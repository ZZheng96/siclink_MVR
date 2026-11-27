"""Fixed public-fold candidate audit. Run with .venv/Scripts/python.exe -B -u."""
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp23', HERE.parent / '23_candidate_oracle/run.py')
x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x)
e = x.e
N, K, DEVICE = 64, 32, 'cuda'


@torch.inference_mode()
def candidates(task, images):
    unseen = task == 'task1'
    data = e.prepare_task1(images) if unseen else e.prepare_seen(task, images, DEVICE)
    # Predictors need only the mouse keys, not the training arrays.
    data['training'] = dict.fromkeys(data['training'])
    rows = sorted(data['rows'], key=lambda r: r['block'])
    pics = np.random.default_rng(23).permutation(data['evaluation_pictures'])[:N]
    assert len(np.unique(pics)) == N
    selections = []
    for r, group in zip(rows, np.array_split(pics, len(rows))):
        idx = np.flatnonzero(np.isin(r['targets'], group))
        assert np.array_equal(np.sort(r['targets'][idx]), np.sort(group))
        selections.append(idx)
    records = []
    if unseen:
        e.predict_task1_seed(data, 42, DEVICE)
        bank = e.render_gb2312_bank(e.FONT_PATH, font_size=52, variation='Regular')
        queries = np.concatenate([r['continuous'][42][idx] for r, idx in zip(rows, selections)])
        def embed(array):
            return F.normalize(x._retrieval_embedding(torch.from_numpy(array[:, None]).to(DEVICE), 'directional2'), dim=1)
        q = embed(queries)
        scores = np.concatenate([(q @ embed(part).T).cpu().numpy() for part in np.array_split(bank, 32)], axis=1)
        ranks = iter(np.argsort(-scores, kind='stable')[:, :K])
        for r, idx in zip(rows, selections):
            for t in r['targets'][idx]:
                ids = next(ranks)
                records.append((r['block'], int(t), ids, int(ids[0])))
    else:
        seeds = e.SEEDS if task == 'task2' else (42,)
        for seed in seeds:
            e.predict_seen_seed(task, data, seed)
        bank = images
        for r, idx in zip(rows, selections):
            logits = np.mean([r['logits'][s] for s in seeds], axis=0, dtype=np.float32)
            scores = .5 * e.row_zscore(logits) + .5 * r['encoding_z']
            assert scores.shape == (len(images), len(images))
            assigned = e.unique_assignment(scores)
            for i, ids in zip(idx, np.argsort(-scores[idx], kind='stable')[:, :K]):
                records.append((r['block'], int(r['targets'][i]), ids, int(assigned[i])))
    return bank, records


def audit(task, images, bank, records, cache):
    # Reuse exp23's ordered binary-pixel cache and official CUDA score_predictions.
    x.KS = (1, K)
    out = x.measure(task, images, bank, records, cache)
    rows = out['rows']
    result = dict(n=len(rows), block_counts={b:v['n'] for b,v in out['by_block'].items()},
                  row_top1_ocf=out['overall']['oracle_ocf'][1], final_top1_ocf=out['overall']['actual_top1_ocf'],
                  oracle32_ocf=out['overall']['oracle_ocf'][K], pair_requests=out['pair_requests'],
                  newly_scored_pairs=out['newly_scored_image_pairs'], identity=None)
    if task != 'task1':
        inside = np.array([r['hit'][K] for r in rows])
        result['identity'] = dict(true_outside32=int((~inside).sum()))
        for key, hit in [('before_assign', [r['hit'][1] for r in rows]), ('after_assign', [r['actual_hit'] for r in rows])]:
            hit = np.asarray(hit)
            result['identity'][key] = dict(correct=int(hit.sum()), accuracy=float(hit.mean()),
                                           outside32_wrong=int((~inside & ~hit).sum()),
                                           inside32_wrong=int((inside & ~hit).sum()),
                                           outside32_correct=int((~inside & hit).sum()))
    return result


def main():
    images = (e.load_images(e.DATA) >= 128).astype(np.float32)
    result = dict(scope='Public training data only; local diagnostic, not blind test.',
                  sampling='exp18 split 42, partition 123; RNG 23 permutes evaluation_pictures; first 64 split evenly in sorted block order, one row per picture.',
                  candidates='exp23 unchanged: task1 seed42 directional2 GB2312; task2 seeds42/202/777, task3 seed42; 50:50 standardized logits/encoding, full-block Hungarian.',
                  metrics='oracle32 and membership use pre-assign ranks; final_top1 uses actual output. Task1 identity unavailable. Official PNG-parity CUDA OCF, sample means.', tasks={})
    cache = {}
    for task in ('task1', 'task2', 'task3'):
        bank, records = candidates(task, images)
        result['tasks'][task] = audit(task, images, bank, records, cache)
        print(task, json.dumps(result['tasks'][task]), flush=True)
        del bank, records
        e.release_memory()
    (HERE / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
