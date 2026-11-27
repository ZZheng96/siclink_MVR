"""Public training tune-fold candidate ceiling; no model selection or submission."""
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval import _retrieval_embedding

spec = importlib.util.spec_from_file_location('exp18', ROOT / 'test/06_fusion_transduction/18_seed_ensemble/run.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
KS = (1, 8, 32, 128)
PER_BLOCK = 4
DEVICE = 'cuda'


def subset(data):
    return np.random.default_rng(23).permutation(data['tune_pictures'])[:PER_BLOCK]


@torch.inference_mode()
def candidates(task, images):
    records = []
    if task == 'task1':
        data = e.prepare_task1(images)
        pics = subset(data)
        e.predict_task1_seed(data, 42, DEVICE)
        bank = e.render_gb2312_bank(e.FONT_PATH, font_size=52, variation='Regular')
        query = np.concatenate([r['continuous'][42][np.isin(r['targets'], pics)] for r in data['rows']])
        query = F.normalize(_retrieval_embedding(torch.from_numpy(query[:, None]).to(DEVICE), 'directional2'), dim=1)
        chunks = []
        for part in np.array_split(bank, 32):
            emb = _retrieval_embedding(torch.from_numpy(part[:, None]).to(DEVICE), 'directional2')
            chunks.append((query @ F.normalize(emb, dim=1).T).cpu().numpy())
        ranks = iter(np.argsort(-np.concatenate(chunks, axis=1), kind='stable')[:, :128])
        for r in data['rows']:
            targets = r['targets'][np.isin(r['targets'], pics)]
            assert len(np.unique(targets)) == len(targets) == PER_BLOCK
            for t in targets:
                ids = next(ranks)
                records.append((r['block'], int(t), ids, int(ids[0])))
    else:
        data = e.prepare_seen(task, images, DEVICE)
        pics = subset(data)
        seeds = e.SEEDS if task == 'task2' else (42,)
        for seed in seeds:
            e.predict_seen_seed(task, data, seed)
        bank = images
        for r in data['rows']:
            logits = np.mean([r['logits'][s] for s in seeds], axis=0, dtype=np.float32)
            scores = .5 * e.row_zscore(logits) + .5 * r['encoding_z']
            assert scores.shape == (len(bank), len(bank))
            assigned = e.unique_assignment(scores)
            selected = np.flatnonzero(np.isin(r['targets'], pics))
            assert len(np.unique(r['targets'][selected])) == len(selected) == PER_BLOCK
            for i in selected:
                ids = np.argsort(-scores[i], kind='stable')[:128]
                records.append((r['block'], int(r['targets'][i]), ids, int(assigned[i])))
    return bank, records, pics.tolist()


def measure(task, images, bank, records, cache):
    # Pack exact binary pixels, preserving prediction/GT order, not class identity.
    def keys(array):
        return [x.tobytes() for x in np.packbits(array.astype(bool).reshape(len(array), -1), axis=1)]
    bk, gt = keys(bank), keys(images)
    missing = {}
    requested = 0
    for _, t, ids, actual in records:
        for i in [*ids, actual]:
            key = bk[i], gt[t]
            requested += 1
            if key not in cache:
                missing[key] = (i, t)
    items = list(missing.items())
    for start in range(0, len(items), 16):
        batch = items[start:start + 16]
        ii, tt = zip(*(v for _, v in batch))
        scores = e.score_predictions(bank[list(ii)], images[list(tt)], device=DEVICE, batch_size=16)
        for (key, _), score in zip(batch, scores):
            cache[key] = float(score)
    rows = []
    for group, t, ids, actual in records:
        values = [cache[bk[i], gt[t]] for i in ids]
        rows.append(dict(block=group, picture=t, actual_index=actual, actual_ocf=cache[bk[actual], gt[t]],
                         actual_hit=None if task == 'task1' else actual == t,
                         candidates=ids.tolist(), oracle={k:max(values[:k]) for k in KS},
                         hit=None if task == 'task1' else {k:bool(t in ids[:k]) for k in KS}))
    def summary(rr):
        return dict(n=len(rr), actual_top1_ocf=float(np.mean([r['actual_ocf'] for r in rr])),
                    actual_accuracy=None if task == 'task1' else float(np.mean([r['actual_hit'] for r in rr])),
                    oracle_ocf={k:float(np.mean([r['oracle'][k] for r in rr])) for k in KS},
                    recall=None if task == 'task1' else {k:float(np.mean([r['hit'][k] for r in rr])) for k in KS})
    return dict(overall=summary(rows), by_block={b:summary([r for r in rows if r['block']==b]) for b in sorted({r['block'] for r in rows})},
                pair_requests=requested, newly_scored_image_pairs=len(items), rows=rows)


def main():
    images = (e.load_images(e.DATA) >= 128).astype(np.float32)
    result = dict(protocol='Public train only; reused tune within fold; non-independent diagnostic, not blind test',
                  split_seed=42, tune_seed=123, subset_seed=23, per_block=PER_BLOCK, ks=KS, tasks={})
    cache = {}
    for task in ('task1','task2','task3'):
        bank, records, pics = candidates(task, images)
        out = measure(task, images, bank, records, cache)
        out['selected_pictures'] = pics
        result['tasks'][task] = out
        (HERE/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        print(task, out['overall'], 'new pairs',out['newly_scored_image_pairs'], flush=True)
        del bank, records
        e.release_memory()


if __name__ == '__main__':
    main()
