"""Run from repository root: .venv/Scripts/python -B -u test/02_task2_cross_day/42_task2_production_paired/run.py"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
TEST = OUT.parent
sys.path[:0] = [str(TEST / '41_forward_cnn_seed_ensemble'), str(TEST), str(TEST.parent)]
import run as x
import evaluate_final as ev
import src.final as prod
import numpy as np
import torch


def main():
    cfgpath = TEST.parent / 'src/final_config.json'
    cfg = json.loads(cfgpath.read_text())
    models = TEST / '.cache/evaluate_final_task23_v1/models'
    frozen = json.loads((OUT / 'ids.json').read_text())
    ids = np.array(frozen['ids'], dtype='<i8')
    assert x.array_sha256(ids) == frozen['sha256'] == '354a5ccbd55f8ca06c25a10158e510d96174685368ee3cce6e99159f199061e3'
    img = (ev.load_images(ev.DATA) >= 128).astype(np.float32)
    blocks = ev.load_feature_blocks(ev.DATA, 'train', bin_ms=50)
    split, parts, _ = ev.public_protocol('task2', blocks, len(img))
    np.testing.assert_array_equal(ids, np.sort(np.random.default_rng(314159).permutation(np.setdiff1d(np.arange(len(img)), np.r_[parts['heldout'],parts['tune']]))[:512]))
    ev.check_training_scope('task2', cfg, blocks, split)
    target = [b for b in blocks if len(split.blocks[b.key].validation_trials)]
    assert [b.key for b in target] == ['mice1/session4', 'mice2/session4']
    scores, obs = [], []
    assign, aggregate = prod.unique_assignment, prod.aggregate_by_target

    def capture(s):
        scores.append(s.copy())
        return assign(s)

    def capture_counts(*args):
        rows = aggregate(*args)
        obs.append(rows[1].copy())
        return rows

    with patch.object(prod, 'unique_assignment', capture), patch.object(prod, 'aggregate_by_target', capture_counts):
        p = prod.predict_task(cfg, 'task2', models, images=img, blocks=blocks, split=split, device=x.DEVICE,
                              reranker_checkpoint=models/'listwise_cnn.pt', candidate_top_n=32)
    assert len(scores) == len(obs) == 2 and p['reranker']
    hist = np.isin(p['targets'], parts['heldout'])
    assert hist.sum() == 1024
    historical = ev.official_evaluate(p['predictions'][hist], img[p['targets'][hist]], p['groups'][hist], x.DEVICE, 16)
    value = historical['ocf_mean']
    print('Historical OCF:', value, flush=True)
    result = dict(historical_ocf=value, historical_error=abs(value-23.824141), ids_sha256=frozen['sha256'],
                  preregistration_sha256=x.sha256(OUT/'preregistration.json'), environment=x.environment(),
                  checkpoint_sha256={str(f.relative_to(models)):x.sha256(f) for f in [models/'listwise_cnn.pt',*sorted((models/'task2').rglob('*.pt')),*sorted((models/'task2').rglob('*.npy'))]}, config_sha256=x.sha256(cfgpath))
    if not result['historical_error'] <= 1e-5:
        x.write_json(OUT/'results.json', dict(result, status='historical_mismatch_stopped'))
        raise RuntimeError('Historical control mismatch; main analysis stopped')
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    bank = torch.from_numpy(img[:,None]).to(x.DEVICE)
    records = {}
    for j,b in enumerate(target):
        source = [a for a in blocks if a.mouse == b.mouse and a.session != 'session4']
        assert [a.session for a in source] == list(x.DAYS[:3])
        counts = []
        for a in source:
            tr = split.blocks[a.key].train_trials
            label = a.trial_targets[tr]
            idx, mean = aggregate(a.features[tr], label)
            np.testing.assert_array_equal(idx, np.arange(len(img)))
            assert (np.bincount(label) == 4).all()
            counts.append(mean)
        pooled = np.mean(counts, axis=0)
        rates = {seed:x.train_cnn_seed(bank, pooled, seed)[0] for seed in x.CNN_SEEDS}
        ensemble = x.assemble_arm_rates(rates)['cnn_seed_ensemble']
        mixed = .8*prod.row_zscore(scores[j]) + .2*prod.row_zscore(x.exp40.poisson_scores(obs[j],ensemble))
        assigned = assign(mixed)
        mask = p['groups'] == b.key
        truth = p['targets'][mask]
        np.testing.assert_array_equal(truth, np.arange(len(img)))
        top = np.argsort(-mixed[ids],axis=1,kind='stable')[:,:32]
        row = {}
        for arm, pred, candidates in [('v5',p['predictions'][mask][ids],p['candidate_indices'][mask][ids]), ('mixed',img[assigned[ids]],top)]:
            ocf = ev.score_predictions(pred,img[ids],device=x.DEVICE,batch_size=16)
            hit = (candidates == ids[:,None]).any(axis=1)
            row[arm] = dict(final_ocf=float(np.mean(ocf,dtype=np.float64)), recall32=float(hit.mean()))
            np.savez_compressed(OUT/f'{b.mouse}_{arm}.npz', ids=ids,ocf=ocf,hit32=hit)
        row['delta'] = {k:row['mixed'][k]-v for k,v in row['v5'].items()}
        row['rate_sha256'] = x.array_sha256(ensemble)
        records[b.mouse] = row
        print(b.mouse,row,flush=True)
    overall = {arm:{k:float(np.mean([r[arm][k] for r in records.values()])) for k in ('final_ocf','recall32')} for arm in ('v5','mixed','delta')}
    advance = all(r['delta']['final_ocf']>0 and r['delta']['recall32']>=0 for r in records.values())
    x.write_json(OUT/'results.json',dict(result,status='complete',per_mouse=records,overall=overall,advance=advance))
    print('overall',overall,'advance',advance)

if __name__ == '__main__':
    main()
