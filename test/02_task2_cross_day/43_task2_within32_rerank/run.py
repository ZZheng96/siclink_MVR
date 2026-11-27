import json
from pathlib import Path
import sys
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
TEST = OUT.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(TEST/'41_forward_cnn_seed_ensemble'), str(TEST), str(TEST.parent)]
import run as x
import evaluate_final as ev
import src.final as prod
import numpy as np
import torch


def main():
    plan = OUT/'preregistration.json'
    assert x.sha256(plan) == '33c7174c2939afb533fe863284abf99b85065fcda3ae0325dcf008628d73c101'
    ids = np.array(json.loads(plan.read_text())['ids'], dtype='<i8')
    prev = TEST/'42_task2_production_paired'
    ref = json.loads((prev/'results.json').read_text())
    cfgpath = TEST.parent/'src/final_config.json'
    cfg = json.loads(cfgpath.read_text())
    models = TEST/'.cache/evaluate_final_task23_v1/models'
    assert x.sha256(cfgpath) == ref['config_sha256']
    for key, digest in ref['checkpoint_sha256'].items():
        assert x.sha256(models/key) == digest
    img = (ev.load_images(ev.DATA) >= 128).astype(np.float32)
    blocks = ev.load_feature_blocks(ev.DATA, 'train', bin_ms=50)
    split, parts, _ = ev.public_protocol('task2', blocks, len(img))
    excluded = np.r_[parts['heldout'],parts['tune'],json.loads((prev/'ids.json').read_text())['ids']]
    pool = np.setdiff1d(np.arange(len(img)), excluded)
    np.testing.assert_array_equal(ids, np.sort(np.random.default_rng(271828).permutation(pool)[:512]))
    ev.check_training_scope('task2', cfg, blocks, split)
    targets = [b for b in blocks if len(split.blocks[b.key].validation_trials)]
    assert [b.key for b in targets] == ['mice1/session4','mice2/session4']
    scores, obs = [], []
    assign, aggregate = prod.unique_assignment, prod.aggregate_by_target
    def capture(s):
        scores.append(s.copy())
        return assign(s)
    def capture_counts(*args):
        pair = aggregate(*args)
        obs.append(pair[1].copy())
        return pair
    with patch.object(prod,'unique_assignment',capture), patch.object(prod,'aggregate_by_target',capture_counts):
        p = prod.predict_task(cfg,'task2',models,images=img,blocks=blocks,split=split,device=x.DEVICE,reranker_checkpoint=models/'listwise_cnn.pt',candidate_top_n=32)
    assert len(scores) == len(obs) == 2 and p['reranker']
    hist = np.isin(p['targets'], parts['heldout'])
    assert hist.sum() == 1024
    value = ev.official_evaluate(p['predictions'][hist],img[p['targets'][hist]],p['groups'][hist],x.DEVICE,16)['ocf_mean']
    result = dict(historical_ocf=value, preregistration_sha256=x.sha256(plan), ids_sha256=x.array_sha256(ids), per_mouse={},advance=False)
    print('Historical OCF:',value,flush=True)
    if not abs(value-23.824141) <= 1e-5:
        x.write_json(OUT/'results.json',dict(result,status='historical_mismatch'))
        return
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    bank = torch.from_numpy(img[:,None]).to(x.DEVICE)
    for j,b in enumerate(targets):
        source = [a for a in blocks if a.mouse == b.mouse and a.session != 'session4']
        assert [a.session for a in source] == list(x.DAYS[:3])
        counts = []
        for a in source:
            tr = split.blocks[a.key].train_trials
            label = a.trial_targets[tr]
            idx, mean = aggregate(a.features[tr], label)
            np.testing.assert_array_equal(idx,np.arange(len(img)))
            assert (np.bincount(label) == 4).all()
            counts.append(mean)
        pooled = np.mean(counts, axis=0)
        rates = {seed:x.train_cnn_seed(bank,pooled,seed)[0] for seed in x.CNN_SEEDS}
        ensemble = x.assemble_arm_rates(rates)['cnn_seed_ensemble']
        digest = x.array_sha256(ensemble)
        assert digest == ref['per_mouse'][b.mouse]['rate_sha256']
        s = scores[j]
        mix = .8*prod.row_zscore(s) + .2*prod.row_zscore(x.exp40.poisson_scores(obs[j],ensemble))
        mask = p['groups'] == b.key
        np.testing.assert_array_equal(p['targets'][mask],np.arange(len(img)))
        assert np.isfinite(mix).all() and np.isfinite(s).all()
        old = p['candidate_indices'][mask]
        r = np.arange(len(img))[:,None]
        desired = np.take_along_axis(old,np.lexsort((old,-mix[r,old]),axis=1),axis=1)
        new = s.copy()
        new[r,desired] = s[r,old]
        order = np.argsort(-new,axis=1,kind='stable')
        actual = order[:,:32]
        badset = np.any(np.sort(old,axis=1)!=np.sort(actual,axis=1),axis=1)
        badorder = np.any(desired!=actual,axis=1)
        row = dict(rows=len(old),set_mismatch=int(badset.sum()),order_mismatch=int(badorder.sum()), rows_with_v5_top32_ties=int((np.diff(s[r,old],axis=1)==0).any(axis=1).sum()),rows_with_mixed_top32_ties=int((np.diff(mix[r,desired],axis=1)==0).any(axis=1).sum()), changed_order=int((old!=actual).any(axis=1).sum()),rate_sha256=digest)
        result['per_mouse'][b.mouse] = row
        if badset.any() or badorder.any():
            x.write_json(OUT/'results.json',dict(result,status='invariant_failure'))
            print(row, flush=True)
            return
        assigned = assign(new)
        per_id = dict(ids=ids, assigned=assigned[ids], original_top32=old, reranked_top32=actual)
        for arm,pred,top in [('v5',p['predictions'][mask][ids],old),('within32',img[assigned[ids]],actual)]:
            ocf = ev.score_predictions(pred,img[ids],device=x.DEVICE,batch_size=16)
            hit = (top[ids]==ids[:,None]).any(axis=1)
            row[arm] = dict(final_ocf=float(np.mean(ocf,dtype=np.float64)),recall32=float(hit.mean()),hits=int(hit.sum()))
            per_id[arm+'_ocf'] = ocf
            per_id[arm+'_hit'] = hit
        row['delta_ocf'] = row['within32']['final_ocf']-row['v5']['final_ocf']
        np.savez_compressed(OUT/f'{b.mouse}.npz',**per_id)
        print(b.mouse,row,flush=True)
    rows = list(result['per_mouse'].values())
    result.update(status='complete',macro_delta=float(np.mean([r['delta_ocf'] for r in rows])),advance=all(r['delta_ocf']>0 for r in rows))
    x.write_json(OUT/'results.json',result)
    print('macro delta:',result['macro_delta'],'advance:',result['advance'])

if __name__ == '__main__':
    main()
