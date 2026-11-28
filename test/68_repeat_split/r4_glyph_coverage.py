"""Rank each true glyph after scoring; pool eight ranks only for evaluation."""
import json
from pathlib import Path
import numpy as np
import r4_benchmark as B


def measure():
    B.CENTERED = B.dct_encode(B.IMAGES, 16)
    B.CENTERED -= B.CENTERED.mean(axis=0, keepdims=True)
    z = B.row_zscore
    ranks = {}
    for m in B.MICE:
        for s in B.SESSIONS:
            d = B.session_scores('v5_50e7', m, s, 50)
            y = d['y']; scores = {'v5_seed1': d['v5_s1'], 'v5_seed3': d['v5_s3']}
            v, l = [], []
            for run, ms in [('e53_10e7',10), ('e53_25e12',25), ('e53_50e9',50)]:
                d = B.session_scores(run, m, s, ms)
                assert np.array_equal(d['y'], y)
                v.append(z(d['v5_s3'])); l.append(z(d['logits_s3']))
            base, e53 = np.mean(v, axis=0), np.mean(l, axis=0)
            d = B.session_scores('v5_250e7', m, s, 250)
            assert np.array_equal(d['y'], y)
            x = np.mean([z(base), z(e53), z(d['logits_s3'])], axis=0)
            row = np.partition(x, -100, axis=1)[:, -100:].mean(1, keepdims=True)
            col = np.partition(x, -100, axis=0)[-100:, :].mean(0, keepdims=True)
            scores.update(e53=e53, multiview=base, current=2*x-row-col)
            for k, a in scores.items():
                t = a[np.arange(len(y)), y, None]
                rank = 1+(a>t).sum(1)+((a==t)&(np.arange(B.K)<y[:,None])).sum(1)
                ranks.setdefault(k, []).append(rank[np.argsort(y)])
    ref = np.min(ranks['current'], axis=0)==1
    out = {}
    union = ref.copy()
    for k, rr in ranks.items():
        best = np.min(rr, axis=0); hit = best==1; union |= hit
        out[k] = dict(correct_trials=int((np.array(rr)==1).sum()), best_rank_counts=np.histogram(best, [1,2,6,33,B.K+1])[0].tolist(), new_vs_current=int((hit & ~ref).sum()), lost_vs_current=int((ref & ~hit).sum()), best_rank_by_target=best.tolist())
    assert out['current']['correct_trials']==572 and ref.sum()==462
    return dict(rank_bins=['1','2-5','6-32','33+'], oracle_any_model=int(union.sum()), models=out)


if __name__ == '__main__':
    result = measure()
    for k, v in result['models'].items(): print(k,{a:b for a,b in v.items() if a!='best_rank_by_target'})
    print('oracle',result['oracle_any_model'])
    Path(__file__).with_name('GLYPH_COVERAGE.json').write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
