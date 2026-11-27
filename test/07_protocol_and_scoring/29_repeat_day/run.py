"""Matched public-train repeat/day prototype diagnostic."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[1] / 'data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset'
C, N, SEED = 2748, 256, 29
IDS = np.sort(np.random.default_rng(SEED).choice(C, N, replace=False))
BOOT = np.random.default_rng(2901).multinomial(N, np.full(N, 1/N), size=2000) / N
FIELDS = ('top1', 'recall32', 'outside32', 'inside32_wrong', 'same_score', 'similar_score', 'margin', 'win', 'tie')


def unit(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.all(norm > 0)
    return x / norm


def features(entry, hz, window, rng):
    assert entry['split'] == 'train' and Path(entry['path']).parts[0] == 'train'
    with np.load(DATA / entry['path']) as z:
        y = np.repeat(z['target'], np.diff(z['group_ptr']))
        ptr, t, u = z['trial_ptr'], z['spike_samples'], z['unit_index']
    nu = entry['n_units']
    assert len(ptr) == len(y)+1 and ptr[0] == 0 and ptr[-1] == len(t) == len(u)
    assert np.array_equal(np.bincount(y), np.full(C, 4))
    assert hz == 30000 and window == 250 and 0 <= t.min() <= t.max() < 7500 and 0 <= u.min() <= u.max() < nu
    trial = np.repeat(np.arange(len(y)), np.diff(ptr))
    flat = (trial*nu + u)*5 + t//1500
    x = np.bincount(flat, minlength=len(y)*nu*5).reshape(len(y), nu*5).astype('f')
    order = rng.permuted(np.argsort(y, kind='stable').reshape(C, 4), axis=1)
    return x[order[IDS]], unit(x[order[:, 2:]].mean(1))


def measure(q, proto, near):
    s = unit(q) @ proto.T
    i = np.arange(N)
    a, b = s[i, IDS], s[i, near]
    rank = 1+(s>a[:, None]).sum(1)+((s==a[:, None]) & (np.arange(C)<IDS[:, None])).sum(1)
    return np.column_stack((rank==1, rank<=32, rank>32, (rank>1)&(rank<=32), a, b, a-b, a>b, a==b)).astype(float)


def stats(x):
    v = x.reshape(-1, len(FIELDS))
    return dict(n=len(v), counts={FIELDS[i]:int(v[:, i].sum()) for i in (0,1,2,3,7,8)}, **dict(zip(FIELDS, v.mean(0).round(7).tolist())))


def interval(x):
    # x has one row per glyph after pairing and averaging over fixed mouse-days.
    lo, hi = np.quantile(BOOT @ x, [.025,.975], axis=0)
    return {f:[round(float(a),7) for a in vals] for f, vals in zip(FIELDS, zip(x.mean(0), lo, hi))}


def main():
    cat = json.loads((DATA/'catalog.json').read_text())
    with np.load(DATA/'train/pictures.npz') as z:
        images = z['images']
    assert images.shape == (C,64,64)
    # Centered pixel correlation is invariant to common foreground/background polarity.
    pix = images.reshape(C,-1).astype('f'); pix = unit(pix-pix.mean(1,keepdims=True))
    visual = pix[IDS] @ pix.T; visual[np.arange(N), IDS] = -np.inf
    near = visual.argmax(1)
    train = sorted((e for e in cat['blocks'] if e['split']=='train'), key=lambda e:(e['mouse'], e['session']))
    rows, pool, pairs = {}, {}, {}
    rng = np.random.default_rng(2900)
    for mouse in sorted({e['mouse'] for e in train}):
        es = [e for e in train if e['mouse']==mouse]
        days = [e['session'] for e in es]; assert days == [f'session{i}' for i in range(1,5)]
        blocks = [features(e, cat['sample_rate_hz'], cat['window_ms'], rng) for e in es]
        for j, day in enumerate(days):
            key = f'{mouse}/{day}'; rows[key] = {'cross_reference': days[j-1], 'same4':None}
            q, same = blocks[j]; cross = blocks[j-1][1]
            for k in (1,2,4):
                query = q[:,:k].mean(1)
                for label, proto in [('same',same),('cross',cross)]:
                    if label=='same' and k==4: continue
                    arm = f'{label}{k}'; x = measure(query, proto, near)
                    rows[key][arm] = stats(x); pool.setdefault(arm, []).append(x)
    agg = {a:stats(np.stack(xs)) for a,xs in pool.items()}; agg['same4'] = None
    bychar = {a:np.mean(xs,axis=0) for a,xs in pool.items()}
    for a,b in [('same1','cross1'),('same2','cross2'),('same2','same1'),('cross2','cross1'),('cross4','cross2')]:
        pairs[a+'-'+b] = interval(bychar[a]-bychar[b])
    result = dict(protocol=dict(split='public train only',seed=SEED, glyphs=IDS.tolist(), visual_neighbors=near.tolist(),visual_rule='highest centered 64x64 pixel cosine excluding self; lowest id on tie',candidates=C,repeat_count_per_glyph_day=4,ref_trials_per_glyph=2,trial_seed=2900,trial_rule='separate permutation per glyph and sorted mouse/day; query prefix, reference last two; cyclic previous day within mouse for cross',decoder='cosine of mean 50ms binned spike counts; no fitted preprocessing; ties lower id first',same4_reason='all four trials consumed; no independent reference',bootstrap='2000 percentile resamples of 256 glyphs (seed 2901); retain all fixed mouse-days and arms together; triples [estimate,lo95,hi95]',errors='outside32: rank>32; inside32_wrong: ranks 2..32; win: same_score>similar_score'),by_mouse_day=rows,pooled=agg,paired=pairs,margin95={a:interval(x)['margin'] for a,x in bychar.items()})
    path = OUT/'results.json'; path.write_text(json.dumps(result,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(path, len(rows),'mouse-days'); print(json.dumps(agg)); print(json.dumps({p:{k:d[k] for k in ('top1','recall32')} for p,d in pairs.items()})); print(json.dumps(result['margin95']))


if __name__ == '__main__':
    main()
