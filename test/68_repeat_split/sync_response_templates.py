"""Continue the synchronous oracle from each fold's saved eight-readout residual."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import sync_oracle as O

B, F = O.B, O.F
NEW = ('total250', 'bins50', 'early_minus_late125')


def features(m, s, r, mode):
    w = dict(zip(NEW, (250, 50, 125)))[mode]
    x = F.counts(m, s, r, w).astype(np.float64)
    if w == 125:
        p = x.reshape(B.K, B.UNITS[m], 2)
        d = p[:, :, 0] - p[:, :, 1]
        return np.sign(d) * np.log1p(abs(d))
    return np.log1p(x)


def fit(m, r, mode):
    local, scale = {}, {}
    for s in B.SESSIONS:
        train = np.stack([features(m, s, t, mode)[np.argsort(F.labels(m, s, t))]
                          for t in range(1, r)])
        local[s] = train.mean(0)
        sd = train.std(axis=(0, 1))
        scale[s] = np.where(sd > 0, sd, 1.)
    pooled = np.mean(list(local.values()), axis=0)
    return {s: ((local[s] + pooled) * .5 / scale[s], scale[s]) for s in B.SESSIONS}


def evaluate(r, out, ref, centered):
    ids = np.array(ref['remaining_ids'])
    assert len(ids) == ref['remaining'] == {3: 1444, 4: 1072}[r]
    assert np.array_equal(np.unique(ids), ids)
    start = len(ids)
    # Keep only the residual submatrices after independent full-library evaluation.
    blocks = {k: [] for k in (*NEW, *O.ORDER)}
    full = {k: [] for k in NEW}
    for m in B.MICE:
        if r == 3:
            F.train_dev(out, m)
        else:
            for name, _, _ in F.RUNS:
                manifest = json.loads((out/name/'bench_manifest.json').read_text())
                assert manifest['protocol'] == 'repeat1-3 train, repeat4 evaluate'
                for seed in B.SEEDS:
                    assert B.sha(B.checkpoint_path(out/name, 'task3', m, seed)) == manifest['checkpoints'][f'{m}/{seed}']
        fits = {k: fit(m, r, k) for k in NEW}
        for s in B.SESSIONS:
            scores = O.score(out, m, s, r, centered)
            for k in NEW:
                proto, sd = fits[k][s]
                x = features(m, s, r, k) / sd
                scores[k] = 2 * (x @ proto.T) - (proto**2).sum(1) - (x**2).sum(1)[:, None]
            # Labels first used here for held-out evaluation, never in scoring.
            y = F.labels(m, s, r)
            rows = np.argsort(y)
            for k, a in scores.items():
                if k in full:
                    full[k].append((a.argmax(1) == y)[rows])
                blocks[k].append(a[np.ix_(rows[ids], ids)])
            del scores, a
    full = {k: F.totals(np.stack(v)) for k, v in full.items()}
    live, cycles = np.arange(start), []
    while len(live):
        before, detail = len(live), {}
        for k in blocks:
            live, detail[k] = O.peel(blocks[k], live)
            if not len(live):
                break
        cycles.append(dict(method_steps=detail, removed=before-len(live), remaining=len(live)))
        print('R', r, 'cycle', len(cycles), 'removed', before-len(live), 'remaining', len(live), flush=True)
        if len(live) == before:
            break
    print('R', r, 'full', full, flush=True)
    return dict(start=start, full=full, cycles=cycles, remaining=len(live), remaining_ids=ids[live].tolist())


if __name__ == '__main__':
    ref = json.loads((F.HERE/'SYNC_ORACLE.json').read_text())
    centered = B.dct_encode(B.IMAGES, 16)
    centered -= centered.mean(0, keepdims=True)
    result = dict(classes=B.K, trials=B.K*len(B.MICE)*len(B.SESSIONS), order=(*NEW, *O.ORDER),
                  protocol='Fold-specific training repeats 1..r-1. Mean log1p counts (signed log1p of early-late for contrast) by class/day, mixed equally with same-mouse all-day class mean; negative squared Euclidean distance divided featurewise by day training variance (zero SD replaced by 1 before scaling). Frozen full-gallery scores. Each method peels synchronously until stalled or empty. Residual IDs and evaluation alignment use truth, not deployable accuracy.')
    result['r4'] = evaluate(4, B.OUT, ref['r4'], centered)
    with TemporaryDirectory() as tmp:
        result['r3'] = evaluate(3, Path(tmp), ref['r3'], centered)
    (F.HERE/'SYNC_RESPONSE_TEMPLATES.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
