"""Frozen ridge brain-to-glyph continuation and paired-row permutation control."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import sync_response_templates as T

B, F, O = T.B, T.F, T.O
NEW = ('ridge_dct16', 'ridge_hv')
OLD = (*T.NEW, *O.ORDER)
ORDER = (*NEW, *OLD)


def fit(m, r, targets):
    gram, rhs, norm = 0, 0, {}
    for s in B.SESSIONS:
        x = np.concatenate([T.features(m, s, t, 'bins50') for t in range(1, r)])
        y = np.concatenate([F.labels(m, s, t) for t in range(1, r)])
        mu, sd = x.mean(0), x.std(0)
        sd[sd == 0] = 1
        norm[s] = mu, sd
        x -= mu
        x /= sd
        gram += x.T @ x
        rhs += x.T @ targets[y]
    alpha = .1 * gram.diagonal().mean()
    gram.flat[::len(gram)+1] += alpha
    return np.linalg.solve(gram, rhs), norm, float(alpha)


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.)


def cycles(ids, blocks):
    live, history = np.arange(len(ids)), []
    while len(live):
        before, detail = len(live), {}
        for k in ORDER:
            live, detail[k] = O.peel(blocks[k], live)
            if not len(live):
                break
        history.append(dict(method_steps=detail, removed=before-len(live), remaining=len(live)))
        if len(live) == before:
            break
    return dict(cycles=history, remaining=len(live), remaining_ids=ids[live].tolist())


def evaluate(out, r, ref, dct, hv):
    ids = np.array(ref['remaining_ids'])
    assert len(ids) == ref['remaining'] == {3:1308, 4:909}[r] and np.array_equal(np.unique(ids), ids)
    rng = np.random.default_rng(42)
    targets = np.concatenate([dct, hv], axis=1).astype(np.float64)
    banks = dict(zip(NEW, (unit(targets[:, :dct.shape[1]]), unit(targets[:, dct.shape[1]:]))))
    old = {k: [] for k in OLD}
    arms = ('observed', 'permuted')
    blocks = {a: {**old, **{k: [] for k in NEW}} for a in arms}
    full = {a: {k: [] for k in NEW} for a in arms}
    result = dict(start=len(ids), alpha={})
    for m in B.MICE:
        if r == 3:
            F.train_dev(out, m)
        else:
            for name, _, _ in F.RUNS:
                man = json.loads((out/name/'bench_manifest.json').read_text())
                assert man['protocol'] == 'repeat1-3 train, repeat4 evaluate'
                for seed in B.SEEDS:
                    assert B.sha(B.checkpoint_path(out/name, 'task3', m, seed)) == man['checkpoints'][f'{m}/{seed}']
        w, norm, result['alpha'][m] = fit(m, r, targets)
        fits = {k: T.fit(m, r, k) for k in T.NEW}
        for s in B.SESSIONS:
            scores = O.score(out, m, s, r, dct)
            for k in T.NEW:
                p, sd = fits[k][s]
                x = T.features(m, s, r, k) / sd
                scores[k] = 2 * (x @ p.T) - (p**2).sum(1) - (x**2).sum(1)[:, None]
            mu, sd = norm[s]
            preds = ((T.features(m, s, r, 'bins50') - mu) / sd) @ w
            perm = rng.permutation(B.K)
            y = F.labels(m, s, r)  # evaluation alignment only
            rows = np.argsort(y)
            for k in OLD:
                old[k].append(scores.pop(k)[np.ix_(rows[ids], ids)])
            for k, p in zip(NEW, np.split(preds, [dct.shape[1]], axis=1)):
                score = unit(p) @ banks[k].T
                top = score.argmax(1)
                for arm, ix in zip(arms, (rows, perm[rows])):
                    full[arm][k].append(top[ix] == y[rows])
                    blocks[arm][k].append(score[np.ix_(ix[ids], ids)])
    for arm in arms:
        stats = {k: F.totals(np.stack(v)) for k, v in full[arm].items()}
        result[arm] = dict(full=stats, **cycles(ids, blocks[arm]))
        print(r, arm, stats, [(c['removed'], c['remaining']) for c in result[arm]['cycles']], flush=True)
    return result


if __name__ == '__main__':
    ref = json.loads((F.HERE/'SYNC_RESPONSE_TEMPLATES.json').read_text())
    assert tuple(ref['order']) == OLD
    dct = B.dct_encode(B.IMAGES, 16)
    dct -= dct.mean(0, keepdims=True)  # identical bank for original soft DCT
    ink = 1. - B.IMAGES
    hv = np.concatenate([ink.sum(2), ink.sum(1)], axis=1)
    hv -= hv.mean(0, keepdims=True)
    res = dict(source='SYNC_RESPONSE_TEMPLATES.json', classes=B.K, trials=B.K*len(B.MICE)*len(B.SESSIONS), order=ORDER,
               protocol='50ms log1p counts; within mouse-day train repeats 1..r-1 mean and population SD (zero -> 1); pool standardized training rows within mouse. Ridge alpha=.1*mean(diag(X.T@X)); separate centered public glyph targets (16x16 DCT and 64 row + 64 column ink sums), separate cosine full-gallery top1. Freeze before label alignment. Same old scores and initial IDs for both arms. Synchronous truth-assisted class removal, not deployable accuracy.',
               control='Single descriptive permutation. Reset NumPy default_rng(42) per fold, draw one full trial-row permutation per mouse/day in benchmark order, shared across both new methods and frozen across cycles. Original eleven unchanged.')
    res['r4'] = evaluate(B.OUT, 4, ref['r4'], dct, hv)
    with TemporaryDirectory() as tmp:
        res['r3'] = evaluate(Path(tmp), 3, ref['r3'], dct, hv)
    (F.HERE/'SYNC_BRAIN_GLYPH.json').write_text(json.dumps(res, indent=2, allow_nan=False)+'\n', encoding='utf-8')
