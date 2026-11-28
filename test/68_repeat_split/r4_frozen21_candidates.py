"""Full-gallery same-trial diagnostic. Run with the project venv."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import sync_extra_decoders as E

B, F, G, T = E.B, E.F, E.G, E.T
NAMES = ('csls', *(k for k in E.OLD if k != 'csls'), *E.WAVE1)
assert len(set(NAMES)) == len(NAMES) == 21


def session(scores, use_features):
    assert set(scores) == set(NAMES)
    base = scores['csls']
    assert all(a.shape == (B.K, B.K) and np.isfinite(a).all() for a in scores.values())
    top = np.stack([scores[k].argmax(1) for k in NAMES], axis=1)
    old = np.argsort(-base, axis=1, kind='stable')[:, :32].copy()
    new = old.copy()
    for i, order in enumerate(old):
        ids, votes = np.unique(top[i, 1:], return_counts=True)
        picks = list(order[:24])
        for j in (*ids[np.lexsort((ids, -base[i, ids], -votes))], *order[24:]):
            if j not in picks:
                picks.append(j)
            if len(picks) == 32:
                break
        new[i] = picks
    x = None
    if use_features:
        x = np.stack([np.take_along_axis(F.Z(scores[k]), new, axis=1) for k in NAMES], axis=2).astype(np.float32)
    return top, old, new, x


def metrics(parts):
    y, b, u, a, n, pred = [np.concatenate(col) for col in zip(*parts)]
    def stat(h):
        return dict(hits=int(h.sum()), unique=len(np.unique(y[h])))
    base, union = stat(b), stat(u)
    return dict(trials=len(y), baseline=base, union=union, rescued=int((u & ~b).sum()),
                new_chars=union['unique']-base['unique'], old32=int(a.sum()), new32=int(n.sum()),
                added=int((n & ~a).sum()), displaced=int((a & ~n).sum()), net=int(n.sum()-a.sum()),
                reranked=None if np.all(pred == -1) else stat(pred == y))


def fold(out, r, banks, weights):
    targets = np.concatenate([banks['d16'], banks['hv']], axis=1)
    parts, xs, ys = {}, [], []
    for m in B.MICE:
        if r == 3:
            F.train_dev(out, m)
        else:
            E.verify_r4(out, m)
        w, norm, _ = G.fit(m, r, targets)
        fits = {k: T.fit(m, r, k) for k in T.NEW}
        wave = E.wave_scorers(1, m, r, banks)
        parts[m] = []
        for s in B.SESSIONS:
            scores = E.O.score(out, m, s, r, banks['d16'])
            for k in T.NEW:
                p, sd = fits[k][s]
                x = T.features(m, s, r, k) / sd
                scores[k] = 2*(x @ p.T) - (p**2).sum(1) - (x**2).sum(1)[:, None]
            mu, sd = norm[s]
            p = ((T.features(m, s, r, 'bins50')-mu)/sd) @ w
            for k, tag, v in zip(G.NEW, ('d16', 'hv'), np.split(p, [banks['d16'].shape[1]], axis=1)):
                scores[k] = E.unit(v) @ E.unit(banks[tag]).T
            scores.update({k: fun(s) for k, fun in wave.items()})
            top, old, new, x = session(scores, r == 3 or weights is not None)
            del scores
            pred = np.full(B.K, -1) if weights is None else new[np.arange(B.K), (x @ weights).argmax(1)]
            y = F.labels(m, s, r)
            assert np.array_equal(np.sort(y), np.arange(B.K))
            nh = new == y[:, None]
            parts[m].append((y, top[:, 0] == y, (top == y[:, None]).any(1),
                             (old == y[:, None]).any(1), nh.any(1), pred))
            if r == 3:
                xs.append(x); ys.append(nh)
        print(f'R{r} {m} complete', flush=True)
    res = metrics(sum(parts.values(), []))
    res['by_mouse'] = {m: metrics(p) for m, p in parts.items()}
    if r == 3 and res['net'] > 0:
        x = np.concatenate(xs).reshape(-1, len(NAMES)).astype(float)
        y = np.concatenate(ys).ravel()
        def loss(w):
            z = x @ w
            return np.logaddexp(0, z).sum()-z[y].sum()+.5*(w@w), x.T@(expit(z)-y)+w
        fit = minimize(loss, np.zeros(len(NAMES)), jac=True, method='L-BFGS-B', options={'maxiter':500})
        assert fit.success, fit.message
        weights = fit.x
    return res, weights


def main():
    banks = E.build_banks()
    with TemporaryDirectory() as tmp:
        r3, w = fold(Path(tmp), 3, banks, None)
    r4, _ = fold(B.OUT, 4, banks, w)
    actual = r4['baseline'] if w is None else r4['reranked']
    result = dict(classes=B.K, baseline='csls', readouts=NAMES, seeds=B.SEEDS,
                  protocol='R3: train R1-2, predict R3; R4: train R1-3, predict R4; verify saved MLP hashes. All trials, original order per mouse/session. No oracle masks. Union is a diagnostic upper bound, not a predictor.',
                  candidates='Protect CSLS top24; remaining 20 top1 votes; sort by votes desc, CSLS desc, ID asc; deduplicate and fill to 32 from baseline. Compare to CSLS top32. All score ties use ascending ID.',
                  gate='Train iff pooled R3 net > 0. Never train on R4-only gains.',
                  reranker='Binary logistic, sum of logistic losses + 0.5 * squared L2 norm, no intercept; full-gallery row-z scores for each candidate only, no ID features; pooled R3 holdout predictions, lock before R4; final tie uses candidate order.',
                  trained=w is not None, locked_weights=None if w is None else w.tolist(), r3=r3, r4=r4,
                  acceptance=dict(scope='R4 single top1, not union/top32', selected='baseline' if w is None else 'reranked', unique_gt=462, hits_ge=572, observed=actual, passed=actual['unique']>462 and actual['hits']>=572))
    text = json.dumps(result, indent=2, allow_nan=False)
    (E.HERE/'R4_FROZEN21_CANDIDATES.json').write_text(text+'\n', encoding='utf-8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
