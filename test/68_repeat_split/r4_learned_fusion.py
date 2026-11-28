"""Learned fusion / recall-preserving rerank / transductive assignment on R4.

A. full-gallery linear stacking of the 21 frozen readouts (per-row z scores).
B. rerank inside the baseline CSLS top 32, so candidate recall cannot drop.
C. transductive one-to-one greedy on the fused probabilities (uses the other
   unlabelled trials of the same block; threshold locked on R3).

Every weight and threshold is fitted/selected on R3 only (models trained on
repeats 1-2).  R4 uses the saved repeat 1-3 checkpoints and is scored once with
the locked values.  Truth labels of R4 are used for nothing else.
"""
import heapq
import json
import os
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

import sync_extra_decoders as E

B, F, G, T = E.B, E.F, E.G, E.T
HERE = E.HERE
DEV = HERE.parents[1] / 'siclink_research_tmp' / 'r3_dev_models'
STAGE = HERE / 'R3_FUSION_STAGE.json'
NAMES = ('csls', *(k for k in E.OLD if k != 'csls'), *E.WAVE1)
assert len(set(NAMES)) == len(NAMES) == 21
TOP = 32           # baseline candidate budget for method B
L2 = 1e-2
TAUS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01)
DELTAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
BASE_UNIQUE, BASE_HITS = 462, 572


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #
def build_scores(out, m, s, r, banks, wg, norm, fits, wave):
    scores = E.O.score(out, m, s, r, banks['d16'])
    for k in T.NEW:
        p, sd = fits[k][s]
        x = T.features(m, s, r, k) / sd
        scores[k] = 2 * (x @ p.T) - (p ** 2).sum(1) - (x ** 2).sum(1)[:, None]
    mu, sd = norm[s]
    pred = ((T.features(m, s, r, 'bins50') - mu) / sd) @ wg
    for k, tag, v in zip(G.NEW, ('d16', 'hv'), np.split(pred, [banks['d16'].shape[1]], axis=1)):
        scores[k] = E.unit(v) @ E.unit(banks[tag]).T
    scores.update({k: fun(s) for k, fun in wave.items()})
    assert set(scores) == set(NAMES)
    for a in scores.values():
        assert a.shape == (B.K, B.K) and np.isfinite(a).all()
    return scores


def rowz(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean(1, keepdims=True)) / np.maximum(a.std(1, keepdims=True), 1e-6)


def gather(scores, cols):
    out = np.empty(cols.shape + (len(NAMES),), dtype=np.float32)
    for i, k in enumerate(NAMES):
        out[:, :, i] = np.take_along_axis(rowz(scores[k]), cols, axis=1)
    return out


def fused_full(scores, w):
    """Weighted sum of the per-row z scores, all 2748 gallery columns."""
    out = np.zeros((B.K, B.K), dtype=np.float32)
    for wk, k in zip(w, NAMES):
        a = scores[k]
        mu = a.mean(1, keepdims=True)
        sd = np.maximum(a.std(1, keepdims=True), 1e-6)
        for j in range(0, B.K, 256):
            out[:, j:j + 256] += np.float32(wk) * ((a[:, j:j + 256] - mu) / sd).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# weight fitting (R3 only)
# --------------------------------------------------------------------------- #
def zpath(r, m, s):
    return DEV / f'z{r}' / f'{m}s{s}.npy'


def load_z(r):
    """(mouse, session, z stack float32, labels) for every block of one fold."""
    out = []
    for m in B.MICE:
        for s in B.SESSIONS:
            out.append((m, s, np.load(zpath(r, m, s)), F.labels(m, s, r)))
    return out


def ce_fit(zs, l2=L2, maxiter=100, init=None):
    """Multinomial logistic regression over the full gallery (truth = its own column).

    The negative set is all 2748 classes of every training trial, so no candidate
    pool bias enters the weights.  The problem is convex; L-BFGS-B from w=csls.
    """
    w0 = np.zeros(len(NAMES))
    w0[0] = 1.0
    if init is not None:
        w0 = np.asarray(init, dtype=np.float64)

    def obj(w):
        loss, grad, n = 0.0, np.zeros_like(w), 0
        w32 = w.astype(np.float32)
        for _, _, z, y in zs:
            f = np.tensordot(w32, z, axes=(0, 0))
            mx = f.max(1, keepdims=True)
            e = np.exp(f - mx)
            ssum = e.sum(1, keepdims=True)
            idx = np.arange(len(y))
            loss += float(np.sum((mx[:, 0] + np.log(ssum[:, 0])) - f[idx, y], dtype=np.float64))
            e /= ssum
            e[idx, y] -= 1.0
            grad += np.tensordot(z, e, axes=([1, 2], [0, 1])).astype(np.float64)
            n += len(y)
        loss = loss / n + l2 * float(w @ w)
        grad = grad / n + 2 * l2 * w
        return loss, grad

    res = minimize(obj, w0, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8))
    return res.x, dict(success=bool(res.success), iterations=int(res.nit),
                       loss=float(res.fun), message=str(res.message))


def hit_table(zs, w):
    """Full-gallery top1 hits per mouse and over the whole fold, from cached z."""
    w32 = np.asarray(w, dtype=np.float32)
    out = {}
    for m in B.MICE:
        hits = base = 0
        for mm, _, z, y in zs:
            if mm != m:
                continue
            hits += int((np.tensordot(w32, z, axes=(0, 0)).argmax(1) == y).sum())
            base += int((z[0].argmax(1) == y).sum())
        out[m] = dict(hits=hits, baseline_hits=base)
    out['all'] = dict(hits=sum(v['hits'] for k, v in out.items()),
                      baseline_hits=sum(v['baseline_hits'] for k, v in out.items()))
    return out



# --------------------------------------------------------------------------- #
# transductive assignment
# --------------------------------------------------------------------------- #
def log_probs(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z / max(float(z.std()), 1e-6)
    mx = z.max(1, keepdims=True)
    return (z - mx - np.log(np.exp(z - mx).sum(1, keepdims=True))).astype(np.float32)


def greedy_chain(logp):
    """All one-to-one fixes in decreasing confidence; returns (rows, cols, logp)."""
    n = len(logp)
    rank = np.argsort(-logp, axis=1, kind='stable')
    ptr = np.zeros(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    fixed = np.full(n, -1, dtype=np.int64)
    heap = [(-float(logp[i, rank[i, 0]]), i) for i in range(n)]
    heapq.heapify(heap)
    rows, cols, probs = [], [], []
    while heap:
        negp, i = heapq.heappop(heap)
        if fixed[i] >= 0:
            continue
        p = ptr[i]
        while p < n and not alive[rank[i, p]]:
            p += 1
        ptr[i] = p
        if p >= n:
            continue
        val = -float(logp[i, rank[i, p]])
        if val > negp + 1e-9:          # stale entry: refresh and re-queue
            heapq.heappush(heap, (val, i))
            continue
        j = int(rank[i, p])
        fixed[i] = j
        alive[j] = False
        rows.append(i)
        cols.append(j)
        probs.append(-val)
        q = p + 1
        while q < n and not alive[rank[i, q]]:
            q += 1
        ptr[i] = q
        if q < n:
            heapq.heappush(heap, (-float(logp[i, rank[i, q]]), i))
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(probs)


def chain_preds(logp, rows, cols, probs, tau):
    n = len(logp)
    m = int((probs >= np.log(tau)).sum())
    alive = np.ones(n, dtype=bool)
    pred = np.full(n, -1, dtype=np.int64)
    alive[cols[:m]] = False
    pred[rows[:m]] = cols[:m]
    left = np.where(pred < 0)[0]
    if len(left):
        keep = np.where(alive)[0]
        pred[left] = keep[logp[left][:, alive].argmax(1)]
    return pred, m


def hungarian(logp):
    r, c = linear_sum_assignment(-logp)
    out = np.empty(len(r), dtype=np.int64)
    out[r] = c
    return out


def dup_assign(fused, base):
    """One-to-one reassignment of rows that share their baseline glyph.

    Rows with a unique baseline prediction keep it.  Every contested row is
    reassigned inside the contest: the free classes are exactly as many as the
    contested rows, so the assignment stays a permutation on that subset.
    """
    n = len(base)
    cnt = np.bincount(base, minlength=n)
    dup = np.where(cnt[base] > 1)[0]
    pred = base.copy()
    if not len(dup):
        return pred
    stable = np.setdiff1d(np.arange(n), dup)
    avail = np.ones(n, dtype=bool)
    avail[base[stable]] = False
    cols = np.where(avail)[0]
    rows, cc = linear_sum_assignment(-fused[np.ix_(dup, cols)])
    pred[dup[rows]] = cols[cc]
    return pred


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def detail(blocks, key):
    """Hits / unique glyphs over the pooled blocks, plus gains vs baseline top1."""
    bh = nh = 0
    bset, nset = set(), set()
    for b in blocks:
        y, pb, pn = b['y'], b['base_pred'], b[key]
        bh += int((pb == y).sum())
        nh += int((pn == y).sum())
        bset |= set(y[pb == y].tolist())
        nset |= set(y[pn == y].tolist())
    return dict(hits=nh, unique=len(nset), baseline_hits=bh, baseline_unique=len(bset),
                gained=len(nset - bset), lost=len(bset - nset),
                passed=bool(len(nset) > BASE_UNIQUE and nh >= BASE_HITS))


def summarise(blocks, keys):
    out = {'all': {k: detail(blocks, k) for k in keys}}
    for m in B.MICE:
        part = [b for b in blocks if b['m'] == m]
        out[m] = {k: detail(part, k) for k in keys}
    return out


def selected(blocks, variants):
    """Pick the tau with the best R3 (unique, hits) per variant and return it."""
    best = {}
    for v in variants:
        scores = []
        for t in TAUS:
            key = f'c_{v}_{t}'
            d = detail(blocks, key)
            scores.append((d['unique'], d['hits'], -t, t))
        scores.sort(reverse=True)
        best[v] = dict(tau=scores[0][3], unique=scores[0][0], hits=scores[0][1],
                       table=[dict(tau=s[3], unique=s[0], hits=s[1]) for s in scores])
    return best


def choose_gate(blocks, variants):
    """Pick the safest one-to-one variant on R3.

    Requirement: hits must not fall below baseline and unique must rise above it.
    Among the valid variants the one with the most R3 hits wins (tie-break unique),
    because the reassignment is transductive and R3->R4 transfer is the main risk.
    """
    base = detail(blocks, 'base_pred')
    rows = [dict(name=v, **detail(blocks, v)) for v in variants]
    ok = [r for r in rows if r['hits'] >= base['hits'] and r['unique'] > base['unique']]
    ok.sort(key=lambda r: (r['hits'], r['unique']), reverse=True)
    return dict(baseline_hits=base['hits'], baseline_unique=base['unique'],
                picked=ok[0] if ok else None, table=rows)


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def prepare(out, r, banks, m, skip_train=False):
    if r == 3:
        if not skip_train:
            F.train_dev(out, m)
            (out / f'ready_{m}.json').write_text('{}')
    else:
        E.verify_r4(out, m)
    targets = np.concatenate([banks['d16'], banks['hv']], axis=1)
    wg, norm, _ = G.fit(m, r, targets)
    fits = {k: T.fit(m, r, k) for k in T.NEW}
    wave = E.wave_scorers(1, m, r, banks)
    return wg, norm, fits, wave


def first_pass(r, out, banks, models):
    """Baseline top32 with its z features; R3 also caches the full z stack."""
    blocks = []
    zdir = DEV / f'z{r}'
    if r == 3:
        zdir.mkdir(parents=True, exist_ok=True)
    for m in B.MICE:
        wg, norm, fits, wave = models[m]
        for s in B.SESSIONS:
            scores = build_scores(out, m, s, r, banks, wg, norm, fits, wave)
            y = F.labels(m, s, r)
            base = scores['csls']
            top = np.argsort(-base, axis=1, kind='stable')[:, :TOP]
            b = dict(m=m, s=s, y=y, top=top, base_pred=base.argmax(1), x32=gather(scores, top))
            if r == 3:
                path = zpath(3, m, s)
                if not path.exists():
                    stack = np.empty((len(NAMES), B.K, B.K), dtype=np.float32)
                    for i, k in enumerate(NAMES):
                        stack[i] = rowz(scores[k])
                    np.save(path, stack)
                    del stack
            blocks.append(b)
            del scores
            print(f'  R{r} first pass {m} s{s}', flush=True)
    return blocks


def second_pass(r, out, banks, models, blocks, w, taus):
    """Full-gallery fusion, candidate rerank and transductive assignment.

    taus maps variant -> tuple of thresholds to record.
    """
    w32 = np.asarray(w, dtype=np.float32)
    ii = np.arange(B.K)
    for b in blocks:
        m, s = b['m'], b['s']
        wg, norm, fits, wave = models[m]
        scores = build_scores(out, m, s, r, banks, wg, norm, fits, wave)
        fused = fused_full(scores, w)
        b['pred_a'] = fused.argmax(1)
        b['pred_b'] = b['top'][ii, (b['x32'] @ w32).argmax(1)]
        for v, logits in (('fused', fused), ('csls', scores['csls'])):
            lp = log_probs(logits)
            rows, cols, probs = greedy_chain(lp)
            for t in taus[v]:
                b[f'c_{v}_{t}'], _ = chain_preds(lp, rows, cols, probs, t)
            b[f'h_{v}'] = hungarian(lp)
            b[f'chain_{v}'] = len(rows)
            del lp
        base, hh = b['base_pred'], b['h_fused']
        b['pred_dup'] = dup_assign(fused, base)
        margin = fused[ii, hh] - fused[ii, base]
        for d in DELTAS:
            b[f'pred_g_{d}'] = np.where(margin >= d, hh, base)
        del scores, fused
        print(f'  R{r} second pass {m} s{s}', flush=True)
    return blocks


def run_r3(banks):
    t0 = time.time()
    DEV.mkdir(parents=True, exist_ok=True)
    done = {m: (DEV / f'ready_{m}.json').exists() for m in B.MICE}
    print(f'R3 dev models cached {done}', flush=True)
    models = {m: prepare(DEV, 3, banks, m, skip_train=done[m]) for m in B.MICE}
    print(f'R3 models ready {time.time() - t0:.0f}s', flush=True)
    blocks = first_pass(3, DEV, banks, models)
    zs = load_z(3)
    print(f'R3 z cached {time.time() - t0:.0f}s', flush=True)
    w, fit_info = ce_fit(zs)
    print(f'R3 CE fit done {time.time() - t0:.0f}s w={np.round(w, 3).tolist()}', flush=True)
    # leave-one-mouse-out check: weights fitted on three mice, scored on the fourth
    loso = {}
    for m in B.MICE:
        tr = [t for t in zs if t[0] != m]
        te = [t for t in zs if t[0] == m]
        wm, _ = ce_fit(tr, maxiter=60)
        loso[m] = dict(table=hit_table(te, wm)['all'],
                       weights=[float(v) for v in wm])
        print(f'R3 LOSO {m} {hit_table(te, wm)["all"]} {time.time() - t0:.0f}s', flush=True)
    print(f'R3 in-sample {hit_table(zs, w)["all"]}', flush=True)
    second_pass(3, DEV, banks, models, blocks, w, {'fused': TAUS, 'csls': TAUS})
    picked = selected(blocks, ('fused', 'csls'))
    gate_variants = ('pred_dup', *(f'pred_g_{d}' for d in DELTAS))
    pick = choose_gate(blocks, gate_variants)
    print(f'R3 gate pick {pick["picked"]}', flush=True)
    keys = ('base_pred', 'pred_a', 'pred_b', 'h_fused', 'h_csls', *gate_variants)
    res = dict(weights=[float(v) for v in w], fit=fit_info, loso=loso,
               tau=picked, pick=pick, evaluation=summarise(blocks, keys),
               chains={f'{b["m"]}s{b["s"]}': {v: int(b[f'chain_{v}']) for v in ('fused', 'csls')}
                       for b in blocks},
               seconds=round(time.time() - t0, 1))
    return res, w, {v: picked[v]['tau'] for v in picked}


def run_r4(banks, w, taus, chosen):
    t0 = time.time()
    models = {m: prepare(B.OUT, 4, banks, m) for m in B.MICE}
    blocks = first_pass(4, B.OUT, banks, models)
    second_pass(4, B.OUT, banks, models, blocks, w,
                {'fused': (taus['fused'],), 'csls': (taus['csls'],)})
    keys = ('base_pred', 'pred_a', 'pred_b', 'h_fused', 'h_csls',
            f'c_fused_{taus["fused"]}', f'c_csls_{taus["csls"]}', chosen,
            'pred_dup', *(f'pred_g_{d}' for d in DELTAS))
    res = summarise(blocks, keys)
    res['hungarian'] = {'fused': detail(blocks, 'h_fused'), 'csls': detail(blocks, 'h_csls')}
    res['chains'] = {f'{b["m"]}s{b["s"]}': {v: int(b[f'chain_{v}']) for v in ('fused', 'csls')}
                     for b in blocks}
    res['seconds'] = round(time.time() - t0, 1)
    return res


def main(stage='all'):
    banks = E.build_banks()
    if STAGE.exists():
        st = json.loads(STAGE.read_text())
        r3, taus = st['r3'], st['taus']
        w = np.asarray(st['weights'], dtype=np.float64)
        print(f'R3 stage cached {STAGE.name}', flush=True)
    else:
        if stage == 'r4':
            raise SystemExit('R3 stage missing; run stage r3 first')
        r3, w, taus = run_r3(banks)
        STAGE.write_text(json.dumps(dict(weights=[float(v) for v in w], taus=taus, r3=r3),
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'R3 stage saved {STAGE.name}', flush=True)
    if stage == 'r3':
        return
    pick = r3.get('pick', {}).get('picked')
    chosen = pick['name'] if pick else 'base_pred'
    r4 = run_r4(banks, w, taus, chosen)

    methods = {'A_full_gallery_stacking': r4['all']['pred_a'],
               'B_top32_rerank': r4['all']['pred_b'],
               'C_greedy_fused': r4['all'][f'c_fused_{taus["fused"]}'],
               'C_greedy_csls': r4['all'][f'c_csls_{taus["csls"]}'],
               'C_hungarian_fused': r4['all']['h_fused'],
               'baseline': r4['all']['base_pred']}
    if chosen != 'base_pred':
        methods[f'D_locked_{chosen}'] = r4['all'][chosen]
    for d in DELTAS:
        methods[f'D_gated_{d}'] = r4['all'][f'pred_g_{d}']
    methods['D_dup'] = r4['all']['pred_dup']
    assert (r4['all']['base_pred']['hits'], r4['all']['base_pred']['unique']) == (BASE_HITS, BASE_UNIQUE), \
        r4['all']['base_pred']
    result = dict(
        classes=B.K, readouts=list(NAMES), seeds=B.SEEDS,
        protocol=('Fit and select only on repeat 3 (models trained on repeats 1-2 into a cached '
                  'scratch dir). Repeat 4 uses the saved repeat 1-3 checkpoints, verified by hash, '
                  'and is scored once with locked weights and thresholds. Pooled over the 8 '
                  'mouse-day blocks of each fold.'),
        fitting=('Weights: multinomial logistic regression over the full 2748-class gallery of '
                 'every R3 trial, features = row-z scores of the 21 readouts, L2=1e-2, L-BFGS-B '
                 'from a csls-only start; leave-one-mouse-out sanity check. Candidate rerank (B) '
                 'applies the same weights inside the CSLS top32.'),
        transductive_note=('C and D use the other unlabelled trials of the same block (one glyph '
                           'per block), so they are transductive unlike A and B. C fixes the most '
                           'confident one-to-one pairs; D reassigns only rows that share their '
                           'baseline glyph (dup) or accepts a Hungarian change only above a fused '
                           'margin delta.  On R3 the D variant is chosen as the valid one with the '
                           'highest hits (hits >= baseline and unique > baseline), i.e. the safest '
                           'option; its delta is then locked for R4.'),
        r3=r3, r4=r4, locked=dict(weights=[float(v) for v in w], tau=taus),
        acceptance=dict(baseline=dict(hits=BASE_HITS, unique=BASE_UNIQUE), methods=methods))
    text = json.dumps(result, indent=2, allow_nan=False)
    (HERE / 'R4_LEARNED_FUSION.json').write_text(text + '\n', encoding='utf-8')
    for name, d in methods.items():
        print(f'R4 {name:26s} hits {d["hits"]:5d} unique {d["unique"]:4d} '
              f'gain +{d["gained"]:4d} loss -{d["lost"]:4d} pass {d["passed"]}', flush=True)
    print(json.dumps(dict(r3=dict(weights=r3['weights'], loso=r3['loso'], tau=r3['tau'])), indent=2))


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
