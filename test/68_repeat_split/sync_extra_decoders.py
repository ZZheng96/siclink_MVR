"""Continue the frozen synchronous oracle with extra closed-form / cheap readouts.

Starts from the residual IDs of SYNC_BRAIN_GLYPH.json (truth-assisted synchronous
oracle) and alternates the original thirteen readouts with new ones until a whole
round removes nothing.  Every readout is fitted on repeats 1..r-1 only and scored
independently over the full 2748-class gallery; labels are first used for
evaluation / truth-assisted removal, never for fitting or score selection.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as torchF

import sync_brain_glyph as G
import sync_response_templates as T
import sync_oracle as O

B, F = T.B, T.F
HERE = F.HERE

OLD = (*G.NEW, *T.NEW, *O.ORDER)  # thirteen existing readouts
WAVE1 = ('ridge_dct32', 'ridge_pix16', 'ridge_stroke', 'corr25', 'corr10',
         'lda50', 'softmax25', 'ridge_tsum50')
WAVE2 = ('binar50', 'eventstats25', 'pca_centroid50')
WAVE3 = ('corr50', 'binar25', 'pca_centroid25', 'lda25')
WAVES = {1: WAVE1, 2: WAVE2, 3: WAVE3}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def log_counts(m, s, r, width):
    return np.log1p(F.counts(m, s, r, width).astype(np.float64))


def labels_concat(m, s, r):
    return np.concatenate([F.labels(m, s, t) for t in range(1, r)])


def std_train(m, s, r, width):
    """Concatenated log1p training rows for one session, standardized."""
    x = np.concatenate([log_counts(m, s, t, width) for t in range(1, r)])
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.
    return (x - mu) / sd, labels_concat(m, s, r), mu, sd


def unit(a):
    return G.unit(np.asarray(a, dtype=np.float64))


def center_unit(a):
    a = a - a.mean(1, keepdims=True)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(n > 0, n, 1.)


# --------------------------------------------------------------------------- #
# public glyph banks (fixed, no labels involved)
# --------------------------------------------------------------------------- #
def stroke_hist(ink, cells=4, bins=8):
    """Gradient-orientation histogram on a 4x4 grid plus cell ink density."""
    gy, gx = np.gradient(ink, axis=(1, 2))
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % (2 * np.pi)
    bucket = np.minimum((ang * (bins / (2 * np.pi))).astype(np.int64), bins - 1)
    step = ink.shape[1] // cells
    parts = []
    for i in range(cells):
        for j in range(cells):
            sl = (slice(None), slice(i * step, (i + 1) * step), slice(j * step, (j + 1) * step))
            m_c, b_c = mag[sl], bucket[sl]
            parts.append(np.stack([(m_c * (b_c == k)).sum((1, 2)) for k in range(bins)], 1))
            parts.append(ink[sl].mean((1, 2))[:, None])
    return np.concatenate(parts, axis=1)


def build_banks():
    images = B.IMAGES
    ink = 1. - images
    d16 = B.dct_encode(images, 16).astype(np.float64)
    d16 -= d16.mean(0, keepdims=True)
    hv = np.concatenate([ink.sum(2), ink.sum(1)], axis=1).astype(np.float64)
    hv -= hv.mean(0, keepdims=True)
    d32 = B.dct_encode(images, 32).astype(np.float64)
    d32 -= d32.mean(0, keepdims=True)
    pix = images.reshape(len(images), 16, 4, 16, 4).mean((2, 4)).reshape(len(images), -1).astype(np.float64)
    pix -= pix.mean(0, keepdims=True)
    stroke = stroke_hist(ink).astype(np.float64)
    stroke -= stroke.mean(0, keepdims=True)
    return dict(d16=d16, hv=hv, d32=d32, pix16=pix, stroke=stroke)


# --------------------------------------------------------------------------- #
# ridge readouts
# --------------------------------------------------------------------------- #
def ridge_fit(m, r, feat, targets):
    gram, rhs, norm = 0, 0, {}
    for s in B.SESSIONS:
        x = np.concatenate([feat(m, s, t) for t in range(1, r)])
        y = labels_concat(m, s, r)
        mu, sd = x.mean(0), x.std(0)
        sd[sd == 0] = 1.
        norm[s] = (mu, sd)
        x = (x - mu) / sd
        gram = gram + x.T @ x
        rhs = rhs + x.T @ targets[y]
    alpha = .1 * np.mean(np.diag(gram))
    gram.flat[::len(gram) + 1] += alpha
    return np.linalg.solve(gram, rhs), norm, float(alpha)


def ridge_score(W, norm, feat, bank, m, s, r):
    mu, sd = norm[s]
    x = (feat(m, s, r) - mu) / sd
    return unit(x @ W) @ bank.T


def feat50(m, s, t):
    return log_counts(m, s, t, 50)


def glyph_scorers(m, r, banks):
    """Ridge on 50 ms log1p counts with three different glyph targets."""
    targets = np.concatenate([banks['d32'], banks['pix16'], banks['stroke']], axis=1)
    w, norm, _ = ridge_fit(m, r, feat50, targets)
    cut = np.cumsum([0, banks['d32'].shape[1], banks['pix16'].shape[1], banks['stroke'].shape[1]])
    out = {}
    for k, a, b, bank in zip(('ridge_dct32', 'ridge_pix16', 'ridge_stroke'),
                             cut[:-1], cut[1:], (banks['d32'], banks['pix16'], banks['stroke'])):
        W, bu = w[:, a:b], unit(bank)
        out[k] = lambda s, W=W, bu=bu, norm=norm: ridge_score(W, norm, feat50, bu, m, s, r)
    return out


def tsum_scorer(m, r, bank):
    """Within-bin standardized temporal mean / variance / maximum, then ridge."""
    stats = {}
    for s in B.SESSIONS:
        x = np.concatenate([log_counts(m, s, t, 50) for t in range(1, r)])
        mu, sd = x.mean(0), x.std(0)
        sd[sd == 0] = 1.
        stats[s] = (mu, sd)

    def feat(mm, s, t):
        mu, sd = stats[s]
        z = ((log_counts(mm, s, t, 50) - mu) / sd).reshape(B.K, B.UNITS[mm], 5)
        return np.concatenate([z.mean(2), z.var(2), z.max(2)], axis=1)

    w, norm, _ = ridge_fit(m, r, feat, bank)
    bu = unit(bank)
    return lambda s: ridge_score(w, norm, feat, bu, m, s, r)


# --------------------------------------------------------------------------- #
# cross-day template correlation readouts
# --------------------------------------------------------------------------- #
def corr_scorer(m, r, width):
    means, norm = [], {}
    for s in B.SESSIONS:
        z, y, mu, sd = std_train(m, s, r, width)
        norm[s] = (mu, sd)
        means.append(z[np.argsort(y)].reshape(B.K, r - 1, -1).mean(1))
    bank = center_unit(np.mean(means, axis=0))
    return lambda s: center_unit((log_counts(m, s, r, width) - norm[s][0]) / norm[s][1]) @ bank.T


def binar_scorer(m, r, width):
    means = []
    for s in B.SESSIONS:
        y = labels_concat(m, s, r)
        b = np.concatenate([(F.counts(m, s, t, width) > 0).astype(np.float64)
                            for t in range(1, r)])
        means.append(b[np.argsort(y)].reshape(B.K, r - 1, -1).mean(1))
    bank = center_unit(np.mean(means, axis=0))
    return lambda s: center_unit((F.counts(m, s, r, width) > 0).astype(np.float64)) @ bank.T


# --------------------------------------------------------------------------- #
# diagonal Gaussian readouts (pooled variance + shrinkage)
# --------------------------------------------------------------------------- #
def diag_solve(z, y, shrink):
    per = len(z) // B.K
    xs = z[np.argsort(y)].reshape(B.K, per, -1)
    mu_c = xs.mean(1)
    var = ((xs - mu_c[:, None, :]) ** 2).mean((0, 1))
    var = (1 - shrink) * var + shrink * float(var.mean())
    scale = 1. / np.sqrt(var)
    return scale, mu_c * scale


def diag_score(scale, mus, z):
    xs = z * scale
    return xs @ mus.T - .5 * (mus ** 2).sum(1)


def lda_scorer(m, r, width, shrink):
    zs, ys, norm = [], [], {}
    for s in B.SESSIONS:
        z, y, mu, sd = std_train(m, s, r, width)
        zs.append(z)
        ys.append(y)
        norm[s] = (mu, sd)
    scale, mus = diag_solve(np.concatenate(zs), np.concatenate(ys), shrink)
    return lambda s: diag_score(scale, mus,
                                (log_counts(m, s, r, width) - norm[s][0]) / norm[s][1])


def event_feature(raw, units, nbins):
    tot = raw.reshape(len(raw), units, nbins).sum(1)
    return np.concatenate([np.log1p(tot),
                           tot / np.maximum(tot.sum(1, keepdims=True), 1e-6),
                           tot / np.maximum(tot.max(1, keepdims=True), 1e-6)], axis=1)


def event_scorer(m, r, width, shrink=.2):
    units, nbins = B.UNITS[m], 250 // width
    zs, ys, norm = [], [], {}
    for s in B.SESSIONS:
        f = np.concatenate([event_feature(F.counts(m, s, t, width).astype(np.float64), units, nbins)
                            for t in range(1, r)])
        mu, sd = f.mean(0), f.std(0)
        sd[sd == 0] = 1.
        zs.append((f - mu) / sd)
        ys.append(labels_concat(m, s, r))
        norm[s] = (mu, sd)
    scale, mus = diag_solve(np.concatenate(zs), np.concatenate(ys), shrink)

    def score(s):
        f = event_feature(F.counts(m, s, r, width).astype(np.float64), units, nbins)
        return diag_score(scale, mus, (f - norm[s][0]) / norm[s][1])
    return score


# --------------------------------------------------------------------------- #
# PCA-whitened nearest centroid
# --------------------------------------------------------------------------- #
def pca_scorer(m, r, width, rank):
    zs, ys, norm = [], [], {}
    for s in B.SESSIONS:
        z, y, mu, sd = std_train(m, s, r, width)
        zs.append(z)
        ys.append(y)
        norm[s] = (mu, sd)
    x = np.concatenate(zs)
    y = np.concatenate(ys)
    mean = x.mean(0)
    xc = x - mean
    cov = xc.T @ xc / len(xc)
    values, vectors = np.linalg.eigh(cov)
    idx = np.argsort(values)[::-1][:rank]
    W = vectors[:, idx] / np.sqrt(np.maximum(values[idx], 1e-8))
    per = len(y) // B.K
    mu_c = (xc @ W)[np.argsort(y)].reshape(B.K, per, rank).mean(1)

    def score(s):
        z = (log_counts(m, s, r, width) - norm[s][0]) / norm[s][1]
        xw = (z - mean) @ W
        return 2 * xw @ mu_c.T - (mu_c ** 2).sum(1)
    return score


# --------------------------------------------------------------------------- #
# light linear softmax (different weight decay from the production MLP)
# --------------------------------------------------------------------------- #
def softmax_scorer(m, r, width=25, epochs=12, wd=1e-2, lr=1e-3):
    xs, ys, norm = [], [], {}
    for s in B.SESSIONS:
        z, y, mu, sd = std_train(m, s, r, width)
        xs.append(torch.from_numpy(z.astype(np.float32)))
        ys.append(torch.from_numpy(y.astype(np.int64)))
        norm[s] = (mu, sd)
    x, y = torch.cat(xs), torch.cat(ys)
    dev = F.H.DEVICE
    torch.manual_seed(42)
    np.random.seed(42)
    model = torch.nn.Linear(x.shape[1], B.K).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator().manual_seed(42)
    for _ in range(epochs):
        for ix in torch.randperm(len(y), generator=gen).split(512):
            loss = torchF.cross_entropy(model(x[ix].to(dev)), y[ix].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def score(s):
        z = (log_counts(m, s, r, width) - norm[s][0]) / norm[s][1]
        with torch.no_grad():
            logits = model(torch.from_numpy(z.astype(np.float32)).to(dev))
        return logits.cpu().numpy().astype(np.float64)
    return score


def wave_scorers(wave, m, r, banks):
    if wave == 1:
        out = glyph_scorers(m, r, banks)
        out['corr25'] = corr_scorer(m, r, 25)
        out['corr10'] = corr_scorer(m, r, 10)
        out['lda50'] = lda_scorer(m, r, 50, .2)
        out['softmax25'] = softmax_scorer(m, r, 25)
        out['ridge_tsum50'] = tsum_scorer(m, r, banks['d32'])
        return out
    if wave == 2:
        return {'binar50': binar_scorer(m, r, 50),
                'eventstats25': event_scorer(m, r, 25),
                'pca_centroid50': pca_scorer(m, r, 50, 128)}
    return {'corr50': corr_scorer(m, r, 50),
            'binar25': binar_scorer(m, r, 25),
            'pca_centroid25': pca_scorer(m, r, 25, 128),
            'lda25': lda_scorer(m, r, 25, .05)}


# --------------------------------------------------------------------------- #
# frozen scores, truth-assisted synchronous peeling
# --------------------------------------------------------------------------- #
def verify_r4(out, m):
    for name, _, _ in F.RUNS:
        manifest = json.loads((out / name / 'bench_manifest.json').read_text())
        assert manifest['protocol'] == 'repeat1-3 train, repeat4 evaluate'
        for seed in B.SEEDS:
            assert B.sha(B.checkpoint_path(out / name, 'task3', m, seed)) == manifest['checkpoints'][f'{m}/{seed}']


def old_blocks(out, r, ids, banks):
    targets = np.concatenate([banks['d16'], banks['hv']], axis=1)
    blocks = {k: [] for k in OLD}
    new_banks = {'ridge_dct16': unit(banks['d16']), 'ridge_hv': unit(banks['hv'])}
    for m in B.MICE:
        if r == 3:
            F.train_dev(out, m)
        else:
            verify_r4(out, m)
        w, norm, _ = G.fit(m, r, targets)
        fits = {k: T.fit(m, r, k) for k in T.NEW}
        for s in B.SESSIONS:
            y = F.labels(m, s, r)
            rows = np.argsort(y)
            scores = O.score(out, m, s, r, banks['d16'])
            for k in T.NEW:
                p, sd = fits[k][s]
                x = T.features(m, s, r, k) / sd
                scores[k] = 2 * (x @ p.T) - (p ** 2).sum(1) - (x ** 2).sum(1)[:, None]
            mu, sd = norm[s]
            preds = ((T.features(m, s, r, 'bins50') - mu) / sd) @ w
            for k in (*T.NEW, *O.ORDER):
                blocks[k].append(scores.pop(k)[np.ix_(rows[ids], ids)])
            for k, p in zip(G.NEW, np.split(preds, [banks['d16'].shape[1]], axis=1)):
                blocks[k].append((unit(p) @ new_banks[k].T)[np.ix_(rows[ids], ids)])
            del scores, preds
        del w, fits, norm
        print(f'old blocks R{r} {m}', flush=True)
    return blocks


def build_wave(wave, r, ids, banks):
    names = WAVES[wave]
    blocks, full = {k: [] for k in names}, {k: [] for k in names}
    for m in B.MICE:
        scorers = wave_scorers(wave, m, r, banks)
        for s in B.SESSIONS:
            y = F.labels(m, s, r)
            rows = np.argsort(y)
            for k in names:
                a = scorers[k](s)
                assert a.shape == (B.K, B.K), (wave, k, a.shape)
                full[k].append((a.argmax(1) == y)[rows])
                blocks[k].append(a[np.ix_(rows[ids], ids)])
                del a
        print(f'wave{wave} R{r} {m}', flush=True)
    return blocks, {k: F.totals(np.stack(full[k])) for k in names}


def probe(blocks, live, names):
    out = {}
    for k in names:
        h = O.hits(blocks[k], live)
        out[k] = dict(residual_unique=int(h.any(0).sum()), residual_hits=int(h.sum()))
    return out


def peel_all(order, blocks, live):
    history = []
    while len(live):
        before, detail = len(live), {}
        for k in order:
            live, detail[k] = O.peel(blocks[k], live)
            if not len(live):
                break
        history.append(dict(method_steps=detail, removed=before - len(live), remaining=len(live)))
        if len(live) == before:
            break
    return history, live


def run_fold(r, out, ref, banks):
    ids = np.array(ref['remaining_ids'], dtype=np.int64)
    assert len(ids) == ref['remaining'] == {3: 1068, 4: 646}[r], (len(ids), ref['remaining'])
    assert np.array_equal(np.unique(ids), ids)
    record = dict(start=len(ids), start_ids=ids.tolist(), waves={})
    blocks = old_blocks(out, r, ids, banks)
    live = np.arange(len(ids))
    record['old_residual_probe'] = probe(blocks, live, OLD)
    order = list(OLD)
    for wave in (1, 2, 3):
        wave_blocks, wave_full = build_wave(wave, r, ids, banks)
        blocks.update(wave_blocks)
        names = WAVES[wave]
        order.extend(names)
        rec = dict(start=len(live),
                   readouts={k: dict(full=wave_full[k], **probe(blocks, live, (k,))[k])
                             for k in names})
        history, live = peel_all(order, blocks, live)
        rec.update(cycles=history, remaining=len(live), remaining_ids=ids[live].tolist())
        record['waves'][wave] = rec
        print(f'R{r} wave{wave}: start {rec["start"]} -> {len(live)}', flush=True)
        if not len(live):
            break
    record['final_remaining'] = len(live)
    record['final_remaining_ids'] = ids[live].tolist()
    record['stalled'] = bool(len(live))
    return record


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def summary_lines(result):
    lines = [
        '# 额外廉价读出：同步 oracle 联合诊断（R4 / R3）', '',
        '> **这是依赖真值的 oracle 联合诊断，不等于可直接部署的单次准确率。**',
        '> 每个评估 trial 的分数在见到标签前已冻结；每轮用真实标签取“该字在 8 个鼠/日内'
        '任一被认出”的并集来移除候选，再用剩余候选继续预测。',
        '> 训练只使用 repeat 1..r-1，不读取评估标签做拟合或选参；各读出独立全库打分，融合权重未调。',
        '',
        '| 折 | 起点 | wave1 后 | wave2 后 | wave3 后 | 最终剩余 |',
        '|---|---:|---:|---:|---:|---:|',
    ]

    def cell(v):
        return '跳过' if v is None else str(v)

    for key, label in (('r4', 'R4'), ('r3', 'R3')):
        rec = result[key]
        vals = []
        for w in (1, 2, 3):
            entry = rec['waves'].get(w)
            vals.append(None if entry is None else entry['remaining'])
        lines.append(f"| {label} | {rec['start']} | " + ' | '.join(cell(v) for v in vals)
                     + f" | {rec['final_remaining']} |")
    lines.append('')

    for key, label in (('r4', 'R4'), ('r3', 'R3')):
        rec = result[key]
        lines += [f'## {label}', '', '| 波次 | 读出 | 全库覆盖 | 全库命中 | 本轮起点残余覆盖 | 本轮起点残余命中 |',
                  '|---|---|---:|---:|---:|---:|']
        for w in (1, 2, 3):
            entry = rec['waves'].get(w)
            if entry is None:
                continue
            for name, stat in entry['readouts'].items():
                lines.append(f"| {w} | `{name}` | {stat['full']['unique']} | {stat['full']['hits']} | "
                             f"{stat['residual_unique']} | {stat['residual_hits']} |")
        lines.append('')
        for w in (1, 2, 3):
            entry = rec['waves'].get(w)
            if entry is None:
                continue
            rounds = [(c['removed'], c['remaining']) for c in entry['cycles']]
            lines.append(f"- wave{w}: 起点 {entry['start']}，逐轮 (新增, 剩余) = {rounds}，"
                         f"结束 {entry['remaining']}。")
        left = rec['final_remaining_ids']
        lines.append(f"- 最终剩余 {rec['final_remaining']} 个字"
                     + (f"：`{left[:40]}`" + (' ...' if len(left) > 40 else '') if left else '。'))
        lines.append('')
    return lines


def main():
    ref = json.loads((HERE / 'SYNC_BRAIN_GLYPH.json').read_text())
    assert tuple(ref['order']) == (*G.NEW, *T.NEW, *O.ORDER)
    banks = build_banks()
    result = dict(
        source='SYNC_BRAIN_GLYPH.json',
        classes=B.K,
        trials=B.K * len(B.MICE) * len(B.SESSIONS),
        old_order=list(OLD),
        waves={str(k): list(v) for k, v in WAVES.items()},
        protocol=('Start from the recorded residual IDs of SYNC_BRAIN_GLYPH.json observed arm. '
                  'Thirteen old readouts are recomputed with the same frozen-score recipe, then extra '
                  'readouts are added wave by wave. All fits use repeats 1..r-1 only; full-gallery scores '
                  'are frozen before label alignment; each cycle removes the union of correctly recognised '
                  'residual classes across the eight mouse-days, then re-peels on the remaining candidates. '
                  'No fusion weights, no per-readout tuning, no re-scoring after masking.'),
        caveat=('Truth-assisted oracle diagnostic. The reported coverage is the union over eight '
                'mouse-days with known labels and sequential candidate removal; it is not a deployable '
                'single-shot accuracy.'),
        readout_protocol=('ridge_dct32 / ridge_pix16 / ridge_stroke: 50 ms log1p counts -> ridge '
                          '(alpha=.1*mean(diag(X.T@X))) to centred DCT32, 16x16 pooled brightness, or a '
                          'fixed 4x4-cell 8-orientation gradient histogram; cosine full gallery. '
                          'corr25 / corr10: per-session training mean/std, cross-day per-character '
                          'template, Pearson correlation. lda50: pooled diagonal variance with 0.2 '
                          'shrinkage. softmax25: light linear softmax (12 epochs, AdamW, weight decay '
                          '1e-2). ridge_tsum50: per-bin/unit standardisation then mean+variance+maximum '
                          'over five 50 ms bins -> ridge to DCT32. Wave2: binned spike-presence '
                          'templates, per-time-block event statistics with diagonal Gaussian, PCA-'
                          'whitened nearest centroid (128 dimensions). Wave3: 50 ms correlation, 25 ms '
                          'spike-presence, 25 ms PCA centroid, 25 ms LDA with 0.05 shrinkage.'),
    )
    result['r4'] = run_fold(4, B.OUT, ref['r4']['observed'], banks)
    with TemporaryDirectory() as tmp:
        result['r3'] = run_fold(3, Path(tmp), ref['r3']['observed'], banks)
    text = json.dumps(result, indent=2, allow_nan=False)
    (HERE / 'SYNC_EXTRA_DECODERS.json').write_text(text + '\n', encoding='utf-8')
    lines = summary_lines(result)
    (HERE / 'SYNC_EXTRA_DECODERS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
