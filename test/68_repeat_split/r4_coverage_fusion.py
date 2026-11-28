"""Full Task3 ensemble plus real half-window logits; select only on repeat 3."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import r4_halfwindow as H
import r4_benchmark as B
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = [('e53_10e7', 10, 7), ('e53_25e12', 25, 12), ('e53_50e9', 50, 9), ('v5_250e7', 250, 7)]
GRID = (0, .1, .25, .5)
Z = B.row_zscore


def path(m, s, r):
    return B.SPLIT / m / f'session{s}_repeat{r}.npz'


def counts(m, s, r, width):
    # No targets, sample IDs or trial alignment in the inference path.
    with np.load(path(m, s, r), allow_pickle=False) as f:
        event = SimpleNamespace(**{k: f[k] for k in ('spike_samples', 'unit_index', 'trial_ptr')},
                                n_units=B.UNITS[m], sample_rate_hz=30000, window_ms=250)
    return B.trial_count_features(event, width)


def labels(m, s, r):
    with np.load(path(m, s, r), allow_pickle=False) as f:
        y = f['target'].astype(np.int64)
        assert np.all(f['repeat_index'] == r) and np.array_equal(np.sort(y), np.arange(B.K))
    return y


def train_dev(out, m):
    for name, width, epochs in RUNS:
        blocks = []
        for s in B.SESSIONS:
            parts = [B.count_features(m, s, r, width) for r in (1, 2)]
            x, y = (np.concatenate(v) for v in zip(*parts))
            blocks.append(B.FeatureBlock(m, f'session{s}', 'train', 'task3', np.arange(len(y)),
                                         np.arange(len(y)+1), y, x, B.UNITS[m], width))
        dropout = .35 if m == 'mice2' and width != 250 else .2
        B.train_task(B.config_for(width, epochs, dropout, name), 'task3', out/name,
                     images=B.IMAGES, blocks=blocks, device=H.DEVICE)


def half_x(m, s, r):
    pairs = counts(m, s, r, 125).reshape(B.K, B.UNITS[m], 2)
    diff = pairs[:, :, 0] - pairs[:, :, 1]
    return np.concatenate([np.log1p(pairs.sum(2)), np.sign(diff)*np.log1p(abs(diff))], axis=1)


def half_fit(m, repeats):
    blocks = {(s, r): ([half_x(m, s, r)], labels(m, s, r)) for s in B.SESSIONS for r in repeats}
    x, y, scales = H.prepare(blocks, 0, repeats)
    return H.fit(x, y, .001).eval(), scales


def full_scores(out, m, s, r, centered):
    v5, mlp = [], []
    for name, width, _ in RUNS:
        root = out/name
        folder = root/'task3'/m
        raw = counts(m, s, r, width)
        with np.load(folder/f'session{s}_normalizer.npz') as f:
            x = B.apply_session_normalizer(raw, (f['mean'], f['scale']))
        logits = np.mean([B.predict_logits(B.load_classifier(B.checkpoint_path(root, 'task3', m, seed),
                         device=H.DEVICE), x, device=H.DEVICE) for seed in B.SEEDS], axis=0)
        if width == 250:
            return np.mean([Z(np.mean(v5, axis=0)), Z(np.mean(mlp, axis=0)), Z(logits)], axis=0)
        enc = B.response_scores(np.log1p(raw), np.load(folder/'encoding_prototypes.npy'), metric='mse', device=H.DEVICE)
        dct = B.softmax_dct_similarity(logits, centered, device=H.DEVICE)
        mlp.append(Z(logits))
        v5.append(Z((mlp[-1] + Z(enc) + Z(dct))/3))


def predict(out, m, s, r, centered, half, weights):
    base = full_scores(out, m, s, r, centered)
    model, scales = half
    mu, sd = scales[s]
    with torch.no_grad():
        h = Z(model(torch.as_tensor((half_x(m, s, r)-mu)/sd, device=H.DEVICE)).cpu().numpy())
    preds = {}
    for w in weights:
        a = (1-w)*base + w*h
        row = np.partition(a, -100, axis=1)[:, -100:].mean(1, keepdims=True)
        col = np.partition(a, -100, axis=0)[-100:].mean(0, keepdims=True)
        preds[w] = (2*a-row-col).argmax(1)
    return preds


def evaluate(out, r, weights, centered):
    hits = {w: [] for w in weights}
    for m in B.MICE:
        if r == 3:
            train_dev(out, m)
        else:
            for name, _, _ in RUNS:
                manifest = json.loads((out/name/'bench_manifest.json').read_text())
                assert manifest['protocol'] == 'repeat1-3 train, repeat4 evaluate'
                for seed in B.SEEDS:
                    assert B.sha(B.checkpoint_path(out/name, 'task3', m, seed)) == manifest['checkpoints'][f'{m}/{seed}']
        half = half_fit(m, range(1, r))
        for s in B.SESSIONS:
            preds = predict(out, m, s, r, centered, half, weights)
            y = labels(m, s, r)
            for w, p in preds.items():
                hits[w].append((p == y)[np.argsort(y)])  # alignment for scoring only
    return {w: np.stack(v) for w, v in hits.items()}


def totals(h):
    return {'unique': int(h.any(0).sum()), 'hits': int(h.sum())}


def changes(base, fused):
    return {'gained': int((fused & ~base).sum()), 'lost': int((base & ~fused).sum()),
            'net': int(fused.sum()-base.sum())}


def main():
    centered = B.dct_encode(B.IMAGES, 16)
    centered -= centered.mean(0, keepdims=True)
    with TemporaryDirectory() as temp:
        dev = evaluate(Path(temp), 3, GRID, centered)
    table = {w: totals(h) for w, h in dev.items()}
    for v in table.values():
        v['eligible'] = v['hits'] >= table[0]['hits']
    w = max((w for w in GRID if table[w]['eligible']), key=lambda w: (table[w]['unique'], -w))
    print('R3 lock:', w, table, flush=True)
    test = evaluate(B.OUT, 4, sorted({0, w}), centered)
    b, f = test[0], test[w]
    result = {'protocol': 'Full r4_iterations comb3, fresh R1-2 fits for R3; manifest-checked R1-3 baseline for R4. Half-window refit each fold, lambda=.001. (1-w)*comb3 + w*row_z(half_logits), then per-session CSLS100, argmax. R3 maximize unique subject to hits>=baseline; smaller weight on tie. R4 historically used for feedback.',
              'mouse_days': 8, 'classes': B.K, 'trials_per_repeat': 8*B.K, 'r3': table, 'locked_weight': w,
              'r4': {'baseline': totals(b), 'fusion': totals(f), 'unique_changes': changes(b.any(0), f.any(0)),
                     'trial_changes': changes(b, f), 'unique_at_least_500': totals(f)['unique'] >= 500,
                     'hits_at_least_572': totals(f)['hits'] >= 572}}
    text = json.dumps(result, indent=2, allow_nan=False)
    (HERE/'COVERAGE_FUSION.json').write_text(text+'\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
