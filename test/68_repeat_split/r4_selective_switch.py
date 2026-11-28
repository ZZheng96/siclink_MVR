"""Select the runner-up switch on R3; compare only the locked rule to current on R4."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import r4_coverage_fusion as F
import numpy as np

B, Z, DEVICE = F.B, F.Z, F.H.DEVICE
GRID = (None, .05, .15, .30)  # conservative first, including off


def scores_and_votes(out, m, s, r, centered):
    # Same comb3 as F.full_scores; retain the three branch votes in the same pass.
    v5, mlp, votes = [], [], []
    for name, width, _ in F.RUNS:
        root = out / name
        folder = root / 'task3' / m
        raw = F.counts(m, s, r, width)
        with np.load(folder / f'session{s}_normalizer.npz') as norm:
            x = B.apply_session_normalizer(raw, (norm['mean'], norm['scale']))
        logits = np.mean([B.predict_logits(B.load_classifier(B.checkpoint_path(root, 'task3', m, seed),
                          device=DEVICE), x, device=DEVICE) for seed in B.SEEDS], axis=0)
        if width == 250:
            current = np.mean([Z(np.mean(v5, axis=0)), Z(np.mean(mlp, axis=0)), Z(logits)], axis=0)
        else:
            assert width in (10, 25, 50)
            votes.append(logits.argmax(1))
            enc = B.response_scores(np.log1p(raw), np.load(folder / 'encoding_prototypes.npy'),
                                    metric='mse', device=DEVICE)
            dct = B.softmax_dct_similarity(logits, centered, device=DEVICE)
            mlp.append(Z(logits))
            v5.append(Z((mlp[-1] + Z(enc) + Z(dct)) / 3))
    row = np.partition(current, -100, axis=1)[:, -100:].mean(1, keepdims=True)
    col = np.partition(current, -100, axis=0)[-100:].mean(0, keepdims=True)
    return 2 * current - row - col, np.stack(votes)


def predict(out, m, s, r, centered, thresholds):
    csls, votes = scores_and_votes(out, m, s, r, centered)
    first = csls.argmax(1)
    rows = np.arange(len(first))
    z = Z(csls)
    best = z[rows, first]
    csls[rows, first] = -np.inf
    second = csls.argmax(1)
    gap = best - z[rows, second]
    supported = (np.stack(votes) == second).sum(0) >= 2
    return {t: first if t is None else np.where(supported & (gap <= t), second, first) for t in thresholds}


def evaluate(out, r, thresholds, centered):
    hits, switches = ({t: [] for t in thresholds} for _ in range(2))
    for m in B.MICE:
        if r == 3:
            F.train_dev(out, m)
        else:
            for name, _, _ in F.RUNS:
                manifest = json.loads((out / name / 'bench_manifest.json').read_text())
                assert manifest['protocol'] == 'repeat1-3 train, repeat4 evaluate'
                for seed in B.SEEDS:
                    assert B.sha(B.checkpoint_path(out / name, 'task3', m, seed)) == manifest['checkpoints'][f'{m}/{seed}']
        for s in B.SESSIONS:
            preds = predict(out, m, s, r, centered, thresholds)
            y = F.labels(m, s, r)
            order = np.argsort(y)
            for t, p in preds.items():
                hits[t].append((p == y)[order])
                switches[t].append((p != preds[None])[order])
    return {t: (np.stack(hits[t]), np.stack(switches[t])) for t in thresholds}


def metrics(base, hits, changed):
    return {**F.totals(hits), 'switched': int(changed.sum()), 'corrected': int((hits & ~base).sum()),
            'spoiled': int((base & ~hits).sum()), 'both_wrong': int((changed & ~base & ~hits).sum())}


def comparison(base, hits, changed):
    return {'baseline': F.totals(base), 'locked': metrics(base, hits, changed),
            'unique_changes': F.changes(base.any(0), hits.any(0)), 'trial_changes': F.changes(base, hits)}


def key(t):
    return 'off' if t is None else str(t)


def main():
    centered = B.dct_encode(B.IMAGES, 16)
    centered -= centered.mean(0, keepdims=True)
    with TemporaryDirectory() as tmp:
        dev = evaluate(Path(tmp), 3, GRID, centered)
    table = {key(t): metrics(dev[None][0], *dev[t]) for t in GRID}
    for row in table.values():
        row['eligible'] = row['hits'] >= table['off']['hits']
    # max keeps the first (most conservative) candidate on equal coverage.
    lock = max((t for t in GRID if table[key(t)]['eligible']), key=lambda t: table[key(t)]['unique'])
    print('R3 lock:', key(lock), table, flush=True)
    test = evaluate(B.OUT, 4, (None,) if lock is None else (None, lock), centered)
    base, (hits, changed) = test[None][0], test[lock]
    r4 = comparison(base, hits, changed)
    assert r4['baseline'] == {'unique': 462, 'hits': 572}, r4['baseline']
    r4['by_mouse'] = {}
    for i, m in enumerate(B.MICE):
        sl = slice(i * len(B.SESSIONS), (i + 1) * len(B.SESSIONS))
        r4['by_mouse'][m] = comparison(base[sl], hits[sl], changed[sl])
    r4['unique_at_least_500'] = r4['locked']['unique'] >= 500
    r4['hits_at_least_572'] = r4['locked']['hits'] >= 572
    result = {'protocol': 'Fresh R1-2 training for R3; R1-3 saved models for R4. Each 10/25/50ms vote is argmax of mean MLP logits over seeds 42/202/777. Switch only to CSLS100 current runner-up with >=2 votes and row-z margin <= threshold. Maximize R3 unique with hits >= baseline; ties favor off then smaller threshold. R4 historically used for feedback.',
              'classes': B.K, 'trials_per_repeat': len(base) * B.K, 'r3': table, 'locked_threshold': key(lock), 'r4': r4}
    text = json.dumps(result, indent=2, allow_nan=False)
    (F.HERE / 'SELECTIVE_SWITCH.json').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
