"""Run from the project venv: python test/68_repeat_split/sync_oracle.py"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from r4_fixed_column_rank import B, F, blend, hits

ORDER = ('csls', 'column', 'mlp10', 'mlp25', 'mlp50', 'mlp250', 'encoding50', 'soft_dct50')


def score(out, m, s, r, centered):
    scores, v5, mlp = {}, [], []
    dev, z = F.H.DEVICE, F.Z
    for name, width, _ in F.RUNS:
        root = out / name
        folder = root / 'task3' / m
        raw = F.counts(m, s, r, width)
        with np.load(folder / f'session{s}_normalizer.npz') as norm:
            x = B.apply_session_normalizer(raw, (norm['mean'], norm['scale']))
        logits = np.mean([B.predict_logits(B.load_classifier(B.checkpoint_path(root, 'task3', m, seed),
                          device=dev), x, device=dev) for seed in B.SEEDS], axis=0)
        scores[f'mlp{width}'] = logits
        if width != 250:
            enc = B.response_scores(np.log1p(raw), np.load(folder / 'encoding_prototypes.npy'), metric='mse', device=dev)
            dct = B.softmax_dct_similarity(logits, centered, device=dev)
            mlp.append(z(logits))
            v5.append(z((mlp[-1] + z(enc) + z(dct)) / 3))
            if width == 50:
                scores.update(encoding50=enc, soft_dct50=dct)
    a = np.mean([z(np.mean(v5, axis=0)), z(np.mean(mlp, axis=0)), z(scores['mlp250'])], axis=0)
    row = np.partition(a, -100, axis=1)[:, -100:].mean(1, keepdims=True)
    col = np.partition(a, -100, axis=0)[-100:, :].mean(0, keepdims=True)
    scores['csls'] = 2*a - row - col
    scores['column'] = blend(scores['csls'])
    return scores


def peel(blocks, ids):
    rounds = []
    while len(ids):
        h = hits(blocks, ids)
        rounds.append(F.totals(h))
        if not h.any():
            break
        ids = ids[~h.any(0)]
    return ids, rounds


def evaluate(out, r, centered, ref):
    blocks = {k: [] for k in ORDER}
    for m in B.MICE:
        if r == 3:
            F.train_dev(out, m)
        else:
            for name, _, _ in F.RUNS:
                manifest = json.loads((out/name/'bench_manifest.json').read_text())
                assert manifest['protocol'] == 'repeat1-3 train, repeat4 evaluate'
                for seed in B.SEEDS:
                    assert B.sha(B.checkpoint_path(out/name, 'task3', m, seed)) == manifest['checkpoints'][f'{m}/{seed}']
        for s in B.SESSIONS:
            scores = score(out, m, s, r, centered)
            order = np.argsort(F.labels(m, s, r))  # alignment for evaluation only
            for k in ORDER:
                blocks[k].append(scores[k][order])
    ids = np.arange(B.K)
    full = {k: F.totals(hits(blocks[k], ids)) for k in ORDER}
    ids, steps = peel(blocks['csls'], ids)
    assert steps == ref['baseline_oracle_rounds'] and ids.tolist() == ref['hard_ids']
    ids, steps = peel(blocks['column'], ids)
    assert steps == ref['fixed_oracle_rounds_from_hard'] and len(ids) == ref['oracle_remaining'] == {3:2208, 4:2069}[r]
    start, cycles = len(ids), []
    while len(ids):
        before, detail = len(ids), {}
        for k in ORDER:
            ids, detail[k] = peel(blocks[k], ids)
            if not len(ids):
                break
        cycles.append(dict(method_steps=detail, removed=before-len(ids), remaining=len(ids)))
        if before == len(ids):
            break
    print('R', r, 'full', full, 'start', start, flush=True)
    for c in cycles:
        print([sum(v['unique'] for v in steps) for steps in c['method_steps'].values()], c['remaining'], flush=True)
    return dict(full=full, start=start, cycles=cycles, remaining=len(ids), remaining_ids=ids.tolist())


if __name__ == '__main__':
    ref = json.loads((F.HERE/'FIXED_COLUMN_RANK.json').read_text())
    centered = B.dct_encode(B.IMAGES, 16)
    centered -= centered.mean(0, keepdims=True)
    result = dict(order=ORDER, classes=B.K, trials=8*B.K, seeds=B.SEEDS,
                  protocol='R1-2 fresh fit -> R3; saved R1-3 -> R4. Full is independent top1. Freeze full scores before evaluation. Ascending ID tie break. Each step removes union of correct IDs across eight mouse-days simultaneously; no refit, reranking transform or reassignment.')
    result['r4'] = evaluate(B.OUT, 4, centered, ref['r4'])
    with TemporaryDirectory() as tmp:
        result['r3'] = evaluate(Path(tmp), 3, centered, ref['r3'])
    (F.HERE/'SYNC_ORACLE.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')
