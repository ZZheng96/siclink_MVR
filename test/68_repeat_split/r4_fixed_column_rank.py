"""Fixed column-rank Gaussian calibration; oracle masks only for conditional evaluation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from r4_selective_switch import B, F, scores_and_votes


def blend(scores):
    g = np.empty_like(scores)
    for j in range(0, B.K, 128):
        cols = slice(j, j + 128)
        ranks = rankdata(scores[:, cols], axis=0, method='average')
        g[:, cols] = ndtri((ranks - .5) / len(scores))
    return .5 * F.Z(scores) + .5 * F.Z(g)


def hits(blocks, ids):
    # Rows already aligned by true class for evaluation only; ids are ascending.
    return np.stack([ids[a[np.ix_(ids, ids)].argmax(1)] == ids for a in blocks])


def compare(b, n):
    def part(x, y):
        return {'baseline': F.totals(x), 'fixed': F.totals(y),
                'coverage_changes': F.changes(x.any(0), y.any(0)), 'hit_changes': F.changes(x, y)}
    k = len(B.SESSIONS)
    return {**part(b, n), 'by_mouse': {m: part(b[i*k:(i+1)*k], n[i*k:(i+1)*k])
                                       for i, m in enumerate(B.MICE)}}


def peel(blocks, ids, h):
    rounds = []
    while True:
        rounds.append(F.totals(h))
        ids = ids[~h.any(0)]
        if rounds[-1]['unique'] == 0 or not len(ids):
            return ids, rounds
        h = hits(blocks, ids)


def evaluate(out, r, centered):
    base, fixed = [], []
    for m in B.MICE:
        if r == 3:
            F.train_dev(out, m)
        else:
            for name, _, _ in F.RUNS:
                manifest = json.loads((out / name / 'bench_manifest.json').read_text())
                assert manifest['protocol'] == 'repeat1-3 train, repeat4 evaluate'
                for seed in B.SEEDS:
                    assert B.sha(B.checkpoint_path(out/name, 'task3', m, seed)) == manifest['checkpoints'][f'{m}/{seed}']
        for s in B.SESSIONS:
            a, _ = scores_and_votes(out, m, s, r, centered)
            n = blend(a)
            order = np.argsort(F.labels(m, s, r))
            base.append(a[order]); fixed.append(n[order])
    ids = np.arange(B.K)
    b, n = hits(base, ids), hits(fixed, ids)
    hard, rounds = peel(base, ids, b)
    if r == 4:
        assert [v['unique'] for v in rounds] == [462, 98, 23, 6, 5, 2, 1, 1, 0] and len(hard) == 2150
    hb, hn = hits(base, hard), hits(fixed, hard)
    left, extra = peel(fixed, hard, hn)
    result = {'full': compare(b, n), 'baseline_oracle_rounds': rounds, 'hard_count': len(hard),
              'hard_ids': hard.tolist(), 'hard_one_round': compare(hb, hn),
              'fixed_oracle_rounds_from_hard': extra, 'oracle_remaining': len(left)}
    print('R', r, {k:v for k,v in result.items() if k != 'hard_ids'}, flush=True)
    return result


if __name__ == '__main__':
    centered = B.dct_encode(B.IMAGES, 16)
    centered -= centered.mean(0, keepdims=True)
    result = {'protocol': 'R4 saved R1-3 models; R3 fresh F.train_dev R1-2. No tuning. Per mouse-day, scipy average column rank of CSLS100 over all unlabeled trials; p=(rank-.5)/2748; .5*row_z(CSLS100)+.5*row_z(ndtri(p)) on all 2748 columns before masks. Smallest class ID wins ties. Synchronous oracle removes union of correct classes across eight mouse-days after each round; residuals computed separately per repeat. No score recomputation after masking.',
              'classes': B.K, 'trials_per_repeat': B.K * len(B.MICE) * len(B.SESSIONS)}
    result['r4'] = evaluate(B.OUT, 4, centered)
    with TemporaryDirectory() as tmp:
        result['r3'] = evaluate(Path(tmp), 3, centered)
    (F.HERE / 'FIXED_COLUMN_RANK.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
