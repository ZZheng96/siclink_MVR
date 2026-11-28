"""Target-aligned representational stability, using repeats 1–3 only."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

import r4_temporal_code as T

OUT = Path(__file__).resolve().with_name("TEMPORAL_STABILITY.json")
FEATURES = ("rate250", "rate50", "temporal25", "temporal10")
REPEATS = (1, 2, 3)
LABEL_SEED = 42
ROW_SEED = 2026


def align(counts, target):
    if counts.shape[0] != T.K or target.shape != (T.K,) or target.dtype.kind not in 'iu':
        raise ValueError("expected one integer target per character")
    order = np.argsort(target)
    if not np.array_equal(target[order], np.arange(T.K)):
        raise ValueError(f"target is not a permutation of 0..{T.K - 1}")
    return counts[order]


def sim_matrix(counts):
    x = np.log1p(counts).reshape(T.K, -1)
    x -= x.mean(0)
    x /= np.maximum(x.std(0), 1e-6)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
    return x @ x.T


def corr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


def analyze(mouse, name, upper):
    label_rng = np.random.default_rng(LABEL_SEED)
    row_rng = np.random.default_rng(ROW_SEED)
    real, shuffled = ({"split_half": {}, "cross_session": {}} for _ in range(2))

    def compare(kind, key, left, right):
        # The same second-side relabeling applies to both axes, preserving geometry.
        p = label_rng.permutation(T.K)
        i, j = upper
        a = left[i, j]
        real[kind][key] = corr(a, right[i, j])
        shuffled[kind][key] = corr(a, right[p[i], p[j]])

    mats = {}
    for s in T.SESSIONS:
        blocks = []
        for r in REPEATS:
            counts, y = T.load_counts(mouse, s, r, T.BINS[name])
            aligned = align(counts, y)
            p = row_rng.permutation(T.K)
            if not np.array_equal(aligned, align(counts[p], y[p])):
                raise AssertionError(f"row-order dependence: {mouse}/{name}/s{s}/r{r}")
            blocks.append(aligned)
            T.CACHE.pop((mouse, s, r, T.BINS[name]), None)
        a, b, c = blocks
        compare("split_half", f"session{s}", sim_matrix(a), sim_matrix((b + c) / 2))
        mats[s] = sim_matrix((a + b + c) / 3)
    for s, t in combinations(T.SESSIONS, 2):
        compare("cross_session", f"s{s}-s{t}", mats[s], mats[t])
    for result in (real, shuffled):
        for key in ("cross_session", "split_half"):
            result[f"{key}_mean"] = float(np.mean(list(result[key].values())))
    real['label_shuffle'] = shuffled
    return real


def main():
    results = {}
    upper = np.triu_indices(T.K, k=1)
    for mouse in T.MICE:
        for name in FEATURES:
            r = analyze(mouse, name, upper)
            results[f"{mouse}/{name}"] = r
            print(mouse, name, *(round(d[k + '_mean'], 6) for k in ('split_half', 'cross_session')
                                 for d in (r, r['label_shuffle'])), flush=True)
    results['_protocol'] = {
        'repeats': REPEATS, 'sessions': T.SESSIONS, 'targets_per_repeat': T.K,
        'split_half': 'R1 vs mean(R2,R3)', 'cross_session': 'mean(R1,R2,R3) on each day',
        'label_shuffle': 'One second-side permutation per comparison; same permutations across mice/features.',
        'label_seed': LABEL_SEED, 'row_seed': ROW_SEED,
        'row_invariance': 'All aligned count tensors exactly equal after joint count/target row permutation.',
    }
    OUT.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
