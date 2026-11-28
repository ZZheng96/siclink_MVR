"""Representational stability of rate code vs temporal code across R4 sessions.

No classifier is trained. For each session, per-character mean responses from repeat1-3
are used to build a character-by-character cosine similarity matrix. Stability is the
correlation between similarity matrices across sessions and between repeat halves.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4_temporal_code as T

OUT = T.OUT
SESSIONS = T.SESSIONS
FEATURES = ("rate250", "rate50", "temporal25", "temporal10")


def mean_response(mouse, session, bin_ms, repeats=(1, 2, 3)):
    arr = np.stack([T.load_counts(mouse, session, r, bin_ms)[0] for r in repeats])
    x = np.log1p(arr.mean(axis=0))
    if bin_ms == 250:
        x = x.sum(axis=1)
    else:
        x = x.reshape(len(x), -1)
    # z-score each feature across characters; removes session-wide gain
    x = (x - x.mean(0, keepdims=True)) / np.maximum(x.std(0, keepdims=True), 1e-6)
    return x.astype(np.float32)


def sim_matrix(x):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
    return x @ x.T


def upper(s):
    iu = np.triu_indices(len(s), k=1)
    return s[iu]


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


results = {}
for mouse in T.MICE:
    for name in FEATURES:
        bin_ms = T.BINS[name]
        mats = {s: sim_matrix(mean_response(mouse, s, bin_ms)) for s in SESSIONS}
        cross = {}
        for i, s1 in enumerate(SESSIONS):
            for s2 in SESSIONS[i + 1:]:
                cross[f"s{s1}-s{s2}"] = corr(upper(mats[s1]), upper(mats[s2]))
        halves = {}
        for s in SESSIONS:
            h1 = sim_matrix(mean_response(mouse, s, bin_ms, (1,)))
            h2h3 = sim_matrix(mean_response(mouse, s, bin_ms, (2, 3)))
            halves[f"session{s}"] = corr(upper(h1), upper(h2h3))
        results[f"{mouse}/{name}"] = {
            "cross_session": cross,
            "cross_session_mean": float(np.mean(list(cross.values()))),
            "split_half": halves,
            "split_half_mean": float(np.mean(list(halves.values()))),
        }
        print(mouse, name, "cross", round(results[f"{mouse}/{name}"]["cross_session_mean"], 4),
              "half", round(results[f"{mouse}/{name}"]["split_half_mean"], 4), flush=True)

(OUT / "TEMPORAL_STABILITY.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
