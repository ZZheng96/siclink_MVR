"""Which feature transform is effective? One fixed nearest-centroid probe, seven transforms."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test/03_task3_single_trial/70_glyph_feature_probe"))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "probe_mod", str(ROOT / "test/03_task3_single_trial/70_glyph_feature_probe/run.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

HASH = "3b7cda2bbfaecb888719b52f3a1d4bb664ad4507299937eb1426d04a915b77f9"
TRANSFORMS = ("base", "l2", "trial_z", "common_mode", "pca_white", "wccn", "unit_top128")
KS = (1, 32, 100)
N_PCA = 256
SHRINK = 0.1
TOP_UNITS = 128


def standardise_fit(x):
    return x.mean(0, dtype=np.float64).astype(np.float32), np.maximum(
        x.std(0, dtype=np.float64).astype(np.float32), 1e-5)


def apply_standardise(x, stats):
    mu, sd = stats
    return (x - mu) / sd


def whiten_matrix(cov, shrink):
    d = cov.shape[0]
    cov = (1 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d, dtype=np.float32)
    w, v = np.linalg.eigh(cov.astype(np.float64))
    w = np.maximum(w, 1e-6)
    return (v / np.sqrt(w)[None, :]).astype(np.float32)


def transform(name, xtr, xte, ytr, stats):
    if name == "base":
        return apply_standardise(xtr, stats), apply_standardise(xte, stats)
    if name == "l2":
        a, b = apply_standardise(xtr, stats), apply_standardise(xte, stats)
        norm = lambda z: z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-6)
        return norm(a), norm(b)
    if name == "trial_z":
        a, b = apply_standardise(xtr, stats), apply_standardise(xte, stats)
        f = lambda z: (z - z.mean(1, keepdims=True)) / np.maximum(z.std(1, keepdims=True), 1e-6)
        return f(a), f(b)
    if name == "common_mode":
        a = xtr - xtr.mean(1, keepdims=True)
        b = xte - xte.mean(1, keepdims=True)
        st = standardise_fit(a)
        return apply_standardise(a, st), apply_standardise(b, st)
    if name == "pca_white":
        a, b = apply_standardise(xtr, stats), apply_standardise(xte, stats)
        mu = a.mean(0)
        ac = a - mu
        u, s, vh = np.linalg.svd(ac, full_matrices=False)
        w = (vh[:N_PCA].T / np.sqrt(np.maximum(s[:N_PCA], 1e-6))[None, :]).astype(np.float32)
        return ac @ w, (b - mu) @ w
    if name == "wccn":
        a, b = apply_standardise(xtr, stats), apply_standardise(xte, stats)
        mu = a.mean(0)
        resid = a - mu
        w = whiten_matrix((resid.T @ resid) / len(a), SHRINK)
        ac = (a - mu) @ w
        return ac, (b - mu) @ w
    raise ValueError(name)


def unit_mask(xtr, ytr, units):
    """Keep the TOP_UNITS units with the largest mean ANOVA F across their bins."""
    mu = xtr.mean(0)
    between = np.zeros(xtr.shape[1])
    within = np.zeros(xtr.shape[1])
    for c in np.unique(ytr):
        rows = xtr[ytr == c]
        between += len(rows) * (rows.mean(0) - mu) ** 2
        within += ((rows - rows.mean(0)) ** 2).sum(0)
    f = between / np.maximum(within / max(len(ytr) - len(np.unique(ytr)), 1), 1e-9)
    per_unit = f.reshape(units, -1).mean(1)
    keep = np.argsort(-per_unit)[:TOP_UNITS]
    mask = np.zeros(xtr.shape[1], dtype=bool)
    bins = xtr.shape[1] // units
    for u in keep:
        mask[u * bins:(u + 1) * bins] = True
    return mask


def centroid_scores(xtr, ytr, xte):
    classes = np.unique(ytr)
    mu = np.stack([xtr[ytr == c].mean(0) for c in classes])
    return -(np.square(xte).sum(1)[:, None] + np.square(mu).sum(1)[None, :] - 2 * xte @ mu.T)


def recall(scores, labels):
    width = max(KS)
    idx = np.argpartition(-scores, width, axis=1)[:, :width]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.take_along_axis(idx, np.argsort(-vals, axis=1), axis=1)
    hit = order == labels[:, None]
    return {k: float(hit[:, :k].any(1).mean()) for k in KS}


def noise_stats(xtr, ytr):
    resid = xtr - np.stack([xtr[ytr == c].mean(0) for c in np.unique(ytr)])[
        np.searchsorted(np.unique(ytr), ytr)]
    total = ((xtr - xtr.mean(0)) ** 2).sum()
    share = 1.0 - float((resid ** 2).sum() / total)
    s = np.linalg.svd(resid, full_matrices=False, compute_uv=False)
    pc1 = float(s[0] ** 2 / (s ** 2).sum())
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, resid.shape[1], size=(4000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    corr = np.corrcoef(resid.T)
    return {"between_class_share_in_sample": share, "residual_pc1_share": pc1,
            "mean_abs_pair_corr": float(np.abs(corr[pairs[:, 0], pairs[:, 1]]).mean())}


def main():
    digest = hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()
    if digest != HASH:
        raise SystemExit("preregistration changed")
    data = {m: probe.load_mouse(m, u) for m, u in probe.MICE.items()}
    result = {"experiment": "76_feature_effectiveness", "preregistration_sha256": HASH,
              "probe": "per-session nearest class mean, euclidean",
              "folds": {}, "summary": {}, "noise": {}}
    for mouse, blocks in data.items():
        per_t = {t: {f"{mouse}/f{k}": {} for k in probe.FOLDS} for t in TRANSFORMS}
        noise_rows = []
        for k in probe.FOLDS:
            for s, block in enumerate(blocks, start=1):
                tr = block["r"] != k
                te = block["r"] == k
                ytr, yte = block["y"][tr], block["y"][te]
                stats = standardise_fit(block["x"][tr])
                for t in TRANSFORMS:
                    if t == "unit_top128":
                        a, b = apply_standardise(block["x"][tr], stats), apply_standardise(
                            block["x"][te], stats)
                        mask = unit_mask(a, ytr, probe.MICE[mouse])
                        a, b = a[:, mask], b[:, mask]
                    else:
                        a, b = transform(t, block["x"][tr], block["x"][te], ytr, stats)
                    sc = recall(centroid_scores(a, ytr, b), yte)
                    cell = per_t[t][f"{mouse}/f{k}"]
                    for kk in KS:
                        cell.setdefault(kk, []).append(sc[kk])
                a_all, _ = transform("base", block["x"][tr], block["x"][te], ytr, stats)
                noise_rows.append(noise_stats(a_all, ytr))
        result["noise"][mouse] = {key: float(np.mean([r[key] for r in noise_rows]))
                                  for key in noise_rows[0]}
        result["summary"][mouse] = {}
        for t in TRANSFORMS:
            agg = {}
            for kk in KS:
                vals = [np.mean(per_t[t][key][kk]) for key in per_t[t]]
                agg[f"recall{kk}"] = float(np.mean(vals))
            wins = sum(np.mean(per_t[t][f"{mouse}/f{k}"][32])
                       > np.mean(per_t["base"][f"{mouse}/f{k}"][32]) for k in probe.FOLDS)
            agg["folds_better_than_base_recall32"] = int(wins)
            agg["recall32_per_fold"] = [float(np.mean(per_t[t][f"{mouse}/f{k}"][32]))
                                        for k in probe.FOLDS]
            result["summary"][mouse][t] = agg
            print(mouse, t, {kk: round(agg[f"recall{kk}"], 4) for kk in KS}, "wins", wins,
                  flush=True)
        print(mouse, "noise", result["noise"][mouse], flush=True)
    result["effective"] = {
        t: all(result["summary"][m][t]["folds_better_than_base_recall32"] >= 3 for m in probe.MICE)
        for t in TRANSFORMS if t != "base"}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print("effective", result["effective"], flush=True)


if __name__ == "__main__":
    main()
