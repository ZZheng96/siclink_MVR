"""Exploratory: can predicted glyph features narrow the 2748-way candidate set?

Feature likelihood = diagonal Gaussian over the z-scored target features, with the
per-feature error variance taken from the inner split (never the held repeat).
Same folds as experiments 70/71/72.
"""

import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test/03_task3_single_trial/70_glyph_feature_probe"))

import importlib.util


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = load("probe", ROOT / "test/03_task3_single_trial/70_glyph_feature_probe/run.py")
exp71 = load("exp71", ROOT / "test/03_task3_single_trial/71_dino_glyph_features/run.py")

KS = (1, 8, 32, 100, 500, 1000)
ALPHAS = (1e2, 1e3, 1e4, 1e5, 1e6)
DINO = "vit_base_patch16_dinov3.lvd1689m"


def recall(scores, labels, ks=KS):
    order = np.argsort(-scores, axis=1, kind="stable")[:, : max(ks)]
    hit = order == labels[:, None]
    return {k: float(hit[:, :k].any(1).mean()) for k in ks}


def standardise(bank):
    return (bank - bank.mean(0)) / np.maximum(bank.std(0), 1e-9)


def main():
    gt = (np.asarray(np.load(probe.CACHE / "pictures.npy")) >= 128).astype(np.float32)
    hand = probe.glyph_features(gt)
    x = torch.from_numpy(gt)[:, None]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    dino = exp71.extract(DINO, x)
    core = dino - dino.mean(0)
    u, s, _ = np.linalg.svd(core, full_matrices=False)
    pcs = standardise((u[:, :64] * s[:64]).astype(np.float32))
    banks = {"hand40": standardise(hand), "dino64": pcs}
    data = {m: probe.load_mouse(m, u_) for m, u_ in probe.MICE.items()}
    result = {"experiment": "73_feature_candidate_recall",
              "protocol_status": "exploratory_not_preregistered",
              "disclosure": "diagnostic derived from experiments 70/71; no held repeat used for alpha or error variance",
              "folds": {}, "summary": {}}
    for mouse, blocks in data.items():
        per_arm = {a: [] for a in ("hand40", "dino64", "template")}
        for k in probe.FOLDS:
            xx = probe.normalize(blocks, k)
            r = np.concatenate([b["r"] for b in blocks])
            y = np.concatenate([b["y"] for b in blocks])
            tr, te = r != k, r == k
            val = [j for j in probe.FOLDS if j != k][0]
            inner = tr & (r != val)
            row = {}
            for arm, bank in banks.items():
                model = probe.Ridge().fit_input(xx[inner]).fit_targets(bank[y[inner]])
                best = max(((a, float(np.median(probe.report(
                    model.predict(xx[r == val], a), bank[y[r == val]], bank[y[inner]].mean(0))[0])))
                    for a in ALPHAS), key=lambda t: t[1])
                inner_r2 = probe.report(model.predict(xx[r == val], best[0]),
                                        bank[y[r == val]], bank[y[inner]].mean(0))[0]
                model = probe.Ridge().fit_input(xx[tr]).fit_targets(bank[y[tr]])
                pred = model.predict(xx[te], best[0])
                sigma2 = np.clip(1.0 - inner_r2, 1e-3, 1.0)
                print(mouse, k, arm, "sigma2 pc1/med", round(float(sigma2[0]), 3),
                      round(float(np.median(sigma2)), 3), flush=True)
                scale = (1.0 / np.sqrt(sigma2)).astype(np.float32)
                ps, bs = pred * scale, bank * scale
                dist = ((ps ** 2).sum(1)[:, None] + (bs ** 2).sum(1)[None, :]
                        - 2.0 * ps @ bs.T)
                row[arm] = recall(-dist, y[te])
                row[arm]["alpha"] = best[0]
                del dist, ps, bs
            # per-session template nearest neighbour, same held rows
            hits = {kk: [] for kk in KS}
            for block in blocks:
                tr_rows = np.flatnonzero(block["r"] != k)
                te_rows = np.flatnonzero(block["r"] == k)
                mu = block["x"][tr_rows].mean(0, dtype=np.float64).astype(np.float32)
                sd = np.maximum(block["x"][tr_rows].std(0, dtype=np.float64).astype(np.float32), 1e-5)
                train = (block["x"][tr_rows] - mu) / sd
                held = (block["x"][te_rows] - mu) / sd
                tmpl = np.stack([train[block["y"][tr_rows] == c].mean(0) for c in range(probe.C)])
                tn = tmpl / np.maximum(np.linalg.norm(tmpl, axis=1, keepdims=True), 1e-9)
                hn = held / np.maximum(np.linalg.norm(held, axis=1, keepdims=True), 1e-9)
                sc = recall(hn @ tn.T, block["y"][te_rows])
                for kk in KS:
                    hits[kk].append(sc[kk])
            row["template"] = {kk: float(np.mean(v)) for kk, v in hits.items()}
            for arm in per_arm:
                per_arm[arm].append(row[arm])
            result["folds"][f"{mouse}/fold{k}"] = row
            print(mouse, k, {a: {kk: round(row[a][kk], 4) for kk in KS} for a in row}, flush=True)
        result["summary"][mouse] = {
            a: {kk: float(np.mean([row[kk] for row in per_arm[a]])) for kk in KS} for a in per_arm}
        print(mouse, "summary", {a: {kk: round(v, 4) for kk, v in d.items()} for a, d in result["summary"][mouse].items()}, flush=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
