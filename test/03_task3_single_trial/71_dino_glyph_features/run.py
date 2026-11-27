"""Exploratory: DINOv3 / DINOv2 CLS features for the 2748 public glyphs.

Reuses the experiment-70 fold structure for the EEG probe. This script was written
after the same diagnostics had been inspected ad hoc, so it is exploratory and not
a pre-registered confirmation.
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

spec = importlib.util.spec_from_file_location(
    "probe", str(ROOT / "test/03_task3_single_trial/70_glyph_feature_probe/run.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

MODELS = {
    "dinov3_b16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov2_b14": "vit_base_patch14_dinov2.lvd142m",
}
WEIGHTS = {
    "vit_base_patch16_dinov3.lvd1689m": ROOT / "models/dinov3_vitb16/model.safetensors",
    "vit_base_patch14_dinov2.lvd142m": ROOT / "models/dinov2_vitb14/model.safetensors",
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
ALPHAS = (1e2, 1e3, 1e4, 1e5, 1e6)


def to_input(binary64):
    x = torch.from_numpy(binary64)[:, None]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    return x


def extract(name, x):
    import timm

    path = WEIGHTS[name]
    overlay = {"file": str(path)} if path.is_file() else None
    model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224,
                              pretrained_cfg_overlay=overlay).cuda().eval()
    out = []
    with torch.no_grad():
        for part in x.split(32):
            part = ((part.cuda() - MEAN) / STD).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(part).float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out).numpy().astype(np.float32)


def kmeans(x, k, seed, iters=60):
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(x[rng.choice(len(x), p=d / d.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        lab = ((x[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[lab == i].mean(0) if (lab == i).any() else centers[i] for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return lab


def ari(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    n = len(a)
    c = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(c, (ai, bi), 1)
    comb = lambda v: (v * (v - 1) / 2).sum()
    ij, ra, cb = comb(c), comb(c.sum(1)), comb(c.sum(0))
    exp = ra * cb / (n * (n - 1) / 2)
    mx = (ra + cb) / 2
    return float((ij - exp) / (mx - exp))


def r2_of(pred, truth, mu):
    return 1 - ((pred - truth) ** 2).sum(0) / np.maximum(((truth - mu) ** 2).sum(0), 1e-12)


def main():
    gt = (np.asarray(np.load(probe.CACHE / "pictures.npy")) >= 128).astype(np.float32)
    n = len(gt)
    hand = probe.glyph_features(gt)
    groups = np.array(probe.GROUPS)
    x = to_input(gt)
    shift = torch.from_numpy(gt)[:, None]
    shift = F.pad(shift, (0, 1, 0, 1))[:, :, 1:, 1:]
    shift = F.interpolate(shift, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    data = {m: probe.load_mouse(m, u) for m, u in probe.MICE.items()}
    result = {
        "experiment": "71_dino_glyph_features",
        "protocol_status": "exploratory_not_preregistered",
        "disclosure": "feature extraction and diagnostics were inspected before this script was written; "
                      "EEG numbers reuse the experiment-70 folds unchanged",
        "data_scope": "public train glyphs and public train EEG only",
        "environment": {"torch": torch.__version__, "timm": __import__("timm").__version__,
                        "torchvision": __import__("torchvision").__version__,
                        "gpu": torch.cuda.get_device_name()},
        "downstream": "test/03_task3_single_trial/70_glyph_feature_probe (same folds, 50 ms ridge)",
        "models": {},
    }
    for tag, name in MODELS.items():
        f = extract(name, x)
        fs = extract(name, shift)
        z = f - f.mean(0)
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
        sim = z @ z.T
        off = sim[~np.eye(n, dtype=bool)]
        zs = fs - f.mean(0)
        zs /= np.maximum(np.linalg.norm(zs, axis=1, keepdims=True), 1e-9)
        s2 = zs @ z.T
        own = np.diag(s2).copy()
        s2[np.arange(n), np.arange(n)] = -9
        fc = f - f.mean(0)
        u, s, _ = np.linalg.svd(fc, full_matrices=False)
        ev = s ** 2 / (s ** 2).sum()
        pcs = (u[:, :64] * s[:64]).astype(np.float32)
        r2_hand = []
        for fold in range(5):
            tr, te = np.arange(n) % 5 != fold, np.arange(n) % 5 == fold
            mu = f[tr].mean(0)
            xc = f[tr] - mu
            w = np.linalg.solve(xc.T @ xc + 1e2 * np.eye(xc.shape[1]), xc.T @ (hand[tr] - hand[tr].mean(0)))
            r2_hand.append(r2_of((f[te] - mu) @ w, hand[te], hand[tr].mean(0)))
        r2_hand = np.median(np.array(r2_hand), 0)
        q = (u[:, :64] * s[:64])
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        cluster = {k: ari(kmeans(q, k, 0), kmeans(q, k, 1)) for k in (8, 16, 32)}
        cell = {
            "shape": list(f.shape),
            "pca_explained": {"pc8": float(ev[:8].sum()), "pc32": float(ev[:32].sum()),
                              "pc128": float(ev[:128].sum())},
            "pairwise_abs_cosine_mean": float(np.abs(off).mean()),
            "shift1px": {"self": float(own.mean()), "best_other": float(s2.max(1).mean()),
                         "self_wins": float((own > s2.max(1)).mean())},
            "handcrafted_recovery_r2": {g: float(np.median(r2_hand[groups == g]))
                                        for g in ("global", "local", "dct")},
            "cluster_ari_two_inits": cluster,
            "eeg_probe": {},
        }
        for mouse, blocks in data.items():
            rows = []
            for k in probe.FOLDS:
                xx = probe.normalize(blocks, k)
                r = np.concatenate([b["r"] for b in blocks])
                y = np.concatenate([b["y"] for b in blocks])
                tr, te = r != k, r == k
                ytr, yte = pcs[y[tr]], pcs[y[te]]
                inner = tr & (r != [j for j in probe.FOLDS if j != k][0])
                inner_val = [j for j in probe.FOLDS if j != k][0]
                model = probe.Ridge().fit_input(xx[inner]).fit_targets(pcs[y[inner]])
                best = max(((a, float(np.median(r2_of(model.predict(xx[r == inner_val], a),
                                                       pcs[y[r == inner_val]], ytr.mean(0)))))
                            for a in ALPHAS), key=lambda t: t[1])
                model = probe.Ridge().fit_input(xx[tr]).fit_targets(ytr)
                r2 = r2_of(model.predict(xx[te], best[0]), yte, ytr.mean(0))
                w = ev[:64] / ev[:64].sum()
                rows.append({"fold": k, "alpha": best[0],
                             "pc1": float(r2[0]), "pc2": float(r2[1]), "pc3": float(r2[2]),
                             "median_pc": float(np.median(r2)),
                             "top8_mean": float(np.sort(r2)[-8:].mean()),
                             "variance_weighted": float((r2 * w).sum())})
                print(tag, mouse, k, rows[-1], flush=True)
            cell["eeg_probe"][mouse] = rows
        result["models"][tag] = cell
        print(tag, {k: tuple(round(float(np.mean([r[f] for r in v])), 3)
                             for f in ("pc1", "pc2", "median_pc", "variance_weighted"))
                    for k, v in cell["eeg_probe"].items()},
              "cluster", cluster, flush=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
