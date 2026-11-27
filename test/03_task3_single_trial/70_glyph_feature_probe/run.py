"""Probe: does a single 50 ms-binned trial carry glyph-feature information?

Frozen protocol in preregistration.json. Public train data only.
"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.fft import dctn
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from src.evaluation import score_predictions
from src.features import trial_count_features
from src.neural import SpikeClassifier
from src.retrieval import unique_assignment

HASH = "1b814058dd59e20de6a09f2730432236cacaeae07edb4ccfc57ade78aad5c31a"
TRAIN = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset/train"
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
MICE = {"mice1": 134, "mice2": 78}
C = 2748
VIEW = 50
FOLDS = (1, 2, 3, 4)
ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
NAMES = (
    ["ink_ratio", "edge_density", "centroid_x", "centroid_y",
     "variance_x", "variance_y", "log_elongation", "symmetry_iou"]
    + [f"block_{i}_{j}" for i in range(4) for j in range(4)]
    + [f"dct_{i}_{j}" for i in range(4) for j in range(4)]
)
GROUPS = ["global"] * 8 + ["local"] * 16 + ["dct"] * 16


def check(cond, msg):
    if not cond:
        raise ValueError(msg)


def load_mouse(mouse, units):
    blocks = []
    for s in range(1, 5):
        with np.load(TRAIN / mouse / f"session{s}.npz") as f:
            frame = SimpleNamespace(
                trial_ptr=f["trial_ptr"], spike_samples=f["spike_samples"],
                unit_index=f["unit_index"], n_units=units,
                sample_rate_hz=30000, window_ms=250)
            y = f["target"].astype(np.int64)
            ids = f["sample_ids"]
        x = np.log1p(trial_count_features(frame, VIEW)).astype(np.float32)
        r = np.empty(len(y), np.int64)
        cnt = np.zeros(C, np.int64)
        for i, t in enumerate(y):
            cnt[t] += 1
            r[i] = cnt[t]
        check((cnt == 4).all() and len(set(ids)) == 4 * C, f"{mouse} session{s} multiplicity")
        blocks.append(dict(x=x, y=y, r=r))
    return blocks


def glyph_features(gt):
    n = len(gt)
    m = gt.astype(np.float64)
    flat = m.reshape(n, -1)
    total = flat.sum(1) + 1e-9
    ink = flat.mean(1)
    edge = np.abs(np.diff(m, axis=2)).mean((1, 2)) + np.abs(np.diff(m, axis=1)).mean((1, 2))
    ys, xs = np.mgrid[0:64, 0:64]
    cx = (flat * xs.ravel()).sum(1) / total / 63.0
    cy = (flat * ys.ravel()).sum(1) / total / 63.0
    vx = (flat * (xs.ravel() - cx[:, None] * 63.0) ** 2).sum(1) / total / 3969.0
    vy = (flat * (ys.ravel() - cy[:, None] * 63.0) ** 2).sum(1) / total / 3969.0
    elong = np.log((np.sqrt(vx) + 1e-6) / (np.sqrt(vy) + 1e-6))
    flip = np.flip(m, axis=2)
    sym = np.minimum(m, flip).sum((1, 2)) / np.maximum(np.maximum(m, flip).sum((1, 2)), 1e-9)
    glob = np.stack([ink, edge, cx, cy, vx, vy, elong, sym], 1)
    local = m.reshape(n, 4, 16, 4, 16).mean((2, 4)).reshape(n, 16)
    dct = dctn(m, axes=(-2, -1), norm="ortho")[:, :4, :4].reshape(n, 16)
    feat = np.concatenate([glob, local, dct], 1)
    return ((feat - feat.mean(0)) / np.maximum(feat.std(0), 1e-9)).astype(np.float32)


class Ridge:
    """Closed-form multi-output ridge with one input factorisation."""

    def fit_input(self, x):
        self.mu = x.mean(0)
        self.xc = x - self.mu
        self.w, self.v = np.linalg.eigh(self.xc.T @ self.xc)
        return self

    def fit_targets(self, y):
        self.b = self.xc.T @ y
        return self

    def predict(self, x, alpha):
        coef = self.v @ ((self.v.T @ self.b) / (self.w[:, None] + alpha))
        return (x - self.mu) @ coef


def scores_of(model, x):
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(x)).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.cat([model(a) for a in x.split(256)]).float().cpu().numpy()


def fit_mlp(x, y, epochs=10):
    torch.manual_seed(42)
    model = SpikeClassifier(x.shape[1], y.shape[1], 1024, 0.0, False)
    model.set_normalization(x)
    model.cuda()
    xt = torch.from_numpy(x).cuda()
    yt = torch.from_numpy(y).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0, fused=True)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.from_numpy(np.random.default_rng(42 + epoch).permutation(len(yt))).cuda()
        for part in order.split(256):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.mse_loss(model(xt[part]), yt[part])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
    return model


def report(pred, truth, train_mean):
    sse = ((pred - truth) ** 2).sum(0)
    sst = ((truth - train_mean) ** 2).sum(0)
    r2 = 1.0 - sse / np.maximum(sst, 1e-12)
    rho = np.array([spearmanr(pred[:, j], truth[:, j]).statistic for j in range(pred.shape[1])])
    return r2, rho


def normalize(blocks, k):
    parts = []
    for b in blocks:
        tr = b["r"] != k
        mu = b["x"][tr].mean(0, dtype=np.float64).astype(np.float32)
        sd = np.maximum(b["x"][tr].std(0, dtype=np.float64).astype(np.float32), 1e-5)
        parts.append((b["x"] - mu) / sd)
    return np.concatenate(parts)


def readout(scores, labels, gt, heldout, device):
    assigned = unique_assignment(scores)
    mask = np.isin(labels, heldout)
    values = score_predictions(gt[assigned[mask]], gt[labels[mask]], device=device, batch_size=16)
    return {
        "ocf": float(values.mean()),
        "top1_raw": float((scores.argmax(1) == labels).mean()),
        "top1_assigned": float((assigned == labels).mean()),
        "n": int(mask.sum()),
    }


def main():
    digest = hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()
    check(digest == HASH, "preregistration changed")
    device = "cuda"
    gt = (np.asarray(np.load(CACHE / "pictures.npy")) >= 128).astype(np.float32)
    feat = glyph_features(gt)
    heldout = np.sort(np.random.default_rng(123).permutation(C)[256:768])
    data = {mouse: load_mouse(mouse, units) for mouse, units in MICE.items()}
    result = {
        "experiment": "70_glyph_feature_probe",
        "preregistration_sha256": HASH,
        "bin_ms": VIEW,
        "feature_names": NAMES,
        "feature_groups": GROUPS,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(),
        "data_scope": "public train only; no hidden test data",
        "folds": {},
        "ocf_fold4": {},
    }
    for mouse, blocks in data.items():
        for k in FOLDS:
            x = normalize(blocks, k)
            r = np.concatenate([b["r"] for b in blocks])
            y = np.concatenate([b["y"] for b in blocks])
            tr, te = r != k, r == k
            check(len(np.unique(y[te])) == C and te.sum() == 4 * C, "held composition")
            ytr, yte = feat[y[tr]], feat[y[te]]
            keys = [j for j in FOLDS if j != k]
            inner_val = keys[0]
            inner_tr = tr & (r != inner_val)
            ridge_inner = Ridge().fit_input(x[inner_tr]).fit_targets(feat[y[inner_tr]])
            best = None
            for alpha in ALPHAS:
                pred = ridge_inner.predict(x[r == inner_val], alpha)
                value = float(np.mean(report(pred, feat[y[r == inner_val]], ytr.mean(0))[0]))
                if best is None or value > best[1]:
                    best = (alpha, value)
            ridge = Ridge().fit_input(x[tr]).fit_targets(ytr)
            pred_ridge = ridge.predict(x[te], best[0])
            r2_ridge, rho_ridge = report(pred_ridge, yte, ytr.mean(0))
            perm = np.random.default_rng(7 + k).permutation(tr.sum())
            ridge_null = Ridge().fit_input(x[tr]).fit_targets(feat[y[tr]][perm])
            r2_null, _ = report(ridge_null.predict(x[te], best[0]), yte, ytr.mean(0))
            model = fit_mlp(x[tr], ytr)
            pred_mlp = scores_of(model, x[te])
            r2_mlp, rho_mlp = report(pred_mlp, yte, ytr.mean(0))
            key = f"{mouse}/fold{k}"
            result["folds"][key] = {
                "alpha": best[0],
                "inner_r2": best[1],
                "ridge_r2": r2_ridge.tolist(),
                "ridge_spearman": rho_ridge.tolist(),
                "mlp_r2": r2_mlp.tolist(),
                "mlp_spearman": rho_mlp.tolist(),
                "null_r2": r2_null.tolist(),
                "null_r2_by_group": {g: float(np.median(r2_null[np.array(GROUPS) == g]))
                                     for g in ("global", "local", "dct")},
                "ridge_r2_by_group": {g: float(np.median(r2_ridge[np.array(GROUPS) == g]))
                                      for g in ("global", "local", "dct")},
                "mlp_r2_by_group": {g: float(np.median(r2_mlp[np.array(GROUPS) == g]))
                                    for g in ("global", "local", "dct")},
            }
            print(key, "alpha", best[0],
                  "ridge", {g: round(v, 4) for g, v in result["folds"][key]["ridge_r2_by_group"].items()},
                  "mlp", {g: round(v, 4) for g, v in result["folds"][key]["mlp_r2_by_group"].items()},
                  "null", {g: round(v, 4) for g, v in result["folds"][key]["null_r2_by_group"].items()},
                  flush=True)
            if k != 4:
                continue
            bank_sq = (feat ** 2).sum(1)
            held_order = np.flatnonzero(r == 4)
            offset = 0
            for s, block in enumerate(blocks, start=1):
                rows = np.flatnonzero(block["r"] == 4)
                check(np.array_equal(held_order[(s - 1) * C:s * C], rows + offset), "row order")
                offset += len(block["y"])
                labels = block["y"][rows]
                held = block["x"][rows]
                tr_rows = np.flatnonzero(block["r"] != 4)
                mu = block["x"][tr_rows].mean(0, dtype=np.float64).astype(np.float32)
                sd = np.maximum(block["x"][tr_rows].std(0, dtype=np.float64).astype(np.float32), 1e-5)
                held_n = (held - mu) / sd
                tr_n = (block["x"][tr_rows] - mu) / sd
                pred = pred_mlp[(s - 1) * C:s * C]
                check(len(pred) == C, "held rows")
                neg = 2 * pred @ feat.T - (pred ** 2).sum(1)[:, None] - bank_sq[None, :]
                templates = np.stack([tr_n[block["y"][tr_rows] == c].mean(0) for c in range(C)])
                tn = templates / np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9)
                hn = held_n / np.maximum(np.linalg.norm(held_n, axis=1, keepdims=True), 1e-9)
                result["ocf_fold4"][f"{mouse}/session{s}"] = {
                    "feature_nn": readout(neg, labels, gt, heldout, device),
                    "template_nn": readout(hn @ tn.T, labels, gt, heldout, device),
                }
                print(mouse, s, result["ocf_fold4"][f"{mouse}/session{s}"], flush=True)
    for mouse, blocks in data.items():
        parts = []
        for block in blocks:
            pooled = np.zeros((C, block["x"].shape[1]), np.float32)
            np.add.at(pooled, block["y"], block["x"])
            parts.append(pooled / 4.0)
        xa = np.concatenate(parts)
        ya = np.tile(np.arange(C), len(blocks))
        r2s = {g: [] for g in ("global", "local", "dct")}
        for fold in range(5):
            tr, te = ya % 5 != fold, ya % 5 == fold
            model = Ridge().fit_input(xa[tr]).fit_targets(feat[ya[tr]])
            r2, _ = report(model.predict(xa[te], 1e3), feat[ya[te]], feat[ya[tr]].mean(0))
            for g in r2s:
                r2s[g].append(float(np.median(r2[np.array(GROUPS) == g])))
        result.setdefault("four_repeat_average", {})[mouse] = {
            g: {"fold_median_r2": v, "mean": float(np.mean(v))} for g, v in r2s.items()
        }
        print(mouse, "4-repeat", {g: round(float(np.mean(v)), 4) for g, v in r2s.items()}, flush=True)
    result["pass_rule_eval"] = evaluate_rule(result)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("pass_rule", result["pass_rule_eval"], flush=True)


def evaluate_rule(result):
    rows = {}
    ok_mice = []
    for mouse in MICE:
        per_fold = {g: [] for g in ("global", "local", "dct")}
        for k in FOLDS:
            cell = result["folds"][f"{mouse}/fold{k}"]
            part = {g: [] for g in per_fold}
            for i, g in enumerate(GROUPS):
                part[g].append([cell["mlp_r2"][i], cell["mlp_spearman"][i], cell["null_r2"][i]])
            for g in per_fold:
                per_fold[g].append(np.array(part[g]))
        med = {g: np.median(np.array(v), axis=0) for g, v in per_fold.items()}
        glob = med["global"]
        null_all = np.concatenate([np.array(v)[:, :, 2].ravel() for v in per_fold.values()])
        null_p99 = float(np.percentile(null_all, 99))
        passing = int(((glob[:, 0] > 0.02) & (glob[:, 1] > 0.10) & (glob[:, 0] > null_p99)).sum())
        ridge_med = float(np.median([np.median(result["folds"][f"{mouse}/fold{k}"]["ridge_r2"][:8])
                                     for k in FOLDS]))
        rows[mouse] = {
            "global_median_r2": float(np.median(glob[:, 0])),
            "global_median_spearman": float(np.median(glob[:, 1])),
            "global_features_passing": passing,
            "local_median_r2": float(np.median(med["local"][:, 0])),
            "dct_median_r2": float(np.median(med["dct"][:, 0])),
            "ridge_global_median_r2": ridge_med,
            "null_p99": null_p99,
        }
        ok_mice.append(passing >= 4 and rows[mouse]["global_median_r2"] > 0.02)
    ocf = {}
    for mouse in MICE:
        for s in range(1, 5):
            cell = result["ocf_fold4"][f"{mouse}/session{s}"]
            ocf.setdefault(mouse, []).append((cell["feature_nn"]["ocf"], cell["template_nn"]["ocf"]))
    ocf_mean = {m: {"feature_nn": float(np.mean([a for a, _ in v])),
                    "template_nn": float(np.mean([b for _, b in v]))} for m, v in ocf.items()}
    ocf_gap = float(np.mean([ocf_mean[m]["feature_nn"] - ocf_mean[m]["template_nn"] for m in MICE]))
    return {
        "per_mouse": rows,
        "ocf_fold4_mean": ocf_mean,
        "ocf_gap_feature_minus_template": ocf_gap,
        "go": bool(all(ok_mice)),
        "ocf_path_rejected": bool(ocf_gap < -1.0),
    }


if __name__ == "__main__":
    main()
