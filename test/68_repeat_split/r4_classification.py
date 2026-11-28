"""Evaluate locked FINAL models on the fourth repeat with non-OCF metrics.

Protocol deviation: the user waived official OCF scoring. The model and epoch were
locked from F1/F2 validation before any R4 label was read. This script unseals R4
once and reports classification, calibration, and pixel metrics only.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_repeat as T

OUT = T.RUN / "R4_CLASSIFICATION.json"
MD = T.RUN / "R4_RESULTS.md"
MICE = (1, 2)
DAYS = (1, 2, 3, 4)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_r4(mouse):
    xs, ys, ss = [], [], []
    for s in DAYS:
        path = T.SPLIT / f"mice{mouse}" / f"session{s}_repeat4.npz"
        with np.load(path, allow_pickle=False) as z:
            y = z["target"].astype(np.int64)
            src = z["source_trial_index"].astype(np.int64)
            rep = z["repeat_index"].astype(np.int64)
            g = z["group_ptr"].astype(np.int64)
            p = z["trial_ptr"].astype(np.int64)
            if not (y.shape == src.shape == rep.shape == (T.K,)):
                raise ValueError(f"bad R4 sample metadata: {path}")
            if not np.array_equal(np.sort(y), np.arange(T.K)):
                raise ValueError(f"R4 target is not a permutation: {path}")
            if not np.all(rep == 4) or not np.array_equal(g, np.arange(T.K + 1)):
                raise ValueError(f"R4 repeat/group mismatch: {path}")
            if len(np.unique(src)) != T.K or ((src < 0) | (src >= 4 * T.K)).any():
                raise ValueError(f"bad R4 source index: {path}")
            xs.append(T.features(mouse, p, z["spike_samples"], z["unit_index"]))
            ys.append(y)
            ss.append(np.full(T.K, s, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)


def predict(net, x, device, batch=1024):
    probs, pixels = [], []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            logits, px = net(x[i:i + batch])
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            if px is not None:
                pixels.append(px.float().cpu().numpy())
    return np.concatenate(probs), (np.concatenate(pixels) if pixels else None)


def row_metrics(y, prob, image):
    n = len(y)
    pred = prob.argmax(1)
    hit32 = (np.argpartition(-prob, 32, axis=1)[:, :32] == y[:, None]).any(1)
    nll = -np.log(np.maximum(prob[np.arange(n), y], 1e-12))
    target = IMAGES[y] >= 128
    px = image.astype(np.float32) / 255
    err = px - target
    mae = np.abs(err).mean((1, 2))
    sq = (err ** 2).mean((1, 2))
    union = ((px >= 0.5) | target).sum((1, 2))
    inter = ((px >= 0.5) & target).sum((1, 2))
    iou = np.where(union, inter / np.maximum(union, 1), 1.0)
    return dict(y=y, pred=pred, hit32=hit32, nll=nll, mae=mae, sq=sq, iou=iou,
                double_empty=union == 0)


def point(rows):
    y, pred = rows["y"], rows["pred"]
    den = np.bincount(y, minlength=T.K) + np.bincount(pred, minlength=T.K)
    tp = np.bincount(y[y == pred], minlength=T.K)
    return {
        "n": int(len(y)),
        "top1": float((pred == y).mean()),
        "recall32": float(rows["hit32"].mean()),
        "macro_f1": float(np.mean(2 * tp / np.maximum(den, 1))),
        "nll": float(rows["nll"].mean()),
        "mae": float(rows["mae"].mean()),
        "rmse": float(np.sqrt(rows["sq"].mean())),
        "iou": float(rows["iou"].mean()),
        "double_empty": int(rows["double_empty"].sum()),
    }


def char_sums(rows):
    y = rows["y"]
    count = np.bincount(y, minlength=T.K).astype(np.float64)
    return {
        "count": count,
        "top1": np.bincount(y, weights=(rows["pred"] == y), minlength=T.K),
        "recall32": np.bincount(y, weights=rows["hit32"], minlength=T.K),
        "nll": np.bincount(y, weights=rows["nll"], minlength=T.K),
        "mae": np.bincount(y, weights=rows["mae"], minlength=T.K),
        "sq": np.bincount(y, weights=rows["sq"], minlength=T.K),
        "iou": np.bincount(y, weights=rows["iou"], minlength=T.K),
    }


def bootstrap(sums_by_mouse, seed=31415, draws=2000):
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, T.K, size=(draws, T.K))
    out = {k: [] for k in ("top1", "recall32", "nll", "mae", "rmse", "iou")}
    for sample in samples:
        w = np.bincount(sample, minlength=T.K)
        mouse_vals = {k: [] for k in out}
        for sums in sums_by_mouse:
            den = float(np.dot(w, sums["count"]))
            for key in ("top1", "recall32", "nll", "mae", "iou"):
                mouse_vals[key].append(float(np.dot(w, sums[key]) / den))
            mouse_vals["rmse"].append(float(np.sqrt(np.dot(w, sums["sq"]) / den)))
        for key in out:
            out[key].append(float(np.mean(mouse_vals[key])))
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    done = json.loads((T.RUN / "FINAL_DONE.json").read_text(encoding="utf-8"))
    lock = json.loads((T.RUN / "DEV_LOCK.json").read_text(encoding="utf-8"))
    epochs = lock["selected_epochs"]

    global IMAGES
    with np.load(T.PICTURES, allow_pickle=False) as z:
        IMAGES = z["images"]
    if IMAGES.shape != (T.K, 64, 64) or IMAGES.dtype != np.uint8:
        raise ValueError("invalid picture library")

    hashes = {}
    for group in done["groups"]:
        hashes[group["scaler_sha256"]] = True
        for ck in group["checkpoints"]:
            hashes[ck["sha256"]] = True
    for mouse in MICE:
        assert sha(T.directory("FINAL", mouse) / "scaler.npz") in hashes
        for model in ("B", "C"):
            for seed in T.SEEDS:
                assert sha(T.checkpoint("FINAL", mouse, model, seed, epochs[model])) in hashes

    result = {
        "protocol": "repeat-holdout v1",
        "deviation": "official OCF not computed at user request; R4 unsealed for non-OCF metrics only",
        "utc": datetime.now(timezone.utc).isoformat(),
        "selected_epochs": epochs,
        "final_done_sha256": sha(T.RUN / "FINAL_DONE.json"),
        "dev_lock_sha256": sha(T.RUN / "DEV_LOCK.json"),
        "r4_files": {f"mice{m}/session{s}_repeat4.npz": sha(T.SPLIT / f"mice{m}" / f"session{s}_repeat4.npz")
                     for m in MICE for s in DAYS},
        "models": {},
    }

    sums = {}
    for model in ("B", "C"):
        per_mouse = {}
        per_seed = {}
        for mouse in MICE:
            x, y, session = load_r4(mouse)
            mean, scale = T.load_scaler("FINAL", mouse)
            xt = torch.from_numpy((x - mean) / scale).to(device)
            prob_sum = np.zeros((len(y), T.K), np.float32)
            pixel_sum = np.zeros((len(y), 4096), np.float32) if model == "C" else None
            seed_rows = []
            for seed in T.SEEDS:
                net = T.load_model("FINAL", mouse, model, seed, epochs[model], device)
                prob, pixels = predict(net, xt, device)
                prob_sum += prob
                if pixels is not None:
                    pixel_sum += pixels
                image = (np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8).reshape(-1, 64, 64)
                         if pixels is not None else IMAGES[prob.argmax(1)])
                seed_rows.append(point(row_metrics(y, prob, image)))
                del net, pixels
            prob = prob_sum / len(T.SEEDS)
            if pixel_sum is not None:
                image = np.rint(np.clip(pixel_sum / len(T.SEEDS), 0, 1) * 255).astype(np.uint8).reshape(-1, 64, 64)
            else:
                image = IMAGES[prob.argmax(1)]
            rows = row_metrics(y, prob, image)
            rows["session"] = session
            day = {f"session{s}": point({k: (v[session == s] if isinstance(v, np.ndarray) else v)
                                         for k, v in rows.items() if k != "session"}) for s in DAYS}
            per_mouse[mouse] = {"metrics": point(rows), "mouse_session": day}
            per_seed[mouse] = {
                "top1_mean": float(np.mean([r["top1"] for r in seed_rows])),
                "top1_sd": float(np.std([r["top1"] for r in seed_rows], ddof=1)),
                "nll_mean": float(np.mean([r["nll"] for r in seed_rows])),
                "nll_sd": float(np.std([r["nll"] for r in seed_rows], ddof=1)),
                "mae_mean": float(np.mean([r["mae"] for r in seed_rows])),
                "mae_sd": float(np.std([r["mae"] for r in seed_rows], ddof=1)),
                "iou_mean": float(np.mean([r["iou"] for r in seed_rows])),
                "iou_sd": float(np.std([r["iou"] for r in seed_rows], ddof=1)),
                "per_seed": [{k: r[k] for k in ("top1", "recall32", "nll", "mae", "rmse", "iou")}
                             for r in seed_rows],
            }
            sums[(model, mouse)] = char_sums(rows)
            print(f"{model} m{mouse}: top1={per_mouse[mouse]['metrics']['top1']:.4f} "
                  f"r32={per_mouse[mouse]['metrics']['recall32']:.4f} "
                  f"nll={per_mouse[mouse]['metrics']['nll']:.4f} "
                  f"mae={per_mouse[mouse]['metrics']['mae']:.4f} "
                  f"iou={per_mouse[mouse]['metrics']['iou']:.4f}", flush=True)
            del x, xt, prob_sum, pixel_sum, prob, image, rows

        macro = {k: float(np.mean([per_mouse[m]["metrics"][k] for m in MICE]))
                 for k in per_mouse[MICE[0]]["metrics"] if k not in ("n", "double_empty")}
        macro["double_empty"] = int(sum(per_mouse[m]["metrics"]["double_empty"] for m in MICE))
        result["models"][model] = {
            "mouse": {str(m): per_mouse[m]["metrics"] for m in MICE},
            "mouse_session": {f"{m}:{s}": per_mouse[m]["mouse_session"][f"session{s}"]
                              for m in MICE for s in DAYS},
            "macro": macro,
            "bootstrap95_macro": bootstrap([sums[(model, m)] for m in MICE]),
            "per_seed": {str(m): per_seed[m] for m in MICE},
        }

    def flat(model):
        return {k: result["models"][model]["macro"][k] for k in ("top1", "recall32", "nll", "mae", "rmse", "iou")}
    result["delta_C_minus_B"] = {k: float(flat("C")[k] - flat("B")[k]) for k in flat("B")}
    result["delta_note"] = "positive is better for top1/recall32/iou; negative is better for nll/mae/rmse"
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# R4 第 4 次 repeat 结果（非官方 OCF）",
        "",
        f"生成时间：{result['utc']}；锁定轮数：B={epochs['B']}，C={epochs['C']}。"
        "模型与轮数在读取 R4 标签前已锁定；本表为一次性评估。",
        "",
        "| 模型 | 鼠 | Top-1 | Recall@32 | macro-F1 | NLL | MAE | RMSE | 前景 IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("B", "C"):
        for mouse in MICE:
            r = result["models"][model]["mouse"][str(mouse)]
            lines.append(f"| {model} | {mouse} | {r['top1']:.4f} | {r['recall32']:.4f} | "
                         f"{r['macro_f1']:.4f} | {r['nll']:.4f} | {r['mae']:.4f} | "
                         f"{r['rmse']:.4f} | {r['iou']:.4f} |")
        r = result["models"][model]["macro"]
        ci = result["models"][model]["bootstrap95_macro"]
        lines.append(f"| {model} | 宏平均 | {r['top1']:.4f} [{ci['top1'][0]:.4f}, {ci['top1'][1]:.4f}] | "
                     f"{r['recall32']:.4f} | {r['macro_f1']:.4f} | {r['nll']:.4f} | "
                     f"{r['mae']:.4f} | {r['rmse']:.4f} | {r['iou']:.4f} |")
    lines += ["", "## 逐日（鼠:session）Top-1 / NLL", "",
              "| 模型 | 组 | Top-1 | Recall@32 | NLL | IoU |", "|---|---|---:|---:|---:|---:|"]
    for model in ("B", "C"):
        for key, r in result["models"][model]["mouse_session"].items():
            lines.append(f"| {model} | {key} | {r['top1']:.4f} | {r['recall32']:.4f} | "
                         f"{r['nll']:.4f} | {r['iou']:.4f} |")
    lines += ["", "## 种子稳定性", "",
              "| 模型 | 鼠 | Top-1 均值±SD | NLL 均值±SD | MAE 均值±SD | IoU 均值±SD |",
              "|---|---:|---:|---:|---:|---:|"]
    for model in ("B", "C"):
        for mouse in MICE:
            r = result["models"][model]["per_seed"][str(mouse)]
            lines.append(f"| {model} | {mouse} | {r['top1_mean']:.4f} ± {r['top1_sd']:.4f} | "
                         f"{r['nll_mean']:.4f} ± {r['nll_sd']:.4f} | "
                         f"{r['mae_mean']:.4f} ± {r['mae_sd']:.4f} | "
                         f"{r['iou_mean']:.4f} ± {r['iou_sd']:.4f} |")
    d = result["delta_C_minus_B"]
    lines += ["", "## C − B（宏平均）", "",
              f"Top-1 {d['top1']:+.4f}，Recall@32 {d['recall32']:+.4f}，NLL {d['nll']:+.4f}，"
              f"MAE {d['mae']:+.4f}，RMSE {d['rmse']:+.4f}，IoU {d['iou']:+.4f}。",
              "", "注：Top-1/Recall@32/IoU 越高越好；NLL/MAE/RMSE 越低越好。"
              "未计算官方 OCF；R4 已解封，后续 OCF 只能作为探索性结果。"]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"macro": {m: result["models"][m]["macro"] for m in ("B", "C")},
                      "delta": d}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
