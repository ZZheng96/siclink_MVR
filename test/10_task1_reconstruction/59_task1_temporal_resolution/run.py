"""Exploratory: does finer temporal resolution improve Task1 continuous reconstruction?

Compares pooled ridge on 10/25/50 ms counts under one fixed protocol (alpha=1.0,
binarization threshold 0.4, GT gallery excluding validation characters).
No hyperparameter search; no training beyond closed-form ridge.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\WORK\siclink")
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions
from src.features import aggregate_by_target
from src.preprocessing import load_cached_blocks
from src.retrieval import retrieve_glyphs
from src.ridge import RidgePath, dct_decode, dct_encode
from src.splits import make_split

CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
OUT = ROOT / "test/10_task1_reconstruction/59_task1_temporal_resolution"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BINS = (10, 25, 50)
ALPHA = 1.0
THRESHOLD = 0.4
CUTOFF = 32


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def build(bin_ms, split, gt, mouse):
    train_x, train_y, eval_x, eval_y, eval_groups = [], [], [], [], []
    for block in load_cached_blocks(CACHE, split="train", bin_ms=bin_ms, mouse=mouse):
        part = split.blocks[block.key]
        targets, counts = aggregate_by_target(
            np.asarray(block.features[part.train_trials], dtype=np.float32),
            block.trial_targets[part.train_trials],
        )
        train_x.append(counts)
        train_y.append(dct_encode(gt[targets], CUTOFF))
        targets_v, counts_v = aggregate_by_target(
            np.asarray(block.features[part.validation_trials], dtype=np.float32),
            block.trial_targets[part.validation_trials],
        )
        keep = np.isin(targets_v, split.validation_pictures)
        eval_x.append(counts_v[keep])
        eval_y.append(gt[targets_v[keep]])
        eval_groups.extend([block.key] * int(keep.sum()))
    return (np.concatenate(train_x), np.concatenate(train_y),
            np.concatenate(eval_x), np.concatenate(eval_y), np.asarray(eval_groups))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pictures = np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False)
    gt = (np.asarray(pictures) >= 128).astype(np.float32)
    blocks25 = load_cached_blocks(CACHE, split="train", bin_ms=25)
    split = make_split(blocks25, "task1", seed=42)
    bank = gt[~np.isin(np.arange(len(gt)), split.validation_pictures)]

    output = {"scope": "exploratory temporal-resolution ablation; pooled closed-form ridge; "
                        "fixed alpha=1.0 and threshold=0.4; not a selection result",
              "bins": list(BINS), "alpha": ALPHA, "threshold": THRESHOLD, "cutoff": CUTOFF,
              "results": {}}
    for bin_ms in BINS:
        per_mouse = {}
        pooled_binary, pooled_retrieval = [], []
        for mouse in ("mice1", "mice2"):
            train_x, train_y, eval_x, eval_y, eval_groups = build(bin_ms, split, gt, mouse)
            print(f"bin={bin_ms} mouse={mouse} train={train_x.shape} eval={eval_x.shape}", flush=True)
            path = RidgePath().fit(np.log1p(train_x), train_y)
            predicted = path.predict(np.log1p(eval_x), ALPHA).astype(np.float32)
            gt_coeff = dct_encode(eval_y, CUTOFF)
            variance = float(dct_encode(gt[split.validation_pictures], CUTOFF).var(axis=0, dtype=np.float32).sum())
            mse = float(((predicted - gt_coeff) ** 2).sum())
            r2 = 1.0 - mse / variance

            continuous = np.asarray(dct_decode(predicted, CUTOFF), dtype=np.float32)
            clipped = np.clip(continuous, 0.0, 1.0)
            binary = (clipped >= THRESHOLD).astype(np.float32)
            binary_scores = score_predictions(binary, eval_y, device=DEVICE, batch_size=16)
            retrieved, _ = retrieve_glyphs(clipped, bank, metric="directional", blur_sigma=0.0, device=DEVICE)
            retrieval_scores = score_predictions(retrieved, eval_y, device=DEVICE, batch_size=16)
            pooled_binary.extend(binary_scores.tolist())
            pooled_retrieval.extend(retrieval_scores.tolist())
            per_mouse[mouse] = {
                "n_features": int(train_x.shape[1]),
                "train_rows": int(len(train_x)),
                "dct_r2": float(r2),
                "dct_sum_mse": mse,
                "binary_ocf": summarize(binary_scores),
                "retrieval_directional_ocf": summarize(retrieval_scores),
                "predicted_image_std": float(clipped.std()),
                "predicted_foreground_fraction": float(binary.mean()),
            }
            print(bin_ms, mouse, "r2", round(r2, 4), "binary", round(float(binary_scores.mean()), 3),
                  "retrieval", round(float(retrieval_scores.mean()), 3), flush=True)
        output["results"][str(bin_ms)] = {
            "per_mouse": per_mouse,
            "macro_binary_ocf": summarize(pooled_binary),
            "macro_retrieval_directional_ocf": summarize(pooled_retrieval),
        }
        print(bin_ms, "macro binary", round(float(np.mean(pooled_binary)), 3),
              "macro retrieval", round(float(np.mean(pooled_retrieval)), 3), flush=True)

    output["gt_reference"] = {
        "foreground_fraction": float(gt[split.validation_pictures].mean()),
        "image_std": float(gt[split.validation_pictures].std()),
    }
    (OUT / "exploratory_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote exploratory_results.json", flush=True)


if __name__ == "__main__":
    main()
