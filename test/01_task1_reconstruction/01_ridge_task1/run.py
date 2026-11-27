"""P0 Task 1: character-held-out repeat-averaged Ridge/DCT decoding."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores, sweep_thresholds
from src.features import aggregate_by_target, load_feature_blocks
from src.ridge import RidgePath, dct_decode, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="bin50_dct32_log")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--cutoff", type=int, default=32)
    parser.add_argument("--transform", choices=("raw", "log"), default="log")
    parser.add_argument("--mouse", choices=("mice1", "mice2"))
    parser.add_argument("--tune-pictures", type=int, default=64)
    parser.add_argument("--threshold-pictures", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=args.bin_ms, mouse=args.mouse)
    split = make_split(blocks, "task1", seed=42)
    rng = np.random.default_rng(123)
    shuffled = rng.permutation(split.validation_pictures)
    tune_pictures = np.sort(shuffled[:args.tune_pictures])
    threshold_pictures = np.sort(shuffled[:args.threshold_pictures])
    evaluation_pictures = np.sort(shuffled[args.tune_pictures:])
    if not len(evaluation_pictures):
        raise ValueError("At least one evaluation picture is required.")

    fitted = []
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    for block in blocks:
        block_split = split.blocks[block.key]
        train_targets, train_x = aggregate_by_target(
            block.features[block_split.train_trials], block.trial_targets[block_split.train_trials]
        )
        validation_targets, validation_x = aggregate_by_target(
            block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
        )
        if args.transform == "log":
            train_x = np.log1p(train_x)
            validation_x = np.log1p(validation_x)
        train_y = dct_encode(images[train_targets], args.cutoff)
        path = RidgePath().fit(train_x, train_y)
        fitted.append({
            "block": block,
            "path": path,
            "validation_x": validation_x,
            "validation_targets": validation_targets,
        })
        print(f"fit {block.key}: train={len(train_x)} validation={len(validation_x)}", flush=True)

    def decode(rows, alpha, pictures=None):
        predictions = []
        targets = []
        groups = []
        for row in rows:
            mask = np.isin(row["validation_targets"], pictures) if pictures is not None else slice(None)
            decoded = dct_decode(row["path"].predict(row["validation_x"][mask], alpha), args.cutoff)
            predictions.append(decoded)
            selected_targets = row["validation_targets"][mask]
            targets.append(images[selected_targets])
            groups.extend([row["block"].key] * len(selected_targets))
        return np.concatenate(predictions), np.concatenate(targets), np.asarray(groups)

    alpha_rows = []
    tune_targets = None
    for alpha in alphas:
        predictions, targets, _ = decode(fitted, alpha, tune_pictures)
        mse = float(np.mean((np.clip(predictions, 0, 1) - targets) ** 2))
        alpha_rows.append({"alpha": alpha, "tune_pixel_mse": mse})
        print(f"alpha={alpha:g}: tune MSE={mse:.8f}", flush=True)
        if tune_targets is None:
            tune_targets = targets
    shortlisted = [row["alpha"] for row in sorted(alpha_rows, key=lambda row: row["tune_pixel_mse"])[:2]]

    sweeps = []
    thresholds = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
    for alpha in shortlisted:
        predictions, targets, _ = decode(fitted, alpha, threshold_pictures)
        for row in sweep_thresholds(predictions, targets, thresholds, device=device, batch_size=args.batch_size):
            row["alpha"] = alpha
            sweeps.append(row)
            print(f"sweep {row}", flush=True)
    best = max(sweeps, key=lambda row: row["ocf_mean"])

    evaluation_predictions, evaluation_targets, evaluation_groups = decode(
        fitted, best["alpha"], evaluation_pictures
    )
    evaluation_scores = score_predictions(
        evaluation_predictions,
        evaluation_targets,
        threshold=best["threshold"],
        device=device,
        batch_size=args.batch_size,
    )
    summary = summarize_scores(evaluation_scores, evaluation_groups)
    continuous_scores = score_predictions(
        evaluation_predictions,
        evaluation_targets,
        device=device,
        batch_size=args.batch_size,
    )
    summary["continuous_without_binarization"] = float(continuous_scores.mean())

    output_dir = ROOT / "test/01_task1_reconstruction/01_ridge_task1"
    artifacts = output_dir / "artifacts" / args.name
    artifacts.mkdir(parents=True, exist_ok=True)
    binary_predictions = (evaluation_predictions >= best["threshold"]).astype(np.uint8) * 255
    np.savez_compressed(
        artifacts / "evaluation_predictions.npz",
        predictions=binary_predictions,
        targets=evaluation_targets.astype(np.uint8) * 255,
        group_ids=evaluation_groups,
    )
    result = {
        "experiment": "01_ridge_task1",
        "name": args.name,
        "config": vars(args),
        "protocol": {
            **split.metadata(),
            "tune_pictures": tune_pictures.tolist(),
            "threshold_pictures": threshold_pictures.tolist(),
            "evaluation_picture_count": len(evaluation_pictures),
            "note": "Public training has 4 repeats; this is a proxy for, not equivalent to, Task 1's 20 repeats.",
        },
        "device": device,
        "alpha_path": alpha_rows,
        "threshold_sweep": sweeps,
        "selected": best,
        "evaluation": summary,
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "elapsed_seconds": time.time() - started,
    }
    result_path = output_dir / f"results_{args.name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"FINAL {result_path}: {summary['ocf_mean']:.6f} OCF; selected={best}", flush=True)


if __name__ == "__main__":
    main()
