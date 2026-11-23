"""P0 Tasks 2/3: Ridge/DCT decoding with direct and crisp-bank outputs."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores, sweep_thresholds
from src.features import aggregate_by_target, load_feature_blocks
from src.retrieval import retrieve_glyphs
from src.ridge import RidgePath, dct_decode, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task2", "task3"))
    parser.add_argument("--name")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--cutoff", type=int, default=32)
    parser.add_argument("--transform", choices=("raw", "log"), default="log")
    parser.add_argument("--session-norm", action="store_true")
    parser.add_argument("--mouse", choices=("mice1", "mice2"))
    parser.add_argument("--tune-pictures", type=int, default=256)
    parser.add_argument("--threshold-pictures", type=int, default=64)
    parser.add_argument("--evaluation-pictures", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def normalize(features):
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    return (features - mean) / np.maximum(scale, 1e-6)


def main():
    args = parse_args()
    name = args.name or f"{args.task}_ridge_{args.transform}" + ("_sessionnorm" if args.session_norm else "")
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=args.bin_ms, mouse=args.mouse)
    split = make_split(blocks, args.task, seed=42)
    rng = np.random.default_rng(123)
    selected = rng.permutation(len(images))
    tune_pictures = np.sort(selected[:args.tune_pictures])
    threshold_pictures = np.sort(selected[:args.threshold_pictures])
    evaluation_pictures = np.sort(selected[args.tune_pictures:args.tune_pictures + args.evaluation_pictures])

    grouped = ({mouse: [block for block in blocks if block.mouse == mouse] for mouse in {b.mouse for b in blocks}}
               if args.task == "task2" else {block.key: [block] for block in blocks})
    rows = []
    for group, group_blocks in grouped.items():
        train_x_parts = []
        train_y_parts = []
        validation_rows = []
        for block in group_blocks:
            block_split = split.blocks[block.key]
            if args.task == "task2":
                if len(block_split.validation_trials):
                    validation_targets, validation_x = aggregate_by_target(
                        block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
                    )
                else:
                    validation_targets = np.empty(0, dtype=block.targets.dtype)
                    validation_x = np.empty((0, block.n_features), dtype=np.float32)
                if len(block_split.train_trials):
                    train_targets, train_x = aggregate_by_target(
                        block.features[block_split.train_trials], block.trial_targets[block_split.train_trials]
                    )
                    if args.transform == "log":
                        train_x = np.log1p(train_x)
                    if args.session_norm:
                        train_x = normalize(train_x)
                    train_x_parts.append(train_x)
                    train_y_parts.append(train_targets)
            else:
                validation_x = block.features[block_split.validation_trials]
                validation_targets = block.trial_targets[block_split.validation_trials]
                train_x = block.features[block_split.train_trials]
                train_targets = block.trial_targets[block_split.train_trials]
                if args.transform == "log":
                    train_x = np.log1p(train_x)
                if args.session_norm:
                    train_x = normalize(train_x)
                train_x_parts.append(train_x)
                train_y_parts.append(train_targets)
            if len(validation_targets):
                validation_rows.append([block.key, validation_x, validation_targets])
        train_x = np.concatenate(train_x_parts)
        train_targets = np.concatenate(train_y_parts)
        train_y = dct_encode(images[train_targets], args.cutoff)
        path = RidgePath().fit(train_x, train_y)
        for validation_row in validation_rows:
            validation_x = validation_row[1]
            if args.transform == "log":
                validation_x = np.log1p(validation_x)
            if args.session_norm:
                validation_x = normalize(validation_x)
            validation_row[1] = validation_x
        rows.append({"group": group, "path": path, "validation": validation_rows})
        print(f"fit {group}: train={len(train_x)}", flush=True)

    def decode(alpha, pictures):
        predictions = []
        targets = []
        groups = []
        for row in rows:
            for key, validation_x, validation_targets in row["validation"]:
                mask = np.isin(validation_targets, pictures)
                selected_targets = validation_targets[mask]
                predictions.append(dct_decode(row["path"].predict(validation_x[mask], alpha), args.cutoff))
                targets.append(images[selected_targets])
                groups.extend([key] * len(selected_targets))
        return np.concatenate(predictions), np.concatenate(targets), np.asarray(groups)

    alphas = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    alpha_rows = []
    for alpha in alphas:
        predictions, targets, _ = decode(alpha, tune_pictures)
        mse = float(np.mean((np.clip(predictions, 0, 1) - targets) ** 2))
        retrieved, indices = retrieve_glyphs(
            predictions, images, metric="cosine", blur_sigma=1.0, device=device
        )
        true_indices = np.concatenate([
            validation_targets[np.isin(validation_targets, tune_pictures)]
            for row in rows for _, _, validation_targets in row["validation"]
        ])
        accuracy = float((indices == true_indices).mean())
        alpha_rows.append({"alpha": alpha, "tune_pixel_mse": mse, "tune_retrieval_top1": accuracy})
        print(f"alpha={alpha:g}: MSE={mse:.8f}, retrieval top1={accuracy:.6f}", flush=True)
    direct_alpha = min(alpha_rows, key=lambda row: row["tune_pixel_mse"])["alpha"]
    retrieval_alpha = max(alpha_rows, key=lambda row: row["tune_retrieval_top1"])["alpha"]

    direct_tune, direct_targets, _ = decode(direct_alpha, threshold_pictures)
    thresholds = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
    threshold_rows = sweep_thresholds(
        direct_tune, direct_targets, thresholds, device=device, batch_size=args.batch_size
    )
    best_threshold = max(threshold_rows, key=lambda row: row["ocf_mean"])

    retrieval_tune, retrieval_targets, _ = decode(retrieval_alpha, tune_pictures)
    true_indices = np.concatenate([
        validation_targets[np.isin(validation_targets, tune_pictures)]
        for row in rows for _, _, validation_targets in row["validation"]
    ])
    retrieval_configs = []
    for metric, blur_sigma in (("cosine", 0.0), ("cosine", 1.0), ("cosine", 2.0), ("mse", 1.0)):
        _, indices = retrieve_glyphs(
            retrieval_tune, images, metric=metric, blur_sigma=blur_sigma, device=device
        )
        retrieval_configs.append({
            "metric": metric,
            "blur_sigma": blur_sigma,
            "tune_top1": float((indices == true_indices).mean()),
        })
    best_retrieval = max(retrieval_configs, key=lambda row: row["tune_top1"])

    evaluation_predictions, evaluation_targets, evaluation_groups = decode(retrieval_alpha, evaluation_pictures)
    retrieved_predictions, retrieved_indices = retrieve_glyphs(
        evaluation_predictions,
        images,
        metric=best_retrieval["metric"],
        blur_sigma=best_retrieval["blur_sigma"],
        device=device,
    )
    true_evaluation_indices = np.concatenate([
        validation_targets[np.isin(validation_targets, evaluation_pictures)]
        for row in rows for _, _, validation_targets in row["validation"]
    ])
    retrieval_scores = score_predictions(
        retrieved_predictions, evaluation_targets, device=device, batch_size=args.batch_size
    )
    retrieval_summary = summarize_scores(retrieval_scores, evaluation_groups)
    retrieval_summary["top1_accuracy"] = float((retrieved_indices == true_evaluation_indices).mean())

    direct_evaluation, direct_evaluation_targets, direct_groups = decode(direct_alpha, evaluation_pictures)
    assert np.array_equal(direct_evaluation_targets, evaluation_targets)
    direct_scores = score_predictions(
        direct_evaluation,
        evaluation_targets,
        threshold=best_threshold["threshold"],
        device=device,
        batch_size=args.batch_size,
    )
    direct_summary = summarize_scores(direct_scores, direct_groups)

    output_dir = ROOT / "test/03_seen_ridge"
    artifacts = output_dir / "artifacts" / name
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / "retrieval_predictions.npz",
        predictions=retrieved_predictions.astype(np.uint8) * 255,
        targets=evaluation_targets.astype(np.uint8) * 255,
        group_ids=evaluation_groups,
    )
    np.savez_compressed(
        artifacts / "direct_predictions.npz",
        predictions=(direct_evaluation >= best_threshold["threshold"]).astype(np.uint8) * 255,
        targets=evaluation_targets.astype(np.uint8) * 255,
        group_ids=evaluation_groups,
    )
    result = {
        "experiment": "03_seen_ridge",
        "name": name,
        "config": vars(args),
        "protocol": {
            **split.metadata(),
            "tune_picture_count": len(tune_pictures),
            "threshold_picture_count": len(threshold_pictures),
            "evaluation_pictures": evaluation_pictures.tolist(),
        },
        "device": device,
        "alpha_path": alpha_rows,
        "threshold_sweep": threshold_rows,
        "retrieval_config_path": retrieval_configs,
        "selected": {
            "direct_alpha": direct_alpha,
            "direct_threshold": best_threshold,
            "retrieval_alpha": retrieval_alpha,
            "retrieval": best_retrieval,
        },
        "evaluation": {"direct": direct_summary, "retrieval": retrieval_summary},
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "elapsed_seconds": time.time() - started,
    }
    result_path = output_dir / f"results_{name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL {result_path}: direct={direct_summary['ocf_mean']:.6f}, "
        f"retrieval={retrieval_summary['ocf_mean']:.6f}, top1={retrieval_summary['top1_accuracy']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
