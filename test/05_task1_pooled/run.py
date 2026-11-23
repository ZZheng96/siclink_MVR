"""P1 Task 1: pool sessions per mouse and compare pooled Ridge with an MLP decoder."""

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
from src.neural import SpikeRegressor, fit_regressor, load_regressor, predict_logits
from src.ridge import PCABasis, RidgePath, dct_decode, dct_encode
from src.retrieval import retrieve_glyphs
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("ridge", "mlp"))
    parser.add_argument("--name")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--cutoff", type=int, default=32)
    parser.add_argument("--basis", choices=("dct", "pca"), default="dct")
    parser.add_argument("--mouse", choices=("mice1", "mice2"))
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-singles", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tune-pictures", type=int, default=64)
    parser.add_argument("--threshold-pictures", type=int, default=16)
    parser.add_argument("--ocf-batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name or f"task1_pooled_{args.model}_{args.basis}{args.cutoff}"
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
    if args.basis == "pca":
        training_pictures = np.flatnonzero(~np.isin(np.arange(len(images)), split.validation_pictures))
        image_basis = PCABasis.fit(images[training_pictures], rank=args.cutoff, device=device)
        encode_images = image_basis.encode
        decode_images = image_basis.decode
    else:
        def encode_images(values):
            return dct_encode(values, args.cutoff)

        def decode_images(values):
            return dct_decode(values, args.cutoff)
    output_dir = ROOT / "test/05_task1_pooled"
    artifacts = output_dir / "artifacts" / name
    artifacts.mkdir(parents=True, exist_ok=True)

    grouped = {mouse: [block for block in blocks if block.mouse == mouse] for mouse in {b.mouse for b in blocks}}
    rows = []
    for group, group_blocks in grouped.items():
        train_x_parts = []
        train_target_parts = []
        validation_rows = []
        for block in group_blocks:
            block_split = split.blocks[block.key]
            train_targets, train_x = aggregate_by_target(
                block.features[block_split.train_trials], block.trial_targets[block_split.train_trials]
            )
            validation_targets, validation_x = aggregate_by_target(
                block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
            )
            train_x_parts.append(train_x)
            train_target_parts.append(train_targets)
            if args.train_singles:
                train_x_parts.append(block.features[block_split.train_trials])
                train_target_parts.append(block.trial_targets[block_split.train_trials])
            validation_rows.append([block.key, validation_x, validation_targets])
        train_x = np.concatenate(train_x_parts)
        train_targets = np.concatenate(train_target_parts)
        train_y = encode_images(images[train_targets])
        row = {"group": group, "validation": validation_rows}
        if args.model == "ridge":
            row["model"] = RidgePath().fit(np.log1p(train_x), train_y)
        else:
            tune_x = []
            tune_y = []
            for _, validation_x, validation_targets in validation_rows:
                mask = np.isin(validation_targets, tune_pictures)
                tune_x.append(validation_x[mask])
                tune_y.append(encode_images(images[validation_targets[mask]]))
            model = SpikeRegressor(train_x.shape[1], train_y.shape[1], args.hidden, args.dropout)
            history = fit_regressor(
                model,
                (train_x, train_y),
                (np.concatenate(tune_x), np.concatenate(tune_y)),
                artifacts / group,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=device,
                seed=42,
            )
            row["model"] = artifacts / group / "regressor.pt"
            row["best"] = min(history, key=lambda item: item["validation_loss"])
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
        print(f"fit {group}: train={len(train_x)}", flush=True)

    def decode(model_value, pictures):
        predictions = []
        targets = []
        groups = []
        for row in rows:
            for key, validation_x, validation_targets in row["validation"]:
                mask = np.isin(validation_targets, pictures)
                if args.model == "ridge":
                    coefficients = row["model"].predict(np.log1p(validation_x[mask]), model_value)
                else:
                    regressor = load_regressor(row["model"], device)
                    coefficients = predict_logits(regressor, validation_x[mask], args.batch_size, device)
                predictions.append(decode_images(coefficients))
                selected_targets = validation_targets[mask]
                targets.append(images[selected_targets])
                groups.extend([key] * len(selected_targets))
        return np.concatenate(predictions), np.concatenate(targets), np.asarray(groups)

    alpha_rows = []
    if args.model == "ridge":
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            predictions, targets, _ = decode(alpha, tune_pictures)
            mse = float(np.mean((np.clip(predictions, 0, 1) - targets) ** 2))
            alpha_rows.append({"alpha": alpha, "tune_pixel_mse": mse})
            print(f"alpha={alpha:g}: MSE={mse:.8f}", flush=True)
        selected_alphas = [row["alpha"] for row in sorted(alpha_rows, key=lambda row: row["tune_pixel_mse"])[:2]]
    else:
        selected_alphas = [None]

    sweeps = []
    for alpha in selected_alphas:
        predictions, targets, _ = decode(alpha, threshold_pictures)
        for result in sweep_thresholds(
            predictions, targets, (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6),
            device=device, batch_size=args.ocf_batch_size,
        ):
            result["alpha"] = alpha
            sweeps.append(result)
    best = max(sweeps, key=lambda row: row["ocf_mean"])
    evaluation_predictions, evaluation_targets, evaluation_groups = decode(best["alpha"], evaluation_pictures)
    scores = score_predictions(
        evaluation_predictions, evaluation_targets, threshold=best["threshold"],
        device=device, batch_size=args.ocf_batch_size,
    )
    summary = summarize_scores(scores, evaluation_groups)
    continuous = score_predictions(
        evaluation_predictions, evaluation_targets, device=device, batch_size=args.ocf_batch_size
    )
    summary["continuous_without_binarization"] = float(continuous.mean())

    bank_indices = np.flatnonzero(~np.isin(np.arange(len(images)), split.validation_pictures))
    bank = images[bank_indices]
    retrieval_tune, retrieval_tune_targets, _ = decode(best["alpha"], threshold_pictures)
    retrieval_sweeps = []
    for metric, blur_sigma in (("cosine", 0.0), ("cosine", 1.0), ("cosine", 2.0), ("mse", 1.0), ("directional", 0.0)):
        retrieved, _ = retrieve_glyphs(
            retrieval_tune, bank, metric=metric, blur_sigma=blur_sigma, device=device
        )
        retrieval_scores = score_predictions(
            retrieved, retrieval_tune_targets, device=device, batch_size=args.ocf_batch_size
        )
        retrieval_sweeps.append({
            "metric": metric,
            "blur_sigma": blur_sigma,
            "ocf_mean": float(retrieval_scores.mean()),
        })
    best_retrieval = max(retrieval_sweeps, key=lambda row: row["ocf_mean"])
    retrieved_evaluation, _ = retrieve_glyphs(
        evaluation_predictions,
        bank,
        metric=best_retrieval["metric"],
        blur_sigma=best_retrieval["blur_sigma"],
        device=device,
    )
    retrieved_scores = score_predictions(
        retrieved_evaluation, evaluation_targets, device=device, batch_size=args.ocf_batch_size
    )
    retrieval_summary = summarize_scores(retrieved_scores, evaluation_groups)

    np.savez_compressed(
        artifacts / "evaluation_predictions.npz",
        predictions=(evaluation_predictions >= best["threshold"]).astype(np.uint8) * 255,
        targets=evaluation_targets.astype(np.uint8) * 255,
        group_ids=evaluation_groups,
    )
    np.savez_compressed(
        artifacts / "retrieval_predictions.npz",
        predictions=retrieved_evaluation.astype(np.uint8) * 255,
        targets=evaluation_targets.astype(np.uint8) * 255,
        group_ids=evaluation_groups,
    )
    result = {
        "experiment": "05_task1_pooled",
        "name": name,
        "config": vars(args),
        "protocol": {
            **split.metadata(),
            "tune_pictures": tune_pictures.tolist(),
            "threshold_pictures": threshold_pictures.tolist(),
            "evaluation_picture_count": len(evaluation_pictures),
            "pooling": "All four sessions per mouse; character identities remain disjoint.",
            "single_trial_augmentation": args.train_singles,
        },
        "training": [{"group": row["group"], "best": row.get("best")} for row in rows],
        "alpha_path": alpha_rows,
        "threshold_sweep": sweeps,
        "selected": {"direct": best, "retrieval": best_retrieval},
        "retrieval_sweep": retrieval_sweeps,
        "evaluation": {"direct": summary, "retrieval": retrieval_summary},
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "elapsed_seconds": time.time() - started,
    }
    result_path = output_dir / f"results_{name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL {result_path}: direct={summary['ocf_mean']:.6f}; "
        f"retrieval={retrieval_summary['ocf_mean']:.6f}; selected={best}",
        flush=True,
    )


if __name__ == "__main__":
    main()
