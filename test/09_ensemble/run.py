"""P2: tune row-normalized ensembles of discriminative and forward-encoding scores."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.final import apply_session_normalizer, fit_session_normalizer
from src.neural import load_classifier, predict_logits
from src.retrieval import unique_assignment
from src.ridge import RidgePath, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def row_zscore(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task2", "task3"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    bank_coefficients = dct_encode(images, 16)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    split = make_split(blocks, args.task, seed=42)
    selected = np.random.default_rng(123).permutation(len(images))
    tune_pictures = np.sort(selected[:256])
    evaluation_pictures = np.sort(selected[256:768])
    if args.task == "task2":
        model_root = ROOT / "test/04_neural_seen/artifacts/task2_mlp_bin50_sessionnorm_assign"
        encoding_alpha = 1.0
    else:
        model_root = ROOT / "test/04_neural_seen/artifacts/task3_mlp_bin50_pooled_sessionnorm_assign"
        encoding_alpha = 0.01
    grouped = {mouse: [block for block in blocks if block.mouse == mouse] for mouse in {b.mouse for b in blocks}}
    rows = []

    for mouse, mouse_blocks in grouped.items():
        model = load_classifier(model_root / mouse / "classifier.pt", device)
        encoding_neural = []
        encoding_targets = []
        for block in mouse_blocks:
            block_split = split.blocks[block.key]
            if len(block_split.train_trials):
                targets, neural = aggregate_by_target(
                    block.features[block_split.train_trials], block.trial_targets[block_split.train_trials]
                )
                encoding_neural.append(np.log1p(neural))
                encoding_targets.append(targets)
        path = RidgePath().fit(bank_coefficients[np.concatenate(encoding_targets)], np.concatenate(encoding_neural))
        prototypes = path.predict(bank_coefficients, encoding_alpha)
        for block in mouse_blocks:
            block_split = split.blocks[block.key]
            if not len(block_split.validation_trials):
                continue
            raw = block.features[block_split.validation_trials]
            raw_targets = block.trial_targets[block_split.validation_trials]
            if args.task == "task2":
                targets, counts = aggregate_by_target(raw, raw_targets)
                normalizer = fit_session_normalizer(raw)
                mlp_counts = apply_session_normalizer(counts, normalizer)
            else:
                targets = raw_targets
                counts = raw
                train_raw = block.features[block_split.train_trials]
                normalizer = fit_session_normalizer(train_raw)
                mlp_counts = apply_session_normalizer(counts, normalizer)
            mlp_scores = predict_logits(model, mlp_counts, device=device)
            encoding_scores = response_scores(np.log1p(counts), prototypes, metric="mse", device=device)
            mlp_z = row_zscore(mlp_scores)
            encoding_z = row_zscore(encoding_scores)
            row = {"block": block.key, "targets": targets, "predictions": {}, "assignments": {}}
            for weight in WEIGHTS:
                combined = weight * mlp_z + (1 - weight) * encoding_z
                row["predictions"][weight] = combined.argmax(axis=1)
                row["assignments"][weight] = unique_assignment(combined)
            rows.append(row)
        print(f"scored {mouse}", flush=True)

    def tune(field):
        output = []
        for weight in WEIGHTS:
            correct = 0
            count = 0
            for row in rows:
                mask = np.isin(row["targets"], tune_pictures)
                correct += int((row[field][weight][mask] == row["targets"][mask]).sum())
                count += int(mask.sum())
            output.append({"mlp_weight": weight, "top1_accuracy": correct / count})
        return output

    independent_tuning = tune("predictions")
    assignment_tuning = tune("assignments")
    best_independent = max(independent_tuning, key=lambda row: row["top1_accuracy"])
    best_assignment = max(assignment_tuning, key=lambda row: row["top1_accuracy"])

    def evaluate(weight, field):
        prediction_parts = []
        target_parts = []
        group_parts = []
        for row in rows:
            mask = np.isin(row["targets"], evaluation_pictures)
            prediction_parts.append(images[row[field][weight][mask]])
            target_parts.append(images[row["targets"][mask]])
            group_parts.extend([row["block"]] * int(mask.sum()))
        predictions = np.concatenate(prediction_parts)
        references = np.concatenate(target_parts)
        groups = np.asarray(group_parts)
        predicted_indices = np.concatenate([
            row[field][weight][np.isin(row["targets"], evaluation_pictures)] for row in rows
        ])
        true_indices = np.concatenate([
            row["targets"][np.isin(row["targets"], evaluation_pictures)] for row in rows
        ])
        scores = score_predictions(predictions, references, device=device, batch_size=args.batch_size)
        summary = summarize_scores(scores, groups)
        summary["top1_accuracy"] = float((predicted_indices == true_indices).mean())
        return summary, predictions, references, groups

    independent_weight = best_independent["mlp_weight"]
    assignment_weight = best_assignment["mlp_weight"]
    independent, predictions, references, groups = evaluate(independent_weight, "predictions")
    assignment, assigned_predictions, _, _ = evaluate(assignment_weight, "assignments")
    output_dir = ROOT / "test/09_ensemble"
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / f"{args.task}_predictions.npz",
        predictions=predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    result = {
        "experiment": "09_ensemble",
        "name": f"{args.task}_mlp_encoding_ensemble",
        "config": {"task": args.task, "encoding_alpha": encoding_alpha, "weights": WEIGHTS},
        "protocol": {"tune_picture_count": len(tune_pictures), "evaluation_pictures": evaluation_pictures.tolist()},
        "independent_tuning": independent_tuning,
        "assignment_tuning": assignment_tuning,
        "selected_independent": best_independent,
        "selected_assignment": best_assignment,
        "evaluation": independent,
        "evaluation_assignment": assignment,
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
    }
    path = output_dir / f"results_{args.task}_ensemble.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL {path}: independent={independent['ocf_mean']:.6f} (w={independent_weight}); "
        f"assignment={assignment['ocf_mean']:.6f} (w={assignment_weight})",
        flush=True,
    )


if __name__ == "__main__":
    main()
