"""P2: unlabeled iterative diagonal CORAL for the encoding ensemble branch."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
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
BASELINE_OCF = {"task2": 23.329312169146995, "task3": 16.57463190021421}
BASELINE_VARIANT = (0, 0.0)
ROUNDS = (1, 2, 3)
STRENGTHS = (0.5, 1.0)
BASELINE_ATOL = 1e-5


def row_zscore(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)


def diagonal_coral(features, assigned, prototypes, strength):
    pseudo_targets = prototypes[assigned]
    mean_x = features.mean(axis=0)
    std_x = features.std(axis=0)
    mean_y = pseudo_targets.mean(axis=0)
    std_y = pseudo_targets.std(axis=0)
    transformed = (features - mean_x) / np.maximum(std_x, 1e-6) * std_y + mean_y
    return (1 - strength) * features + strength * transformed


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
    selected_pictures = np.random.default_rng(123).permutation(len(images))
    tune_pictures = np.sort(selected_pictures[:256])
    evaluation_pictures = np.sort(selected_pictures[256:768])
    if args.task == "task2":
        model_root = ROOT / "test/20_task2_cross_day/04_neural_seen/artifacts/task2_mlp_bin50_sessionnorm_assign"
        encoding_alpha = 1.0
    else:
        model_root = ROOT / "test/20_task2_cross_day/04_neural_seen/artifacts/task3_mlp_bin50_pooled_sessionnorm_assign"
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
            features = np.log1p(counts)
            mlp_z = row_zscore(predict_logits(model, mlp_counts, device=device))
            encoding_z = row_zscore(response_scores(features, prototypes, metric="mse", device=device))
            combined = 0.5 * mlp_z + 0.5 * encoding_z
            rows.append({
                "block": block.key,
                "targets": targets,
                "features": features,
                "prototypes": prototypes,
                "mlp_z": mlp_z,
                "assignments": {BASELINE_VARIANT: unique_assignment(combined)},
            })
        print(f"scored {mouse}", flush=True)

    def evaluate(variant, pictures):
        prediction_parts = []
        target_parts = []
        group_parts = []
        predicted_index_parts = []
        true_index_parts = []
        for row in rows:
            mask = np.isin(row["targets"], pictures)
            indices = row["assignments"][variant]
            prediction_parts.append(images[indices[mask]])
            target_parts.append(images[row["targets"][mask]])
            group_parts.extend([row["block"]] * int(mask.sum()))
            predicted_index_parts.append(indices[mask])
            true_index_parts.append(row["targets"][mask])
        predictions = np.concatenate(prediction_parts)
        references = np.concatenate(target_parts)
        groups = np.asarray(group_parts)
        predicted_indices = np.concatenate(predicted_index_parts)
        true_indices = np.concatenate(true_index_parts)
        scores = score_predictions(predictions, references, device=device, batch_size=args.batch_size)
        summary = summarize_scores(scores, groups)
        summary["top1_accuracy"] = float((predicted_indices == true_indices).mean())
        return summary, predictions, references, groups

    baseline, _, _, _ = evaluate(BASELINE_VARIANT, evaluation_pictures)
    expected_baseline = BASELINE_OCF[args.task]
    baseline_error = abs(baseline["ocf_mean"] - expected_baseline)
    print(
        f"BASELINE {args.task}: actual={baseline['ocf_mean']:.10f}; "
        f"expected={expected_baseline:.10f}; abs_error={baseline_error:.3g}",
        flush=True,
    )
    if baseline_error > BASELINE_ATOL:
        raise RuntimeError(
            f"Baseline OCF mismatch for {args.task}: actual={baseline['ocf_mean']}, "
            f"expected={expected_baseline}, abs_error={baseline_error}, tolerance={BASELINE_ATOL}. "
            "Stopping before iterative alignment and tuning."
        )

    variants = [BASELINE_VARIANT] + [(rounds, strength) for strength in STRENGTHS for rounds in ROUNDS]
    for row in rows:
        for strength in STRENGTHS:
            assigned = row["assignments"][BASELINE_VARIANT]
            for rounds in ROUNDS:
                aligned = diagonal_coral(row["features"], assigned, row["prototypes"], strength)
                encoding_z = row_zscore(response_scores(aligned, row["prototypes"], metric="mse", device=device))
                combined = 0.5 * row["mlp_z"] + 0.5 * encoding_z
                assigned = unique_assignment(combined)
                row["assignments"][(rounds, strength)] = assigned
        print(f"aligned {row['block']}", flush=True)

    tuning = []
    for variant in variants:
        summary, _, _, _ = evaluate(variant, tune_pictures)
        rounds, strength = variant
        record = {"rounds": rounds, "strength": strength, **summary}
        tuning.append(record)
        print(f"TUNE rounds={rounds}; strength={strength}; ocf={summary['ocf_mean']:.10f}", flush=True)

    best = max(tuning, key=lambda row: row["ocf_mean"])
    selected_variant = (best["rounds"], best["strength"])
    evaluation, predictions, references, groups = evaluate(selected_variant, evaluation_pictures)
    output_dir = ROOT / "test/60_fusion_transduction/13_iterative_alignment"
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / f"{args.task}_predictions.npz",
        predictions=predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    result = {
        "experiment": "13_iterative_alignment",
        "name": f"{args.task}_iterative_diagonal_coral",
        "config": {
            "task": args.task,
            "encoding_alpha": encoding_alpha,
            "mlp_weight": 0.5,
            "rounds": ROUNDS,
            "strengths": STRENGTHS,
            "selection_metric": "tune_ocf_mean",
        },
        "protocol": {
            "split_seed": 42,
            "picture_seed": 123,
            "tune_picture_count": len(tune_pictures),
            "tune_pictures": tune_pictures.tolist(),
            "evaluation_picture_count": len(evaluation_pictures),
            "evaluation_pictures": evaluation_pictures.tolist(),
        },
        "baseline_reproduction": {
            "expected_ocf": expected_baseline,
            "actual_ocf": baseline["ocf_mean"],
            "absolute_error": baseline_error,
            "absolute_tolerance": BASELINE_ATOL,
            "passed": True,
        },
        "tuning": tuning,
        "selected": best,
        "evaluation": evaluation,
        "evaluation_baseline": baseline,
        "top1": {
            "selected": evaluation["top1_accuracy"],
            "baseline": baseline["top1_accuracy"],
        },
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
    }
    result_path = output_dir / f"results_{args.task}_alignment.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL {result_path}: selected={evaluation['ocf_mean']:.10f}; "
        f"baseline={baseline['ocf_mean']:.10f}; variant={selected_variant}",
        flush=True,
    )


if __name__ == "__main__":
    main()
