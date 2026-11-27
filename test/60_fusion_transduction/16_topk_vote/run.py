"""P2: softmax-weighted pixel voting over assignment-aware top-k candidates."""

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
from src.final import apply_session_normalizer, fit_session_normalizer, row_zscore
from src.neural import load_classifier, predict_logits
from src.retrieval import unique_assignment
from src.ridge import RidgePath, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
K_VALUES = (1, 3, 5, 10, 20)
TEMPERATURES = (0.5, 1.0, 2.0)
THRESHOLDS = (0.25, 0.5, 0.75)
BASELINE_OCF = {"task2": 23.329312169146995, "task3": 16.57463190021421}
BASELINE_ATOL = 1e-5


def topk_vote(images, scores, assigned, k, temperature, threshold, block_size):
    """Render binary votes without materializing all candidates for all samples."""
    output = []
    for start in range(0, len(scores), block_size):
        stop = min(start + block_size, len(scores))
        block_scores = scores[start:stop]
        block_assigned = assigned[start:stop]
        count = len(block_scores)
        candidates = np.empty((count, k), dtype=np.int64)
        candidates[:, 0] = block_assigned

        # Exclude the assignment before selecting exactly k-1 additional candidates.
        unassigned_scores = block_scores.copy()
        unassigned_scores[np.arange(count), block_assigned] = -np.inf
        additional = k - 1
        candidates[:, 1:] = np.argpartition(
            unassigned_scores, -additional, axis=1
        )[:, -additional:]

        logits = np.take_along_axis(block_scores, candidates, axis=1) / temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        vote = np.zeros((count, *images.shape[1:]), dtype=np.float32)
        for candidate_index in range(k):
            vote += probabilities[:, candidate_index, None, None] * images[candidates[:, candidate_index]]
        output.append((vote >= threshold).astype(np.float32))
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task2", "task3"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
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
            mlp_scores = row_zscore(predict_logits(model, mlp_counts, device=device))
            encoding_scores = row_zscore(response_scores(np.log1p(counts), prototypes, metric="mse", device=device))
            scores = 0.5 * mlp_scores + 0.5 * encoding_scores
            rows.append({
                "block": block.key,
                "targets": targets,
                "scores": scores,
                "assigned": unique_assignment(scores),
            })
        print(f"scored {mouse}", flush=True)

    def predictions_for(k, temperature, threshold, pictures):
        prediction_parts = []
        target_parts = []
        group_parts = []
        correct = 0
        for row in rows:
            mask = np.isin(row["targets"], pictures)
            if not mask.any():
                continue
            if k == 1:
                predictions = images[row["assigned"][mask]]
                correct += int((row["assigned"][mask] == row["targets"][mask]).sum())
            else:
                predictions = topk_vote(
                    images,
                    row["scores"][mask],
                    row["assigned"][mask],
                    k,
                    temperature,
                    threshold,
                    args.batch_size,
                )
            prediction_parts.append(predictions)
            target_parts.append(images[row["targets"][mask]])
            group_parts.extend([row["block"]] * int(mask.sum()))
        predictions = np.concatenate(prediction_parts).astype(np.float32, copy=False)
        references = np.concatenate(target_parts)
        groups = np.asarray(group_parts)
        scores = score_predictions(predictions, references, device=device, batch_size=args.batch_size)
        summary = summarize_scores(scores, groups)
        if k == 1:
            summary["top1_accuracy"] = float(correct / len(predictions))
        return summary, predictions, references, groups

    # Evaluate the held-out baseline first so a reproduction mismatch stops all tuning.
    baseline, baseline_predictions, baseline_references, baseline_groups = predictions_for(
        1, None, None, evaluation_pictures
    )
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
            f"expected={expected_baseline}, abs_error={baseline_error}, tolerance={BASELINE_ATOL}."
        )

    def tune_record(k, temperature, threshold, summary):
        record = {
            "k": k,
            "temperature": temperature,
            "threshold": threshold,
            "n": summary["n"],
            "ocf_mean": summary["ocf_mean"],
        }
        if "top1_accuracy" in summary:
            record["top1_accuracy"] = summary["top1_accuracy"]
        tuning.append(record)
        print(f"TUNE {record}", flush=True)

    tuning = []
    baseline_tune, _, _, _ = predictions_for(1, None, None, tune_pictures)
    tune_record(1, None, None, baseline_tune)
    for k in K_VALUES[1:]:
        for temperature in TEMPERATURES:
            for threshold in THRESHOLDS:
                summary, _, _, _ = predictions_for(k, temperature, threshold, tune_pictures)
                tune_record(k, temperature, threshold, summary)

    best = max(tuning, key=lambda row: row["ocf_mean"])
    if best["k"] == 1:
        evaluation = baseline
        predictions = baseline_predictions
        references = baseline_references
        groups = baseline_groups
    else:
        evaluation, predictions, references, groups = predictions_for(
            best["k"], best["temperature"], best["threshold"], evaluation_pictures
        )

    output_dir = ROOT / "test/60_fusion_transduction/16_topk_vote"
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / f"{args.task}_predictions.npz",
        predictions=predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    result = {
        "experiment": "16_topk_vote",
        "name": f"{args.task}_assignment_aware_topk_pixel_vote",
        "config": {
            "task": args.task,
            "encoding_alpha": encoding_alpha,
            "mlp_weight": 0.5,
            "encoding_weight": 0.5,
            "score_normalization": "row_zscore",
            "assignment": "unique_assignment",
            "k_values": list(K_VALUES),
            "temperatures": list(TEMPERATURES),
            "thresholds": list(THRESHOLDS),
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
        "note": "Candidates always include the assignment; no inter-candidate similarity is computed.",
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
    }
    result_path = output_dir / f"results_{args.task}_topk.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL {result_path}: selected={evaluation['ocf_mean']:.10f}; "
        f"baseline={baseline['ocf_mean']:.10f}; variant={best}",
        flush=True,
    )


if __name__ == "__main__":
    main()
