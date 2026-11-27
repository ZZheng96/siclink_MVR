"""Test Task 3 three-repeat mean augmentation without changing inference or fusion."""

import gc
import json
import sys
import time
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
from src.neural import SpikeClassifier, fit_classifier, load_classifier, predict_logits
from src.retrieval import unique_assignment
from src.ridge import RidgePath, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
OUTPUT_DIR = ROOT / "test/30_task3_single_trial/22_task3_mean_augmentation"
ARTIFACTS = OUTPUT_DIR / "artifacts"
BASELINE_ROOT = ROOT / "test/20_task2_cross_day/04_neural_seen/artifacts/task3_mlp_bin50_pooled_sessionnorm_assign"
BASELINE_OCF = 16.57463190021421
BASELINE_ATOL = 1e-5
MLP_WEIGHT = 0.5
ENCODING_ALPHA = 0.01


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def row_zscore(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)


def fused_assignment(model, blocks, device):
    predictions = []
    for block in blocks:
        logits = predict_logits(model, block["validation_x"], device=device)
        encoding_scores = response_scores(
            np.log1p(block["raw_validation_x"]), block["prototypes"], metric="mse", device=device
        )
        combined = MLP_WEIGHT * row_zscore(logits) + (1 - MLP_WEIGHT) * row_zscore(encoding_scores)
        predictions.append(unique_assignment(combined))
    return predictions


def prepare_mouse(mouse, mouse_blocks, split, bank_coefficients, device):
    train_x_parts = []
    train_y_parts = []
    validation_x_parts = []
    validation_y_parts = []
    encoding_neural_parts = []
    encoding_target_parts = []
    validation_blocks = []
    for block in mouse_blocks:
        block_split = split.blocks[block.key]
        raw_train_x = block.features[block_split.train_trials]
        train_targets = block.trial_targets[block_split.train_trials]
        mean_targets, raw_mean_x = aggregate_by_target(raw_train_x, train_targets)
        normalizer = fit_session_normalizer(raw_train_x)
        train_x_parts.extend((
            apply_session_normalizer(raw_train_x, normalizer),
            apply_session_normalizer(raw_mean_x, normalizer),
        ))
        train_y_parts.extend((train_targets, mean_targets))
        encoding_neural_parts.append(np.log1p(raw_mean_x))
        encoding_target_parts.append(mean_targets)

        raw_validation_x = block.features[block_split.validation_trials]
        validation_targets = block.trial_targets[block_split.validation_trials]
        validation_x = apply_session_normalizer(raw_validation_x, normalizer)
        validation_x_parts.append(validation_x)
        validation_y_parts.append(validation_targets)
        validation_blocks.append({
            "key": block.key,
            "raw_validation_x": raw_validation_x,
            "validation_x": validation_x,
            "targets": validation_targets,
        })

    encoding_path = RidgePath().fit(
        bank_coefficients[np.concatenate(encoding_target_parts)], np.concatenate(encoding_neural_parts)
    )
    prototypes = encoding_path.predict(bank_coefficients, ENCODING_ALPHA)
    for block in validation_blocks:
        block["prototypes"] = prototypes
    baseline = load_classifier(BASELINE_ROOT / mouse / "classifier.pt", device)
    baseline_predictions = fused_assignment(baseline, validation_blocks, device)
    del baseline
    release_memory()
    for block, predictions in zip(validation_blocks, baseline_predictions):
        block["baseline_predictions"] = predictions
    return {
        "mouse": mouse,
        "train_x": np.concatenate(train_x_parts),
        "train_y": np.concatenate(train_y_parts),
        "validation_x": np.concatenate(validation_x_parts),
        "validation_y": np.concatenate(validation_y_parts),
        "validation_blocks": validation_blocks,
        "single_trial_rows": sum(len(block.trial_targets[split.blocks[block.key].train_trials]) for block in mouse_blocks),
        "mean_rows": len(np.concatenate(train_y_parts)) - sum(
            len(block.trial_targets[split.blocks[block.key].train_trials]) for block in mouse_blocks
        ),
    }


def evaluate(prepared, images, pictures, field, device):
    prediction_indices = []
    target_indices = []
    groups = []
    for mouse_data in prepared.values():
        for block in mouse_data["validation_blocks"]:
            mask = np.isin(block["targets"], pictures)
            prediction_indices.append(block[field][mask])
            target_indices.append(block["targets"][mask])
            groups.extend([block["key"]] * int(mask.sum()))
    prediction_indices = np.concatenate(prediction_indices)
    target_indices = np.concatenate(target_indices)
    predictions = images[prediction_indices]
    references = images[target_indices]
    scores = score_predictions(predictions, references, device=device, batch_size=16)
    summary = summarize_scores(scores, np.asarray(groups))
    summary["top1_accuracy"] = float((prediction_indices == target_indices).mean())
    return summary, predictions, references, np.asarray(groups)


def main():
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    images = (load_images(DATA) >= 128).astype(np.float32)
    bank_coefficients = dct_encode(images, 16)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    split = make_split(blocks, "task3", seed=42)
    selected = np.random.default_rng(123).permutation(len(images))
    tune_pictures = np.sort(selected[:256])
    evaluation_pictures = np.sort(selected[256:768])

    prepared = {}
    for mouse in sorted({block.mouse for block in blocks}):
        mouse_blocks = [block for block in blocks if block.mouse == mouse]
        prepared[mouse] = prepare_mouse(mouse, mouse_blocks, split, bank_coefficients, device)
        print(f"PREPARED {mouse}: baseline predictions and unchanged encoding complete", flush=True)

    baseline_tune, _, _, _ = evaluate(prepared, images, tune_pictures, "baseline_predictions", device)
    baseline_evaluation, baseline_predictions, references, groups = evaluate(
        prepared, images, evaluation_pictures, "baseline_predictions", device
    )
    baseline_error = abs(baseline_evaluation["ocf_mean"] - BASELINE_OCF)
    print(
        f"BASELINE tune={baseline_tune['ocf_mean']:.10f}; held-out={baseline_evaluation['ocf_mean']:.10f}; "
        f"expected={BASELINE_OCF:.10f}; abs_error={baseline_error:.3g}",
        flush=True,
    )
    if baseline_error > BASELINE_ATOL:
        raise RuntimeError(f"Task 3 baseline mismatch: {baseline_evaluation['ocf_mean']} vs {BASELINE_OCF}")

    training = []
    for mouse, mouse_data in prepared.items():
        tune_mask = np.isin(mouse_data["validation_y"], tune_pictures)
        output = ARTIFACTS / "models" / mouse
        torch.manual_seed(42)
        np.random.seed(42)
        model = SpikeClassifier(
            n_features=mouse_data["train_x"].shape[1], n_classes=len(images),
            hidden=512, dropout=0.2, apply_log=False,
        )
        history = fit_classifier(
            model,
            (mouse_data["train_x"], mouse_data["train_y"]),
            (mouse_data["validation_x"][tune_mask], mouse_data["validation_y"][tune_mask]),
            output,
            epochs=30,
            batch_size=256,
            learning_rate=3e-4,
            weight_decay=1e-3,
            device=device,
            patience=8,
            seed=42,
        )
        del model
        release_memory()
        candidate = load_classifier(output / "classifier.pt", device)
        candidate_predictions = fused_assignment(candidate, mouse_data["validation_blocks"], device)
        for block, predictions in zip(mouse_data["validation_blocks"], candidate_predictions):
            block["candidate_predictions"] = predictions
        best = max(history, key=lambda row: row["validation_accuracy"])
        training.append({
            "mouse": mouse,
            "single_trial_rows": mouse_data["single_trial_rows"],
            "added_three_repeat_mean_rows": mouse_data["mean_rows"],
            "total_rows": len(mouse_data["train_x"]),
            "epochs_run": len(history),
            "checkpoint_epoch": best["epoch"],
            "best_validation_accuracy": best["validation_accuracy"],
            "checkpoint": str((output / "classifier.pt").relative_to(ROOT)),
        })
        del candidate
        release_memory()
        print(f"CANDIDATE {mouse}: {training[-1]}", flush=True)

    candidate_tune, _, _, _ = evaluate(prepared, images, tune_pictures, "candidate_predictions", device)
    candidate_wins = candidate_tune["ocf_mean"] > baseline_tune["ocf_mean"]
    selected_name = "candidate" if candidate_wins else "baseline"
    if candidate_wins:
        candidate_evaluation, selected_predictions, references, groups = evaluate(
            prepared, images, evaluation_pictures, "candidate_predictions", device
        )
    else:
        candidate_evaluation = None
        selected_predictions = baseline_predictions

    selected_path = ARTIFACTS / "task3_mean_augmentation_selected.npz"
    np.savez_compressed(
        selected_path,
        predictions=selected_predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    result = {
        "experiment": "22_task3_mean_augmentation",
        "name": "task3_single_trials_plus_three_repeat_means",
        "score_scope": "fixed local validation protocol only; not a Kaggle score",
        "config": {
            "task": "task3",
            "seed": 42,
            "bin_ms": 50,
            "architecture": "mlp",
            "hidden": 512,
            "dropout": 0.2,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "batch_size": 256,
            "maximum_epochs": 30,
            "patience": 8,
            "training_change_only": "append per-session/target means of the three permitted training repeats; retain all single trials",
            "validation": "unchanged held-out single trial, excluded from every training mean",
            "mlp_weight": MLP_WEIGHT,
            "encoding_alpha": ENCODING_ALPHA,
            "assignment": True,
            "selection_metric": "tune official OCF; ties retain baseline",
        },
        "protocol": {
            **split.metadata(),
            "tune_picture_count": len(tune_pictures),
            "evaluation_picture_count": len(evaluation_pictures),
            "held_out_rule": "baseline reproduction plus candidate only if candidate wins on tune",
        },
        "training": training,
        "baseline": {
            "expected_ocf": BASELINE_OCF,
            "actual_ocf": baseline_evaluation["ocf_mean"],
            "absolute_error": baseline_error,
            "absolute_tolerance": BASELINE_ATOL,
            "passed": True,
            "tune": baseline_tune,
            "evaluation": baseline_evaluation,
        },
        "tuning": [
            {"variant": "baseline", **baseline_tune},
            {"variant": "candidate", **candidate_tune},
        ],
        "selected": {
            "variant": selected_name,
            "tune": candidate_tune if candidate_wins else baseline_tune,
        },
        "evaluation": candidate_evaluation if candidate_wins else baseline_evaluation,
        "candidate_evaluation": candidate_evaluation,
        "candidate_held_out_excluded": not candidate_wins,
        "evaluation_delta_ocf": (
            candidate_evaluation["ocf_mean"] - baseline_evaluation["ocf_mean"] if candidate_wins else 0.0
        ),
        "selected_npz": str(selected_path.relative_to(ROOT)),
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "elapsed_seconds": time.time() - started,
    }
    result_path = OUTPUT_DIR / "results_task3_mean_augmentation.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"FINAL {result_path}: baseline_tune={baseline_tune['ocf_mean']:.6f}; "
        f"candidate_tune={candidate_tune['ocf_mean']:.6f}; selected={selected_name}; "
        f"evaluation={result['evaluation']['ocf_mean']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
