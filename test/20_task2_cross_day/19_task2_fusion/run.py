"""Training-free Task2 MLP/encoding fusion-weight calibration."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

OUTPUT_DIR = ROOT / "test/20_task2_cross_day/19_task2_fusion"
ARTIFACTS = OUTPUT_DIR / "artifacts"
RESULTS_PATH = OUTPUT_DIR / "results_task2_fusion.json"
SELECTED_NPZ = ARTIFACTS / "task2_fusion_selected.npz"
WEIGHTS = (0.4, 0.5, 0.6, 0.7)
BASELINE_WEIGHT = 0.5
BASELINE_OCF = 24.33936155396989
BASELINE_ATOL = 1e-5


def load_experiment18():
    path = ROOT / "test/60_fusion_transduction/18_seed_ensemble/run.py"
    spec = importlib.util.spec_from_file_location("experiment18_seed_ensemble", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load experiment18 from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment18 = load_experiment18()
SEEDS = experiment18.SEEDS


def assign_seen_weight(data, weight):
    for row in data["rows"]:
        mean_logits = np.mean(
            np.stack([row["logits"][seed] for seed in SEEDS]),
            axis=0,
            dtype=np.float32,
        )
        mlp_z = experiment18.row_zscore(mean_logits)
        combined = weight * mlp_z + (1.0 - weight) * row["encoding_z"]
        row["assignments"][weight] = experiment18.unique_assignment(combined)
    print(f"ASSIGNED weight={weight:.1f} rows={len(data['rows'])}", flush=True)


def weight_record(weight, summary):
    return {
        "weight": float(weight),
        "seeds": list(SEEDS),
        "n": summary["n"],
        "ocf_mean": summary["ocf_mean"],
        "ocf_std": summary["ocf_std"],
        "top1_accuracy": summary["top1_accuracy"],
    }


def main():
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE {device}; seeds={SEEDS}; weights={WEIGHTS}; training=none", flush=True)

    images = (experiment18.load_images(experiment18.DATA) >= 128).astype(np.float32)
    if len(images) != 2748:
        raise ValueError(f"Expected 2748 training pictures, found {len(images)}.")
    data = experiment18.prepare_seen("task2", images, device)
    if len(data["tune_pictures"]) != 256 or len(data["evaluation_pictures"]) != 512:
        raise RuntimeError("Unexpected tune/held-out picture counts.")
    for seed in SEEDS:
        experiment18.predict_seen_seed("task2", data, seed)

    assign_seen_weight(data, BASELINE_WEIGHT)
    baseline_evaluation, baseline_details = experiment18.evaluate_seen(
        data, BASELINE_WEIGHT, data["evaluation_pictures"], return_details=True
    )
    baseline_error = abs(baseline_evaluation["ocf_mean"] - BASELINE_OCF)
    print(
        f"BASELINE weight=0.5: actual={baseline_evaluation['ocf_mean']:.14f}; "
        f"expected={BASELINE_OCF:.14f}; abs_error={baseline_error:.3g}",
        flush=True,
    )
    if baseline_error > BASELINE_ATOL:
        raise RuntimeError(
            f"Baseline mismatch: actual={baseline_evaluation['ocf_mean']}, "
            f"expected={BASELINE_OCF}, abs_error={baseline_error}, tolerance={BASELINE_ATOL}."
        )
    baseline = {
        "weight": BASELINE_WEIGHT,
        "expected_ocf": BASELINE_OCF,
        "actual_ocf": baseline_evaluation["ocf_mean"],
        "absolute_error": baseline_error,
        "absolute_tolerance": BASELINE_ATOL,
        "passed": True,
        "evaluation": baseline_evaluation,
    }

    tuning = []
    for weight in WEIGHTS:
        if weight != BASELINE_WEIGHT:
            assign_seen_weight(data, weight)
        summary, _ = experiment18.evaluate_seen(data, weight, data["tune_pictures"])
        record = weight_record(weight, summary)
        tuning.append(record)
        print(
            f"TUNE weight={weight:.1f} n={record['n']} OCF={record['ocf_mean']:.14f} "
            f"top1={record['top1_accuracy']:.10f}",
            flush=True,
        )
    selected = max(tuning, key=lambda row: row["ocf_mean"])
    selected_weight = selected["weight"]
    if selected_weight == BASELINE_WEIGHT:
        evaluation = baseline_evaluation
        details = baseline_details
    else:
        evaluation, details = experiment18.evaluate_seen(
            data, selected_weight, data["evaluation_pictures"], return_details=True
        )

    np.savez_compressed(
        SELECTED_NPZ,
        predictions=details["predictions"].astype(np.uint8) * 255,
        targets=details["references"].astype(np.uint8) * 255,
        groups=details["groups"],
        predicted_indices=details["predicted_indices"],
        target_indices=details["target_indices"],
        pictures=details["pictures"],
        weight=np.asarray(selected_weight, dtype=np.float64),
        seeds=np.asarray(SEEDS, dtype=np.int64),
    )

    mice = sorted(data["training"])
    checkpoints = {
        str(seed): {
            mouse: str(experiment18.seen_checkpoint("task2", seed, mouse).relative_to(ROOT))
            for mouse in mice
        }
        for seed in SEEDS
    }
    held_out_weights = [BASELINE_WEIGHT]
    if selected_weight != BASELINE_WEIGHT:
        held_out_weights.append(selected_weight)
    result = {
        "experiment": "19_task2_fusion",
        "name": "task2_training_free_mlp_fusion_weight_calibration",
        "config": {
            "task": "task2",
            "seeds": list(SEEDS),
            "weights": list(WEIGHTS),
            "training": "none; all existing checkpoints reused",
            "checkpoints": checkpoints,
            "bin_ms": 50,
            "encoding_alpha": data["encoding_alpha"],
            "logits_ensemble": "equal-weight mean of raw logits from seeds 42, 202, 777, then row_zscore",
            "fusion": "weight * MLP ensemble row_zscore + (1 - weight) * fixed encoding row_zscore",
            "post_fusion_zscore": False,
            "assignment": "unique_assignment",
            "selection_metric": "tune official OCF mean only",
        },
        "protocol": {
            "split_seed": 42,
            "picture_seed": 123,
            "tune_picture_count": len(data["tune_pictures"]),
            "held_out_picture_count": len(data["evaluation_pictures"]),
            "evaluation_picture_count": len(data["evaluation_pictures"]),
            "assignment_scope": (
                "unique_assignment over every complete 2748-picture validation block "
                "before tune/held-out subsetting"
            ),
            "held_out_rule": (
                "weight=0.5 baseline plus only the tune-selected weight; "
                "reuse the baseline evaluation when the selected weight is 0.5"
            ),
            "held_out_evaluation_weights": held_out_weights,
            "selected_npz_scope": "held-out rows for the tune-selected weight",
        },
        "baseline": baseline,
        "tuning": tuning,
        "selected": selected,
        "evaluation": evaluation,
        "delta": evaluation["ocf_mean"] - baseline["actual_ocf"],
        "top1": {
            "baseline": baseline_evaluation["top1_accuracy"],
            "selected": evaluation["top1_accuracy"],
            "delta": evaluation["top1_accuracy"] - baseline_evaluation["top1_accuracy"],
        },
        "selected_npz": str(SELECTED_NPZ.relative_to(ROOT)),
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "Kaggle_baseline": 21.09590,
        "score_scope": "fixed local validation protocol only; not a Kaggle score",
        "elapsed_seconds": time.perf_counter() - started,
    }
    RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"FINAL selected_weight={selected_weight:.1f} held_out_OCF={evaluation['ocf_mean']:.14f} "
        f"baseline={baseline['actual_ocf']:.14f} delta={result['delta']:.14f} "
        f"top1={evaluation['top1_accuracy']:.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
