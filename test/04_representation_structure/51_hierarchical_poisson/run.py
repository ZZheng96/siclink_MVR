"""Development-only hierarchical image-prior/empirical Poisson candidate."""

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.preprocessing import file_sha256, load_cached_blocks, validate_split
from src.ridge import RidgePath, dct_encode
from src.splits import make_split

OUT = Path(__file__).resolve().parent
PREREGISTRATION = OUT / "preregistration.json"
RESULTS = OUT / "development_results.json"
SELECTION = OUT / "selection.json"
EXPERIMENT48 = ROOT / "test/06_fusion_transduction/48_high_capacity_generalization"
RUN_48 = EXPERIMENT48 / "run.py"
RESULTS_48 = EXPERIMENT48 / "development_results.json"
SELECTION_48 = EXPERIMENT48 / "selection.json"
MICE = ("mice1", "mice2")
BLOCKS = tuple(f"{mouse}/session{session}" for mouse in MICE for session in range(1, 5))
TASKS = ("task2", "task3")
PRODUCTION = "production_recipe"
ARM = "hierarchical_poisson"
N_CLASSES = 2748
DCT_CUTOFF = 16
BIN_MS = 25
TRAIN_SEED = 42
ALPHA = 1.0
PSEUDO_COUNT = 3.0
RATE_FLOOR = 1e-4
GAIN_OFFSET = 0.01
GAIN_CLIP = (0.25, 4.0)
TASK2_FOLDS = ("session1", "session2", "session3")
TASK3_FOLDS = ("20261126", "20261127")
REQUIRED_NONNEGATIVE_FOLDS = {"task2": 2, "task3": 2}
EVALUATION_EPOCHS = (1, 2, 3, 4, 5, 7, 9, 12, 16, 20)
METRICS = (
    "raw_top1_accuracy",
    "raw_recall_at_32",
    "raw_top1_official_ocf",
    "assigned_final_ocf",
)
EXPECTED_EVALUATION_ROWS = {
    "task2": {"mouse_fold": 256, "mouse": 768, "fold": 512, "macro": 1536},
    "task3": {"mouse_fold": 1024, "mouse": 2048, "fold": 2048, "macro": 4096},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_experiment48_code():
    """Reuse evaluation and aggregation code, without importing any experiment 48 metrics."""
    raw = RUN_48.read_bytes()
    source = raw.decode("utf-8")
    require(
        'count_key = "n_rows" if level == "mouse_fold" else "n_evaluation_rows"' in source
        and 'row["n_evaluation_rows"] = sum(member[count_key] for member in members)' in source,
        "Experiment 48 aggregate_rows is not the corrected n_evaluation_rows version",
    )
    spec = importlib.util.spec_from_file_location("siclink_experiment48_for51", RUN_48)
    require(spec is not None and spec.loader is not None, "Cannot load experiment 48 code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(raw).hexdigest()


EXP48, EXP48_CODE_SHA256 = load_experiment48_code()


def development_subset():
    return np.sort(np.random.default_rng(20261128).permutation(N_CLASSES)[:256]).astype(np.int64)


def array_sha256(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float32).tobytes()).hexdigest()


def array_audit(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    require(matrix.ndim == 2 and np.isfinite(matrix).all(), "Cannot audit a nonfinite/non-matrix rate")
    values = matrix.astype(np.float64, copy=False)
    return {
        "shape": list(matrix.shape),
        "sha256_float32": array_sha256(matrix),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_preregistration():
    raw = PREREGISTRATION.read_bytes()
    value = json.loads(raw)
    require(value["experiment"] == OUT.name, "Preregistration experiment mismatch")
    require(value["freeze_revision"] == 1, "Unexpected preregistration revision")
    require(
        value["status"] == "preregistered_before_real_training_or_real_metrics",
        "Real fitting requires the frozen preregistration status",
    )
    data = value["data"]
    require(data["cache"] == "test/.cache/shared_preprocessing_v1", "Cache path changed")
    require(data["public_partition"] == "train only", "Only public train data are permitted")
    require(data["cache_access"] == "read-only; never rebuild the cache", "Cache must be read-only")
    require(data["mice"] == list(MICE) and data["blocks"] == list(BLOCKS), "Block contract changed")
    require(data["n_classes"] == N_CLASSES and data["subset_size"] == 256, "Class/subset changed")
    require(
        data["development_picture_subset"]
        == "np.sort(np.random.default_rng(20261128).permutation(2748)[:256])",
        "Development subset changed",
    )
    folds = value["folds"]
    require(folds["task2"]["fold_ids"] == list(TASK2_FOLDS), "Task2 folds changed")
    require(folds["task3"]["fold_ids"] == list(TASK3_FOLDS), "Task3 folds changed")
    require("session4" not in folds["task2"]["fold_ids"], "Task2 session4 is prohibited")
    require("42" not in folds["task3"]["fold_ids"], "Task3 seed42 is prohibited")
    arm = value["arm"]
    require(arm["arms_in_this_experiment"] == [ARM] and arm["training_epochs"] == 0,
            "The sole deterministic arm changed")
    require(arm["hyperparameter_search"] is False, "Hyperparameter search is prohibited")
    prior = value["image_prior"]
    require(prior["alpha"] == ALPHA and prior["rate_floor"] == RATE_FLOOR, "Image prior changed")
    posterior = value["empirical_posterior"]
    require(posterior["pseudo_count"] == PSEUDO_COUNT and not posterior["pseudo_count_search"],
            "Pseudo-count contract changed")
    calibration = value["unlabeled_target_calibration"]
    require(calibration["offset"] == GAIN_OFFSET and calibration["clip"] == list(GAIN_CLIP),
            "Gain contract changed")
    require(calibration["clip_search"] is False, "Gain clipping cannot be searched")
    scoring = value["poisson_scoring"]
    require("4*calibrated_rate.sum" in scoring["task2"], "Task2 four-repeat score changed")
    require("repeat aggregation is prohibited" in scoring["task3"].lower(), "Task3 single-trial score changed")
    require("Never construct" in scoring["candidate_matrix"], "Three-dimensional guard missing")
    gate = value["advancement_gate"]
    require(gate["macro_minimum_absolute_gain"] == 3.0, "Macro gate changed")
    require(gate["required_nonnegative_folds"] == REQUIRED_NONNEGATIVE_FOLDS, "Fold gate changed")
    reference = value["reference48"]
    require(reference["required_sources"] == [
        "test/06_fusion_transduction/48_high_capacity_generalization/development_results.json",
        "test/06_fusion_transduction/48_high_capacity_generalization/selection.json",
    ], "Experiment 48 reference contract changed")
    require(value["outputs"]["model_artifacts"] == "none retained", "Model artifacts are prohibited")
    prohibited = value["prohibited"]
    require("Task2 session4 validation or final evaluation" in prohibited, "Task2 prohibition missing")
    require("Task3 split seed 42 evaluation" in prohibited, "Task3 prohibition missing")
    require("dependence on experiment 49 or 50 outputs" in prohibited, "Independence declaration missing")
    return value, hashlib.sha256(raw).hexdigest()


def load_reference48(results_path=RESULTS_48, selection_path=SELECTION_48):
    """Validate the production comparator before any real cache or CUDA operation."""
    results_path, selection_path = Path(results_path), Path(selection_path)
    missing = [str(path) for path in (results_path, selection_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required experiment 48 comparator files are missing: {missing}")
    results_raw, selection_raw = results_path.read_bytes(), selection_path.read_bytes()
    results, selection = json.loads(results_raw), json.loads(selection_raw)
    results_hash = hashlib.sha256(results_raw).hexdigest()
    selection_hash = hashlib.sha256(selection_raw).hexdigest()
    require(results.get("experiment") == "48_high_capacity_generalization", "Unexpected reference results")
    require(selection.get("experiment") == "48_high_capacity_generalization", "Unexpected reference selection")
    require(results.get("status") == "development_results_complete", "Reference results incomplete")
    require(selection.get("status") == "development_selection_complete", "Reference selection incomplete")
    require(selection.get("development_results_sha256") == results_hash, "Reference hash linkage failed")
    require(selection.get("preregistration_sha256") == results.get("preregistration_sha256"),
            "Reference preregistration hash mismatch")
    require(results.get("development_picture_subset") == development_subset().tolist(),
            "Reference development subset mismatch")
    selected_epochs = {}
    for task in TASKS:
        task_selection = selection["tasks"][task]
        epoch = task_selection["selected_epochs"][PRODUCTION]
        require(isinstance(epoch, int) and not isinstance(epoch, bool) and epoch in EVALUATION_EPOCHS,
                f"Invalid reference production epoch for {task}")
        independently_selected = EXP48.select_epoch(
            results["macro"], task, PRODUCTION, expected_count=len(EVALUATION_EPOCHS),
        )
        require(independently_selected["epoch"] == epoch, "Reference is not production's best macro epoch")
        selected_epochs[task] = epoch
        selected_macro = task_selection["selected_macro"][PRODUCTION]
        result_macro = EXP48.find_aggregate(results["macro"], task, PRODUCTION, epoch)
        for metric in METRICS:
            require(selected_macro[metric] == result_macro[metric],
                    f"Reference selected macro {metric} mismatch for {task}")
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        for mouse in MICE:
            row = EXP48.find_aggregate(results["mouse"], task, PRODUCTION, epoch, mouse=mouse)
            require(row["n_evaluation_rows"] == EXPECTED_EVALUATION_ROWS[task]["mouse"],
                    "Reference mouse row-count mismatch")
        for fold in fold_ids:
            row = EXP48.find_aggregate(results["fold"], task, PRODUCTION, epoch, fold=fold)
            require(row["n_evaluation_rows"] == EXPECTED_EVALUATION_ROWS[task]["fold"],
                    "Reference fold row-count mismatch")
        require(result_macro["n_evaluation_rows"] == EXPECTED_EVALUATION_ROWS[task]["macro"],
                "Reference macro row-count mismatch")
    return {
        "results": results,
        "selected_epochs": selected_epochs,
        "results_sha256": results_hash,
        "selection_sha256": selection_hash,
    }


def target_statistics(raw, targets):
    """Return sorted per-target raw-count sums and repeat counts."""
    raw = np.asarray(raw, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    require(raw.ndim == 2 and targets.ndim == 1 and len(raw) == len(targets) and len(targets),
            "Invalid target-statistic inputs")
    require(np.isfinite(raw).all() and np.all(raw >= 0), "Raw counts must be finite and nonnegative")
    order = np.argsort(targets, kind="stable")
    sorted_targets, sorted_raw = targets[order], raw[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_targets)) + 1))
    counts = np.diff(np.concatenate((starts, [len(targets)]))).astype(np.int64)
    sums = np.add.reduceat(sorted_raw, starts, axis=0, dtype=np.float32)
    return sorted_targets[starts], sums, counts


def complete_targets(targets, counts, n_classes, expected_repeats):
    require(np.array_equal(targets, np.arange(n_classes, dtype=np.int64)), "Target coverage/order mismatch")
    require(np.all(counts == expected_repeats), "Per-target training repeat count mismatch")


def fit_image_prior(bank, target_parts, mean_parts, *, expected_rows):
    targets = np.concatenate(target_parts).astype(np.int64, copy=False)
    means = np.concatenate(mean_parts).astype(np.float32, copy=False)
    require(len(targets) == expected_rows and len(means) == expected_rows, "Image-prior row count mismatch")
    require(np.all((targets >= 0) & (targets < len(bank))), "Image-prior target outside gallery")
    path = RidgePath().fit(bank[targets], np.log1p(means))
    predicted_log = path.predict(bank, ALPHA)
    require(np.isfinite(predicted_log).all(), "Nonfinite image-prior prediction")
    with np.errstate(over="raise", invalid="raise"):
        prior = np.maximum(np.expm1(predicted_log), RATE_FLOOR).astype(np.float32)
    require(prior.shape == (len(bank), means.shape[1]) and np.isfinite(prior).all(),
            "Image-prior rate construction failed")
    return prior


def posterior_rates(sum_train_counts, n, prior):
    sums = np.asarray(sum_train_counts, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    require(sums.shape == prior.shape and sums.ndim == 2 and n > 0, "Posterior shape/count mismatch")
    posterior = (sums + PSEUDO_COUNT * prior) / float(n + PSEUDO_COUNT)
    return np.maximum(posterior, RATE_FLOOR).astype(np.float32)


def feature_marginal(raw):
    raw = np.asarray(raw, dtype=np.float32)
    require(raw.ndim == 2 and len(raw) and np.isfinite(raw).all(), "Invalid unlabeled marginal input")
    return raw.mean(axis=0, dtype=np.float64).astype(np.float32)


def calibrate_from_marginals(posterior, source_marginal, target_marginal):
    source = np.asarray(source_marginal, dtype=np.float32)
    target = np.asarray(target_marginal, dtype=np.float32)
    require(posterior.ndim == 2 and posterior.shape[1] == len(source) == len(target),
            "Marginal/rate feature mismatch")
    raw_gain = (target + GAIN_OFFSET) / (source + GAIN_OFFSET)
    gain = np.clip(raw_gain, *GAIN_CLIP).astype(np.float32)
    calibrated = np.maximum(posterior * gain, RATE_FLOOR).astype(np.float32)
    require(np.isfinite(raw_gain).all() and np.isfinite(calibrated).all(), "Invalid target calibration")
    return calibrated, raw_gain.astype(np.float32), gain


def construction_audit(entry):
    return {
        "block": entry["block"],
        "n_train_single_trials": entry["source_n"],
        "n_unlabeled_target_single_trials": entry["target_n"],
        "n_train_repeats_per_glyph": entry["n_repeats"],
        "target_marginal_uses_labels": False,
        "validation_labels_used_only_after_rate_construction": True,
        "source_marginal": array_audit(entry["source_marginal"][None, :]),
        "target_marginal": array_audit(entry["target_marginal"][None, :]),
        "empirical_rate": array_audit(entry["empirical_rate"]),
        "image_prior_rate": array_audit(entry["prior"]),
        "posterior_rate": array_audit(entry["posterior"]),
        "unclipped_gain": array_audit(entry["raw_gain"][None, :]),
        "gain": {
            **array_audit(entry["gain"][None, :]),
            "lower_clip_count": int(np.sum(entry["raw_gain"] < GAIN_CLIP[0])),
            "upper_clip_count": int(np.sum(entry["raw_gain"] > GAIN_CLIP[1])),
        },
        "calibrated_rate": array_audit(entry["rate"]),
        "repeat_multiplier_in_score": entry["repeats"],
    }


def prepare_task2(mouse_blocks, split, mouse, fold, subset, bank, *, n_classes=N_CLASSES):
    """Build one held-session Task2 posterior and its four-repeat raw observations."""
    target_parts, mean_parts, train_audit = [], [], []
    empirical_sum = None
    source_sum = None
    source_n = 0
    held = None
    held_raw = None
    for block in mouse_blocks:
        part = split.blocks[block.key]
        if block.session == fold:
            require(len(part.train_trials) == 0 and len(part.validation_trials) == block.n_trials,
                    "Task2 target session is not fully held out")
            held = block
            held_raw = np.asarray(block.features[part.validation_trials], dtype=np.float32)
            continue
        require(len(part.validation_trials) == 0 and len(part.train_trials) == block.n_trials,
                "Task2 training session leakage")
        raw = np.asarray(block.features[part.train_trials], dtype=np.float32)
        labels = np.asarray(block.trial_targets[part.train_trials], dtype=np.int64)
        targets, sums, counts = target_statistics(raw, labels)
        complete_targets(targets, counts, n_classes, 4)
        target_parts.append(targets)
        mean_parts.append(sums / counts[:, None])
        empirical_sum = sums.copy() if empirical_sum is None else empirical_sum + sums
        source_sum = raw.sum(axis=0, dtype=np.float64) if source_sum is None else source_sum + raw.sum(
            axis=0, dtype=np.float64
        )
        source_n += len(raw)
        train_audit.append({"block": block.key, "train_trials": len(raw), "validation_trials": 0,
                            "prior_target_mean_rows": n_classes, "train_repeats_per_glyph": 4})
    require(held is not None and len(target_parts) == 3, "Task2 must have three training sessions")
    prior = fit_image_prior(bank, target_parts, mean_parts, expected_rows=3 * n_classes)
    n_repeats = 12
    empirical_rate = (empirical_sum / n_repeats).astype(np.float32)
    posterior = posterior_rates(empirical_sum, n_repeats, prior)
    source_marginal = (source_sum / source_n).astype(np.float32)
    target_marginal = feature_marginal(held_raw)
    calibrated, raw_gain, gain = calibrate_from_marginals(posterior, source_marginal, target_marginal)
    held_part = split.blocks[held.key]
    grouped_targets, observations, grouped_counts = target_statistics(
        held_raw, held.trial_targets[held_part.validation_trials],
    )
    complete_targets(grouped_targets, grouped_counts, n_classes, 4)
    sorted_targets = grouped_targets
    entry = {
        "block": held.key, "observations": observations, "targets": sorted_targets,
        "rate": calibrated, "repeats": 4, "prior": prior, "empirical_rate": empirical_rate,
        "posterior": posterior, "source_marginal": source_marginal, "target_marginal": target_marginal,
        "raw_gain": raw_gain, "gain": gain, "source_n": source_n, "target_n": len(held_raw),
        "n_repeats": n_repeats,
    }
    audit = {
        "task": "task2", "mouse": mouse, "fold": fold,
        "prior_fit": {"train_rows": 3 * n_classes, "alpha": ALPHA,
                      "source": "three training sessions' per-target four-repeat raw means"},
        "train_blocks": train_audit,
        "validation_blocks": [{"block": held.key, "train_trials": 0,
                               "unlabeled_target_marginal_trials": len(held_raw),
                               "grouped_validation_rows": n_classes, "repeats_summed": 4}],
        "rates": [construction_audit(entry)],
    }
    return {"task": "task2", "mouse": mouse, "fold": fold, "subset": subset,
            "validation": [entry], "audit": audit}


def prepare_task3(mouse_blocks, split, mouse, fold, subset, bank, *, n_classes=N_CLASSES):
    """Fit one four-block prior but retain four independent Task3 posteriors/gains."""
    target_parts, mean_parts, block_data = [], [], []
    for block in mouse_blocks:
        part = split.blocks[block.key]
        require(len(part.train_trials) == 3 * n_classes and len(part.validation_trials) == n_classes,
                "Task3 train/validation trial counts changed")
        raw = np.asarray(block.features[part.train_trials], dtype=np.float32)
        labels = np.asarray(block.trial_targets[part.train_trials], dtype=np.int64)
        targets, sums, counts = target_statistics(raw, labels)
        complete_targets(targets, counts, n_classes, 3)
        target_parts.append(targets)
        mean_parts.append(sums / counts[:, None])
        raw_validation = np.asarray(block.features[part.validation_trials], dtype=np.float32)
        block_data.append((block, part, raw, raw_validation, sums))
    require(len(block_data) == 4, "Task3 requires four blocks")
    prior = fit_image_prior(bank, target_parts, mean_parts, expected_rows=4 * n_classes)
    entries = []
    train_audit = []
    for block, part, raw, raw_validation, sums in block_data:
        n_repeats = 3
        empirical_rate = (sums / n_repeats).astype(np.float32)
        posterior = posterior_rates(sums, n_repeats, prior)
        source_marginal = feature_marginal(raw)
        target_marginal = feature_marginal(raw_validation)
        calibrated, raw_gain, gain = calibrate_from_marginals(posterior, source_marginal, target_marginal)
        validation_targets = np.asarray(block.trial_targets[part.validation_trials], dtype=np.int64)
        order = np.argsort(validation_targets, kind="stable")
        observations, sorted_targets = raw_validation[order], validation_targets[order]
        require(np.array_equal(sorted_targets, np.arange(n_classes)), "Task3 validation target coverage mismatch")
        entry = {
            "block": block.key, "observations": observations, "targets": sorted_targets,
            "rate": calibrated, "repeats": 1, "prior": prior, "empirical_rate": empirical_rate,
            "posterior": posterior, "source_marginal": source_marginal,
            "target_marginal": target_marginal, "raw_gain": raw_gain, "gain": gain,
            "source_n": len(raw), "target_n": len(raw_validation), "n_repeats": n_repeats,
        }
        entries.append(entry)
        train_audit.append({"block": block.key, "train_trials": len(raw),
                            "unlabeled_target_marginal_trials": len(raw_validation),
                            "prior_target_mean_rows": n_classes, "train_repeats_per_glyph": 3,
                            "validation_rows": n_classes, "repeat_aggregation": False})
    audit = {
        "task": "task3", "mouse": mouse, "fold": fold,
        "prior_fit": {"train_rows": 4 * n_classes, "alpha": ALPHA,
                      "source": "four blocks' independent per-target three-repeat raw means"},
        "train_blocks": train_audit,
        "empirical_posterior_policy": "separate per block; never pooled across four blocks",
        "rates": [construction_audit(entry) for entry in entries],
    }
    return {"task": "task3", "mouse": mouse, "fold": fold, "subset": subset,
            "validation": entries, "audit": audit}


def poisson_scores(observations, rate, repeats, device, *, batch_size=256):
    """Score all classes with float32 two-dimensional GEMMs only."""
    observations = np.asarray(observations, dtype=np.float32)
    rate = np.asarray(rate, dtype=np.float32)
    require(observations.ndim == 2 and rate.ndim == 2 and observations.shape[1] == rate.shape[1],
            "Poisson inputs must be matched two-dimensional matrices")
    require(isinstance(repeats, (int, np.integer)) and repeats > 0, "Repeat multiplier must be positive")
    require(np.isfinite(observations).all() and np.isfinite(rate).all() and np.all(rate > 0),
            "Poisson observations/rates are invalid")
    output = np.empty((len(observations), len(rate)), dtype=np.float32)
    device = torch.device(device)
    x = torch.as_tensor(np.ascontiguousarray(observations), dtype=torch.float32, device=device)
    rate_tensor = torch.as_tensor(np.ascontiguousarray(rate), dtype=torch.float32, device=device)
    log_rate = torch.log(rate_tensor)
    rate_sum = rate_tensor.sum(dim=1, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            scores = torch.mm(x[start:stop], log_rate.t())
            require(scores.ndim == 2 and scores.shape == (stop - start, len(rate)),
                    "Full-candidate score matrix has an invalid shape")
            scores = scores - float(repeats) * rate_sum
            require(scores.dtype == torch.float32, "Poisson GEMM must remain float32")
            output[start:stop] = scores.cpu().numpy()
    require(np.isfinite(output).all(), "Nonfinite full-candidate Poisson scores")
    return output


def evaluate_prepared(prepared, ocf_cache, *, device, batch_size=256):
    rows = []
    for entry in prepared["validation"]:
        targets = entry["targets"]
        full_scores = poisson_scores(
            entry["observations"], entry["rate"], entry["repeats"], device, batch_size=batch_size,
        )
        require(full_scores.shape == (len(targets), len(entry["rate"])),
                "Not all query/class scores were constructed")
        positions = EXP48.subset_positions(targets, prepared["subset"])
        logits, evaluation_targets = full_scores[positions], targets[positions]
        require(logits.shape == (len(prepared["subset"]), len(entry["rate"])),
                "Development evaluation matrix does not retain every candidate")
        metrics = EXP48.evaluate_logits(logits, evaluation_targets, ocf_cache)
        rows.append({
            "task": prepared["task"], "arm": ARM, "mouse": prepared["mouse"],
            "fold": prepared["fold"], "block": entry["block"], "epoch": 1,
            "n_rows": len(evaluation_targets),
            "full_score_shape": list(full_scores.shape),
            "evaluation_score_shape": list(logits.shape),
            "repeat_multiplier": entry["repeats"],
            "three_dimensional_score_tensor_constructed": False,
            **metrics,
        })
    return rows


def reference_metadata(reference):
    return {
        "development_results": "test/06_fusion_transduction/48_high_capacity_generalization/development_results.json",
        "selection": "test/06_fusion_transduction/48_high_capacity_generalization/selection.json",
        "development_results_sha256": reference["results_sha256"],
        "selection_sha256": reference["selection_sha256"],
        "run_code_sha256": EXP48_CODE_SHA256,
        "arm": PRODUCTION,
        "selected_epochs": reference["selected_epochs"],
    }


def build_selection(results, reference48):
    """Compare the single deterministic candidate to production's own 48 epoch."""
    tasks = {}
    for task in TASKS:
        candidate = EXP48.find_aggregate(results["macro"], task, ARM, 1)
        production_epoch = reference48["selected_epochs"][task]
        production = EXP48.find_aggregate(
            reference48["results"]["macro"], task, PRODUCTION, production_epoch,
        )
        per_mouse = {}
        for mouse in MICE:
            p = EXP48.find_aggregate(
                reference48["results"]["mouse"], task, PRODUCTION, production_epoch, mouse=mouse,
            )
            h = EXP48.find_aggregate(results["mouse"], task, ARM, 1, mouse=mouse)
            per_mouse[mouse] = float(h["assigned_final_ocf"] - p["assigned_final_ocf"])
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        per_fold = {}
        for fold in fold_ids:
            p = EXP48.find_aggregate(
                reference48["results"]["fold"], task, PRODUCTION, production_epoch, fold=fold,
            )
            h = EXP48.find_aggregate(results["fold"], task, ARM, 1, fold=fold)
            per_fold[fold] = float(h["assigned_final_ocf"] - p["assigned_final_ocf"])
        macro_delta = float(candidate["assigned_final_ocf"] - production["assigned_final_ocf"])
        nonnegative_folds = sum(delta >= 0 for delta in per_fold.values())
        required_folds = REQUIRED_NONNEGATIVE_FOLDS[task]
        gates = {
            "macro_assigned_ocf_delta_at_least_3": macro_delta >= 3.0,
            "both_mouse_deltas_nonnegative": all(delta >= 0 for delta in per_mouse.values()),
            "fold_deltas_nonnegative": nonnegative_folds >= required_folds,
        }
        tasks[task] = {
            "production_selected_epoch": production_epoch,
            "candidate_epoch_selection": "none; one deterministic fit with aggregate adapter epoch=1",
            "selected_macro": {PRODUCTION: production, ARM: candidate},
            "hierarchical_poisson_minus_production": {
                "macro_assigned_final_ocf": macro_delta,
                "per_mouse_assigned_final_ocf": per_mouse,
                "per_fold_assigned_final_ocf": per_fold,
            },
            "nonnegative_fold_count": nonnegative_folds,
            "required_nonnegative_fold_count": required_folds,
            "gates": gates,
            "advance": all(gates.values()),
            "next_step": "development gate passed; separate pristine/final authorization still required"
                         if all(gates.values()) else "stop; no alternate model or hyperparameter search",
        }
    return {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_selection_complete",
        "development_reuse": "development reuse only; not pristine validation; not a Kaggle result",
        "reference48": reference_metadata(reference48),
        "development_reference_metrics_used": True,
        "final_or_heldout_metrics_used": False,
        "tasks": tasks,
        "advance": all(task["advance"] for task in tasks.values()),
    }


def run_experiment():
    preregistration, preregistration_hash = validate_preregistration()
    reference48 = load_reference48()
    require(not RESULTS.exists() and not SELECTION.exists(),
            "Development outputs already exist; refusing an unpreregistered repeated run")
    require(torch.cuda.is_available(), "The preregistered real development run requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    manifest, manifest_hash = EXP48.preflight_cache()
    subset = development_subset()
    print("Loading read-only public raw uint16 counts and gallery images", flush=True)
    blocks = load_cached_blocks(ROOT / "test/.cache/shared_preprocessing_v1", split="train", bin_ms=BIN_MS)
    require(tuple(block.key for block in blocks) == tuple(sorted(BLOCKS)), "Loaded block set mismatch")
    require(all(block.features.dtype == np.uint16 and block.bin_ms == BIN_MS for block in blocks),
            "Real counts must be shared 25ms uint16")
    images = (np.load(ROOT / "test/.cache/shared_preprocessing_v1/pictures.npy",
                      mmap_mode="r", allow_pickle=False) >= 128).astype(np.float32)
    require(images.shape == (N_CLASSES, 64, 64), "Gallery image cache mismatch")
    bank = dct_encode(images, DCT_CUTOFF).astype(np.float32)
    require(bank.shape == (N_CLASSES, DCT_CUTOFF * DCT_CUTOFF), "DCT16 bank mismatch")
    ocf_cache = EXP48.OcfPairCache(images, device="cuda", batch_size=32)
    block_rows, fold_audits = [], []
    started = time.time()
    models_completed = 0
    total_models = (len(TASK2_FOLDS) + len(TASK3_FOLDS)) * len(MICE)
    for task in TASKS:
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        for fold in fold_ids:
            split = make_split(
                blocks, task, seed=TRAIN_SEED if task == "task2" else int(fold),
                validation_session=fold if task == "task2" else "session4",
            )
            validate_split(split, blocks)
            require(split.validation_session != "session4" if task == "task2" else split.seed != 42,
                    "Prohibited final fold encountered")
            for mouse in MICE:
                mouse_blocks = [block for block in blocks if block.mouse == mouse]
                require(len(mouse_blocks) == 4, f"Missing sessions for {mouse}")
                prepare = prepare_task2 if task == "task2" else prepare_task3
                prepared = prepare(mouse_blocks, split, mouse, fold, subset, bank)
                fold_audits.append(prepared["audit"])
                print(f"START deterministic_model={models_completed + 1}/{total_models} "
                      f"task={task} fold={fold} mouse={mouse} arm={ARM}", flush=True)
                block_rows.extend(evaluate_prepared(prepared, ocf_cache, device=device))
                models_completed += 1
                print(f"DONE deterministic_model={models_completed}/{total_models} "
                      f"elapsed_seconds={time.time() - started:.1f}", flush=True)
                prepared.clear()
                gc.collect()
    require(models_completed == total_models and len(block_rows) == 22, "Incomplete deterministic result grid")
    results = {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_results_complete",
        "data_policy": "public train only; development reuse; no Task2 session4 or Task3 seed42 final evaluation",
        "development_reuse": "development reuse only; not pristine validation; not a Kaggle result",
        "preregistration_sha256": preregistration_hash,
        "cache_manifest_sha256": manifest_hash,
        "reference48": reference_metadata(reference48),
        "run": {
            "device": torch.cuda.get_device_name(device),
            "implementation_sha256": file_sha256(__file__),
            "bin_ms": BIN_MS,
            "raw_count_dtype": "uint16",
            "arm": ARM,
            "training_epochs": 0,
            "hyperparameter_or_epoch_search": False,
            "deterministic_models_completed": models_completed,
            "model_artifacts_retained": False,
            "score_matmul": "CUDA float32 torch.mm with TF32 disabled; two-dimensional matrices only",
            "gallery": {"shape": list(images.shape), "sha256": array_sha256(images),
                        "binarization_threshold": 128, "dct_cutoff": DCT_CUTOFF,
                        "dct_bank_sha256": array_sha256(bank)},
            "ocf_pair_cache": {"storage": "memory only", "scoring_calls": ocf_cache.calls,
                               "unique_pairs_scored": ocf_cache.scored_pairs},
            "elapsed_seconds": float(time.time() - started),
        },
        "development_picture_subset": subset.tolist(),
        "fold_audits": fold_audits,
        "block": block_rows,
        "mouse_fold": EXP48.aggregate_rows(block_rows, "mouse_fold"),
        "mouse": EXP48.aggregate_rows(block_rows, "mouse"),
        "fold": EXP48.aggregate_rows(block_rows, "fold"),
        "macro": EXP48.aggregate_rows(block_rows, "macro"),
    }
    require(len(results["mouse_fold"]) == 10 and len(results["mouse"]) == 4,
            "Mouse aggregation grid mismatch")
    require(len(results["fold"]) == 5 and len(results["macro"]) == 2,
            "Fold/macro aggregation grid mismatch")
    for task in TASKS:
        for level in EXPECTED_EVALUATION_ROWS[task]:
            rows = [row for row in results[level] if row["task"] == task]
            require(all(row["n_evaluation_rows"] == EXPECTED_EVALUATION_ROWS[task][level] for row in rows),
                    f"Experiment 51 {level} n_evaluation_rows mismatch for {task}")
    write_json(RESULTS, results)
    selection = build_selection(results, reference48)
    selection["preregistration_sha256"] = preregistration_hash
    selection["development_results_sha256"] = file_sha256(RESULTS)
    write_json(SELECTION, selection)
    require(hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest() == preregistration_hash,
            "Preregistration changed during the run")
    print(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print(f"Wrote {RESULTS} and {SELECTION}; no model artifacts retained", flush=True)


def synthetic_gate_results():
    reference_results = {"macro": [], "mouse": [], "fold": []}
    candidate_results = {"macro": [], "mouse": [], "fold": []}
    selected_epochs = {}
    for task in TASKS:
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        production_epoch = 2
        selected_epochs[task] = production_epoch
        production_common = {
            "task": task, "arm": PRODUCTION, "epoch": production_epoch,
            "raw_top1_accuracy": 0.1, "raw_recall_at_32": 0.2,
            "raw_top1_official_ocf": 20.0, "assigned_final_ocf": 50.0,
        }
        candidate_common = {
            "task": task, "arm": ARM, "epoch": 1,
            "raw_top1_accuracy": 0.2, "raw_recall_at_32": 0.4,
            "raw_top1_official_ocf": 24.0, "assigned_final_ocf": 53.1,
        }
        reference_results["macro"].append(dict(production_common))
        candidate_results["macro"].append(dict(candidate_common))
        for mouse in MICE:
            reference_results["mouse"].append({**production_common, "mouse": mouse})
            candidate_results["mouse"].append({**candidate_common, "mouse": mouse})
        for fold in fold_ids:
            reference_results["fold"].append({**production_common, "fold": fold})
            candidate_results["fold"].append({**candidate_common, "fold": fold})
    reference = {
        "results": reference_results,
        "selected_epochs": selected_epochs,
        "results_sha256": "synthetic-reference-results",
        "selection_sha256": "synthetic-reference-selection",
    }
    return candidate_results, reference


def smoke_test():
    """Synthetic-only numerical/isolation checks; never read real data or experiment 48 metrics."""
    validate_preregistration()

    observations = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    rates = np.array([[0.5, 2.0], [4.0, 1.0]], dtype=np.float32)
    expected = np.array([
        [np.log(0.5) + 2 * np.log(2.0) - 2.5, np.log(4.0) + 2 * np.log(1.0) - 5.0],
        [3 * np.log(2.0) - 2.5, 3 * np.log(1.0) - 5.0],
    ], dtype=np.float32)
    mm_dimensions = []
    original_mm = torch.mm

    def checked_mm(left, right):
        require(left.ndim == 2 and right.ndim == 2, "A three-dimensional GEMM was attempted")
        mm_dimensions.append((tuple(left.shape), tuple(right.shape)))
        return original_mm(left, right)

    torch.mm = checked_mm
    try:
        actual = poisson_scores(observations, rates, 1, "cpu", batch_size=1)
        actual_four = poisson_scores(observations, rates, 4, "cpu", batch_size=2)
    finally:
        torch.mm = original_mm
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    np.testing.assert_allclose(
        actual_four, expected - 3 * rates.sum(axis=1, dtype=np.float32)[None, :], atol=2e-6, rtol=0,
    )
    require(mm_dimensions and all(len(left) == len(right) == 2 for left, right in mm_dimensions),
            "Two-dimensional full-candidate GEMM check failed")

    sums = np.array([[6.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    prior_example = np.array([[2.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    posterior_example = posterior_rates(sums, 3, prior_example)
    np.testing.assert_allclose(
        sums / 3, np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32), atol=0, rtol=0,
    )
    np.testing.assert_allclose(
        posterior_example, np.array([[2.0, 2.0], [0.5, 1.0]], dtype=np.float32), atol=0, rtol=0,
    )
    calibrated, raw_gain, gain = calibrate_from_marginals(
        posterior_example, np.array([0.0, 3.0], dtype=np.float32),
        np.array([100.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(raw_gain, np.array([100.01 / 0.01, 1.01 / 3.01], dtype=np.float32),
                               atol=2e-6, rtol=0)
    require(gain[0] == 4.0 and np.isfinite(calibrated).all(), "Fixed gain clipping failed")

    n_classes, repeats = 6, 4
    subset = np.arange(n_classes, dtype=np.int64)
    blocks = [replace(block, features=np.asarray(block.features, dtype=np.uint16))
              for block in EXP48.synthetic_blocks(n_classes=n_classes, repeats=repeats)]
    mice1_blocks = [block for block in blocks if block.mouse == "mice1"]
    rng = np.random.default_rng(772)
    images = (rng.random((n_classes, 64, 64)) > 0.82).astype(np.float32)
    images[np.arange(n_classes), np.arange(n_classes) % 64, :] = 1
    bank = dct_encode(images, 16).astype(np.float32)

    task2_split = make_split(blocks, "task2", seed=42, validation_session="session1")
    prepared2 = prepare_task2(mice1_blocks, task2_split, "mice1", "session1", subset, bank,
                              n_classes=n_classes)
    require(prepared2["audit"]["prior_fit"]["train_rows"] == 3 * n_classes,
            "Task2 prior must use three per-session mean banks")
    entry2 = prepared2["validation"][0]
    require(entry2["n_repeats"] == 12 and entry2["repeats"] == 4,
            "Task2 n=12/four-repeat scoring policy failed")
    require(entry2["source_n"] == 3 * n_classes * 4 and entry2["target_n"] == n_classes * 4,
            "Task2 source/target single-trial marginal counts failed")
    held = next(block for block in mice1_blocks if block.session == "session1")
    held_part = task2_split.blocks[held.key]
    grouped_targets, expected_grouped, grouped_counts = target_statistics(
        held.features[held_part.validation_trials], held.trial_targets[held_part.validation_trials],
    )
    complete_targets(grouped_targets, grouped_counts, n_classes, repeats)
    np.testing.assert_array_equal(entry2["observations"], expected_grouped)
    direct2 = expected_grouped @ np.log(entry2["rate"]).T - 4 * entry2["rate"].sum(axis=1)
    np.testing.assert_allclose(
        poisson_scores(entry2["observations"], entry2["rate"], 4, "cpu"), direct2,
        atol=2e-4, rtol=2e-5,
    )

    shifted_blocks = [replace(block, features=block.features + np.uint16(1000))
                      if block.key == held.key else block for block in mice1_blocks]
    shifted2 = prepare_task2(shifted_blocks, task2_split, "mice1", "session1", subset, bank,
                             n_classes=n_classes)["validation"][0]
    for key in ("prior", "empirical_rate", "posterior", "source_marginal"):
        np.testing.assert_array_equal(entry2[key], shifted2[key])
    require(not np.array_equal(entry2["target_marginal"], shifted2["target_marginal"])
            and not np.array_equal(entry2["gain"], shifted2["gain"]),
            "Held-out counts should alter only target calibration, not training-derived rates")

    permuted_blocks = [replace(block, targets=np.roll(block.targets, 1))
                       if block.key == held.key else block for block in mice1_blocks]
    permuted2 = prepare_task2(permuted_blocks, task2_split, "mice1", "session1", subset, bank,
                              n_classes=n_classes)["validation"][0]
    for key in ("prior", "empirical_rate", "posterior", "source_marginal", "target_marginal",
                "raw_gain", "gain", "rate"):
        np.testing.assert_array_equal(entry2[key], permuted2[key])
    require(prepared2["audit"]["rates"][0]["target_marginal_uses_labels"] is False,
            "Target-marginal label-isolation audit missing")

    task3_split = make_split(blocks, "task3", seed=20261126)
    prepared3 = prepare_task3(mice1_blocks, task3_split, "mice1", "20261126", subset, bank,
                              n_classes=n_classes)
    require(prepared3["audit"]["prior_fit"]["train_rows"] == 4 * n_classes,
            "Task3 prior must use four per-block mean banks")
    require(len(prepared3["validation"]) == 4, "Task3 requires four independent block posteriors")
    require(any(not np.array_equal(prepared3["validation"][0]["posterior"], entry["posterior"])
                for entry in prepared3["validation"][1:]),
            "Task3 block posteriors were incorrectly pooled")
    for entry in prepared3["validation"]:
        require(entry["n_repeats"] == 3 and entry["repeats"] == 1,
                "Task3 n=3/single-trial scoring policy failed")
        require(entry["source_n"] == 3 * n_classes and entry["target_n"] == n_classes,
                "Task3 block marginal counts failed")
        block = next(block for block in mice1_blocks if block.key == entry["block"])
        part = task3_split.blocks[block.key]
        raw_validation = np.asarray(block.features[part.validation_trials], dtype=np.float32)
        labels = block.trial_targets[part.validation_trials]
        order = np.argsort(labels, kind="stable")
        np.testing.assert_array_equal(entry["observations"], raw_validation[order])

    ocf = EXP48.OcfPairCache(images, device="cpu", batch_size=2)
    rows2 = evaluate_prepared(prepared2, ocf, device="cpu", batch_size=2)
    rows3 = evaluate_prepared(prepared3, ocf, device="cpu", batch_size=2)
    require(len(rows2) == 1 and len(rows3) == 4, "Smoke block grid failed")
    for row in rows2 + rows3:
        require(row["full_score_shape"] == [n_classes, n_classes]
                and row["evaluation_score_shape"] == [n_classes, n_classes],
                "Smoke did not evaluate the full candidate axis")
        require(row["three_dimensional_score_tensor_constructed"] is False,
                "Three-dimensional score tensor audit failed")
        require(all(np.isfinite(row[metric]) for metric in METRICS), "Nonfinite smoke metric")
    require(rows2[0]["repeat_multiplier"] == 4
            and all(row["repeat_multiplier"] == 1 for row in rows3), "Score repeat multipliers changed")
    for level in ("mouse_fold", "mouse", "fold", "macro"):
        aggregated = EXP48.aggregate_rows(rows2 + rows3, level)
        for row in aggregated:
            expected_rows = n_classes if row["task"] == "task2" else 4 * n_classes
            require(row["n_evaluation_rows"] == expected_rows,
                    f"Corrected aggregation count failed at {level}/{row['task']}")

    candidate_results, reference = synthetic_gate_results()
    positive = build_selection(candidate_results, reference)
    require(positive["advance"] and all(positive["tasks"][task]["advance"] for task in TASKS),
            "Positive advancement-gate case failed")
    negative_macro = json.loads(json.dumps(candidate_results))
    for level in ("macro", "mouse", "fold"):
        for row in negative_macro[level]:
            row["assigned_final_ocf"] = 52.9
    failed_macro = build_selection(negative_macro, reference)
    require(not failed_macro["advance"] and all(
        not task["gates"]["macro_assigned_ocf_delta_at_least_3"] for task in failed_macro["tasks"].values()
    ), "Negative macro gate case failed")
    negative_mouse = json.loads(json.dumps(candidate_results))
    for row in negative_mouse["mouse"]:
        if row["mouse"] == "mice2":
            row["assigned_final_ocf"] = 49.0
    require(not build_selection(negative_mouse, reference)["advance"],
            "Negative per-mouse gate case failed")
    negative_fold = json.loads(json.dumps(candidate_results))
    for row in negative_fold["fold"]:
        if row["task"] == "task3" and row["fold"] == "20261127":
            row["assigned_final_ocf"] = 49.0
    failed_fold = build_selection(negative_fold, reference)
    require(failed_fold["tasks"]["task2"]["advance"] and not failed_fold["tasks"]["task3"]["advance"],
            "Task3 2/2 nonnegative-fold gate case failed")
    print(
        "SMOKE OK: hand Poisson; empirical/prior/posterior/gain; train-validation and label isolation; "
        "Task2 four-repeat/Task3 single; full 2-D candidate GEMM; 48 evaluation/aggregation; +/- gates",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        run_experiment()


if __name__ == "__main__":
    main()
