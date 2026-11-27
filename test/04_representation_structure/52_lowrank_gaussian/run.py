"""Preregistered development-only low-rank Gaussian candidate."""

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
ARM = "lowrank_gaussian"
N_CLASSES = 2748
BIN_MS = 25
TRAIN_SEED = 42
TASK2_FOLDS = ("session1", "session2", "session3")
TASK3_FOLDS = ("20261126", "20261127")
RANKS = {"task2": 4, "task3": 1}
KAPPAS = {"task2": 1.0 / 3.0, "task3": 4.0 / 3.0}
TEMPLATE_REPEATS = {"task2": 12, "task3": 3}
QUERY_REPEATS = {"task2": 4, "task3": 1}
VARIANCE_FLOOR = 1e-6
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
    """Reuse evaluation and aggregation code without importing experiment 48 metrics."""
    raw = RUN_48.read_bytes()
    source = raw.decode("utf-8")
    require(
        'count_key = "n_rows" if level == "mouse_fold" else "n_evaluation_rows"' in source
        and 'row["n_evaluation_rows"] = sum(member[count_key] for member in members)' in source,
        "Experiment 48 aggregate_rows is not the corrected n_evaluation_rows version",
    )
    spec = importlib.util.spec_from_file_location("siclink_experiment48_for52", RUN_48)
    require(spec is not None and spec.loader is not None, "Cannot load experiment 48 code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(raw).hexdigest()


EXP48, EXP48_CODE_SHA256 = load_experiment48_code()


def development_subset():
    return np.sort(np.random.default_rng(20261128).permutation(N_CLASSES)[:256]).astype(np.int64)


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def array_audit(array):
    array = np.asarray(array)
    require(array.ndim in (1, 2) and array.size and np.isfinite(array).all(),
            "Cannot audit a nonfinite or non-vector/matrix value")
    values = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": array_sha256(array),
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
    require(
        data["loader"] == "src.preprocessing.load_cached_blocks(cache, split='train', bin_ms=25)",
        "Cache loader contract changed",
    )
    require(data["cache_access"] == "read-only; never rebuild the cache", "Cache must be read-only")
    require(data["public_partition"] == "train only", "Only public train data are permitted")
    require(data["mice"] == list(MICE) and data["blocks"] == list(BLOCKS), "Block contract changed")
    require(data["n_classes"] == N_CLASSES and data["subset_size"] == 256, "Class/subset changed")
    require(
        data["development_picture_subset"]
        == "np.sort(np.random.default_rng(20261128).permutation(2748)[:256])",
        "Development subset changed",
    )
    require(data["same_subset_for_every_block_fold_and_mouse"] is True, "Subset must be shared")
    folds = value["folds"]
    require(folds["task2"]["fold_ids"] == list(TASK2_FOLDS), "Task2 folds changed")
    require(folds["task3"]["fold_ids"] == list(TASK3_FOLDS), "Task3 folds changed")
    require("session4" not in folds["task2"]["fold_ids"], "Task2 session4 is prohibited")
    require("42" not in folds["task3"]["fold_ids"], "Task3 seed42 is prohibited")
    arm = value["arm"]
    require(arm["name"] == ARM and arm["arms_in_this_experiment"] == [ARM], "Candidate arm changed")
    require(arm["deterministic_fit_only"] is True and arm["training_epochs"] == 0,
            "The candidate must be deterministic and epoch-free")
    require(arm["ranks"] == RANKS and arm["rank_search"] is False, "Rank contract changed")
    require(arm["hyperparameter_search"] is False, "Hyperparameter search is prohibited")
    transform = value["transform_and_templates"]
    require(transform["transform"].startswith("z=log1p(raw_count)"), "log1p transform changed")
    require("12 aligned source trials" in transform["task2_source_alignment"],
            "Task2 12-repeat source template changed")
    require("10992 unlabeled" in transform["task2_target_alignment"],
            "Task2 unlabeled target marginal changed")
    require("three raw-z training trials" in transform["task3_templates"],
            "Task3 independent three-repeat templates changed")
    residual = value["residual_covariance"]
    require(residual["formula"] == "C=(E^T E)/(N-G), where E contains the specified residuals and G is the number of fitted class or block-class templates.",
            "Residual covariance formula changed")
    require(residual["loading"] == "L=U_rank*sqrt(max(lambda_rank,0)), shape n_features by rank.",
            "Loading formula changed")
    require(residual["variance_floor"] == VARIANCE_FLOOR and residual["variance_floor_search"] is False,
            "Variance floor changed")
    predictive = value["predictive_likelihood"]
    require(predictive["task2"] == {
        "n_template_repeats": 12, "n_query_repeats": 4, "kappa": "1/4+1/12=1/3",
    }, "Task2 predictive covariance changed")
    require(predictive["task3"] == {
        "n_template_repeats": 3, "n_query_repeats": 1, "kappa": "1+1/3=4/3",
    }, "Task3 predictive covariance changed")
    require("A=L/sqrt(d)" in predictive["woodbury"] and "H=I+A^T A" in predictive["woodbury"],
            "Woodbury formula changed")
    require("Never construct" in predictive["candidate_matrix"], "Three-dimensional guard missing")
    require(predictive["score_modification"].startswith("None:"), "Score modification is prohibited")
    evaluation = value["evaluation"]
    require("corrected aggregate_rows unchanged" in evaluation["implementation"],
            "Experiment 48 evaluation reuse contract changed")
    require(evaluation["metrics"] == list(METRICS), "Metric contract changed")
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
    require("dependence on experiment 49, 50, or 51 outputs" in prohibited,
            "Independence declaration missing")
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


def log1p_counts(raw):
    raw = np.asarray(raw)
    require(raw.ndim == 2 and len(raw) and np.isfinite(raw).all() and np.all(raw >= 0),
            "Raw counts must be a finite nonnegative matrix")
    return np.log1p(raw.astype(np.float64))


def target_statistics(features, targets):
    """Return sorted float64 sums and repeat counts without altering input rows."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    require(features.ndim == 2 and targets.ndim == 1 and len(features) == len(targets) and len(targets),
            "Invalid target-statistic inputs")
    require(np.isfinite(features).all(), "Features must be finite")
    order = np.argsort(targets, kind="stable")
    sorted_targets, sorted_features = targets[order], features[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_targets)) + 1))
    counts = np.diff(np.concatenate((starts, [len(targets)]))).astype(np.int64)
    sums = np.add.reduceat(sorted_features, starts, axis=0, dtype=np.float64)
    return sorted_targets[starts], sums, counts


def complete_targets(targets, counts, n_classes, expected_repeats):
    require(np.array_equal(targets, np.arange(n_classes, dtype=np.int64)), "Target coverage/order mismatch")
    require(np.all(counts == expected_repeats), "Per-target repeat count mismatch")


def fit_lowrank(residuals, *, n, g, rank, n_features):
    """Fit C=E^TE/(N-G), then its descending rank eigenpairs and diagonal floor."""
    require(isinstance(n, int) and isinstance(g, int) and n > g > 0, "Invalid N/G degrees of freedom")
    require(isinstance(rank, int) and 0 < rank <= n_features, "Invalid covariance rank")
    covariance = np.zeros((n_features, n_features), dtype=np.float64)
    rows = 0
    for residual in residuals:
        residual = np.asarray(residual, dtype=np.float64)
        require(residual.ndim == 2 and residual.shape[1] == n_features and np.isfinite(residual).all(),
                "Residuals must be finite two-dimensional matrices")
        covariance += residual.T @ residual
        rows += len(residual)
    require(rows == n, "Residual row count does not equal N")
    covariance /= float(n - g)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    retained_eigenvalues = eigenvalues[-rank:][::-1].copy()
    retained_vectors = eigenvectors[:, -rank:][:, ::-1]
    loading = retained_vectors * np.sqrt(np.maximum(retained_eigenvalues, 0.0))[None, :]
    unfloored = np.diag(covariance).copy() - np.sum(loading * loading, axis=1)
    diagonal = np.maximum(unfloored, VARIANCE_FLOOR)
    require(loading.shape == (n_features, rank) and diagonal.shape == (n_features,),
            "Loading/diagonal shape mismatch")
    require(np.isfinite(loading).all() and np.isfinite(diagonal).all(), "Nonfinite covariance fit")
    return {
        "covariance": covariance,
        "eigenvalues": retained_eigenvalues,
        "loading": loading,
        "diagonal": diagonal,
        "diagonal_floor_count": int(np.sum(unfloored < VARIANCE_FLOOR)),
        "n": n,
        "g": g,
        "df": n - g,
        "rank": rank,
    }


def model_audit(model, templates, task):
    return {
        "N": model["n"],
        "G": model["g"],
        "df": model["df"],
        "n_template_repeats": TEMPLATE_REPEATS[task],
        "n_query_repeats": QUERY_REPEATS[task],
        "kappa": KAPPAS[task],
        "rank": model["rank"],
        "retained_eigenvalues_descending": model["eigenvalues"].tolist(),
        "diagonal_floor": VARIANCE_FLOOR,
        "diagonal_floor_count": model["diagonal_floor_count"],
        "templates": {key: array_audit(value) for key, value in templates.items()},
        "covariance": array_audit(model["covariance"]),
        "loading": array_audit(model["loading"]),
        "diagonal": array_audit(model["diagonal"]),
        "covariance_formula": "C=E^TE/(N-G)",
        "loading_formula": "L=U_rank*sqrt(max(lambda_rank,0))",
        "diagonal_formula": "d=max(diag(C)-rowsum(L^2),1e-6)",
        "float64_sufficient_statistics_and_eigendecomposition": True,
    }


def prepare_task2(mouse_blocks, split, mouse, fold, subset, *, n_classes=N_CLASSES):
    """Fit three aligned source sessions, then form label-aware four-repeat queries."""
    source_parts = []
    audit_blocks = []
    held = held_part = held_z = None
    n_features = mouse_blocks[0].n_features
    for block in mouse_blocks:
        require(block.n_features == n_features, "Feature dimension changed across source blocks")
        part = split.blocks[block.key]
        if block.session == fold:
            require(len(part.train_trials) == 0 and len(part.validation_trials) == block.n_trials,
                    "Task2 target session is not fully held out")
            held, held_part = block, part
            held_z = log1p_counts(block.features[part.validation_trials])
            continue
        require(len(part.validation_trials) == 0 and len(part.train_trials) == block.n_trials,
                "Task2 training session leakage")
        z = log1p_counts(block.features[part.train_trials])
        labels = np.asarray(block.trial_targets[part.train_trials], dtype=np.int64)
        targets, sums, counts = target_statistics(z, labels)
        complete_targets(targets, counts, n_classes, 4)
        offset = z.mean(axis=0, dtype=np.float64)
        source_parts.append({
            "block": block.key, "z": z, "labels": labels, "sums": sums,
            "counts": counts, "offset": offset,
        })
        audit_blocks.append({
            "block": block.key, "role": "train_source", "train_trials": len(z),
            "validation_trials": 0, "train_repeats_per_glyph": 4,
            "offset_source": "all unlabeled train_trials in this source block",
        })
    require(held is not None and len(source_parts) == 3, "Task2 must have three source sessions")
    reference_offset = np.mean([part["offset"] for part in source_parts], axis=0, dtype=np.float64)
    target_offset = held_z.mean(axis=0, dtype=np.float64)
    aligned_sum = np.zeros_like(source_parts[0]["sums"], dtype=np.float64)
    total_counts = np.zeros(n_classes, dtype=np.int64)
    for part in source_parts:
        aligned_sum += part["sums"] + part["counts"][:, None] * (reference_offset - part["offset"])
        total_counts += part["counts"]
    require(np.all(total_counts == 12), "Task2 global templates require exactly 12 repeats")
    templates = aligned_sum / total_counts[:, None]
    n = sum(len(part["z"]) for part in source_parts)
    g = n_classes
    residuals = (
        part["z"] - part["offset"] + reference_offset - templates[part["labels"]]
        for part in source_parts
    )
    model = fit_lowrank(residuals, n=n, g=g, rank=RANKS["task2"], n_features=n_features)

    # Local validation labels are first accessed here, after every fitted statistic is fixed.
    validation_labels = np.asarray(held.trial_targets[held_part.validation_trials], dtype=np.int64)
    targets, sums, counts = target_statistics(held_z, validation_labels)
    complete_targets(targets, counts, n_classes, 4)
    queries = sums / counts[:, None] - target_offset + reference_offset
    positions = EXP48.subset_positions(targets, subset)
    entry = {
        "block": held.key,
        "observations": queries[positions],
        "targets": targets[positions],
        "templates": templates,
        "n_template_repeats": 12,
        "n_query_repeats": 4,
        "repeat_aggregation": True,
    }
    audit_blocks.append({
        "block": held.key, "role": "validation_target", "train_trials": 0,
        "unlabeled_target_marginal_trials": len(held_z), "validation_trials": len(held_z),
        "grouped_validation_rows": n_classes, "evaluation_rows": len(positions),
        "query_repeats": 4, "target_marginal_uses_labels": False,
        "labels_used_only_after_model_fitting": True,
    })
    audit = {
        "task": "task2", "mouse": mouse, "fold": fold,
        "train_validation_blocks": audit_blocks,
        "source_n": n, "target_n": len(held_z),
        "reference_offset": array_audit(reference_offset),
        "target_offset": array_audit(target_offset),
        "target_marginal_uses_labels": False,
        "target_marginal_label_isolation": "mean of all 10992 unlabeled held-session log1p trials",
        "validation_labels_used_only_after_templates_covariance_and_target_marginal": True,
        **model_audit(model, {"global_12_repeat": templates}, "task2"),
    }
    return {
        "task": "task2", "mouse": mouse, "fold": fold, "subset": subset,
        "templates": templates, "reference_offset": reference_offset, "target_offset": target_offset,
        "validation": [entry], "audit": audit, "kappa": KAPPAS["task2"], **model,
    }


def prepare_task3(mouse_blocks, split, mouse, fold, subset, *, n_classes=N_CLASSES):
    """Fit four independent block templates and pool only their within-block residuals."""
    source_parts = []
    audit_blocks = []
    n_features = mouse_blocks[0].n_features
    for block in mouse_blocks:
        require(block.n_features == n_features, "Feature dimension changed across Task3 blocks")
        part = split.blocks[block.key]
        require(len(part.train_trials) == 3 * n_classes and len(part.validation_trials) == n_classes,
                "Task3 train/validation trial counts changed")
        z = log1p_counts(block.features[part.train_trials])
        labels = np.asarray(block.trial_targets[part.train_trials], dtype=np.int64)
        targets, sums, counts = target_statistics(z, labels)
        complete_targets(targets, counts, n_classes, 3)
        templates = sums / counts[:, None]
        source_parts.append({
            "block": block, "part": part, "z": z, "labels": labels, "templates": templates,
        })
        audit_blocks.append({
            "block": block.key, "role": "train_source", "train_trials": len(z),
            "train_repeats_per_glyph": 3, "validation_trials": len(part.validation_trials),
            "target_marginal_used": False,
        })
    require(len(source_parts) == 4, "Task3 requires four blocks")
    n = sum(len(part["z"]) for part in source_parts)
    g = 4 * n_classes
    residuals = (part["z"] - part["templates"][part["labels"]] for part in source_parts)
    model = fit_lowrank(residuals, n=n, g=g, rank=RANKS["task3"], n_features=n_features)

    entries = []
    for part in source_parts:
        block, block_split = part["block"], part["part"]
        validation_z = log1p_counts(block.features[block_split.validation_trials])
        validation_labels = np.asarray(block.trial_targets[block_split.validation_trials], dtype=np.int64)
        order = np.argsort(validation_labels, kind="stable")
        sorted_targets = validation_labels[order]
        require(np.array_equal(sorted_targets, np.arange(n_classes, dtype=np.int64)),
                "Task3 validation must contain exactly one trial per class")
        positions = EXP48.subset_positions(sorted_targets, subset)
        entries.append({
            "block": block.key,
            "observations": validation_z[order][positions],
            "targets": sorted_targets[positions],
            "templates": part["templates"],
            "n_template_repeats": 3,
            "n_query_repeats": 1,
            "repeat_aggregation": False,
        })
        audit_blocks.append({
            "block": block.key, "role": "validation_single_trial", "train_trials": 0,
            "validation_trials": len(validation_z), "evaluation_rows": len(positions),
            "query_repeats": 1, "repeat_aggregation": False,
            "labels_used_only_after_model_fitting": True,
        })
    audit = {
        "task": "task3", "mouse": mouse, "fold": fold,
        "train_validation_blocks": audit_blocks,
        "source_n": n, "target_n": 4 * n_classes,
        "template_policy": "independent three-repeat template for each block; no cross-block template pooling",
        "residual_policy": "pool within-block residuals over four blocks only",
        "target_marginal_uses_labels": False,
        "target_marginal_used": False,
        "validation_labels_used_only_after_templates_and_covariance": True,
        **model_audit(
            model, {part["block"].key: part["templates"] for part in source_parts}, "task3",
        ),
    }
    return {
        "task": "task3", "mouse": mouse, "fold": fold, "subset": subset,
        "validation": entries, "audit": audit, "kappa": KAPPAS["task3"], **model,
    }


def lowrank_scores(observations, templates, loading, diagonal, kappa, device, *, batch_size=256):
    """Expand Woodbury Mahalanobis scores using only two-dimensional float32 GEMMs."""
    observations = np.asarray(observations, dtype=np.float64)
    templates = np.asarray(templates, dtype=np.float64)
    loading = np.asarray(loading, dtype=np.float64)
    diagonal = np.asarray(diagonal, dtype=np.float64)
    require(observations.ndim == 2 and templates.ndim == 2 and loading.ndim == 2 and diagonal.ndim == 1,
            "Woodbury inputs must be vector/two-dimensional matrices")
    require(observations.shape[1] == templates.shape[1] == loading.shape[0] == len(diagonal),
            "Woodbury feature dimensions do not match")
    require(len(observations) and len(templates) and loading.shape[1] > 0,
            "Woodbury query/class/rank axes must be nonempty")
    require(np.isfinite(observations).all() and np.isfinite(templates).all()
            and np.isfinite(loading).all() and np.isfinite(diagonal).all() and np.all(diagonal > 0),
            "Woodbury inputs must be finite with positive diagonal")
    require(isinstance(batch_size, int) and batch_size > 0 and np.isfinite(kappa) and kappa > 0,
            "Invalid batch size or kappa")
    rank = loading.shape[1]
    a64 = loading / np.sqrt(diagonal)[:, None]
    h64 = np.eye(rank, dtype=np.float64) + a64.T @ a64
    h_inverse64 = np.linalg.solve(h64, np.eye(rank, dtype=np.float64))
    h_inverse64 = (h_inverse64 + h_inverse64.T) * 0.5

    device = torch.device(device)
    x = torch.as_tensor(np.ascontiguousarray(observations, dtype=np.float32), device=device)
    mu = torch.as_tensor(np.ascontiguousarray(templates, dtype=np.float32), device=device)
    a = torch.as_tensor(np.ascontiguousarray(a64, dtype=np.float32), device=device)
    sqrt_d = torch.as_tensor(np.ascontiguousarray(np.sqrt(diagonal), dtype=np.float32), device=device)
    h_inverse = torch.as_tensor(np.ascontiguousarray(h_inverse64, dtype=np.float32), device=device)
    require(x.dtype == mu.dtype == a.dtype == sqrt_d.dtype == h_inverse.dtype == torch.float32,
            "Woodbury scoring must use float32 tensors")
    output = np.empty((len(observations), len(templates)), dtype=np.float32)
    with torch.inference_mode():
        mu_w = mu / sqrt_d
        mu_a = torch.mm(mu_w, a)
        mu_h = torch.mm(mu_a, h_inverse)
        class_q = (mu_w * mu_w).sum(dim=1) - (mu_h * mu_a).sum(dim=1)
        require(class_q.ndim == 1 and len(class_q) == len(templates), "Class-term shape mismatch")
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            query_w = x[start:stop] / sqrt_d
            query_a = torch.mm(query_w, a)
            query_h = torch.mm(query_a, h_inverse)
            query_q = (query_w * query_w).sum(dim=1) - (query_h * query_a).sum(dim=1)
            base_cross = torch.mm(query_w, mu_w.t())
            lowrank_cross = torch.mm(query_a, mu_h.t())
            mahalanobis = (
                query_q[:, None] + class_q[None, :] - 2.0 * (base_cross - lowrank_cross)
            )
            scores = -0.5 * mahalanobis / float(kappa)
            require(scores.ndim == 2 and scores.shape == (stop - start, len(templates))
                    and scores.dtype == torch.float32, "Full-candidate score matrix has an invalid shape")
            output[start:stop] = scores.cpu().numpy()
    require(np.isfinite(output).all(), "Nonfinite full-candidate Woodbury scores")
    return output


def evaluate_prepared(prepared, ocf_cache, *, device, batch_size=256):
    rows = []
    for entry in prepared["validation"]:
        targets = entry["targets"]
        positions = EXP48.subset_positions(targets, prepared["subset"])
        require(np.array_equal(positions, np.arange(len(targets))), "Validation rows are not subset-sorted")
        observations = entry["observations"][positions]
        evaluation_targets = targets[positions]
        full_scores = lowrank_scores(
            observations, entry["templates"], prepared["loading"], prepared["diagonal"],
            prepared["kappa"], device, batch_size=batch_size,
        )
        require(full_scores.shape == (len(evaluation_targets), len(entry["templates"])),
                "Not all query/class scores were constructed")
        logits = full_scores
        require(logits.shape == (len(prepared["subset"]), len(entry["templates"])),
                "Development evaluation matrix does not retain every candidate")
        metrics = EXP48.evaluate_logits(logits, evaluation_targets, ocf_cache)
        rows.append({
            "task": prepared["task"], "arm": ARM, "mouse": prepared["mouse"],
            "fold": prepared["fold"], "block": entry["block"], "epoch": 1,
            "n_rows": len(evaluation_targets),
            "full_score_shape": list(full_scores.shape),
            "evaluation_score_shape": list(logits.shape),
            "n_template_repeats": entry["n_template_repeats"],
            "n_query_repeats": entry["n_query_repeats"],
            "repeat_aggregation": entry["repeat_aggregation"],
            "kappa": prepared["kappa"],
            "rank": prepared["rank"],
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
    """Compare the deterministic candidate to production's own selected 48 epoch."""
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
            "lowrank_gaussian_minus_production": {
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
    _preregistration, preregistration_hash = validate_preregistration()
    require(not RESULTS.exists() and not SELECTION.exists(),
            "Development outputs already exist; refusing an unpreregistered repeated run")
    reference48 = load_reference48()
    require(torch.cuda.is_available(), "The preregistered real development run requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    _manifest, manifest_hash = EXP48.preflight_cache()
    subset = development_subset()
    print("Loading read-only public raw uint16 counts and gallery images", flush=True)
    blocks = load_cached_blocks(ROOT / "test/.cache/shared_preprocessing_v1", split="train", bin_ms=BIN_MS)
    require(tuple(block.key for block in blocks) == tuple(sorted(BLOCKS)), "Loaded block set mismatch")
    require(all(block.features.dtype == np.uint16 and block.bin_ms == BIN_MS for block in blocks),
            "Real counts must be shared 25ms uint16")
    images = (np.load(ROOT / "test/.cache/shared_preprocessing_v1/pictures.npy",
                      mmap_mode="r", allow_pickle=False) >= 128).astype(np.float32)
    require(images.shape == (N_CLASSES, 64, 64), "Gallery image cache mismatch")
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
                print(f"START deterministic_model={models_completed + 1}/{total_models} "
                      f"task={task} fold={fold} mouse={mouse} arm={ARM}", flush=True)
                prepare = prepare_task2 if task == "task2" else prepare_task3
                prepared = prepare(mouse_blocks, split, mouse, fold, subset)
                fold_audits.append(prepared["audit"])
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
            "transform": "log1p in float64; float32 copies only for scoring",
            "arm": ARM,
            "training_epochs": 0,
            "hyperparameter_or_epoch_search": False,
            "ranks": RANKS,
            "kappas": KAPPAS,
            "deterministic_models_completed": models_completed,
            "model_artifacts_retained": False,
            "covariance": "C=E^TE/(N-G); descending float64 eigenpairs; floored residual diagonal",
            "score_matmul": "CUDA float32 torch.mm with TF32 disabled; expanded Woodbury; two-dimensional matrices only",
            "gallery": {"shape": list(images.shape), "sha256_float32": array_sha256(images),
                        "binarization_threshold": 128, "use": "official OCF only"},
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
                    f"Experiment 52 {level} n_evaluation_rows mismatch for {task}")
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


def check_covariance_contract(prepared, task):
    require(prepared["rank"] == RANKS[task] and prepared["loading"].shape[1] == RANKS[task],
            f"{task} rank contract failed")
    require(len(prepared["eigenvalues"]) == RANKS[task], "Retained eigenvalue count mismatch")
    require(np.all(np.diff(prepared["eigenvalues"]) <= 0), "Eigenvalues are not descending")
    require(prepared["kappa"] == KAPPAS[task], f"{task} kappa contract failed")
    diagonal_c = np.diag(prepared["covariance"])
    reconstructed = np.sum(prepared["loading"] ** 2, axis=1) + prepared["diagonal"]
    unfloored = diagonal_c - np.sum(prepared["loading"] ** 2, axis=1)
    retained = unfloored >= VARIANCE_FLOOR
    np.testing.assert_allclose(reconstructed[retained], diagonal_c[retained], atol=1e-12, rtol=1e-10)
    require(np.all(prepared["diagonal"] >= VARIANCE_FLOOR), "Covariance diagonal floor failed")
    require(prepared["df"] == prepared["n"] - prepared["g"], "Covariance degrees of freedom failed")


def smoke_test():
    """Synthetic-only numerical/isolation checks; never read real data or experiment 48 metrics."""
    validate_preregistration()

    observations = np.array([[0.2, 1.1, -0.4], [1.5, -0.7, 0.3]], dtype=np.float64)
    templates = np.array([[0.1, -0.2, 0.8], [1.0, 0.4, -0.5]], dtype=np.float64)
    loading = np.array([[0.4, -0.2], [0.1, 0.3], [0.5, 0.2]], dtype=np.float64)
    diagonal = np.array([0.7, 1.2, 0.9], dtype=np.float64)
    covariance = loading @ loading.T + np.diag(diagonal)
    direct = np.empty((len(observations), len(templates)), dtype=np.float64)
    for query_index, query in enumerate(observations):
        for class_index, template in enumerate(templates):
            residual = query - template
            mahalanobis = residual @ np.linalg.solve(covariance, residual)
            direct[query_index, class_index] = -0.5 * mahalanobis / KAPPAS["task3"]

    mm_dimensions = []
    original_mm = torch.mm

    def checked_mm(left, right):
        require(left.ndim == 2 and right.ndim == 2, "A three-dimensional GEMM was attempted")
        mm_dimensions.append((tuple(left.shape), tuple(right.shape)))
        return original_mm(left, right)

    torch.mm = checked_mm
    try:
        actual = lowrank_scores(
            observations, templates, loading, diagonal, KAPPAS["task3"], "cpu", batch_size=1,
        )
        actual_batched = lowrank_scores(
            observations, templates, loading, diagonal, KAPPAS["task3"], "cpu", batch_size=2,
        )
    finally:
        torch.mm = original_mm
    np.testing.assert_allclose(actual, direct, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual_batched, direct, atol=2e-5, rtol=2e-5)
    require(actual.shape == (2, 2) and np.isfinite(actual).all(), "Dense hand-check score shape failed")
    require(mm_dimensions and all(len(left) == len(right) == 2 for left, right in mm_dimensions),
            "Two-dimensional full-candidate GEMM check failed")

    n_classes, repeats = 6, 4
    subset = np.arange(n_classes, dtype=np.int64)
    blocks = [replace(block, features=np.asarray(block.features, dtype=np.uint16))
              for block in EXP48.synthetic_blocks(n_classes=n_classes, repeats=repeats)]
    mice1_blocks = [block for block in blocks if block.mouse == "mice1"]
    raw_example = np.array([[0, 1, 3], [2, 5, 8]], dtype=np.uint16)
    np.testing.assert_array_equal(log1p_counts(raw_example), np.log1p(raw_example.astype(np.float64)))

    task2_split = make_split(blocks, "task2", seed=42, validation_session="session1")
    prepared2 = prepare_task2(
        mice1_blocks, task2_split, "mice1", "session1", subset, n_classes=n_classes,
    )
    check_covariance_contract(prepared2, "task2")
    require(prepared2["n"] == 3 * n_classes * repeats and prepared2["g"] == n_classes,
            "Task2 N/G contract failed")
    require(prepared2["audit"]["n_template_repeats"] == 12
            and prepared2["audit"]["n_query_repeats"] == 4, "Task2 repeat contract failed")
    entry2 = prepared2["validation"][0]
    require(entry2["n_template_repeats"] == 12 and entry2["n_query_repeats"] == 4
            and entry2["repeat_aggregation"] is True, "Task2 four-repeat query policy failed")
    held = next(block for block in mice1_blocks if block.session == "session1")
    held_part = task2_split.blocks[held.key]
    held_z = log1p_counts(held.features[held_part.validation_trials])
    np.testing.assert_array_equal(prepared2["target_offset"], held_z.mean(axis=0, dtype=np.float64))
    grouped_targets, grouped_sums, grouped_counts = target_statistics(
        held_z, held.trial_targets[held_part.validation_trials],
    )
    complete_targets(grouped_targets, grouped_counts, n_classes, 4)
    expected_queries = grouped_sums / 4 - prepared2["target_offset"] + prepared2["reference_offset"]
    np.testing.assert_allclose(entry2["observations"], expected_queries, atol=0, rtol=0)

    shifted_blocks = [replace(block, features=block.features + np.uint16(1000))
                      if block.key == held.key else block for block in mice1_blocks]
    shifted2 = prepare_task2(
        shifted_blocks, task2_split, "mice1", "session1", subset, n_classes=n_classes,
    )
    for key in ("templates", "reference_offset", "covariance", "loading", "diagonal"):
        np.testing.assert_array_equal(prepared2[key], shifted2[key])
    require(not np.array_equal(prepared2["target_offset"], shifted2["target_offset"])
            and not np.array_equal(entry2["observations"], shifted2["validation"][0]["observations"]),
            "Held-out features should alter only target alignment and queries")

    permuted_blocks = [replace(block, targets=np.roll(block.targets, 1))
                       if block.key == held.key else block for block in mice1_blocks]
    permuted2 = prepare_task2(
        permuted_blocks, task2_split, "mice1", "session1", subset, n_classes=n_classes,
    )
    for key in ("templates", "reference_offset", "target_offset", "covariance", "loading", "diagonal"):
        np.testing.assert_array_equal(prepared2[key], permuted2[key])
    require(prepared2["audit"]["target_marginal_uses_labels"] is False,
            "Target-marginal label-isolation audit missing")

    training_shifted = []
    for block in mice1_blocks:
        if block.key == "mice1/session2":
            training_shifted.append(replace(block, features=block.features + np.uint16(7)))
        else:
            training_shifted.append(block)
    changed2 = prepare_task2(
        training_shifted, task2_split, "mice1", "session1", subset, n_classes=n_classes,
    )
    require(not np.array_equal(prepared2["templates"], changed2["templates"])
            and not np.array_equal(prepared2["covariance"], changed2["covariance"]),
            "Changing training features must change templates and covariance")

    task3_split = make_split(blocks, "task3", seed=20261126)
    prepared3 = prepare_task3(
        mice1_blocks, task3_split, "mice1", "20261126", subset, n_classes=n_classes,
    )
    check_covariance_contract(prepared3, "task3")
    require(prepared3["n"] == 4 * 3 * n_classes and prepared3["g"] == 4 * n_classes,
            "Task3 pooled residual N/G contract failed")
    require(len(prepared3["validation"]) == 4, "Task3 requires four independent block evaluations")
    require(any(not np.array_equal(prepared3["validation"][0]["templates"], entry["templates"])
                for entry in prepared3["validation"][1:]), "Task3 templates were incorrectly pooled")
    for entry in prepared3["validation"]:
        require(entry["n_template_repeats"] == 3 and entry["n_query_repeats"] == 1
                and entry["repeat_aggregation"] is False, "Task3 single-trial policy failed")
        block = next(block for block in mice1_blocks if block.key == entry["block"])
        part = task3_split.blocks[block.key]
        validation_z = log1p_counts(block.features[part.validation_trials])
        labels = block.trial_targets[part.validation_trials]
        order = np.argsort(labels, kind="stable")
        np.testing.assert_array_equal(entry["observations"], validation_z[order])

    shifted_validation_blocks = []
    for block in mice1_blocks:
        if block.key == "mice1/session1":
            features = block.features.copy()
            features[task3_split.blocks[block.key].validation_trials] += np.uint16(1000)
            shifted_validation_blocks.append(replace(block, features=features))
        else:
            shifted_validation_blocks.append(block)
    shifted3 = prepare_task3(
        shifted_validation_blocks, task3_split, "mice1", "20261126", subset, n_classes=n_classes,
    )
    for key in ("covariance", "loading", "diagonal"):
        np.testing.assert_array_equal(prepared3[key], shifted3[key])
    for original_entry, shifted_entry in zip(prepared3["validation"], shifted3["validation"]):
        np.testing.assert_array_equal(original_entry["templates"], shifted_entry["templates"])
    require(not np.array_equal(prepared3["validation"][0]["observations"],
                               shifted3["validation"][0]["observations"]),
            "Task3 validation features should alter queries but not fitted statistics")

    rng = np.random.default_rng(772)
    images = (rng.random((n_classes, 64, 64)) > 0.82).astype(np.float32)
    images[np.arange(n_classes), np.arange(n_classes) % 64, :] = 1
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
    require(rows2[0]["n_query_repeats"] == 4 and rows2[0]["repeat_aggregation"] is True
            and all(row["n_query_repeats"] == 1 and row["repeat_aggregation"] is False for row in rows3),
            "Evaluation repeat policies changed")
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
        "SMOKE OK: hand dense inverse; covariance/rank/kappa; train-validation and label isolation; "
        "Task2 four-repeat/Task3 single; full 2-D candidate GEMM; OCF/aggregation; +/- gates",
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
