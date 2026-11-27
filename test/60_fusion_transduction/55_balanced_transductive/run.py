"""Preregistered development-only balanced-transductive experiment (experiment 55)."""

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.features import FeatureBlock, aggregate_by_target, aggregate_samples
from src.neural import SpikeClassifier, predict_logits
from src.preprocessing import CACHE_SCHEMA, file_sha256, load_cached_blocks, validate_split
from src.splits import make_split

OUT = Path(__file__).resolve().parent
PREREGISTRATION = OUT / "preregistration.json"
RESULTS = OUT / "development_results.json"
SELECTION = OUT / "selection.json"
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
DATA = (
    ROOT
    / "data/siclink-mouse-visual-reconstruction-bci-challenge"
    / "siclink_chinese_character_decoding_dataset"
)
REFERENCE48 = ROOT / "test/60_fusion_transduction/48_high_capacity_generalization"
REFERENCE48_RESULTS = REFERENCE48 / "development_results.json"
REFERENCE48_SELECTION = REFERENCE48 / "selection.json"
REFERENCE48_PREREGISTRATION = REFERENCE48 / "preregistration.json"

MICE = ("mice1", "mice2")
BLOCKS = tuple(f"{mouse}/session{session}" for mouse in MICE for session in range(1, 5))
TASKS = ("task2", "task3")
TASK2_FOLDS = ("session1", "session2", "session3")
TASK3_FOLDS = ("20261126", "20261127")
N_CLASSES = 2748
BIN_MS = 50
WINDOW_MS = 250
N_TRIALS = 10992
TRAIN_SEED = 42
SUBSET_SEED = 20261128
SUBSET_SIZE = 256
BASE_EPOCHS = {"task2": 7, "task3": 9}
REFERENCE48_PRODUCTION_EPOCHS = {"task2": 5, "task3": 12}
ARM = "balanced_transductive"
PRODUCTION_ARM = "production_recipe"
METRICS = (
    "raw_top1_accuracy",
    "raw_recall_at_32",
    "raw_top1_official_ocf",
    "assigned_final_ocf",
)
HIDDEN = 512
DROPOUT = {"mice1": 0.2, "mice2": 0.35}
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
GRADIENT_CLIP_NORM = 5.0
TRANSDUCTION_ROUNDS = 2
FINE_TUNE_EPOCHS = 2
FINE_TUNE_LEARNING_RATE = 3e-5
FINE_TUNE_WEIGHT_DECAY = 1e-3
SOFT_TARGET_WEIGHT = 0.5
TEMPERATURE = 0.05
SINKHORN_ITERATIONS = 20
MACRO_MINIMUM_GAIN = 3.0
REQUIRED_NONNEGATIVE_FOLDS = {"task2": 2, "task3": 2}
EXPECTED_FEATURE_DIMENSIONS = {"mice1": 670, "mice2": 390}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_reference48_module():
    path = REFERENCE48 / "run.py"
    spec = importlib.util.spec_from_file_location("experiment48_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP48 = load_reference48_module()


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
        data["loader"] == "src.preprocessing.load_cached_blocks(cache, split='train', bin_ms=50)",
        "Cache loader contract changed",
    )
    require(data["cache_access"] == "read-only; never rebuild the cache", "Cache must be read-only")
    require(data["public_partition"] == "train only", "Only public train data are permitted")
    require(data["view_bin_ms"] == BIN_MS, "Base view changed")
    require(data["mice"] == list(MICE) and data["blocks"] == list(BLOCKS), "Block contract changed")
    require(
        data["n_classes"] == N_CLASSES and data["subset_size"] == SUBSET_SIZE,
        "Class/subset contract changed",
    )
    require(
        data["development_picture_subset"]
        == "np.sort(np.random.default_rng(20261128).permutation(2748)[:256])",
        "Development picture subset changed",
    )
    require(
        data["row_preparation"]
        == "Experiment 48 prepare_task2 and prepare_task3 unchanged on the 50ms block set.",
        "Row preparation contract changed",
    )
    folds = value["folds"]
    require(folds["task2"]["fold_ids"] == list(TASK2_FOLDS), "Task2 folds changed")
    require(folds["task3"]["fold_ids"] == list(TASK3_FOLDS), "Task3 folds changed")
    require("session4" not in folds["task2"]["fold_ids"], "Task2 session4 validation is prohibited")
    require("42" not in folds["task3"]["fold_ids"], "Task3 seed42 evaluation is prohibited")
    arm = value["arm"]
    require(arm["name"] == ARM and arm["arms_in_this_experiment"] == [ARM], "Arm contract changed")
    require(arm["base_view_bin_ms"] == BIN_MS, "Base view changed")
    require(arm["base_epochs"] == BASE_EPOCHS, "Base epochs changed")
    require(arm["transduction_rounds"] == TRANSDUCTION_ROUNDS, "Round count changed")
    require(arm["extra_hyperparameter_search"] is False, "Parameter search is prohibited")
    require(arm["weight_search"] is False, "Weight search is prohibited")
    training = value["base_training"]
    require(training["class"] == "src.neural.SpikeClassifier", "Model class changed")
    for fragment in (
        "hidden=512",
        "dropout mice1=0.2/mice2=0.35",
        "label_smoothing=0.05",
        "weight_decay=0.001",
        "learning_rate=3e-4",
        "batch_size=256",
        "apply_log=False",
        "gradient_clip_norm=5.0",
        "training_seed=42",
        "bf16 autocast",
        "TF32 disabled",
    ):
        require(fragment in training["configuration"], f"Base recipe fragment changed: {fragment}")
    queries = value["transductive_unlabeled_queries"]
    require("group_ptr/trial_sample_indices" in queries["task2"], "Task2 query contract changed")
    require("2748 unlabeled single-trial rows" in queries["task3"], "Task3 query contract changed")
    require("Targets are read only after" in queries["label_isolation"], "Label isolation changed")
    sinkhorn = value["sinkhorn_balancing"]
    require("0.05" in sinkhorn["input"], "Temperature changed")
    require("1.0 per class" in sinkhorn["column_target"], "Column target changed")
    require(sinkhorn["iterations"] == SINKHORN_ITERATIONS, "Sinkhorn iterations changed")
    require(sinkhorn["search"].startswith("none"), "Sinkhorn search is prohibited")
    fine = value["fine_tuning"]
    require(fine["rounds"] == TRANSDUCTION_ROUNDS, "Round count changed")
    require(fine["per_round_epochs"] == FINE_TUNE_EPOCHS, "Fine-tune epochs changed")
    require(fine["learning_rate"] == FINE_TUNE_LEARNING_RATE, "Fine-tune learning rate changed")
    require(fine["weight_decay"] == FINE_TUNE_WEIGHT_DECAY, "Fine-tune weight decay changed")
    require(fine["batch_size"] == BATCH_SIZE, "Fine-tune batch size changed")
    require(fine["soft_target_weight"] == SOFT_TARGET_WEIGHT, "Soft target weight changed")
    require(fine["initialization"] == "continue from the previous round's weights; never reset",
            "Fine-tune initialization changed")
    require(fine["search"] == "none", "Fine-tune search is prohibited")
    evaluation = value["inference_and_evaluation"]
    require(evaluation["metrics"] == list(METRICS), "Metric set changed")
    require("argmax and unique_assignment" in evaluation["final_prediction"],
            "Final prediction contract changed")
    gate = value["advancement_gate"]
    require(gate["macro_minimum_absolute_gain"] == MACRO_MINIMUM_GAIN, "Advancement gate changed")
    require(gate["required_nonnegative_folds"] == REQUIRED_NONNEGATIVE_FOLDS,
            "Fold gate changed")
    require("Both mouse-level" in gate["per_mouse"], "Per-mouse gate changed")
    require(value["outputs"]["model_artifacts"] == "none retained", "Model artifacts must not be retained")
    require(
        value["outputs"]["development_results"] == f"test/{OUT.name}/development_results.json"
        and value["outputs"]["selection"] == f"test/{OUT.name}/selection.json",
        "Output paths changed",
    )
    prohibited = value["prohibited"]
    for item in (
        "Task2 session4 validation or final evaluation",
        "Task3 split seed 42 evaluation",
        "held-out labels in grouping, normalization, Sinkhorn, pseudo-labels or fine-tuning",
        "repeat aggregation in Task3",
        "temperature, iteration, soft-weight, round or learning-rate search",
        "production changes or Kaggle submission",
    ):
        require(item in prohibited, f"Prohibited item missing: {item}")
    return value, hashlib.sha256(raw).hexdigest()


def preflight_cache():
    """Check cache integrity without creating, rebuilding, or modifying it."""
    manifest_path = CACHE / "manifest.json"
    require(manifest_path.is_file(), f"Shared cache is not ready: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == CACHE_SCHEMA, "Unsupported cache schema")
    require(Path(manifest["source_root"]).resolve() == DATA.resolve(), "Cache source root mismatch")
    require(BIN_MS in manifest.get("bin_ms", []), "The cache has no 50ms counts")
    require(manifest.get("window_ms") == WINDOW_MS, "Cache window changed")
    require(manifest.get("image_shape") == [N_CLASSES, 64, 64], "Gallery image shape mismatch")
    pictures = CACHE / "pictures.npy"
    require(pictures.is_file(), "Cached gallery images are missing")
    records = [record for record in manifest["blocks"] if record.get("split") == "train"]
    by_key = {f"{record['mouse']}/{record['session']}": record for record in records}
    require(len(records) == len(BLOCKS) and set(by_key) == set(BLOCKS), "Train cache blocks are incomplete")
    for key in BLOCKS:
        record = by_key[key]
        require(record["task"] == "train", f"Unexpected cached task for {key}")
        require(record["n_samples"] == N_TRIALS and record["n_trials"] == N_TRIALS,
                f"Trial count changed for {key}")
        count = record["counts"][str(BIN_MS)]
        require(count["dtype"] == "uint16" and count["shape"][0] == N_TRIALS,
                f"Count contract changed for {key}")
        require(count["shape"][1] == EXPECTED_FEATURE_DIMENSIONS[record["mouse"]],
                f"50ms feature dimension changed for {key}")
        directory = CACHE / "blocks" / record["id"]
        count_path = directory / count["file"]
        require(count_path.is_file() and (directory / "index.npz").is_file(),
                f"Cache files missing for {key}")
        require(file_sha256(count_path) == count["sha256"], f"Cache hash mismatch for {key}")
    return manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def validate_reference48(subset):
    """Validate the frozen experiment-48 comparator before any cache/CUDA access."""
    hashes = {
        "development_results": file_sha256(REFERENCE48_RESULTS),
        "selection": file_sha256(REFERENCE48_SELECTION),
        "preregistration": file_sha256(REFERENCE48_PREREGISTRATION),
    }
    results = json.loads(REFERENCE48_RESULTS.read_text(encoding="utf-8"))
    selection = json.loads(REFERENCE48_SELECTION.read_text(encoding="utf-8"))
    require(selection["development_results_sha256"] == hashes["development_results"],
            "Experiment 48 development-results hash link is broken")
    require(selection["preregistration_sha256"] == hashes["preregistration"],
            "Experiment 48 preregistration hash link is broken")
    require(results["preregistration_sha256"] == selection["preregistration_sha256"],
            "Experiment 48 result/preregistration linkage failed")
    require(selection["experiment"] == REFERENCE48.name == results["experiment"],
            "Experiment 48 identity mismatch")
    require(results["development_picture_subset"] == subset.tolist(),
            "Experiment 48 development subset mismatch")
    require(results["run"]["bin_ms"] == 25, "Experiment 48 view changed")
    require(
        len(results["block"]) == 440
        and len(results["mouse_fold"]) == 200
        and len(results["mouse"]) == 80
        and len(results["fold"]) == 100
        and len(results["macro"]) == 40,
        "Experiment 48 aggregation grid mismatch",
    )
    production = [row for row in results["block"] if row["arm"] == PRODUCTION_ARM]
    require(len(production) == 220, "Experiment 48 production rows are incomplete")
    recomputed_macro = EXP48.aggregate_rows(production, "macro")
    for task in TASKS:
        epoch = REFERENCE48_PRODUCTION_EPOCHS[task]
        require(selection["tasks"][task]["selected_epochs"][PRODUCTION_ARM] == epoch,
                f"Experiment 48 production epoch changed for {task}")
        recomputed = EXP48.select_epoch(recomputed_macro, task, PRODUCTION_ARM, expected_count=10)
        recorded = selection["tasks"][task]["selected_macro"][PRODUCTION_ARM]
        require(recomputed["epoch"] == epoch, f"Experiment 48 epoch linkage failed for {task}")
        require(
            abs(recomputed["assigned_final_ocf"] - recorded["assigned_final_ocf"]) < 1e-9
            and abs(recomputed["raw_top1_accuracy"] - recorded["raw_top1_accuracy"]) < 1e-12,
            f"Experiment 48 macro linkage failed for {task}",
        )
    return {
        "results": results,
        "selection": selection,
        "production_blocks": production,
        "hashes": hashes,
    }


def fit_session_normalizer(features):
    transformed = np.log1p(np.maximum(features, 0))
    mean = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = transformed.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1e-5)


def apply_session_normalizer(features, normalizer):
    mean, scale = normalizer
    return ((np.log1p(np.maximum(features, 0)) - mean) / scale).astype(np.float32)


def train_base_model(prepared, device, *, epochs, seed=TRAIN_SEED):
    """Production-recipe MLP trained to a fixed base epoch; parameters stay in memory."""
    train_x = prepared["train_x"]
    train_y = prepared["train_y"]
    require(np.isfinite(train_x).all() and len(train_x) == len(train_y), "Invalid training matrix")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpikeClassifier(
        n_features=train_x.shape[1], n_classes=prepared["n_classes"], hidden=HIDDEN,
        dropout=DROPOUT[prepared["mouse"]], apply_log=False,
    )
    model.set_normalization(train_x)
    np.testing.assert_array_equal(
        model.feature_mean.cpu().numpy(),
        train_x.mean(axis=0, dtype=np.float64).astype(np.float32),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(train_y))
        model.train()
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            batch_x = torch.from_numpy(np.ascontiguousarray(train_x[indices], dtype=np.float32)).to(device)
            batch_y = torch.from_numpy(train_y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(batch_x)
                loss = F.cross_entropy(logits.float(), batch_y, label_smoothing=LABEL_SMOOTHING)
            require(bool(torch.isfinite(loss).item()), "Nonfinite base training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(
            f"BASE task={prepared['task']} fold={prepared['fold']} mouse={prepared['mouse']} "
            f"epoch={epoch}/{epochs} train_loss={total_loss / len(train_y):.6f}",
            flush=True,
        )
    return model


def normalized_row_softmax(logits, temperature=TEMPERATURE):
    """Row z-score, fixed temperature, row softmax; returns a 2-D class matrix."""
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.ndim == 2 and np.isfinite(logits).all(), "Invalid unlabeled logits")
    centered = logits - logits.mean(axis=1, keepdims=True)
    scale = np.maximum(logits.std(axis=1, keepdims=True), 1e-12)
    scaled = centered / scale / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def sinkhorn_balance(probabilities, column_target, iterations=SINKHORN_ITERATIONS):
    """Row sums 1 and every class column summing to the balanced column target."""
    matrix = np.array(probabilities, dtype=np.float64, copy=True)
    require(matrix.ndim == 2 and np.isfinite(matrix).all(), "Invalid Sinkhorn input")
    require(column_target > 0, "Invalid column target")
    for _ in range(iterations):
        matrix /= np.maximum(matrix.sum(axis=0, keepdims=True), 1e-300) / column_target
        matrix /= matrix.sum(axis=1, keepdims=True)
    row_error = float(np.abs(matrix.sum(axis=1) - 1.0).max())
    column_error = float(np.abs(matrix.sum(axis=0) - column_target).max())
    return matrix, row_error, column_error


def soft_cross_entropy(logits, targets):
    log_probabilities = F.log_softmax(logits.float(), dim=1)
    return -(targets * log_probabilities).sum(dim=1).mean()


def fine_tune_epochs(model, train_x, train_y, validation_x, validation_targets, device, *,
                     epochs=FINE_TUNE_EPOCHS, learning_rate=FINE_TUNE_LEARNING_RATE,
                     weight_decay=FINE_TUNE_WEIGHT_DECAY, batch_size=BATCH_SIZE,
                     soft_weight=SOFT_TARGET_WEIGHT, seed=TRAIN_SEED):
    """Continue training with hard-CE(train) + soft_weight * soft-CE(validation, P)."""
    require(validation_targets.shape == (len(validation_x), model.n_classes),
            "Sinkhorn matrix must span the full candidate axis")
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    validation_tensor = torch.from_numpy(
        np.ascontiguousarray(validation_x, dtype=np.float32)
    ).to(device)
    target_tensor = torch.from_numpy(
        np.ascontiguousarray(validation_targets, dtype=np.float32)
    ).to(device)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(train_y))
        validation_order = np.random.default_rng(seed + 7919 + epoch).permutation(len(validation_x))
        hard_total = 0.0
        soft_total = 0.0
        for step, start in enumerate(range(0, len(order), batch_size)):
            indices = order[start:start + batch_size]
            positions = np.take(
                validation_order, (np.arange(len(indices)) + step * batch_size) % len(validation_x)
            )
            batch_x = torch.from_numpy(np.ascontiguousarray(train_x[indices], dtype=np.float32)).to(device)
            batch_y = torch.from_numpy(train_y[indices]).to(device)
            batch_positions = torch.from_numpy(positions).to(device)
            batch_validation = validation_tensor[batch_positions]
            batch_targets = target_tensor[batch_positions]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(batch_x)
                model.eval()
                validation_logits = model(batch_validation)
                model.train()
                hard = F.cross_entropy(logits.float(), batch_y)
                soft = soft_cross_entropy(validation_logits, batch_targets)
            require(bool(torch.isfinite(hard).item()) and bool(torch.isfinite(soft).item()),
                    "Nonfinite fine-tuning loss")
            (hard + soft_weight * soft).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            hard_total += hard.item() * len(indices)
            soft_total += soft.item() * len(indices)
        history.append({
            "epoch": epoch,
            "hard_cross_entropy": hard_total / len(train_y),
            "soft_cross_entropy": soft_total / len(train_y),
        })
        print(
            f"FINE epoch={epoch} hard_ce={history[-1]['hard_cross_entropy']:.6f} "
            f"soft_ce={history[-1]['soft_cross_entropy']:.6f}",
            flush=True,
        )
    return history


def unlabeled_validation_sets(task, prepared, split, blocks_by_key, fold):
    """Unlabeled evaluation-block rows: no labels are read here."""
    sets = []
    if task == "task2":
        key = f"{prepared['mouse']}/{fold}"
        block = blocks_by_key[key]
        part = split.blocks[key]
        require(len(part.validation_trials) == block.n_trials,
                "Task2 held session is not entirely unlabeled")
        raw = np.asarray(block.features, dtype=np.float32)
        # group_ptr defines the only label-free grouping; the cached blocks store one
        # trial per sample, so this preserves the 10992 unlabeled trial rows.
        rows = aggregate_samples(raw, block.group_ptr, average=True)
        require(len(rows) == block.n_samples, "Task2 unlabeled row count changed")
        sets.append({
            "block_key": key,
            "block_label": fold,
            "rows": apply_session_normalizer(rows, prepared["normalizers"][fold]),
        })
    else:
        for key in sorted(prepared["validation"]):
            block = blocks_by_key[key]
            part = split.blocks[key]
            require(len(part.validation_trials) == prepared["n_classes"],
                    f"Task3 validation rows changed for {key}")
            raw = np.asarray(block.features[part.validation_trials], dtype=np.float32)
            sets.append({
                "block_key": key,
                "block_label": key,
                "rows": apply_session_normalizer(raw, prepared["normalizers"][block.session]),
            })
        require(len(sets) == 4, "Task3 requires four unlabeled validation blocks")
    for entry in sets:
        require(entry["rows"].ndim == 2 and np.isfinite(entry["rows"]).all(),
                "Unlabeled rows must be a finite 2-D matrix")
    return sets


def run_transduction(model, prepared, validation_sets, device, *, rounds=TRANSDUCTION_ROUNDS,
                     temperature=TEMPERATURE, iterations=SINKHORN_ITERATIONS,
                     soft_weight=SOFT_TARGET_WEIGHT, seed=TRAIN_SEED):
    """Two rounds of Sinkhorn-balanced soft pseudo-label fine-tuning on unlabeled rows."""
    audits = []
    previous = None
    for round_index in range(1, rounds + 1):
        probabilities = []
        records = []
        for entry in validation_sets:
            logits = predict_logits(model, entry["rows"], batch_size=BATCH_SIZE, device=device)
            require(logits.ndim == 2 and logits.shape == (len(entry["rows"]), prepared["n_classes"]),
                    "Unlabeled scores must span the full 2-D candidate axis")
            soft = normalized_row_softmax(logits, temperature)
            column_target = len(entry["rows"]) / prepared["n_classes"]
            matrix, row_error, column_error = sinkhorn_balance(soft, column_target, iterations)
            hard = matrix.argmax(axis=1)
            agreement = None
            if previous is not None:
                agreement = float(np.mean(hard == previous[entry["block_key"]]))
            probabilities.append(matrix)
            records.append({
                "round": round_index,
                "block": entry["block_label"],
                "rows": int(len(entry["rows"])),
                "column_target": float(column_target),
                "row_marginal_max_error": row_error,
                "column_marginal_max_error": column_error,
                "mean_row_max_probability": float(matrix.max(axis=1).mean()),
                "pseudo_label_top1_agreement_with_previous_round": agreement,
            })
        validation_x = np.concatenate([entry["rows"] for entry in validation_sets])
        targets = np.concatenate(probabilities).astype(np.float32, copy=False)
        require(targets.shape == (len(validation_x), prepared["n_classes"]),
                "Concatenated pseudo-label matrix changed shape")
        history = fine_tune_epochs(
            model, prepared["train_x"], prepared["train_y"], validation_x, targets, device,
            soft_weight=soft_weight, seed=seed + 1009 * round_index,
        )
        for record in records:
            record["fine_tune_history"] = history
        audits.extend(records)
        previous = {
            entry["block_key"]: probabilities[index].argmax(axis=1)
            for index, entry in enumerate(validation_sets)
        }
        del probabilities, targets
    return model, audits


def evaluated_full_rows(task, prepared, entry, block, part, fold):
    """Full unlabeled-score rows for final evaluation; labels are read only here."""
    if task == "task2":
        require(len(part.validation_trials) == block.n_trials,
                "Task2 held session must stay fully held out")
        raw = np.asarray(block.features, dtype=np.float32)
        labels = np.asarray(block.trial_targets, dtype=np.int64)
        targets, means = aggregate_by_target(raw, labels, method="mean")
        normalizer = prepared["normalizers"][fold]
        return apply_session_normalizer(means, normalizer), targets
    labels = np.asarray(block.trial_targets[part.validation_trials], dtype=np.int64)
    return entry["rows"], labels


def subset_scores(full_logits, row_targets, subset):
    """Score the full class axis, then keep the frozen development subset rows."""
    require(full_logits.ndim == 2 and full_logits.shape[1] == N_CLASSES,
            "Scores must span all candidates on a 2-D axis")
    order = np.argsort(row_targets, kind="stable")
    sorted_targets = row_targets[order]
    positions = EXP48.subset_positions(sorted_targets, subset)
    return full_logits[order][positions], sorted_targets[positions], order, positions


def evaluate_gate(task, macro_delta, per_mouse, per_fold):
    fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
    require(set(per_mouse) == set(MICE), f"Per-mouse deltas incomplete for {task}")
    require(set(per_fold) == set(fold_ids), f"Per-fold deltas incomplete for {task}")
    required = REQUIRED_NONNEGATIVE_FOLDS[task]
    nonnegative = sum(1 for value in per_fold.values() if value >= 0.0)
    gates = {
        "macro_assigned_ocf_delta_at_least_3": bool(macro_delta >= MACRO_MINIMUM_GAIN),
        "both_mouse_deltas_nonnegative": bool(all(value >= 0.0 for value in per_mouse.values())),
        "required_nonnegative_folds": bool(nonnegative >= required),
    }
    return {
        "gates": gates,
        "nonnegative_fold_count": nonnegative,
        "required_nonnegative_fold_count": required,
        "advance": all(gates.values()),
    }


def summarize_transduction(results, task):
    rows = [row for row in results["transduction_audit"] if row["task"] == task]
    require(bool(rows), f"Missing transduction audit for {task}")
    summary = {}
    for round_index in range(1, TRANSDUCTION_ROUNDS + 1):
        entries = [entry for row in rows for entry in row["rounds"] if entry["round"] == round_index]
        require(bool(entries), f"Missing round {round_index} audit for {task}")
        agreements = [
            entry["pseudo_label_top1_agreement_with_previous_round"]
            for entry in entries
            if entry["pseudo_label_top1_agreement_with_previous_round"] is not None
        ]
        summary[f"round_{round_index}"] = {
            "n_block_fits": len(entries),
            "max_row_marginal_error": float(max(entry["row_marginal_max_error"] for entry in entries)),
            "max_column_marginal_error": float(
                max(entry["column_marginal_max_error"] for entry in entries)
            ),
            "mean_pseudo_label_top1_agreement_with_previous_round": (
                float(np.mean(agreements)) if agreements else None
            ),
        }
    return summary


def build_selection(results, reference, hashes, implementation_sha256):
    macro_rows = results["macro"]
    mouse_rows = results["mouse"]
    fold_rows = results["fold"]
    reference_macro = EXP48.aggregate_rows(reference["production_blocks"], "macro")
    reference_mouse = EXP48.aggregate_rows(reference["production_blocks"], "mouse")
    reference_fold = EXP48.aggregate_rows(reference["production_blocks"], "fold")
    tasks = {}
    for task in TASKS:
        base_epoch = BASE_EPOCHS[task]
        final_epoch = base_epoch + TRANSDUCTION_ROUNDS * FINE_TUNE_EPOCHS
        production_epoch = REFERENCE48_PRODUCTION_EPOCHS[task]
        base_macro = EXP48.find_aggregate(macro_rows, task, ARM, base_epoch)
        final_macro = EXP48.find_aggregate(macro_rows, task, ARM, final_epoch)
        production_macro = EXP48.find_aggregate(
            reference_macro, task, PRODUCTION_ARM, production_epoch
        )
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        per_mouse = {}
        per_mouse_base = {}
        for mouse in MICE:
            final_row = EXP48.find_aggregate(mouse_rows, task, ARM, final_epoch, mouse=mouse)
            base_row = EXP48.find_aggregate(mouse_rows, task, ARM, base_epoch, mouse=mouse)
            reference_row = EXP48.find_aggregate(
                reference_mouse, task, PRODUCTION_ARM, production_epoch, mouse=mouse
            )
            per_mouse[mouse] = float(final_row["assigned_final_ocf"] - reference_row["assigned_final_ocf"])
            per_mouse_base[mouse] = float(
                final_row["assigned_final_ocf"] - base_row["assigned_final_ocf"]
            )
        per_fold = {}
        per_fold_base = {}
        for fold in fold_ids:
            final_row = EXP48.find_aggregate(fold_rows, task, ARM, final_epoch, fold=fold)
            base_row = EXP48.find_aggregate(fold_rows, task, ARM, base_epoch, fold=fold)
            reference_row = EXP48.find_aggregate(
                reference_fold, task, PRODUCTION_ARM, production_epoch, fold=fold
            )
            per_fold[fold] = float(final_row["assigned_final_ocf"] - reference_row["assigned_final_ocf"])
            per_fold_base[fold] = float(final_row["assigned_final_ocf"] - base_row["assigned_final_ocf"])
        macro_delta = float(final_macro["assigned_final_ocf"] - production_macro["assigned_final_ocf"])
        gate = evaluate_gate(task, macro_delta, per_mouse, per_fold)
        tasks[task] = {
            "base_epoch": base_epoch,
            "transductive_epoch": final_epoch,
            "base_macro": base_macro,
            "transductive_macro": final_macro,
            "reference48_production_epoch": production_epoch,
            "reference48_production_macro": production_macro,
            "versus_reference48_production": {
                "macro_assigned_final_ocf": macro_delta,
                "per_mouse_assigned_final_ocf": per_mouse,
                "per_fold_assigned_final_ocf": per_fold,
            },
            "versus_same_fold_50ms_base": {
                "macro_assigned_final_ocf": float(
                    final_macro["assigned_final_ocf"] - base_macro["assigned_final_ocf"]
                ),
                "macro_raw_recall_at_32": float(
                    final_macro["raw_recall_at_32"] - base_macro["raw_recall_at_32"]
                ),
                "per_mouse_assigned_final_ocf": per_mouse_base,
                "per_fold_assigned_final_ocf": per_fold_base,
            },
            "nonnegative_fold_count": gate["nonnegative_fold_count"],
            "required_nonnegative_fold_count": gate["required_nonnegative_fold_count"],
            "gates": gate["gates"],
            "advance": gate["advance"],
            "transduction_rounds": summarize_transduction(results, task),
            "next_step": (
                "development gate passed; separate pristine/final authorization still required"
                if gate["advance"] else
                "stop; do not search temperature, iteration count, soft weight, round count "
                "or learning rate"
            ),
        }
    overall = all(tasks[task]["advance"] for task in TASKS)
    return {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_selection_complete",
        "arm": ARM,
        "hashes": {
            "preregistration_sha256": hashes["preregistration_sha256"],
            "cache_manifest_sha256": hashes["cache_manifest_sha256"],
            "reference48_development_results_sha256": hashes["reference48_development_results"],
            "reference48_selection_sha256": hashes["reference48_selection"],
            "reference48_preregistration_sha256": hashes["reference48_preregistration"],
            "development_results_sha256": file_sha256(RESULTS),
            "implementation_sha256": implementation_sha256,
        },
        "development_picture_subset": results["development_picture_subset"],
        "tasks": tasks,
        "advance": overall,
        "next_step": (
            "development gate passed for both tasks; separate pristine/final authorization "
            "still required"
            if overall else
            "stop; at least one task gate failed, so no temperature, iteration count, soft "
            "weight, round count or learning-rate search is permitted"
        ),
        "historical_or_final_metrics_used": False,
        "model_artifacts_retained": False,
        "protocol_deviations": results["protocol_deviations"],
    }


def run_experiment():
    preregistration, preregistration_hash = validate_preregistration()
    require(not RESULTS.exists() and not SELECTION.exists(),
            "Development outputs already exist; refusing a repeated selection run")
    require(torch.cuda.is_available(), "The preregistered bf16 CUDA experiment requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    manifest, manifest_hash = preflight_cache()
    subset = np.sort(np.random.default_rng(SUBSET_SEED).permutation(N_CLASSES)[:SUBSET_SIZE])
    subset = subset.astype(np.int64)
    require(len(subset) == SUBSET_SIZE and len(np.unique(subset)) == SUBSET_SIZE,
            "Invalid development picture subset")
    reference = validate_reference48(subset)
    print("Loading read-only public train blocks and gallery images", flush=True)
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=BIN_MS)
    require(tuple(block.key for block in blocks) == tuple(sorted(BLOCKS)), "Loaded block set mismatch")
    for block in blocks:
        require(block.n_features == EXPECTED_FEATURE_DIMENSIONS[block.mouse],
                f"Unexpected 50ms feature width for {block.key}")
    blocks_by_key = {block.key: block for block in blocks}
    images = (np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False) >= 128).astype(np.float32)
    require(images.shape == (N_CLASSES, 64, 64), "Gallery image cache mismatch")
    ocf_cache = EXP48.OcfPairCache(images, device="cuda", batch_size=32)
    block_rows = []
    transduction_audit = []
    fold_audits = []
    started = time.time()
    fits_completed = 0
    for task in TASKS:
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        base_epoch = BASE_EPOCHS[task]
        final_epoch = base_epoch + TRANSDUCTION_ROUNDS * FINE_TUNE_EPOCHS
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
                prepare = EXP48.prepare_task2 if task == "task2" else EXP48.prepare_task3
                prepared = prepare(mouse_blocks, split, mouse, fold, subset)
                fold_audits.append(prepared["audit"])
                print(
                    f"START fit={fits_completed + 1}/10 task={task} fold={fold} mouse={mouse} "
                    f"train_rows={len(prepared['train_y'])}", flush=True,
                )
                model = train_base_model(prepared, device, epochs=base_epoch)
                base_logits = {}
                base_metrics = {}
                for key, (validation_x, validation_y) in prepared["validation"].items():
                    logits = predict_logits(model, validation_x, batch_size=BATCH_SIZE, device=device)
                    require(logits.ndim == 2 and logits.shape == (len(validation_y), N_CLASSES),
                            "Base scores must span the full candidate axis")
                    base_logits[key] = logits
                    metrics = EXP48.evaluate_logits(logits, validation_y, ocf_cache)
                    base_metrics[key] = metrics
                    block_rows.append({
                        "task": task, "arm": ARM, "mouse": mouse, "fold": fold,
                        "block": fold if task == "task2" else key, "epoch": base_epoch,
                        "n_rows": len(validation_y), **metrics,
                    })
                validation_sets = unlabeled_validation_sets(
                    task, prepared, split, blocks_by_key, fold
                )
                if task == "task2":
                    require(len(validation_sets[0]["rows"]) == N_TRIALS,
                            "Task2 unlabeled rows must stay at one row per unlabeled trial")
                model, round_audit = run_transduction(model, prepared, validation_sets, device)
                transduction_audit.append({
                    "task": task, "mouse": mouse, "fold": fold,
                    "base_epoch": base_epoch, "base_metrics": base_metrics,
                    "rounds": round_audit,
                })
                for entry in validation_sets:
                    block = blocks_by_key[entry["block_key"]]
                    part = split.blocks[entry["block_key"]]
                    full_rows, row_targets = evaluated_full_rows(
                        task, prepared, entry, block, part, fold
                    )
                    full_logits = predict_logits(model, full_rows, batch_size=BATCH_SIZE, device=device)
                    require(full_logits.ndim == 2 and full_logits.shape == (len(full_rows), N_CLASSES),
                            "Final scores must span the full 2-D candidate axis")
                    scores, targets, order, positions = subset_scores(
                        full_logits, row_targets, subset
                    )
                    require(np.array_equal(targets, subset), "Final subset rows are not subset-sorted")
                    stored_key = fold if task == "task2" else entry["block_key"]
                    np.testing.assert_allclose(
                        full_rows[order][positions], prepared["validation"][stored_key][0],
                        atol=1e-5, rtol=0,
                    )
                    metrics = EXP48.evaluate_logits(scores, targets, ocf_cache)
                    base_scores = base_logits.pop(stored_key)
                    base_metrics[stored_key]["final_top1_agreement_with_base"] = float(
                        np.mean(scores.argmax(axis=1) == base_scores.argmax(axis=1))
                    )
                    block_rows.append({
                        "task": task, "arm": ARM, "mouse": mouse, "fold": fold,
                        "block": fold if task == "task2" else entry["block_key"],
                        "epoch": final_epoch, "n_rows": len(targets), **metrics,
                    })
                require(not base_logits, "Base logits were not consumed exactly once")
                fits_completed += 1
                print(
                    f"DONE fit={fits_completed}/10 task={task} fold={fold} mouse={mouse} "
                    f"elapsed_seconds={time.time() - started:.1f}", flush=True,
                )
                del model, prepared, validation_sets
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    require(fits_completed == 10, "Not all preregistered fits completed")
    require(len(block_rows) == 44, "Incomplete block/checkpoint result grid")
    results = {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_results_complete",
        "data_policy": (
            "public train only; development reuse; no Task2 session4 or Task3 seed42 final "
            "evaluation; held-out labels used only for final metrics"
        ),
        "preregistration_sha256": preregistration_hash,
        "cache_manifest_sha256": manifest_hash,
        "reference48": {
            "development_results_sha256": reference["hashes"]["development_results"],
            "selection_sha256": reference["hashes"]["selection"],
            "preregistration_sha256": reference["hashes"]["preregistration"],
            "production_epochs": REFERENCE48_PRODUCTION_EPOCHS,
            "view_bin_ms": 25,
        },
        "run": {
            "device": torch.cuda.get_device_name(device),
            "bin_ms": BIN_MS,
            "window_ms": WINDOW_MS,
            "feature_dimensions": {
                mouse: EXPECTED_FEATURE_DIMENSIONS[mouse] for mouse in MICE
            },
            "training_seed": TRAIN_SEED,
            "base_epochs": BASE_EPOCHS,
            "transductive_epochs": {
                task: BASE_EPOCHS[task] + TRANSDUCTION_ROUNDS * FINE_TUNE_EPOCHS for task in TASKS
            },
            "transduction": {
                "rounds": TRANSDUCTION_ROUNDS,
                "epochs_per_round": FINE_TUNE_EPOCHS,
                "learning_rate": FINE_TUNE_LEARNING_RATE,
                "weight_decay": FINE_TUNE_WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "soft_target_weight": SOFT_TARGET_WEIGHT,
                "temperature": TEMPERATURE,
                "sinkhorn_iterations": SINKHORN_ITERATIONS,
            },
            "fits_completed": fits_completed,
            "model_artifacts_retained": False,
            "ocf_pair_cache": {
                "storage": "memory only",
                "scoring_calls": ocf_cache.calls,
                "unique_pairs_scored": ocf_cache.scored_pairs,
            },
            "elapsed_seconds": float(time.time() - started),
        },
        "development_picture_subset": subset.tolist(),
        "protocol_deviations": [
            {
                "item": "task2_unlabeled_row_grouping",
                "preregistered": (
                    "group the held session's 10992 trials into 2748 rows by sample using only "
                    "group_ptr/trial_sample_indices and average raw counts per sample"
                ),
                "observed": (
                    "every cached train block stores one trial per sample (group_ptr diffs are all "
                    "1, n_samples = n_trials = 10992), so the prescribed grouping cannot produce "
                    "2748 rows and cannot give the frozen one-row-per-class column target of 1.0 "
                    "without using held-out labels"
                ),
                "implemented": (
                    "kept the label-free 10992 unlabeled trial rows and used the structurally "
                    "implied balanced column target rows/n_classes = 4.0; final evaluation still "
                    "uses experiment 48's official four-repeat mean rows"
                ),
                "labels_used_in_grouping": False,
                "parameters_searched": False,
            },
            {
                "item": "feature_width",
                "preregistered": "50ms bins over the 250ms window (preregistration loader bin_ms=50)",
                "observed": (
                    "50ms counting gives 134 x 5 = 670 (mice1) and 78 x 5 = 390 (mice2); the "
                    "1340/780 widths belong to the 25ms view"
                ),
                "implemented": "kept the frozen bin_ms=50 loader; no view change",
                "labels_used_in_grouping": False,
                "parameters_searched": False,
            },
        ],
        "fold_audits": fold_audits,
        "block": block_rows,
        "mouse_fold": EXP48.aggregate_rows(block_rows, "mouse_fold"),
        "mouse": EXP48.aggregate_rows(block_rows, "mouse"),
        "fold": EXP48.aggregate_rows(block_rows, "fold"),
        "macro": EXP48.aggregate_rows(block_rows, "macro"),
        "transduction_audit": transduction_audit,
    }
    require(len(results["mouse_fold"]) == 20 and len(results["mouse"]) == 8,
            "Mouse aggregation grid mismatch")
    require(len(results["fold"]) == 10 and len(results["macro"]) == 4,
            "Fold/macro aggregation grid mismatch")
    write_json(RESULTS, results)
    implementation_sha256 = file_sha256(Path(__file__).resolve())
    hashes = {
        "preregistration_sha256": preregistration_hash,
        "cache_manifest_sha256": manifest_hash,
        "reference48_development_results": reference["hashes"]["development_results"],
        "reference48_selection": reference["hashes"]["selection"],
        "reference48_preregistration": reference["hashes"]["preregistration"],
    }
    selection = build_selection(results, reference, hashes, implementation_sha256)
    write_json(SELECTION, selection)
    require(hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest() == preregistration_hash,
            "Preregistration changed during the run")
    print(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print(f"Wrote {RESULTS} and {SELECTION}; no model artifacts retained", flush=True)


def synthetic_blocks(n_classes=6, repeats=4, n_features=12, seed=9182):
    rng = np.random.default_rng(seed)
    blocks = []
    for mouse_index, mouse in enumerate(MICE):
        for session_index in range(1, 5):
            targets = np.arange(n_classes, dtype=np.int64)
            features = rng.poisson(0.4 + mouse_index, size=(n_classes * repeats, n_features)).astype(np.float32)
            features += (np.repeat(targets, repeats)[:, None]
                         == np.arange(n_features)[None, :] % n_classes) * 3
            blocks.append(FeatureBlock(
                mouse=mouse, session=f"session{session_index}", split="train", task="train",
                sample_ids=np.arange(len(targets), dtype=np.int64),
                group_ptr=np.arange(0, n_classes * repeats + 1, repeats, dtype=np.int64),
                targets=targets, features=features, n_units=n_features // 5, bin_ms=BIN_MS,
            ))
    return sorted(blocks, key=lambda block: (block.mouse, block.session))


def permute_heldout_labels(blocks, split, seed=5):
    """Permute labels only inside fully held-out blocks; the split itself is unchanged."""
    rng = np.random.default_rng(seed)
    output = []
    for block in blocks:
        part = split.blocks[block.key]
        if len(part.validation_trials) == block.n_trials:
            targets = np.asarray(block.targets, dtype=np.int64).copy()
            permuted = targets[rng.permutation(len(targets))]
            require(not np.array_equal(permuted, targets), "Synthetic label permutation is vacuous")
            output.append(replace(block, targets=permuted))
        else:
            output.append(block)
    return sorted(output, key=lambda block: (block.mouse, block.session))


def smoke_test():
    """Synthetic-only checks; never load real blocks or compute real development metrics."""
    validate_preregistration()
    n_classes = 6
    repeats = 4
    subset = np.arange(n_classes, dtype=np.int64)
    blocks = synthetic_blocks(n_classes, repeats)
    block = blocks[0]
    raw = np.asarray(block.features[: n_classes], dtype=np.float32)
    normalizer = fit_session_normalizer(raw)
    shifted = raw + 1e6
    np.testing.assert_array_equal(normalizer[0], fit_session_normalizer(raw)[0])
    require(not np.array_equal(normalizer[0], fit_session_normalizer(shifted)[0]),
            "Synthetic normalizer isolation check is vacuous")
    np.testing.assert_allclose(
        apply_session_normalizer(raw, normalizer).mean(axis=0), 0, atol=2e-6
    )

    task2_split = make_split(blocks, "task2", seed=42, validation_session="session1")
    prepared2 = EXP48.prepare_task2(
        [item for item in blocks if item.mouse == "mice1"], task2_split, "mice1", "session1",
        subset, n_classes=n_classes,
    )
    require(prepared2["audit"]["single_trial_rows"] == 3 * n_classes * repeats,
            "Task2 single-row policy failed")
    require(len(prepared2["train_y"]) == 3 * n_classes * (repeats + 1), "Task2 total rows failed")
    task3_split = make_split(blocks, "task3", seed=20261126)
    prepared3 = EXP48.prepare_task3(
        [item for item in blocks if item.mouse == "mice1"], task3_split, "mice1", "20261126",
        subset, n_classes=n_classes,
    )
    require(len(prepared3["train_y"]) == 4 * 3 * n_classes, "Task3 single-trial train policy failed")
    require(prepared3["audit"]["four_repeat_mean_rows"] == 0, "Task3 repeat aggregation is prohibited")

    # Label-free unlabeled rows: full candidate axis, exactly 2-D.
    blocks_by_key = {item.key: item for item in blocks}
    sets2 = unlabeled_validation_sets("task2", prepared2, task2_split, blocks_by_key, "session1")
    require(len(sets2) == 1 and len(sets2[0]["rows"]) == n_classes,
            "Task2 unlabeled rows must stay at one row per unlabeled sample")
    sets3 = unlabeled_validation_sets("task3", prepared3, task3_split, blocks_by_key, "20261126")
    require(len(sets3) == 4, "Task3 unlabeled block count failed")
    for entry in sets2 + sets3:
        rows = entry["rows"]
        require(rows.ndim == 2 and rows.shape[1] == prepared2["train_x"].shape[1],
                "Unlabeled rows must be a 2-D feature matrix")
    for prepared, sets in ((prepared2, sets2), (prepared3, sets3)):
        for entry in sets:
            require(len(entry["rows"]) / n_classes >= 1.0, "Invalid balanced row prior")

    # (i) Sinkhorn row/column marginals.
    rng = np.random.default_rng(4242)
    logits = rng.normal(size=(n_classes * repeats, n_classes)).astype(np.float32) * 4.0
    soft = normalized_row_softmax(logits)
    np.testing.assert_allclose(soft.sum(axis=1), 1.0, atol=1e-12)
    require(soft.ndim == 2 and soft.shape[1] == n_classes, "Soft targets must span all candidates")
    column_target = len(logits) / n_classes
    errors = []
    for iterations in (1, 5, SINKHORN_ITERATIONS):
        matrix, row_error, column_error = sinkhorn_balance(soft, column_target, iterations)
        require(matrix.shape == logits.shape and np.isfinite(matrix).all() and (matrix >= 0).all(),
                "Sinkhorn matrix is invalid")
        require(row_error < 1e-9, "Sinkhorn rows must sum to one")
        errors.append(column_error)
    require(errors[-1] <= errors[0],
            f"Sinkhorn column marginals did not improve with iterations: {errors}")
    balanced_logits = np.zeros((n_classes * repeats, n_classes), dtype=np.float32)
    balanced_logits[np.arange(len(balanced_logits)), np.arange(len(balanced_logits)) // repeats] = 8.0
    matrix, row_error, column_error = sinkhorn_balance(
        normalized_row_softmax(balanced_logits), column_target, SINKHORN_ITERATIONS
    )
    require(row_error < 1e-9 and column_error < 1e-9,
            f"Balanced-prior Sinkhorn margins must close exactly: {row_error}, {column_error}")

    # (ii) Label isolation: permuting fully held-out labels changes nothing that is fitted.
    permuted_blocks = permute_heldout_labels(blocks, task2_split)
    permuted_split = make_split(permuted_blocks, "task2", seed=42, validation_session="session1")
    permuted2 = EXP48.prepare_task2(
        [item for item in permuted_blocks if item.mouse == "mice1"], permuted_split, "mice1",
        "session1", subset, n_classes=n_classes,
    )
    np.testing.assert_array_equal(prepared2["train_x"], permuted2["train_x"])
    np.testing.assert_array_equal(prepared2["train_y"], permuted2["train_y"])
    np.testing.assert_array_equal(
        prepared2["normalizers"]["session1"][0], permuted2["normalizers"]["session1"][0]
    )
    np.testing.assert_array_equal(
        prepared2["normalizers"]["session1"][1], permuted2["normalizers"]["session1"][1]
    )
    original_validation = prepared2["validation"]["session1"]
    permuted_validation = permuted2["validation"]["session1"]
    require(not np.array_equal(original_validation[0], permuted_validation[0]),
            "Synthetic held-out label permutation did not reach the validation rows")
    require(np.array_equal(np.sort(original_validation[1]), np.sort(permuted_validation[1])),
            "Synthetic held-out label permutation changed the target multiset")

    torch.manual_seed(31337)
    model = SpikeClassifier(
        n_features=prepared2["train_x"].shape[1], n_classes=n_classes, hidden=8,
        dropout=0.0, apply_log=False,
    )
    model.set_normalization(prepared2["train_x"])
    state = {key: value.clone() for key, value in model.state_dict().items()}
    outcomes = []
    for prepared in (prepared2, permuted2):
        candidate = SpikeClassifier(
            n_features=prepared2["train_x"].shape[1], n_classes=n_classes, hidden=8,
            dropout=0.0, apply_log=False,
        )
        candidate.load_state_dict(state)
        candidate.set_normalization(prepared2["train_x"])
        validation = unlabeled_validation_sets(
            "task2", prepared, task2_split, blocks_by_key, "session1"
        )
        candidate, _ = run_transduction(
            candidate, prepared, validation, torch.device("cpu"), rounds=1,
            seed=11,
        )
        outcomes.append((
            normalized_row_softmax(
                predict_logits(candidate, validation[0]["rows"], batch_size=8, device="cpu")
            ),
            {key: value.clone() for key, value in candidate.state_dict().items()},
        ))
    np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
    for key in outcomes[0][1]:
        require(bool(torch.equal(outcomes[0][1][key], outcomes[1][1][key])),
                f"Fine-tuning changed fitted weights under a label permutation: {key}")

    # (iii) Soft cross-entropy against a hand computation.
    logits = torch.from_numpy(np.asarray(rng.normal(size=(4, n_classes)), dtype=np.float32))
    targets = rng.random((4, n_classes))
    targets /= targets.sum(axis=1, keepdims=True)
    target_tensor = torch.from_numpy(targets.astype(np.float32))
    manual = float(
        -(target_tensor
          * torch.log_softmax(logits, dim=1)).sum(dim=1).mean().item()
    )
    computed = float(soft_cross_entropy(logits, target_tensor).item())
    require(abs(manual - computed) < 1e-6, "Soft cross-entropy does not match the hand calculation")
    one_hot = F.one_hot(torch.arange(4) % n_classes, n_classes).float()
    require(abs(float(soft_cross_entropy(logits, one_hot).item())
                - float(F.cross_entropy(logits, torch.arange(4) % n_classes).item())) < 1e-6,
            "One-hot soft cross-entropy must equal hard cross-entropy")

    # (iv) Full-candidate 2-D axis, no 3-D tensors anywhere in scoring.
    batch = torch.from_numpy(np.ascontiguousarray(prepared2["train_x"][:3], dtype=np.float32))
    with torch.inference_mode():
        model.eval()
        output = model(batch)
    require(output.ndim == 2 and output.shape[1] == n_classes, "Model output must be 2-D over all candidates")
    for name, value in (("logits", logits.numpy()), ("soft", soft), ("probabilities", targets)):
        require(np.asarray(value).ndim == 2, f"{name} must stay 2-D")

    # Official OCF scoring path on synthetic images.
    image_rng = np.random.default_rng(772)
    images = (image_rng.random((n_classes, 64, 64)) > 0.82).astype(np.uint8) * 255
    images[np.arange(n_classes), np.arange(n_classes) % 64, :] = 255
    ocf = EXP48.OcfPairCache(images, device="cpu", batch_size=2)
    scores = np.asarray(rng.normal(size=(n_classes, n_classes)), dtype=np.float32)
    metrics = EXP48.evaluate_logits(scores, np.arange(n_classes), ocf)
    require(set(metrics) == set(METRICS) and all(np.isfinite(v) for v in metrics.values()),
            "Synthetic official-OCF metrics failed")

    # (v) Aggregation counts and gate polarity.
    block_rows = []
    for task in TASKS:
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        blocks_per_fold = 1 if task == "task2" else 4
        for epoch in (BASE_EPOCHS[task],
                      BASE_EPOCHS[task] + TRANSDUCTION_ROUNDS * FINE_TUNE_EPOCHS):
            for fold in fold_ids:
                for mouse in MICE:
                    for index in range(blocks_per_fold):
                        block_rows.append({
                            "task": task, "arm": ARM, "mouse": mouse, "fold": fold,
                            "block": f"{fold}/{index}", "epoch": epoch, "n_rows": 256,
                            "raw_top1_accuracy": 0.1, "raw_recall_at_32": 0.2,
                            "raw_top1_official_ocf": 20.0, "assigned_final_ocf": 18.0,
                        })
    macro = EXP48.aggregate_rows(block_rows, "macro")
    require(len(macro) == 4, "Smoke macro grid failed")
    task2_macro = EXP48.find_aggregate(macro, "task2", ARM, BASE_EPOCHS["task2"])
    task3_macro = EXP48.find_aggregate(macro, "task3", ARM, BASE_EPOCHS["task3"])
    require(task2_macro["n_evaluation_rows"] == 6 * 256, "Task2 macro row count failed")
    require(task3_macro["n_evaluation_rows"] == 16 * 256, "Task3 macro row count failed")
    mouse_fold = EXP48.aggregate_rows(block_rows, "mouse_fold")
    require(len(mouse_fold) == 20, "Smoke mouse_fold grid failed")
    require(all(row["n_evaluation_rows"] == (256 if row["task"] == "task2" else 1024)
                for row in mouse_fold), "Smoke mouse_fold row propagation failed")

    positive = evaluate_gate(
        "task2", 3.5, {"mice1": 0.4, "mice2": 0.1},
        {"session1": 0.2, "session2": -0.1, "session3": 0.5},
    )
    require(positive["advance"], "Positive advancement-gate smoke case failed")
    negative_cases = (
        ("task2", 2.9, {"mice1": 0.4, "mice2": 0.1}, {"session1": 0.2, "session2": 0.1, "session3": 0.3}),
        ("task2", 3.1, {"mice1": 0.4, "mice2": -0.01}, {"session1": 0.2, "session2": 0.1, "session3": 0.3}),
        ("task2", 3.1, {"mice1": 0.4, "mice2": 0.1}, {"session1": 0.2, "session2": -0.1, "session3": -0.2}),
        ("task3", 3.1, {"mice1": 0.4, "mice2": 0.1}, {"20261126": 0.2, "20261127": -0.2}),
        ("task3", 2.0, {"mice1": 0.4, "mice2": 0.1}, {"20261126": 0.2, "20261127": 0.2}),
    )
    for task, delta, per_mouse, per_fold in negative_cases:
        require(not evaluate_gate(task, delta, per_mouse, per_fold)["advance"],
                "Negative advancement-gate smoke case passed")
    print("SMOKE OK: Sinkhorn marginals, label isolation, soft CE, 2-D candidate axis, "
          "aggregation counts and gates", flush=True)


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
