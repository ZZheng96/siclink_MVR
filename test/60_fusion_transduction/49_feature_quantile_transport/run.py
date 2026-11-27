"""Preregistered development-only test of unlabeled per-feature quantile transport."""

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions
from src.features import FeatureBlock, aggregate_by_target
from src.neural import SpikeClassifier, predict_logits
from src.preprocessing import CACHE_SCHEMA, file_sha256, load_cached_blocks, validate_split
from src.retrieval import unique_assignment
from src.splits import make_split

OUT = Path(__file__).resolve().parent
PREREGISTRATION = OUT / "preregistration.json"
RESULTS = OUT / "development_results.json"
SELECTION = OUT / "selection.json"
SELECTION_48 = ROOT / "test/60_fusion_transduction/48_high_capacity_generalization/selection.json"
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
MICE = ("mice1", "mice2")
BLOCKS = tuple(f"{mouse}/session{session}" for mouse in MICE for session in range(1, 5))
TASKS = ("task2", "task3")
ARMS = ("baseline_zscore", "feature_quantile_transport")
N_CLASSES = 2748
BIN_MS = 25
REPEATS = 4
TRAIN_SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
GRADIENT_CLIP_NORM = 5.0
N_QUANTILES = 33
TASK2_FOLDS = ("session1", "session2", "session3")
TASK3_FOLDS = ("20261126", "20261127")
EVALUATION_EPOCHS_48 = (1, 2, 3, 4, 5, 7, 9, 12, 16, 20)
REQUIRED_NONNEGATIVE_FOLDS = {"task2": 2, "task3": 2}
MODEL = {
    "hidden": 512,
    "dropout": {"mice1": 0.2, "mice2": 0.35},
    "label_smoothing": 0.05,
    "weight_decay": 0.001,
    "learning_rate": LEARNING_RATE,
}
METRICS = (
    "raw_top1_accuracy",
    "raw_recall_at_32",
    "raw_top1_official_ocf",
    "assigned_final_ocf",
)


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


def array_sha256(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float32).tobytes()).hexdigest()


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
    require(data["n_classes"] == N_CLASSES and data["subset_size"] == 256, "Class/subset contract changed")
    require(
        data["development_picture_subset"]
        == "np.sort(np.random.default_rng(20261128).permutation(2748)[:256])",
        "Development picture subset changed",
    )
    require(data["same_subset_for_every_block_fold_mouse_arm_and_epoch"] is True, "Subset must be shared")
    folds = value["folds"]
    require(folds["task2"]["fold_ids"] == list(TASK2_FOLDS), "Task2 folds changed")
    require(folds["task3"]["fold_ids"] == list(TASK3_FOLDS), "Task3 folds changed")
    require("session4" not in folds["task2"]["fold_ids"], "Task2 session4 validation is prohibited")
    require("42" not in folds["task3"]["fold_ids"], "Task3 seed42 evaluation is prohibited")
    model = value["model"]
    require(model["class"] == "src.neural.SpikeClassifier", "Model class changed")
    require(model["recipe"].startswith("Fixed to the experiment 48 production_recipe"), "Model recipe changed")
    require(model["hidden"] == MODEL["hidden"] and model["dropout"] == MODEL["dropout"], "Architecture changed")
    require(model["label_smoothing"] == MODEL["label_smoothing"], "Label smoothing changed")
    require(
        model["weight_decay"] == MODEL["weight_decay"] and model["learning_rate"] == LEARNING_RATE,
        "Optimizer changed",
    )
    require(set(value["arms"]) == set(ARMS), "Arm set changed")
    transport = value["quantile_transport"]
    require(transport["n_quantiles"] == N_QUANTILES, "Quantile count changed")
    require(transport["quantile_grid"] == "np.linspace(0.0, 1.0, 33), fixed; never searched", "Quantile grid changed")
    require("np.quantile" in transport["estimator"] and "np.interp" in transport["estimator"], "Estimator changed")
    require("train-fold rows only" in transport["reference_isolation"], "Reference isolation contract changed")
    require("identity" in transport["degenerate_feature"], "Degenerate-feature contract changed")
    epoch_policy = value["epoch_policy"]
    require(epoch_policy["source"] == "test/60_fusion_transduction/48_high_capacity_generalization/selection.json", "Epoch source changed")
    require(epoch_policy["arm"] == "production_recipe", "Epoch source arm changed")
    require("never re-selects epochs or arms" in epoch_policy["no_research"], "Epoch no-research contract changed")
    training = value["shared_training_contract"]
    require(training["n_classes"] == N_CLASSES and training["apply_log"] is False, "Model input contract changed")
    require(training["optimizer"] == "AdamW" and training["batch_size"] == BATCH_SIZE, "Training contract changed")
    require(training["training_seed"] == TRAIN_SEED, "Training seed changed")
    require(training["gradient_clip_norm"] == GRADIENT_CLIP_NORM, "Gradient clipping changed")
    require(value["advancement_gate"]["macro_minimum_absolute_gain"] == 3.0, "Advancement gate changed")
    require(value["outputs"]["model_artifacts"] == "none retained", "Model artifacts must not be retained")
    prohibited = value["prohibited"]
    require("Task2 session4 validation or final evaluation" in prohibited, "Task2 prohibition missing")
    require("Task3 split seed 42 evaluation" in prohibited, "Task3 prohibition missing")
    require(
        "validation or target-session rows inside any quantile reference" in prohibited,
        "Reference isolation prohibition missing",
    )
    return value, hashlib.sha256(raw).hexdigest()


def parse_fixed_epochs(value):
    """Extract the two production_recipe epoch integers from an experiment 48 selection."""
    require(value.get("experiment") == "48_high_capacity_generalization", "Unexpected epoch source file")
    epochs = {}
    for task in TASKS:
        epoch = value["tasks"][task]["selected_epochs"]["production_recipe"]
        require(
            isinstance(epoch, int) and not isinstance(epoch, bool) and epoch in EVALUATION_EPOCHS_48,
            f"Invalid experiment 48 epoch for {task}",
        )
        epochs[task] = epoch
    return epochs


def load_fixed_epochs(path=SELECTION_48):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Experiment 48 selection is required before --run: {path}. "
            "Experiment 49 never searches epochs; finish experiment 48 first."
        )
    raw = path.read_bytes()
    return parse_fixed_epochs(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def preflight_cache():
    """Check cache integrity without creating, rebuilding, or modifying it."""
    manifest_path = CACHE / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Shared cache is not ready: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == CACHE_SCHEMA, "Unsupported cache schema")
    require(Path(manifest["source_root"]).resolve() == DATA.resolve(), "Cache source root mismatch")
    require(BIN_MS in manifest.get("bin_ms", []), "The cache has no 25ms counts")
    require(manifest.get("image_shape") == [N_CLASSES, 64, 64], "Gallery image shape mismatch")
    pictures = CACHE / "pictures.npy"
    require(pictures.is_file(), "Cached gallery images are missing")
    records = [record for record in manifest["blocks"] if record.get("split") == "train"]
    by_key = {f"{record['mouse']}/{record['session']}": record for record in records}
    require(len(records) == len(BLOCKS) and set(by_key) == set(BLOCKS), "Train cache blocks are incomplete")
    for key in BLOCKS:
        record = by_key[key]
        require(record["task"] == "train", f"Unexpected cached task for {key}")
        require(record["n_samples"] == 10992 and record["n_trials"] == 10992, f"Trial count changed for {key}")
        count = record["counts"][str(BIN_MS)]
        require(count["dtype"] == "uint16" and count["shape"][0] == 10992, f"Count contract changed for {key}")
        directory = CACHE / "blocks" / record["id"]
        count_path = directory / count["file"]
        require(count_path.is_file() and (directory / "index.npz").is_file(), f"Cache files missing for {key}")
        require(file_sha256(count_path) == count["sha256"], f"Cache hash mismatch for {key}")
    return manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def fit_session_normalizer(features):
    transformed = np.log1p(np.maximum(features, 0))
    mean = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = transformed.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1e-5)


def apply_session_normalizer(features, normalizer):
    mean, scale = normalizer
    return ((np.log1p(np.maximum(features, 0)) - mean) / scale).astype(np.float32)


def subset_positions(targets, subset):
    targets = np.asarray(targets, dtype=np.int64)
    subset = np.asarray(subset, dtype=np.int64)
    require(np.array_equal(targets, np.sort(targets)), "Aggregated targets must be sorted")
    positions = np.searchsorted(targets, subset)
    require(np.all(positions < len(targets)), "A development target is absent")
    require(np.array_equal(targets[positions], subset), "Development target mapping failed")
    return positions


def build_transport(source, reference, n_quantiles=N_QUANTILES):
    """Per-feature source->reference linear interpolation knots from unlabeled rows only."""
    source = np.asarray(source, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    require(source.ndim == 2 and reference.ndim == 2, "Transport inputs must be 2-D")
    require(len(source) > 0 and len(reference) > 0, "Transport inputs must be nonempty")
    require(source.shape[1] == reference.shape[1], "Transport feature mismatch")
    probabilities = np.linspace(0.0, 1.0, n_quantiles)
    source_knots = np.quantile(source.astype(np.float64), probabilities, axis=0)
    reference_knots = np.quantile(reference.astype(np.float64), probabilities, axis=0)
    knots = []
    for feature in range(source.shape[1]):
        source_column = source_knots[:, feature]
        reference_column = reference_knots[:, feature]
        if source_column[-1] <= source_column[0]:
            knots.append(None)  # Degenerate source feature: identity mapping.
            continue
        unique_source, inverse, counts = np.unique(source_column, return_inverse=True, return_counts=True)
        if len(unique_source) < len(source_column):
            # Consolidate duplicate source knots to the mean of their reference knots.
            sums = np.zeros(len(unique_source), dtype=np.float64)
            np.add.at(sums, inverse, reference_column)
            reference_column = sums / counts
        knots.append((unique_source, reference_column))
    return {"n_quantiles": n_quantiles, "n_features": source.shape[1], "knots": knots}


def apply_transport(matrix, transport):
    matrix = np.asarray(matrix, dtype=np.float32)
    require(
        matrix.ndim == 2 and matrix.shape[1] == transport["n_features"],
        "Transport application shape mismatch",
    )
    output = matrix.astype(np.float64)
    for feature, knot in enumerate(transport["knots"]):
        if knot is None:
            continue
        unique_source, reference_column = knot
        output[:, feature] = np.interp(matrix[:, feature], unique_source, reference_column)
    return output.astype(np.float32)


def _task2_session_rows(block, part, n_classes):
    """Baseline z-score singles and four-repeat means for one Task2 session."""
    raw = np.asarray(block.features[part.train_trials if len(part.train_trials) else part.validation_trials],
                     dtype=np.float32)
    labels = np.asarray(block.trial_targets[part.train_trials if len(part.train_trials) else part.validation_trials],
                        dtype=np.int64)
    normalizer = fit_session_normalizer(raw)
    singles = apply_session_normalizer(raw, normalizer)
    mean_targets, mean_raw = aggregate_by_target(raw, labels, method="mean")
    require(len(mean_targets) == n_classes, f"Repeat grouping failed for {block.key}")
    means = apply_session_normalizer(mean_raw, normalizer)
    return normalizer, singles, means, mean_targets, labels, len(raw)


def prepare_task2(mouse_blocks, split, mouse, fold, subset, arm, *, n_classes=N_CLASSES):
    """Baseline rows; the quantile arm additionally transports each session to a train-only reference."""
    require(arm in ARMS, f"Unknown arm {arm}")
    feature_parts, target_parts = [], []
    audit_blocks = []
    normalizers = {}
    train_sessions = []
    target = None
    for block in mouse_blocks:
        part = split.blocks[block.key]
        if block.session == fold:
            require(len(part.train_trials) == 0 and len(part.validation_trials) == block.n_trials,
                    f"Incomplete target session for {block.key}")
            normalizer, singles, means, mean_targets, _, n_trials = _task2_session_rows(
                block, part, n_classes)
            normalizers[block.session] = normalizer
            target = {"session": block.session, "singles": singles, "means": means,
                      "mean_targets": mean_targets, "n_trials": n_trials}
            audit_blocks.append({
                "block": block.key, "role": "validation", "train_trials": 0,
                "validation_trials": n_trials, "validation_rows": len(subset),
                "normalizer_source": "all unlabeled trials in the target session",
                "validation_row_policy": "four-repeat raw-count arithmetic mean, then target-session normalization",
            })
        else:
            require(len(part.validation_trials) == 0 and len(part.train_trials) == block.n_trials,
                    f"Training session leakage for {block.key}")
            normalizer, singles, means, mean_targets, labels, n_trials = _task2_session_rows(
                block, part, n_classes)
            normalizers[block.session] = normalizer
            train_sessions.append({"session": block.session, "singles": singles, "means": means,
                                   "mean_targets": mean_targets, "labels": labels, "n_trials": n_trials})
            audit_blocks.append({
                "block": block.key, "role": "train", "train_trials": n_trials,
                "single_trial_rows": len(singles), "four_repeat_mean_rows": len(means),
                "normalizer_source": "this fold's train_trials in this session only",
            })
    require(target is not None and len(train_sessions) == 3, "Task2 fold preparation failed")
    transport_audit = None
    if arm == "feature_quantile_transport":
        reference = np.concatenate([session["singles"] for session in train_sessions]).astype(np.float32)
        maps = []
        for session in train_sessions:
            transport = build_transport(session["singles"], reference)
            feature_parts.extend((apply_transport(session["singles"], transport),
                                  apply_transport(session["means"], transport)))
            target_parts.extend((session["labels"], session["mean_targets"]))
            maps.append({"session": session["session"], "source_rows": len(session["singles"]),
                         "applied_to": ["single_trial_rows", "four_repeat_mean_rows"]})
        target_transport = build_transport(target["singles"], reference)
        transported_target_means = apply_transport(target["means"], target_transport)
        maps.append({"session": target["session"], "source_rows": len(target["singles"]),
                     "source": "all unlabeled target-session trials",
                     "applied_to": ["validation four_repeat_mean_rows"]})
        transport_audit = {
            "n_quantiles": N_QUANTILES,
            "reference": "concatenated baseline z-score single-trial rows of the three training sessions",
            "reference_rows": int(len(reference)),
            "reference_sha256": array_sha256(reference),
            "reference_contains_target_session": False,
            "maps": maps,
        }
        positions = subset_positions(target["mean_targets"], subset)
        validation_x = transported_target_means[positions]
    else:
        for session in train_sessions:
            feature_parts.extend((session["singles"], session["means"]))
            target_parts.extend((session["labels"], session["mean_targets"]))
        positions = subset_positions(target["mean_targets"], subset)
        validation_x = target["means"][positions]
    validation_y = target["mean_targets"][positions]
    train_x = np.concatenate(feature_parts).astype(np.float32, copy=False)
    train_y = np.concatenate(target_parts).astype(np.int64, copy=False)
    single_rows = sum(len(session["singles"]) for session in train_sessions)
    mean_rows = sum(len(session["means"]) for session in train_sessions)
    require(len(train_x) == single_rows + mean_rows, "Task2 row count mismatch")
    require(np.array_equal(validation_y, subset), "Task2 validation rows are not subset-sorted")
    return {
        "task": "task2", "mouse": mouse, "fold": fold, "n_classes": n_classes,
        "train_x": train_x, "train_y": train_y,
        "validation": {fold: (validation_x, validation_y)},
        "normalizers": normalizers,
        "audit": {
            "task": "task2", "mouse": mouse, "fold": fold, "arm": arm,
            "single_trial_rows": single_rows, "four_repeat_mean_rows": mean_rows,
            "train_rows": len(train_x), "validation_rows_per_block": len(subset),
            "blocks": audit_blocks,
            "quantile_transport": transport_audit,
        },
    }


def prepare_task3(mouse_blocks, split, mouse, fold, subset, arm, *, n_classes=N_CLASSES):
    """Single trials only; the quantile arm transports each block's validation rows toward its train rows."""
    require(arm in ARMS, f"Unknown arm {arm}")
    feature_parts, target_parts = [], []
    validation = {}
    normalizers = {}
    audit_blocks = []
    for block in mouse_blocks:
        part = split.blocks[block.key]
        raw = np.asarray(block.features[part.train_trials], dtype=np.float32)
        labels = np.asarray(block.trial_targets[part.train_trials], dtype=np.int64)
        require(len(raw) == 3 * n_classes, f"Task3 must have three train trials per glyph in {block.key}")
        require(len(part.validation_trials) == n_classes, f"Task3 validation count changed in {block.key}")
        normalizer = fit_session_normalizer(raw)
        normalizers[block.session] = normalizer
        singles = apply_session_normalizer(raw, normalizer)
        feature_parts.append(singles)  # reference->reference is identity: keep baseline train rows.
        target_parts.append(labels)
        raw_validation = np.asarray(block.features[part.validation_trials], dtype=np.float32)
        validation_labels = np.asarray(block.trial_targets[part.validation_trials], dtype=np.int64)
        order = np.argsort(validation_labels, kind="stable")
        sorted_labels = validation_labels[order]
        sorted_validation = apply_session_normalizer(raw_validation[order], normalizer)
        transport_audit = None
        if arm == "feature_quantile_transport":
            transport = build_transport(sorted_validation, singles)
            sorted_validation = apply_transport(sorted_validation, transport)
            transport_audit = {
                "n_quantiles": N_QUANTILES,
                "reference": "this block's train-fold baseline z-score single-trial rows",
                "reference_rows": int(len(singles)),
                "reference_sha256": array_sha256(singles),
                "source": "this block's unlabeled validation single-trial rows",
                "source_rows": int(len(sorted_validation)),
                "applied_to": ["validation single-trial rows"],
                "training_rows_transformed": False,
            }
        positions = subset_positions(sorted_labels, subset)
        validation[block.key] = (sorted_validation[positions], sorted_labels[positions])
        audit_blocks.append({
            "block": block.key, "role": "train_and_validation", "train_trials": len(raw),
            "single_trial_rows": len(singles), "four_repeat_mean_rows": 0,
            "validation_trials": len(raw_validation), "validation_rows": len(subset),
            "normalizer_source": "this block's train_trials only",
            "validation_row_policy": "strictly one unmerged trial per target",
            "quantile_transport": transport_audit,
        })
    train_x = np.concatenate(feature_parts).astype(np.float32, copy=False)
    train_y = np.concatenate(target_parts).astype(np.int64, copy=False)
    require(len(train_x) == 4 * 3 * n_classes and len(validation) == 4, "Task3 fold preparation failed")
    require(all(np.array_equal(targets, subset) for _, targets in validation.values()),
            "Task3 validation rows are not subset-sorted")
    return {
        "task": "task3", "mouse": mouse, "fold": fold, "n_classes": n_classes,
        "train_x": train_x, "train_y": train_y, "validation": validation,
        "normalizers": normalizers,
        "audit": {
            "task": "task3", "mouse": mouse, "fold": fold, "arm": arm,
            "single_trial_rows": len(train_x), "four_repeat_mean_rows": 0,
            "train_rows": len(train_x), "validation_rows_per_block": len(subset),
            "blocks": audit_blocks,
        },
    }


class OcfPairCache:
    """Cache only requested official-OCF pairs in memory; never write metric caches."""

    def __init__(self, images, *, device="cpu", batch_size=32):
        self.images = np.asarray(images)
        self.device = str(device)
        self.batch_size = batch_size
        self.values = {}
        self.calls = 0
        self.scored_pairs = 0

    def score(self, targets, predictions):
        targets = np.asarray(targets, dtype=np.int64)
        predictions = np.asarray(predictions, dtype=np.int64)
        require(targets.shape == predictions.shape and targets.ndim == 1, "OCF pairs must be aligned vectors")
        new = []
        seen = set()
        for target, prediction in zip(targets.tolist(), predictions.tolist()):
            pair = (target, prediction)
            if pair not in self.values and pair not in seen:
                new.append(pair)
                seen.add(pair)
        if new:
            true_ids = np.fromiter((pair[0] for pair in new), dtype=np.int64, count=len(new))
            predicted_ids = np.fromiter((pair[1] for pair in new), dtype=np.int64, count=len(new))
            scores = score_predictions(
                self.images[predicted_ids], self.images[true_ids],
                device=self.device, batch_size=self.batch_size,
            )
            require(len(scores) == len(new) and np.isfinite(scores).all(), "Official OCF scoring failed")
            self.values.update(zip(new, map(float, scores)))
            self.calls += 1
            self.scored_pairs += len(new)
        return np.asarray([self.values[(int(t), int(p))] for t, p in zip(targets, predictions)], dtype=np.float64)


def evaluate_logits(logits, targets, ocf_cache):
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    require(logits.ndim == 2 and len(logits) == len(targets) and np.isfinite(logits).all(), "Invalid raw logits")
    raw = logits.argmax(axis=1).astype(np.int64)
    top = np.argsort(-logits, axis=1, kind="stable")[:, :min(32, logits.shape[1])]
    recall = np.any(top == targets[:, None], axis=1)
    assigned = unique_assignment(logits)
    require(len(np.unique(assigned)) == len(assigned), "Hungarian assignment is not unique")
    raw_ocf = ocf_cache.score(targets, raw)
    assigned_ocf = ocf_cache.score(targets, assigned)
    return {
        "raw_top1_accuracy": float(np.mean(raw == targets)),
        "raw_recall_at_32": float(np.mean(recall)),
        "raw_top1_official_ocf": float(raw_ocf.mean()),
        "assigned_final_ocf": float(assigned_ocf.mean()),
    }


def fit_fold(prepared, arm, ocf_cache, *, epochs, device, batch_size=BATCH_SIZE, seed=TRAIN_SEED):
    """Fit one mouse/fold/arm for the fixed epoch count without saving model parameters."""
    device = torch.device(device)
    train_x = prepared["train_x"]
    train_y = prepared["train_y"]
    n_classes = prepared["n_classes"]
    require(isinstance(epochs, int) and epochs >= 1, "A fixed positive epoch count is required")
    require(np.isfinite(train_x).all() and len(train_x) == len(train_y), "Invalid training matrix")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpikeClassifier(
        n_features=train_x.shape[1], n_classes=n_classes, hidden=MODEL["hidden"],
        dropout=MODEL["dropout"].get(prepared["mouse"], 0.0),
        apply_log=False,
    )
    model.set_normalization(train_x)
    expected_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(model.feature_mean.cpu().numpy(), expected_mean)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MODEL["learning_rate"], weight_decay=MODEL["weight_decay"],
        fused=device.type == "cuda",
    )
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(train_y))
        model.train()
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_x = torch.from_numpy(np.ascontiguousarray(train_x[indices], dtype=np.float32)).to(device)
            batch_y = torch.from_numpy(train_y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch_x)
                loss = F.cross_entropy(logits.float(), batch_y, label_smoothing=MODEL["label_smoothing"])
            require(bool(torch.isfinite(loss).item()), "Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(
            f"TRAIN task={prepared['task']} fold={prepared['fold']} mouse={prepared['mouse']} "
            f"arm={arm} epoch={epoch}/{epochs} train_loss={total_loss / len(train_y):.6f}", flush=True,
        )
    rows = []
    for block, (validation_x, validation_y) in prepared["validation"].items():
        raw_logits = predict_logits(model, validation_x, batch_size=batch_size, device=device)
        metrics = evaluate_logits(raw_logits, validation_y, ocf_cache)
        rows.append({
            "task": prepared["task"], "arm": arm, "mouse": prepared["mouse"],
            "fold": prepared["fold"], "block": block, "epoch": epochs,
            "n_rows": len(validation_y), **metrics,
        })
    print(
        f"EVAL task={prepared['task']} fold={prepared['fold']} mouse={prepared['mouse']} "
        f"arm={arm} epoch={epochs} " +
        json.dumps({name: float(np.mean([row[name] for row in rows])) for name in METRICS}),
        flush=True,
    )
    require(len(rows) == len(prepared["validation"]), "Fixed-epoch evaluation rows are incomplete")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def aggregate_rows(block_rows, level):
    keys_by_level = {
        "mouse_fold": ("task", "arm", "mouse", "fold"),
        "mouse": ("task", "arm", "mouse"),
        "fold": ("task", "arm", "fold"),
        "macro": ("task", "arm"),
    }
    require(level in keys_by_level, "Unknown aggregation level")
    keys = keys_by_level[level]
    source = block_rows if level == "mouse_fold" else aggregate_rows(block_rows, "mouse_fold")
    groups = defaultdict(list)
    for row in source:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key in sorted(groups):
        members = groups[group_key]
        row = dict(zip(keys, group_key))
        epochs = {member["epoch"] for member in members}
        require(len(epochs) == 1, "Aggregated rows must share the fixed epoch")
        row["epoch"] = epochs.pop()
        row["n_source_cells"] = len(members)
        if level == "mouse_fold":
            row["n_evaluation_rows"] = sum(member["n_rows"] for member in members)
        else:
            row["n_evaluation_rows"] = sum(member["n_evaluation_rows"] for member in members)
        for metric in METRICS:
            row[metric] = float(np.mean([member[metric] for member in members], dtype=np.float64))
        output.append(row)
    return output


def find_aggregate(rows, task, arm, **identifiers):
    matches = [row for row in rows if row["task"] == task and row["arm"] == arm
               and all(row[key] == value for key, value in identifiers.items())]
    require(len(matches) == 1, "Expected one aggregate selection row")
    return matches[0]


def build_selection(results, fixed_epochs, epoch_source_sha256):
    macro = results["macro"]
    mouse_rows = results["mouse"]
    fold_rows = results["fold"]
    tasks = {}
    for task in TASKS:
        epoch = fixed_epochs[task]
        baseline = find_aggregate(macro, task, "baseline_zscore")
        transported = find_aggregate(macro, task, "feature_quantile_transport")
        require(baseline["epoch"] == epoch and transported["epoch"] == epoch,
                f"Fixed-epoch mismatch for {task}")
        per_mouse = {}
        for mouse in MICE:
            b = find_aggregate(mouse_rows, task, "baseline_zscore", mouse=mouse)
            q = find_aggregate(mouse_rows, task, "feature_quantile_transport", mouse=mouse)
            per_mouse[mouse] = float(q["assigned_final_ocf"] - b["assigned_final_ocf"])
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        per_fold = {}
        for fold in fold_ids:
            b = find_aggregate(fold_rows, task, "baseline_zscore", fold=fold)
            q = find_aggregate(fold_rows, task, "feature_quantile_transport", fold=fold)
            per_fold[fold] = float(q["assigned_final_ocf"] - b["assigned_final_ocf"])
        macro_delta = float(transported["assigned_final_ocf"] - baseline["assigned_final_ocf"])
        required_folds = REQUIRED_NONNEGATIVE_FOLDS[task]
        nonnegative_folds = sum(delta >= 0 for delta in per_fold.values())
        gates = {
            "macro_assigned_ocf_delta_at_least_3": macro_delta >= 3.0,
            "both_mouse_deltas_nonnegative": all(delta >= 0 for delta in per_mouse.values()),
            "fold_deltas_nonnegative": (
                nonnegative_folds >= required_folds and nonnegative_folds > len(fold_ids) / 2
            ),
        }
        tasks[task] = {
            "fixed_epoch": epoch,
            "arms": {"baseline_zscore": baseline, "feature_quantile_transport": transported},
            "quantile_transport_minus_baseline": {
                "macro_assigned_final_ocf": macro_delta,
                "per_mouse_assigned_final_ocf": per_mouse,
                "per_fold_assigned_final_ocf": per_fold,
            },
            "nonnegative_fold_count": nonnegative_folds,
            "required_nonnegative_fold_count": required_folds,
            "gates": gates,
            "advance": all(gates.values()),
            "next_step": "development gate passed; separate pristine/final authorization still required"
                         if all(gates.values()) else "stop; no alternate transform or hyperparameter search",
        }
    return {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_selection_complete",
        "development_reuse": "development reuse only; not pristine validation; not a Kaggle result",
        "epoch_source": {
            "path": "test/60_fusion_transduction/48_high_capacity_generalization/selection.json",
            "sha256": epoch_source_sha256,
            "arm": "production_recipe",
            "fixed_epochs": fixed_epochs,
        },
        "tasks": tasks,
    }


def run_experiment():
    preregistration, preregistration_hash = validate_preregistration()
    fixed_epochs, epoch_source_sha256 = load_fixed_epochs()
    print(f"Fixed epochs from experiment 48 production_recipe: {fixed_epochs}", flush=True)
    require(not RESULTS.exists() and not SELECTION.exists(),
            "Development outputs already exist; refusing an unpreregistered repeated selection run")
    require(torch.cuda.is_available(), "The preregistered bf16 CUDA experiment requires CUDA")
    device = torch.device("cuda")
    manifest, manifest_hash = preflight_cache()
    subset = np.sort(np.random.default_rng(20261128).permutation(N_CLASSES)[:256]).astype(np.int64)
    require(len(subset) == 256 and len(np.unique(subset)) == 256, "Invalid development picture subset")
    print("Loading read-only public train blocks and gallery images", flush=True)
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=BIN_MS)
    require(tuple(block.key for block in blocks) == tuple(sorted(BLOCKS)), "Loaded block set mismatch")
    images = (np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False) >= 128).astype(np.float32)
    require(images.shape == (N_CLASSES, 64, 64), "Gallery image cache mismatch")
    ocf_cache = OcfPairCache(images, device="cuda", batch_size=32)
    block_rows = []
    fold_audits = []
    started = time.time()
    fits_completed = 0
    total_fits = (len(TASK2_FOLDS) + len(TASK3_FOLDS)) * len(MICE) * len(ARMS)
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
                for arm in ARMS:
                    prepare = prepare_task2 if task == "task2" else prepare_task3
                    prepared = prepare(mouse_blocks, split, mouse, fold, subset, arm)
                    fold_audits.append(prepared["audit"])
                    print(
                        f"START fit={fits_completed + 1}/{total_fits} task={task} fold={fold} mouse={mouse} "
                        f"arm={arm} fixed_epoch={fixed_epochs[task]} train_rows={len(prepared['train_y'])}",
                        flush=True,
                    )
                    block_rows.extend(
                        fit_fold(prepared, arm, ocf_cache, epochs=fixed_epochs[task], device=device)
                    )
                    fits_completed += 1
                    print(f"DONE fit={fits_completed}/{total_fits} elapsed_seconds={time.time() - started:.1f}",
                          flush=True)
                    prepared.clear()
                    gc.collect()
    require(fits_completed == total_fits, "Not all preregistered fits completed")
    require(len(block_rows) == 12 + 32, "Incomplete block result grid")
    results = {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_results_complete",
        "data_policy": "public train only; development reuse; no Task2 session4 or Task3 seed42 final evaluation",
        "development_reuse": "development reuse only; not pristine validation; not a Kaggle result",
        "preregistration_sha256": preregistration_hash,
        "cache_manifest_sha256": manifest_hash,
        "epoch_source": {
            "path": "test/60_fusion_transduction/48_high_capacity_generalization/selection.json",
            "sha256": epoch_source_sha256,
            "arm": "production_recipe",
            "fixed_epochs": fixed_epochs,
        },
        "run": {
            "device": torch.cuda.get_device_name(device),
            "bin_ms": BIN_MS,
            "training_seed": TRAIN_SEED,
            "n_quantiles": N_QUANTILES,
            "fits_completed": fits_completed,
            "model_artifacts_retained": False,
            "ocf_pair_cache": {
                "storage": "memory only", "scoring_calls": ocf_cache.calls,
                "unique_pairs_scored": ocf_cache.scored_pairs,
            },
            "elapsed_seconds": float(time.time() - started),
        },
        "development_picture_subset": subset.tolist(),
        "fold_audits": fold_audits,
        "block": block_rows,
        "mouse_fold": aggregate_rows(block_rows, "mouse_fold"),
        "mouse": aggregate_rows(block_rows, "mouse"),
        "fold": aggregate_rows(block_rows, "fold"),
        "macro": aggregate_rows(block_rows, "macro"),
    }
    require(len(results["mouse_fold"]) == 20 and len(results["mouse"]) == 8, "Mouse aggregation grid mismatch")
    require(len(results["fold"]) == 10 and len(results["macro"]) == 4, "Fold/macro aggregation grid mismatch")
    write_json(RESULTS, results)
    selection = build_selection(results, fixed_epochs, epoch_source_sha256)
    selection["preregistration_sha256"] = preregistration_hash
    selection["development_results_sha256"] = file_sha256(RESULTS)
    selection["historical_or_final_metrics_used"] = False
    write_json(SELECTION, selection)
    require(hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest() == preregistration_hash,
            "Preregistration changed during the run")
    print(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print(f"Wrote {RESULTS} and {SELECTION}; no model artifacts retained", flush=True)


def synthetic_blocks(n_classes=6, repeats=4):
    rng = np.random.default_rng(9182)
    blocks = []
    for mouse_index, mouse in enumerate(MICE):
        for session_index in range(1, 5):
            targets = np.tile(np.arange(n_classes, dtype=np.int64), repeats)
            targets = targets[rng.permutation(len(targets))]
            features = rng.poisson(0.4 + mouse_index, size=(len(targets), 9)).astype(np.float32)
            features += (targets[:, None] == np.arange(features.shape[1])[None, :] % n_classes) * 3
            blocks.append(FeatureBlock(
                mouse=mouse, session=f"session{session_index}", split="train", task="train",
                sample_ids=np.arange(len(targets), dtype=np.int64),
                group_ptr=np.arange(len(targets) + 1, dtype=np.int64), targets=targets,
                features=features, n_units=9, bin_ms=25,
            ))
    return sorted(blocks, key=lambda block: (block.mouse, block.session))


def smoke_quantile_transport_unit():
    """Duplicate-safe knots, degenerate identity, and source->reference distribution movement."""
    rng = np.random.default_rng(315)
    reference = rng.normal(size=(4000, 7)).astype(np.float32)
    source = rng.normal(loc=3.0, scale=2.0, size=(3000, 7)).astype(np.float32)
    source[:, 3] = np.repeat([1.0, 2.0, 2.0, 5.0], 750)  # duplicate source knots
    source[:, 5] = 4.25  # degenerate source feature
    reference[:, 5] = rng.normal(size=4000)
    transport = build_transport(source, reference)
    require(transport["knots"][5] is None, "Degenerate feature must keep an identity mapping")
    mapped = apply_transport(source, transport)
    require(np.isfinite(mapped).all(), "Duplicate source knots produced nonfinite values")
    np.testing.assert_array_equal(mapped[:, 5], source[:, 5], "Degenerate feature is not identity")
    probabilities = np.linspace(0.0, 1.0, N_QUANTILES)
    reference_knots = np.quantile(reference[:, 3].astype(np.float64), probabilities)
    require(
        mapped[:, 3].min() >= reference_knots[0] - 1e-4 and mapped[:, 3].max() <= reference_knots[-1] + 1e-4,
        "Duplicate-knot column escaped the reference knot range",
    )
    tied = apply_transport(np.array([[1.0], [1.0], [1.0], [2.0]], dtype=np.float32),
                           build_transport(np.array([[1.0], [1.0], [1.0], [2.0]], dtype=np.float32),
                                           np.array([[0.0], [0.0], [10.0], [10.0]], dtype=np.float32)))
    require(np.isfinite(tied).all() and tied[3, 0] >= tied[0, 0], "Tied source knots are not handled safely")
    for column in (0, 1, 2, 4, 6):
        moved = np.quantile(mapped[:, column], probabilities)
        target_knots = np.quantile(reference[:, column], probabilities)
        source_knots = np.quantile(source[:, column], probabilities)
        require(np.max(np.abs(source_knots - target_knots)) > 1.0, "Synthetic shift check is vacuous")
        np.testing.assert_allclose(moved, target_knots, atol=0.05,
                                   err_msg="Transported source quantiles do not match the reference")
    identity = apply_transport(reference, build_transport(reference, reference))
    np.testing.assert_allclose(identity, reference, atol=1e-5,
                               err_msg="reference->reference transport is not identity")


def smoke_test():
    """Synthetic-only checks; never load real blocks or compute real development metrics."""
    validate_preregistration()
    smoke_quantile_transport_unit()
    try:
        load_fixed_epochs(OUT / "definitely_missing_selection.json")
    except FileNotFoundError:
        pass
    else:
        raise ValueError("A missing experiment 48 selection must fail immediately")
    epochs = parse_fixed_epochs({
        "experiment": "48_high_capacity_generalization",
        "tasks": {task: {"selected_epochs": {"production_recipe": 7}} for task in TASKS},
    })
    require(epochs == {"task2": 7, "task3": 7}, "Epoch handoff parsing failed")
    for bad in (6, "7", None):
        try:
            parse_fixed_epochs({
                "experiment": "48_high_capacity_generalization",
                "tasks": {task: {"selected_epochs": {"production_recipe": bad}} for task in TASKS},
            })
        except (ValueError, KeyError, TypeError):
            pass
        else:
            raise ValueError("Invalid epoch handoff was accepted")
    n_classes = 6
    repeats = 4
    subset = np.arange(n_classes, dtype=np.int64)
    blocks = synthetic_blocks(n_classes, repeats)
    block = blocks[0]
    train_raw = np.asarray(block.features[:12], dtype=np.float32)
    normalizer = fit_session_normalizer(train_raw)
    shifted = np.asarray(block.features[12:], dtype=np.float32) + 1e6
    np.testing.assert_array_equal(normalizer[0], fit_session_normalizer(train_raw)[0])
    require(not np.array_equal(normalizer[0], fit_session_normalizer(shifted)[0]),
            "Synthetic normalizer isolation check is vacuous")
    np.testing.assert_allclose(
        apply_session_normalizer(train_raw, normalizer).mean(axis=0), 0, atol=2e-6
    )
    task2_split = make_split(blocks, "task2", seed=42, validation_session="session1")
    mice1_blocks = [b for b in blocks if b.mouse == "mice1"]
    prepared2_base = prepare_task2(mice1_blocks, task2_split, "mice1", "session1", subset,
                                   "baseline_zscore", n_classes=n_classes)
    prepared2_qt = prepare_task2(mice1_blocks, task2_split, "mice1", "session1", subset,
                                 "feature_quantile_transport", n_classes=n_classes)
    require(prepared2_base["audit"]["single_trial_rows"] == 3 * n_classes * repeats, "Task2 single-row policy failed")
    require(prepared2_base["audit"]["four_repeat_mean_rows"] == 3 * n_classes, "Task2 mean-row policy failed")
    require(len(prepared2_base["train_y"]) == 3 * n_classes * (repeats + 1), "Task2 total rows failed")
    np.testing.assert_array_equal(prepared2_base["train_y"], prepared2_qt["train_y"])
    require(prepared2_base["train_x"].shape == prepared2_qt["train_x"].shape, "QT changed Task2 row counts")
    # Reference isolation: the Task2 reference is exactly the three training sessions' singles.
    expected_reference = []
    for train_block in mice1_blocks:
        if train_block.session == "session1":
            continue
        raw = np.asarray(train_block.features, dtype=np.float32)
        expected_reference.append(apply_session_normalizer(raw, fit_session_normalizer(raw)))
    expected_reference = np.concatenate(expected_reference).astype(np.float32)
    qt_audit = prepared2_qt["audit"]["quantile_transport"]
    require(qt_audit is not None and qt_audit["reference_contains_target_session"] is False,
            "Task2 reference audit missing")
    require(qt_audit["reference_rows"] == len(expected_reference), "Task2 reference row count mismatch")
    require(qt_audit["reference_sha256"] == array_sha256(expected_reference),
            "Task2 reference is not isolated to the training sessions")
    require(qt_audit["maps"][-1]["session"] == "session1"
            and qt_audit["maps"][-1]["source_rows"] == n_classes * repeats,
            "Task2 target map must use all unlabeled target singles")
    shifted_blocks = []
    for b in mice1_blocks:
        features = b.features + (1e3 if b.session == "session1" else 0.0)
        shifted_blocks.append(FeatureBlock(
            mouse=b.mouse, session=b.session, split=b.split, task=b.task, sample_ids=b.sample_ids,
            group_ptr=b.group_ptr, targets=b.targets, features=features.astype(np.float32),
            n_units=b.n_units, bin_ms=b.bin_ms,
        ))
    shifted_qt = prepare_task2(shifted_blocks, task2_split, "mice1", "session1", subset,
                               "feature_quantile_transport", n_classes=n_classes)
    require(shifted_qt["audit"]["quantile_transport"]["reference_sha256"] == qt_audit["reference_sha256"],
            "Target-session rows leaked into the Task2 reference")
    target_block = next(b for b in blocks if b.key == "mice1/session1")
    raw_target = np.asarray(target_block.features, dtype=np.float32)
    mean_targets, mean_raw = aggregate_by_target(raw_target, target_block.trial_targets, method="mean")
    expected_validation = apply_session_normalizer(mean_raw, prepared2_base["normalizers"]["session1"])
    np.testing.assert_array_equal(prepared2_base["validation"]["session1"][0], expected_validation)
    task3_split = make_split(blocks, "task3", seed=20261126)
    prepared3_base = prepare_task3(mice1_blocks, task3_split, "mice1", "20261126", subset,
                                   "baseline_zscore", n_classes=n_classes)
    prepared3_qt = prepare_task3(mice1_blocks, task3_split, "mice1", "20261126", subset,
                                 "feature_quantile_transport", n_classes=n_classes)
    require(len(prepared3_base["train_y"]) == 4 * 3 * n_classes, "Task3 single-trial train policy failed")
    require(prepared3_base["audit"]["four_repeat_mean_rows"] == 0, "Task3 repeat aggregation is prohibited")
    require(all(len(x) == n_classes for x, _ in prepared3_base["validation"].values()),
            "Task3 validation rows failed")
    # Task3 training identity: QT keeps the exact baseline training rows.
    np.testing.assert_array_equal(prepared3_base["train_x"], prepared3_qt["train_x"])
    np.testing.assert_array_equal(prepared3_base["train_y"], prepared3_qt["train_y"])
    for train_block in mice1_blocks:
        qt_block = next(row for row in prepared3_qt["audit"]["blocks"] if row["block"] == train_block.key)
        raw = np.asarray(train_block.features[task3_split.blocks[train_block.key].train_trials], dtype=np.float32)
        reference = apply_session_normalizer(raw, fit_session_normalizer(raw))
        require(qt_block["quantile_transport"]["reference_sha256"] == array_sha256(reference),
                f"Task3 reference is not isolated to the train fold of {train_block.key}")
        require(qt_block["quantile_transport"]["training_rows_transformed"] is False,
                "Task3 training rows must stay on the baseline branch")
    model = SpikeClassifier(prepared3_base["train_x"].shape[1], n_classes, hidden=8, dropout=0, apply_log=False)
    model.set_normalization(prepared3_base["train_x"])
    np.testing.assert_array_equal(
        model.feature_mean.numpy(), prepared3_base["train_x"].mean(axis=0, dtype=np.float64).astype(np.float32)
    )
    rng = np.random.default_rng(772)
    images = (rng.random((n_classes, 64, 64)) > 0.82).astype(np.uint8) * 255
    images[np.arange(n_classes), np.arange(n_classes) % 64, :] = 255
    ocf = OcfPairCache(images, device="cpu", batch_size=2)
    pairs_t = np.array([0, 1, 2, 0], dtype=np.int64)
    pairs_p = np.array([0, 2, 2, 1], dtype=np.int64)
    cached = ocf.score(pairs_t, pairs_p)
    direct = score_predictions(images[pairs_p], images[pairs_t], device="cpu", batch_size=2)
    np.testing.assert_array_equal(cached, direct)
    rows2 = fit_fold(prepared2_qt, "feature_quantile_transport", ocf, epochs=2, device="cpu", batch_size=16)
    rows3 = fit_fold(prepared3_qt, "feature_quantile_transport", ocf, epochs=2, device="cpu", batch_size=16)
    rows3b = fit_fold(prepared3_base, "baseline_zscore", ocf, epochs=2, device="cpu", batch_size=16)
    require(len(rows2) == 1 and len(rows3) == 4 and len(rows3b) == 4, "Smoke evaluation grid failed")
    require(all(row["epoch"] == 2 for row in rows2 + rows3 + rows3b), "Fixed-epoch rows failed")
    for row in rows2 + rows3 + rows3b:
        require(all(np.isfinite(row[metric]) for metric in METRICS), "Nonfinite smoke metric")
    for level, expected in (("mouse_fold", 3), ("mouse", 3), ("fold", 3), ("macro", 3)):
        require(len(aggregate_rows(rows2 + rows3 + rows3b, level)) == expected,
                f"Aggregation grid failed at {level}")
    # Advancement gates: one positive case and structural negative cases.
    fixed_epochs = {"task2": 7, "task3": 7}
    synthetic_results = {"macro": [], "mouse": [], "fold": []}

    def fill(results, qt_assigned, mouse_overrides=None, fold_overrides=None):
        for task in TASKS:
            fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
            for arm in ARMS:
                assigned = 50.0 if arm == "baseline_zscore" else qt_assigned
                common = {
                    "task": task, "arm": arm, "epoch": fixed_epochs[task], "n_rows": 256,
                    "raw_top1_accuracy": 0.1, "raw_recall_at_32": 0.2,
                    "raw_top1_official_ocf": 20.0, "assigned_final_ocf": assigned,
                }
                results["macro"].append(dict(common))
                for mouse in MICE:
                    value = assigned
                    if arm != "baseline_zscore" and mouse_overrides and mouse in mouse_overrides:
                        value = mouse_overrides[mouse]
                    results["mouse"].append({**common, "mouse": mouse, "assigned_final_ocf": value})
                for fold in fold_ids:
                    value = assigned
                    if arm != "baseline_zscore" and fold_overrides and fold in fold_overrides:
                        value = fold_overrides[fold]
                    results["fold"].append({**common, "fold": fold, "assigned_final_ocf": value})

    fill(synthetic_results, 53.1)
    positive = build_selection(synthetic_results, fixed_epochs, "0" * 64)
    require(all(positive["tasks"][task]["advance"] for task in TASKS),
            "Positive advancement-gate smoke case failed")
    for overrides, label in (
        ({"qt_assigned": 52.9}, "macro below +3"),
        ({"mouse_overrides": {"mice2": 49.0}}, "negative mouse delta"),
        ({"fold_overrides": {"session2": 49.0, "session3": 49.0, "20261127": 49.0}}, "fold majority failure"),
    ):
        failed_results = {"macro": [], "mouse": [], "fold": []}
        fill(failed_results, overrides.get("qt_assigned", 53.1),
             mouse_overrides=overrides.get("mouse_overrides"),
             fold_overrides=overrides.get("fold_overrides"))
        failed = build_selection(failed_results, fixed_epochs, "0" * 64)
        require(all(not failed["tasks"][task]["advance"] for task in TASKS),
                f"Negative advancement-gate smoke case failed: {label}")
    print("SMOKE OK: reference isolation, duplicate/degenerate knots, Task3 train identity, distribution movement, "
          "fixed-epoch grid, official OCF, selection and gates", flush=True)


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
