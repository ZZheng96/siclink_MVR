"""Preregistered development-only comparison of two MLP training recipes."""

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
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
MICE = ("mice1", "mice2")
BLOCKS = tuple(f"{mouse}/session{session}" for mouse in MICE for session in range(1, 5))
TASKS = ("task2", "task3")
ARMS = ("production_recipe", "high_capacity_noreg")
N_CLASSES = 2748
BIN_MS = 25
REPEATS = 4
TRAIN_SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 20
EVALUATION_EPOCHS = (1, 2, 3, 4, 5, 7, 9, 12, 16, 20)
LEARNING_RATE = 3e-4
GRADIENT_CLIP_NORM = 5.0
TASK2_FOLDS = ("session1", "session2", "session3")
TASK3_FOLDS = ("20261126", "20261127")
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
    expected_arms = {
        "production_recipe": (512, {"mice1": 0.2, "mice2": 0.35}, 0.05, 0.001),
        "high_capacity_noreg": (1024, {"mice1": 0.0, "mice2": 0.0}, 0.0, 0.0),
    }
    for arm, (hidden, dropout, smoothing, decay) in expected_arms.items():
        row = value["arms"][arm]
        require(row["class"] == "src.neural.SpikeClassifier", f"Model class changed for {arm}")
        require(row["hidden"] == hidden and row["dropout"] == dropout, f"Architecture changed for {arm}")
        require(row["label_smoothing"] == smoothing, f"Label smoothing changed for {arm}")
        require(row["weight_decay"] == decay and row["learning_rate"] == LEARNING_RATE, f"Optimizer changed for {arm}")
    training = value["shared_training_contract"]
    require(training["n_classes"] == N_CLASSES and training["apply_log"] is False, "Model input contract changed")
    require(training["optimizer"] == "AdamW" and training["batch_size"] == BATCH_SIZE, "Training contract changed")
    require(training["training_seed"] == TRAIN_SEED and training["max_epochs"] == MAX_EPOCHS, "Seed/epoch contract changed")
    require(training["evaluation_epochs"] == list(EVALUATION_EPOCHS), "Evaluation checkpoints changed")
    require(training["gradient_clip_norm"] == GRADIENT_CLIP_NORM, "Gradient clipping changed")
    require(value["advancement_gate"]["macro_minimum_absolute_gain"] == 3.0, "Advancement gate changed")
    require(value["outputs"]["model_artifacts"] == "none retained", "Model artifacts must not be retained")
    prohibited = value["prohibited"]
    require("Task2 session4 validation or final evaluation" in prohibited, "Task2 prohibition missing")
    require("Task3 split seed 42 evaluation" in prohibited, "Task3 prohibition missing")
    return value, hashlib.sha256(raw).hexdigest()


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


def prepare_task2(mouse_blocks, split, mouse, fold, subset, *, n_classes=N_CLASSES):
    """Fit train-only session normalizers; locally group target repeats for validation."""
    feature_parts, target_parts = [], []
    audit_blocks = []
    normalizers = {}
    validation_x = validation_y = None
    single_rows = mean_rows = 0
    for block in mouse_blocks:
        part = split.blocks[block.key]
        if block.session == fold:
            require(len(part.train_trials) == 0 and len(part.validation_trials) == block.n_trials,
                    f"Incomplete target session for {block.key}")
            raw = np.asarray(block.features[part.validation_trials], dtype=np.float32)
            labels = np.asarray(block.trial_targets[part.validation_trials], dtype=np.int64)
            normalizer = fit_session_normalizer(raw)
            normalizers[block.session] = normalizer
            mean_targets, mean_raw = aggregate_by_target(raw, labels, method="mean")
            require(len(mean_targets) == n_classes, f"Target repeat grouping failed for {block.key}")
            positions = subset_positions(mean_targets, subset)
            validation_x = apply_session_normalizer(mean_raw[positions], normalizer)
            validation_y = mean_targets[positions]
            audit_blocks.append({
                "block": block.key, "role": "validation", "train_trials": 0,
                "validation_trials": len(raw), "validation_rows": len(positions),
                "normalizer_source": "all unlabeled trials in the target session",
                "validation_row_policy": "four-repeat raw-count arithmetic mean, then target-session normalization",
            })
        else:
            require(len(part.validation_trials) == 0 and len(part.train_trials) == block.n_trials,
                    f"Training session leakage for {block.key}")
            raw = np.asarray(block.features[part.train_trials], dtype=np.float32)
            labels = np.asarray(block.trial_targets[part.train_trials], dtype=np.int64)
            normalizer = fit_session_normalizer(raw)
            normalizers[block.session] = normalizer
            singles = apply_session_normalizer(raw, normalizer)
            mean_targets, mean_raw = aggregate_by_target(raw, labels, method="mean")
            require(len(mean_targets) == n_classes, f"Training repeat grouping failed for {block.key}")
            means = apply_session_normalizer(mean_raw, normalizer)
            feature_parts.extend((singles, means))
            target_parts.extend((labels, mean_targets))
            single_rows += len(singles)
            mean_rows += len(means)
            audit_blocks.append({
                "block": block.key, "role": "train", "train_trials": len(raw),
                "single_trial_rows": len(singles), "four_repeat_mean_rows": len(means),
                "normalizer_source": "this fold's train_trials in this session only",
            })
    require(validation_x is not None and len(feature_parts) == 6, "Task2 fold preparation failed")
    train_x = np.concatenate(feature_parts).astype(np.float32, copy=False)
    train_y = np.concatenate(target_parts).astype(np.int64, copy=False)
    require(len(train_x) == single_rows + mean_rows, "Task2 row count mismatch")
    require(np.array_equal(validation_y, subset), "Task2 validation rows are not subset-sorted")
    return {
        "task": "task2", "mouse": mouse, "fold": fold, "n_classes": n_classes,
        "train_x": train_x, "train_y": train_y,
        "validation": {fold: (validation_x, validation_y)},
        "normalizers": normalizers,
        "audit": {
            "task": "task2", "mouse": mouse, "fold": fold,
            "single_trial_rows": single_rows, "four_repeat_mean_rows": mean_rows,
            "train_rows": len(train_x), "validation_rows_per_block": len(subset),
            "blocks": audit_blocks,
        },
    }


def prepare_task3(mouse_blocks, split, mouse, fold, subset, *, n_classes=N_CLASSES):
    """Use only single trials, with each block normalizer fitted on its train trials."""
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
        feature_parts.append(singles)
        target_parts.append(labels)
        raw_validation = np.asarray(block.features[part.validation_trials], dtype=np.float32)
        validation_labels = np.asarray(block.trial_targets[part.validation_trials], dtype=np.int64)
        order = np.argsort(validation_labels, kind="stable")
        sorted_labels = validation_labels[order]
        positions = subset_positions(sorted_labels, subset)
        sorted_validation = apply_session_normalizer(raw_validation[order], normalizer)
        validation[block.key] = (sorted_validation[positions], sorted_labels[positions])
        audit_blocks.append({
            "block": block.key, "role": "train_and_validation", "train_trials": len(raw),
            "single_trial_rows": len(singles), "four_repeat_mean_rows": 0,
            "validation_trials": len(raw_validation), "validation_rows": len(subset),
            "normalizer_source": "this block's train_trials only",
            "validation_row_policy": "strictly one unmerged trial per target",
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
            "task": "task3", "mouse": mouse, "fold": fold,
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


def fit_fold(prepared, arm, arm_config, ocf_cache, *, device, max_epochs=MAX_EPOCHS,
             evaluation_epochs=EVALUATION_EPOCHS, batch_size=BATCH_SIZE, seed=TRAIN_SEED):
    """Fit one mouse/fold/arm without saving model parameters."""
    device = torch.device(device)
    train_x = prepared["train_x"]
    train_y = prepared["train_y"]
    n_classes = prepared["n_classes"]
    require(np.isfinite(train_x).all() and len(train_x) == len(train_y), "Invalid training matrix")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpikeClassifier(
        n_features=train_x.shape[1], n_classes=n_classes, hidden=arm_config["hidden"],
        dropout=arm_config["dropout"][prepared["mouse"]] if prepared["mouse"] in arm_config["dropout"] else 0.0,
        apply_log=False,
    )
    model.set_normalization(train_x)
    expected_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(model.feature_mean.cpu().numpy(), expected_mean)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arm_config["learning_rate"], weight_decay=arm_config["weight_decay"],
        fused=device.type == "cuda",
    )
    rows = []
    for epoch in range(1, max_epochs + 1):
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
                loss = F.cross_entropy(
                    logits.float(), batch_y, label_smoothing=arm_config["label_smoothing"]
                )
            require(bool(torch.isfinite(loss).item()), "Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * len(indices)
        train_loss = total_loss / len(train_y)
        if epoch in evaluation_epochs:
            for block, (validation_x, validation_y) in prepared["validation"].items():
                raw_logits = predict_logits(model, validation_x, batch_size=batch_size, device=device)
                metrics = evaluate_logits(raw_logits, validation_y, ocf_cache)
                rows.append({
                    "task": prepared["task"], "arm": arm, "mouse": prepared["mouse"],
                    "fold": prepared["fold"], "block": block, "epoch": epoch,
                    "n_rows": len(validation_y), **metrics,
                })
            print(
                f"EVAL task={prepared['task']} fold={prepared['fold']} mouse={prepared['mouse']} "
                f"arm={arm} epoch={epoch} train_loss={train_loss:.6f} " +
                json.dumps({name: float(np.mean([row[name] for row in rows if row['epoch'] == epoch]))
                            for name in METRICS}),
                flush=True,
            )
        else:
            print(
                f"TRAIN task={prepared['task']} fold={prepared['fold']} mouse={prepared['mouse']} "
                f"arm={arm} epoch={epoch} train_loss={train_loss:.6f}", flush=True,
            )
    expected_rows = len(prepared["validation"]) * len(evaluation_epochs)
    require(len(rows) == expected_rows, "Checkpoint evaluation rows are incomplete")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def aggregate_rows(block_rows, level):
    keys_by_level = {
        "mouse_fold": ("task", "arm", "mouse", "fold", "epoch"),
        "mouse": ("task", "arm", "mouse", "epoch"),
        "fold": ("task", "arm", "fold", "epoch"),
        "macro": ("task", "arm", "epoch"),
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
        row["n_source_cells"] = len(members)
        count_key = "n_rows" if level == "mouse_fold" else "n_evaluation_rows"
        row["n_evaluation_rows"] = sum(member[count_key] for member in members)
        for metric in METRICS:
            row[metric] = float(np.mean([member[metric] for member in members], dtype=np.float64))
        output.append(row)
    return output


def selection_key(row):
    return (-row["assigned_final_ocf"], -row["raw_recall_at_32"],
            -row["raw_top1_accuracy"], row["epoch"])


def select_epoch(rows, task, arm, *, expected_count=None):
    candidates = [row for row in rows if row["task"] == task and row["arm"] == arm]
    require(bool(candidates), f"Missing macro history for {task}/{arm}")
    if expected_count is not None:
        require(len(candidates) == expected_count, f"Incomplete macro history for {task}/{arm}")
    return dict(min(candidates, key=selection_key))


def find_aggregate(rows, task, arm, epoch, **identifiers):
    matches = [row for row in rows if row["task"] == task and row["arm"] == arm
               and row["epoch"] == epoch and all(row[key] == value for key, value in identifiers.items())]
    require(len(matches) == 1, "Expected one aggregate selection row")
    return matches[0]


def build_selection(results):
    macro = results["macro"]
    mouse_rows = results["mouse"]
    fold_rows = results["fold"]
    tasks = {}
    for task in TASKS:
        selected = {
            arm: select_epoch(macro, task, arm, expected_count=len(EVALUATION_EPOCHS)) for arm in ARMS
        }
        ordered = sorted(((arm, selected[arm]) for arm in ARMS), key=lambda item: selection_key(item[1]))
        winner = ordered[0][0]
        production = selected["production_recipe"]
        high = selected["high_capacity_noreg"]
        per_mouse = {}
        for mouse in MICE:
            p = find_aggregate(mouse_rows, task, "production_recipe", production["epoch"], mouse=mouse)
            h = find_aggregate(mouse_rows, task, "high_capacity_noreg", high["epoch"], mouse=mouse)
            per_mouse[mouse] = float(h["assigned_final_ocf"] - p["assigned_final_ocf"])
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        per_fold = {}
        for fold in fold_ids:
            p = find_aggregate(fold_rows, task, "production_recipe", production["epoch"], fold=fold)
            h = find_aggregate(fold_rows, task, "high_capacity_noreg", high["epoch"], fold=fold)
            per_fold[fold] = float(h["assigned_final_ocf"] - p["assigned_final_ocf"])
        macro_delta = float(high["assigned_final_ocf"] - production["assigned_final_ocf"])
        required_folds = 2
        nonnegative_folds = sum(delta >= 0 for delta in per_fold.values())
        gates = {
            "high_capacity_is_winner": winner == "high_capacity_noreg",
            "macro_assigned_ocf_delta_at_least_3": macro_delta >= 3.0,
            "both_mouse_deltas_nonnegative": all(delta >= 0 for delta in per_mouse.values()),
            "strict_majority_fold_deltas_nonnegative": (
                nonnegative_folds >= required_folds and nonnegative_folds > len(fold_ids) / 2
            ),
        }
        tasks[task] = {
            "selected_epochs": {arm: selected[arm]["epoch"] for arm in ARMS},
            "selected_macro": selected,
            "winner": winner,
            "winner_ordering": ["assigned_final_ocf", "raw_recall_at_32", "raw_top1_accuracy", "earlier_epoch"],
            "high_capacity_minus_production": {
                "macro_assigned_final_ocf": macro_delta,
                "per_mouse_assigned_final_ocf": per_mouse,
                "per_fold_assigned_final_ocf": per_fold,
            },
            "nonnegative_fold_count": nonnegative_folds,
            "required_nonnegative_fold_count": required_folds,
            "gates": gates,
            "advance": all(gates.values()),
            "next_step": "development gate passed; separate pristine/final authorization still required"
                         if all(gates.values()) else "stop; no alternate hyperparameter search",
        }
    return {"schema": 1, "experiment": OUT.name, "status": "development_selection_complete", "tasks": tasks}


def run_experiment():
    preregistration, preregistration_hash = validate_preregistration()
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
    arm_configs = preregistration["arms"]
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
                prepared = prepare(mouse_blocks, split, mouse, fold, subset)
                fold_audits.append(prepared["audit"])
                for arm in ARMS:
                    print(
                        f"START fit={fits_completed + 1}/20 task={task} fold={fold} mouse={mouse} arm={arm} "
                        f"train_rows={len(prepared['train_y'])}", flush=True,
                    )
                    block_rows.extend(fit_fold(prepared, arm, arm_configs[arm], ocf_cache, device=device))
                    fits_completed += 1
                    print(f"DONE fit={fits_completed}/20 elapsed_seconds={time.time() - started:.1f}", flush=True)
                prepared.clear()
                gc.collect()
    require(fits_completed == 20, "Not all preregistered fits completed")
    require(len(block_rows) == 440, "Incomplete block/checkpoint result grid")
    results = {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_results_complete",
        "data_policy": "public train only; development reuse; no Task2 session4 or Task3 seed42 final evaluation",
        "preregistration_sha256": preregistration_hash,
        "cache_manifest_sha256": manifest_hash,
        "run": {
            "device": torch.cuda.get_device_name(device),
            "bin_ms": BIN_MS,
            "training_seed": TRAIN_SEED,
            "evaluation_epochs": list(EVALUATION_EPOCHS),
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
    require(len(results["mouse_fold"]) == 200 and len(results["mouse"]) == 80,
            "Mouse aggregation grid mismatch")
    require(len(results["fold"]) == 100 and len(results["macro"]) == 40, "Fold/macro aggregation grid mismatch")
    write_json(RESULTS, results)
    selection = build_selection(results)
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


def smoke_test():
    """Synthetic-only checks; never load real blocks or compute real development metrics."""
    validate_preregistration()
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
    prepared2 = prepare_task2(
        [b for b in blocks if b.mouse == "mice1"], task2_split, "mice1", "session1", subset,
        n_classes=n_classes,
    )
    require(prepared2["audit"]["single_trial_rows"] == 3 * n_classes * repeats, "Task2 single-row policy failed")
    require(prepared2["audit"]["four_repeat_mean_rows"] == 3 * n_classes, "Task2 mean-row policy failed")
    require(len(prepared2["train_y"]) == 3 * n_classes * (repeats + 1), "Task2 total rows failed")
    target_block = next(b for b in blocks if b.key == "mice1/session1")
    raw_target = np.asarray(target_block.features, dtype=np.float32)
    mean_targets, mean_raw = aggregate_by_target(raw_target, target_block.trial_targets, method="mean")
    expected_validation = apply_session_normalizer(mean_raw, prepared2["normalizers"]["session1"])
    np.testing.assert_array_equal(prepared2["validation"]["session1"][0], expected_validation)
    task3_split = make_split(blocks, "task3", seed=20261126)
    prepared3 = prepare_task3(
        [b for b in blocks if b.mouse == "mice1"], task3_split, "mice1", "20261126", subset,
        n_classes=n_classes,
    )
    require(len(prepared3["train_y"]) == 4 * 3 * n_classes, "Task3 single-trial train policy failed")
    require(prepared3["audit"]["four_repeat_mean_rows"] == 0, "Task3 repeat aggregation is prohibited")
    require(all(len(x) == n_classes for x, _ in prepared3["validation"].values()), "Task3 validation rows failed")
    model = SpikeClassifier(prepared3["train_x"].shape[1], n_classes, hidden=8, dropout=0, apply_log=False)
    model.set_normalization(prepared3["train_x"])
    np.testing.assert_array_equal(
        model.feature_mean.numpy(), prepared3["train_x"].mean(axis=0, dtype=np.float64).astype(np.float32)
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
    config = {
        "hidden": 8, "dropout": {"mice1": 0.0, "mice2": 0.0},
        "label_smoothing": 0.0, "weight_decay": 0.0, "learning_rate": 3e-4,
    }
    rows2 = fit_fold(prepared2, "smoke", config, ocf, device="cpu", max_epochs=2,
                     evaluation_epochs=(1, 2), batch_size=16)
    rows3 = fit_fold(prepared3, "smoke", config, ocf, device="cpu", max_epochs=2,
                     evaluation_epochs=(1, 2), batch_size=16)
    require(len(rows2) == 2 and len(rows3) == 8, "Smoke checkpoint grid failed")
    for row in rows2 + rows3:
        require(all(np.isfinite(row[metric]) for metric in METRICS), "Nonfinite smoke metric")
    for level, count_key in (("mouse_fold", "n_rows"), ("mouse", "n_evaluation_rows"),
                             ("fold", "n_evaluation_rows"), ("macro", "n_evaluation_rows")):
        aggregated = aggregate_rows(rows2 + rows3, level)
        require(all(row["n_evaluation_rows"] > 0 for row in aggregated),
                f"Aggregation count propagation failed at {level}: {count_key}")
    sample = {"task": "task2", "arm": "a", "epoch": 1,
              "raw_top1_accuracy": 0.1, "raw_recall_at_32": 0.2,
              "raw_top1_official_ocf": 10.0, "assigned_final_ocf": 20.0}
    later = {**sample, "epoch": 2}
    require(select_epoch([later, sample], "task2", "a")["epoch"] == 1, "Earlier-epoch tie break failed")
    better_recall = {**sample, "epoch": 3, "raw_recall_at_32": 0.3}
    require(select_epoch([sample, better_recall], "task2", "a")["epoch"] == 3, "Recall tie break failed")
    better_top1 = {**sample, "epoch": 4, "raw_top1_accuracy": 0.2}
    require(select_epoch([sample, better_top1], "task2", "a")["epoch"] == 4, "Top1 tie break failed")
    synthetic_results = {"macro": [], "mouse": [], "fold": []}
    for task in TASKS:
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        for arm in ARMS:
            assigned = 50.0 if arm == "production_recipe" else 53.1
            for epoch in EVALUATION_EPOCHS:
                common = {
                    "task": task, "arm": arm, "epoch": epoch, "n_rows": 256,
                    "raw_top1_accuracy": 0.1, "raw_recall_at_32": 0.2,
                    "raw_top1_official_ocf": 20.0, "assigned_final_ocf": assigned,
                }
                synthetic_results["macro"].append(dict(common))
                for mouse in MICE:
                    synthetic_results["mouse"].append({**common, "mouse": mouse})
                for fold in fold_ids:
                    synthetic_results["fold"].append({**common, "fold": fold})
    synthetic_selection = build_selection(synthetic_results)
    require(all(synthetic_selection["tasks"][task]["advance"] for task in TASKS),
            "Positive advancement-gate smoke case failed")
    failed_results = json.loads(json.dumps(synthetic_results))
    for row in failed_results["mouse"]:
        if row["arm"] == "high_capacity_noreg" and row["mouse"] == "mice2":
            row["assigned_final_ocf"] = 49.0
    failed_selection = build_selection(failed_results)
    require(all(not failed_selection["tasks"][task]["advance"] for task in TASKS),
            "Negative per-mouse advancement-gate smoke case failed")
    print("SMOKE OK: normalization isolation, Task2/Task3 rows, raw logits, Hungarian, official OCF, selection and gates",
          flush=True)


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
