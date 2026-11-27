"""Preregistered development-only multiresolution logit ensemble (experiment 53)."""

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.features import FeatureBlock
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
REFERENCE48 = ROOT / "test/06_fusion_transduction/48_high_capacity_generalization"

MICE = ("mice1", "mice2")
BLOCKS = tuple(f"{mouse}/session{session}" for mouse in MICE for session in range(1, 5))
TASKS = ("task2", "task3")
VIEWS = (10, 25, 50)
TEMPORAL_BINS = {10: 25, 25: 10, 50: 5}
WINDOW_MS = 250
N_CLASSES = 2748
ARM = "multires_ensemble"
CONSISTENCY_VIEW = 25
CONSISTENCY_ATOL = 1e-5
SUBSET_SIZE = 256
TASK2_FOLDS = ("session1", "session2", "session3")
TASK3_FOLDS = ("20261126", "20261127")


def load_reference_module():
    """Import experiment 48's run module read-only (its directory name is not an identifier)."""
    path = REFERENCE48 / "run.py"
    spec = importlib.util.spec_from_file_location("experiment48_reference_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = load_reference_module()
prepare_task2 = REF.prepare_task2
prepare_task3 = REF.prepare_task3
OcfPairCache = REF.OcfPairCache
evaluate_logits = REF.evaluate_logits
aggregate_rows = REF.aggregate_rows
select_epoch = REF.select_epoch
find_aggregate = REF.find_aggregate
METRICS = REF.METRICS
EVALUATION_EPOCHS = REF.EVALUATION_EPOCHS
MAX_EPOCHS = REF.MAX_EPOCHS
BATCH_SIZE = REF.BATCH_SIZE
TRAIN_SEED = REF.TRAIN_SEED
GRADIENT_CLIP_NORM = REF.GRADIENT_CLIP_NORM

PRODUCTION_CONFIG = {
    "hidden": 512,
    "dropout": {"mice1": 0.2, "mice2": 0.35},
    "label_smoothing": 0.05,
    "weight_decay": 0.001,
    "learning_rate": 3e-4,
}


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


def view_arm(view):
    return f"view{view}"


def fold_ids(task):
    return TASK2_FOLDS if task == "task2" else TASK3_FOLDS


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
        data["loader"]
        == "src.preprocessing.load_cached_blocks(cache, split='train', bin_ms=b) for b in (10, 25, 50)",
        "Cache loader contract changed",
    )
    require(data["cache_access"] == "read-only; never rebuild the cache", "Cache must be read-only")
    require(data["public_partition"] == "train only", "Only public train data are permitted")
    require(data["views"] == list(VIEWS), "View set changed")
    require(data["window_ms"] == WINDOW_MS, "Window changed")
    require(
        {int(key): int(count) for key, count in data["temporal_bins"].items()} == TEMPORAL_BINS,
        "Temporal bin contract changed",
    )
    require(data["mice"] == list(MICE) and data["blocks"] == list(BLOCKS), "Block contract changed")
    require(data["n_classes"] == N_CLASSES and data["subset_size"] == SUBSET_SIZE, "Class/subset contract changed")
    require(
        data["development_picture_subset"]
        == "np.sort(np.random.default_rng(20261128).permutation(2748)[:256])",
        "Development picture subset changed",
    )
    require(
        data["same_subset_for_every_view_block_fold_mouse_and_epoch"] is True,
        "The subset must be shared by every view, block, fold, mouse and epoch",
    )
    folds = value["folds"]
    require(folds["task2"]["fold_ids"] == list(TASK2_FOLDS), "Task2 folds changed")
    require(folds["task3"]["fold_ids"] == list(TASK3_FOLDS), "Task3 folds changed")
    require("session4" not in folds["task2"]["fold_ids"], "Task2 session4 validation is prohibited")
    require("42" not in folds["task3"]["fold_ids"], "Task3 seed42 evaluation is prohibited")
    arm = value["arm"]
    require(arm["name"] == ARM and arm["arms_in_this_experiment"] == [ARM], "Arm contract changed")
    require(arm["views"] == list(VIEWS), "Arm views changed")
    require(arm["training_epochs"] == MAX_EPOCHS, "Arm epoch count changed")
    require(arm["view_weight_search"] is False, "View weight search is prohibited")
    require(arm["extra_hyperparameter_search"] is False, "Hyperparameter search is prohibited")
    require(
        arm["fusion_weighting"] == "equal arithmetic mean of per-view row z-scores; fixed, never fitted",
        "Fusion weighting changed",
    )
    training = value["model_and_training"]
    require(training["class"] == "src.neural.SpikeClassifier", "Model class changed")
    require(training["evaluation_epochs"] == list(EVALUATION_EPOCHS), "Evaluation checkpoints changed")
    for token in (
        "hidden=512",
        "dropout mice1=0.2/mice2=0.35",
        "label_smoothing=0.05",
        "weight_decay=0.001",
        "learning_rate=3e-4",
        "batch_size=256",
        "apply_log=False",
        "gradient_clip_norm=5.0",
        "training_seed=42",
        "max_epochs=20",
        "AdamW",
    ):
        require(token in training["per_view_configuration"], f"Training contract token changed: {token}")
    require("experiment 48 select_epoch unchanged" in training["epoch_selection"],
            "Epoch selection rule changed")
    require(
        "no cross-view epoch coupling and no per-fold selection" in training["epoch_selection"],
        "Epoch selection decoupling changed",
    )
    require("absolute tolerance 1e-5" in training["consistency_check"], "25ms tolerance changed")
    fusion = value["fusion"]
    require(
        fusion["normalization"] == "Row-wise z-score per view: (x - row_mean) / max(row_std, 1e-6).",
        "Fusion normalization changed",
    )
    require(
        fusion["combination"] == "Arithmetic mean of the three z-scored logit matrices.",
        "Fusion combination changed",
    )
    require("No top-32 restriction" in fusion["no_fusion"], "Fusion restriction changed")
    gate = value["advancement_gate"]
    require(gate["macro_minimum_absolute_gain"] == 3.0, "Advancement gate changed")
    require(gate["required_nonnegative_folds"] == {"task2": 2, "task3": 2}, "Fold gate changed")
    require(value["outputs"]["model_artifacts"] == "none retained", "Model artifacts must not be retained")
    require(
        value["outputs"]["overwrite_policy"] == "Refuse --run if either output already exists.",
        "Overwrite policy changed",
    )
    prohibited = value["prohibited"]
    require("Task2 session4 validation or final evaluation" in prohibited, "Task2 prohibition missing")
    require("Task3 split seed 42 evaluation" in prohibited, "Task3 prohibition missing")
    require(
        "validation labels for normalization, training, epoch selection inside a fold, or fusion weights"
        in prohibited,
        "Validation-label prohibition missing",
    )
    return value, hashlib.sha256(raw).hexdigest()


def validate_reference48():
    """Validate the read-only 48 comparator before any cache or CUDA access."""
    results_path = REFERENCE48 / "development_results.json"
    selection_path = REFERENCE48 / "selection.json"
    preregistration_path = REFERENCE48 / "preregistration.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    results_hash = file_sha256(results_path)
    selection_hash = file_sha256(selection_path)
    preregistration_hash = file_sha256(preregistration_path)
    require(
        selection.get("development_results_sha256") == results_hash,
        "Experiment 48 selection does not link to experiment 48 results",
    )
    require(
        selection.get("preregistration_sha256") == preregistration_hash,
        "Experiment 48 selection does not link to experiment 48 preregistration",
    )
    require(
        results.get("preregistration_sha256") == preregistration_hash,
        "Experiment 48 results do not link to experiment 48 preregistration",
    )
    require(selection.get("historical_or_final_metrics_used") is False,
            "Experiment 48 recorded prohibited metrics")
    require(results.get("experiment") == REFERENCE48.name, "Experiment 48 results are mislabelled")
    require(selection.get("experiment") == REFERENCE48.name, "Experiment 48 selection is mislabelled")
    subset = np.sort(np.random.default_rng(20261128).permutation(N_CLASSES)[:SUBSET_SIZE]).astype(np.int64)
    require(
        np.array_equal(np.asarray(results["development_picture_subset"], dtype=np.int64), subset),
        "Experiment 48 development subset changed",
    )
    require(
        (len(results["block"]), len(results["mouse_fold"]), len(results["mouse"]),
         len(results["fold"]), len(results["macro"])) == (440, 200, 80, 100, 40),
        "Experiment 48 aggregation grid changed",
    )
    production = [row for row in results["block"] if row["arm"] == "production_recipe"]
    require(len(production) == 220, "Experiment 48 production block rows changed")
    require(all(row["n_rows"] == SUBSET_SIZE for row in production), "Experiment 48 row count changed")
    keys = {(row["task"], row["fold"], row["mouse"], row["block"], row["epoch"]) for row in production}
    require(len(keys) == len(production), "Experiment 48 production keys are not unique")
    selected_epochs = {}
    for task, expected in (("task2", 60), ("task3", 160)):
        require(sum(row["task"] == task for row in production) == expected,
                f"Experiment 48 production row count changed for {task}")
        selected = select_epoch(results["macro"], task, "production_recipe",
                                expected_count=len(EVALUATION_EPOCHS))
        require(selection["tasks"][task]["selected_epochs"]["production_recipe"] == selected["epoch"],
                f"Experiment 48 selected epoch linkage changed for {task}")
        require(selection["tasks"][task]["selected_macro"]["production_recipe"]["epoch"] == selected["epoch"],
                f"Experiment 48 selected macro linkage changed for {task}")
        selected_epochs[task] = int(selected["epoch"])
    reference = {
        "development_results_sha256": results_hash,
        "selection_sha256": selection_hash,
        "preregistration_sha256": preregistration_hash,
        "production_selected_epochs": selected_epochs,
    }
    return results, reference


def preflight_cache():
    """Check read-only cache integrity for all three views without rebuilding anything."""
    manifest_path = CACHE / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Shared cache is not ready: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == CACHE_SCHEMA, "Unsupported cache schema")
    require(Path(manifest["source_root"]).resolve() == DATA.resolve(), "Cache source root mismatch")
    require(manifest.get("window_ms") == WINDOW_MS, "Cache window changed")
    require(sorted(int(value) for value in manifest["bin_ms"]) == list(VIEWS), "Cache bin set changed")
    require(manifest.get("image_shape") == [N_CLASSES, 64, 64], "Gallery image shape mismatch")
    require((CACHE / "pictures.npy").is_file(), "Cached gallery images are missing")
    records = [record for record in manifest["blocks"] if record.get("split") == "train"]
    by_key = {f"{record['mouse']}/{record['session']}": record for record in records}
    require(len(records) == len(BLOCKS) and set(by_key) == set(BLOCKS), "Train cache blocks are incomplete")
    for key in BLOCKS:
        record = by_key[key]
        require(record["task"] == "train", f"Unexpected cached task for {key}")
        require(record["n_samples"] == 10992 and record["n_trials"] == 10992, f"Trial count changed for {key}")
        for view in VIEWS:
            count = record["counts"][str(view)]
            require(count["dtype"] == "uint16", f"Count dtype changed for {key}/{view}")
            require(count["shape"] == [10992, record["n_units"] * TEMPORAL_BINS[view]],
                    f"Feature dimension changed for {key}/{view}")
            directory = CACHE / "blocks" / record["id"]
            count_path = directory / count["file"]
            require(count_path.is_file() and (directory / "index.npz").is_file(), f"Cache files missing for {key}")
            require(file_sha256(count_path) == count["sha256"], f"Cache hash mismatch for {key}/{view}")
    return manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def fit_fold_logits(prepared, arm, arm_config, ocf_cache, *, device, max_epochs=MAX_EPOCHS,
                    evaluation_epochs=EVALUATION_EPOCHS, batch_size=BATCH_SIZE, seed=TRAIN_SEED):
    """Experiment 48 fit_fold, unchanged training, plus retained raw validation logits."""
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
    logits = {}
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
                output = model(batch_x)
                loss = F.cross_entropy(
                    output.float(), batch_y, label_smoothing=arm_config["label_smoothing"]
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
                require(
                    raw_logits.shape == (len(validation_y), n_classes)
                    and raw_logits.dtype == np.float32 and np.isfinite(raw_logits).all(),
                    "Unexpected validation logit matrix",
                )
                logits[(block, epoch)] = np.array(raw_logits, dtype=np.float32, copy=True)
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
    return rows, logits


def check_consistency_25ms(rows25, reference_results):
    """Compare every 25ms block row with experiment 48 production_recipe at atol=1e-5."""
    expected = {
        (row["task"], row["fold"], row["mouse"], row["block"], row["epoch"]): row
        for row in reference_results["block"] if row["arm"] == "production_recipe"
    }
    observed = {
        (row["task"], row["fold"], row["mouse"], row["block"], row["epoch"]): row for row in rows25
    }
    require(set(observed) == set(expected),
            "The 25ms view does not cover the experiment 48 production grid exactly")
    maxima = {metric: 0.0 for metric in METRICS}
    for key in sorted(expected):
        reference = expected[key]
        current = observed[key]
        require(current["n_rows"] == reference["n_rows"], f"25ms row count changed for {key}")
        for metric in METRICS:
            difference = abs(float(current[metric]) - float(reference[metric]))
            require(np.isfinite(difference), f"Nonfinite 25ms consistency difference for {key}/{metric}")
            maxima[metric] = max(maxima[metric], difference)
    overall = max(maxima.values())
    report = {
        "view": CONSISTENCY_VIEW,
        "rows_compared": len(expected),
        "absolute_tolerance": CONSISTENCY_ATOL,
        "max_abs_difference_by_metric": maxima,
        "max_abs_difference": overall,
        "passed": bool(overall <= CONSISTENCY_ATOL),
    }
    require(report["passed"],
            "25ms consistency check failed: max abs difference "
            f"{overall:.3e} exceeds atol {CONSISTENCY_ATOL:.0e} "
            f"({json.dumps(maxima)}); aborting before fusion")
    return report


def row_zscore(matrix):
    x = np.asarray(matrix, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] >= 2 and np.isfinite(x).all(), "Row z-score needs a finite 2D matrix")
    mean = x.mean(axis=1, keepdims=True)
    scale = np.maximum(x.std(axis=1, keepdims=True), 1e-6)
    return (x - mean) / scale


def fuse_views(matrices):
    """Equal arithmetic mean of the three per-view row z-scores; never fitted."""
    require(len(matrices) == len(VIEWS), "Fusion requires exactly the three preregistered views")
    shapes = {np.shape(matrix) for matrix in matrices}
    require(len(shapes) == 1, "Fused view matrices must share one shape")
    zscored = [row_zscore(matrix) for matrix in matrices]
    return np.mean(np.stack(zscored, axis=0), axis=0, dtype=np.float64)


def gate_result(task, fused, production):
    folds = fold_ids(task)
    required = 2
    macro_delta = float(fused["macro"]["assigned_final_ocf"] - production["macro"]["assigned_final_ocf"])
    recall_delta = float(fused["macro"]["raw_recall_at_32"] - production["macro"]["raw_recall_at_32"])
    per_mouse = {
        mouse: float(fused["mouse"][mouse]["assigned_final_ocf"] - production["mouse"][mouse]["assigned_final_ocf"])
        for mouse in MICE
    }
    per_fold = {
        fold: float(fused["fold"][fold]["assigned_final_ocf"] - production["fold"][fold]["assigned_final_ocf"])
        for fold in folds
    }
    nonnegative = sum(delta >= 0 for delta in per_fold.values())
    gates = {
        "macro_assigned_final_ocf_delta_at_least_3": bool(macro_delta >= 3.0),
        "both_mouse_deltas_nonnegative": bool(all(delta >= 0 for delta in per_mouse.values())),
        "required_nonnegative_folds": bool(nonnegative >= required),
    }
    return {
        "macro_assigned_final_ocf_delta": macro_delta,
        "macro_raw_recall_at_32_delta": recall_delta,
        "per_mouse_assigned_final_ocf_delta": per_mouse,
        "per_fold_assigned_final_ocf_delta": per_fold,
        "nonnegative_fold_count": int(nonnegative),
        "required_nonnegative_fold_count": int(required),
        "gates": gates,
        "advance": bool(all(gates.values())),
    }


def collect_aggregates(aggregates, task, arm, epoch):
    return {
        "macro": find_aggregate(aggregates["macro"], task, arm, epoch),
        "mouse": {mouse: find_aggregate(aggregates["mouse"], task, arm, epoch, mouse=mouse) for mouse in MICE},
        "fold": {fold: find_aggregate(aggregates["fold"], task, arm, epoch, fold=fold) for fold in fold_ids(task)},
    }


def run_experiment():
    preregistration, preregistration_hash = validate_preregistration()
    require(not RESULTS.exists() and not SELECTION.exists(),
            "Development outputs already exist; refusing an unpreregistered repeated run")
    require(torch.cuda.is_available(), "The preregistered bf16 CUDA experiment requires CUDA")
    reference_results, reference = validate_reference48()
    manifest, manifest_hash = preflight_cache()
    implementation_hash = file_sha256(Path(__file__).resolve())
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.backends.cuda.matmul.allow_tf32 is False, "TF32 must stay disabled")
    subset = np.sort(np.random.default_rng(20261128).permutation(N_CLASSES)[:SUBSET_SIZE]).astype(np.int64)
    require(len(subset) == SUBSET_SIZE and len(np.unique(subset)) == SUBSET_SIZE, "Invalid development subset")
    require(np.array_equal(subset, np.asarray(reference_results["development_picture_subset"], dtype=np.int64)),
            "Development subset differs from experiment 48")

    print("Loading read-only public train blocks and gallery images", flush=True)
    images = (np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False) >= 128).astype(np.float32)
    require(images.shape == (N_CLASSES, 64, 64), "Gallery image cache mismatch")
    ocf_cache = OcfPairCache(images, device="cuda", batch_size=32)

    started = time.time()
    fits_completed = 0
    block_rows = []
    view_macro_history = []
    view_selection = {}
    view_fold_audits = []
    retained_logits = {}
    targets = {}
    consistency_report = None
    total_fits = len(VIEWS) * (len(TASK2_FOLDS) * len(MICE) + len(TASK3_FOLDS) * len(MICE))

    for view in VIEWS:
        blocks = load_cached_blocks(CACHE, split="train", bin_ms=view)
        require(tuple(block.key for block in blocks) == tuple(sorted(BLOCKS)), "Loaded block set mismatch")
        for block in blocks:
            require(
                block.features.shape == (10992, block.n_units * TEMPORAL_BINS[view]),
                f"Cached feature shape mismatch for {block.key} at {view}ms",
            )
        view_rows = []
        view_logits = {}
        for task in TASKS:
            for fold in fold_ids(task):
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
                    view_fold_audits.append({
                        "view": view,
                        "feature_dimension": int(prepared["train_x"].shape[1]),
                        "temporal_bins": TEMPORAL_BINS[view],
                        **prepared["audit"],
                    })
                    for block, (_, validation_y) in prepared["validation"].items():
                        key = (task, fold, mouse, block)
                        if key not in targets:
                            targets[key] = np.array(validation_y, dtype=np.int64, copy=True)
                        else:
                            require(np.array_equal(targets[key], validation_y),
                                    f"Validation targets changed for {key}")
                    print(
                        f"START fit={fits_completed + 1}/{total_fits} view={view}ms task={task} fold={fold} "
                        f"mouse={mouse} features={prepared['train_x'].shape[1]} "
                        f"train_rows={len(prepared['train_y'])}",
                        flush=True,
                    )
                    rows, logits = fit_fold_logits(
                        prepared, view_arm(view), PRODUCTION_CONFIG, ocf_cache, device=device,
                    )
                    for (block, epoch), matrix in logits.items():
                        view_logits[(task, fold, mouse, block, epoch)] = matrix
                    view_rows.extend(rows)
                    fits_completed += 1
                    print(f"DONE fit={fits_completed}/{total_fits} "
                          f"elapsed_seconds={time.time() - started:.1f}", flush=True)
                    prepared.clear()
                    del logits
                    gc.collect()
        macro_rows = aggregate_rows(view_rows, "macro")
        view_macro_history.extend(macro_rows)
        for task in TASKS:
            selected = select_epoch(macro_rows, task, view_arm(view), expected_count=len(EVALUATION_EPOCHS))
            view_selection[(view, task)] = dict(selected)
            retained = {
                (fold, mouse, block): matrix
                for (task_key, fold, mouse, block, epoch), matrix in view_logits.items()
                if task_key == task and epoch == selected["epoch"]
            }
            require(len(retained) == len(fold_ids(task)) * len(MICE) * (1 if task == "task2" else 4),
                    f"Incomplete retained logits for view {view} {task}")
            retained_logits[(view, task)] = retained
            print(
                f"SELECT view={view}ms task={task} epoch={selected['epoch']} "
                f"macro_assigned_final_ocf={selected['assigned_final_ocf']:.6f} "
                f"macro_recall_at_32={selected['raw_recall_at_32']:.6f}",
                flush=True,
            )
        block_rows.extend(view_rows)
        require(len(block_rows) == (VIEWS.index(view) + 1) * 220, "Incomplete view block grid")
        if view == CONSISTENCY_VIEW:
            consistency_report = check_consistency_25ms(view_rows, reference_results)
            print(
                f"CONSISTENCY view=25ms rows={consistency_report['rows_compared']} "
                f"max_abs_diff={consistency_report['max_abs_difference']:.3e} passed=true",
                flush=True,
            )
        del view_rows, view_logits, macro_rows
        gc.collect()

    require(fits_completed == total_fits, "Not all preregistered fits completed")
    require(total_fits == 30, "Unexpected preregistered fit count")
    require(consistency_report is not None and consistency_report["passed"],
            "Missing successful 25ms consistency report before fusion")
    require(len(block_rows) == 660, "Incomplete block/checkpoint result grid")

    fused_rows = []
    for task in TASKS:
        for fold in fold_ids(task):
            for mouse in MICE:
                mouse_block_keys = ([fold] if task == "task2"
                                    else [block for block in BLOCKS if block.startswith(f"{mouse}/")])
                require(len(mouse_block_keys) == (1 if task == "task2" else 4),
                        f"Unexpected fused block count for {task}/{mouse}")
                for block in mouse_block_keys:
                    matrices = [retained_logits[(view, task)][(fold, mouse, block)] for view in VIEWS]
                    fused_matrix = fuse_views(matrices)
                    require(fused_matrix.shape == (SUBSET_SIZE, N_CLASSES), "Unexpected fused matrix shape")
                    validation_y = targets[(task, fold, mouse, block)]
                    metrics = evaluate_logits(fused_matrix, validation_y, ocf_cache)
                    fused_rows.append({
                        "task": task, "arm": ARM, "mouse": mouse, "fold": fold,
                        "block": block, "epoch": 0, "n_rows": len(validation_y), **metrics,
                    })
        print(f"FUSED task={task} rows={len(fused_rows)}", flush=True)
    require(len(fused_rows) == 22, "Incomplete fused result grid")

    fused_aggregates = {
        "block": fused_rows,
        "mouse_fold": aggregate_rows(fused_rows, "mouse_fold"),
        "mouse": aggregate_rows(fused_rows, "mouse"),
        "fold": aggregate_rows(fused_rows, "fold"),
        "macro": aggregate_rows(fused_rows, "macro"),
    }
    require(len(fused_aggregates["macro"]) == 2, "Fused macro grid mismatch")

    tasks_output = {}
    production_output = {}
    gate_output = {}
    for task in TASKS:
        epoch = reference["production_selected_epochs"][task]
        production = collect_aggregates(reference_results, task, "production_recipe", epoch)
        fused = collect_aggregates(fused_aggregates, task, ARM, 0)
        production_output[task] = production
        gate_output[task] = gate_result(task, fused, production)
        tasks_output[task] = {
            "selected_view_epochs": {str(view): view_selection[(view, task)]["epoch"] for view in VIEWS},
            "selected_view_macro": {
                str(view): {name: view_selection[(view, task)][name] for name in METRICS} for view in VIEWS
            },
            "multires_ensemble_macro": {name: fused["macro"][name] for name in METRICS},
            "production_recipe_macro": {name: production["macro"][name] for name in METRICS},
            "gate": gate_output[task],
        }

    overall_advance = bool(all(gate_output[task]["advance"] for task in TASKS))
    results = {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_results_complete",
        "data_policy": "public train only; development reuse; no Task2 session4 or Task3 seed42 final evaluation",
        "preregistration_sha256": preregistration_hash,
        "cache_manifest_sha256": manifest_hash,
        "implementation_sha256": implementation_hash,
        "reference48": reference,
        "run": {
            "device": torch.cuda.get_device_name(device),
            "views": list(VIEWS),
            "temporal_bins": {str(view): TEMPORAL_BINS[view] for view in VIEWS},
            "window_ms": WINDOW_MS,
            "training_seed": TRAIN_SEED,
            "evaluation_epochs": list(EVALUATION_EPOCHS),
            "fits_completed": fits_completed,
            "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
            "model_artifacts_retained": False,
            "ocf_pair_cache": {
                "storage": "memory only",
                "scoring_calls": ocf_cache.calls,
                "unique_pairs_scored": ocf_cache.scored_pairs,
            },
            "elapsed_seconds": float(time.time() - started),
        },
        "development_picture_subset": subset.tolist(),
        "consistency_25ms": consistency_report,
        "view_fold_audits": view_fold_audits,
        "view_selection": {
            task: {
                str(view): {
                    "epoch": view_selection[(view, task)]["epoch"],
                    "macro": {name: view_selection[(view, task)][name] for name in METRICS},
                }
                for view in VIEWS
            }
            for task in TASKS
        },
        "view_macro_history": view_macro_history,
        "block": block_rows,
        "fused": fused_aggregates,
        "production48": production_output,
        "deltas_and_gates": gate_output,
        "advance": overall_advance,
    }
    require(hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest() == preregistration_hash,
            "Preregistration changed during the run")
    write_json(RESULTS, results)

    selection = {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_selection_complete",
        "preregistration_sha256": preregistration_hash,
        "development_results_sha256": file_sha256(RESULTS),
        "implementation_sha256": implementation_hash,
        "reference48": reference,
        "arm": ARM,
        "fusion": {
            "views": list(VIEWS),
            "normalization": "row-wise z-score per view: (x - row_mean) / max(row_std, 1e-6)",
            "combination": "equal arithmetic mean of the three z-scored logit matrices",
            "weights_fitted": False,
            "candidate_restriction": "none; full 256x2748 matrices",
        },
        "epoch_ordering": ["assigned_final_ocf", "raw_recall_at_32", "raw_top1_accuracy", "earlier_epoch"],
        "consistency_25ms": consistency_report,
        "tasks": tasks_output,
        "overall_advance": overall_advance,
        "historical_or_final_metrics_used": False,
        "next_step": "development gate passed; separate pristine/final authorization still required"
                     if overall_advance else "stop; no alternate view set, fusion weight or epoch rule",
    }
    require(hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest() == preregistration_hash,
            "Preregistration changed during the run")
    require(file_sha256(Path(__file__).resolve()) == implementation_hash,
            "Implementation changed during the run")
    write_json(SELECTION, selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print(f"Wrote {RESULTS} and {SELECTION}; no model artifacts retained", flush=True)


def synthetic_blocks(n_classes=6, repeats=4, units=9, bin_ms=25):
    rng = np.random.default_rng(9182)
    blocks = []
    for mouse_index, mouse in enumerate(MICE):
        for session_index in range(1, 5):
            targets = np.tile(np.arange(n_classes, dtype=np.int64), repeats)
            targets = targets[rng.permutation(len(targets))]
            features = rng.poisson(0.4 + mouse_index, size=(len(targets), units)).astype(np.float32)
            features += (targets[:, None] == np.arange(features.shape[1])[None, :] % n_classes) * 3
            blocks.append(FeatureBlock(
                mouse=mouse, session=f"session{session_index}", split="train", task="train",
                sample_ids=np.arange(len(targets), dtype=np.int64),
                group_ptr=np.arange(len(targets) + 1, dtype=np.int64), targets=targets,
                features=features, n_units=units, bin_ms=bin_ms,
            ))
    return sorted(blocks, key=lambda block: (block.mouse, block.session))


def synthetic_gate_inputs(macro_ocf, mouse_ocf, fold_ocf, folds):
    fused = {
        "macro": {"assigned_final_ocf": macro_ocf, "raw_recall_at_32": 0.2},
        "mouse": {mouse: {"assigned_final_ocf": mouse_ocf[mouse], "raw_recall_at_32": 0.2} for mouse in MICE},
        "fold": {fold: {"assigned_final_ocf": fold_ocf[fold], "raw_recall_at_32": 0.2} for fold in folds},
    }
    production = {
        "macro": {"assigned_final_ocf": 50.0, "raw_recall_at_32": 0.2},
        "mouse": {mouse: {"assigned_final_ocf": 50.0, "raw_recall_at_32": 0.2} for mouse in MICE},
        "fold": {fold: {"assigned_final_ocf": 50.0, "raw_recall_at_32": 0.2} for fold in folds},
    }
    return fused, production


def smoke_test():
    """Synthetic-only checks; never read the real cache, real metrics or 48 outputs."""
    validate_preregistration()
    rng = np.random.default_rng(4242)

    matrix = rng.normal(size=(5, 9)).astype(np.float32)
    zscored = row_zscore(matrix)
    np.testing.assert_allclose(zscored.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(zscored.std(axis=1), 1.0, atol=1e-9)
    wide_matrix = matrix.astype(np.float64)
    np.testing.assert_allclose(
        zscored,
        (wide_matrix - wide_matrix.mean(axis=1, keepdims=True))
        / np.maximum(wide_matrix.std(axis=1, keepdims=True), 1e-6),
        atol=1e-12,
    )
    flat = np.full((2, 4), 3.25, dtype=np.float32)
    require(np.all(np.abs(row_zscore(flat)) <= 1e-12), "Constant rows must map to zero through the 1e-6 clamp")

    a, b, c = (rng.normal(size=(6, 11)).astype(np.float32) for _ in range(3))
    fused = fuse_views([a, b, c])
    np.testing.assert_allclose(fused, (row_zscore(a) + row_zscore(b) + row_zscore(c)) / 3.0, atol=1e-12)
    np.testing.assert_allclose(fuse_views([a, a, a]), row_zscore(a), atol=1e-12)
    np.testing.assert_allclose(fuse_views([b, b, b]), row_zscore(b), atol=1e-12)
    require(not np.allclose(fused, row_zscore(a)), "Synthetic fusion equality check is vacuous")
    for sample, scale in ((a, 7.0), (b, 0.5)):
        np.testing.assert_allclose(row_zscore(sample), row_zscore(sample * scale), atol=1e-5)
    require(np.isfinite(fused).all(), "Fused matrix must stay finite")

    smoke_classes = 6
    images = (rng.random((smoke_classes, 64, 64)) > 0.82).astype(np.uint8) * 255
    images[np.arange(smoke_classes), np.arange(smoke_classes) % 64, :] = 255
    ocf = OcfPairCache(images, device="cpu", batch_size=2)
    logits = np.eye(smoke_classes, dtype=np.float32) * 5.0 + rng.normal(size=(smoke_classes, smoke_classes))
    smoke_targets = np.arange(smoke_classes, dtype=np.int64)
    direct = evaluate_logits(logits, smoke_targets, ocf)
    single_view = evaluate_logits(fuse_views([logits, logits, logits]), smoke_targets, ocf)
    for name in METRICS:
        require(abs(direct[name] - single_view[name]) <= 1e-12,
                f"Single-view fusion must equal that view's own score for {name}")

    wide = np.full((3, 40), -10.0, dtype=np.float32)
    wide[0, 39] = 5.0
    wide[1, 5] = 5.0
    wide[2, 33] = 5.0
    wide_targets = np.array([39, 5, 33], dtype=np.int64)
    wide_images = (rng.random((40, 64, 64)) > 0.82).astype(np.uint8) * 255
    wide_images[np.arange(40), np.arange(40) % 64, :] = 255
    wide_metrics = evaluate_logits(wide, wide_targets, OcfPairCache(wide_images, device="cpu", batch_size=4))
    require(wide_metrics["raw_top1_accuracy"] == 1.0, "Full-candidate top-1 axis failed")
    require(wide_metrics["raw_recall_at_32"] == 1.0, "Full-candidate recall axis failed")
    assigned = unique_assignment(wide)
    require(len(np.unique(assigned)) == 3 and set(assigned.tolist()) == {39, 5, 33},
            "Assignment must use the full 2D candidate axis")
    wide_fused = fuse_views([wide, wide, wide])
    require(np.argmax(wide_fused, axis=1).tolist() == [39, 5, 33],
            "Fusion must preserve the full candidate axis")

    smoke_rows2 = []
    smoke_rows3 = []
    for view in VIEWS:
        for epoch in (1, 2):
            for fold in TASK2_FOLDS:
                for mouse in MICE:
                    smoke_rows2.append({
                        "task": "task2", "arm": view_arm(view), "mouse": mouse, "fold": fold,
                        "block": fold, "epoch": epoch, "n_rows": SUBSET_SIZE,
                        **{name: 0.1 for name in METRICS},
                    })
            for fold in TASK3_FOLDS:
                for mouse in MICE:
                    for block in BLOCKS:
                        if block.startswith(f"{mouse}/"):
                            smoke_rows3.append({
                                "task": "task3", "arm": view_arm(view), "mouse": mouse, "fold": fold,
                                "block": block, "epoch": epoch, "n_rows": SUBSET_SIZE,
                                **{name: 0.1 for name in METRICS},
                            })
    aggregated = aggregate_rows(smoke_rows2 + smoke_rows3, "mouse_fold")
    task2_cells = [row for row in aggregated if row["task"] == "task2"]
    task3_cells = [row for row in aggregated if row["task"] == "task3"]
    require(len(task2_cells) == 2 * len(VIEWS) * 3 * 2, "Task2 mouse_fold count propagation failed")
    require(len(task3_cells) == 2 * len(VIEWS) * 2 * 2, "Task3 mouse_fold count propagation failed")
    require(all(row["n_evaluation_rows"] == SUBSET_SIZE for row in task2_cells), "Task2 cell row count failed")
    require(all(row["n_evaluation_rows"] == 4 * SUBSET_SIZE for row in task3_cells), "Task3 cell row count failed")
    expected_totals = {
        "mouse": {"task2": 3 * 256, "task3": 2 * 4 * 256},
        "fold": {"task2": 2 * 256, "task3": 2 * 4 * 256},
        "macro": {"task2": 2 * 3 * 256, "task3": 2 * 2 * 4 * 256},
    }
    for level, totals in expected_totals.items():
        level_rows = aggregate_rows(smoke_rows2 + smoke_rows3, level)
        by_task = {
            row["task"]: row["n_evaluation_rows"]
            for row in level_rows if row["arm"] == view_arm(10) and row["epoch"] == 1
        }
        require(by_task == totals, f"Count propagation failed at {level}: {by_task} != {totals}")

    macro_history = aggregate_rows(smoke_rows2 + smoke_rows3, "macro")
    require(select_epoch(macro_history, "task2", view_arm(10))["epoch"] == 1, "Earlier-epoch tie break failed")
    boosted = [dict(row) for row in macro_history]
    for row in boosted:
        if row["task"] == "task2" and row["arm"] == view_arm(50) and row["epoch"] == 2:
            row["assigned_final_ocf"] += 5.0
    require(select_epoch(boosted, "task2", view_arm(50))["epoch"] == 2, "Higher-macro epoch selection failed")
    require(select_epoch(macro_history, "task3", view_arm(25))["epoch"] == 1, "Earlier-epoch task3 tie break failed")
    require(len({row["epoch"] for row in macro_history
                if row["task"] == "task2" and row["arm"] == view_arm(25)}) == 2,
            "Per-view epoch history must not be coupled across views")

    folds2 = TASK2_FOLDS
    positive = synthetic_gate_inputs(53.5, {"mice1": 50.4, "mice2": 50.2},
                                     {fold: 50.3 for fold in folds2}, folds2)
    positive_result = gate_result("task2", *positive)
    require(positive_result["advance"], "Positive advancement gate smoke case failed")
    require(positive_result["macro_raw_recall_at_32_delta"] == 0.0, "Recall delta smoke failed")

    macro_negative = synthetic_gate_inputs(52.9, {"mice1": 50.4, "mice2": 50.2},
                                          {fold: 50.3 for fold in folds2}, folds2)
    require(not gate_result("task2", *macro_negative)["advance"], "Macro-negative gate smoke case failed")

    mouse_negative = synthetic_gate_inputs(53.5, {"mice1": 50.4, "mice2": 49.9},
                                           {fold: 50.3 for fold in folds2}, folds2)
    require(not gate_result("task2", *mouse_negative)["advance"], "Mouse-negative gate smoke case failed")

    fold_negative = synthetic_gate_inputs(
        53.5, {"mice1": 50.4, "mice2": 50.2},
        {"session1": 49.5, "session2": 49.6, "session3": 50.4}, folds2,
    )
    fold_result = gate_result("task2", *fold_negative)
    require(not fold_result["advance"], "Fold-negative gate smoke case failed")
    require(fold_result["nonnegative_fold_count"] == 1, "Fold-negative count failed")

    folds3 = TASK3_FOLDS
    task3_fold_negative = synthetic_gate_inputs(
        53.5, {"mice1": 50.4, "mice2": 50.2},
        {"20261126": 49.9, "20261127": 50.3}, folds3,
    )
    require(not gate_result("task3", *task3_fold_negative)["advance"],
            "Task3 2/2 fold gate smoke case failed")
    task3_positive = synthetic_gate_inputs(
        53.2, {"mice1": 50.1, "mice2": 50.0},
        {"20261126": 50.0, "20261127": 50.2}, folds3,
    )
    require(gate_result("task3", *task3_positive)["advance"], "Task3 positive gate smoke case failed")

    blocks = synthetic_blocks()
    subset = np.arange(6, dtype=np.int64)
    task2_split = make_split(blocks, "task2", seed=42, validation_session="session1")
    prepared2 = prepare_task2(
        [block for block in blocks if block.mouse == "mice1"], task2_split, "mice1", "session1", subset,
        n_classes=6,
    )
    config = {
        "hidden": 8, "dropout": {"mice1": 0.0, "mice2": 0.0},
        "label_smoothing": 0.0, "weight_decay": 0.0, "learning_rate": 3e-4,
    }
    rows, logits = fit_fold_logits(prepared2, view_arm(10), config, ocf, device="cpu", max_epochs=2,
                                   evaluation_epochs=(1, 2), batch_size=16)
    require(len(rows) == 2 and len(logits) == 2, "Logit capture checkpoint grid failed")
    for (block_key, epoch), matrix in logits.items():
        require(matrix.shape == (6, 6) and matrix.dtype == np.float32, "Captured logit matrix contract failed")
        require(block_key == "session1" and epoch in (1, 2), "Captured logit key contract failed")
    require(all(np.isfinite(row[metric]) for row in rows for metric in METRICS), "Nonfinite smoke metric")
    print("SMOKE OK: row z-score, equal-weight fusion, single-view identity, full-candidate axis, "
          "count propagation, logit capture, per-view epoch selection and positive/negative gates", flush=True)


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
