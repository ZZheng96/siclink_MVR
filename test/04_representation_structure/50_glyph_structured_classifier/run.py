"""Preregistered development-only fixed-gallery DCT16 cosine classifier."""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.neural import SpikeClassifier, predict_logits
from src.preprocessing import file_sha256, load_cached_blocks, validate_split
from src.ridge import dct_encode
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
ARM = "structured_dct16"
N_CLASSES = 2748
DCT_CUTOFF = 16
DCT_DIMENSION = DCT_CUTOFF * DCT_CUTOFF
BIN_MS = 25
TRAIN_SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 20
EVALUATION_EPOCHS = (1, 2, 3, 4, 5, 7, 9, 12, 16, 20)
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.05
GRADIENT_CLIP_NORM = 5.0
HIDDEN = 512
DROPOUT = {"mice1": 0.2, "mice2": 0.35}
LOG_SCALE_MIN = math.log(1.0)
LOG_SCALE_MAX = math.log(100.0)
TASK2_FOLDS = ("session1", "session2", "session3")
TASK3_FOLDS = ("20261126", "20261127")
REQUIRED_NONNEGATIVE_FOLDS = {"task2": 2, "task3": 2}
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
    """Import only code, never experiment 48 metrics, and pin the reused implementation."""
    raw = RUN_48.read_bytes()
    source = raw.decode("utf-8")
    require(
        'count_key = "n_rows" if level == "mouse_fold" else "n_evaluation_rows"' in source
        and 'row["n_evaluation_rows"] = sum(member[count_key] for member in members)' in source,
        "Experiment 48 aggregate_rows is not the corrected n_evaluation_rows version",
    )
    spec = importlib.util.spec_from_file_location("siclink_experiment48_for50", RUN_48)
    require(spec is not None and spec.loader is not None, "Cannot load experiment 48 code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(raw).hexdigest()


EXP48, EXP48_CODE_SHA256 = load_experiment48_code()


def development_subset():
    return np.sort(np.random.default_rng(20261128).permutation(N_CLASSES)[:256]).astype(np.int64)


def array_sha256(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float32).tobytes()).hexdigest()


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
    require(data["same_subset_for_every_block_fold_mouse_and_epoch"] is True, "Subset must be shared")
    require("experiment 48 prepare_task2 and prepare_task3 unchanged" in data["row_preparation"],
            "Experiment 48 row preparation must be reused")
    folds = value["folds"]
    require(folds["task2"]["fold_ids"] == list(TASK2_FOLDS), "Task2 folds changed")
    require(folds["task3"]["fold_ids"] == list(TASK3_FOLDS), "Task3 folds changed")
    require("session4" not in folds["task2"]["fold_ids"], "Task2 session4 validation is prohibited")
    require("42" not in folds["task3"]["fold_ids"], "Task3 seed42 evaluation is prohibited")
    model = value["model"]
    require(model["trunk_class"] == "src.neural.SpikeClassifier", "Trunk class changed")
    require(model["hidden"] == HIDDEN and model["dropout"] == DROPOUT, "Trunk architecture changed")
    require(model["embedding_dimension"] == DCT_DIMENSION, "Embedding dimension changed")
    require(model["apply_log"] is False and model["free_2748_way_linear_head"] is False,
            "The free classifier head must be removed")
    head = value["structured_head"]
    require(head["arm"] == ARM and head["low_frequency_shape"] == [16, 16], "DCT16 head changed")
    require(head["target_dimension"] == DCT_DIMENSION, "DCT target dimension changed")
    require(head["initial_scale"] == 10.0 and head["scale_bounds"] == [1.0, 100.0],
            "Logit-scale initialization or bounds changed")
    require(head["embedding_normalization_epsilon"] == 1e-12, "Embedding normalization changed")
    require("registered buffer" in head["gallery_storage"] and "requires_grad=false" in head["gallery_storage"],
            "DCT gallery must be a fixed buffer")
    loss = value["loss"]
    require(loss["function"] == "torch.nn.functional.cross_entropy on float32 logits", "Loss changed")
    require(loss["classes_per_batch"] == N_CLASSES and loss["label_smoothing"] == LABEL_SMOOTHING,
            "Full-gallery cross entropy contract changed")
    require(loss["mse_image_regression"] is False, "MSE image regression is prohibited")
    training = value["shared_training_contract"]
    require(training["n_classes"] == N_CLASSES, "Class contract changed")
    require(training["optimizer"] == "AdamW" and training["batch_size"] == BATCH_SIZE,
            "Optimizer/batch contract changed")
    require(training["learning_rate"] == LEARNING_RATE and training["weight_decay"] == WEIGHT_DECAY,
            "Optimizer hyperparameters changed")
    require(training["training_seed"] == TRAIN_SEED and training["max_epochs"] == MAX_EPOCHS,
            "Seed/epoch contract changed")
    require(training["evaluation_epochs"] == list(EVALUATION_EPOCHS), "Evaluation checkpoints changed")
    require(training["gradient_clip_norm"] == GRADIENT_CLIP_NORM, "Gradient clipping changed")
    selection = value["selection"]
    require(selection["required_sources"] == [
        "test/06_fusion_transduction/48_high_capacity_generalization/development_results.json",
        "test/06_fusion_transduction/48_high_capacity_generalization/selection.json",
    ], "Experiment 48 source contract changed")
    require("before CUDA" in selection["missing_source"], "Missing sources must fail before CUDA/data loading")
    gate = value["advancement_gate"]
    require(gate["macro_minimum_absolute_gain"] == 3.0, "Advancement gate changed")
    require(gate["required_nonnegative_folds"] == REQUIRED_NONNEGATIVE_FOLDS, "Fold gate changed")
    require(value["outputs"]["model_artifacts"] == "none retained", "Model artifacts must not be retained")
    prohibited = value["prohibited"]
    require("Task2 session4 validation or final evaluation" in prohibited, "Task2 prohibition missing")
    require("Task3 split seed 42 evaluation" in prohibited, "Task3 prohibition missing")
    require("MSE image-regression loss" in prohibited, "MSE prohibition missing")
    require("dependence on experiment 49 outputs" in prohibited, "Experiment 49 independence missing")
    return value, hashlib.sha256(raw).hexdigest()


def build_dct_gallery(binary_images):
    """Return fixed row-normalized DCT16 targets; zero glyphs remain exact zero rows."""
    images = np.asarray(binary_images, dtype=np.float32)
    require(images.ndim == 3 and images.shape[1:] == (64, 64), "Gallery must be (N,64,64)")
    require(np.isfinite(images).all() and np.all((images == 0) | (images == 1)),
            "DCT gallery input must be finite and binary")
    coefficients = dct_encode(images, cutoff=DCT_CUTOFF).astype(np.float64)
    norms = np.linalg.norm(coefficients, axis=1, keepdims=True)
    gallery = np.divide(
        coefficients, norms, out=np.zeros_like(coefficients), where=norms > 0,
    ).astype(np.float32)
    require(gallery.shape == (len(images), DCT_DIMENSION) and np.isfinite(gallery).all(),
            "DCT gallery construction failed")
    require(np.all(gallery[norms[:, 0] == 0] == 0), "Zero-vector guard failed")
    return gallery


class StructuredGlyphClassifier(nn.Module):
    """Production SpikeClassifier trunk with a fixed full-gallery cosine head."""

    def __init__(self, n_features, gallery, *, hidden=HIDDEN, dropout=0.2):
        super().__init__()
        gallery_tensor = torch.as_tensor(np.asarray(gallery), dtype=torch.float32).clone()
        require(gallery_tensor.ndim == 2 and gallery_tensor.shape[1] == DCT_DIMENSION,
                "Structured gallery has an invalid shape")
        require(bool(torch.isfinite(gallery_tensor).all()), "Structured gallery must be finite")
        self.trunk = SpikeClassifier(
            n_features=n_features, n_classes=DCT_DIMENSION, hidden=hidden,
            dropout=dropout, apply_log=False,
        )
        self.register_buffer("dct_gallery", gallery_tensor)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def set_normalization(self, features):
        self.trunk.set_normalization(features)

    def forward(self, features):
        embedding = F.normalize(self.trunk(features).float(), p=2, dim=1, eps=1e-12)
        cosine_logits = embedding @ self.dct_gallery.T
        scale = self.log_scale.clamp(min=LOG_SCALE_MIN, max=LOG_SCALE_MAX).exp()
        return cosine_logits * scale


def fit_fold(prepared, gallery, ocf_cache, *, device, max_epochs=MAX_EPOCHS,
             evaluation_epochs=EVALUATION_EPOCHS, batch_size=BATCH_SIZE, seed=TRAIN_SEED,
             hidden=HIDDEN, dropout=None):
    """Fit one structured mouse/fold model without retaining model parameters."""
    device = torch.device(device)
    train_x = prepared["train_x"]
    train_y = prepared["train_y"]
    n_classes = prepared["n_classes"]
    require(np.isfinite(train_x).all() and len(train_x) == len(train_y), "Invalid training matrix")
    require(len(gallery) == n_classes and train_y.min() >= 0 and train_y.max() < n_classes,
            "Gallery/label contract mismatch")
    require(1 <= max_epochs <= MAX_EPOCHS and set(evaluation_epochs) <= set(range(1, max_epochs + 1)),
            "Invalid epoch grid")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dropout is None:
        dropout = DROPOUT[prepared["mouse"]]
    model = StructuredGlyphClassifier(
        train_x.shape[1], gallery, hidden=hidden, dropout=dropout,
    )
    model.set_normalization(train_x)
    expected_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(model.trunk.feature_mean.cpu().numpy(), expected_mean)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
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
                require(logits.shape == (len(indices), n_classes), "Full-gallery logits shape mismatch")
                loss = F.cross_entropy(logits.float(), batch_y, label_smoothing=LABEL_SMOOTHING)
            require(bool(torch.isfinite(loss).item()), "Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * len(indices)
        train_loss = total_loss / len(train_y)
        if epoch in evaluation_epochs:
            checkpoint_rows = []
            for block, (validation_x, validation_y) in prepared["validation"].items():
                raw_logits = predict_logits(model, validation_x, batch_size=batch_size, device=device)
                require(raw_logits.shape == (len(validation_y), n_classes),
                        "Full-gallery evaluation logits shape mismatch")
                metrics = EXP48.evaluate_logits(raw_logits, validation_y, ocf_cache)
                checkpoint_rows.append({
                    "task": prepared["task"], "arm": ARM, "mouse": prepared["mouse"],
                    "fold": prepared["fold"], "block": block, "epoch": epoch,
                    "n_rows": len(validation_y), **metrics,
                })
            rows.extend(checkpoint_rows)
            print(
                f"EVAL task={prepared['task']} fold={prepared['fold']} mouse={prepared['mouse']} "
                f"arm={ARM} epoch={epoch} train_loss={train_loss:.6f} " +
                json.dumps({name: float(np.mean([row[name] for row in checkpoint_rows]))
                            for name in METRICS}),
                flush=True,
            )
        else:
            print(
                f"TRAIN task={prepared['task']} fold={prepared['fold']} mouse={prepared['mouse']} "
                f"arm={ARM} epoch={epoch} train_loss={train_loss:.6f}", flush=True,
            )
    require(len(rows) == len(prepared["validation"]) * len(evaluation_epochs),
            "Checkpoint evaluation rows are incomplete")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def load_reference48(results_path=RESULTS_48, selection_path=SELECTION_48):
    """Load and validate the two required references before any cache/GPU operation."""
    results_path = Path(results_path)
    selection_path = Path(selection_path)
    missing = [str(path) for path in (results_path, selection_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Experiment 48 development_results.json and selection.json are required before --run; "
            f"missing: {missing}. Experiment 50 exits before cache, real-data, or CUDA access."
        )
    results_raw = results_path.read_bytes()
    selection_raw = selection_path.read_bytes()
    results = json.loads(results_raw)
    selection = json.loads(selection_raw)
    results_hash = hashlib.sha256(results_raw).hexdigest()
    selection_hash = hashlib.sha256(selection_raw).hexdigest()
    require(results.get("experiment") == "48_high_capacity_generalization",
            "Unexpected experiment 48 development results")
    require(selection.get("experiment") == "48_high_capacity_generalization",
            "Unexpected experiment 48 selection")
    require(results.get("status") == "development_results_complete", "Experiment 48 results are incomplete")
    require(selection.get("status") == "development_selection_complete", "Experiment 48 selection is incomplete")
    require(selection.get("development_results_sha256") == results_hash,
            "Experiment 48 selection/results hash mismatch")
    require(selection.get("preregistration_sha256") == results.get("preregistration_sha256"),
            "Experiment 48 preregistration hash mismatch")
    require(results.get("development_picture_subset") == development_subset().tolist(),
            "Experiment 48 development subset mismatch")
    selected_epochs = {}
    for task in TASKS:
        task_selection = selection["tasks"][task]
        epoch = task_selection["selected_epochs"][PRODUCTION]
        require(isinstance(epoch, int) and not isinstance(epoch, bool) and epoch in EVALUATION_EPOCHS,
                f"Invalid experiment 48 production epoch for {task}")
        selected_epochs[task] = epoch
        selected_macro = task_selection["selected_macro"][PRODUCTION]
        result_macro = EXP48.find_aggregate(results["macro"], task, PRODUCTION, epoch)
        require(selected_macro["epoch"] == epoch, "Experiment 48 selected macro epoch mismatch")
        for metric in METRICS:
            require(selected_macro[metric] == result_macro[metric],
                    f"Experiment 48 selected macro {metric} mismatch for {task}")
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        for level in EXPECTED_EVALUATION_ROWS[task]:
            if level == "mouse_fold":
                cells = [(mouse, fold) for mouse in MICE for fold in fold_ids]
                for mouse, fold in cells:
                    row = EXP48.find_aggregate(
                        results[level], task, PRODUCTION, epoch, mouse=mouse, fold=fold,
                    )
                    require(row["n_evaluation_rows"] == EXPECTED_EVALUATION_ROWS[task][level],
                            f"Experiment 48 mouse-fold row-count mismatch for {task}/{mouse}/{fold}")
            elif level == "mouse":
                for mouse in MICE:
                    row = EXP48.find_aggregate(results[level], task, PRODUCTION, epoch, mouse=mouse)
                    require(row["n_evaluation_rows"] == EXPECTED_EVALUATION_ROWS[task][level],
                            f"Experiment 48 mouse row-count mismatch for {task}/{mouse}")
            elif level == "fold":
                for fold in fold_ids:
                    row = EXP48.find_aggregate(results[level], task, PRODUCTION, epoch, fold=fold)
                    require(row["n_evaluation_rows"] == EXPECTED_EVALUATION_ROWS[task][level],
                            f"Experiment 48 fold row-count mismatch for {task}/{fold}")
            else:
                require(result_macro["n_evaluation_rows"] == EXPECTED_EVALUATION_ROWS[task][level],
                        f"Experiment 48 macro row-count mismatch for {task}")
    return {
        "results": results,
        "selected_epochs": selected_epochs,
        "results_sha256": results_hash,
        "selection_sha256": selection_hash,
    }


def build_selection(results, reference48):
    """Select structured epochs and compare against production at its own 48 epochs."""
    tasks = {}
    for task in TASKS:
        structured = EXP48.select_epoch(
            results["macro"], task, ARM, expected_count=len(EVALUATION_EPOCHS),
        )
        production_epoch = reference48["selected_epochs"][task]
        production = EXP48.find_aggregate(
            reference48["results"]["macro"], task, PRODUCTION, production_epoch,
        )
        per_mouse = {}
        for mouse in MICE:
            p = EXP48.find_aggregate(
                reference48["results"]["mouse"], task, PRODUCTION, production_epoch, mouse=mouse,
            )
            s = EXP48.find_aggregate(results["mouse"], task, ARM, structured["epoch"], mouse=mouse)
            per_mouse[mouse] = float(s["assigned_final_ocf"] - p["assigned_final_ocf"])
        fold_ids = TASK2_FOLDS if task == "task2" else TASK3_FOLDS
        per_fold = {}
        for fold in fold_ids:
            p = EXP48.find_aggregate(
                reference48["results"]["fold"], task, PRODUCTION, production_epoch, fold=fold,
            )
            s = EXP48.find_aggregate(results["fold"], task, ARM, structured["epoch"], fold=fold)
            per_fold[fold] = float(s["assigned_final_ocf"] - p["assigned_final_ocf"])
        macro_delta = float(structured["assigned_final_ocf"] - production["assigned_final_ocf"])
        nonnegative_folds = sum(delta >= 0 for delta in per_fold.values())
        required_folds = REQUIRED_NONNEGATIVE_FOLDS[task]
        gates = {
            "macro_assigned_ocf_delta_at_least_3": macro_delta >= 3.0,
            "both_mouse_deltas_nonnegative": all(delta >= 0 for delta in per_mouse.values()),
            "fold_deltas_nonnegative": nonnegative_folds >= required_folds,
        }
        tasks[task] = {
            "selected_epochs": {PRODUCTION: production_epoch, ARM: structured["epoch"]},
            "selected_macro": {PRODUCTION: production, ARM: structured},
            "structured_selection_ordering": [
                "assigned_final_ocf", "raw_recall_at_32", "raw_top1_accuracy", "earlier_epoch",
            ],
            "structured_minus_production": {
                "macro_assigned_final_ocf": macro_delta,
                "per_mouse_assigned_final_ocf": per_mouse,
                "per_fold_assigned_final_ocf": per_fold,
            },
            "nonnegative_fold_count": nonnegative_folds,
            "required_nonnegative_fold_count": required_folds,
            "gates": gates,
            "advance": all(gates.values()),
            "next_step": "development gate passed; separate pristine/final authorization still required"
                         if all(gates.values()) else "stop; no alternate structured head or hyperparameter search",
        }
    return {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_selection_complete",
        "development_reuse": "development reuse only; not pristine validation; not a Kaggle result",
        "reference48": {
            "development_results": "test/06_fusion_transduction/48_high_capacity_generalization/development_results.json",
            "selection": "test/06_fusion_transduction/48_high_capacity_generalization/selection.json",
            "development_results_sha256": reference48["results_sha256"],
            "selection_sha256": reference48["selection_sha256"],
            "run_code_sha256": EXP48_CODE_SHA256,
            "arm": PRODUCTION,
            "selected_epochs": reference48["selected_epochs"],
        },
        "development_reference_metrics_used": True,
        "final_or_heldout_metrics_used": False,
        "tasks": tasks,
        "advance": all(task["advance"] for task in tasks.values()),
    }


def run_experiment():
    preregistration, preregistration_hash = validate_preregistration()
    reference48 = load_reference48()
    require(not RESULTS.exists() and not SELECTION.exists(),
            "Development outputs already exist; refusing an unpreregistered repeated selection run")
    require(torch.cuda.is_available(), "The preregistered bf16 CUDA experiment requires CUDA")
    device = torch.device("cuda")
    manifest, manifest_hash = EXP48.preflight_cache()
    subset = development_subset()
    print("Loading read-only public train blocks and gallery images", flush=True)
    blocks = load_cached_blocks(ROOT / "test/.cache/shared_preprocessing_v1", split="train", bin_ms=BIN_MS)
    require(tuple(block.key for block in blocks) == tuple(sorted(BLOCKS)), "Loaded block set mismatch")
    images = (np.load(ROOT / "test/.cache/shared_preprocessing_v1/pictures.npy",
                      mmap_mode="r", allow_pickle=False) >= 128).astype(np.float32)
    require(images.shape == (N_CLASSES, 64, 64), "Gallery image cache mismatch")
    gallery = build_dct_gallery(images)
    require(gallery.shape == (N_CLASSES, DCT_DIMENSION), "Full DCT gallery shape mismatch")
    ocf_cache = EXP48.OcfPairCache(images, device="cuda", batch_size=32)
    block_rows = []
    fold_audits = []
    started = time.time()
    fits_completed = 0
    total_fits = (len(TASK2_FOLDS) + len(TASK3_FOLDS)) * len(MICE)
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
                prepare = EXP48.prepare_task2 if task == "task2" else EXP48.prepare_task3
                prepared = prepare(mouse_blocks, split, mouse, fold, subset)
                fold_audits.append(prepared["audit"])
                print(
                    f"START fit={fits_completed + 1}/{total_fits} task={task} fold={fold} mouse={mouse} "
                    f"arm={ARM} train_rows={len(prepared['train_y'])}", flush=True,
                )
                block_rows.extend(fit_fold(prepared, gallery, ocf_cache, device=device))
                fits_completed += 1
                print(f"DONE fit={fits_completed}/{total_fits} elapsed_seconds={time.time() - started:.1f}",
                      flush=True)
                prepared.clear()
                gc.collect()
    require(fits_completed == total_fits, "Not all preregistered structured fits completed")
    require(len(block_rows) == 220, "Incomplete structured block/checkpoint result grid")
    results = {
        "schema": 1,
        "experiment": OUT.name,
        "status": "development_results_complete",
        "data_policy": "public train only; development reuse; no Task2 session4 or Task3 seed42 final evaluation",
        "development_reuse": "development reuse only; not pristine validation; not a Kaggle result",
        "preregistration_sha256": preregistration_hash,
        "cache_manifest_sha256": manifest_hash,
        "reference48": {
            "development_results": "test/06_fusion_transduction/48_high_capacity_generalization/development_results.json",
            "selection": "test/06_fusion_transduction/48_high_capacity_generalization/selection.json",
            "development_results_sha256": reference48["results_sha256"],
            "selection_sha256": reference48["selection_sha256"],
            "run_code_sha256": EXP48_CODE_SHA256,
            "arm": PRODUCTION,
            "selected_epochs": reference48["selected_epochs"],
        },
        "run": {
            "device": torch.cuda.get_device_name(device),
            "implementation_sha256": file_sha256(__file__),
            "bin_ms": BIN_MS,
            "training_seed": TRAIN_SEED,
            "evaluation_epochs": list(EVALUATION_EPOCHS),
            "fits_completed": fits_completed,
            "model_artifacts_retained": False,
            "gallery": {
                "shape": list(gallery.shape),
                "sha256": array_sha256(gallery),
                "binarization_threshold": 128,
                "dct_cutoff": DCT_CUTOFF,
                "zero_rows": int(np.sum(np.linalg.norm(gallery, axis=1) == 0)),
            },
            "ocf_pair_cache": {
                "storage": "memory only", "scoring_calls": ocf_cache.calls,
                "unique_pairs_scored": ocf_cache.scored_pairs,
            },
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
    require(len(results["mouse_fold"]) == 100 and len(results["mouse"]) == 40,
            "Mouse aggregation grid mismatch")
    require(len(results["fold"]) == 50 and len(results["macro"]) == 20,
            "Fold/macro aggregation grid mismatch")
    for task in TASKS:
        for level in EXPECTED_EVALUATION_ROWS[task]:
            rows = [row for row in results[level] if row["task"] == task]
            require(all(row["n_evaluation_rows"] == EXPECTED_EVALUATION_ROWS[task][level] for row in rows),
                    f"Experiment 50 {level} n_evaluation_rows mismatch for {task}")
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
    """Metric-free-looking synthetic rows used solely to exercise selection logic."""
    reference_results = {"macro": [], "mouse": [], "fold": []}
    structured_results = {"macro": [], "mouse": [], "fold": []}
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
        reference_results["macro"].append(dict(production_common))
        for mouse in MICE:
            reference_results["mouse"].append({**production_common, "mouse": mouse})
        for fold in fold_ids:
            reference_results["fold"].append({**production_common, "fold": fold})
        for epoch in EVALUATION_EPOCHS:
            structured_common = {
                "task": task, "arm": ARM, "epoch": epoch,
                "raw_top1_accuracy": 0.1, "raw_recall_at_32": 0.2,
                "raw_top1_official_ocf": 20.0, "assigned_final_ocf": 53.1,
            }
            structured_results["macro"].append(dict(structured_common))
            for mouse in MICE:
                structured_results["mouse"].append({**structured_common, "mouse": mouse})
            for fold in fold_ids:
                structured_results["fold"].append({**structured_common, "fold": fold})
    reference = {
        "results": reference_results,
        "selected_epochs": selected_epochs,
        "results_sha256": "synthetic-reference-results",
        "selection_sha256": "synthetic-reference-selection",
    }
    return structured_results, reference


def smoke_test():
    """Synthetic-only checks; never read real metrics, real blocks, or real gallery images."""
    validate_preregistration()
    n_classes = 6
    repeats = 4
    subset = np.arange(n_classes, dtype=np.int64)
    blocks = EXP48.synthetic_blocks(n_classes=n_classes, repeats=repeats)
    mice1_blocks = [block for block in blocks if block.mouse == "mice1"]
    task2_split = make_split(blocks, "task2", seed=42, validation_session="session1")
    prepared2 = EXP48.prepare_task2(
        mice1_blocks, task2_split, "mice1", "session1", subset, n_classes=n_classes,
    )
    require(prepared2["audit"]["single_trial_rows"] == 3 * n_classes * repeats,
            "Task2 single-row policy changed")
    require(prepared2["audit"]["four_repeat_mean_rows"] == 3 * n_classes,
            "Task2 mean-row policy changed")
    task3_split = make_split(blocks, "task3", seed=20261126)
    prepared3 = EXP48.prepare_task3(
        mice1_blocks, task3_split, "mice1", "20261126", subset, n_classes=n_classes,
    )
    require(prepared3["audit"]["four_repeat_mean_rows"] == 0, "Task3 repeat aggregation is prohibited")
    require(len(prepared3["train_y"]) == 4 * 3 * n_classes, "Task3 single-trial rows changed")

    rng = np.random.default_rng(772)
    images = (rng.random((n_classes, 64, 64)) > 0.82).astype(np.float32)
    images[np.arange(n_classes), np.arange(n_classes) % 64, :] = 1
    ocf_images = images.copy()
    images[0] = 0
    gallery = build_dct_gallery(images)
    require(gallery.shape == (n_classes, DCT_DIMENSION), "DCT gallery shape failed")
    require(np.all(gallery[0] == 0), "Zero DCT row is not protected")
    nonzero_norms = np.linalg.norm(gallery[1:].astype(np.float64), axis=1)
    np.testing.assert_allclose(nonzero_norms, 1, atol=1e-6, rtol=0)
    original_gallery = gallery.copy()

    model = StructuredGlyphClassifier(
        prepared3["train_x"].shape[1], gallery, hidden=8, dropout=0,
    )
    model.set_normalization(prepared3["train_x"])
    require(model.trunk.network[-1].out_features == DCT_DIMENSION, "Trunk is not 256-dimensional")
    require(not model.dct_gallery.requires_grad, "DCT gallery unexpectedly requires gradients")
    require("dct_gallery" not in dict(model.named_parameters()), "DCT gallery is trainable")
    np.testing.assert_allclose(model.log_scale.exp().item(), 10.0, atol=1e-6, rtol=0)
    batch_x = torch.from_numpy(prepared3["train_x"][:8])
    batch_y = torch.from_numpy(prepared3["train_y"][:8])
    logits = model(batch_x)
    require(logits.shape == (8, n_classes), "Every batch must score every gallery class")
    require(float(logits.abs().max()) <= 10.001, "Initial cosine logit scale is invalid")
    loss = F.cross_entropy(logits.float(), batch_y, label_smoothing=LABEL_SMOOTHING)
    loss.backward()
    require(model.dct_gallery.grad is None, "Gradient flowed into the DCT gallery")
    require(model.log_scale.grad is not None, "Logit scale received no gradient")
    require(any(parameter.grad is not None for name, parameter in model.named_parameters()
                if name.startswith("trunk.")), "Trunk received no gradient")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer.step()
    np.testing.assert_array_equal(model.dct_gallery.numpy(), original_gallery)
    saved_log_scale = model.log_scale.detach().clone()
    with torch.no_grad():
        model.log_scale.fill_(math.log(1000.0))
    clamped_logits = model(batch_x)
    require(float(clamped_logits.abs().max()) <= 100.01, "Logit-scale upper clamp failed")
    with torch.no_grad():
        model.log_scale.copy_(saved_log_scale)

    ocf = EXP48.OcfPairCache(ocf_images, device="cpu", batch_size=2)
    rows2 = fit_fold(
        prepared2, gallery, ocf, device="cpu", max_epochs=1, evaluation_epochs=(1,),
        batch_size=16, hidden=8, dropout=0,
    )
    rows3 = fit_fold(
        prepared3, gallery, ocf, device="cpu", max_epochs=1, evaluation_epochs=(1,),
        batch_size=16, hidden=8, dropout=0,
    )
    require(len(rows2) == 1 and len(rows3) == 4, "Smoke checkpoint grid failed")
    for row in rows2 + rows3:
        require(all(np.isfinite(row[metric]) for metric in METRICS), "Nonfinite synthetic metric")
    for level in ("mouse_fold", "mouse", "fold", "macro"):
        aggregated = EXP48.aggregate_rows(rows2 + rows3, level)
        require(all(row["n_evaluation_rows"] > 0 for row in aggregated),
                f"Aggregation count propagation failed at {level}")
        for row in aggregated:
            expected = 6 if row["task"] == "task2" else 24
            require(row["n_evaluation_rows"] == expected,
                    f"Corrected aggregation count failed at {level}/{row['task']}")

    sample = {
        "task": "task2", "arm": ARM, "epoch": 1,
        "raw_top1_accuracy": 0.1, "raw_recall_at_32": 0.2,
        "raw_top1_official_ocf": 10.0, "assigned_final_ocf": 20.0,
    }
    require(EXP48.select_epoch([{**sample, "epoch": 2}, sample], "task2", ARM)["epoch"] == 1,
            "Earlier-epoch tie break failed")
    require(EXP48.select_epoch(
        [sample, {**sample, "epoch": 3, "raw_recall_at_32": 0.3}], "task2", ARM,
    )["epoch"] == 3, "Recall32 tie break failed")
    require(EXP48.select_epoch(
        [sample, {**sample, "epoch": 4, "raw_top1_accuracy": 0.2}], "task2", ARM,
    )["epoch"] == 4, "Top1 tie break failed")

    structured_results, reference = synthetic_gate_results()
    positive = build_selection(structured_results, reference)
    require(positive["advance"] and all(positive["tasks"][task]["advance"] for task in TASKS),
            "Positive advancement-gate case failed")
    require(all(positive["tasks"][task]["selected_epochs"][ARM] == 1 for task in TASKS),
            "Shared-epoch macro selection failed")
    require(all(positive["tasks"][task]["selected_epochs"][PRODUCTION] == 2 for task in TASKS),
            "Production's own experiment 48 epoch was not used")
    negative_mouse = json.loads(json.dumps(structured_results))
    for row in negative_mouse["mouse"]:
        if row["mouse"] == "mice2":
            row["assigned_final_ocf"] = 49.0
    require(not build_selection(negative_mouse, reference)["advance"],
            "Negative per-mouse gate case failed")
    negative_fold = json.loads(json.dumps(structured_results))
    for row in negative_fold["fold"]:
        if row["task"] == "task3" and row["fold"] == "20261127":
            row["assigned_final_ocf"] = 49.0
    failed_fold = build_selection(negative_fold, reference)
    require(failed_fold["tasks"]["task2"]["advance"] and not failed_fold["tasks"]["task3"]["advance"],
            "Task3 2/2 nonnegative-fold gate failed")
    print(
        "SMOKE OK: fixed normalized DCT16 gallery including zero rows, full-gallery CE gradients, "
        "scale clamp, experiment 48 rows/OCF/Hungarian/aggregation, epoch selection and all gates",
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
