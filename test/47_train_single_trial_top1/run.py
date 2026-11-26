"""Exact public-train single-trial top-1 capacity acceptance for two mice."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.neural import SpikeClassifier, predict_logits
from src.preprocessing import CACHE_SCHEMA, file_sha256, load_cached_blocks

OUT = Path(__file__).resolve().parent
PREREGISTRATION = OUT / "preregistration.json"
ARTIFACTS = OUT / "artifacts" / "models"
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
MICE = ("mice1", "mice2")
BLOCKS = tuple(f"{mouse}/session{session}" for mouse in MICE for session in range(1, 5))
N_CLASSES = 2748
TRIALS_PER_BLOCK = 10992
TRIALS_PER_MOUSE = 43968
TRIALS_TOTAL = 87936
REPEATS_PER_BLOCK = 4
BIN_MS = 25
HIDDEN = 1024
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
LABEL_SMOOTHING = 0.0
DROPOUT = 0.0
MAX_EPOCHS = 100
GRADIENT_CLIP_NORM = 5.0
SEED = 42


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_preregistration():
    value = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    require(value["experiment"] == OUT.name, "Preregistration experiment name mismatch")
    require(
        value["status"] == "preregistered_before_real_training_or_real_metrics",
        "Real fitting requires the frozen preregistration status",
    )
    data = value["data"]
    require(data["bin_ms"] == BIN_MS and data["cache_schema"] == CACHE_SCHEMA, "Cache contract changed")
    require(data["blocks"] == list(BLOCKS), "Training block contract changed")
    require(data["target_classes"] == N_CLASSES, "Target count contract changed")
    require(data["trials_per_block"] == TRIALS_PER_BLOCK, "Per-block trial contract changed")
    require(data["trials_per_mouse"] == TRIALS_PER_MOUSE, "Per-mouse trial contract changed")
    require(data["trials_total"] == TRIALS_TOTAL, "Overall trial contract changed")
    require(data["validation_fold"] is None, "This experiment must not introduce a validation fold")
    require(not data["split_manifest_required"], "This all-train experiment must not require a split")
    require(not data["validation_or_test_labels_used"], "Validation/test labels are forbidden")
    training = value["model_and_training"]
    require(training["hidden"] == HIDDEN, "Hidden width differs from preregistration")
    require(training["dropout"] == DROPOUT, "Dropout must be disabled")
    require(training["weight_decay"] == WEIGHT_DECAY, "Weight decay must be disabled")
    require(training["loss"] == "cross_entropy(label_smoothing=0.0)", "Label smoothing must be disabled")
    require(training["learning_rate"] == LEARNING_RATE, "Learning rate differs from preregistration")
    require(training["batch_size"] == BATCH_SIZE, "Batch size differs from preregistration")
    require(training["max_epochs_per_mouse"] == MAX_EPOCHS, "Epoch limit differs from preregistration")
    require(training["seed"] == SEED, "Seed differs from preregistration")
    return value


def preflight_cache():
    """Read-only readiness/integrity checks; never create or rebuild the cache."""
    manifest_path = CACHE / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Shared cache is not ready: {manifest_path}. "
            "Wait for the main cache task; experiment 47 never builds it."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == CACHE_SCHEMA, "Unsupported cache schema")
    require(
        Path(manifest.get("source_root", "")).resolve() == DATA.resolve(),
        "Cache source root does not match the public dataset",
    )
    require(BIN_MS in manifest.get("bin_ms", []), "The shared cache has no 25ms counts")
    records = [record for record in manifest.get("blocks", []) if record.get("split") == "train"]
    by_key = {f"{record['mouse']}/{record['session']}": record for record in records}
    require(set(by_key) == set(BLOCKS) and len(records) == len(BLOCKS), "Cache train blocks are incomplete")
    for key in BLOCKS:
        record = by_key[key]
        require(record.get("task") == "train", f"Unexpected task for {key}")
        require(record.get("n_trials") == TRIALS_PER_BLOCK, f"Wrong trial count for {key}")
        require(record.get("n_samples") == TRIALS_PER_BLOCK, f"Wrong sample count for {key}")
        count = record.get("counts", {}).get(str(BIN_MS))
        require(count is not None, f"Missing 25ms count record for {key}")
        require(count.get("dtype") == "uint16", f"Raw counts are not uint16 for {key}")
        require(
            len(count.get("shape", [])) == 2 and count["shape"][0] == TRIALS_PER_BLOCK,
            f"Wrong cached count shape for {key}",
        )
        directory = CACHE / "blocks" / record["id"]
        count_path = directory / count["file"]
        require(count_path.is_file(), f"Missing cached counts: {count_path}")
        require((directory / "index.npz").is_file(), f"Missing cached index for {key}")
        require(file_sha256(count_path) == count["sha256"], f"Cached count hash mismatch for {key}")
    return manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def fit_session_normalizer(features):
    transformed = np.log1p(np.maximum(features, 0))
    mean = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = transformed.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1e-5)


def apply_session_normalizer(features, normalizer):
    mean, scale = normalizer
    return ((np.log1p(np.maximum(features, 0)) - mean) / scale).astype(np.float32)


def prepare_mouse(mouse, mouse_blocks, *, n_classes=N_CLASSES, repeats=REPEATS_PER_BLOCK):
    """Normalize each complete training session, then concatenate without aggregation."""
    feature_parts = []
    target_parts = []
    normalizers = {}
    audit = []
    n_features = None
    for block in mouse_blocks:
        require(block.mouse == mouse, f"Wrong mouse block supplied to {mouse}")
        raw = np.asarray(block.features, dtype=np.float32)
        targets = np.asarray(block.trial_targets, dtype=np.int64)
        require(raw.ndim == 2 and len(raw) == len(targets), f"Feature/target mismatch in {block.key}")
        require(len(raw) == n_classes * repeats, f"Unexpected trial count in {block.key}")
        require(np.isfinite(raw).all() and raw.min(initial=0) >= 0, f"Invalid raw counts in {block.key}")
        require(
            len(targets) and targets.min() >= 0 and targets.max() < n_classes,
            f"Out-of-range training target in {block.key}",
        )
        counts = np.bincount(targets, minlength=n_classes)
        require(
            len(counts) == n_classes and np.all(counts == repeats),
            f"Every target must have exactly {repeats} original trials in {block.key}",
        )
        if n_features is None:
            n_features = raw.shape[1]
        require(raw.shape[1] == n_features, f"Feature dimension changed in {block.key}")
        normalizer = fit_session_normalizer(raw)
        normalized = apply_session_normalizer(raw, normalizer)
        require(np.isfinite(normalized).all(), f"Nonfinite normalized features in {block.key}")
        feature_parts.append(normalized)
        target_parts.append(targets)
        normalizers[block.session] = normalizer
        audit.append({
            "block": block.key,
            "train_trials": len(raw),
            "target_classes": n_classes,
            "original_trials_per_target": repeats,
            "normalizer_source": "all original public train trials in this session only",
        })
    features = np.concatenate(feature_parts).astype(np.float32, copy=False)
    targets = np.concatenate(target_parts).astype(np.int64, copy=False)
    require(len(features) == len(mouse_blocks) * n_classes * repeats, "Concatenation changed row count")
    return {
        "mouse": mouse,
        "features": features,
        "targets": targets,
        "normalizers": normalizers,
        "audit": audit,
    }


@torch.inference_mode()
def exact_top1(model, features, targets, *, batch_size=BATCH_SIZE, device="cpu"):
    """Evaluate every row with eval-mode hard argmax; return unrounded integer counts."""
    device = torch.device(device)
    require(len(features) == len(targets) and len(targets) > 0, "Exact evaluation requires aligned nonempty data")
    model.eval()
    correct = 0
    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        batch_x = torch.from_numpy(
            np.ascontiguousarray(features[start:stop], dtype=np.float32)
        ).to(device)
        batch_y = torch.from_numpy(np.asarray(targets[start:stop], dtype=np.int64)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch_x)
        correct += int((logits.argmax(dim=1) == batch_y).sum().item())
    total = len(targets)
    return {"correct": correct, "total": total, "accuracy": correct / total}


def save_checkpoint(path, model, *, mouse, epoch, metrics, normalizers):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    session_normalization = {
        session: {
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
        }
        for session, (mean, scale) in normalizers.items()
    }
    checkpoint = {
        "mouse": mouse,
        "n_features": model.n_features,
        "n_classes": model.n_classes,
        "hidden": model.hidden,
        "dropout": model.dropout,
        "apply_log": model.apply_log,
        "architecture": "mlp",
        "n_units": None,
        "n_bins": None,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "session_normalization": session_normalization,
        "session_normalization_input": "raw 25ms single-trial counts; log1p then session mean/std",
        "epoch": epoch,
        "seed": SEED,
        "bin_ms": BIN_MS,
        "train_correct": metrics["correct"],
        "train_total": metrics["total"],
        "train_accuracy": metrics["accuracy"],
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def fit_mouse(prepared, *, n_classes=N_CLASSES, device, output=None, max_epochs=MAX_EPOCHS,
              batch_size=BATCH_SIZE, hidden=HIDDEN, learning_rate=LEARNING_RATE, seed=SEED):
    """Zero-smoothing, zero-decay fitting with full original-train evaluation each epoch."""
    require(max_epochs > 0, "Epoch limit must be positive")
    features = prepared["features"]
    targets = prepared["targets"]
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpikeClassifier(
        n_features=features.shape[1],
        n_classes=n_classes,
        hidden=hidden,
        dropout=DROPOUT,
        apply_log=False,
    )
    model.set_normalization(features)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    history = []
    output = Path(output) if output is not None else None
    for epoch in range(1, max_epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(targets))
        model.train()
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_x = torch.from_numpy(
                np.ascontiguousarray(features[indices], dtype=np.float32)
            ).to(device)
            batch_y = torch.from_numpy(targets[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch_x)
                loss = F.cross_entropy(
                    logits.float(), batch_y, label_smoothing=LABEL_SMOOTHING
                )
            require(bool(torch.isfinite(loss).item()), f"Nonfinite loss for {prepared['mouse']} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * len(indices)
        metrics = exact_top1(model, features, targets, batch_size=batch_size, device=device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(targets),
            "train_correct": metrics["correct"],
            "train_total": metrics["total"],
            "train_accuracy": metrics["accuracy"],
            "evaluation": "model.eval exact top1 on every original single training trial",
        }
        history.append(row)
        if output is not None:
            save_checkpoint(
                output / "classifier.pt", model, mouse=prepared["mouse"], epoch=epoch,
                metrics=metrics, normalizers=prepared["normalizers"],
            )
            write_json(output / "history.json", history)
        print(f"TRAIN {prepared['mouse']} {json.dumps(row, allow_nan=False)}", flush=True)
        if metrics["correct"] == metrics["total"]:
            break
    return model, history


def overall_status(correct, total):
    return "passed" if correct == TRIALS_TOTAL and total == TRIALS_TOTAL else "failed"


def run_real(epochs_override=None):
    preregistration = validate_preregistration()
    manifest, cache_manifest_sha256 = preflight_cache()
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=BIN_MS)
    require(tuple(block.key for block in blocks) == BLOCKS, "Unexpected ordered public train blocks")
    for block in blocks:
        require(block.split == "train" and block.task == "train", f"Non-train data entered: {block.key}")
        require(block.n_trials == TRIALS_PER_BLOCK, f"Wrong block length: {block.key}")
        require(block.n_samples == TRIALS_PER_BLOCK, f"Wrong sample length: {block.key}")
        require(np.all(np.diff(block.group_ptr) == 1), f"Train rows were unexpectedly grouped: {block.key}")
    max_epochs = MAX_EPOCHS if epochs_override is None else epochs_override
    is_preregistered = epochs_override is None
    protocol_status = "preregistered" if is_preregistered else "non_preregistered_debug"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    write_json(OUT / "results.json", {
        "experiment": OUT.name,
        "status": "running",
        "protocol_status": protocol_status,
        "is_preregistered_protocol": is_preregistered,
        "epochs_limit": max_epochs,
        "overall": {"correct": 0, "total": TRIALS_TOTAL, "correct_over_total": f"0/{TRIALS_TOTAL}", "accuracy": 0.0},
    })
    print(
        f"REAL RUN device={device}; bin_ms={BIN_MS}; max_epochs={max_epochs}; "
        f"protocol={protocol_status}; cache splits are not used",
        flush=True,
    )
    per_mouse = {}
    for mouse in MICE:
        mouse_blocks = [block for block in blocks if block.mouse == mouse]
        require(len(mouse_blocks) == 4, f"Expected four sessions for {mouse}")
        prepared = prepare_mouse(mouse, mouse_blocks)
        require(len(prepared["targets"]) == TRIALS_PER_MOUSE, f"Wrong total trials for {mouse}")
        output = ARTIFACTS / mouse
        model, history = fit_mouse(
            prepared, device=device, output=output, max_epochs=max_epochs
        )
        last = history[-1]
        per_mouse[mouse] = {
            "correct": last["train_correct"],
            "total": last["train_total"],
            "accuracy": last["train_accuracy"],
            "epochs": len(history),
            "early_stopped_at_100_percent": last["train_correct"] == last["train_total"],
            "checkpoint": (output / "classifier.pt").relative_to(ROOT).as_posix(),
            "history": (output / "history.json").relative_to(ROOT).as_posix(),
            "sessions": prepared["audit"],
        }
        del model, prepared
        if device.type == "cuda":
            torch.cuda.empty_cache()
        correct_so_far = sum(row["correct"] for row in per_mouse.values())
        total_so_far = sum(row["total"] for row in per_mouse.values())
        if correct_so_far == TRIALS_TOTAL and total_so_far == TRIALS_TOTAL:
            break
    correct = sum(row["correct"] for row in per_mouse.values())
    total = sum(row["total"] for row in per_mouse.values())
    status = overall_status(correct, total)
    result = {
        "experiment": OUT.name,
        "status": status,
        "goal_met": status == "passed",
        "protocol_status": protocol_status,
        "is_preregistered_protocol": is_preregistered,
        "epochs_limit": max_epochs,
        "default_preregistered_epochs": MAX_EPOCHS,
        "epochs_override_used": not is_preregistered,
        "overall": {
            "correct": correct,
            "total": total,
            "correct_over_total": f"{correct}/{total}",
            "accuracy": correct / total,
        },
        "per_mouse": per_mouse,
        "protocol": {
            "bin_ms": BIN_MS,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "label_smoothing": LABEL_SMOOTHING,
            "weight_decay": WEIGHT_DECAY,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "validation_fold_used": False,
            "split_manifest_used": False,
            "mean_augmentation_used": False,
            "hungarian_assignment_used": False,
            "image_or_image_index_prior_used": False,
        },
        "cache": {
            "path": CACHE.relative_to(ROOT).as_posix(),
            "manifest_sha256": cache_manifest_sha256,
            "source_root": manifest["source_root"],
        },
        "preregistration_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "Training-set capacity and label-pipeline acceptance only; no generalization claim.",
    }
    write_json(OUT / "results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


def smoke_test():
    """Tiny synthetic fits test labels, full eval, normalization, and actual early stopping."""
    validate_preregistration()
    rng = np.random.default_rng(2026)
    n_classes = 4
    repeats = 3
    max_epochs = 40
    synthetic_mice = ("synthetic1", "synthetic2")
    blocks_by_mouse = {}
    originals = {}
    for mouse_index, mouse in enumerate(synthetic_mice):
        blocks = []
        for session_index in range(2):
            targets = np.tile(np.arange(n_classes, dtype=np.int64), repeats)
            order = rng.permutation(len(targets))
            targets = targets[order]
            features = rng.poisson(0.1, size=(len(targets), n_classes)).astype(np.float32)
            signature = (targets * 3 + mouse_index) % n_classes
            features[np.arange(len(targets)), signature] += 12 + mouse_index + session_index
            key = f"{mouse}/session{session_index + 1}"
            block = SimpleNamespace(
                mouse=mouse,
                session=f"session{session_index + 1}",
                key=key,
                features=features,
                trial_targets=targets,
            )
            originals[key] = features.copy()
            blocks.append(block)
        blocks_by_mouse[mouse] = blocks

    histories = {}
    total_correct = 0
    total_rows = 0
    for mouse in synthetic_mice:
        prepared = prepare_mouse(mouse, blocks_by_mouse[mouse], n_classes=n_classes, repeats=repeats)
        expected_targets = np.concatenate([block.trial_targets for block in blocks_by_mouse[mouse]])
        np.testing.assert_array_equal(prepared["targets"], expected_targets)
        assert len(prepared["targets"]) == 2 * n_classes * repeats
        for block in blocks_by_mouse[mouse]:
            np.testing.assert_array_equal(block.features, originals[block.key])
        model, history = fit_mouse(
            prepared,
            n_classes=n_classes,
            device="cpu",
            output=None,
            max_epochs=max_epochs,
            batch_size=8,
            hidden=16,
            learning_rate=0.01,
            seed=7,
        )
        last = history[-1]
        assert last["train_correct"] == last["train_total"] == len(expected_targets)
        assert last["train_accuracy"] == 1.0
        assert len(history) < max_epochs, "Smoke test did not exercise early stopping"
        assert not model.training
        independent_logits = predict_logits(model, prepared["features"], batch_size=7, device="cpu")
        independent = exact_top1(model, prepared["features"], prepared["targets"], batch_size=7, device="cpu")
        assert independent["correct"] == int(
            (independent_logits.argmax(axis=1) == prepared["targets"]).sum()
        )
        expected_mean = prepared["features"].mean(axis=0, dtype=np.float64).astype(np.float32)
        np.testing.assert_array_equal(model.feature_mean.detach().cpu().numpy(), expected_mean)
        histories[mouse] = history
        total_correct += last["train_correct"]
        total_rows += last["train_total"]
    assert total_correct == total_rows == 2 * 2 * n_classes * repeats
    assert overall_status(TRIALS_TOTAL, TRIALS_TOTAL) == "passed"
    assert overall_status(TRIALS_TOTAL - 1, TRIALS_TOTAL) == "failed"
    print(json.dumps({
        "status": "passed",
        "synthetic_only": True,
        "real_data_accessed": False,
        "real_training_started": False,
        "cache_accessed": False,
        "checks": [
            "nonordered synthetic target IDs remain aligned with every original single-trial row",
            "session log1p normalization leaves raw cached counts unchanged",
            "model normalization exactly matches its own mouse's concatenated training features",
            "eval-mode full-array exact top1 agrees with independent predict_logits argmax",
            "both tiny synthetic mouse fits reach their own 100% target and stop before the epoch cap",
            "overall status is failed for 87935/87936 and passed only for 87936/87936",
        ],
        "synthetic_total": {"correct": total_correct, "total": total_rows, "accuracy": total_correct / total_rows},
        "tiny_training_histories": histories,
    }, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def positive_epochs(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--epochs must be a positive integer")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Run the read-only-cache real experiment")
    mode.add_argument("--smoke-test", action="store_true", help="Run tiny synthetic CPU tests only")
    parser.add_argument(
        "--epochs", type=positive_epochs, default=None,
        help="Debug-only epoch override; any explicit value marks real results non-preregistered",
    )
    args = parser.parse_args()
    if args.smoke_test and args.epochs is not None:
        parser.error("--epochs is only a debug override for --run; smoke-test epochs are internally fixed")
    return args


def main():
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        run_real(epochs_override=args.epochs)


if __name__ == "__main__":
    main()
