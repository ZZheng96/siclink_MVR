"""Paired Task3 single-trial training with one frozen count-dropout candidate."""
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

from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions
from src.features import aggregate_by_target, load_feature_blocks
from src.final import apply_session_normalizer, fit_session_normalizer
from src.neural import SpikeClassifier, predict_logits
from src.retrieval import unique_assignment
from src.ridge import RidgePath, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
OUT = Path(__file__).resolve().parent
PREREGISTRATION = OUT / "preregistration.json"
MANIFEST = OUT / "manifest.json"
ARTIFACTS = OUT / "artifacts"
MICE = ("mice1", "mice2")
BLOCKS = tuple(f"{mouse}/session{session}" for mouse in MICE for session in range(1, 5))
ARMS = ("baseline", "count_dropout")
SEED = 42
PICTURE_SEED = 123
N_GLYPHS = 2748
K = 32
EPOCHS = {"mice1": 7, "mice2": 7}
HIDDEN = 512
DROPOUT = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
ENCODING_ALPHA = 0.01
PRESERVE_ORIGINAL_PROBABILITY = 0.5
BINOMIAL_RETENTION_PROBABILITY = 0.5


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_record(path):
    path = Path(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def environment(device=None):
    result = {
        "python": sys.executable,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
    }
    if device is not None:
        device = torch.device(device)
        result["device"] = str(device)
        result["gpu"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return result


def validate_preregistration():
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    fixed = preregistration["fixed_training_contract"]
    candidate = preregistration["frozen_count_dropout_candidate"]
    assert preregistration["experiment"] == OUT.name
    assert preregistration["status"] == "preregistered_before_real_training_or_real_metrics"
    assert fixed["training_seeds"] == [SEED]
    assert fixed["epochs_by_mouse"] == EPOCHS
    assert fixed["batch_size"] == BATCH_SIZE
    assert candidate["preserve_original_probability"] == PRESERVE_ORIGINAL_PROBABILITY
    assert candidate["binomial_retention_probability"] == BINOMIAL_RETENTION_PROBABILITY
    assert preregistration["data_and_split"]["gallery_size"] == N_GLYPHS
    assert preregistration["data_and_split"]["blocks"] == list(BLOCKS)
    assert preregistration["picture_subsets"]["construction"] == (
        "order = numpy.random.default_rng(123).permutation(2748)"
    )
    assert preregistration["frozen_gates"]["numeric_tolerance"] == 0.0
    return preregistration


def row_zscore(scores):
    scores = np.asarray(scores, dtype=np.float32)
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), 1e-6
    )


def validate_counts(features):
    features = np.asarray(features)
    if not np.isfinite(features).all() or np.any(features < 0):
        raise ValueError("Raw spike counts must be finite and nonnegative")
    if not np.array_equal(features, np.floor(features)):
        raise ValueError("Count dropout requires integer raw spike counts")


def perturb_counts(raw, rng):
    """Preserve each row with p=.5; otherwise apply independent Binomial(x,.5)."""
    counts = np.asarray(raw, dtype=np.int64)
    preserve = rng.random(len(counts)) < PRESERVE_ORIGINAL_PROBABILITY
    output = counts.copy()
    changed = ~preserve
    if changed.any():
        output[changed] = rng.binomial(counts[changed], BINOMIAL_RETENTION_PROBABILITY)
    return output, preserve


def augment_batch(raw, normalized, block_ids, means, scales, rng):
    features = np.array(normalized, dtype=np.float32, copy=True)
    perturbed, preserve = perturb_counts(raw, rng)
    changed = ~preserve
    if changed.any():
        selected_blocks = block_ids[changed]
        features[changed] = (
            np.log1p(perturbed[changed]) - means[selected_blocks]
        ) / scales[selected_blocks]
    return features, preserve


def fit_arm(mouse, arm, data, n_classes, device, output=None, *, epochs=None,
            batch_size=BATCH_SIZE, hidden=HIDDEN, dropout=DROPOUT,
            learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, seed=SEED):
    if arm not in ARMS:
        raise ValueError(arm)
    device = torch.device(device)
    epochs = EPOCHS[mouse] if epochs is None else epochs
    raw = data["train_raw"]
    normalized = data["train_x"]
    targets = data["train_y"]
    block_ids = data["train_block_ids"]
    if not (len(raw) == len(normalized) == len(targets) == len(block_ids)):
        raise ValueError("Paired training arrays are misaligned")

    torch.manual_seed(seed)
    model = SpikeClassifier(
        n_features=normalized.shape[1], n_classes=n_classes,
        hidden=hidden, dropout=dropout, apply_log=False,
    )
    model.set_normalization(normalized)
    model.to(device)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    history = []
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(targets))
        noise_rng = np.random.default_rng(1000 + seed + epoch)
        model.train()
        total_loss = 0.0
        total_correct = 0
        original_rows = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            if arm == "count_dropout":
                batch_x, preserve = augment_batch(
                    raw[indices], normalized[indices], block_ids[indices],
                    data["normalizer_means"], data["normalizer_scales"], noise_rng,
                )
                original_rows += int(preserve.sum())
            else:
                batch_x = normalized[indices]
                original_rows += len(indices)
            features = torch.from_numpy(np.ascontiguousarray(batch_x, dtype=np.float32)).to(device)
            batch_targets = torch.from_numpy(targets[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(features)
                loss = F.cross_entropy(logits.float(), batch_targets, label_smoothing=0.05)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss for {mouse}/{arm} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(indices)
            total_correct += int((logits.argmax(dim=1) == batch_targets).sum().item())
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(targets),
            "train_accuracy": total_correct / len(targets),
            "original_rows": original_rows,
            "thinned_rows": len(targets) - original_rows,
        }
        history.append(row)
        print(f"TRAIN {mouse}/{arm} {json.dumps(row)}", flush=True)

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "classifier.pt"
        torch.save({
            "n_features": model.n_features,
            "n_classes": model.n_classes,
            "hidden": model.hidden,
            "dropout": model.dropout,
            "apply_log": model.apply_log,
            "architecture": "mlp",
            "n_units": None,
            "n_bins": None,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "epoch": epochs,
            "seed": seed,
            "arm": arm,
        }, checkpoint)
        write_json(output / "history.json", history)
    return model, history


def prepare_mouse(mouse, mouse_blocks, split, n_classes):
    train_raw_parts = []
    train_x_parts = []
    train_y_parts = []
    train_block_parts = []
    normalizer_means = []
    normalizer_scales = []
    encoding_parts = []
    rows = []
    audit = []
    n_features = None
    for block_index, block in enumerate(mouse_blocks):
        part = split.blocks[block.key]
        raw = np.asarray(block.features[part.train_trials], dtype=np.float32)
        targets = np.asarray(block.trial_targets[part.train_trials], dtype=np.int64)
        validation_raw = np.asarray(block.features[part.validation_trials], dtype=np.float32)
        validation_targets = np.asarray(block.trial_targets[part.validation_trials], dtype=np.int64)
        validate_counts(raw)
        validate_counts(validation_raw)
        if np.intersect1d(part.train_trials, part.validation_trials).size:
            raise ValueError(f"Train/validation trial overlap in {block.key}")
        if len(raw) != 3 * n_classes or len(validation_raw) != n_classes:
            raise ValueError(f"Unexpected Task3 trial counts in {block.key}")
        if not np.all(np.bincount(targets, minlength=n_classes) == 3):
            raise ValueError(f"Expected exactly three training repeats per glyph in {block.key}")
        if not np.array_equal(np.sort(validation_targets), np.arange(n_classes)):
            raise ValueError(f"Incomplete validation gallery in {block.key}")
        if targets.min() < 0 or targets.max() >= n_classes:
            raise ValueError(f"Out-of-range training labels in {block.key}")
        if n_features is None:
            n_features = raw.shape[1]
        if raw.shape[1] != n_features or validation_raw.shape[1] != n_features:
            raise ValueError(f"Feature mismatch in {block.key}")

        normalizer = fit_session_normalizer(raw)
        normalized = apply_session_normalizer(raw, normalizer).astype(np.float32)
        mean_targets, mean_raw = aggregate_by_target(raw, targets)
        if not np.array_equal(mean_targets, np.arange(n_classes)):
            raise ValueError(f"Incomplete training glyph means in {block.key}")
        encoding_parts.append(np.log1p(mean_raw).astype(np.float32))
        train_raw_parts.append(raw)
        train_x_parts.append(normalized)
        train_y_parts.append(targets)
        train_block_parts.append(np.full(len(raw), block_index, dtype=np.int64))
        normalizer_means.append(normalizer[0])
        normalizer_scales.append(normalizer[1])
        rows.append({
            "block": block.key,
            "mouse": mouse,
            "raw": validation_raw,
            "targets": validation_targets,
            "normalizer": normalizer,
        })
        audit.append({
            "block": block.key,
            "train_trials": len(part.train_trials),
            "validation_trials": len(part.validation_trials),
            "train_validation_overlap": 0,
            "train_repeats_per_glyph": 3,
            "validation_repeats_per_glyph": 1,
            "normalizer_source": "train_trials only",
        })
    return {
        "mouse": mouse,
        "train_raw": np.concatenate(train_raw_parts),
        "train_x": np.concatenate(train_x_parts),
        "train_y": np.concatenate(train_y_parts),
        "train_block_ids": np.concatenate(train_block_parts),
        "normalizer_means": np.stack(normalizer_means).astype(np.float32),
        "normalizer_scales": np.stack(normalizer_scales).astype(np.float32),
        "encoding_parts": encoding_parts,
        "rows": rows,
        "audit": audit,
    }


def fit_shared_encoding(prepared, bank_coefficients):
    n_classes = len(bank_coefficients)
    for mouse, data in prepared.items():
        encoding_targets = np.tile(np.arange(n_classes), len(data["encoding_parts"]))
        ridge = RidgePath().fit(
            bank_coefficients[encoding_targets], np.concatenate(data["encoding_parts"])
        )
        data["prototypes"] = ridge.predict(bank_coefficients, ENCODING_ALPHA).astype(np.float32)
        data.pop("encoding_parts")
        print(f"ENCODING {mouse}: fitted from train_trials only", flush=True)


def continuous_image_scores(logits, centered_bank):
    normalized_bank = F.normalize(centered_bank, dim=1)
    output = []
    for start in range(0, len(logits), 64):
        batch = torch.from_numpy(logits[start:start + 64]).to(centered_bank.device)
        probabilities = F.softmax(batch, dim=1)
        expected = F.normalize(probabilities @ centered_bank, dim=1)
        output.append((expected @ normalized_bank.T).cpu().numpy())
    return np.concatenate(output)


def rank_scores(scores):
    """Labels are deliberately absent: ranking and assignment use scores only."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite matrix")
    top = np.argsort(-scores, axis=1, kind="stable")[:, :min(K, scores.shape[1])].copy()
    assigned = unique_assignment(scores)
    if len(np.unique(assigned)) != len(assigned):
        raise ValueError("Assignment is not one-to-one")
    return top, assigned


@torch.inference_mode()
def rank_full_blocks(prepared, models, centered_bank, device):
    rankings = {arm: [] for arm in ARMS}
    for mouse in MICE:
        data = prepared[mouse]
        for row in data["rows"]:
            raw = row["raw"]
            normalized = apply_session_normalizer(raw, row["normalizer"]).astype(np.float32)
            encoding = response_scores(
                np.log1p(raw), data["prototypes"], metric="mse", device=device
            )
            encoding_z = row_zscore(encoding)
            for arm in ARMS:
                logits = predict_logits(models[mouse][arm], normalized, device=device)
                image = continuous_image_scores(logits, centered_bank)
                scores = (row_zscore(logits) + encoding_z + row_zscore(image)) / 3.0
                top, assigned = rank_scores(scores)
                rankings[arm].append({
                    "block": row["block"],
                    "mouse": mouse,
                    "targets": row["targets"],
                    "top32": top,
                    "assigned": assigned,
                })
            print(f"RANKED {row['block']}: full gallery, both arms", flush=True)
    return rankings


def subset_records(rankings, pictures, n_classes):
    pictures = np.asarray(pictures, dtype=np.int64)
    output = {}
    for arm, parts in rankings.items():
        selected = []
        for part in parts:
            if not np.array_equal(np.sort(part["targets"]), np.arange(n_classes)):
                raise ValueError(f"Incomplete full-block targets in {part['block']}")
            keep = np.flatnonzero(np.isin(part["targets"], pictures))
            keep = keep[np.argsort(part["targets"][keep], kind="stable")]
            if not np.array_equal(part["targets"][keep], pictures):
                raise ValueError(f"Unexpected picture subset in {part['block']}")
            selected.append({
                "block": part["block"],
                "mouse": part["mouse"],
                "targets": part["targets"][keep],
                "top32": part["top32"][keep],
                "assigned": part["assigned"][keep],
            })
        output[arm] = {
            "targets": np.concatenate([part["targets"] for part in selected]),
            "top32": np.concatenate([part["top32"] for part in selected]),
            "assigned": np.concatenate([part["assigned"] for part in selected]),
            "groups": np.concatenate([
                np.full(len(part["targets"]), part["block"]) for part in selected
            ]),
            "mice": np.concatenate([
                np.full(len(part["targets"]), part["mouse"]) for part in selected
            ]),
        }
    return output


def fill_ocf_cache(records_by_arm, images, cache, device):
    pairs = np.unique(np.concatenate([
        np.column_stack((records["assigned"], records["targets"]))
        for records in records_by_arm.values()
    ]).astype(np.int64), axis=0)
    missing = [tuple(map(int, pair)) for pair in pairs if tuple(map(int, pair)) not in cache]
    print(f"OCF assigned pairs={len(pairs)} new={len(missing)}", flush=True)
    for start in range(0, len(missing), 512):
        chunk = missing[start:start + 512]
        predicted = np.fromiter((pair[0] for pair in chunk), dtype=np.int64)
        targets = np.fromiter((pair[1] for pair in chunk), dtype=np.int64)
        scores = score_predictions(
            images[predicted], images[targets], device=str(device), batch_size=16
        )
        for pair, score in zip(chunk, scores):
            cache[pair] = float(score)


def metric_summary(records, cache):
    targets = records["targets"]
    top = records["top32"]
    assigned = records["assigned"]
    inside = (top == targets[:, None]).any(axis=1)
    recall1 = top[:, 0] == targets
    correct = assigned == targets
    final_values = np.asarray([
        cache[(int(predicted), int(target))]
        for predicted, target in zip(assigned, targets)
    ])

    def summarize(mask):
        n = int(mask.sum())
        if not n:
            raise ValueError("Cannot summarize an empty metric group")
        return {
            "n": n,
            "recall_at_1": float(recall1[mask].mean()),
            "recall_at_32": float(inside[mask].mean()),
            "final_accuracy": float(correct[mask].mean()),
            "final_ocf": float(final_values[mask].mean()),
            "recall_at_1_count": int(recall1[mask].sum()),
            "recall_at_32_count": int(inside[mask].sum()),
            "final_correct_count": int(correct[mask].sum()),
        }

    return {
        "overall": summarize(np.ones(len(targets), dtype=bool)),
        "by_mouse": {
            mouse: summarize(records["mice"] == mouse)
            for mouse in sorted(set(records["mice"].tolist()))
        },
        "by_block": {
            block: summarize(records["groups"] == block)
            for block in sorted(set(records["groups"].tolist()))
        },
    }


def paired_deltas(metrics):
    fields = ("recall_at_1", "recall_at_32", "final_accuracy", "final_ocf")
    output = {}
    for scope in ("overall", "by_mouse", "by_block"):
        if scope == "overall":
            output[scope] = {
                field: metrics["count_dropout"][scope][field] - metrics["baseline"][scope][field]
                for field in fields
            }
        else:
            output[scope] = {
                group: {
                    field: metrics["count_dropout"][scope][group][field]
                    - metrics["baseline"][scope][group][field]
                    for field in fields
                }
                for group in metrics["baseline"][scope]
            }
    return output


def evaluate_gate(metrics, stage):
    if stage not in ("tune", "heldout"):
        raise ValueError(stage)
    checks = {}
    for mouse in MICE:
        baseline = metrics["baseline"]["by_mouse"][mouse]
        candidate = metrics["count_dropout"]["by_mouse"][mouse]
        recall_delta = candidate["recall_at_32"] - baseline["recall_at_32"]
        ocf_delta = candidate["final_ocf"] - baseline["final_ocf"]
        if stage == "tune":
            recall_pass = recall_delta >= 0.0
        else:
            recall_pass = recall_delta > 0.0
        ocf_pass = ocf_delta >= 0.0
        checks[mouse] = {
            "recall_at_32_delta": recall_delta,
            "recall_at_32_pass": bool(recall_pass),
            "final_ocf_delta": ocf_delta,
            "final_ocf_pass": bool(ocf_pass),
        }
    return {
        "stage": stage,
        "passed": all(
            check["recall_at_32_pass"] and check["final_ocf_pass"]
            for check in checks.values()
        ),
        "checks": checks,
    }


def manifest_payload(status, device=None):
    dependencies = [
        PREREGISTRATION,
        Path(__file__),
        ROOT / "src/dataloader.py",
        ROOT / "src/encoding.py",
        ROOT / "src/evaluation.py",
        ROOT / "src/features.py",
        ROOT / "src/final.py",
        ROOT / "src/neural.py",
        ROOT / "src/retrieval.py",
        ROOT / "src/ridge.py",
        ROOT / "src/splits.py",
    ]
    result = {
        "experiment": OUT.name,
        "status": status,
        "preregistration_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "reproduce_from_repository_root": (
            ".venv\\Scripts\\python.exe -B -u test\\45_task3_repeat_dropout\\run.py --run"
        ),
        "synthetic_smoke_command": (
            ".venv\\Scripts\\python.exe -B test\\45_task3_repeat_dropout\\run.py --smoke-test"
        ),
        "dependencies": [file_record(path) for path in dependencies],
        "expected_compute_scale": json.loads(
            PREREGISTRATION.read_text(encoding="utf-8")
        )["expected_real_compute_scale"],
    }
    if device is not None:
        result["environment"] = environment(device)
    if status != "running":
        outputs = [OUT / "tune.json", OUT / "results.json"]
        outputs.extend(sorted(ARTIFACTS.glob("models/*/*/classifier.pt")))
        outputs.extend(sorted(ARTIFACTS.glob("models/*/*/history.json")))
        result["outputs"] = [file_record(path) for path in outputs if path.is_file()]
    return result


def write_manifest(status, device=None):
    write_json(MANIFEST, manifest_payload(status, device))


def run_real():
    preregistration = validate_preregistration()
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_manifest("running", device)
    print(f"REAL RUN device={device}; frozen arms={ARMS}; seeds=[{SEED}]", flush=True)

    images = (load_images(DATA) >= 128).astype(np.float32)
    if images.shape != (N_GLYPHS, 64, 64):
        raise ValueError("Unexpected public training gallery")
    bank_coefficients = dct_encode(images, 16).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    if tuple(block.key for block in blocks) != BLOCKS:
        raise ValueError("Expected the eight ordered public training blocks")
    split = make_split(blocks, "task3", seed=SEED)
    order = np.random.default_rng(PICTURE_SEED).permutation(N_GLYPHS)
    tune_pictures = np.sort(order[:256])
    heldout_pictures = np.sort(order[256:768])
    if np.intersect1d(tune_pictures, heldout_pictures).size:
        raise ValueError("Tune and heldout pictures overlap")

    prepared = {}
    for mouse in MICE:
        prepared[mouse] = prepare_mouse(
            mouse, [block for block in blocks if block.mouse == mouse], split, N_GLYPHS
        )
    fit_shared_encoding(prepared, bank_coefficients)
    models = {mouse: {} for mouse in MICE}
    training = []
    for mouse in MICE:
        for arm in ARMS:
            output = ARTIFACTS / "models" / arm / mouse
            model, history = fit_arm(
                mouse, arm, prepared[mouse], N_GLYPHS, device, output=output
            )
            models[mouse][arm] = model
            training.append({
                "mouse": mouse,
                "arm": arm,
                "seed": SEED,
                "epochs": len(history),
                "train_rows": len(prepared[mouse]["train_y"]),
                "mean_rows_added": 0,
                "validation_rows_used_for_training": 0,
                "checkpoint_epoch": len(history),
                "checkpoint": (output / "classifier.pt").relative_to(ROOT).as_posix(),
                "history": (output / "history.json").relative_to(ROOT).as_posix(),
            })

    centered_bank = torch.from_numpy(
        bank_coefficients - bank_coefficients.mean(axis=0, keepdims=True)
    ).to(device)
    rankings = rank_full_blocks(prepared, models, centered_bank, device)
    tune_records = subset_records(rankings, tune_pictures, N_GLYPHS)
    ocf_cache = {}
    fill_ocf_cache(tune_records, images, ocf_cache, device)
    tune_metrics = {
        arm: metric_summary(tune_records[arm], ocf_cache) for arm in ARMS
    }
    tune_gate = evaluate_gate(tune_metrics, "tune")
    protocol = {
        "data_scope": "public train only; no hidden test, labels, IDs, or submissions",
        "split_seed": SEED,
        "picture_seed": PICTURE_SEED,
        "tune_pictures": tune_pictures.tolist(),
        "heldout_pictures": heldout_pictures.tolist(),
        "training_trials_per_block": 8244,
        "validation_trials_per_block": 2748,
        "assignment_scope": "complete 2748-row full-gallery unique_assignment before either subset",
        "heldout_used_before_tune_gate": False,
        "historical_reuse_disclosure": preregistration["picture_subsets"]["historical_reuse_disclosure"],
    }
    common = {
        "experiment": OUT.name,
        "preregistration_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "protocol": protocol,
        "training": training,
        "data_integrity": {
            mouse: prepared[mouse]["audit"] for mouse in MICE
        },
        "tune": tune_metrics,
        "tune_delta_count_dropout_minus_baseline": paired_deltas(tune_metrics),
        "tune_gate": tune_gate,
    }
    write_json(OUT / "tune.json", dict(common, tune_locked_before_heldout=True))
    print(f"TUNE GATE passed={tune_gate['passed']}", flush=True)

    if not tune_gate["passed"]:
        result = dict(
            common,
            status="stopped_tune_gate",
            advance=False,
            heldout=None,
            heldout_evaluated_variants=[],
            stop_reason="At least one preregistered per-mouse tune condition failed",
            elapsed_seconds=time.perf_counter() - started,
            environment=environment(device),
        )
        write_json(OUT / "results.json", result)
        write_manifest("stopped_tune_gate", device)
        return result

    heldout_records = subset_records(rankings, heldout_pictures, N_GLYPHS)
    fill_ocf_cache(heldout_records, images, ocf_cache, device)
    heldout_metrics = {
        arm: metric_summary(heldout_records[arm], ocf_cache) for arm in ARMS
    }
    heldout_gate = evaluate_gate(heldout_metrics, "heldout")
    result = dict(
        common,
        status="complete_gate_passed" if heldout_gate["passed"] else "complete_gate_failed",
        advance=heldout_gate["passed"],
        heldout=heldout_metrics,
        heldout_delta_count_dropout_minus_baseline=paired_deltas(heldout_metrics),
        heldout_gate=heldout_gate,
        heldout_evaluated_variants=list(ARMS),
        elapsed_seconds=time.perf_counter() - started,
        environment=environment(device),
        production_integrated=False,
    )
    write_json(OUT / "results.json", result)
    write_manifest(result["status"], device)
    print(f"FINAL heldout gate passed={heldout_gate['passed']}", flush=True)
    return result


def smoke_test():
    validate_preregistration()
    rng = np.random.default_rng(2026)
    n_classes = 6
    features = rng.poisson(2.0, size=(4 * n_classes, 4)).astype(np.float32) + 1
    block = SimpleNamespace(
        mouse="mice1", session="session1", key="mice1/session1",
        sample_ids=np.arange(n_classes), group_ptr=np.arange(n_classes + 1) * 4,
        targets=np.arange(n_classes), trial_targets=np.repeat(np.arange(n_classes), 4),
        features=features, n_samples=n_classes, n_trials=4 * n_classes, n_features=4,
    )
    split = make_split([block], "task3", seed=SEED)
    part = split.blocks[block.key]
    prepared = prepare_mouse("mice1", [block], split, n_classes)
    assert len(part.train_trials) == 18 and len(part.validation_trials) == 6
    assert not np.intersect1d(part.train_trials, part.validation_trials).size
    assert len(prepared["train_raw"]) == 18

    changed_block = SimpleNamespace(**vars(block))
    changed_block.features = features.copy()
    changed_block.features[part.validation_trials] += 1000
    changed = prepare_mouse("mice1", [changed_block], split, n_classes)
    np.testing.assert_array_equal(prepared["train_raw"], changed["train_raw"])
    np.testing.assert_array_equal(prepared["train_x"], changed["train_x"])
    np.testing.assert_array_equal(prepared["normalizer_means"], changed["normalizer_means"])
    np.testing.assert_array_equal(prepared["normalizer_scales"], changed["normalizer_scales"])

    count_probe = np.tile(features[:8], (8, 1))
    thinned_probe, probe_preserve = perturb_counts(count_probe, np.random.default_rng(19))
    assert probe_preserve.any() and (~probe_preserve).any()
    assert np.all(thinned_probe >= 0) and np.all(thinned_probe <= count_probe)
    np.testing.assert_array_equal(
        thinned_probe[probe_preserve], count_probe[probe_preserve]
    )

    raw = prepared["train_raw"][:8]
    normalized = prepared["train_x"][:8]
    raw_before = raw.copy()
    normalized_before = normalized.copy()
    augmented, preserve = augment_batch(
        raw, normalized, prepared["train_block_ids"][:8],
        prepared["normalizer_means"], prepared["normalizer_scales"],
        np.random.default_rng(7),
    )
    np.testing.assert_array_equal(raw, raw_before)
    np.testing.assert_array_equal(normalized, normalized_before)
    np.testing.assert_array_equal(augmented[preserve], normalized[preserve])
    repeated, repeated_preserve = augment_batch(
        raw, normalized, prepared["train_block_ids"][:8],
        prepared["normalizer_means"], prepared["normalizer_scales"],
        np.random.default_rng(7),
    )
    np.testing.assert_array_equal(preserve, repeated_preserve)
    np.testing.assert_array_equal(augmented, repeated)
    assert np.isfinite(augmented).all()

    models = {}
    histories = {}
    for arm in ARMS:
        models[arm], histories[arm] = fit_arm(
            "mice1", arm, prepared, n_classes, "cpu", output=None,
            epochs=2, batch_size=8, hidden=8, dropout=0.2,
        )
        assert len(histories[arm]) == 2
        assert all(row["original_rows"] + row["thinned_rows"] == 18 for row in histories[arm])
        if arm == "baseline":
            assert all(row["thinned_rows"] == 0 for row in histories[arm])
    validation_x = apply_session_normalizer(
        prepared["rows"][0]["raw"], prepared["rows"][0]["normalizer"]
    ).astype(np.float32)
    first = predict_logits(models["count_dropout"], validation_x, device="cpu")
    second = predict_logits(models["count_dropout"], validation_x, device="cpu")
    np.testing.assert_array_equal(first, second)
    assert not models["count_dropout"].training

    synthetic_rankings = {arm: [] for arm in ARMS}
    for block_index, block_name in enumerate(BLOCKS):
        mouse = block_name.split("/")[0]
        for arm_index, arm in enumerate(ARMS):
            scores = rng.normal(size=(n_classes, n_classes)).astype(np.float32)
            scores += np.eye(n_classes, dtype=np.float32) * (1.0 + arm_index)
            top, assigned = rank_scores(scores)
            assert len(np.unique(assigned)) == n_classes
            synthetic_rankings[arm].append({
                "block": block_name, "mouse": mouse, "targets": np.arange(n_classes),
                "top32": top, "assigned": assigned,
            })
    records = subset_records(synthetic_rankings, np.arange(2), n_classes)
    cache = {}
    for arm_records in records.values():
        for predicted, target in zip(arm_records["assigned"], arm_records["targets"]):
            cache[(int(predicted), int(target))] = 1.0 if predicted == target else 0.25
    synthetic_metrics = {
        arm: metric_summary(records[arm], cache) for arm in ARMS
    }
    assert synthetic_metrics["baseline"]["overall"]["n"] == 16
    assert set(synthetic_metrics["baseline"]["by_mouse"]) == set(MICE)
    assert set(synthetic_metrics["baseline"]["by_block"]) == set(BLOCKS)
    assert all(value["n"] == 2 for value in synthetic_metrics["baseline"]["by_block"].values())

    def gate_metrics(recall, ocf):
        row = {"recall_at_32": recall, "final_ocf": ocf}
        return {"by_mouse": {"mice1": dict(row), "mice2": dict(row)}}

    passing_tune = evaluate_gate({
        "baseline": gate_metrics(0.5, 10.0),
        "count_dropout": gate_metrics(0.5, 10.0),
    }, "tune")
    assert passing_tune["passed"]
    passing_heldout = evaluate_gate({
        "baseline": gate_metrics(0.5, 10.0),
        "count_dropout": gate_metrics(0.6, 10.0),
    }, "heldout")
    assert passing_heldout["passed"]
    assert not evaluate_gate({
        "baseline": gate_metrics(0.5, 10.0),
        "count_dropout": gate_metrics(0.5, 10.1),
    }, "heldout")["passed"]
    assert not evaluate_gate({
        "baseline": gate_metrics(0.5, 10.0),
        "count_dropout": gate_metrics(0.6, 9.9),
    }, "tune")["passed"]

    print(json.dumps({
        "status": "passed",
        "real_data_accessed": False,
        "real_training_started": False,
        "checks": [
            "task3 split has three train trials and one disjoint validation trial per glyph",
            "validation responses cannot affect training arrays or session normalizers",
            "count dropout respects 0 <= thinned <= raw, is reproducible, and keeps original rows exact",
            "two-epoch tiny baseline/candidate CPU fits execute without validation checkpointing",
            "candidate inference is deterministic and uses eval mode without count noise",
            "score-only full assignment and two-mouse/eight-block metric grouping execute",
            "tune noninferiority and heldout strict-recall/noninferior-OCF gates execute",
        ],
        "tiny_training_histories": histories,
    }, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Run the full preregistered real-data experiment")
    mode.add_argument("--smoke-test", action="store_true", help="Run synthetic CPU tests only")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        run_real()


if __name__ == "__main__":
    main()
