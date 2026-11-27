"""Preregistered Task2 forward-CNN seed ensemble; public training data only.

The default command runs the full four-fold experiment. Use --smoke-test for
small synthetic checks without loading the dataset or training any model.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREREG_PATH = OUT / "preregistration.json"
spec = importlib.util.spec_from_file_location(
    "exp40_for_cnn_seed_ensemble", ROOT / "test/05_model_architecture/40_forward_cnn_day_readout/run.py")
exp40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp40)
exp25 = exp40.exp25
exp24 = exp40.exp24
e = exp40.e
DEVICE = exp40.DEVICE
DAYS = ("session1", "session2", "session3", "session4")
MICE = ("mice1", "mice2")
CNN_SEEDS = (42, 202, 777)
ARMS = ("existing_cnn", "cnn_seed_ensemble")
METRICS = ("final_ocf", "final_top1", "recall_at_1", "recall_at_32", "oracle32_ocf")
REPEATS = 4
BATCH = 128
EPOCHS = 12
REPRODUCE_COMMAND = ".venv\\Scripts\\python.exe -B -u test\\41_forward_cnn_seed_ensemble\\run.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def validate_preregistration(prereg):
    if prereg.get("status") != "preregistered_before_training":
        raise RuntimeError("Preregistration is missing or not locked")
    contract = prereg["fixed_cnn_contract"]
    if contract["epochs"] != EPOCHS or contract["batch_size"] != BATCH:
        raise RuntimeError("CNN epochs or batch size differs from preregistration")
    if contract["optimizer"] != "AdamW(lr=0.002, weight_decay=0.001)":
        raise RuntimeError("CNN optimizer differs from preregistration")
    arms = prereg["fixed_arms"]
    if arms["existing_cnn"]["cnn_seeds"] != [42]:
        raise RuntimeError("Existing-CNN seed differs from preregistration")
    if arms["cnn_seed_ensemble"]["cnn_seeds"] != list(CNN_SEEDS):
        raise RuntimeError("CNN seed set or order differs from preregistration")
    if tuple(e.SEEDS) != CNN_SEEDS:
        raise RuntimeError("Shared MLP seeds differ from the fixed three-seed protocol")
    if prereg["shared_mlp_and_fusion"]["seeds"] != list(CNN_SEEDS):
        raise RuntimeError("Preregistered MLP seeds differ from implementation")
    subsets = prereg["picture_subsets"]
    if subsets["subset_seed"] != 123 or subsets["tune"]["size"] != 256 \
            or subsets["heldout"]["size"] != 512:
        raise RuntimeError("Picture subsets differ from preregistration")


def train_cnn_seed(bank, counts, seed):
    """Experiment 24 training with seed as the only changed argument."""
    torch.manual_seed(seed)
    # Per-glyph sufficient statistics: all trials have equal exposure/repeats.
    y = torch.as_tensor(counts, dtype=torch.float32, device=DEVICE)
    model = exp24.Encoder(y.mean(0)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    history = []
    for _ in range(EPOCHS):
        total = 0.0
        for indices in torch.randperm(len(y), device=DEVICE).split(BATCH):
            rate = model(bank[indices])
            loss = F.poisson_nll_loss(rate, y[indices], log_input=False, full=False)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite CNN training loss for seed {seed}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
        history.append(total / len(y))
    with torch.no_grad():
        rates = torch.cat([model(images) for images in bank.split(BATCH)]).cpu().numpy()
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise RuntimeError(f"Non-finite or non-positive CNN rate for seed {seed}")
    del model, optimizer, y
    e.release_memory()
    return rates, history


def assemble_arm_rates(seed_rates):
    """Reuse seed42 and average predicted rates, never log-rates or scores."""
    if set(seed_rates) != set(CNN_SEEDS):
        raise ValueError("Expected exactly the preregistered three CNN seed rates")
    existing_rate = seed_rates[42]
    ensemble_rate = np.mean(
        np.stack([seed_rates[seed] for seed in CNN_SEEDS], axis=0),
        axis=0, dtype=np.float32)
    if not np.isfinite(ensemble_rate).all() or np.any(ensemble_rate <= 0):
        raise RuntimeError("Non-finite or non-positive ensemble rate")
    rates = {"existing_cnn": existing_rate, "cnn_seed_ensemble": ensemble_rate}
    if rates["existing_cnn"] is not seed_rates[42]:
        raise RuntimeError("Existing-CNN arm must reference the one seed42 rate array")
    return rates


def run_mouse(group, split, held_day, bank, subsets):
    mouse = group[0].mouse
    if [block.session for block in group] != list(DAYS):
        raise ValueError(f"Expected four ordered days for {mouse}")
    source = [block for block in group if block.session != held_day]
    target_block = next(block for block in group if block.session == held_day)
    gallery_size = len(bank)

    means, xs, ys = [], [], []
    repeats = []
    for block in source:
        part = split.blocks[block.key]
        if not len(part.train_trials) or len(part.validation_trials):
            raise ValueError(f"Invalid source split for {block.key}")
        raw = block.features[part.train_trials]
        labels = block.trial_targets[part.train_trials]
        target_ids, counts = e.aggregate_by_target(raw, labels)
        np.testing.assert_array_equal(target_ids, np.arange(gallery_size))
        means.append(counts)
        repeats.extend(np.unique(np.bincount(labels)).tolist())
        normalizer = e.fit_session_normalizer(raw)
        xs.extend((e.apply_session_normalizer(raw, normalizer),
                   e.apply_session_normalizer(counts, normalizer)))
        ys.extend((labels, target_ids))
    if len(set(repeats)) != 1 or repeats[0] != REPEATS:
        raise ValueError("Expected four repeats per source glyph/day")
    pooled_mean = np.mean(means, axis=0)

    # Seed42 occurs once in this loop. Both arms reuse its predicted rate below.
    seed_rates, loss_histories = {}, {}
    for seed in CNN_SEEDS:
        seed_rates[seed], loss_histories[str(seed)] = train_cnn_seed(
            bank, pooled_mean, seed)
    rates = assemble_arm_rates(seed_rates)
    seed_rate_hashes = {str(seed): array_sha256(rate) for seed, rate in seed_rates.items()}
    arm_rate_hashes = {arm: array_sha256(rate) for arm, rate in rates.items()}
    if arm_rate_hashes["existing_cnn"] != seed_rate_hashes["42"]:
        raise RuntimeError("Existing-CNN rate is not the shared seed42 prediction")

    target_part = split.blocks[target_block.key]
    if len(target_part.train_trials) or not len(target_part.validation_trials):
        raise ValueError(f"Invalid target split for {target_block.key}")
    raw = target_block.features[target_part.validation_trials]
    labels = target_block.trial_targets[target_part.validation_trials]
    targets, observations = e.aggregate_by_target(raw, labels)
    np.testing.assert_array_equal(targets, np.arange(gallery_size))
    target_x = e.apply_session_normalizer(observations, e.fit_session_normalizer(raw))

    # This one MLP ensemble is shared by both CNN arms in the mouse-fold.
    x = np.concatenate(xs).astype(np.float32, copy=False)
    y = np.concatenate(ys).astype(np.int64, copy=False)
    logits, mlp_records = [], []
    for seed in e.SEEDS:
        epochs = e.EPOCHS["task2"][mouse]
        model = exp25.fit_mlp(x, y, seed, epochs, gallery_size)
        logits.append(e.predict_logits(model, target_x, device=DEVICE))
        mlp_records.append({"seed": seed, "epochs_run": epochs, "checkpoint_selection": "none"})
        del model
        e.release_memory()
    mlp = e.row_zscore(np.mean(np.stack(logits), axis=0, dtype=np.float32))
    del logits, x, y

    target_mean, target_sums, target_constants = exp40.repeat_statistics(
        target_block, target_part.validation_trials, gallery_size)
    np.testing.assert_array_equal(target_mean, observations)
    constant_rate = pooled_mean.mean(axis=0, dtype=np.float32)
    nll = {
        arm: exp40.full_nll(observations, target_sums, target_constants, rates[arm])
        for arm in ARMS
    }
    nll["constant_rate"] = exp40.full_nll(
        observations, target_sums, target_constants, constant_rate)

    fused, assigned = {}, {}
    for arm in ARMS:
        # Average rates first; only then form this arm's Poisson scores.
        scores = exp40.poisson_scores(observations, rates[arm])
        fused[arm] = 0.5 * mlp + 0.5 * e.row_zscore(scores)
        assigned[arm] = e.unique_assignment(fused[arm])
        if len(np.unique(assigned[arm])) != gallery_size:
            raise RuntimeError(f"{arm} final assignment is not unique")
        del scores

    metrics = {arm: {} for arm in ARMS}
    for subset_name, pictures in subsets.items():
        for arm in ARMS:
            metrics[arm][subset_name] = exp40.measure_subset(
                targets, fused[arm], assigned[arm], pictures)
    deltas = {
        subset_name: {
            metric: metrics["cnn_seed_ensemble"][subset_name][metric]
                    - metrics["existing_cnn"][subset_name][metric]
            for metric in METRICS
        }
        for subset_name in subsets
    }
    record = {
        "mouse": mouse,
        "held_session": held_day,
        "train_sessions": [block.session for block in source],
        "nll": nll,
        "nll_delta_cnn_seed_ensemble_minus_existing_cnn":
            nll["cnn_seed_ensemble"] - nll["existing_cnn"],
        "nll_delta_cnn_seed_ensemble_minus_constant_rate":
            nll["cnn_seed_ensemble"] - nll["constant_rate"],
        "metrics": metrics,
        "heldout_delta_cnn_seed_ensemble_minus_existing_cnn": deltas["heldout"],
        "tune_delta_cnn_seed_ensemble_minus_existing_cnn": deltas["tune"],
    }
    training = {
        "mouse": mouse,
        "held_session": held_day,
        "source_blocks": [block.key for block in source],
        "target_block": target_block.key,
        "units": target_block.n_units,
        "unit_time_outputs": target_block.n_features,
        "source_glyphs_per_day": gallery_size,
        "source_repeats_per_glyph": REPEATS,
        "target_repeats_per_glyph": REPEATS,
        "cnn_training_order": list(CNN_SEEDS),
        "cnn_loss_history_by_seed": loss_histories,
        "cnn_rate_sha256_by_seed": seed_rate_hashes,
        "arm_rate_sha256": arm_rate_hashes,
        "seed42_training_count": 1,
        "seed42_same_array_used_by_both_arms": True,
        "ensemble_aggregation": "arithmetic mean of independent unit-by-time predicted rates before Poisson scoring",
        "mlp": mlp_records,
        "mlp_shared_by_cnn_arms": True,
    }
    return record, training


def check_historical_seed42(record, historical):
    """Validate the paired control against experiment 40; never substitute it."""
    old = next((item for item in historical["records"]
                if item["mouse"] == record["mouse"]
                and item["held_session"] == record["held_session"]), None)
    if old is None:
        raise RuntimeError("Missing experiment 40 seed42 control record")
    errors = {}
    for subset in ("tune", "heldout"):
        for metric in METRICS:
            actual = record["metrics"]["existing_cnn"][subset][metric]
            expected = old["metrics"]["existing_cnn"][subset][metric]
            errors[f"{subset}.{metric}"] = abs(actual - expected)
    errors["nll"] = abs(record["nll"]["existing_cnn"] - old["nll"]["existing_cnn"])
    maximum = max(errors.values())
    result = {
        "mouse": record["mouse"],
        "held_session": record["held_session"],
        "maximum_absolute_error": maximum,
        "absolute_errors": errors,
        "tolerance": 1e-5,
        "passed": maximum <= 1e-5,
        "historical_values_substituted": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"Existing-CNN reproduction failed: {result}")
    return result


def mean_metric(records, arm, subset, metric):
    return float(np.mean([record["metrics"][arm][subset][metric] for record in records]))


def summarize(records):
    fold_pooled = {}
    for day in DAYS:
        fold = [record for record in records if record["held_session"] == day]
        fold_pooled[day] = {
            arm: {
                subset: {metric: mean_metric(fold, arm, subset, metric) for metric in METRICS}
                for subset in ("tune", "heldout")
            }
            for arm in ARMS
        }
        fold_pooled[day]["heldout_delta"] = {
            metric: fold_pooled[day]["cnn_seed_ensemble"]["heldout"][metric]
                    - fold_pooled[day]["existing_cnn"]["heldout"][metric]
            for metric in METRICS
        }
        fold_pooled[day]["nll"] = {
            arm: float(np.mean([record["nll"][arm] for record in fold]))
            for arm in ("existing_cnn", "cnn_seed_ensemble", "constant_rate")
        }
        fold_pooled[day]["nll_delta_ensemble_minus_existing"] = (
            fold_pooled[day]["nll"]["cnn_seed_ensemble"]
            - fold_pooled[day]["nll"]["existing_cnn"])
    macro = {
        arm: {
            subset: {
                metric: float(np.mean([fold_pooled[day][arm][subset][metric] for day in DAYS]))
                for metric in METRICS
            }
            for subset in ("tune", "heldout")
        }
        for arm in ARMS
    }
    macro["heldout_delta"] = {
        metric: macro["cnn_seed_ensemble"]["heldout"][metric]
                - macro["existing_cnn"]["heldout"][metric]
        for metric in METRICS
    }
    macro["nll"] = {
        arm: float(np.mean([fold_pooled[day]["nll"][arm] for day in DAYS]))
        for arm in ("existing_cnn", "cnn_seed_ensemble", "constant_rate")
    }
    macro["nll_delta_ensemble_minus_existing"] = (
        macro["nll"]["cnn_seed_ensemble"] - macro["nll"]["existing_cnn"])
    per_mouse = {}
    for mouse in MICE:
        group = [record for record in records if record["mouse"] == mouse]
        per_mouse[mouse] = {
            arm: {
                subset: {metric: mean_metric(group, arm, subset, metric) for metric in METRICS}
                for subset in ("tune", "heldout")
            }
            for arm in ARMS
        }
        per_mouse[mouse]["heldout_delta"] = {
            metric: per_mouse[mouse]["cnn_seed_ensemble"]["heldout"][metric]
                    - per_mouse[mouse]["existing_cnn"]["heldout"][metric]
            for metric in METRICS
        }
        per_mouse[mouse]["nll"] = {
            arm: float(np.mean([record["nll"][arm] for record in group]))
            for arm in ("existing_cnn", "cnn_seed_ensemble", "constant_rate")
        }
        per_mouse[mouse]["nll_delta_ensemble_minus_existing"] = (
            per_mouse[mouse]["nll"]["cnn_seed_ensemble"]
            - per_mouse[mouse]["nll"]["existing_cnn"])
    return fold_pooled, macro, per_mouse


def advancement_gate(records, fold_pooled, macro, per_mouse):
    fold_deltas = {day: fold_pooled[day]["heldout_delta"]["final_ocf"] for day in DAYS}
    positive_folds = [day for day, delta in fold_deltas.items() if delta > 0]
    per_mouse_checks = {
        mouse: {
            "final_ocf_delta": values["heldout_delta"]["final_ocf"],
            "recall_at_32_delta": values["heldout_delta"]["recall_at_32"],
            "final_ocf_passed": values["heldout_delta"]["final_ocf"] > 0,
            "recall_at_32_passed": values["heldout_delta"]["recall_at_32"] >= 0,
            "passed": values["heldout_delta"]["final_ocf"] > 0
                      and values["heldout_delta"]["recall_at_32"] >= 0,
        }
        for mouse, values in per_mouse.items()
    }
    per_mouse_nll = {
        mouse: {
            "existing_cnn": values["nll"]["existing_cnn"],
            "cnn_seed_ensemble": values["nll"]["cnn_seed_ensemble"],
            "constant_rate": values["nll"]["constant_rate"],
            "delta_ensemble_minus_existing": values["nll_delta_ensemble_minus_existing"],
            "passed": values["nll_delta_ensemble_minus_existing"] <= 0,
        }
        for mouse, values in per_mouse.items()
    }
    nll_mouse_folds = [
        {
            "mouse": record["mouse"],
            "held_session": record["held_session"],
            "existing_cnn": record["nll"]["existing_cnn"],
            "cnn_seed_ensemble": record["nll"]["cnn_seed_ensemble"],
            "constant_rate": record["nll"]["constant_rate"],
            "delta_ensemble_minus_existing":
                record["nll_delta_cnn_seed_ensemble_minus_existing_cnn"],
            "delta_ensemble_minus_constant_rate":
                record["nll_delta_cnn_seed_ensemble_minus_constant_rate"],
        }
        for record in records
    ]
    macro_nll_delta = macro["nll_delta_ensemble_minus_existing"]
    checks = {
        "macro_final_ocf_delta": macro["heldout_delta"]["final_ocf"],
        "macro_final_ocf_passed": macro["heldout_delta"]["final_ocf"] > 0,
        "positive_fold_count": len(positive_folds),
        "positive_folds": positive_folds,
        "fold_consistency_passed": len(positive_folds) >= 3,
        "fold_final_ocf_deltas": fold_deltas,
        "per_mouse_decoding": per_mouse_checks,
        "per_mouse_decoding_passed": all(value["passed"] for value in per_mouse_checks.values()),
        "macro_nll_delta_ensemble_minus_existing": macro_nll_delta,
        "macro_nll_passed": macro_nll_delta < 0,
        "per_mouse_nll": per_mouse_nll,
        "per_mouse_nll_passed": all(value["passed"] for value in per_mouse_nll.values()),
        "nll_mouse_folds": nll_mouse_folds,
    }
    checks["decoding_gate_passed"] = (
        checks["macro_final_ocf_passed"] and checks["fold_consistency_passed"]
        and checks["per_mouse_decoding_passed"])
    checks["nll_gate_passed"] = checks["macro_nll_passed"] and checks["per_mouse_nll_passed"]
    checks["advance"] = checks["decoding_gate_passed"] and checks["nll_gate_passed"]
    checks["production_modified"] = False
    return checks


def environment():
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": DEVICE,
        "cuda_device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    }


def make_manifest(started, records):
    dependencies = [
        PREREG_PATH,
        OUT / "run.py",
        ROOT / "test/05_model_architecture/40_forward_cnn_day_readout/run.py",
        ROOT / "test/05_model_architecture/40_forward_cnn_day_readout/results.json",
        ROOT / "test/07_protocol_and_scoring/25_rotating_sessions/run.py",
        ROOT / "test/05_model_architecture/24_forward_cnn/run.py",
        ROOT / "test/06_fusion_transduction/18_seed_ensemble/run.py",
    ]
    outputs = [OUT / "results.json", OUT / "training_summary.json"]
    manifest = {
        "experiment": "41_forward_cnn_seed_ensemble",
        "status": "complete",
        "reproduce_from_repository_root": REPRODUCE_COMMAND,
        "preregistration_sha256": sha256(PREREG_PATH),
        "git_commit_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": environment(),
        "dependencies": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in dependencies
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
        "mouse_fold_count": len(records),
        "cnn_seeds": list(CNN_SEEDS),
        "seed42_training_count_per_mouse_fold": 1,
        "cnn_rates_or_checkpoints_saved": False,
        "ocf_cache": {
            "implementation": "test/05_model_architecture/40_forward_cnn_day_readout/run.py cached_ocf",
            "scope": "current process, shared across arms and folds",
            "historical_persistent_cache_reused": False,
            "entries_at_manifest": len(exp40.OCF_CACHE),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "artifact_manifest.json", manifest)


def main():
    started = time.perf_counter()
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    validate_preregistration(prereg)
    prereg_hash = sha256(PREREG_PATH)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    images = (e.load_images(e.DATA) >= 128).astype(np.float32)
    if images.shape != (2748, 64, 64):
        raise ValueError("Unexpected public training gallery")
    exp40.IMAGES = images
    exp40.OCF_CACHE.clear()
    bank = torch.from_numpy(images[:, None]).to(DEVICE)
    order = np.random.default_rng(123).permutation(len(images))
    subsets = {"tune": np.sort(order[:256]), "heldout": np.sort(order[256:768])}
    historical = json.loads(
        (ROOT / "test/05_model_architecture/40_forward_cnn_day_readout/results.json").read_text(encoding="utf-8"))
    for name in subsets:
        np.testing.assert_array_equal(subsets[name], historical["picture_ids"][name])

    blocks = e.load_feature_blocks(e.DATA, "train", bin_ms=50)
    if sorted(block.key for block in blocks) != [
            f"{mouse}/{day}" for mouse in MICE for day in DAYS]:
        raise ValueError("Expected all eight public mouse/day training blocks")
    records, training_records, reproduction = [], [], {}
    for day in DAYS:
        split = e.make_split(blocks, "task2", seed=42, validation_session=day)
        for mouse in MICE:
            print(f"START {day} {mouse}", flush=True)
            group = [block for block in blocks if block.mouse == mouse]
            record, training = run_mouse(group, split, day, bank, subsets)
            records.append(record)
            training_records.append(training)
            key = f"{mouse}/{day}"
            reproduction[key] = check_historical_seed42(record, historical)
            write_json(OUT / "training_summary.json", {
                "experiment": "41_forward_cnn_seed_ensemble",
                "status": "running",
                "preregistration_sha256": prereg_hash,
                "completed_mouse_folds": len(training_records),
                "records": training_records,
                "historical_seed42_reproduction": reproduction,
            })
            print("DONE", day, mouse, json.dumps({
                "nll": record["nll"],
                "heldout_delta": record["heldout_delta_cnn_seed_ensemble_minus_existing_cnn"],
                "seed42_reproduction_max_abs_error": reproduction[key]["maximum_absolute_error"],
            }), flush=True)

    fold_pooled, macro, per_mouse = summarize(records)
    gate = advancement_gate(records, fold_pooled, macro, per_mouse)
    env = environment()
    result = {
        "experiment": "41_forward_cnn_seed_ensemble",
        "status": "complete",
        "preregistration_sha256": prereg_hash,
        "protocol": {
            "folds": "four leave-one-day-out folds x two mice; all three CNN seeds and shared three-seed MLP retrained, including session4",
            "cnn_seeds": list(CNN_SEEDS),
            "seed42_trained_once_and_rate_shared_by_both_arms": True,
            "ensemble_rate": "arithmetic mean of unit-by-time predicted rates before Poisson scoring",
            "target_day_responses_in_rate_fitting": False,
            "hidden_labels_or_id_alignment": False,
            "assignment": "full 2748-row Hungarian unique assignment per arm before picture subsetting",
            "oracle_recall_and_final_are_separate": True,
            "full_nll_includes_log_factorial": True,
            "constant_rate_control_retained": True,
            "historical_reuse_limitation": prereg["picture_subsets"]["historical_reuse_limitation"],
        },
        "picture_ids": {name: values.tolist() for name, values in subsets.items()},
        "environment": env,
        "records": records,
        "fold_pooled": fold_pooled,
        "macro_mean_of_four_folds": macro,
        "per_mouse_mean_of_four_folds": per_mouse,
        "historical_seed42_reproduction_against_experiment40": reproduction,
        "advancement_gate": gate,
        "advance": gate["advance"],
        "production_modified": False,
        "reproduce_command": REPRODUCE_COMMAND,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "results.json", result)
    write_json(OUT / "training_summary.json", {
        "experiment": "41_forward_cnn_seed_ensemble",
        "status": "complete",
        "preregistration_sha256": prereg_hash,
        "environment": env,
        "configuration": {
            "cnn_seeds": list(CNN_SEEDS),
            "cnn_epochs": EPOCHS,
            "cnn_batch_size": BATCH,
            "cnn_optimizer": "AdamW(lr=0.002, weight_decay=0.001)",
            "mlp_seeds": list(e.SEEDS),
            "mlp_epochs_by_mouse": e.EPOCHS["task2"],
            "source_repeats": REPEATS,
            "full_evaluation_nll_includes_log_factorial": True,
            "ensemble_rate_aggregation": "arithmetic",
        },
        "completed_mouse_folds": len(training_records),
        "records": training_records,
        "historical_seed42_reproduction_against_experiment40": reproduction,
        "runtime_seconds": time.perf_counter() - started,
    })
    make_manifest(started, records)
    print("MACRO_HELDOUT", json.dumps(macro["heldout_delta"]), flush=True)
    print("GATE", json.dumps(gate), flush=True)
    print(f"WROTE {OUT / 'results.json'}", flush=True)


def synthetic_records(ocf_deltas=None, nll_deltas=None, recall32_deltas=None):
    """Small in-memory records for gate logic; no data or model is touched."""
    ocf_deltas = ocf_deltas or {day: 0.01 for day in DAYS}
    nll_deltas = nll_deltas or {mouse: -0.01 for mouse in MICE}
    recall32_deltas = recall32_deltas or {mouse: 0.0 for mouse in MICE}
    records = []
    for day in DAYS:
        for mouse in MICE:
            base = {metric: 0.5 for metric in METRICS}
            ensemble = dict(base)
            ensemble["final_ocf"] += ocf_deltas[day]
            ensemble["recall_at_32"] += recall32_deltas[mouse]
            metrics = {
                "existing_cnn": {"tune": dict(base), "heldout": dict(base)},
                "cnn_seed_ensemble": {"tune": dict(ensemble), "heldout": dict(ensemble)},
            }
            nll = {
                "existing_cnn": 2.0,
                "cnn_seed_ensemble": 2.0 + nll_deltas[mouse],
                "constant_rate": 3.0,
            }
            records.append({
                "mouse": mouse,
                "held_session": day,
                "nll": nll,
                "nll_delta_cnn_seed_ensemble_minus_existing_cnn": nll_deltas[mouse],
                "nll_delta_cnn_seed_ensemble_minus_constant_rate":
                    nll["cnn_seed_ensemble"] - nll["constant_rate"],
                "metrics": metrics,
            })
    return records


def evaluate_synthetic_gate(records):
    fold_pooled, macro, per_mouse = summarize(records)
    return advancement_gate(records, fold_pooled, macro, per_mouse)


def smoke_test():
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    validate_preregistration(prereg)

    seed_rates = {
        42: np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32),
        202: np.array([[3.0, 4.0], [6.0, 10.0]], dtype=np.float32),
        777: np.array([[5.0, 6.0], [8.0, 12.0]], dtype=np.float32),
    }
    rates = assemble_arm_rates(seed_rates)
    expected = np.mean(np.stack(list(seed_rates.values())), axis=0, dtype=np.float32)
    np.testing.assert_array_equal(rates["cnn_seed_ensemble"], expected)
    assert rates["existing_cnn"] is seed_rates[42]
    assert not np.allclose(rates["cnn_seed_ensemble"], np.exp(np.mean(
        np.log(np.stack(list(seed_rates.values()))), axis=0)))

    # One glyph, one bin, four counts [0,1,2,3], rate=2:
    # NLL = (4*2 - 6*log(2) + log(0!*1!*2!*3!))/4.
    mean_counts = np.array([[1.5]], dtype=np.float32)
    sums = np.array([[6.0]], dtype=np.float32)
    constants = np.array([[math.log(12.0)]], dtype=np.float32)
    nll = exp40.full_nll(mean_counts, sums, constants, np.array([[2.0]], dtype=np.float32))
    expected_nll = (8.0 - 6.0 * math.log(2.0) + math.log(12.0)) / 4.0
    np.testing.assert_allclose(nll, expected_nll, rtol=0, atol=1e-6)
    without_factorial = (8.0 - 6.0 * math.log(2.0)) / 4.0
    assert abs(nll - without_factorial) > 0.5

    passing = synthetic_records(
        ocf_deltas={"session1": 0.02, "session2": 0.01, "session3": 0.0, "session4": 0.01})
    gate = evaluate_synthetic_gate(passing)
    assert gate["advance"] and gate["positive_fold_count"] == 3

    too_few_positive_folds = synthetic_records(
        ocf_deltas={"session1": 0.02, "session2": -0.01, "session3": 0.0, "session4": 0.01})
    gate = evaluate_synthetic_gate(too_few_positive_folds)
    assert not gate["fold_consistency_passed"] and not gate["advance"]

    # Macro NLL improves through mice2, but mice1 violates its own <=0 condition.
    mouse_nll_failure = synthetic_records(nll_deltas={"mice1": 0.02, "mice2": -0.06})
    gate = evaluate_synthetic_gate(mouse_nll_failure)
    assert gate["macro_nll_passed"] and not gate["per_mouse_nll_passed"] and not gate["advance"]

    equal_macro_nll = synthetic_records(nll_deltas={"mice1": 0.0, "mice2": 0.0})
    gate = evaluate_synthetic_gate(equal_macro_nll)
    assert not gate["macro_nll_passed"] and not gate["advance"]

    recall_failure = synthetic_records(recall32_deltas={"mice1": -0.01, "mice2": 0.0})
    gate = evaluate_synthetic_gate(recall_failure)
    assert not gate["per_mouse_decoding_passed"] and not gate["advance"]

    print(json.dumps({
        "status": "passed",
        "checks": [
            "preregistration_configuration",
            "seed42_single_array_reuse",
            "arithmetic_rate_not_log_rate_ensemble",
            "full_poisson_nll_includes_log_factorial",
            "three_of_four_fold_rule",
            "macro_and_per_mouse_nll_vetoes",
            "per_mouse_recall32_veto",
        ],
        "dataset_loaded": False,
        "models_trained": False,
        "gpu_kernels_launched_by_smoke": False,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke_test:
        smoke_test()
    else:
        main()
