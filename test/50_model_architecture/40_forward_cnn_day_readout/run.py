"""Preregistered Task2 source-day CNN readout experiment; public training data only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from scipy.special import gammaln
import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREREG_PATH = OUT / "preregistration.json"
spec = importlib.util.spec_from_file_location("exp25_for_day_readout", ROOT / "test/70_protocol_and_scoring/25_rotating_sessions/run.py")
exp25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp25)
exp24 = exp25.exp24
e = exp25.e
DEVICE = exp25.DEVICE
DAYS = ("session1", "session2", "session3", "session4")
MICE = ("mice1", "mice2")
ARMS = ("existing_cnn", "day_readout")
METRICS = ("final_ocf", "final_top1", "recall_at_1", "recall_at_32", "oracle32_ocf")
REPEATS = 4
BATCH = 128
EPOCHS = 12
CNN_SEED = 42
OCF_CACHE = {}
IMAGES = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class DayReadout(nn.Module):
    def __init__(self, mean_rate, n_days):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, 8, kernel, stride=2, padding=kernel // 2),
                nn.SiLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(8, 12, 3, padding=1),
                nn.SiLU(),
                nn.AvgPool2d(4),
            )
            for kernel in (3, 7)
        )
        self.head = nn.Linear(384, mean_rate.size)
        with torch.no_grad():
            self.head.bias.copy_(torch.as_tensor(mean_rate).clamp_min(0.001).expm1().log())
        self.gain = nn.Parameter(torch.ones(n_days, mean_rate.size))
        self.bias = nn.Parameter(torch.zeros(n_days, mean_rate.size))

    def forward(self, images, day):
        features = torch.cat([branch(images).flatten(1) for branch in self.branches], dim=1)
        base_log_rate = torch.log(F.softplus(self.head(features)) + 1e-6)
        return torch.exp(self.gain[day] * base_log_rate + self.bias[day])


def repeat_statistics(block, trial_indices, gallery_size):
    labels = block.trial_targets[trial_indices]
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    expected = np.repeat(np.arange(gallery_size), REPEATS)
    if not np.array_equal(sorted_labels, expected):
        raise ValueError(f"Expected {REPEATS} ordered repeats per glyph in {block.key}")
    raw = block.features[trial_indices[order]].reshape(gallery_size, REPEATS, block.n_features)
    sums = raw.sum(axis=1, dtype=np.float32)
    constants = gammaln(raw + 1).sum(axis=1, dtype=np.float32)
    return sums / REPEATS, sums, constants


@torch.no_grad()
def predict_day_rates(model, bank, n_days):
    days = []
    for day in range(n_days):
        parts = [model(images, day) for images in bank.split(BATCH)]
        days.append(torch.cat(parts))
    if not all(torch.isfinite(rate).all() and torch.all(rate > 0) for rate in days):
        raise RuntimeError("Non-finite or non-positive predicted source-day rate")
    source_means = [float(rate.mean().cpu()) for rate in days]
    target = torch.stack(days).mean(dim=0).cpu().numpy()
    return target, source_means


def train_day_readout(bank, source_sums, source_constants, pooled_mean):
    torch.manual_seed(CNN_SEED)
    model = DayReadout(pooled_mean.mean(axis=0), len(source_sums)).to(DEVICE)
    decay = [parameter for name, parameter in model.named_parameters() if name not in ("gain", "bias")]
    readout = [model.gain, model.bias]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.001},
        {"params": readout, "weight_decay": 0.0},
    ], lr=0.002)
    sums = torch.from_numpy(np.stack(source_sums)).to(DEVICE)
    constants = torch.from_numpy(np.stack(source_constants)).to(DEVICE)
    history = []
    for _ in range(EPOCHS):
        total = 0.0
        for day in range(len(source_sums)):
            for indices in torch.randperm(len(bank), device=DEVICE).split(BATCH):
                rate = model(bank[indices], day)
                loss = ((REPEATS * rate - sums[day, indices] * torch.log(rate)
                         + constants[day, indices]) / REPEATS).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite day-readout training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(indices)
        history.append(total / (len(source_sums) * len(bank)))
    target_rate, source_rate_means = predict_day_rates(model, bank, len(source_sums))
    summary = {
        "full_nll_history": history,
        "gain_mean_absolute_deviation_from_1": float((model.gain.detach() - 1).abs().mean().cpu()),
        "bias_mean_absolute": float(model.bias.detach().abs().mean().cpu()),
        "source_predicted_rate_means": source_rate_means,
        "target_rate_is_arithmetic_mean_of_predicted_source_rates": True,
    }
    del model, optimizer, sums, constants
    e.release_memory()
    return target_rate, summary


def full_nll(mean_counts, sums, constants, rate):
    rate = np.asarray(rate, dtype=np.float32)
    if rate.ndim == 1 and rate.shape == (mean_counts.shape[1],):
        rate = np.broadcast_to(rate, mean_counts.shape)
    if rate.shape != mean_counts.shape or not np.isfinite(rate).all() or np.any(rate <= 0):
        raise ValueError("Invalid rates for full Poisson NLL")
    terms = (REPEATS * rate - sums * np.log(rate) + constants) / REPEATS
    return float(terms.mean(dtype=np.float64))


def poisson_scores(observations, rate):
    rate = np.maximum(rate, 1e-6)
    return ((observations.astype(np.float32) @ np.log(rate).T - rate.sum(1))
            * REPEATS).astype(np.float32)


def cached_ocf(predicted, targets):
    predicted = np.asarray(predicted, dtype=np.int64).ravel()
    targets = np.asarray(targets, dtype=np.int64).ravel()
    if predicted.shape != targets.shape:
        raise ValueError("Mismatched OCF pair arrays")
    pairs = list(zip(predicted.tolist(), targets.tolist()))
    missing = sorted({pair for pair in pairs if pair not in OCF_CACHE})
    for start in range(0, len(missing), 256):
        chunk = missing[start:start + 256]
        pp = [pair[0] for pair in chunk]
        tt = [pair[1] for pair in chunk]
        scores = e.score_predictions(IMAGES[pp], IMAGES[tt], device=DEVICE, batch_size=64)
        for pair, score in zip(chunk, scores):
            OCF_CACHE[pair] = float(score)
    return np.asarray([OCF_CACHE[pair] for pair in pairs], dtype=np.float64)


def measure_subset(targets, fused, assigned, pictures):
    mask = np.isin(targets, pictures)
    true = targets[mask]
    scores = fused[mask]
    prediction = assigned[mask]
    truth_score = scores[np.arange(len(true)), true, None]
    candidate_ids = np.arange(fused.shape[1])
    ranks = (1 + (scores > truth_score).sum(1)
             + ((scores == truth_score) & (candidate_ids < true[:, None])).sum(1))
    top32 = np.argsort(-scores, axis=1, kind="stable")[:, :32]
    oracle_pairs = cached_ocf(top32.ravel(), np.repeat(true, 32)).reshape(len(true), 32)
    final_ocf = cached_ocf(prediction, true)
    return {
        "n": int(len(true)),
        "final_ocf": float(final_ocf.mean()),
        "final_top1": float((prediction == true).mean()),
        "recall_at_1": float((ranks == 1).mean()),
        "recall_at_32": float((ranks <= 32).mean()),
        "oracle32_ocf": float(oracle_pairs.max(axis=1).mean()),
    }


def run_mouse(group, split, held_day, bank, coeff, subsets):
    mouse = group[0].mouse
    if [block.session for block in group] != list(DAYS):
        raise ValueError(f"Expected four ordered days for {mouse}")
    source = [block for block in group if block.session != held_day]
    target_block = next(block for block in group if block.session == held_day)
    gallery_size = len(bank)

    # Match experiment 25 through the existing arm's complete unlabeled scores.
    means, ids, repeats = [], [], []
    xs, ys = [], []
    for block in source:
        part = split.blocks[block.key]
        if not len(part.train_trials) or len(part.validation_trials):
            raise ValueError(f"Invalid source split for {block.key}")
        raw = block.features[part.train_trials]
        labels = block.trial_targets[part.train_trials]
        target_ids, counts = e.aggregate_by_target(raw, labels)
        np.testing.assert_array_equal(target_ids, np.arange(gallery_size))
        means.append(counts)
        ids.append(target_ids)
        repeats.extend(np.unique(np.bincount(labels)).tolist())
        normalizer = e.fit_session_normalizer(raw)
        xs.extend((e.apply_session_normalizer(raw, normalizer),
                   e.apply_session_normalizer(counts, normalizer)))
        ys.extend((labels, target_ids))
    if len(set(repeats)) != 1 or repeats[0] != REPEATS:
        raise ValueError("Expected four repeats per source glyph/day")
    pooled_mean = np.mean(means, axis=0)
    ridge = e.RidgePath().fit(
        np.tile(coeff, (len(means), 1)), np.log1p(np.concatenate(means))
    ).predict(coeff, 1.0)
    existing_rate, existing_loss = exp24.train(bank, pooled_mean)

    target_part = split.blocks[target_block.key]
    if len(target_part.train_trials) or not len(target_part.validation_trials):
        raise ValueError(f"Invalid target split for {target_block.key}")
    raw = target_block.features[target_part.validation_trials]
    labels = target_block.trial_targets[target_part.validation_trials]
    targets, observations = e.aggregate_by_target(raw, labels)
    np.testing.assert_array_equal(targets, np.arange(gallery_size))
    target_x = e.apply_session_normalizer(observations, e.fit_session_normalizer(raw))
    x = np.concatenate(xs).astype(np.float32, copy=False)
    y = np.concatenate(ys).astype(np.int64, copy=False)
    logits = []
    mlp_records = []
    for seed in e.SEEDS:
        epochs = e.EPOCHS["task2"][mouse]
        model = exp25.fit_mlp(x, y, seed, epochs, gallery_size)
        logits.append(e.predict_logits(model, target_x, device=DEVICE))
        mlp_records.append({"seed": seed, "epochs_run": epochs, "checkpoint_selection": "none"})
        del model
        e.release_memory()
    mlp = e.row_zscore(np.mean(np.stack(logits), axis=0, dtype=np.float32))
    del logits, x, y
    existing_scores = poisson_scores(observations, existing_rate)
    existing_fused = 0.5 * mlp + 0.5 * e.row_zscore(existing_scores)
    existing_assigned = e.unique_assignment(existing_fused)
    if len(np.unique(existing_assigned)) != gallery_size:
        raise RuntimeError("Existing-CNN final assignment is not unique")
    del existing_scores

    # Only source-day responses enter the candidate fit. No metric is read before it.
    source_sums, source_constants = [], []
    for source_index, block in enumerate(source):
        part = split.blocks[block.key]
        check_mean, sums, constants = repeat_statistics(block, part.train_trials, gallery_size)
        np.testing.assert_array_equal(check_mean, means[source_index])
        source_sums.append(sums)
        source_constants.append(constants)
    day_rate, day_summary = train_day_readout(bank, source_sums, source_constants, pooled_mean)
    constant_rate = pooled_mean.mean(axis=0, dtype=np.float32)
    target_mean, target_sums, target_constants = repeat_statistics(
        target_block, target_part.validation_trials, gallery_size)
    np.testing.assert_array_equal(target_mean, observations)

    rates = {"existing_cnn": existing_rate, "day_readout": day_rate}
    nll = {name: full_nll(observations, target_sums, target_constants, rate)
           for name, rate in rates.items()}
    nll["constant_rate"] = full_nll(observations, target_sums, target_constants, constant_rate)
    metrics = {name: {} for name in ARMS}
    for subset_name, pictures in subsets.items():
        metrics["existing_cnn"][subset_name] = measure_subset(
            targets, existing_fused, existing_assigned, pictures)
    day_scores = poisson_scores(observations, day_rate)
    day_fused = 0.5 * mlp + 0.5 * e.row_zscore(day_scores)
    day_assigned = e.unique_assignment(day_fused)
    if len(np.unique(day_assigned)) != gallery_size:
        raise RuntimeError("Day-readout final assignment is not unique")
    for subset_name, pictures in subsets.items():
        metrics["day_readout"][subset_name] = measure_subset(
            targets, day_fused, day_assigned, pictures)
    deltas = {
        subset_name: {metric: metrics["day_readout"][subset_name][metric]
                      - metrics["existing_cnn"][subset_name][metric] for metric in METRICS}
        for subset_name in subsets
    }
    record = {
        "mouse": mouse,
        "held_session": held_day,
        "train_sessions": [block.session for block in source],
        "nll": nll,
        "nll_delta_day_readout_minus_constant": nll["day_readout"] - nll["constant_rate"],
        "nll_delta_day_readout_minus_existing_cnn": nll["day_readout"] - nll["existing_cnn"],
        "metrics": metrics,
        "heldout_delta_day_readout_minus_existing_cnn": deltas["heldout"],
        "tune_delta_day_readout_minus_existing_cnn": deltas["tune"],
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
        "existing_cnn_historical_loss_history": existing_loss,
        "day_readout": day_summary,
        "mlp": mlp_records,
        "target_rate_source": "CNN predictions only; no observed target-day or averaged observed source-day prototype",
    }
    return record, training


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
            metric: fold_pooled[day]["day_readout"]["heldout"][metric]
                    - fold_pooled[day]["existing_cnn"]["heldout"][metric]
            for metric in METRICS
        }
        fold_pooled[day]["nll"] = {
            arm: float(np.mean([record["nll"][arm] for record in fold]))
            for arm in ("existing_cnn", "day_readout", "constant_rate")
        }
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
        metric: macro["day_readout"]["heldout"][metric] - macro["existing_cnn"]["heldout"][metric]
        for metric in METRICS
    }
    macro["nll"] = {
        arm: float(np.mean([fold_pooled[day]["nll"][arm] for day in DAYS]))
        for arm in ("existing_cnn", "day_readout", "constant_rate")
    }
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
            metric: per_mouse[mouse]["day_readout"]["heldout"][metric]
                    - per_mouse[mouse]["existing_cnn"]["heldout"][metric]
            for metric in METRICS
        }
    return fold_pooled, macro, per_mouse


def check_historical_baseline(day, fold_pooled, historical):
    if day == "session4":
        return None
    subset_map = {"tune": "tune", "heldout": "held_out"}
    metric_map = {"final_ocf": "ocf", "final_top1": "top1",
                  "recall_at_1": "recall1", "recall_at_32": "recall32"}
    errors = {}
    for subset, old_subset in subset_map.items():
        for metric, old_metric in metric_map.items():
            actual = fold_pooled[day]["existing_cnn"][subset][metric]
            expected = historical["new_folds"][day]["cnn_poisson"][old_subset][old_metric]
            errors[f"{subset}.{metric}"] = abs(actual - expected)
    maximum = max(errors.values())
    result = {"maximum_absolute_error": maximum, "absolute_errors": errors, "passed": maximum <= 1e-5}
    if not result["passed"]:
        raise RuntimeError(f"Existing-CNN reproduction failed for {day}: {result}")
    return result


def advancement_gate(records, fold_pooled, macro, per_mouse):
    fold_deltas = {day: fold_pooled[day]["heldout_delta"]["final_ocf"] for day in DAYS}
    positive_folds = [day for day, delta in fold_deltas.items() if delta > 0]
    per_mouse_checks = {
        mouse: {
            "final_ocf_delta": values["heldout_delta"]["final_ocf"],
            "recall_at_32_delta": values["heldout_delta"]["recall_at_32"],
            "passed": values["heldout_delta"]["final_ocf"] > 0
                      and values["heldout_delta"]["recall_at_32"] >= 0,
        }
        for mouse, values in per_mouse.items()
    }
    nll_checks = [
        {
            "mouse": record["mouse"],
            "held_session": record["held_session"],
            "day_readout": record["nll"]["day_readout"],
            "constant_rate": record["nll"]["constant_rate"],
            "delta": record["nll_delta_day_readout_minus_constant"],
            "passed": record["nll"]["day_readout"] < record["nll"]["constant_rate"],
        }
        for record in records
    ]
    checks = {
        "macro_final_ocf_delta": macro["heldout_delta"]["final_ocf"],
        "macro_final_ocf_passed": macro["heldout_delta"]["final_ocf"] > 0,
        "positive_fold_count": len(positive_folds),
        "positive_folds": positive_folds,
        "fold_consistency_passed": len(positive_folds) >= 3,
        "fold_final_ocf_deltas": fold_deltas,
        "per_mouse": per_mouse_checks,
        "per_mouse_passed": all(value["passed"] for value in per_mouse_checks.values()),
        "nll_mouse_folds": nll_checks,
        "nll_passed_count": sum(check["passed"] for check in nll_checks),
        "nll_day_conditioning_passed": all(check["passed"] for check in nll_checks),
    }
    metric_passed = (checks["macro_final_ocf_passed"] and checks["fold_consistency_passed"]
                     and checks["per_mouse_passed"])
    checks["metric_gate_passed"] = metric_passed
    checks["day_conditioning_supported"] = checks["nll_day_conditioning_passed"]
    checks["advance"] = metric_passed and checks["nll_day_conditioning_passed"]
    checks["production_modified"] = False
    return checks


def make_manifest(started, records):
    dependencies = [
        PREREG_PATH,
        OUT / "run.py",
        ROOT / "test/60_fusion_transduction/18_seed_ensemble/run.py",
        ROOT / "test/50_model_architecture/24_forward_cnn/run.py",
        ROOT / "test/70_protocol_and_scoring/25_rotating_sessions/run.py",
    ]
    outputs = [OUT / "results.json", OUT / "training_summary.json"]
    manifest = {
        "experiment": "40_forward_cnn_day_readout",
        "status": "complete",
        "reproduce_from_repository_root": ".venv\\Scripts\\python.exe -B -u test\\40_forward_cnn_day_readout\\run.py",
        "preregistration_sha256": sha256(PREREG_PATH),
        "git_commit_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": DEVICE,
            "cuda_device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        },
        "dependencies": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in dependencies
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
        "mouse_fold_count": len(records),
        "checkpoints_saved": False,
        "ocf_pair_cache_entries_at_manifest": len(OCF_CACHE),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "artifact_manifest.json", manifest)


def main():
    global IMAGES
    started = time.perf_counter()
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_training":
        raise RuntimeError("Preregistration is missing or not locked")
    prereg_hash = sha256(PREREG_PATH)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    IMAGES = (e.load_images(e.DATA) >= 128).astype(np.float32)
    if IMAGES.shape != (2748, 64, 64):
        raise ValueError("Unexpected public training gallery")
    bank = torch.from_numpy(IMAGES[:, None]).to(DEVICE)
    coeff = e.dct_encode(IMAGES, 16)
    order = np.random.default_rng(123).permutation(len(IMAGES))
    subsets = {"tune": np.sort(order[:256]), "heldout": np.sort(order[256:768])}
    blocks = e.load_feature_blocks(e.DATA, "train", bin_ms=50)
    if sorted(block.key for block in blocks) != [f"{mouse}/{day}" for mouse in MICE for day in DAYS]:
        raise ValueError("Expected all eight public mouse/day training blocks")
    historical = json.loads((ROOT / "test/70_protocol_and_scoring/25_rotating_sessions/results.json").read_text(encoding="utf-8"))
    records, training_records, reproduction = [], [], {}
    for day in DAYS:
        split = e.make_split(blocks, "task2", seed=42, validation_session=day)
        for mouse in MICE:
            print(f"START {day} {mouse}", flush=True)
            group = [block for block in blocks if block.mouse == mouse]
            record, training = run_mouse(group, split, day, bank, coeff, subsets)
            records.append(record)
            training_records.append(training)
            write_json(OUT / "training_summary.json", {
                "experiment": "40_forward_cnn_day_readout",
                "status": "running",
                "preregistration_sha256": prereg_hash,
                "completed_mouse_folds": len(training_records),
                "records": training_records,
            })
            print("DONE", day, mouse, json.dumps({
                "nll": record["nll"],
                "heldout_delta": record["heldout_delta_day_readout_minus_existing_cnn"],
            }), flush=True)
        fold_records = [record for record in records if record["held_session"] == day]
        partial_fold = {
            arm: {
                subset: {metric: mean_metric(fold_records, arm, subset, metric) for metric in METRICS}
                for subset in ("tune", "heldout")
            }
            for arm in ARMS
        }
        reproduction[day] = check_historical_baseline(day, {day: partial_fold}, historical)
        if reproduction[day] is not None:
            print("BASELINE_REPRODUCTION", day, reproduction[day]["maximum_absolute_error"], flush=True)

    fold_pooled, macro, per_mouse = summarize(records)
    gate = advancement_gate(records, fold_pooled, macro, per_mouse)
    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": DEVICE,
        "cuda_device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    }
    result = {
        "experiment": "40_forward_cnn_day_readout",
        "status": "complete",
        "preregistration_sha256": prereg_hash,
        "protocol": {
            "folds": "four leave-one-day-out folds x two mice; all CNNs and three-seed MLPs retrained",
            "target_day_responses_in_prototypes_or_rate_fitting": False,
            "target_day_rate": "arithmetic mean of model-predicted source-day rates for day_readout",
            "hidden_labels_or_id_alignment": False,
            "assignment": "full 2748-row Hungarian unique assignment before picture subsetting",
            "oracle_and_final_are_separate": True,
            "historical_reuse_limitation": prereg["picture_subsets"]["historical_reuse_limitation"],
        },
        "picture_ids": {name: values.tolist() for name, values in subsets.items()},
        "environment": environment,
        "records": records,
        "fold_pooled": fold_pooled,
        "macro_mean_of_four_folds": macro,
        "per_mouse_mean_of_four_folds": per_mouse,
        "baseline_reproduction_against_experiment25": reproduction,
        "advancement_gate": gate,
        "advance": gate["advance"],
        "production_modified": False,
        "reproduce_command": ".venv\\Scripts\\python.exe -B -u test\\40_forward_cnn_day_readout\\run.py",
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "results.json", result)
    write_json(OUT / "training_summary.json", {
        "experiment": "40_forward_cnn_day_readout",
        "status": "complete",
        "preregistration_sha256": prereg_hash,
        "environment": environment,
        "configuration": {
            "cnn_seed": CNN_SEED,
            "cnn_epochs": EPOCHS,
            "cnn_batch_size": BATCH,
            "mlp_seeds": list(e.SEEDS),
            "mlp_epochs_by_mouse": e.EPOCHS["task2"],
            "source_repeats": REPEATS,
            "full_training_nll_includes_log_factorial": True,
        },
        "completed_mouse_folds": len(training_records),
        "records": training_records,
        "baseline_reproduction_against_experiment25": reproduction,
        "runtime_seconds": time.perf_counter() - started,
    })
    make_manifest(started, records)
    print("MACRO_HELDOUT", json.dumps(macro["heldout_delta"]), flush=True)
    print("GATE", json.dumps(gate), flush=True)
    print(f"WROTE {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
