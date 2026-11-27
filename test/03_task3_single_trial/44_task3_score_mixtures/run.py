"""Frozen public Task3 evaluation of v5 score-top-k weighted glyph mixtures."""

import json
import platform
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import src.final as final
from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions, summarize_scores
from src.features import load_feature_blocks
from src.final import DATA, row_zscore
from src.retrieval import softmax_dct_similarity, unique_assignment
from src.ridge import dct_encode
from src.splits import make_split

OUTPUT = ROOT / "test/03_task3_single_trial/44_task3_score_mixtures"
PREREGISTRATION = OUTPUT / "preregistration.json"


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def relative_path(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def stable_topk(scores, k):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not 1 <= k <= scores.shape[1]:
        raise ValueError("Expected a 2-D score matrix and a valid k.")
    return np.argsort(-scores, axis=1, kind="stable")[:, :k].astype(np.int64)


def render_mixtures(images, ranked_indices, ranked_scores, *, k, temperature, threshold, chunk_size=128):
    """Render independent per-row softmax-weighted binary glyph mixtures."""
    images = np.asarray(images, dtype=np.float32)
    ranked_indices = np.asarray(ranked_indices, dtype=np.int64)
    ranked_scores = np.asarray(ranked_scores, dtype=np.float32)
    if ranked_indices.shape != ranked_scores.shape or ranked_indices.ndim != 2:
        raise ValueError("Ranked indices and scores must have matching 2-D shapes.")
    if not 1 <= k <= ranked_indices.shape[1] or temperature <= 0 or not 0 <= threshold <= 1:
        raise ValueError("Invalid mixture configuration.")
    output = np.empty((len(ranked_indices), *images.shape[1:]), dtype=np.float32)
    for start in range(0, len(output), chunk_size):
        stop = min(start + chunk_size, len(output))
        selected = ranked_indices[start:stop, :k]
        selected_scores = ranked_scores[start:stop, :k]
        logits = (selected_scores - selected_scores.max(axis=1, keepdims=True)) / temperature
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        mixed = np.zeros((stop - start, *images.shape[1:]), dtype=np.float32)
        for candidate in range(k):
            mixed += weights[:, candidate, None, None] * images[selected[:, candidate]]
        output[start:stop] = mixed >= threshold
    return output


def grid_configurations(preregistration):
    family = preregistration["candidate_family"]
    return [
        {
            "grid_index": index,
            "k": k,
            "temperature": temperature,
            "threshold": threshold,
        }
        for index, (k, temperature, threshold) in enumerate(
            (k, temperature, threshold)
            for k in family["k_values"]
            for temperature in family["temperatures"]
            for threshold in family["thresholds"]
        )
    ]


def select_locked_index(candidate_rows):
    """Maximum tune OCF, retaining the first frozen-grid configuration on ties."""
    return max(
        range(len(candidate_rows)),
        key=lambda index: (candidate_rows[index]["tune"]["ocf_mean"], -index),
    )


def advancement(per_mouse_deltas, per_group_deltas):
    mouse_pass = all(value > 0 for value in per_mouse_deltas.values())
    positive_groups = sum(value > 0 for value in per_group_deltas.values())
    return {
        "both_mice_strictly_positive": mouse_pass,
        "positive_group_count": positive_groups,
        "required_positive_groups": 5,
        "advance": bool(mouse_pass and positive_groups >= 5),
    }


def official_scores(predictions, targets, device, batch_size):
    return score_predictions(
        predictions, targets, device=device, batch_size=batch_size
    )


def metric_bundle(scores, groups, mice, targets, score_top1, assigned=None):
    result = summarize_scores(scores, groups)
    result["per_mouse"] = {
        mouse: summarize_scores(scores[mice == mouse])
        for mouse in sorted(set(mice.tolist()))
    }
    result["score_top1_identity_accuracy"] = float(np.mean(score_top1 == targets))
    if assigned is not None:
        result["assigned_identity_accuracy"] = float(np.mean(assigned == targets))
    else:
        result["final_mixed_image_identity_accuracy"] = None
        result["identity_note"] = (
            "A weighted multi-glyph image has no single final predicted identity; "
            "score-top1 is only a common diagnostic of the underlying ranker."
        )
    result["metric"] = "official_ocf_png_parity"
    return result


def validate_frozen_config(config, preregistration):
    task = config["task3"]
    if config["name"] != preregistration["frozen_inputs"]["config_name"]:
        raise RuntimeError("Production config name differs from preregistration.")
    if not task["assignment"] or not task["prototype_continuous"]["enabled"]:
        raise RuntimeError("Task3 assignment and continuous branch must be enabled.")
    if task["prototype_continuous"]["temperature"] != 1.0:
        raise RuntimeError("The frozen continuous branch requires temperature=1.")
    if task.get("seeds", [42]) != [42]:
        raise RuntimeError("Only the frozen seed-42 Task3 checkpoint is permitted.")
    if task.get("reranker", {}).get("enabled", False):
        raise RuntimeError("Task3 reranker use is forbidden in this experiment.")


def capture_production(config, task3_models, images, blocks, split, device, candidate_top_n):
    """Run unmodified production inference while capturing and independently checking scores."""
    records = []
    pending = {}
    coefficients = dct_encode(images, 16)
    expected_centered = coefficients - coefficients.mean(axis=0, keepdims=True)
    prototype_cache = {}
    original_softmax = final.softmax_dct_similarity
    original_assignment = final.unique_assignment

    def forbidden_reranker(*args, **kwargs):
        raise RuntimeError("A reranker was invoked despite the frozen Task3 no-reranker protocol.")

    def capturing_softmax(logits, centered_coefficients, device=None, batch_size=64):
        index = len(records)
        if index >= len(blocks) or index in pending:
            raise RuntimeError("Unexpected Task3 continuous-branch invocation order.")
        np.testing.assert_array_equal(centered_coefficients, expected_centered)
        continuous = original_softmax(logits, centered_coefficients, device=device, batch_size=batch_size)
        pending[index] = {
            "mlp_logits": np.asarray(logits, dtype=np.float32).copy(),
            "continuous": np.asarray(continuous, dtype=np.float32).copy(),
        }
        return continuous

    def capturing_assignment(scores):
        index = len(records)
        if index >= len(blocks) or index not in pending:
            raise RuntimeError("Unexpected Task3 assignment invocation order.")
        block = blocks[index]
        trial_indices = split.blocks[block.key].validation_trials
        captured_scores = np.asarray(scores, dtype=np.float32)
        if captured_scores.shape != (len(trial_indices), len(images)):
            raise RuntimeError(f"Unexpected score shape for {block.key}: {captured_scores.shape}")
        captured = pending.pop(index)
        if block.mouse not in prototype_cache:
            prototype_cache[block.mouse] = np.load(
                task3_models / block.mouse / "encoding_prototypes.npy", allow_pickle=False
            )
        raw_counts = block.features[trial_indices]
        encoding = response_scores(
            np.log1p(raw_counts), prototype_cache[block.mouse], metric="mse", device=device
        )
        continuous = original_softmax(
            captured["mlp_logits"], expected_centered, device=device
        )
        np.testing.assert_allclose(
            continuous, captured["continuous"], rtol=1e-5, atol=1e-6
        )
        reconstructed = (
            row_zscore(captured["mlp_logits"])
            + row_zscore(encoding)
            + row_zscore(continuous)
        ) / 3.0
        np.testing.assert_allclose(reconstructed, captured_scores, rtol=1e-5, atol=1e-6)
        production_assigned = original_assignment(captured_scores)
        independent_assigned = original_assignment(reconstructed)
        np.testing.assert_array_equal(production_assigned, independent_assigned)
        ranked = stable_topk(captured_scores, candidate_top_n)
        ranked_scores = np.take_along_axis(captured_scores, ranked, axis=1)
        records.append({
            "block_key": block.key,
            "mouse": block.mouse,
            "targets": np.asarray(block.trial_targets[trial_indices], dtype=np.int64),
            "assigned": production_assigned.astype(np.int64),
            "score_top1": ranked[:, 0].copy(),
            "ranked_indices": ranked,
            "ranked_scores": ranked_scores.astype(np.float32),
            "score_max_abs_error": float(np.max(np.abs(reconstructed - captured_scores))),
            "continuous_max_abs_error": float(np.max(np.abs(continuous - captured["continuous"]))),
            "independent_assignment_match": True,
        })
        return production_assigned

    with (
        mock.patch.object(final, "softmax_dct_similarity", capturing_softmax),
        mock.patch.object(final, "unique_assignment", capturing_assignment),
        mock.patch.object(final, "apply_listwise_reranker", forbidden_reranker),
    ):
        prediction = final.predict_task(
            config,
            "task3",
            task3_models.parent,
            images=images,
            blocks=blocks,
            split=split,
            device=device,
            reranker_checkpoint=None,
            candidate_top_n=candidate_top_n,
            manifest_output=None,
        )
    if len(records) != len(blocks) or pending or prediction["reranker"] is not None:
        raise RuntimeError("Production capture did not complete exactly once per block without a reranker.")

    offset = 0
    for record in records:
        count = len(record["targets"])
        part = slice(offset, offset + count)
        np.testing.assert_array_equal(
            prediction["candidate_indices"][part], record["ranked_indices"]
        )
        np.testing.assert_array_equal(
            prediction["predictions"][part], images[record["assigned"]]
        )
        np.testing.assert_array_equal(prediction["targets"][part], record["targets"])
        offset += count
    if offset != len(prediction["predictions"]):
        raise RuntimeError("Captured rows do not cover the production prediction exactly.")
    return records, prediction


def fixed_partitions(n_images, preregistration):
    order = np.random.default_rng(123).permutation(n_images)
    tune = np.sort(order[:256])
    heldout = np.sort(order[256:768])
    expected = preregistration["data_scope"]["expected_rows"]
    if n_images != 2748 or len(tune) != 256 or len(heldout) != 512:
        raise RuntimeError("The fixed picture partition differs from preregistration.")
    if np.intersect1d(tune, heldout).size:
        raise RuntimeError("Tune and heldout picture IDs overlap.")
    return {"tune": tune, "heldout": heldout}, expected


def save_tune_score_archive(path, score_matrix, configurations, targets, groups, mice):
    np.savez_compressed(
        path,
        ocf=score_matrix,
        configurations=np.asarray([
            [row["k"], row["temperature"], row["threshold"]] for row in configurations
        ], dtype=np.float64),
        configuration_columns=np.asarray(["k", "temperature", "threshold"]),
        target_indices=targets,
        group_ids=groups,
        mouse_ids=mice,
    )


def build_manifest(preregistration, config_path, task3_models, blocks, selection_hash, elapsed):
    source_paths = [
        PREREGISTRATION,
        OUTPUT / "run.py",
        OUTPUT / "smoke_test.py",
        config_path,
        ROOT / "src/final.py",
        ROOT / "src/evaluation.py",
        ROOT / "src/retrieval.py",
        ROOT / "src/splits.py",
        ROOT / "src/features.py",
        ROOT / "src/dataloader.py",
        ROOT / "src/encoding.py",
        ROOT / "src/neural.py",
        ROOT / "src/ridge.py",
    ]
    model_paths = []
    for mouse in sorted({block.mouse for block in blocks}):
        for name in preregistration["frozen_inputs"]["required_per_mouse_artifacts"]:
            model_paths.append(task3_models / mouse / name)
    catalog_path = DATA / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    data_paths = [catalog_path, DATA / "train/pictures.npz"]
    data_paths.extend(DATA / entry["path"] for entry in catalog["blocks"] if entry["split"] == "train")

    def hashed(paths):
        return [
            {
                "path": relative_path(path),
                "sha256": final.file_sha256(path),
                "size_bytes": Path(path).stat().st_size,
            }
            for path in paths
        ]

    output_paths = [
        OUTPUT / "selection.json",
        OUTPUT / "results.json",
        OUTPUT / "tune_candidate_scores.npz",
        OUTPUT / "baseline_scores.npz",
        OUTPUT / "locked_candidate_scores.npz",
    ]
    return {
        "experiment": preregistration["name"],
        "preregistration_sha256": final.file_sha256(PREREGISTRATION),
        "selection_sha256_at_lock": selection_hash,
        "command": preregistration["execution"]["full_command"],
        "pre_run_checks": preregistration["execution"]["pre_run_checks"],
        "working_directory": str(ROOT),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "code_and_config_inputs": hashed(source_paths),
        "model_inputs": hashed(model_paths),
        "public_train_data_inputs": hashed(data_paths),
        "outputs": hashed(output_paths),
        "elapsed_seconds_including_manifest_inputs": elapsed,
        "scope": preregistration["scope_limits"],
        "historical_reuse": preregistration["data_scope"]["historical_reuse"],
    }


def main():
    started = time.perf_counter()
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["status"] != "frozen_before_implementation_and_scoring":
        raise RuntimeError("The preregistration is not frozen.")
    config_path = ROOT / preregistration["frozen_inputs"]["config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_frozen_config(config, preregistration)
    task3_models = ROOT / preregistration["frozen_inputs"]["models"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = preregistration["execution"]["ocf_batch_size"]
    configurations = grid_configurations(preregistration)
    if len(configurations) != preregistration["candidate_family"]["grid_size"]:
        raise RuntimeError("Candidate grid size differs from preregistration.")
    max_k = max(row["k"] for row in configurations)
    if max_k != preregistration["execution"]["candidate_top_n_for_consistency_check"]:
        raise RuntimeError("Candidate capture width differs from preregistration.")

    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=config["task3"]["bin_ms"])
    split = make_split(blocks, "task3", seed=42)
    partitions, expected = fixed_partitions(len(images), preregistration)
    if len(blocks) != expected["blocks"]:
        raise RuntimeError(f"Expected {expected['blocks']} Task3 public blocks, got {len(blocks)}.")
    training_scope = {}
    for block in blocks:
        block_split = split.blocks[block.key]
        if np.intersect1d(block_split.train_trials, block_split.validation_trials).size:
            raise RuntimeError(f"Train/validation leakage in {block.key}.")
        if len(block_split.validation_trials) != expected["validation_per_block"]:
            raise RuntimeError(f"Unexpected validation row count for {block.key}.")
        for artifact in preregistration["frozen_inputs"]["required_per_mouse_artifacts"]:
            path = task3_models / block.mouse / artifact
            if not path.is_file():
                raise FileNotFoundError(path)
        training_scope[block.key] = {
            "train_trials": len(block_split.train_trials),
            "validation_trials": len(block_split.validation_trials),
        }

    print(f"device={device}; candidates={len(configurations)}; max_k={max_k}", flush=True)
    records, production = capture_production(
        config, task3_models, images, blocks, split, device, max_k
    )
    targets = np.concatenate([record["targets"] for record in records])
    groups = np.concatenate([
        np.full(len(record["targets"]), record["block_key"]) for record in records
    ])
    mice = np.concatenate([
        np.full(len(record["targets"]), record["mouse"]) for record in records
    ])
    assigned = np.concatenate([record["assigned"] for record in records])
    score_top1 = np.concatenate([record["score_top1"] for record in records])
    ranked_indices = np.concatenate([record["ranked_indices"] for record in records])
    ranked_scores = np.concatenate([record["ranked_scores"] for record in records])
    np.testing.assert_array_equal(production["candidate_indices"], ranked_indices)
    np.testing.assert_array_equal(production["targets"], targets)
    np.testing.assert_array_equal(production["groups"], groups)
    del production

    masks = {name: np.isin(targets, pictures) for name, pictures in partitions.items()}
    for name, mask in masks.items():
        expected_total = expected[f"{name}_total"]
        if mask.sum() != expected_total:
            raise RuntimeError(f"Unexpected {name} total: {mask.sum()} != {expected_total}")
        for record in records:
            block_mask = groups == record["block_key"]
            count = np.sum(mask & block_mask)
            if count != expected[f"{name}_per_block"]:
                raise RuntimeError(f"Unexpected {name} rows for {record['block_key']}: {count}")

    tune_mask = masks["tune"]
    tune_targets = targets[tune_mask]
    tune_groups = groups[tune_mask]
    tune_mice = mice[tune_mask]
    tune_target_images = images[tune_targets]
    tune_baseline_scores = official_scores(
        images[assigned[tune_mask]], tune_target_images, device, batch_size
    )
    baseline_tune = metric_bundle(
        tune_baseline_scores,
        tune_groups,
        tune_mice,
        tune_targets,
        score_top1[tune_mask],
        assigned[tune_mask],
    )
    print(f"baseline tune OCF={baseline_tune['ocf_mean']:.9f}", flush=True)

    candidate_rows = []
    tune_score_matrix = np.empty((len(configurations), len(tune_targets)), dtype=np.float64)
    for configuration in configurations:
        index = configuration["grid_index"]
        predictions = render_mixtures(
            images,
            ranked_indices[tune_mask],
            ranked_scores[tune_mask],
            k=configuration["k"],
            temperature=configuration["temperature"],
            threshold=configuration["threshold"],
        )
        scores = official_scores(predictions, tune_target_images, device, batch_size)
        tune_score_matrix[index] = scores
        metrics = metric_bundle(
            scores,
            tune_groups,
            tune_mice,
            tune_targets,
            score_top1[tune_mask],
        )
        row = {**configuration, "tune": metrics}
        candidate_rows.append(row)
        print(
            f"tune {index + 1:02d}/{len(configurations)} "
            f"k={configuration['k']} T={configuration['temperature']} "
            f"threshold={configuration['threshold']} OCF={metrics['ocf_mean']:.9f}",
            flush=True,
        )
        del predictions, scores

    save_tune_score_archive(
        OUTPUT / "tune_candidate_scores.npz",
        tune_score_matrix,
        configurations,
        tune_targets,
        tune_groups,
        tune_mice,
    )
    locked_index = select_locked_index(candidate_rows)
    locked_row = candidate_rows[locked_index]
    locked_configuration = {
        key: locked_row[key] for key in ("grid_index", "k", "temperature", "threshold")
    }
    selection = {
        "experiment": preregistration["name"],
        "preregistration_sha256": final.file_sha256(PREREGISTRATION),
        "selection_rule": preregistration["selection_and_advancement"]["tuning"],
        "tie_break": preregistration["candidate_family"]["grid_order_and_tie_break"],
        "data_scope": preregistration["data_scope"],
        "training_scope": training_scope,
        "picture_ids": {name: values.tolist() for name, values in partitions.items()},
        "device": device,
        "baseline_tune": baseline_tune,
        "candidates": candidate_rows,
        "locked_index": locked_index,
        "locked_configuration": locked_configuration,
        "locked_candidate_tune": locked_row["tune"],
        "heldout_candidate_scored_before_this_lock": False,
    }
    write_json(OUTPUT / "selection.json", selection)
    selection_hash = final.file_sha256(OUTPUT / "selection.json")
    print(f"locked configuration: {locked_configuration}; selection sha256={selection_hash}", flush=True)

    heldout_mask = masks["heldout"]
    heldout_targets = targets[heldout_mask]
    heldout_groups = groups[heldout_mask]
    heldout_mice = mice[heldout_mask]
    heldout_target_images = images[heldout_targets]
    heldout_baseline_scores = official_scores(
        images[assigned[heldout_mask]], heldout_target_images, device, batch_size
    )
    baseline_heldout = metric_bundle(
        heldout_baseline_scores,
        heldout_groups,
        heldout_mice,
        heldout_targets,
        score_top1[heldout_mask],
        assigned[heldout_mask],
    )
    locked_heldout_predictions = render_mixtures(
        images,
        ranked_indices[heldout_mask],
        ranked_scores[heldout_mask],
        k=locked_configuration["k"],
        temperature=locked_configuration["temperature"],
        threshold=locked_configuration["threshold"],
    )
    locked_heldout_scores = official_scores(
        locked_heldout_predictions, heldout_target_images, device, batch_size
    )
    locked_heldout = metric_bundle(
        locked_heldout_scores,
        heldout_groups,
        heldout_mice,
        heldout_targets,
        score_top1[heldout_mask],
    )
    if final.file_sha256(OUTPUT / "selection.json") != selection_hash:
        raise RuntimeError("selection.json changed after heldout scoring began.")

    np.savez_compressed(
        OUTPUT / "baseline_scores.npz",
        tune_ocf=tune_baseline_scores,
        heldout_ocf=heldout_baseline_scores,
        tune_target_indices=tune_targets,
        heldout_target_indices=heldout_targets,
        tune_group_ids=tune_groups,
        heldout_group_ids=heldout_groups,
        tune_mouse_ids=tune_mice,
        heldout_mouse_ids=heldout_mice,
    )
    np.savez_compressed(
        OUTPUT / "locked_candidate_scores.npz",
        tune_ocf=tune_score_matrix[locked_index],
        heldout_ocf=locked_heldout_scores,
        configuration=np.asarray([
            locked_configuration["k"],
            locked_configuration["temperature"],
            locked_configuration["threshold"],
        ]),
        configuration_columns=np.asarray(["k", "temperature", "threshold"]),
        tune_target_indices=tune_targets,
        heldout_target_indices=heldout_targets,
        tune_group_ids=tune_groups,
        heldout_group_ids=heldout_groups,
        tune_mouse_ids=tune_mice,
        heldout_mouse_ids=heldout_mice,
    )

    mouse_comparison = {}
    for mouse in sorted(set(mice.tolist())):
        baseline_value = baseline_heldout["per_mouse"][mouse]["ocf_mean"]
        candidate_value = locked_heldout["per_mouse"][mouse]["ocf_mean"]
        mouse_comparison[mouse] = {
            "baseline_ocf": baseline_value,
            "locked_candidate_ocf": candidate_value,
            "delta": candidate_value - baseline_value,
        }
    group_comparison = {}
    for group in sorted(set(groups.tolist())):
        baseline_value = baseline_heldout["groups"][group]["ocf_mean"]
        candidate_value = locked_heldout["groups"][group]["ocf_mean"]
        group_comparison[group] = {
            "baseline_ocf": baseline_value,
            "locked_candidate_ocf": candidate_value,
            "delta": candidate_value - baseline_value,
        }
    advance = advancement(
        {mouse: row["delta"] for mouse, row in mouse_comparison.items()},
        {group: row["delta"] for group, row in group_comparison.items()},
    )
    results = {
        "experiment": preregistration["name"],
        "preregistration_sha256": final.file_sha256(PREREGISTRATION),
        "selection_sha256_at_lock": selection_hash,
        "historical_reuse_disclosure": preregistration["data_scope"]["historical_reuse"],
        "hypothesis_status": preregistration["hypothesis_status"],
        "device": device,
        "production_consistency": {
            "entrypoint": "src.final.predict_task",
            "reranker": None,
            "candidate_top_n": max_k,
            "all_production_predictions_equal_images_at_captured_assignment": True,
            "all_production_candidates_equal_stable_score_topk": True,
            "all_independent_assignments_match": True,
            "score_reconstruction_tolerances": {"rtol": 1e-5, "atol": 1e-6},
            "blocks": [
                {
                    "block_key": record["block_key"],
                    "rows": len(record["targets"]),
                    "score_max_abs_error": record["score_max_abs_error"],
                    "continuous_max_abs_error": record["continuous_max_abs_error"],
                    "independent_assignment_match": record["independent_assignment_match"],
                }
                for record in records
            ],
        },
        "locked_configuration": locked_configuration,
        "baseline": {"tune": baseline_tune, "heldout": baseline_heldout},
        "locked_candidate": {"tune": locked_row["tune"], "heldout": locked_heldout},
        "heldout_comparison": {
            "overall": {
                "baseline_ocf": baseline_heldout["ocf_mean"],
                "locked_candidate_ocf": locked_heldout["ocf_mean"],
                "delta": locked_heldout["ocf_mean"] - baseline_heldout["ocf_mean"],
            },
            "per_mouse": mouse_comparison,
            "per_group": group_comparison,
            **advance,
            "rule": preregistration["selection_and_advancement"]["advance_rule"],
            "next_step": (
                "Advance only according to this frozen rule."
                if advance["advance"]
                else "Stop; no further candidate search or heldout retuning under this protocol."
            ),
        },
        "oracle": None,
        "elapsed_seconds_before_manifest": time.perf_counter() - started,
    }
    write_json(OUTPUT / "results.json", results)
    manifest = build_manifest(
        preregistration,
        config_path,
        task3_models,
        blocks,
        selection_hash,
        time.perf_counter() - started,
    )
    write_json(OUTPUT / "manifest.json", manifest)
    print(
        f"heldout overall baseline={baseline_heldout['ocf_mean']:.9f} "
        f"locked={locked_heldout['ocf_mean']:.9f} "
        f"delta={locked_heldout['ocf_mean'] - baseline_heldout['ocf_mean']:.9f}; "
        f"advance={advance['advance']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
