"""Experiment 67: per-task paired candidates on the established submission protocol.

Uses exactly the protocol of the two Kaggle submissions (`test/evaluate_final.py`):
fixed public split (seed 42), the RNG-123 tune/heldout picture partition, whole-block
assignment before picture subsetting, and official OCF. v5 models are reused from the
frozen evaluation cache, so the base arm must reproduce the recorded v5 numbers.

Arms per task:
  base       : frozen v5 prediction path, unchanged
  sinkhorn   : base scores + log Sinkhorn-balanced prior (T=0.05, 20 iterations, alpha=1)
  multires   : v5 path with the MLP z-score replaced by the equal mean of the 10/25/50 ms
               views (extra views trained with the same v5 recipe and fixed epochs)

Selection uses tune only; heldout is reported separately and never selects. No
production change and no Kaggle submission happens here.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from evaluate_final import public_protocol
from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.final import (
    DATA,
    apply_session_normalizer,
    checkpoint_path,
    fit_session_normalizer,
    row_zscore,
    train_task,
)
from src.neural import load_classifier, load_regressor, predict_logits
from src.reranker import apply_listwise_reranker, load_listwise_reranker
from src.retrieval import render_gb2312_bank, retrieve_glyphs, softmax_dct_similarity, unique_assignment
from src.ridge import dct_decode, dct_encode

V5 = ROOT / "test/.cache/evaluate_final_task23_v1/models"
V5_TASK1 = ROOT / "test/.cache/evaluate_final_task1_v2/models"
EXPECTED = {"task1": 18.437061915748913, "task2": 23.824141005605483, "task3": 16.7598316835003}
VIEWS = (10, 25, 50)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "test/70_protocol_and_scoring/67_taskwise_best")
    parser.add_argument("--tasks", nargs="+", choices=("task1", "task2", "task3"), default=["task2", "task3"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ocf-batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sinkhorn_prior(scores, *, temperature=0.05, iterations=20):
    """Return log balanced-assignment prior from row-z scores; unlabeled inputs only.

    Performed in the log domain so near one-hot rows cannot overflow during column
    normalisation. Each cell holds exactly one example of every class in these blocks.
    """
    scores = np.asarray(scores, dtype=np.float64)
    centered = (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), 1e-6)
    log_probabilities = centered / temperature

    def normalise(values, axis):
        peak = values.max(axis=axis, keepdims=True)
        return values - (peak + np.log(np.exp(values - peak).sum(axis=axis, keepdims=True)))

    log_probabilities = normalise(log_probabilities, 1)
    for _ in range(iterations):
        log_probabilities = normalise(log_probabilities, 0)
        log_probabilities = normalise(log_probabilities, 1)
    return log_probabilities


def load_view_models(directory, task, mouse, seeds, device):
    return [load_classifier(checkpoint_path(Path(directory), task, mouse, seed), device) for seed in seeds]


def task_scores(config, task, models, blocks, split, images, device, views=(50,), multires_dir=None):
    """Rebuild v5 scores per validation block, optionally with extra MLP views."""
    task_config = config[task]
    seeds = task_config.get("seeds", [42])
    reranker = None
    if task == "task2" and task_config.get("reranker", {}).get("enabled", False):
        reranker = load_listwise_reranker(Path(models) / "listwise_cnn.pt", device)
    centered_coefficients = None
    if task == "task3" and task_config.get("prototype_continuous", {}).get("enabled", False):
        coefficients = dct_encode(images, 16)
        centered_coefficients = coefficients - coefficients.mean(axis=0, keepdims=True)
    view_blocks = {view: (blocks if view == task_config["bin_ms"] else
                          {b.key: b for b in load_feature_blocks(DATA, "train", bin_ms=view)})
                   for view in views}
    models_by_mouse = {}
    output = {}
    for block in blocks:
        part = split.blocks[block.key]
        trial_indices = part.validation_trials
        if not len(trial_indices):
            continue
        if block.mouse not in models_by_mouse:
            primary = [load_classifier(checkpoint_path(Path(models), task, block.mouse, seed), device)
                       for seed in seeds]
            extra = {}
            for view in views:
                if view == task_config["bin_ms"]:
                    continue
                extra[view] = load_view_models(
                    Path(multires_dir) / f"view{view}", task, block.mouse, [42], device)[0]
            models_by_mouse[block.mouse] = (primary, extra)
        primary, extra = models_by_mouse[block.mouse]
        if task == "task2":
            raw_validation = block.features[trial_indices]
            targets, raw_counts = aggregate_by_target(raw_validation, block.trial_targets[trial_indices])
            normalizer = fit_session_normalizer(raw_validation)
        else:
            raw_counts = block.features[trial_indices]
            with np.load(Path(models) / task / block.mouse / f"{block.session}_normalizer.npz") as data:
                normalizer = (data["mean"], data["scale"])
            targets = block.trial_targets[trial_indices]
        counts = apply_session_normalizer(raw_counts, normalizer)
        mlp_scores = np.mean(
            np.stack([predict_logits(model, counts, device=device) for model in primary]),
            axis=0, dtype=np.float32,
        )
        view_zs = [row_zscore(mlp_scores)]
        for view in views:
            if view == task_config["bin_ms"]:
                continue
            view_block = view_blocks[view][block.key]
            view_trials = split.blocks[view_block.key].validation_trials
            view_raw = view_block.features[view_trials]
            if task == "task2":
                _, view_counts = aggregate_by_target(view_raw, view_block.trial_targets[view_trials])
                view_counts = apply_session_normalizer(view_counts, fit_session_normalizer(view_raw))
            else:
                with np.load(Path(multires_dir) / f"view{view}" / task / block.mouse /
                             f"{block.session}_normalizer.npz") as data:
                    view_counts = apply_session_normalizer(view_raw, (data["mean"], data["scale"]))
            view_zs.append(row_zscore(predict_logits(extra[view], view_counts, device=device)))
        mlp_z = np.mean(np.stack(view_zs), axis=0) if len(view_zs) > 1 else view_zs[0]
        prototypes = np.load(Path(models) / task / block.mouse / "encoding_prototypes.npy")
        encoding_z = row_zscore(response_scores(np.log1p(raw_counts), prototypes, metric="mse", device=device))
        if task == "task3":
            continuous = softmax_dct_similarity(mlp_scores, centered_coefficients, device=device)
            scores = (mlp_z + encoding_z + row_zscore(continuous)) / 3.0
        else:
            weight = task_config.get("mlp_weight", 1.0)
            scores = weight * mlp_z + (1 - weight) * encoding_z
        if reranker is not None:
            reranker_config = task_config["reranker"]
            scores = apply_listwise_reranker(
                reranker, scores, mlp_z, encoding_z, block.mouse, images, device,
                top_n=reranker_config["top_n"], k=reranker_config["k"],
                batch_size=reranker_config["inference_batch_size"],
            )
        output[block.key] = {"scores": scores.astype(np.float64), "targets": np.asarray(targets, dtype=np.int64)}
    return output


def task1_continuous(config, models, blocks, split, device):
    """Return aggregated v5 continuous reconstructions per block."""
    task_config = config["task1"]
    seeds = task_config.get("seeds", [42])
    by_mouse = {}
    output = {}
    for block in blocks:
        part = split.blocks[block.key]
        trial_indices = part.validation_trials
        if not len(trial_indices):
            continue
        if block.mouse not in by_mouse:
            by_mouse[block.mouse] = [load_regressor(checkpoint_path(Path(models), "task1", block.mouse, seed), device)
                                     for seed in seeds]
        targets, counts = aggregate_by_target(block.features[trial_indices],
                                              block.trial_targets[trial_indices])
        coefficients = np.mean(
            np.stack([predict_logits(model, counts, device=device) for model in by_mouse[block.mouse]]),
            axis=0, dtype=np.float32,
        )
        output[block.key] = {"continuous": dct_decode(coefficients, 32),
                             "targets": np.asarray(targets, dtype=np.int64)}
    return output


def ocf(predictions, targets, device, batch_size):
    return summarize_scores(score_predictions(predictions, targets, device=device,
                                             batch_size=batch_size))["ocf_mean"]


def evaluate_rows(prediction_by_block, images, partitions, device, batch_size):
    """Return OCF for each picture partition using whole-block predictions."""
    rows = {}
    for name, pictures in partitions.items():
        joined_predictions, joined_targets = [], []
        for value in prediction_by_block.values():
            mask = np.isin(value["targets"], pictures)
            joined_predictions.append(value["predictions"][mask])
            joined_targets.append(value["targets"][mask])
        rows[name] = {"ocf": ocf(np.concatenate(joined_predictions), images[np.concatenate(joined_targets)],
                                 device, batch_size),
                      "rows": int(sum(len(v) for v in joined_predictions))}
    return rows


def predict_from_scores(score_by_block, images):
    return {key: {"predictions": images[unique_assignment(value["scores"])],
                  "targets": value["targets"]}
            for key, value in score_by_block.items()}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / f"results_{'_'.join(args.tasks)}.json"
    if results_path.exists() and not args.overwrite:
        raise SystemExit(f"{results_path} exists; pass --overwrite.")
    started = time.perf_counter()
    config = json.loads((ROOT / "src/final_config.json").read_text(encoding="utf-8"))
    images = (load_images(DATA) >= 128).astype(np.float32)
    device = args.device
    results = {"experiment": "67_taskwise_best", "config": config["name"],
               "protocol": "test/evaluate_final.py fixed split seed42 + RNG123 partitions",
               "selection": "tune selects; heldout reported only", "tasks": {}}

    for task in args.tasks:
        bin_ms = config[task]["bin_ms"]
        blocks = load_feature_blocks(DATA, "train", bin_ms=bin_ms)
        split, partitions, protocol = public_protocol(task, blocks, len(images))
        models = V5_TASK1 if task == "task1" else V5
        task_result = {"protocol": protocol, "partitions": {k: len(v) for k, v in partitions.items()},
                       "arms": {}}

        if task == "task1":
            bank = render_gb2312_bank(Path(r"C:\Windows\Fonts") / config[task]["font_file"],
                                      font_size=config[task]["font_size"],
                                      variation=config[task]["font_variation"])
            continuous_by_block = task1_continuous(config, models, blocks, split, device)
            prediction = {key: {"predictions": retrieve_glyphs(value["continuous"], bank,
                                                               metric="directional2", device=device)[0],
                                "targets": value["targets"]}
                          for key, value in continuous_by_block.items()}
            task_result["arms"]["base"] = evaluate_rows(prediction, images, partitions, device,
                                                        args.ocf_batch_size)
            task_result["note"] = ("Scale calibration cannot change directional retrieval: the metric "
                                   "normalises the linear directional field, so gain is a no-op.")
        else:
            score_sets = {"base": task_scores(config, task, models, blocks, split, images, device)}
            score_sets["sinkhorn"] = {
                key: {"scores": value["scores"] + sinkhorn_prior(value["scores"]),
                      "targets": value["targets"]}
                for key, value in score_sets["base"].items()
            }
            multires_dir = args.output / "models" / task
            view_config = copy.deepcopy(config)
            view_config[task]["seeds"] = [42]
            if task == "task2":
                view_config[task]["reranker"]["enabled"] = False
            for view in VIEWS:
                if view == bin_ms:
                    continue
                view_config[task]["bin_ms"] = view
                view_blocks = load_feature_blocks(DATA, "train", bin_ms=view)
                train_task(view_config, task, multires_dir / f"view{view}", images=images,
                           blocks=view_blocks, split=split, skip_existing=True, device=device)
                del view_blocks
            score_sets["multires"] = task_scores(config, task, models, blocks, split, images, device,
                                                 views=VIEWS, multires_dir=multires_dir)
            for name, scores in score_sets.items():
                prediction = predict_from_scores(scores, images)
                task_result["arms"][name] = evaluate_rows(prediction, images, partitions, device,
                                                          args.ocf_batch_size)
                print(f"{task} {name}: tune={task_result['arms'][name]['tune']['ocf']:.6f} "
                      f"heldout={task_result['arms'][name]['heldout']['ocf']:.6f}", flush=True)
            base_heldout = task_result["arms"]["base"]["heldout"]["ocf"]
            if abs(base_heldout - EXPECTED[task]) > 1e-6:
                task_result["reproduction_error"] = base_heldout - EXPECTED[task]
                print(f"WARNING {task} base heldout {base_heldout:.6f} != expected "
                      f"{EXPECTED[task]:.6f}", flush=True)
            ranked = sorted((name for name in task_result["arms"] if name != "base"),
                            key=lambda name: -task_result["arms"][name]["tune"]["ocf"])
            task_result["selected"] = ranked[0] if ranked else "base"
            task_result["selected_tune_delta"] = (
                task_result["arms"][task_result["selected"]]["tune"]["ocf"]
                - task_result["arms"]["base"]["tune"]["ocf"])
            task_result["selected_heldout_delta"] = (
                task_result["arms"][task_result["selected"]]["heldout"]["ocf"] - base_heldout)
        results["tasks"][task] = task_result
        results_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")

    heldout = {}
    for task, value in results["tasks"].items():
        name = value.get("selected", "base")
        heldout[task] = {"arm": name, "ocf": value["arms"][name]["heldout"]["ocf"]}
    results["per_task_selection"] = heldout
    results["combined_heldout_mean"] = float(np.mean([row["ocf"] for row in heldout.values()]))
    results["v5_heldout_mean"] = float(np.mean([EXPECTED[task] for task in heldout]))
    results["elapsed_seconds"] = time.perf_counter() - started
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selection": heldout,
                      "combined_heldout_mean": results["combined_heldout_mean"],
                      "v5_heldout_mean": results["v5_heldout_mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
