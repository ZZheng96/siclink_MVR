"""Experiment 65: does a glyph-family posterior improve identity assignment?

Frozen paired test on development folds only. One 50 ms production-recipe classifier is
trained per mouse/fold; the only difference between arms is how the final 2748-way scores
are formed before the Hungarian assignment:

  base            : raw logits
  soft            : row z-scored logits + lambda * row z-scored log cluster posterior
  hard            : row z-scored logits restricted to the top-K cluster posterior families

The cluster partition is computed once from the 2748 public gallery images with the
OCF-aligned directional embedding; it uses no labels and is identical for train and test.

Gate (pre-registered): macro assigned OCF delta >= +3.0 versus base, both mouse deltas
nonnegative, and nonnegative fold deltas for at least 2/3 Task2 folds and 2/2 Task3 folds.
No weight, epoch or cluster-count search beyond the listed fixed grid.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions
from src.features import aggregate_by_target
from src.neural import SpikeClassifier, predict_logits
from src.preprocessing import load_cached_blocks
from src.retrieval import _retrieval_embedding, unique_assignment
from src.ridge import dct_encode
from src.splits import make_split
from src.v6 import _apply_norm, _fit_norm

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
N_CLASSES = 2748
DEV_PICTURES = np.sort(np.random.default_rng(20261128).permutation(N_CLASSES)[:256])
K_GRID = (8, 16, 32)
LAMBDAS = (0.25, 0.5, 1.0)
HARD_GRID = ((8, 3), (32, 5))
DROPOUT = {"mice1": 0.2, "mice2": 0.35}
EPOCHS = {"task2": 7, "task3": 9}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "test/40_representation_structure/65_hierarchical_cluster_decoding")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ocf-batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def kmeans(features, k, *, seed, device, iterations=100):
    x = features.to(device=device, dtype=torch.float32)
    n = len(x)
    generator = np.random.default_rng(seed)
    centers = [x[int(generator.integers(n))]]
    squared = ((x - centers[0]) ** 2).sum(dim=1)
    for _ in range(1, k):
        probabilities = (squared / squared.sum()).cpu().numpy()
        index = int(generator.choice(n, p=probabilities))
        centers.append(x[index])
        squared = torch.minimum(squared, ((x - centers[-1]) ** 2).sum(dim=1))
    centers = torch.stack(centers)
    assignment = torch.full((n,), -1, device=device)
    for _ in range(iterations):
        distances = torch.cdist(x, centers)
        updated = distances.argmin(dim=1)
        if torch.equal(updated, assignment):
            break
        assignment = updated
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device)
        sums.index_add_(0, assignment, x)
        counts.index_add_(0, assignment, torch.ones(n, device=device))
        empty = counts == 0
        centers = sums / counts.clamp_min(1)[:, None]
        if empty.any():
            farthest = torch.topk(distances.min(dim=1).values, int(empty.sum())).indices
            centers[empty] = x[farthest]
    return assignment.cpu().numpy().astype(np.int64)


def fit_model(x, y, mouse, epochs, device):
    torch.manual_seed(42)
    model = SpikeClassifier(x.shape[1], N_CLASSES, hidden=512, dropout=DROPOUT[mouse], apply_log=False)
    model.set_normalization(x)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3,
                                  fused=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(42 + epoch).permutation(len(y))
        total = 0.0
        for start in range(0, len(order), 256):
            index = order[start:start + 256]
            inputs = torch.from_numpy(x[index]).to(device)
            labels = torch.from_numpy(y[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(inputs).float(), labels, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(index)
        print(f"  epoch={epoch}/{epochs} loss={total / len(y):.6f}", flush=True)
    return model


def prepare_task2(blocks, split, mouse):
    features, targets = [], []
    validation = {}
    for block in blocks:
        if block.mouse != mouse:
            continue
        part = split.blocks[block.key]
        raw = np.asarray(block.features[part.train_trials], dtype=np.float32)
        labels = np.asarray(block.trial_targets[part.train_trials], dtype=np.int64)
        if len(raw):
            normalizer = _fit_norm(raw)
            features.append(_apply_norm(raw, normalizer))
            targets.append(labels)
            grouped_targets, means = aggregate_by_target(raw, labels)
            features.append(_apply_norm(means, normalizer))
            targets.append(grouped_targets)
        if len(part.validation_trials):
            raw_validation = np.asarray(block.features[part.validation_trials], dtype=np.float32)
            labels_validation = np.asarray(block.trial_targets[part.validation_trials], dtype=np.int64)
            normalizer = _fit_norm(raw_validation)
            grouped_targets, means = aggregate_by_target(raw_validation, labels_validation)
            validation[block.key] = (_apply_norm(means, normalizer), grouped_targets)
    return np.concatenate(features), np.concatenate(targets), validation


def prepare_task3(blocks, split, mouse):
    features, targets = [], []
    validation = {}
    for block in blocks:
        if block.mouse != mouse:
            continue
        part = split.blocks[block.key]
        raw = np.asarray(block.features[part.train_trials], dtype=np.float32)
        labels = np.asarray(block.trial_targets[part.train_trials], dtype=np.int64)
        normalizer = _fit_norm(raw)
        features.append(_apply_norm(raw, normalizer))
        targets.append(labels)
        raw_validation = np.asarray(block.features[part.validation_trials], dtype=np.float32)
        labels_validation = np.asarray(block.trial_targets[part.validation_trials], dtype=np.int64)
        validation[block.key] = (_apply_norm(raw_validation, normalizer), labels_validation)
    return np.concatenate(features), np.concatenate(targets), validation


def cluster_inputs(logits, clusters, k, lam=None, top_clusters=None):
    standardized = (logits - logits.mean(axis=1, keepdims=True)) / np.maximum(
        logits.std(axis=1, keepdims=True), 1e-6
    )
    if lam is None and top_clusters is None:
        return logits
    membership = np.zeros((N_CLASSES, k), dtype=np.float64)
    membership[np.arange(N_CLASSES), clusters] = 1.0
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    cluster_probabilities = probabilities @ membership
    log_cluster = np.log(np.maximum(cluster_probabilities, 1e-300))
    standardization = (log_cluster - log_cluster.mean(axis=1, keepdims=True)) / np.maximum(
        log_cluster.std(axis=1, keepdims=True), 1e-6
    )
    if top_clusters is not None:
        keep = np.argsort(-cluster_probabilities, axis=1)[:, :top_clusters]
        allowed_clusters = np.zeros((len(logits), k), dtype=bool)
        allowed_clusters[np.arange(len(logits))[:, None], keep] = True
        allowed_glyphs = allowed_clusters[:, clusters]
        return np.where(allowed_glyphs, standardized, -1e9)
    return standardized + lam * standardization[:, clusters]


class OcfCache:
    """Cache official OCF for (prediction index, target index) gallery pairs."""

    def __init__(self, images, device, batch_size):
        self.images = images
        self.device = device
        self.batch_size = batch_size
        self.store = {}

    def score(self, predictions, targets):
        pairs = np.unique(np.stack([predictions, targets], axis=1), axis=0)
        missing = [tuple(int(v) for v in pair) for pair in pairs
                   if (int(pair[0]), int(pair[1])) not in self.store]
        if missing:
            array = np.asarray(missing, dtype=np.int64)
            values = score_predictions(
                self.images[array[:, 0]], self.images[array[:, 1]],
                device=str(self.device), batch_size=self.batch_size,
            )
            for pair, value in zip(missing, values):
                self.store[pair] = float(value)
        return np.asarray([self.store[(int(p), int(t))] for p, t in zip(predictions, targets)])


def evaluate_fold(task, logits_by_block, images, cache, clusters, config):
    """Return assigned official OCF per block and the subset metrics."""
    results = {}
    for key, (logits, targets) in logits_by_block.items():
        scores = cluster_inputs(logits, clusters, config.get("k"),
                                lam=config.get("lam"), top_clusters=config.get("top_clusters"))
        indices = unique_assignment(scores)
        mask = np.isin(targets, DEV_PICTURES)
        values = cache.score(indices[mask], targets[mask])
        order = np.argsort(-scores, axis=1)[:, :32]
        results[key] = {
            "rows": int(mask.sum()),
            "assigned_ocf": float(values.mean()),
            "assigned_top1_accuracy": float(np.mean(indices == targets)),
            "recall_at_32_from_scores": float(np.mean((order == targets[:, None]).any(axis=1))),
        }
    return results


def build_summary(fold_results, configs, folds):
    """Aggregate development folds exactly as the gate expects."""
    summary = {"tasks": {}, "selection": {}, "advance": False}
    for task in ("task2", "task3"):
        keys = [f"{task}/{fold}" for current, fold in folds if current == task]
        per_config = {}
        for config in configs:
            name = config["name"]
            macro, mouse_delta, fold_delta = {}, {}, {}
            for key in keys:
                rows = fold_results[key][name]
                mouse_delta[key] = {
                    mouse: np.mean([v["assigned_ocf"] for k, v in rows.items() if k.startswith(mouse)])
                    - np.mean([v["assigned_ocf"] for k, v in fold_results[key]["base"].items()
                               if k.startswith(mouse)])
                    for mouse in ("mice1", "mice2")
                }
                fold_delta[key] = np.mean([v["assigned_ocf"] for v in rows.values()]) - \
                    np.mean([v["assigned_ocf"] for v in fold_results[key]["base"].values()])
                macro[key] = np.mean([v["assigned_ocf"] for v in rows.values()])
            per_config[name] = {
                "macro": float(np.mean(list(macro.values()))),
                "fold_macro": {k.split("/")[1]: float(v) for k, v in macro.items()},
                "mouse_delta": {k.split("/")[1]: {m: float(v) for m, v in d.items()}
                                for k, d in mouse_delta.items()},
                "fold_delta": {k.split("/")[1]: float(v) for k, v in fold_delta.items()},
            }
        base_macro = per_config["base"]["macro"]
        for name, row in per_config.items():
            row["macro_delta"] = row["macro"] - base_macro
        candidates = [name for name in per_config if name != "base"]
        best = max(candidates, key=lambda candidate: per_config[candidate]["macro"])
        row = per_config[best]
        both_mice = all(value >= 0 for fold in row["mouse_delta"].values()
                        for value in fold.values())
        positive_folds = sum(1 for value in row["fold_delta"].values() if value >= 0)
        summary["tasks"][task] = {
            "base_macro": base_macro,
            "per_config": per_config,
            "selected": best,
            "selected_macro_delta": row["macro_delta"],
            "selected_both_mice_nonnegative": both_mice,
            "selected_nonnegative_folds": positive_folds,
            "selected_fold_count": len(row["fold_delta"]),
            "pass": bool(row["macro_delta"] >= 3.0 and both_mice and positive_folds >= 2),
        }
        summary["selection"][task] = best
        print(f"{task}: best={best} delta={row['macro_delta']:+.4f} "
              f"pass={summary['tasks'][task]['pass']}", flush=True)
    summary["advance"] = all(summary["tasks"][task]["pass"] for task in ("task2", "task3"))
    return summary


def main():
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "development_results.json"
    if results_path.exists() and not args.overwrite:
        raise SystemExit(f"{results_path} exists; pass --overwrite.")
    started = time.perf_counter()
    device = torch.device(args.device)
    images = (load_images(DATA) >= 128).astype(np.float32)
    embedding = torch.from_numpy(np.asarray(images, dtype=np.float32)[:, None]).to(device)
    embedding = _retrieval_embedding(embedding, "directional_multi").float()
    clusters = {k: kmeans(embedding, k, seed=0, device=device) for k in K_GRID}
    del embedding
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=50)
    cache = OcfCache(images, device, args.ocf_batch_size)
    configs = [{"name": "base"}]
    for k in K_GRID:
        for lam in LAMBDAS:
            configs.append({"name": f"soft_k{k}_l{lam}", "k": k, "lam": lam})
    for k, top in HARD_GRID:
        configs.append({"name": f"hard_k{k}_top{top}", "k": k, "top_clusters": top})

    results = {
        "experiment": "65_hierarchical_cluster_decoding",
        "status": "development_paired_test",
        "scope": "public train blocks only; no hidden test data; development pictures only",
        "clusters": {"features": "directional_multi k-means on the 2748 public gallery",
                     "k_grid": list(K_GRID), "seed": 0},
        "development_pictures": DEV_PICTURES.tolist(),
        "configs": configs,
        "folds": {},
        "gate": {
            "macro_minimum_absolute_gain": 3.0,
            "both_mice_nonnegative": True,
            "required_nonnegative_folds": {"task2": 2, "task3": 2},
        },
    }
    folds = [("task2", f"session{s}") for s in (1, 2, 3)] + \
            [("task3", seed) for seed in (20261126, 20261127)]
    for task, fold in folds:
        split = (make_split(blocks, "task2", seed=42, validation_session=fold) if task == "task2"
                 else make_split(blocks, "task3", seed=fold))
        logits_by_block = {}
        for mouse in ("mice1", "mice2"):
            prepare = prepare_task2 if task == "task2" else prepare_task3
            x, y, validation = prepare(blocks, split, mouse)
            print(f"{task}/{fold}/{mouse}: train={len(y)}", flush=True)
            model = fit_model(x, y, mouse, EPOCHS[task], device)
            for key, (features, targets) in validation.items():
                logits_by_block[key] = (
                    predict_logits(model, features, device=device).astype(np.float64), targets
                )
            del model, x, y, validation
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_results = {}
        for config in configs:
            fold_results[config["name"]] = evaluate_fold(
                task, logits_by_block, images, cache, clusters[config.get("k", K_GRID[0])], config
            )
        results["folds"][f"{task}/{fold}"] = fold_results
        base_value = np.mean([v["assigned_ocf"] for v in fold_results["base"].values()])
        ranked = {c["name"]: np.mean([v["assigned_ocf"] for v in fold_results[c["name"]].values()])
                  for c in configs}
        best_name = max(ranked, key=ranked.get)
        print(f"{task}/{fold} base={base_value:.4f} best={best_name} {ranked[best_name]:.4f}", flush=True)

    summary = build_summary(results["folds"], configs, folds)
    results["summary"] = summary
    results["elapsed_seconds"] = time.perf_counter() - started
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"advance": summary["advance"],
                      "selection": summary["selection"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
