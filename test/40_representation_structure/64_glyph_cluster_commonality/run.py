"""Exploratory: do visually clustered glyphs share a common neural response?

Descriptive analysis only. It clusters the 2748 public binary glyph images with the
OCF-aligned directional field embedding, measures within-cluster versus between-cluster
neural response similarity per training session with a label-permutation null, and reads
cluster-level posteriors from the already-trained fixed-split 50 ms classifiers.

No model is trained, selected or changed here. The fixed heldout pictures were reused
before, so the cluster-decoding numbers are exploratory, not pristine confirmation.
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
from src.features import aggregate_by_target
from src.neural import load_classifier, predict_logits
from src.preprocessing import load_cached_blocks
from src.retrieval import _retrieval_embedding
from src.ridge import dct_encode
from src.splits import make_split
from src.v6 import _apply_norm, _directory, _fit_norm

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
V6_MODELS = ROOT / "test/70_protocol_and_scoring/63_v6_fixed/models"
K_GRID = (8, 16, 32, 64, 128)
PRIMARY_K = 32
N_PERMUTATIONS = 200
SEED = 20261126


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "test/40_representation_structure/64_glyph_cluster_commonality")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def image_embedding(images, device):
    tensor = torch.from_numpy(np.asarray(images, dtype=np.float32)[:, None]).to(device)
    return _retrieval_embedding(tensor, "directional_multi").float()


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


def adjusted_rand(a, b):
    _, first = np.unique(a, return_inverse=True)
    _, second = np.unique(b, return_inverse=True)
    table = np.zeros((first.max() + 1, second.max() + 1))
    np.add.at(table, (first, second), 1)

    def pairs(values):
        return float((values * (values - 1) / 2).sum())

    total = pairs(table)
    rows = pairs(table.sum(axis=1))
    columns = pairs(table.sum(axis=0))
    n = len(a)
    expected = rows * columns / (n * (n - 1) / 2)
    maximum = 0.5 * (rows + columns)
    return 1.0 if maximum == expected else (total - expected) / (maximum - expected)


def neural_representation(block):
    """Per-picture mean response, feature z-scored across pictures and row L2-normalised."""
    raw = np.asarray(block.features, dtype=np.float32)
    transformed = np.log1p(np.maximum(raw, 0))
    mean = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(transformed.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)
    normalized = (transformed - mean) / scale
    targets, grouped = aggregate_by_target(normalized, np.asarray(block.trial_targets))
    if len(targets) != 2748:
        raise RuntimeError(f"Unexpected picture count for {block.key}.")
    grouped = grouped - grouped.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(grouped, axis=1, keepdims=True)
    return (grouped / np.maximum(norm, 1e-8)).astype(np.float32)


def coherence(similarity, labels, k, device, generator):
    """Observed within/between similarity and a label-permutation null."""
    labels_t = torch.from_numpy(np.asarray(labels)).to(device)
    sim = torch.from_numpy(similarity).to(device)
    sim.fill_diagonal_(0.0)
    n = len(labels)

    def split_statistic(assignment):
        one_hot = F.one_hot(assignment, k).to(torch.float32)
        block_sums = one_hot.T @ sim @ one_hot
        sizes = one_hot.sum(dim=0)
        within = (block_sums.diagonal() / (sizes * (sizes - 1))).mean()
        between_pairs = n * n - (sizes * sizes).sum()
        between = (block_sums.sum() - block_sums.diagonal().sum()) / between_pairs
        return float(within), float(between)

    within, between = split_statistic(labels_t)
    null = np.asarray([
        split_statistic(torch.from_numpy(generator.permutation(labels)).to(device))[0]
        for _ in range(N_PERMUTATIONS)
    ])
    return {
        "within_mean_similarity": within,
        "between_mean_similarity": between,
        "delta": within - between,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "z": float((within - null.mean()) / max(null.std(), 1e-12)),
        "p_permutation": float((1 + np.sum(null >= within)) / (1 + len(null))),
    }


def rank_vector(values):
    return np.argsort(np.argsort(values)).astype(np.float64)


def correlation_from_ranks(rank_a, rank_b):
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    denominator = np.sqrt((rank_a ** 2).sum() * (rank_b ** 2).sum())
    return float(rank_a @ rank_b / denominator) if denominator else 0.0


def cluster_posteriors(task, blocks, split, device):
    """Cluster inputs come from the fixed-split 50 ms model only; targets are scoring-only."""
    rows = []
    for block in blocks:
        trials = np.asarray(split.blocks[block.key].validation_trials, dtype=np.int64)
        if not len(trials):
            continue
        directory = _directory(V6_MODELS, task, block.mouse, 50)
        model = load_classifier(directory / "classifier.pt", device=device)
        if task == "task2":
            raw = np.asarray(block.features, dtype=np.float32)
            normalizer = _fit_norm(raw)
            targets, means = aggregate_by_target(
                raw, np.asarray(block.targets)[block.trial_sample_indices]
            )
            logits = predict_logits(model, _apply_norm(means, normalizer), device=device)
            row_targets = targets
        else:
            raw = np.asarray(block.features[trials], dtype=np.float32)
            with np.load(directory / f"{block.session}_normalizer.npz", allow_pickle=False) as saved:
                normalizer = saved["mean"], saved["scale"]
            logits = predict_logits(model, _apply_norm(raw, normalizer), device=device)
            row_targets = np.asarray(block.trial_targets[trials], dtype=np.int64)
        logits = torch.from_numpy(np.asarray(logits, dtype=np.float64)).to(device)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        rows.append((block.key, probabilities, row_targets))
        del model, logits
    return rows


def evaluate_clusters(rows, clusters, k):
    out = {}
    membership = np.zeros((2748, k), dtype=np.float64)
    membership[np.arange(2748), clusters] = 1.0
    for key, probabilities, targets in rows:
        cluster_probabilities = probabilities @ membership
        order = np.argsort(-cluster_probabilities, axis=1)
        true_cluster = clusters[targets]
        ranks = np.argmax(order == true_cluster[:, None], axis=1)
        glyph_order = np.argsort(-probabilities, axis=1)[:, :32]
        out[key] = {
            "rows": len(targets),
            "cluster_top1": float(np.mean(ranks == 0)),
            "cluster_top3": float(np.mean(ranks < 3)),
            "cluster_top5": float(np.mean(ranks < 5)),
            "glyph_top1": float(np.mean(np.argmax(probabilities, axis=1) == targets)),
            "glyph_recall_at_32": float(np.mean((glyph_order == targets[:, None]).any(axis=1))),
            "true_cluster_share_in_top32_glyphs": float(np.mean(
                clusters[glyph_order] == true_cluster[:, None]
            )),
        }
    return out


def main():
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    if results_path.exists() and not args.overwrite:
        raise SystemExit(f"{results_path} exists; pass --overwrite.")
    started = time.perf_counter()
    device = torch.device(args.device)
    images = (load_images(DATA) >= 128).astype(np.float32)
    if len(images) != 2748:
        raise RuntimeError("Expected the 2748 public gallery images.")
    embedding = image_embedding(images, device)
    image_rdm = (1.0 - (embedding @ embedding.T).cpu().numpy()).astype(np.float32)
    image_ranks = rank_vector(image_rdm[np.triu_indices(len(images), k=1)])

    results = {
        "experiment": "64_glyph_cluster_commonality",
        "status": "exploratory_descriptive",
        "scope": "public train blocks only; no model training, selection or production change",
        "heldout_disclosure": (
            "Cluster decoding reuses the fixed-split checkpoints whose public heldout was "
            "inspected in earlier experiments; these numbers are not pristine confirmation."
        ),
        "clustering": {
            "features": "directional_multi embedding of the 2748 public gallery images",
            "k_grid": list(K_GRID),
            "primary_k": PRIMARY_K,
            "clusters": {},
        },
        "coherence": {},
        "cross_session_neural_rdm": {},
        "cluster_decoding": {},
    }

    assignments = {}
    dct_features = torch.from_numpy(dct_encode(images, 16))
    for k in K_GRID:
        primary = kmeans(embedding, k, seed=0, device=device)
        alternate = kmeans(embedding, k, seed=1, device=device)
        dct_assignment = kmeans(dct_features, k, seed=0, device=device)
        assignments[k] = primary
        sizes = np.bincount(primary, minlength=k).tolist()
        results["clustering"]["clusters"][str(k)] = {
            "sizes": sizes,
            "stability_ari_seed0_vs_seed1": adjusted_rand(primary, alternate),
            "robustness_ari_directional_vs_dct16": adjusted_rand(primary, dct_assignment),
        }
        print(f"k={k}: min={min(sizes)} max={max(sizes)} "
              f"ari_seed={adjusted_rand(primary, alternate):.4f} "
              f"ari_dct={adjusted_rand(primary, dct_assignment):.4f}", flush=True)

    blocks = load_cached_blocks(CACHE, split="train", bin_ms=50)
    generator = np.random.default_rng(SEED)
    neural_ranks = {}
    for block in blocks:
        representation = neural_representation(block)
        similarity = representation @ representation.T
        rdm = 1.0 - similarity
        upper = np.triu_indices(len(representation), k=1)
        neural_ranks[block.key] = rank_vector(rdm[upper])
        row = {
            "mouse": block.mouse,
            "pictures": len(representation),
            "rsa_image_vs_neural_spearman": correlation_from_ranks(image_ranks, neural_ranks[block.key]),
        }
        for k in K_GRID:
            row[f"k{k}"] = coherence(similarity, assignments[k], k, device, generator)
        results["coherence"][block.key] = row
        print(f"{block.key}: rsa={row['rsa_image_vs_neural_spearman']:+.4f} "
              f"k32_delta={row['k32']['delta']:+.4f} z={row['k32']['z']:+.2f}", flush=True)
        del similarity, rdm, representation

    for mouse in ("mice1", "mice2"):
        keys = [block.key for block in blocks if block.mouse == mouse]
        values = [
            correlation_from_ranks(neural_ranks[keys[i]], neural_ranks[keys[j]])
            for i in range(len(keys)) for j in range(i + 1, len(keys))
        ]
        results["cross_session_neural_rdm"][mouse] = {
            "pairs": len(values), "mean": float(np.mean(values)),
            "min": float(np.min(values)), "max": float(np.max(values)),
        }
        print(f"{mouse} cross-session neural RDM: mean={np.mean(values):+.4f}", flush=True)

    for task in ("task2", "task3"):
        split = make_split(blocks, task, seed=42)
        rows = cluster_posteriors(task, blocks, split, device)
        per_k = {str(k): evaluate_clusters(rows, assignments[k], k) for k in K_GRID}
        results["cluster_decoding"][task] = {
            "reference_chance_top1": 1.0 / PRIMARY_K,
            "blocks": per_k,
        }
        primary = per_k[str(PRIMARY_K)]
        print(f"{task}: k32 cluster_top1={np.mean([v['cluster_top1'] for v in primary.values()]):.4f} "
              f"top3={np.mean([v['cluster_top3'] for v in primary.values()]):.4f} "
              f"glyph_recall32={np.mean([v['glyph_recall_at_32'] for v in primary.values()]):.4f}",
              flush=True)

    np.savez_compressed(output / "clusters.npz", **{f"k{k}": assignments[k] for k in K_GRID})
    results["elapsed_seconds"] = time.perf_counter() - started
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {results_path}", flush=True)


if __name__ == "__main__":
    main()
