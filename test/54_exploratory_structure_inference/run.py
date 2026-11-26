"""Exploratory diagnostic: how much of Task3 error comes from argmax-only inference?

exploratory; not preregistered; not used for production selection.

The script reuses experiment 48's Task3 preparation, OCF pair cache, official metric
aggregation and the production training recipe, trains exactly nine epochs, captures
the 256x2748 raw validation logits, and compares structure-aware inference policies.
No model parameters, no epoch selection and no production artefacts are produced.
"""

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions
from src.neural import SpikeClassifier, predict_logits
from src.preprocessing import load_cached_blocks, validate_split
from src.retrieval import glyph_similarity, unique_assignment  # unique_assignment shared with exp 48
from src.splits import make_split

OUT = Path(__file__).resolve().parent
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
REFERENCE48 = ROOT / "test/48_high_capacity_generalization"
RESULTS = OUT / "exploratory_results.json"
SIMILARITY_CACHE = OUT / "structure_similarity_directional_multi.npz"

MICE = ("mice1", "mice2")
BLOCKS = tuple(f"{mouse}/session{session}" for mouse in MICE for session in range(1, 5))
TASK3_FOLDS = ("20261126", "20261127")
N_CLASSES = 2748
SUBSET_SIZE = 256
BIN_MS = 50
TOP_K = 32
TRAINING_EPOCH = 9
BATCH_SIZE = 256
TRAIN_SEED = 42
GRADIENT_CLIP_NORM = 5.0
DISCLAIMER = "exploratory; not preregistered; not used for production selection"

PRODUCTION_CONFIG = {
    "hidden": 512,
    "dropout": {"mice1": 0.2, "mice2": 0.35},
    "label_smoothing": 0.05,
    "weight_decay": 1e-3,
    "learning_rate": 3e-4,
}

STRATEGIES = (
    "argmax",
    "hungarian",
    "top32_medoid",
    "top32_prob_medoid",
    "top32_mean_image",
    "pS_argmax",
    "pS_hungarian",
    "zlogits_plus_pS",
)
METRICS = ("raw_top1_accuracy", "raw_recall_at_32", "raw_top1_official_ocf", "assigned_final_ocf")


def load_reference_module():
    """Import experiment 48's run module read-only (its directory name is not an identifier)."""
    path = REFERENCE48 / "run.py"
    spec = importlib.util.spec_from_file_location("experiment48_reference_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = load_reference_module()
prepare_task3 = REF.prepare_task3
OcfPairCache = REF.OcfPairCache
subset_positions = REF.subset_positions
aggregate_rows = REF.aggregate_rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sanitize(value):
    """Convert numpy scalars and replace non-finite floats by null for strict JSON."""
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sanitize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


# --------------------------------------------------------------------------------------------
# Structure similarity and inference policies
# --------------------------------------------------------------------------------------------


def load_or_build_similarity(images, path, *, device):
    """Return a 0-100 structural similarity matrix, reusing the local npz cache when present."""
    count = len(images)
    path = Path(path)
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            similarity = np.asarray(data["S"], dtype=np.float32)
            metric = str(data["metric"])
            scale = float(data["scale"])
        require(metric == "directional_multi", "Similarity cache metric changed")
        require(scale == 100.0, "Similarity cache scale changed")
        require(similarity.shape == (count, count), "Similarity cache shape mismatch")
        require(np.isfinite(similarity).all(), "Similarity cache contains non-finite values")
        return similarity
    raw = np.asarray(glyph_similarity(images, metric="directional_multi", device=device), dtype=np.float64)
    require(raw.shape == (count, count) and np.isfinite(raw).all(), "Structural similarity failed")
    similarity = ((np.clip(raw, -1.0, 1.0) + 1.0) * 50.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, S=similarity, metric=np.array("directional_multi"), scale=np.array(100.0),
    )
    return similarity


def similarity_report(similarity):
    diagonal = np.diag(similarity).astype(np.float64)
    return {
        "metric": "directional_multi",
        "scale": "0-100 = 50 * (cosine + 1)",
        "shape": list(similarity.shape),
        "diagonal_mean": float(diagonal.mean()),
        "diagonal_min": float(diagonal.min()),
        "mean": float(np.asarray(similarity, dtype=np.float64).mean()),
        "min": float(similarity.min()),
        "max": float(similarity.max()),
    }


def softmax_rows(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def top_k_indices(logits, k=TOP_K):
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.ndim == 2 and 0 < k <= logits.shape[1], "Invalid top-k request")
    return np.argsort(-logits, axis=1, kind="stable")[:, :k]


def row_zscore(matrix):
    x = np.asarray(matrix, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] >= 2 and np.isfinite(x).all(), "Row z-score needs a finite matrix")
    return (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)


def medoid_scores(top, similarity, weights=None):
    """Score each row's top-k candidate by structural agreement with its own candidate set.

    The unweighted variant sums the other k-1 candidates (self excluded); the weighted
    variant maximizes sum_c w_c * S[candidate, c], which includes the candidate's own weight.
    """
    n_rows, k = top.shape
    output = np.empty((n_rows, k), dtype=np.float64)
    for row in range(n_rows):
        candidates = top[row]
        block = np.asarray(similarity[np.ix_(candidates, candidates)], dtype=np.float64)
        if weights is None:
            output[row] = block.sum(axis=1) - np.diag(block)
        else:
            output[row] = block @ np.asarray(weights[row], dtype=np.float64)
    return output


def embed_top_scores(top, values, n_classes):
    """Place top-k scores in a full-width matrix; other classes receive a strong penalty."""
    scores = np.full((top.shape[0], n_classes), -1e6, dtype=np.float64)
    scores[np.arange(top.shape[0])[:, None], top] = values
    return scores


def build_strategy(name, logits, similarity, images, top):
    """Return (score matrix or None, class predictions or None, image predictions or None)."""
    n_rows, n_classes = logits.shape
    wide = np.asarray(logits, dtype=np.float64)
    rows = np.arange(n_rows)
    if name == "argmax":
        return wide, wide.argmax(axis=1).astype(np.int64), None
    if name == "hungarian":
        return wide, unique_assignment(wide), None
    if name in ("top32_medoid", "top32_prob_medoid"):
        weights = None
        if name == "top32_prob_medoid":
            weights = softmax_rows(np.take_along_axis(wide, top, axis=1))
        values = medoid_scores(top, similarity, weights)
        scores = embed_top_scores(top, values, n_classes)
        return scores, top[rows, values.argmax(axis=1)].astype(np.int64), None
    if name == "top32_mean_image":
        weights = softmax_rows(np.take_along_axis(wide, top, axis=1))
        mean_image = np.einsum(
            "nk,nkij->nij", weights, np.asarray(images[top], dtype=np.float64),
        )
        return None, None, (mean_image >= 0.5).astype(np.float32)
    probabilities = softmax_rows(wide).astype(np.float32)
    expected = (probabilities @ np.asarray(similarity, dtype=np.float32)).astype(np.float64)
    if name == "pS_argmax":
        return expected, expected.argmax(axis=1).astype(np.int64), None
    if name == "pS_hungarian":
        return expected, unique_assignment(expected), None
    if name == "zlogits_plus_pS":
        scores = row_zscore(wide) + row_zscore(expected)
        return scores, scores.argmax(axis=1).astype(np.int64), None
    raise ValueError(f"Unknown inference strategy: {name}")


def evaluate_strategy(scores, predictions, image_predictions, targets, top, images,
                      ocf_cache, *, device):
    """Official OCF for one strategy; image outputs cannot be deduplicated by assignment."""
    recall = float(np.mean(np.any(top == targets[:, None], axis=1)))
    if image_predictions is not None:
        ocf = score_predictions(image_predictions, images[targets], device=device, batch_size=32)
        require(np.isfinite(ocf).all(), "Official OCF produced non-finite image scores")
        mean_ocf = float(np.mean(ocf))
        return {
            "raw_top1_accuracy": float("nan"),
            "raw_recall_at_32": float("nan"),
            "raw_top1_official_ocf": mean_ocf,
            "assigned_final_ocf": mean_ocf,
        }
    predicted_ocf = ocf_cache.score(targets, predictions)
    assigned = unique_assignment(scores)
    require(len(np.unique(assigned)) == len(assigned), "Assignment is not unique")
    return {
        "raw_top1_accuracy": float(np.mean(predictions == targets)),
        "raw_recall_at_32": recall,
        "raw_top1_official_ocf": float(predicted_ocf.mean()),
        "assigned_final_ocf": float(ocf_cache.score(targets, assigned).mean()),
    }


def block_diagnostics(logits, targets, similarity, top, medoid_predictions):
    """Quantify candidate-set structure and the cost of the argmax-only choice."""
    pairwise = np.empty(len(targets), dtype=np.float64)
    for row in range(len(targets)):
        candidates = top[row]
        block = np.asarray(similarity[np.ix_(candidates, candidates)], dtype=np.float64)
        off_diagonal = block.sum() - np.trace(block)
        pairwise[row] = off_diagonal / (len(candidates) * (len(candidates) - 1))
    hit = np.any(top == targets[:, None], axis=1)
    argmax_predictions = np.asarray(logits, dtype=np.float64).argmax(axis=1)
    errors = argmax_predictions != targets
    return {
        "top32_pairwise_similarity_mean": float(pairwise.mean()),
        "recall_at_32": float(hit.mean()),
        "n_rows": int(len(targets)),
        "n_recall_at_32_hits": int(hit.sum()),
        "medoid_accuracy_given_hit": float(np.mean(medoid_predictions[hit] == targets[hit]))
        if hit.any() else float("nan"),
        "argmax_errors": errors,
        "argmax_predictions": argmax_predictions,
    }


# --------------------------------------------------------------------------------------------
# Training (production recipe, exactly nine epochs, logits retained in memory)
# --------------------------------------------------------------------------------------------


def fit_fold_logits(prepared, arm_config, *, device, epochs=TRAINING_EPOCH,
                    batch_size=BATCH_SIZE, seed=TRAIN_SEED):
    """Experiment 48 production training, stopped at epoch 9; no model artefacts are kept."""
    device = torch.device(device)
    train_x = prepared["train_x"]
    train_y = prepared["train_y"]
    n_classes = prepared["n_classes"]
    require(np.isfinite(train_x).all() and len(train_x) == len(train_y), "Invalid training matrix")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpikeClassifier(
        n_features=train_x.shape[1], n_classes=n_classes, hidden=arm_config["hidden"],
        dropout=arm_config["dropout"][prepared["mouse"]],
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
    for epoch in range(1, epochs + 1):
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
                loss = F.cross_entropy(output.float(), batch_y, label_smoothing=arm_config["label_smoothing"])
            require(bool(torch.isfinite(loss).item()), "Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(
            f"TRAIN fold={prepared['fold']} mouse={prepared['mouse']} epoch={epoch} "
            f"train_loss={total_loss / len(train_y):.6f}",
            flush=True,
        )
    logits = {}
    for block, (validation_x, _) in prepared["validation"].items():
        raw = predict_logits(model, validation_x, batch_size=batch_size, device=device)
        require(
            raw.shape == (SUBSET_SIZE, n_classes) and raw.dtype == np.float32 and np.isfinite(raw).all(),
            "Unexpected validation logit matrix",
        )
        logits[block] = np.array(raw, dtype=np.float32, copy=True)
    require(len(logits) == 4, "Task3 must produce four validation blocks")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits


# --------------------------------------------------------------------------------------------
# Aggregation and comparison
# --------------------------------------------------------------------------------------------


def find_row(rows, **identifiers):
    matches = [row for row in rows if all(row[key] == value for key, value in identifiers.items())]
    require(len(matches) == 1, f"Expected one row for {identifiers}")
    return matches[0]


def build_comparison(macro, mouse, fold):
    baseline_macro = find_row(macro, arm="hungarian")
    ranked = sorted(
        macro, key=lambda row: (-row["assigned_final_ocf"], -row["raw_top1_official_ocf"], row["arm"])
    )
    macro_output = {}
    for row in macro:
        macro_output[row["arm"]] = {
            "raw_top1_accuracy": row["raw_top1_accuracy"],
            "raw_recall_at_32": row["raw_recall_at_32"],
            "raw_top1_official_ocf": row["raw_top1_official_ocf"],
            "assigned_final_ocf": row["assigned_final_ocf"],
            "delta_assigned_final_ocf_vs_hungarian": float(
                row["assigned_final_ocf"] - baseline_macro["assigned_final_ocf"]
            ),
            "delta_raw_top1_official_ocf_vs_hungarian": float(
                row["raw_top1_official_ocf"] - baseline_macro["raw_top1_official_ocf"]
            ),
        }
    per_mouse = {
        row["arm"]: {
            mouse_key: float(row["assigned_final_ocf"]
                             - find_row(mouse, arm="hungarian", mouse=mouse_key)["assigned_final_ocf"])
            for mouse_key in MICE
        }
        for row in mouse
    }
    per_mouse = {arm: per_mouse[arm] for arm in STRATEGIES}
    per_fold = {
        row["arm"]: {
            fold_key: float(row["assigned_final_ocf"]
                            - find_row(fold, arm="hungarian", fold=fold_key)["assigned_final_ocf"])
            for fold_key in TASK3_FOLDS
        }
        for row in fold
    }
    per_fold = {arm: per_fold[arm] for arm in STRATEGIES}
    best = ranked[0]
    return {
        "baseline": "hungarian",
        "ranking_metric": "macro assigned_final_ocf",
        "macro": macro_output,
        "per_mouse_delta_assigned_final_ocf": per_mouse,
        "per_fold_delta_assigned_final_ocf": per_fold,
        "best_strategy_by_macro_assigned_final_ocf": best["arm"],
        "best_macro_assigned_final_ocf": best["assigned_final_ocf"],
        "best_macro_delta_vs_hungarian": float(
            best["assigned_final_ocf"] - baseline_macro["assigned_final_ocf"]
        ),
    }


# --------------------------------------------------------------------------------------------
# Real run
# --------------------------------------------------------------------------------------------


def preflight_cache():
    manifest_path = CACHE / "manifest.json"
    require(manifest_path.is_file(), f"Shared cache is not ready: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(BIN_MS in [int(value) for value in manifest["bin_ms"]], "The cache has no 50ms counts")
    require(manifest.get("image_shape") == [N_CLASSES, 64, 64], "Gallery image shape mismatch")
    require((CACHE / "pictures.npy").is_file(), "Cached gallery images are missing")
    records = [record for record in manifest["blocks"] if record.get("split") == "train"]
    keys = {f"{record['mouse']}/{record['session']}" for record in records}
    require(len(records) == len(BLOCKS) and keys == set(BLOCKS), "Train cache blocks are incomplete")
    return manifest


def prepare_task3_fold(blocks, fold, mouse, subset):
    split = make_split(blocks, "task3", seed=int(fold), validation_session="session4")
    validate_split(split, blocks)
    require(split.seed != 42, "The prohibited Task3 seed 42 fold was requested")
    mouse_blocks = [block for block in blocks if block.mouse == mouse]
    require(len(mouse_blocks) == 4, f"Missing sessions for {mouse}")
    return prepare_task3(mouse_blocks, split, mouse, fold, subset)


def run_experiment():
    require(torch.cuda.is_available(), "The production bf16 recipe requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.backends.cuda.matmul.allow_tf32 is False, "TF32 must stay disabled")
    manifest = preflight_cache()
    subset = np.sort(np.random.default_rng(20261128).permutation(N_CLASSES)[:SUBSET_SIZE]).astype(np.int64)
    require(len(subset) == SUBSET_SIZE, "Invalid development subset")

    print("Loading read-only public train blocks (50ms) and gallery images", flush=True)
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=BIN_MS)
    require(tuple(block.key for block in blocks) == tuple(sorted(BLOCKS)), "Loaded block set mismatch")
    images = (np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False) >= 128).astype(np.float32)
    require(images.shape == (N_CLASSES, 64, 64), "Gallery image cache mismatch")
    similarity = load_or_build_similarity(images, SIMILARITY_CACHE, device="cuda")
    require(similarity.shape == (N_CLASSES, N_CLASSES), "Structural similarity shape mismatch")
    report = similarity_report(similarity)
    print(f"SIMILARITY {json.dumps(report)}", flush=True)
    ocf_cache = OcfPairCache(images, device="cuda", batch_size=32)

    started = time.time()
    block_rows = []
    diagnostics = []
    argmax_error_ocf = []
    for fold in TASK3_FOLDS:
        for mouse in MICE:
            prepared = prepare_task3_fold(blocks, fold, mouse, subset)
            print(
                f"START fold={fold} mouse={mouse} train_rows={len(prepared['train_y'])} "
                f"features={prepared['train_x'].shape[1]}",
                flush=True,
            )
            logits_by_block = fit_fold_logits(prepared, PRODUCTION_CONFIG, device=device)
            for block, logits in logits_by_block.items():
                targets = np.asarray(prepared["validation"][block][1], dtype=np.int64)
                top = top_k_indices(logits, TOP_K)
                reference = REF.evaluate_logits(logits, targets, ocf_cache)
                strategy_metrics = {}
                for name in STRATEGIES:
                    scores, predictions, image_predictions = build_strategy(
                        name, logits, similarity, images, top
                    )
                    metrics = evaluate_strategy(
                        scores, predictions, image_predictions, targets, top, images,
                        ocf_cache, device="cuda",
                    )
                    strategy_metrics[name] = metrics
                    block_rows.append({
                        "task": "task3", "arm": name, "mouse": mouse, "fold": fold,
                        "block": block, "epoch": TRAINING_EPOCH, "n_rows": len(targets), **metrics,
                    })
                require(
                    abs(strategy_metrics["argmax"]["raw_top1_accuracy"] - reference["raw_top1_accuracy"]) <= 1e-12
                    and abs(strategy_metrics["argmax"]["raw_top1_official_ocf"]
                            - reference["raw_top1_official_ocf"]) <= 1e-5,
                    "argmax strategy disagrees with experiment 48 evaluate_logits",
                )
                require(
                    abs(strategy_metrics["hungarian"]["raw_top1_official_ocf"]
                        - reference["assigned_final_ocf"]) <= 1e-5,
                    "hungarian strategy disagrees with experiment 48 evaluate_logits",
                )
                medoid_predictions = build_strategy("top32_medoid", logits, similarity, images, top)[1]
                diag = block_diagnostics(logits, targets, similarity, top, medoid_predictions)
                error_mask = diag["argmax_errors"]
                error_scores = ocf_cache.score(
                    targets[error_mask], diag["argmax_predictions"][error_mask],
                ) if error_mask.any() else np.empty(0, dtype=np.float64)
                argmax_error_ocf.extend(float(value) for value in error_scores.tolist())
                diagnostics.append({
                    "fold": fold, "mouse": mouse, "block": block,
                    "top32_pairwise_similarity_mean": diag["top32_pairwise_similarity_mean"],
                    "recall_at_32": diag["recall_at_32"],
                    "n_rows": diag["n_rows"],
                    "n_recall_at_32_hits": diag["n_recall_at_32_hits"],
                    "medoid_accuracy_given_hit": diag["medoid_accuracy_given_hit"],
                    "argmax_error_count": int(error_mask.sum()),
                    "argmax_error_ocf_mean": float(error_scores.mean()) if len(error_scores) else float("nan"),
                    "argmax_error_ocf_median": float(np.median(error_scores)) if len(error_scores)
                    else float("nan"),
                    "argmax_error_ocf_p90": float(np.percentile(error_scores, 90)) if len(error_scores)
                    else float("nan"),
                })
                print(
                    f"EVAL fold={fold} mouse={mouse} block={block} "
                    f"hungarian_assigned={strategy_metrics['hungarian']['assigned_final_ocf']:.3f} "
                    f"recall32={diag['recall_at_32']:.4f} "
                    f"medoid_given_hit={diag['medoid_accuracy_given_hit']:.4f}",
                    flush=True,
                )
            prepared.clear()
    require(len(block_rows) == len(STRATEGIES) * len(TASK3_FOLDS) * len(MICE) * 4,
            "Incomplete block/strategy result grid")
    require(len(diagnostics) == len(TASK3_FOLDS) * len(MICE) * 4, "Incomplete diagnostic grid")

    mouse_fold = aggregate_rows(block_rows, "mouse_fold")
    mouse_rows = aggregate_rows(block_rows, "mouse")
    fold_rows = aggregate_rows(block_rows, "fold")
    macro_rows = aggregate_rows(block_rows, "macro")
    require(len(macro_rows) == len(STRATEGIES), "Macro grid mismatch")
    comparison = build_comparison(macro_rows, mouse_rows, fold_rows)
    error_values = np.asarray(argmax_error_ocf, dtype=np.float64)
    pooled = {
        "pooled_argmax_error_count": int(len(error_values)),
        "pooled_argmax_error_ocf_mean": float(error_values.mean()) if len(error_values) else float("nan"),
        "pooled_argmax_error_ocf_median": float(np.median(error_values)) if len(error_values)
        else float("nan"),
        "pooled_argmax_error_ocf_p90": float(np.percentile(error_values, 90)) if len(error_values)
        else float("nan"),
        "mean_top32_pairwise_similarity": float(np.mean(
            [row["top32_pairwise_similarity_mean"] for row in diagnostics])),
        "mean_recall_at_32": float(np.mean([row["recall_at_32"] for row in diagnostics])),
        "mean_medoid_accuracy_given_hit": float(np.nanmean(
            [row["medoid_accuracy_given_hit"] for row in diagnostics])),
    }
    results = {
        "schema": 1,
        "experiment": OUT.name,
        "status": "exploratory_complete",
        "disclaimer": DISCLAIMER,
        "data_policy": "public train only; development folds 20261126/20261127; no final evaluation",
        "config": {
            "bin_ms": BIN_MS,
            "task": "task3",
            "fold_ids": list(TASK3_FOLDS),
            "mice": list(MICE),
            "n_classes": N_CLASSES,
            "subset_size": SUBSET_SIZE,
            "training_epoch": TRAINING_EPOCH,
            "epoch_selection": "none",
            "top_k": TOP_K,
            "model_and_training": {
                "class": "src.neural.SpikeClassifier",
                "hidden": PRODUCTION_CONFIG["hidden"],
                "dropout": PRODUCTION_CONFIG["dropout"],
                "label_smoothing": PRODUCTION_CONFIG["label_smoothing"],
                "weight_decay": PRODUCTION_CONFIG["weight_decay"],
                "learning_rate": PRODUCTION_CONFIG["learning_rate"],
                "batch_size": BATCH_SIZE,
                "seed": TRAIN_SEED,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
                "model_artifacts_retained": False,
            },
            "strategies": list(STRATEGIES),
            "strategy_notes": {
                "top32_medoid": "top-32 candidates; maximize S sum over the other 31 (self excluded)",
                "top32_prob_medoid": "top-32 candidates; maximize sum_c softmax(logit)_c * S[candidate, c]",
                "top32_mean_image": "softmax-weighted mean of the top-32 glyph images, binarized at 0.5",
                "pS_argmax": "row softmax(T=1) over all 2748 classes times the 0-100 S matrix",
                "pS_hungarian": "same pS field, unique_assignment",
                "zlogits_plus_pS": "equal-weight sum of row z-scores of logits and of pS",
                "assigned_final_ocf": "official OCF after unique_assignment of the strategy's score field; "
                                      "for top32_mean_image the binarized image cannot be assigned, so its "
                                      "assigned value equals its raw value",
                "raw_top1_accuracy": "null for top32_mean_image (image output, not a class index)",
                "raw_recall_at_32": "null for top32_mean_image (candidate-restriction metrics do not apply)",
            },
        },
        "cache": {"path": "test/.cache/shared_preprocessing_v1", "manifest_schema": manifest.get("schema")},
        "similarity": {**report, "cache_file": SIMILARITY_CACHE.name},
        "run": {
            "device": torch.cuda.get_device_name(device),
            "elapsed_seconds": float(time.time() - started),
            "ocf_pair_cache": {
                "storage": "memory only",
                "scoring_calls": ocf_cache.calls,
                "unique_pairs_scored": ocf_cache.scored_pairs,
            },
        },
        "diagnostics": {"per_block": diagnostics, "pooled": pooled},
        "block": block_rows,
        "mouse_fold": mouse_fold,
        "mouse": mouse_rows,
        "fold": fold_rows,
        "macro": macro_rows,
        "comparison": comparison,
    }
    write_json(RESULTS, results)
    print("SUMMARY " + json.dumps({
        "macro": {
            arm: {
                "assigned_final_ocf": comparison["macro"][arm]["assigned_final_ocf"],
                "raw_top1_official_ocf": comparison["macro"][arm]["raw_top1_official_ocf"],
                "delta": comparison["macro"][arm]["delta_assigned_final_ocf_vs_hungarian"],
            }
            for arm in STRATEGIES
        },
        "pooled": pooled,
        "best": comparison["best_strategy_by_macro_assigned_final_ocf"],
        "best_delta": comparison["best_macro_delta_vs_hungarian"],
    }, ensure_ascii=False), flush=True)
    print(f"Wrote {RESULTS}; {DISCLAIMER}", flush=True)


# --------------------------------------------------------------------------------------------
# Smoke test (synthetic six-class data only; never reads the real cache)
# --------------------------------------------------------------------------------------------


def synthetic_images(n_classes=6, seed=5150):
    rng = np.random.default_rng(seed)
    images = np.zeros((n_classes, 64, 64), dtype=np.float32)
    for index in range(n_classes):
        top = 6 + 8 * index
        images[index, top:top + 5, :] = 1.0
        left = (11 * index) % 64
        images[index, :, left:left + 4] = 1.0
        images[index, 40 + index, 30:34] = 1.0
    return images, rng


def smoke_test():
    """Validate every strategy's shape, medoid logic, pS matmul, binarisation and OCF call."""
    images, rng = synthetic_images()
    n_classes = len(images)
    similarity_raw = np.asarray(glyph_similarity(images, metric="directional_multi", device="cpu"))
    require(similarity_raw.shape == (n_classes, n_classes), "Smoke similarity failed")
    np.testing.assert_allclose(similarity_raw, similarity_raw.T, atol=1e-4)
    similarity = ((np.clip(similarity_raw, -1.0, 1.0) + 1.0) * 50.0).astype(np.float32)
    np.testing.assert_allclose(similarity, similarity.T, atol=1e-3)
    require(np.all(np.diag(similarity) > 50.0), "Self similarity should exceed the scale midpoint")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke_similarity.npz"
        first = load_or_build_similarity(images, path, device="cpu")
        require(path.is_file(), "Similarity cache was not written")
        second = load_or_build_similarity(images, path, device="cpu")
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, similarity, atol=1e-4)

    targets = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    logits = np.full((n_classes, n_classes), -4.0, dtype=np.float32)
    logits[rng.permutation(n_classes), rng.permutation(n_classes)] = 8.0
    logits[np.arange(n_classes), np.arange(n_classes)] = 9.0
    logits[0, 1] = 9.5
    top = top_k_indices(logits, 4)
    require(top.shape == (n_classes, 4), "Smoke top-k shape failed")
    require(np.all(np.take_along_axis(logits, top, axis=1) >= -4.0), "Smoke top-k values failed")

    ocf_cache = OcfPairCache(images, device="cpu", batch_size=2)
    reference = REF.evaluate_logits(logits, targets, ocf_cache)
    outputs = {}
    for name in STRATEGIES:
        scores, predictions, image_predictions = build_strategy(name, logits, similarity, images, top)
        if image_predictions is None:
            require(scores is not None and scores.shape == (n_classes, n_classes),
                    f"Score shape failed for {name}")
            require(predictions.shape == (n_classes,) and predictions.dtype == np.int64,
                    f"Prediction shape failed for {name}")
            require(predictions.min() >= 0 and predictions.max() < n_classes, f"Prediction range failed for {name}")
        else:
            require(image_predictions.shape == (n_classes, 64, 64), "Mean-image shape failed")
            require(set(np.unique(image_predictions).tolist()) <= {0.0, 1.0}, "Mean-image binarisation failed")
        metrics = evaluate_strategy(
            scores, predictions, image_predictions, targets, top, images, ocf_cache, device="cpu",
        )
        outputs[name] = metrics
        require(np.isfinite(metrics["raw_top1_official_ocf"]), f"Non-finite OCF for {name}")
        require(0.0 <= metrics["raw_top1_official_ocf"] <= 100.0, f"OCF out of range for {name}")
        if not np.isnan(metrics["raw_top1_accuracy"]):
            require(0.0 <= metrics["raw_top1_accuracy"] <= 1.0, f"Accuracy out of range for {name}")
        if name not in ("top32_mean_image",):
            require(len(np.unique(unique_assignment(scores))) == n_classes, f"Assignment failed for {name}")

    require(abs(outputs["argmax"]["raw_top1_accuracy"] - reference["raw_top1_accuracy"]) <= 1e-12,
            "Smoke argmax cross-check failed")
    require(abs(outputs["hungarian"]["raw_top1_official_ocf"] - reference["assigned_final_ocf"]) <= 1e-5,
            "Smoke hungarian cross-check failed")
    require(np.isnan(outputs["top32_mean_image"]["raw_top1_accuracy"]),
            "Mean-image accuracy must be undefined")
    require(outputs["top32_medoid"]["raw_top1_official_ocf"] >= 0.0, "Mean-image OCF missing")

    medoid_values = medoid_scores(top, similarity)
    row = 0
    candidates = top[row]
    manual = np.asarray(similarity[np.ix_(candidates, candidates)], dtype=np.float64)
    np.testing.assert_allclose(medoid_values[row], manual.sum(axis=1) - np.diag(manual), atol=1e-4)
    weights = softmax_rows(np.take_along_axis(np.asarray(logits, dtype=np.float64), top, axis=1))
    weighted = medoid_scores(top, similarity, weights)
    np.testing.assert_allclose(weighted[row], manual @ weights[row], atol=1e-3)

    expected = (softmax_rows(np.asarray(logits, dtype=np.float64)).astype(np.float32)
                @ similarity.astype(np.float32)).astype(np.float64)
    pS_scores, pS_predictions, _ = build_strategy("pS_argmax", logits, similarity, images, top)
    np.testing.assert_allclose(pS_scores, expected, atol=1e-3)
    require(np.array_equal(pS_predictions, expected.argmax(axis=1).astype(np.int64)),
            "pS argmax mismatch")

    zscores = row_zscore(np.asarray(logits, dtype=np.float64))
    fused = zscores + row_zscore(expected)
    np.testing.assert_allclose(build_strategy("zlogits_plus_pS", logits, similarity, images, top)[0],
                               fused, atol=1e-6)
    np.testing.assert_allclose(row_zscore(logits).mean(axis=1), 0.0, atol=1e-12)

    block_rows = []
    for block_index, mouse in enumerate(MICE):
        for name in STRATEGIES:
            metrics = outputs[name]
            block_rows.append({
                "task": "task3", "arm": name, "mouse": mouse, "fold": "20261126",
                "block": f"{mouse}/session1", "epoch": TRAINING_EPOCH, "n_rows": n_classes, **metrics,
            })
    aggregated = aggregate_rows(block_rows, "macro")
    require(len(aggregated) == len(STRATEGIES), "Smoke aggregation grid failed")
    mean_image_macro = next(row for row in aggregated if row["arm"] == "top32_mean_image")
    require(np.isnan(mean_image_macro["raw_top1_accuracy"]), "NaN propagation through aggregation failed")
    sanitized = sanitize(mean_image_macro)
    require(sanitized["raw_top1_accuracy"] is None, "JSON sanitisation of NaN failed")

    diagnostics = block_diagnostics(
        logits, targets, similarity, top,
        build_strategy("top32_medoid", logits, similarity, images, top)[1],
    )
    require(0.0 <= diagnostics["recall_at_32"] <= 1.0, "Smoke recall failed")
    require(0.0 <= diagnostics["top32_pairwise_similarity_mean"] <= 100.0,
            "Smoke pairwise similarity failed")
    require(len(diagnostics["argmax_errors"]) == n_classes, "Smoke argmax error mask failed")
    print(
        "SMOKE OK: similarity symmetry and cache, all strategy shapes, medoid and weighted-medoid logic, "
        "pS matmul, z-score fusion, mean-image binarisation, official OCF, aggregation and NaN sanitisation",
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
