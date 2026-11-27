"""Task 3 single-trial retrieval comparison on public training data only."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions
from src.features import aggregate_by_target, load_feature_blocks
from src.final import apply_session_normalizer, fit_session_normalizer
from src.neural import load_classifier, predict_logits
from src.retrieval import unique_assignment
from src.ridge import RidgePath, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
OUT = ROOT / "test/30_task3_single_trial/32_task3_retrieval"
MODEL_ROOT = ROOT / "test/20_task2_cross_day/04_neural_seen/artifacts/task3_mlp_bin50_pooled_sessionnorm_assign"
SEED = 42
PICTURE_SEED = 123
K = 32
ENCODING_ALPHA = 0.01
WINDOWS = ((0, 2), (2, 4), (4, 5), (0, 5))
WINDOW_NAMES = ("0-100ms", "100-200ms", "200-250ms", "0-250ms")
VARIANTS = ("baseline", "multiscale", "lda_reliability", "prototype_continuous")


def row_zscore(scores):
    scores = np.asarray(scores, dtype=np.float32)
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)


def time_slice(features, n_units, window):
    start, stop = window
    return np.ascontiguousarray(
        features.reshape(len(features), n_units, 5)[:, :, start:stop].reshape(len(features), -1)
    )


def prepare(blocks, split, bank_coefficients, device):
    contexts = {}
    rows = []
    for mouse in sorted({block.mouse for block in blocks}):
        mouse_blocks = [block for block in blocks if block.mouse == mouse]
        n_units = mouse_blocks[0].n_units
        n_features = n_units * 5
        model = load_classifier(MODEL_ROOT / mouse / "classifier.pt", device)
        if model.n_features != n_features:
            raise ValueError(f"Frozen classifier feature mismatch for {mouse}")
        encoding_parts = {window: [] for window in WINDOWS}
        class_means = []
        residual_square = np.zeros(n_features, dtype=np.float64)
        residual_df = 0
        for block in mouse_blocks:
            part = split.blocks[block.key]
            train_raw = block.features[part.train_trials]
            train_targets = block.trial_targets[part.train_trials]
            if not np.all(np.bincount(train_targets, minlength=len(bank_coefficients)) == 3):
                raise ValueError(f"Expected three training repeats per glyph in {block.key}")
            mean_targets, mean_x = aggregate_by_target(train_raw, train_targets)
            if not np.array_equal(mean_targets, np.arange(len(bank_coefficients))):
                raise ValueError(f"Incomplete training gallery in {block.key}")
            normalizer = fit_session_normalizer(train_raw)
            train_normalized = apply_session_normalizer(train_raw, normalizer)
            ids, means = aggregate_by_target(train_normalized, train_targets)
            if not np.array_equal(ids, mean_targets):
                raise ValueError(f"LDA class order mismatch in {block.key}")
            residual_square += np.square(train_normalized - means[train_targets], dtype=np.float64).sum(axis=0)
            residual_df += len(train_raw) - len(ids)
            class_means.append(means)
            for window in WINDOWS:
                encoding_parts[window].append(np.log1p(time_slice(mean_x, n_units, window)))

            validation_trials = part.validation_trials
            rows.append({
                "block": block.key,
                "mouse": mouse,
                "n_units": n_units,
                "targets": block.trial_targets[validation_trials],
                "raw": block.features[validation_trials],
                "normalizer": normalizer,
            })

        prototypes = {}
        for window in WINDOWS:
            neural = np.concatenate(encoding_parts[window])
            targets = np.tile(np.arange(len(bank_coefficients)), len(mouse_blocks))
            path = RidgePath().fit(bank_coefficients[targets], neural)
            prototypes[window] = path.predict(bank_coefficients, ENCODING_ALPHA).astype(np.float32)
            print(f"PREPARED {mouse} encoding {WINDOW_NAMES[WINDOWS.index(window)]}", flush=True)
            del path, neural
        within = (residual_square / residual_df).astype(np.float32)
        # Fixed 50% shrinkage toward an isotropic within-class variance.
        lda_scale = 0.5 * within + 0.5 * float(np.median(within))
        lda_prototype = np.mean(class_means, axis=0, dtype=np.float32)
        contexts[mouse] = {
            "model": model,
            "prototypes": prototypes,
            "lda_prototype": lda_prototype,
            "lda_weight": (1.0 / np.maximum(lda_scale, 1e-8)).astype(np.float32),
            "lda_within_variance_median": float(np.median(within)),
            "lda_residual_df": int(residual_df),
        }
        del class_means, encoding_parts
    rows.sort(key=lambda row: row["block"])
    return rows, contexts


def weighted_mse_scores(observations, prototypes, weight, device):
    prototype = torch.from_numpy(prototypes).to(device)
    weight_tensor = torch.from_numpy(weight).to(device)
    prototype_square = (prototype.square() * weight_tensor).sum(dim=1)
    output = []
    for start in range(0, len(observations), 128):
        batch = torch.from_numpy(observations[start:start + 128]).to(device)
        scores = (2 * (batch * weight_tensor) @ prototype.T - prototype_square
                  - (batch.square() * weight_tensor).sum(dim=1, keepdim=True))
        output.append(scores.float().cpu().numpy())
    return np.concatenate(output)


def continuous_image_scores(logits, centered_bank, device):
    # A fixed T=1 softmax expectation; no temperature or fusion weight is tuned.
    output = []
    for start in range(0, len(logits), 64):
        batch = torch.from_numpy(logits[start:start + 64]).to(device)
        probabilities = F.softmax(batch, dim=1)
        expected = F.normalize(probabilities @ centered_bank, dim=1)
        output.append((expected @ F.normalize(centered_bank, dim=1).T).cpu().numpy())
    return np.concatenate(output)


def base_query(row, context, indices, device):
    raw = row["raw"][indices]
    normalized = apply_session_normalizer(raw, row["normalizer"])
    logits = predict_logits(context["model"], normalized, device=device)
    encoding = response_scores(
        np.log1p(raw), context["prototypes"][(0, 5)], metric="mse", device=device
    )
    return {
        "row": row,
        "targets": row["targets"][indices],
        "raw": raw,
        "normalized": normalized,
        "logits": logits,
        "mlp_z": row_zscore(logits),
        "encoding_z": row_zscore(encoding),
    }


def rank_variant(query, name, context, centered_bank, device):
    if name == "baseline":
        scores = 0.5 * query["mlp_z"] + 0.5 * query["encoding_z"]
    elif name == "multiscale":
        parts = []
        for window in WINDOWS:
            if window == (0, 5):
                parts.append(query["encoding_z"])
            else:
                raw = np.log1p(time_slice(query["raw"], query["row"]["n_units"], window))
                response = response_scores(
                    raw, context["prototypes"][window], metric="mse", device=device
                )
                parts.append(row_zscore(response))
        multiscale = np.mean(parts, axis=0, dtype=np.float32)
        scores = 0.5 * query["mlp_z"] + 0.5 * multiscale
    elif name == "lda_reliability":
        lda = weighted_mse_scores(
            query["normalized"], context["lda_prototype"], context["lda_weight"], device
        )
        scores = 0.5 * query["mlp_z"] + 0.5 * row_zscore(lda)
    elif name == "prototype_continuous":
        image = continuous_image_scores(query["logits"], centered_bank, device)
        scores = (query["mlp_z"] + query["encoding_z"] + row_zscore(image)) / 3.0
    else:
        raise ValueError(name)
    if not np.isfinite(scores).all():
        raise ValueError(f"Nonfinite {name} scores")
    top32 = np.argsort(-scores, axis=1, kind="stable")[:, :K]
    assigned = unique_assignment(scores)
    return {
        "block": query["row"]["block"],
        "targets": query["targets"].copy(),
        "top32": top32,
        "assigned": assigned,
    }


def concatenate_records(parts):
    return {
        "targets": np.concatenate([part["targets"] for part in parts]),
        "top32": np.concatenate([part["top32"] for part in parts]),
        "assigned": np.concatenate([part["assigned"] for part in parts]),
        "groups": np.concatenate([
            np.full(len(part["targets"]), part["block"]) for part in parts
        ]),
    }


def rank_set(rows, contexts, pictures, names, bank_coefficients, device, full_block_assignment=False):
    centered_bank = torch.from_numpy(
        bank_coefficients - bank_coefficients.mean(axis=0, keepdims=True)
    ).to(device)
    queries = []
    for row in rows:
        if full_block_assignment:
            indices = np.arange(len(row["targets"]))
        else:
            indices = np.flatnonzero(np.isin(row["targets"], pictures))
            if not np.array_equal(row["targets"][indices], pictures):
                raise ValueError(f"Unexpected picture subset in {row['block']}")
        queries.append(base_query(row, contexts[row["mouse"]], indices, device))
    output = {}
    for name in names:
        parts = []
        for query in queries:
            part = rank_variant(query, name, contexts[query["row"]["mouse"]], centered_bank, device)
            if full_block_assignment:
                keep = np.flatnonzero(np.isin(part["targets"], pictures))
                if not np.array_equal(part["targets"][keep], pictures):
                    raise ValueError(f"Unexpected picture subset in {part['block']}")
                part = {
                    "block": part["block"],
                    "targets": part["targets"][keep],
                    "top32": part["top32"][keep],
                    "assigned": part["assigned"][keep],
                }
            parts.append(part)
        output[name] = concatenate_records(parts)
        print(f"RANKED {name}: n={len(output[name]['targets'])}", flush=True)
    return output


def fill_ocf_cache(record_groups, images, cache, device, include_top32=True):
    pair_parts = []
    for records in record_groups.values():
        top = records["top32"]
        if include_top32:
            pair_parts.append(np.column_stack((top.ravel(), np.repeat(records["targets"], K))))
        pair_parts.append(np.column_stack((records["assigned"], records["targets"])))
    pairs = np.unique(np.concatenate(pair_parts).astype(np.int64), axis=0)
    missing = [tuple(map(int, pair)) for pair in pairs if tuple(map(int, pair)) not in cache]
    print(f"OCF unique pairs={len(pairs)} new={len(missing)}", flush=True)
    for start in range(0, len(missing), 512):
        chunk = missing[start:start + 512]
        predicted = np.fromiter((pair[0] for pair in chunk), dtype=np.int64)
        targets = np.fromiter((pair[1] for pair in chunk), dtype=np.int64)
        scores = score_predictions(images[predicted], images[targets], device=device, batch_size=32)
        for pair, score in zip(chunk, scores):
            cache[pair] = float(score)
        if start % 16384 == 0:
            print(f"OCF scored {min(start + len(chunk), len(missing))}/{len(missing)}", flush=True)


def metric_summary(records, cache):
    targets = records["targets"]
    top = records["top32"]
    assigned = records["assigned"]
    inside = (top == targets[:, None]).any(axis=1)
    correct = assigned == targets
    oracle_values = np.asarray([
        cache[(int(predicted), int(target))]
        for predicted, target in zip(top.ravel(), np.repeat(targets, K))
    ]).reshape(len(targets), K)
    final_values = np.asarray([
        cache[(int(predicted), int(target))] for predicted, target in zip(assigned, targets)
    ])

    def summarize(mask):
        n = int(mask.sum())
        return {
            "n": n,
            "true_recall_at_32": float(inside[mask].mean()),
            "true_inside_32_count": int(inside[mask].sum()),
            "true_outside_32_count": int((~inside[mask]).sum()),
            "true_inside_32_not_selected_count": int((inside[mask] & ~correct[mask]).sum()),
            "final_correct_count": int(correct[mask].sum()),
            "final_accuracy": float(correct[mask].mean()),
            "oracle32_ocf": float(oracle_values[mask].max(axis=1).mean()),
            "final_ocf": float(final_values[mask].mean()),
        }

    all_rows = np.ones(len(targets), dtype=bool)
    return {
        "overall": summarize(all_rows),
        "by_block": {
            block: summarize(records["groups"] == block)
            for block in sorted(set(records["groups"].tolist()))
        },
    }


def assignment_summary(records, cache):
    targets = records["targets"]
    top = records["top32"]
    assigned = records["assigned"]
    inside = (top == targets[:, None]).any(axis=1)
    correct = assigned == targets
    final_values = np.asarray([
        cache[(int(predicted), int(target))] for predicted, target in zip(assigned, targets)
    ])

    def summarize(mask):
        return {
            "true_inside_32_not_selected_count": int((inside[mask] & ~correct[mask]).sum()),
            "final_correct_count": int(correct[mask].sum()),
            "final_accuracy": float(correct[mask].mean()),
            "final_ocf": float(final_values[mask].mean()),
        }

    return {
        "overall": summarize(np.ones(len(targets), dtype=bool)),
        "by_block": {
            block: summarize(records["groups"] == block)
            for block in sorted(set(records["groups"].tolist()))
        },
    }


def recompute_full_block():
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = OUT / "results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    images = (load_images(DATA) >= 128).astype(np.float32)
    bank_coefficients = dct_encode(images, 16)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    split = make_split(blocks, "task3", seed=SEED)
    rows, contexts = prepare(blocks, split, bank_coefficients, device)
    heldout_pictures = np.asarray(result["protocol"]["heldout_pictures"], dtype=np.int64)
    names = tuple(dict.fromkeys(("baseline", result["selected"])))
    records = rank_set(
        rows, contexts, heldout_pictures, names, bank_coefficients, device,
        full_block_assignment=True,
    )
    cache = {}
    fill_ocf_cache(records, images, cache, device, include_top32=False)
    recomputed = {name: assignment_summary(records[name], cache) for name in names}
    original_scores = result.get("full_block_recompute", {}).get(
        "original_subset_assignment_final_ocf",
        {name: result["heldout"][name]["overall"]["final_ocf"] for name in names},
    )
    for name in names:
        result["heldout"][name]["overall"].update(recomputed[name]["overall"])
        for block, metrics in recomputed[name]["by_block"].items():
            result["heldout"][name]["by_block"][block].update(metrics)

    reference = json.loads((ROOT / "test/20_task2_cross_day/09_ensemble/results_task3_ensemble.json").read_text(
        encoding="utf-8"
    ))["evaluation_assignment"]
    baseline = result["heldout"]["baseline"]
    baseline_delta = baseline["overall"]["final_ocf"] - reference["ocf_mean"]
    block_deltas = {
        block: metrics["final_ocf"] - reference["groups"][block]["ocf_mean"]
        for block, metrics in baseline["by_block"].items()
    }
    result["protocol"]["tune_assignment_scope"] = (
        "per-block rectangular Hungarian assignment inside the fixed 256-picture tune subset; "
        "recall/oracle selection metrics are independent of assignment"
    )
    result["protocol"]["heldout_assignment_scope"] = (
        "Hungarian unique_assignment over every complete 2748-row validation block before "
        "selecting the fixed 512 heldout pictures"
    )
    result["protocol"]["assignment_scope"] = (
        "tune uses its fixed picture subset; final heldout uses full-block assignment before subsetting"
    )
    result["full_block_recompute"] = {
        "command": ".venv/Scripts/python.exe -B -u test/30_task3_single_trial/32_task3_retrieval/run.py --recompute-full-block",
        "reason": "Make baseline and selected candidate use exp09's full-block Hungarian assignment scope.",
        "model_or_fusion_parameters_changed": False,
        "selection_changed": False,
        "original_subset_assignment_final_ocf": original_scores,
        "baseline_exp09_reference_ocf": reference["ocf_mean"],
        "baseline_minus_exp09": baseline_delta,
        "baseline_block_minus_exp09": block_deltas,
        "baseline_matches_exp09_within_1e-12": bool(
            abs(baseline_delta) <= 1e-12
            and all(abs(value) <= 1e-12 for value in block_deltas.values())
        ),
        "candidate_minus_baseline": (
            result["heldout"][result["selected"]]["overall"]["final_ocf"]
            - baseline["overall"]["final_ocf"]
        ),
        "environment": {
            "python": sys.executable,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": device,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"FINAL {path}", flush=True)
    for name in names:
        print(f"FULL_BLOCK {name} {json.dumps(result['heldout'][name]['overall'])}", flush=True)
    print(f"BASELINE_MINUS_EXP09 {baseline_delta}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-full-block", action="store_true")
    args = parser.parse_args()
    if args.recompute_full_block:
        recompute_full_block()
        return
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    if images.shape != (2748, 64, 64):
        raise ValueError("Unexpected public training gallery")
    bank_coefficients = dct_encode(images, 16)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    if sorted(block.key for block in blocks) != [
        f"{mouse}/session{session}" for mouse in ("mice1", "mice2") for session in range(1, 5)
    ]:
        raise ValueError("Expected all eight public training blocks")
    split = make_split(blocks, "task3", seed=SEED)
    order = np.random.default_rng(PICTURE_SEED).permutation(len(images))
    tune_pictures = np.sort(order[:256])
    heldout_pictures = np.sort(order[256:768])
    rows, contexts = prepare(blocks, split, bank_coefficients, device)

    tune_records = rank_set(rows, contexts, tune_pictures, VARIANTS, bank_coefficients, device)
    ocf_cache = {}
    fill_ocf_cache(tune_records, images, ocf_cache, device)
    tuning = {name: metric_summary(tune_records[name], ocf_cache) for name in VARIANTS}
    for name in VARIANTS:
        print(f"TUNE {name} {json.dumps(tuning[name]['overall'])}", flush=True)

    def selection_key(name):
        row = tuning[name]["overall"]
        return row["true_recall_at_32"], row["oracle32_ocf"], row["final_ocf"]

    selected = max(VARIANTS, key=selection_key)
    protocol = {
        "data_scope": "public train only; no competition test files read",
        "task": "task3 single held-out trial",
        "split_seed": SEED,
        "picture_seed": PICTURE_SEED,
        "tune_pictures": tune_pictures.tolist(),
        "heldout_pictures": heldout_pictures.tolist(),
        "tune_reused_from_seed42_checkpoint_validation": True,
        "assignment_scope": "tune uses its fixed picture subset; final heldout uses full-block assignment before subsetting",
        "tune_assignment_scope": "per-block rectangular Hungarian assignment inside the fixed 256-picture tune subset; recall/oracle selection metrics are independent of assignment",
        "heldout_assignment_scope": "Hungarian unique_assignment over every complete 2748-row validation block before selecting the fixed 512 heldout pictures",
        "candidate_ranking": "pre-assignment stable descending scores; 32 candidates",
        "ocf": "official PNG-parity OCF on binary public training glyphs",
        "heldout_used_for_selection": False,
        "heldout_selection_scope": "current experiment's winner selection only",
        "heldout_used_for_current_winner_selection": False,
        "heldout_used_for_candidate_design": True,
        "heldout_reused_before_this_experiment": True,
        "heldout_selection_statement": "Only tune labels/metrics were used for the current winner choice. Selection was persisted before any heldout inference or scoring in this run. After selection, only baseline and the selected variant were ranked/evaluated on heldout; losing variants received no heldout evaluation.",
        "historical_heldout_disclosure": "The same 512 heldout IDs were evaluated in earlier local experiments including exp18 and exp30, and exp30's single-trial recall audit motivated these candidate families. Thus no heldout metric enters the current lexicographic winner choice, but the set is historically reused, influenced candidate design, and is not a pristine confirmatory holdout.",
        "limitation": "This reused local split is not an untouched external validation or a blind competition-test result.",
    }
    definitions = {
        "baseline": "0.5*z(frozen MLP logits)+0.5*z(full-window ridge neural-prototype MSE)",
        "multiscale": "0.5*z(MLP)+0.5*mean(z(ridge neural-prototype MSE) over 0-100,100-200,200-250,0-250 ms)",
        "lda_reliability": "0.5*z(MLP)+0.5*z(diagonal-LDA neural-prototype score); inverse within-class variance with fixed 50% shrinkage to its median; session-normalized train trials only",
        "prototype_continuous": "equal-weight z(MLP), z(full neural prototype), z(cosine of fixed T=1 softmax-expected DCT16 continuous image against centered glyph DCT16)",
        "frozen_classifiers": {
            mouse: str((MODEL_ROOT / mouse / "classifier.pt").relative_to(ROOT))
            for mouse in sorted(contexts)
        },
        "encoding_alpha": ENCODING_ALPHA,
        "selection_rule": "lexicographic maximum of tune true recall@32, tune oracle32 OCF, tune final OCF; fixed VARIANTS order breaks exact ties; baseline is an allowed outcome",
        "candidate_order": list(VARIANTS),
        "lda_training_diagnostics": {
            mouse: {
                "within_variance_median": context["lda_within_variance_median"],
                "residual_degrees_of_freedom": context["lda_residual_df"],
            }
            for mouse, context in contexts.items()
        },
    }
    selection = {
        "experiment": "32_task3_retrieval",
        "protocol": protocol,
        "definitions": definitions,
        "tune": tuning,
        "selected": selected,
        "selected_tune_key": selection_key(selected),
    }
    (OUT / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"SELECTION LOCKED: {selected}", flush=True)

    # Only now touch heldout neural trials, and only for baseline plus the winner.
    heldout_names = tuple(dict.fromkeys(("baseline", selected)))
    heldout_records = rank_set(
        rows, contexts, heldout_pictures, heldout_names, bank_coefficients, device,
        full_block_assignment=True,
    )
    fill_ocf_cache(heldout_records, images, ocf_cache, device)
    heldout = {name: metric_summary(heldout_records[name], ocf_cache) for name in heldout_names}
    result = dict(selection)
    result["heldout"] = heldout
    result["heldout_evaluated_variants"] = list(heldout_names)
    result["environment"] = {
        "python": sys.executable,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": device,
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    path = OUT / "results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"FINAL {path}", flush=True)
    for name, value in heldout.items():
        print(f"HELDOUT {name} {json.dumps(value['overall'])}", flush=True)


if __name__ == "__main__":
    main()
