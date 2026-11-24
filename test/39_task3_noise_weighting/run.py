"""Frozen contrastive cosine blends with experiment 35's prototype_continuous."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("task3_exp35", OUT.parent / "35_task3_contrastive/run.py")
exp35 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp35)

VARIANTS = ("v5", "noise_weighted")


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_encoder(mouse, n_features, device):
    path = exp35.ARTIFACTS / mouse / "contrastive.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = dict(n_features=int(n_features), hidden=exp35.HIDDEN, embedding=exp35.EMBEDDING,
                    dropout=exp35.DROPOUT, temperature=exp35.TEMPERATURE, epoch=exp35.EPOCHS,
                    seed=exp35.SEED)
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value}, got {checkpoint.get(key)!r}")
    encoder = exp35.ContrastiveEncoder(n_features)
    encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    return encoder.to(device).eval(), {**expected, "checkpoint": path.relative_to(exp35.ROOT).as_posix()}


def weighted_mse_scores(observations, prototypes, inverse_variance, device):
    observed = torch.from_numpy(np.asarray(observations, dtype=np.float32)).to(device)
    prototype = torch.from_numpy(np.asarray(prototypes, dtype=np.float32)).to(device)
    weights = torch.from_numpy(np.asarray(inverse_variance, dtype=np.float32)).to(device)
    output = []
    for start in range(0, len(observed), 128):
        batch = observed[start:start + 128]
        output.append((-((batch[:, None, :] - prototype[None, :, :]).square() * weights).mean(dim=2)).cpu().numpy())
    return np.concatenate(output)


def build_contexts(blocks, split, bank, device):
    rows, contexts, checkpoints = [], {}, {}
    glyphs = np.arange(exp35.N_GLYPHS)
    for mouse in sorted({b.mouse for b in blocks}):
        sessions = [b for b in blocks if b.mouse == mouse]
        classifier = exp35.load_classifier(exp35.MODEL_ROOT / mouse / "classifier.pt", device)
        if any(b.n_features != classifier.n_features for b in sessions):
            raise ValueError(f"Frozen classifier feature mismatch for {mouse}")
        model, checkpoints[mouse] = load_encoder(mouse, sessions[0].n_features, device)
        encoding_parts = []
        for block in sessions:
            part = split.blocks[block.key]
            raw = block.features[part.train_trials]
            targets = block.trial_targets[part.train_trials]
            ids, mean_x = exp35.aggregate_by_target(raw, targets)
            if not np.array_equal(ids, glyphs) or not np.all(np.bincount(targets) == 3):
                raise ValueError(f"Expected three training repeats per glyph in {block.key}")
            log_raw = np.log1p(raw)
            glyph_variance = np.zeros(raw.shape[1], dtype=np.float64)
            for glyph in glyphs:
                glyph_variance += np.var(log_raw[targets == glyph], axis=0, ddof=1)
            glyph_variance /= len(glyphs)
            block_variance = glyph_variance
            shrunk_variance = 0.5 * block_variance + 0.5 * block_variance.mean()
            inverse_variance = 1.0 / np.maximum(shrunk_variance, 1e-6)
            encoding_parts.append(np.log1p(mean_x))
            rows.append(dict(block=block.key, mouse=mouse, normalizer=exp35.fit_session_normalizer(raw),
                             targets=block.trial_targets[part.validation_trials],
                             raw=block.features[part.validation_trials],
                             inverse_variance=inverse_variance.astype(np.float32)))
        ridge = exp35.RidgePath().fit(bank[np.tile(glyphs, len(sessions))], np.concatenate(encoding_parts))
        contexts[mouse] = dict(classifier=classifier, contrastive=model,
                               prototypes=ridge.predict(bank, exp35.ENCODING_ALPHA).astype(np.float32))
    return sorted(rows, key=lambda r: r["block"]), contexts, checkpoints


@torch.inference_mode()
def evaluate(rows, contexts, pictures, names, centered, images, cache, device, full=False):
    parts = {name: [] for name in names}
    for row in rows:
        targets = row["targets"]
        subset = np.flatnonzero(np.isin(targets, pictures))
        if not np.array_equal(targets[subset], pictures):
            raise ValueError(f"Unexpected picture subset in {row['block']}")
        if full and not np.array_equal(np.sort(targets), np.arange(exp35.N_GLYPHS)):
            raise ValueError(f"Incomplete validation block {row['block']}")
        indices = np.arange(len(targets)) if full else subset
        context = contexts[row["mouse"]]
        query = exp35.base_query(row, context, indices, device)
        image = exp35.continuous_image_scores(query["logits"], centered)
        v5 = (query["mlp_z"] + query["encoding_z"] + exp35.row_zscore(image)) / 3.0
        weighted = weighted_mse_scores(np.log1p(row["raw"][indices]), context["prototypes"],
                                       row["inverse_variance"], device)
        weighted_z = exp35.row_zscore(weighted)
        weighted_score = (query["mlp_z"] + weighted_z + exp35.row_zscore(image)) / 3.0
        keep = subset if full else slice(None)
        for name in names:
            scores = v5 if name == "v5" else weighted_score
            assigned = exp35.unique_assignment(scores)
            top = np.argsort(-scores[keep], axis=1, kind="stable")[:, :exp35.K].copy()
            parts[name].append(dict(block=row["block"], mouse=row["mouse"], targets=query["targets"][keep],
                                    assigned=assigned[keep], top32=top))
    records = {name: exp35.concatenate_records(group) for name, group in parts.items()}
    exp35.fill_ocf_cache(records, images, cache, device)
    return {name: exp35.metric_summary(record, cache) for name, record in records.items()}


def main():
    started = time.perf_counter()
    selection_path = OUT / "selection.json"
    prereg = json.loads(selection_path.read_text(encoding="utf-8"))["preregistration"]
    if tuple(prereg["candidates"]) != VARIANTS:
        raise ValueError("Preregistration disagrees with fixed candidates")
    selection = {"preregistration": prereg}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = (exp35.load_images(exp35.DATA) >= 128).astype(np.float32)
    if images.shape != (exp35.N_GLYPHS, 64, 64):
        raise ValueError("Unexpected public training gallery")
    bank = exp35.dct_encode(images, 16).astype(np.float32)
    blocks = exp35.load_feature_blocks(exp35.DATA, "train", bin_ms=50)
    if sorted(b.key for b in blocks) != [f"{m}/session{s}" for m in ("mice1", "mice2") for s in range(1, 5)]:
        raise ValueError("Expected all eight public training blocks")
    split = exp35.make_split(blocks, "task3", seed=exp35.SEED)
    rows, contexts, checkpoints = build_contexts(blocks, split, bank, device)
    del blocks
    order = np.random.default_rng(exp35.PICTURE_SEED).permutation(exp35.N_GLYPHS)
    tune, heldout = np.sort(order[:256]), np.sort(order[256:768])
    centered = torch.from_numpy(bank - bank.mean(axis=0, keepdims=True)).to(device)
    cache = {}
    tuning = evaluate(rows, contexts, tune, VARIANTS, centered, images, cache, device)
    metrics_order = tuple(prereg["maximize_tune_overall_lexicographically"])
    keys = {name: tuple(tuning[name]["overall"][metric] for metric in metrics_order) for name in VARIANTS}
    baseline_ocf = tuning["v5"]["overall"]["final_ocf"]
    candidate_ocf = tuning["noise_weighted"]["overall"]["final_ocf"]
    selection["tune"] = tuning
    selection["tune_keys"] = keys
    selection["gate"] = dict(baseline_v5_final_ocf=baseline_ocf,
                              noise_weighted_final_ocf=candidate_ocf,
                              strict_final_ocf_improvement=candidate_ocf > baseline_ocf)
    if candidate_ocf <= baseline_ocf:
        write_json(selection_path, selection)
        write_json(OUT / "results.json", dict(experiment=OUT.name, **selection,
                                               heldout=None, heldout_evaluated_variants=[],
                                               reason="Strict tune final OCF gate failed",
                                               elapsed_seconds=time.perf_counter() - started))
        return

    winner = "noise_weighted"
    selection["locked_selection"] = dict(selected=winner, selected_tune_key=keys[winner])
    write_json(selection_path, selection)

    heldout = np.sort(order[1280:1792])
    names = ("v5", winner)
    metrics = evaluate(rows, contexts, heldout, names, centered, images, cache, device, full=True)
    protocol = {
        "data_scope": "public train only", "task": "task3 single held-out trial",
        "split_seed": exp35.SEED, "picture_seed": exp35.PICTURE_SEED,
        "tune_pictures": tune.tolist(), "heldout_pictures": heldout.tolist(),
        "training_index_scope": "train_trials only for variance, session normalizers, and ridge prototypes; frozen public train checkpoints",
        "variance_estimator": "Per glyph sample variance across exactly three public train_trials on log1p responses, averaged across glyphs separately within each block, then 50% shrinkage toward the block mean variance",
        "forbidden_inputs": "No validation residuals, ID matching, or hidden data",
        "tune_assignment_scope": "per-block rectangular Hungarian on 256 tune rows against all glyphs",
        "heldout_assignment_scope": "full-block 2748-row Hungarian before retaining permutation positions [1280:1792]",
        "candidate_ranking": "stable descending scores over all 2748 glyphs; top 32 before assignment",
        "ocf": "official PNG-parity OCF on binary public training glyphs",
        "heldout_used_for_selection": False,
        "heldout_selection_statement": "Only v5 and the strictly tune-superior noise_weighted candidate are evaluated on the fixed 512 set.",
        "historical_data_disclosure": "The 512 permutation slice may have been evaluated by earlier local experiments. It is not treated as pristine.",
    }
    write_json(OUT / "results.json", dict(
        experiment=OUT.name, **selection, protocol=protocol, contrastive_checkpoints=checkpoints,
        heldout=metrics, heldout_evaluated_variants=list(names), elapsed_seconds=time.perf_counter() - started,
        environment=dict(python=sys.executable, torch=str(torch.__version__), numpy=np.__version__,
                         device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None),
    ))


if __name__ == "__main__":
    main()
