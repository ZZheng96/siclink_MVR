"""Task1 temporal candidate coverage on the fixed public split; no test-set inference."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions
from src.features import aggregate_by_target, load_feature_blocks
from src.neural import SpikeRegressor, fit_regressor, load_regressor, predict_logits
from src.retrieval import _retrieval_embedding, render_gb2312_bank
from src.ridge import dct_decode, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
BASELINE_ROOT = ROOT / "test/05_task1_pooled/artifacts/task1_pooled_mlp_bin25_singles"
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
MODEL_ROOT = HERE / "artifacts/models"
SELECTION_PATH = HERE / "artifacts/selection.json"
RESULTS_PATH = HERE / "results.json"
SEED = 42
PICTURE_SEED = 123
TOP_K = 32
BASELINE = "baseline"
CANDIDATES = ("pool50", "delay50", "delay50_pool50_reliable")
VARIANTS = (BASELINE, *CANDIDATES)
EPOCHS = {"mice1": 8, "mice2": 11}
BASELINE_TUNE_OCF = 16.811784552406262
BASELINE_HELDOUT_OCF = 18.624308986308705
BASELINE_ATOL = 1e-5


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prepare_protocol(blocks):
    split = make_split(blocks, "task1", seed=SEED)
    shuffled = np.random.default_rng(PICTURE_SEED).permutation(split.validation_pictures)
    model_validation_pictures = np.sort(shuffled[:64])
    tune_pictures = np.sort(shuffled[:16])
    heldout_pictures = np.sort(shuffled[64:])
    if (len(model_validation_pictures), len(tune_pictures), len(heldout_pictures)) != (64, 16, 211):
        raise ValueError("Fixed Task1 64/16/211 picture partition was not reproduced")
    if not set(tune_pictures).issubset(set(model_validation_pictures)):
        raise ValueError("TUNE16 must remain the predefined subset of model validation")
    if set(model_validation_pictures) & set(heldout_pictures):
        raise ValueError("Model validation and heldout pictures overlap")

    rows = []
    training = {}
    for mouse in sorted({block.mouse for block in blocks}):
        train_x_parts = []
        train_target_parts = []
        validation_x_parts = []
        validation_target_parts = []
        mouse_blocks = [block for block in blocks if block.mouse == mouse]
        for block in mouse_blocks:
            part = split.blocks[block.key]
            train_x = block.features[part.train_trials]
            train_targets = block.trial_targets[part.train_trials]
            if set(train_targets) & set(split.validation_pictures):
                raise ValueError(f"Training labels leak validation pictures in {block.key}")
            mean_targets, mean_x = aggregate_by_target(train_x, train_targets)
            validation_targets, validation_x = aggregate_by_target(
                block.features[part.validation_trials], block.trial_targets[part.validation_trials]
            )
            train_x_parts.extend((mean_x, train_x))
            train_target_parts.extend((mean_targets, train_targets))
            keep = np.isin(validation_targets, model_validation_pictures)
            validation_x_parts.append(validation_x[keep])
            validation_target_parts.append(validation_targets[keep])
            rows.append({
                "block": block.key,
                "mouse": mouse,
                "n_units": block.n_units,
                "targets": validation_targets,
                "counts": validation_x,
            })
        training[mouse] = {
            "n_units": mouse_blocks[0].n_units,
            "train_x": np.concatenate(train_x_parts),
            "train_targets": np.concatenate(train_target_parts).astype(np.int64, copy=False),
            "validation_x": np.concatenate(validation_x_parts),
            "validation_targets": np.concatenate(validation_target_parts).astype(np.int64, copy=False),
        }
    rows.sort(key=lambda row: row["block"])
    return {
        "rows": rows,
        "training": training,
        "validation_picture_count": len(split.validation_pictures),
        "model_validation_pictures": model_validation_pictures,
        "tune_pictures": tune_pictures,
        "heldout_pictures": heldout_pictures,
    }


def repeat_icc(features, targets):
    """Single-feature-repeat ICC(1,1), estimated separately within each session."""
    targets = np.asarray(targets)
    unique, means = aggregate_by_target(features, targets)
    counts = np.bincount(targets, minlength=int(targets.max()) + 1)[unique]
    if not np.all(counts == counts[0]) or counts[0] < 2:
        raise ValueError("Reliability estimation requires equal within-session repeats")
    repeats = int(counts[0])
    grand = np.asarray(means, dtype=np.float64).mean(axis=0)
    between_ss = np.square(np.asarray(means, dtype=np.float64) - grand).sum(axis=0) * repeats
    total_ss = np.square(features, dtype=np.float64).sum(axis=0)
    fitted_ss = np.square(means, dtype=np.float64).sum(axis=0) * repeats
    within_ss = np.maximum(total_ss - fitted_ss, 0.0)
    between_var = between_ss / (len(unique) - 1)
    within_var = within_ss / (len(features) - len(unique))
    denominator = between_var + (repeats - 1) * within_var
    icc = np.divide(
        between_var - within_var,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    return np.clip(icc, 0.0, 1.0).astype(np.float32)


def training_diagnostics(blocks, protocol):
    """Use only source-training trials; never inspect TUNE/heldout responses here."""
    split = make_split(blocks, "task1", seed=SEED)
    masks = {}
    diagnostics = {}
    for mouse in sorted(protocol["training"]):
        mouse_blocks = [block for block in blocks if block.mouse == mouse]
        n_units = mouse_blocks[0].n_units
        session_rates = []
        session_iccs = []
        for block in mouse_blocks:
            part = split.blocks[block.key]
            x = block.features[part.train_trials]
            targets = block.trial_targets[part.train_trials]
            session_rates.append(x.reshape(len(x), n_units, 10).mean(axis=0, dtype=np.float64))
            session_iccs.append(repeat_icc(x, targets).reshape(n_units, 10))
        rates = np.mean(session_rates, axis=0)
        iccs = np.stack(session_iccs)
        unit_reliability = iccs.mean(axis=(0, 2))
        selected_count = (n_units + 1) // 2
        selected = np.sort(np.argsort(-unit_reliability, kind="stable")[:selected_count])
        masks[mouse] = selected.astype(np.int64)
        active = rates.mean(axis=1) > 1e-8
        normalized = rates[active] / np.maximum(rates[active].mean(axis=1, keepdims=True), 1e-12)
        bins25 = []
        for index in range(10):
            bins25.append({
                "bin": index,
                "start_ms": index * 25,
                "end_ms": (index + 1) * 25,
                "spikes_per_unit_trial": float(rates[:, index].mean()),
                "active_unit_normalized_rate": float(normalized[:, index].mean()),
                "repeat_icc_mean": float(iccs[:, :, index].mean()),
                "repeat_icc_median": float(np.median(iccs[:, :, index])),
            })
        pooled_rates = rates.reshape(n_units, 5, 2).mean(axis=2)
        pooled_icc = iccs.reshape(4, n_units, 5, 2).mean(axis=3)
        bins50 = [{
            "bin": index,
            "start_ms": index * 50,
            "end_ms": (index + 1) * 50,
            "spikes_per_unit_trial": float(pooled_rates[:, index].mean()),
            "repeat_icc_mean_of_25ms_bins": float(pooled_icc[:, :, index].mean()),
        } for index in range(5)]
        diagnostics[mouse] = {
            "source_sessions": [block.key for block in mouse_blocks],
            "n_units": n_units,
            "reliability_selected_count": len(selected),
            "reliability_selected_unit_indices": selected.tolist(),
            "reliability_selected_mean_icc": float(unit_reliability[selected].mean()),
            "reliability_excluded_mean_icc": float(np.delete(unit_reliability, selected).mean()),
            "bins25": bins25,
            "pooled50_summary": bins50,
        }
        print(f"TRAIN TIMING {mouse} {json.dumps(bins25)}", flush=True)
        print(f"RELIABILITY {mouse} selected={len(selected)}/{n_units}", flush=True)
    return masks, diagnostics


def transform_features(features, n_units, variant, selected_units=None):
    features = np.asarray(features, dtype=np.float32)
    if variant == BASELINE:
        return features
    temporal = features.reshape(len(features), n_units, 10)
    if variant == "pool50":
        temporal = temporal.reshape(len(features), n_units, 5, 2).sum(axis=3, dtype=np.float32)
    elif variant == "delay50":
        temporal = temporal[:, :, 2:]
    elif variant == "delay50_pool50_reliable":
        if selected_units is None:
            raise ValueError("Reliability mask is required")
        temporal = temporal[:, selected_units, 2:]
        temporal = temporal.reshape(len(features), len(selected_units), 4, 2).sum(axis=3, dtype=np.float32)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return np.ascontiguousarray(temporal.reshape(len(features), -1))


def checkpoint_path(variant, mouse):
    if variant == BASELINE:
        return BASELINE_ROOT / mouse / "regressor.pt"
    return MODEL_ROOT / variant / mouse / "regressor.pt"


def train_candidates(protocol, images, masks, device, reuse_existing=False):
    image_coefficients = dct_encode(images, 32)
    records = []
    for variant in CANDIDATES:
        for mouse in sorted(protocol["training"]):
            base = protocol["training"][mouse]
            validation_x = transform_features(
                base["validation_x"], base["n_units"], variant, masks[mouse]
            )
            validation_y = image_coefficients[base["validation_targets"]]
            output = MODEL_ROOT / variant / mouse
            checkpoint = checkpoint_path(variant, mouse)
            history_path = output / "history.json"
            reused = bool(reuse_existing and checkpoint.is_file() and history_path.is_file())
            if reused:
                model = load_regressor(checkpoint, "cpu")
                if model.n_features != validation_x.shape[1] or model.n_outputs != 1024:
                    raise ValueError(f"Existing checkpoint feature mismatch: {checkpoint}")
                history = json.loads(history_path.read_text(encoding="utf-8"))
                del model
                print(f"REUSE {variant}/{mouse}: checkpoint={checkpoint}", flush=True)
            else:
                train_x = transform_features(
                    base["train_x"], base["n_units"], variant, masks[mouse]
                )
                train_y = image_coefficients[base["train_targets"]]
                print(
                    f"TRAIN {variant}/{mouse}: train={train_x.shape} validation={validation_x.shape}",
                    flush=True,
                )
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                model = SpikeRegressor(train_x.shape[1], 1024, hidden=512, dropout=0.4)
                history = fit_regressor(
                    model,
                    (train_x, train_y),
                    (validation_x, validation_y),
                    output,
                    epochs=EPOCHS[mouse],
                    batch_size=256,
                    learning_rate=3e-4,
                    weight_decay=0.01,
                    device=device,
                    seed=SEED,
                )
                del model, train_x, train_y
            best = min(history, key=lambda row: row["validation_loss"])
            records.append({
                "variant": variant,
                "mouse": mouse,
                "n_features": validation_x.shape[1],
                "train_rows": len(base["train_x"]),
                "model_validation_rows": len(validation_x),
                "checkpoint_epoch": best["epoch"],
                "best_validation_loss": best["validation_loss"],
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "reused_existing_checkpoint": reused,
                "history": history,
            })
            del validation_x, validation_y
            release_memory()
    return records


@torch.inference_mode()
def make_bank_embedding(bank, device):
    # Keep the baseline's original full-bank FFT batch; chunking can change near-tie ranks.
    images = torch.from_numpy(bank[:, None]).to(device)
    embedding = F.normalize(_retrieval_embedding(images, "directional2"), dim=1)
    if embedding.shape != (6763, 12288):
        raise ValueError(f"Unexpected directional2 bank embedding: {tuple(embedding.shape)}")
    return embedding


@torch.inference_mode()
def continuous_scores(continuous, bank_embedding, device):
    output = []
    for start in range(0, len(continuous), 64):
        batch = torch.from_numpy(continuous[start:start + 64, None]).to(device)
        query = F.normalize(_retrieval_embedding(batch, "directional2"), dim=1)
        output.append((query @ bank_embedding.T).float().cpu().numpy())
    return np.concatenate(output)


def predict_parts(protocol, variant, pictures, masks, bank_embedding, device):
    models = {
        mouse: load_regressor(checkpoint_path(variant, mouse), device)
        for mouse in sorted(protocol["training"])
    }
    parts = []
    for row in protocol["rows"]:
        indices = np.flatnonzero(np.isin(row["targets"], pictures))
        targets = row["targets"][indices]
        if not np.array_equal(targets, pictures):
            raise ValueError(f"Unexpected ordered picture subset in {row['block']}")
        if variant == BASELINE:
            # Reproduce exp18/21 exactly: infer the complete 275-picture block, then subset.
            coefficients = predict_logits(models[row["mouse"]], row["counts"], device=device)
            continuous = dct_decode(coefficients, 32)[indices]
        else:
            # Losing candidates never receive heldout inference.
            counts = transform_features(
                row["counts"][indices], row["n_units"], variant, masks[row["mouse"]]
            )
            coefficients = predict_logits(models[row["mouse"]], counts, device=device)
            continuous = dct_decode(coefficients, 32)
        parts.append({
            "block": row["block"],
            "mouse": row["mouse"],
            "targets": targets,
            "pictures": targets.copy(),
            "scores": continuous_scores(continuous, bank_embedding, device),
        })
    for model in models.values():
        del model
    del models
    release_memory()
    print(f"RANKING SCORES {variant}: pictures={len(pictures)} blocks={len(parts)}", flush=True)
    return parts


def parts_to_records(parts, baseline_parts=None, device="cuda"):
    if baseline_parts is not None and len(parts) != len(baseline_parts):
        raise ValueError("Candidate and baseline block counts differ")
    output = {"targets": [], "pictures": [], "groups": [], "mice": [], "top32": []}
    for index, part in enumerate(parts):
        scores = part["scores"]
        if baseline_parts is not None:
            baseline = baseline_parts[index]
            if part["block"] != baseline["block"] or not np.array_equal(part["targets"], baseline["targets"]):
                raise ValueError("Candidate/baseline rows are not aligned")
            scores = 0.5 * (scores + baseline["scores"])
        top = torch.topk(
            torch.from_numpy(scores).to(device), k=TOP_K, dim=1, largest=True, sorted=True
        ).indices.cpu().numpy().astype(np.int64)
        output["targets"].append(part["targets"])
        output["pictures"].append(part["pictures"])
        output["groups"].append(np.full(len(top), part["block"]))
        output["mice"].append(np.full(len(top), part["mouse"]))
        output["top32"].append(top)
    return {key: np.concatenate(value) for key, value in output.items()}


def fill_ocf_cache(record_groups, images, bank, cache, device):
    pair_parts = []
    for records in record_groups.values():
        pair_parts.append(np.column_stack((
            records["top32"].reshape(-1), np.repeat(records["pictures"], TOP_K)
        )))
    pairs = np.unique(np.concatenate(pair_parts).astype(np.int64), axis=0)
    missing = [tuple(map(int, pair)) for pair in pairs if tuple(map(int, pair)) not in cache]
    print(f"OCF unique_pairs={len(pairs)} new={len(missing)} cached={len(cache)}", flush=True)
    for start in range(0, len(missing), 512):
        chunk = missing[start:start + 512]
        predicted = np.fromiter((pair[0] for pair in chunk), dtype=np.int64)
        targets = np.fromiter((pair[1] for pair in chunk), dtype=np.int64)
        scores = score_predictions(bank[predicted], images[targets], device=device, batch_size=32)
        for pair, score in zip(chunk, scores):
            cache[pair] = float(score)
        if start % 8192 == 0 or start + len(chunk) == len(missing):
            print(f"OCF scored {min(start + len(chunk), len(missing))}/{len(missing)}", flush=True)


def metric_summary(records, cache):
    targets = records["pictures"]
    top = records["top32"]
    values = np.asarray([
        cache[(int(predicted), int(target))]
        for predicted, target in zip(top.reshape(-1), np.repeat(targets, TOP_K))
    ], dtype=np.float64).reshape(len(targets), TOP_K)

    def summarize(mask):
        current = values[mask]
        best_rank = current.argmax(axis=1) + 1
        top1 = current[:, 0]
        oracle32 = current.max(axis=1)
        return {
            "n": int(mask.sum()),
            "top1_ocf": float(top1.mean()),
            "oracle_top1_ocf": float(top1.mean()),
            "oracle_top32_ocf": float(oracle32.mean()),
            "oracle32_gain_over_top1_ocf": float((oracle32 - top1).mean()),
            "oracle_best_rank_mean": float(best_rank.mean()),
        }

    all_rows = np.ones(len(targets), dtype=bool)
    return {
        "overall": summarize(all_rows),
        "by_mouse": {
            mouse: summarize(records["mice"] == mouse)
            for mouse in sorted(set(records["mice"].tolist()))
        },
        "by_block": {
            block: summarize(records["groups"] == block)
            for block in sorted(set(records["groups"].tolist()))
        },
    }


def selection_key(summary):
    row = summary["overall"]
    return row["oracle_top32_ocf"], row["top1_ocf"]


def exact_identity_diagnostic(images, bank):
    lookup = {
        np.packbits(np.asarray(glyph, dtype=np.uint8)).tobytes(): index
        for index, glyph in enumerate(bank)
    }
    matches = sum(
        np.packbits(np.asarray(image, dtype=np.uint8)).tobytes() in lookup for image in images
    )
    return {
        "exact_training_to_bank_pixel_matches": int(matches),
        "identity_recall_available": matches == len(images),
        "reason": None if matches == len(images) else (
            "Public training glyphs do not have exact pixel matches in the rendered bank; "
            "oracle OCF is reported as candidate coverage, not identity recall."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Resume with matching candidate checkpoints/histories from an earlier invocation.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This experiment requires the official CUDA OCF implementation")
    images = (load_images(DATA) >= 128).astype(np.float32)
    if images.shape != (2748, 64, 64):
        raise ValueError("Unexpected public training gallery")
    bank = render_gb2312_bank(FONT_PATH, font_size=52, variation="Regular")
    if bank.shape != (6763, 64, 64):
        raise ValueError("The independent strict-GB2312 bank must contain 6763 glyphs")
    blocks = load_feature_blocks(DATA, "train", bin_ms=25)
    if sorted(block.key for block in blocks) != [
        f"{mouse}/session{session}" for mouse in ("mice1", "mice2") for session in range(1, 5)
    ]:
        raise ValueError("Expected all eight public training blocks")
    protocol = prepare_protocol(blocks)
    masks, timing = training_diagnostics(blocks, protocol)
    training = train_candidates(
        protocol, images, masks, device, reuse_existing=args.reuse_existing
    )

    bank_embedding = make_bank_embedding(bank, device)
    tune_baseline_parts = predict_parts(
        protocol, BASELINE, protocol["tune_pictures"], masks, bank_embedding, device
    )
    tune_records = {BASELINE: parts_to_records(tune_baseline_parts, device=device)}
    for variant in CANDIDATES:
        parts = predict_parts(protocol, variant, protocol["tune_pictures"], masks, bank_embedding, device)
        tune_records[variant] = parts_to_records(parts, tune_baseline_parts, device)
        del parts
    del bank_embedding
    release_memory()

    ocf_cache = {}
    fill_ocf_cache(tune_records, images, bank, ocf_cache, device)
    tuning = {name: metric_summary(tune_records[name], ocf_cache) for name in VARIANTS}
    baseline_tune_error = abs(tuning[BASELINE]["overall"]["top1_ocf"] - BASELINE_TUNE_OCF)
    if baseline_tune_error > BASELINE_ATOL:
        raise RuntimeError(
            f"Baseline TUNE reproduction failed: {baseline_tune_error} > {BASELINE_ATOL}"
        )
    for name in VARIANTS:
        print(f"TUNE {name} {json.dumps(tuning[name]['overall'])}", flush=True)

    selected = max(VARIANTS, key=lambda name: selection_key(tuning[name]))
    selected_key = selection_key(tuning[selected])
    selection = {
        "experiment": "33_task1_temporal_candidates",
        "data_scope": "public train only; competition test files are never read",
        "protocol": {
            "task": "task1 character-held-out split",
            "split_seed": SEED,
            "picture_seed": PICTURE_SEED,
            "validation_picture_count": protocol["validation_picture_count"],
            "model_validation_pictures": protocol["model_validation_pictures"].tolist(),
            "tune_pictures": protocol["tune_pictures"].tolist(),
            "heldout_pictures": protocol["heldout_pictures"].tolist(),
            "tune_picture_count": 16,
            "heldout_picture_count": 211,
            "model_checkpoint_selection": "minimum DCT-MSE on the fixed 64 model-validation pictures",
            "winner_selection_count": 1,
            "selection_rule": (
                "One lexicographic maximum over baseline and three candidates: TUNE16 oracle_top32_ocf, "
                "then TUNE16 top1_ocf; fixed VARIANTS order breaks exact ties; baseline is allowed."
            ),
            "heldout_rule": (
                "Baseline follows the original full-275-block inference path and is subset before any "
                "metric; losing candidates receive TUNE inference only. Selection is persisted before "
                "selected-candidate heldout inference and before all heldout OCF scoring. Only baseline "
                "and the selected variant receive heldout retrieval or OCF evaluation."
            ),
            "heldout_used_for_selection": False,
            "historical_heldout_disclosure": (
                "This fixed local heldout set was used by earlier experiments; it is historically reused, "
                "not a pristine external confirmation, although it plays no role in this run's winner choice."
            ),
        },
        "config": {
            "source_bin_ms": 25,
            "stimulus_window_ms": [0, 250],
            "bin_rule": "floor(spike_samples/(30000*bin_ms/1000)); bin k starts at k*bin_ms after stimulus onset",
            "pool50_rule": "sum adjacent nonoverlapping 25ms spike-count bins; exactly equivalent to direct 50ms counting",
            "reliability_rule": (
                "Per-session training-trial ICC(1,1) from character repeats, clipped to [0,1]; average over "
                "four sessions and ten bins; fixed top half of units per mouse; no TUNE/heldout labels or responses"
            ),
            "variants": {
                "baseline": "frozen seed42 full 0-250ms 25ms-bin DCT32 MLP; directional2 top32",
                "pool50": "retrained all-unit 0-250ms five-50ms-bin DCT32 MLP; equal-score fusion with baseline",
                "delay50": "retrained all-unit 50-250ms eight-25ms-bin DCT32 MLP; equal-score fusion with baseline",
                "delay50_pool50_reliable": (
                    "retrained top-half reliable-unit 50-250ms four-50ms-bin DCT32 MLP; "
                    "equal-score fusion with baseline"
                ),
            },
            "candidate_fusion": "0.5*baseline directional2 cosine + 0.5*temporal-model directional2 cosine; no weight sweep",
            "training": "same seed42 DCT32-MSE MLP as baseline; hidden512 dropout0.4 AdamW lr3e-4 weight_decay0.01",
            "epochs": EPOCHS,
            "retrieval_bank": "independent complete strict-GB2312, NotoSansSC Regular size52, 6763 glyphs",
            "retrieval_metric": "directional2 cosine",
            "top_k": TOP_K,
            "ocf": "official PNG-parity OCF between candidate bank glyph and public target image",
            "identity_recall": exact_identity_diagnostic(images, bank),
            "latency_limitation": "No pre-stimulus interval is available; bin profiles and repeat ICC diagnose timing but do not prove physiological onset latency.",
        },
        "training_diagnostics": timing,
        "training": training,
        "tune": tuning,
        "selected": selected,
        "selected_tune_key": list(selected_key),
        "baseline_tune_verification": {
            "expected_top1_ocf": BASELINE_TUNE_OCF,
            "actual_top1_ocf": tuning[BASELINE]["overall"]["top1_ocf"],
            "absolute_error": baseline_tune_error,
            "absolute_tolerance": BASELINE_ATOL,
            "passed": True,
        },
    }
    write_json(SELECTION_PATH, selection)
    print(f"SELECTION LOCKED {selected}; key={selected_key}; path={SELECTION_PATH}", flush=True)

    heldout_names = tuple(dict.fromkeys((BASELINE, selected)))
    bank_embedding = make_bank_embedding(bank, device)
    heldout_baseline_parts = predict_parts(
        protocol, BASELINE, protocol["heldout_pictures"], masks, bank_embedding, device
    )
    heldout_records = {BASELINE: parts_to_records(heldout_baseline_parts, device=device)}
    if selected != BASELINE:
        selected_parts = predict_parts(
            protocol, selected, protocol["heldout_pictures"], masks, bank_embedding, device
        )
        heldout_records[selected] = parts_to_records(selected_parts, heldout_baseline_parts, device)
        del selected_parts
    del bank_embedding
    release_memory()
    if set(heldout_records) != set(heldout_names):
        raise RuntimeError("Heldout evaluation contains a losing variant")
    fill_ocf_cache(heldout_records, images, bank, ocf_cache, device)
    heldout = {name: metric_summary(heldout_records[name], ocf_cache) for name in heldout_names}
    baseline_heldout_error = abs(
        heldout[BASELINE]["overall"]["top1_ocf"] - BASELINE_HELDOUT_OCF
    )
    if baseline_heldout_error > BASELINE_ATOL:
        raise RuntimeError(
            f"Baseline heldout reproduction failed: {baseline_heldout_error} > {BASELINE_ATOL}"
        )

    result = dict(selection)
    result["heldout"] = heldout
    result["heldout_evaluated_variants"] = list(heldout_names)
    result["baseline_heldout_verification"] = {
        "checkpoint_root": str(BASELINE_ROOT.relative_to(ROOT)),
        "expected_top1_ocf": BASELINE_HELDOUT_OCF,
        "actual_top1_ocf": heldout[BASELINE]["overall"]["top1_ocf"],
        "absolute_error": baseline_heldout_error,
        "absolute_tolerance": BASELINE_ATOL,
        "passed": True,
    }
    result["heldout_selected_minus_baseline"] = {
        metric: heldout[selected]["overall"][metric] - heldout[BASELINE]["overall"][metric]
        for metric in ("top1_ocf", "oracle_top32_ocf")
    }
    result["environment"] = {
        "python": sys.executable,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0),
    }
    result["artifacts"] = {"selection_json": str(SELECTION_PATH.relative_to(ROOT))}
    result["elapsed_seconds"] = time.perf_counter() - started
    write_json(RESULTS_PATH, result)
    print(f"FINAL {RESULTS_PATH}", flush=True)
    for name in heldout_names:
        print(f"HELDOUT {name} {json.dumps(heldout[name])}", flush=True)
    print(f"DELTA {json.dumps(result['heldout_selected_minus_baseline'])}", flush=True)


if __name__ == "__main__":
    main()
