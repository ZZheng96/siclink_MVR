"""Falsifiable GB2312 font calibration before any Task 1 font-bank retrieval."""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.neural import load_regressor, predict_logits
from src.retrieval import retrieve_glyphs
from src.ridge import dct_decode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
MODEL_ROOT = ROOT / "test/05_task1_pooled/artifacts/task1_pooled_mlp_bin25_singles"
OUTPUT_DIR = ROOT / "test/15_font_calibration"
ARTIFACTS = OUTPUT_DIR / "artifacts"
CALIBRATION_PATH = OUTPUT_DIR / "calibration.json"
RESULTS_PATH = OUTPUT_DIR / "results_task1_font.json"
SELECTED_NPZ = ARTIFACTS / "task1_font_selected.npz"
CALIBRATION_NPZ = ARTIFACTS / "stage_a_best_matches.npz"

FONT_SPECS = (
    {"file": "NotoSansSC-VF.ttf", "collection_index": 0, "variation": "Regular"},
    {"file": "simsun.ttc", "collection_index": 0, "variation": None},
    {"file": "simhei.ttf", "collection_index": 0, "variation": None},
    {"file": "simkai.ttf", "collection_index": 0, "variation": None},
    {"file": "simfang.ttf", "collection_index": 0, "variation": None},
    {"file": "msyh.ttc", "collection_index": 0, "variation": None},
    {"file": "Deng.ttf", "collection_index": 0, "variation": None},
)
FONT_DIRECTORY = Path(r"C:\Windows\Fonts")
FONT_SIZES = (40, 44, 48, 52, 56, 60)
CALIBRATION_SAMPLE_SEED = 15001
CALIBRATION_OCF_SEED = 15002
RANDOM_GLYPH_SEED = 15003
CALIBRATION_SAMPLE_COUNT = 256
CALIBRATION_OCF_COUNT = 64
BASELINE_OCF = 18.4831207873
BASELINE_ATOL = 1e-5

# These criteria are fixed before rendering or inspecting any calibration scores.
MIN_MEDIAN_COSINE = 0.80
MIN_MATCHED_OCF_MEAN = 50.0
MIN_OCF_MARGIN_OVER_RANDOM = 20.0
MIN_OCF_RATIO_OVER_RANDOM = 2.0


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def gb2312_characters():
    characters = []
    undefined = []
    for lead in range(0xB0, 0xF8):
        for trail in range(0xA1, 0xFF):
            try:
                character = bytes((lead, trail)).decode("gb2312")
            except UnicodeDecodeError:
                undefined.append({"lead": f"0x{lead:02X}", "trail": f"0x{trail:02X}"})
            else:
                if not ("\u4e00" <= character <= "\u9fff"):
                    raise ValueError(f"Unexpected non-Han GB2312 decode at {lead:02X}{trail:02X}: {character!r}")
                characters.append(character)
    assert lead == 0xF7 and trail == 0xFE
    slot_count = (0xF7 - 0xB0 + 1) * (0xFE - 0xA1 + 1)
    if slot_count != 6768 or len(characters) + len(undefined) != slot_count:
        raise RuntimeError("GB2312 grid accounting failed.")
    if len(characters) != len(set(characters)):
        raise RuntimeError("GB2312 decoding produced duplicate characters.")
    return characters, {
        "encoding": "gb2312",
        "decode_errors": "strict",
        "lead_range": "B0-F7",
        "trail_range": "A1-FE",
        "grid_slot_count": slot_count,
        "decoded_character_count": len(characters),
        "undefined_slot_count": len(undefined),
        "undefined_slots": undefined,
        "candidate_order": "ascending lead, then ascending trail; undefined slots omitted",
        "note": "The grid has 6768 code positions but only 6763 strict-gb2312 Han characters; D7FA-D7FE are undefined.",
    }


def font_inventory():
    inventory = []
    for order, spec in enumerate(FONT_SPECS):
        path = FONT_DIRECTORY / spec["file"]
        row = {
            "font_order": order,
            **spec,
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "license_status": "unverified_for_this_local_file",
            "redistribution_assumed": False,
            "license_note": (
                "Local presence permits this experiment only and does not establish redistribution rights. "
                "No font file is copied into outputs. Verify the exact font license before redistributing fonts "
                "or a rendered bank; Windows system fonts must not be treated as redistributable."
            ),
        }
        inventory.append(row)
    return inventory


def load_font(config):
    font = ImageFont.truetype(
        config["path"], config["font_size"], index=config["collection_index"]
    )
    if config["variation"] is not None:
        font.set_variation_by_name(config["variation"])
    return font


def render_glyph(character, font):
    # Center each thresholded glyph's own ink bbox, independently of advance width.
    side = max(128, 3 * font.size)
    origin = font.size
    canvas = Image.new("L", (side, side), 0)
    ImageDraw.Draw(canvas).text((origin, origin), character, font=font, fill=255)
    binary = np.asarray(canvas, dtype=np.uint8) >= 128
    ys, xs = np.nonzero(binary)
    output = np.zeros((64, 64), dtype=np.uint8)
    if not len(xs):
        return output
    glyph = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if glyph.shape[0] > 64 or glyph.shape[1] > 64:
        raise ValueError(f"Glyph does not fit 64x64 at font size {font.size}: {glyph.shape}")
    y = (64 - glyph.shape[0]) // 2
    x = (64 - glyph.shape[1]) // 2
    output[y:y + glyph.shape[0], x:x + glyph.shape[1]] = glyph
    return output


def render_bank(characters, config):
    font = load_font(config)
    bank = np.zeros((len(characters), 64, 64), dtype=np.uint8)
    for index, character in enumerate(characters):
        bank[index] = render_glyph(character, font)
    return bank


def binary_vectors_32(images):
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (64, 64):
        raise ValueError("Expected images shaped (N, 64, 64).")
    pooled = images.reshape(len(images), 32, 2, 32, 2).mean(axis=(2, 4))
    return (pooled >= 0.5).astype(np.float32).reshape(len(images), -1)


def release_arrays():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cosine_matches(bank, calibration_images, characters, device):
    candidate_vectors = binary_vectors_32(bank)
    calibration_vectors = binary_vectors_32(calibration_images)
    candidate_vectors /= np.maximum(np.linalg.norm(candidate_vectors, axis=1, keepdims=True), 1e-12)
    calibration_vectors /= np.maximum(np.linalg.norm(calibration_vectors, axis=1, keepdims=True), 1e-12)
    nearest_parts = []
    best_parts = []
    if device == "cuda":
        candidate_tensor = torch.from_numpy(candidate_vectors).to(device)
        calibration_tensor = torch.from_numpy(calibration_vectors).to(device)
        for start in range(0, len(calibration_tensor), 32):
            similarities = calibration_tensor[start:start + 32] @ candidate_tensor.T
            best, nearest = similarities.max(dim=1)
            best_parts.append(best.cpu().numpy())
            nearest_parts.append(nearest.cpu().numpy())
        del similarities, best, nearest, candidate_tensor, calibration_tensor
    else:
        for start in range(0, len(calibration_vectors), 32):
            similarities = calibration_vectors[start:start + 32] @ candidate_vectors.T
            nearest_parts.append(similarities.argmax(axis=1))
            best_parts.append(similarities.max(axis=1))
        del similarities
    nearest_indices = np.concatenate(nearest_parts).astype(np.int64)
    best_cosines = np.concatenate(best_parts).astype(np.float64)
    counts = np.bincount(nearest_indices, minlength=len(characters))
    unique_count = int(np.count_nonzero(counts))
    result = {
        "n": len(nearest_indices),
        "median_best_cosine": float(np.median(best_cosines)),
        "mean_best_cosine": float(best_cosines.mean()),
        "std_best_cosine": float(best_cosines.std()),
        "min_best_cosine": float(best_cosines.min()),
        "p10_best_cosine": float(np.quantile(best_cosines, 0.1)),
        "p90_best_cosine": float(np.quantile(best_cosines, 0.9)),
        "top1_unique_candidate_count": unique_count,
        "top1_duplicate_occupancy_rate": float((len(nearest_indices) - unique_count) / len(nearest_indices)),
        "top1_images_on_repeated_candidate_rate": float(np.mean(counts[nearest_indices] > 1)),
        "top1_max_occupancy": int(counts.max(initial=0)),
        "empty_candidate_count": int(np.sum(~bank.any(axis=(1, 2)))),
    }
    del candidate_vectors, calibration_vectors, counts
    release_arrays()
    return result, nearest_indices, best_cosines


def ranked_configurations(rows):
    return sorted(rows, key=lambda row: (
        -row["median_best_cosine"],
        -row["mean_best_cosine"],
        row["top1_duplicate_occupancy_rate"],
        row["font_order"],
        row["font_size"],
    ))


def ocf_summary(predictions, references, device):
    predictions = np.asarray(predictions, dtype=np.float32)
    scores = score_predictions(predictions, references, device=device, batch_size=8)
    return {
        "n": len(scores),
        "ocf_mean": float(scores.mean()),
        "ocf_median": float(np.median(scores)),
        "ocf_std": float(scores.std()),
    }, scores


def stage_a(images, split, characters, character_metadata, inventory, device):
    rng_validation = np.random.default_rng(123)
    calibration_validation = np.sort(rng_validation.permutation(2748)[:227])
    all_indices = np.arange(len(images))
    remaining = np.setdiff1d(all_indices, calibration_validation)
    # The additional Task 1 exclusion prevents any test11/test14 tune/held-out leakage.
    eligible = np.setdiff1d(remaining, split.validation_pictures)
    if np.intersect1d(eligible, calibration_validation).size or np.intersect1d(eligible, split.validation_pictures).size:
        raise RuntimeError("Calibration leakage check failed.")
    calibration_indices = np.sort(
        np.random.default_rng(CALIBRATION_SAMPLE_SEED).permutation(eligible)[:CALIBRATION_SAMPLE_COUNT]
    )
    ocf_positions = np.sort(
        np.random.default_rng(CALIBRATION_OCF_SEED).permutation(len(calibration_indices))[:CALIBRATION_OCF_COUNT]
    )
    calibration_images = images[calibration_indices]
    configurations = []
    for font_row in inventory:
        if not font_row["exists"]:
            continue
        for font_size in FONT_SIZES:
            configurations.append({
                "config_id": f"font{font_row['font_order']}_size{font_size}",
                "font_order": font_row["font_order"],
                "font_file": font_row["file"],
                "path": font_row["path"],
                "collection_index": font_row["collection_index"],
                "variation": font_row["variation"],
                "font_size": font_size,
            })

    result = {
        "experiment": "15_font_calibration",
        "stage": "A",
        "status": "stage_a_running",
        "comparable": False,
        "score_scope": "local calibration/validation only; not a Kaggle score",
        "character_set": character_metadata,
        "font_inventory": inventory,
        "rendering": {
            "library": "Pillow",
            "output": "binary white-on-black uint8 64x64 with values {0,1}",
            "raster_threshold": 128,
            "centering": "each thresholded glyph ink bbox centered independently with floor((64-extent)/2)",
            "fitted_offsets": False,
            "font_sizes": list(FONT_SIZES),
            "downsample": "nonoverlapping 2x2 area mean, threshold >=0.5 to binary 32x32",
            "similarity": "float32 L2-normalized 32x32 binary-vector cosine nearest neighbor",
            "batch_size": 32,
        },
        "protocol": {
            "calibration_validation_rule": "set(np.random.default_rng(123).permutation(2748)[:227])",
            "calibration_validation_picture_count": len(calibration_validation),
            "calibration_validation_pictures": calibration_validation.tolist(),
            "task1_split_seed": 42,
            "task1_validation_picture_count": len(split.validation_pictures),
            "task1_validation_pictures": split.validation_pictures.tolist(),
            "validation_intersection_count": int(np.intersect1d(calibration_validation, split.validation_pictures).size),
            "remaining_after_227_exclusion_count": len(remaining),
            "eligible_after_additional_task1_exclusion_count": len(eligible),
            "calibration_sample_seed": CALIBRATION_SAMPLE_SEED,
            "calibration_picture_count": len(calibration_indices),
            "calibration_pictures": calibration_indices.tolist(),
            "calibration_ocf_seed": CALIBRATION_OCF_SEED,
            "calibration_ocf_positions": ocf_positions.tolist(),
            "calibration_ocf_picture_count": len(ocf_positions),
            "calibration_ocf_pictures": calibration_indices[ocf_positions].tolist(),
            "random_glyph_seed": RANDOM_GLYPH_SEED,
            "validation_use": "Neither the specified 227 calibration exclusions nor any of the 275 Task 1 validation pictures is rendered, matched, or used to select rendering parameters in Stage A.",
        },
        "preregistered_selection_rules": {
            "best_configuration": "highest median_best_cosine; ties by highest mean_best_cosine, lower top1_duplicate_occupancy_rate, earlier font order, then smaller font size",
            "top1_duplicate_occupancy_rate": "(N - number of unique top1 candidate indices) / N",
            "stage_b_candidates": "the first at most 3 configurations in the same calibration ranking",
            "stage_b_selection": "maximum official OCF on test11 tune=shuffled[:16] only; ties retain calibration rank",
            "held_out_rule": "evaluate only the tune-selected font configuration on shuffled[64:], plus the fixed independent training-bank baseline",
        },
        "preregistered_stage_a_gate": {
            "minimum_median_best_cosine": MIN_MEDIAN_COSINE,
            "minimum_matched_ocf_mean": MIN_MATCHED_OCF_MEAN,
            "minimum_ocf_margin_over_random": MIN_OCF_MARGIN_OVER_RANDOM,
            "minimum_ocf_ratio_over_random": MIN_OCF_RATIO_OVER_RANDOM,
            "combination": "all four criteria are required; otherwise save calibration.json and stop without Task 1 font retrieval",
        },
        "configuration_count": len(configurations),
        "configurations": [],
    }
    write_json(CALIBRATION_PATH, result)
    matches = {}
    for index, config in enumerate(configurations, 1):
        started = time.perf_counter()
        print(f"STAGE A [{index}/{len(configurations)}] {config['config_id']}", flush=True)
        bank = render_bank(characters, config)
        metrics, nearest, best = cosine_matches(bank, calibration_images, characters, device)
        row = {**config, **metrics, "elapsed_seconds": time.perf_counter() - started}
        result["configurations"].append(row)
        matches[config["config_id"]] = (nearest, best)
        write_json(CALIBRATION_PATH, result)
        print(
            f"  median={row['median_best_cosine']:.6f} mean={row['mean_best_cosine']:.6f} "
            f"duplicate={row['top1_duplicate_occupancy_rate']:.6f} seconds={row['elapsed_seconds']:.2f}",
            flush=True,
        )
        del bank
        release_arrays()

    ranked = ranked_configurations(result["configurations"])
    best_config = ranked[0]
    nearest, best_cosines = matches[best_config["config_id"]]
    rng_random = np.random.default_rng(RANDOM_GLYPH_SEED)
    random_indices = rng_random.integers(0, len(characters), size=len(ocf_positions))
    matched_indices = nearest[ocf_positions]
    font = load_font(best_config)
    matched = np.stack([render_glyph(characters[index], font) for index in matched_indices])
    random_glyphs = np.stack([render_glyph(characters[index], font) for index in random_indices])
    references = calibration_images[ocf_positions]
    matched_summary, matched_scores = ocf_summary(matched, references, device)
    random_summary, random_scores = ocf_summary(random_glyphs, references, device)
    margin = matched_summary["ocf_mean"] - random_summary["ocf_mean"]
    ratio = matched_summary["ocf_mean"] / max(random_summary["ocf_mean"], 1e-12)
    checks = {
        "median_cosine": best_config["median_best_cosine"] >= MIN_MEDIAN_COSINE,
        "matched_ocf_absolute": matched_summary["ocf_mean"] >= MIN_MATCHED_OCF_MEAN,
        "ocf_margin_over_random": margin >= MIN_OCF_MARGIN_OVER_RANDOM,
        "ocf_ratio_over_random": ratio >= MIN_OCF_RATIO_OVER_RANDOM,
    }
    passed = all(checks.values())
    result["best_configuration"] = best_config
    result["calibration_ranking"] = [row["config_id"] for row in ranked]
    result["official_ocf_check"] = {
        "config_id": best_config["config_id"],
        "matched": matched_summary,
        "random_glyph_control": random_summary,
        "ocf_margin_over_random": margin,
        "ocf_ratio_over_random": ratio,
        "checks": checks,
        "passed": passed,
    }
    result["stage_b_candidate_config_ids"] = [row["config_id"] for row in ranked[:3]]
    result["stage_b_eligible"] = passed
    result["status"] = "stage_a_passed" if passed else "stopped_stage_a_no_strong_match"
    result["conclusion"] = (
        "Stage A passed; only the three preselected calibration configurations may enter Stage B."
        if passed else
        "No strong public-font reconstruction was established under the preregistered gate; stop before full Task 1."
    )
    np.savez_compressed(
        CALIBRATION_NPZ,
        calibration_pictures=calibration_indices,
        candidate_indices=nearest,
        candidate_codepoints=np.asarray([ord(characters[index]) for index in nearest], dtype=np.int32),
        best_cosines=best_cosines,
        ocf_positions=ocf_positions,
        ocf_pictures=calibration_indices[ocf_positions],
        matched_candidate_indices=matched_indices,
        random_candidate_indices=random_indices,
        matched_ocf_scores=matched_scores,
        random_ocf_scores=random_scores,
    )
    write_json(CALIBRATION_PATH, result)
    print(f"STAGE A BEST {best_config['config_id']}: {best_config}", flush=True)
    print(f"STAGE A OCF matched={matched_summary} random={random_summary} gate={passed}", flush=True)
    del matched, random_glyphs, references
    release_arrays()
    return result if passed else None


def build_task1_rows(blocks, split, device):
    rows = []
    for block in blocks:
        block_split = split.blocks[block.key]
        targets, counts = aggregate_by_target(
            block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
        )
        model = load_regressor(MODEL_ROOT / block.mouse / "regressor.pt", device)
        predictions = dct_decode(predict_logits(model, counts, device=device), 32)
        rows.append({"block": block.key, "targets": targets, "predictions": predictions})
        print(f"TASK1 CONTINUOUS {block.key}: {predictions.shape}", flush=True)
    return rows


def evaluate_directional2(rows, images, bank, pictures, device, return_details=False):
    prediction_parts = []
    target_parts = []
    group_parts = []
    picture_parts = []
    candidate_parts = []
    for row in rows:
        mask = np.isin(row["targets"], pictures)
        predictions, indices = retrieve_glyphs(
            row["predictions"][mask], bank, metric="directional2", blur_sigma=1.0, device=device
        )
        prediction_parts.append(predictions)
        target_parts.append(images[row["targets"][mask]])
        group_parts.extend([row["block"]] * int(mask.sum()))
        picture_parts.append(row["targets"][mask])
        candidate_parts.append(indices)
    predictions = np.concatenate(prediction_parts)
    references = np.concatenate(target_parts)
    groups = np.asarray(group_parts)
    scores = score_predictions(predictions, references, device=device, batch_size=16)
    summary = summarize_scores(scores, groups)
    if not return_details:
        return summary, None
    return summary, {
        "predictions": predictions,
        "references": references,
        "groups": groups,
        "pictures": np.concatenate(picture_parts),
        "candidate_indices": np.concatenate(candidate_parts),
    }


def stage_b(stage_a_result, images, blocks, split, characters, device):
    shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
    tune_pictures = np.sort(shuffled[:16])
    evaluation_pictures = np.sort(shuffled[64:])
    rows = build_task1_rows(blocks, split, device)
    training_bank = images[np.flatnonzero(~np.isin(np.arange(len(images)), split.validation_pictures))]
    baseline, _ = evaluate_directional2(rows, images, training_bank, evaluation_pictures, device)
    baseline_error = abs(baseline["ocf_mean"] - BASELINE_OCF)
    print(
        f"BASELINE actual={baseline['ocf_mean']:.10f} expected={BASELINE_OCF:.10f} error={baseline_error:.3g}",
        flush=True,
    )
    if baseline_error > BASELINE_ATOL:
        raise RuntimeError(f"Independent directional2 baseline mismatch: {baseline_error} > {BASELINE_ATOL}")

    by_id = {row["config_id"]: row for row in stage_a_result["configurations"]}
    tuning = []
    for config_id in stage_a_result["stage_b_candidate_config_ids"]:
        config = by_id[config_id]
        started = time.perf_counter()
        bank = render_bank(characters, config).astype(np.float32)
        summary, _ = evaluate_directional2(rows, images, bank, tune_pictures, device)
        row = {
            "config_id": config_id,
            "font_file": config["font_file"],
            "font_size": config["font_size"],
            "variation": config["variation"],
            "calibration_median_best_cosine": config["median_best_cosine"],
            "n": summary["n"],
            "ocf_mean": summary["ocf_mean"],
            "ocf_std": summary["ocf_std"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        tuning.append(row)
        print(f"TUNE {row}", flush=True)
        del bank
        release_arrays()
    selected = max(tuning, key=lambda row: row["ocf_mean"])
    selected_config = by_id[selected["config_id"]]
    selected_bank = render_bank(characters, selected_config).astype(np.float32)
    evaluation, details = evaluate_directional2(
        rows, images, selected_bank, evaluation_pictures, device, return_details=True
    )
    np.savez_compressed(
        SELECTED_NPZ,
        predictions=np.rint(np.clip(details["predictions"], 0, 1) * 255).astype(np.uint8),
        targets=details["references"].astype(np.uint8) * 255,
        group_ids=details["groups"],
        picture_indices=details["pictures"],
        candidate_indices=details["candidate_indices"],
        candidate_codepoints=np.asarray(
            [ord(characters[index]) for index in details["candidate_indices"]], dtype=np.int32
        ),
        candidate_characters=np.asarray([characters[index] for index in details["candidate_indices"]]),
        selected_config_json=np.asarray(json.dumps(selected_config, ensure_ascii=False)),
    )
    result = {
        "experiment": "15_font_calibration",
        "name": "task1_full_gb2312_font_bank_directional2",
        "comparable": False,
        "score_scope": "local validation only; not a Kaggle score",
        "config": {
            "task": "task1",
            "bin_ms": 25,
            "dct_cutoff": 32,
            "checkpoints": str(MODEL_ROOT.relative_to(ROOT)),
            "continuous_predictions": "same independent per-block rows and directional2 retrieval as test11/test14",
            "retrieval_metric": "directional2",
            "retrieval_blur_sigma": 1.0,
            "candidate_bank": "all strict-gb2312 decoded Han characters rendered with the selected public-font configuration",
            "candidate_count": len(characters),
            "training_picture_bank_used_for_selected": False,
            "selection_metric": "tune_ocf_mean",
        },
        "protocol": {
            "split_seed": 42,
            "picture_seed": 123,
            "tune_pictures": tune_pictures.tolist(),
            "tune_picture_count": len(tune_pictures),
            "evaluation_pictures": evaluation_pictures.tolist(),
            "evaluation_picture_count": len(evaluation_pictures),
            "baseline_bank_picture_count": len(training_bank),
            "calibration_json": str(CALIBRATION_PATH.relative_to(ROOT)),
            "held_out_rule": "Only the tune-selected font was evaluated on evaluation pictures; no held-out feedback selected a font or size.",
        },
        "tuning": tuning,
        "selected": selected,
        "evaluation": evaluation,
        "evaluation_baseline": baseline,
        "evaluation_delta_ocf": evaluation["ocf_mean"] - baseline["ocf_mean"],
        "groups": {
            "selected": evaluation["groups"],
            "evaluation_baseline": baseline["groups"],
        },
        "baseline_reproduction": {
            "expected_ocf": BASELINE_OCF,
            "actual_ocf": baseline["ocf_mean"],
            "absolute_error": baseline_error,
            "absolute_tolerance": BASELINE_ATOL,
            "passed": True,
        },
        "selected_npz": str(SELECTED_NPZ.relative_to(ROOT)),
        "license_risk": (
            "No font binaries are redistributed. Exact local font licenses are unverified; Windows system fonts "
            "are not assumed redistributable. Verify font and rendered-bank licensing before external distribution."
        ),
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
    }
    write_json(RESULTS_PATH, result)
    print(
        f"FINAL {RESULTS_PATH}: selected={selected['config_id']} OCF={evaluation['ocf_mean']:.10f} "
        f"baseline={baseline['ocf_mean']:.10f} delta={result['evaluation_delta_ocf']:.10f}",
        flush=True,
    )
    del selected_bank, training_bank
    release_arrays()


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    characters, character_metadata = gb2312_characters()
    images = (load_images(DATA) >= 128).astype(np.float32)
    if len(images) != 2748:
        raise ValueError(f"Expected 2748 training pictures, found {len(images)}.")
    blocks = load_feature_blocks(DATA, "train", bin_ms=25)
    split = make_split(blocks, "task1", seed=42)
    inventory = font_inventory()
    print(f"DEVICE {device}; GB2312 characters={len(characters)}; fonts={inventory}", flush=True)
    stage_a_result = stage_a(images, split, characters, character_metadata, inventory, device)
    if stage_a_result is None:
        print(f"STOP Stage A gate failed. Results: {CALIBRATION_PATH}", flush=True)
        return
    stage_b(stage_a_result, images, blocks, split, characters, device)


if __name__ == "__main__":
    main()
