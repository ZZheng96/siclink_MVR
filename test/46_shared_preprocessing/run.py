"""Build and verify the shared NPZ-derived data cache and frozen split manifests."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_blocks
from src.features import trial_count_features
from src.preprocessing import (
    file_sha256,
    load_cached_blocks,
    load_split_manifest,
    prepare_cache,
    save_split_manifest,
)
from src.splits import make_split

OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
BINS = (10, 25, 50)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_real_cache():
    manifest = prepare_cache(DATA, CACHE, bin_ms=BINS)
    checks = {}
    total_trials = 0
    for width in BINS:
        cached = load_cached_blocks(CACHE, split="train", bin_ms=width)
        original = load_blocks(DATA, "train")
        original = sorted(original, key=lambda block: (block.mouse, block.session))
        assert len(cached) == len(original) == 8
        for cached_block, original_block in zip(cached, original):
            assert cached_block.key == f"{original_block.mouse}/{original_block.session}"
            expected = trial_count_features(original_block, width)
            np.testing.assert_array_equal(cached_block.features, expected)
            np.testing.assert_array_equal(cached_block.targets, original_block.targets)
            np.testing.assert_array_equal(cached_block.sample_ids, original_block.sample_ids)
            np.testing.assert_array_equal(cached_block.group_ptr, original_block.group_ptr)
            np.testing.assert_array_equal(cached_block.trial_sample_indices, np.arange(len(expected)))
            labels, counts = np.unique(cached_block.trial_targets, return_counts=True)
            np.testing.assert_array_equal(labels, np.arange(2748))
            np.testing.assert_array_equal(counts, np.full(2748, 4))
            total_trials += cached_block.n_trials if width == BINS[0] else 0
        checks[f"bin_{width}_exact_train_equality"] = True
    assert total_trials == 87936
    checks["train_trials_total"] = total_trials

    pictures = np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False)
    assert pictures.shape == (2748, 64, 64)
    checks["pictures_shape"] = list(pictures.shape)
    checks["source_files_hashed"] = len(manifest["sources"])

    split_records = {}
    for task in ("task1", "task2", "task3"):
        blocks = load_cached_blocks(CACHE, split="train", bin_ms=25)
        split = make_split(blocks, task, seed=42)
        path = CACHE / "splits" / f"{task}_seed42.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_split_manifest(path, split, blocks)
        loaded = load_split_manifest(path, blocks)
        fresh = make_split(blocks, task, seed=42)
        assert loaded.task == fresh.task and loaded.seed == fresh.seed
        np.testing.assert_array_equal(loaded.validation_pictures, fresh.validation_pictures)
        for key in fresh.blocks:
            for field in ("train_samples", "validation_samples", "train_trials", "validation_trials"):
                np.testing.assert_array_equal(
                    getattr(loaded.blocks[key], field), getattr(fresh.blocks[key], field)
                )
        split_records[task] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_sha256(path),
            "blocks": len(loaded.blocks),
            "round_trip_exact": True,
        }
    checks["splits"] = split_records
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Build and verify real public data.")
    args = parser.parse_args()
    if not args.run:
        parser.error("Use --run; this experiment intentionally has no metric-only mode.")
    preregistration = json.loads((OUT / "preregistration.json").read_text(encoding="utf-8"))
    assert preregistration["status"] == "preregistered_before_real_cache_or_real_metrics"
    assert preregistration["cache"]["bin_ms"] == list(BINS)
    checks = verify_real_cache()
    result = {
        "experiment": OUT.name,
        "status": "passed",
        "cache": str(CACHE.relative_to(ROOT)),
        "manifest_sha256": file_sha256(CACHE / "manifest.json"),
        "checks": checks,
        "production_changed": False,
        "ocf_computed": False,
    }
    write_json(OUT / "results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
