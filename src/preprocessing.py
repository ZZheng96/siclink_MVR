"""Auditable, reusable conversion of event NPZ files into trial-level arrays."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from src.dataloader import SpikeBlock, load_catalog, load_images
from src.features import FeatureBlock, trial_count_features
from src.splits import BlockSplit, ProtocolSplit

CACHE_SCHEMA = 1
SPLIT_SCHEMA = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_array(path, array):
    temporary = path.with_suffix(path.suffix + ".tmp")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=array.dtype, shape=array.shape
    )
    output[:] = array
    output.flush()
    del output
    os.replace(temporary, path)


def _write_index(path, arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _source_records(root):
    root = Path(root)
    catalog = load_catalog(root)
    paths = [root / "catalog.json", root / "train/pictures.npz"]
    paths.extend(root / entry["path"] for entry in catalog["blocks"])
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]


def _validate_block(block, n_images):
    if block.group_ptr.ndim != 1 or block.trial_ptr.ndim != 1:
        raise ValueError("group_ptr and trial_ptr must be one-dimensional.")
    if len(block.group_ptr) != len(block) + 1 or block.group_ptr[0] != 0:
        raise ValueError("Invalid sample grouping.")
    if block.trial_ptr[0] != 0 or block.trial_ptr[-1] != len(block.spike_samples):
        raise ValueError("Invalid trial event boundaries.")
    if np.any(np.diff(block.group_ptr) <= 0) or np.any(np.diff(block.trial_ptr) < 0):
        raise ValueError("Trial pointers must be ordered and samples non-empty.")
    if block.group_ptr[-1] != len(block.trial_ptr) - 1:
        raise ValueError("Sample and trial pointers disagree.")
    if len(block.unit_index) != len(block.spike_samples):
        raise ValueError("Spike times and unit indices are misaligned.")
    if len(block.unit_index):
        if block.unit_index.min() < 0 or block.unit_index.max() >= block.n_units:
            raise ValueError("Unit index is outside the block.")
    if block.targets is not None:
        if block.targets.shape != (len(block),):
            raise ValueError("Expected one target per training sample row.")
        if block.targets.min() < 0 or block.targets.max() >= n_images:
            raise ValueError("Training target is outside pictures.npz.")


def prepare_cache(root, cache_root, *, bin_ms=(10, 25, 50)):
    """Extract every catalog block once and store exact integer counts as memmaps."""
    root = Path(root).resolve()
    cache_root = Path(cache_root).resolve()
    bins = tuple(sorted(set(int(value) for value in bin_ms)))
    if not bins or any(value <= 0 for value in bins):
        raise ValueError("bin_ms must contain positive integers.")
    sources = _source_records(root)
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") == CACHE_SCHEMA
            and manifest.get("source_root") == str(root)
            and manifest.get("sources") == sources
            and manifest.get("bin_ms") == list(bins)
        ):
            return manifest
        raise ValueError(f"Existing cache is stale or incompatible: {cache_root}")
    if cache_root.exists():
        raise ValueError(f"Cache directory exists without a valid manifest: {cache_root}")

    staging = cache_root.with_name(cache_root.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        catalog = load_catalog(root)
        images = load_images(root)
        _write_array(staging / "pictures.npy", images)
        records = []
        for entry in catalog["blocks"]:
            block = SpikeBlock(
                root / entry["path"], entry, catalog["sample_rate_hz"], catalog["window_ms"]
            )
            _validate_block(block, len(images))
            name = "__".join(
                (entry["split"], entry["task"], entry["mouse"], entry["session"])
            )
            destination = staging / "blocks" / name
            destination.mkdir(parents=True)
            arrays = {
                "sample_ids": block.sample_ids,
                "group_ptr": block.group_ptr.astype(np.int64),
                "trial_ptr": block.trial_ptr.astype(np.int64),
                "trial_sample_indices": np.repeat(
                    np.arange(len(block), dtype=np.int64), np.diff(block.group_ptr)
                ),
            }
            if block.targets is not None:
                arrays["targets"] = block.targets.astype(np.int64)
            _write_index(destination / "index.npz", arrays)
            count_records = {}
            for width in bins:
                counts = trial_count_features(block, width)
                if counts.max(initial=0) > np.iinfo(np.uint16).max:
                    raise ValueError("Count does not fit the uint16 cache representation.")
                filename = f"counts_bin{width}.npy"
                _write_array(destination / filename, counts.astype(np.uint16))
                count_records[str(width)] = {
                    "file": filename,
                    "shape": list(counts.shape),
                    "dtype": "uint16",
                    "sha256": file_sha256(destination / filename),
                }
            records.append({
                **{key: entry[key] for key in (
                    "split", "task", "mouse", "session", "path", "n_units", "n_samples"
                )},
                "id": name,
                "n_trials": len(block.trial_ptr) - 1,
                "counts": count_records,
            })
            print(
                f"cached {entry['split']}/{entry['task']}/{block.mouse}/{block.session}: "
                f"{len(block.trial_ptr) - 1} trials, bins={bins}",
                flush=True,
            )
        manifest = {
            "schema": CACHE_SCHEMA,
            "source_root": str(root),
            "sample_rate_hz": catalog["sample_rate_hz"],
            "window_ms": catalog["window_ms"],
            "image_shape": list(images.shape),
            "bin_ms": list(bins),
            "sources": sources,
            "blocks": records,
            "data_policy": (
                "Only deterministic raw counts and row mappings are shared. Fold-specific "
                "normalization, prototypes, PCA and model parameters must be fitted on train_trials only."
            ),
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, cache_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def load_cached_blocks(cache_root, *, split="train", task=None, bin_ms=25, mouse=None):
    """Load memmapped FeatureBlocks without recomputing spike counts."""
    cache_root = Path(cache_root)
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != CACHE_SCHEMA:
        raise ValueError("Unsupported preprocessing cache schema.")
    if str(bin_ms) not in {str(value) for value in manifest["bin_ms"]}:
        raise ValueError(f"bin_ms={bin_ms} is not present in the cache.")
    output = []
    for record in manifest["blocks"]:
        if record["split"] != split:
            continue
        if task is not None and record["task"] != task:
            continue
        if mouse is not None and record["mouse"] != mouse:
            continue
        directory = cache_root / "blocks" / record["id"]
        with np.load(directory / "index.npz", allow_pickle=False) as index:
            arrays = {key: index[key] for key in index.files}
        count_record = record["counts"][str(bin_ms)]
        features = np.load(directory / count_record["file"], mmap_mode="r", allow_pickle=False)
        if list(features.shape) != count_record["shape"]:
            raise ValueError(f"Cached shape mismatch for {record['id']}.")
        output.append(FeatureBlock(
            mouse=record["mouse"],
            session=record["session"],
            split=split,
            task=record["task"],
            sample_ids=arrays["sample_ids"],
            group_ptr=arrays["group_ptr"],
            targets=arrays.get("targets"),
            features=features,
            n_units=record["n_units"],
            bin_ms=bin_ms,
        ))
    if not output:
        raise ValueError("No cached blocks match the request.")
    return sorted(output, key=lambda block: (block.mouse, block.session))


def validate_split(split, blocks):
    """Check disjoint indices and task-specific target/session/repeat isolation."""
    by_key = {block.key: block for block in blocks}
    if set(by_key) != set(split.blocks):
        raise ValueError("Split blocks do not match cached blocks.")
    for key, part in split.blocks.items():
        block = by_key[key]
        if np.intersect1d(part.train_trials, part.validation_trials).size:
            raise ValueError(f"Train/validation trial leakage in {key}.")
        if len(part.train_trials) and (
            part.train_trials.min() < 0 or part.train_trials.max() >= block.n_trials
        ):
            raise ValueError(f"Invalid train trial index in {key}.")
        if len(part.validation_trials) and (
            part.validation_trials.min() < 0 or part.validation_trials.max() >= block.n_trials
        ):
            raise ValueError(f"Invalid validation trial index in {key}.")
        combined = np.concatenate((part.train_trials, part.validation_trials))
        if len(combined) != block.n_trials or len(np.unique(combined)) != block.n_trials:
            raise ValueError(f"Split does not partition every trial exactly once in {key}.")
        train_targets = block.trial_targets[part.train_trials]
        validation_targets = block.trial_targets[part.validation_trials]
        if split.task == "task1":
            if np.intersect1d(train_targets, validation_targets).size:
                raise ValueError(f"Character leakage in {key}.")
            if set(validation_targets) - set(split.validation_pictures.tolist()):
                raise ValueError(f"Unexpected validation character in {key}.")
        elif split.task == "task2":
            is_validation = block.session == split.validation_session
            if is_validation != (len(part.train_trials) == 0):
                raise ValueError(f"Session leakage in {key}.")
            if is_validation != (len(part.validation_trials) == block.n_trials):
                raise ValueError(f"Incomplete held-out session in {key}.")
        elif split.task == "task3":
            validation_counts = np.unique(validation_targets, return_counts=True)[1]
            if len(validation_targets) != 2748 or np.any(validation_counts != 1):
                raise ValueError(f"Task3 must hold out every glyph once in {key}.")
            train_counts = np.unique(train_targets, return_counts=True)[1]
            if len(train_counts) != 2748 or np.any(train_counts != 3):
                raise ValueError(f"Task3 must retain three train repeats per glyph in {key}.")
        else:
            raise ValueError(f"Unknown task: {split.task}")


def save_split_manifest(path, split, blocks):
    validate_split(split, blocks)
    value = {
        "schema": SPLIT_SCHEMA,
        **split.metadata(),
        "blocks": {
            key: {
                "train_samples": part.train_samples.tolist(),
                "validation_samples": part.validation_samples.tolist(),
                "train_trials": part.train_trials.tolist(),
                "validation_trials": part.validation_trials.tolist(),
            }
            for key, part in split.blocks.items()
        },
    }
    _write_json(Path(path), value)
    return value


def load_split_manifest(path, blocks):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value["schema"] != SPLIT_SCHEMA:
        raise ValueError("Unsupported split manifest schema.")
    split = ProtocolSplit(
        task=value["task"],
        seed=value["seed"],
        validation_session=value["validation_session"],
        validation_pictures=np.asarray(value["validation_pictures"], dtype=np.int64),
        blocks={
            key: BlockSplit(
                train_samples=np.asarray(part["train_samples"], dtype=np.int64),
                validation_samples=np.asarray(part["validation_samples"], dtype=np.int64),
                train_trials=np.asarray(part["train_trials"], dtype=np.int64),
                validation_trials=np.asarray(part["validation_trials"], dtype=np.int64),
            )
            for key, part in value["blocks"].items()
        },
    )
    validate_split(split, blocks)
    return split
