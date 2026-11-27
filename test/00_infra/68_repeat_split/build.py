"""Split every public training block into one file per presentation ordinal.

For each mouse and session the 2,748 characters are each presented 4 times in shuffled
order. This script numbers the occurrences of a character in time order (1st, 2nd, 3rd,
4th time it appeared) and writes four NPZ files per session, so each file holds exactly
one single-trial recording per character.

Output schema matches the official block files: sample_ids, group_ptr, trial_ptr,
spike_samples, unit_index, target, plus repeat_index and source_trial_index.
Only public training data is read; nothing is copied from the competition test split.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_catalog

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
DEFAULT_OUTPUT = ROOT / "data/derived/repeat_split_v1"
REPEATS = 4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_block(path, destination, mouse, session):
    with np.load(path, allow_pickle=False) as data:
        sample_ids = data["sample_ids"]
        group_ptr = data["group_ptr"]
        trial_ptr = data["trial_ptr"]
        spike_samples = data["spike_samples"]
        unit_index = data["unit_index"]
        targets = data["target"]

    n_trials = len(trial_ptr) - 1
    trial_targets = targets[np.repeat(np.arange(len(group_ptr) - 1), np.diff(group_ptr))]
    if len(trial_targets) != n_trials:
        raise RuntimeError(f"Trial/target mismatch in {path}.")

    occurrence = np.zeros(targets.max() + 1, dtype=np.int64)
    buckets = [[] for _ in range(REPEATS)]
    for trial in range(n_trials):
        target = int(trial_targets[trial])
        ordinal = int(occurrence[target])
        if ordinal >= REPEATS:
            raise RuntimeError(f"{path}: character {target} presented more than {REPEATS} times.")
        buckets[ordinal].append(trial)
        occurrence[target] += 1
    if not np.all(occurrence == REPEATS):
        raise RuntimeError(f"{path}: not every character has exactly {REPEATS} presentations.")

    records = []
    for ordinal, trials in enumerate(buckets, start=1):
        trials = np.asarray(trials, dtype=np.int64)
        counts = np.diff(trial_ptr)[trials]
        starts = trial_ptr[trials]
        trial_ptr_out = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        events = np.concatenate([np.arange(start, start + count) for start, count in zip(starts, counts)])
        output = destination / f"{session}_repeat{ordinal}.npz"
        np.savez(
            output,
            sample_ids=sample_ids[trials],
            group_ptr=np.arange(len(trials) + 1, dtype=np.int64),
            trial_ptr=trial_ptr_out,
            spike_samples=spike_samples[events],
            unit_index=unit_index[events],
            target=trial_targets[trials].astype(np.int64),
            repeat_index=np.full(len(trials), ordinal, dtype=np.int64),
            source_trial_index=trials,
        )
        present = np.unique(trial_targets[trials])
        if len(present) != len(trials) or len(present) != len(occurrence):
            raise RuntimeError(f"{output}: repeat {ordinal} does not contain each character once.")
        records.append({
            "file": str(output.relative_to(ROOT)),
            "mouse": mouse,
            "session": session,
            "repeat_index": ordinal,
            "n_samples": int(len(trials)),
            "n_events": int(len(events)),
            "sha256": sha256(output),
        })
        print(f"wrote {output.name}: samples={len(trials)} events={len(events)}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"{manifest_path} exists; pass --overwrite.")
    catalog = load_catalog(DATA)
    records = []
    for entry in sorted((e for e in catalog["blocks"] if e["split"] == "train"),
                        key=lambda e: (e["mouse"], e["session"])):
        destination = args.output / entry["mouse"]
        destination.mkdir(parents=True, exist_ok=True)
        records.extend(split_block(DATA / entry["path"], destination, entry["mouse"], entry["session"]))
    args.output.joinpath("manifest.json").write_text(json.dumps({
        "schema": "repeat_split_v1",
        "rule": ("occurrences of each character are numbered in presentation (time) order; "
                 "repeat k contains the k-th presentation of every character"),
        "source_root": str(DATA),
        "files": records,
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} files to {args.output}", flush=True)


if __name__ == "__main__":
    main()
