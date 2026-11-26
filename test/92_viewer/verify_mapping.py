"""Verify that each heldout prediction row maps to the exact public train trials shown in the viewer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer_data as v  # noqa: E402
from src.splits import make_split  # noqa: E402


def main():
    failures = 0
    split_cache = {t: make_split(v.stub_blocks(), t, seed=v.SPLIT_SEED) for t in v.TASKS}
    for task in v.TASKS:
        h = v.heldout(task)
        rows, split = h["rows"], split_cache[task]
        problems = []
        if not h["sequence_ok"]:
            problems.append("row order (group_ids, target_indices) does not match the replayed split")
        if abs(h["mean"] - h["recorded_mean"]) > 1e-6:
            problems.append(f"mean OCF {h['mean']} != recorded {h['recorded_mean']}")
        if h["oracle"] is not None and abs(h["oracle_mean"] - h["recorded_oracle_mean"]) > 1e-6:
            problems.append(f"oracle mean {h['oracle_mean']} != recorded {h['recorded_oracle_mean']}")
        seen = set()
        bank = v.pictures()
        for r in rows:
            if r["trials"] is None:
                problems.append(f"row {r['row']}: unresolved")
                continue
            d = v.load_block("train", "train", r["mouse"], r["session"])
            key = f"{r['mouse']}/{r['session']}"
            sb = split.blocks[key]
            trials = np.asarray(r["trials"])
            sample_of = np.searchsorted(d["group_ptr"], trials, side="right") - 1
            if not np.all(d["target"][sample_of] == r["target"]):
                problems.append(f"row {r['row']}: trial target != GT target")
            if not np.array_equal(h["gt"][r["row"]], bank[r["target"]]):
                problems.append(f"row {r['row']}: GT image != pictures[target]")
            if not np.isin(trials, sb.validation_trials).all() or np.isin(trials, sb.train_trials).any():
                problems.append(f"row {r['row']}: trial not in validation split or leaked into training split")
            expected = {"task1": 4, "task2": 4, "task3": 1}[task]
            if len(trials) != expected:
                problems.append(f"row {r['row']}: {len(trials)} trials, expected {expected}")
            if task == "task2":
                all_s4 = np.flatnonzero(d["target"] == r["target"])
                if r["session"] != "session4" or not np.array_equal(np.sort(trials), all_s4):
                    problems.append(f"row {r['row']}: Task2 trials are not all session4 repeats of the target")
            if task == "task1":
                # all validation trials of this target in the block form exactly this group
                vt = sb.validation_trials
                same = vt[d["target"][np.searchsorted(d["group_ptr"], vt, side="right") - 1] == r["target"]]
                if not np.array_equal(np.sort(trials), np.sort(same)):
                    problems.append(f"row {r['row']}: Task1 group differs from all validation trials of the target")
            for t in trials:
                if (key, int(t)) in seen:
                    problems.append(f"row {r['row']}: trial {t} used by two rows")
                seen.add((key, int(t)))
        failures += len(problems)
        status = "OK" if not problems else f"FAIL ({len(problems)})"
        print(f"{task}: rows={len(rows)} resolved={sum(r['trials'] is not None for r in rows)} "
              f"trials/row={sorted({len(r['trials'] or []) for r in rows})} distinct_trials={len(seen)} "
              f"meanOCF={h['mean']:.4f} (recorded {h['recorded_mean']:.4f}) -> {status}")
        for p in problems[:10]:
            print("   ", p)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
