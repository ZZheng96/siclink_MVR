"""Nested leave-one-session-out validation of the frozen Task2 K=8 reranker."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "test/31_task2_rerank/run.py"
spec = importlib.util.spec_from_file_location("exp31_nested", SOURCE)
exp31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp31)
e = exp31.e

DAYS = ("session1", "session2", "session3")
FROZEN_CONFIG = "cnn_k8_global_assignment"
BASELINE = "original_global_assignment"
METRICS = ("ocf", "overall_top1", "original_top32_recall", "original_top32_conditional_top1")


def measure_rows(rows, images):
    """Score each complete prediction block once; all picture metrics are slices of these arrays."""
    measured = []
    for row in rows:
        part = {"mouse": row["mouse"], "targets": row["targets"], "inside32": row["inside32"]}
        predictions = {
            BASELINE: e.unique_assignment(row["base"]),
            FROZEN_CONFIG: e.unique_assignment(exp31.rerank_scores(row, 8)),
        }
        for name, predicted in predictions.items():
            part[name] = {
                "predicted": predicted,
                "ocf": e.score_predictions(images[predicted], images[row["targets"]],
                                           device=exp31.DEVICE, batch_size=16),
            }
        measured.append(part)
    return measured


def summarize(measured, name, pictures):
    selected = []
    for part in measured:
        mask = np.isin(part["targets"], pictures)
        selected.append({key: value[mask] if isinstance(value, np.ndarray) else value
                         for key, value in part.items()})
        for arm in (BASELINE, FROZEN_CONFIG):
            selected[-1][arm] = {key: value[mask] for key, value in part[arm].items()}
    n = sum(len(part["targets"]) for part in selected)
    condition_n = sum(int(part["inside32"].sum()) for part in selected)
    conditional_hits = sum(int(np.sum((part[name]["predicted"] == part["targets"]) & part["inside32"]))
                           for part in selected)
    return {
        "n": n,
        "ocf": float(np.concatenate([part[name]["ocf"] for part in selected]).mean()),
        "overall_top1": sum(int(np.sum(part[name]["predicted"] == part["targets"]))
                            for part in selected) / n,
        "original_top32_recall": condition_n / n,
        "original_top32_conditional_top1": conditional_hits / condition_n if condition_n else None,
    }


def main():
    started = time.perf_counter()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    images = (e.load_images(e.DATA) >= 128).astype(np.float32)
    coefficients = e.dct_encode(images, 16)
    blocks = e.load_feature_blocks(e.DATA, "train", bin_ms=50)
    order = np.random.default_rng(123).permutation(len(images))
    subsets = {"all": np.arange(len(images)), "tune": np.sort(order[:256]),
               "heldout": np.sort(order[256:768])}
    rows = {}
    for mouse in ("mice1", "mice2"):
        group = [block for block in blocks if block.mouse == mouse]
        for day in DAYS:
            train_days = tuple(other for other in DAYS if other != day)
            row, _ = exp31.make_score_row(group, day, train_days, images, coefficients)
            row.update(mouse=mouse, block=f"{mouse}/{day}")
            rows[(mouse, day)] = row

    folds = []
    original_cache_path = exp31.cache_path
    for outer_day in DAYS:
        # Inner OOF models must also exclude the outer target day's labels.
        # Thus each inner score row has only the one remaining session as training data.
        exp31.cache_path = lambda mouse, day, outer=outer_day: (
            HERE / "artifacts/scores" / outer / f"{mouse}_{day}.npz"
        )
        training_rows = []
        inner_training_records = []
        for mouse in ("mice1", "mice2"):
            group = [block for block in blocks if block.mouse == mouse]
            for inner_day in DAYS:
                if inner_day == outer_day:
                    continue
                inner_train_days = tuple(day for day in DAYS if day not in (outer_day, inner_day))
                row, record = exp31.make_score_row(group, inner_day, inner_train_days, images, coefficients)
                row.update(mouse=mouse, block=f"{mouse}/{inner_day}")
                training_rows.append(row)
                inner_training_records.append(record)
        exp31.cache_path = original_cache_path
        target_rows = [rows[(mouse, outer_day)] for mouse in ("mice1", "mice2")]
        model, history, counts, _ = exp31.fit_reranker(training_rows, images)
        exp31.predict_utilities(model, target_rows, images)
        measured = measure_rows(target_rows, images)
        fold = {
            "outer_session": outer_day,
            "reranker_training_sessions": [day for day in DAYS if day != outer_day],
            "reranker_training_examples": int(sum(counts.values())),
            "reranker_training_examples_by_mouse_day": counts,
            "reranker_epoch_loss": history,
            "inner_oof_base_training": inner_training_records,
            "base_model_training": (
                "Target-day OOF base models exclude the outer target session. Every inner OOF base model "
                "used for reranker fitting excludes both its inner score day and the outer target day."
            ),
            "metrics": {},
        }
        for subset_name, pictures in subsets.items():
            fold["metrics"][subset_name] = {}
            for name in (BASELINE, FROZEN_CONFIG):
                fold["metrics"][subset_name][name] = {
                    "pooled": summarize(measured, name, pictures),
                    "by_mouse": {
                        mouse: summarize([part for part in measured if part["mouse"] == mouse], name, pictures)
                        for mouse in ("mice1", "mice2")
                    },
                }
            fold["metrics"][subset_name]["delta_reranker_minus_baseline"] = {
                metric: fold["metrics"][subset_name][FROZEN_CONFIG]["pooled"][metric]
                        - fold["metrics"][subset_name][BASELINE]["pooled"][metric]
                for metric in METRICS
            }
        folds.append(fold)
        checkpoint = HERE / f"artifacts/{outer_day}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        del model, measured, target_rows
        e.release_memory()
        print(f"OUTER {outer_day}: " + json.dumps(fold["metrics"]["heldout"]["delta_reranker_minus_baseline"]),
              flush=True)

    macro = {}
    for subset_name in subsets:
        macro[subset_name] = {}
        for name in (BASELINE, FROZEN_CONFIG):
            macro[subset_name][name] = {
                metric: float(np.mean([fold["metrics"][subset_name][name]["pooled"][metric]
                                       for fold in folds]))
                for metric in METRICS
            }
        macro[subset_name]["delta_reranker_minus_baseline"] = {
            metric: macro[subset_name][FROZEN_CONFIG][metric] - macro[subset_name][BASELINE][metric]
            for metric in METRICS
        }
    result = {
        "experiment": "34_task2_nested_rerank",
        "protocol": {
            "outer_folds": list(DAYS),
            "target_day_excluded_from_base_training": True,
            "target_day_excluded_from_reranker_training": True,
            "target_day_excluded_from_all_inner_oof_base_training": True,
            "inner_oof_training_days_per_score_row": 1,
            "fixed_selection": FROZEN_CONFIG,
            "selection_source": "test/31_task2_rerank/artifacts/selection.json (session4 tune only)",
            "no_new_model_or_K_selection": True,
            "subsets": {name: pictures.tolist() for name, pictures in subsets.items()},
            "assignment_scope": "full 2748-row block before picture subsetting",
            "note": "The historical tune/heldout IDs are reused for comparability, not used for selection here.",
        },
        "folds": folds,
        "macro_mean_of_three_outer_days": macro,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
                        "device": exp31.DEVICE},
    }
    path = HERE / "results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("MACRO", json.dumps(macro, indent=2), flush=True)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
