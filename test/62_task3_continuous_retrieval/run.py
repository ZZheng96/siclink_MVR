"""Exploratory: Task3 continuous reconstruction + directional retrieval vs classification.

Task3 has never been evaluated as continuous reconstruction; prior work used 2748-way
classification.  This tests whether structure retrieval can beat the 15.86 single-view
classification baseline under the frozen development protocol.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\WORK\siclink")
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions
from src.features import aggregate_by_target
from src.preprocessing import load_cached_blocks
from src.retrieval import retrieve_glyphs
from src.ridge import RidgePath, dct_decode, dct_encode
from src.splits import make_split

CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
OUT = ROOT / "test/62_task3_continuous_retrieval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIN_MS = 50
CUTOFF = 32
ALPHA = 1.0
FOLDS = ("20261126", "20261127")
MICE = ("mice1", "mice2")
SUBSET_SEED = 20261128
SUBSET_SIZE = 256


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pictures = np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False)
    gt = (np.asarray(pictures) >= 128).astype(np.float32)
    subset = np.sort(np.random.default_rng(SUBSET_SEED).permutation(len(gt))[:SUBSET_SIZE])
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=BIN_MS)

    output = {"scope": "exploratory; Task3 continuous DCT regression + directional retrieval; "
                        "fixed alpha=1.0, cutoff=32; compared to 15.86 classification",
              "bin_ms": BIN_MS, "folds": list(FOLDS), "subset_size": SUBSET_SIZE, "results": {}}
    for fold in FOLDS:
        split = make_split(blocks, "task3", seed=int(fold))
        for mouse in MICE:
            train_x, train_y, rows, truths = [], [], [], []
            for block in [b for b in blocks if b.mouse == mouse]:
                part = split.blocks[block.key]
                targets, counts = aggregate_by_target(
                    np.asarray(block.features[part.train_trials], dtype=np.float32),
                    block.trial_targets[part.train_trials],
                )
                train_x.append(counts)
                train_y.append(dct_encode(gt[targets], CUTOFF))
            train_x = np.concatenate(train_x)
            train_y = np.concatenate(train_y)
            path = RidgePath().fit(np.log1p(train_x), train_y)
            for block in [b for b in blocks if b.mouse == mouse]:
                part = split.blocks[block.key]
                trials = part.validation_trials
                labels = block.trial_targets[trials]
                counts = np.asarray(block.features[trials], dtype=np.float32)
                predictions = path.predict(np.log1p(counts), ALPHA).astype(np.float32)
                order = np.argsort(labels, kind="stable")
                keep = np.isin(labels[order], subset)
                rows.append(predictions[order][keep])
                truths.append(gt[labels[order][keep]])
            coefficients = np.concatenate(rows)
            truth = np.concatenate(truths)
            continuous = np.clip(np.asarray(dct_decode(coefficients, CUTOFF), dtype=np.float32), 0.0, 1.0)
            retrieved, _ = retrieve_glyphs(continuous, gt, metric="directional", blur_sigma=0.0, device=DEVICE)
            scores = score_predictions(retrieved, truth, device=str(DEVICE), batch_size=16)
            gt_coeff = dct_encode(truth, CUTOFF)
            mse = float(((coefficients - gt_coeff) ** 2).mean())
            variance = float(gt_coeff.var(axis=0).mean())
            key = f"{fold}/{mouse}"
            output["results"][key] = {
                "n": int(len(scores)),
                "rows": int(len(coefficients)),
                "dct_r2_approx": float(1.0 - mse / variance),
                "retrieval_ocf": summarize(scores),
                "predicted_image_std": float(continuous.std()),
            }
            print(key, "r2", round(output["results"][key]["dct_r2_approx"], 4),
                  "retrieval", round(float(scores.mean()), 3), flush=True)

    all_means = [output["results"][k]["retrieval_ocf"]["mean"] for k in output["results"]]
    output["macro_retrieval_ocf"] = float(np.mean(all_means))
    output["classification_baseline_single_view"] = 15.8648
    (OUT / "exploratory_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("macro", output["macro_retrieval_ocf"], flush=True)


if __name__ == "__main__":
    main()
