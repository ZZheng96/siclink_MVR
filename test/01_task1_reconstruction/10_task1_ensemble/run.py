"""P2: blend bin25/bin50 DCT-MLP outputs before directional glyph retrieval."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.neural import load_regressor, predict_logits
from src.retrieval import retrieve_glyphs
from src.ridge import dct_decode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks25 = {block.key: block for block in load_feature_blocks(DATA, "train", bin_ms=25)}
    blocks50 = {block.key: block for block in load_feature_blocks(DATA, "train", bin_ms=50)}
    blocks = [blocks25[key] for key in sorted(blocks25)]
    split = make_split(blocks, "task1", seed=42)
    shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
    tune_pictures = np.sort(shuffled[:16])
    evaluation_pictures = np.sort(shuffled[64:])
    bank = images[np.flatnonzero(~np.isin(np.arange(len(images)), split.validation_pictures))]
    rows = []
    for block in blocks:
        block_split = split.blocks[block.key]
        targets, counts25 = aggregate_by_target(
            block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
        )
        block50 = blocks50[block.key]
        targets50, counts50 = aggregate_by_target(
            block50.features[block_split.validation_trials], block50.trial_targets[block_split.validation_trials]
        )
        assert np.array_equal(targets, targets50)
        model25 = load_regressor(
            ROOT / "test/01_task1_reconstruction/05_task1_pooled/artifacts/task1_pooled_mlp_bin25_singles" / block.mouse / "regressor.pt", device
        )
        model50 = load_regressor(
            ROOT / "test/01_task1_reconstruction/05_task1_pooled/artifacts/task1_pooled_mlp_dct32" / block.mouse / "regressor.pt", device
        )
        prediction25 = dct_decode(predict_logits(model25, counts25, device=device), 32)
        prediction50 = dct_decode(predict_logits(model50, counts50, device=device), 32)
        rows.append({"block": block.key, "targets": targets, "p25": prediction25, "p50": prediction50})

    def evaluate(weight, pictures):
        predictions = []
        targets = []
        groups = []
        for row in rows:
            mask = np.isin(row["targets"], pictures)
            continuous = weight * row["p25"][mask] + (1 - weight) * row["p50"][mask]
            retrieved, _ = retrieve_glyphs(continuous, bank, metric="directional", device=device)
            predictions.append(retrieved)
            targets.append(images[row["targets"][mask]])
            groups.extend([row["block"]] * int(mask.sum()))
        predictions = np.concatenate(predictions)
        references = np.concatenate(targets)
        groups = np.asarray(groups)
        scores = score_predictions(predictions, references, device=device, batch_size=16)
        return summarize_scores(scores, groups), predictions, references, groups

    tuning = []
    for weight in WEIGHTS:
        summary, _, _, _ = evaluate(weight, tune_pictures)
        tuning.append({"bin25_weight": weight, "ocf_mean": summary["ocf_mean"]})
    best = max(tuning, key=lambda row: row["ocf_mean"])
    weight = best["bin25_weight"]
    summary, predictions, references, groups = evaluate(weight, evaluation_pictures)
    output_dir = ROOT / "test/01_task1_reconstruction/10_task1_ensemble"
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / "task1_predictions.npz",
        predictions=predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    result = {
        "experiment": "10_task1_ensemble",
        "name": "task1_bin25_bin50_ensemble",
        "config": {"task": "task1", "weights": WEIGHTS},
        "protocol": {"tune_pictures": tune_pictures.tolist(), "evaluation_picture_count": len(evaluation_pictures)},
        "tuning": tuning,
        "selected": best,
        "evaluation": summary,
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
    }
    path = output_dir / "results_task1_ensemble.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"FINAL {path}: OCF={summary['ocf_mean']:.6f}; bin25_weight={weight}", flush=True)


if __name__ == "__main__":
    main()
