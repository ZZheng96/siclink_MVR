"""P2: select directional retrieval resolution on a separate tuning character set."""

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
METRICS = ("cosine", "mse", "directional2", "directional", "directional8", "directional_multi")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=25)
    split = make_split(blocks, "task1", seed=42)
    shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
    tune_pictures = np.sort(shuffled[:16])
    evaluation_pictures = np.sort(shuffled[64:])
    bank = images[np.flatnonzero(~np.isin(np.arange(len(images)), split.validation_pictures))]
    rows = []
    for block in blocks:
        block_split = split.blocks[block.key]
        targets, counts = aggregate_by_target(
            block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
        )
        model = load_regressor(
            ROOT / "test/01_task1_reconstruction/05_task1_pooled/artifacts/task1_pooled_mlp_bin25_singles" / block.mouse / "regressor.pt", device
        )
        predictions = dct_decode(predict_logits(model, counts, device=device), 32)
        rows.append({"block": block.key, "targets": targets, "predictions": predictions})

    def evaluate(metric, pictures):
        prediction_parts = []
        target_parts = []
        group_parts = []
        for row in rows:
            mask = np.isin(row["targets"], pictures)
            predictions, _ = retrieve_glyphs(
                row["predictions"][mask], bank, metric=metric, blur_sigma=1.0, device=device
            )
            prediction_parts.append(predictions)
            target_parts.append(images[row["targets"][mask]])
            group_parts.extend([row["block"]] * int(mask.sum()))
        predictions = np.concatenate(prediction_parts)
        references = np.concatenate(target_parts)
        groups = np.asarray(group_parts)
        scores = score_predictions(predictions, references, device=device, batch_size=16)
        return summarize_scores(scores, groups), predictions, references, groups

    tuning = []
    for metric in METRICS:
        summary, _, _, _ = evaluate(metric, tune_pictures)
        tuning.append({"metric": metric, "ocf_mean": summary["ocf_mean"]})
        print(tuning[-1], flush=True)
    best = max(tuning, key=lambda row: row["ocf_mean"])
    summary, predictions, references, groups = evaluate(best["metric"], evaluation_pictures)
    output_dir = ROOT / "test/01_task1_reconstruction/11_retrieval_ablation"
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / "task1_predictions.npz",
        predictions=predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    result = {
        "experiment": "11_retrieval_ablation",
        "name": "task1_directional_pooling",
        "config": {"task": "task1", "metrics": METRICS},
        "protocol": {"tune_pictures": tune_pictures.tolist(), "evaluation_picture_count": len(evaluation_pictures)},
        "tuning": tuning,
        "selected": best,
        "evaluation": summary,
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
    }
    path = output_dir / "results_task1_retrieval.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"FINAL {path}: OCF={summary['ocf_mean']:.6f}; selected={best}", flush=True)


if __name__ == "__main__":
    main()
