"""P2: structural medoid postprocessing of ensemble scores and global assignments."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.final import apply_session_normalizer, fit_session_normalizer, row_zscore
from src.neural import load_classifier, predict_logits
from src.retrieval import glyph_similarity, unique_assignment
from src.ridge import RidgePath, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def structural_medoid(scores, assigned, similarity, weight, k=20):
    output = np.empty(len(scores), dtype=np.int64)
    for index, row in enumerate(scores):
        candidates = np.argpartition(row, -k)[-k:]
        if assigned[index] not in candidates:
            candidates = np.append(candidates, assigned[index])
        values = row[candidates]
        probabilities = np.exp(values - values.max())
        probabilities /= probabilities.sum()
        local_similarity = similarity[np.ix_(candidates, candidates)]
        expected = local_similarity @ probabilities
        normalized = (values - values.mean()) / max(values.std(), 1e-6)
        output[index] = candidates[np.argmax(expected + weight * normalized)]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task2", "task3"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    bank_coefficients = dct_encode(images, 16)
    similarity = glyph_similarity(images, metric="directional2", device=device)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    split = make_split(blocks, args.task, seed=42)
    selected = np.random.default_rng(123).permutation(len(images))
    tune_pictures = np.sort(selected[:256])
    evaluation_pictures = np.sort(selected[256:768])
    if args.task == "task2":
        model_root = ROOT / "test/20_task2_cross_day/04_neural_seen/artifacts/task2_mlp_bin50_sessionnorm_assign"
        encoding_alpha = 1.0
    else:
        model_root = ROOT / "test/20_task2_cross_day/04_neural_seen/artifacts/task3_mlp_bin50_pooled_sessionnorm_assign"
        encoding_alpha = 0.01
    grouped = {mouse: [block for block in blocks if block.mouse == mouse] for mouse in {b.mouse for b in blocks}}
    rows = []
    for mouse, mouse_blocks in grouped.items():
        model = load_classifier(model_root / mouse / "classifier.pt", device)
        encoding_neural = []
        encoding_targets = []
        for block in mouse_blocks:
            block_split = split.blocks[block.key]
            if len(block_split.train_trials):
                targets, neural = aggregate_by_target(
                    block.features[block_split.train_trials], block.trial_targets[block_split.train_trials]
                )
                encoding_neural.append(np.log1p(neural))
                encoding_targets.append(targets)
        path = RidgePath().fit(bank_coefficients[np.concatenate(encoding_targets)], np.concatenate(encoding_neural))
        prototypes = path.predict(bank_coefficients, encoding_alpha)
        for block in mouse_blocks:
            block_split = split.blocks[block.key]
            if not len(block_split.validation_trials):
                continue
            raw = block.features[block_split.validation_trials]
            raw_targets = block.trial_targets[block_split.validation_trials]
            if args.task == "task2":
                targets, counts = aggregate_by_target(raw, raw_targets)
                normalizer = fit_session_normalizer(raw)
                mlp_counts = apply_session_normalizer(counts, normalizer)
            else:
                targets = raw_targets
                counts = raw
                normalizer = fit_session_normalizer(block.features[block_split.train_trials])
                mlp_counts = apply_session_normalizer(counts, normalizer)
            mlp_scores = row_zscore(predict_logits(model, mlp_counts, device=device))
            encoding_scores = row_zscore(response_scores(np.log1p(counts), prototypes, metric="mse", device=device))
            scores = 0.5 * mlp_scores + 0.5 * encoding_scores
            rows.append({
                "block": block.key,
                "targets": targets,
                "scores": scores,
                "assigned": unique_assignment(scores),
            })
        print(f"scored {mouse}", flush=True)

    def predictions_for(weight, pictures, baseline=False):
        prediction_parts = []
        target_parts = []
        group_parts = []
        index_parts = []
        true_parts = []
        for row in rows:
            mask = np.isin(row["targets"], pictures)
            indices = row["assigned"] if baseline else structural_medoid(row["scores"], row["assigned"], similarity, weight)
            prediction_parts.append(images[indices[mask]])
            target_parts.append(images[row["targets"][mask]])
            group_parts.extend([row["block"]] * int(mask.sum()))
            index_parts.append(indices[mask])
            true_parts.append(row["targets"][mask])
        predictions = np.concatenate(prediction_parts)
        references = np.concatenate(target_parts)
        groups = np.asarray(group_parts)
        predicted_indices = np.concatenate(index_parts)
        true_indices = np.concatenate(true_parts)
        scores = score_predictions(predictions, references, device=device, batch_size=args.batch_size)
        summary = summarize_scores(scores, groups)
        summary["top1_accuracy"] = float((predicted_indices == true_indices).mean())
        return summary, predictions, references, groups

    tuning = []
    for weight in (0.0, 0.1, 0.5, 1.0, 2.0):
        summary, _, _, _ = predictions_for(weight, tune_pictures)
        tuning.append({"neural_weight": weight, "ocf_mean": summary["ocf_mean"], "top1_accuracy": summary["top1_accuracy"]})
        print(tuning[-1], flush=True)
    best = max(tuning, key=lambda row: row["ocf_mean"])
    summary, predictions, references, groups = predictions_for(best["neural_weight"], evaluation_pictures)
    baseline, _, _, _ = predictions_for(0, evaluation_pictures, baseline=True)
    output_dir = ROOT / "test/60_fusion_transduction/12_structural_medoid"
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / f"{args.task}_predictions.npz",
        predictions=predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    result = {
        "experiment": "12_structural_medoid",
        "name": f"{args.task}_structural_medoid",
        "config": {"task": args.task, "candidate_count": 20, "encoding_alpha": encoding_alpha},
        "protocol": {"tune_picture_count": len(tune_pictures), "evaluation_pictures": evaluation_pictures.tolist()},
        "tuning": tuning,
        "selected": best,
        "evaluation": summary,
        "evaluation_baseline": baseline,
        "note": "Structural medoid may break one-to-one assignment; it is evaluated as an independent postprocessor.",
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
    }
    path = output_dir / f"results_{args.task}_medoid.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL {path}: medoid={summary['ocf_mean']:.6f}; baseline={baseline['ocf_mean']:.6f}; "
        f"selected={best}",
        flush=True,
    )


if __name__ == "__main__":
    main()
