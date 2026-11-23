"""P1: official conditional DDPM evaluated with strict task-specific OCF validation."""

import argparse
from copy import deepcopy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.ddpm import build_model, diffusion_scheduler, sample_images, training_loss
from src.evaluation import score_predictions, summarize_scores, sweep_thresholds
from src.features import aggregate_by_target, load_feature_blocks
from src.retrieval import retrieve_glyphs
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task1", "task2", "task3"))
    parser.add_argument("--name")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--mouse", choices=("mice1", "mice2"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--ocf-batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name or f"{args.task}_ddpm_e{args.epochs}"
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=args.bin_ms, mouse=args.mouse)
    split = make_split(blocks, args.task, seed=42)
    if args.task == "task1":
        shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
        tune_pictures = np.sort(shuffled[:64])
        threshold_pictures = np.sort(shuffled[:16])
        evaluation_pictures = np.sort(shuffled[64:])
    else:
        selected = np.random.default_rng(123).permutation(len(images))
        tune_pictures = np.sort(selected[:256])
        threshold_pictures = np.sort(selected[:64])
        evaluation_pictures = np.sort(selected[256:768])
    output_dir = ROOT / "test/07_ddpm"
    artifacts = output_dir / "artifacts" / name
    artifacts.mkdir(parents=True, exist_ok=True)

    grouped = {mouse: [block for block in blocks if block.mouse == mouse] for mouse in {b.mouse for b in blocks}}
    rows = []
    for group, group_blocks in grouped.items():
        train_x_parts = []
        train_y_parts = []
        validation_x_parts = []
        validation_y_parts = []
        validation_keys = []
        for block in group_blocks:
            block_split = split.blocks[block.key]
            if len(block_split.train_trials):
                train_x_parts.append(block.features[block_split.train_trials])
                train_y_parts.append(block.trial_targets[block_split.train_trials])
            if len(block_split.validation_trials):
                if args.task in ("task1", "task2"):
                    validation_targets, validation_x = aggregate_by_target(
                        block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
                    )
                else:
                    validation_x = block.features[block_split.validation_trials]
                    validation_targets = block.trial_targets[block_split.validation_trials]
                validation_x_parts.append(validation_x)
                validation_y_parts.append(validation_targets)
                validation_keys.extend([block.key] * len(validation_targets))
        train_x = np.concatenate(train_x_parts)
        train_targets = np.concatenate(train_y_parts)
        validation_x = np.concatenate(validation_x_parts)
        validation_targets = np.concatenate(validation_y_parts)
        train_images = images[train_targets][:, None]
        dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_images))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        model = build_model(train_x.shape[1]).to(device, memory_format=torch.channels_last)
        ema = deepcopy(model).eval().requires_grad_(False)
        scheduler = diffusion_scheduler()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, fused=device.type == "cuda")
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for counts, targets in loader:
                counts = counts.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).mul_(2).sub_(1)
                optimizer.zero_grad(set_to_none=True)
                loss = training_loss(model, scheduler, targets, counts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    for average, parameter in zip(ema.parameters(), model.parameters()):
                        average.lerp_(parameter, 0.001)
                total_loss += loss.item() * len(counts)
            history.append({"epoch": epoch, "train_loss": total_loss / len(train_x)})
            print(f"{group} {history[-1]}", flush=True)
        torch.save({"inputs": train_x.shape[1], "state_dict": ema.state_dict(), "epoch": args.epochs}, artifacts / f"{group.replace('/', '_')}.pt")
        ema.eval()
        prediction_parts = []
        generator = torch.Generator(device=device).manual_seed(742)
        with torch.inference_mode():
            for start in range(0, len(validation_x), args.sample_batch_size):
                counts = torch.from_numpy(validation_x[start:start + args.sample_batch_size]).to(device, dtype=torch.float32)
                samples = sample_images(ema, scheduler, counts, generator)
                prediction_parts.append(samples[:, 0].float().cpu().numpy())
        predictions = np.concatenate(prediction_parts)
        rows.append({
            "predictions": predictions,
            "targets": validation_targets,
            "groups": np.asarray(validation_keys),
        })
        del model, ema, optimizer, loader, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def collect(pictures):
        predictions = []
        targets = []
        groups = []
        for row in rows:
            mask = np.isin(row["targets"], pictures)
            predictions.append(row["predictions"][mask])
            targets.append(images[row["targets"][mask]])
            groups.extend(row["groups"][mask].tolist())
        return np.concatenate(predictions), np.concatenate(targets), np.asarray(groups)

    threshold_predictions, threshold_targets, _ = collect(threshold_pictures)
    threshold_rows = sweep_thresholds(
        threshold_predictions, threshold_targets, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        device=str(device), batch_size=args.ocf_batch_size,
    )
    best_threshold = max(threshold_rows, key=lambda row: row["ocf_mean"])
    evaluation_predictions, evaluation_targets, evaluation_groups = collect(evaluation_pictures)
    scores = score_predictions(
        evaluation_predictions, evaluation_targets, threshold=best_threshold["threshold"],
        device=str(device), batch_size=args.ocf_batch_size,
    )
    direct_summary = summarize_scores(scores, evaluation_groups)
    continuous = score_predictions(
        evaluation_predictions, evaluation_targets, device=str(device), batch_size=args.ocf_batch_size
    )
    direct_summary["continuous_without_binarization"] = float(continuous.mean())
    bank_indices = np.flatnonzero(~np.isin(np.arange(len(images)), split.validation_pictures)) if args.task == "task1" else np.arange(len(images))
    retrieved, _ = retrieve_glyphs(
        evaluation_predictions, images[bank_indices], metric="directional", device=str(device)
    )
    retrieval_scores = score_predictions(
        retrieved, evaluation_targets, device=str(device), batch_size=args.ocf_batch_size
    )
    retrieval_summary = summarize_scores(retrieval_scores, evaluation_groups)
    np.savez_compressed(
        artifacts / "evaluation_predictions.npz",
        predictions=(evaluation_predictions >= best_threshold["threshold"]).astype(np.uint8) * 255,
        targets=evaluation_targets.astype(np.uint8) * 255,
        group_ids=evaluation_groups,
    )
    result = {
        "experiment": "07_ddpm",
        "name": name,
        "config": vars(args),
        "protocol": {
            **split.metadata(),
            "tune_picture_count": len(tune_pictures),
            "evaluation_picture_count": len(evaluation_pictures),
            "pooling": "Per mouse across permitted training sessions",
        },
        "threshold_sweep": threshold_rows,
        "selected_threshold": best_threshold,
        "evaluation": {"direct": direct_summary, "retrieval": retrieval_summary},
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "elapsed_seconds": time.time() - started,
    }
    result_path = output_dir / f"results_{name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL {result_path}: direct={direct_summary['ocf_mean']:.6f}; "
        f"retrieval={retrieval_summary['ocf_mean']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
