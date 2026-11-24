"""Fixed-weight Task 3 replication on the next 512 IDs of the same permutation."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("task3_fusion_36", OUT.parent / "36_task3_fusion/run.py")
exp36 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp36)
exp35 = exp36.exp35

WEIGHTS = {"v5": 0.0, "blend_0.25": 0.25}


def main():
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = (exp35.load_images(exp35.DATA) >= 128).astype(np.float32)
    if images.shape != (exp35.N_GLYPHS, 64, 64):
        raise ValueError("Unexpected public training gallery")
    bank = exp35.dct_encode(images, 16).astype(np.float32)
    blocks = exp35.load_feature_blocks(exp35.DATA, "train", bin_ms=50)
    if sorted(b.key for b in blocks) != [f"{m}/session{s}" for m in ("mice1", "mice2") for s in range(1, 5)]:
        raise ValueError("Expected all eight public training blocks")
    split = exp35.make_split(blocks, "task3", seed=exp35.SEED)
    rows, contexts, checkpoints = exp36.build_contexts(blocks, split, bank, device)
    del blocks

    order = np.random.default_rng(exp35.PICTURE_SEED).permutation(exp35.N_GLYPHS)
    prior_36 = np.sort(order[256:768])
    pictures = np.sort(order[768:1280])
    if np.intersect1d(prior_36, pictures).size:
        raise ValueError("Replication pictures overlap experiment 36 scoring pictures")
    centered = torch.from_numpy(bank - bank.mean(axis=0, keepdims=True)).to(device)
    normalized = F.normalize(centered, dim=1)
    records = {name: [] for name in WEIGHTS}
    with torch.inference_mode():
        for row in rows:
            targets = row["targets"]
            subset = np.flatnonzero(np.isin(targets, pictures))
            if not np.array_equal(targets[subset], pictures):
                raise ValueError(f"Unexpected picture subset in {row['block']}")
            if not np.array_equal(np.sort(targets), np.arange(exp35.N_GLYPHS)):
                raise ValueError(f"Incomplete validation block {row['block']}")
            context = contexts[row["mouse"]]
            query = exp35.base_query(row, context, np.arange(len(targets)), device)
            image = exp35.continuous_image_scores(query["logits"], centered)
            v5 = (query["mlp_z"] + query["encoding_z"] + exp35.row_zscore(image)) / 3.0
            cosine_z = exp35.row_zscore(
                exp35.contrastive_scores(query["normalized"], context["contrastive"], normalized)
            )
            for name, weight in WEIGHTS.items():
                scores = v5 if weight == 0 else (1 - weight) * v5 + weight * cosine_z
                assigned = exp35.unique_assignment(scores)
                top = np.argsort(-scores[subset], axis=1, kind="stable")[:, :exp35.K].copy()
                records[name].append({
                    "block": row["block"], "mouse": row["mouse"], "targets": targets[subset],
                    "assigned": assigned[subset], "top32": top,
                })
    records = {name: exp35.concatenate_records(parts) for name, parts in records.items()}
    cache = {}
    assigned_pairs = np.unique(np.concatenate([
        np.column_stack((record["assigned"], record["targets"])) for record in records.values()
    ]).astype(np.int64), axis=0)
    for start in range(0, len(assigned_pairs), 512):
        pairs = assigned_pairs[start:start + 512]
        scores = exp35.score_predictions(
            images[pairs[:, 0]], images[pairs[:, 1]], device=str(device), batch_size=32
        )
        for pair, score in zip(pairs, scores):
            cache[tuple(map(int, pair))] = float(score)
    def assigned_summary(record):
        targets = record["targets"]
        correct = record["assigned"] == targets
        final_values = np.asarray([
            cache[(int(predicted), int(target))]
            for predicted, target in zip(record["assigned"], targets)
        ])

        def summarize(mask):
            n = int(mask.sum())
            return {
                "n": n,
                "final_correct_count": int(correct[mask].sum()),
                "final_accuracy": float(correct[mask].mean()),
                "final_ocf": float(final_values[mask].mean()),
            }

        return {
            "overall": summarize(np.ones(len(targets), dtype=bool)),
            "by_mouse": {
                mouse: summarize(record["mice"] == mouse)
                for mouse in sorted(set(record["mice"].tolist()))
            },
            "by_block": {
                block: summarize(record["groups"] == block)
                for block in sorted(set(record["groups"].tolist()))
            },
        }

    metrics = {name: assigned_summary(record) for name, record in records.items()}

    protocol = {
        "data_scope": "public train only",
        "task": "task3 single held-out trial",
        "split_seed": exp35.SEED,
        "picture_seed": exp35.PICTURE_SEED,
        "experiment_36_scoring_pictures": prior_36.tolist(),
        "replication_pictures": pictures.tolist(),
        "picture_partition": "same default_rng(123) permutation as experiment 36; replication uses sorted permutation slice [768:1280]",
        "nonoverlap_with_experiment_36_scoring_pictures": True,
        "training_index_scope": "train_trials only for session normalizers and ridge prototypes; frozen public train checkpoints",
        "training_checkpoints_and_full_block_assignment_shared_with_experiment_36": True,
        "assignment_scope": "full-block 2748-row Hungarian before retaining 512 replication pictures",
        "candidate_ranking": "stable descending scores over all 2748 glyphs; top 32 before assignment",
        "ocf": "official PNG-parity OCF on binary public training glyphs, scored with experiment 35 exact scoring utility",
        "evaluated_variants": ["v5", "blend_0.25"],
        "blend_0.25_weight": 0.25,
        "independence_disclosure": "Scoring glyph IDs do not overlap experiment 36. Training checkpoints, training context construction, and full-block assignment protocol are shared, so this is not a completely independent replication.",
    }
    output = {
        "experiment": OUT.name,
        "protocol": protocol,
        "contrastive_checkpoints": checkpoints,
        "heldout": metrics,
        "heldout_evaluated_variants": list(WEIGHTS),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": sys.executable,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    temporary = OUT / "results.json.tmp"
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUT / "results.json")


if __name__ == "__main__":
    main()
