"""Fixed-epoch supervised DCT contrastive retrieval for Task 3, public train only."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions
from src.features import aggregate_by_target, load_feature_blocks
from src.final import apply_session_normalizer, fit_session_normalizer
from src.neural import load_classifier, predict_logits
from src.retrieval import unique_assignment
from src.ridge import RidgePath, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
OUT = ROOT / "test/40_representation_structure/35_task3_contrastive"
ARTIFACTS = OUT / "artifacts"
MODEL_ROOT = ROOT / "test/20_task2_cross_day/04_neural_seen/artifacts/task3_mlp_bin50_pooled_sessionnorm_assign"
SEED = 42
PICTURE_SEED = 123
N_GLYPHS = 2748
K = 32
ENCODING_ALPHA = 0.01
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.1
HIDDEN = 512
EMBEDDING = 256
DROPOUT = 0.1
VARIANTS = ("baseline", "prototype_continuous", "contrastive")


def row_zscore(scores):
    scores = np.asarray(scores, dtype=np.float32)
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)


class ContrastiveEncoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.network = nn.Sequential(
            nn.Linear(n_features, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, EMBEDDING),
        )

    def forward(self, features):
        return F.normalize(self.network(features), dim=1)


def fit_contrastive(mouse, train_blocks, normalized_bank, device):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = ContrastiveEncoder(train_blocks[0]["features"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    bank = torch.from_numpy(normalized_bank).to(device)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        block_losses = {}
        for block in train_blocks:
            # Sample exactly one permitted source trial per glyph, then shuffle glyphs.
            chosen = np.asarray([
                rng.choice(block["glyph_trials"][glyph]) for glyph in range(N_GLYPHS)
            ], dtype=np.int64)
            glyph_order = rng.permutation(N_GLYPHS)
            chosen = chosen[glyph_order]
            block_loss = 0.0
            for start in range(0, N_GLYPHS, BATCH_SIZE):
                rows = chosen[start:start + BATCH_SIZE]
                glyph_ids = glyph_order[start:start + BATCH_SIZE]
                features = torch.from_numpy(block["features"][rows]).to(device)
                targets = bank[torch.from_numpy(glyph_ids).to(device)]
                labels = torch.arange(len(rows), device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    embeddings = model(features)
                    logits = embeddings.float() @ targets.float().T / TEMPERATURE
                    loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                block_loss += loss.item() * len(rows)
                total_loss += loss.item() * len(rows)
            block_losses[block["block"]] = block_loss / N_GLYPHS
        row = {
            "epoch": epoch,
            "train_loss": total_loss / (N_GLYPHS * len(train_blocks)),
            "by_block_train_loss": block_losses,
        }
        if not np.isfinite(row["train_loss"]):
            raise ValueError(f"Nonfinite contrastive loss for {mouse}")
        history.append(row)
        print(
            f"CONTRASTIVE {mouse} epoch={epoch}/{EPOCHS} loss={row['train_loss']:.8f}",
            flush=True,
        )

    output = ARTIFACTS / mouse
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "contrastive.pt"
    torch.save({
        "n_features": model.n_features,
        "hidden": HIDDEN,
        "embedding": EMBEDDING,
        "dropout": DROPOUT,
        "temperature": TEMPERATURE,
        "epoch": EPOCHS,
        "seed": SEED,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }, checkpoint)
    (output / "history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return model, {
        "n_features": model.n_features,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "source_train_rows": int(sum(len(block["features"]) for block in train_blocks)),
        "source_train_rows_by_block": {
            block["block"]: int(len(block["features"])) for block in train_blocks
        },
        "sampled_rows_per_epoch": int(N_GLYPHS * len(train_blocks)),
        "sampled_rows_per_epoch_by_block": {block["block"]: N_GLYPHS for block in train_blocks},
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "history_path": str((output / "history.json").relative_to(ROOT)),
        "history": history,
    }


def prepare(blocks, split, bank_coefficients, device):
    centered = bank_coefficients - bank_coefficients.mean(axis=0, keepdims=True)
    normalized_bank = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    normalized_bank = normalized_bank.astype(np.float32)
    contexts = {}
    training = {}
    rows = []
    for mouse in sorted({block.mouse for block in blocks}):
        mouse_blocks = [block for block in blocks if block.mouse == mouse]
        n_features = mouse_blocks[0].n_features
        classifier = load_classifier(MODEL_ROOT / mouse / "classifier.pt", device)
        if classifier.n_features != n_features:
            raise ValueError(f"Frozen classifier feature mismatch for {mouse}")
        encoding_parts = []
        train_blocks = []
        for block in mouse_blocks:
            part = split.blocks[block.key]
            train_raw = block.features[part.train_trials]
            train_targets = block.trial_targets[part.train_trials]
            if not np.all(np.bincount(train_targets, minlength=N_GLYPHS) == 3):
                raise ValueError(f"Expected three source training repeats per glyph in {block.key}")
            mean_targets, mean_x = aggregate_by_target(train_raw, train_targets)
            if not np.array_equal(mean_targets, np.arange(N_GLYPHS)):
                raise ValueError(f"Incomplete training gallery in {block.key}")
            encoding_parts.append(np.log1p(mean_x))
            normalizer = fit_session_normalizer(train_raw)
            train_normalized = apply_session_normalizer(train_raw, normalizer).astype(np.float32)
            glyph_trials = [np.flatnonzero(train_targets == glyph) for glyph in range(N_GLYPHS)]
            if any(len(indices) != 3 for indices in glyph_trials):
                raise ValueError(f"Unexpected glyph sampling groups in {block.key}")
            train_blocks.append({
                "block": block.key,
                "features": train_normalized,
                "glyph_trials": glyph_trials,
            })
            validation_trials = part.validation_trials
            rows.append({
                "block": block.key,
                "mouse": mouse,
                "targets": block.trial_targets[validation_trials],
                "raw": block.features[validation_trials],
                "normalizer": normalizer,
            })

        encoding_targets = np.tile(np.arange(N_GLYPHS), len(mouse_blocks))
        ridge = RidgePath().fit(
            bank_coefficients[encoding_targets], np.concatenate(encoding_parts)
        )
        prototypes = ridge.predict(bank_coefficients, ENCODING_ALPHA).astype(np.float32)
        del ridge, encoding_parts
        print(f"PREPARED {mouse} full-window encoding", flush=True)
        contrastive, training[mouse] = fit_contrastive(
            mouse, train_blocks, normalized_bank, device
        )
        contexts[mouse] = {
            "classifier": classifier,
            "prototypes": prototypes,
            "contrastive": contrastive,
        }
        del train_blocks
    rows.sort(key=lambda row: row["block"])
    return rows, contexts, training


def continuous_image_scores(logits, centered_bank):
    normalized_bank = F.normalize(centered_bank, dim=1)
    output = []
    for start in range(0, len(logits), 64):
        batch = torch.from_numpy(logits[start:start + 64]).to(centered_bank.device)
        probabilities = F.softmax(batch, dim=1)
        expected = F.normalize(probabilities @ centered_bank, dim=1)
        output.append((expected @ normalized_bank.T).cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def contrastive_scores(features, model, normalized_bank):
    output = []
    for start in range(0, len(features), 128):
        batch = torch.from_numpy(features[start:start + 128]).to(normalized_bank.device)
        with torch.autocast(
            device_type=normalized_bank.device.type, dtype=torch.bfloat16,
            enabled=normalized_bank.device.type == "cuda",
        ):
            embeddings = model(batch)
        output.append((embeddings.float() @ normalized_bank.float().T).cpu().numpy())
    return np.concatenate(output)


def base_query(row, context, indices, device):
    raw = row["raw"][indices]
    normalized = apply_session_normalizer(raw, row["normalizer"]).astype(np.float32)
    logits = predict_logits(context["classifier"], normalized, device=device)
    encoding = response_scores(
        np.log1p(raw), context["prototypes"], metric="mse", device=device
    )
    return {
        "row": row,
        "targets": row["targets"][indices],
        "normalized": normalized,
        "logits": logits,
        "mlp_z": row_zscore(logits),
        "encoding_z": row_zscore(encoding),
    }


def rank_variant(query, name, context, centered_bank, normalized_bank):
    if name == "baseline":
        scores = 0.5 * query["mlp_z"] + 0.5 * query["encoding_z"]
    elif name == "prototype_continuous":
        image = continuous_image_scores(query["logits"], centered_bank)
        scores = (query["mlp_z"] + query["encoding_z"] + row_zscore(image)) / 3.0
    elif name == "contrastive":
        image = contrastive_scores(
            query["normalized"], context["contrastive"], normalized_bank
        )
        scores = (query["mlp_z"] + query["encoding_z"] + row_zscore(image)) / 3.0
    else:
        raise ValueError(name)
    if not np.isfinite(scores).all():
        raise ValueError(f"Nonfinite {name} scores")
    return {
        "block": query["row"]["block"],
        "mouse": query["row"]["mouse"],
        "targets": query["targets"].copy(),
        "top32": np.argsort(-scores, axis=1, kind="stable")[:, :K],
        "assigned": unique_assignment(scores),
    }


def concatenate_records(parts):
    return {
        "targets": np.concatenate([part["targets"] for part in parts]),
        "top32": np.concatenate([part["top32"] for part in parts]),
        "assigned": np.concatenate([part["assigned"] for part in parts]),
        "groups": np.concatenate([
            np.full(len(part["targets"]), part["block"]) for part in parts
        ]),
        "mice": np.concatenate([
            np.full(len(part["targets"]), part["mouse"]) for part in parts
        ]),
    }


def rank_set(rows, contexts, pictures, names, bank_coefficients, device, full_block_assignment=False):
    centered_bank = torch.from_numpy(
        bank_coefficients - bank_coefficients.mean(axis=0, keepdims=True)
    ).to(device)
    normalized_bank = F.normalize(centered_bank, dim=1)
    queries = []
    for row in rows:
        if full_block_assignment:
            indices = np.arange(len(row["targets"]))
            if len(indices) != N_GLYPHS or len(np.unique(row["targets"])) != N_GLYPHS:
                raise ValueError(f"Expected a complete 2748-row block in {row['block']}")
        else:
            indices = np.flatnonzero(np.isin(row["targets"], pictures))
            if not np.array_equal(row["targets"][indices], pictures):
                raise ValueError(f"Unexpected picture subset in {row['block']}")
        queries.append(base_query(row, contexts[row["mouse"]], indices, device))
    output = {}
    for name in names:
        parts = []
        for query in queries:
            part = rank_variant(
                query, name, contexts[query["row"]["mouse"]], centered_bank, normalized_bank
            )
            if full_block_assignment:
                keep = np.flatnonzero(np.isin(part["targets"], pictures))
                if not np.array_equal(part["targets"][keep], pictures):
                    raise ValueError(f"Unexpected heldout subset in {part['block']}")
                part = {
                    "block": part["block"],
                    "mouse": part["mouse"],
                    "targets": part["targets"][keep],
                    "top32": part["top32"][keep],
                    "assigned": part["assigned"][keep],
                }
            parts.append(part)
        output[name] = concatenate_records(parts)
        print(f"RANKED {name}: n={len(output[name]['targets'])}", flush=True)
    return output


def fill_ocf_cache(record_groups, images, cache, device):
    pair_parts = []
    for records in record_groups.values():
        top = records["top32"]
        pair_parts.append(np.column_stack((top.ravel(), np.repeat(records["targets"], K))))
        pair_parts.append(np.column_stack((records["assigned"], records["targets"])))
    pairs = np.unique(np.concatenate(pair_parts).astype(np.int64), axis=0)
    missing = [tuple(map(int, pair)) for pair in pairs if tuple(map(int, pair)) not in cache]
    print(f"OCF unique pairs={len(pairs)} new={len(missing)}", flush=True)
    for start in range(0, len(missing), 512):
        chunk = missing[start:start + 512]
        predicted = np.fromiter((pair[0] for pair in chunk), dtype=np.int64)
        targets = np.fromiter((pair[1] for pair in chunk), dtype=np.int64)
        scores = score_predictions(images[predicted], images[targets], device=str(device), batch_size=32)
        for pair, score in zip(chunk, scores):
            cache[pair] = float(score)
        if start % 16384 == 0:
            print(f"OCF scored {min(start + len(chunk), len(missing))}/{len(missing)}", flush=True)


def metric_summary(records, cache):
    targets = records["targets"]
    top = records["top32"]
    assigned = records["assigned"]
    inside = (top == targets[:, None]).any(axis=1)
    correct = assigned == targets
    oracle_values = np.asarray([
        cache[(int(predicted), int(target))]
        for predicted, target in zip(top.ravel(), np.repeat(targets, K))
    ]).reshape(len(targets), K)
    final_values = np.asarray([
        cache[(int(predicted), int(target))] for predicted, target in zip(assigned, targets)
    ])

    def summarize(mask):
        n = int(mask.sum())
        return {
            "n": n,
            "true_recall_at_32": float(inside[mask].mean()),
            "true_inside_32_count": int(inside[mask].sum()),
            "true_outside_32_count": int((~inside[mask]).sum()),
            "true_inside_32_not_selected_count": int((inside[mask] & ~correct[mask]).sum()),
            "final_correct_count": int(correct[mask].sum()),
            "final_accuracy": float(correct[mask].mean()),
            "oracle32_ocf": float(oracle_values[mask].max(axis=1).mean()),
            "final_ocf": float(final_values[mask].mean()),
        }

    return {
        "overall": summarize(np.ones(len(targets), dtype=bool)),
        "by_mouse": {
            mouse: summarize(records["mice"] == mouse)
            for mouse in sorted(set(records["mice"].tolist()))
        },
        "by_block": {
            block: summarize(records["groups"] == block)
            for block in sorted(set(records["groups"].tolist()))
        },
    }


def main():
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = (load_images(DATA) >= 128).astype(np.float32)
    if images.shape != (N_GLYPHS, 64, 64):
        raise ValueError("Unexpected public training gallery")
    bank_coefficients = dct_encode(images, 16).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    if sorted(block.key for block in blocks) != [
        f"{mouse}/session{session}" for mouse in ("mice1", "mice2") for session in range(1, 5)
    ]:
        raise ValueError("Expected all eight public training blocks")
    split = make_split(blocks, "task3", seed=SEED)
    order = np.random.default_rng(PICTURE_SEED).permutation(N_GLYPHS)
    tune_pictures = np.sort(order[:256])
    heldout_pictures = np.sort(order[256:768])
    rows, contexts, training = prepare(blocks, split, bank_coefficients, device)

    tune_records = rank_set(rows, contexts, tune_pictures, VARIANTS, bank_coefficients, device)
    ocf_cache = {}
    fill_ocf_cache(tune_records, images, ocf_cache, device)
    tuning = {name: metric_summary(tune_records[name], ocf_cache) for name in VARIANTS}
    for name in VARIANTS:
        print(f"TUNE {name} {json.dumps(tuning[name]['overall'])}", flush=True)

    def selection_key(name):
        row = tuning[name]["overall"]
        return row["true_recall_at_32"], row["oracle32_ocf"], row["final_ocf"]

    selected = max(VARIANTS, key=selection_key)
    protocol = {
        "data_scope": "public train only; no competition test files read",
        "task": "task3 single held-out trial",
        "split_seed": SEED,
        "picture_seed": PICTURE_SEED,
        "tune_pictures": tune_pictures.tolist(),
        "heldout_pictures": heldout_pictures.tolist(),
        "training_index_scope": "only split.blocks[block.key].train_trials for encoding, normalization, and contrastive fitting",
        "tune_reused_from_seed42_checkpoint_validation": True,
        "tune_assignment_scope": "per-block rectangular Hungarian assignment inside the fixed 256-picture tune subset; recall/oracle selection metrics are independent of assignment",
        "heldout_assignment_scope": "Hungarian unique_assignment over every complete 2748-row validation block before selecting the fixed 512 heldout pictures",
        "assignment_scope": "tune uses its fixed picture subset; final heldout uses full-block assignment before subsetting",
        "candidate_ranking": "pre-assignment stable descending scores over all 2748 glyphs; 32 candidates",
        "ocf": "official PNG-parity OCF on binary public training glyphs",
        "heldout_used_for_selection": False,
        "heldout_used_for_candidate_design": True,
        "heldout_reused_before_this_experiment": True,
        "heldout_selection_statement": "The winner was locked in selection.json from tune metrics before heldout inference or scoring. Only baseline and the locked winner were then evaluated on heldout; losing variants received no heldout evaluation.",
        "historical_heldout_disclosure": "The fixed 512 IDs were evaluated in earlier local experiments and informed candidate design. They are not a pristine confirmatory holdout, although no heldout metric enters this run's lexicographic selection.",
    }
    definitions = {
        "baseline": "0.5*z(frozen MLP logits)+0.5*z(full-window ridge neural-prototype MSE)",
        "prototype_continuous": "current v5: equal-weight z(MLP), z(full neural prototype), z(cosine of fixed T=1 softmax-expected DCT16 against centered glyph DCT16)",
        "contrastive": "fixed new fusion: equal-weight z(MLP), z(full neural prototype), z(cosine of supervised contrastive embedding against normalized centered glyph DCT16)",
        "contrastive_training": "per mouse and per epoch, sample one source train trial for every glyph in each block, shuffle glyphs within block, unique-glyph batch InfoNCE against that block's DCT16 targets; fixed final epoch, no validation-based model or epoch selection",
        "architecture": "Linear(n_features,512)+LayerNorm+SiLU+Dropout(0.1)+Linear(512,256)+L2 normalize",
        "encoding_alpha": ENCODING_ALPHA,
        "frozen_classifiers": {
            mouse: str((MODEL_ROOT / mouse / "classifier.pt").relative_to(ROOT))
            for mouse in sorted(contexts)
        },
        "selection_rule": "lexicographic maximum of tune true recall@32, tune oracle32 OCF, tune final OCF; fixed candidate order breaks exact ties; baseline is an allowed outcome",
        "candidate_order": list(VARIANTS),
    }
    selection = {
        "experiment": "35_task3_contrastive",
        "command": ".venv/Scripts/python.exe -B -u test/40_representation_structure/35_task3_contrastive/run.py",
        "protocol": protocol,
        "definitions": definitions,
        "training": training,
        "tune": tuning,
        "selected": selected,
        "selected_tune_key": selection_key(selected),
    }
    (OUT / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"SELECTION LOCKED: {selected}", flush=True)

    heldout_names = tuple(dict.fromkeys(("baseline", selected)))
    heldout_records = rank_set(
        rows, contexts, heldout_pictures, heldout_names, bank_coefficients, device,
        full_block_assignment=True,
    )
    fill_ocf_cache(heldout_records, images, ocf_cache, device)
    heldout = {name: metric_summary(heldout_records[name], ocf_cache) for name in heldout_names}
    result = dict(selection)
    result["heldout"] = heldout
    result["heldout_evaluated_variants"] = list(heldout_names)
    result["environment"] = {
        "python": sys.executable,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    path = OUT / "results.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"FINAL {path}", flush=True)
    for name, value in heldout.items():
        print(f"HELDOUT {name} {json.dumps(value['overall'])}", flush=True)


if __name__ == "__main__":
    main()
