"""Task2 listwise CNN similar-glyph reranking with a frozen held-out protocol."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "25_rotating_sessions/run.py"
spec = importlib.util.spec_from_file_location("exp25_for_rerank", SOURCE)
exp25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp25)
e = exp25.e

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DAYS = ("session1", "session2", "session3")
KS = (8, 16, 32)
TOP_N = 32
CNN_SEED = 31
CNN_EPOCHS = 12
CNN_BATCH = 32
EMBEDDING_DIM = 48
EXPECTED_TUNE_OCF = 25.831461860596946
EXPECTED_HELDOUT_OCF = 24.33936155396989
BASELINE_ATOL = 1e-5

CONFIGS = [
    {"name": "original_top1", "method": "original", "k": 1, "global_assignment": False},
    {"name": "original_global_assignment", "method": "original", "k": 1, "global_assignment": True},
] + [
    {"name": f"cnn_k{k}" + ("_global_assignment" if assigned else ""),
     "method": "cnn", "k": k, "global_assignment": assigned}
    for k in KS for assigned in (False, True)
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def local_zscore(values):
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(values.std(axis=1, keepdims=True), 1e-6)


def cache_path(mouse, score_day):
    return OUT / "artifacts/scores" / f"{mouse}_{score_day}.npz"


def save_score_cache(path, row, train_sessions):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            targets=row["targets"], base=row["base"], top=row["top"], numeric=row["numeric"],
            inside32=row["inside32"], train_days=np.asarray([int(day[-1]) for day in train_sessions]),
        )
    temporary.replace(path)


def load_score_cache(path, train_sessions):
    with np.load(path, allow_pickle=False) as data:
        expected_days = [int(day[-1]) for day in train_sessions]
        if data["train_days"].tolist() != expected_days:
            raise ValueError(f"Stale score cache: {path}")
        row = {key: data[key] for key in ("targets", "base", "top", "numeric", "inside32")}
    if row["base"].shape != (2748, 2748) or row["top"].shape != (2748, TOP_N):
        raise ValueError(f"Invalid score cache shapes: {path}")
    if row["numeric"].shape != (2748, TOP_N, 10):
        raise ValueError(f"Invalid numeric feature cache: {path}")
    return row


def make_score_row(group, score_day, train_sessions, images, coefficients):
    mouse = group[0].mouse
    path = cache_path(mouse, score_day)
    if path.is_file():
        print(f"SCORE CACHE {mouse} {score_day}", flush=True)
        return load_score_cache(path, train_sessions), {"mouse": mouse, "score_session": score_day,
                                                         "train_sessions": list(train_sessions), "cache_reused": True}

    train_x_parts, train_y_parts = [], []
    encoding_x_parts, encoding_target_parts = [], []
    target_raw = target_labels = None
    for block in group:
        if block.session not in train_sessions and block.session != score_day:
            continue
        raw = block.features
        labels = block.trial_targets
        targets, counts = e.aggregate_by_target(raw, labels)
        np.testing.assert_array_equal(targets, np.arange(len(images)))
        normalizer = e.fit_session_normalizer(raw)
        if block.session in train_sessions:
            encoding_x_parts.append(np.log1p(counts))
            encoding_target_parts.append(targets)
            train_x_parts.extend((e.apply_session_normalizer(raw, normalizer),
                                  e.apply_session_normalizer(counts, normalizer)))
            train_y_parts.extend((labels, targets))
        if block.session == score_day:
            target_raw = raw
            target_labels = labels
            target_counts = counts
            target_x = e.apply_session_normalizer(counts, normalizer)
    if target_raw is None or len(encoding_x_parts) != len(train_sessions):
        raise ValueError(f"Invalid fold for {mouse}: train={train_sessions}, score={score_day}")

    ridge = e.RidgePath().fit(
        coefficients[np.concatenate(encoding_target_parts)], np.concatenate(encoding_x_parts)
    ).predict(coefficients, 1.0)
    ridge_scores = e.response_scores(np.log1p(target_counts), ridge, metric="mse", device=DEVICE)
    ridge_z = e.row_zscore(ridge_scores).astype(np.float32)
    del ridge, ridge_scores

    logits = []
    if score_day == "session4":
        for seed in e.SEEDS:
            checkpoint = e.seen_checkpoint("task2", seed, mouse)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model = e.load_classifier(checkpoint, DEVICE)
            logits.append(e.predict_logits(model, target_x, device=DEVICE))
            del model
            e.release_memory()
            print(f"FINAL BASELINE {mouse} seed={seed}", flush=True)
    else:
        train_x = np.concatenate(train_x_parts).astype(np.float32, copy=False)
        train_y = np.concatenate(train_y_parts).astype(np.int64, copy=False)
        for seed in e.SEEDS:
            model = exp25.fit_mlp(train_x, train_y, seed, e.EPOCHS["task2"][mouse], len(images))
            logits.append(e.predict_logits(model, target_x, device=DEVICE))
            del model
            e.release_memory()
            print(f"OOF BASELINE {mouse} score={score_day} train={train_sessions} seed={seed}", flush=True)
        del train_x, train_y
    mlp_z = e.row_zscore(np.mean(np.stack(logits), axis=0, dtype=np.float32)).astype(np.float32)
    del logits
    base = (0.5 * mlp_z + 0.5 * ridge_z).astype(np.float32)

    top = np.argsort(-base, axis=1, kind="stable")[:, :TOP_N].astype(np.int16)
    gather = lambda values: np.take_along_axis(values, top.astype(np.int64), axis=1)
    b, m, r = gather(base), gather(mlp_z), gather(ridge_z)
    rows = np.arange(len(base))[:, None]
    rank = np.broadcast_to(np.arange(TOP_N, dtype=np.float32)[None, :] / (TOP_N - 1), b.shape)
    margin = np.broadcast_to(b[:, :1] - b[:, 1:2], b.shape)
    mouse_flag = np.full(b.shape, float(mouse == "mice2"), dtype=np.float32)
    numeric = np.stack((b, m, r, b - b[:, :1], local_zscore(b), local_zscore(m), local_zscore(r),
                        rank, margin, mouse_flag), axis=-1).astype(np.float32)
    targets = np.asarray(target_labels)
    # aggregate_by_target returns sorted unique glyph IDs, not trial labels.
    targets = np.arange(len(images), dtype=np.int64)
    inside32 = np.any(top == targets[:, None], axis=1)
    row = {"mouse": mouse, "block": f"{mouse}/{score_day}", "targets": targets,
           "base": base, "top": top, "numeric": numeric, "inside32": inside32}
    save_score_cache(path, row, train_sessions)
    record = {"mouse": mouse, "score_session": score_day, "train_sessions": list(train_sessions),
              "cache_reused": False, "oof_raw_top1_all_glyphs": float(np.mean(base.argmax(1) == targets)),
              "oof_base_recall32_all_glyphs": float(inside32.mean()),
              "train_rows_including_trial_and_mean_rows": int(sum(len(x) for x in train_x_parts)),
              "mlp_epochs_per_seed": None if score_day == "session4" else e.EPOCHS["task2"][mouse]}
    del mlp_z, ridge_z, b, m, r, numeric, target_raw
    e.release_memory()
    return row, record


class ListwiseGlyphReranker(nn.Module):
    def __init__(self, numeric_dim=10):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, 8, kernel, stride=2, padding=kernel // 2), nn.SiLU(), nn.AvgPool2d(2),
                nn.Conv2d(8, 12, 3, padding=1), nn.SiLU(), nn.AvgPool2d(4),
            ) for kernel in (3, 7)
        )
        self.embedding = nn.Linear(384, EMBEDDING_DIM)
        self.head = nn.Sequential(
            nn.Linear(4 * EMBEDDING_DIM + numeric_dim + 1, 96), nn.SiLU(), nn.Dropout(0.1), nn.Linear(96, 1)
        )

    def forward(self, glyphs, numeric):
        batch, count = glyphs.shape[:2]
        flat = glyphs.reshape(batch * count, 1, 64, 64)
        embedded = torch.cat([branch(flat).flatten(1) for branch in self.branches], dim=1)
        embedded = F.normalize(self.embedding(embedded), dim=1).reshape(batch, count, EMBEDDING_DIM)
        weights = F.softmax(numeric[:, :, 0].float(), dim=1)
        context = torch.bmm(weights.unsqueeze(1), embedded).squeeze(1)
        context = F.normalize(context, dim=1)
        context = context.unsqueeze(1).expand_as(embedded)
        similarity = (embedded * context).sum(dim=2, keepdim=True)
        features = torch.cat((embedded, context, embedded * context, torch.abs(embedded - context),
                              similarity, numeric), dim=2)
        return self.head(features).squeeze(2).float()


def training_dataset(rows):
    candidate_parts, numeric_parts, target_parts = [], [], []
    counts = {}
    for row in rows:
        mask = row["inside32"]
        top = row["top"][mask].astype(np.int64)
        targets = row["targets"][mask]
        positive = np.argmax(top == targets[:, None], axis=1)
        if not np.all(top[np.arange(len(top)), positive] == targets):
            raise RuntimeError("Positive candidate extraction failed")
        candidate_parts.append(top)
        numeric_parts.append(row["numeric"][mask])
        target_parts.append(positive.astype(np.int64))
        counts[row["block"]] = int(mask.sum())
    return TensorDataset(
        torch.from_numpy(np.concatenate(candidate_parts)),
        torch.from_numpy(np.concatenate(numeric_parts)),
        torch.from_numpy(np.concatenate(target_parts)),
    ), counts


def fit_reranker(rows, images):
    dataset, counts = training_dataset(rows)
    if not len(dataset):
        raise RuntimeError("No original-top32 training examples")
    torch.manual_seed(CNN_SEED)
    np.random.seed(CNN_SEED)
    model = ListwiseGlyphReranker().to(DEVICE)
    torch.manual_seed(CNN_SEED)
    loader = DataLoader(dataset, batch_size=CNN_BATCH, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3, fused=DEVICE == "cuda")
    bank = torch.from_numpy(images[:, None]).to(DEVICE)
    history = []
    model.train()
    for epoch in range(1, CNN_EPOCHS + 1):
        total = 0.0
        for candidates, numeric, positive in loader:
            candidates = candidates.to(DEVICE)
            numeric = numeric.to(DEVICE)
            positive = positive.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(bank[candidates], numeric)
                loss = F.cross_entropy(logits, positive)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(positive)
        history.append(total / len(dataset))
        print(f"RERANKER epoch={epoch}/{CNN_EPOCHS} loss={history[-1]:.8f}", flush=True)
    checkpoint = OUT / "artifacts/listwise_cnn.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "seed": CNN_SEED, "epochs": CNN_EPOCHS, "embedding_dim": EMBEDDING_DIM,
                "numeric_dim": 10, "top_n": TOP_N}, checkpoint)
    model.eval()
    del bank, loader, dataset
    e.release_memory()
    return model, history, counts, checkpoint


@torch.inference_mode()
def predict_utilities(model, rows, images):
    bank = torch.from_numpy(images[:, None]).to(DEVICE)
    for row in rows:
        output = []
        for start in range(0, len(row["targets"]), 128):
            candidates = torch.from_numpy(row["top"][start:start + 128].astype(np.int64)).to(DEVICE)
            numeric = torch.from_numpy(row["numeric"][start:start + 128]).to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(bank[candidates], numeric)
            output.append(logits.float().cpu().numpy())
        row["utilities"] = np.concatenate(output).astype(np.float32)
    del bank
    e.release_memory()


def rerank_scores(row, k):
    scores = row["base"].copy()
    top = row["top"].astype(np.int64)
    utilities = row["utilities"]
    order = np.argsort(-utilities[:, :k], axis=1, kind="stable")
    ranked_candidates = np.take_along_axis(top[:, :k], order, axis=1)
    original_values = np.take_along_axis(row["base"], top[:, :k], axis=1)
    rows = np.arange(len(scores))[:, None]
    scores[rows, ranked_candidates] = original_values
    return scores


def build_predictions(rows):
    predictions = {row["mouse"]: {} for row in rows}
    for row in rows:
        mouse = row["mouse"]
        predictions[mouse]["original_top1"] = row["base"].argmax(axis=1).astype(np.int64)
        predictions[mouse]["original_global_assignment"] = e.unique_assignment(row["base"])
        for k in KS:
            scores = rerank_scores(row, k)
            stem = f"cnn_k{k}"
            predictions[mouse][stem] = scores.argmax(axis=1).astype(np.int64)
            predictions[mouse][stem + "_global_assignment"] = e.unique_assignment(scores)
            del scores
        print(f"PREDICTIONS {mouse}: original and K={KS}, with/without full-block assignment", flush=True)
    return predictions


def evaluate(rows, predictions, name, images, pictures):
    by_mouse = {}
    all_scores = []
    total_hits = total_correct = total_n = total_condition_n = 0
    for row in rows:
        mouse = row["mouse"]
        mask = np.isin(row["targets"], pictures)
        if mask.sum() != len(pictures):
            raise ValueError(f"Unexpected picture subset for {mouse}")
        targets = row["targets"][mask]
        predicted = predictions[mouse][name][mask]
        scores = e.score_predictions(images[predicted], images[targets], device=DEVICE, batch_size=16)
        condition = row["inside32"][mask]
        condition_n = int(condition.sum())
        condition_hits = int(np.sum((predicted == targets) & condition))
        hits = int(np.sum(predicted == targets))
        by_mouse[mouse] = {
            "n": int(mask.sum()), "ocf": float(scores.mean()), "ocf_std": float(scores.std()),
            "overall_top1": hits / int(mask.sum()),
            "original_top32_condition_n": condition_n,
            "original_top32_condition_hits": condition_hits,
            "original_top32_conditional_top1": condition_hits / condition_n if condition_n else None,
            "original_top32_recall": condition_n / int(mask.sum()),
        }
        all_scores.append(scores)
        total_hits += hits
        total_correct += condition_hits
        total_n += int(mask.sum())
        total_condition_n += condition_n
    pooled_scores = np.concatenate(all_scores)
    return {
        "pooled": {
            "n": total_n, "ocf": float(pooled_scores.mean()), "ocf_std": float(pooled_scores.std()),
            "overall_top1": total_hits / total_n,
            "original_top32_condition_n": total_condition_n,
            "original_top32_condition_hits": total_correct,
            "original_top32_conditional_top1": total_correct / total_condition_n if total_condition_n else None,
            "original_top32_recall": total_condition_n / total_n,
        },
        "by_mouse": by_mouse,
    }


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    images = (e.load_images(e.DATA) >= 128).astype(np.float32)
    if images.shape != (2748, 64, 64):
        raise ValueError(f"Unexpected image bank: {images.shape}")
    coefficients = e.dct_encode(images, 16)
    blocks = e.load_feature_blocks(e.DATA, "train", bin_ms=50)
    if {block.mouse for block in blocks} != {"mice1", "mice2"}:
        raise ValueError("Expected both training mice")

    oof_rows, final_rows, training_records = [], [], []
    for mouse in ("mice1", "mice2"):
        group = [block for block in blocks if block.mouse == mouse]
        if sorted(block.session for block in group) != ["session1", "session2", "session3", "session4"]:
            raise ValueError(f"Expected four training sessions for {mouse}")
        for score_day in DAYS:
            train_sessions = tuple(day for day in DAYS if day != score_day)
            row, record = make_score_row(group, score_day, train_sessions, images, coefficients)
            oof_rows.append(row)
            training_records.append(record)
        row, record = make_score_row(group, "session4", DAYS, images, coefficients)
        final_rows.append(row)
        training_records.append(record)
    del blocks

    model, history, training_counts, checkpoint = fit_reranker(oof_rows, images)
    predict_utilities(model, final_rows, images)
    predictions = build_predictions(final_rows)

    order = np.random.default_rng(123).permutation(len(images))
    tune_pictures = np.sort(order[:256])
    heldout_pictures = np.sort(order[256:768])
    tune = {config["name"]: evaluate(final_rows, predictions, config["name"], images, tune_pictures)
            for config in CONFIGS}
    tune_baseline = tune["original_global_assignment"]["pooled"]["ocf"]
    tune_error = abs(tune_baseline - EXPECTED_TUNE_OCF)
    if tune_error > BASELINE_ATOL:
        raise RuntimeError(f"Frozen tune baseline mismatch: actual={tune_baseline}, error={tune_error}")

    # This is the only model/configuration selection step. No held-out labels or scores are used.
    selected = max(
        CONFIGS,
        key=lambda config: (
            tune[config["name"]]["pooled"]["ocf"],
            tune[config["name"]]["pooled"]["original_top32_conditional_top1"] or 0.0,
            tune[config["name"]]["pooled"]["overall_top1"],
        ),
    )
    selection = {
        "selected": selected,
        "selection_metric": "highest tune official OCF; ties: conditional top32 accuracy, overall top1, declaration order",
        "tune_baseline_reproduction": {"expected_ocf": EXPECTED_TUNE_OCF, "actual_ocf": tune_baseline,
                                       "absolute_error": tune_error, "passed": True},
        "heldout_policy": "Evaluate only original raw/global baselines and this frozen selection; never reselect on heldout.",
        "tune": tune,
    }
    write_json(OUT / "artifacts/selection.json", selection)
    print(f"FROZEN SELECTION {selected}", flush=True)

    heldout = {
        "original_top1": evaluate(final_rows, predictions, "original_top1", images, heldout_pictures),
        "original_global_assignment": evaluate(final_rows, predictions, "original_global_assignment", images, heldout_pictures),
        "selected": evaluate(final_rows, predictions, selected["name"], images, heldout_pictures),
    }
    heldout_baseline = heldout["original_global_assignment"]["pooled"]["ocf"]
    heldout_error = abs(heldout_baseline - EXPECTED_HELDOUT_OCF)
    if heldout_error > BASELINE_ATOL:
        raise RuntimeError(f"Frozen heldout baseline mismatch: actual={heldout_baseline}, error={heldout_error}")

    result = {
        "experiment": "31_task2_rerank",
        "method": "listwise CNN over original top32 similar-glyph candidates; topK score permutation",
        "protocol": {
            "task": "task2", "bin_ms": 50, "split_seed": 42, "picture_seed": 123,
            "tune_picture_count": len(tune_pictures), "heldout_picture_count": len(heldout_pictures),
            "cnn_training": "session1-3 only; each score row predicted by ridge and three MLPs fitted on the other two training days",
            "tune": "session4 RNG123 first 256 glyphs; sole K/assignment selection set",
            "heldout": "session4 RNG123 next 512 glyphs; frozen baseline and selected configuration only",
            "assignment_scope": "Hungarian unique_assignment on every complete 2748-row block before picture subsetting",
            "conditional_metric": "hits/n restricted to samples whose true glyph is in frozen fused base pre-assignment top32; common denominator for every arm",
        },
        "baseline": {
            "name": "frozen exp18 Task2 ridge_mse + MLP seeds 42/202/777, row-zscore 0.5/0.5",
            "mlp_checkpoint_reuse": True, "ridge_alpha": 1.0,
            "tune_reproduction": selection["tune_baseline_reproduction"],
            "heldout_reproduction": {"expected_ocf": EXPECTED_HELDOUT_OCF, "actual_ocf": heldout_baseline,
                                     "absolute_error": heldout_error, "passed": True},
        },
        "oof_base_training": training_records,
        "reranker": {
            "seed": CNN_SEED, "fixed_epochs": CNN_EPOCHS, "batch_size": CNN_BATCH,
            "learning_rate": 1e-3, "weight_decay": 1e-3, "dropout": 0.1,
            "embedding_dim": EMBEDDING_DIM, "eligible_training_examples": int(sum(training_counts.values())),
            "eligible_by_mouse_day": training_counts, "epoch_loss": history,
            "checkpoint": str(checkpoint.relative_to(e.ROOT)),
            "numeric_features": ["base", "mlp_z", "ridge_z", "base_gap", "base_top32_z", "mlp_top32_z",
                                 "ridge_top32_z", "rank", "top1_margin", "mouse_indicator"],
            "visual_context": "base-score softmax weighted CNN embedding of original top32",
            "eligibility": "training lists require truth in original OOF top32; inference is applied to every row",
            "score_conversion": "permute the original topK base score values according to CNN utility; no extra blend weight",
        },
        "tune": tune,
        "selected": selected,
        "heldout": heldout,
        "heldout_delta_selected_minus_original_global": {
            key: heldout["selected"]["pooled"][key] - heldout["original_global_assignment"]["pooled"][key]
            for key in ("ocf", "overall_top1", "original_top32_conditional_top1")
        },
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": DEVICE},
    }
    write_json(OUT / "results.json", result)
    print("RESULT", json.dumps({"selected": selected, "heldout": heldout,
                                "delta": result["heldout_delta_selected_minus_original_global"]}, indent=2), flush=True)
    print(f"WROTE {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
