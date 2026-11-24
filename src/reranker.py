"""Listwise similar-glyph reranking for the complete Task 2 pipeline."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class ListwiseGlyphReranker(nn.Module):
    """Score candidates using glyph embeddings, list context and numeric features."""

    def __init__(self, numeric_dim=10, embedding_dim=48):
        super().__init__()
        self.numeric_dim = numeric_dim
        self.embedding_dim = embedding_dim
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, 8, kernel, stride=2, padding=kernel // 2), nn.SiLU(), nn.AvgPool2d(2),
                nn.Conv2d(8, 12, 3, padding=1), nn.SiLU(), nn.AvgPool2d(4),
            ) for kernel in (3, 7)
        )
        self.embedding = nn.Linear(384, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(4 * embedding_dim + numeric_dim + 1, 96),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 1),
        )

    def forward(self, glyphs, numeric):
        batch, count = glyphs.shape[:2]
        flat = glyphs.reshape(batch * count, 1, 64, 64)
        embedded = torch.cat([branch(flat).flatten(1) for branch in self.branches], dim=1)
        embedded = F.normalize(self.embedding(embedded), dim=1).reshape(
            batch, count, self.embedding_dim
        )
        weights = F.softmax(numeric[:, :, 0].float(), dim=1)
        context = torch.bmm(weights.unsqueeze(1), embedded).squeeze(1)
        context = F.normalize(context, dim=1)
        context = context.unsqueeze(1).expand_as(embedded)
        similarity = (embedded * context).sum(dim=2, keepdim=True)
        features = torch.cat(
            (embedded, context, embedded * context, torch.abs(embedded - context), similarity, numeric),
            dim=2,
        )
        return self.head(features).squeeze(2).float()


def _local_zscore(values):
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(
        values.std(axis=1, keepdims=True), 1e-6
    )


def build_candidate_features(base_scores, mlp_z, encoding_z, mouse, top_n=32):
    """Build the stable base-score top-N list and its ten numeric channels."""
    base_scores = np.asarray(base_scores, dtype=np.float32)
    mlp_z = np.asarray(mlp_z, dtype=np.float32)
    encoding_z = np.asarray(encoding_z, dtype=np.float32)
    if base_scores.ndim != 2 or base_scores.shape != mlp_z.shape or base_scores.shape != encoding_z.shape:
        raise ValueError("Base and branch scores must have matching 2-D shapes.")
    if not np.isfinite(base_scores).all() or not np.isfinite(mlp_z).all() or not np.isfinite(encoding_z).all():
        raise ValueError("Reranker scores must be finite.")
    if not 1 <= top_n <= base_scores.shape[1]:
        raise ValueError("top_n must be between one and the candidate count.")
    top = np.argsort(-base_scores, axis=1, kind="stable")[:, :top_n].astype(np.int16)
    gather = lambda values: np.take_along_axis(values, top.astype(np.int64), axis=1)
    base, mlp, encoding = gather(base_scores), gather(mlp_z), gather(encoding_z)
    rank = np.broadcast_to(
        np.arange(top_n, dtype=np.float32)[None, :] / max(top_n - 1, 1), base.shape
    )
    margin = np.broadcast_to(base[:, :1] - base[:, 1:2], base.shape) if top_n > 1 else np.zeros_like(base)
    mouse_flag = np.full(base.shape, float(mouse == "mice2"), dtype=np.float32)
    numeric = np.stack(
        (base, mlp, encoding, base - base[:, :1], _local_zscore(base), _local_zscore(mlp),
         _local_zscore(encoding), rank, margin, mouse_flag),
        axis=-1,
    ).astype(np.float32)
    return top, numeric


@torch.inference_mode()
def predict_utilities(model, top, numeric, images, device, batch_size=128):
    """Predict one utility per top-N candidate without changing candidate membership."""
    device = torch.device(device)
    model.eval()
    bank = torch.from_numpy(np.asarray(images, dtype=np.float32)[:, None]).to(device)
    output = []
    for start in range(0, len(top), batch_size):
        candidates = torch.from_numpy(top[start:start + batch_size].astype(np.int64)).to(device)
        features = torch.from_numpy(numeric[start:start + batch_size]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(bank[candidates], features)
        output.append(logits.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def permute_topk_scores(base_scores, top, utilities, k=8):
    """Permute original base values among the top-K candidates; keep every other score."""
    if not 1 <= k <= top.shape[1]:
        raise ValueError("k must be between one and the shortlist width.")
    scores = np.asarray(base_scores).copy()
    top = np.asarray(top).astype(np.int64)
    order = np.argsort(-np.asarray(utilities)[:, :k], axis=1, kind="stable")
    ranked_candidates = np.take_along_axis(top[:, :k], order, axis=1)
    original_values = np.take_along_axis(scores, top[:, :k], axis=1)
    rows = np.arange(len(scores))[:, None]
    scores[rows, ranked_candidates] = original_values
    return scores


def apply_listwise_reranker(model, base_scores, mlp_z, encoding_z, mouse, images, device,
                            *, top_n=32, k=8, batch_size=128):
    """Return base scores after the frozen top-K permutation; assignment stays downstream."""
    top, numeric = build_candidate_features(base_scores, mlp_z, encoding_z, mouse, top_n)
    utilities = predict_utilities(model, top, numeric, images, device, batch_size)
    return permute_topk_scores(base_scores, top, utilities, k)


def training_dataset(rows):
    """Use only public OOF rows whose true glyph occurs in the original top-N list."""
    candidate_parts, numeric_parts, target_parts = [], [], []
    counts = {}
    for index, row in enumerate(rows):
        top = np.asarray(row["top"]).astype(np.int64)
        targets = np.asarray(row["targets"])
        mask = np.any(top == targets[:, None], axis=1)
        selected = top[mask]
        selected_targets = targets[mask]
        positive = np.argmax(selected == selected_targets[:, None], axis=1)
        if not np.all(selected[np.arange(len(selected)), positive] == selected_targets):
            raise RuntimeError("Positive candidate extraction failed.")
        candidate_parts.append(selected)
        numeric_parts.append(np.asarray(row["numeric"])[mask])
        target_parts.append(positive.astype(np.int64))
        counts[row.get("block", str(index))] = int(mask.sum())
    if not sum(counts.values()):
        raise RuntimeError("No original-top-N training examples.")
    return TensorDataset(
        torch.from_numpy(np.concatenate(candidate_parts)),
        torch.from_numpy(np.concatenate(numeric_parts)),
        torch.from_numpy(np.concatenate(target_parts)),
    ), counts


def fit_listwise_reranker(rows, images, output, *, seed=31, epochs=12, batch_size=32,
                          learning_rate=1e-3, weight_decay=1e-3, top_n=32, device=None):
    """Fit the listwise CNN from public OOF examples and save a portable checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dataset, counts = training_dataset(rows)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ListwiseGlyphReranker().to(device)
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=device.type == "cuda"
    )
    bank = torch.from_numpy(np.asarray(images, dtype=np.float32)[:, None]).to(device)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for candidates, numeric, positive in loader:
            candidates = candidates.to(device)
            numeric = numeric.to(device)
            positive = positive.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(bank[candidates], numeric)
                loss = F.cross_entropy(logits, positive)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(positive)
        history.append(total / len(dataset))
        print(f"RERANKER epoch={epoch}/{epochs} loss={history[-1]:.8f}", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "seed": seed,
        "epochs": epochs,
        "embedding_dim": model.embedding_dim,
        "numeric_dim": model.numeric_dim,
        "top_n": top_n,
    }, output)
    model.eval()
    return model, {"eligible_training_examples": int(sum(counts.values())),
                   "eligible_by_mouse_day": counts, "epoch_loss": history}


def load_listwise_reranker(path, device=None):
    """Load a listwise checkpoint without importing any experimental module."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ListwiseGlyphReranker(
        numeric_dim=checkpoint.get("numeric_dim", 10),
        embedding_dim=checkpoint.get("embedding_dim", 48),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()
