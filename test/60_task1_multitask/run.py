"""Exploratory ablation: does a classification auxiliary head improve Task1 DCT reconstruction?

Two arms share identical data, trunk, optimizer and epochs:
  A) regression only (v5 configuration)
  B) regression + 0.1 * cross-entropy on the 2473 training characters
Fixed protocol: Task1 seed42 character holdout, 4-repeat means per target per block,
held-out 211 pictures.  Reports corrected DCT R2, binary OCF and directional retrieval OCF.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(r"d:\WORK\siclink")
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions
from src.features import aggregate_by_target
from src.preprocessing import load_cached_blocks
from src.retrieval import retrieve_glyphs
from src.ridge import dct_decode, dct_encode
from src.splits import make_split

CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
OUT = ROOT / "test/60_task1_multitask"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = 512
DROPOUT = 0.4
WEIGHT_DECAY = 0.01
LR = 3e-4
BATCH = 256
EPOCHS = 20
EVAL_EPOCHS = (4, 6, 8, 10, 12, 16, 20)
CUTOFF = 32
THRESHOLD = 0.4
LAMBDA = 0.1
SEED = 42
MICE = ("mice1", "mice2")


def build(split, gt, mouse):
    train_x, train_dct, train_label = [], [], []
    eval_x, eval_gt = [], []
    classes = np.array(sorted(set(np.unique(np.concatenate([b.targets for b in load_cached_blocks(CACHE, split="train", bin_ms=25, mouse=mouse)]))).difference(
        set(split.validation_pictures.tolist()))), dtype=np.int64)
    lookup = {int(c): i for i, c in enumerate(classes)}
    for block in load_cached_blocks(CACHE, split="train", bin_ms=25, mouse=mouse):
        part = split.blocks[block.key]
        targets, counts = aggregate_by_target(
            np.asarray(block.features[part.train_trials], dtype=np.float32),
            block.trial_targets[part.train_trials],
        )
        keep = np.isin(targets, classes)
        targets, counts = targets[keep], counts[keep]
        train_x.append(counts)
        train_dct.append(dct_encode(gt[targets], CUTOFF))
        train_label.append(np.array([lookup[int(t)] for t in targets], dtype=np.int64))
        vt, vc = aggregate_by_target(
            np.asarray(block.features[part.validation_trials], dtype=np.float32),
            block.trial_targets[part.validation_trials],
        )
        keep_v = np.isin(vt, split.validation_pictures)
        eval_x.append(vc[keep_v])
        eval_gt.append(gt[vt[keep_v]])
    return (np.concatenate(train_x), np.concatenate(train_dct), np.concatenate(train_label),
            np.concatenate(eval_x), np.concatenate(eval_gt), len(classes), classes)


class MultiTask(nn.Module):
    def __init__(self, n_features, n_classes, *, use_classifier):
        super().__init__()
        self.use_classifier = use_classifier
        self.trunk = nn.Sequential(
            nn.Linear(n_features, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.reg = nn.Linear(HIDDEN, CUTOFF * CUTOFF)
        self.cls = nn.Linear(HIDDEN, n_classes) if use_classifier else None

    def set_normalization(self, features):
        mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(features.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("scale", torch.from_numpy(scale))

    def forward(self, x):
        h = self.trunk((x.float() - self.mean.float()) / self.scale.float())
        return self.reg(h), (self.cls(h) if self.cls is not None else None)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def run_arm(train_x, train_dct, train_label, eval_x, eval_gt, n_classes, bank, gt_mean, gt_var, use_classifier):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = MultiTask(train_x.shape[1], n_classes, use_classifier=use_classifier).to(DEVICE)
    model.set_normalization(train_x)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    history = {}
    for epoch in range(1, EPOCHS + 1):
        order = np.random.default_rng(SEED + epoch).permutation(len(train_x))
        model.train()
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            bx = torch.from_numpy(np.ascontiguousarray(train_x[idx])).to(DEVICE)
            by = torch.from_numpy(np.ascontiguousarray(train_dct[idx])).to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                reg, logits = model(bx)
                loss = F.mse_loss(reg.float(), by)
                if logits is not None:
                    bl = torch.from_numpy(train_label[idx]).to(DEVICE)
                    loss = loss + LAMBDA * F.cross_entropy(logits.float(), bl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch in EVAL_EPOCHS:
            model.eval()
            with torch.inference_mode():
                chunks = []
                for start in range(0, len(eval_x), BATCH):
                    bx = torch.from_numpy(np.ascontiguousarray(eval_x[start:start + BATCH])).to(DEVICE)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        reg, _ = model(bx)
                    chunks.append(reg.float().cpu().numpy())
            predicted = np.concatenate(chunks).astype(np.float32)
            mse = float(((predicted - dct_encode(eval_gt, CUTOFF)) ** 2).sum())
            r2 = 1.0 - mse / gt_var
            continuous = np.clip(np.asarray(dct_decode(predicted, CUTOFF), dtype=np.float32), 0.0, 1.0)
            binary = (continuous >= THRESHOLD).astype(np.float32)
            b = score_predictions(binary, eval_gt, device=str(DEVICE), batch_size=16)
            retrieved, _ = retrieve_glyphs(continuous, bank, metric="directional", blur_sigma=0.0, device=DEVICE)
            r = score_predictions(retrieved, eval_gt, device=str(DEVICE), batch_size=16)
            history[epoch] = {"dct_r2": float(r2), "binary_ocf": summarize(b), "retrieval_ocf": summarize(r)}
            print(f"arm={'multi' if use_classifier else 'reg'} epoch={epoch} r2={r2:.4f} "
                  f"binary={float(b.mean()):.3f} retrieval={float(r.mean()):.3f}", flush=True)
    return history


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pictures = np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False)
    gt = (np.asarray(pictures) >= 128).astype(np.float32)
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=25)
    split = make_split(blocks, "task1", seed=42)
    bank = gt[~np.isin(np.arange(len(gt)), split.validation_pictures)]
    gt_coeff = dct_encode(gt[split.validation_pictures], CUTOFF)
    gt_var = float(((gt_coeff - gt_coeff.mean(axis=0)) ** 2).sum())

    output = {"scope": "exploratory multitask ablation; Task1 char holdout; not a selection result",
              "config": {"hidden": HIDDEN, "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
                         "lr": LR, "batch": BATCH, "epochs": EPOCHS, "eval_epochs": list(EVAL_EPOCHS),
                         "lambda": LAMBDA, "cutoff": CUTOFF, "threshold": THRESHOLD},
              "arms": {}}
    for arm, use_classifier in (("regression_only", False), ("multitask", True)):
        per_mouse = {}
        pooled = {}
        for mouse in MICE:
            train_x, train_dct, train_label, eval_x, eval_gt, n_classes, _ = build(split, gt, mouse)
            history = run_arm(train_x, train_dct, train_label, eval_x, eval_gt, n_classes,
                              bank, None, gt_var, use_classifier)
            per_mouse[mouse] = {"n_classes": int(n_classes), "history": history}
        output["arms"][arm] = per_mouse
    (OUT / "exploratory_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote exploratory_results.json", flush=True)


if __name__ == "__main__":
    main()
