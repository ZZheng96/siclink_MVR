"""Compact nonlinear classifiers for seen-character neural decoding."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SpikeClassifier(nn.Module):
    """Map binned spike counts to seen-character logits."""

    def __init__(self, n_features, n_classes=2748, hidden=512, dropout=0.2, apply_log=True):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden = hidden
        self.dropout = dropout
        self.apply_log = apply_log
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_scale", torch.ones(n_features))
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def set_normalization(self, features):
        features = np.asarray(features, dtype=np.float32)
        transformed = np.log1p(np.maximum(features, 0)) if self.apply_log else features
        mean = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = transformed.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_mean.copy_(torch.from_numpy(mean).to(self.feature_mean.device))
        self.feature_scale.copy_(torch.from_numpy(np.maximum(scale, 1e-5)).to(self.feature_scale.device))

    def forward(self, features):
        if self.apply_log:
            features = torch.log1p(features.clamp_min(0))
        features = (features - self.feature_mean) / self.feature_scale
        return self.network(features)


class SpikeTemporalClassifier(SpikeClassifier):
    """Learn unit-specific temporal kernels before character classification."""

    def __init__(self, n_features, n_units, n_bins, n_classes=2748, hidden=512, dropout=0.2, apply_log=True):
        super().__init__(n_features, n_classes, hidden, dropout, apply_log)
        if n_units * n_bins != n_features:
            raise ValueError("n_units*n_bins must equal n_features.")
        self.n_units = n_units
        self.n_bins = n_bins
        self.architecture = "temporal"
        self.network = nn.Identity()
        channels = 4
        self.temporal = nn.Conv1d(
            n_units, n_units * channels, kernel_size=3, padding=1, groups=n_units
        )
        flattened = n_units * channels * n_bins
        self.head = nn.Sequential(
            nn.LayerNorm(flattened),
            nn.Linear(flattened, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, features):
        if self.apply_log:
            features = torch.log1p(features.clamp_min(0))
        features = (features - self.feature_mean) / self.feature_scale
        features = features.reshape(-1, self.n_units, self.n_bins)
        features = torch.nn.functional.silu(self.temporal(features))
        return self.head(features.flatten(1))


def _loader(features, targets, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(features, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def _evaluate(model, loader, device):
    model.eval()
    correct = 0
    total_loss = 0.0
    count = 0
    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(features)
                loss = nn.functional.cross_entropy(logits.float(), targets)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total_loss += loss.item() * len(targets)
            count += len(targets)
    return {"loss": total_loss / count, "accuracy": correct / count}


def fit_classifier(model, train, validation, output, *, epochs=30, batch_size=256,
                   learning_rate=3e-4, weight_decay=1e-3, device=None, patience=8, seed=42):
    """Fit with best-validation-accuracy checkpointing and return epoch history."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y = train
    val_x, val_y = validation
    model.set_normalization(train_x)
    model.to(device)
    train_loader = _loader(train_x, train_y, batch_size, True)
    val_loader = _loader(val_x, val_y, batch_size, False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=device.type == "cuda"
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "classifier.pt"
    history = []
    best_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(features)
                loss = nn.functional.cross_entropy(logits.float(), targets, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(targets)
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
        validation_metrics = _evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_x),
            "train_accuracy": train_correct / len(train_x),
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            stale_epochs = 0
            torch.save({
                "n_features": model.n_features,
                "n_classes": model.n_classes,
                "hidden": model.hidden,
                "dropout": model.dropout,
                "apply_log": model.apply_log,
                "architecture": getattr(model, "architecture", "mlp"),
                "n_units": getattr(model, "n_units", None),
                "n_bins": getattr(model, "n_bins", None),
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_accuracy": best_accuracy,
            }, checkpoint)
        else:
            stale_epochs += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale_epochs >= patience:
            break
    return history


def fit_classifier_fixed_epochs(model, train, *, epochs, batch_size=256, learning_rate=3e-4,
                                weight_decay=1e-3, device=None, seed=42):
    """Fit an OOF base learner for a fixed number of epochs without validation labels."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y = train
    model.set_normalization(train_x)
    model.to(device).train()
    torch.manual_seed(seed)
    loader = _loader(train_x, train_y, batch_size, True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=device.type == "cuda"
    )
    history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(features)
                loss = nn.functional.cross_entropy(logits.float(), targets, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(features)
        history.append(total_loss / len(train_x))
        print(f"OOF MLP epoch={epoch}/{epochs} loss={history[-1]:.8f}", flush=True)
    return model, history


@torch.inference_mode()
def predict_logits(model, features, batch_size=512, device=None):
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.eval()
    output = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(np.asarray(features[start:start + batch_size], dtype=np.float32)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch)
        output.append(logits.float().cpu().numpy())
    return np.concatenate(output)


def load_classifier(path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture", "mlp") == "temporal":
        model = SpikeTemporalClassifier(
            checkpoint["n_features"], checkpoint["n_units"], checkpoint["n_bins"],
            checkpoint["n_classes"], checkpoint["hidden"], checkpoint["dropout"], checkpoint.get("apply_log", True),
        )
    else:
        model = SpikeClassifier(
            checkpoint["n_features"], checkpoint["n_classes"], checkpoint["hidden"], checkpoint["dropout"],
            checkpoint.get("apply_log", True),
        )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


class SpikeRegressor(SpikeClassifier):
    """Use the same normalized MLP trunk for continuous DCT regression."""

    def __init__(self, n_features, n_outputs, hidden=1024, dropout=0.2, apply_log=True):
        super().__init__(n_features, n_outputs, hidden, dropout, apply_log)
        self.n_outputs = n_outputs


def fit_regressor(model, train, validation, output, *, epochs=50, batch_size=256,
                  learning_rate=3e-4, weight_decay=1e-3, device=None, patience=10, seed=42):
    """Fit continuous outputs with best-validation-MSE checkpointing."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y = train
    val_x, val_y = validation
    model.set_normalization(train_x)
    model.to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(np.asarray(train_x, dtype=np.float32)),
        torch.from_numpy(np.asarray(train_y, dtype=np.float32)),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_features = torch.from_numpy(np.asarray(val_x, dtype=np.float32))
    val_targets = torch.from_numpy(np.asarray(val_y, dtype=np.float32))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=device.type == "cuda"
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predictions = model(features)
                loss = nn.functional.mse_loss(predictions.float(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(targets)
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for start in range(0, len(val_features), batch_size):
                features = val_features[start:start + batch_size].to(device)
                targets = val_targets[start:start + batch_size].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    predictions = model(features)
                    loss = nn.functional.mse_loss(predictions.float(), targets)
                val_loss += loss.item() * len(targets)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_x),
            "validation_loss": val_loss / len(val_x),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            stale_epochs = 0
            torch.save({
                "n_features": model.n_features,
                "n_outputs": model.n_outputs,
                "hidden": model.hidden,
                "dropout": model.dropout,
                "apply_log": model.apply_log,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_loss": best_loss,
            }, output / "regressor.pt")
        else:
            stale_epochs += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale_epochs >= patience:
            break
    return history


def load_regressor(path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SpikeRegressor(
        checkpoint["n_features"], checkpoint["n_outputs"], checkpoint["hidden"], checkpoint["dropout"],
        checkpoint.get("apply_log", True),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()
