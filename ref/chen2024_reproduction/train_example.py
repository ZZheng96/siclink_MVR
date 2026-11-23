"""Train the ported Chen et al. (2024) decoder on the authors' example data.

2026-11-24移植/修改自beizai/Decoding（train.py）。
Paper: https://doi.org/10.1371/journal.pcbi.1012297
Source: https://github.com/beizai/Decoding
Source commit: 631ebbcec0eedb907dd1ec24a908c4f99324c219
License: GPL-3.0; see LICENSE.

Scope: this is a shortened reproduction on the authors' 3200-frame example file,
not the full 400-epoch run reported in the paper. It exists to verify that the
ported model trains end to end and improves a pixel-level metric on held-out
frames. SSIM/PSNR are the paper's metrics, not the competition OCF.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn

from model import Chen2024Decoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "downloads" / "chen2024_training"


def load_split(prefix):
    images = np.load(DATA / f"movie_03_{prefix}_pic_1999_VISp_800.npy")
    spikes = np.load(DATA / f"movie_03_{prefix}_spike_1999_VISp_800.npy")
    assert images.shape[1:] == (256, 256) and spikes.shape[1] == 800
    assert len(images) == len(spikes)
    return (
        torch.from_numpy(spikes).float(),
        torch.from_numpy(images).float().unsqueeze(1),
    )


@torch.no_grad()
def evaluate(model, spikes, images, device, batch_size):
    model.eval()
    ssim_values, psnr_values, mse_values = [], [], []
    for start in range(0, len(spikes), batch_size):
        stop = min(start + batch_size, len(spikes))
        _, prediction = model(spikes[start:stop].to(device))
        prediction = prediction.clamp(0.0, 1.0).cpu().numpy()[:, 0]
        reference = images[start:stop].cpu().numpy()[:, 0]
        for index in range(len(prediction)):
            ssim_values.append(
                structural_similarity(reference[index], prediction[index], data_range=1.0)
            )
            psnr_values.append(
                peak_signal_noise_ratio(reference[index], prediction[index], data_range=1.0)
            )
            mse_values.append(float(np.mean((reference[index] - prediction[index]) ** 2)))
    # Match the paper's evaluation window: skip the first five frames.
    return {
        "ssim_mean": float(np.mean(ssim_values[5:])),
        "psnr_mean_db": float(np.mean(psnr_values[5:])),
        "mse_mean": float(np.mean(mse_values[5:])),
        "frames_scored": len(ssim_values[5:]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--output", type=str, default="runs/example_short")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_spikes, train_images = load_split("train")
    test_spikes, test_images = load_split("test")
    print(f"device={device} train={len(train_spikes)} test={len(test_spikes)}", flush=True)

    model = Chen2024Decoder(ncell=800).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    baseline = evaluate(model, test_spikes, test_images, device, args.eval_batch_size)
    print("initial " + json.dumps(baseline), flush=True)

    history = [{"epoch": 0, "train_loss": None, **baseline}]
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_spikes))
        total, count = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            if len(index) < 2:
                continue
            spikes = train_spikes[index].to(device)
            target = train_images[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            _, prediction = model(spikes)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
            count += len(index)
        train_loss = total / max(count, 1)
        metrics = evaluate(model, test_spikes, test_images, device, args.eval_batch_size)
        elapsed = time.time() - started
        record = {"epoch": epoch, "train_loss": train_loss, **metrics,
                  "elapsed_s": round(elapsed, 1)}
        history.append(record)
        print(json.dumps(record), flush=True)
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "ncell": 800},
            output_dir / "checkpoint.pt",
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
