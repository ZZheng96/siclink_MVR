"""Minimal CPU-only software test using synthetic tensors, not research results.

2026-11-24移植/修改自beizai/Decoding（train.py）。
Paper: https://doi.org/10.1371/journal.pcbi.1012297
Source: https://github.com/beizai/Decoding
Source commit: 631ebbcec0eedb907dd1ec24a908c4f99324c219
License: GPL-3.0; see LICENSE.
"""

import os

# Prevent this software test from selecting a CUDA device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn

from model import Chen2024Decoder


def main():
    torch.manual_seed(0)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    ncell = 8
    batch_size = 2
    model = Chen2024Decoder(ncell=ncell).cpu()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)

    # Synthetic values exercise the software only; they are not scientific data.
    spikes = torch.randn(batch_size, ncell, device="cpu")
    target = torch.rand(batch_size, 1, 256, 256, device="cpu")

    optimizer.zero_grad(set_to_none=True)
    dense_out, cae_out = model(spikes)

    assert dense_out.shape == (batch_size, 1, 64, 64)
    assert cae_out.shape == (batch_size, 1, 256, 256)
    assert dense_out.device.type == "cpu" and cae_out.device.type == "cpu"
    assert torch.isfinite(dense_out).all(), "dense_out contains non-finite values"
    assert torch.isfinite(cae_out).all(), "cae_out contains non-finite values"
    assert torch.all((dense_out >= 0) & (dense_out <= 1)), (
        "dense_out must be bounded by sigmoid"
    )

    # Match the original end-to-end MSE objective for one software-test step.
    loss = nn.functional.mse_loss(cae_out, target)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"

    tested_weight = model.cae.model[-2].weight
    weight_before = tested_weight.detach().clone()
    optimizer.step()
    assert not torch.equal(weight_before, tested_weight), "optimizer did not update weights"

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("PASS: CPU forward, backward, and one Adam optimizer step")
    print(f"dense_out shape: {tuple(dense_out.shape)}")
    print(f"cae_out shape: {tuple(cae_out.shape)}")
    print(f"dense_out range: [{dense_out.min().item():.6f}, {dense_out.max().item():.6f}]")
    print(f"finite MSE loss: {loss.item():.6f}")
    print(f"trainable parameters (ncell={ncell}): {parameter_count:,}")
    print("Synthetic tensors were used for software testing only, not research results.")


if __name__ == "__main__":
    main()
