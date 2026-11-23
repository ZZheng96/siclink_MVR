"""Chen et al. (2024) decoding model, ported from Keras to PyTorch.

2026-11-24移植/修改自beizai/Decoding（train.py）。
Paper: https://doi.org/10.1371/journal.pcbi.1012297
Source: https://github.com/beizai/Decoding
Source commit: 631ebbcec0eedb907dd1ec24a908c4f99324c219
License: GPL-3.0; see LICENSE.
"""

import torch
from torch import nn


class DenseDecoder(nn.Module):
    """Port of dense_decoder(ncell); input is N x ncell."""

    def __init__(self, ncell: int = 800):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(ncell, 16384),
            nn.BatchNorm1d(16384, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(16384, 8192),
            nn.BatchNorm1d(8192, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8192, 4096),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keras Reshape((64, 64, 1)) becomes channels-first N x 1 x 64 x 64.
        return self.model(x).reshape(x.shape[0], 1, 64, 64)


class CAE(nn.Module):
    """Port of CAE((64, 64, 1)); input/output use PyTorch NCHW layout."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            # Keras UpSampling2D defaults to nearest-neighbour interpolation.
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=3),
            # The original cae_out is BN only: no activation/sigmoid follows it.
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Chen2024Decoder(nn.Module):
    """End-to-end decoder returning (dense_out, cae_out)."""

    def __init__(self, ncell: int = 800):
        super().__init__()
        self.dense = DenseDecoder(ncell)
        self.cae = CAE()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dense_out = self.dense(x)
        cae_out = self.cae(dense_out)
        return dense_out, cae_out
