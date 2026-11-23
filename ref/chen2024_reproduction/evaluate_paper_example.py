"""Reproduce the reported reconstruction metrics on the authors' example output.

2026-11-24移植/修改自beizai/Decoding（test.py）。
Paper: https://doi.org/10.1371/journal.pcbi.1012297
Source: https://github.com/beizai/Decoding
Source commit: 631ebbcec0eedb907dd1ec24a908c4f99324c219
License: GPL-3.0; see LICENSE.

This only scores the authors' published prediction file against the authors'
published test images. It does not train a model and is not a competition score.
"""

import json
from pathlib import Path

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "downloads" / "chen2024_training"
RESULT = ROOT / "downloads" / "chen2024_results"

ground_truth = np.load(DATA / "movie_03_test_pic_1999_VISp_800.npy")
prediction = np.load(RESULT / "movie_03_iter_1999_VISp_800_400.npy")

assert ground_truth.shape == (400, 256, 256)
assert prediction.shape == (400, 256, 256, 1)

# test.py evaluates frames 5..399 (395 frames), skipping the first five.
frames = range(5, 400)
ssim_values, psnr_values, mse_values = [], [], []
for index in frames:
    reference = ground_truth[index]
    estimate = prediction[index, :, :, 0]
    # test.py used multichannel=True on a single channel; 2-D SSIM is equivalent.
    ssim_values.append(structural_similarity(reference, estimate, data_range=1.0))
    psnr_values.append(peak_signal_noise_ratio(reference, estimate, data_range=1.0))
    mse_values.append(mean_squared_error(reference, estimate))

summary = {
    "scored_frames": len(list(frames)),
    "ssim_mean": float(np.mean(ssim_values)),
    "psnr_mean_db": float(np.mean(psnr_values)),
    "mse_mean": float(np.mean(mse_values)),
    "ssim_median": float(np.median(ssim_values)),
    "psnr_median_db": float(np.median(psnr_values)),
    "mse_median": float(np.median(mse_values)),
    "ground_truth_pixel_range": [float(ground_truth.min()), float(ground_truth.max())],
    "prediction_pixel_range": [float(prediction.min()), float(prediction.max())],
    "note": "Authors' published prediction vs published test images; not a competition OCF score.",
}
print(json.dumps(summary, indent=2))
