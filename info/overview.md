# SiClink Mouse Visual Reconstruction BCI Competition — Overview

- Source: https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/overview
- Archived: 2026-11-23 (Asia/Shanghai)
- Host: [Global SiClink](https://www.kaggle.com/globalsiclink)
- Competition type: Community Prediction Competition
- Tag: Custom Metric
- Points/medals: Does not award Points or Medals

> This document preserves the substantive English text and tables from the Overview tab. Relative dates and participation counts are snapshots and may change.

## Overview

Vision is a primary channel through which biological organisms perceive the world, and decoding visual neural activity is a critical step toward bridging biological perception and silicon-based intelligence. Can we reconstruct the exact visual structure of what an animal sees directly from its brain activity?

In this challenge, your goal is to reconstruct 64 × 64 grayscale images of Chinese characters (white character on a black background) presented to a mouse from its visual cortex spiking activity (250 ms recording window per presentation).

Unlike natural scenes where coarse textures can often mask local reconstruction errors, Chinese characters are composed of structured strokes and shared radicals, where distinct characters may differ by only one or two local strokes. This makes them an ideal benchmark for testing whether a neural decoder can recover both global spatial layout and fine-grained local structure—laying the groundwork for high-precision visual brain-computer interfaces (BCIs).

## Description

### The Three-Task Generalization Protocol

A practical visual BCI must generalize beyond the exact stimuli, recording days, and trial-averaging conditions used during calibration. To systematically evaluate your model across three key axes of variation, every submission is tested on three complementary tasks:

| Task | Generalization Axis | Characters | Session | Trials per Sample | Test Samples | Weight |
|---|---|---|---|---|---:|---:|
| Task 1 | Zero-Shot Visual Generalization | Unseen during training | Same session as training (session1–4) | 20 repeated trials | 2,720 | 1/3 (33.3%) |
| Task 2 | Cross-Day Neural Transfer | Seen during training | New recording day (session5) | 4 repeated trials | 5,496 | 1/3 (33.3%) |
| Task 3 | Single-Trial Reliability | Seen during training | Same session as training (session1–4) | 1 single trial | 21,984 | 1/3 (33.3%) |

**Task 1 (Within-Session, Unseen Characters — 1/3 weight):** Reconstruct a novel Chinese character that never appeared in the training set, using 20 repeated presentations recorded within the same session (session1–session4). This tests whether your model learns a composable mapping from neural activity to visual strokes rather than memorizing a fixed dictionary of characters.

**Task 2 (Cross-Day, Seen Characters — 1/3 weight):** Reconstruct a familiar character from the training set using 4 repeated presentations recorded on a different day (session5) from the same mouse. Because neural responses and baseline firing properties drift across days, this task evaluates how well your model extracts stable, day-invariant visual representations.

**Task 3 (Within-Session, Single-Trial Seen Characters — 1/3 weight):** Reconstruct a familiar character from a single 250 ms trial held out from the training sessions (session1–session4). While averaging across repeats boosts signal-to-noise ratio, real-time BCIs must operate one presentation at a time.

Generate exactly one 64 × 64 grayscale image per `sample_id`, not per trial. In Tasks 1 and 2, you may use any subset or combination of the repeated trials within that sample.

## Evaluation

### How Scoring Works

Each reconstructed 64 × 64 single-channel grayscale PNG (`<sample_id>.png`, white character on a black background) is paired with its hidden ground-truth character image and evaluated using OCF (Oriented Contour Fidelity), a multi-scale structural metric tailored for stroke and contour reconstruction. OCF evaluates how accurately your reconstructed image captures the stroke orientations and multi-scale contour energy of the target character on a 0 to 100 scale (higher is better):

1. **Preprocessing:** Your submitted PNG (uint8, 0–255) is normalized to [0, 1] by dividing by 255. The ground-truth character image is binarized at threshold 128 (1 for foreground strokes, 0 for background).
2. **Small-Shift & Deformation Alignment:** Before comparing contours, the reference character is aligned to your prediction allowing minor rigid adjustments (rotations within ±10°, translations within ±2 pixels) and a smooth local displacement (up to 2 pixels). This ensures your model is judged on structural legibility rather than sub-pixel registration offsets.
3. **Multi-Scale Directional Contour Comparison:** Across three spatial scales (σ ∈ {1, 2, 4} pixels) and four stroke orientations (0°, 45°, 90°, 135°), OCF extracts local directional contour energy and measures the directional overlap energy hσ(x) between your prediction (aσ(x)) and the aligned reference (bσ(x)).
4. **Per-Image Score (0–100):** The per-image OCF score is the ratio of total overlapping contour energy to total union contour energy across all scales and pixels:

$$
OCF(P,T)=100\times\frac{\sum_{\sigma,x}h_\sigma(x)}{\sum_{\sigma,x}\left(a_\sigma(x)+b_\sigma(x)-h_\sigma(x)\right)}
$$

An exact match to the binarized reference scores 100, while a blank (all-zero) prediction against a character scores 0. The complete, exact CPU (`src/ocf/ocf_native.py`) and GPU (`src/ocf/ocf_cuda.py`) implementations are provided in the participant package so you can compute official OCF scores locally during validation.

### Task & Final Score Aggregation

For each task $j \in \{1,2,3\}$ containing $N_j$ test samples ($N_1=2,720$, $N_2=5,496$, $N_3=21,984$), the task score $S_j$ is the arithmetic mean of the per-image OCF scores $s_{j,i}$:

$$
S_j=\frac{1}{N_j}\sum_{i=1}^{N_j}s_{j,i}
$$

The final competition score is the equal-weighted average of the three task scores:

$$
S_{final}=\frac{S_1+S_2+S_3}{3}
$$

### Submission Format

First generate the PNG archive `submission.zip` using the participant package, containing the three task directories (`task1/`, `task2/`, `task3/`) at its root, with one 64 × 64 single-channel grayscale PNG (mode="L") named `<sample_id>.png` for every test sample:

```text
submission.zip
├── task1/                  # 2,720 grayscale PNGs (64 x 64)
│   ├── test_task1_pic0.png
│   ├── ...
│   └── test_task1_pic2719.png
├── task2/                  # 5,496 grayscale PNGs (64 x 64)
│   ├── test_task2_pic0.png
│   ├── ...
│   └── test_task2_pic5495.png
└── task3/                  # 21,984 grayscale PNGs (64 x 64)
    ├── test_task3_pic0.png
    ├── ...
    └── test_task3_pic21983.png
```

**Kaggle upload format:** Convert this PNG archive using `convert_to_kaggle.py` from `kaggle_submission_adapter/kaggle_submission_adapter/` in the [Data files](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data):

```bash
python -m pip install Pillow
python convert_to_kaggle.py submission.zip kaggle_submission.zip
```

Upload `kaggle_submission.zip`, **not** the original PNG archive. It contains one `submission.csv` with 30,200 rows and exactly two columns:

| Column | Content |
|---|---|
| `sample_id` | The exact test sample ID, such as `test_task1_pic0`. |
| `prediction` | Standard base64 encoding of the sample's 64 × 64, 8-bit grayscale PNG. |

The converter validates the IDs and image formats and preserves every pixel value. Only the submission container changes; OCF scoring and the equal task weights are unchanged. A format example with placeholder black images is supplied as `sample_submission.csv`.

All 30,200 samples contribute to the public leaderboard; there is no separate held-out private test set. Public leaderboard scores and final prize eligibility remain subject to the review and verification requirements in the [Competition Rules](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/rules).

## Prizes

The total cash prize pool is USD 10,000 (gross, before tax), allocated to the top 10 eligible teams as follows:

| Final Rank | Prize per Team (USD, Before Tax) |
|---|---:|
| 1st | $4,000 |
| 2nd | $2,000 |
| 3rd | $1,500 |
| 4th | $1,000 |
| 5th | $500 |
| 6th–10th | $200 each |

Prizes are awarded per team, not per team member. For details on prize eligibility, acceptance requirements, and distribution, please refer to the prize provisions in the Competition Rules.

## Community & Support

Join the SiClink Neural Decoding Challenge | Community Telegram group to meet other participants, find teammates, and ask getting-started questions. English and Chinese are both welcome. Joining is optional and does not affect participation or prize eligibility.

For questions about data, scoring, or competition rules, please post in Kaggle Discussions so everyone can see the answer. Official announcements and clarifications will be posted on Kaggle.

Please do not upload competition data, prediction files, or unpublished solution code to Telegram. Do not send identity documents, tax forms, or payment details in the group or through unsolicited direct messages.

- [Join the Telegram community](https://t.me/+JZQP16EDuE01MjY1)
- [Kaggle Discussions](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/discussion)

## Timeline

- **Start:** The page displayed “3 days ago”; its tooltip was `Sat Nov 21 2026 01:32:07 GMT+0800 (China Standard Time)`, equivalent to 2026-11-20 17:32:07 UTC.
- **Final submission deadline:** January 20, 2027 at 10:00 UTC. The page tooltip was `Wed Jan 20 2027 18:00:00 GMT+0800 (China Standard Time)`.

## Citation

SiClink. SiClink Mouse Visual Reconstruction BCI Competition. https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge, 2026. Kaggle.

## Participation Snapshot

The following figures were visible when this page was archived on 2026-11-23 and will change over time:

- 133 Entrants
- 44 Participants
- 44 Teams
- 88 Submissions
