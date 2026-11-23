# Data — Dataset Description and Official Instructions

- Source: https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data
- Archived: 2026-11-23 (Asia/Shanghai)
- Files shown by Kaggle: 87
- Total size shown by Kaggle: 249.66 MB
- Types shown by Kaggle: `npz`, `py`, `md` + 3 others
- License: Subject to Competition Rules

## Experimental Setup & Dataset Summary

Neural spiking activity was recorded from the visual cortex of 2 mice (`mice1` and `mice2`) across 10 recording sessions (5 sessions per mouse) while the animals viewed 64 × 64 grayscale Chinese character images:

- **Stimulus Presentation:** Each character image was displayed for 250 ms per trial. Neural spike times are aligned to stimulus onset (0 to 250 ms, recorded at 30,000 Hz).
- **Recorded Neural Units:** `mice1` has 134 recorded units and `mice2` has 78 recorded units. Unit indices are consistent across all sessions for the same mouse.
- **Session & Stimulus Split:**
  - In each session, the mouse viewed 2,748 seen characters (5 repeats each) and 340 unseen characters (20 repeats each), totaling 20,540 trials per session.
  - `session1` to `session4` (4 sessions per mouse): Used for the shared training set (`train/`, using 4 of the 5 repeats per seen character), Task 1 (`test/task1/`, the 20 repeats of the 340 unseen characters), and Task 3 (`test/task3/`, the remaining 5th repeat of the 2,748 seen characters).
  - `session5` (1 held-out cross-day session per mouse): Recorded on a different day and reserved exclusively for Task 2 (`test/task2/`, 4 repeats for each of the 2,748 seen characters).

| Dataset Split | Directory / Path | Sessions Covered | Characters per Session | Trials per Sample | Total Samples (Images) | Total Trials |
|---|---|---|---|---|---:|---:|
| Shared Training Set | `train/` | mice1 & mice2, session1–4 | 2,748 seen | 1 trial per sample (4 repeats stored as 4 samples) | 87,936 | 87,936 |
| Task 1 Test (Unseen Characters) | `test/task1/` | mice1 & mice2, session1–4 | 340 unseen | 20 trials grouped per sample | 2,720 | 54,400 |
| Task 2 Test (Cross-Day Seen) | `test/task2/` | mice1 & mice2, session5 | 2,748 seen | 4 trials grouped per sample | 5,496 | 21,984 |
| Task 3 Test (Single-Trial Seen) | `test/task3/` | mice1 & mice2, session1–4 | 2,748 seen | 1 trial per sample | 21,984 | 21,984 |
| Total Test Set | `test/` | All 10 sessions | — | — | 30,200 | 98,368 |

## Directory Structure

The Data files contain the following participant resources:

- **`participant_package_en.zip` (Code & Demo Package):** Contains the official data loader (`src/dataloader.py`), local OCF scoring module (`src/ocf/`), submission packager (`src/submission.py`), a runnable conditional DDPM baseline (`example/`), and a tiny demo dataset (`data/`) for quick pipeline testing.
- **`siclink_chinese_character_decoding_dataset/` (Official Full Dataset):** Contains the full training and test recordings across all 10 sessions.
- **`_/` (Chinese Participant Guide):** Chinese-language participant materials.
- **`kaggle_submission_adapter/kaggle_submission_adapter/`:** Contains `convert_to_kaggle.py`, its README, and a 30,200-row `sample_submission.csv` with placeholder black images. No test answers are included.

In the examples below, `official_data/` is a local alias for the full dataset directory containing `catalog.json`; it is not an additional download. The English starter code is under `participant_package_en/participant_package_en/` in Kaggle Data.

```text
participant_package_en/           # 1. Starter Code & Demo Package
├── README.md                     # Detailed guide and code walkthrough
├── BACKGROUND_AND_TASKS.md       # Scientific background and task definitions
├── SCORING.md                    # OCF scoring formula and local evaluation guide
├── data/                         # Tiny demo subset (8 train trials + 2 test samples/task)
├── src/
│   ├── dataloader.py             # Official PyTorch Dataset loader (load_dataset)
│   ├── spike_counts.py           # Optional time-binning helper (count_spikes)
│   ├── submission.py             # Verifies and packages PNGs into submission.zip
│   └── ocf/                      # Official OCF metric (CPU & CUDA implementations)
└── example/                      # Conditional DDPM baseline (train.py, predict.py, model.py)

official_data/                    # 2. Official Full Dataset (Extract to a separate directory)
├── catalog.json                  # Master metadata index (window_ms=250, image_size=[64, 64])
├── train/                        # Shared training set (87,936 samples)
│   ├── pictures.npz              # 2,748 ground-truth 64x64 uint8 character images
│   ├── mice1/
│   │   └── session1.npz ... session4.npz
│   └── mice2/
│       └── session1.npz ... session4.npz
└── test/                         # Unlabelled test sets (30,200 samples / 98,368 trials)
    ├── task1/                    # Within-session unseen characters (2,720 samples)
    │   ├── mice1/ (session1.npz ... session4.npz)
    │   └── mice2/ (session1.npz ... session4.npz)
    ├── task2/                    # Cross-day seen characters (5,496 samples)
    │   ├── mice1/ (session5.npz)
    │   └── mice2/ (session5.npz)
    └── task3/                    # Within-session single-trial seen characters (21,984 samples)
        ├── mice1/ (session1.npz ... session4.npz)
        └── mice2/ (session1.npz ... session4.npz)
```

## Loading Data with the Official Data Loader

You do not need to manually parse the raw `.npz` files. Simply pass the directory containing `catalog.json` to `load_dataset()` from `src.dataloader`. It automatically converts raw spike events into clean binary spike matrices of shape `(n_trials, n_units, n_time_bins)`:

- **Rows (`n_units`):** Recorded neural units for that mouse (134 for `mice1`, 78 for `mice2`).
- **Columns (`n_time_bins`):** Time bins across the 250 ms presentation window. With the default `bin_ms=1` (1 ms per bin), each trial has 250 columns. An entry is 1 if the unit fired at least once in that time bin, and 0 otherwise.

```python
import torch
from src.dataloader import load_dataset

# Point to the folder containing catalog.json (use "data" for the bundled demo)
data_dir = "/path/to/official_data"

# 1. Load the shared training set (87,936 samples)
train_set = load_dataset(data_dir, split="train", bin_ms=1)
train_sample = train_set[0]
print(train_sample["sample_id"])       # e.g., 'train_pic0'
print(train_sample["trials"].shape)    # (1, 134, 250) -> (n_trials, n_units, n_time_bins)
print(train_sample["image"].shape)     # (64, 64) uint8 ground-truth character image (0..255)

# 2. Load test samples for Task 1, Task 2, or Task 3
for task in ("task1", "task2", "task3"):
    test_set = load_dataset(data_dir, split="test", task=task, bin_ms=1)
    sample = test_set[0]
    print(f"{task}: {len(test_set)} samples | first ID={sample['sample_id']} | trials shape={sample['trials'].shape}")
```

### What Each Sample Dictionary Contains

| Key | Type | Description |
|---|---|---|
| `sample_id` | str | Unique sample ID (e.g., `train_pic0`, `test_task1_pic0`). Use `<sample_id>.png` as your output filename for test predictions. |
| `mouse` | str | Subject identifier (`mice1` or `mice2`). |
| `session` | str | Recording session identifier (`session1` through `session5`). |
| `n_units` | int | Number of neural units for this mouse (134 for `mice1`, 78 for `mice2`). |
| `trials` | `np.ndarray` (uint8) | Binary spike array of shape `(n_trials, n_units, n_time_bins)`. `n_trials` is 1 for train and task3, 20 for task1, and 4 for task2. |
| `image` | `np.ndarray` (uint8) | (Training set only) Ground-truth 64 × 64 grayscale character image with pixel values in 0–255. |

## Quickstart: Train the DDPM Baseline, Score Locally & Build `submission.zip`

From inside `participant_package_en/`, you can verify your environment on the bundled demo data (`data/`) in seconds, and then run on the full official dataset:

```bash
# 1. Install dependencies
python -m pip install -r example/requirements.txt

# 2. Quick test on the bundled demo data (1 epoch)
python -m example.train --data data --output runs_demo --epochs 1
python -m example.predict --data data --checkpoints runs_demo --output submission_demo

# 3. Train and generate submission.zip on the full official dataset
python -m example.train --data "/path/to/official_data" --output runs_full --epochs 10
python -m example.predict --data "/path/to/official_data" --checkpoints runs_full --output submission
```

To evaluate your predictions locally on a validation split using the official OCF metric (supports both CPU and `"cuda:0"`):

```python
import numpy as np
from src.ocf.score import score, score_png

# Score a batch of float predictions in [0, 1] against binary targets (reference >= 128)
# Shape can be (64, 64), (N, 64, 64), or (N, 1, 64, 64)
batch_scores = score(predictions_float01, reference_uint8 >= 128, device="cpu")
print("Mean OCF score:", batch_scores.mean())

# Or score a saved 64x64 grayscale PNG file directly against a 64x64 uint8 reference image
single_score = score_png("validation_pred.png", reference_uint8, device="cpu")
```

### Before Converting the PNG Package, Verify That

- `submission.zip` contains `task1/` (2,720 PNGs), `task2/` (5,496 PNGs), and `task3/` (21,984 PNGs) directly at its root (30,200 PNG files in total).
- Every image is a 64 × 64 single-channel grayscale PNG (mode="L", white character on a black background) named `<sample_id>.png`.

Note: `src/submission.py` (called automatically by `example/predict.py`) validates all sample IDs, image dimensions, and color modes before writing `submission.zip`.

## Submit to Kaggle

The participant package generates the PNG archive `submission.zip`. Do not upload that PNG archive directly to Kaggle. Convert it with the adapter supplied in Data:

```bash
python -m pip install Pillow
python /path/to/kaggle_submission_adapter/kaggle_submission_adapter/convert_to_kaggle.py submission.zip kaggle_submission.zip
```

Upload `kaggle_submission.zip`, which contains one `submission.csv` with exactly 30,200 rows and two columns: `sample_id,prediction`. Each prediction is a base64-encoded 64 × 64 grayscale PNG. The converter preserves every pixel value; it only changes the container to the format accepted by Kaggle.

The sample submission contains placeholder black images, not a model baseline or ground truth. All 30,200 samples are scored on the public leaderboard using the equal mean of the three task-level OCF means. Scores and prize eligibility remain subject to the review and verification requirements in the Competition Rules.
