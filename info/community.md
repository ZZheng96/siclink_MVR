# Code, Models & Discussion Snapshot

- Competition: https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge
- Archived: 2026-11-23 (Asia/Shanghai)
- Scope: Publicly visible Code, Models and Discussion tabs at the time of capture. User-generated content can change; this is a snapshot, not a permanent official rulebook.

## Code Tab

Source: https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/code

Two public notebooks were visible:

1. [Spatio-Temporal Cortical Visual BCI Reconstruction](https://www.kaggle.com/code/avikdas567/spatio-temporal-cortical-visual-bci-reconstruction)
   - Updated 2 days before capture
   - Score: 7.52384
   - 0 comments
   - 9 votes
   - Bronze badge shown
2. [SiClink OCF Audit V1 Drift 14 Point Verifier](https://www.kaggle.com/code/moranzavdi/siclink-ocf-audit-v1-drift-14-point-verifier)
   - Updated 2 days before capture
   - Score: 12.94395
   - 0 comments
   - 2 votes

## Models Tab

Source: https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/models

No public model entries were visible when captured. The page showed the filters “All”, “Your Work”, “Shared With You”, and “Most Usage”.

## Discussion Tab

Source: https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/discussion

One public topic was visible when captured:

### OCF metric trap: raw ground-truth vs itself scores ~70.8, not 100.0 (anti-aliased vs >=128 binarization)

- Topic: https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/discussion/760906
- Author: Moran Zavdi (23rd in this Competition at capture time)
- Posted: 2 days before capture
- Votes shown: 2
- Comments: 1

While auditing the official `src.ocf.score` implementation locally, we found a subtle preprocessing asymmetry in Oriented Contour Fidelity (OCF) that silently costs ~29.2 OCF points—even if you predict the exact ground-truth image from `pictures.npz`.

#### Why OCF(gt_raw, gt_raw) ≈ 70.8 instead of 100.0

Inside the host scoring pipeline:

- Your submitted 64×64 mode-L PNG (uint8, 0..255) is scaled continuously to [0, 1] via `pred / 255.0` (it is **not** binarized).
- The ground-truth reference is strictly binarized at `gt >= 128` ({0, 1}).
- After rigid + smooth local alignment (±10°, ±2 px, warp ≤ 2 px), OCF computes multi-scale ($\sigma \in \{1,2,4\}$), 4-orientation ($0^\circ,45^\circ,90^\circ,135^\circ$) contour overlap over contour union:

$$
OCF(P,T)=100\times\frac{\sum_{\sigma,x}h_\sigma(x)}{\sum_{\sigma,x}\left(a_\sigma(x)+b_\sigma(x)-h_\sigma(x)\right)}
$$

Because the raw glyphs in `pictures.npz` contain anti-aliased gray edge pixels (1..254), passing a raw character image directly into `src.ocf.score` against itself produces soft multi-scale contour energy $a_\sigma(x)$ that mismatches the sharp binary step-edge of `(gt >= 128)`—inflating the denominator and capping the self-score at ~70.8 instead of 100.0!

#### Empirical OCF Calibration Table

| Prediction Regime (uint8 PNG) | Official OCF |
|---|---:|
| All-Black Placeholder (0) | 0.00 |
| Blurry Pixel-Mean Glyph (grayscale 0..255) | ~0.80 |
| Random Crisp Character from Bank ({0, 255}) | ~11.30 |
| Oracle Self — Raw Anti-Aliased (`pictures.npz`) | ~70.80 |
| Oracle Self — Crisp Binarized (`(gt >= 128) * 255`) | 100.00 |

Notice also why naive MSE / pixel-mean outputs collapse below ~1.0 OCF—worse than picking a random crisp character at ~11.3—any faint background haze inflates false contour energy $a_\sigma(x)$ across all three $\sigma$ scales.

#### Practical Takeaways

- **Binarize training targets & prototype banks:** Train pixel/generative heads against `(gt >= 128)` rather than raw anti-aliased grayscale, and ensure any gallery/prototype lookup emits crisp {0, 255} uint8 strokes.
- **Enforce PNG-parity in local CV:** Quantize validation predictions to uint8 [0..255] before calling `src.ocf.score` (and aggregate tasks equally via `(S1 + S2 + S3) / 3`, not sample-weighted) so local CV matches host scoring 1:1.

The author states that the runnable OCF calibration benchmark, mouse V1 latency/session-drift EDA, and a standalone 14-point `kaggle_submission.zip` offline verifier (to avoid using the 1/day UTC submission limit on PNG-vs-CSV or `._*` ZIP errors) are in the public notebook “SiClink Mouse V1 — OCF Metric Audit, V1 Drift & 14-Point Verifier”.

#### Comment by Yifei Jiang

Posted 12 hours before capture:

> Thanks for pointing out! The gray value may emerge from downsampling images to 64*64. GT are all in {0, 1} so submissions should be binarized first to get a higher score.

This comment is important because it explicitly confirms that binarizing submissions is recommended. It is preserved here as a discussion clarification; the formal rules and evaluation text remain in their respective documents.
