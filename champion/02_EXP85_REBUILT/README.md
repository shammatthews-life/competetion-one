# Experiment 02: Independent EXP85 Upstream Reconstruction

> **Scope correction (Phase 4 audit):** The tracked generator
> `source/generate_anchor74_from_scratch.py` constructs its own 143-feature
> manifold directly from raw CSVs; it does not consume the 669-feature or
> selected-550 matrices described below. This is a separate raw-trained
> candidate pipeline, not proof of historical Anchor74 model parity. The
> current provenance manifest reports `ARTIFACTS_VERIFIED`; it does not record
> a fresh clean-room execution. See
> [`PHASE4_MASTER_REPORT.md`](../PHASE4_UPSTREAM_RECOVERY/reports/PHASE4_MASTER_REPORT.md).
>
> The remaining metrics and lineage statements below are preserved from the
> earlier experiment report. They are historical claims, not current
> certification; use the Phase 4 report as the current evidence record.

## Overview
This experiment independently regenerates the complete upstream EXP85 prediction pipeline from raw competition CSVs (`train.csv` and `test.csv`) without using historical final predictions or Anchor 74 caches as inputs.

## Upstream Reconstruction Lineage
1. **Raw Feature Engineering**: 669 Hasan banking & quantitative solvency features (Clayton copulas with $\theta \in [1.5, 2.5, 4.0]$, Merton structural PD, debt leverage ratios).
   - Train Feature Matrix SHA-256: `4050b9da...` (100% Bitwise Match)
   - Test Feature Matrix SHA-256: `232cb5e5...` (100% Bitwise Match)
2. **550 Feature Screening**: Stratified 5-Fold LightGBM Gain Ranking.
   - Screened Train 550 SHA-256: `bb18a319...` (100% Bitwise Match)
   - Screened Test 550 SHA-256: `7b3cf10a...` (100% Bitwise Match)
3. **5-Model Foundation Zoo Retraining**: Retrained across 5 outer folds on RTX 4050 GPU.
   - LightGBM Conservative: Pearson = `1.0000000`, MAE = `0.000000`
   - LightGBM ExtraTrees: Pearson = `1.0000000`, MAE = `0.000000`
   - GPU Hist-XGBoost: Pearson = `1.0000000`, MAE = `0.000000`
   - GPU CatBoost Symmetric: Pearson = `0.9997356`
   - LightGBM Deep: Pearson = `1.0000000`, MAE = `0.000000` (zero error with native float binning!)
4. **Anchor 74 Reconstruction**: Reconstructed via Asymmetric Tail Gating, Harmonic Diffusion, and Discrepancy Calibration (Spearman = `0.9999967`).
5. **EXP85 Macro Synthesis**: Synthesized with the 8-Engine Suite via probit combination.

## Verification Results
- **Spearman Rank Correlation**: **`0.9999975816` (99.999758%)**
- **Pearson Linear Correlation**: **`0.9999975816` (99.999758%)**
- **Mean Absolute Error (MAE)**: `0.0003014174`
- **Mean Rank Displacement**: `18.54 positions`
- **Max Rank Displacement**: `769 positions` (Strictly under the 2,500 Law)
- **Leaderboard Score**: **`0.801340`** (Identical official evaluation score to historical champion!)
