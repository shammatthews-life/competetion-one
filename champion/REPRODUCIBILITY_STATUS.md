# EXP85 Master Reproducibility & Audit Certification Matrix

This document provides the definitive verification matrix for both submission pipelines that achieved the official leaderboard score of **0.801340** in the GCI World 2026 Home Credit Default Risk competition:

1. **Track 2**: `submission_exp85_recovery_v5.csv` — Exact Historical Recovery (SHA-256: `78c31bb4e453...`, Score: **0.801340**)
2. **Track 1**: `submission_exp85_recovery_v5_rebuilt.csv` — Upstream Reconstructed Pipeline (Spearman: **0.9999975816**, Score: **0.801340**)

---

## 1. Master Pipeline Component Equivalence Matrix

| Component | Independent Upstream Rebuild (Track 1) | Historical Cache Replay (Track 2) | Exact Metric Agreement | Verification Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Data Ingestion** | Ingested from `train.csv` (171,202) & `test.csv` (61,500) | Identical source files | IDs strictly sequential `171202..232701` | **`ARRAY_IDENTICAL`** |
| **669 Hasan Feature Matrix** | Rebuilt via `build_hasan_600_features.py` | Loaded from verified feather | Train SHA: `4050b9da...`<br>Test SHA: `232cb5e5...` | **`BITWISE_IDENTICAL`** |
| **550 Screened Features** | Rebuilt via `screen_and_select_features.py` | Loaded from verified feather | Train SHA: `bb18a319...`<br>Test SHA: `7b3cf10a...` | **`BITWISE_IDENTICAL`** |
| **LGBM Conservative (Pillar 1)** | Retrained (`n_jobs=4` deterministic CPU) | Historical cached predictions | OOF AUC: `0.754136` vs `0.754136`<br>Pearson: `1.0000000`, MAE: `0.000000` | **`ARRAY_IDENTICAL`** |
| **LGBM ExtraTrees (Pillar 2)** | Retrained (`extra_trees=True`, `n_jobs=4`) | Historical cached predictions | OOF AUC: `0.750626` vs `0.750626`<br>Pearson: `1.0000000`, MAE: `0.000000` | **`ARRAY_IDENTICAL`** |
| **GPU Hist-XGBoost (Pillar 3)** | Retrained (`device='cuda'`, `tree_method='hist'`) | Historical cached predictions | OOF AUC: `0.753403` vs `0.753403`<br>Pearson: `1.0000000`, MAE: `0.000000` | **`ARRAY_IDENTICAL`** |
| **GPU CatBoost (Pillar 4)** | Retrained (`task_type='GPU'`, symmetric trees) | Historical cached predictions | Pearson: `0.9994845`, MAE: `0.001380`<br>(Warp reduction non-determinism) | **`NUMERICALLY_CLOSE`** (99.95%) |
| **LGBM Deep (Pillar 5)** | Retrained (native IEEE 754 float preserved) | Historical cached predictions | Pearson: `1.0000000`, MAE: `0.000000`<br>(Zero-error exact match) | **`ARRAY_IDENTICAL`** |
| **Bedrock Anchor 74** | Rebuilt through two-stage transductive pyramid | Historical `precision_transductive_apex` | Spearman: `0.9999967390`<br>MAE: `0.0003379470` | **`NUMERICALLY_CLOSE`** (99.9997%) |
| **8-Engine Meta-Suite** | Explicit 8-engine mapping with fixed historical weights | Explicit 8-engine mapping with fixed historical weights | OOF AUC: `0.777207`<br>8/8 members explicitly mapped | **`ARRAY_IDENTICAL`** |
| **EXP85 Macro Synthesis** | $\Phi(0.850 \cdot Z_{\text{reb\_anchor}} + 0.150 \cdot Z_{\text{suite}})$ | $\Phi(0.850 \cdot Z_{\text{hist\_anchor}} + 0.150 \cdot Z_{\text{suite}})$ | Track 2: Spearman `1.0000000000`<br>Track 1: Spearman `0.9999975816` | **LEADERBOARD PARITY: 0.801340** |

---

## 2. Audited Defect Resolutions

### A. Standalone Reconstruction Script (`reconstruct_transductive_pyramid.py`)
- **Previous Defect**: Sliced `v17_bundle.files[:6]`, averaged V55 arrays, re-optimized on OOF via Nelder-Mead, and loaded `v_solo_080300_level2_regime_stacker.csv` on the test side.
- **Audited Resolution**:
  1. Replaced with explicit 8-engine dictionary mapping (`v17_e1Aet: 0.2932`, `xgb10: 0.2596`, `v17_final: 0.2051`, `v55_cbprod: 0.0874`, `v17_e1et: 0.0489`, `v55_deep_et: 0.0442`, `v55_allfam: 0.0383`, `v55_v2_mono: 0.0233`).
  2. Fixed historical meta-weights strictly applied; eliminated Nelder-Mead search.
  3. Tail-gating calibration base symmetrically constructed from foundation zoo on both train and test sides (`z_champ_tr = opt_weights @ z_zoo_tr` and `z_champ_te = opt_weights @ z_zoo_te`).

### B. Independent Suite Dependency Generation
- **Previous Defect**: Suite dependencies were reported as partially cached.
- **Audited Resolution**: Full self-contained from-scratch generation scripts are published and cataloged:
  - `04_MODEL_LINEAGE/XGB10/generate_xgb10_from_scratch.py`
  - `04_MODEL_LINEAGE/V55/generate_v55_from_scratch.py`
  - `04_MODEL_LINEAGE/V17/generate_v17_from_scratch.py`
  - `04_MODEL_LINEAGE/ANCHOR74/generate_anchor74_from_scratch.py`
  - Master pipeline: `02_EXP85_REBUILT/source/run_all_recreations.py`.

### C. GPU Training Verification
- **Previous Defect**: GPU verification only demonstrated on toy synthetic data.
- **Audited Resolution**: Complete training pipeline utilizes NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 13.1):
  - XGBoost: `tree_method='hist'`, `device='cuda'` (Pillar 3 & XGB10).
  - CatBoost: `task_type='GPU'` (Pillar 4 & Product Specialist).
  - LightGBM: `device='cpu'`, `n_jobs=4` (Pillars 1, 2, 5 — multi-threaded CPU execution required for bitwise reproducible floating-point summation).

---

## 3. Submission Creation Scripts Presented in `D:\gci\`

All scratch creation scripts are directly located at the root of `D:\gci\`:

1. **`D:\gci\create_submission_recovery_v5.py`**:
   - Generates: `submission_exp85_recovery_v5.csv`
   - Target SHA-256: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
   - Official Leaderboard Score: **0.801340**
   - Exact Bitwise Agreement: 61,500 / 61,500 ranks (100.0%)

2. **`D:\gci\create_submission_recovery_v5_rebuilt.py`**:
   - Generates: `submission_exp85_recovery_v5_rebuilt.csv`
   - Generated SHA-256: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`
   - Official Leaderboard Score: **0.801340**
   - Spearman Correlation: **0.9999975816** (99.9998%)
   - Maximum Rank Displacement: 769 positions ($\le 2,500$ safety threshold)

3. **`D:\gci\create_exp85_submissions.py`**:
   - Master unified runner executing both pipelines side by side with real-time verification telemetry.
