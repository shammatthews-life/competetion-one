# EXP85 Phase 3: Mismatch Diagnosis & Forensic Root-Cause Log

## Executive Methodology

Under strict anti-false-success protocols, this document logs every numerical divergence, feature rank disagreement, or stochastic training shift encountered during the autonomous source reconstruction.

---

## 1. Workstream 2: Feature Regeneration Diagnostics

### 669 Hasan Feature Reconstruction:
- **Hypothesis**: The complete 669 features can be deterministically reconstructed from raw `inpuy/train.csv` and `inpuy/test.csv` using the mathematical transformations in `build_hasan_600_features.py`.
- **Empirical Execution**: Executed `run_hasan_600_builder.py` generating `independent_rebuild/hasan_features_tr.feather` and `hasan_features_te.feather`.
- **Exhaustive Comparison**:
  * Shape: `(171,202, 669)` train, `(61,500, 669)` test. Exact shape match.
  * Column Order: 669 / 669 identical column names in identical sequence.
  * Value Equality: Checked across all 114,534,138 train cells and 41,143,500 test cells.
  * Discrepancies: **0 / 669 columns discrepant**.
  * File SHA-256 Hashes:
    - Train Feather SHA-256: `4050b9dace8ca8c40cde99f6b4b1fd62c8d2fdf03086961fc3ddaaf0658dd347` (Exact match with `cache/hasan_features_tr.feather`)
    - Test Feather SHA-256: `232cb5e5f5a30d3607b5acf6a5dedeec17f9cff18bc162d449305e7b82024beb` (Exact match with `cache/hasan_features_te.feather`)
- **Diagnosis**: **ZERO MISMATCH. Fully verified bitwise identical from raw data.**

---

## 2. Workstream 2 Step B: 550 Feature Screening Diagnostics

- **Hypothesis**: Running 5-fold Stratified LightGBM gain ranking on the regenerated 669-feature matrix will recover the exact 550 features selected into `hasan_selected_550_tr.feather`.
- **Sensitivity Factors**:
  * LightGBM multithreading (`n_jobs=4`) can produce non-deterministic floating-point reduction across parallel histogram bins.
  * Even minor floating-point differences around rank 545–555 could alter the cutoff boundary.
- **Verification Plan**:
  1. Compare feature importances against `cache/hasan_feature_importance.csv`.
  2. Compute Jaccard overlap between selected features and historical 550 features.
  3. Verify cell values for all common features.

---

## 3. Workstream 3: Model Zoo Training Diagnostics (Upcoming)

- **Models**:
  1. LightGBM Conservative (`lgb_cons`)
  2. LightGBM ExtraTrees (`lgb_et`)
  3. GPU XGBoost Hist (`xgb`)
  4. GPU CatBoost Symmetric (`cat`)
  5. LightGBM Deep (`lgb_deep`)
- **GPU Determinism Controls**:
  * CUDA histogram accumulation on NVIDIA RTX GPUs is subject to non-deterministic warp scheduling unless `deterministic=True` or single-threaded reductions are enforced.
  * Array correlation vs exact byte matching will be separately tracked and quantified.

