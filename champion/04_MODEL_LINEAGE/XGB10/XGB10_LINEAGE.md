# XGB10 Model Lineage & Dual-Model Distinction

This document clarifies the exact architecture, feature space, and validation metrics of XGBoost in the EXP85 champion pipeline, resolving earlier documentation ambiguities regarding reported OOF AUC values.

---

## 1. Dual-Model Architectural Distinction

Two distinct XGBoost models operate within the EXP85 hierarchy:

| Dimension | Model A: 8-Engine Suite Member (`xgb10`) | Model B: 5-Model Foundation Zoo (`test_xgb`) |
| :--- | :--- | :--- |
| **Pipeline Level** | Level 1: 8-Engine Meta-Suite Member | Foundation Level: Bedrock Anchor 74 Pillar 3 |
| **Target Artifacts** | `oof_10f_xgb.npy`, `test_10f_xgb.npy` | `hasan_expanded_zoo.npz['test_xgb']` |
| **Feature Space** | **239 Features** (Multi-view financial manifold) | **550 Screened Features** (LightGBM gain screening) |
| **Cross-Validation** | **10-Fold Stratified Cross-Validation** | **5-Fold Stratified Cross-Validation** |
| **OOF ROC-AUC** | **`0.774331`** | **`0.753403`** |
| **Role in EXP85** | Suite Member 2 (**Weight: 25.96%**) | Pillar 3 in Anchor 74 Probit Blend (**Weight: 24.0%**) |
| **Source Script** | `generate_xgb10_from_scratch.py` | `train_independent_zoo.py` |

> [!NOTE]
> Historical documentation reported `0.774331` for the 10-fold standalone suite model and `0.753403` for the 5-fold 550-feature zoo model. Both numbers are mathematically correct and correspond to these two distinct models.

---

## 2. 239-Feature Manifold Specification

The standalone 10-fold XGBoost model uses a specialized 239-feature manifold constructed in `build_xgb_features`:
1. **Primary Financial Ratios**: Debt-to-income, annuity-to-credit, payment surplus, goods price differentials.
2. **External Source Interactions**: Nan-safe statistical moments, cross-source copula products, differences, and weighted composites.
3. **Age & Tenure Geometry**: Days birth, days registration, days ID publish, phone change intervals, and tenure ratios.
4. **Solvency Indicators**: Disposable income, living cost estimates, credit term multipliers.
5. **Categorical Encodings**: Normalized frequency encodings and integer factorizations across all 8 demographic categorical variables.

---

## 3. Hardware Acceleration & CUDA Verification

- **Algorithm**: `tree_method='hist'`
- **Device**: `device='cuda'` (NVIDIA GeForce RTX 4050 Laptop GPU, CUDA 13.1)
- **Fallback**: Automatic deterministic CPU fallback if CUDA runtime is unavailable.
- **Precision**: 32-bit floating point array export (`float32`).
