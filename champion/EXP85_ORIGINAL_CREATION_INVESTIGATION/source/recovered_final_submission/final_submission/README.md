# GCI Global 2026: Home Credit Default Risk (Assignment 1038)
## Solution Documentation & Reproduction Guide

### 1. Performance Overview
- **Omnicampus Public Leaderboard**: **`0.80041` ROC-AUC**
- **Out-of-Fold Cross-Validation**: **`0.802330` ROC-AUC**

---

### 2. File Verification & Cryptographic Hashes (SHA-256)
- **Final Submission File**: `submission_final.csv`
  - SHA-256: `8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b`
  - Shape: 61,500 test predictions + 1 header line (`SK_ID_CURR,TARGET`)
- **Standalone Reproduction Script**: `reproduce_final_submission.py`

---

### 3. Pipeline Architecture
The solution employs a four-tiered hierarchical synthesis:
1. **Diverse Base Streams (16 Unbundled Models & 9 Macro Pillars)**:
   - Neural Dual-Manifolds (`TabM-32` and `ResNet` on 646 features, evaluated in both Logit and Probit spaces).
   - 9 GBDT Families (`LightGBM`, `XGBoost`, `CatBoost` across depth-4, depth-6, conservative, and deep architectures on 648 and 595 Buhlmann feature sets).
   - Hybrid DAE and Relational Ensembles (`LGBM-DAE-491`, `TabM-463`, `Heterogeneous-Zoo`, `Buhlmann-Tier2`).
2. **Quantile-Matched Empirical Log-Odds**:
   - Monotonically maps out-of-fold uniform ranks into empirical log-odds distributions calibrated to the test probabilities.
3. **Leak-Free Relational Entity Graph Diffusion**:
   - Strict target-free entity graph constructed via applicant entity matching (`DAYS_BIRTH`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `CODE_GENDER`, `NAME_EDUCATION_TYPE`).
   - Propagates target-free risk differentials (`z_target_free_all`) across confirmed borrower links without target-encoding leakage.
   - Applies transductive partner coupling across 320 test-to-test confirmed repeat-borrower pairs.
4. **Non-Linear Level-2 Meta-Stacker**:
   - Out-of-fold multi-seed L2 Logistic Regression + shallow residual LightGBM utilizing continuous `EXT_SOURCE` missingness regime interactions (`disagreement_neural_gbdt * ext_missing`) and inter-model dispersion.

---

### 4. Reproduction Instructions
To reproduce `submission_final.csv`:
1. Ensure the feature matrices and fold predictions are present in the `output/` directory.
2. Run:
   ```bash
   python reproduce_final_submission.py
   ```
3. The script will output `submission_final.csv` matching SHA-256 `8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b`.
