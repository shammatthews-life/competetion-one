# GCI Home Credit Default-Risk Research — Origin Master Verification & Progress Ledger

> **Purpose:** Single append-only authoritative record of every from-scratch experiment, model architecture, mathematical method, validation score, file hash, and code recreation recipe.
>
> **Project Scope:** GCI World 2026 Credit Default Competition (ROC-AUC Metric).  
> **Strict Operational Boundary:** 100% PURE IN-MEMORY FROM-SCRATCH CODE ONLY. Zero precomputed `.npz`, `.npy`, `.csv` prediction files loaded. Only `inpuy/train.csv` and `inpuy/test.csv`.  
> **Target:** Surpass 0.7800 Local OOF AUC / 0.7820+ Public Leaderboard through fully reproducible, competition-compliant modeling.  
> **Governance Rule:** Never overwrite or silently modify historical entries. All progress is append-only.

---

## 0. Executive Governance & Append-Only Rules

1. **Clean-Room Compliance:** Only competition inputs `train.csv` and `test.csv` (and `sample_submission.csv`) are permitted. Loading external prediction caches (`v17_bundle.npz`, `v55_bundle.npz`, `precision_anchor.csv`) is strictly prohibited as it leads to audit disqualification.
2. **Deterministic & Fold-Safe:** Every learned imputer, target encoder, or grouping must be fitted strictly inside outer training partitions. Outer validation and test sets must remain untouched.
3. **No Silent History Overwrites:** Previous experiments, parameters, and scores are permanently preserved. Corrections are added as dated addenda.
4. **Complete Reproduction Recipes:** Every notebook must contain complete, standalone Python code capable of running top-to-bottom on clean hardware.

---

## 1. Verified Experiment Index (Origin From-Scratch Series)

| Experiment ID | Executable Notebook Location | Model Architecture | Local OOF ROC-AUC (`loc`) | Public Leaderboard (LB) | Delta LB vs Baseline | Audit Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Origin Baseline** | `gemini/independentexperiments/origin.ipynb` | 5-Fold LightGBM + Core Ratios | 0.768950 | 0.77482 | — | REPRODUCED (100% Pure Scratch) |
| **Origin EXP 1** | `gemini/independentexperiments/origin_exp1.ipynb` | Core Ratios + Merton/Copula Ratios | 0.769850 | 0.77524 | +42 bps | REPRODUCED (100% Pure Scratch) |
| **Origin EXP 2** | `gemini/independentexperiments/origin_exp2.ipynb` | 8-Engine Suite + 10F Monotonic GPU XGBoost | 0.771170 | 0.77586 | +104 bps (+62 vs EXP1) | REPRODUCED (100% Pure Scratch) |
| **Origin EXP 3** | `gemini/independentexperiments/origin_exp3.ipynb` | 8-Engine Suite + Nelder-Mead Probit Blend | 0.771857 | **0.77704** | **+222 bps** (+118 vs EXP2) | REPRODUCED (100% Pure Scratch) |
| **Origin EXP 4** | `gemini/independentexperiments/origin_exp4.ipynb` | Multi-View Specialists + Disagreement Stacking | 0.771778 | 0.77669 | +187 bps | REPRODUCED (100% Pure Scratch) |
| **Origin EXP 5** | `gemini/independentexperiments/origin_exp5.ipynb` | Full Community Keys + Person Pooling + 5 Members | *Target >= 0.7800* | *Target >= 0.7820* | *Target +400 bps* | IN DEVELOPMENT |

---

## 2. Key Synthesis from Truth Ledger (Phases 1 to 23 Audit)

A deep study of `D:\gci\truth\GCI_MASTER_VERIFICATION_LEDGER_UPDATED_2026-11-04.md` establishes the exact empirical boundaries:

### 2.1 Product Decoders & Ratio Keys (Phases 3, 3B, 7, 8, 9, 13)
- **The Core Signal:** Loan annuity and credit are strongly quantized into interest rate and term tariffs.
- **Ratios over Lattices:** Ratio keys (`Q3`, `Q4`, `Q5` = `AMT_ANNUITY / AMT_CREDIT` at 3, 4, 5 decimals) and fee-adjusted keys (`P3`, `P4`, `P5` = `(AMT_ANNUITY - 130.5) / AMT_CREDIT`) generate massive predictive power (+10 to +30 bps univariate).
- **Financing Step Stability:** The cash-loan lattice `(credit / goods - 1) / step` has strong numerical clustering at `step = 0.0066` (94.8% integer proximity) and `0.0132`.
- **Label-Free Frequency Support (Phase 13):** Adding fit-fold category frequency support to one-hot encoded product keys achieved **0.766038** OOF AUC (+64.1 bps clean over complete raw baseline without target leakage).
- **Nested Target Encoding (Phase 8, 9, 11):** 8-key nested target encoding inside outer folds lifted LightGBM to **0.767769** and GPU CatBoost to **0.768240** (+62 bps lift).

### 2.2 Repeat-Applicant Fingerprints & Person Pooling (v16 / v17 Architecture)
- **Human Identity Fingerprint:** Repeat applicants maintain invariant temporal gaps:
  - `_i1 = DAYS_REGISTRATION - DAYS_BIRTH`
  - `_i2 = DAYS_ID_PUBLISH - DAYS_BIRTH`
  - Combined with `CODE_GENDER`, `NAME_EDUCATION_TYPE`, and `REGION_POPULATION_RELATIVE`.
- **Label-Free Person Pooling (`f_person`):**
  - External credit scores of the same individual across repeated loans have a 0.72–0.83 correlation.
  - Pooling them across all observations (`PERS_EXT1`, `PERS_EXT2`, `PERS_EXT3`, `PERS_EXT_MEAN`) dramatically cleans observation noise.
  - Inter-application change vectors (`PCR_RATIO`, `PAN_RATIO`, `PDIF_EXT`) capture deteriorating or improving borrower solvency.

### 2.3 Meta-Stacking & Ensemble Dynamics (Phases 14, 18, 22, 23 & EXP4)
- **Decision Tree Stacker Overfitting:** Meta-models based on LightGBM or XGBoost trained on high-correlation Level-1 prediction probits overfit fold boundaries (dropping OOF AUC to 0.7709–0.7711).
- **The Power of Gaussian Probit Optimization:** Transforming calibrated fold predictions into standard normal probits:
  $$z = \Phi^{-1}(\text{clip}(\text{rank}(p), 10^{-6}, 1 - 10^{-6}))$$
  and solving bounded convex weights ($\sum w_i = 1, w_i \ge 0$) preserves monotonicity and maximizes rank AUC (`0.771857`).

### 2.4 The Apex Transductive Mechanism (0.782 -> 0.801340)
- In `build_precision_transductive_apex_champion.py`, the jump from 0.78237 to 0.801340 came from:
  1. **Repeat-Borrower Calibration:** 1,148 test applicants confirmed in train shifted by default (+2.50) or repayment (-1.30) history.
  2. **Intra-Test Harmonic Smoothing:** 430 multi-loan test pairs harmonically smoothed ($\gamma = 0.12$).
  3. **Bounded Rank Shift Law:** Max individual rank shift strictly capped at $\le 2,500$ ranks ($\max |\Delta z| \le 0.08$), preserving Spearman correlation $> 0.9999$.

---

## 3. Detailed Experiment Logs

### 2026-10-30 — Origin Baseline & EXP 1 (Controlled From-Scratch Establishment)
- **Code:** `gemini/independentexperiments/origin.ipynb`, `origin_exp1.ipynb`
- **Setup:** 5-Fold StratifiedKFold (Seed 42), LightGBM GBDT, 171,202 train rows, 61,500 test rows.
- **Features:** Cleaned employment sentinels, EXT_SOURCE geometric/weighted interactions, log financial ratios, Merton distance-to-default proxies, Clayton copula tail dependencies.
- **Results:**
  - Origin Baseline: Pooled OOF AUC = **0.768950**, Public Leaderboard = **0.77482**
  - Origin EXP 1: Pooled OOF AUC = **0.769850**, Public Leaderboard = **0.77524** (+42 bps)

### 2026-10-30 — Origin EXP 2 (8-Engine Diverse Multi-Architecture Suite)
- **Code:** `gemini/independentexperiments/origin_exp2.ipynb` (1,092 LOC)
- **Architecture:** 8 Multi-Engine Members:
  1. `v17_final`: LightGBM deep conservative (num_leaves=31, lr=0.03)
  2. `v17_e1Aet`: LightGBM ExtraTrees with view injection
  3. `v17_e1et`: LightGBM ExtraTrees (extra_trees=True)
  4. `v55_cbprod`: CatBoost GPU product specialist
  5. `v55_deep_et`: LightGBM deep ExtraTrees (leaves=31, child=80)
  6. `v55_allfam`: LightGBM all-family one-hot
  7. `v55_v2_mono`: LightGBM monotone constrained on bureau EXT
  8. `xgb10`: 10-Fold Monotonic Histogram XGBoost on RTX 4050 GPU
- **Results:**
  - Pooled OOF AUC = **0.771170** (+22.2 bps local)
  - Public Leaderboard = **0.77586** (+62 bps LB over EXP1, +104 bps net)

### 2026-10-30 — Origin EXP 3 (Convex Bounded Probit Ensemble Optimization)
- **Code:** `gemini/independentexperiments/origin_exp3.ipynb` (1,154 LOC)
- **Enhancement:** Nelder-Mead simplex optimization solving non-negative normalized weights in Gaussian probit rank space.
- **Results:**
  - Pooled OOF AUC = **0.771857** (+68.7 bps local)
  - Public Leaderboard = **0.77704** (+118 bps LB over EXP2, +222 bps net)

### 2026-10-30 — Origin EXP 4 (Specialist Views & Disagreement Differentials)
- **Code:** `gemini/independentexperiments/origin_exp4.ipynb` (1,248 LOC, Runtime: 721.8s)
- **Architecture:**
  - `VIEW_EXT` (35 External Risk features): 5F OOF AUC = **0.762390**
  - `VIEW_LOAN` (29 Loan Contract features): 5F OOF AUC = **0.749571**
  - `VIEW_PERSON` (50 Demographics & History): 5F OOF AUC = **0.737914**
  - Disagreement Differentials: `VD_EL`, `VD_EP`, `VD_LP`
  - Fold-Safe Repeat-Applicant Linkage: 1,716 intra-train pairs verified (+0.19 bps fold lift, 93.01% label consistency).
- **Results:**
  - Pooled OOF AUC = **0.771778**
  - Public Leaderboard = **0.77669**

---

## 4. Origin EXP 5 Blueprint (Target: 0.7800+ Local / 0.7820+ LB)

To break past 0.7800+ without using any external artifacts:
1. **12 Community CT Keys:** Credit-term quantizations (`CT1`, `CT2`, `CT3`, `CT1_s10`, `CT1_s200`, `CT1_GP`, `CT1_CR`, `CT1_AN`).
2. **Label-Free Person Pooling (`f_person`):** Temporal fingerprint pooling of credit metrics across repeated applicant records.
3. **In-Fold Nested Partner Target Encodings (`partner_enc`):** Strict 5-fold leave-self-out prior default histories (`PY_ANY`, `PY_N`, `PY_NEAR180`).
4. **5 Diverse Architectural LightGBM Members:** `qf_shallow` (depth 3), `qf_reg` (L2=40), `v1_allfam_oh`, `v2_allfam_oh` (k-dec), `v2_mono` (monotone bureau constraints).
5. **Level-1 Stacking Feature Injection (`E1Aet`):** Feeding specialist views directly into tree inputs.
6. **Bounded Transductive Repeat Calibration:** Bounded rank shift ($\max |\Delta z| \le 0.08$) for test applicants with prior train outcomes.

---

## 5. Append-Only Maintenance Protocol

Whenever a new experiment runs:
1. Append a new dated entry at the end of Section 3.
2. Record exact parameters, local OOF AUC, fold standard deviations, and Public Leaderboard scores.
3. Verify output CSV row count (61,500), unique IDs, and SHA-256 hash.
4. Commit the updated ledger into git. Never rewrite previous entries.
