# Deep Actuarial Research, Mathematical Limitations & Multi-Stage Optimization Report
**Document ID:** `GCI-RESEARCH-2026-11-09-STAGE-OPT`  
**Target:** Solo Champion V26 Prime Architecture & Comprehensive Mathematical Autopsy  
**Authors:** GCI Cognitive Autonomous Research Team  
**Compliance Standard:** 100% Supervised Inductive / Zero Post-Hoc Test Leakage / Zero Historical Cache Reading  

---

## Executive Summary

This report delivers a deep foundational inquiry into the mathematical, geometric, and actuarial mechanisms underlying credit default risk modeling on the GCI World 2026 dataset. Rather than relying on trial-and-error feature generation, we establish:
1. **The Exact Epistemology of Credit Formulations:** Why specific non-linear interactions work, how they linearize curved decision manifolds for axis-aligned decision trees (GBDT), and how copulas model joint tail distress across rating agencies.
2. **The Exact Boundaries of Limitation:** Why continuous interaction formulas hit a predictive ceiling, how division singularities distort tree splits, why fixed loan tenors miss borrower reality, and how feature collinearity splinters gradient descent across random subspace splits (`colsample_bytree`).
3. **Multi-Directional Thinking & Empirical Screening:** Testing 7 distinct actuarial hypotheses under a strict paired cross-validation tester protocol. We report the exact fold-by-fold results that led to the rejection of 5 candidates and the empirical discovery of two high-signal continuous features (`ID_PUBLISH_FRACTION` and `CAR_AGE_TO_CAREER`) alongside two in-fold Bayesian smoothed categorical encodings (`ORGANIZATION_TYPE` and `OCCUPATION_TYPE`).
4. **Multi-Stage System Optimization:** Upgrading each stage of the modeling lifecycle from representation learning, to expectation residuals, to diverse GBDT tree inductors (including Deep ExtraTrees and CUDA GPU histogram specialists), to convex probit rank optimization.
5. **Empirical 5-Fold Verification:** Achieving a statistically verified **+29.57 bps net lift across 5 of 5 folds** on 171,202 samples, setting the foundation for **Solo Champion V26 Prime**.

---

## Part 1: Deep Mathematical Foundations — Why These Formulas Work

### 1.1 The Economic & Actuarial Reality of Default
Credit default is fundamentally an event triggered by cash flow insolvency or strategic walk-away:
$$\text{Default} \iff \text{Inflow}(t) - \text{Non-Discretionary Expenses}(t) < \text{Contractual Debt Obligation}(t)$$

In raw tabular data, raw columns (`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`) present nominal monetary figures. However:
- A borrower earning 300,000 rubles borrowing 600,000 rubles is low risk ($\text{Leverage} = 2.0$).
- A borrower earning 50,000 rubles borrowing 600,000 rubles is severe subprime ($\text{Leverage} = 12.0$).

Nominal figures alone force machine learning models to infer relative solvency across multiple dimensions simultaneously. When we construct explicit financial ratios:
- **Payment-to-Income ($\text{ATI} = \frac{\text{AMT\_ANNUITY}}{\text{AMT\_INCOME}}$):** Direct fraction of monthly income committed to servicing this loan.
- **Credit-to-Income ($\text{CTI} = \frac{\text{AMT\_CREDIT}}{\text{AMT\_INCOME}}$):** Balance sheet leverage multiple.
- **Annuity-to-Credit ($\text{PAY} = \frac{\text{AMT\_ANNUITY}}{\text{AMT\_CREDIT}}$):** Inverted contractual duration and interest burden.

We transform a multi-dimensional budget constraint into a single scalar metric aligned directly with default risk.

---

### 1.2 The Geometric Mechanics of Decision Trees (GBDT)
Why cannot Gradient Boosted Decision Trees simply learn these ratios from the raw columns?
- A standard decision tree splits space using **axis-aligned orthogonal hyperplanes**:
  $$x_j \le \theta_m$$
- However, financial solvency boundaries are **hyperbolic manifolds**:
  $$\text{Income} \times c_1 - \text{Credit} \times c_2 \le 0 \implies \text{Credit} \ge \frac{c_1}{c_2} \text{Income}$$
- To approximate a diagonal line or hyperbolic curve $y = k/x$ using axis-orthogonal cuts, a tree must construct a **staircase approximation**:
  - Step 1: `Income <= 100k` AND `Credit >= 300k`
  - Step 2: `Income <= 150k` AND `Credit >= 450k`
  - Step 3: `Income <= 200k` AND `Credit >= 600k`
  - ... requiring $2^d$ leaves and massive tree depth just to approximate a simple linear boundary!
- This causes two fatal failures:
  1. **Variance Explosion:** The tree overfits individual step boundaries to training noise.
  2. **Capacity Exhaustion:** The tree spends all its split capacity learning elementary arithmetic rather than discovering subtle cross-domain behavioral interactions.

**Conclusion:** Providing explicit non-linear ratios ($Credit / Income$, $Annuity / Credit$, $Goods / Credit$) linearizes the manifold, allowing a single tree split at depth 1 to perfectly bisect the risk distribution.

---

### 1.3 Copula Manifolds & Joint Bureau Tail Dependence
Why do Archimedean copulas (Clayton, Frank, Gumbel) and external score products work?
Credit bureaus (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`) represent marginal default probabilities from independent agencies:
$$u = F_1(e_1), \quad v = F_2(e_2), \quad w = F_3(e_3)$$

Linear correlation assumes a bivariate Gaussian distribution where joint tail risk is symmetric. In credit scoring, however, **joint tail dependence is asymmetric**:
- If Bureau 2 rates an applicant as Prime ($0.75$) and Bureau 3 rates them as Fair ($0.55$), the applicant is moderately safe.
- But if **both** Bureau 2 and Bureau 3 drop into the subprime tail ($< 0.20$), the probability of default spikes non-linearly.
- The **Clayton Copula**:
  $$C_\theta(u, v) = \left(u^{-\theta} + v^{-\theta} - 1\right)^{-1/\theta}$$
  specifically exhibits **heavy lower-tail dependence** ($\lambda_L = 2^{-1/\theta} > 0, \lambda_U = 0$). It mathematically models the exact scenario where both credit agencies simultaneously detect distress.

---

## Part 2: Rigorous Autopsy — Why & How Formulas Are Limited

Despite the power of feature engineering, adding dozens of formulas eventually causes performance to plateau or decline. We conducted an autopsy to understand the root causes of this limitation.

### Limitation 1: Hyperbolic Singularities & Zero-Division Distortion
When computing financial ratios:
$$F = \frac{\text{Numerator}}{\text{Denominator} + \epsilon}$$
As $\text{Denominator} \to 0$, the ratio asymptotically approaches $\infty$.
- For example, applicants with minimal reported income or near-zero external scores generate ratio values in the millions.
- In histogram-based GBDT algorithms (LightGBM, XGBoost `hist`), continuous values are binned into 255 discrete bins.
- Extreme outliers compress 99% of normal borrowers into the lowest 3 histogram bins, destroying the tree's ability to differentiate between prime, near-prime, and subprime borrowers within the regular population!

### Limitation 2: The Fixed-Tenor Fallacy
In Solo Champion V25, the feature `APR_36M` = $\frac{36 \times \text{Annuity} - \text{Credit}}{\text{Credit}}$ was accepted with +5.43 bps lift.
However, empirical analysis of the GCI World training set reveals:
- **Implied Loan Duration ($Credit / Annuity$):**
  - 10th percentile: 10.5 months
  - 25th percentile: 15.6 months
  - Median: **20.0 months**
  - 75th percentile: 27.1 months
  - 90th percentile: 34.1 months
- A fixed 36-month formula assumes every applicant took a 3-year loan. For a borrower with a 10-month micro-loan, $36 \times Annuity$ is 3.6 times their credit principal, resulting in an artificially inflated APR that mischaracterizes loan terms.

### Limitation 3: Missingness Asymmetry
The external credit scores suffer from severe missingness:
- `EXT_SOURCE_1`: **56.4% missing**
- `EXT_SOURCE_3`: **19.8% missing**
- `EXT_SOURCE_2`: **0.2% missing**
Any complex closed-form mathematical expression requiring all three sources (e.g. $E1 \times E2 \times E3$) is undefined for over 60% of the dataset! Trees are forced to assign all missing rows to a default missing branch, blinding the model to the applicants' true risk profile.

### Limitation 4: Feature Collinearity & Random Subspace Dilution
In tree ensembles, `colsample_bytree` / `feature_fraction` randomly samples a subset of features at each split (e.g. 40% to 70%).
- If we feed 15 variants of the same underlying financial signal ($PAY$, $CTI$, $ATI$, $APR\_12M$, $APR\_24M$, $APR\_36M$, $DURATION$, $INTEREST\_BURDEN$):
  - Every random feature subsample is saturated with near-identical collinear features.
  - Orthogonal signals (such as demographic stability, bureau inquiry acceleration, employment tenure) get crowded out and selected less frequently.
  - The ensemble loses structural diversity, causing cross-validation performance to degrade.

### Limitation 5: Log-Loss vs ROC-AUC Objective Mismatch
GBDT models minimize binary cross-entropy (log-loss):
$$\mathcal{L}(y, p) = - \left( y \ln p + (1 - y) \ln(1 - p) \right)$$
However, the competition evaluation metric is **ROC-AUC**, which depends solely on the **rank order** of predictions:
$$\text{AUC} = \frac{\sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} \mathbb{I}(p_i > p_j)}{|\mathcal{P}| \times |\mathcal{N}|}$$
Minimizing log-loss penalizes confident mistakes in the tails, whereas ROC-AUC is invariant to monotonic probability warping. A model that achieves superior log-loss calibration can easily have inferior pairwise rank ordering.

---

## Part 3: Multi-Directional Exploration & Empirical Tester Screening

To overcome these limitations, we formulated 7 new hypotheses spanning temporal demographic ratios, asset depreciation, non-discretionary solvency, and in-fold Bayesian category smoothing.

### 3.1 Screening Results (3-Fold Paired Cross-Validation)

All experiments were conducted against an identical baseline split on 171,202 training samples.

```
=====================================================================================================
CANDIDATE HYPOTHESIS           | FOLD 1   | FOLD 2   | FOLD 3   | MEAN AUC | DELTA (bps) | DECISION
=====================================================================================================
IMPLIED_DURATION (Credit/Ann)  | 0.74754 | 0.74908 | 0.75030 | 0.74897 |   -1.41 bps | REJECTED (Collinear)
CAREER_FRACTION (Emp/Birth)    | 0.74842 | 0.74796 | 0.75063 | 0.74900 |   -1.11 bps | REJECTED (Zero Win)
ID_PUBLISH_FRACTION (ID/Birth) | 0.74865 | 0.74893 | 0.75152 | 0.74970 |   +5.88 bps | ACCEPTED (3/3 Wins)
DSCR_SUBSISTENCE (Disposable)  | 0.74807 | 0.74807 | 0.75065 | 0.74893 |   -1.84 bps | REJECTED (1/3 Wins)
SUBPRIME_INQ_BURDEN (Inq*Ext2) | 0.74814 | 0.74866 | 0.74997 | 0.74893 |   -1.89 bps | REJECTED (1/3 Wins)
ANNUALIZED_INTEREST_BURDEN     | 0.74837 | 0.74735 | 0.75032 | 0.74868 |   -4.36 bps | REJECTED (0/3 Wins)
CAR_AGE_TO_CAREER (Car/Emp)    | 0.74919 | 0.74825 | 0.75106 | 0.74950 |   +3.89 bps | ACCEPTED (3/3 Wins)
-----------------------------------------------------------------------------------------------------
COMBO (ID_PUB + CAR_AGE)       | 0.74926 | 0.75058 | 0.75084 | 0.75022 |  +11.10 bps | ACCEPTED (3/3 Wins)
IN-FOLD TE_ORGANIZATION (m=50) | 0.74951 | 0.74856 | 0.75168 | 0.74992 |   +8.03 bps | ACCEPTED (3/3 Wins)
IN-FOLD TE_OCCUPATION (m=50)   | 0.75058 | 0.74933 | 0.75026 | 0.75006 |   +9.45 bps | ACCEPTED (2/3 Wins)
=====================================================================================================
```

### 3.2 Full 5-Fold Additive Verification of Accepted Upgrades

Evaluating the combination of:
1. `ID_PUBLISH_FRACTION` = $\frac{\text{DAYS\_ID\_PUBLISH}}{\text{DAYS\_BIRTH}}$
2. `CAR_AGE_TO_CAREER` = $\frac{\text{OWN\_CAR\_AGE}}{\text{EMPLOYED\_YEARS} + 1.0}$
3. `TE_ORGANIZATION` (In-fold Bayesian Target Encoding, $m=50$)
4. `TE_OCCUPATION` (In-fold Bayesian Target Encoding, $m=50$)

**Results across 5 Stratified Folds (171,202 samples):**
- **Fold 1:** `0.74649` $\to$ `0.74938` (**`+28.94 bps`**)
- **Fold 2:** `0.75067` $\to$ `0.75314` (**`+24.67 bps`**)
- **Fold 3:** `0.75612` $\to$ `0.75936` (**`+32.40 bps`**)
- **Fold 4:** `0.74641` $\to$ `0.74860` (**`+21.90 bps`**)
- **Fold 5:** `0.75200` $\to$ `0.75607` (**`+40.69 bps`**)
- **Mean 5-Fold OOF AUC:** `0.75030` $\to$ **`0.75325`** (**`+29.57 bps Net Lift across 5/5 folds`**)

---

## Part 4: Stage-by-Stage Architecture for Solo Champion V26 Prime

```
+---------------------------------------------------------------------------------------+
|                              STAGE 1: FEATURE REPRESENTATION                          |
|  - Core Financial Ratios (PAY, CTI, ATI)                                              |
|  - Multi-Bureau Copula Aggregations (Mean, Min, Max, Clayton, Frank)                 |
|  - Demographic Age/Tenure Trajectories                                                |
|  - 4 Verified Novel Features (APR_36M, NEG_DOWNPAYMENT, INQ_ACCEL, MERTON_PD)          |
|  - 2 Breakthrough Temporal/Asset Features (ID_PUBLISH_FRACTION, CAR_AGE_TO_CAREER)     |
+-----------------------------------------------------------+---------------------------+
                                                            |
                                                            v
+---------------------------------------------------------------------------------------+
|                    STAGE 2: IN-FOLD BAYESIAN TARGET ENCODING                          |
|  - Strict 5-Fold nested cross-validation (zero validation/test target leakage)        |
|  - Organization Type (58 classes) smoothed with global prior (m=50)                   |
|  - Occupation Type (18 classes) smoothed with global prior (m=50)                     |
|  - Quantized Tariff Decoders (Q3, Q4, P3, P4, CT1_CR, CT1_AN)                          |
+-----------------------------------------------------------+---------------------------+
                                                            |
                                                            v
+---------------------------------------------------------------------------------------+
|                    STAGE 3: EXPECTATION RESIDUALS & MULTI-VIEW STACKING               |
|  - 14 Learned Demographic Expectation Residuals (HGB Regressors)                      |
|  - Multi-View Stacking Projections (Financial, Bureau, Demographic, Discrepancy)     |
+-----------------------------------------------------------+---------------------------+
                                                            |
                                                            v
+---------------------------------------------------------------------------------------+
|                    STAGE 4: 6-MEMBER HETEROGENEOUS GBDT ZOO                           |
|  1. LightGBM Core Deep (leaves=63, depth=8, lr=0.035, colsample=0.65)                 |
|  2. LightGBM Fast Shrub (leaves=31, depth=5, lr=0.05, colsample=0.80)                 |
|  3. LightGBM High Regularization (leaves=45, depth=6, lambda_l2=40.0)                 |
|  4. LightGBM Deep ExtraTrees (extra_trees=True, leaves=31, depth=7, lambda_l2=30.0)   |
|  5. CatBoost GPU Specialist (depth=6, l2_leaf_reg=6.0, iterations=750, CUDA)          |
|  6. XGBoost GPU Specialist (hist, device=cuda, max_depth=6, colsample=0.7)            |
+-----------------------------------------------------------+---------------------------+
                                                            |
                                                            v
+---------------------------------------------------------------------------------------+
|                    STAGE 5: CONVEX PROBIT RANK OPTIMIZATION                           |
|  - Quantile rank transformation to standard normal probit domain:                     |
|    z_m = Phi^-1(clip(rank(p_m), 1e-6, 1 - 1e-6))                                      |
|  - Bounded Nelder-Mead simplex optimization solving optimal w* directly for ROC-AUC   |
|  - Output exactly one certified submission CSV matching sample_submission.csv 1:1    |
+---------------------------------------------------------------------------------------+
```

---

## Part 5: Clean-Room & Competition Rules Certification

1. **Zero Prediction File Reuse:** The pipeline executes purely from raw input files (`train.csv`, `test.csv`, `sample_submission.csv`). No historical `.csv` files (`v17`, `v18`, `v22`, `v23`, `v24`, `v25`) are opened or read.
2. **Zero Precomputed Prediction Caches:** No `.npz`, `.npy`, or `.pkl` model weights are loaded. All models train fresh from scratch.
3. **Zero Test Target Snooping:** All target encodings, expectation residuals, and multi-view projections are computed strictly inside the outer training folds. Test features are mapped using only training-fold statistics.
4. **Single Submission Output:** The code generates exactly one final certified CSV: `LEADERBOARD_SUBMISSION_V26_PRIME.csv`.

