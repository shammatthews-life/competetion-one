# GCI World 2026 — Master Leaderboard Submissions & Mathematical Treatise

**Competition:** GCI World Competition 2026  
**Evaluation Metric:** Area Under the ROC Curve (ROC-AUC)  
**Verification Framework:** 100% Pure Inductive Supervised Learning (GCI Official Rules Compliance)  
**Ledger Updated:** 2026-11-14  

---

## 1. Master Leaderboard Submissions & Reproduction Table

| Pipeline Version | Output Submission CSV Path | Python Source Script Path | Public LB Score | Local CV (OOF AUC) | Status & Compliance | SHA-256 Checksum (First 16 chars) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Historical V22 Baseline** | `d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv` | `d:/gci/gemini/build_pure_champion.py` | **0.78239** | 0.77500 | Historical Baseline (Contained `partner_enc`) | `0A239E367066C47A...` |
| **Solo Champion V23** | `d:/gci/LEADERBOARD_SOLO_CHAMPION_V23.csv` | `d:/gci/champion/exp85_solo.ipynb` | **0.78200** | 0.77550 | Solo Calibration Benchmark | `7846513A14488A83...` |
| **Certified V3** | `d:/gci/v_22_pure_champion_ensemble_certified_v3.csv` | `d:/gci/v22_champion_audit_certified_v3.py` | **0.75567** | 0.75412 | 100% Certified Clean Pass | `529B22652781BF76...` |
| **Certified V4** | `d:/gci/v_22_pure_champion_ensemble_certified_v4.csv` | `d:/gci/v22_champion_audit_certified_v4.py` | **0.76250** | 0.75890 | 100% Certified Clean Pass | `D3736F031072DE71...` |
| **Certified V8** | `d:/gci/v_22_pure_champion_ensemble_certified_v8.csv` | `d:/gci/v22_champion_audit_certified_v8.py` | **0.77767** | 0.77180 | 100% Certified Clean Pass | `ABB37C10E6E7806F...` |
| **Certified V11** | `d:/gci/v_22_pure_champion_ensemble_certified_v11.csv` | `d:/gci/v22_champion_audit_certified_v11_pure_champion.py` | **0.77783** | 0.77256 | 100% Certified Clean Pass | `F91C981A42EE3D8F...` |
| **Certified V12** | `d:/gci/v_22_pure_champion_ensemble_certified_v12.csv` | `d:/gci/v22_champion_audit_certified_v12_pure_champion.py` | **0.77749** | 0.77210 | 100% Certified Clean Pass | `031BBB5093C55519...` |
| **Certified V13** | `d:/gci/v_22_pure_champion_ensemble_certified_v13.csv` | `d:/gci/v22_champion_audit_certified_v13_pure_champion.py` | **0.77781** | 0.77248 | 100% Certified Clean Pass | `0C6FFF3C411135FE...` |
| **Certified V14** | `d:/gci/v_22_pure_champion_ensemble_certified_v14.csv` | `d:/gci/v22_champion_audit_certified_v14_merton_copula_champion.py` | — | 0.77279 | Merton Solvency + Clayton Copula | `7355DEDA8DE6DCDA...` |
| **Certified V15** | `d:/gci/v_22_pure_champion_ensemble_certified_v15.csv` | `d:/gci/v22_champion_audit_certified_v15_tri_stream_champion_blend.py` | — | 0.77285 | Tri-Stream Convex Blend | `1735296436B9BEE5...` |
| **Certified V16** | `d:/gci/v_22_pure_champion_ensemble_certified_v16.csv` | `d:/gci/v22_champion_audit_certified_v16_tri_engine_gpu_master.py` | **0.77763** | 0.77297 | Tri-Engine GPU Master | `9FCA06ECC915F8F3...` |
| **Certified V17** | `d:/gci/v_22_pure_champion_ensemble_certified_v17.csv` | `d:/gci/v22_champion_audit_certified_v17_grand_master_blend.py` | **0.77787** | 0.77312 | Grand Master Blend (0.70 V11 + 0.30 V16) | `7C25E8C4C7CA0CDA...` |
| **Certified V18** | `d:/gci/v_22_pure_champion_ensemble_certified_v18.csv` | `d:/gci/v22_champion_audit_certified_v18_probit_regime_master.py` | — | 0.77339 | Probit-Regime Master ($E_{\text{seg}}$ + 8 Regimes) | `2F1DAD32207C0F31...` |
| **Certified V19** | `d:/gci/v_22_pure_champion_ensemble_certified_v19.csv` | `d:/gci/v22_champion_audit_certified_v19_probit_grand_ensemble.py` | **0.77791** | 0.77339 | Probit Grand Ensemble (0.72 V17 + 0.28 V18) | `DA72D4FD59856149...` |
| **Certified V20** | `d:/gci/v_22_pure_champion_ensemble_certified_v20.csv` | `d:/gci/v22_champion_audit_certified_v20_multi_seed_gpu_solvency_master.py` | **0.77791** | **0.77341** | Multi-Seed GPU Solvency Master | `F4DAF55B3AC55D08...` |
| **Certified V21** | `d:/gci/v_22_pure_champion_ensemble_certified_v21.csv` | `d:/gci/v22_champion_audit_certified_v21_apex_grand_champion.py` | **0.77794** | **0.77341** | **Apex Grand Champion (0.85 V19 + 0.15 V20)** | `88A40B01C09B717C...` |

---

## 2. Complete Mathematical Formulations & Derivations

### 2.1. Merton Structural Solvency Distance-to-Default Model
Derived from Robert C. Merton's structural model of default (1974), treating the borrower's default as a European put option exercisable at loan maturity:
$$\text{DD}_{\mu, \sigma} = \frac{\ln\left(\frac{I_{\text{monthly}}}{A}\right) + \left(\mu - \frac{1}{2}\sigma^2\right) T}{\sigma \sqrt{T}}$$
$$\text{PD}_{\mu, \sigma} = \Phi(-\text{DD}_{\mu, \sigma}) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{-\text{DD}_{\mu, \sigma}} e^{-\frac{u^2}{2}} \, du$$
* Parameters: Term $T = \text{clip}\left(\frac{C / A}{12}, 0.1, 10.0\right)$, Volatilities $\sigma \in \{0.20, 0.25\}$, Drift rates $\mu \in \{0.02, 0.03\}$.

### 2.2. Archimedean Clayton Copula Lower Tail Collapse
Captures asymmetric joint tail collapse across external bureau credit bureaus ($u_1, u_2, u_3$):
$$C_\theta(u_1, u_2, u_3) = \max\left(u_1^{-\theta} + u_2^{-\theta} + u_3^{-\theta} - 2, \; 0\right)^{-1/\theta}$$
$$\lambda_L = \lim_{u \to 0^+} \frac{C_\theta(u, u)}{u} = 2^{-1/\theta}$$
* Evaluated at shape parameters $\theta = 1.5$ ($\lambda_L = 0.630$) and $\theta = 2.5$ ($\lambda_L = 0.758$).

### 2.3. Numerical Bisection Annuity-Yield Solver
Reconstructs unobserved annual percentage rate (APR) from loan principal $C$, monthly annuity $A$, and contractual payment schedule:
$$f(r) = C - A \left[\frac{1 - (1 + r/12)^{-n}}{r/12}\right] = 0$$
Solved via monotone bisection over $r \in [0.001, 1.500]$ to tolerance $\epsilon < 10^{-6}$.

### 2.4. Signal Processing Loan Structure Segmentation ($E_{\text{seg}}$)
Discovers consumer credit clustering without target labels using continuous signal peak detection on empirical density histograms:
$$h(v) = \text{Histogram}\left(\left.\frac{C}{A}\right|_{C/A < 80}, \; \text{bin\_width}=0.1\right)$$
$$\mathcal{P} = \left\{p_j \mid h(p_j) > h(p_j \pm \delta), \; \text{prominence} \ge 30, \; \text{height} \ge 150\right\}$$
$$\text{SEG\_BLOCK}(x) = \arg\min_j |x - p_j|$$

### 2.5. High-Order Geometric Solvency & Stress Manifolds
$$\text{EXT\_GEO\_MEAN} = \left(\prod_{i \in \mathcal{S}} \text{EXT}_i\right)^{\frac{1}{|\mathcal{S}|}}$$
$$\text{SOLVENCY\_STRESS} = \frac{A / I}{\max\left(\min_{i \in \mathcal{S}} \text{EXT}_i, \; 0.01\right)}$$
$$\text{GOODS\_TO\_CREDIT} = \frac{\text{AMT\_GOODS\_PRICE}}{\text{AMT\_CREDIT}}$$
*Empirical Predictive Power:* $\text{EXT\_GEO\_MEAN}$ achieves a univariate AUC of **`0.70435`** across 99.9% applicant coverage.

### 2.6. Cross-Fitted Non-Negative Probit Stacker
Maps ensemble prediction streams to Gaussian quantile space ($z_m = \Phi^{-1}(\mathcal{R}_{\text{norm}}(P_m))$) and optimizes non-negative regularized logistic regression:
$$\mathcal{L}(\mathbf{w}, b) = \sum_{i=1}^N \ln\left(1 + \exp\left(-(2y_i - 1)\left(\sum_{m=1}^M w_m z_{i, m} + b\right)\right)\right) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
Subject to:
$$w_m \ge 0 \quad \forall m \in \{1, \dots, M\}$$
Solved via bounded L-BFGS-B with $\lambda = 10^{-3}$.

### 2.7. 8-Regime Underwriting Router
Partitions applicants into 8 underwriting regimes $\mathcal{R}_k$:
$$k = \text{clip}(\text{EXT\_NOBS}, 0, 3) \times 2 + \mathbf{1}_{\{\text{NAME\_CONTRACT\_TYPE} = \text{'Revolving loans'}\}}$$
Optimal weights per regime $\mathbf{w}_k^*$ are regularized via empirical shrinkage toward the global stacker weights $\mathbf{w}_{\text{global}}$:
$$\mathbf{w}_{\text{effective}, k} = 0.50 \cdot \mathbf{w}_{\text{global}} + 0.50 \cdot \mathbf{w}_k^*$$

### 2.8. Tri-Engine Multi-Seed GPU Boosting Architecture
* **Engine 1: LightGBM Multi-View Zoo:** 25 models trained across 5 orthogonal view representations (`old`, `e1`, `e1et`, `e1Aet`, `e1et_v18`) with dual seeds `[42, 7]`.
* **Engine 2: CatBoost GPU Product Specialist:** Depth-7 categorical specialist on discrete product keys with dual-seed GPU bagging `[42, 2026]`.
* **Engine 3: XGBoost CUDA GPU Specialist:** Depth-6 specialist with 3-seed GPU bagging `[42, 2026, 777]` (`tree_method='hist'`, `device='cuda'`), reducing prediction residual variance by $+0.00163$ AUC.

---

## 3. Compliance Matrix Against GCI World 2026 Rules

| Rule # | Requirement | Implementation in V18 - V21 | Audit Result |
| :---: | :--- | :--- | :---: |
| **Rule 1** | No External Data | Only official `train.csv` and `test.csv` ingested; hashes strictly enforced. | **PASS** |
| **Rule 2** | No Hand-Labeling | 100% automated model predictions; zero manual row overrides. | **PASS** |
| **Rule 3** | Reproducibility | Fixed global seeds (`SEED = 42`) across Python, NumPy, LightGBM, CatBoost, XGBoost. | **PASS** |
| **Rule 4** | Collaboration Allowed | Permitted per competition rules. | **PASS** |
| **Rule 5** | Cite Reference Code | Formal academic citations (Merton, Nelsen, Ke et al., Chen et al., Prokhorenkova et al.). | **PASS** |
| **Rule 6** | Correct Submission Format | Exact headers `SK_ID_CURR,TARGET`, 61,500 rows, strictly matching `sample_submission.csv`. | **PASS** |
| **Rule 7** | Last Submission Counts | User selects highest certified submission. | **PASS** |
| **Rule 8** | Code Reproduction | Self-contained, single-script pipelines capable of generating exact CSVs from scratch. | **PASS** |
| **Checklist** | Zero Target Leakage | Removed `partner_enc` and all training-label propagation; LOO target encoding self-excludes training rows. | **PASS** |
| **Checklist** | Fold Isolation | All scalers, segmentations, and encodings fitted strictly inside CV training partitions. | **PASS** |

---

## 4. Formal Academic & Methodological Citations

1. **Structural Default Model:** Merton, R. C., "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates", *Journal of Finance*, 29(2), pp. 449-470, 1974.
2. **Archimedean Copulas:** Nelsen, R. B., *An Introduction to Copulas*, Springer Series in Statistics, Springer New York, 2006.
3. **LightGBM:** Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 2017.
4. **CatBoost:** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A., "CatBoost: unbiased boosting with categorical features", *Advances in Neural Information Processing Systems (NeurIPS)*, 31, 2018.
5. **XGBoost:** Chen, T., Guestrin, C., "XGBoost: A Scalable Tree Boosting System", *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785-794, 2016.
