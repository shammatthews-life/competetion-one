# GCI World 2026: Credit Default Risk Modeling

Clean-room, reproducible, strictly supervised inductive competition pipeline generating automated predictions strictly from `train.csv`, `test.csv`, and `sample_submission.csv`.

## Certified Production Pipeline (V28 Audit-Certified)

- **Main Executable Script:** `solo_champion_v28_certified.py`
- **Interactive Notebook:** `solo_champion_v28_certified.ipynb`
- **Pinned Dependencies:** `requirements.txt`
- **Output Submission:** `LEADERBOARD_SUBMISSION_V28_CERTIFIED.csv`
- **Research & Verification Documentation:**
  - `ORIGIN_MASTER_VERIFICATION_LEDGER.md`
  - `truth/reports/DEEP_ACTUARIAL_RESEARCH_AND_STAGE_OPTIMIZATION.md`

### Quick Run

```bash
pip install -r requirements.txt
python solo_champion_v28_certified.py --data_dir "D:\gci\inpuy" --output_csv "LEADERBOARD_SUBMISSION_V28_CERTIFIED.csv"
```

## Compliance & Validation Guarantees

1. **Zero External Data / Historical CSV Reads:** Loads solely `train.csv`, `test.csv`, and `sample_submission.csv`.
2. **Zero Hand-Labeling & Target Propagation:** No heuristic repeat-applicant target transfer (`partner_enc`) and no manual EDA-conditional test weights.
3. **Strictly Inductive Preprocessing:** Residual regressors, medians, frequency counts, group deviations, winsorization bounds, scalers, Bayesian target encodings, and probit CDF curves are all fitted strictly inside training folds (never using `test.csv` rows).
4. **Zero In-Sample Stacking & Unbiased Validation:** All stacking views use row-aligned inner 3-fold out-of-fold predictions; ensemble weights are evaluated strictly via outer 5-fold nested cross-validation on untouched validation folds.

## Original Sources & Citations

- **LightGBM:** Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30, 3146–3154.
- **CatBoost:** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS 31, 6638–6648.
- **XGBoost:** Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD, 785–794.
- **Scikit-learn:** Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
- **SciPy:** Virtanen, P., et al. (2020). *SciPy 1.0: fundamental algorithms for scientific computing in Python.* Nature Methods 17, 261–272.
- **PyTorch:** Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 32, 8024–8035.
- **Competition Baseline & Data Dictionary:** GCI World 2026 / Home Credit Default Risk materials (`Competition/README.ipynb`, `Competition/tutorial.ipynb`, `HomeCredit_columns_description.xlsx`, Omnicampus / Matsuo-Iwasawa Lab).

