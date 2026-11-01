# Prediction-Gap Diagnostic

- Run: `20261102T001500Z_stage0_full_covariates`
- Status: `VALID_DIAGNOSTIC_ONLY`
- Raw test labels: none; no training or tuning was performed.
- All comparisons were joined to the exact raw test IDs.

| Pair | Pearson | Spearman | MAE | RMSE | Max abs error | Mean rank displacement | p95 rank displacement | Max rank displacement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase2_vs_historical_champion | 0.7782506970 | 0.9157084581 | 0.41111346 | 0.46865093 | 0.9401703 | 5392.1051 | 15026.1000 | 52044 |
| v5_hybrid_vs_historical_champion | 0.9999975816 | 0.9999975816 | 0.00030141741 | 0.00063487705 | 0.012504065 | 18.5372 | 80.0000 | 769 |
| phase2_vs_v5_hybrid | 0.7782540515 | 0.9157128812 | 0.41111342 | 0.46865075 | 0.94023534 | 5392.0868 | 15020.0500 | 52100 |

Grouped statistics are in `grouped_prediction_differences.csv`. They describe disagreement with a reference prediction, not subgroup default risk or model error against labels.

No candidate was selected or promoted from this report. Historical test predictions are not ground truth.
