# EXP09 — Prediction Diversity and Champion Bridge

EXP09 audits existing OOF/test prediction artifacts and evaluates fixed-weight combinations. It does not retrain models or modify prior experiment directories.

## Run

```powershell
$runDir = "runs\exp09_prediction_diversity_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python run_exp09.py --data-dir D:\gci\inpuy --run-dir $runDir
python verify_exp09.py --run-dir $runDir
python -m unittest discover -s tests -v
```

The runner refuses to proceed unless the raw input hashes match the frozen competition data. It requires exact prediction output hashes, input provenance, row counts, and fold membership against a seed-specific matched Phase 2 reference.

## Method

- Uses already-saved OOF and test arrays from EXP04–EXP08 and the GPU raw-XGBoost / interaction screens.
- Uses the strict matched Phase 2 reference separately for outer seeds 20261101 and 42.
- Uses only fixed equal-weight blends: 50:50 for pairs and 1/3 each for triples.
- Does not fit meta-models or choose weights from outer-validation labels.
- Reports the selected best pair/triple as exploratory because ranking many combinations causes selection bias.
- Reports residual/error overlap and risk-decile performance to identify complementary error signal.
- Does not produce a submission unless a candidate passes both first-seed and independent-seed gates.

`ARTIFACT_INVENTORY.csv` records provenance and eligibility; `PREDICTION_DIVERSITY.csv` records correlation, residual, risk-decile, error-overlap, and ranking-overlap diagnostics; `FOLD_METRICS.csv` records paired fold AUC; and each run's `SHA256_MANIFEST.json` records all inputs, sources, and outputs.
