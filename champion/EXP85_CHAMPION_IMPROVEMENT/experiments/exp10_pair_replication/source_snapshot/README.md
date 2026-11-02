# EXP08 — Historical Feature Bridge

EXP08 recovers the historical executable's 669 feature definitions, excludes
the nine lineage-unsafe features, then evaluates the remaining 660 under the
same five outer folds and fold-local Phase 2 preprocessing used by EXP07.

## Run

From the repository root, use the existing experiment virtual environment:

```powershell
& '.\champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\.venv\Scripts\python.exe' `
  '.\champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp08_historical_feature_bridge\run_exp08.py' `
  --data-dir 'D:\gci\inpuy' `
  --run-dir '.\champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp08_historical_feature_bridge\runs\exp08_historical_seed20261101_v1'
```

Only a new or empty run directory is accepted. Failed attempts are preserved
with a `FAILED.json` and log. The default run is matched to the strict
20261101 Phase 2 reference. Seed 42 is available for independent replication
only after the advancement gate passes.

## Validation controls

- Five fixed outer folds; each outer fold has its own inner stratified
  stopping/selection split.
- Phase 2 fold-local encodings are rebuilt and audited for each partition.
- Historical Clayton ranks are fit on the current model-fit rows and applied
  to validation/test rows.
- Individual-family screening, feature selection, early stopping and blend
  weights use inner validation only. Outer labels are used only for final
  metrics.
- The candidate uses LightGBM on CPU, CatBoost on verified CUDA, and XGBoost
  on CUDA only when a real CUDA preflight succeeds. XGBoost never falls back
  silently to CPU.
- Historical infinities are retained and counted. `HOUSING_STABILITY` is
  excluded from model input because its historical denominator can be zero;
  any other generated infinity fails the run.
- Nine pooled signature/timeline or non-nested target-memory features are
  inventoried but not evaluated.
- The historical champion, frozen Phase 2 release, and earlier experiments
  are read-only inputs.

## Artifacts

The run directory contains OOF/test predictions for the matched baseline,
candidate, each historical family and the nested blend; fold indices and
metrics; feature importance; model configuration; GPU telemetry; a
submission; and SHA-256 manifests. `FEATURE_INVENTORY.csv` and
`FEATURE_PARITY.md` document feature lineage and reconstruction parity.
`verify_exp08.py` independently checks run integrity, fold alignment, AUCs,
and submission IDs.

An OOF improvement is not an official leaderboard result. Promotion requires
the predeclared pooled-AUC, fold-consistency and standard-error gate, followed
by an independent-seed replication.
