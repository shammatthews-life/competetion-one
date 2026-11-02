# EXP10 — Independent Replication of the EXP09 Best Pair

EXP10 reproduces only the exact fixed pair selected by EXP09:

`equal_exp07_signature__exp08_lgbcat`

The original EXP08 runner and historical feature builder are archived byte-for-byte
under `source_snapshot/original/`; the execution copies and EXP07 runner are in
`source_snapshot/`. The isolated runners add support for outer seed `20261102`,
records the newly generated fold assignment rather than comparing it to old folds,
and disables component submission CSVs. Model and feature logic are not changed.
The historical feature source is checked against the SHA-256 recorded by EXP08.

## Run

From this directory in PowerShell:

```powershell
$runDir = "runs\exp10_pair_seed20261102_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python run_exp10.py --data-dir D:\gci\inpuy --run-dir $runDir
python verify_exp10.py --run-dir $runDir
python -m unittest discover -s tests -v
```

If both component runs completed but the parent runner was interrupted before
writing the combined evidence, finalize into a new directory without retraining:

```powershell
python run_exp10.py --data-dir D:\gci\inpuy `
  --run-dir runs\exp10_pair_seed20261102_final `
  --reuse-components-from runs\exp10_pair_seed20261102
```

The run trains the exact EXP07 signature model and the EXP08 fold-local
LightGBM/CatBoost component on the same five `StratifiedKFold` folds with seed
`20261102`. EXP08's saved three-model mean is separated back into its exact
LightGBM/CatBoost component by subtracting its saved CUDA-XGBoost component.
The pair is always a fixed 50:50 mean. No weights or model parameters are
selected against outer-validation labels.

All reported AUC comparisons use the single fresh Phase 2 OOF/test stream from
the EXP07 run. It is saved in `shared_baseline_contract.npz` with the ordered
IDs, labels, and fold membership. Component-local Phase 2 predictions are
checked and preserved as diagnostics, but are never substituted into pair
metrics. If independently trained component baselines differ, the exact
discrepancy is reported rather than calibrated or hidden.

The run fails rather than falling back to CPU if required CUDA preflights fail.
LightGBM uses CPU because the installed LightGBM build is CPU-only. GPU utilization,
VRAM, source configs, fold predictions, and source/output hashes are preserved in
the component run directories and `SHA256_MANIFEST.json`.

`candidate_submission.csv` is created only if the pair clears both predeclared
OOF thresholds: at least `+0.001` pooled AUC and at least four of five folds
improved. The independent verifier writes `VERIFICATION.json` and adds its hash
to the run manifest. OOF AUC is not an official leaderboard score.
