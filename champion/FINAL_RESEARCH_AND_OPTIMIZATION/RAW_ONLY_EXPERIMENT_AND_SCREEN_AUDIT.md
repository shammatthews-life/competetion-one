# Raw-Only Experiment and Feature-Screen Audit

## Decision

The raw-only baseline completed and its candidate artifacts passed structural and hash validation. It achieved a fixed five-fold OOF AUC of **0.7625457907**. This is a new experiment, not a reconstruction of EXP85, and it does **not** establish or reproduce the historical 0.801340 score.

The supplied feature-screen code's known ablations were directionally reproduced, but its random-noise control failed the code's own stated tolerance in both the one-seed and three-seed runs. Candidate rankings from that screen are therefore not promoted as validated results.

The historical rebuild remains **INCOMPLETE**: the known V20 producer and 18 V080041 upstream dependencies have not been recovered. No candidate was submitted to the competition.

## Origin notebook findings

The origin inspected was `D:\gci\champion\exp85_solo.pre_independent_wiring_backup.ipynb`. It contains distinct execution paths and should not be run top-to-bottom as if it were a single raw-only pipeline:

- Its raw-only cells build new features and train fresh OOF/test models without loading the historical V17, V55, XGBoost, precision-anchor, or target caches. Those are alternative experiments; they do not recreate the original model bytes.
- Other cells search historical locations, load or regenerate V17/V55/XGBoost artifacts, and synthesize the final blend from `cache/v_17_full_suite_bundle.npz`, `cache/v55_members.npz`, and `cache/test_10f_xgb.npy`.
- The final verification cell compares against a historical reference. That comparison is not a training step and cannot establish raw reconstruction.

The notebook's coexistence of raw-only generators and cache-recovery/synthesis cells is a provenance hazard. New experiments should remain isolated from it.

## Completed independent raw-only baseline

| Item | Evidence |
|---|---|
| Raw inputs | `D:\gci\inpuy\train.csv` (171,202 rows), `test.csv` (61,500 rows), and `sample_submission.csv` |
| Historical prediction inputs | None, recorded in the run dependency manifest |
| Transductive feature inputs | None |
| Features | 102 rowwise features; LightGBM one-hot frame has 201 columns |
| CV | Stratified 5-fold, shuffle enabled, seed 42 |
| LightGBM | CPU; five folds; OOF AUC **0.7614474490** |
| CatBoost | GPU:0; five folds; OOF AUC **0.7601360676** |
| Fixed blend | Predeclared 50/50 probability average; OOF AUC **0.7625457907** |
| Runtime | 736.717 seconds; exit code 0 |
| Hardware | NVIDIA GeForce RTX 4050 Laptop GPU, driver 592.82, 6,141 MiB; CatBoost GPU training completed. LightGBM used CPU because this installed build does not support its GPU backend. |
| Official score | Not evaluated |

Artifacts are in `runs/raw_baseline_20261101_154924/`. The submission candidate contains 61,500 rows, exactly the sample-submission columns, unique IDs in exact test order, and finite predictions in [0, 1]. Its values were independently recomputed from the saved model predictions using float64 arithmetic; the CSV-rounding maximum absolute difference is below `5.1e-11`. The stored blend's OOF AUC and all eight generated-output hashes in the run manifest were independently verified.

Key hashes:

- Train: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- Test: `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`
- Baseline source: `c49aab451d1500e1bd7e0d618a764efeb1815c172f74e1bc5174e38a793ff3fe`
- Candidate CSV: `2ba6f3d0677851417e55265f74b707f9f25d2233cdad772559e67f27b78ba2fc`

The prediction comparison to `v_20261027_optimal_macro_apex_v85.csv` is only an ID-aligned reference comparison. Its differing prediction scale makes MAE/RMSE unsuitable as score evidence; neither correlation nor this comparison is an official evaluation.

## Supplied feature-screen research

The archived `gci_core.py` / `fast_screen.py` code was run with the raw competition CSVs, an empty per-run cache root, and no discovered HGB prediction arrays (`HGB_PATH=None`, `HGB_OOF=None`). XGBoost 3.4.1 used `device=cuda`, `tree_method=hist`. The feature builder reported 1,047 decoded product values versus 5,335 uncorrected five-decimal ratio values, and 6,149 strict / 9,650 loose partner rows.

The five-fold one-seed validation took 407.136 seconds (exit 0):

| Check | Mean paired AUC delta | Result |
|---|---:|---|
| Remove partner-label features | -0.00140 | Directionally matches the documented effect |
| Remove `TE_*` and `CNT_*` features | -0.00941 | Directionally matches; this removes more than product-key encodings alone |
| Add random noise | -0.00079 | Outside the documented ±0.0003 control tolerance |
| Add duplicate features | -0.00017 | Within the documented tolerance |

The three-seed control-only run took 660.588 seconds (exit 0):

| Check | Mean paired AUC delta | Standard error | Per-fold deltas |
|---|---:|---:|---|
| Add random noise | -0.00036 | 0.00012 | -0.00071, -0.00056, -0.00019, -0.00002, -0.00032 |
| Add duplicate features | -0.00004 | 0.00010 | +0.00002, -0.00035, +0.00019, +0.00009, -0.00016 |

The noise control remains outside tolerance and is negative in every fold. Since the proxy uses `colsample_bytree=0.5`, adding noise can displace useful features during column sampling; the result shows that the screen's control criterion does not currently distinguish this model-capacity effect from ordinary null behavior. The validation gate is **not passed**. Do not treat its `PROMISING` label as sufficient evidence; first correct or replace the control design, then validate any selected feature block with the intended production CV model.

The exact outputs and logs are in:

- `runs/research_screen_validation_20261101_162000/`
- `runs/research_screen_controls_3seed_20261101_163000/`

## Remaining blockers and next action

This work does not recover the original V20 writer, the 18 upstream V080041 producers, historical model checkpoints, or the complete original training configuration. It does not resolve the earlier CatBoost device discrepancy, train the original complete model suite, or obtain an official competition score. Therefore, the original raw-data EXP85 reconstruction is still **not certified**.

The next smallest corrective action is to repair the feature screen's null-control design and rerun its controls before using it to select features. In parallel, the decisive historical-lineage work remains recovery of the original V20 producer and each missing V080041 dependency; do not substitute historical prediction files and call that a raw rebuild.
