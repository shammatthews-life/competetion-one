# Full Raw-Component Execution Report

## Result

**Verdict: D. INCOMPLETE.** The current-source raw components ran successfully
and produced a valid isolated candidate. They did not reproduce the historical
champion or establish the historical model lineage. No leaderboard submission
was made and no official score is claimed.

## Execution boundary

- Repository snapshot: `48b66288dfb262b1a1572bace1e70e41da573ef6`.
- Run directory: `runs/raw_full_gpu_20261031_235900` (identifier only; exact
  command times and durations are recorded in `execution.json` and logs).
- Raw inputs: `D:\gci\inpuy\train.csv` and `test.csv`; 171,202 training rows
  and 61,500 test rows.
- SHA-256: train
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`;
  test `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`.
- The isolated component directory was new, commands used full mode without
  `--fast`, and no historical prediction cache or checkpoint was read to
  generate candidate predictions.
- Generator commands, working directory, logs, exit codes, elapsed seconds,
  source hashes, dependency versions, raw-input hashes, and output hashes are
  recorded in `runs/raw_full_gpu_20261031_235900/execution.json` and
  `runs/raw_full_gpu_20261031_235900/logs/`.

## Generator results

| Component | Exit | Elapsed | Logged full-run validation | Output SHA-256 |
|---|---:|---:|---|---|
| XGB10 | 0 | 49.953 s | CUDA confirmed; OOF AUC 0.761020 | `oof_10f_xgb.npy`: `6d1b4c862fa07acd96d719c1d040fd6eb9daf63cbb0b7f51febedc8d50c8e0d4`; `test_10f_xgb.npy`: `d44e4e741d92ea2fff17deaba50fd58f22dad8378be5f68454e6c1a574534303` |
| V55 | 0 | 644.500 s | CatBoost GPU confirmed; seven members completed, CatBoost Product Specialist OOF AUC 0.750755 | `v55_members.npz`: `4e47c98870bd7aed9120976b174b76e46b81c260d8d1eb9f2d4cee228d3a9ca2` |
| V17 | 0 | 530.578 s | CatBoost GPU confirmed; final fusion OOF AUC 0.758336 | `v_17_full_suite_bundle.npz`: `323672967636c675eaf3c586778db16f685355b033afae58ac6bb688a996b56f` |
| Anchor74 (current Git source) | 0 | 1,657.844 s | XGBoost `cuda`, CatBoost `GPU`; output has 61,500 finite targets | `v_20261027_precision_transductive_apex.csv`: `ab541d868d193eea6426abf8924472890480498d7599102103fd6bb7302bb9b6` |
| Labels | n/a | generated during XGB10 | Exact equality with raw `TARGET` | `y_train.npy`: `2af2e56de31b22d326cd8fffe56b20cdd0b9c13cccddd0b4e8a50a97a5b984e2` |

Total recorded orchestration elapsed time was `2,894.797` seconds. Component
validation confirmed expected row counts, finite values, prediction ranges,
unique raw IDs, and exact generated-label equality.

The orchestration source was amended after its long-running process started
to add stricter prediction-range/scalar checks; that process retained the
pre-amendment code in memory. To close this provenance gap, the current
`validate_components` implementation was run independently against all
completed arrays after training (exit 0, `0.453` s), and the current strict
submission validator was rerun (exit 0, `4.734` s). Their source hashes,
commands, times, output hashes, and reports are recorded in
`runs/raw_full_gpu_20261031_235900/reports/post_run_checks.json`.

## Device evidence and warnings

- XGBoost CUDA and CatBoost GPU preflight passed. Runtime logs confirmed CUDA
  for XGB10, V17 CatBoost, Anchor74 XGBoost, and GPU for V55/Anchor74 CatBoost.
- The installed LightGBM build does not include GPU tree support. Its GPU
  preflight failed with exit code 1 and the explicit
  `GPU Tree Learner was not enabled in this build` error, recorded in
  `runs/raw_full_gpu_20261031_235900/logs/lightgbm_gpu_preflight.txt`.
  LightGBM members ran on CPU as configured, not via a silent fallback.
- Sampled NVIDIA telemetry contained 518 samples, peak utilization of 100%,
  peak memory of 4,970 MiB, and 7.15% average utilization across the entire run
  (including idle periods). The full sample log is retained at
  `runs/raw_full_gpu_20261031_235900/logs/gpu_telemetry.csv`.
- Anchor74 XGBoost logged an explicit device-mismatch warning in
  `predict_proba`: the CUDA booster received CPU pandas data and selected its
  DMatrix prediction path. The same run log confirms the XGBoost model
  completed on CUDA; this warning is disclosed rather than treated as proof
  of GPU inference.
- NumPy logged all-NaN warnings for rows with missing EXT_SOURCE fields. The
  component and final strict validators found finite prediction outputs.

## Interpretation

This is a successful full-mode execution of the **available current-source
generators**, not a full historical rebuild. The current Anchor74 source builds
143 features and does not save its feature matrices; Phase 3 evidence describes
669 features and a selected 550-feature input. The separate 695-feature
production-lineage report belongs to another source snapshot. Historical folds,
training settings, and fitted checkpoints are not fully recovered.

V20's original writer remains unavailable. Fifteen V080041 upstream prediction
or feature artifacts remain unavailable after recovering `meta.json` and
deriving labels/IDs from raw CSVs. Those branches are not direct inputs to the
v85 formula, but their lineage remains uncertified. See
`FULL_DEPENDENCY_GRAPH.md` and `MISSING_DEPENDENCIES.csv`.

All 10 audit unit tests passed, and `compileall` passed. Their logs, plus the
post-run validation/comparison commands, exit codes, runtimes, and
source/report hashes, are retained under
`runs/raw_full_gpu_20261031_235900/logs/` and
`runs/raw_full_gpu_20261031_235900/reports/post_run_checks.json`.
