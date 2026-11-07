# Status

## Current verdict

**GC-01, TE-01 and both XGBoost screens are complete; none is promoted. E-INT-01 is the next predeclared feature screen against Phase 2.**

## Verified

- Repository clone was clean before work at `0e1bcbea881f926b07aaa6916328cfd242684a51`, matching `origin/main`.
- Frozen Phase 2 verifier passes: 129 release files, candidate SHA `2470cecc...7411fa`, raw input hashes, schema, IDs/order and `[0,1]` predictions.
- Historical champion SHA on disk matches `78c31bb4...e6a947`.
- Hybrid V5 candidate SHA on disk matches `91aa95c9...b792482`.
- Raw input hashes match the frozen release: train `b8dd5a61...c458ecf1`, test `dfb7aeb9...d2d42c65`, sample `6e5ad587...dffe7db`.
- NVIDIA GeForce RTX 4050 Laptop GPU is present (6,141 MiB, driver 592.82). Current Python 3.12.10 has NumPy/Pandas/SciPy/sklearn/XGBoost/CatBoost/LightGBM; PyTorch and TabM absent.
- Historical original folders `D:\gci\champion` and `D:\gci\gemini` have no Git metadata; clone history is available and was inspected.
- 18 distinct named pipelines/experiment families are catalogued with lineage qualifications in `HISTORICAL_PROJECT_INVENTORY.md`.
- Existing forensic result classifies V5 as hybrid; raw Track1B at Spearman `0.8921289`; Phase 3 final source has historical cache dependencies.

## Completed controlled work

- Stage 0 read-only diagnostic compared Phase 2 and hybrid V5 to the historical champion, aligned by all 61,500 raw test IDs.
- Corrected Stage 0 run includes all raw test covariates for missingness slices, exact input/source hashes, and output hashes.
- The complete research test suite currently passes (16/16); the frozen nested target-encoding tests pass (3/3).
- The new 76-feature peer-context builder passed a real-data first-fold preflight in 9.35 seconds; it uses only outer-fit covariates and leave-one-out means for training rows.
- GC-01 five-fold seed-42 screen is complete and rejected: OOF AUC fell by `0.0016302` on all five folds; exact-fold, OOF and candidate integrity checks passed. GPU utilization peaked at 57% and VRAM use at 4,903 MiB.
- TE-01 core categorical nested target-encoding screen is complete in a unique seed-42 run directory. It scored `0.7709211321` OOF AUC (`-0.0006403102` vs Phase 2), with 1/5 positive folds. Independent run-manifest, fold, OOF, candidate, GPU, and target-encoding test checks passed.
- Package SHA-256 manifest was generated and verified after the experiment and report updates.

## Completed model-diversity screen: XGB-01/02

- XGBoost 3.4.1 reports `USE_CUDA=true`; a direct `device=cuda:0`, `tree_method=hist` smoke fit passed on the RTX 4050 (CUDA 13.3 build, driver 592.82).
- Full real-data feature preflight used the frozen Phase 2 rowwise builder: 171,202 train rows, 61,500 test rows, 102 rowwise columns; fold-1 matrices have 201 features and 132.7 MiB total CSR storage. Preflight completed in 1.56 seconds. Process working set was 613 MiB at snapshot, peak 883 MiB; private bytes were 1.34 GiB.
- The raw-data directory has the three expected competition CSVs plus an unrelated image; no relational Home Credit tables. The supplied `files.zip` contains only `fast_screen.py` and `gci_core.py`; `final_submission.rar` contains a README, a replay script and a historical submission CSV, not auxiliary input tables. Neither archive's historical prediction/replay artifacts are XGB-01 inputs.
- Current Python has no PyTorch; TabM therefore remains feasibility-blocked pending an isolated dependency/environment plan. No package was installed.
- XGB-01 standalone OOF AUC was `0.7613696` vs Phase 2 `0.7715614`; the fixed 80/20 rank blend gained only `+0.0001020` (3/5 folds).
- The one predeclared replication on seed `20261101` scored XGBoost `0.7602661` vs Phase 2 `0.7709638`; the unchanged blend gained `+0.0000251` (2/5 folds). Residual correlations were about `0.994` on both seeds. No promotion or further seed/configuration sweep.
- Both CUDA runs completed in about 30.7 seconds each. Six telemetry samples per run showed maximum utilization 77% / 69%, peak VRAM 318 MiB, and all folds reported device `cuda:0`. Process peak working set was about 1.12 GiB; peak private bytes about 1.73 GiB. Both run manifests and candidate integrity checks passed.
- The final archived submission replay code was inspected: it loads precomputed `group_a_oof_10stream.npy` and `group_a_test_10stream.npy`. This is evidence for historical cached replay, not an independent raw model result.
- E-INT-01 will now test five predeclared EXT_SOURCE/affordability interactions on the exact Phase 2 seed-42 folds. Its advancement gate is fixed before training: mean paired AUC gain `>=0.0002`, at least 4/5 positive folds, and mean gain greater than twice its standard error. Only a pass earns one independent-seed replication.

## Explicitly not done

- No new model candidate was promoted and no leaderboard score was claimed; the frozen Phase 2 candidate remains unchanged.
- No historical/reference/test prediction used as a training target or tuning signal.
- No historical cache arrays were copied into this workspace; model outputs here were generated by the recorded experiments.
- No change to historical champion or frozen Phase 2.
- The read-only diagnostic did not need GPU computation. GC-01 and TE-01 used explicit GPU CatBoost and CPU LightGBM, with timestamped utilization/VRAM sampling. TE-01 recorded 40 samples, max 58% utilization, and peak 4,903 MiB VRAM.

## Blockers to full historical raw rebuild

Missing or unverified: ID sidecars for historical suite arrays, exact historical producer source/config/folds for V17/V55/XGB10, the complete raw ancestry for the prior Level-2/optimal Anchor inputs, and a platform record binding 0.801340 to an exact upload hash.

## Stage 0 result

The final source-verified run is `runs/20261102T003000Z_stage0_final/` and evaluated all 33 test columns without a target. Phase 2 versus the historical champion has Pearson `0.7782506970`, Spearman `0.9157084581`, MAE `0.41111346`, and RMSE `0.46865093`; hybrid V5 versus the champion has Pearson/Spearman `0.9999975816`, MAE `0.0003014174`, and RMSE `0.0006348771`. This confirms the known gap and hybrid replay, not score reproduction by Phase 2.

An earlier immutable run, `runs/20261102T000000Z_correlation_gap_stage0/`, is superseded because its missingness slice counted only a selected subset of covariates. Prediction-wide metrics were unaffected. Do not use that run's grouped slices.

## GC-01 result

The immutable run `runs/20261101T141000Z_phase2_group_context/` uses 76 peer-group features. Phase 2 OOF AUC was `0.7715614423`; the extended candidate scored `0.7699312062` (`-0.0016302361`), with paired fold deltas negative 5/5. Both individual model OOF AUCs also declined. This feature block is rejected; no second-seed run is warranted. CPU process peak RAM was not captured; GPU VRAM and utilization were sampled.

## TE-01 result

The immutable run `runs/20261101T142000Z_core_category_te/` adds nested smoothed encodings for four core categories. Phase 2 OOF AUC was `0.7715614423`; TE-01 scored `0.7709211321` (`-0.0006403102`), with 1/5 paired folds positive. Independent verification confirms exact folds, run-manifest hashes, recomputed OOF metrics, candidate schema/IDs/probability range, and 40 GPU telemetry samples (58% max utilization; 4,903 MiB peak VRAM). The screen is rejected; no second seed is warranted. No official score was evaluated.
