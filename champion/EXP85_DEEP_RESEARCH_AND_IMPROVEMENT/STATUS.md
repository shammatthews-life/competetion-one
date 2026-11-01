# Status

## Current verdict

**Historical review completed to evidence available; the ID-aligned correlation-gap diagnostic ran successfully; no new model training was justified in this first stage.**

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

## In progress

- Stage 0 read-only diagnostic compared Phase 2 and hybrid V5 to the historical champion, aligned by all 61,500 raw test IDs.
- Corrected Stage 0 run includes all raw test covariates for missingness slices, exact input/source hashes, and output hashes.
- Four synthetic diagnostic tests and Python syntax checks pass.
- Package SHA-256 manifest generation and verification are implemented; final manifest remains to be generated after documentation and code are frozen.

## Explicitly not done

- No model trained, candidate selected, or leaderboard score claimed.
- No historical/reference/test prediction used as a training target or tuning signal.
- No cache arrays copied into this workspace.
- No change to historical champion or frozen Phase 2.
- No GPU computation was needed for this read-only diagnostic.

## Blockers to full historical raw rebuild

Missing or unverified: ID sidecars for historical suite arrays, exact historical producer source/config/folds for V17/V55/XGB10, the complete raw ancestry for the prior Level-2/optimal Anchor inputs, and a platform record binding 0.801340 to an exact upload hash.

## Stage 0 result

The final source-verified run is `runs/20261102T003000Z_stage0_final/` and evaluated all 33 test columns without a target. Phase 2 versus the historical champion has Pearson `0.7782506970`, Spearman `0.9157084581`, MAE `0.41111346`, and RMSE `0.46865093`; hybrid V5 versus the champion has Pearson/Spearman `0.9999975816`, MAE `0.0003014174`, and RMSE `0.0006348771`. This confirms the known gap and hybrid replay, not score reproduction by Phase 2.

An earlier immutable run, `runs/20261102T000000Z_correlation_gap_stage0/`, is superseded because its missingness slice counted only a selected subset of covariates. Prediction-wide metrics were unaffected. Do not use that run's grouped slices.
