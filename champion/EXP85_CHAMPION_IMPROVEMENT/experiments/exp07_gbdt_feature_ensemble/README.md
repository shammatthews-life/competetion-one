# EXP07 — GBDT signature features and nested blending

EXP07 tests one feature hypothesis rather than repeating the earlier generic
ratio, category-peer, category-encoding, polynomial-interaction, or raw-XGB
screens. It adds a fold-local repeated-application signature frequency and a
strictly cross-fitted signature target rate to the frozen Phase 2 feature
pipeline.

The signature uses two day-difference fields, gender, and education. It is
not an applicant ID or verified borrower identity. The historical Hasan
feature builder contains a related key and globally generated memory
features; EXP07 reimplements only the key idea and does not reuse historical
cached matrices or target-derived outputs.

See [FEATURE_INVENTORY.md](./FEATURE_INVENTORY.md) for availability,
duplication, rationale, and leakage classification.
See [RESULTS.md](./RESULTS.md) for both seed runs, the independent replication,
artifact hashes, and the final rejected decision.

## Validation and blend protocol

- Five exact outer folds per seed, verified against frozen Phase 2 fold
  artifacts (`20261101` and `42`).
- Inner validation is drawn only from outer-fit rows.
- Seed `20261101` uses the independently rebuilt strict Phase 2 reference
  from EXP06. Seed `42` refits the frozen Phase 2 models inside this runner,
  selecting iteration counts only on inner validation; the archived seed-42
  predictions are not used as baseline predictions.
- Each inner model and each final outer-fit model receives separately
  generated Phase 2 features and signature features. Target encodings for
  training rows are cross-fitted; validation/test encodings use only the
  corresponding fit partition's labels.
- LightGBM and CatBoost use the frozen Phase 2 configurations. LightGBM is
  explicitly CPU; CatBoost is explicitly GPU and must report the RTX 4050.
- A 50:50 LightGBM/CatBoost blend forms the signature-feature candidate. For
  each outer fold, a candidate-vs-Phase-2 blending weight is selected from a
  fixed 0.05 grid using only that fold's inner-validation AUC. The weight is
  frozen before scoring the outer validation fold.
- XGBoost is not repeated: two prior CUDA screens showed material standalone
  underperformance and negligible, inconsistent fixed-rank-blend lift.
- Promotion requires at least `+0.001` pooled OOF AUC and four of five
  positive fold deltas, with mean paired-fold gain exceeding two standard
  errors. A positive screen is only a replication candidate, not promotion.

## Run and verify

Use the isolated experiment environment:

```powershell
champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\.venv\Scripts\python.exe champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp07_gbdt_feature_ensemble\run_exp07.py --data-dir D:\gci\inpuy
champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\.venv\Scripts\python.exe champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp07_gbdt_feature_ensemble\verify_exp07.py --run-dir champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp07_gbdt_feature_ensemble\runs\exp07_signature_seed20261101_v1
```

The independent seed-42 replication uses a separate, new run directory:

```powershell
champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\.venv\Scripts\python.exe champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp07_gbdt_feature_ensemble\run_exp07.py --data-dir D:\gci\inpuy --outer-seed 42 --run-dir champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp07_gbdt_feature_ensemble\runs\exp07_signature_seed42_v2
champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\.venv\Scripts\python.exe champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp07_gbdt_feature_ensemble\verify_exp07.py --run-dir champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp07_gbdt_feature_ensemble\runs\exp07_signature_seed42_v2
```

The runner refuses to reuse a non-empty run directory, checks input/reference
hashes and exact fold indices, requires an idle-enough RTX 4050 before
training, and does not silently fall back from CatBoost GPU. Every run
preserves feature definitions, fold metrics, OOF/test predictions, candidate
submissions, GPU evidence, model checkpoints, logs, and a complete hash
manifest. No official leaderboard result is inferred from OOF validation.
