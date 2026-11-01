# Model Card — Independent Phase 2 Pipeline

## Intended use and status

The package implements the independently trained Phase 2 development candidate for binary Home Credit `TARGET` prediction. It produces a test submission only after raw-input, frozen-reference, and submission-integrity checks pass. It has no official competition evaluation and must not be described as the historical EXP85 reconstruction.

## Inputs and features

The pipeline reads the original raw training, test, and sample-submission CSVs. It validates their schemas, target, ID uniqueness/order, and pinned SHA-256 hashes. It retains source rows and missing-value behavior. The exact Phase 1 features and the selected pooled unlabeled fee-frequency computation are described in `config/feature_config.json` and source.

The four Phase 2 target encodings are nested fold-safe encodings for rounded fee ratios Q3/Q4/Q5 and decoded product tier K. Inner-training rows are encoded out-of-fold; outer validation and test rows are encoded from outer-training labels only. No test target exists or is consumed.

## Models and ensemble

- LightGBM: binary GBDT, CPU, five outer folds, seed `42 + fold`.
- CatBoost: GPU 0, five outer folds, seed `42 + fold`.
- Both models use the archived hyperparameters and 150-round early stopping.
- Test predictions are averaged across the five fold models for each learner.
- Final output is `0.5 * LightGBM probability + 0.5 * CatBoost probability`; no post-processing.

Full parameter values are in `config/model_config.json`. Actual fold device and GPU telemetry are recorded per run.

## Validation

The baseline evidence is seed-42 five-fold OOF AUC `0.7715614423106346`, compared with matched raw baseline `0.7625457907047022`. A separate second-seed gate previously reported pooled Phase 2 AUC `0.7711413866258005` and fold-local Phase 2 AUC `0.7709638203765881`. These are prior results, not claims that a new run reproduced them.

Each new execution retrains from raw data, compares its exact fold assignments, OOF metrics and predictions, test distribution, and submission with the immutable frozen reference, and records the result. It does not use prediction arrays as training inputs.

## Limitations

- The selected recipe pools train/test covariates to compute fee ratio counts (transductive, unlabeled frequency features).
- CatBoost GPU training may vary slightly by driver/runtime; reproduction tolerances are explicit.
- AUC and file integrity do not establish an official leaderboard score.
- The frozen candidate and historical-cache reconstruction remain unchanged.
