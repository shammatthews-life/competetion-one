# Model Card: Independent EXP85 Phase 2 Candidate

## Status and intended use

Status: **FROZEN_DEVELOPMENT_CANDIDATE**.

This model is an independent raw-data development candidate for the GCI World 2026 Home Credit Default Risk task. It is preserved for reproducibility, inspection, and a future user-directed competition decision. It is not certified as an official competition winner and has no official evaluation result.

## Model and data

- Prediction target: binary `TARGET`.
- Candidate output: 61,500 application IDs with probabilities in `[0, 1]`.
- Models: fixed 50/50 probability blend of LightGBM and CatBoost.
- Feature source: raw application train/test columns; engineered fee/product features plus nested smoothed encodings of decoded ratio/product keys.
- Original outer validation: five-fold stratified CV, seed 42.
- Nested target encoding: five inner folds, seed 7; outer-training rows use inner OOF encodings and validation/test rows use outer-training labels only.
- Fee ratio frequency construction in the selected original candidate pools unlabeled train/test covariates (transductive). A second-seed fold-local sensitivity test retained nearly all the gain.
- Historical prediction files are not model inputs.

Exact raw-input locations and hashes are in `HASHES/INPUT_HASHES.json`. The datasets themselves are excluded from this Git release.

## Performance evidence

- Original seed-42 OOF AUC: `0.7715614423106346`.
- Matched raw-only baseline OOF AUC: `0.7625457907047022`.
- Original-seed difference: `+0.0090156516059324`.
- Second-seed raw baseline: `0.7620389005999739`.
- Second-seed pooled Phase 2: `0.7711413866258005` (`+0.0091024860258265`; 5/5 folds positive).
- Second-seed fold-local Phase 2: `0.7709638203765881` (`+0.0089249197766141`; 5/5 folds positive).
- Fold-local frequency decoder versus disabled control: `+0.0031900391533038` OOF AUC, 5/5 folds positive.

These are development cross-validation metrics. The second-seed test is a stability check, not an official test-set label evaluation.

## Limitations and cautions

- No official submission or competition evaluation was performed; the official score is `NOT_EVALUATED`.
- This candidate does not reproduce or claim the historical EXP85 score of `0.801340`.
- The second-seed variants are validation evidence and do not replace the original selected candidate.
- Pooled unlabeled covariate frequencies are transductive; results may not transfer to unrelated cohorts.
- The candidate's high or low correlation to another submission does not establish prediction equality, model lineage, or leaderboard equivalence.
- Model checkpoints were not included in this archive; use the documented code and raw inputs to retrain in a separate workspace.

## Integrity

Frozen candidate SHA-256:
`2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`.

The required verification command and the contents covered by the freeze manifest are documented in `README.md` and `FREEZE_MANIFEST.json`.
