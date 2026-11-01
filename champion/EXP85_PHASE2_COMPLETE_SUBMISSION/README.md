# EXP85 Phase 2 Complete Submission

This isolated package retrains the frozen independent Phase 2 recipe from the raw Home Credit `train.csv`, `test.csv`, and `sample_submission.csv`. It does not import historical EXP85 prediction caches or alter the frozen Phase 2 release.

## Selected recipe recovered from source

- Phase 1 derives the original row-level credit, annuity, income, family, external-score, Merton, and bureau features; it also decodes the 129.5 annuity fee and the archived `0.0132` credit/goods product tier.
- The selected seed-42 Phase 2 run uses pooled **unlabeled** train/test covariate frequencies for the fee decoder.
- Phase 2 keys are `FEE_Q3` (smoothing 100), `FEE_Q4` (30), `FEE_Q5` (30), and `PRODUCT_K` (50).
- Five stratified outer folds use seed 42. Outer-training rows get inner five-fold OOF encodings with seed 7. Outer-validation and test encodings use outer-fit labels only.
- LightGBM trains on CPU; CatBoost uses GPU 0. Both use the archived parameters, random seeds, and early stopping.
- The exact ensemble is the arithmetic mean of the two models' raw class-1 probabilities. There is no rank transform or post-processing.

For key `k` and fit-fold prior `p`, the encoding is:

`TE(k) = (sum(target for key k) + smoothing * p) / (count(key k) + smoothing)`

An unseen key receives `p`; each key also contributes `log1p(fit-fold count)`. The model fit rows receive inner-OOF encodings, excluding each row's own target.

## Run from the repository root

The ordered Jupyter front end is [EXP85_PHASE2_COMPLETE_SUBMISSION.ipynb](./EXP85_PHASE2_COMPLETE_SUBMISSION.ipynb). It calls the same validated modules and configurations in this package; it intentionally does not copy or fork model logic. Keep the notebook beside `source/`, `scripts/`, and `config/`. It is not a standalone single-file Kaggle notebook unless those dependencies are bundled with it.

The notebook's input preflight is safe to run on its own. Its final training cell performs a full five-fold run and creates a new timestamped candidate only after the frozen-reference gates pass.

```powershell
python -B champion\EXP85_PHASE2_COMPLETE_SUBMISSION\scripts\run_full_pipeline.py
```

Default raw inputs are read from `D:\gci\inpuy`. Override with `--data-dir`. Every invocation creates a unique, non-overwriting run folder. The full mode writes and verifies the candidate inside the run first; it copies the candidate to the package's `submissions/<run-id>/` only after the frozen reproduction gate passes.

```powershell
python -B champion\EXP85_PHASE2_COMPLETE_SUBMISSION\scripts\reproduce_validation.py
```

This second command performs the same five-fold raw-data retraining and reference comparison but does not promote a submission. It is intentionally not an inexpensive score-only shortcut.

Verify a completed run or submission independently:

```powershell
python -B champion\EXP85_PHASE2_COMPLETE_SUBMISSION\scripts\verify_pipeline.py `
  --run-dir champion\EXP85_PHASE2_COMPLETE_SUBMISSION\runs\<run-id>

python -B champion\EXP85_PHASE2_COMPLETE_SUBMISSION\scripts\verify_submission.py `
  --submission champion\EXP85_PHASE2_COMPLETE_SUBMISSION\submissions\<run-id>\submission_phase2_complete.csv

python -B champion\EXP85_PHASE2_COMPLETE_SUBMISSION\scripts\package_manifest.py verify
```

Install the exact validated Python package versions from `requirements.txt` only in a compatible environment. This pipeline does not silently fall back from GPU CatBoost to CPU. The available RTX 4050 is smoke-tested before model fitting, and GPU utilization/memory are sampled during each CatBoost fold.

## Outputs and provenance

Each timestamped run stores model files, fold indices, feature names/dtypes, OOF/test predictions for both component models and their blend, per-fold metrics, feature importances, submission, comparison report, raw-input/package/config hashes, hashes of the authoritative frozen source files, software/device evidence, and an output SHA-256 manifest. `package_manifest.py create` writes a package-level inventory; `package_manifest.py verify` fails on missing, extra, or changed files. The frozen candidate, frozen OOF arrays, and raw inputs are read-only references.

The original frozen seed-42 candidate has SHA-256 `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa` and documented OOF AUC `0.7715614423106346`. A new run is accepted only if its five folds match exactly, global OOF AUC differs by at most 0.001, every fold AUC by at most 0.003, and OOF/test/submission correlations to the frozen references are at least 0.99. The execution report records each value; the candidate is not required to be byte-identical because GPU training may be nondeterministic.

These are development-validation metrics, not an official competition score. Submission to the competition remains a separate user decision.
