# Controlled Experiment Roadmap

## Shared protocol

- Frozen baseline: Phase 2 candidate and source snapshot under `EXP85_PHASE2_FROZEN_RELEASE`.
- Primary metrics: saved OOF ROC-AUC, mean per-fold AUC, paired per-fold delta, and a predeclared stability summary.
- Folds: persist exact fold indices; never mix a candidate with baseline predictions from different folds.
- Encoding/selection/stacking/calibration: fit entirely inside outer training; inner OOF for any target-derived or meta features.
- Inputs: record SHA-256 for raw train/test/sample and every source/config/artifact.
- Outputs: OOF and test prediction arrays, IDs, fold assignments, per-fold metrics, logs, wall time, device and observed GPU evidence.
- One change per experiment; every run folder unique. Candidate/test-prediction similarity to the historical champion is diagnostic only and never a selection target.
- Promotion requires at least one independent outer seed; improvements must be consistent and pass leakage/integrity tests. The baseline candidate remains frozen regardless.

## Stages

| Stage | Experiment | Why now | Single controlled change | Validation and stop rule | Cost |
|---:|---|---|---|---|---|
| 0 | Historical/test correlation gap slices | Required before new training; reveals where candidate families disagree | No training; compare Phase 2 and V5 to champion by verified IDs and test-covariate slices | Verify hashes/ID sets; no labels; report descriptive metrics only | Seconds, CPU |
| 1 | Historical source and cache ID-binding audit | Resolves largest lineage blocker | Add provenance, not model changes | No memberwise alignment claims unless producer order/IDs are verified | Low–medium |
| 2 | Third-seed Phase 2 stability | Two prior seeds already pass; check another independent split before broader experiments | Same features, models, blend; seed only changes outer folds | Matched baseline, per-fold deltas, leakage tests; reject unstable lift | ~20–25 min prior gate; GPU CatBoost |
| 3 | One Phase 2 feature-block ablation | Quantify actual contribution after a third-seed baseline | Remove exactly one block: fee decoder, product key, or target-encoding family | Same folds/models and same code except feature block; paired OOF comparison | Medium |
| 4 | Nested historical 550 selector benchmark | Check fidelity versus unbiased validation separately | Nested feature selection only | Outer labels never influence selected columns; compare to fixed-screen arm | High; CPU under current LightGBM |
| 5 | Raw XGBoost complementarity | Only if no leakage and the selected feature baseline is stable | Add one raw XGB model with Phase 2 features; no weight tuning on outer validation | OOF meta-stacking in inner folds; compare Phase 2 and previous best on same splits | Medium; CUDA, memory preflight |
| 6 | Missingness interaction ablation | Historical regime evidence exists | Add one predeclared EXT_SOURCE missingness feature block | Same folds; no test/champion guidance | Low–medium |
| 7 | TabM feasibility, then model study | Published modern architecture but no local Torch | Feasibility-only environment/RAM pilot, then full candidate if feasible | No install or training until CUDA/PyTorch plan is approved and memory budget proven | High; GPU, 6 GiB |
| 8 | Graph/transductive methods | Historical source suggests links; not validated | Only one source-verified graph signal | Fold-local graph; test-covariate-only; strict negative controls | Medium/high |

## Current first execution

Stage 0 is complete as a diagnostic-only run. It verified ID integrity and quantified the prediction gap before any experiment could be tempted to optimize toward the historical test vector. This is not model training and cannot promote a candidate.

## GPU execution gate

Before any GPU experiment:

1. Capture `nvidia-smi` device name, driver, total/free memory.
2. Confirm library build/version and requested device.
3. Run a tiny explicit-device preflight in a unique temporary location; verify the model's training log/config actually says CUDA/GPU.
4. Sample utilization/memory during real training and save timestamped evidence.
5. Abort on initialization/device error; no CPU fallback or reused failure output.

Current audit: XGBoost/CatBoost packages installed; PyTorch absent; LightGBM GPU learner absent. The RTX 4050 has about 6 GiB. No model was trained during Stage 0.
