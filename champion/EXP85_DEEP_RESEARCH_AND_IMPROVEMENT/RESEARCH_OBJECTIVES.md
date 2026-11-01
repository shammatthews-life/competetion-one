# Research Objectives and Guardrails

## Objective

Recover the EXP85 component lineage at artifact level, explain why raw and hybrid candidates diverge, then use a fixed, leakage-safe protocol to improve the frozen independent Phase 2 baseline. Preserve useful historical reconstruction as a separate fidelity track.

## Research questions

1. Which exact producer, source revision, feature schema, fold assignments, parameters, hardware, and artifacts created each historical component?
2. Which components are actually regenerated from raw `train.csv`/`test.csv`, which are reconstructed post-processors, and which remain historical cache replays?
3. What portions of the historical test-prediction gap can be explained by validated, ID-aligned components versus unverified positional arrays?
4. Which independent feature/model additions improve AUC across multiple fixed seeds without compromising label isolation?

## Track separation

- **Independent model track:** raw train/test features, fold-safe OOF, frozen Phase 2 baseline, no historical predictions as features or training targets.
- **Historical fidelity track:** cached historical components may be used for replay and audit only; every output is labeled `HYBRID_REPLAY` or `ARTIFACT_REPLAY_ONLY` where appropriate.
- **Prediction-gap diagnostics:** historical submissions may be compared by ID as descriptive outputs only. They are not ground-truth labels, cannot be used to fit, tune, select, or calibrate a model.
- **Official evaluation:** no official score is inferred without an immutable competition submission record.

## Hard protection rules

1. Do not edit the historical champion, frozen Phase 2 release, or prior successful correlation runs.
2. Never save new results into existing experiment directories.
3. Before any training, verify raw input hashes, source/config hashes, fold identity, target isolation, output path uniqueness, and hardware preflight.
4. Validation labels are used only for per-fold metrics. Feature selection, target encoding, calibration, and stacker fitting stay inside training folds.
5. Test labels do not exist in this dataset and must never be synthesized from submissions or the historical score.
6. Disable silent CPU fallback. A model configured for CUDA must fail before training if the actual CUDA backend is unavailable.
7. Keep each experiment to one scientific change; save OOF/test vectors, IDs, fold assignments, per-fold metrics, environment, timings, and hashes.
8. Reject one-seed or inconsistent-fold improvements as provisional; do not promote an experiment based on historical-test prediction similarity.

## Acceptance criteria

An independent candidate can only be promoted if it:

- beats the matched Phase 2 raw-only baseline on predeclared folds;
- repeats on an independent outer seed, with per-fold deltas and no unexplained negative fold;
- passes unit tests for label isolation and artifact integrity;
- uses explicit, recorded device configuration and observed GPU evidence when a GPU path is claimed;
- reproduces all saved summaries from OOF arrays;
- has a unique ID-aligned submission with finite `[0,1]` predictions and a SHA-256.

These are development-validation criteria, not leaderboard guarantees.
