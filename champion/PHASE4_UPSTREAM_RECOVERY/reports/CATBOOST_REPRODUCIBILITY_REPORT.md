# CatBoost Fold-5 Reproducibility Audit

## Scope and input identity

This was a bounded diagnostic, not a full EXP85 retraining. The run trained
five same-seed GPU repeats and one CPU control on fold 5.

- Run evidence: `runs/catboost_fold5_gpu5_20261031_222432/execution.json`
- Total run time: 230.609 seconds; five GPU fits plus one CPU fit.
- CatBoost 1.2.10; Python 3.12.10; RTX 4050 Laptop GPU, driver 592.82.
- Training matrix: `(171202, 550)`; test matrix: `(61500, 550)`.
- Historical and current train/test matrices were exact; feature order SHA-256:
  `bfe1b359e4bbdaa6d4114571300fc75f494910c96d279c16e2ce5f84df483bd4`.
- Labels match raw `train.csv`; feature-row maps match raw train/test ID order.
- Split: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, fold 5;
  CatBoost model seed 46.
- Requested parameters: 750 iterations, learning rate 0.04, depth 6,
  L2 5, Bayesian bootstrap, bagging temperature 1, Logloss, early stopping 60.
- Device evidence: model metadata reports GPU for all five GPU fits and CPU for
  the CPU control. GPU telemetry and model/prediction hashes are in the run
  evidence; large model and array files are intentionally not committed.

## Five GPU repeats

| Measure | Observed range |
|---|---:|
| Validation AUC | 0.7536403014–0.7536404623 |
| Pairwise GPU validation Spearman | minimum 0.9999999977 |
| Pairwise GPU test Spearman | minimum 0.9999999977 |
| Maximum pairwise GPU validation MAE | 5.74e-7 |
| Maximum pairwise GPU test MAE | 6.05e-7 |
| Maximum pairwise GPU test absolute difference | 3.26e-4 |
| Maximum pairwise GPU test rank displacement | 53 |

Compared to the historical OOF reference, all five runs are essentially
identical in rank (Spearman `0.9999999981` to `0.9999999997`; MAE
`1.61e-7` to `4.77e-7`). Compared to the current candidate OOF, their
correlations are approximately `0.99975172` and MAE approximately `0.0010874`.
Thus these five repeats consistently reproduce the historical-like result,
not the candidate-like result.

## CPU control and causal limits

The CPU control's validation AUC is `0.7490954`; its validation Spearman against
GPU repeat 1 is `0.9873858` (MAE `0.0080859`). This is not a clean device-only
comparison: the effective CPU and GPU settings differ, including border count
(254 CPU, 128 GPU), and CPU-only defaults such as model-shrink and sampling
settings. Requested parameters alone do not eliminate these effective
differences.

The earlier two-repeat run reportedly produced one historical-like and one
candidate-like fit. This five-repeat run did not reproduce that split. The new
evidence therefore supersedes any claim that GPU nondeterminism is established
as the cause. It remains a hypothesis, not a proven explanation.

This evidence supersedes any earlier report wording that presents CUDA warp
reduction as the established cause; the historical Phase 4 report remains
preserved at
[`PHASE4_ZERO_GAP_OPTIMIZATION_REPORT.md`](../../03_PHASE_HISTORY/PHASE_4/PHASE4_ZERO_GAP_OPTIMIZATION_REPORT.md).

## Conclusion

**CatBoost exact historical reproducibility is not established, and the cause
of the prior divergence remains unknown.** Inputs and row alignment are
strongly verified; five GPU repeats are stable, but historical model/runtime
provenance and an apples-to-apples control are missing. No full model suite or
blend was retrained or tuned.
