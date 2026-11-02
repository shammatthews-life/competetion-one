# EXP06 — FT-Transformer results

## Decision

**REJECTED.** The independently trained candidate did not meet the
predeclared advancement gate and must not replace the historical champion or
be submitted as an improvement. No official leaderboard score was obtained.

The complete run is in
`runs/exp06_ft_transformer_seed20261101_v5/`. Its OOF predictions, test
predictions, submission, fold indices and metrics, checkpoints, telemetry,
training-source snapshot, experiment summary, and SHA-256 manifest are
preserved there.

## Experiment

- Architecture: FT-Transformer with one token per numerical feature,
  categorical embeddings, a CLS token, two attention heads, and two
  Transformer encoder layers.
- Data: 171,202 permitted raw training rows and 61,500 raw test rows.
- Features: 122, generated from the frozen Phase 2 feature pipeline.
- Validation: the same five outer folds as the Phase 2, EXP04, and EXP05
  references. The 15% inner validation split came only from each outer-fit
  partition. It selected the epoch count; each final model was refit on the
  complete outer-fit partition. Outer validation labels were not used for
  stopping or selection.
- Model configuration: 16-dimensional tokens, feed-forward dimension 64,
  dropout 0.15, AdamW, learning rate 0.0003, batch size 2,048, maximum 30
  epochs, patience 8, and no mixed precision.
- CUDA: PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 4050 Laptop GPU.
  CUDA preflight and training succeeded; there was no CPU fallback.

## Fold results

| Fold | FT OOF AUC | Matched Phase 2 AUC | Delta | Selected epochs | Runtime (s) | Peak GPU | Peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.76124928 | 0.76606922 | -0.00481995 | 26 | 599.4 | 99% | 1,254 |
| 2 | 0.76775267 | 0.77263702 | -0.00488435 | 23 | 571.9 | 99% | 1,360 |
| 3 | 0.76705980 | 0.77142459 | -0.00436479 | 29 | 570.8 | 98% | 1,338 |
| 4 | 0.76277468 | 0.76853623 | -0.00576155 | 20 | 513.2 | 99% | 1,282 |
| 5 | 0.76926809 | 0.77330287 | -0.00403479 | 30 | 597.8 | 98% | 1,330 |

| Metric | FT-Transformer | Matched Phase 2 | Difference |
|---|---:|---:|---:|
| Pooled OOF ROC-AUC | 0.76402234 | 0.77034958 | -0.00632725 |
| Mean fold ROC-AUC | 0.76562090 | 0.77039399 | -0.00477308 |
| Paired folds improved | 0/5 | — | — |

The matched benchmark is the fixed 50:50 mean of the frozen Phase 2
LightGBM/CatBoost configurations. Each model's iteration count was selected
on inner validation and refit on the full outer-fit rows. The archived Phase
2 score of 0.77096382 used outer-fold early stopping and is therefore only a
protocol-mismatched diagnostic, not the decision baseline.

Secondary comparisons using their preserved OOF artifacts:

- EXP04 CatBoost Lossguide: 0.76803271; EXP06 delta -0.00401038.
- EXP05 residual MLP: 0.76372731; EXP06 delta +0.00029503, with only 2/5
  paired folds improved. This does not satisfy the matched-baseline gate.

The predeclared promotion gate required at least +0.001 pooled AUC and
improvement on at least four of five folds. Neither condition was met.

## Diagnostics and artifact validation

- Total run time: 2,860.8 seconds (47.7 minutes).
- FT GPU telemetry: 1,136 samples; peak utilization 99%; peak device memory
  reported by `nvidia-smi` 1,360 MiB; peak PyTorch allocation 742 MiB.
- OOF prediction distribution: mean 0.077805, standard deviation 0.082602,
  median 0.050021, 1st/99th percentiles 0.009952/0.413441.
- Test prediction distribution: mean 0.086282, standard deviation 0.088091,
  median 0.057117, 1st/99th percentiles 0.010938/0.419717.
- Correlation to the historical champion's test predictions is diagnostic
  only: Pearson 0.785235 and Spearman 0.905122. It was not used for training
  or selection.
- Fold-local inner validation AUC, selected epochs, outer-fold AUC, training
  runtime, and GPU/VRAM usage are recorded per fold. Standard tree gain
  importance is not applicable to this architecture; no attention map is
  presented as feature importance.
- Submission schema is exactly `SK_ID_CURR,TARGET`; all 61,500 IDs are unique
  and in sample-submission order. Predictions are finite and in [0, 1].
- Candidate submission SHA-256:
  `ab1ae5ed9616eed5730c16323a183049e5953f1e45be8d5862ccab4c0d16c12b`
- The independent audit verified 8 input/reference hashes, 5 source hashes,
  and 35 output hashes. The training-time runner snapshot matches the source
  hash recorded before training.
- Focused tests: 7 passed and `py_compile` passed. The repository-wide
  `git diff --check` also reports pre-existing trailing whitespace in
  unrelated dirty files; the staged EXP06-only diff is checked separately.

## Next step

Do not tune this FT-Transformer configuration. Keep the 0.801340 historical
champion immutable and evaluate a genuinely different candidate only under
the same inner-only, matched-fold protocol. The official 0.80314 target
remains unmet.
