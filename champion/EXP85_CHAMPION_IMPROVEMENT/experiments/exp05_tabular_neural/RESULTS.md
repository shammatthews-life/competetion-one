# EXP05 results

## Decision

**Rejected.** The independent residual MLP candidate did not improve either
OOF reference. It is not a replacement for the historical champion, and no
official leaderboard score was obtained.

## Candidate

- Run: `runs/exp05_cuda_seed20261101_v1/`
- Data: 171,202 raw training rows and 61,500 raw test rows.
- Features: 122 rowwise and fold-local Phase 2 features; categorical
  embeddings with fold-fit numeric preprocessing.
- Validation: five matched Phase 2/EXP04 outer folds; inner-only epoch
  selection, followed by refitting on each complete outer-fit partition.
- OOF ROC-AUC: **0.7637273078**.
- Fold ROC-AUC: `0.7595053037`, `0.7682467965`, `0.7673489463`,
  `0.7653673122`, `0.7686943193`.

| OOF comparator | Reference AUC | Candidate delta | Folds improved |
|---|---:|---:|---:|
| Phase 2 fold-local blend | 0.7709638204 | -0.0072365126 | 0/5 |
| EXP04 CatBoost Lossguide | 0.7680327122 | -0.0043054044 | 1/5 |

Phase 2 used outer-validation early stopping and is therefore not a perfectly
method-matched comparator; its score may be optimistic. EXP04 used inner-only
stopping. No outer-validation label was used for EXP05 feature generation,
preprocessing, stopping, or fitting.

## Execution and reproducibility

- Device: NVIDIA GeForce RTX 4050 Laptop GPU, 6,140 MiB reported memory.
- Runtime: 293.8 seconds total; five folds ran sequentially.
- GPU telemetry: 409 samples, peak utilization 85%, peak device memory used
  281 MiB; PyTorch peak allocated memory 113 MiB.
- CUDA environment: PyTorch `2.11.0+cu128`, CUDA build `12.8`; preflight GPU
  forward/backward smoke fit passed. CPU fallback is disabled.
- Fixed split/model seeds, deterministic PyTorch algorithms, and deterministic
  CUDA GEMM configuration were used. Four focused tests passed, including
  seeded model-initialization reproducibility. A second full training run was
  not performed.
- All 41 source/input/output records in the run manifest verified against
  their SHA-256 values.
- Raw input SHA-256: train
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`,
  test
  `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`,
  sample submission
  `6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db`.

## Submission artifact

The candidate CSV has 61,500 unique, ordered test IDs and columns `SK_ID_CURR`,
`TARGET`. Prediction range is `[0.0051131994, 0.5315931439]`; SHA-256:
`b89423e99f1255ab90b6252d97e04e8d3d5cb7b783152f8f6d0877a6409b678d`.

The test predictions correlate with the frozen champion at Pearson `0.80395`
and Spearman `0.90934`; this is diagnostic only and was not used for training
or model selection. EXP05's OOF predictions have mean `0.08093`, standard
deviation `0.08339`, and median `0.04983`; test predictions have mean
`0.08961`, standard deviation `0.08825`, and median `0.05695`.
