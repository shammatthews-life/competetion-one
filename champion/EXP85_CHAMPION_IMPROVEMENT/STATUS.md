# EXP85 Champion Improvement Status

## Objective

Keep the historical EXP85 implementation and references immutable while
testing isolated, leakage-aware improvements toward a public ROC-AUC of
`0.80314`. No official competition submission is made automatically.

## Completed

- Snapshotted the historical final-synthesis source, seven direct input
  artifacts, upstream source files, and frozen champion reference.
- Verified the cached `y_train.npy` labels against raw `train.csv`; raw
  competition files were hashed, not copied into the workspace.
- The derived label array is locally available for reproducibility but
  explicitly excluded from Git; raw competition rows are not committed.
- Replayed the historical 85:15 synthesis from the snapshot. All 61,500 test
  IDs matched the raw test order, the predictions were valid, and the output
  was byte-identical to the frozen champion:
  `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`.
- Ran the first predeclared five-fold, outer-fit-only logistic meta-stacker
  screen over the eight cached OOF streams. It was **rejected**:
  - Fixed-suite pooled OOF AUC: `0.7772070078`
  - Cross-fit meta pooled OOF AUC: `0.7771115234`
  - Mean paired-fold delta: `-0.0000813250` (SE `0.0000494806`)
  - Positive folds: `1/5`
  - Test candidate: not generated
- Ran focused tests: `3 passed`; Python compilation passed.

## Limitations

- Baseline replay uses historical predictions and does not retrain upstream
  models; it is not an independent raw-data reconstruction.
- Historical OOF fold sidecars and producer manifests are unavailable.
- Anchor74 has no verified aligned OOF vector, so the first experiment cannot
  validate a change to the full 85:15 ensemble.
- Historical fixed suite weights were selected using the complete OOF labels;
  the fold comparison is not fully nested.
- The historical `0.801340` score is reported by project evidence but is not
  independently tied to the frozen CSV hash by an official upload receipt.
- The checked research virtual environment has CPU-only PyTorch; no GPU neural
  experiment was executed.

## Next action

Do not tune the rejected experiment. First seek a defensible fold-safe OOF
Anchor74 stream or specify a genuinely new raw-data model with a valid
full-stack evaluation. Any candidate must remain isolated and must not be
submitted automatically. Until then, the improvement target is unverified.
