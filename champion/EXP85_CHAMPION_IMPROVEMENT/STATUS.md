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
- Prior raw-XGBoost screens were reviewed rather than repeated: standalone OOF
  AUCs were `0.7613696` and `0.7602661`; their fixed 80:20 diagnostic blend
  lifts over Phase 2 were only `0.0001020` and `0.0000251`, inconsistent and
  not promoted. No extra seed or weight sweep was run.

## Next action

Do not tune the rejected EXP01 experiment. EXP02 now implements an independent
raw-data surrogate OOF screen because a verified, leakage-safe historical
Anchor74 OOF stream remains unavailable. It is not a historical reconstruction
and cannot produce a submission.

## EXP02: raw-data Anchor74 surrogate OOF

- Implemented the five-family raw-data OOF runner and fold-local graph
  calibration in `experiments/exp02_anchor74_oof/`.
- The existing raw feature builder reports 143 features but returns 695. The
  runner now validates 695 and documents the source discrepancy.
- Replaced row-wise folds with `StratifiedGroupKFold` using the exact entity
  key consumed by the repeat-borrower graph. Full raw-data preflight produced
  five validation folds with zero entity-group overlap; fold sizes were
  34,240-34,241 and validation positive rates were approximately 0.0807.
- Eight workspace tests pass, including grouped-fold isolation and ensuring
  held-out labels do not affect fold-local graph calibration. Python AST
  parsing passes.
- Two prior partial runs are preserved and explicitly marked
  `ABORTED_INVALID_PARTIAL` with their retained log/telemetry hashes. Their
  fold AUCs are not accepted as experiment results.
- Historical suite caches may be used only for clearly labeled hybrid
  diagnostics; their fold lineage is unverified. The completed run below
  produced no candidate or submission.
- Raw competition files remain outside the repository. The OOF archive does
  not serialize raw IDs, entity keys, or target labels.

### Completed run: `20261101T183230Z`

- All five grouped folds completed in 1,101.1 seconds. Per-fold group overlap
  was `[0, 0, 0, 0, 0]`; all model streams had complete, finite OOF coverage.
- Pooled AUCs: LightGBM conservative `0.7581163`, LightGBM ExtraTrees
  `0.7566237`, XGBoost `0.7566817`, CatBoost `0.7559647`, LightGBM deep
  `0.7551001`, and graph-calibrated surrogate anchor `0.7597101`.
- The 85:15 surrogate-anchor/historical-suite diagnostic blend scored
  `0.7643409`; the cross-fit two-stream logistic diagnostic scored
  `0.7771967`. These figures are not a validated full-EXP85 comparison:
  historical suite OOF fold provenance is unavailable.
- XGBoost and CatBoost CUDA preflights passed. Telemetry recorded 1,092
  samples, 100% peak utilization, and 4,903 MiB peak memory with no monitor
  error. LightGBM remained CPU-based.
- Reported output hashes, all five fold index sets, raw-input/source hashes,
  group isolation, prediction finiteness, and the archive's omission of IDs,
  group keys, and labels were independently verified.
- Outcome remains `DIAGNOSTIC_ONLY_NO_CANDIDATE`; official score is
  `NOT_EVALUATED`. No submission was generated.

### Next concrete action

Do not tune or promote EXP02. The next valid historical 85:15 evaluation
requires leakage-safe OOF predictions from the actual Anchor74 upstream stack
and verified fold lineage for all historical suite OOF streams, or independent
retraining of those historical producers. Until then, keep EXP02 diagnostic
only and do not claim historical Anchor74 parity or an official score.
