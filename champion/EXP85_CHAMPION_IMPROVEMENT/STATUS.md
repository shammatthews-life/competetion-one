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

## EXP03: historical expanded-zoo dependency rebuild

- Dependency tracing found that the historical `hasan_expanded_zoo.npz` is
  present (SHA-256
  `73951fa46ea73403460d62c0427d4e3b696d935230ac74154600902e60c84df4`) but
  lacks producer-run provenance. The direct EXP85 inputs were present; this
  is an unverified artifact, not a missing file.
- Added an isolated generator using the four hash-pinned historical feature,
  feature-selection, and model-zoo scripts. It creates a new cache, extracts
  labels directly from raw `train.csv`, and fails closed if any historical
  base-model prediction cache is present before training.
- Added expected feature/NPZ schemas, source and raw-input hash gates,
  fold-index manifests, fixed split configuration, GPU preflights, telemetry,
  array/feature parity metrics, downstream path isolation, and provenance
  tests.
- Six focused tests and Python compilation passed. XGBoost 3.4.1
  (`cuda:0`) and CatBoost 1.2.10 (`GPU`) both passed preflight on the RTX 4050.
- The raw-data rebuild completed in 1,019.2 seconds. Every stage exited 0.
  The generated 11-key NPZ is valid, finite, has the expected 171,202/61,500
  row vectors and matches raw label order. Its SHA-256 is
  `0a619389f223a4cd93779f61b9d4bd7b600335d2fcf986afbcbf4eacf181e879`.
- Generated 669-feature and selected-550 train/test matrices match the
  historical references exactly in values, column order, dtypes, NaNs, and
  infinity positions. The source intentionally emits 34 train / 16 test
  infinities in `HOUSING_STABILITY`; validation now accepts them only because
  the entire matrices match the historical references exactly.
- Eight of ten model prediction streams match the old zoo byte-for-byte:
  LightGBM Conservative, LightGBM ExtraTrees, XGBoost, and Deep LightGBM,
  both OOF and test. CatBoost is not exact: OOF Spearman `0.9996243`, MAE
  `0.0006702`, max difference `0.0984592`; test Spearman `0.9999277`, MAE
  `0.0007278`, max difference `0.0294115`.
- A second fixed-seed GPU run confirmed CatBoost is not bitwise reproducible:
  fold 2 OOF AUC changed from `0.751470` to `0.751814`, while the Deep
  LightGBM streams repeated exactly. The repeat happened to be much closer to
  the historical CatBoost arrays (OOF MAE `4.45e-7`; test MAE `3.31e-7`) but
  was still not exact. The repeat was retained as a reproducibility check, not
  selected to tune parity; see `repeat_parity_report.json`.
- The isolated GPU telemetry captured 474 samples on the RTX 4050: peak GPU
  utilization 100%, peak memory 4,903 MiB, peak temperature 64 C. XGBoost
  preflight selected `cuda:0`; CatBoost preflight reported `GPU`.
- With the first independently rebuilt zoo, the original final 85:15 script
  stopped at its unchanged mean-rank-shift guard: `149.4283`, just below the
  required `[150, 250]`; it wrote no CSV. We did not alter the formula or
  bypass the assertion.
- A separate integration replay using the second raw zoo run passed all
  seven original downstream stages, including the unchanged final guard
  (`185.35` mean rank shift). It produced a 61,500-row CSV only inside the
  ignored run runtime. This chain is **HYBRID_HISTORICAL_CACHE_RECONSTRUCTION**:
  it starts from historical anchors and reuses V17/V55/XGB10/Grand Master 79.
  The generated output hash is
  `13b24617a8fd5eafd51eefe6f1a33b290b4efc5be157125cdb15648d3f1763c0`.
  Against the frozen champion it has exact ID coverage, Spearman
  `0.9999999872`, MAE `0.00003163`, max absolute difference `0.00063415`,
  mean rank displacement `1.95`, max `39`; it is not exact and has no official
  leaderboard score. It is not promoted as a submission.
- No output score is represented as an official leaderboard result. The
  frozen historical champion and Phase 2 release remain untouched.
- A separate full raw-component audit already trained XGB10, V55, V17, and a
  raw-data Anchor74 surrogate; its candidate hash is
  `2a5d2dfc2504f86a6d6c6d42bfc63fff7f62a10ea66135b1fee44982f9a348fe`.
  The candidate is classified `RAW_COMPONENT_CANDIDATE;
  HISTORICAL_PARITY_NOT_ESTABLISHED`; against the frozen champion its
  Spearman is `0.8915697`, MAE `0.0999145`, and RMSE `0.1344311`. These raw
  rebuilds do not reproduce the historical architecture and should not be
  repeated as a substitute for recovering the missing upstream lineage.

### Next concrete action

Recover the missing historical neural/ensemble model inputs or their original
producer source, weights, configuration, and fold manifest. Workspace-wide
search and all local Git branches contain no copies of the four arrays listed
in `RECONSTRUCTION_STATUS.md`. Until the exact 080041 ancestry and every
final-synthesis member can be reproduced from raw data, do not call either
generated candidate an independent historical submission or claim the
`0.801340` score.

## Independent improvement update (2026-11-02)

The improvement track has started separately from blocked historical
reconstruction. EXP04 tested GPU CatBoost Lossguide using only the available
Home Credit application train/test CSVs and fold-local Phase 2 features.
Historical predictions were not training inputs or labels; the frozen
champion was read only after training for a diagnostic comparison.

- Data: 171,202 labeled rows, 61,500 test rows, 122 features. Raw input hashes
  and complete code/output hashes are in the run manifest.
- Validation: five matched folds, seed `20261101`; inner-fit early stopping
  followed by full outer-fit refitting. OOF ROC-AUC was `0.7680327122`,
  compared with `0.7709638204` from the saved Phase 2 OOF vector
  (`-0.0029311082`); all five paired folds declined.
- The Phase 2 comparator selected stopping iterations using its outer
  validation folds, so it may be optimistic relative to EXP04's stricter
  inner-only stopping; the negative comparison is conservative.
- Hardware: all folds used the RTX 4050 GPU; 205 telemetry samples, 99% peak
  utilization and 4,041 MiB peak VRAM.
- Result: the candidate passed 61,500-row schema/ID checks, but failed the
  predeclared advancement gate and is **not promoted**. SHA-256:
  `a718e1b55e319a35ffe9945d1a7fa222e06eb1794e59aa26e7ce8579f4b31ef9`.
- Validation: five focused tests passed; the output manifest independently
  verified 19 run outputs, five inputs, and five source files.
- The first exploratory run is retained and marked invalid because it used
  the outer validation labels for early stopping. Only the strict run above
  is considered a valid OOF screen.

This local OOF result does not measure the historical champion's
`0.801340` public score or establish any new leaderboard score. Do not tune
Lossguide on these same folds or treat its candidate as a replacement. The
next credible improvement requires either a new, predeclared feature/model
source or recovered auxiliary data; the historical relational archives and
auxiliary tables remain unavailable.
