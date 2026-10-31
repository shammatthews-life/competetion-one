# Final Candidate Validation

## Candidate boundary

The candidate was generated only from the four fresh full-mode raw components
in `runs/raw_full_gpu_20261031_235900/components`. It was not submitted,
copied over a prior submission, or blended with historical caches.

## Schema and formula checks

| Check | Result |
|---|---|
| Rows | 61,500 |
| Columns | `SK_ID_CURR`, `TARGET` |
| IDs unique | Pass |
| ID set equals raw `test.csv` | Pass |
| Row order equals raw `test.csv` | Pass |
| Targets finite and within `[0,1]` | Pass |
| Targets unique ordinal percentiles | Pass |
| Candidate SHA-256 | `2a5d2dfc2504f86a6d6c6d42bfc63fff7f62a10ea66135b1fee44982f9a348fe` |
| Independent formula recomputation | Pass; max absolute difference `4.9594e-11`, within ten-decimal CSV rounding |
| Official leaderboard evaluation | Not performed; score is unknown |

The formula uses the fixed eight-member rank/probit suite weights, followed by
the `0.85 * Anchor74 + 0.15 * suite` probit blend and ordinal projection.
Historical caches and the frozen champion are comparison-only inputs.

## Comparison to frozen and prior candidates

All metrics below are computed after exact ID alignment.

| Reference | Pearson | Spearman | MAE | RMSE | Max abs difference | Mean rank displacement | Max rank displacement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen 0.801340 champion reference | 0.891570 | 0.891570 | 0.099914 | 0.134431 | 0.820455 | 6,144.74 | 50,458 |
| Previous raw candidate | 0.902144 | 0.902144 | 0.095043 | 0.127708 | 0.722846 | 5,845.11 | 44,455 |

The candidate is not an exact reproduction; correlation is not a leaderboard
score and does not establish historical parity.

## Historical component comparisons

Fresh Anchor74 versus the historical Anchor74 CSV, aligned by exact raw test
ID: Pearson/Spearman `0.869363`, MAE `0.110323`, RMSE `0.147555`, maximum
absolute difference `0.814677`, mean rank displacement `6,784.91`, and maximum
rank displacement `50,103`.

The post-run comparison contains all 27 OOF/test member-array pairs. Member
Spearman spans `0.857506`–`0.938947`; none is an exact rank match. Historical
prediction arrays contain no ID sidecars, so OOF comparison is positional after
checking exact raw label identity, and test-array comparison is positional
without independently provable per-row ID alignment.

The fresh V55 `test_cb_product` is rank-normalized before storage while the
historical array stores raw probabilities. Raw-value MAE/RMSE for that pair
are therefore not directly comparable; the report supplies rank-percentile
metrics, and its Spearman is `0.900908`, not exact historical ordering.

Full details and per-array hashes are in
`runs/raw_full_gpu_20261031_235900/reports/historical_output_comparison.json`.
Strict row/schema/range/formula checks are in
`runs/raw_full_gpu_20261031_235900/reports/final_validation_strict.json`;
component validation is in
`runs/raw_full_gpu_20261031_235900/reports/post_run_component_validation.json`.

## Earliest proven divergence and next action

The earliest proven observable difference is feature schema: the current
Anchor74 generator in the executed Git snapshot builds 143 features, while
Phase 3 reports 669 features and a selected 550-feature input. This execution
did not persist its in-memory feature matrix, so it cannot identify the first
cell or transformation that diverges. A separate 695-feature production
lineage report and its `SOLVENCY` finding belong to a different source
snapshot; they must not be attributed to this 143-feature generator.

**Next smallest corrective action:** run a feature-only diagnostic for the
exact 143-feature source builder, persist its train/test matrices and ID map
without fitting models, and compare shared columns against the persisted Phase
3 matrices after exact ID alignment. Only then identify the first differing
formula/dtype/missing-value treatment and decide whether the historical model
configuration can be recovered. Do not tune the blend or train another model
as a substitute for that comparison.

## Decision

**D. INCOMPLETE.** A fresh raw-component candidate exists and passes its
submission-format and formula checks, but the original feature/model lineage,
historical folds/configuration, V20 writer, and separate V080041 upstream
artifacts are not fully recovered. The official `0.801340` score has not been
independently reproduced or evaluated.
