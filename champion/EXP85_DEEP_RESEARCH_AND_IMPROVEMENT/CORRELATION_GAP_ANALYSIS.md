# Correlation Gap Analysis

## Purpose and restriction

This analysis compares prediction artifacts to the historical champion solely as a diagnostic of lineage and prediction differences. The champion is **not** a source of ground-truth labels, a training target, a calibration target, or a model-selection objective. No candidate can be promoted because it is more correlated with the champion.

## Verified baseline facts

- Historical champion: 61,500 IDs, SHA-256 `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`.
- Hybrid V5: 61,500 IDs, SHA-256 `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`; Pearson `0.9999975815867518`, Spearman `0.9999975815867501`, MAE `0.0003014174`, RMSE `0.0006348771`, maximum error `0.0125041`, mean/median/p95/max rank displacement `18.54/7/80/769`.
- Raw Track1B: SHA-256 `9b9f7dded04ae606b8d0ca389154304f9365bf628589e1dba4e5bba652cc6f6b`; Pearson/Spearman `0.8921288944`, MAE `0.0996704`, RMSE `0.1340840`, maximum error `0.8212033`, mean/max rank displacement `6129.73/50504`.
- Frozen Phase 2: candidate SHA-256 `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`; independent raw-only. The Stage 0 comparison below was generated against the exact raw test IDs; it is not assumed to be close to the champion.

All CSV comparisons are one-to-one aligned by `SK_ID_CURR`. The legacy high correlation was also replayed positionally only after proving unique IDs and exact row order.

## Component decomposition constraints

The existing forensic report records positional correlations of current production members to historical caches, because the historical NPZ/NPY arrays do not embed IDs. These values are useful clues but are **not ID-aligned certified component comparisons**:

| Historical member | Reported positional Spearman | Reported MAE | Qualification |
|---|---:|---:|---|
| V17 `e1Aet` | 0.929469 | 0.075187 | Index-order only |
| XGB10 | 0.937408 | 0.019429 | Index-order only |
| V17 `final` | 0.951633 | 0.061693 | Index-order only |
| V55 CatBoost product | 0.900804 | 0.025355 | Index-order only |
| V17 `e1et` | 0.929471 | 0.075230 | Index-order only |
| V55 deep ET | 0.921398 | 0.079931 | Index-order only |
| V55 all-family | 0.926560 | 0.076797 | Index-order only |
| V55 monotone | 0.900071 | 0.092644 | Index-order only |

Do not present these as correctly aligned borrower-level results without a producer-verified ID binding.

The strongest supported synthesis attribution is the fixed-input swap diagnostic: historical Phase 3 Anchor74 + historical suite reproduces V5, while substituting the current production Anchor reduces Spearman to `0.9031292`; substituting only the current suite while retaining the historical Anchor yields `0.9986817`. This supports component-identity mismatch, especially Anchor74, as the primary reason V5 and raw Track1B differ. It does not explain the exact per-borrower residual or certify raw-only lineage.

## New read-only gap diagnostic

`source/correlation_gap_diagnostics.py` aligns the frozen Phase 2 and hybrid V5 submissions with the reference and all raw test rows, verifies the frozen input hashes, and writes full metrics plus error slices under one unique `runs/<run-id>/` folder. Slice definitions are predeclared: reference-prediction deciles, EXT_SOURCE missingness regimes, all-feature missingness bands, income quartiles, age groups, contract type, and gender. It records counts, MAE, RMSE, signed bias, maximum absolute error, and global-rank displacement summaries.

### Corrected full-covariate run

Run: `runs/20261102T003000Z_stage0_final/`. All rows match the raw test ID set; both candidate ID orders are checked and aligned by ID. Test/sample order, exact two-column submission schemas, finite `[0,1]` predictions, and SHA-256 hashes passed. The run read all 33 test columns and verified the test table contains no target column.

| Pair | Pearson | Spearman | MAE | RMSE | Max absolute error | Mean rank displacement | p95 rank displacement | Max rank displacement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 2 vs historical champion | 0.7782506970 | 0.9157084581 | 0.41111346 | 0.46865093 | 0.94017030 | 5392.1051 | 15026.1 | 52044 |
| Hybrid V5 vs historical champion | 0.9999975816 | 0.9999975816 | 0.0003014174 | 0.0006348771 | 0.01250407 | 18.5372 | 80 | 769 |
| Phase 2 vs hybrid V5 | 0.7782540515 | 0.9157128812 | 0.41111342 | 0.46865075 | 0.94023534 | 5392.0868 | 15020.05 | 52100 |

The immutable first run, `runs/20261102T000000Z_correlation_gap_stage0/`, is superseded only for grouped slices: its missingness count used a restricted subset of test columns. Prediction-wide values remain the same. The final run reads all raw test covariates.

These slices describe where two **predictions** disagree. The test data has no labels; no subgroup default rate, AUC, or causal explanation is inferred. The run is not an experiment to optimize against the champion. The report, grouped table, run manifest, and their hashes are stored in the run directory.

## What remains unidentifiable

The successful hybrid's cached suite vectors cannot be aligned to `SK_ID_CURR` from those arrays alone. Thus a valid ID-aligned component-vs-champion decomposition for historical V17/V55/XGB10 is not presently available. The final-output metric is certifiable by IDs; the memberwise historical metric is not.
