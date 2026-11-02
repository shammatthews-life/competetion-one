# EXP09 — Prediction Diversity and Champion Bridge Results

- Status: **COMPLETED_SCREEN — REJECTED**
- Prediction-only evaluation; no models were retrained.
- All reported AUCs are OOF metrics, not public leaderboard scores.

## Strict matched comparisons

| Stream / strategy | Seed | Pooled OOF AUC | Delta vs Phase 2 | Positive folds |
|---|---:|---:|---:|---:|
| phase2 | 20261101 | 0.77034958 | +0.00000000 | 0/5 |
| exp04 | 20261101 | 0.76803271 | -0.00231687 | 0/5 |
| exp05 | 20261101 | 0.76372731 | -0.00662228 | 0/5 |
| exp06 | 20261101 | 0.76402234 | -0.00632725 | 0/5 |
| exp07_signature | 20261101 | 0.77138426 | +0.00103468 | 4/5 |
| exp08_historical | 20261101 | 0.77021653 | -0.00013306 | 1/5 |
| exp08_lgbcat | 20261101 | 0.77030950 | -0.00004008 | 2/5 |
| exp08_xgb | 20261101 | 0.76899815 | -0.00135143 | 0/5 |
| raw_xgb_main | 20261101 | 0.76026612 | -0.01008346 | 0/5 |
| exp07_nested_blend | 20261101 | 0.77136671 | +0.00101713 | 4/5 |
| exp08_nested_blend | 20261101 | 0.77054757 | +0.00019798 | 4/5 |
| Best fixed two-candidate screen: equal_exp07_signature__exp08_lgbcat | 20261101 | 0.77135704 | +0.00100746 | 5/5 |
| Best fixed three-candidate screen: equal_exp04__exp07_signature__exp08_lgbcat | 20261101 | 0.77095722 | +0.00060764 | 4/5 |

## First-screen leader and replication

- Strategy: `exp07_signature`.
- Main-seed result: 0.77138426 (delta +0.00103468; 4/5 positive folds).
- Independent seed result: 0.77182936 (delta +0.00058078; 4/5 positive folds).

Candidate pair/triple search results are exploratory: the best combination is selected after comparing multiple fixed combinations, so its first-screen estimate is selection-biased. No outer-fold labels fitted any weights; all blend weights are fixed arithmetic means.

## Independent seed screen

| Stream | Pooled OOF AUC | Delta vs seed-42 Phase 2 | Positive folds | 50:50 with Phase 2 |
|---|---:|---:|---:|---:|
| exp07_signature | 0.77182936 | +0.00058078 | 4/5 | 0.77195766 |
| raw_xgb_replication | 0.76136957 | -0.00987900 | 0/5 | 0.77008229 |
| phase2_plus_exp07_signature_equal50 | 0.77195766 | +0.00070908 | 5/5 | — |
| phase2_plus_raw_xgb_replication_equal50 | 0.77008229 | -0.00116628 | 1/5 | — |

## Controls and exclusions

- Every candidate stream was required to have matching train/test/sample SHA-256 records, matching prediction-output hashes, expected row counts, a complete five-fold assignment, and exact fold membership matching its seed-specific Phase 2 reference.
- OOF rows were verified against each source's saved per-fold predictions at their exact validation indices, or against an ID-bearing OOF sidecar. Exact input hashes and ordered train/test ID hashes were also checked.
- Test arrays were admitted to test-ensemble inventory only when a source submission ID sidecar or five per-fold test predictions matched; the strict Phase 2 test streams lack such direct ID sidecars, so no candidate submission was created.
- Fixed equal-weight blending was used; no weight or meta-model was fitted on OOF or outer validation labels.
- Logistic stacking was not attempted because homogeneous inner-validation predictions were unavailable for every candidate.
- The historical champion archive had a cached test submission but no independently verifiable champion OOF/component predictions. It was excluded from ensembling.
- The interaction-screen arrays were excluded from OOF comparisons because the saved run has neither an ID-bearing OOF sidecar nor per-fold OOF prediction arrays.
- No candidate submission was generated because no strategy passed the independent-seed gate.

## Artifact integrity

- Train ID SHA-256: `efa7351b882ec649b980d55e85e87ff8728a6f0afd80fcb25c44a6a10753c855`.
- Test ID SHA-256: `0cfa5599319972ebaa2abe80fd84afcebf85e85913d1b5ee84d7545c27675787`.
- Discovered prediction artifacts: 98.
- Input/source/output SHA-256 records: `exp09_prediction_diversity_v4/sha256_manifest.json`.
- Detailed artifact exclusions and per-stream provenance are in `ARTIFACT_INVENTORY.csv`.
- Per-fold paired metrics are in `FOLD_METRICS.csv`; diversity diagnostics are in `PREDICTION_DIVERSITY.csv`.

## Decision

**REJECTED** — the best first-screen strategy does not have a verified matching independent-seed replicate that also clears the +0.001 pooled / 4-of-5-fold gate. Preserve the historical champion and frozen Phase 2 release.

The historical public score (0.801340) and target (0.80314) are not established by these OOF results. No official leaderboard evaluation was performed.
