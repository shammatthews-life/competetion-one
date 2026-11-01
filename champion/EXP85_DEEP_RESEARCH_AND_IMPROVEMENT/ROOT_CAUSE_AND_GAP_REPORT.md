# Historical EXP85 Gap: Root Cause and Remaining Unknowns

## Finding

The strongest supported explanation for the raw reconstruction's gap is **component-lineage mismatch**, not a failed 85/15 blend formula. Reusing the intended historical inputs closely reproduces V5; replacing those inputs with newly generated, similarly named components does not.

The existing forensic report records the following fixed-input swaps:

| Anchor input | Suite input | Spearman vs. champion |
|---|---|---:|
| Phase 3 high-correlation Anchor74 | Historical cached suite | 0.9999975816 |
| Current raw Anchor74 | Historical cached suite | 0.9031292 |
| Phase 3 high-correlation Anchor74 | Current rebuilt suite | 0.9986817 |
| Current raw Anchor74 | Current rebuilt suite | 0.8921289 |

This isolates the largest observed divergence to the Anchor74 component identity. It does **not** establish why the current Anchor74 differs, nor prove that the historical suite caches are ID-aligned.

## Verified lineage evidence

- The high-correlation V5 uses Phase 3 Anchor74 SHA-256 `315b0bd80e4698240a7a902648a157d84da1480dea77901cdb60892502a7b55b` and historical cached V17/V55/XGB10 suite predictions.
- A same-named current Anchor74 has SHA-256 `6c3207977abc21091b64a7446f979c17ed2ce9aaabf18b3acb2065569c559bc8`; another same-named file has SHA-256 `04deaa2e3e89c13bbfbe759f36aabf34609856cd7bb5c47b65ef3c2b715134d8`. They are not interchangeable with the high-correlation artifact.
- The fixed formula `0.85 * Z_Anchor74 + 0.15 * Z_Suite` reproduces the saved high-correlation V5 to its 10-decimal CSV precision only when the intended Anchor74 and historical suite predictions are selected.
- Historical suite `.npz`/`.npy` arrays have no embedded `SK_ID_CURR` or trusted row-order sidecars. Their memberwise comparisons remain positional, not certified borrower-aligned comparisons.
- The Phase 3 reconstruction source consumes cached suite arrays. A raw-input command line or successful execution does not remove that cache dependency.
- The reported exact original score is associated with a historical artifact/replay claim; this work has not independently uploaded or evaluated any candidate.

Detailed source paths, artifact comparisons, and limitations are documented in the prior [correlation replication root-cause report](../EXP85_SUBMISSION_CORRELATION_REPLICATION/evidence/ROOT_CAUSE_REPORT_original.md), [model lineage](MODEL_LINEAGE.md), and [correlation-gap analysis](CORRELATION_GAP_ANALYSIS.md).

## Gap to Phase 2

Phase 2 is a separate raw-only model family: LightGBM/CatBoost with application-level rowwise features, decoded fee/product features, and nested smoothed encodings. It reaches a previously validated seed-42 OOF AUC of `0.7715614423`, but its predictions have Pearson `0.7782506970` and Spearman `0.9157084581` versus the historical champion's **unlabeled test predictions**.

The latter correlations describe prediction disagreement only. They are not measures of AUC, default-risk error, or official score. They cannot identify the cause of the OOF/leaderboard performance difference.

## Audit of the supplied notebook's Stage 9 cell

The attached Stage 9 code calculates Pearson/Spearman and rank equality directly on the `TARGET` arrays after reading CSVs. In the shown cell it does not first validate exact schemas, unique IDs, equal ID sets, or reorder predictions to a canonical `test.csv` ID vector. Therefore its Track 1 correlation/rank figures are not independently certified borrower-aligned by that cell. A separate auditor subsequently performed the ID-aligned V5/champion comparison and reproduced the documented `0.9999975816` result.

The displayed Track 2 candidate SHA is identical to the champion SHA, which does establish byte identity for those two files if the hashes were computed as shown. It does not show that an upload received a score. Printing the historical string `0.801340 (CONFIRMED)` is not an official evaluation record. The notebook cell contains no platform request/result, submission identifier, or binding between an evaluated upload and that hash. Treat `0.801340` as a historical reported score, not a new verification.

This distinction does not invalidate the later verified artifact/hash evidence; it narrows what the notebook output alone proves.

Plausible but unresolved contributors include unavailable upstream models, different feature generations, fold assignments, parameters, target encodings, model stochasticity, missing-value semantics, and historical transductive post-processing. The evidence supports component identity and Anchor74 ancestry as major fidelity blockers; it does not support assigning the residual to any one unverified mechanism.

## Reproducible versus unavailable

| Capability/component | Classification | Evidence / boundary |
|---|---|---|
| Phase 2 raw-only pipeline | `RAW_REPRODUCIBLE` | Frozen code, raw hashes, saved folds/OOF and integrity verifier. Official score remains unknown. |
| Fee/product feature builder | `RAW_REPRODUCIBLE` | Source and label-independence tests are preserved. Pooled frequency counts are transductive; a fold-local sensitivity exists. |
| Nested Phase 2 target encoding | `RAW_REPRODUCIBLE` | Inner OOF for outer-training rows; validation/test mappings use outer-training labels. |
| Historical V5 fixed blend formula | `PARTIALLY_REPRODUCIBLE` | Formula replay works with exact inputs; it is not itself a raw model producer. |
| Historical Phase 3 Anchor74 artifact | `CACHE_DEPENDENT` | High-fidelity output consumes prior apex and zoo predictions; exact complete raw ancestry is unresolved. |
| Historical V17/V55/XGB10 predictions | `CACHE_DEPENDENT` | Exact arrays exist, but producer/config/folds and trusted ID bindings are incomplete. |
| Historical V20 producer and missing upstream stacker dependencies | `MISSING` | Prior audits did not recover full original producer lineage. |
| Relational Home Credit tables | `NOT_APPLICABLE` to the current available inputs | The current raw-data directory contains only application train/test/sample CSVs. No related-table aggregation experiment is possible from these files. |
| Official evaluation of a new candidate | `MISSING` | No current official submission/evaluation is claimed. |

## Root-cause limits and next evidence

To improve historical reproducibility, recover exact producer revisions, configs, folds, feature schemas, and a trusted ID order for each cached suite member. Keep that fidelity track separate from raw-only validation. Do not repair lineage by fitting against historical test predictions.

For generalization, the immediate independent experiment tests category peer aggregates with strict outer-fold isolation. Its decision is based only on matched OOF labels, not champion similarity.
