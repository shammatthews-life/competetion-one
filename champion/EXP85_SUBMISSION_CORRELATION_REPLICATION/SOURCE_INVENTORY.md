# Source Inventory and Forensic Findings

All sources below were read-only inputs. The copies under `source/` are SHA-256 checked against the original paths.

| Source | Original path | SHA-256 | Finding |
|---|---|---|---|
| V5 reconstruction notebook | `D:\gci\champion\02_EXP85_REBUILT\notebooks\exp85_recovery_v5.ipynb` | `d7d16d00599061b041e2b85619eb4dbc3f3607d437766c9c1e72d928a913f7b7` | Direct source of the reported Stage 8 output and Stage 9 comparison. |
| Existing ID-aware comparison auditor | `D:\gci\champion\EXP85_ISOLATED_REPAIR\scripts\audit_prediction_correlations.py` | `0e1c7862c9974451d5e2ab539a6f05ba917421a57d7e2fb00e7621b97e529653` | Separately validates ID sets, joins by ID, and computes Pearson/Spearman/error/rank metrics. Its missing/extra counter labels are reversed on invalid sets; it still fails closed, and valid-pair counts are unaffected. It has no p95 rank displacement or explicit probability-bound check. |
| Existing auditor tests | `D:\gci\champion\EXP85_ISOLATED_REPAIR\scripts\test_prediction_correlations.py` | `78a5db39ac678404721a473f6b1eb3f478558973d4d22aa267a358ee2b6e08de` | Synthetic tests pass on the preserved, unmodified auditor. |
| Forensic analysis implementation | `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\correlation_failure_investigation\analyze_existing_artifacts.py` | `ad54049367bf4738638a2ece6813d89b5114b4efb484f14ef27ee25cae438fe8` | Source of the read-only forensic component comparisons and lineage analysis. |
| Forensic root-cause report | `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\correlation_failure_investigation\ROOT_CAUSE_REPORT.md` | Recorded in `REPLICATION_MANIFEST.json` | Identifies the high-correlation route as a hybrid cache replay and separates the nearby values. |
| Historical evidence JSON | `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\correlation_failure_investigation\audit_evidence.json` | Recorded in `REPLICATION_MANIFEST.json` | Contains source and artifact hashes used in the forensic audit. |
| Historical submission generator | `D:\gci\champion\create_submission_recovery_v5_rebuilt.py` | `fbeaecbf28c4ab96459d8cda32b1997b4788faf6674221942c889208087ba2ed` | Its default hybrid mode loads historical caches. It is not the source of the notebook's exact high-correlation Anchor path: it prefers `02_EXP85_REBUILT\rebuilt_artifacts`, whose same-named Anchor74 has a different hash. Do not use it as a drop-in producer for the notebook's candidate. |
| Phase 3 pyramid source | `D:\gci\champion\02_EXP85_REBUILT\source\reconstruct_transductive_pyramid.py` | `a99bd2ba02ed673a6660eb67c39fb16e80d9d01d4560e02ef3ba3ae0fefbd0f4` | The forensic report finds historical suite-cache reads/fallbacks in this route; its final candidate is mixed-lineage. |
| New comparison runner | `source/correlation_runner.py` | Captured per run and in the internal hash manifest | ID-aligned and legacy Stage 9 modes; fail-closed checks, hashes, full metrics, unique run output, and lineage notes. |

## Original notebook behavior recovered

Notebook Stage 8 wrote `D:\gci\champion\08_SUBMISSIONS\submission_exp85_recovery_v5_rebuilt.csv`. It combined `y_a_reb` with `test_suite_z` using an 85/15 probit blend, then used `rankdata(..., method="ordinal")` to create the final percentile target and wrote ten decimal places.

Stage 9 read `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv` and both saved submissions. It computed Pearson/Spearman and MAE from raw prediction values. It created ordinal ranks for agreement/displacement but did **not** join predictions by `SK_ID_CURR`. The two files' ID order was subsequently verified to be exactly identical, so the reported pair is numerically reproducible without positional ambiguity.

The same notebook's Stage 7 loads V17 and V55 bundles plus the XGB10 test vector from `D:\gci\champion\01_EXP85_RECOVERY\cache\`. Stage 6 selects the Anchor74 under `02_EXP85_REBUILT\artifacts\`. The forensic report verifies these exact inputs and identifies the resulting submission as a hybrid historical-cache replay.

## Existing auditor finding

The older isolated auditor is more robust than notebook Stage 9 for ID association: it rejects duplicates and mismatched ID sets and calculates statistics after an ID join. Its invalid-set labels invert “missing from candidate” and “extra in candidate” when constructing error counts. Since it rejects either nonzero count and the replicated pair has exact sets, this did not alter its primary metrics. The new runner reports these counts with correct set-difference semantics and adds rank p95, probability bounds, provenance fields, and a distinct legacy mode.
