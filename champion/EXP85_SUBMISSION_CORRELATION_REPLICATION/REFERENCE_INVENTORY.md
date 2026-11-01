# Reference and Candidate Inventory

All comparisons below use `reference_artifacts/historical_champion.csv`, copied byte-for-byte from `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`, SHA-256 `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`. The second reference `D:\gci\gemini\v_20261027_optimal_macro_apex_v85.csv` has the same 1,291,519-byte file and SHA-256 `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`; it is a duplicate identity, not a second independent reference.

| Candidate identity | Original path | SHA-256 | Pearson | Spearman | MAE | RMSE | Mean / median / p95 / max rank displacement | Lineage interpretation |
|---|---|---|---:|---:|---:|---:|---|---|
| Track 1 V5 rebuilt | `D:\gci\champion\08_SUBMISSIONS\submission_exp85_recovery_v5_rebuilt.csv` | `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482` | 0.9999975815867518 | 0.9999975815867501 | 0.0003014174102764 | 0.0006348770547015 | 18.5372 / 7 / 80 / 769 | **Hybrid historical-cache replay.** Rebuilt Phase 3 Anchor74 plus historical cached V17/V55/XGB10 suite. |
| Phase 3 rebuilt | `D:\gci\gemini\champion\phase3_reconstruction\independent_rebuild\submission_exp85_rebuilt.csv` | `8165ad2c33c1b1a367bf01752f740e240d04af8bbf999356dc65f611191abc46` | 0.9999592592916924 | 0.9999628751933789 | 0.0025133036257398 | 0.0031345775902674 | 109.0083 / 78 / 313 / 2,091 | Phase 3 synthesis; forensic report finds historical suite cache dependencies. |
| Archived full rebuilt | `D:\gci\champion\08_SUBMISSIONS\submission_exp85_full_rebuilt.csv` | `6bfc518b7abc208ff7e95c5dd8029b8d418d7ae3c1c94f9cdc6c4499598bc042` | 0.9982094981549333 | 0.9982094981549220 | 0.0122605168882992 | 0.0172747496103094 | 754.0218 / 538 / 2,182 / 9,878 | Separate archived candidate; this audit does not recertify its producer lineage. |
| `submission_final.csv` | `D:\gci\gemini\final_submission\submission_final.csv` | `8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b` | 0.9969658880270195 | 0.9969658880270557 | 0.0154847825853724 | 0.0224872659151002 | 952.3170 / 675 / 2,692.05 / 16,367 | Separate artifact; its model/source lineage is not established by this comparison. |
| Current production Track1B | `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\runs\track1b_1793445445960\submission_track1b.csv` | `9b9f7dded04ae606b8d0ca389154304f9365bf628589e1dba4e5bba652cc6f6b` | 0.8921288944399958 | 0.8921288944402191 | 0.0996703591777561 | 0.1340839945279868 | 6,129.7271 / 4,599 / 17,135.05 / 50,504 | Separate raw-production candidate; not equivalent to historical cached components. |

All five candidates have 61,500 rows, unique IDs, zero missing/extra IDs against the reference, and exact original ID order. The first execution and independent rerun produce the same metrics. Full-precision outputs, input hashes, command/config/source hashes, and run timestamps are stored under each matching run folder.

## Duplicate and stale records

- `D:\gci\champion\02_EXP85_REBUILT\submissions\submission_exp85_recovery_v5_rebuilt.csv` is byte-identical to the V5 candidate above (same SHA-256); it is not a separate experiment.
- `D:\gci\champion\08_SUBMISSIONS\SUBMISSION_RESULTS.md` lists SHA-256 `e979a04a...` for the V5 filename. That does not match the actual on-disk candidate hash `91aa95c9...`; the stale report hash is not used as a substitute.
- The user's reference `submission_final.csv` is a distinct 8e8158... file and does not produce the 0.99999758 result against the historical champion.
- The 0.9999628752 and 0.9982094982 values belong to distinct candidate files and run records; neither is the V5 candidate.

The correlation outputs do not confirm any candidate's official leaderboard score. The source documents contain official-score assertions with inconsistent file hashes; those claims require separate platform evidence and are not inferred here.
