# EXP85 Submission Correlation Replication

- Run ID: `20261101T184300_notebook_stage9_v5`
- Status: **VALID**
- Started (UTC): `2026-11-01T13:25:12.928735+00:00`
- Git commit: `fb42cdd2bd9a333b0fa3a6bda975c8beaf763b64`
- Method: `notebook_stage9`
- Command exit code: `0`

## Inputs

- Reference: `D:\gci\_gci_push_tmp\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\reference_artifacts\historical_champion.csv`
- Reference source path: `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`
- Reference SHA-256: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
- Candidate: `D:\gci\_gci_push_tmp\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\candidate_artifacts\track1_v5_rebuilt.csv`
- Candidate source path: `D:\gci\champion\08_SUBMISSIONS\submission_exp85_recovery_v5_rebuilt.csv`
- Candidate SHA-256: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`

## Alignment

- Matched rows: `61500`
- Duplicate ID rows (reference/candidate): `0` / `0`
- Missing candidate IDs / extra candidate IDs: `0` / `0`
- Original row order matched: `True`

## Metrics

- Pearson: `0.99999758158675178`
- Spearman: `0.99999758158675012`
- MAE: `0.00030141741027642276`
- RMSE: `0.00063487705470147496`
- Maximum absolute prediction error: `0.012504065100000017`
- Rank displacement mean / median / p95 / max: `18.537170731707317` / `7` / `80` / `769`
- Exact rank agreement rows: `3093`
- Rank method: `ordinal`
- Alignment method: original notebook row-position comparison after exact ID-order validation

## Limitations

Hybrid historical-cache replay; the Anchor74 was a reconstructed artifact, but the eight-engine V17/V55/XGB10 suite uses historical cached predictions. This is not a full raw-only rebuild.
Correlation does not prove prediction equality or model lineage. No official score is inferred from this local comparison.

- Script SHA-256: `7a732b4a9dd9897df4fa9d99dae0415ada6bfc5a29e8486ae6595f3f7d9d5726`
- Configuration SHA-256: `a898bbab3ba8552097a1fde407bd0df76f0f61710befe52fc34268ea1383c632`
