# EXP85 Submission Correlation Replication

- Run ID: `20261101T184500_phase3`
- Status: **VALID**
- Started (UTC): `2026-11-01T13:25:15.799891+00:00`
- Git commit: `fb42cdd2bd9a333b0fa3a6bda975c8beaf763b64`
- Method: `id_aligned`
- Command exit code: `0`

## Inputs

- Reference: `D:\gci\_gci_push_tmp\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\reference_artifacts\historical_champion.csv`
- Reference source path: `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`
- Reference SHA-256: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
- Candidate: `D:\gci\_gci_push_tmp\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\candidate_artifacts\track1_phase3_rebuilt.csv`
- Candidate source path: `D:\gci\gemini\champion\phase3_reconstruction\independent_rebuild\submission_exp85_rebuilt.csv`
- Candidate SHA-256: `8165ad2c33c1b1a367bf01752f740e240d04af8bbf999356dc65f611191abc46`

## Alignment

- Matched rows: `61500`
- Duplicate ID rows (reference/candidate): `0` / `0`
- Missing candidate IDs / extra candidate IDs: `0` / `0`
- Original row order matched: `True`

## Metrics

- Pearson: `0.99995925929169238`
- Spearman: `0.9999628751933789`
- MAE: `0.0025133036257398377`
- RMSE: `0.0031345775902673961`
- Maximum absolute prediction error: `0.033965638800000031`
- Rank displacement mean / median / p95 / max: `109.00832520325203` / `78` / `313` / `2091`
- Exact rank agreement rows: `288`
- Rank method: `average`
- Alignment method: one-to-one ID join, sorted by ID

## Limitations

Phase 3 synthesis uses historical V17/V55/XGB10 suite caches; mixed lineage.
Correlation does not prove prediction equality or model lineage. No official score is inferred from this local comparison.

- Script SHA-256: `7a732b4a9dd9897df4fa9d99dae0415ada6bfc5a29e8486ae6595f3f7d9d5726`
- Configuration SHA-256: `e5d9b801ae4f50c4e7c5d93732bc55948d4b90d2e4b19aa7c4c86704795bae19`
