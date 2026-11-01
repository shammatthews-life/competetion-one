# EXP85 Submission Correlation Replication

- Run ID: `20261101T184800_current_track1b`
- Status: **VALID**
- Started (UTC): `2026-11-01T13:25:23.969605+00:00`
- Git commit: `fb42cdd2bd9a333b0fa3a6bda975c8beaf763b64`
- Method: `id_aligned`
- Command exit code: `0`

## Inputs

- Reference: `D:\gci\_gci_push_tmp\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\reference_artifacts\historical_champion.csv`
- Reference source path: `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`
- Reference SHA-256: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
- Candidate: `D:\gci\_gci_push_tmp\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\candidate_artifacts\track1b_current_production.csv`
- Candidate source path: `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\runs\track1b_1793445445960\submission_track1b.csv`
- Candidate SHA-256: `9b9f7dded04ae606b8d0ca389154304f9365bf628589e1dba4e5bba652cc6f6b`

## Alignment

- Matched rows: `61500`
- Duplicate ID rows (reference/candidate): `0` / `0`
- Missing candidate IDs / extra candidate IDs: `0` / `0`
- Original row order matched: `True`

## Metrics

- Pearson: `0.89212889443999577`
- Spearman: `0.89212889444021914`
- MAE: `0.099670359177756115`
- RMSE: `0.13408399452798678`
- Maximum absolute prediction error: `0.82120325199999999`
- Rank displacement mean / median / p95 / max: `6129.727089430894` / `4599` / `17135.049999999996` / `50504`
- Exact rank agreement rows: `2`
- Rank method: `average`
- Alignment method: one-to-one ID join, sorted by ID

## Limitations

Current production pipeline trained from raw inputs; known to be materially different from historical champion.
Correlation does not prove prediction equality or model lineage. No official score is inferred from this local comparison.

- Script SHA-256: `7a732b4a9dd9897df4fa9d99dae0415ada6bfc5a29e8486ae6595f3f7d9d5726`
- Configuration SHA-256: `e5d9b801ae4f50c4e7c5d93732bc55948d4b90d2e4b19aa7c4c86704795bae19`
