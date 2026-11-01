# EXP85 Submission Correlation Replication

- Run ID: `20261101T133259Z_submission_final`
- Status: **VALID**
- Started (UTC): `2026-11-01T13:33:02.026648+00:00`
- Git commit: `fb42cdd2bd9a333b0fa3a6bda975c8beaf763b64`
- Method: `id_aligned`
- Command exit code: `0`

## Inputs

- Reference: `D:\gci\_gci_push_tmp\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\reference_artifacts\historical_champion.csv`
- Reference source path: `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`
- Reference SHA-256: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
- Candidate: `D:\gci\_gci_push_tmp\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\candidate_artifacts\submission_final.csv`
- Candidate source path: `D:\gci\gemini\final_submission\submission_final.csv`
- Candidate SHA-256: `8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b`

## Alignment

- Matched rows: `61500`
- Duplicate ID rows (reference/candidate): `0` / `0`
- Missing candidate IDs / extra candidate IDs: `0` / `0`
- Original row order matched: `True`

## Prediction validation

- Expected rows: `61500`
- Reference min / max: `8.1301000000000003e-06` / `0.9999918699`
- Candidate min / max: `1.62599e-05` / `0.99998374010000002`
- All predictions finite and within `[0, 1]`: `True`

## Metrics

- Pearson: `0.99696588802701946`
- Spearman: `0.99696588802705566`
- MAE: `0.015484782585372359`
- RMSE: `0.022487265915100176`
- Maximum absolute prediction error: `0.26612784949999996`
- Rank displacement mean / median / p95 / max: `952.31700813008126` / `675` / `2692.0499999999956` / `16367`
- Exact rank agreement rows: `41`
- Rank method: `average`
- Alignment method: one-to-one ID join, sorted by ID

## Limitations

Submission origin not established by this comparison.
Correlation does not prove prediction equality or model lineage. No official score is inferred from this local comparison.

- Script SHA-256: `bbf4282cb7e2d393be0590718d370bbc8fb80d2d84503eab7b395ff885454e2c`
- Configuration SHA-256: `e5d9b801ae4f50c4e7c5d93732bc55948d4b90d2e4b19aa7c4c86704795bae19`
