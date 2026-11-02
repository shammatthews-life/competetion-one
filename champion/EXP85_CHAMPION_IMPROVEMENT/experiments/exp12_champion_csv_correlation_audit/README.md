# EXP12 — Champion CSV correlation audit

This is a read-only audit of CSV files found recursively under `D:\gci\champion`,
`D:\gci\_gci_push_tmp\champion`, and `D:\gci\gemini`. The historical champion is
read-only and must match its pinned SHA-256 before the audit proceeds. Each
candidate is aligned by its identifier; matching row counts or row order alone
never establish comparability. Training/OOF predictions are not correlated
with the test-set champion.

Run from the repository root:

```powershell
python champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp12_champion_csv_correlation_audit\source\run_exp12.py
python champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp12_champion_csv_correlation_audit\source\verify_exp12.py
python -m unittest discover -s champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp12_champion_csv_correlation_audit\tests -v
```

The runner only writes inside this EXP12 directory and excludes that directory
from its recursive input scan, so reruns do not ingest their own outputs.
`CSV_INVENTORY.csv` includes every discovered CSV, including excluded datasets,
features, reports, and invalid files, with an explicit reason. Correlation
results are descriptive prediction comparisons, not evidence of AUC or public
leaderboard improvement.

The matrix and pairwise table deduplicate byte-identical CSVs and identical
ID-aligned prediction streams while retaining every source path in the
inventory and duplicate report. Kendall tau is computed for each champion
comparison; pairwise Pearson and Spearman matrices are vectorized for practical
coverage. No labels or test outcomes are inferred from predictions.
