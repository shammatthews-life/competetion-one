# Correlation Methodology

## Artifact verification

Before metrics are computed, each CSV is hashed independently and read with explicit ID/prediction columns. Blank IDs, duplicate IDs, nonnumeric or nonfinite predictions, and predictions outside `[0, 1]` invalidate the comparison. The report records row counts, set differences, original row-order equality, matched rows, and both SHA-256 values.

## Preferred method: `id_aligned`

The runner rejects duplicate IDs and any missing/extra IDs, performs a one-to-one join keyed by `SK_ID_CURR`, sorts the joined rows by ID, and then computes all metrics. It reports Pearson and Spearman correlation, MAE, RMSE, maximum absolute error, and mean/median/95th percentile/maximum absolute rank displacement. Average ranks are used for displacement ties.

This is the preferred method for new comparisons. A high correlation is not equality, lineage proof, or official score evidence.

## Exact legacy method: `notebook_stage9`

The historical notebook compared raw prediction arrays in file row order and ranked with `rankdata(method="ordinal")`. This mode preserves that computation but adds a guard: the two files must first have the same unique IDs in precisely the same order. If not, it refuses the legacy calculation rather than silently correlating different applicants.

The primary old result is reproduced in `runs/20261101T184300_notebook_stage9_v5/`. In this dataset, the exact-order validation passes, and the ID-aligned result matches to the full precision shown in the saved JSON.

## Legacy isolated auditor

The preserved `audit_prediction_correlations.py` is run unchanged on the same pair in `runs/20261101T184200_original_auditor_v5_rebuilt/`. It validates IDs, merges by ID, uses Pearson and Spearman on paired raw values, and uses average ranks. It reproduces the same reported ten-decimal correlations. Its current tests also pass.

## Interpretation of the metrics

- Pearson describes linear association in the raw saved probabilities.
- Spearman describes monotone rank association; it does not imply values are equal.
- Rank displacement describes per-ID rank movement under the stated tie method.
- MAE/RMSE/max error describe prediction-value differences.
- None of these calculates an official leaderboard result.

No normalization or calibration is applied by the comparison runners. Rank/probit calibration belongs to the candidate-generation pipeline, not the correlation calculation. The notebook itself generated the candidate using the 85/15 probit composition and final ordinal projection before it measured raw-target correlation.
