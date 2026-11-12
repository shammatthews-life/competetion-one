# Independent Raw XGBoost Diversity Screen

- Run: `20261101T144300Z_raw_xgb_diversity_seed42`
- Decision: `PROMISING_SCREEN_REQUIRES_INDEPENDENT_SEED`
- Same exact five seed-42 folds as frozen Phase 2.
- Candidate predictions were generated from raw application train/test CSVs only.
- No auxiliary tables, historical predictions, or target-derived features were used for XGBoost training.
- The fixed 80/20 rank blend is a diagnostic only; its weight was not tuned and it is not promoted.

Frozen Phase 2 OOF AUC: `0.7715614423`
Independent XGBoost OOF AUC: `0.7613695677`
XGBoost delta: `-0.0101918746`
Fixed 80/20 rank-blend OOF AUC: `0.7716634770`
Fixed blend delta: `+0.0001020347`
OOF prediction Pearson correlation: `0.945496`
OOF residual Pearson correlation: `0.994500`
Squared-error Pearson correlation: `0.994796`

| Fold | Phase 2 AUC | XGBoost AUC | XGB delta | Fixed blend AUC | Blend delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.76827281 | 0.75928425 | -0.00898856 | 0.76865977 | +0.00038695 |
| 2 | 0.77412739 | 0.76205328 | -0.01207411 | 0.77400391 | -0.00012348 |
| 3 | 0.77616376 | 0.76550459 | -0.01065917 | 0.77617762 | +0.00001386 |
| 4 | 0.76956077 | 0.75724158 | -0.01231919 | 0.76911484 | -0.00044593 |
| 5 | 0.77001386 | 0.76311009 | -0.00690377 | 0.77071853 | +0.00070467 |

- Runtime: `30.7` seconds.
- GPU samples: `6`; max utilization `77%`; peak VRAM `318` MiB.
- Candidate SHA-256: `39ea94a113cbfb0493cb1fbdf21f20ffee3fdd4f440ef14b8f799bb9f2860ebc`
- Official competition score: **NOT EVALUATED**.
