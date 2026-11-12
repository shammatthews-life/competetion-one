# Independent XGBoost Seed Replication

- Run: `20261101T144500Z_raw_xgb_diversity_seed20261101`
- Decision: `REPLICATED_POSITIVE_DIAGNOSTIC_ONLY`
- New five-fold seed 20261101; exact frozen Phase 2 fold indices and fold-local OOF reference.
- Same XGBoost configuration, feature builder, categorical handling, and fixed 80/20 diagnostic as the seed-42 screen.

Frozen Phase 2 OOF AUC: `0.7709638204`
Independent XGBoost OOF AUC: `0.7602661213`
XGBoost delta: `-0.0106976990`
Fixed 80/20 rank-blend OOF AUC: `0.7709889247`
Blend delta: `+0.0000251043`
Positive blend folds: `2/5`
Residual correlation: `0.994353`

| Fold | Phase 2 AUC | XGBoost AUC | XGB delta | Fixed blend AUC | Blend delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.76608191 | 0.75483797 | -0.01124393 | 0.76604698 | -0.00003492 |
| 2 | 0.77302333 | 0.76333992 | -0.00968341 | 0.77322678 | +0.00020345 |
| 3 | 0.77175481 | 0.76083437 | -0.01092044 | 0.77189215 | +0.00013734 |
| 4 | 0.76869469 | 0.75762980 | -0.01106489 | 0.76856283 | -0.00013186 |
| 5 | 0.77539333 | 0.76464964 | -0.01074370 | 0.77527119 | -0.00012214 |

- Runtime: `30.7` seconds.
- GPU samples: `6`; max utilization `69%`; peak VRAM `318` MiB.
- Candidate SHA-256: `0de101afce5a3ed1345f56e28fad73ec09296ae399deab11a47804a321605d38`
- Official score: **NOT EVALUATED**.
