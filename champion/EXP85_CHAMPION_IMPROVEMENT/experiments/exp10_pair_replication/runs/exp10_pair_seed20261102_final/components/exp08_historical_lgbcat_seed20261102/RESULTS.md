# EXP08 — Historical Feature Bridge Results

- Run: `exp08_historical_lgbcat_seed20261102`
- Status: **COMPLETED — REJECTED**
- Outer seed: `20261102`
- OOF AUC and public leaderboard AUC are different evaluation measures.

## Matched OOF comparison

| Model | Pooled OOF AUC | Delta vs Phase 2 | Five outer-fold AUCs |
|---|---:|---:|---|
| Strict Phase 2 | 0.77070815 | — | 0.769709, 0.777952, 0.766308, 0.773479, 0.766453 |
| Historical candidate | 0.77085788 | +0.00014973 | 0.770720, 0.777822, 0.765980, 0.773347, 0.766740 |
| Nested Phase 2 blend | 0.77104075 | +0.00033260 | 0.770484, 0.778185, 0.766390, 0.773585, 0.766895 |

### Paired nested-blend deltas by fold

| Fold | Phase 2 AUC | Blend AUC | Paired delta |
|---:|---:|---:|---:|
| 1 | 0.769709 | 0.770484 | +0.000774 |
| 2 | 0.777952 | 0.778185 | +0.000232 |
| 3 | 0.766308 | 0.766390 | +0.000081 |
| 4 | 0.773479 | 0.773585 | +0.000106 |
| 5 | 0.766453 | 0.766895 | +0.000443 |

## Historical family evaluation

| Family | Pooled OOF AUC | Fold AUCs | Positive folds vs Phase 2 |
|---|---:|---|---:|
| A_FINANCIAL | 0.76967247 | 0.769099, 0.776645, 0.764463, 0.772854, 0.765902 | 0 / 5 |
| B_RISK | 0.76987478 | 0.768338, 0.776374, 0.764859, 0.773746, 0.766542 | 2 / 5 |
| C_TEMPORAL | 0.76946382 | 0.768856, 0.775442, 0.765572, 0.771357, 0.766818 | 1 / 5 |
| D_DEMOGRAPHIC | 0.76999097 | 0.769244, 0.776222, 0.765193, 0.773172, 0.766459 | 1 / 5 |
| E_INQUIRY | 0.76972600 | 0.769445, 0.776200, 0.765047, 0.772645, 0.765764 | 0 / 5 |
| F_EXTERNAL | 0.76927548 | 0.768750, 0.776024, 0.763934, 0.772292, 0.766051 | 0 / 5 |
| H_INTERACTIONS | 0.76835231 | 0.767977, 0.775044, 0.762577, 0.771179, 0.765423 | 0 / 5 |

## Controls and evidence

- Reconstructed 660 eligible features from 669 historical formulas; 9 lineage-unsafe formulas excluded.
- Feature family selection, feature importance cutoffs, early stopping, and blend weights were selected on inner validation only.
- CatBoost GPU preflight: `{'available': True, 'task_type': 'GPU', 'device': 'CUDA:0', 'iterations': 2}`.
- CUDA XGBoost preflight: `{'available': True, 'version': '3.4.1', 'device': 'CUDA', 'parameters': {'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cuda', 'max_depth': 2, 'verbosity': 0}, 'cpu_fallback_used': False}`.
- GPU peak utilization: 100%; peak memory: 4901 MiB; samples: 1566.
- Advancement gate: pooled delta +0.00033260; 5 positive folds; mean paired delta +0.00032728 (SE 0.00012880); decision **REJECTED**.

## Previous experiment diagnostics

| Reference | Pooled OOF AUC | EXP08 blend delta |
|---|---:|---:|
| EXP04_EXP05_EXP06 | not recomputed | not comparable |

## Submission and reproducibility

- Component-only replication; no submission was emitted.
- Ordered test IDs validated: `True`.
- Runtime: 1763.1s total; 1759.4s across fold training.
- Input, source, reference, run-output, and feature-inventory hashes are recorded in `sha256_manifest.json`.
- The independent verifier is `verify_exp08.py`.

No official public leaderboard evaluation was performed.
