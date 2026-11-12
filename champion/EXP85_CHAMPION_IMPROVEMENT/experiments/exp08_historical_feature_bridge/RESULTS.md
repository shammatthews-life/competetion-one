# EXP08 — Historical Feature Bridge Results

- Run: `exp08_historical_seed20261101_v3`
- Status: **COMPLETED — REJECTED**
- Outer seed: `20261101`
- OOF AUC and public leaderboard AUC are different evaluation measures.

## Matched OOF comparison

| Model | Pooled OOF AUC | Delta vs Phase 2 | Five outer-fold AUCs |
|---|---:|---:|---|
| Strict Phase 2 | 0.77034958 | — | 0.766069, 0.772637, 0.771425, 0.768536, 0.773303 |
| Historical candidate | 0.77021653 | -0.00013306 | 0.765937, 0.771766, 0.771958, 0.768305, 0.773241 |
| Nested Phase 2 blend | 0.77054757 | +0.00019798 | 0.766172, 0.772529, 0.771909, 0.768738, 0.773571 |

### Paired nested-blend deltas by fold

| Fold | Phase 2 AUC | Blend AUC | Paired delta |
|---:|---:|---:|---:|
| 1 | 0.766069 | 0.766172 | +0.000103 |
| 2 | 0.772637 | 0.772529 | -0.000108 |
| 3 | 0.771425 | 0.771909 | +0.000484 |
| 4 | 0.768536 | 0.768738 | +0.000202 |
| 5 | 0.773303 | 0.773571 | +0.000268 |

## Historical family evaluation

| Family | Pooled OOF AUC | Fold AUCs | Positive folds vs Phase 2 |
|---|---:|---|---:|
| A_FINANCIAL | 0.76924664 | 0.764901, 0.771728, 0.771469, 0.766990, 0.771512 | 1 / 5 |
| B_RISK | 0.76943618 | 0.765617, 0.772000, 0.771164, 0.767832, 0.770922 | 0 / 5 |
| C_TEMPORAL | 0.76916915 | 0.764803, 0.770641, 0.772186, 0.766811, 0.771815 | 1 / 5 |
| D_DEMOGRAPHIC | 0.76909692 | 0.765341, 0.770965, 0.770920, 0.767167, 0.771424 | 0 / 5 |
| E_INQUIRY | 0.76919173 | 0.764820, 0.771752, 0.771596, 0.767686, 0.770406 | 1 / 5 |
| F_EXTERNAL | 0.76900470 | 0.764449, 0.771004, 0.771063, 0.767409, 0.771421 | 0 / 5 |
| H_INTERACTIONS | 0.76811857 | 0.763008, 0.770719, 0.769811, 0.765462, 0.771941 | 0 / 5 |

## Controls and evidence

- Reconstructed 660 eligible features from 669 historical formulas; 9 lineage-unsafe formulas excluded.
- Feature family selection, feature importance cutoffs, early stopping, and blend weights were selected on inner validation only.
- CatBoost GPU preflight: `{'available': True, 'task_type': 'GPU', 'device': 'CUDA:0', 'iterations': 2}`.
- CUDA XGBoost preflight: `{'available': True, 'version': '3.4.1', 'device': 'CUDA', 'parameters': {'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cuda', 'max_depth': 2, 'verbosity': 0}, 'cpu_fallback_used': False}`.
- GPU peak utilization: 100%; peak memory: 5121 MiB; samples: 1511.
- Advancement gate: pooled delta +0.00019798; 4 positive folds; mean paired delta +0.00018965 (SE 0.00009724); decision **REJECTED**.

## Previous experiment diagnostics

| Reference | Pooled OOF AUC | EXP08 blend delta |
|---|---:|---:|
| EXP04_CatBoost_Lossguide | 0.76803271 | +0.00251486 |
| EXP05_Residual_MLP | 0.76372731 | +0.00682026 |
| EXP06_FT_Transformer | 0.76402234 | +0.00652523 |

## Submission and reproducibility

- Submission rows/schema: 61500 / `SK_ID_CURR, TARGET`.
- Submission SHA-256: `3c488eaa5cdc828cee3a475ed6c9bcf04f255ccd74a5ef1a047b03ca680a54ac`.
- Ordered test IDs validated: `True`.
- Runtime: 1670.4s total; 1661.6s across fold training.
- Input, source, reference, run-output, and feature-inventory hashes are recorded in `sha256_manifest.json`.
- The independent verifier is `verify_exp08.py`.

No official public leaderboard evaluation was performed.
