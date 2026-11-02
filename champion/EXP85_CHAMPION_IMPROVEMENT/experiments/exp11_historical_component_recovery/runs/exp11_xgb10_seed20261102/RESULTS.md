# EXP11 — Historical XGB10 Component Recovery

- Decision: **REJECTED**.
- This is an OOF research result, not an official leaderboard score.
- No submission was created and the frozen historical champion was not modified.

## Lineage and method

- Selected stream: EXP85 `xgb10`, historically weighted 25.96% within its 8-engine suite.
- Historical reference: cached OOF AUC recomputes to 0.77433064; it is provenance only and its fold IDs are unavailable.
- EXP11: raw-data XGBoost refit using the available historical generator and five matched folds.
- Lineage discrepancy: the available source emits 173 full-data features, although documentation claims 239; per-fold feature counts are recorded in the summary.
- Fixed historical tree parameters; fold-local categorical mappings and nested early stopping prevent outer-validation contamination.
- RTX 4050 CUDA was required; CPU fallback is rejected.

## Pooled OOF comparison

| Stream | Pooled OOF AUC | Delta vs Phase 2 |
|---|---:|---:|
| Shared Phase 2 | 0.77076772 | +0.00000000 |
| Strict XGB10 refit | 0.75988817 | -0.01087955 |

## Per-fold metrics

| Fold | Phase 2 AUC | XGB10 AUC | Delta |
|---:|---:|---:|---:|
| 1 | 0.76970934 | 0.75762172 | -0.01208762 |
| 2 | 0.77795252 | 0.76427790 | -0.01367462 |
| 3 | 0.76660063 | 0.75668093 | -0.00991970 |
| 4 | 0.77349758 | 0.76262572 | -0.01087186 |
| 5 | 0.76645277 | 0.75847188 | -0.00798089 |

## Diagnostics

- Prediction correlation with Phase 2: 0.944479.
- Residual correlation: 0.994499.
- Absolute-error correlation: 0.992489.
- Phase 2 / XGB10 MAE: 0.133900 / 0.135297.
- EXT_SOURCE missingness subgroup AUCs are recorded in `experiment_summary.json`.
- RTX 4050 peak utilization / VRAM: 92% / 187 MiB; runtime 66.0s.

## Advancement decision

- Required gain: +0.001000; observed: -0.01087955.
- Required positive folds: 4/5; observed: 0/5.
- OOF threshold gate: **FAIL**; the final research gate also requires independent saved-model replay.
- See `experiment_summary.json`, `component_lineage.json`, `SHA256_MANIFEST.json`, and `VERIFICATION.json` for machine-readable evidence.

EXP11 does not establish an official public score or prove the preserved 0.801340 champion has been beaten.
