# Validation Summary

## Decision

**PASS — freeze the original Phase 2 candidate as a development candidate.**

The verification gate addressed repeatability on a second five-fold seed, a same-fold raw baseline, fold-local fee-frequency sensitivity, target-encoding label boundaries, candidate integrity, and observed GPU activity.

## Metrics

| Run / variant | OOF AUC | Delta vs matched raw baseline | Positive paired folds |
|---|---:|---:|---:|
| Original seed 42 raw baseline | 0.7625457907047022 | — | — |
| Original seed 42 Phase 2 | 0.7715614423106346 | +0.0090156516059324 | 5/5 |
| Seed 20261101 raw baseline | 0.7620389005999739 | — | — |
| Seed 20261101 Phase 2, pooled frequencies | 0.7711413866258005 | +0.0091024860258265 | 5/5 |
| Seed 20261101 Phase 2, fold-local frequencies | 0.7709638203765881 | +0.0089249197766141 | 5/5 |
| Seed 20261101 frequency decoder disabled | 0.7677737812232842 | +0.0057348806233103 | 5/5 |

Fold-local frequency decoding versus its disabled control improved OOF AUC by `+0.0031900391533038`, positive on 5/5 paired folds.

## Integrity and leakage checks

- Frozen candidate SHA-256 matches `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`.
- Candidate schema: `SK_ID_CURR,TARGET`; 61,500 rows; unique IDs; exact raw test/sample order.
- Prediction values are finite and within `[0, 1]`.
- OOF AUC was independently recomputed from saved prediction arrays.
- Inner OOF target encodings exclude a training row's own label; validation/test encodings use outer-fit labels only.
- Fold-local frequency reference IDs were independently verified against fold membership.
- 18 unit tests passed (baseline 7, Phase 1 3, Phase 2 3, final gate 5).
- CatBoost used explicitly configured GPU:0 with `nvidia-smi` sampling during every fit; LightGBM used CPU.
- No historical submission or champion predictions were used to train this independent model.

## Interpretation

The gate supports repeatable development-CV improvement across two tested seeds. It does not establish historical EXP85 reconstruction, prediction equality with the champion, an official leaderboard result, or generalization to an unrelated cohort.

The authoritative detail is in `REPORTS/FINAL_PHASE2_GATE_REPORT.md` and `EVIDENCE/FINAL_PHASE2_GATE_EVIDENCE.json`. The frozen candidate is not the seed-20261101 validation output; it remains the original seed-42 Phase 2 CSV.
