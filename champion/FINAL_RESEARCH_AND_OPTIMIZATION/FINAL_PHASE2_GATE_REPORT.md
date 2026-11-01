# Final Phase 2 Validation Gate

## Verdict: PASS

The independent Phase 2 improvement repeated on a second, predeclared five-fold seed. The fold-local fee-frequency version also beat a matched decoder-disabled control on all five folds. Target encodings, candidate integrity, raw-input hashes, and per-fold GPU use passed verification.

This PASS applies to the requested independent-model validation gate only. It is not 99% prediction correlation with the historical champion, does not reproduce the historical 0.801340 leaderboard score, and is not an official competition evaluation.

## Same-seed comparison

The new outer seed was **20261101** (the prior runs used seed 42). Baseline, pooled Phase 2, and fold-local Phase 2 were trained with identical outer folds and the same model hyperparameters; model random seeds were derived from the new outer seed and fold number. The fixed 50/50 LightGBM/CatBoost blend was retained without weight tuning.

| Model/features | OOF AUC | Mean fold AUC | Paired fold delta vs raw baseline | Improved folds |
|---|---:|---:|---|---:|
| Raw-only baseline | 0.7620389006 | 0.7620242894 | — | — |
| Phase 2, pooled unlabeled fee frequencies | 0.7711413866 | 0.7711683434 | +0.011128, +0.008804, +0.008587, +0.008457, +0.008744 | 5/5 |
| Phase 2, fold-local fee frequencies | 0.7709638204 | 0.7709896152 | +0.010298, +0.008352, +0.009197, +0.008468, +0.008511 | 5/5 |

Relative to the original seed-42 OOF comparison, the Phase 2 blend improved by `+0.0090156516` (0.7625457907 to 0.7715614423). On the new seed, the pooled and fold-local Phase 2 variants improved by `+0.0091024860` and `+0.0089249198` against their newly trained, same-fold baseline. Both improved on all five folds.

## Fold-local fee-frequency ablation

To isolate the decoder's contribution, the fold-local Phase 2 stack was compared with a same-seed, same-fold control that disabled frequency-derived decoding. The control set `FEE_DECODED` and both frequency counts to zero, reverted decoded-ratio features and target-encoding keys to the raw annuity/credit ratio, and retained rowwise features, product-key features, the pointwise fee-adjusted ratio, and nested target encodings.

| Variant | OOF AUC | Mean fold AUC |
|---|---:|---:|
| Frequency decoder disabled | 0.7677737812 | 0.7678028548 |
| Fold-local frequency decoder enabled | 0.7709638204 | 0.7709896152 |
| Enabled minus disabled | **+0.0031900392** | **+0.0031867604 mean paired fold delta** |

Paired fold deltas were `+0.0038824`, `+0.0036642`, `+0.0024466`, `+0.0017002`, and `+0.0042403`: **5/5 positive**. This supports a useful contribution from the frequency-decoding block in the tested feature/model stack. It does not imply the block's value will be identical on other cohorts.

The local frequency map for each outer fold used only that fold's fit-row IDs and covariates. Independent verification recomputed each reference-ID hash from the fold assignments and checked that validation and test IDs were absent from the reference. No target is used in this unsupervised frequency transform.

## Leakage and schema audit

- The raw train file has 34 columns and test has 33; the only train-only column is `TARGET`. Predictor columns otherwise match exactly.
- Features are constructed from the raw application train/test CSVs. The ID and target are not passed as model features; no payment, repayment, delinquency, default, outcome, or label columns are present in test predictors.
- Outer-training target encodings are inner five-fold OOF encodings; an encoded training row is not used to fit its own target map.
- Outer validation and test target encodings use outer-training labels only. Fit/apply rows are disjoint; changing apply-row labels cannot alter their encodings.
- The new-seed baseline and both Phase 2 variants use identical reconstructed folds. AUC was recalculated independently from saved OOF arrays.
- No historical champion predictions or labels were read for this gate. No weights were tuned against held-out labels.

## Candidate integrity and preserved selection

The selected candidate remains the original Phase 2 artifact; the second-seed outputs are validation evidence and do not replace it:

- Path: `runs/phase02_key_te_20261101_181500/submission_key_target_encoding.csv`
- SHA-256: `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`
- 61,500 rows; exactly `SK_ID_CURR,TARGET`; all IDs unique and in exact test/sample order.
- Finite predictions in [0, 1]; saved candidate values match the test blend within CSV rounding (`< 5.1e-11` maximum absolute difference).

The new-seed pooled and fold-local candidates and the frequency-disabled control were also independently checked. Their run manifests, raw input hashes, source hashes, fold assignments, blended predictions, candidate schemas, ID ordering, uniqueness, finite/range checks, and CSV-to-array agreement passed.

## GPU evidence and execution

CatBoost was explicitly configured for GPU:0 and LightGBM ran on CPU. `nvidia-smi` was sampled every 0.5 seconds during each CatBoost fit:

- New-seed main gate: 15 CatBoost fits, 1,709 samples total; per-fold maximum GPU utilization ranged from 43% to 99%, with reported used memory from 5,031 to 5,366 MiB.
- Frequency-disabled ablation: five CatBoost fits; per-fold maximum utilization ranged from 52% to 95%.
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, driver 592.82, 6,141 MiB.

The complete new-seed gate ran in **1,327.31 seconds** (22.1 minutes). The focused decoder ablation ran in **330.21 seconds**. Independent verification recalculated the metrics, checked 27 main-run outputs and 13 ablation outputs plus all input/source/comparison hashes, and completed without fitting models. All commands completed successfully after correcting an initial runner startup issue: a reporting-only `device` field was removed from CatBoost constructor arguments before the successful run; no output from that failed initialization was reused.

Five final-gate unit tests passed, including explicit checks of local frequency reference boundaries, fee-feature label independence, target-encoding self-exclusion, apply-label independence, and the disabled-decoder control. The earlier baseline, Phase 1, and Phase 2 tests also pass (18 tests total across the four suites).

Machine-readable evidence: `FINAL_PHASE2_GATE_EVIDENCE.json`. Reproduction code: `experiments/phase_04_final_validation_gate/`. No competition submission was made; official score remains `NOT_EVALUATED`.

## Decision

**Freeze the existing Phase 2 candidate as the independent raw-data submission candidate.** The new evidence confirms its improvement is repeatable and that fold-local frequency construction preserves the gain without using validation/test covariates in the frequency reference. Do not change or submit the candidate until the user decides.
