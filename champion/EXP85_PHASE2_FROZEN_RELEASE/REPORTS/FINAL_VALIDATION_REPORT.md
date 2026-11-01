# Final Validation Report

## Result

The strongest supported independent raw-data candidate is the fixed 50/50 LightGBM/CatBoost blend from Phase 2, using label-safe nested target encodings. It achieved **0.7715614423 OOF AUC**, an increase of **0.0090156516** over the frozen raw-only baseline (**0.7625457907**).

This is a local five-fold development-CV result. It is not the historical EXP85 0.801340 leaderboard score. No submission was sent to the competition, and no official score was evaluated.

## Final second-seed gate

The independent final gate used a new five-fold seed (20261101) and retrained a raw baseline, pooled-frequency Phase 2, and fold-local-frequency Phase 2 on identical folds. Their recomputed OOF AUCs were 0.7620389006, 0.7711413866, and 0.7709638204 respectively. Pooled and fold-local Phase 2 improved on the matched raw baseline on 5/5 folds (+0.0091024860 and +0.0089249198 full-OOF deltas). In a controlled ablation, fold-local frequency decoding exceeded the decoder-disabled variant by +0.0031900392 OOF AUC on 5/5 folds.

The verifier independently recalculated OOF AUCs from saved arrays, verified the 27 main-run and 13 ablation output records, raw/source hashes, fold splits, target-encoding boundaries, frequency-map references, candidate schema/IDs/range, CSV-to-array agreement, and GPU observations. It returned PASS without retraining. The original Phase 2 candidate was not overwritten; its SHA-256 remains `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`.

The detailed report is `FINAL_PHASE2_GATE_REPORT.md`; machine-readable independent results are in `FINAL_PHASE2_GATE_EVIDENCE.json`. The gate's PASS confirms the stated raw-only development validation criteria. It does not establish 99% correlation to the historical champion or reproduce its official score.

## Candidate and integrity

- Candidate file: `runs/phase02_key_te_20261101_181500/submission_key_target_encoding.csv`
- Candidate SHA-256: `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`
- OOF blend SHA-256: `f9a8ffd7f42c1ee93b53a560d0f0f7acc255a286b8297c5cacfeae4ccabe0466`
- Test blend SHA-256: `9b73c8612183da00fbd325d04f993b7a7f3be2c56841b2a4672f6fc029d5c3fe`
- Candidate schema: `SK_ID_CURR,TARGET`; 61,500 rows; 61,500 unique IDs; exact test/sample ID order; finite probabilities in [0, 1].
- Recomputed candidate-to-test-blend maximum absolute error: `4.999961555895993e-11`, consistent with CSV decimal formatting.
- Every baseline and Phase 1/2/3 output hash was verified: 8 + 23 + 15 + 12 records. All recorded input hashes were checked, including raw train/test/sample inputs and recorded source/summary dependencies (4 + 3 + 8 + 6 input records).
- Saved two- and three-model blends were independently recomputed. Candidate schema, ID uniqueness/order, bounds, and row counts were checked for all three saved experiment submissions.
- Fold assignment arrays from Phase 2 and Phase 3 match a fresh StratifiedKFold reconstruction exactly: five splits, shuffle enabled, seed 42.

Full machine-readable results are in `VALIDATION_EVIDENCE.json` and `MODEL_LINEAGE_MANIFEST.json`. The executable inference-only check is `source/validate_experiment_artifacts.py`.

## Experiment scores

| Candidate | OOF AUC | Mean fold AUC | Mean fold lift vs baseline | Positive folds |
|---|---:|---:|---:|---:|
| Raw baseline, fixed 50/50 | 0.7625457907 | 0.7625729373 | — | — |
| Fee/product feature block, fixed 50/50 | 0.7648301137 | 0.7648746393 | +0.0023017020 | 5/5 |
| Nested key encodings, fixed 50/50 | **0.7715614423** | **0.7716277181** | **+0.0090547809** | **5/5** |
| XGBoost only | 0.7704971413 | 0.7705579146 | +0.0079849773 | 5/5 |
| Equal 1/3 three-model ensemble | 0.7716955766 | 0.7717630132 | +0.0091900759 | 5/5 |

The three-model ensemble raises full OOF AUC by only `0.0001341343` versus Phase 2 and improves just 3/5 individual folds. It is not adopted. The Phase 2 candidate is the stronger, simpler and more independently supported choice.

## Execution and hardware

Completed model-training runs (as recorded in their experiment summaries):

| Run | Status | Runtime | Hardware evidence |
|---|---|---:|---|
| Raw baseline | Completed | 736.717 s | LightGBM CPU; CatBoost GPU:0 |
| Phase 1 fee/product features | Completed | 403.784 s | LightGBM CPU; CatBoost configured through the same GPU training helper |
| Phase 2 nested encodings | Completed | 331.003 s | LightGBM CPU and CatBoost GPU:0 recorded across all folds |
| Phase 3 XGBoost | Completed | 67.008 s | All five saved fold configurations specify `device=cuda:0` and `grow_gpu_hist` |

The recorded device was an NVIDIA GeForce RTX 4050 Laptop GPU, driver 592.82, 6,141 MiB. LightGBM ran on CPU because the installed build uses its CPU backend. The final artifact validator did not train models; it completed in about 1.6 seconds with exit code 0.

## Tests

The repository's standard-library unittest suites were run directly because `pytest` is not installed:

- Phase 1 decoded-features tests: **3 passed**.
- Phase 2 nested-encoding/leakage-safety tests: **3 passed**.
- Raw baseline feature tests: **7 passed**.
- Total: **13 passed**.

The artifact validator command was:

```powershell
python champion\FINAL_RESEARCH_AND_OPTIMIZATION\source\validate_experiment_artifacts.py
```

Result: **PASS**, exit code 0, approximately 1.6 seconds; no training was performed.

## Interpretation and next step

The target encoding is nested and leakage-safe with respect to labels. The original fee ratio frequency decoder is transductive: pooled unlabeled train/test covariates influence each row's decoder features. The new-seed fold-local sensitivity retained nearly all of the pooled-frequency lift; the selected original candidate remains unchanged.

This work does not resolve historical V20/V080041 lineage blockers, rebuild the frozen EXP85 model chain, or establish an official competition score. Preserve the historical champion, Recovery V5 replay, and this raw-data candidate as distinct artifacts.
