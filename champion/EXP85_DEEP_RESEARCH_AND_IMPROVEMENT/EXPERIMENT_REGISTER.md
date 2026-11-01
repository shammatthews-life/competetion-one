# Experiment Register

| ID | Hypothesis / change | Validation | Status | Result / decision |
|---|---|---|---|---|
| Stage 0 | ID-aligned comparison of Phase 2 and hybrid V5 against historical champion; no fitting | Exact 61,500 raw test IDs; no labels | Complete | Confirms prediction divergence and hybrid replay only; see `CORRELATION_GAP_ANALYSIS.md`. |
| GC-01 | Add fold-local category peer means/deviations to Phase 2 features | Same five seed-42 folds as frozen Phase 2; 76 feature columns; nested TE unchanged | Complete; rejected | OOF AUC `0.7699312062` vs `0.7715614423` Phase 2 (`-0.0016302361`); all five folds and both individual model components declined. No second seed. |
| TE-01 | Add nested target encodings for organization, occupation, income type, and education | Same exact five seed-42 folds; existing nested 5-fold encoder, inner seed 7 | Complete; rejected | OOF AUC `0.7709211321` vs `0.7715614423` Phase 2 (`-0.0006403102`); positive on 1/5 folds. Independent hash/fold/OOF/schema/ID/range/GPU checks passed. |

## GC-01 integrity contract

- Reference: frozen Phase 2 OOF vector and exact saved fold indices.
- Inputs: only hash-verified raw train/test/sample CSVs; no historical prediction caches or champion.
- Group keys: income type, education, occupation, organization.
- Numeric values: income, credit, annuity, goods price, EXT_SOURCE_1..3, age, employment duration.
- Fold boundary: group summaries use outer training rows only. Every fit row's own numeric value is removed from its group/global peer mean. Validation/test values do not contribute to fit statistics.
- Scoring: OOF ROC-AUC only, both pooled and per identical fold; candidate ID/schema/range checks.
- Hardware: explicit CPU LightGBM and GPU CatBoost; timestamped `nvidia-smi` sampling.
- Promotion: impossible from this seed alone; an independent predeclared seed is required.
- Official competition score: not evaluated.

## GC-01 outcome

The run completed in `336.4` seconds. Phase 2 + peer aggregates scored `0.7699312062`, `-0.0016302361` versus frozen Phase 2; paired deltas were negative on 5/5 folds. Component OOF AUC also fell for LightGBM (`-0.0012503`) and CatBoost (`-0.0021441`). The new features were used (about 23.7% of LightGBM gain and 30.2% of CatBoost importance) but did not generalize better on these folds. It is rejected, with no second-seed spend.

GPU telemetry: 67 samples, maximum utilization 57%, peak VRAM 4,903 MiB of 6,141 MiB. CPU process peak working-set sampling did not attach to the run; it is recorded as unavailable, not inferred. Per-fold elapsed times remain in the model run summary.

Each completed or failed run remains immutable under `runs/`, with its own code/input/output hashes and failure explanation if applicable.

## TE-01 outcome

The immutable run `runs/20261101T142000Z_core_category_te/` adds nested, smoothed encodings for organization, occupation, income type, and education using the existing Phase 2 inner 5-fold encoder (seed 7). The matched seed-42 pooled OOF AUC is `0.7709211321`, `-0.0006403102` versus frozen Phase 2; only 1/5 paired folds improved. The screen is rejected and no independent seed is warranted.

Independent verification recomputed pooled and per-fold AUC from saved OOF vectors, confirmed all five fit/validation index arrays exactly match the frozen release, verified every run-manifest input/output SHA-256, and checked the candidate's 61,500-row schema, unique ordered IDs, and finite `[0,1]` probabilities. Target-encoding tests passed (3/3); the complete research test suite passed (11/11). GPU telemetry recorded 40 samples, maximum utilization 58%, and peak VRAM use 4,903 MiB of 6,141 MiB. LightGBM ran on CPU and CatBoost on GPU:0. Official score was not evaluated.

The model runner recomputed candidate features and predictions from the raw CSVs. Previously saved baseline/Phase 1 OOF vectors are declared as comparison-only inputs by the frozen runner; they were not prediction-generation inputs. The frozen Phase 2 OOF was used only as the matched validation reference.
