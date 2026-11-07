# Experiment Register

| ID | Hypothesis / change | Validation | Status | Result / decision |
|---|---|---|---|---|
| Stage 0 | ID-aligned comparison of Phase 2 and hybrid V5 against historical champion; no fitting | Exact 61,500 raw test IDs; no labels | Complete | Confirms prediction divergence and hybrid replay only; see `CORRELATION_GAP_ANALYSIS.md`. |
| GC-01 | Add fold-local category peer means/deviations to Phase 2 features | Same five seed-42 folds as frozen Phase 2; 76 feature columns; nested TE unchanged | Complete; rejected | OOF AUC `0.7699312062` vs `0.7715614423` Phase 2 (`-0.0016302361`); all five folds and both individual model components declined. No second seed. |
| TE-01 | Add nested target encodings for organization, occupation, income type, and education | Same exact five seed-42 folds; existing nested 5-fold encoder, inner seed 7 | Complete; rejected | OOF AUC `0.7709211321` vs `0.7715614423` Phase 2 (`-0.0006403102`); positive on 1/5 folds. Independent hash/fold/OOF/schema/ID/range/GPU checks passed. |
| XGB-01 | Independently train a GPU XGBoost model on raw rowwise application features with fold-fit one-hot categories | Exact five seed-42 folds; comparison-only fixed 80/20 rank blend; no validation-tuned weight | Complete; not promoted | XGB OOF AUC `0.7613696` vs Phase 2 `0.7715614` (`-0.0101919`). Fixed blend delta only `+0.0001020`, positive on 3/5 folds; requires replication. |
| XGB-02 | Replicate the unchanged XGB-01 configuration against frozen Phase 2 OOF on the independent seed | Exact frozen seed-20261101 folds; fixed 80/20 rank blend; no weight tuning | Complete; not promoted | XGB OOF AUC `0.7602661` vs Phase 2 `0.7709638` (`-0.0106977`). Blend delta `+0.0000251`, positive on 2/5 folds; gain is too small/inconsistent for promotion. |
| E-INT-01 | Add five fixed rowwise EXT_SOURCE/affordability interactions to Phase 2 | Seed-42 exact Phase 2 folds; compare paired fold AUCs with predeclared screen gate | Complete; rejected | OOF AUC `0.7712424` vs Phase 2 `0.7715614` (`-0.0003190`); mean paired-fold delta `-0.0003193`, positive on 2/5 folds. Gate failed; no second seed. |

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

## XGB-01 / XGB-02 outcome

The independent GPU XGBoost screen completed on seed 42 in `30.7` seconds, and its single predeclared replication completed on frozen seed `20261101` in `30.7` seconds. Both trained all five folds on CUDA (`cuda:0`); the direct tiny CUDA smoke fit also passed. OOF AUC was `0.7613695677` vs Phase 2 `0.7715614423` on seed 42 and `0.7602661213` vs fold-local Phase 2 `0.7709638204` on seed 20261101. The XGB model alone is materially weaker.

A fixed, untuned 80/20 rank diagnostic blend gained only `0.0001020347` AUC on seed 42 (3/5 positive folds) and `0.0000251043` on seed 20261101 (2/5 positive folds). OOF prediction correlations were `0.94550` and `0.94422`; squared-error correlations were `0.99480` and `0.99473`, indicating strongly shared error patterns. The second-seed result does not establish reliable complementarity; no blend/candidate is promoted and no third seed or weight sweep is authorized.

The run manifests, exact Phase 2 fold matches, saved OOF AUCs, ordered IDs/schema/probability bounds, and GPU samples were independently verified. Each run recorded six GPU samples; maximum utilization was 77% / 69%, and peak device memory was 318 MiB. Windows process peak working sets were about 1.12 GiB; peak private bytes about 1.73 GiB. The XGB test CSVs are diagnostic raw-only outputs, not submission recommendations.

## E-INT-01 outcome

The immutable run `runs/20261101T145000Z_epoly_interactions_seed42/` added five pre-specified, rowwise, label-free interactions to the frozen Phase 2 feature builder. It used the exact seed-42 Phase 2 folds and unchanged LightGBM/CatBoost configurations. Phase 2 OOF AUC recomputed from its saved OOF was `0.7715614423`; E_POLY scored `0.7712424145` (`-0.0003190278`). Mean paired-fold delta was `-0.0003192742` (SE `0.0002908994`), with only 2/5 positive folds. The predeclared advancement gate failed; no independent-seed replication or feature tuning is warranted.

Leakage review confirmed the wrapper rejects a `TARGET` column, adds no ID feature, and reads only rowwise application covariates. The frozen Phase 2 runner drops `TARGET` before building features; nested training encodings use disjoint inner-fit/apply rows, while outer validation and test encodings use outer-fit labels only. Exact folds, raw/reference/source/candidate hashes, recomputed AUC, 61,500 ordered test IDs, schema and probability bounds passed independent verification. Runtime was `284.0` seconds; LightGBM used CPU and CatBoost used GPU:0 on all five folds. Fifty-six telemetry samples showed maximum GPU utilization `44%` and peak VRAM `5,036 MiB`; peak process working set was `1.63 GiB`. Candidate SHA-256 is `6f2290bbff346d14997df9134c748c81c6f7bab41ee2f783d05e978dd4caae81`. This candidate is diagnostic and rejected, not a submission recommendation; official score was not evaluated.
