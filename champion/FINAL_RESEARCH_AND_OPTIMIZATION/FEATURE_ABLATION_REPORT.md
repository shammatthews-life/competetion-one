# Feature-Block Ablation Report

## Method

Blocks were added sequentially while keeping the five outer folds, LightGBM/CatBoost configurations, and fixed 50/50 probability blend constant. Each score is generated from held-out OOF predictions. There was no OOF weight fitting. This measures the incremental value of each block in this pipeline; it is not a causal attribution to every individual feature.

## Results

| Model/block | Mean fold AUC without block | Mean fold AUC with block | Mean paired delta | Positive folds |
|---|---:|---:|---:|---:|
| LightGBM: fee/product feature block | 0.7614895 | 0.7632860 | +0.0017964 | 5/5 |
| CatBoost: fee/product feature block | 0.7601607 | 0.7628614 | +0.0027007 | 5/5 |
| 50/50 blend: fee/product feature block | 0.7625729 | 0.7648746 | +0.0023017 | 5/5 |
| LightGBM: nested key target encodings on fee/product model | 0.7632860 | 0.7704179 | +0.0071319 | 5/5 |
| CatBoost: nested key target encodings on fee/product model | 0.7628614 | 0.7704818 | +0.0076204 | 5/5 |
| 50/50 blend: nested key target encodings on fee/product model | 0.7648746 | 0.7716277 | +0.0067531 | 5/5 |
| Equal 1/3 XGBoost ensemble member, relative to Phase 2 | 0.7715614 | 0.7716956 | +0.0001341 full-OOF AUC | 3/5 |

Full OOF AUC for the selected Phase 2 blend is **0.7715614**, versus **0.7625458** for the frozen raw-only baseline. Mean fold AUC and full OOF AUC differ slightly because AUC is not a linear average across folds.

## Feature details

Phase 1 added twelve label-free values derived from:

- The archived 129.5 annuity fee hypothesis and frequency competition between rounded raw and fee-adjusted annuity/credit ratios.
- Rounded decoded annuity/credit tiers.
- The archived 0.0132 credit/goods product-key hypothesis, nearest integer key, validity flag, and key distance.

The fee decoder marked 158,042 rows (about 74.98% of cash-loan rows); the product key was valid for 135,609 rows (about 64.34% of cash-loan rows). These counts are diagnostic evidence that the hypotheses occur in this data, not proof that the underlying contract mechanism is universal.

Phase 2 added smoothed target encodings and log-counts for decoded ratio tiers Q3/Q4/Q5 and the integer product key. Smoothing values were fixed from the supplied source code (100, 30, 30, 50). Cross-fitting was nested by outer fold and inner seed 7.

The Phase 2 gain was concentrated in decoded fee ratio encodings. In summed fold feature importance, `TE_FEE_Q4` and `TE_FEE_Q5` were among the top three features for both model families. Importance is diagnostic, not an ablation result.

## Limits

The fee and product features were added as one combined block, so the current runs do not establish how much gain comes from the fee decoder versus the product-key decoder individually. Similarly, Phase 2 evaluates the four target-encoding keys as one block; importance suggests Q4/Q5 are influential, but removing individual keys has not yet been tested. Do not claim per-feature causality from these grouped comparisons.

The Phase 1/2 fee-frequency transform uses all available unlabeled train/test covariates, matching the transductive research recipe. It does not use labels, but CV includes held-out rows in the unlabeled frequency table. Therefore these scores validate the stated transductive competition pipeline, not an inductive future-cohort pipeline. A fold-local covariate-frequency sensitivity run is recommended before final submission.

The supplied `fast_screen.py` candidate rankings were not used: its random-noise control missed its stated tolerance, including after three model seeds.

## Second-seed frequency sensitivity and ablation

On outer seed 20261101, the fold-local frequency version scored 0.7709638204 OOF AUC, versus 0.7711413866 for pooled frequencies and 0.7620389006 for a newly trained same-fold raw baseline. The fold-local variant improved over baseline on all five folds and was only 0.0001775662 below pooled frequencies.

A controlled fold-local decoder ablation retained rowwise and product-key features, the pointwise fee-adjusted ratio, and nested target encodings while disabling frequency-derived decoding. The decoder-enabled variant scored 0.7709638204 versus 0.7677737812 disabled, a +0.0031900392 OOF AUC difference, positive on all five paired folds. This measures the frequency-decoding block in the tested feature stack; it does not separate every underlying formula or establish performance on other cohorts.

The fold-local sensitivity addresses the transductive-frequency caveat for this validation, but the selected original seed-42 submission remains the pooled-frequency Phase 2 candidate. The new-seed variants are validation evidence and do not replace it.
