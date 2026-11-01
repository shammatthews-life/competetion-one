# Feature Lineage and Leakage Boundaries

## Historical feature systems

### Hasan 669 → selected 550

Project reports describe 669 raw-derived features including financial ratios, demographic/time intervals, interactions, copula-style joint-risk transforms, and Merton-inspired default-distance features. `screen_and_select_features.py` ranks features using LightGBM gain and exports 550. The Phase 3 parity report gives matching hashes for its reconstructed matrices:

- 669 train `4050b9da...65dd347`; test `232cb5e5...024beb`.
- selected 550 train `bb18a319...57eacd`; test `7b3cf10a...d82482`.

These are report-supplied hashes and historical artifacts; this stage did not regenerate multi-gigabyte matrices. A full independent check must hash the exact current files and verify source/config/schema. A global feature screening performed before fold evaluation is not an unbiased OOF feature-selection procedure; use inner-fold screening for any new CV claim.

### V17, V55, and XGB10 views

- V17 includes base, E1, E1A, group/view and demographic-residual feature manifolds; a final multi-view fusion is ranked before entering the suite.
- V55 includes core/extended/demographic/monotone views and a CatBoost product-specialist view.
- XGB10 has a separately documented 239-feature financial/solvency/demographic manifold.

The exact historical view arrays and their feature builder revisions are not equivalent merely because reconstructed generators accept raw CSVs.

### Independent Phase 2 features

The frozen candidate adds decoded fee/annuity-credit ratio features, product tiers/keys, and nested smoothed target encodings. For outer-training rows, encodings are inner OOF; outer validation/test keys use only outer-training labels. Pooling unlabeled train/test covariates for frequency decoding is explicitly transductive; a fold-local sensitivity test retained nearly all lift.

## Leakage checklist for future work

| Transformation | Training-fold requirements | Validation/test rules |
|---|---|---|
| Numeric imputation/scaling | Fit on fold training only where model is sensitive | Transform validation/test from fitted parameters |
| Count/frequency features | Fold-local counts for strict inductive estimate; pooled unlabeled train/test only as a separately labeled transductive test | No labels in counts; do not claim inductive CV if validation covariates enter counts |
| Target encoding | Inner OOF for each outer-training row; smoothing tuned only in inner data | Outer validation/test mappings fit on outer training labels only |
| Feature selection | Selection nested inside outer training, preferably inner folds | Never screen with all outer labels before scoring that outer validation |
| Model stacking | Base OOF rows only for meta-model training; fold-local base models for outer validation | No in-fold base predictions used as stacker inputs |
| Calibration/thresholding | Fit only on training/OOF predictions inside outer fold | Apply frozen fold-trained calibrator to validation/test |
| Test graph/transductive smoothing | Re-create graph from the permitted unlabeled cohort; no labels from validation/test nodes | Report as transductive; do not use champion predictions to define labels |
| Historical candidate comparison | No fit/tune path at all | ID-align and label as diagnostic-only |

## Infinity handling

The Phase 4 report documents `HOUSING_STABILITY` infinities and a LightGBM behavior change when infinities were replaced with NaN. This is a concrete transformation-parity hazard: `+inf` and missing are not interchangeable. Each builder must define explicit handling and record counts before model fitting; do not globally replace infinities without a model-specific reason.
