# EXP07 feature availability and hypothesis inventory

## Selected experiment block

| Feature | Formula | Source columns | Leakage risk and control | Availability | Rationale |
|---|---|---|---|---|---|
| `EXP07_SIG_COUNT_LOG` | `log1p(number of other fitting rows with the same application signature)` | `int(DAYS_BIRTH - DAYS_REGISTRATION)`, `int(DAYS_BIRTH - DAYS_ID_PUBLISH)`, `CODE_GENDER`, `NAME_EDUCATION_TYPE` | Covariate-only count. Counts for fitting rows use only the current fitting partition and subtract that row; validation/test counts map from fitting rows only. | Available from the raw application CSVs; fold-local implementation is reproducible. | Represents repeated demographic/time signatures without asserting that records are the same borrower or using the applicant ID. |
| `EXP07_SIG_TARGET_RATE` | `(signature target sum + 20 * fit-partition target mean) / (signature count + 20)` | The same signature plus `TARGET` from fitting rows only | High leakage risk if a row contributes its own label. Each fit row is encoded by a five-fold cross-fit that excludes its whole held-out subfold; outer validation/test use outer-fit labels only. The inner stopping pipeline rebuilds it from inner-fit labels only. | Available from raw training data; included only as nested cross-fitted features. | Tests whether a small repeated-signature cohort carries residual risk information not represented by the existing broad-category encodings. |

The signature is the exact key pattern documented in the historical Hasan
feature builder, but EXP07 does **not** copy its globally pooled frequency or
seed-42 supervised-memory implementation. The EXP07 implementation is
independently defined and fold-local. The key is a coarse applicant
signature, not verified person identity; no borrower-link claim is made.

## Existing and reviewed feature sources

| Source family | Availability classification | EXP07 treatment |
|---|---|---|
| Raw `train.csv`, `test.csv`, `sample_submission.csv` | Available, SHA-256 verified | Sole feature/label inputs; applicant IDs are used only for alignment and submissions. |
| Frozen Phase 2 rowwise features, fee/product decoding, and nested target encodings | Available and reproducible | Fixed baseline feature pipeline; source files are hash-checked and not modified. |
| Phase 2 strict inner-stopping OOF and test reference | Available and integrity-verified | Fixed comparison and blend anchor; never used as labels or model features. |
| Current Phase 2 122-feature schema | Available and reproducible | Audited for duplication; the new signature fields are appended, not substituted. |
| Historical 669-feature builder | Source and mathematical notes available locally; generated matrices are not part of this experiment | Read-only design reference. Historical global feature selection and pooled/supervised group features are not imported as validated inputs. |
| Historical V17/V55/XGB10 caches and producer lineage | Historically cached only / lineage incomplete | Excluded from training, feature generation, weight selection, and candidate selection. |
| Bureau, previous-application, POS, installment, and credit-card tables | Missing from the available raw-data directory | No relational aggregates are fabricated. |
| Generic category peer aggregates, core-category target encodings, five EXT_SOURCE/affordability interactions | Available as completed negative screens | Not repeated: GC-01, TE-01, and E-INT-01 declined versus Phase 2. |
| Raw CUDA XGBoost ensemble member | Available as two completed screens, with negligible and inconsistent blend lift | Not rerun or weight-swept: XGB-01/02 standalone performance was about 0.010 AUC lower and prior fixed-blend lift was not stable. |

## Historical design references and boundary

- `D:\\gci\\champion\\05_FEATURE_ENGINEERING\\FEATURE_BUILDERS\\build_hasan_600_features.py`
- `D:\\gci\\champion\\05_FEATURE_ENGINEERING\\FEATURES_669\\FEATURES_669_MATHEMATICAL_SPECIFICATION.md`
- Frozen raw feature builder:
  `champion/EXP85_PHASE2_FROZEN_RELEASE/SOURCE/project/source/run_independent_baseline.py`

The local Hasan builder's exact key uses the integer-cast time differences,
gender, and education above. Its historical target-memory generation is not accepted
as leakage-safe for this experiment: EXP07 independently cross-fits every
training-row encoding inside the current fit partition.
