# EXP85 Phase 2 Complete Pipeline — Execution Report

- Status: **PASS**
- Mode: `full`
- Run ID: `phase2_complete_raw_20261101T211310`
- Runtime seconds: `206.662`
- OOF AUC: `0.7715805073`
- Frozen OOF AUC: `0.7715614423`
- OOF AUC delta: `+0.0000190650`
- Submission SHA-256: `0b0fd01b5150362c9887033a4b202c6ea7c51443cf42b566237a3055d117835c`
- Reproduction gate: **PASS**
- Official competition score: **NOT EVALUATED**

## Per-fold validation

| Fold | LightGBM AUC | CatBoost AUC | Blend AUC | Frozen blend AUC | Delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.76656212 | 0.76781919 | 0.76827280 | 0.76827281 | -0.00000001 |
| 2 | 0.77305641 | 0.77282859 | 0.77410892 | 0.77412739 | -0.00001846 |
| 3 | 0.77608815 | 0.77390699 | 0.77616390 | 0.77616376 | +0.00000014 |
| 4 | 0.76808564 | 0.76859726 | 0.76956077 | 0.76956077 | +0.00000000 |
| 5 | 0.76829693 | 0.76940101 | 0.77012264 | 0.77001386 | +0.00010878 |

## Device evidence

- LightGBM: `CPU`
- CatBoost: `GPU:0`
- GPU preflight: `PASS`
- GPU telemetry samples: `149`

## Reproduction comparison

- OOF blend correlation: `0.99989898`
- Test blend correlation: `0.99997961`
- Submission correlation: `0.99997961`
- Promoted submission: `D:\gci\_gci_push_tmp\champion\EXP85_PHASE2_COMPLETE_SUBMISSION\submissions\phase2_complete_raw_20261101T211310\submission_phase2_complete.csv`

## Provenance

- Raw training/test/sample inputs were hash-verified and no historical EXP85 predictions were used.
- The frozen Phase 2 candidate and its saved OOF/test outputs were read-only comparison references.
- Authoritative frozen baseline/Phase 1/Phase 2 source SHA-256 values are recorded in the JSON report and run manifest.
- Full raw-data retraining is independent of historical-cache reconstruction.
