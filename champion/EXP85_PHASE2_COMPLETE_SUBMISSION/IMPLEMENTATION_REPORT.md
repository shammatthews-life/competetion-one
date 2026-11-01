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
- GPU telemetry: 149 samples; mean utilization `47.35%`, peak `99%`, peak memory used `5003 MiB` (GPU total `6141 MiB`).

## Inputs and blend

- Train rows: `171202`; test/submission rows: `61500`.
- Input SHA-256: train `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`; test `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`; sample `6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db`.
- Final blend: `0.5 * LightGBM probability + 0.5 * CatBoost probability`; no post-processing.
- Reproduction limits: global OOF AUC delta `<= 0.001`; per-fold delta `<= 0.003`; OOF/test/submission Pearson `>= 0.99`. All checks passed.

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

## Executed validation commands

| Check | Exit code | Result |
|---|---:|---|
| `scripts\run_full_pipeline.py --data-dir D:\gci\inpuy --run-id phase2_complete_raw_20261101T211310` | 0 | PASS |
| `scripts\verify_pipeline.py --run-dir runs\phase2_complete_raw_20261101T211310 --data-dir D:\gci\inpuy` | 0 | PASS; 39 run files verified |
| `scripts\verify_submission.py --submission submissions\phase2_complete_raw_20261101T211310\submission_phase2_complete.csv --data-dir D:\gci\inpuy` | 0 | PASS; 61,500 rows |
| Frozen release `REPRODUCTION\verify_freeze.py` with raw CSV paths | 0 | PASS; original Phase 2 candidate and 129 manifest files unchanged |
| `python -B -m unittest discover -s tests -p test_*.py -v` | 0 | 22 tests passed |
| Python AST parse over package source/scripts/tests | 0 | 24 files parsed |
| `scripts\package_manifest.py create`, then `verify` | 0 | PASS; 75 package files |

The JSON execution report contains full per-fold metrics, input/source/config hashes, software versions, fold indices, GPU sample telemetry, and prediction comparisons.
