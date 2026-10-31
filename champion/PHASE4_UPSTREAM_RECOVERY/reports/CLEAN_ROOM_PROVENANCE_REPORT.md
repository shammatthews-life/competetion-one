# Clean-Room Provenance Audit

## Current canonical manifest

`02_EXP85_REBUILT/rebuilt_artifacts/RECONSTRUCTION_PROVENANCE_MANIFEST.json`
has SHA-256
`72390623f5e65832cfa0bb17f8eaced80506ef5304f62e34bd916d6826f0bce5`.
Its recorded status is `ARTIFACTS_VERIFIED`; its recorded Git commit is
`269affc900484d0a6131475ac855c33f3032f58f`. This status verifies existing
artifacts and does not establish that the four generators executed in a clean,
isolated build.

The older `CLEAN_ROOM_VERIFICATION_REPORT.md` names
`02_EXP85_REBUILT/rebuilt_artifacts/clean_room_build/RECONSTRUCTION_PROVENANCE_MANIFEST.json`.
That path is absent from the current tree and Git history. Git tracks only the
canonical `rebuilt_artifacts/RECONSTRUCTION_PROVENANCE_MANIFEST.json`.
Consequently, the older clean-room execution claims lack a verifiable manifest
and subprocess telemetry at the claimed path.

Although the canonical manifest contains cache-access fields, its recorded
`execution_telemetry` is the string "Verified existing artifacts via deep
uniform structural and numerical validation." Those access fields are not
process-level observations; the manifest records no OS file-access tracing.

Additional source mismatch: the current Anchor74 generator SHA-256 is
`db3fa4148a681d6e0c7029bf908dca12dab8b0708329937daf59a94be1eed022` and
constructs 143 features. Historical reports describe a different 669/550
feature pipeline. The presence of raw CSV reads at the generator entry point
does not prove historical feature/model parity. The current manifest also
contains XGB/V55 source hashes that correspond to earlier source revisions,
not current source.

## Isolated smoke-test status

`clean_room_smoke_runner.py` runs generators with `--fast` in copied staging
directories. Its contract hard-codes
`SMOKE_ONLY_NOT_CERTIFIED` and `raw_rebuild_certified: false`; success can
validate wiring and output shape only. It cannot certify full raw
reconstruction.

The completed isolated run at
`runs/clean_room_smoke_retry_20261031_230700/execution.json` recorded
`SMOKE_COMPLETE_NOT_CERTIFIED`, with `raw_rebuild_certified: false`. All four
fast-mode subprocesses exited 0 in a total of 585.749 seconds:

| Generator | Seconds | Exit | Output evidence |
|---|---:|---:|---|
| XGB10 | 17.359 | 0 | OOF/test arrays finite; expected row counts |
| V55 | 182.140 | 0 | 14 expected member keys; arrays finite |
| V17 | 63.906 | 0 | 13 expected bundle keys; arrays finite |
| Anchor74 | 322.344 | 0 | 61,500 rows; finite targets; IDs exactly match raw test order |

Input SHA-256: `train.csv` =
`b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`,
`test.csv` =
`dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`.
The generated Anchor74 CSV SHA-256 is
`0add9f20003460224e837f4173f02937611c153b77078ebb177bc90ff5890f20`.
All four staged generator source hashes are recorded in the execution JSON.

Device evidence in the logs reports CUDA/GPU for XGB10 and Anchor74 GPU model
stages, and NVIDIA CUDA GPU for CatBoost in V55/V17. XGBoost also emitted a
device-mismatch warning and said prediction fell back to `DMatrix`; the log
does not support claiming that every prediction operation used the GPU. This
smoke-only run is not historical parity evidence.

The first attempt under `runs/clean_room_smoke/` is explicitly marked
`INCOMPLETE_NOT_CERTIFIED`; it has only three successful steps. The completed
retry is the smoke evidence summarized above. Both run records are preserved.

## Required distinction

- `ARTIFACTS_VERIFIED`: checks files already present.
- Smoke complete: fast-mode subprocesses complete; not historical parity and
  not a full rebuild.
- `RAW_REBUILD_CERTIFIED`: requires all four successful, non-fast generator
  executions in an empty output directory, current source and raw-input hashes,
  and validation of every expected output.

No report, smoke test, hybrid replay, or candidate score overrides those
requirements. Current status remains **not raw-rebuild certified**.
