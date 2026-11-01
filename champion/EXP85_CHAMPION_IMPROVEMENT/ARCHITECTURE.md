# Historical EXP85 architecture and provenance

## Authoritative executable synthesis

The source identified by the historical dependency manifest as the EXP85
generator is `D:\gci\gemini\build_exp85_optimal_macro_apex.py`. Its
SHA-256 is recorded in `baseline/BASELINE_MANIFEST.json`. The historical
`D:\gci\gemini` tree has no Git metadata, so an original producer commit
cannot be asserted; the snapshot records the Git commit of this reconstruction
workspace separately.

This source executes the final ensemble synthesis, not every upstream model
trainer. Its direct inputs are:

| Input | Role | Provenance classification |
|---|---|---|
| `v_20261027_precision_transductive_apex.csv` | Anchor74 test predictions; receives 85% ensemble weight | Historical prediction cache |
| `v_17_full_suite_bundle.npz` | V17 OOF/test streams | Historical prediction cache |
| `v55_members.npz` | V55 OOF/test streams | Historical prediction cache |
| `oof_10f_xgb.npy`, `test_10f_xgb.npy` | XGB10 OOF/test stream | Historical prediction cache |
| `y_train.npy` | Training labels used for suite OOF telemetry | Derived label cache; independently checked against raw `train.csv` |
| `v_20261027_grand_master_macro_stacker_apex.csv` | EXP79 comparison/telemetry only; not an input to the final prediction formula | Historical prediction |
| `train.csv`, `test.csv`, `sample_submission.csv` | Source data and row-order references | Raw competition data, hashed but not copied into this workspace |

## Transformations

For each suite member, the script maps predictions to average ranks
`(rank - 0.5) / N`, applies a normal quantile transform with clipping at
`1e-6`, and combines eight streams with normalized fixed weights:

| Stream | Weight before normalization |
|---|---:|
| V17 `e1Aet` | 0.2932 |
| XGB10 | 0.2596 |
| V17 final | 0.2051 |
| V55 CatBoost product | 0.0874 |
| V17 `e1et` | 0.0489 |
| V55 deep ExtraTrees | 0.0442 |
| V55 all-family one-hot | 0.0383 |
| V55 monotone | 0.0233 |

The script transforms Anchor74 to probit space, computes
`0.85 * Anchor74_probit + 0.15 * suite_probit`, converts the result to strict
ordinal ranks, and writes `(rank - 0.5) / 61,500` with ten decimal places.
The isolated replay retains these transformations and weights; it only
redirects reads to the frozen snapshot and redirects writes into a unique
workspace run directory.

## Upstream ancestry and scope

Anchor74 is the precision transductive apex. Its formula consumes the
historical optimal transductive apex and historical zoo predictions; earlier
Level-2 and bedrock construction also consumed saved V080200/V080041 anchors.
Those source scripts and their known upstream chain are fingerprinted in the
baseline manifest where available. The final EXP85 script does **not** load
V20 or V080041 directly; their missing original producer ancestry matters to
independently rebuilding Anchor74, not to executing the final formula from
the preserved Anchor74 CSV.

The standalone “exact reproduction” notebook embeds compressed historical
champion ranks. It is an output replay, not an independent training pipeline,
and is not treated as an eligible raw-data baseline.

## Execution class and score evidence

The executable final synthesis is classified as **historical-cache-assisted
execution**. A successful local byte-for-byte replay validates the final
formula and input alignment, not an independent rebuild of the historical
models or Anchor74.

The source and prior audit describe the EXP85 submission as scoring `0.801340`.
The original source project contains no Git commit metadata, and the reviewed
competition log does not provide an independently verified upload receipt
binding that score to the frozen CSV hash. The user's target of `0.80314` is
therefore a separate, unverified public-leaderboard objective.
