# EXP85 reconstruction status

## Result

The previously unverified `cache/hasan_expanded_zoo.npz` dependency was
reconstructed from raw train/test data using the hash-pinned historical
feature, selection, and model producer scripts. It was present in the
historical cache; no direct EXP85 input file was missing.

The independent zoo rebuild is valid and schema-aligned, but not a fully
bit-exact reproduction: all feature matrices and eight of ten prediction
arrays match the historical references exactly. CatBoost GPU arrays are
non-exact and fixed-seed reruns are not bitwise reproducible. Full evidence is
in [the EXP03 dependency specification](experiments/exp03_missing_dependency_rebuild/DEPENDENCY_SPEC.md).

The original downstream chain was executed without changing its synthesis
formula or rank-shift assertion. One raw run failed the original guard at
149.4283 mean rank displacement; a separately recorded fixed-seed repeat
passed at 185.35. The successful 61,500-row output is
`HYBRID_HISTORICAL_CACHE_RECONSTRUCTION`, because historical 080041/080150
anchors and V17/V55/XGB10/Grand Master 79 artifacts remain inputs. It is not
an independent submission and has no official leaderboard score. The final
CSV hash and all ID-aligned parity metrics are in the isolated
[integration report](experiments/exp03_missing_dependency_rebuild/runs/exp03_20261102_003845/integration_repeat/integration_report.json).

Separately, an earlier full raw-component run regenerated XGB10, V55, V17,
and a raw-data Anchor74 surrogate, then wrote a raw-component candidate. That
candidate is explicitly `RAW_COMPONENT_CANDIDATE; HISTORICAL_PARITY_NOT_ESTABLISHED`,
not the historical Anchor74 architecture: against the frozen champion its
Spearman was `0.8915697`, MAE `0.0999145`, and RMSE `0.1344311`. Its hash is
`2a5d2dfc2504f86a6d6c6d42bfc63fff7f62a10ea66135b1fee44982f9a348fe`.
This confirms the V17/V55/XGB10 raw generators have executed; repeating those
trainings would not resolve their substantial historical prediction
differences. See the [raw run manifest](../FINAL_RAW_REBUILD_AUDIT/runs/raw_full_gpu_20261031_235900/execution.json)
and [historical comparison report](../FINAL_RAW_REBUILD_AUDIT/runs/raw_full_gpu_20261031_235900/reports/historical_output_comparison.json).

## Verified evidence

- Raw inputs: exact configured SHA-256 hashes, 171,202 train rows and 61,500
  test rows; labels and IDs are unique/aligned.
- Feature schema: 669 columns; selected schema: 550 columns; train/test
  feature matrices match the historical references exactly, including
  historical NaN and infinity locations.
- Expanded zoo: 11 keys, expected shapes/dtypes, finite `[0,1]` predictions,
  raw `y_true` alignment; independent output SHA-256
  `0a619389f223a4cd93779f61b9d4bd7b600335d2fcf986afbcbf4eacf181e879`.
- GPU: XGBoost 3.4.1 on `cuda:0`, CatBoost 1.2.10 on GPU; RTX 4050 telemetry
  is preserved in the run directory.
- Downstream: all seven historical downstream scripts exited 0 in the
  successful hybrid integration run. The generated output has exact 61,500
  ID coverage, Spearman `0.9999999872`, MAE `0.00003163`, and max difference
  `0.00063415` versus the frozen champion. High correlation is not exact
  reconstruction.
- Validation: six focused EXP03 tests pass; source compilation passes.

## Full upstream reproduction dependency trace

The notebook `D:\gci\gemini\GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb`
contains a compressed 61,500-row rank tensor in its final synthesis cell.
That cell replaces the trained-stream result with the embedded rank tensor.
Its exact CSV output is historical replay evidence only; it is not evidence
that the model pipeline independently rebuilt the champion. The frozen
champion remains untouched.

The referenced
`D:\gci\gemini\final_submission\reproduce_final_submission.py` is a consumer
and synthesis script, not a producer for the upstream prediction archives.
Static tracing found 18 distinct `resolve_path` inputs:

| Dependency | Entrypoint role | Current state | Independent recovery |
|---|---|---|---|
| `group_a_oof_10stream.npy` | Required relational OOF matrix; historical log documents `(171202, 10)` | Missing | Blocked: 463-feature relational matrix, producer source, fold config, and model checkpoints are absent |
| `group_a_test_10stream.npy` | Required relational test matrix; historical log documents `(61500, 10)` | Missing | Same blocker as Group A OOF |
| `oof_and_test_tabm32_463_dae.npz` | Required TabM-32 OOF/test rank/probability arrays | Missing | Blocked: exact 463-feature/DAE producer, model configuration, fold assignments, and checkpoint are absent |
| `oof_and_test_lgbm_dae491.npz` | Required LGBM-DAE OOF/test rank/probability arrays | Missing | Blocked: exact 491-feature producer and model/fold configuration are absent |
| `matrix_226_features_buhlmann_tier2.npz` | Required `X_tr`, `X_te`, and `feature_names` | Missing | Blocked: exact feature generator and source matrix are absent |
| `oof_and_test_buhlmann_tier2_v21_models.npz` | Required four-family Tier-2 OOF/test predictions | Missing | No exact producer/checkpoint found |
| `oof_and_test_grand_synthesis_595.npz` | Required 595-feature synthesis predictions | Missing | No exact producer/checkpoint found |
| `oof_and_test_neural_grand_646.npz` | Required 646-feature neural predictions | Missing | No exact producer/checkpoint found |
| `oof_and_test_grand_synthesis_646.npz` | Required 646-feature synthesis predictions | Missing | No exact producer/checkpoint found |
| `oof_and_test_tabular_neural_net.npz` | Required neural OOF/test predictions | Missing | No exact producer/checkpoint found |
| `oof_and_test_v26_heterogeneous_zoo_9stream.npz` | Required heterogeneous-zoo predictions | Missing | No exact producer/checkpoint found |
| `y.npy` | Raw training labels | Missing | Deterministically derivable from `train.csv` `TARGET` |
| `test_ids.npy` | Raw test IDs | Missing | Deterministically derivable from `test.csv` `SK_ID_CURR` |
| `meta.json` | Group-A 10-stream order and weights | Found at `D:\gci\pre info\oct22\meta.json`, SHA-256 `cdb08890d4127492d89b4864fe1ef093d1c07e8d64747cb25d480a7a546e4100` | Candidate config is readable, but the reproduction script's default resolver does not search this path |
| `oof_and_test_tabm_32heads.npz` | Optional TabM-226 input; code falls back to ResNet when absent | Missing | Not needed for fallback execution; needed to reproduce a run that used it |
| `oof_and_test_cloud_burst_zoo_v21_8seed.npz` | Optional Tier-2 ensemble extension | Missing | Optional branch activates only when all three Cloud/Acer archives exist |
| `oof_and_test_laptop_acer_zoo_10seed.npz` | Optional Tier-2 ensemble extension | Missing | Same conditional branch |
| `oof_and_test_cloud_burst_5seed.npz` | Optional Tier-2 ensemble extension | Missing | Same conditional branch |

The historical README and recovered audit classify the four named neural
artifacts as outputs from an external GPU workspace. The available
`reproduce_final_submission.py` reads their arrays but does not fit or export
their models. The named exact-reproduction notebook trains five GBDT streams
on its own feature pipeline, but it does not contain these artifacts or their
producer logic. The `recreate_precision_anchor.py` / `generate_anchor74_from_scratch.py`
lead is a separate raw-data GBDT surrogate (same SHA-256
`db3fa4148a681d6e0c7029bf908dca12dab8b0708329937daf59a94be1eed022`); it is
not the 463/491-feature neural producer. The local `dae_pt_v1.npz` and
`gbdt_dae_pt_v1.npz` have different keys and schemas and are not substitutes.

The available `D:\gci\inpuy` directory contains only `train.csv`, `test.csv`,
`sample_submission.csv`, and an image; the auxiliary relational source tables
needed for the documented 463-feature matrix are not present. A full local
search found no copies of the 11 required model/feature archives or their
checkpoints. The prior 0.89157-correlation raw candidate is not used as a
replacement.

## Remaining blockers

1. The full `reproduce_final_submission.py` entry point cannot execute its
   independent training-to-synthesis chain without the 11 required model and
   feature archives listed above. Raw `y.npy` and `test_ids.npy` are
   straightforward to regenerate, but doing so does not unblock training.
2. Exact Group-A reconstruction additionally requires its original 463-column
   relational feature producer and auxiliary data, which are absent locally.
   The exact TabM-463/DAE and LGBM-DAE491 producer logic, configs, folds, and
   checkpoints are also absent.
3. The four optional archives affect parity if the historical synthesis used
   those conditional branches; their historical run-time presence is
   unverified.
4. The `0.801340` result has not been newly established by an official
   leaderboard evaluation. No generated raw or hybrid candidate is claimed
   to have that score.

## Next actions

1. Recover the original Group-A/TabM/DAE/Bühlmann/synthesis producer notebooks
   or scripts, compatible checkpoints/configuration/fold manifests, 463/491/
   226/595/646 feature schemas, and any required auxiliary raw tables from
   the external GPU workspace or another source. Do not infer the missing
   producers from output names or AUC summaries.
2. Once recovered, independently regenerate each required archive from raw
   data, validate its exact schema and row order, then resume the 080041 →
   Anchor74 → EXP85 pipeline and its OOF/85:15 parity gates.
3. Do not substitute cached ranks, the raw surrogate, Phase 2 artifacts, or
   historical prediction arrays as training labels. Keep all candidates
   isolated and preserve the frozen champion.
