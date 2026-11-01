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

## Remaining blockers

1. The original `submission_final.csv` producer references neural/ensemble
   artifacts that are absent from the workspace and every local Git branch:
   `group_a_oof_10stream.npy`, `group_a_test_10stream.npy`,
   `oof_and_test_tabm32_463_dae.npz`, and `oof_and_test_lgbm_dae491.npz`.
   Workspace-wide path searches and local all-branch Git history searches
   found no copies. Their exact weights/configuration and complete producer
   lineage remain unavailable, so the 080041 anchor cannot be independently
   rebuilt with the available evidence.
2. Raw-data V17, V55, and XGB10 generators have already been executed in the
   separate full-component audit and their outputs were consumed by its raw
   candidate. They are non-exact against historical arrays, and the raw
   Anchor74 surrogate also differs materially. Recovering the historical
   producer inputs/configuration and fold lineage—not repeating those raw
   runs—is required for historical parity.
3. CatBoost GPU is not bitwise repeatable under the preserved fixed-seed
   source on this RTX 4050. Historical CatBoost OOF/test arrays are not exact
   in the independent run; the second run is close but remains non-exact.
4. The `0.801340` result is not newly established by an official leaderboard
   evaluation. No generated hybrid CSV is claimed to have that score.

## Next actions

1. Recover the missing neural/ensemble inputs or their original model
   checkpoints, source/configuration, and fold manifests from a source outside
   the local workspace/history. Without them, the exact 080041 producer
   cannot be reconstructed; do not replace it with a surrogate and claim
   historical parity.
2. If the exact upstream evidence is recovered, rebuild 080041 and the
   subsequent V080150 → V080200 → Level-2 → asymmetric/transductive → Anchor74
   chain from raw inputs, then regenerate the original final synthesis and
   verify it against the frozen reference.
3. Keep both generated candidates isolated; do not promote either as a final
   submission or mix either with Phase 2.
