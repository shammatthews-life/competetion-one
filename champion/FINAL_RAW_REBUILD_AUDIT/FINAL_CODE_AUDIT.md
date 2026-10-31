# Final EXP85 Code and Input Audit

## Scope and evidence boundary

The full run recorded in `runs/raw_full_gpu_20261031_235900/execution.json`
invokes the four `generate_*_from_scratch.py` programs directly. The run uses
an empty component-output directory, explicit `D:\gci\inpuy\train.csv` and
`test.csv` inputs, full mode (no `--fast`), and a GPU-only preflight. It does
not invoke a submission writer with fixed legacy output paths.

The audit distinguishes a fresh raw-input run from historical model parity.
Fresh training from the available source is not proof that its feature schema,
fold assignments, model settings, framework versions, or predictions match
the historical producers.

## Script-by-script audit

| Script | Inputs and transitive reads | Model and validation configuration | OOF/test inputs and outputs | Classification |
|---|---|---|---|---|
| `02_EXP85_REBUILT/source/generate_xgb10_from_scratch.py` | Explicit raw `train.csv` and `test.csv`; the feature builder concatenates raw train/test rows for the stated 239-feature manifold. No model checkpoint or prediction-cache reads appear in the source. It imports no local helper module. | Hist XGBoost, 10-fold `StratifiedKFold(shuffle=True, random_state=42)`, 450 requested estimators, fold model seed `42 + fold`, early stopping 30. GPU probe and selected `device="cuda"` are required by the wrapper. | Produces OOF predictions, test predictions, and labels as `.npy` files in the isolated component directory. | Raw-input model training, not a copy or repackage. Historical 550-feature inputs are a different model lineage. |
| `02_EXP85_REBUILT/source/generate_v55_from_scratch.py` | Explicit raw `train.csv` and `test.csv`; no historical `.npy`, `.npz`, checkpoint, or local helper reads found. | Seven-model suite; 5-fold stratified CV, shuffle and seed 42. The full-mode tree/boosting iteration settings are defined in the source. CatBoost performs a GPU probe and selects GPU when available; the run wrapper rejects a reported CPU fallback. | Produces `v55_members.npz`, containing each member's OOF and test predictions. | Raw-input model training. The bundle is newly generated, not a historical cache copied into place. |
| `02_EXP85_REBUILT/source/generate_v17_from_scratch.py` | Explicit raw `train.csv` and `test.csv`; no historical prediction or checkpoint reads and no local helper imports found. | Five streams; 5-fold stratified CV, shuffle and seed 42. The full-mode LightGBM/CatBoost settings and CatBoost seed schedule are in source. The run wrapper rejects a reported CPU CatBoost fallback. | Produces `v_17_full_suite_bundle.npz` with OOF/test arrays and its derived final stream. | Raw-input model training. It is not the historical V17 cache's provenance. |
| `02_EXP85_REBUILT/source/generate_anchor74_from_scratch.py` | Explicit raw `train.csv` and `test.csv`; combines them during feature construction. No historical prediction/cache reads or local helper imports found. | Current Hasan feature builder; 5-fold stratified model zoo with fixed seeds, LightGBM, ExtraTrees, XGBoost and CatBoost components; the source also defines a 5-fold KFold feature transformation. CUDA/GPU selection for XGBoost and CatBoost is checked in the captured run log. | Produces the Anchor74-style `SK_ID_CURR,TARGET` CSV. It does not persist the train/test feature matrices. | Fresh raw-input training, but this Git snapshot constructs 143 features, not Phase 3's 669 or selected 550. A separate local production-lineage report describes a 695-feature builder; that is a different source snapshot and is not the generator executed here. |
| `02_EXP85_REBUILT/source/run_all_recreations.py` | Reads raw train labels and test IDs for output validation; launches the four generators, then reads their generated arrays/CSV to verify shapes, ranges, and hashes. No model checkpoints are trained by the orchestrator itself. | Selects full, fast-smoke, or verify-only modes. A verify-only result checks existing files; it is not evidence that the files were trained during that invocation. | Creates a provenance manifest after generator execution or artifact verification. | Orchestration and validation, not a model producer. This audit bypassed it to maintain direct, explicit commands and independent run logging. |
| `champion/create_exp85_submissions.py` | Reads the raw test ID list, historical Anchor74 and historical suite artifacts for Track 2, rebuilt Anchor74 for Track 1A, and a provenance manifest plus regenerated V17/V55/XGB files for Track 1B. Reads the frozen benchmark for comparison. | No model training or fold assignment; performs rank/probit synthesis and threshold checks. | Writes three track submissions to fixed paths under `D:\gci`; Track 1B consumes manifest-certified component files. | Synthesis/repackaging only. Not invoked because it writes outside the isolated run and its thresholds are not a raw-rebuild certificate. |
| `champion/create_submission_recovery_v5_rebuilt.py` | `hybrid` mode uses rebuilt Anchor74 plus historical V17/V55/XGB arrays; `full_scratch` mode reads the corresponding arrays from the rebuild directory. It also reads raw test IDs and a historical benchmark. | No model training; fixed 8-member rank/probit blend followed by ordinal rank projection. | Writes Recovery V5 hybrid and/or full-scratch candidate to fixed legacy output paths. | Synthesis only. Its hybrid output remains historical-cache dependent; neither mode was invoked in this audit. |

## Exact candidate formula

The isolated wrapper reproduces the source-level 8-member recipe without
calling the legacy writer:

1. Rank each member independently using average ranks and map to
   `(rank - 0.5) / n`.
2. Apply clipped probit transforms.
3. Normalize the fixed weights:
   `v17_e1Aet=.2932`, `xgb10=.2596`, `v17_final=.2051`,
   `v55_cbprod=.0874`, `v17_e1et=.0489`, `v55_deep_et=.0442`,
   `v55_allfam=.0383`, `v55_v2_mono=.0233`.
4. Blend `0.85 * probit(Anchor74 rank) + 0.15 * suite_probit`.
5. Apply ordinal rank projection to make 61,500 unique output targets.

The validator contains a separate formula implementation and compares the
written ten-decimal CSV with recomputed values after exact raw test-ID
alignment. Unit tests cover the full 61,500-row path, member mapping, ID
reordering, duplicate IDs, non-finite targets, and metric calculation.

## External packages and helper-file trace

- The four generators are standalone scripts; their imports are standard
  library plus NumPy, pandas, SciPy (where used), scikit-learn, LightGBM,
  XGBoost, and/or CatBoost. No generator imports a local helper or reads a
  checkpoint/prediction array. Their model-data `read_csv` calls load the
  explicit raw train/test CSV paths.
- `run_all_recreations.py` imports local
  `02_EXP85_REBUILT/source/provenance_contract.py`, then launches the generators
  with `subprocess`. The helper classifies execution mode and validates manifest
  fields; it does not train models or load prediction arrays itself. The
  orchestrator reads the generated arrays only for verification and produces
  its JSON manifest.
- `create_exp85_submissions.py` and
  `create_submission_recovery_v5_rebuilt.py` add the generator source directory
  to `sys.path` and import `provenance_contract.py`. They separately read
  historical caches and/or rebuilt arrays, then write fixed-path CSVs under
  `D:\gci`; they were not run here.
- `provenance_contract.py` checks provenance status, command records, source
  hash string shape, raw-input hash string shape, and artifact names. It does
  not independently recompute every manifest hash against the current files.
  This audit instead hashes raw inputs, generator source, logs, outputs, and
  reference files directly.
- The legacy v85 builder in `gemini/build_exp85_optimal_macro_apex.py` reads
  cached suite predictions plus Anchor74; Champion79 is diagnostic/reference
  material, not part of the final 8-member blend calculation. It writes to a
  fixed path and was not executed.

## Historical artifact branches

- **Recovery V5** is a hybrid replay and requires historical cached V17, V55,
  and XGB predictions. The reported exact historical match remains evidence
  for replay, not independent upstream model regeneration.
- **V20** at `D:\gci\gemini\v_20_apex_grand_champion_submission.csv` has SHA-256
  `41c39e302e9bc25c520157e9d1fb22cc61c3cafcc3754a6629fcd1e29321f6c0`, 61,500
  unique raw-test-order IDs, and finite `[0,1]` targets. No recovered original
  writer or training run exists in the prior search evidence; its model
  lineage remains unknown.
- **V080041** at `D:\gci\gemini\v_solo_080041_pure_champion.csv` has SHA-256
  `8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b` and is
  byte-identical to the available `submission_final.csv` export. The export
  writer is known, but its upstream stacker depends on 18 named artifacts.
  Follow-up inventory found `meta.json` locally and labels/IDs derivable from
  raw CSVs; 15 historical prediction/feature artifacts remain unavailable in
  the searched roots. These V080041 dependencies are not direct inputs to the
  separate v85 8-member formula.

## Known parity gaps and limits

- Current Anchor74 has 143 features; historical Phase 3 reports 669 and a
  selected model input of 550. The current generator does not persist its
  feature matrix, so this run cannot establish cell-level feature parity.
- The workstation also contains a separate 695-feature production lineage
  report under `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION`. It reports a
  first shared-feature divergence at `SOLVENCY`; that source/report is not the
  48b66288 Git snapshot audited and executed in this folder. The 143-feature
  versus 695-feature distinction must not be collapsed into one pipeline.
- The raw generators define current folds/seeds/settings in code, but the
  original historical producer settings and runtime provenance are not
  complete enough to assert identical training.
- Historical CatBoost discrepancies are not resolved by a successful
  GPU-only preflight or a new candidate run.
- No historical model checkpoint is available as an inference-only parity
  control in this run.
- No leaderboard submission was made; no official score is inferred.

## Full-run result

The full-mode raw candidate run completed in `2,894.797` seconds. All four
generators exited successfully and wrote only to
`runs/raw_full_gpu_20261031_235900/components`. It used raw inputs with
SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
(`train.csv`) and
`dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`
(`test.csv`). There was no `--fast` flag and no historical cache was used to
generate candidate predictions.

The output is a valid, independently recomputed raw-component candidate, not a
reproduction of the frozen champion: its ID-aligned Spearman is `0.891570`,
and the fresh Anchor74 Spearman against the historical Anchor74 is `0.869363`.
The 143-feature current Anchor74 builder is not cellwise comparable to the
historical 669/550-feature matrices because this generator does not persist its
matrix. The separate 695-feature lineage report is from another source
snapshot and remains separate evidence.

The missing V20 producer, 15 V080041 prediction/feature artifacts, unknown
historical fold assignments/model settings, unresolved CatBoost discrepancy,
and lack of an official evaluation leave the original historical lineage
incomplete. The raw candidate run does not resolve those gaps.
