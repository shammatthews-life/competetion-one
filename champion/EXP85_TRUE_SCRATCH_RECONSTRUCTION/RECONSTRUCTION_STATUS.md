# EXP85 Historical Reconstruction Status

## Overall

**Full independent raw reconstruction: BLOCKED by unrecovered historical
upstream model inputs.** Phase 0 verifier passed after correcting the feature
evidence records and one dependency-graph edge. Its final run reported no
verification errors; all five focused tests and JSON validation passed.

The frozen champion and separate Phase 2 release remain untouched. No local
candidate is claimed to have the official `0.801340` score; no official
leaderboard evaluation record is available in this evidence set.

## Stage status

| Stage | Result | Evidence / limitation |
|---|---|---|
| Raw data and row alignment | PASS | Train/test hashes are pinned; 171,202 train and 61,500 test rows; saved labels match raw target order. |
| 669 raw features | PASS | Historical source `build_hasan_600_features.py` was hash-pinned and its feature statements were replayed on raw CSVs. Train/test schemas, order, dtypes, missing masks, and all feature cells match canonical matrices exactly. |
| Historical 550-feature selection | PASS | Five-fold LightGBM selector replay ran on regenerated features; all 550 names and their order match the historical selector output. |
| Feature-file hash discrepancy | RESOLVED | The canonical files and independent rebuild copies have their expected SHA-256 values. A separate replay produced different Feather byte hashes but exact matrix values; hashes and paths are separately recorded, never silently substituted. |
| Five-family raw zoo | PASS as a raw-trained candidate; PARTIAL historical parity | Five-fold training ran for 1,283.656 seconds. LightGBM conservative, LightGBM ExtraTrees, XGBoost, and deep LightGBM OOF/test predictions match the historical cache exactly. CatBoost does not: OOF Pearson `0.9999078`, Spearman `0.9999501`, MAE `0.00021784`; test Pearson `0.9999803`, Spearman `0.9999903`, MAE `0.00024531`. Historical CatBoost runtime/device/checkpoint metadata is unavailable. |
| V17 raw candidate | EXECUTED; historical parity unresolved | The isolated full raw run generated all five V17 streams and the ensemble from raw inputs. Historical `resid16.pkl` producer and exact schema remain unresolved; the historical cache has no fold/ID sidecars. |
| V55 raw candidate | EXECUTED; historical parity unresolved | The isolated full raw run generated seven V55 members from raw inputs. Historical member/cache lineage is not proven identical. |
| XGB10 raw candidate | EXECUTED; historical parity unresolved | The isolated full raw run generated 10-fold XGBoost OOF/test arrays from raw inputs. Historical producer/fold sidecars are unavailable. |
| Level-2 -> asymmetric -> optimal bedrock | HYBRID candidate exists; NOT an independent historical rebuild | A prior run trained on the raw zoo but consumed historical V080200/V080041 anchors. It is explicitly classified as hybrid and is not eligible as the final independent submission. |
| Precision apex / Anchor74 | Historical formula replay PASS; independent raw rebuild BLOCKED | The formula replay matches historical rank ordering when supplied the historical zoo and optimal-apex reference. It does not regenerate the missing upstream baseline/model predictions. |
| EXP85 final synthesis | Historical-cache replay PASS; independent raw parity NOT achieved | The fixed 85:15 formula is known. A prior full raw-component candidate had ID-aligned Spearman `0.891570` against the frozen champion and is not a historical-parity result. |

## Executed run evidence

The runs below predate this status update and were inspected rather than
repeated. Full commands, configurations, source/input hashes, fold details,
versions, device evidence, and output hashes remain in their original run
manifests/logs.

| Run | Executed work | Result |
|---|---|---|
| `feature_parity_20261031_200506` | AST-isolated historical 669-feature statements on raw CSVs; feature-by-feature comparison | Exit 0; 27.796 s; exact four-matrix value parity |
| `selector_replay_20261031_202500` | Five-fold LightGBM feature-selection replay | Exit 0; 238.485 s; 550/550 selected names and order match |
| `anchor74_historical_zoo_20261031_215500_retry1` | Five-family five-fold zoo training using regenerated selected features | Exit 0; 1,283.656 s; manifest records raw inputs, folds, parameters, versions, GPU preflight, telemetry, and predictions |
| `raw_full_gpu_20261031_235900` | Raw XGB10, V55, V17, Anchor74-style candidate, blend, and validation | Exit 0; 2,894.797 s; candidate SHA-256 `2a5d2dfc2504f86a6d6c6d42bfc63fff7f62a10ea66135b1fee44982f9a348fe`; Spearman `0.891570` vs frozen champion |
| `bedrock_chain_rebuild_20261101` | Level-2/asymmetric/optimal stages with raw-trained zoo and recovered historical base anchors | Stage commands exited 0; output is hybrid by construction; optimal-apex candidate SHA-256 `4c5b02ac972297822e33e39e44e5631c1cebad80bd014a02955e70ff0875df34` |
| `bedrock_chain_historical_zoo_replay_20261101` | Historical Level-2/asymmetric/optimal formula replay using cached zoo and recovered anchors | Stage outputs match their historical files exactly; this is a historical-cache replay, not an independent rebuild |
| `historical_anchor74_full_chain_replay_20261101_retry2` | Historical Anchor74 formula replay and ID-aligned comparison | Exit 0; 61,500 IDs verified; Pearson 1.0, Spearman `0.9999999999999998`, MAE `2.50e-11`; historical zoo cache used |
| `phase0_audit_20261101` | Phase 0 evidence verifier after graph correction | Exit 0; verdict `PHASE_0_CHECKS_PASS_FULL_RAW_RECONSTRUCTION_BLOCKED`; 43 nodes, 56 edges, 39 artifacts, 37 source files, 63 array members; no verification errors; five focused tests passed |

## Current hard blockers and recovery attempts

### 1. V20 producer is unrecovered

- Available prediction: `D:\gci\gemini\v_20_apex_grand_champion_submission.csv`,
  SHA-256 `41c39e302e9bc25c520157e9d1fb22cc61c3cafcc3754a6629fcd1e29321f6c0`.
- The original writer, training configuration, and model ancestry were not
  found in the local source/notebook search or non-shallow `git log --all`
  history. The file is consumed by the V080150 builder.
- Replaying later transforms from the V20 CSV would remain a historical-cache
  reconstruction, not an independent raw rebuild.

### 2. V080041 learned inputs are missing

- The saved V080041 submission is only an export of
  `submission_final.csv`; the export script does not train its source models.
- The supplied `final_submission.rar` contains the README, reproduction
  script, and saved submission, but not the missing learned arrays/features.
- The reproduction inventory has 18 named inputs: 15 learned prediction or
  feature artifacts remain unrecovered; labels and IDs are recoverable from
  raw CSVs, and `meta.json` was found elsewhere.
- These inputs prevent raw reconstruction of V080041, which contributes the
  95% baseline to V080150. V080150 feeds V080200 and the later bedrock chain.

### 3. VGold and downstream historical cache provenance

`train_v_gold_grand_master_apex.py` is available, but its source consumes
historical V17/V55/XGBoost predictions and earlier candidate submissions.
The saved VGold CSV therefore does not certify its full raw training lineage.

### 4. CatBoost parity metadata

The raw-trained CatBoost candidate is close but not exact. The historical
cache has no per-fold checkpoints, package lock, driver/runtime record, or
GPU telemetry with which to reproduce its numerical environment exactly.

### Recovery routes already attempted

- Raw inputs, source trees, notebooks, artifacts, and the full non-shallow
  local Git history were searched; the Gemini tree census counted 38,195 files
  (including environment files).
- The supplied final-submission archive was inventoried; it contains the
  saved final CSV and replay code, not the 15 missing learned artifacts.
- Historical consumer and producer scripts, V17/V55/XGB10 recreation scripts,
  feature builders, selector, zoo trainers, apex builders, run manifests, and
  prior reconstruction reports were inspected.
- Exact V080150 -> V080200 and subsequent formula replays were executed in
  separate run directories. They validate arithmetic only when historical
  cached anchors/predictions are supplied.
- Local Git searches for V20 and original artifact-writer strings found audit
  and consumer references, not the missing original producer.

## What is needed to continue the independent historical rebuild

Recover the original V20 producer/model configuration and the 15 missing
V080041 prediction/feature inputs together with their producers/configurations.
For exact CatBoost prediction parity, recover its historical runtime/device
metadata or fitted fold models. Without these dependencies, there is no
permitted independent route to the original V080150/V080200 bedrock values;
substituting saved predictions would make the result hybrid. Official
leaderboard evaluation is separately required before claiming score `0.801340`.

## Result categories

- **Raw feature and zoo components:** independently regenerated from raw data;
  CatBoost prediction parity remains incomplete.
- **V17/V55/XGB10 raw candidates:** independently generated, but not certified
  as exact historical cache producers.
- **Bedrock/Anchor74 formula replay:** historical-cache reconstruction.
- **Prior V5 result:** hybrid historical-cache reconstruction.
- **Full independent historical EXP85 reconstruction and official score:** not
  achieved.
