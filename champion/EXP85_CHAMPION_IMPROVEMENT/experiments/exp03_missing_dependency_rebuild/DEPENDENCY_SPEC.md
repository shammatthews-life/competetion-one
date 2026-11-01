# EXP03 dependency specification

## What is actually missing?

Nothing read directly by `build_exp85_optimal_macro_apex.py` is absent. Its
seven direct inputs are the precision Anchor74 CSV, Grand Master 79 CSV, V17
and V55 NPZ bundles, XGB10 OOF/test arrays, and the training labels; all are
present locally. The historical precision and optimal-apex scripts also read
`cache/hasan_expanded_zoo.npz`, which exists at
`D:\gci\gemini\cache\hasan_expanded_zoo.npz` (SHA-256
`73951fa46ea73403460d62c0427d4e3b696d935230ac74154600902e60c84df4`).

That zoo cache is therefore **available but provenance-unverified**, not
missing. It is the dependency this experiment reconstructs independently.
Two source paths can write the same filename; the archive itself does not
identify which producer ran. EXP03 uses the historical selected-550-feature
producer chain whose source snapshots are preserved and hash-pinned here.

## Producer-consumer table

| Dependency | Original producer / evidence | Present? | Raw-data regeneration | EXP03 status |
|---|---|---:|---|---|
| `hasan_features_tr.feather`, `hasan_features_te.feather` (669 columns) | `baseline/source/build_hasan_600_features.py`; exact snapshot matches `D:\gci\gemini\build_hasan_600_features.py` | Yes | Yes, raw `train.csv` and `test.csv` | Rebuilt in isolated runtime; compare feature order, dtypes, and values |
| `y_train.npy` | Raw `TARGET` extraction; also has a `feature_builder.py` candidate writer | Yes | Yes, directly from raw `train.csv` | Regenerated in runtime; int64 to match historical trainer input |
| `hasan_selected_550_tr.feather`, `hasan_selected_550_te.feather` | `baseline/source/screen_and_select_features.py`; five-fold LightGBM gain ranking | Yes | Yes, from rebuilt features and raw labels | Rebuilt from scratch; selection is label-dependent as in the historical code |
| `hasan_zoo_predictions.npz` | `baseline/source/train_hasan_master_zoo.py` | Yes | Yes, from selected features and raw labels | Rebuilt from scratch; isolated cache is checked empty before training |
| `hasan_expanded_zoo.npz` (11 keys: five model OOF/test pairs and `y_true`) | `baseline/source/train_catboost_and_deep_lgb.py`; another self-contained candidate writer also exists | Yes | Yes, using original selected-feature chain and GPU-supported models | Independently rebuilt; feature matrices exact and 8/10 model arrays exact; CatBoost OOF/test are close but not exact |
| `v_solo_080041_pure_champion.csv` | `solo_grand_master_apex_080.py`, from `submission_final.csv` | Yes | Not independently from raw data; source champion is a prediction artifact | Historical input/reference |
| `v_gold_grand_master_apex.csv`, `v_20_apex_grand_champion_submission.csv` | Historical candidate CSVs consumed by the 080150 producer | Yes | Their complete raw-only provenance is not established in this task | Historical inputs to the isolated downstream replay |
| `v_solo_080150_micro_apex_fusion.csv` | `solo_grand_master_apex_080.py` | Yes | Partially; it also needs the three historical prediction inputs above | Re-generated in isolation; classified hybrid |
| `v_solo_080200_grand_master_apex.csv` | `build_v80200_apex.py` | Yes | Yes, once 080150 is supplied | Re-generated using hybrid 080150 input |
| `v_solo_080300_level2_regime_stacker.csv` | `build_level2_regime_meta_stacker.py` | Yes | Yes, once 080200 and pure 080041 are supplied | Re-generated using the new zoo and historical anchors |
| Asymmetric / optimal-transductive / precision Anchor74 CSVs | Their correspondingly named historical scripts | Yes | Yes, from prior anchor plus zoo and raw tables | Re-generated sequentially in isolation |
| V17/V55/XGB10/Grand Master 79 inputs for final EXP85 synthesis | Historical caches and CSV; exact source lineage is outside this zoo dependency | Yes | Raw-only V17/V55/XGB10 generators have executed in `FINAL_RAW_REBUILD_AUDIT`; their outputs were consumed by a separate raw-component candidate, not this historical-anchor replay | Hybrid inputs reused here; raw candidate comparison is non-exact and does not establish historical parity |
| `submission_final.csv` producer's remote neural/synthesis arrays | `final_submission/reproduce_final_submission.py`; audit manifest lists missing remote inputs | Some upstream arrays absent | No exact raw-only regeneration demonstrated; model artifacts / remote outputs unavailable | Genuine upstream provenance blocker; no replacement model is claimed |

## Dependency graph

```text
train.csv + test.csv
  -> 669 features
  -> 5-fold feature importance -> selected 550 columns
  -> LightGBM Conservative + LightGBM ExtraTrees + GPU XGBoost
  -> CatBoost GPU + Deep LightGBM
  -> hasan_expanded_zoo.npz
       -> Level-2 regime meta-stacker (also needs historical 080200 and 080041)
       -> asymmetric tail gate
       -> optimal transductive apex
       -> precision transductive apex / Anchor74
       -> final 85:15 EXP85 synthesis (also needs V17, V55, XGB10 and 079)

submission_final.csv + v_gold + v20
  -> historical 080041 / 080150 starting anchors
  -> 080200
  -> Level-2 stacker
```

## Fidelity boundary

The zoo generator preserves the available historical code and its seed,
five-fold split, early-stopping, and feature-selection behavior. The historical
feature builder creates target-informed UIDY features by its own five-fold
scheme; this is not nested inside the subsequent zoo folds. Historical parity
therefore does not imply leakage-safe OOF validation.

The completed raw-data rebuild produced SHA-256
`0a619389f223a4cd93779f61b9d4bd7b600335d2fcf986afbcbf4eacf181e879`.
The 669-feature and selected-550 matrices matched all historical values and
schemas exactly, including their 34 train / 16 test infinities. Eight model
arrays were byte-exact. CatBoost OOF parity was Pearson `0.9992049`,
Spearman `0.9996243`, MAE `0.0006702`, maximum difference `0.0984592`;
test parity was Pearson `0.9998449`, Spearman `0.9999277`, MAE `0.0007278`,
maximum difference `0.0294115`. These are approximation results, not exact
CatBoost parity.

A second fixed-seed CatBoost GPU run was not bitwise identical to the first
(fold 2 OOF AUC changed by `0.000344`), confirming run-to-run variance. The
second run was very close but still not exact against the historical arrays.
The downstream chain passed all seven source stages when tested with this
second raw output; its final CSV remains classified as a hybrid replay because
the other historical anchors and suite bundles were reused.

The 080150 chain begins from `submission_final.csv`, `v_gold`, and `v20`.
`submission_final.csv` is available and hash-verified, but its original
reproduction script references remote neural/ensemble arrays absent from the
machine. Consequently the downstream replay can validate integration and
numerical parity from historical starting anchors, but cannot certify a
complete independent raw-data reconstruction of the entire EXP85 lineage.
