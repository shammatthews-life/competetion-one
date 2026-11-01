# Track1B Correlation Failure — Forensic Root-Cause Report

**Status:** Root cause identified from existing artifacts. No model was trained, no original Gemini file was changed, and no existing submission or run output was overwritten.

## Executive finding

The current `0.8921289` result is not evidence that the EXP85 85/15 formula is wrong. The formula, eight member names, weights, rank normalization, probit clipping, and final ordinal projection match the earlier high-correlation implementation. The input lineage does not.

The earlier Track1 submission on disk is reproducibly **0.9999975816 Spearman** (99.9998% when rounded). Its actual inputs were:

1. The Phase 3 rebuilt Anchor74 at `D:\gci\champion\02_EXP85_REBUILT\artifacts\v_20261027_precision_transductive_apex.csv`, SHA-256 `315b0bd80e4698240a7a902648a157d84da1480dea77901cdb60892502a7b55b`.
2. The historical cached V17, V55, and XGB10 suite predictions in `D:\gci\champion\01_EXP85_RECOVERY\cache\`.

The current Track1B instead used:

1. A separately generated direct Anchor74 artifact, SHA-256 `6c3207977abc21091b64a7446f979c17ed2ce9aaabf18b3acb2065569c559bc8`.
2. Newly retrained production V17, V55, and XGB10 outputs.

Applying the unchanged historical formula to the earlier anchor and historical suite reproduces **all 61,500** values of the prior high-correlation CSV at its 10-decimal output precision. Replacing the high-correlation anchor with the current production anchor while keeping the historical suite drops Spearman to `0.9031292`; replacing only the suite while retaining the high-correlation anchor yields `0.9986817`. Using both current input families yields the observed `0.8921289`. This isolates the main regression to component lineage—especially Anchor74—not blending weights.

The high-correlation route is a **hybrid replay**, not a fully raw-data rebuild: it consumes historical suite prediction caches. Its measured correlation is valid, but it does not certify fresh training of every component.

The nearby figures refer to different artifacts: Phase 3 reports `0.9999982` for the asymmetric-tail intermediate, `0.9999979` for the optimal transductive anchor, and `0.9999628752` for its final reconstructed submission. The saved V5 rebuilt Track1 submission is `0.9999975816`. These should not be substituted for one another.

## 1. Frozen evidence and exact run provenance

The machine-readable inventory, including SHA-256 values for every listed source, artifact, raw input, and current run manifest, is [audit_evidence.json](./audit_evidence.json). The reproducible, read-only audit is [analyze_existing_artifacts.py](./analyze_existing_artifacts.py).

### Submissions and anchor alternatives

| Artifact | Exact path | SHA-256 | ID-aligned Spearman vs frozen champion |
|---|---|---|---:|
| Frozen champion | `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv` | `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947` | 1.0000000 |
| Prior high-correlation Track1 | `D:\gci\champion\08_SUBMISSIONS\submission_exp85_recovery_v5_rebuilt.csv` | `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482` | **0.9999975816** |
| Duplicate prior Track1 | `D:\gci\champion\02_EXP85_REBUILT\submissions\submission_exp85_recovery_v5_rebuilt.csv` | `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482` | **0.9999975816** |
| Current production Track1B | `runs\track1b_1793445445960\submission_track1b.csv` | `9b9f7dded04ae606b8d0ca389154304f9365bf628589e1dba4e5bba652cc6f6b` | **0.8921288944** |
| High-correlation Phase 3 / `02\artifacts` Anchor74 | `D:\gci\champion\02_EXP85_REBUILT\artifacts\v_20261027_precision_transductive_apex.csv` | `315b0bd80e4698240a7a902648a157d84da1480dea77901cdb60892502a7b55b` | 0.9977761 vs champion; 0.9999967390 vs historical Anchor74 |
| Current production Anchor74 | `runs\recovered_anchor74_1793444526421\v_20261027_precision_transductive_apex.csv` | `6c3207977abc21091b64a7446f979c17ed2ce9aaabf18b3acb2065569c559bc8` | 0.8907616 vs champion; 0.8691278 vs historical Anchor74 |
| Same-name `02\rebuilt_artifacts` Anchor74 | `D:\gci\champion\02_EXP85_REBUILT\rebuilt_artifacts\v_20261027_precision_transductive_apex.csv` | `04deaa2e3e89c13bbfbe759f36aabf34609856cd7bb5c47b65ef3c2b715134d8` | 0.8907894 vs champion; 0.8691564 vs historical Anchor74 |
| `clean_room_build` Anchor74 | `D:\gci\champion\02_EXP85_REBUILT\rebuilt_artifacts\clean_room_build\v_20261027_precision_transductive_apex.csv` | `c1f2634ca8c84d8a650a1478a0114dbfe0e919e45d949f772e4fe74079827b84` | 0.8750233 vs champion; 0.8544954 vs historical Anchor74 |

The prior high-correlation CSVs are byte-identical to each other. `08_SUBMISSIONS\SUBMISSION_RESULTS.md` lists a different hash (`e979a04a...`) for that filename; this does not match the file currently on disk. The on-disk hash above was recomputed from the actual CSV and agrees with the separate prediction comparison and recovery script's expected `91aa95c9...` hash.

### Current production runs used by the low-correlation output

All current raw input hashes match the production manifests: train `b8dd5a61...` (171,202 rows), test `dfb7aeb9...` (61,500 rows).

| Component | Run directory / manifest | Exit / elapsed | Output SHA-256 |
|---|---|---:|---|
| XGB10 | `runs\recovered_xgb10_1793441428099\execution_manifest.json` (`c52d5db0...`) | 0 / 69.28 s | `test_10f_xgb.npy`: `b05cfaf46c8718ff67446ab198ca0244da6bc2bfcdd348736cb3f1973884dd71` |
| V55 | `runs\recovered_v55_1793443036653\execution_manifest.json` (`aefde7e4...`) | 0 / 778.02 s | `v55_members.npz`: `bff5dd29e537f62f47585c7d5f27157db9e1330bcfa951dfbaaafdf39c04442c` |
| V17 | `runs\recovered_v17_1793443853760\execution_manifest.json` (`6f0e4f03...`) | 0 / 645.08 s | `v_17_full_suite_bundle.npz`: `41911ba036bd9f9b6e2f40caf1f37ce30820be21bdf9a1208de23a33e4a2d3eb` |
| Anchor74 | `runs\recovered_anchor74_1793444526421\execution_manifest.json` (`22c292b3...`) | 0 / 890.33 s | CSV: `6c3207977abc21091b64a7446f979c17ed2ce9aaabf18b3acb2065569c559bc8` |
| Track1B | `runs\track1b_1793445445960\track1b_manifest.json` (`eaeb49ab...`) | generated, 61,500 rows | CSV: `9b9f7dded04ae606b8d0ca389154304f9365bf628589e1dba4e5bba652cc6f6b` |

The full source hashes, output hashes, and manifest records are in the JSON evidence file.

## 2. ID alignment and row-order proof

- The raw test file has 61,500 unique IDs, exactly `171202` through `232701`.
- The frozen champion, both prior high-correlation CSV copies, current Track1B, and all inspected Anchor74 CSVs have unique IDs and exactly the raw test ID set: **zero missing and zero extra IDs**.
- All CSV-to-CSV metrics in this report align by `SK_ID_CURR`; the audit fails if IDs are duplicated or sets differ.
- Current Anchor74 is explicitly ID-reindexed to raw test order by the current builder; its run manifest also verifies exact in-order equality.
- V17/V55/XGB NPZ/NPY bundles do **not** contain `SK_ID_CURR`. The recovered generators read the raw test frame without sorting, concatenate train then test with `ignore_index=True`, and predict on the test tail in that order. This establishes source-level order for the current runs. Historical NPZ caches have no embedded IDs or ID sidecar binding them to row identities, so historical member comparisons below are reported as **positional only**, not as ID-aligned claims.

## 3. Formula audit — not the source of the regression

The historical V5 notebook's Stage 7/8 and current [`src\build_track1b.py`](../src/build_track1b.py) use the same eight members and weights:

| Suite member | Weight |
|---|---:|
| V17 `test_e1Aet` | 0.2932 |
| XGB10 test predictions | 0.2596 |
| V17 `final_test` | 0.2051 |
| V55 `test_cb_product` | 0.0874 |
| V17 `test_e1et` | 0.0489 |
| V55 `test_deep_et` | 0.0442 |
| V55 `test_v2_allfam_oh` | 0.0383 |
| V55 `test_v2_mono` | 0.0233 |

Both use `(average_rank - 0.5) / n`, `norm.ppf(clip(u, 1e-6, 1-1e-6))`, a weighted sum of suite probits, `0.85 * probit(anchor) + 0.15 * suite_probit`, then `(ordinal_rank - 0.5) / n`. The weights sum to exactly 1.0, so the historical normalization step has no effect. No weight tuning or alternative formula was used in this audit.

Using the fixed formula on the high-correlation anchor plus historical caches reproduces all 61,500 prior Track1 values after the original 10-decimal CSV rounding. The independent metrics against the champion are Pearson/Spearman `0.9999975816`, MAE `0.0003014174`, RMSE `0.0006348771`, maximum difference `0.0125041`, mean rank displacement `18.54`, maximum `769`.

## 4. Component-level divergence

These are positional comparisons of current production suite arrays against the historical cached arrays with the same canonical key. They are **not ID-aligned** because the NPZ/NPY files contain no IDs.

| Member | Spearman | MAE | RMSE | Max abs diff | Mean / max rank displacement |
|---|---:|---:|---:|---:|---:|
| V17 `e1Aet` | 0.929469 | 0.075187 | 0.108342 | 0.806390 | 4,626.5 / 49,611 |
| XGB10 | 0.937408 | 0.019429 | 0.033485 | 0.783539 | 4,422.4 / 48,243 |
| V17 `final` | 0.951633 | 0.061693 | 0.089785 | 0.671632 | 3,794.1 / 41,305 |
| V55 CatBoost product | 0.900804 | 0.025355 | 0.041132 | 0.737251 | 5,777.2 / 53,958 |
| V17 `e1et` | 0.929471 | 0.075230 | 0.108338 | 0.805450 | 4,629.1 / 49,575 |
| V55 `deep_et` | 0.921398 | 0.079931 | 0.114458 | 0.807925 | 4,915.7 / 49,687 |
| V55 all-family | 0.926560 | 0.076797 | 0.110636 | 0.793389 | 4,723.0 / 48,793 |
| V55 monotone | 0.900071 | 0.092644 | 0.129055 | 0.844349 | 5,697.6 / 51,927 |

Anchor74 is the largest individual problem: the current Anchor74 is only `0.8691278` Spearman against the historical Anchor74, while the Phase 3 Anchor74 used in the successful high-correlation route is `0.9999967390`. Since Anchor74 supplies 85% of the macro in probit space, substituting the wrong Anchor74 has a large effect. The suite also diverges from cached historical members, and cannot be treated as equivalent just because all eight keys exist.

Fixed-weight input swaps (diagnostic ablations, not tuning):

| Anchor / suite inputs | Spearman vs champion | MAE | Mean / max rank displacement | 10dp matches to prior high-corr CSV |
|---|---:|---:|---:|---:|
| Phase 3 Anchor74 + historical cached suite | **0.9999975816** | 0.0003014 | 18.54 / 769 | **61,500** |
| Phase 3 Anchor74 + current production suite | 0.9986817181 | 0.0101083 | 621.66 / 8,078 | 69 |
| Current production Anchor74 + historical cached suite | 0.9031292271 | 0.0946219 | 5,819.25 / 46,655 | 15 |
| Current production Anchor74 + current production suite | **0.8921288944** | 0.0996704 | 6,129.73 / 50,504 | 5 |

The last row independently reproduces the current production Track1B rank order (within one rank for all but two rows), confirming that the failure is in the inputs selected for composition, not a missing or misordered formula stage.

## 5. Feature and architecture parity

The high-correlation Anchor74 lineage is the Phase 3 Hasan feature/model route:

- Raw train/test -> 669 features -> gain-selected 550-feature matrix -> five-model zoo -> transductive tail/graph/discrepancy pyramid -> Anchor74.
- Phase 3 rebuilt 669 train/test matrices match `D:\gci\gemini\cache\` files byte-for-byte, including hashes, column order, dtypes, and values.
- Phase 3 selected 550 train/test matrices also match their historical cache files byte-for-byte.
- The matrices preserve 34 train and 16 test positive-infinity feature values; this is intentional in the Phase 3 LightGBM path. Phase 4's experiment notes specifically warn that converting these infinities to NaN changes the deep LightGBM behavior.

The current direct Anchor74 run is a different branch:

- It was produced by recovered `recreate_precision_anchor.py`, not by the full Phase 3 `reconstruct_transductive_pyramid.py` output used in the high-correlation notebook.
- Current feature audit: 695 columns, schema hash `697144f4dfcca46436baef3695e9ce8d1de09a5e201e81ddf515d21ed60bbe40`, and zero infinities.
- Compared with the Phase 3 669-name set, only 599 names overlap; 96 names are current-only and 70 Phase 3 names are absent. Only 506 names overlap the selected 550 list. The order and schema are not equal.
- The current V17/V55 generators also build their own feature manifolds. Passing their expected internal schema checks does not prove equivalence to historical cached model predictions.

The source history confirms the path split:

- The successful V5 notebook explicitly selects `02_EXP85_REBUILT\artifacts\v_20261027_precision_transductive_apex.csv` for its rebuilt anchor and `01_EXP85_RECOVERY\cache\` for the eight-engine suite.
- The current runner instead executes the direct recovered component generators and passes their new outputs to Track1B.
- The standalone `create_submission_recovery_v5_rebuilt.py` currently prefers `02_EXP85_REBUILT\rebuilt_artifacts\` when that directory exists. That same-named Anchor74 file has the low-correlation `04de...` hash, not the successful `315b...` artifact selected by the notebook. The two paths must not be interchanged based on the filename.

## 6. First divergence and root cause

**First divergence:** Anchor74 artifact resolution, before Track1B synthesis. The current pipeline treats a direct recovered `recreate_precision_anchor.py` run as interchangeable with the Phase 3 rebuilt Anchor74. The byte hashes and ID-aligned metrics disprove that equivalence. The current pipeline then also substitutes newly retrained V17/V55/XGB10 outputs for the historical cached suite members. The unchanged formula faithfully combines the wrong lineage.

**Most likely root cause:** component identity was keyed by the shared labels `anchor74`, `v17`, `v55`, and `xgb10`, rather than by their exact producer, architecture, feature schema, and artifact hash. A same-named Anchor74 artifact in a different directory belongs to a different reconstruction. The new production manifest verifies the selected generator ran successfully from the raw CSVs, but it does not certify that the generator is the specific high-correlation transductive Anchor74 producer.

This is a supported root cause for the correlation regression. The exact reasons why each current V17/V55/XGB member differs from the cache—historical code revision, fold assignment, model parameters, hardware nondeterminism, or data treatment—remain unresolved because the historical NPZ caches contain neither IDs nor a producer manifest. Do not attribute those differences to one unverified cause.

## 7. Required correction and gates before another production run

1. Pin and hash the intended Anchor74 producer and artifact. For the known high-correlation route, use the Phase 3/`02\artifacts` Anchor74 hash `315b...`; reject the `04de...`, `c1f...`, and current production `6c32...` artifacts for this fidelity target.
2. Reproduce the high-correlation composition from existing inputs first: Phase 3 Anchor74 plus the historical suite cache, using the fixed formula above. Preserve the result as a **hybrid historical-fidelity** output and disclose the cache lineage.
3. Keep the **raw rebuild** claim separate. Before replacing any historical suite member with a retrained member, reconcile it against its historical counterpart with verified row IDs, features, parameters, folds, seeds, and numeric treatment. NPZ positional correlation alone is insufficient certification.
4. Fix path resolution so `artifacts` and `rebuilt_artifacts` cannot be selected interchangeably. Require an expected source hash, feature-schema hash, and artifact hash in the component manifest—not just the generic component name.
5. Before any training: test that the fixed-formula in-memory composition reproduces the frozen high-correlation CSV at 10 decimals and its metrics; test that the wrong Anchor74 hash is rejected; test unique and exact ID sets; attach an ID sidecar/hash to every newly generated NPZ/NPY test vector.

No model training was done for this investigation. `RAW_REBUILD_CERTIFIED` remains **false**: the known `0.9999975816` route depends on historical cached suite predictions, while the current fully executed production inputs do not reproduce the historical component lineage.
