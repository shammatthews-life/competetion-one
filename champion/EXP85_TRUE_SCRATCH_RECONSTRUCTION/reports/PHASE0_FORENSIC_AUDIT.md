# EXP85 Phase 0 Forensic Audit

## Gate

**Phase 0 evidence checks pass; full raw reconstruction remains blocked.**

The earlier audit's four feature hash conflicts are resolved: canonical
historical matrices and the independent raw rebuild have exact cell-level
parity, while one replay emitted equivalent Feather files with different
serialization hashes. Raw zoo training and downstream bedrock stages have
also been executed in isolated prior runs. The historical pipeline is still
not independently reproducible end-to-end because upstream V080041/V20/VGold
model ancestry is not fully recoverable, and one zoo family (CatBoost) does
not exactly match its historical cached predictions.

This is a Phase 0 evidence report. No model was trained, no experiment or
correlation comparison was rerun, no submission was made, and no existing
candidate/cache/champion/Phase 2 output was changed. This audit's outputs are
confined to this directory.

## 1. Evidence boundaries and preserved results

The historical champion is the immutable reference:

- File: `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`
- SHA-256: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
- 61,500 rows; expected columns `SK_ID_CURR,TARGET`.

The earlier V5 result is an **ID-aligned hybrid cache reconstruction**, not a
raw-only rebuild. Its recorded candidate is
`D:\gci\champion\08_SUBMISSIONS\submission_exp85_recovery_v5_rebuilt.csv`,
SHA-256 `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`.
The prior audit reports Pearson `0.9999975815867518`, Spearman
`0.9999975815867501`, MAE `0.0003014174`, maximum absolute difference
`0.0125040651`, 3,093 exact prediction matches, and maximum rank displacement
769 against the frozen champion. This audit did not recompute those metrics.

That V5 candidate uses the historical V17, V55, and XGB10 prediction caches,
plus a **different rebuilt Anchor74 generation** (SHA-256
`315b0bd80e4698240a7a902648a157d84da1480dea77901cdb60892502a7b55b`), not
the original historical Anchor74 CSV (SHA-256
`e5869c8be91b5a69f8cbedfa36e35185d4ce3dd6b781df01440b42fe6d009484`). The
candidate is byte-identical to the earlier V5 candidate, not to the champion.
The numerical correlation therefore must not be described as a full raw-data
rebuild or as proof of the official competition score.

A separate prior full-mode raw-component candidate is recorded with
SHA-256 `2a5d2dfc2504f86a6d6c6d42bfc63fff7f62a10ea66135b1fee44982f9a348fe`.
Its ID-aligned Spearman against the historical champion was `0.891570`.
Those previous runs are evidence only; this Phase 0 audit did not rerun them.

The independent Phase 2 release remains a separate experiment and was not
read as a model input or modified.

## 2. Raw inputs and alignment authority

| Input | SHA-256 | Audit use |
|---|---|---|
| `D:\gci\inpuy\train.csv` | `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1` | Raw training features and labels |
| `D:\gci\inpuy\test.csv` | `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65` | Raw test features and authoritative test ID order |
| `D:\gci\inpuy\sample_submission.csv` | `6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db` | Submission schema reference |

The runnable verifier checks raw/champion/intermediate submission IDs and
order, cached-array lengths, and the pinned feature-parity and selector
manifests. Prediction `.npy`/`.npz` files do not contain row IDs or fold
assignments, so matching lengths alone cannot prove row-level alignment.
Pickle inputs are fingerprinted but never deserialized.

The measured validation passed: raw train has 171,202 rows and raw test has
61,500; `y_train.npy` exactly matches raw `TARGET` order; raw test, sample
submission, historical champion, historical V85 output, prior V5 candidate,
and prior raw candidate have unique IDs in exact raw-test order. Their
submission targets are finite and within `[0,1]`.

## 3. Historical dependency graph

Machine-readable nodes, edges, hashes, producer/consumer status, and synthesis
formula are in [dependency_graph.json](../manifests/dependency_graph.json).
Core path:

```text
train.csv / test.csv
  ├─> exact 669-feature raw rebuild ─> five-fold 550-feature selector replay
  │      └─> raw-trained five-family zoo candidate
  │             ├─> historical-cache comparison (4/5 families exact; CatBoost non-exact)
  │             └─> raw bedrock candidate (hybrid: historical base anchor consumed)
  ├─> historical V080041 + V20 + VGold
  │      └─> V080150 ─> V080200 ─> Level-2 stacker
  │             └─> asymmetric gated apex ─> optimal transductive apex
  │                    └─> precision apex / Anchor74
  ├─> historical five-family zoo cache ────────────────┘
  ├─> V17 trainer/recreation ─> cached V17 bundle
  ├─> V55 trainer/recreation ─> cached V55 members
  └─> XGB10 trainer/recreation ─> cached OOF/test predictions

Anchor74 + eight fixed rank/probit suite members
  └─> 0.85 * Anchor74 + 0.15 * suite
        └─> ordinal rank projection ─> historical EXP85 V85 submission
```

The final historical blend uses eight suite weights:

| Member | Weight |
|---|---:|
| `v17_e1Aet` | 0.2932 |
| `xgb10` | 0.2596 |
| `v17_final` | 0.2051 |
| `v55_cbprod` | 0.0874 |
| `v17_e1et` | 0.0489 |
| `v55_deep_et` | 0.0442 |
| `v55_allfam` | 0.0383 |
| `v55_v2_mono` | 0.0233 |

Each member is mapped by average rank to `(rank - 0.5) / n`, clipped and
probit-transformed, then combined with the fixed weights. The final value is
the ordinal rank of `0.85 * probit(Anchor74 rank) + 0.15 * suite_probit`,
scaled as `(rank - 0.5) / 61500`. The historical builder reads the
Grand-Master macro-apex CSV for comparison/telemetry; it is not part of the
blend formula. The original 85/15 formula and suite weights are therefore
recoverable; producer parity is not.

### Core cached-array schema

These are verified historical cache schemas, not proof of their producers:

| Artifact | Members / arrays | Shape and dtype | ID/fold/checkpoint metadata |
|---|---|---|---|
| `v_17_full_suite_bundle.npz` | `oof_old`, `oof_e1`, `oof_e1et`, `oof_e1Aet`, `oof_cb` | `(171202,)`, `float32` | None |
| same | `test_old`, `test_e1`, `test_e1et`, `test_e1Aet` | `(61500,)`, `float64` | None |
| same | `test_cb` | `(61500,)`, `float32` | None |
| same | `final_oof`, `final_test`, `final_auc` | `(171202,)` float64; `(61500,)` float64; scalar float64 | None |
| `v55_members.npz` | Seven `oof_*` members | each `(171202,)`, `float32` | None |
| same | Six `test_*` members | each `(61500,)`, `float64`; `test_cb_product` is `float32` | None |
| `oof_10f_xgb.npy`, `test_10f_xgb.npy` | OOF/test | `(171202,)`, `(61500,)`, both `float32` | None |
| `y_train.npy` | labels | `(171202,)`, `int32` | Re-creatable from raw labels |
| `hasan_expanded_zoo.npz` | Five `oof_*` and five `test_*` model-family arrays | OOF `(171202,)`, test `(61500,)`, `float64` | No IDs, fold metadata, or fitted models |
| same | `y_true` | `(171202,)`, `int64` | Labels only; no fold metadata |

All listed prediction values were reported finite in the prior read-only
artifact inspection. The verifier emits measured array schemas for named
files and records any hash mismatch without substituting a sibling generation.

## 4. Component status and exact gaps

| Component | Classification | Evidence and limitation |
|---|---|---|
| Raw train/test inputs | `REPRODUCED_FROM_RAW` | Present; hashes pinned above. |
| Historical 669-feature and selected 550-feature matrices | `REPRODUCED_FROM_RAW` | All four matrices match historical schema, column order, dtypes, missing masks, and every cell; 0 differing cells and maximum absolute difference 0. |
| Hasan five-family zoo | `REPRODUCED_FROM_RAW` candidate / `HISTORICAL_CACHE_ONLY` reference | An isolated five-fold raw training run exactly matches four families' OOF/test predictions; CatBoost is highly correlated but not exact. The historical cache has no IDs, folds, fitted models, or original run manifest. |
| Anchor74 formula | `HYBRID_HISTORICAL_CACHE_RECONSTRUCTION` for prior replay; raw end-to-end rebuild unresolved | The historical formula stages replay exactly when supplied cached zoo predictions and upstream anchors. A separate 143-feature raw candidate reached only 0.869128 rank correlation to historical Anchor74. The 669/550 and separate 695-feature lineages must not be conflated. |
| V17 | `HISTORICAL_CACHE_ONLY` | Historical bundle exists. A raw generator was executed in the previous full raw run, but it does not establish historical parity. The exact `D:\gci\codex\v_3_cache\resid16.pkl` dependency exists (44,680,005 bytes; SHA-256 `a0f20faa770250b5ed53e22fc23c7b65247bd4d08e0109f160c827f46f4eab12`). Static pickle disassembly identifies a pandas DataFrame with 32 named residual columns; exact shape/dtypes and producer lineage remain unverified. |
| V55 | `HISTORICAL_CACHE_ONLY` | Historical bundle and raw recreation source exist. The original producer consumes cached features/residuals/views/encodings and member artifacts; exact historical execution provenance is absent. |
| XGB10 | `HISTORICAL_CACHE_ONLY` | Historical OOF/test arrays and a raw writer exist. Arrays lack row IDs/folds; original producer/configuration is not verified. |
| V85 final blend | `SOURCE_AVAILABLE_UNTESTED` for a new independent rebuild; formula itself is recovered | The 85/15 formula and member weights are explicit and reproduced by prior cache replay. Do not treat formula replay as model regeneration. |
| Prior V5 candidate | `HISTORICAL_CACHE_ONLY` | Reproduces prior V5 output and approximately 0.99999758 champion correlation using historical V17/V55/XGB10 caches and a distinct rebuilt Anchor input. |
| V20 | `CONFIG_MISSING` | Prediction CSV exists, but no original writer/model ancestry was found in local source, archive, or non-shallow Git history. It feeds V080150 and then the Anchor74 bedrock chain. |
| V080041 | `MISSING_ARTIFACT` | Fifteen learned prediction/feature inputs remain unrecovered. V080041 is an upstream input to V080150; see [missing_artifacts.csv](../manifests/missing_artifacts.csv). |

The source snapshots are not one coherent executable lineage. The checked-in
`D:\gci\_gci_push_tmp\gemini\champion\exp85_solo.ipynb` is 28,611 bytes with
SHA-256 `2640eaa9b828c9b9485497f9ce35060bd4d89db179820a5a42bb153039862312`.
The separate archive file
`D:\gci\gemini\champion\exp85_solo.ipynb` is 68,547 bytes with SHA-256
`52996cf1a2c9b018195672ea00bc316cd6cf01bab8018f9956b55d8cae22c9a3`. They
are different snapshots, not a hash discrepancy at the same path. The earlier
cell inventory applies to the checked-in snapshot and reported zero executed
cells. The archive notebook contains a raw-input feature/training path, but
its source inspection is not evidence that it generated the historical
artifacts. Existing notebooks with labels such as “clean-room” or “exact
reproducible” are not independently trusted by their names. Their source and
artifact lineage must be reconciled before using one as the authoritative
historical trainer.

### Feature hashes: serialization difference, not matrix difference

The canonical historical matrices are under `D:\gci\gemini\cache`; matching
byte-for-byte copies are also present under
`D:\gci\gemini\champion\phase3_reconstruction\independent_rebuild`. The
feature-parity run wrote separate Feather files under
`D:\gci\champion\EXP85_MASTER_RECOVERY\runs\feature_parity_20261031_200506`.
Those files have different byte hashes, but the pinned parity report
(`636ba1f46ecbf7ef8c8b7aceb5b9a07ce009c91e4406c770589f0dc9d3d3154c`)
measures exact data equality:

| Comparison | Shape | Schema/order | Dtype/missing-mask mismatches | Differing cells | Max abs. difference |
|---|---:|---|---:|---:|---:|
| 669 train | 171,202 x 669 | Exact | 0 / 0 | 0 | 0 |
| 669 test | 61,500 x 669 | Exact | 0 / 0 | 0 | 0 |
| 550 train | 171,202 x 550 | Exact | 0 / 0 | 0 | 0 |
| 550 test | 61,500 x 550 | Exact | 0 / 0 | 0 | 0 |

The selector was also retrained from regenerated raw features using the
recovered five-fold LightGBM configuration. Its manifest records the same 550
features in the same order as the historical selection. The replay's Feather
hashes are retained separately in the dependency graph; they are not silently
substituted for the canonical cache hashes. The measured disagreement was a
serialization-level hash difference, not a feature-value discrepancy.

## 5. Requested blocker categories

**A. Missing cached artifacts**

- Historical learner checkpoints, fold assignments, and prediction row-ID
  sidecars are absent from the V17/V55/XGB10/zoo prediction caches.
- The V17 residual input exists at the exact trainer path and is hash-pinned,
  but only its pandas DataFrame type and 32 residual-column names were
  statically inspected; exact shape/dtypes and producer lineage remain
  unverified.
- Fifteen learned artifacts required by the separate V080041 stacker are
  unrecovered. They block raw regeneration of its saved V080041 output.
- V20's original producer and model lineage remain missing. V20 and V080041
  both feed V080150, an upstream anchor in the Anchor74 bedrock chain.

**B. Missing artifact generators**

- Producer code for the historical five-family zoo is available as recovered
  source, but the original full run manifest/environment is missing.
- Raw recreation scripts exist for V17/V55/XGB10 and Anchor74 candidates.
  Their presence does not establish they generated the canonical historical
  cache values.
- V20's original writer was not located.

**C. Incomplete reconstruction functions**

- The fixed eight-member/85-15 synthesis is complete and its prior cache replay
  passed. The incomplete work is upstream model regeneration and lineage, not
  the final arithmetic.
- Anchor74's historical output can be formula-replayed from cached zoo output;
  that does not regenerate the zoo from raw data.

**D. Incorrect paths or schema assumptions**

- V17's source expects `D:\gci\codex\v_3_cache\resid16.pkl`; the exact path is
  present and hash-pinned. Static disassembly identifies a pandas DataFrame
  and 32 residual columns, but the pickle was not deserialized and exact shape,
  dtypes, and producer remain unverified. A same-name `.feather` elsewhere is
  a different file/format and is not a substitute.
- Test/OOF arrays have lengths matching expected rows but do not carry IDs or
  folds. Positional alignment remains an explicit assumption.
- Multiple cache generations exist. At least one V17/V55 cache-recreation
  directory has hashes different from the canonical root cache; generation
  choice must be pinned by hash.

**E. Incorrect component assembly**

- No blend-weight or final-formula defect was found in the established V85
  synthesis. The prior 85/15 result was reconstructed with the recorded
  precision-sensitive calculations.
- A different Anchor74 generation was used for V5, so V5 is not a byte-exact
  replay of the full historical champion.
- The Grand-Master apex read by the final builder is diagnostic/telemetry, not
  a hidden blend input.

**F. Missing provenance/integrity checks**

- Historical model execution manifests do not pin original environment,
  device, checkpoints, or fold definitions. The Anchor74 replay manifest has
  `training_manifest: null`.
- Raw feature and selector replay manifests pin sources, raw inputs, folds,
  package versions, feature order, and output hashes. Historical model caches
  still lack equivalent producer manifests and sidecars.
- The checked-in and archive `exp85_solo.ipynb` snapshots differ; the original
  producer notebook is not identified.
- Historical suite caches lack test IDs and folds; array order cannot be
  independently certified.
- The reported `0.801340` is present in local historical score records, but
  no official platform response was independently verified in this Phase 0
  audit.

**G. Differences between generations**

- Canonical versus cache-recreation copies of V17/V55/XGB10 and Anchor74 are
  not assumed equivalent; each file hash is preserved separately.
- Feature-parity Feather files have distinct serialization hashes but exact
  cell-level parity with canonical historical feature matrices.
- Historical Anchor74 is documented as 669 engineered / 550 selected
  features. A separate raw candidate uses 143 features; another production
  lineage describes 695 features. These refer to distinct source snapshots.
- The previous full raw-component candidate did not establish historical
  parity (Spearman `0.891570` to the final champion). The separate
  raw-trained zoo reproduces four of five historical families exactly;
  CatBoost remains non-exact.

## 6. Repository and execution record

The working Git repository is `D:\gci\_gci_push_tmp`, HEAD
`a77b9ba113f1260250b436d1d756fdb83b1b1f9f`, non-shallow. The worktree already
contained unrelated Phase 2 and research changes before this audit; they were
left untouched. No commit, staging, or push was performed.

The final verifier invocation was
`python D:\gci\_gci_push_tmp\champion\EXP85_TRUE_SCRATCH_RECONSTRUCTION\source\run_phase0_audit.py`
from Python 3.12.10 / NumPy environment recorded in the execution log. It ran
from `2026-11-01T17:14:58.553462+00:00` through
`2026-11-01T17:31:46.782244+00:00` (9.906078 seconds) and returned **exit
code 0** with verdict
`PHASE_0_CHECKS_PASS_FULL_RAW_RECONSTRUCTION_BLOCKED` and no verification
errors. The focused command
`python -m unittest discover -s tests -p test_phase0_audit.py -v` returned
**exit code 0** (5 tests passed). The run inventoried 43 graph nodes, 56
edges, 39 artifact paths, 37 source files, and 63 array members. All recorded
Git commands, including the
selected `git log -S` searches, returned exit code 0. Those history searches
matched the V5 reconstruction/audit commits (including `79da8eb1`) and do not
establish original model-training producer commits.

The verifier is
[run_phase0_audit.py](../source/run_phase0_audit.py) and produces:

- `manifests/artifact_inventory.csv`
- `manifests/source_inventory.csv`
- `manifests/array_schema.csv`
- `manifests/data_validation.json`
- `logs/phase0_execution.json`

The execution log records timestamps, Python/platform/NumPy versions,
measured hashes, Git command/exit-code results, focused test command/output,
and any verification errors. The full Gemini archive census found 38,195
files / 5,681,313,604 bytes (including `.venv` and hidden files); this is a
discovery count, not a claim that every vendor/environment file is a project
source. Core source files and artifacts are fingerprinted individually by the
verifier.

## 7. Gate decision and next authorized step

The Phase 0 verifier now checks the pinned raw-data alignment, feature-value
parity evidence, artifact/source hashes, and focused tests. Raw feature,
selector, zoo-training, and bedrock-chain runs already exist in isolated
workspaces; repeating them with the same evidence would not resolve the
remaining historical ancestry gaps. A strict raw rebuild of the historical
Anchor74 chain is currently blocked by:

1. The original producer/configuration for V20 is absent from local source,
   archived implementations, and non-shallow Git history.
2. Fifteen learned V080041 upstream prediction/feature artifacts are absent
   from the supplied archive and searched workspaces; the available script
   only exports the saved submission.
3. VGold has source code and a saved output, but its historical execution
   provenance depends on cached V17/V55 streams and earlier submissions.

The recovered V080150 -> V080200 formula replay is exact, but starts from
cached V080150. Downstream Level-2/asymmetric/optimal/precision formula
replays also consume historical zoo predictions and/or that cached bedrock.
The raw-trained zoo is a valuable independent component result, not a
substitute for missing historical predictions. See
[RECONSTRUCTION_STATUS.md](../RECONSTRUCTION_STATUS.md) for completed stages,
execution evidence, and the precise continuation blocker.

The three result categories must remain distinct:

1. **Historical-cache reconstruction:** high-correlation V5 result exists;
   not raw-only and not byte-identical to the champion.
2. **Independent raw-data model:** Phase 2 remains frozen and separate.
3. **Full raw-data historical pipeline reconstruction:** not achieved; the
   historical 0.801340 score has not been independently reproduced here.
