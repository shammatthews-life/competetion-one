# EXP85 Unrecovered Producer Findings

## Decision

**Status: INCOMPLETE.** The checked-out V85 artifacts can reproduce the frozen
historical CSV, but the available evidence does not establish a raw-data rebuild.
No training, model inference, tuning, candidate submission, or frozen-champion
modification was performed in this investigation.

## V20 producer

- The V20 submission exists at `gemini/v_20_apex_grand_champion_submission.csv`;
  its recorded SHA-256 is
  `41c39e302e9bc25c520157e9d1fb22cc61c3cafcc3754a6629fcd1e29321f6c0`.
- Its ID/row-order and target validation are recorded in the
  [V20/V080041 lineage audit](../../PHASE4_UPSTREAM_RECOVERY/reports/V20_V080041_LINEAGE_REPORT.md).
- Source references found in the repository are consumers/evaluators. The
  local repository is not shallow; `git log --all -S` found no original V20
  writer in the searched Python/notebook history. The filename itself is not
  proof of a producer.
- **Conclusion:** V20 is an available prediction artifact with unrecovered
  production lineage. Replaying or copying it would not independently rebuild
  its models.

## V080041 upstream dependencies

- `solo_grand_master_apex_080.py` exports/copies `submission_final.csv` to
  `v_solo_080041_pure_champion.csv`; it does not explain how the source
  predictions were trained.
- The supplied `final_submission.rar` inventory contains only
  `README.md`, `reproduce_final_submission.py`, and `submission_final.csv`.
  The reproduction script loads upstream predictions and a feature matrix;
  it is not a raw-data model trainer.
- The archived reproduction script is byte-identical to the repository copy
  (`gemini/final_submission/reproduce_final_submission.py`), SHA-256
  `8a90d233723639297b1a433ee474b37ea1d00879fdb518ac39ee75a0fe575662`.
- The current
  [missing-dependency inventory](../../FINAL_RAW_REBUILD_AUDIT/MISSING_DEPENDENCIES.csv)
  has 18 named inputs: 15 historical prediction/feature artifacts remain
  unrecovered, `meta.json` was found outside the legacy resolver, and labels
  plus test IDs are derivable from the raw CSVs. The two raw arrays do not
  replace the unavailable learned predictions.
- The archive's `submission_final.csv` matches the V080041 artifact, SHA-256
  `8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b`.
- **Conclusion:** the exact final export is reproducible from its saved CSV;
  the model ancestry and a raw rebuild are not.

## V8 KNN cache producer

The historical V8 trainer consumes `knn_feature_tr.npy` and
`knn_feature_te.npy` as `KNN_250_RISK_RATE`; it does not construct them.
Other model scripts consume the same arrays under alternate column labels.
Both cache copies currently available are byte-identical:

| Artifact | Shape / dtype | SHA-256 |
|---|---|---|
| `knn_feature_tr.npy` | `(171202,)`, `float32`, finite | `6bca1368faaefe669f6ac0e0755e72e038ec89fd55dab1058a7b71345ecb503b` |
| `knn_feature_te.npy` | `(61500,)`, `float32`, finite | `9cfad6a1774dff8824916296801ee65639ee018cf2eeb35731be1fc01ff61908` |

The arrays are present in both `D:\gci\gemini\cache` and the clone's
`gemini/cache`; matching hashes establish copy identity only, not provenance.
No exact producer was found in the local Python/notebook sources or Git
history. The supplied V15 research source implements a different smoothed
target-rate method: nested folds with `k=50` over three EXT features and `k=100`
over EXT plus three ratios. Its README describes a rejected 500-neighbour
experiment. Neither establishes the historical 250-neighbour cache algorithm,
feature space, fold treatment, or seed.

**Conclusion:** the cached KNN values exist, but the writer/configuration needed
to regenerate them exactly remains unknown. Do not silently substitute the
available V15 KNN recipe.

## Origin notebook and historical replay

- The user-designated
  [origin notebook](../../exp85_solo.pre_independent_wiring_backup.ipynb) is
  unchanged at SHA-256
  `1ccfab28ef8534fc176c757c79250547c03a041a922851d683c82d0df1241d23`.
  It has 12 code cells and no saved execution counts or outputs, so it records
  source, not a verifiable original run. The code-cell comparison is in the
  [notebook lineage report](./NOTEBOOK_LINEAGE_COMPARISON.md).
- The
  [historical V85 replay verification](./HISTORICAL_V85_REPLAY_VERIFICATION.json)
  recomputed the fixed blend from cached Anchor74, V17, V55, and XGB10
  predictions and matched the frozen V85 CSV byte-for-byte. This is strong
  historical replay evidence, not independent model regeneration.
- The
  [Git LFS integrity check](./GEMINI_LFS_SOURCE_VERIFICATION.md) passed for
  26/26 checked-out payloads and local LFS objects. The separate
  29-entry feature-audit SHA manifest was also checked with zero mismatches.
  These checks preserve artifact integrity; they do not prove model lineage.

## Search and execution record

| Check | Result | Exit | Elapsed |
|---|---|---:|---:|
| `git rev-parse --is-shallow-repository` | `false` | 0 | Not timed |
| `git log --all -S'knn_feature_tr.npy' -- '*.py' '*.ipynb'` | No producer commit found | 0 | 0.344 s |
| `git log --all -S'v_20_apex_grand_champion_submission.csv' -- '*.py' '*.ipynb'` | Two audit commits only; no original producer found | 0 | 0.231 s |
| `python -m py_compile` on both audit scripts | Passed | 0 | 0.507 s |
| `git diff --check` | Passed | 0 | 0.191 s |
| `python source/verify_lfs_source_integrity.py` | 26/26 LFS worktree hashes and local object hashes match; origin notebook baseline matches | 0 | 2.484 s |
| `python source/verify_feature_audit_manifest.py --root <bridge> --manifest <bridge>\manifests\SHA256_MANIFEST.csv` | 29/29 sizes and SHA-256 values match | 0 | 1.328 s |
| Historical V85 replay verifier | Exact frozen-CSV byte match; cached inputs used | 0 | 0.628 s |
| Alternate cache namespace diagnostic | Did not match; wrong cache namespace | 0 | 0.666 s |

The alternate namespace diagnostic used `gemini/champion/cache` rather than the
historical `gemini/cache` inputs and must not be substituted.

The manifest verifier recorded its invocation, exit code, elapsed time, input
manifest hash, and per-file expected/actual sizes and hashes in
[`FEATURE_AUDIT_SHA256_VERIFICATION.json`](./FEATURE_AUDIT_SHA256_VERIFICATION.json).

| Evidence | SHA-256 |
|---|---|
| `verify_lfs_source_integrity.py` | `dc68014dace735fc58d85dfc1bde81fc5da8303937bf3614ba8d1afdf988fb4f` |
| `GEMINI_LFS_SOURCE_VERIFICATION.json` | `6af2b004f0aa83bcc0e257735ec53fa52abc8913f49d8184f0d2ee474b5a97f5` |
| `verify_feature_audit_manifest.py` | `1e37da4691cba793fbe118a71286400d3093f99f24900242c7d031e9dea3c0bd` |
| `FEATURE_AUDIT_SHA256_VERIFICATION.json` | `91f3edb080e9831641333808b3534ebad312ddbde6d39546d00cca87f92a777f` |

## Smallest next corrective action

1. Recover the exact V8 KNN source/configuration and its raw feature/fold
   inputs, or label V8/V17/V35 descendants as requiring historical cache input.
2. Obtain the 15 missing V080041 learned prediction/feature artifacts together
   with their original producer code/configurations.
3. Recover the V20 writer and parent-model lineage.
4. Only then run an isolated full-production rebuild with source/input hashes,
   fold/device logs, and validated outputs.

Until those dependencies are recovered, preserve V85 as a separate historical
cached replay and do not claim a full raw rebuild or official-score
reproduction.

## External source hashes

| User-provided source | Bytes | SHA-256 |
|---|---:|---|
| `final_submission.rar` | 401747 | `3c61377f272d1230d80de1e98bdd70401fdd1b57bac2fc1cf39922512f0a3e51` |
| `files.zip` | 15519 | `6a4f1eeb309f8df59dfc3dccf078cc360c318858b6dc70241a78b8716c631db8` |
| `GCI_Research_Progress_Report_2026-10-22.docx` | 47420 | `236b561e805cc737075d2855b7f3b13b45485220907bf91f27ef7359ad0a7853` |
| `GCI_Community_Research_Full_Extraction_and_Triage.docx` | 224937 | `e306685f015b4604160f0001ef3ba7c158f85351256e96fea62b92d9583f8ed2` |
| `solution_v5_gci_research.py` | 140761 | `4f0f1e9208f99ef2350af05f9c3da98eaae922de729295f2e0c410186924ae74` |
| `README_HONORS_V15.md` | 7714 | `57a935aafcd4b785b15870f907c61ee17b88deabfb4ff69e8602e6adc87d0f25` |
| `gci_v25_raw_dump.py` | 52437 | `e0ee005ab9f7738cb85bde95afd18f42da87485d2df33a5923d368608014c5fb` |
