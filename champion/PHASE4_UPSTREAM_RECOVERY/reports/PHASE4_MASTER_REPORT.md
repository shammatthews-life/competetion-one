# EXP85 Phase 4 Recovery — Master Report

## Executive verdict

**Raw rebuild certification is not supported by the evidence currently in Git.**
The existing canonical provenance manifest says `ARTIFACTS_VERIFIED`; it records
checks of existing files, not a clean-room execution of the four generators.
The clean-room manifest path claimed by the older report is absent from both the
working tree and Git history. Do not treat either the 0.999997 hybrid benchmark
or the 0.998209 raw-trained candidate as a certified raw rebuild.

An isolated fast-mode smoke run did complete all four generators with exit code
0 and exact Anchor74 test-ID order. It is explicitly
`SMOKE_COMPLETE_NOT_CERTIFIED`; no full EXP85 retraining was launched. The frozen
champion and historical submissions were not modified.

## Three remaining investigations

| Area | Verified result | Remaining gap |
|---|---|---|
| CatBoost | Five same-seed GPU fold-5 fits are mutually extremely stable and all closely match historical OOF. The current matrix, labels, row maps, fold split, and requested model settings were verified. | The earlier two-run split was not reproduced. The CPU control has different effective settings, so it cannot isolate device effects. Historical serialized model/runtime/driver evidence is unavailable. Root cause remains unresolved. |
| V20 | `v_20_apex_grand_champion_submission.csv` exists, has 61,500 unique IDs in raw-test order, and has SHA-256 `41c39e302e9bc25c520157e9d1fb22cc61c3cafcc3754a6629fcd1e29321f6c0`. | No exact writer or generating run was found. The located source references are consumers, not provenance. |
| V080041 | Its CSV is byte-identical to `submission_final.csv`, SHA-256 `8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b`; both align with raw test IDs. The available writer reads the final submission and writes the V080041 output. | The final submission's upstream stacker depends on 18 named files that were not found in the searched roots. Its original model lineage cannot be independently replayed from currently located files. |

Detailed evidence: [CatBoost report](./CATBOOST_REPRODUCIBILITY_REPORT.md),
[dependency graph](./UPSTREAM_DEPENDENCY_GRAPH.md),
[V20/V080041 lineage](./V20_V080041_LINEAGE_REPORT.md),
[clean-room audit](./CLEAN_ROOM_PROVENANCE_REPORT.md), and
[certification verdict](./RAW_REBUILD_VERDICT.md). Hashes for the report,
source, tests, run execution records, and directly modified provenance files
are recorded in `../manifests/PHASE4_SHA256_MANIFEST.json`.

## Keep the benchmarks separate

| Experiment | Recorded SHA-256 | Recorded comparison | Interpretation |
|---|---|---|---|
| Track 1A / Recovery V5 hybrid | `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482` | Spearman `0.9999975816` to the frozen champion | Uses historical cached suite predictions; a successful hybrid replay, not an independent raw rebuild. |
| Track 1B candidate | `6bfc518b7abc208ff7e95c5dd8029b8d418d7ae3c1c94f9cdc6c4499598bc042` | Spearman `0.9982094982` to the frozen champion | Valid ID-aligned candidate; raw lineage is not certified. |
| Frozen champion | `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947` | Historical reference | Must remain unchanged. |

These are previously recorded evidence values, not newly recalculated metrics.

## Reproducibility safeguards added

- The clean-room runner distinguishes smoke mode, artifact verification, and
  full raw-rebuild certification. Smoke or verify-only runs cannot authorize
  full-scratch submission synthesis.
- Full certification requires four successful, non-fast generator executions
  in an empty output directory, with source/raw-input/output hashes.
- Submission builders reject a manifest that does not satisfy this contract.
- Run outputs and copied raw data are excluded from Git; compact execution
  metadata and reports are retained.

## Next smallest safe actions

1. Recover the historical V20 writer/run metadata and the 18 missing stacker
   dependencies, or document that those upstreams are irrecoverable.
2. Obtain historical CatBoost model and runtime provenance before attributing
   its discrepancy to GPU nondeterminism.
3. Only if a full rebuild is explicitly authorized, run the four full-mode
   generators into a new empty isolated destination and verify all IDs,
   outputs, source hashes, and logs before changing certification status.

The current result remains **not certified**; no missing artifacts were
fabricated and no cached predictions were relabeled as raw-generated.
