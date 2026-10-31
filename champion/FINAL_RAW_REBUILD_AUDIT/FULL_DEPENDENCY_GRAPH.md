# EXP85 Full Dependency Graph

## Fresh raw-candidate path executed in this audit

```text
Raw train.csv + test.csv
  ├─> generate_xgb10_from_scratch.py
  │     ├─> y_train.npy
  │     ├─> oof_10f_xgb.npy
  │     └─> test_10f_xgb.npy
  ├─> generate_v55_from_scratch.py
  │     └─> v55_members.npz (fresh OOF/test member predictions)
  ├─> generate_v17_from_scratch.py
  │     └─> v_17_full_suite_bundle.npz (fresh OOF/test member predictions)
  └─> generate_anchor74_from_scratch.py
        └─> v_20261027_precision_transductive_apex.csv

Fresh Anchor74 + fresh 8 selected test members
  └─> isolated 85/15 probit/rank formula
        └─> candidate_submission.csv
              ├─> independent formula recomputation
              ├─> exact raw test ID/schema/range checks
              └─> ID-aligned comparisons to frozen champion and prior raw candidate
```

The generator processes use only the explicit raw CSV paths as prediction
generation data sources. The isolated component directory begins empty. The
run log records command, working directory, exit code, elapsed time, source
hashes, raw input hashes, output hashes, dependency versions, and GPU
preflight/device evidence. This proves execution provenance for the candidate
produced by the current source; it does not prove historical model parity.

## Historical replay and other branches

```text
Historical Recovery V5
  ├─> historical Anchor74 / recovered Phase 3 anchor artifacts
  ├─> cached V17 predictions ─┐
  ├─> cached V55 predictions ─┼─> fixed 8-member formula ─> Recovery V5 replay
  └─> cached XGB predictions ─┘

V20 CSV ──> consumers/evaluators
  └─> original writer and model ancestry not recovered

V080041 CSV
  └─> known export from submission_final.csv
        └─> legacy stacker
              ├─> 15 unavailable historical prediction/feature bundles
              ├─> meta.json (found locally, outside resolver search locations)
              └─> y.npy/test_ids.npy (regenerable from raw CSVs)
```

The V20 and V080041 branches are not direct inputs to the v85 8-member formula.
Their unresolved lineages still block certification of those separate
artifacts, but do not by themselves prevent this raw-component candidate from
being generated.

## Completed run evidence

The full-mode XGB10, V55, V17, and current-source Anchor74 generators each
exited `0` in the isolated run. The candidate was synthesized from those fresh
outputs and independently recomputed; no historical prediction cache entered
candidate generation. See `RAW_REBUILD_EXECUTION_REPORT.md` for per-generator
commands, times, hashes, device observations, and warnings, and
`FINAL_SUBMISSION_VALIDATION.md` for schema and ID-aligned comparisons.
Successful component execution does not close the historical feature/model
lineage gaps shown above.

## Dependency classification

| Dependency | Needed for isolated v85 candidate | Historical requirement | Result |
|---|---:|---|---|
| Raw `train.csv`, `test.csv` | Yes | Also foundational historical inputs | Present; hashes recorded in execution evidence |
| Current XGB10, V55, V17 and Anchor74 generator source | Yes | May differ from original producers | Present and executed in full mode |
| Historical V17/V55/XGB prediction caches | No for this run | Required for exact Recovery V5 replay | Historical replay remains a separate hybrid benchmark |
| Historical 669/550-feature Anchor74 inputs/matrices | No for this run | Required to prove historical Anchor74 parity | No exact raw/model parity established |
| V20 original producer/checkpoint/configuration | No for this run | Required to certify V20 lineage | Not recovered |
| 15 V080041 historical prediction/feature arrays | No for this run | Required for faithful V080041 stacker replay | Not recovered in searched roots |
| `meta.json` for V080041 | No for this run | Needed by legacy stacker configuration | Found at `D:\gci\pre info\oct22\meta.json`, but not in resolver's configured candidate paths |
| `y.npy`, `test_ids.npy` for V080041 | No for this run | Required only by that branch | Values are derivable from raw `train.csv` and `test.csv`; historical array files were not found |
| Frozen champion | Comparison only | Not a training input | Read-only reference; not blended into candidate |
| Historical v85 submission | Comparison only | Not a training input | Not overwritten or submitted |
