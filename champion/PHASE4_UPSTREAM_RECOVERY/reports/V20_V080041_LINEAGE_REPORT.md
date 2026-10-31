# V20 and V080041 Lineage Audit

## V20 artifact

- Path: `D:\gci\gemini\v_20_apex_grand_champion_submission.csv`
- SHA-256: `41c39e302e9bc25c520157e9d1fb22cc61c3cafcc3754a6629fcd1e29321f6c0`
- Size: 1,677,103 bytes.
- Validation: 61,500 rows, unique IDs, IDs exactly match raw `test.csv` order,
  finite targets in `[0, 1]`.
- Located source references are consumers in evaluation, synthesis and later
  build scripts. No exact writer was found in the searched local roots or
  repository history. The artifact is available, but its production lineage is
  **unknown**, not certified.

## V080041 artifact

- Path: `D:\gci\gemini\v_solo_080041_pure_champion.csv`
- SHA-256: `8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b`
- Exact byte match: `scratch/final_sub/final_submission/submission_final.csv`,
  same SHA-256 and size (1,291,519 bytes).
- Both files contain 61,500 unique IDs in raw test order and finite targets
  ranging from `1.62599e-05` to `0.9999837401`.
- The available `solo_grand_master_apex_080.py` reads `submission_final.csv`
  and writes the V080041 artifact. This proves the final copy/export lineage,
  not the model provenance of `submission_final.csv`.
- Writer source SHA-256:
  `38917ee975beeb5abb1ff522490732d42e351ef161fc179e0b5bf6ad44871756`.
- Legacy final stacker source SHA-256:
  `8a90d233723639297b1a433ee474b37ea1d00879fdb518ac39ee75a0fe575662`.

## Missing upstream inputs

`reproduce_final_submission.py` resolves 18 named file dependencies. The exact
names, consumer lines, search roots, and missing status are listed in
`MISSING_UPSTREAM_DEPENDENCIES.csv`; none was found in the configured search
roots. These include historical prediction bundles, metadata, a 226-feature
matrix archive, labels and test IDs. Labels and IDs can be recovered from raw
CSV, but doing so does not replace the missing historical prediction arrays.

The audit did not fabricate inputs, infer a V20 writer from consumer code, or
claim that a copy of the final V080041 output rebuilds its upstream models.
Machine-readable evidence is in `upstream_lineage_evidence.json`.
