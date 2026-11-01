# Replication Status

**Status: NUMERIC RESULT REPRODUCED; LINEAGE CLASSIFICATION CORRECTED**

- `0.9999975816` Spearman and Pearson are reproduced from the exact V5 candidate (`91aa95c9...`) against the frozen historical champion (`78c31bb4...`).
- Original notebook row order is independently confirmed: all 61,500 unique IDs match, with no missing or extra IDs.
- The original isolated audit script independently reproduces the same result after ID alignment.
- The source of the high-correlation candidate is hybrid: Phase 3 Anchor74 plus historical V17/V55/XGB10 suite caches. It is not a full independent raw-data rebuild.
- `0.9999628752` is the separate Phase 3 candidate (`8165ad2c...`).
- `0.9982094982` is the separate archived full-rebuild candidate (`6bfc518b...`); its provenance was not recertified by this comparison.
- `0.9969658880` is the separate `submission_final.csv` artifact (`8e8158ae...`).
- `0.8921288944` is the separate current production Track1B candidate (`9b9f7dde...`).
- Nine new synthetic runner tests pass; the original correlation auditor's synthetic cases pass.
- No historical source, submission, frozen Phase 2 candidate, or champion file was modified.
- No official score was evaluated during this work.

The detailed runs, inputs, and full-precision metrics are indexed in `REPLICATION_MANIFEST.json`.
