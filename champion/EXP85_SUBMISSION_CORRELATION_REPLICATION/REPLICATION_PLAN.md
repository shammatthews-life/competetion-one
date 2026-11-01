# Replication Plan

## Completed stages

1. Inspected local repository state and confirmed the frozen Phase 2 release was already committed and pushed before correlation work began.
2. Preserved Phase 2 in a separate immutable release folder; its candidate hash and release verifier pass.
3. Created this separate correlation project and inventoried the historical notebook, existing auditor, reports, references, candidates, and raw historical cache dependencies.
4. Reproduced the exact notebook's Track 1 V5 comparison and reran the original ID-aware auditor on the same pair.
5. Compared distinct historical candidates against the same immutable champion reference using a new ID-aligned runner.
6. Recorded the source/candidate hashes, scripts, run commands, method, and full-precision metrics.

## Comparison order

1. Reproduce the original notebook's positional Stage 9 computation only after verifying identical, unique ID order.
2. Repeat that pair using explicit ID joins; report all error and rank statistics.
3. Compare the Phase 3 `submission_exp85_rebuilt.csv`, archived full rebuild, `submission_final.csv`, and current Track1B separately. Never merge their metrics into one result.
4. Compare the original reports and the exact on-disk file hashes; report disagreements without replacing missing or stale files.
5. Assess model lineage independently of the numerical correlation.

## Stop conditions

- Missing artifact, hash mismatch, duplicate IDs, differing ID sets, nonfinite/out-of-range predictions, or a reused run directory is a hard failure.
- Do not retrain models to reproduce a correlation.
- Do not change the frozen Phase 2 candidate, historical champion, or any source artifact.
- Do not tune against the historical reference or infer an official score from local metrics.

## Next milestone

The numeric result is now reproduced. The next precise task is a read-only, component-manifest audit of the historical cache inputs and the Phase 3 Anchor74 producer used by the notebook. The historical V17/V55/XGB10 arrays have no embedded IDs or ID sidecars, so they cannot presently be independently rebound to IDs from their files alone. Do not claim a raw-only rebuild until that lineage gap is resolved.
