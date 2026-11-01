# Next Steps

## Immediate

1. Keep the high-correlation V5 result classified as a hybrid cache replay.
2. Preserve the exact historical cache hashes and the exact Phase 3 Anchor74 hash with any future reproduction.
3. Do not use the generic name `Anchor74` as an identity: pin producer path, source hash, feature/schema hash, and prediction hash.

## To validate lineage

- Determine whether historical V17, V55, and XGB10 caches have reliable producer manifests or per-row ID sidecars. Current forensic evidence says the arrays do not embed IDs or a sidecar linking rows to IDs, so their mapping remains positional based on their generation order.
- Reconstruct the notebook's eight suite members from the exact caches and prove that the synthesized candidate matches all 61,500 values after ten-decimal output formatting. Existing root-cause analysis reports that this is already reproduced, but retain it as hybrid evidence.
- For any raw-only claim, train/derive every suite component without historical predictions and compare each output by verified IDs, fold assignments, feature schema, hyperparameters, seeds, and numeric treatment. Stop on any missing producer/configuration dependency.
- Resolve the stale official submission hash by locating the platform's immutable submission record or original uploaded file; do not infer from a filename or score table.

## Explicitly out of scope

- Do not retrain Phase 2.
- Do not submit a candidate.
- Do not optimize the blend against the historical champion.
- Do not alter the champion, V5 candidate, or Phase 2 frozen release.
