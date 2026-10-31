# Raw-Rebuild Verdict

## Verdict: not certified

The available evidence does not justify `RAW_REBUILD_CERTIFIED=true`.
Classification is **not certified with currently verifiable execution
provenance**; this is not a claim that the historical build never ran, only
that its claimed clean-room evidence is not present at the recorded location
or in Git history.

## Basis

1. The tracked canonical manifest is `ARTIFACTS_VERIFIED`, not a generator
   execution record.
2. The older report's claimed `clean_room_build` manifest is absent from the
   repository tree and Git history.
3. The current Anchor74 generator's 143-feature construction is not the
   historical 669/550 feature path described by the older reports.
4. The 0.9999975816 Track 1A result reuses historical cached suite predictions;
   it is a hybrid benchmark, not evidence that every model was reconstructed
   from raw data.
5. Track 1B's 0.9982094982 candidate is a valid candidate result, but score
   similarity alone does not establish execution lineage.
6. V20 lacks a located writer; V080041 is a copy/export of `submission_final`,
   whose 18 upstream inputs are unavailable in the searched roots.
7. CatBoost's five new GPU repeats are stable and historical-like, but do not
   resolve the earlier split or establish its cause.

## What this verdict does not say

- It does not invalidate the historical hybrid benchmark or the candidate CSV.
- It does not claim that raw-trained generators cannot produce useful models.
- It does not infer a root cause from correlation or feature count alone.
- It does not authorize relabeling historical caches as raw-generated.

No full retraining, blend tuning, or overwrite of the frozen champion was
performed. Reclassification requires a newly recorded full-mode run with
current source hashes, raw input hashes, command/exit/runtime logs, and all
expected outputs verified against raw row IDs.
