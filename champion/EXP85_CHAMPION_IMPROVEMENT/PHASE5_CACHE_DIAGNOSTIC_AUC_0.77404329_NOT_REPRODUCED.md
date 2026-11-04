# Phase 5 cache diagnostic — AUC 0.77404329 (not reproduced)

**Status: diagnostic only. This is not a verified V16 result, champion score, or
valid improvement estimate.**

The value `0.77404329` is the AUC calculated by applying the saved fixed V16
blend formula to locally available candidate OOF caches. The cache files do not
include row IDs, fold assignments, or a provenance manifest binding them to the
executed V16 notebook and exact input data. The diagnostic is therefore not a
reproduction.

The saved V16 notebook output claims a fixed-formula AUC of `0.77649`. The
candidate-cache diagnostic is `0.00244671` lower, but the values cannot be
treated as an apples-to-apples comparison until the cache lineage is verified.

The final EXP85 champion comparison is blocked: its exact final OOF vector,
fold map, and Anchor74 out-of-fold counterpart were not recovered. No model was
trained for this audit, no improvement is claimed, and no champion or
submission was changed.

Detailed evidence, inventory, execution log, and captured terminal output are
preserved in the local recovery run:

`D:\gci\truth\experiments\phase5_champion_reproduction\runs\recovery_20261104T053622Z\`

The local candidate-cache score table is
`V16_CACHE_SCORE_COMPARISON.csv` in that run directory. The score names this
file for searchability; it does not upgrade the diagnostic's verification
status.
