# Upstream Dependency Graph

This graph separates the raw-generator path, the legacy submission path, and
the available V20/V080041 files. A line from a script to an artifact means the
source consumes or emits it only where explicitly described below; it does not
assert that every upstream model can be rebuilt.

```text
Raw train.csv + test.csv
  └─ 02_EXP85_REBUILT/source/run_all_recreations.py
       ├─ generate_xgb10_from_scratch.py ──> XGB10 OOF/test arrays
       ├─ generate_v55_from_scratch.py ────> V55 member archive
       ├─ generate_v17_from_scratch.py ────> V17 bundle
       └─ generate_anchor74_from_scratch.py -> Anchor74 CSV
          (tracked canonical manifest currently records ARTIFACTS_VERIFIED,
           not these four executions)

18 unavailable named inputs
  ├─ historical OOF/test bundles, metadata and feature matrix
  ├─ raw labels / test IDs (also derivable from raw CSVs)
  └─ consumed by gemini/final_submission/reproduce_final_submission.py
       └─ writes submission_final.csv
            └─ read and exported by solo_grand_master_apex_080.py
                 └─ v_solo_080041_pure_champion.csv

v_20_apex_grand_champion_submission.csv
  └─ consumed by build/evaluation/synthesis scripts;
     exact producer not found in the searched source/history
```

## Evidence basis

- The four raw generator names and six expected outputs are defined in
  `02_EXP85_REBUILT/source/run_all_recreations.py`.
- The existing canonical manifest is
  `02_EXP85_REBUILT/rebuilt_artifacts/RECONSTRUCTION_PROVENANCE_MANIFEST.json`
  and says `ARTIFACTS_VERIFIED`. It does not contain a clean-room subprocess
  ledger.
- `audit_upstream_lineage.py` parses literal `resolve_path(...)` references
  from the legacy submission source and searches the Gemini tree, clone,
  Downloads, and Documents locations for exact filenames.
- The resulting `MISSING_UPSTREAM_DEPENDENCIES.csv` records 18 missing inputs.
- The audit verified exact V080041/final-submission byte equality and ID
  order. The V20 file is present and ID-aligned, but its source mentions found
  in Python are consumer references, not a writer.

See [V20/V080041 lineage evidence](./V20_V080041_LINEAGE_REPORT.md) and
[clean-room provenance](./CLEAN_ROOM_PROVENANCE_REPORT.md).
