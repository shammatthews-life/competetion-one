# Phase 2 Freeze Report

## Status

**FROZEN_DEVELOPMENT_CANDIDATE**

This label means the selected independent development candidate and its validation evidence have been archived and integrity-checked. It does not mean an official competition submission was made, the historical EXP85 model lineage was recovered, or the official score was reproduced.

## Provenance

- Source repository: `https://github.com/shammatthews-life/competetion-one`
- Source commit before freeze: `db867d015d522e49aa895eff2630a71e7a4525a7`
- Freeze release commit: `6cd8bd8d86e066e27d8d59395c2cccfb564e5659`
- GitHub release commit: `https://github.com/shammatthews-life/competetion-one/commit/6cd8bd8d86e066e27d8d59395c2cccfb564e5659`
- Candidate source path: `D:\gci\_gci_push_tmp\champion\FINAL_RESEARCH_AND_OPTIMIZATION\runs\phase02_key_te_20261101_181500\submission_key_target_encoding.csv`
- Frozen path: `CANDIDATE/submission_key_target_encoding.csv`
- Candidate SHA-256: `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`
- Row count: 61,500; columns: `SK_ID_CURR,TARGET`
- Freeze creation timestamp: recorded in `FREEZE_MANIFEST.json`.
- Freeze verification: **PASS**; 129 internal release files verified, candidate/source byte identity confirmed, all three raw input hashes matched, and IDs/schema/range checks passed.
- Preserved model and gate unit tests: **18 passed**; freeze verifier synthetic tests: **4 passed**.

The frozen candidate was copied without transformation. The included verifier hashes both the archive copy and the original source path and requires byte identity.

## Model validation

Original five-fold seed 42:

- Raw baseline OOF AUC: `0.7625457907047022`.
- Phase 2 OOF AUC: `0.7715614423106346`.
- Matched lift: `+0.0090156516059324`.

Independent outer seed 20261101:

- Matched raw baseline OOF AUC: `0.7620389005999739`.
- Pooled-frequency Phase 2 OOF AUC: `0.7711413866258005`; lift `+0.0091024860258265`; 5/5 folds positive.
- Fold-local Phase 2 OOF AUC: `0.7709638203765881`; lift `+0.0089249197766141`; 5/5 folds positive.
- Fold-local decoder versus disabled control: `+0.0031900391533038` OOF AUC; 5/5 folds positive.
- Unit tests: 18 passed.
- Independent gate verification: PASS.

All metrics are development cross-validation, not official leaderboard results.

## Integrity verification

Run the exact verification command from the repository root:

```powershell
python -B champion\EXP85_PHASE2_FROZEN_RELEASE\REPRODUCTION\verify_freeze.py
```

It verifies every release file against `FREEZE_MANIFEST.json`; checks that required release files are present and no unmanifested release files have appeared; hashes the original and archived candidate; checks all three raw input hashes; validates the candidate header, 61,500 rows, unique IDs, test/sample ID set and exact order, finite values, and `[0, 1]` range. Missing paths or mismatches are reported as errors and result in a nonzero exit.

Input hashes are listed in `HASHES/INPUT_HASHES.json`. The raw train/test/sample CSVs are external and intentionally not committed.

## Reproduction scope and limitations

The original source scripts, their relative directory layout, tests, baseline and Phase 1 OOF prerequisites, final gate source, fold definitions, saved predictions, reports, and execution evidence are archived. The Phase 1/2 historical comparison dependencies are raw-only outputs from the documented baseline and Phase 1 experiments; the Phase 2 training lineage reports no historical prediction inputs.

Full retraining needs compatible Python packages, raw input CSVs, and GPU support for CatBoost. Model checkpoints and raw input data are intentionally excluded. Re-running training must use a separate workspace and unique run directories. Never overwrite this release or the original candidate.

The selected candidate uses pooled unlabeled train/test fee-frequency statistics; a fold-local sensitivity test is preserved and performed separately. The user has not authorized submission. Official competition score remains `NOT_EVALUATED`; historical EXP85 `0.801340` is not claimed.
