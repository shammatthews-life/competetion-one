# EXP85 Phase 2 Frozen Release

**Release label:** `FROZEN_DEVELOPMENT_CANDIDATE`

This folder preserves the independently trained Phase 2 development candidate and the evidence supporting its validation gate. It is an archival snapshot, not an official competition submission or a claim that the historical EXP85 score was reproduced.

## Frozen artifact

- Candidate: `CANDIDATE/submission_key_target_encoding.csv`
- SHA-256: `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`
- OOF AUC (original five-fold seed 42): `0.7715614423106346`
- Matched raw baseline OOF AUC: `0.7625457907047022`
- Second-seed gate: PASS
- Official competition score: NOT EVALUATED

The original candidate in the working project is expected at:
`D:\gci\_gci_push_tmp\champion\FINAL_RESEARCH_AND_OPTIMIZATION\runs\phase02_key_te_20261101_181500\submission_key_target_encoding.csv`.
The original and archived candidate must remain byte-identical.

## Layout

- `SOURCE/project/`: original relative source layout, source scripts, unit tests, and minimal raw-only Phase 0/1 OOF prerequisites used by the Phase 2 experiment.
- `CANDIDATE/`: frozen Phase 2, Phase 1, and baseline submissions.
- `REPORTS/`: development, ablation, leakage, validation, and final gate reports.
- `EVIDENCE/`: run summaries, predictions, fold definitions, manifests, and execution evidence; model checkpoints are intentionally excluded.
- `HASHES/`: input hashes and the generated freeze manifest.
- `REPRODUCTION/`: integrity verifier, manifest generator, and verifier tests.

Raw input datasets are deliberately not included. Expected inputs are documented with SHA-256 values in `HASHES/INPUT_HASHES.json`.

## Verify the freeze

From the repository root, run:

```powershell
python -B champion\EXP85_PHASE2_FROZEN_RELEASE\REPRODUCTION\verify_freeze.py
```

The verifier checks the internal manifest, candidate hash, original candidate hash, input hashes, test/sample ID ordering, duplicate/missing/extra IDs, schema, row count, and finite probability range. If a required original input or source candidate is unavailable, it reports that exact path and exits nonzero; it never silently treats missing evidence as a pass.

To rerun the verifier tests:

```powershell
python -B -m unittest discover -s champion\EXP85_PHASE2_FROZEN_RELEASE\REPRODUCTION -p "test_*.py"
```

## Reproduce model development

The preserved scripts are laid out under `SOURCE/project/` using their original relative directory structure. The original model-development sequence was baseline (five folds, seed 42), Phase 1, then Phase 2 (nested target encodings, inner seed 7). The Phase 1/2 scripts compare against raw-only OOF artifacts; the minimal OOF prerequisites are included under `SOURCE/project/runs/`.

Full retraining requires the original Python dependencies, the raw CSVs, and an available supported GPU for CatBoost. It is not part of the integrity verification command and must write to a separate disposable workspace/run directory; never run it over this release or the frozen source candidate. The original final second-seed verification is in `SOURCE/project/experiments/phase_04_final_validation_gate/`; its saved runs and independent evidence are in `EVIDENCE/second_seed/`.

## Integrity and change control

`FREEZE_MANIFEST.json` covers every file in this release except itself. The manifest generator and verifier are included in the release. Historical EXP85 artifacts, the official-score claim, and the original Phase 2 source candidate are outside the scope of any edits made here.

See `REPORTS/FREEZE_REPORT.md` for freeze provenance, validation metrics, commit details, and limitations; see `MODEL_CARD.md` for intended use and risk boundaries.
