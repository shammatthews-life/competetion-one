# EXP85 Deep Research and Improvement

This is a new, separate evidence workspace for historical-lineage research and controlled development. It does not replace, modify, or certify the frozen Phase 2 candidate or historical EXP85 champion.

## Protected inputs

- Frozen Phase 2 release: `..\EXP85_PHASE2_FROZEN_RELEASE\`; candidate SHA-256 `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`.
- Correlation replication: `..\EXP85_SUBMISSION_CORRELATION_REPLICATION\`; pushed commit `0e1bcbea881f926b07aaa6916328cfd242684a51`.
- Historical champion: `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`; expected SHA-256 `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`.
- Raw train/test/sample inputs are external, read-only, and hash-checked from the frozen release.

All new files, diagnostics, and future run outputs belong here. Historical model caches are referenced by exact path and hash; they are not duplicated. Each run must use a unique directory. No test labels or historical predictions may be used for training or model selection.

## Current evidence-based conclusion

The reported `0.9999975816` V5/champion correlation is reproducible, but that V5 artifact combines a reconstructed historical Anchor74 with historical V17/V55/XGB10 prediction caches. It is a hybrid replay, not an independent raw-only rebuild. Later direct raw component executions are real raw-data runs, but their composed Track1B output had Spearman `0.8921288944` against the champion; the formula was not the main source of that discrepancy.

Phase 2 is the strongest validated independent raw-data candidate found so far: five-fold OOF AUC `0.7715614423` on seed 42, independently confirmed on seed 20261101, and preserved immutably. This is not an official score or a reconstruction of historical EXP85.

The first execution in this workspace was a **read-only, ID-aligned prediction-gap diagnostic**. Two fixed five-fold feature screens (category-peer aggregates and nested core-category encodings) have since completed on the exact saved seed-42 folds; both were rejected for lower OOF AUC than frozen Phase 2. Candidate predictions were regenerated from raw train/test data; historical predictions were not used for feature generation or model selection. See `EXPERIMENT_REGISTER.md`; no model is promoted without a positive screen and an independent-seed replication.

## Layout

- `HISTORICAL_PROJECT_INVENTORY.md`, `HISTORICAL_ARCHITECTURE.md`, `MODEL_LINEAGE.md`, `FEATURE_LINEAGE.md`: source-backed archaeology.
- `CORRELATION_GAP_ANALYSIS.md`: candidate comparisons and grouped residual diagnostics.
- `ROOT_CAUSE_AND_GAP_REPORT.md`: source-backed explanation of historical component divergence and notebook audit limitations.
- `CURRENT_STATE.md`, `RESEARCH_FINDINGS.md`, `OPPORTUNITY_RANKING.md`: verified position and prioritized methods.
- `EXPERIMENT_REGISTER.md`, `BEST_MODEL_REPORT.md`, `NEGATIVE_FINDINGS.md`, `NEXT_ACTIONS.md`: experiments, decisions and limits.
- `RESEARCH_REFERENCES.md`, `OPPORTUNITY_MATRIX.md`, `EXPERIMENT_ROADMAP.md`: external research and controlled next steps.
- `source/`: read-only inventory and diagnostic tools.
- `research/`: technique notes.
- `experiments/`: protocols and commands.
- `runs/`: unique immutable execution directories.
- `reports/`: human-readable results.
- `evidence/`: initial-state and provenance evidence.
- `tests/`: synthetic unit tests.
- `hashes/`: generated internal SHA-256 manifest.

See `STATUS.md` for completed work, active runs, blockers, and the distinction between historical fidelity and independent model development.
