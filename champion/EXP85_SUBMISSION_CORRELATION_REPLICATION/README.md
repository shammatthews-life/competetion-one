# EXP85 Submission Correlation Replication

This project independently audits the historical high-correlation result without training models or modifying any historical file. The validated Phase 2 candidate is separately preserved in `../EXP85_PHASE2_FROZEN_RELEASE/` and is not modified here.

## Result in brief

The reported near-`0.99999` result is reproduced for the exact historical pair:

- Reference: `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`
- Reference SHA-256: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
- Candidate: `D:\gci\champion\08_SUBMISSIONS\submission_exp85_recovery_v5_rebuilt.csv`
- Candidate SHA-256: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`
- Reproduced Pearson: `0.9999975815867518`
- Reproduced Spearman: `0.9999975815867501`
- Aligned rows: 61,500; duplicates, missing IDs, extra IDs: all zero.

The original notebook calculated metrics positionally without an explicit ID join. A forensic audit confirms that both files use the exact same raw test ID set and row order. The new ID-aligned runner produces the same result. The historical claim is therefore reproduced numerically, but its label as an independent raw rebuild is not supported: the recipe uses a reconstructed Phase 3 Anchor74 together with historical cached V17, V55, and XGB10 suite predictions. It is a **hybrid historical-cache replay**, not a full raw-only rebuild.

Other near-identical-looking figures are from different candidates. Exact identities and reproduced metrics are in `REFERENCE_INVENTORY.md` and `reports/REPLICATION_RESULTS.md`.

No official score is inferred from correlation. The official historical champion score is context only; none of these local correlation runs submitted an artifact.

## Historical V5 reconstruction

The original notebook's eight-member suite and 85/15 probit/ordinal-rank merge are now executable through `source/reconstruct_historical_v5.py`. Run `20261101T151558Z_historical_cache_v5` reproduced the previous V5 candidate byte-for-byte:

- Candidate: `runs/20261101T151558Z_historical_cache_v5/submission_historical_cache_v5.csv`
- SHA-256: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`
- Pearson / Spearman against the historical champion: `0.9999975815867518` / `0.9999975815867501`
- Exact candidate bytes and values matched the pinned previous V5 artifact; 61,500 IDs, order, schema, uniqueness, finite values, and probability bounds passed.

The replay preserves the original `float32` arithmetic through rank normalization and probit for float32 cache members; promoting these arrays early changes low-order values and can change the final ordinal ordering. Run inputs, hashes, dependency graph, metrics, and command exit code are recorded in the run's JSON, Markdown, and manifest files. The earlier failed replay remains preserved as a separate run.

## Contents

- `source/`: original notebook/audit/source snapshots, the fail-closed comparison runner, and the historical V5 reconstruction runner.
- `reference_artifacts/` and `candidate_artifacts/`: byte-verified copies of the exact comparison inputs.
- `runs/`: unique, non-overwriting result directories with JSON/Markdown output and execution provenance.
- `evidence/`: prior forensic report and its machine-readable evidence.
- `experiments/`: reproducible commands for known candidate comparisons.
- `tests/`: synthetic alignment, error, and metric tests.
- `hashes/`: generated internal SHA-256 manifest.

The folder's `.gitattributes` disables line-ending conversion so the byte-level evidence and SHA-256 manifests remain valid across checkouts.

## Reproduce the primary result

From the repository root, using the copies in this folder:

```powershell
python -B champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\correlation_runner.py `
  --reference champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\reference_artifacts\historical_champion.csv `
  --candidate champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\candidate_artifacts\track1_v5_rebuilt.csv `
  --run-dir champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\runs\YOUR_UNIQUE_RUN_ID `
  --method notebook_stage9 `
  --reference-source-path D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv `
  --candidate-source-path D:\gci\champion\08_SUBMISSIONS\submission_exp85_recovery_v5_rebuilt.csv `
  --lineage-note "Hybrid replay; historical V17/V55/XGB10 cached predictions are used."
```

`notebook_stage9` reproduces the notebook's raw-vector comparison and ordinal rank displacement, but fails closed unless candidate and reference IDs are unique, equal, and in identical order. The default `id_aligned` method joins by ID and uses average ranks for displacement. Both methods enforce exactly 61,500 rows by default; `--expected-rows` is available for synthetic/test inputs. A run directory must not already exist.

Run the replication and reconstruction tests (17 pass):

```powershell
python -B -m unittest discover -s champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\tests -p "test_*.py"
```

Run the historical-cache reconstruction from the repository root. This writes to a new run directory and fails if that directory already exists:

```powershell
python -B champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\reconstruct_historical_v5.py
```

Run the preserved original auditor's synthetic suite:

```powershell
python -B champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\original_isolated_audit\test_prediction_correlations.py
```

Verify the complete replication artifact inventory:

```powershell
python -B champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\verify_replication.py
```

After adding or rerunning evidence, regenerate both manifests with:

```powershell
python -B champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\build_manifests.py
```

## Current conclusion

The `0.9999975816` result is genuinely reproduced against the exact historical champion reference, to the originally reported ten decimal places. It is not evidence that all model components were independently regenerated, that predictions are equal, or that the candidate's official score was 0.801340. The exact cached-component lineage and distinct neighboring metrics are documented; unresolved provenance is not filled by substitution.
