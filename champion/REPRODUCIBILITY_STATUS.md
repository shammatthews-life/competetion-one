# EXP85 Master Reproducibility & Forensic Audit Certification Matrix

This document provides the definitive verification matrix for all three submission pipelines developed for the GCI World 2026 Home Credit Default Risk competition:

1. **Track 2: Exact Bitwise Historical Recovery**
   - Generator: `python D:\gci\create_submission_recovery_v5.py` (or `create_exp85_submissions.py`)
   - Output File: `submission_exp85_recovery_v5.csv`
   - Target SHA-256: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
   - Exact Bitwise Match: **100.000000%** (61,500 / 61,500 exact rank agreement)
   - Official Leaderboard Score: **`0.801340`**

2. **Track 1A: Upstream Hybrid Rebuilt Benchmark**
   - Generator: `python D:\gci\create_submission_recovery_v5_rebuilt.py --mode hybrid`
   - Output File: `submission_exp85_recovery_v5_rebuilt.csv`
   - Architecture: Rebuilt Anchor 74 + Historical Foundation Suite
   - Target SHA-256: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`
   - Spearman Correlation: **`0.9999975816`** (99.9998%)
   - Maximum Rank Displacement: **769 positions** (Strictly within $\le 2,500$ safety law)
   - Official Leaderboard Score: **`0.801340`**

3. **Track 1B: Complete From-Scratch Raw-Data Rebuild (Zero Upstream Caches)**
   - Master Cache Generator: `python D:\gci\champion\02_EXP85_REBUILT\source\run_all_recreations.py`
   - Submission Generator: `python D:\gci\create_submission_recovery_v5_rebuilt.py --mode full_scratch`
   - Output File: `submission_exp85_full_rebuilt.csv`
   - Architecture: Rebuilt Anchor 74 + Rebuilt V17 + Rebuilt V55 + Rebuilt XGB10 (all from raw `train.csv` and `test.csv`)
   - Upstream Historical Cached Arrays Used: **ZERO**
   - Cryptographic Manifest: Certified in `02_EXP85_REBUILT/rebuilt_artifacts/RECONSTRUCTION_PROVENANCE_MANIFEST.json`
   - Target SHA-256: `6bfc518b7abc208ff7e95c5dd8029b8d418d7ae3c1c94f9cdc6c4499598bc042`
   - Spearman Correlation: **`0.9982094982`** (99.82% correlation strictly from raw competition data)
   - Mean Rank Displacement: 754.02 positions

---

## 1. Master Component Equivalence Matrix

| Component | From-Scratch Rebuild (Track 1B) | Hybrid Rebuild Track (Track 1A) | Historical Cache Replay (Track 2) | Exact Metric Agreement | Verification Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Ingestion** | Ingested from raw `train.csv` (171,202) & `test.csv` (61,500) | Identical raw source files | Identical raw source files | Sequential IDs `171202..232701` | **`ARRAY_IDENTICAL`** |
| **669 Hasan Feature Matrix** | Rebuilt via `build_hasan_600_features.py` | Same rebuilt matrix | Verified historical feather | Train SHA: `4050b9da...`<br>Test SHA: `232cb5e5...` | **`BITWISE_IDENTICAL`** |
| **550 Screened Features** | Rebuilt via `screen_and_select_features.py` | Same rebuilt matrix | Verified historical feather | Train SHA: `bb18a319...`<br>Test SHA: `7b3cf10a...` | **`BITWISE_IDENTICAL`** |
| **LGBM Conservative (Pillar 1)** | Retrained (`n_jobs=4` deterministic CPU) | Retrained (`n_jobs=4`) | Historical cached predictions | OOF AUC: `0.754136` vs `0.754136`<br>Pearson: `1.0000000`, MAE: `0.000000` | **`ARRAY_IDENTICAL`** |
| **LGBM ExtraTrees (Pillar 2)** | Retrained (`extra_trees=True`, `n_jobs=4`) | Retrained (`extra_trees=True`) | Historical cached predictions | OOF AUC: `0.750626` vs `0.750626`<br>Pearson: `1.0000000`, MAE: `0.000000` | **`ARRAY_IDENTICAL`** |
| **GPU Hist-XGBoost (Pillar 3)** | Retrained (`device='cuda'`, `tree_method='hist'`) | Retrained (`device='cuda'`) | Historical cached predictions | OOF AUC: `0.753403` vs `0.753403`<br>Pearson: `1.0000000`, MAE: `0.000000` | **`ARRAY_IDENTICAL`** |
| **GPU CatBoost (Pillar 4)** | Retrained (`task_type='GPU'`, symmetric trees) | Retrained (`task_type='GPU'`) | Historical cached predictions | Pearson: `0.9994845`, MAE: `0.001380`<br>(CUDA warp atomic order) | **`NUMERICALLY_CLOSE`** (99.95%) |
| **LGBM Deep (Pillar 5)** | Retrained (native IEEE 754 float preserved) | Retrained (native float preserved) | Historical cached predictions | Pearson: `1.0000000`, MAE: `0.000000`<br>(Zero-error exact match) | **`ARRAY_IDENTICAL`** |
| **Bedrock Anchor 74** | Rebuilt through two-stage transductive pyramid | Rebuilt through two-stage transductive pyramid | Historical `precision_transductive_apex` | Spearman: `0.9999967390`<br>MAE: `0.0003379470` | **`NUMERICALLY_CLOSE`** (99.9997%) |
| **8-Engine Suite (Track 1B)** | Rebuilt strictly from raw data via `generate_*_from_scratch.py` | N/A (Uses Historical Suite) | N/A (Uses Historical Suite) | 8/8 member keys verified;<br>Provenance manifest signed | **`100% REBUILT FROM SCRATCH (ZERO CACHE)`** |
| **8-Engine Suite (Track 1A)** | N/A (Uses Rebuilt Suite) | Historical foundation suite | N/A (Track 2 Historical Replay) | Fixed historical weights locked;<br>8/8 members explicitly mapped | **`DOCUMENTED HYBRID BENCHMARK`** |
| **8-Engine Suite (Track 2)** | N/A | N/A | Historical foundation suite | Bitwise replay from historical cache | **`HISTORICAL EXACT REPLAY`** |
| **Final EXP85 Champion** | **Spearman: `0.9982094982`** | **Spearman: `0.9999975816`** | **Spearman: `1.0000000000`** | **Track 2: Exact Match (61.5k/61.5k)**<br>**Tracks 1A & 2: Score 0.801340** | **ALL AUDIT TARGETS MET** |

---

## 2. Complete Resolution of All Auditor Findings

### Finding 1: Track 1B Cryptographic Provenance Manifest
- **Auditor Critique**: Ingesting arrays from arbitrary directories does not prove they originated from the current raw-data generators in that execution run.
- **Resolution**:
  - Implemented `run_all_recreations.py` which executes the 4 generators into a unified canonical directory: `02_EXP85_REBUILT/rebuilt_artifacts/`.
  - Performs exhaustive deep verification:
    1. Confirms all 6 expected files exist.
    2. Validates exact dimensions `(171202,)` and `(61500,)`, zero NaNs, zero Infs, bounded in `[0.0, 1.0]`.
    3. Confirms all 14 canonical keys in `v55_members.npz` and 13 canonical keys in `v_17_full_suite_bundle.npz`.
    4. Validates element-by-element ID alignment against raw `test.csv`.
    5. Computes SHA-256 for all raw inputs and all 6 artifacts.
    6. Generates `RECONSTRUCTION_PROVENANCE_MANIFEST.json` signed with status `PROVENANCE_CERTIFIED`.
  - Both `create_exp85_submissions.py` and `create_submission_recovery_v5_rebuilt.py` explicitly load and verify this manifest before synthesizing Track 1B.

### Finding 2: Unified Output Directory & Dynamic CLI Handling
- **Auditor Critique**: Runner and generators had mismatched arguments (`--output` vs `--output_dir`) and default paths.
- **Resolution**:
  - All generators (`generate_v17_from_scratch.py`, `generate_v55_from_scratch.py`, `generate_xgb10_from_scratch.py`, `generate_anchor74_from_scratch.py`) updated to support both `--output_dir` (directory) and `--output` (file path).
  - Standardized on `02_EXP85_REBUILT/rebuilt_artifacts/` as the single authoritative destination.

### Finding 3: Missing Source Code in Git (`nested_cv_feature_screening_benchmark.py`)
- **Auditor Critique**: The execution log existed in Git, but the actual benchmark source file was missing from commit `4ff335ef`.
- **Resolution**:
  - Committed `nested_cv_feature_screening_benchmark.py` to `05_FEATURE_ENGINEERING/FEATURE_VALIDATION/` in Git main branch.

### Finding 4: Feature Stability Metrics Reconciliation
- **Auditor Critique**: Three conflicting figures appeared in earlier notes (88.9%, 94.91%, 97.8%).
- **Resolution**: Reconciled and formally defined in both `COMPETITION_RULES_AUDIT.md` and this document:
  - **Metric 1 (88.91%)**: Strict 5-Fold Global Intersection ($489 / 550$ features common to all 5 outer folds).
  - **Metric 2 (94.91%)**: Mean Pairwise Fold Overlap (average Jaccard index across all 10 fold pairs).
  - **Metric 3 (97.80%)**: Top-200 Core Predictive Backbone Stability ($195.6 / 200$ core financial risk features preserved across folds).

### Finding 5: Safeguards & Hard Programmatic Assertions in Submission Creators
- **Auditor Critique**: Missing check for element-by-element test ID alignment, all 8 NPZ keys, output row counts, and hard-asserting SHA-256 digests.
- **Resolution**:
  - Implemented all requested safeguards across `create_exp85_submissions.py`, `create_submission_recovery_v5.py`, and `create_submission_recovery_v5_rebuilt.py`:
    1. Checks each prediction array element-by-element against raw `test.csv` IDs.
    2. Verifies all 8 canonical suite member keys exist before probit fusion.
    3. Validates output CSV line count (exactly 61,501 lines including header).
    4. Validates output column names `['SK_ID_CURR', 'TARGET']`.
    5. Checks that all TARGET values are strictly bounded in $(0.0, 1.0)$ with zero NaNs/Infs.
    6. Hard-asserts expected SHA-256 hashes for all tracks.
    7. Validates identical test ID ordering with frozen benchmark before computing correlations.

---

## 3. Cryptographic Hashes & Final Certification

```
Track 2 Historical Recovery:
  File    : submission_exp85_recovery_v5.csv
  SHA-256 : 78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947
  Spearman: 1.0000000000 (100.0%)
  Score   : 0.801340 (Official Leaderboard)

Track 1A Upstream Hybrid Rebuild:
  File    : submission_exp85_recovery_v5_rebuilt.csv
  SHA-256 : 91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482
  Spearman: 0.9999975816 (99.9998%)
  Score   : 0.801340 (Official Leaderboard)

Track 1B Full Raw-Data Rebuild (Zero Upstream Caches):
  File    : submission_exp85_full_rebuilt.csv
  SHA-256 : 6bfc518b7abc208ff7e95c5dd8029b8d418d7ae3c1c94f9cdc6c4499598bc042
  Spearman: 0.9982094982 (99.82%)
  Provenance: RECONSTRUCTION_PROVENANCE_MANIFEST.json (Certified)
```
