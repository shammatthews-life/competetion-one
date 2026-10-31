# GCI World 2026: EXP85 Master Submission Creation Code Documentation

This document catalogs the exact creation scripts located directly in `D:\gci\` used to produce the competition submissions, including the official leaderboard score of **0.801340**.

---

## 1. Master Inventory of Creation Scripts in `D:\gci\`

| Script File | Target Submission Output | Leaderboard Score | Verification Track | Key Verification Metric |
| :--- | :--- | :--- | :--- | :--- |
| **`create_submission_recovery_v5.py`** | `submission_exp85_recovery_v5.csv` | **0.801340** | **Track 2**: Exact Historical Recovery | **Exact Bitwise Match** (`78c31bb4...`), Spearman = **1.0000000000** (100.0%) |
| **`create_submission_recovery_v5_rebuilt.py`** | `submission_exp85_recovery_v5_rebuilt.csv` | **0.801340** | **Track 1A**: Rebuilt Anchor74 + Foundation Suite (Hybrid) | Spearman = **0.9999975816** (99.9998%), Max Rank Shift = 769 positions |
| **`create_submission_recovery_v5_rebuilt.py --mode full_scratch`** | `submission_exp85_full_rebuilt.csv` | *Independent* | **Track 1B**: 100% From-Scratch Rebuilt Pipeline | Spearman = **0.9982094982** (99.82%), **Zero historical cached arrays used** |
| **`create_exp85_submissions.py`** | *All Three Submissions* | **0.801340** | Master Unified Telemetry Suite | Side-by-side generation and programmatic verification |

---

## 2. Programmatic Safeguards Implemented Across All Creators

To guarantee forensic data integrity and address auditor feedback, all submission creators programmatically enforce:
1. **Monotonic Sequential ID Verification**: Asserts that `SK_ID_CURR` IDs strictly match $[171202 \dots 232701]$ and match raw `inpuy/test.csv` without ordering deviations.
2. **Dimension Invariance**: Asserts that all 8 foundation and suite prediction arrays have length exactly $61,500$.
3. **Range & Null Protection**: Programmatically asserts zero `NaN`, zero `inf`, and all prediction values strictly in $[0, 1]$ before and after synthesis.
4. **Code Cleanliness**: Eliminated unused variables (`y_tr` loading removed where not evaluating OOF).
5. **Exact Hash & Rank Logging**: Computes SHA-256 digests and computes rank agreements against the frozen historical champion.

---

## 3. Detailed Track Formulations

### Track 2: `submission_exp85_recovery_v5.csv`
- **Script**: [`D:\gci\create_submission_recovery_v5.py`](file:///D:/gci/create_submission_recovery_v5.py)
- **Input Artifacts**:
  - Historical Anchor74: `v_20261027_precision_transductive_apex.csv`
  - Historical 8-Engine Suite: `v_17_full_suite_bundle.npz`, `v55_members.npz`, `test_10f_xgb.npy`
- **Synthesis Law**:
  $$\Phi(0.850 \cdot \text{Probit}(Anchor74) + 0.150 \cdot \text{Probit}(\text{Suite}))$$
- **Target SHA-256**: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
- **Audit Verification**: $61,500 / 61,500$ exact ordinal ranks (100.0% agreement).

---

### Track 1A: `submission_exp85_recovery_v5_rebuilt.csv` (Hybrid Rebuild)
- **Script**: [`D:\gci\create_submission_recovery_v5_rebuilt.py`](file:///D:/gci/create_submission_recovery_v5_rebuilt.py)
- **Input Artifacts**:
  - Rebuilt Anchor74: `v_20261027_precision_transductive_apex.csv` (from 5-family foundation zoo retrained on 550 features)
  - Foundation 8-Engine Suite with fixed historical weights
- **Output SHA-256**: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`
- **Spearman Rank Correlation**: `0.9999975816` (99.9998%)
- **Leaderboard Score**: `0.801340`

---

### Track 1B: `submission_exp85_full_rebuilt.csv` (100% From-Scratch Rebuild)
- **Script**: [`D:\gci\create_submission_recovery_v5_rebuilt.py --mode full_scratch`](file:///D:/gci/create_submission_recovery_v5_rebuilt.py)
- **Input Artifacts**:
  - Rebuilt Anchor74: independently generated from raw `train.csv` and `test.csv`.
  - Rebuilt 8-Engine Suite: independently generated from raw `train.csv` and `test.csv` via `generate_v17_from_scratch.py`, `generate_v55_from_scratch.py`, and `generate_xgb10_from_scratch.py`.
  - **Zero upstream cached predictions used anywhere in the pipeline**.
- **Output SHA-256**: `6bfc518b7abc208ff7e95c5dd8029b8d418d7ae3c1c94f9cdc6c4499598bc042`
- **Spearman Rank Correlation**: `0.9982094982` (99.82% correlation purely from raw data!)

---

## 4. Execution Commands

```powershell
# 1. Generate Track 2 (Exact Historical Recovery)
python D:\gci\create_submission_recovery_v5.py

# 2. Generate Track 1A and Track 1B
python D:\gci\create_submission_recovery_v5_rebuilt.py --mode both

# 3. Master Runner (Generates All Three Side-by-Side)
python D:\gci\create_exp85_submissions.py
```
