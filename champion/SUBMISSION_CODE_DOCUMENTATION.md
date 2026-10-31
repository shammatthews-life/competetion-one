# GCI World 2026: EXP85 Submission Creation Code Documentation

This document catalogs the exact creation scripts located directly in `D:\gci\` used to produce both leaderboard-winning submissions (**Score: 0.801340**).

---

## 1. Inventory of Creation Scripts in `D:\gci\`

| Script File | Target Submission Output | Leaderboard Score | Verification Track | Key Verification Metric |
| :--- | :--- | :--- | :--- | :--- |
| **`create_submission_recovery_v5.py`** | `submission_exp85_recovery_v5.csv` | **0.801340** | Track 2: Exact Historical Recovery | **Exact Bitwise Match** (`78c31bb4...`), Spearman = 1.0000000000 |
| **`create_submission_recovery_v5_rebuilt.py`** | `submission_exp85_recovery_v5_rebuilt.csv` | **0.801340** | Track 1: Independent Rebuilt Pipeline | Spearman = **0.9999975816** (99.9998%), Max Rank Shift = 769 |
| **`create_exp85_submissions.py`** | *Both Submissions* | **0.801340** | Unified Execution & Telemetry Suite | Programmatic parity and SHA-256 assertions |

---

## 2. Track Details & Mathematical Formulation

### Track 2: `submission_exp85_recovery_v5.csv`
- **Script**: [`D:\gci\create_submission_recovery_v5.py`](file:///D:/gci/create_submission_recovery_v5.py)
- **Inputs**:
  - Historical Anchor74: `v_20261027_precision_transductive_apex.csv`
  - Historical 8-Engine Meta-Suite: `v_17_full_suite_bundle.npz`, `v55_members.npz`, `test_10f_xgb.npy`
- **Weights**:
  - `v17_e1Aet`: 0.2932
  - `xgb10`: 0.2596
  - `v17_final`: 0.2051
  - `v55_cbprod`: 0.0874
  - `v17_e1et`: 0.0489
  - `v55_deep_et`: 0.0442
  - `v55_allfam`: 0.0383
  - `v55_v2_mono`: 0.0233
- **Synthesis**:
  $$\Phi(0.850 \cdot \text{Probit}(Anchor74) + 0.150 \cdot \text{Probit}(\text{Suite}))$$
- **Target SHA-256**: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`
- **Output Agreement**: 61,500 / 61,500 exact ordinal ranks (100.0%).

---

### Track 1: `submission_exp85_recovery_v5_rebuilt.csv`
- **Script**: [`D:\gci\create_submission_recovery_v5_rebuilt.py`](file:///D:/gci/create_submission_recovery_v5_rebuilt.py)
- **Inputs**:
  - Rebuilt Anchor74: `v_20261027_precision_transductive_apex.csv` (independently generated from 5-model zoo trained on screened 550 features)
  - Upgraded 8-Engine Meta-Suite with fixed historical weights
- **Synthesis**:
  $$\Phi(0.850 \cdot \text{Probit}(Anchor74_{\text{rebuilt}}) + 0.150 \cdot \text{Probit}(\text{Suite}))$$
- **Output SHA-256**: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`
- **Spearman Correlation**: `0.9999975816` (99.9998%)
- **Leaderboard Score**: `0.801340`

---

## 3. How to Run

To regenerate both files directly in `D:\gci\`:

```powershell
# 1. Generate Track 2 (Exact Historical Recovery)
python D:\gci\create_submission_recovery_v5.py

# 2. Generate Track 1 (Independent Upstream Rebuilt)
python D:\gci\create_submission_recovery_v5_rebuilt.py

# 3. Or run the unified master generator
python D:\gci\create_exp85_submissions.py
```
