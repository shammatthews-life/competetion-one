# GCI World 2026 Champion Archive: EXP85 Recovery & Autonomous Reconstruction

**Official Competition Evaluation Score: `0.801340` (Rank #12 worldwide, Top 0.44%)**  
**Repository**: `https://github.com/shammatthews-life/competetion-one` (Main Branch)  
**Target Submission**: `submission_exp85_recovery_v5.csv` & `submission_exp85_recovery_v5_rebuilt.csv`  

---

## Complete Project Structure

```text
champion/
├── README.md
├── MASTER_MANIFEST.json
├── MASTER_FILE_INVENTORY.csv
├── SHA256SUMS.txt
├── REPRODUCIBILITY_STATUS.md
├── COMPETITION_RULES_AUDIT.md
│
├── 00_HISTORICAL_REFERENCE/
│   ├── EXP85_HISTORICAL_CHAMPION.csv
│   ├── HISTORICAL_SHA256.txt
│   ├── HISTORICAL_SCORE.md
│   └── HISTORICAL_LINEAGE.md
│
├── 01_EXP85_RECOVERY/
│   ├── README.md
│   ├── notebooks/
│   ├── source/
│   ├── config/
│   ├── artifacts/
│   ├── cache/
│   ├── reports/
│   └── submissions/
│
├── 02_EXP85_REBUILT/
│   ├── README.md
│   ├── notebooks/
│   ├── source/
│   ├── config/
│   ├── features/
│   ├── models/
│   ├── artifacts/
│   ├── cache/
│   ├── reports/
│   └── submissions/
│
├── 03_PHASE_HISTORY/
│   ├── PHASE_1/
│   ├── PHASE_2/
│   ├── PHASE_3/
│   └── PHASE_4/
│
├── 04_MODEL_LINEAGE/
│   ├── FOUNDATION_MODELS/
│   ├── ANCHOR74/
│   ├── V17/
│   ├── V55/
│   ├── XGB10/
│   └── EXP85_SYNTHESIS/
│
├── 05_FEATURE_ENGINEERING/
│   ├── FEATURES_669/
│   ├── FEATURES_550/
│   ├── FEATURE_BUILDERS/
│   └── FEATURE_VALIDATION/
│
├── 06_REPRODUCIBILITY/
│   ├── REBUILD_INSTRUCTIONS.md
│   ├── DEPENDENCIES.txt
│   ├── ENVIRONMENT.json
│   ├── HASH_VALIDATION.csv
│   ├── PREDICTION_COMPARISON.csv
│   └── REPRODUCTION_TESTS.md
│
├── 07_COMPETITION_COMPLIANCE/
│   ├── OFFICIAL_RULES_SNAPSHOT.md
│   ├── DATA_POLICY_AUDIT.md
│   ├── VALIDATION_POLICY_AUDIT.md
│   ├── MODEL_AND_EXTERNAL_DATA_AUDIT.md
│   ├── SUBMISSION_FORMAT_AUDIT.md
│   ├── POTENTIAL_VIOLATIONS.md
│   └── FINAL_COMPLIANCE_VERDICT.md
│
├── 08_SUBMISSIONS/
│   ├── submission_exp85_recovery_v5.csv
│   ├── submission_exp85_recovery_v5_rebuilt.csv
│   ├── SUBMISSION_RESULTS.md
│   └── SUBMISSION_HASHES.txt
│
└── 09_EVIDENCE/
    ├── LEADERBOARD_EVIDENCE/
    ├── EXECUTION_LOGS/
    ├── HISTORICAL_COMPARISONS/
    └── SCREENSHOTS/
```

## Two Independent Verification Tracks
1. **Track 1 (Autonomous Source Reconstruction)**:
   - Full upstream regeneration from raw `train.csv` and `test.csv`.
   - Feature engineering (669 features) -> screening (550 features) -> 5-model zoo -> Anchor 74 pyramid -> 85/15 synthesis.
   - Spearman Correlation: **`0.9999975816`** | Leaderboard Score: **`0.801340`**.
2. **Track 2 (Exact Bit-for-Bit Dependency Synthesis)**:
   - Evaluates verified historical foundation artifacts via probit formula.
   - SHA-256: `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947` (Exact byte match).
   - Spearman Correlation: **`1.0000000000` (100.0%)** | Leaderboard Score: **`0.801340`**.
