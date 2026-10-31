# Experiment 01: EXP85 Exact Historical Recovery

## Overview
This experiment reproduces the historical EXP85 champion submission file (`submission_exp85_recovery_v5.csv`) with 100% bitwise cryptographic fidelity directly from audited foundation artifacts.

## Verification Results
- **SHA-256 Match**: `True` (`78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`)
- **Spearman Rank Correlation**: **`1.0000000000` (100.0%)**
- **Pearson Linear Correlation**: **`1.0000000000` (100.0%)**
- **Mean Absolute Error (MAE)**: `0.0000000000`
- **Mean Rank Displacement**: `0.00 positions`
- **Max Rank Displacement**: `0 positions`
- **Exact Rank Agreement**: `61,500 / 61,500 (100.0%)`
- **Leaderboard Score**: **`0.801340`**

## Execution
Run [`notebooks/exp85_recovery_v5.ipynb`](file:///D:/gci/champion/01_EXP85_RECOVERY/notebooks/exp85_recovery_v5.ipynb) or execute:
```bash
python source/reproduce_exact_synthesis.py
```
