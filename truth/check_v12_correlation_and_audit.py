"""
================================================================================
GCI WORLD 2026: V12 CORRELATION & FIDELITY AUDIT TOOL
================================================================================
Audits rank correlation and compliance for v_22_pure_champion_ensemble_certified_v12.csv.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

V12_PATH = r"d:/gci/v_22_pure_champion_ensemble_certified_v12.csv"
SAMPLE_PATH = r"d:/gci/inpuy/sample_submission.csv"

assert os.path.exists(V12_PATH), f"Target submission not found: {V12_PATH}"
sub12 = pd.read_csv(V12_PATH)
sample = pd.read_csv(SAMPLE_PATH)

print("=" * 80)
print("AUDIT & VERIFICATION: v_22_pure_champion_ensemble_certified_v12.csv")
print("=" * 80)

# Schema Assertions
assert len(sub12) == 61500, f"Expected 61,500 rows, got {len(sub12)}"
assert sub12["SK_ID_CURR"].is_unique, "Duplicate SK_ID_CURR detected!"
assert sub12["SK_ID_CURR"].equals(sample["SK_ID_CURR"]), "ID alignment mismatch with sample_submission.csv!"
assert not sub12["TARGET"].isna().any(), "NaN values detected in predictions!"
assert np.isfinite(sub12["TARGET"].values).all(), "Non-finite values detected!"
assert (sub12["TARGET"] >= 0.0).all() and (sub12["TARGET"] <= 1.0).all(), "Predictions outside [0, 1] bounds!"

sha = hashlib.sha256(open(V12_PATH, "rb").read()).hexdigest().upper()
print(f"[PASS] Exact Schema Verified: 61,500 rows, unique IDs, finite in [0, 1]")
print(f"[PROVENANCE] SHA-256: {sha}\n")

# Rank Correlation Ledger
BENCHMARKS = [
    ("LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv (LB 0.78239)", r"d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv"),
    ("LEADERBOARD_SOLO_CHAMPION_V23.csv (Solo Champion)", r"d:/gci/LEADERBOARD_SOLO_CHAMPION_V23.csv"),
    ("v_22_pure_champion_ensemble_certified_v11.csv (V11 Pure)", r"d:/gci/v_22_pure_champion_ensemble_certified_v11.csv"),
    ("v_22_pure_champion_ensemble_certified_v8.csv (V8)", r"d:/gci/v_22_pure_champion_ensemble_certified_v8.csv")
]

print("-" * 80)
print(f"{'Benchmark Target':55s} | {'Spearman Rho':12s} | {'Pearson r':10s}")
print("-" * 80)

for name, path in BENCHMARKS:
    if os.path.exists(path):
        b_df = pd.read_csv(path)
        rho, _ = spearmanr(sub12["TARGET"], b_df["TARGET"])
        r, _ = pearsonr(sub12["TARGET"], b_df["TARGET"])
        print(f"{name:55s} | {rho*100:10.4f}%   | {r*100:8.4f}%")
print("-" * 80)

