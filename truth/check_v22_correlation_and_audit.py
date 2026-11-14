"""
================================================================================
GCI WORLD 2026: DIAGNOSTIC AUDIT & CORRELATION BENCHMARK FOR V22
================================================================================
Compares v_22_pure_champion_ensemble_certified_v22.csv against:
  1. LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv (0.78239 LB Benchmark)
  2. submission_4_v22_champion_078239.csv
  3. LEADERBOARD_SOLO_CHAMPION_V23.csv (0.78200 Benchmark)
  4. v_22_pure_champion_ensemble_certified_v11.csv
  5. v_22_pure_champion_ensemble_certified_v14.csv
  6. v_22_pure_champion_ensemble_certified_v16.csv
  7. v_22_pure_champion_ensemble_certified_v17.csv
  8. v_22_pure_champion_ensemble_certified_v18.csv
  9. v_22_pure_champion_ensemble_certified_v19.csv
  10. v_22_pure_champion_ensemble_certified_v20.csv
  11. v_22_pure_champion_ensemble_certified_v21.csv
"""

import os
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

v22_path = r"d:/gci/v_22_pure_champion_ensemble_certified_v22.csv"
if not os.path.exists(v22_path):
    print(f"Artifact not found yet: {v22_path}")
    exit(0)

v22_df = pd.read_csv(v22_path)
v22_preds = v22_df["TARGET"].values
v22_sha = hashlib.sha256(open(v22_path, "rb").read()).hexdigest().upper()

print("=" * 80)
print("V22 SUBMISSION ARTIFACT INTEGRITY CHECK")
print("=" * 80)
print(f"File Path: {v22_path}")
print(f"Row Count: {len(v22_df):,}")
print(f"Unique IDs: {v22_df['SK_ID_CURR'].nunique():,}")
print(f"Missing Values: {v22_df.isna().sum().sum()}")
print(f"Finite Values: {np.isfinite(v22_preds).all()}")
print(f"Value Range: [{v22_preds.min():.6f}, {v22_preds.max():.6f}]")
print(f"SHA-256 Hash: {v22_sha}")

benchmarks = [
    ("LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv", r"d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv"),
    ("submission_4_v22_champion_078239.csv", r"d:/gci/submission_4_v22_champion_078239.csv"),
    ("LEADERBOARD_SOLO_CHAMPION_V23.csv", r"d:/gci/LEADERBOARD_SOLO_CHAMPION_V23.csv"),
    ("v_22_pure_champion_ensemble_certified_v11.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v11.csv"),
    ("v_22_pure_champion_ensemble_certified_v14.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v14.csv"),
    ("v_22_pure_champion_ensemble_certified_v16.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v16.csv"),
    ("v_22_pure_champion_ensemble_certified_v17.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v17.csv"),
    ("v_22_pure_champion_ensemble_certified_v18.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v18.csv"),
    ("v_22_pure_champion_ensemble_certified_v19.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v19.csv"),
    ("v_22_pure_champion_ensemble_certified_v20.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v20.csv"),
    ("v_22_pure_champion_ensemble_certified_v21.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v21.csv")
]

print("\n" + "=" * 80)
print("DIAGNOSTIC RANK CORRELATION MATRIX (V22 vs BENCHMARKS)")
print("=" * 80)

for bname, bpath in benchmarks:
    if os.path.exists(bpath):
        bdf = pd.read_csv(bpath)
        sp_r, _ = spearmanr(v22_preds, bdf["TARGET"].values)
        pe_r, _ = pearsonr(v22_preds, bdf["TARGET"].values)
        print(f"  vs {bname:45s} | Spearman: {sp_r:.6f} ({sp_r*100:.3f}%) | Pearson: {pe_r:.6f} ({pe_r*100:.3f}%)")
    else:
        print(f"  vs {bname:45s} | NOT FOUND")
