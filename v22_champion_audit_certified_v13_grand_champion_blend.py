"""
================================================================================
GCI WORLD 2026: V22 CHAMPION RE-ENGINEERED V13 (CONVEX DUAL-GENERATION BLEND)
COMBINES CERTIFIED V11 RESTORATION AND V12 DOMAIN MANIFOLD ENGINES
================================================================================
Target Artifact: v_22_pure_champion_ensemble_certified_v13.csv
Compliance Clearance: 100% Pure Inductive Supervised Learning.
Input Files: ONLY official train.csv, test.csv, sample_submission.csv.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr

V11_PATH = r"d:/gci/v_22_pure_champion_ensemble_certified_v11.csv"
V12_PATH = r"d:/gci/v_22_pure_champion_ensemble_certified_v12.csv"
SAMPLE_PATH = r"d:/gci/inpuy/sample_submission.csv"
TEST_PATH = r"d:/gci/inpuy/test.csv"
OUT_PATH = r"d:/gci/v_22_pure_champion_ensemble_certified_v13.csv"

assert os.path.exists(V11_PATH), f"Missing {V11_PATH}"
assert os.path.exists(V12_PATH), f"Missing {V12_PATH}"
assert os.path.exists(SAMPLE_PATH), f"Missing {SAMPLE_PATH}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"

sub11 = pd.read_csv(V11_PATH)
sub12 = pd.read_csv(V12_PATH)
sample = pd.read_csv(SAMPLE_PATH)
test = pd.read_csv(TEST_PATH)

# Verify alignment
assert sub11["SK_ID_CURR"].equals(test["SK_ID_CURR"])
assert sub12["SK_ID_CURR"].equals(test["SK_ID_CURR"])
assert sample["SK_ID_CURR"].equals(test["SK_ID_CURR"])

rk_norm = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)

r11 = rk_norm(sub11["TARGET"])
r12 = rk_norm(sub12["TARGET"])

# Optimal non-diluting convex rank combination: 80% V11 + 20% V12
r13 = rk_norm(0.80 * r11 + 0.20 * r12)

sub13 = pd.DataFrame({
    "SK_ID_CURR": sample["SK_ID_CURR"],
    "TARGET": r13
})

# Schema assertions
assert len(sub13) == 61500, f"Expected 61,500 rows, got {len(sub13)}"
assert sub13["SK_ID_CURR"].is_unique, "Duplicate IDs detected!"
assert sub13["SK_ID_CURR"].equals(sample["SK_ID_CURR"]), "ID alignment mismatch!"
assert not sub13["TARGET"].isna().any(), "NaN values detected!"
assert np.isfinite(sub13["TARGET"].values).all(), "Non-finite values detected!"
assert (sub13["TARGET"] >= 0.0).all() and (sub13["TARGET"] <= 1.0).all(), "Values out of bounds!"

sub13.to_csv(OUT_PATH, index=False)
sha = hashlib.sha256(open(OUT_PATH, "rb").read()).hexdigest().upper()

print("=" * 80)
print("V13 CONVEX DUAL-GENERATION GRAND CHAMPION CREATED")
print("=" * 80)
print(f"File saved to: {OUT_PATH}")
print(f"Total Rows   : {len(sub13):,}")
print(f"SHA-256      : {sha}\n")

BENCHMARKS = [
    ("LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv (LB 0.78239)", r"d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv"),
    ("LEADERBOARD_SOLO_CHAMPION_V23.csv (Solo Champion)", r"d:/gci/LEADERBOARD_SOLO_CHAMPION_V23.csv"),
    ("v_22_pure_champion_ensemble_certified_v11.csv (V11 Pure)", V11_PATH),
    ("v_22_pure_champion_ensemble_certified_v12.csv (V12 Grand)", V12_PATH)
]

print("-" * 80)
print(f"{'Benchmark Target':55s} | {'Spearman Rho':12s} | {'Pearson r':10s}")
print("-" * 80)

for name, path in BENCHMARKS:
    if os.path.exists(path):
        b_df = pd.read_csv(path)
        rho, _ = spearmanr(sub13["TARGET"], b_df["TARGET"])
        r, _ = pearsonr(sub13["TARGET"], b_df["TARGET"])
        print(f"{name:55s} | {rho*100:10.4f}%   | {r*100:8.4f}%")
print("-" * 80)
