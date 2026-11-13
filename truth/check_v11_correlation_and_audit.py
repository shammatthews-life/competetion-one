"""
================================================================================
GCI WORLD 2026: V11 INDEPENDENT CORRELATION & RULE VERIFICATION TOOL
================================================================================
Comprehensive mathematical evaluation of submission CSV against all benchmark
artefacts (LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22, submission_4_v22_champion_078239,
LEADERBOARD_SOLO_CHAMPION_V23, V8, V9).
Evaluates overall rank correlation (Spearman rho), Pearson correlation, mean absolute
rank error, extreme rank deviations, repeat borrower segment alignment, and
strict adherence to competition rules.
================================================================================
"""

import sys
import os
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata

def evaluate_submission(sub_path):
    print("=" * 80)
    print(f"GCI WORLD 2026: SUBMISSION AUDIT & CORRELATION ANALYSIS")
    print(f"Target Artifact: {sub_path}")
    print("=" * 80)

    if not os.path.exists(sub_path):
        print(f"[!] ERROR: Target artifact '{sub_path}' does not exist.")
        return

    # Check file size & SHA-256
    sha = hashlib.sha256(open(sub_path, "rb").read()).hexdigest().upper()
    size = os.path.getsize(sub_path)
    print(f"[*] Artifact SHA-256: {sha}")
    print(f"[*] Artifact File Size: {size:,} bytes")

    sub_df = pd.read_csv(sub_path)
    print(f"[*] Total Rows: {len(sub_df):,}")
    print(f"[*] Unique SK_ID_CURR: {sub_df['SK_ID_CURR'].nunique():,}")
    assert len(sub_df) == 61500, f"Expected 61,500 rows, got {len(sub_df)}"
    assert sub_df["SK_ID_CURR"].nunique() == 61500, "Duplicate IDs detected"
    assert not sub_df["TARGET"].isna().any(), "Target contains NaNs"
    assert np.isfinite(sub_df["TARGET"].values).all(), "Target contains Infinities"
    assert (sub_df["TARGET"] >= 0.0).all() and (sub_df["TARGET"] <= 1.0).all(), "Values out of [0, 1] range"
    print("[+] All structural schema, uniqueness, finite, and boundary checks: PASSED")

    # Load test and train for repeat borrower breakdown
    te = pd.read_csv(r"d:/gci/inpuy/test.csv")
    tr = pd.read_csv(r"d:/gci/inpuy/train.csv")

    i1_te = te["DAYS_REGISTRATION"] - te["DAYS_BIRTH"]
    i2_te = te["DAYS_ID_PUBLISH"] - te["DAYS_BIRTH"]
    k_te = te[["CODE_GENDER", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE", "FLAG_MOBIL"]].astype(str).agg("|".join, axis=1) + "|" + i1_te.astype(str) + "|" + i2_te.astype(str)

    i1_tr = tr["DAYS_REGISTRATION"] - tr["DAYS_BIRTH"]
    i2_tr = tr["DAYS_ID_PUBLISH"] - tr["DAYS_BIRTH"]
    k_tr = tr[["CODE_GENDER", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE", "FLAG_MOBIL"]].astype(str).agg("|".join, axis=1) + "|" + i1_tr.astype(str) + "|" + i2_tr.astype(str)

    tr_keys = set(k_tr)
    is_repeat = k_te.isin(tr_keys).values

    rk_sub = (rankdata(sub_df["TARGET"]) - 0.5) / len(sub_df)

    benchmarks = [
        ("LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv", r"d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv"),
        ("submission_4_v22_champion_078239.csv", r"d:/gci/submission_4_v22_champion_078239.csv"),
        ("LEADERBOARD_SOLO_CHAMPION_V23.csv", r"d:/gci/LEADERBOARD_SOLO_CHAMPION_V23.csv"),
        ("v_22_pure_champion_ensemble_certified_v8.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v8.csv"),
        ("v_22_pure_champion_ensemble_certified_v9.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v9.csv"),
    ]

    print("\n" + "-" * 80)
    print(f"{'Benchmark Target':<48} {'Spearman rho':<14} {'Pearson r':<14} {'Match %'}")
    print("-" * 80)

    for bname, bpath in benchmarks:
        if not os.path.exists(bpath):
            continue
        bdf = pd.read_csv(bpath)
        rho, _ = spearmanr(sub_df["TARGET"], bdf["TARGET"])
        pr, _ = pearsonr(sub_df["TARGET"], bdf["TARGET"])
        print(f"{bname:<48} {rho:<14.6f} {pr:<14.6f} {rho*100:.3f}%")

        if bname == "LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv":
            rk_b = (rankdata(bdf["TARGET"]) - 0.5) / len(bdf)
            diff = np.abs(rk_sub - rk_b)
            rho_nonrep, _ = spearmanr(rk_sub[~is_repeat], rk_b[~is_repeat])
            rho_rep, _ = spearmanr(rk_sub[is_repeat], rk_b[is_repeat])
            print(f"\n  [Deep Segment Breakdown vs Original Champion]:")
            print(f"    - Non-Repeat Applicants ({np.sum(~is_repeat):,} rows / {np.sum(~is_repeat)/len(sub_df)*100:.2f}%):")
            print(f"        Spearman rho: {rho_nonrep:.6f} ({rho_nonrep*100:.3f}%) | Mean Abs Rank Error: {np.mean(diff[~is_repeat]):.5f}")
            print(f"    - Repeat Historical Applicants ({np.sum(is_repeat):,} rows / {np.sum(is_repeat)/len(sub_df)*100:.2f}%):")
            print(f"        Spearman rho: {rho_rep:.6f} ({rho_rep*100:.3f}%) | Mean Abs Rank Error: {np.mean(diff[is_repeat]):.5f}")
            print(f"    - Mean Absolute Rank Deviation: {np.mean(diff):.5f}")
            print(f"    - Extreme Rank Deviations (> 0.10): {np.sum(diff > 0.10):,} rows ({np.sum(diff > 0.10)/len(sub_df)*100:.2f}%)")

    print("-" * 80)
    print("[*] Evaluation Complete.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else r"d:/gci/v_22_pure_champion_ensemble_certified_v11.csv"
    evaluate_submission(target)
