#!/usr/bin/env python3
"""
================================================================================
GCI WORLD 2026: SUBMISSION CORRELATION DIAGNOSTIC UTILITY
================================================================================
Diagnostic tool for comparing prediction rankings between model submissions.

IMPORTANT METHODOLOGICAL NOTE:
Spearman rank correlation and top-decile overlap quantify similarity in applicant
risk rankings between two prediction files. However:
1. High rank correlation indicates concordance in relative ordering, but does
   NOT guarantee identical test ROC-AUC or identical leaderboard score.
2. Prediction correlation does NOT prove or evaluate competition rule compliance;
   compliance is verified strictly through independent code audit and execution checks.
================================================================================
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent.parent

BENCHMARKS = [
    ("v22_champion_078239", ROOT / "submission_4_v22_champion_078239.csv"),
    ("v22_pure_champion_lb1", ROOT / "LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv"),
    ("v22_reproduced", ROOT / "v_22_pure_champion_ensemble_reproduced.csv"),
    ("v22_scratch", ROOT / "v_22_pure_champion_ensemble_scratch.csv"),
    ("v22_certified_v2", ROOT / "v_22_pure_champion_ensemble_certified_v2.csv"),
    ("v28_certified", ROOT / "LEADERBOARD_SUBMISSION_V28_CERTIFIED.csv"),
    ("v27_grandmaster", ROOT / "LEADERBOARD_SUBMISSION_V27_GRANDMASTER.csv")
]

def audit_file_correlation(target_path: Path):
    if not target_path.exists():
        print(f"[ERROR] Target file does not exist: {target_path}")
        return 1

    print("=" * 80)
    print(f"GCI WORLD 2026: PREDICTION SIMILARITY DIAGNOSTIC FOR {target_path.name}")
    print("=" * 80)

    df_target = pd.read_csv(target_path)
    if "SK_ID_CURR" not in df_target.columns or "TARGET" not in df_target.columns:
        print("[FAIL] Invalid schema: Must contain 'SK_ID_CURR' and 'TARGET'")
        return 1

    # Strict integrity checks on target file
    assert df_target["SK_ID_CURR"].is_unique, "Target file has non-unique SK_ID_CURR"
    assert np.isfinite(df_target["TARGET"].values).all(), "Target file contains NaN or Inf values"
    assert (df_target["TARGET"].values >= 0.0).all() and (df_target["TARGET"].values <= 1.0).all(), "TARGET values out of [0, 1] bounds"

    print(f"Target file: {target_path.name}")
    print(f"Rows: {len(df_target):,}")
    print(f"TARGET Stats -> Min: {df_target['TARGET'].min():.6f}, Max: {df_target['TARGET'].max():.6f}, "
          f"Mean: {df_target['TARGET'].mean():.6f}, Std: {df_target['TARGET'].std():.6f}")
    print("-" * 80)
    print(f"{'Benchmark Name':<30} | {'Spearman Rho':<14} | {'Pearson R':<12} | {'Top-10% Overlap':<15} | Concordance")
    print("-" * 80)

    for name, bench_path in BENCHMARKS:
        if not bench_path.exists():
            print(f"{name:<30} | [NOT FOUND on disk]")
            continue

        df_bench = pd.read_csv(bench_path)
        if "SK_ID_CURR" not in df_bench.columns or "TARGET" not in df_bench.columns:
            print(f"{name:<30} | Invalid schema in benchmark file")
            continue

        assert df_bench["SK_ID_CURR"].is_unique, f"Benchmark {name} has non-unique SK_ID_CURR"
        assert np.isfinite(df_bench["TARGET"].values).all(), f"Benchmark {name} contains non-finite values"

        if len(df_bench) != len(df_target):
            print(f"{name:<30} | Length mismatch ({len(df_bench):,} vs {len(df_target):,})")
            continue

        # Strict set equality of applicant IDs
        assert set(df_target["SK_ID_CURR"]) == set(df_bench["SK_ID_CURR"]), f"ID set mismatch between target and benchmark {name}"

        # Strict 1-to-1 ID alignment check
        if (df_bench["SK_ID_CURR"].values == df_target["SK_ID_CURR"].values).all():
            y_t = df_target["TARGET"].values
            y_b = df_bench["TARGET"].values
        else:
            merged = pd.merge(df_target, df_bench, on="SK_ID_CURR", how="inner", suffixes=('_target', '_bench'))
            if len(merged) != len(df_target):
                print(f"{name:<30} | Incomplete ID match: only {len(merged):,} of {len(df_target):,} matched")
                continue
            y_t = merged["TARGET_target"].values
            y_b = merged["TARGET_bench"].values

        rho, _ = spearmanr(y_t, y_b)
        pr, _ = pearsonr(y_t, y_b)

        # Top 10% highest predicted risk overlap
        k = int(0.10 * len(y_t))
        top_t = set(np.argsort(y_t)[-k:])
        top_b = set(np.argsort(y_b)[-k:])
        overlap = len(top_t & top_b) / k

        status = "VERY HIGH" if rho >= 0.95 else ("MODERATE" if rho >= 0.85 else "LOW")
        print(f"{name:<30} | {rho:.6f}       | {pr:.6f}     | {overlap*100:6.2f}%         | {status}")

    print("-" * 80)
    print("[DIAGNOSTIC NOTE] This tool measures ranking similarity between predictions.")
    print("                  High correlation indicates similar relative applicant risk orderings,")
    print("                  but does not guarantee identical leaderboard AUC or prove rule compliance.")
    print("================================================================================")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit submission prediction similarity against benchmarks")
    parser.add_argument("target", nargs="?", default="v_22_pure_champion_ensemble_certified_v2.csv",
                        help="Name or path of target CSV to test")
    args = parser.parse_args()

    target_file = Path(args.target)
    if not target_file.is_absolute():
        target_file = ROOT / target_file

    sys.exit(audit_file_correlation(target_file))
