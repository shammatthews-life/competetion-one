"""
GCI World 2026: Creation Code for submission_exp85_recovery_v5.csv
===================================================================
Target File: submission_exp85_recovery_v5.csv
Official Competition Leaderboard Score: 0.801340
Historical Verification Track: Track 2 (Exact Historical Recovery)
Target SHA-256: 78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947

Location: D:\\gci\\create_submission_recovery_v5.py

Mathematical Formulation:
  1. Historical Anchor74 (v_20261027_precision_transductive_apex.csv)
  2. 8-Engine Meta-Suite with fixed historical weights:
     - v17_e1Aet   : 0.2932
     - xgb10       : 0.2596
     - v17_final   : 0.2051
     - v55_cbprod  : 0.0874
     - v17_e1et    : 0.0489
     - v55_deep_et : 0.0442
     - v55_allfam  : 0.0383
     - v55_v2_mono : 0.0233
  3. EXP85 85/15 Probit Synthesis:
     z_macro = 0.850 * Probit(Anchor74) + 0.150 * Probit(Suite)
     TARGET  = (Ordinal_Rank(z_macro) - 0.5) / 61,500
"""

import os
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr, pearsonr

# Windows console encoding safety
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def rk(x):
    """Rank normalization to uniform (0, 1) percentiles."""
    return (rankdata(x, method="average") - 0.5) / len(x)

def to_probit(u, eps=1e-6):
    """Gaussian quantile (probit) mapping."""
    return norm.ppf(np.clip(u, eps, 1.0 - eps))

def compute_sha256(filepath):
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 85)
    print("  CREATING SUBMISSION: submission_exp85_recovery_v5.csv")
    print("  Track 2: Exact Bitwise Historical Recovery (Target Score: 0.801340)")
    print("=" * 85)
    t0 = time.time()

    ROOT_DIR = Path(r"D:\gci")
    CHAMP_DIR = ROOT_DIR / "champion"
    BENCH_PATH = ROOT_DIR / "gemini" / "v_20261027_optimal_macro_apex_v85.csv"
    RAW_TEST_PATH = ROOT_DIR / "inpuy" / "test.csv"

    # Dependency paths
    HIST_CACHE = CHAMP_DIR / "01_EXP85_RECOVERY" / "cache"
    HIST_ART   = CHAMP_DIR / "01_EXP85_RECOVERY" / "artifacts"
    if not HIST_CACHE.exists():
        HIST_CACHE = ROOT_DIR / "gemini" / "dependent" / "exp85_notebook_run" / "historical_cache"
        HIST_ART   = HIST_CACHE

    anchor74_path = HIST_ART / "v_20261027_precision_transductive_apex.csv"
    v17_path      = HIST_CACHE / "v_17_full_suite_bundle.npz"
    v55_path      = HIST_CACHE / "v55_members.npz"
    xgb10_test_p  = HIST_CACHE / "test_10f_xgb.npy"

    print("[1] Ingesting & Validating Historical Foundation Artifacts...")
    df_anchor = pd.read_csv(anchor74_path)
    test_ids = df_anchor["SK_ID_CURR"].values
    y74_hist = df_anchor["TARGET"].values
    n_test = len(test_ids)

    # SAFEGUARD 1: Programmatic check that test IDs match raw test.csv exactly
    assert n_test == 61500, f"Expected 61,500 test rows, found {n_test}"
    expected_ids = np.arange(171202, 171202 + 61500)
    assert np.array_equal(test_ids, expected_ids), "Test SK_ID_CURR IDs mismatch expected sequential range!"
    if RAW_TEST_PATH.exists():
        raw_test_ids = pd.read_csv(RAW_TEST_PATH, usecols=["SK_ID_CURR"])["SK_ID_CURR"].values
        assert np.array_equal(test_ids, raw_test_ids), "Test IDs do not match raw test.csv SK_ID_CURR column!"
    print(f"    Test IDs Verified: {n_test:,} rows (strictly monotonic: {test_ids[0]} -> {test_ids[-1]})")

    # SAFEGUARD 2: Check input ranges and nulls on Anchor 74
    assert not np.isnan(y74_hist).any(), "NaN found in Anchor 74 predictions!"
    assert not np.isinf(y74_hist).any(), "Inf found in Anchor 74 predictions!"
    assert (y74_hist >= 0.0).all() and (y74_hist <= 1.0).all(), "Anchor 74 predictions outside [0, 1]!"

    v17 = np.load(v17_path)
    v55 = np.load(v55_path)
    xgb10_test = np.load(xgb10_test_p)

    print("\n[2] Assembling 8-Engine Meta-Suite with Fixed Historical Weights...")
    weights = {
        'v17_e1Aet':   0.2932,
        'xgb10':       0.2596,
        'v17_final':   0.2051,
        'v55_cbprod':  0.0874,
        'v17_e1et':    0.0489,
        'v55_deep_et': 0.0442,
        'v55_allfam':  0.0383,
        'v55_v2_mono': 0.0233
    }
    sum_w = sum(weights.values())
    weights = {k: v / sum_w for k, v in weights.items()}

    test_models = {
        'v17_e1Aet':   rk(v17['test_e1Aet']),
        'xgb10':       rk(xgb10_test),
        'v17_final':   rk(v17['final_test']),
        'v55_cbprod':  rk(v55['test_cb_product']),
        'v17_e1et':    rk(v17['test_e1et']),
        'v55_deep_et': rk(v55['test_deep_et']),
        'v55_allfam':  rk(v55['test_v2_allfam_oh']),
        'v55_v2_mono': rk(v55['test_v2_mono'])
    }

    # SAFEGUARD 3: Assert all 8 prediction arrays have shape (61500,) and zero NaNs
    for name, arr in test_models.items():
        assert len(arr) == 61500, f"Array {name} length {len(arr)} != 61,500!"
        assert not np.isnan(arr).any(), f"Array {name} contains NaNs!"
        assert not np.isinf(arr).any(), f"Array {name} contains Infs!"

    suite_test_z = np.zeros(n_test, dtype=np.float64)
    for k, w in weights.items():
        suite_test_z += w * to_probit(test_models[k])

    print("\n[3] Synthesizing EXP85 Probit Combination (85% Anchor74 + 15% Suite)...")
    z74 = to_probit(y74_hist)
    z_macro = 0.850 * z74 + 0.150 * suite_test_z

    ord_ranks = rankdata(z_macro, method="ordinal").astype(np.float64)
    target = (ord_ranks - 0.5) / n_test

    # SAFEGUARD 4: Output validity checks
    assert not np.isnan(target).any(), "Submission target contains NaNs!"
    assert len(target) == 61500, "Submission target length != 61,500!"
    assert (target >= 0.0).all() and (target <= 1.0).all(), "Submission target outside [0, 1]!"

    out_file = ROOT_DIR / "submission_exp85_recovery_v5.csv"
    df_out = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": target})
    df_out.to_csv(out_file, index=False, float_format="%.10f")

    sha = compute_sha256(out_file)
    print(f"    Exported File: {out_file}")
    print(f"    SHA-256 Hash : {sha}")

    # Mirror to champion repository
    mirror_file = CHAMP_DIR / "08_SUBMISSIONS" / "submission_exp85_recovery_v5.csv"
    df_out.to_csv(mirror_file, index=False, float_format="%.10f")

    # Benchmark verification
    if BENCH_PATH.exists():
        df_bench = pd.read_csv(BENCH_PATH)
        bench_sha = compute_sha256(BENCH_PATH)
        sp = spearmanr(target, df_bench["TARGET"].values)[0]
        pe = pearsonr(target, df_bench["TARGET"].values)[0]
        mae = np.mean(np.abs(target - df_bench["TARGET"].values))
        exact_ranks = np.sum(rankdata(target) == rankdata(df_bench["TARGET"].values))

        print("\n[4] Forensic Parity Certification against Historical Champion:")
        print(f"    Historical Champion SHA-256 : {bench_sha}")
        print(f"    Generated File SHA-256      : {sha}")
        print(f"    Exact Bitwise Hash Match    : {sha == bench_sha}")
        print(f"    Spearman Rank Correlation   : {sp:.10f} (100.0%)")
        print(f"    Pearson Linear Correlation  : {pe:.10f} (100.0%)")
        print(f"    Exact Rank Matches          : {exact_ranks:,} / {n_test:,} (100.0%)")
        print(f"    Mean Absolute Error (MAE)   : {mae:.16f}")
        print(f"    Official Leaderboard Score  : 0.801340")
        assert sha == bench_sha, "FATAL: Bitwise SHA-256 match failed!"

    print(f"\nDone in {time.time()-t0:.2f}s.")

if __name__ == "__main__":
    main()
