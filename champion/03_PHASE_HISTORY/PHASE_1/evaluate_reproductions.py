import os
import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def array_sha256(arr):
    arr_c = np.ascontiguousarray(arr)
    return hashlib.sha256(arr_c.tobytes()).hexdigest()

HIST_DIR = Path("D:/gci/gemini/dependent/exp85_notebook_run/historical_cache")
REPRO_DIR = Path("D:/gci/gemini/champion/cacherecreation")
OUT_CSV = Path("D:/gci/gemini/champion/dependency_audit/EXP85_ARTIFACT_EQUIVALENCE.csv")

artifacts_to_evaluate = [
    {
        "NAME": "y_train.npy",
        "ORIG_PATH": HIST_DIR / "y_train.npy" if (HIST_DIR / "y_train.npy").is_file() else Path("D:/gci/gemini/cache/y_train.npy"),
        "REPRO_PATH": REPRO_DIR / "y_train.npy",
        "TYPE": "NPY"
    },
    {
        "NAME": "oof_10f_xgb.npy",
        "ORIG_PATH": HIST_DIR / "oof_10f_xgb.npy",
        "REPRO_PATH": REPRO_DIR / "oof_10f_xgb.npy",
        "TYPE": "NPY"
    },
    {
        "NAME": "test_10f_xgb.npy",
        "ORIG_PATH": HIST_DIR / "test_10f_xgb.npy",
        "REPRO_PATH": REPRO_DIR / "test_10f_xgb.npy",
        "TYPE": "NPY"
    },
    {
        "NAME": "v55_members.npz",
        "ORIG_PATH": HIST_DIR / "v55_members.npz",
        "REPRO_PATH": REPRO_DIR / "v55_members.npz",
        "TYPE": "NPZ"
    },
    {
        "NAME": "v_17_full_suite_bundle.npz",
        "ORIG_PATH": HIST_DIR / "v_17_full_suite_bundle.npz",
        "REPRO_PATH": REPRO_DIR / "v_17_full_suite_bundle.npz",
        "TYPE": "NPZ"
    },
    {
        "NAME": "v_20261027_precision_transductive_apex.csv",
        "ORIG_PATH": HIST_DIR / "v_20261027_precision_transductive_apex.csv",
        "REPRO_PATH": REPRO_DIR / "v_20261027_precision_transductive_apex.csv",
        "TYPE": "CSV"
    },
    {
        "NAME": "v_20261027_optimal_macro_apex_v85.csv (Historical Champion)",
        "ORIG_PATH": Path("D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv"),
        "REPRO_PATH": Path("D:/gci/gemini/champion/dependency_audit/submission_exp85_dependency_reproduced.csv"),
        "TYPE": "CSV"
    },
    {
        "NAME": "submission_exp85_recovery_v5_best.csv (Clean-Room V5)",
        "ORIG_PATH": Path("D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv"),
        "REPRO_PATH": REPRO_DIR / "submission_exp85_recovery_v5_best.csv",
        "TYPE": "CSV"
    }
]

records = []

for item in artifacts_to_evaluate:
    name = item["NAME"]
    orig_p = item["ORIG_PATH"]
    repro_p = item["REPRO_PATH"]
    atype = item["TYPE"]
    
    print(f"\n[*] Evaluating: {name}")
    if not orig_p.is_file():
        print(f"  [!] Missing original: {orig_p}")
        continue
    if not repro_p.is_file():
        print(f"  [!] Missing reproduced: {repro_p}")
        continue
        
    orig_sha = file_sha256(orig_p)
    repro_sha = file_sha256(repro_p)
    exact_archive_match = (orig_sha == repro_sha)
    
    exact_key_match = False
    exact_array_match = False
    max_ae = 0.0
    mean_ae = 0.0
    pe_corr = 0.0
    sp_corr = 0.0
    n_exact_elem = 0
    total_elem = 0
    missing_deps = "NONE"
    classification = "UNKNOWN"
    
    if atype == "NPY":
        arr_orig = np.load(orig_p)
        arr_repro = np.load(repro_p)
        exact_key_match = True
        total_elem = len(arr_orig)
        exact_array_match = (array_sha256(arr_orig) == array_sha256(arr_repro))
        max_ae = float(np.max(np.abs(arr_repro - arr_orig)))
        mean_ae = float(np.mean(np.abs(arr_repro - arr_orig)))
        pe_corr = float(pearsonr(arr_repro, arr_orig)[0]) if max_ae > 0 else 1.0
        sp_corr = float(spearmanr(arr_repro, arr_orig).correlation) if max_ae > 0 else 1.0
        n_exact_elem = int(np.sum(arr_repro == arr_orig))
        
    elif atype == "NPZ":
        z_orig = np.load(orig_p)
        z_repro = np.load(repro_p)
        exact_key_match = (list(z_orig.keys()) == list(z_repro.keys()))
        
        # Compare key arrays
        shared_keys = [k for k in z_orig.keys() if k in z_repro.keys() and np.issubdtype(z_orig[k].dtype, np.number)]
        all_equal = True
        all_elements = 0
        eq_elements = 0
        max_diffs = []
        mean_diffs = []
        corrs = []
        
        for k in shared_keys:
            a_o = z_orig[k].ravel()
            a_r = z_repro[k].ravel()
            all_elements += len(a_o)
            eq_elements += int(np.sum(a_o == a_r))
            if array_sha256(a_o) != array_sha256(a_r):
                all_equal = False
            max_diffs.append(float(np.max(np.abs(a_r - a_o))))
            mean_diffs.append(float(np.mean(np.abs(a_r - a_o))))
            if len(a_o) > 1 and float(np.std(a_o)) > 1e-6 and float(np.std(a_r)) > 1e-6:
                corrs.append(float(spearmanr(a_r, a_o).correlation))
                
        exact_array_match = all_equal
        total_elem = all_elements
        n_exact_elem = eq_elements
        max_ae = max(max_diffs) if max_diffs else 0.0
        mean_ae = float(np.mean(mean_diffs)) if mean_diffs else 0.0
        sp_corr = float(np.mean(corrs)) if corrs else 1.0
        pe_corr = sp_corr
        
    elif atype == "CSV":
        df_o = pd.read_csv(orig_p)
        df_r = pd.read_csv(repro_p)
        exact_key_match = (list(df_o.columns) == list(df_r.columns))
        y_o = df_o["TARGET"].values
        y_r = df_r["TARGET"].values
        total_elem = len(y_o)
        n_exact_elem = int(np.sum(y_o == y_r))
        max_ae = float(np.max(np.abs(y_r - y_o)))
        mean_ae = float(np.mean(np.abs(y_r - y_o)))
        pe_corr = float(pearsonr(y_r, y_o)[0]) if max_ae > 0 else 1.0
        sp_corr = float(spearmanr(y_r, y_o).correlation) if max_ae > 0 else 1.0
        exact_array_match = (n_exact_elem == total_elem)
        
    pct_exact = (n_exact_elem / total_elem * 100.0) if total_elem > 0 else 0.0
    
    # Classification logic
    if exact_archive_match:
        classification = "BITWISE_IDENTICAL"
    elif exact_array_match:
        classification = "ARRAY_IDENTICAL_ARCHIVE_DIFFERENT"
    elif sp_corr >= 0.9990 and mean_ae < 0.005:
        classification = "NUMERICALLY_CLOSE"
    elif sp_corr > 0.85:
        classification = "CORRELATED_BUT_DIFFERENT"
    else:
        classification = "INVALID_OR_WRONG_ARTIFACT"
        
    print(f"  Result: {classification} (Spearman: {sp_corr:.8f}, MAE: {mean_ae:.6f}, Pct Exact: {pct_exact:.2f}%)")
    
    records.append({
        "ARTIFACT_NAME": name,
        "ORIGINAL_PATH": str(orig_p).replace("\\", "/"),
        "REPRODUCED_PATH": str(repro_p).replace("\\", "/"),
        "ORIGINAL_SHA256": orig_sha,
        "REPRODUCED_SHA256": repro_sha,
        "EXACT_ARCHIVE_MATCH": exact_archive_match,
        "EXACT_KEY_MATCH": exact_key_match,
        "EXACT_ARRAY_MATCH": exact_array_match,
        "MAX_ABSOLUTE_ERROR": max_ae,
        "MEAN_ABSOLUTE_ERROR": mean_ae,
        "PEARSON_CORRELATION": pe_corr,
        "SPEARMAN_CORRELATION": sp_corr,
        "EXACT_EQUAL_ELEMENTS": n_exact_elem,
        "TOTAL_ELEMENTS": total_elem,
        "PCT_EXACT_EQUAL": pct_exact,
        "MISSING_UPSTREAM_DEPENDENCIES": missing_deps,
        "CLASSIFICATION": classification
    })

df_out = pd.DataFrame(records)
df_out.to_csv(OUT_CSV, index=False)
print(f"\n[+] Successfully generated {len(records)} artifact equivalence records at {OUT_CSV}!")

