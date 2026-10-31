import os
import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, spearmanr

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def array_sha256(arr):
    # Ensure C-contiguous before hashing raw buffer
    arr_c = np.ascontiguousarray(arr)
    return hashlib.sha256(arr_c.tobytes()).hexdigest()

print("=" * 85)
print("  PHASE 4: MASTER ARTIFACT VERIFICATION & ARRAY-LEVEL INTEGRITY AUDIT")
print("=" * 85)

# 1. Historical Champion Target
champ_path = Path("D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv")
print(f"[*] Target Benchmark File: {champ_path}")
print(f"    - File Exists: {champ_path.is_file()}")
print(f"    - File SHA-256: {file_sha256(champ_path)}")
df_bench = pd.read_csv(champ_path)
print(f"    - Rows: {len(df_bench):,}, Columns: {list(df_bench.columns)}")
print(f"    - Unique Ranks: {df_bench['TARGET'].nunique():,}")
print(f"    - Range: [{df_bench['TARGET'].min():.8f}, {df_bench['TARGET'].max():.8f}] | Mean: {df_bench['TARGET'].mean():.8f}")

# 2. Check all copies of the 5 input files across workspace
targets = [
    "v_20261027_precision_transductive_apex.csv",
    "v_17_full_suite_bundle.npz",
    "v55_members.npz",
    "oof_10f_xgb.npy",
    "test_10f_xgb.npy",
    "y_train.npy"
]

print("\n" + "=" * 85)
print("  INSPECTING INSTANCES OF THE 6 FOUNDATION ARTIFACTS")
print("=" * 85)

inventory = pd.read_csv("D:/gci/gemini/champion/dependency_audit/EXP85_DEPENDENCY_INVENTORY.csv")

for t in targets:
    print(f"\n>>> TARGET ARTIFACT: {t} <<<")
    matches = inventory[inventory["FILE_NAME"] == t]
    print(f"Found {len(matches)} instance(s) across workspace:")
    for _, row in matches.iterrows():
        p = Path(row["PATH"])
        print(f"  Path    : {p}")
        print(f"  Size    : {row['SIZE_BYTES']:,} bytes | Modified: {row['LAST_MODIFIED']}")
        print(f"  SHA-256 : {row['SHA256']}")
        
        # Detailed internal inspection
        if t.endswith(".npz"):
            data = np.load(p)
            print(f"  Keys ({len(data.keys())}): {list(data.keys())}")
            for k in data.keys():
                arr = data[k]
                finite = np.all(np.isfinite(arr)) if np.issubdtype(arr.dtype, np.number) else True
                h_arr = array_sha256(arr)
                if np.issubdtype(arr.dtype, np.number):
                    min_val = f"{np.min(arr):.4f}"
                    max_val = f"{np.max(arr):.4f}"
                else:
                    min_val = str(arr.ravel()[0]) if len(arr) > 0 else "EMPTY"
                    max_val = str(arr.ravel()[-1]) if len(arr) > 0 else "EMPTY"
                print(f"    Key '{k}': shape={arr.shape}, dtype={arr.dtype}, finite={finite}, array_sha256={h_arr[:16]}... min={min_val}, max={max_val}")
        elif t.endswith(".npy"):
            arr = np.load(p)
            finite = np.all(np.isfinite(arr))
            h_arr = array_sha256(arr)
            print(f"  Shape: {arr.shape}, dtype={arr.dtype}, finite={finite}, array_sha256={h_arr[:16]}... min={np.min(arr):.4f}, max={np.max(arr):.4f}")
        elif t.endswith(".csv"):
            df = pd.read_csv(p)
            print(f"  Shape: {df.shape}, cols={list(df.columns)}, unique={df['TARGET'].nunique()}, min={df['TARGET'].min():.6f}, max={df['TARGET'].max():.6f}")
