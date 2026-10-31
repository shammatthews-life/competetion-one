import os
import sys
import time
import hashlib
import csv
from pathlib import Path

ROOT_DIR = Path("D:/gci")
OUT_CSV = Path("D:/gci/gemini/champion/dependency_audit/EXP85_DEPENDENCY_INVENTORY.csv")

TARGET_EXTENSIONS = {
    ".py", ".ipynb", ".npz", ".npy", ".csv", ".feather", ".pkl",
    ".pickle", ".json", ".parquet", ".joblib", ".pt", ".pth", ".cbm", ".txt", ".md"
}

IGNORE_DIRS = {
    ".git", ".venv", "__pycache__", ".ipynb_checkpoints", "node_modules", ".pytest_cache"
}

def get_sha256(filepath, max_size_mb=500):
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > max_size_mb:
        return "TOO_LARGE_SKIPPED"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def categorize_file(path_str, name):
    p = path_str.lower()
    if "historical_cache" in p or "name5_dependent_code" in p or ("gemini" in p and "v_20261027" in name.lower()):
        return "ORIGINAL_ARTIFACT"
    elif "backup" in p or "backup" in name.lower():
        return "BACKUP"
    elif "reproduced" in p or "cacherecreation" in p or "recovery" in p or "rebuilt" in name.lower():
        return "GENERATED_REPLACEMENT"
    elif p.endswith(".py") or p.endswith(".ipynb"):
        return "CODE_OR_PIPELINE"
    elif p.endswith(".md") or p.endswith(".txt"):
        return "DOCUMENTATION_OR_LOG"
    else:
        return "UNKNOWN_DATA_OR_CACHE"

def guess_creator_and_consumer(path_str, name):
    name_l = name.lower()
    creator = "UNKNOWN"
    consumer = "UNKNOWN"
    
    if "optimal_macro_apex_v85" in name_l:
        creator = "build_exp85_optimal_macro_apex.py"
        consumer = "competition_submission / evaluation"
    elif "precision_transductive_apex" in name_l:
        creator = "build_precision_transductive_apex_champion.py"
        consumer = "build_exp85_optimal_macro_apex.py"
    elif "optimal_transductive_apex" in name_l:
        creator = "build_optimal_transductive_apex_champion.py"
        consumer = "build_precision_transductive_apex_champion.py"
    elif "asymmetric_tail_gated_apex" in name_l:
        creator = "build_asymmetric_convex_gated_champion.py"
        consumer = "build_optimal_transductive_apex_champion.py"
    elif "level2_regime_stacker" in name_l:
        creator = "build_level2_regime_meta_stacker.py"
        consumer = "build_asymmetric_convex_gated_champion.py"
    elif "v_17_full_suite_bundle" in name_l:
        creator = "train_full_v16_suite.py / recreate_v17_bundle.py"
        consumer = "build_exp85_optimal_macro_apex.py"
    elif "v55_members" in name_l:
        creator = "train_clean_honors_champion.py / recreate_v55_members.py"
        consumer = "build_exp85_optimal_macro_apex.py"
    elif "10f_xgb" in name_l:
        creator = "recreate_xgb10_and_labels.py / exp42 monotonic gpu xgb"
        consumer = "build_exp85_optimal_macro_apex.py"
    elif "hasan_expanded_zoo" in name_l:
        creator = "train_catboost_and_deep_lgb.py + train_hasan_master_zoo.py"
        consumer = "build_precision_transductive_apex_champion.py"
    elif "hasan_selected_550" in name_l:
        creator = "screen_and_select_features.py"
        consumer = "train_catboost_and_deep_lgb.py / train_hasan_master_zoo.py"
    elif "hasan_features" in name_l:
        creator = "build_hasan_600_features.py"
        consumer = "screen_and_select_features.py"
    elif "train.csv" in name_l or "test.csv" in name_l:
        creator = "Kaggle Home Credit Raw Ingestion"
        consumer = "All feature engineering & baseline training"
    elif "exp85_recovery" in name_l:
        creator = "recovery pipeline"
        consumer = "evaluation audit"
    
    return creator, consumer

print(f"[*] Starting comprehensive recursive scan of {ROOT_DIR}...")
t0 = time.time()
file_records = []
total_scanned = 0

for root, dirs, files in os.walk(ROOT_DIR):
    # Skip ignored dirs in-place
    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in TARGET_EXTENSIONS:
            total_scanned += 1
            full_path = Path(root) / file
            try:
                stat = full_path.stat()
                size_bytes = stat.st_size
                mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                sha256 = get_sha256(full_path)
                category = categorize_file(str(full_path), file)
                creator, consumer = guess_creator_and_consumer(str(full_path), file)
                
                file_records.append({
                    "PATH": str(full_path).replace("\\", "/"),
                    "FILE_NAME": file,
                    "EXTENSION": ext,
                    "SIZE_BYTES": size_bytes,
                    "LAST_MODIFIED": mtime_str,
                    "SHA256": sha256,
                    "CATEGORY": category,
                    "UPSTREAM_CREATOR": creator,
                    "DOWNSTREAM_CONSUMER": consumer
                })
            except Exception as e:
                print(f"[!] Error reading {full_path}: {e}")

print(f"[+] Scanned {total_scanned} files in {time.time()-t0:.2f}s.")
print(f"[*] Writing inventory to {OUT_CSV}...")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "PATH", "FILE_NAME", "EXTENSION", "SIZE_BYTES", "LAST_MODIFIED", "SHA256",
        "CATEGORY", "UPSTREAM_CREATOR", "DOWNSTREAM_CONSUMER"
    ])
    writer.writeheader()
    writer.writerows(file_records)

print(f"[+] Successfully wrote {len(file_records)} inventory rows to {OUT_CSV}!")

