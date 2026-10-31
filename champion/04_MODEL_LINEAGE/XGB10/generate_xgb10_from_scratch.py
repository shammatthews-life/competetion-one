"""
recreate_xgb10_and_labels.py
================================================================================
GCI World 2026 - 100% Self-Contained 10-Fold XGBoost & Label Generator
================================================================================
Rule Compliance : 100% Strictly Independent (train.csv and test.csv ONLY)
Target Artifacts:
  - cache/oof_10f_xgb.npy  : 10-Fold Hist-XGBoost OOF Predictions (171,202 rows, float32)
  - cache/test_10f_xgb.npy : 10-Fold Hist-XGBoost Test Predictions (61,500 rows, float32)
  - cache/y_train.npy      : Ground Truth Training Labels (171,202 rows, int32)
================================================================================
"""

from pathlib import Path
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "inpuy"
OUTPUT_DIR = SCRIPT_DIR


def build_xgb_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Constructs the 239-feature multi-view manifold directly from train and test CSVs.
    Zero upstream cached files used.
    """
    n_train = len(df_train)
    n_test = len(df_test)
    df_all = pd.concat([df_train.drop(columns=["TARGET"]), df_test], ignore_index=True)

    CATS = [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
    ]

    emp_raw = df_all["DAYS_EMPLOYED"].values.astype(np.float64)
    days_emp = np.where(emp_raw == 365243.0, np.nan, emp_raw)

    inc = df_all["AMT_INCOME_TOTAL"].values.astype(np.float64)
    cred = df_all["AMT_CREDIT"].values.astype(np.float64)
    ann = df_all["AMT_ANNUITY"].values.astype(np.float64)
    goods = df_all["AMT_GOODS_PRICE"].fillna(df_all["AMT_CREDIT"]).values.astype(np.float64)

    birth = df_all["DAYS_BIRTH"].values.astype(np.float64)
    reg = df_all["DAYS_REGISTRATION"].values.astype(np.float64)
    id_pub = df_all["DAYS_ID_PUBLISH"].values.astype(np.float64)
    phone = df_all["DAYS_LAST_PHONE_CHANGE"].values.astype(np.float64)

    e1 = df_all["EXT_SOURCE_1"].values.astype(np.float64)
    e2 = df_all["EXT_SOURCE_2"].values.astype(np.float64)
    e3 = df_all["EXT_SOURCE_3"].values.astype(np.float64)

    F = {}
    F["DAYS_EMPLOYED_CLEAN"] = days_emp.astype(np.float32)
    F["IS_RETIRED"] = (emp_raw == 365243.0).astype(np.float32)
    F["MISS_CNT"] = df_all.isna().sum(axis=1).values.astype(np.float32)

    # Demographics & Relative Event Ages
    F["AGE_YEARS"] = (-birth / 365.25).astype(np.float32)
    F["EMPLOYED_YEARS"] = (-days_emp / 365.25).astype(np.float32)
    F["EMP_TO_AGE"] = (days_emp / (birth + 1.0)).astype(np.float32)
    F["REG_TO_AGE"] = (reg / (birth + 1.0)).astype(np.float32)
    F["ID_TO_AGE"] = (id_pub / (birth + 1.0)).astype(np.float32)
    F["PHONE_TO_AGE"] = (phone / (birth + 1.0)).astype(np.float32)

    # Banking & Solvency
    term = cred / (ann + 1.0)
    F["CREDIT_TERM"] = term.astype(np.float32)
    F["EAR"] = np.clip(24.0 * (ann * term - cred) / (cred * (term + 1.0) + 1.0), -1.0, 5.0).astype(np.float32)
    surplus = inc - 12.0 * ann
    F["SURPLUS"] = surplus.astype(np.float32)
    F["SURPLUS_TO_INC"] = (surplus / (inc + 1.0)).astype(np.float32)
    F["TRUE_COST"] = (ann * term - cred).astype(np.float32)
    F["TRUE_COST_RATIO"] = ((ann * term - cred) / (cred + 1.0)).astype(np.float32)
    F["LTV"] = np.clip(cred / (goods + 1.0), 0.0, 5.0).astype(np.float32)
    F["DTI"] = np.clip(cred / (inc + 1.0), 0.0, 20.0).astype(np.float32)
    F["DEBT_BURDEN"] = np.clip(ann / (inc / 12.0 + 1.0), 0.0, 5.0).astype(np.float32)
    F["DSCR"] = np.clip(((inc / 12.0) * 0.65) / (ann + 1.0), 0.0, 20.0).astype(np.float32)
    F["BUFFER_MONTHS"] = np.clip(surplus / (12.0 * ann + 1.0), -10.0, 50.0).astype(np.float32)
    F["INC_PER_FAM"] = (inc / (df_all["CNT_FAM_MEMBERS"].fillna(1.0) + 1e-3)).values.astype(np.float32)

    # External Sources Triad
    ext_stack = np.column_stack([e1, e2, e3])
    F["EXT_MEAN"] = np.nanmean(ext_stack, axis=1).astype(np.float32)
    F["EXT_MIN"] = np.nanmin(ext_stack, axis=1).astype(np.float32)
    F["EXT_MAX"] = np.nanmax(ext_stack, axis=1).astype(np.float32)
    F["EXT_STD"] = np.nan_to_num(np.nanstd(ext_stack, axis=1), nan=0.0).astype(np.float32)
    F["EXT_W"] = (np.nan_to_num(0.2 * e1) + np.nan_to_num(0.4 * e2) + np.nan_to_num(0.4 * e3)).astype(np.float32)
    F["EXT_DIFF_12"] = (e1 - e2).astype(np.float32)
    F["EXT_DIFF_23"] = (e2 - e3).astype(np.float32)
    F["EXT_PROD"] = (np.nan_to_num(e1, nan=0.5) * np.nan_to_num(e2, nan=0.5) * np.nan_to_num(e3, nan=0.5)).astype(np.float32)

    # Log transforms
    F["LOG_INCOME"] = np.log1p(inc).astype(np.float32)
    F["LOG_CREDIT"] = np.log1p(cred).astype(np.float32)
    F["LOG_ANNUITY"] = np.log1p(ann).astype(np.float32)

    # Merton Distance-to-Default
    T_years = np.clip(term / 12.0, 0.1, 10.0)
    disp_ratio = np.maximum(0.01, (inc / 12.0) / (ann + 1.0))
    for mu, sigma, tag in [(0.02, 0.20, "20"), (0.03, 0.25, "25")]:
        dd = (np.log(disp_ratio) + (mu - 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        F[f"MERTON_DD_{tag}"] = np.clip(dd, -10.0, 10.0).astype(np.float32)

    # One-hot encodings for categoricals
    df_oh = pd.get_dummies(df_all[CATS].astype("category"), dummy_na=True, dtype=np.float32)
    df_oh.columns = [f"OH_{i}_{c}" for i, c in enumerate(df_oh.columns)]

    # Retain raw numeric columns
    raw_num = df_all.select_dtypes(include=[np.number]).drop(columns=["SK_ID_CURR"], errors="ignore")
    for c in raw_num.columns:
        if c not in F:
            F[c] = raw_num[c].values.astype(np.float32)

    X_full = pd.concat([pd.DataFrame(F), df_oh], axis=1)
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean_cols = [c.replace(':', '_').replace('|', '_').replace(' ', '_').replace('-', '_') for c in X_full.columns]
    X_full.columns = clean_cols

    X_tr = X_full.iloc[:n_train].reset_index(drop=True)
    X_te = X_full.iloc[n_train:].reset_index(drop=True)
    return X_tr, X_te


def generate_xgb_and_labels(fast=False, output_dir=OUTPUT_DIR):
    """
    Train 10-fold Hist-XGBoost and extract ground truth y_train.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    print(f"[*] Ingesting raw train dataset: {train_path}")
    print(f"[*] Ingesting raw test dataset : {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    y_train = df_train["TARGET"].values.astype(np.int32)
    n_train = len(df_train)
    n_test = len(df_test)

    # Save y_train.npy immediately
    y_out_path = output_dir / "y_train.npy"
    np.save(y_out_path, y_train)
    print(f"[+] Extracted and saved y_train.npy: shape={y_train.shape}, dtype={y_train.dtype} ({y_out_path.stat().st_size:,} bytes)")

    print("[*] Engineering multi-view features in memory from raw data...")
    X_tr, X_te = build_xgb_features(df_train, df_test)
    print(f"[+] Feature matrix built in {time.time()-t0:.1f}s: {X_tr.shape[1]} features")

    # Hardware acceleration check
    try:
        xgb_test_m = xgb.XGBClassifier(tree_method="hist", device="cuda")
        xgb_test_m.fit(np.zeros((10, 2)), np.zeros(10))
        xgb_device = "cuda"
        print("[*] XGBoost Compute Device: NVIDIA CUDA GPU")
    except Exception:
        xgb_device = "cpu"
        print("[*] XGBoost Compute Device: CPU")

    n_splits = 3 if fast else 10
    n_trees = 150 if fast else 450
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_xgb = np.zeros(n_train, dtype=np.float32)
    test_xgb = np.zeros(n_test, dtype=np.float32)

    print(f"\n[*] Training {n_splits}-Fold Hist-XGBoost (n_estimators={n_trees}, device={xgb_device})...")
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_train), 1):
        m = xgb.XGBClassifier(
            n_estimators=n_trees, learning_rate=0.035, max_depth=6,
            subsample=0.80, colsample_bytree=0.55, reg_alpha=3.0, reg_lambda=8.0,
            min_child_weight=30, tree_method="hist", device=xgb_device,
            eval_metric="auc", early_stopping_rounds=30, random_state=42 + fold
        )
        m.fit(X_tr.iloc[trn_idx], y_train[trn_idx], eval_set=[(X_tr.iloc[val_idx], y_train[val_idx])], verbose=False)
        p_val = m.predict_proba(X_tr.iloc[val_idx])[:, 1]
        oof_xgb[val_idx] = p_val
        test_xgb += m.predict_proba(X_te)[:, 1].astype(np.float32) / float(n_splits)
        f_auc = roc_auc_score(y_train[val_idx], p_val)
        print(f"    Fold {fold:2d}/{n_splits} AUC: {f_auc:.6f}")

    auc_oof = roc_auc_score(y_train, oof_xgb)
    print(f"\n[+] 10-Fold XGBoost Complete: OOF AUC = {auc_oof:.6f} ({time.time()-t0:.1f}s)")

    oof_out = output_dir / "oof_10f_xgb.npy"
    test_out = output_dir / "test_10f_xgb.npy"
    np.save(oof_out, oof_xgb)
    np.save(test_out, test_xgb)

    print("=" * 80)
    print("  Artifacts Generated From Scratch Successfully!")
    print(f"  - y_train.npy      : {y_out_path} ({y_out_path.stat().st_size:,} bytes)")
    print(f"  - oof_10f_xgb.npy  : {oof_out} ({oof_out.stat().st_size:,} bytes)")
    print(f"  - test_10f_xgb.npy : {test_out} ({test_out.stat().st_size:,} bytes)")
    print(f"  Total Runtime: {time.time()-t0:.1f}s")
    print("=" * 80)
    return oof_xgb, test_xgb, y_train


def main():
    parser = argparse.ArgumentParser(description="Generate 10-fold XGBoost and y_train strictly from raw data")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save artifacts")
    args = parser.parse_args()

    out_d = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    generate_xgb_and_labels(fast=args.fast, output_dir=out_d)


if __name__ == "__main__":
    main()
