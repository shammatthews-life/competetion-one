"""
generate_v55_from_scratch.py / recreate_v55_members.py
================================================================================
GCI World 2026 - 100% Self-Contained V55 Multi-Architecture Suite Generator
================================================================================
Rule Compliance : 100% Strictly Independent (train.csv and test.csv ONLY)
Target Artifact : cache/v55_members.npz

Architecture & Member Role Specification:
  Trains all 7 distinct V55 multi-architecture models strictly from raw data:
    1. qf_shallow   : LightGBM Shallow Reg (num_leaves=7, max_depth=3, extra_trees=True)
    2. qf_reg       : LightGBM Conservative Reg (num_leaves=11, min_child_samples=300, lambda_l2=40)
    3. v1_allfam_oh : LightGBM All-Family OH (num_leaves=15, feature_fraction=0.3, extra_trees=True)
    4. v2_allfam_oh : LightGBM V2 Demographics OH (num_leaves=15, feature_fraction=0.3) [SUITE MEMBER: 3.83%]
    5. v2_mono      : LightGBM Monotone Shallow-Reg (monotone_constraints on EXT)       [SUITE MEMBER: 2.33%]
    6. deep_et      : LightGBM Deep ExtraTrees (num_leaves=31, extra_trees=True)        [SUITE MEMBER: 4.42%]
    7. cb_product   : CatBoost Product Specialist (depth=6, l2=8.0, categorical trees)   [SUITE MEMBER: 8.74%]

Validation & Symmetry Integrity:
  - Standard test predictions are generated via fold-averaging across held-out folds,
    preserving mathematical symmetry between OOF evaluation and test inference.
================================================================================
"""

from pathlib import Path
import sys
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR

rk = lambda v: (rankdata(v, method="average") - 0.5) / len(v)

CATS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
]

def resolve_data_dir(cli_dir=None):
    """Robustly resolve competition input directory."""
    if cli_dir:
        p = Path(cli_dir)
        if p.is_dir() and (p / "train.csv").is_file():
            return p
    candidates = [
        Path(r"D:\gci\inpuy"),
        SCRIPT_DIR.parents[2] / "inpuy",
        SCRIPT_DIR.parents[1] / "inpuy",
        SCRIPT_DIR.parent / "inpuy",
        Path.cwd() / "inpuy",
        Path.cwd().parent / "inpuy",
    ]
    for c in candidates:
        if c.is_dir() and (c / "train.csv").is_file():
            return c
    raise FileNotFoundError("Could not find competition 'inpuy' directory containing train.csv and test.csv")

def build_feature_manifolds(df_train, df_test):
    """
    Constructs multi-architecture feature manifolds from raw train and test data.
    """
    t0 = time.time()
    n_train = len(df_train)
    n_test = len(df_test)
    df_all = pd.concat([df_train.drop(columns=["TARGET"]), df_test], ignore_index=True)

    # Core Features
    df_core = pd.DataFrame(index=df_all.index)
    inc = df_all["AMT_INCOME_TOTAL"].values.astype(np.float32)
    cred = df_all["AMT_CREDIT"].values.astype(np.float32)
    ann = df_all["AMT_ANNUITY"].values.astype(np.float32)
    goods = df_all["AMT_GOODS_PRICE"].fillna(df_all["AMT_CREDIT"]).values.astype(np.float32)

    df_core["AMT_INCOME_TOTAL"] = inc
    df_core["AMT_CREDIT"] = cred
    df_core["AMT_ANNUITY"] = ann
    df_core["AMT_GOODS_PRICE"] = goods
    df_core["CREDIT_INCOME_RATIO"] = cred / (inc + 1e-5)
    df_core["ANNUITY_INCOME_RATIO"] = ann / (inc + 1e-5)
    df_core["CREDIT_TERM"] = ann / (cred + 1e-5)
    df_core["GOODS_PRICE_DIFF"] = goods - cred

    emp_raw = df_all["DAYS_EMPLOYED"].values.astype(np.float32)
    days_emp = np.where(emp_raw == 365243.0, np.nan, emp_raw)
    df_core["DAYS_BIRTH"] = df_all["DAYS_BIRTH"].values.astype(np.float32)
    df_core["DAYS_EMPLOYED"] = days_emp
    df_core["DAYS_EMPLOYED_PERC"] = days_emp / (df_core["DAYS_BIRTH"].values + 1e-5)
    df_core["DAYS_REGISTRATION"] = df_all["DAYS_REGISTRATION"].values.astype(np.float32)
    df_core["DAYS_ID_PUBLISH"] = df_all["DAYS_ID_PUBLISH"].values.astype(np.float32)

    ext1 = df_all["EXT_SOURCE_1"].values.astype(np.float32)
    ext2 = df_all["EXT_SOURCE_2"].values.astype(np.float32)
    ext3 = df_all["EXT_SOURCE_3"].values.astype(np.float32)
    df_core["EXT_SOURCE_1"] = ext1
    df_core["EXT_SOURCE_2"] = ext2
    df_core["EXT_SOURCE_3"] = ext3

    ext_stack = np.column_stack([ext1, ext2, ext3])
    df_core["EXT_MEAN"] = np.nanmean(ext_stack, axis=1)
    df_core["EXT_STD"] = np.nanstd(ext_stack, axis=1)
    df_core["EXT_PROD"] = ext1 * ext2 * ext3
    df_core["EXT_W"] = (np.nan_to_num(ext1) * 2.0 + np.nan_to_num(ext2) * 3.0 + np.nan_to_num(ext3) * 4.0) / 9.0

    # Categorical Frequency & Codes
    for c in CATS:
        s = df_all[c].astype(str).fillna("MISSING")
        df_core[f"FREQ_{c}"] = s.map(s.value_counts(normalize=True)).values.astype(np.float32)
        df_core[f"CODE_{c}"] = s.astype("category").cat.codes.values.astype(np.float32)

    # One-Hot Demographics with JSON-safe feature names
    import re
    prefixes = [f"OH_{c}" for c in CATS]
    df_oh = pd.get_dummies(df_all[CATS], prefix=prefixes, drop_first=True, dtype=np.float32)
    df_oh.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df_oh.columns]
    df_all_comb = pd.concat([df_core, df_oh], axis=1)
    df_all_comb.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df_all_comb.columns]

    # Additional interaction features for v2 models
    q_ratio = (ann / (cred + 1e-5)).astype(np.float32)
    df_all_comb["Q_RATIO"] = q_ratio
    df_all_comb["Q_CRED_GOODS"] = (cred / (goods + 1e-5)).astype(np.float32)

    # Monotone feature subset
    mono_cols = [c for c in df_core.columns if not c.startswith("OH_")]
    df_mono = df_core[mono_cols].copy()

    # CatBoost frame
    cb_cols = [c for c in df_all.columns if c not in ("SK_ID_CURR", "TARGET")]
    cb_frame = df_all[cb_cols].copy()
    cb_cats = [c for c in CATS if c in cb_frame.columns]
    for c in cb_cats:
        cb_frame[c] = cb_frame[c].fillna("MISSING").astype(str)

    manifolds = {
        "X_core_tr": df_core.iloc[:n_train].reset_index(drop=True),
        "X_core_te": df_core.iloc[n_train:].reset_index(drop=True),
        "X_all_tr": df_all_comb.iloc[:n_train].reset_index(drop=True),
        "X_all_te": df_all_comb.iloc[n_train:].reset_index(drop=True),
        "X_mono_tr": df_mono.iloc[:n_train].reset_index(drop=True),
        "X_mono_te": df_mono.iloc[n_train:].reset_index(drop=True),
        "cb_tr": cb_frame.iloc[:n_train].reset_index(drop=True),
        "cb_te": cb_frame.iloc[n_train:].reset_index(drop=True),
        "cb_cats": cb_cats,
    }
    print(f"[+] Feature manifolds constructed in {time.time()-t0:.1f}s")
    return manifolds

def train_v55_suite_from_scratch(fast=False, output_dir=None, data_dir=None, use_full_retrain=False):
    t_start = time.time()
    data_path = resolve_data_dir(data_dir)
    print("=" * 80)
    print("  GCI World 2026: V55 Multi-Architecture Suite Generator (100% Raw Data)")
    print("=" * 80)
    print(f"[*] Ingesting raw competition datasets from: {data_path}")

    df_train = pd.read_csv(data_path / "train.csv")
    df_test = pd.read_csv(data_path / "test.csv")
    y_train = df_train["TARGET"].values.astype(np.int32)
    n_train = len(df_train)
    n_test = len(df_test)

    # Determine safe output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR / "smoke_test" if fast else OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "v55_members.npz"

    manifolds = build_feature_manifolds(df_train, df_test)
    n_splits = 3 if fast else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(n_train), y_train))
    print(f"\n[*] Training 7 Distinct Models across {n_splits}-Fold Stratified CV (fast={fast})...")

    ext_neg_names = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_MEAN", "EXT_W", "EXT_PROD"}
    mono_constraints = [-1 if c in ext_neg_names else 0 for c in manifolds["X_mono_tr"].columns]

    # Hardware acceleration check for CatBoost
    try:
        cb_probe = CatBoostClassifier(iterations=5, task_type="GPU", verbose=False)
        cb_probe.fit(pd.DataFrame({"a": ["x", "y"] * 5, "b": [1, 2] * 5}), np.array([0, 1] * 5), cat_features=["a"])
        cb_task = "GPU"
        print("[*] CatBoost Compute Device: NVIDIA CUDA GPU")
    except Exception:
        cb_task = "CPU"
        print("[*] CatBoost Compute Device: CPU")

    models_spec = [
        ("qf_shallow", manifolds["X_core_tr"], manifolds["X_core_te"],
         {"objective": "binary", "learning_rate": 0.05, "num_leaves": 7, "max_depth": 3,
          "min_child_samples": 150, "feature_fraction": 0.5, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "extra_trees": True, "n_jobs": 4, "verbose": -1, "seed": 42},
         400 if fast else 1300),

        ("qf_reg", manifolds["X_core_tr"], manifolds["X_core_te"],
         {"objective": "binary", "learning_rate": 0.03, "num_leaves": 11,
          "min_child_samples": 300, "feature_fraction": 0.4, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 40, "extra_trees": True, "n_jobs": 4, "verbose": -1, "seed": 43},
         500 if fast else 1939),

        ("v1_allfam_oh", manifolds["X_all_tr"], manifolds["X_all_te"],
         {"objective": "binary", "learning_rate": 0.04, "num_leaves": 15,
          "min_child_samples": 150, "feature_fraction": 0.3, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "extra_trees": True, "n_jobs": 4, "verbose": -1, "seed": 44},
         450 if fast else 1563),

        ("v2_allfam_oh", manifolds["X_all_tr"], manifolds["X_all_te"],
         {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15,
          "min_child_samples": 150, "feature_fraction": 0.3, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "extra_trees": False, "n_jobs": 4, "verbose": -1, "seed": 45},
         450 if fast else 1500),

        ("v2_mono", manifolds["X_mono_tr"], manifolds["X_mono_te"],
         {"objective": "binary", "learning_rate": 0.02, "num_leaves": 15,
          "min_child_samples": 150, "feature_fraction": 0.5, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "monotone_constraints": mono_constraints,
          "extra_trees": True, "n_jobs": 4, "verbose": -1, "seed": 46},
         450 if fast else 1571),

        ("deep_et", manifolds["X_all_tr"], manifolds["X_all_te"],
         {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31,
          "min_child_samples": 80, "feature_fraction": 0.3, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "extra_trees": True, "n_jobs": 4, "verbose": -1, "seed": 47},
         400 if fast else 1200),
    ]

    bundle_dict = {}
    oof_metrics = {}

    for idx, (name, X_tr, X_te, params, rounds) in enumerate(models_spec, 1):
        print(f"\n[Model {idx}/7] Training LightGBM Member: {name} ({rounds} rounds, {X_tr.shape[1]} features)...")
        t_m = time.time()
        oof_pred = np.zeros(n_train, dtype=np.float32)
        test_fold_pred = np.zeros(n_test, dtype=np.float64)

        for fold, (trn_idx, val_idx) in enumerate(folds, 1):
            dtrain = lgb.Dataset(X_tr.iloc[trn_idx], label=y_train[trn_idx])
            dval = lgb.Dataset(X_tr.iloc[val_idx], label=y_train[val_idx])
            p_fold = dict(params)
            p_fold["seed"] = params["seed"] + fold

            model = lgb.train(p_fold, dtrain, num_boost_round=rounds, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(30, verbose=False)])

            oof_pred[val_idx] = model.predict(X_tr.iloc[val_idx])
            test_fold_pred += model.predict(X_te) / float(n_splits)

        if use_full_retrain:
            full_preds = []
            for sd in [42, 7]:
                m_full = lgb.train({**params, "seed": sd}, lgb.Dataset(X_tr, y_train), num_boost_round=int(rounds * 1.1))
                full_preds.append(rk(m_full.predict(X_te)))
            final_test = rk(0.50 * rk(test_fold_pred) + 0.50 * np.mean(full_preds, axis=0))
        else:
            # Symmetric fold-averaged test prediction
            final_test = rk(test_fold_pred)

        auc = roc_auc_score(y_train, oof_pred)
        oof_metrics[name] = auc
        print(f"      -> {name:15s} Complete | OOF AUC: {auc:.6f} ({time.time()-t_m:.1f}s)")
        bundle_dict[f"oof_{name}"] = oof_pred
        bundle_dict[f"test_{name}"] = final_test.astype(np.float64)

    # Model 7: CatBoost Product Specialist
    print(f"\n[Model 7/7] Training CatBoost Product Specialist (cb_product)...")
    t_cb = time.time()
    oof_cb = np.zeros(n_train, dtype=np.float32)
    test_cb = np.zeros(n_test, dtype=np.float32)
    cb_tr = manifolds["cb_tr"]
    cb_te = manifolds["cb_te"]
    cb_cats = manifolds["cb_cats"]
    cb_rounds = 200 if fast else 400

    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        cb_model = CatBoostClassifier(
            iterations=cb_rounds, depth=6, learning_rate=0.045, loss_function="Logloss",
            eval_metric="AUC", l2_leaf_reg=8.0, random_seed=42 + fold, task_type=cb_task, verbose=False
        )
        cb_model.fit(
            cb_tr.iloc[trn_idx], y_train[trn_idx], cat_features=cb_cats,
            eval_set=(cb_tr.iloc[val_idx], y_train[val_idx]), early_stopping_rounds=30, verbose=False
        )
        oof_cb[val_idx] = cb_model.predict_proba(cb_tr.iloc[val_idx])[:, 1]
        test_cb += cb_model.predict_proba(cb_te)[:, 1].astype(np.float32) / float(n_splits)

    auc_cb = roc_auc_score(y_train, oof_cb)
    oof_metrics["cb_product"] = auc_cb
    print(f"      -> cb_product     Complete | OOF AUC: {auc_cb:.6f} ({time.time()-t_cb:.1f}s)")
    bundle_dict["oof_cb_product"] = oof_cb
    bundle_dict["test_cb_product"] = rk(test_cb).astype(np.float32)

    np.savez_compressed(out_file, **bundle_dict)
    print("\n" + "=" * 80)
    print(f"  V55 Multi-Architecture Suite Bundle Generated Successfully!")
    print(f"  Output Artifact: {out_file} ({out_file.stat().st_size:,} bytes)")
    print(f"  Total Runtime:   {time.time()-t_start:.1f}s")
    print("=" * 80)
    return bundle_dict

def main():
    parser = argparse.ArgumentParser(description="Generate V55 Multi-Architecture Suite strictly from raw data")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode")
    parser.add_argument("--data_dir", type=str, default=None, help="Explicit path to raw competition data ('inpuy')")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save bundle")
    parser.add_argument("--use_full_retrain", action="store_true", help="Enable dual full-data retraining for LightGBM test")
    args = parser.parse_args()

    train_v55_suite_from_scratch(fast=args.fast, output_dir=args.output_dir, data_dir=args.data_dir, use_full_retrain=args.use_full_retrain)

if __name__ == "__main__":
    main()
