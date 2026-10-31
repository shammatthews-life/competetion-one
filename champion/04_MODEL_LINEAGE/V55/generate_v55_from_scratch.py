"""
recreate_v55_members.py
================================================================================
GCI World 2026 - 100% Self-Contained V55 Multi-Architecture Generator
================================================================================
Target Artifact : cache/v55_members.npz
Destination     : D:/gci/gemini/champion/cacherecreation/
Rule Compliance : 100% Strictly Independent (train.csv and test.csv ONLY)

Architecture:
  Trains all 7 distinct V55 multi-architecture models strictly from raw data:
    1. qf_shallow   : LightGBM Shallow Reg (num_leaves=7, max_depth=3, lr=0.05, extra_trees=True)
    2. qf_reg       : LightGBM Conservative Reg (num_leaves=11, min_child_samples=300, lambda_l2=40)
    3. v1_allfam_oh : LightGBM All-Family OH (num_leaves=15, feature_fraction=0.3, extra_trees=True)
    4. v2_allfam_oh : LightGBM V2 Demographics OH (num_leaves=15, feature_fraction=0.3)
    5. v2_mono      : LightGBM Monotone Shallow-Reg (genuine monotone_constraints on EXT features)
    6. deep_et      : LightGBM Deep ExtraTrees (num_leaves=31, extra_trees=True)
    7. cb_product   : CatBoost Oblivious Product Specialist (depth=6, l2=8.0, categorical features)
  Includes dual-seed full-data retraining and uniform rank projection for test predictions.
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
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "inpuy"
OUTPUT_DIR = SCRIPT_DIR

rk = lambda v: (rankdata(v, method="average") - 0.5) / len(v)


def load_raw_data(data_dir=DATA_DIR):
    """Load only competition train.csv and test.csv."""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Required files {train_path} and {test_path} not found!")

    print(f"[*] Ingesting raw train dataset: {train_path}")
    print(f"[*] Ingesting raw test dataset : {test_path}")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def build_feature_manifolds(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Constructs the 5 distinct feature manifolds strictly from raw train and test data:
      - Xco: Core features + One-Hot encodings
      - Xao: Core + Extended ratios + One-Hot encodings
      - Xaok: Core + Extended + Demographic interaction terms
      - Xck: Monotonic feature subset
      - cb_tr / cb_te: CatBoost product-cell matrix
    """
    t0 = time.time()
    print("[*] Engineering multi-view feature manifolds directly in memory from raw data...")

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

    # 1. Clean & Sentinels
    F_core = {}
    F_core["DAYS_EMPLOYED_CLEAN"] = days_emp.astype(np.float32)
    F_core["IS_RETIRED"] = (emp_raw == 365243.0).astype(np.float32)
    F_core["MISS_CNT"] = df_all.isna().sum(axis=1).values.astype(np.float32)

    # 2. Demographics & Relative Event Ages
    F_core["AGE_YEARS"] = (-birth / 365.25).astype(np.float32)
    F_core["EMPLOYED_YEARS"] = (-days_emp / 365.25).astype(np.float32)
    F_core["EMP_TO_AGE"] = (days_emp / (birth + 1.0)).astype(np.float32)
    F_core["REG_TO_AGE"] = (reg / (birth + 1.0)).astype(np.float32)
    F_core["ID_TO_AGE"] = (id_pub / (birth + 1.0)).astype(np.float32)
    F_core["PHONE_TO_AGE"] = (phone / (birth + 1.0)).astype(np.float32)

    # 3. Financial Solvency & Banking Ratios
    term = cred / (ann + 1.0)
    F_core["CREDIT_TERM"] = term.astype(np.float32)
    F_core["EAR"] = np.clip(24.0 * (ann * term - cred) / (cred * (term + 1.0) + 1.0), -1.0, 5.0).astype(np.float32)
    surplus = inc - 12.0 * ann
    F_core["SURPLUS"] = surplus.astype(np.float32)
    F_core["SURPLUS_TO_INC"] = (surplus / (inc + 1.0)).astype(np.float32)
    F_core["TRUE_COST"] = (ann * term - cred).astype(np.float32)
    F_core["TRUE_COST_RATIO"] = ((ann * term - cred) / (cred + 1.0)).astype(np.float32)
    F_core["LTV"] = np.clip(cred / (goods + 1.0), 0.0, 5.0).astype(np.float32)
    F_core["DTI"] = np.clip(cred / (inc + 1.0), 0.0, 20.0).astype(np.float32)
    F_core["DEBT_BURDEN"] = np.clip(ann / (inc / 12.0 + 1.0), 0.0, 5.0).astype(np.float32)
    F_core["DSCR"] = np.clip(((inc / 12.0) * 0.65) / (ann + 1.0), 0.0, 20.0).astype(np.float32)
    F_core["SOLVENCY"] = ((inc / 12.0) / (ann + 1.0)).astype(np.float32)
    F_core["BUFFER_MONTHS"] = np.clip(surplus / (12.0 * ann + 1.0), -10.0, 50.0).astype(np.float32)
    F_core["INC_PER_FAM"] = (inc / (df_all["CNT_FAM_MEMBERS"].fillna(1.0) + 1e-3)).values.astype(np.float32)

    # Log transforms
    F_core["LOG_INCOME"] = np.log1p(inc).astype(np.float32)
    F_core["LOG_CREDIT"] = np.log1p(cred).astype(np.float32)
    F_core["LOG_ANNUITY"] = np.log1p(ann).astype(np.float32)
    F_core["LOG_GOODS"] = np.log1p(goods).astype(np.float32)

    # 4. External Sources Triad
    ext_stack = np.column_stack([e1, e2, e3])
    F_core["EXT_MEAN"] = np.nanmean(ext_stack, axis=1).astype(np.float32)
    F_core["EXT_MIN"] = np.nanmin(ext_stack, axis=1).astype(np.float32)
    F_core["EXT_MAX"] = np.nanmax(ext_stack, axis=1).astype(np.float32)
    F_core["EXT_STD"] = np.nan_to_num(np.nanstd(ext_stack, axis=1), nan=0.0).astype(np.float32)
    F_core["EXT_W"] = (np.nan_to_num(0.2 * e1) + np.nan_to_num(0.4 * e2) + np.nan_to_num(0.4 * e3)).astype(np.float32)
    F_core["EXT_DIFF_12"] = (e1 - e2).astype(np.float32)
    F_core["EXT_DIFF_23"] = (e2 - e3).astype(np.float32)
    F_core["EXT_DIFF_13"] = (e1 - e3).astype(np.float32)
    F_core["EXT_PROD"] = (np.nan_to_num(e1, nan=0.5) * np.nan_to_num(e2, nan=0.5) * np.nan_to_num(e3, nan=0.5)).astype(np.float32)

    # 5. Merton Structural Distance-to-Default
    T_years = np.clip(term / 12.0, 0.1, 10.0)
    disp_ratio = np.maximum(0.01, (inc / 12.0) / (ann + 1.0))
    for mu, sigma, tag in [(0.02, 0.20, "20"), (0.03, 0.25, "25")]:
        dd = (np.log(disp_ratio) + (mu - 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        F_core[f"MERTON_DD_{tag}"] = np.clip(dd, -10.0, 10.0).astype(np.float32)

    # 6. Raw Numeric Features
    raw_num = df_all.select_dtypes(include=[np.number]).drop(columns=["SK_ID_CURR"], errors="ignore")
    for c in raw_num.columns:
        if c not in F_core:
            F_core[c] = raw_num[c].values.astype(np.float32)

    df_core = pd.DataFrame(F_core)
    df_oh = pd.get_dummies(df_all[CATS].astype("category"), dummy_na=True, dtype=np.float32)
    df_oh.columns = [f"OH_{i}_{c}" for i, c in enumerate(df_oh.columns)]

    # Manifold 1: Xco (Core OH)
    Xco_all = pd.concat([df_core, df_oh], axis=1).replace([np.inf, -np.inf], np.nan)
    Xco_all.columns = [c.replace(':', '_').replace('|', '_').replace(' ', '_').replace('-', '_') for c in Xco_all.columns]

    # Manifold 2: Xao (All-Family OH with interaction terms)
    F_ext = {}
    for c in ["OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE"]:
        grp_inc = df_all.groupby(c)["AMT_INCOME_TOTAL"].transform("mean").values
        F_ext[f"DEV_INC_{c[:4]}"] = (inc - grp_inc).astype(np.float32)
        F_ext[f"RAT_INC_{c[:4]}"] = (inc / (grp_inc + 1.0)).astype(np.float32)
    Xao_all = pd.concat([Xco_all, pd.DataFrame(F_ext)], axis=1)

    # Manifold 3: Xaok (V2 OH with product schedule ratios)
    q_ratio = (ann / (cred + 1e-5)).astype(np.float32)
    F_v2 = {"Q_RATIO": q_ratio, "Q_CRED_GOODS": (cred / (goods + 1e-5)).astype(np.float32)}
    Xaok_all = pd.concat([Xao_all, pd.DataFrame(F_v2)], axis=1)

    # Manifold 4: Xck (Monotone Feature Set)
    mono_cols = [c for c in df_core.columns if not c.startswith("OH_")]
    Xck_all = df_core[mono_cols].copy()

    # Manifold 5: CatBoost Product-Cell Frame
    cb_cols = [c for c in df_all.columns if c not in ("SK_ID_CURR", "TARGET")]
    cb_frame = df_all[cb_cols].copy()
    cb_cats = [c for c in CATS if c in cb_frame.columns]
    for c in cb_cats:
        cb_frame[c] = cb_frame[c].fillna("MISSING").astype(str)

    manifolds = {
        "Xco": Xco_all.iloc[:n_train].reset_index(drop=True),
        "Xcot": Xco_all.iloc[n_train:].reset_index(drop=True),
        "Xao": Xao_all.iloc[:n_train].reset_index(drop=True),
        "Xaot": Xao_all.iloc[n_train:].reset_index(drop=True),
        "Xaok": Xaok_all.iloc[:n_train].reset_index(drop=True),
        "Xaokt": Xaok_all.iloc[n_train:].reset_index(drop=True),
        "Xck": Xck_all.iloc[:n_train].reset_index(drop=True),
        "Xckt": Xck_all.iloc[n_train:].reset_index(drop=True),
        "cb_tr": cb_frame.iloc[:n_train].reset_index(drop=True),
        "cb_te": cb_frame.iloc[n_train:].reset_index(drop=True),
        "cb_cats": cb_cats,
    }
    print(f"[+] Multi-manifold construction complete in {time.time()-t0:.1f}s")
    return manifolds


def train_v55_suite_from_scratch(fast=False, output_path=None):
    """
    Train all 7 distinct V55 multi-architecture models strictly from raw data.
    """
    t_start = time.time()
    df_train, df_test = load_raw_data()
    y_train = df_train["TARGET"].values.astype(np.int32)
    n_train = len(df_train)
    n_test = len(df_test)

    if output_path is None:
        output_path = OUTPUT_DIR / "v55_members.npz"

    manifolds = build_feature_manifolds(df_train, df_test)
    n_splits = 3 if fast else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(n_train), y_train))
    print(f"\n[*] Training 7 Distinct Models across {n_splits}-Fold Stratified CV (fast={fast})...")

    ext_neg_names = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_MEAN", "EXT_W", "EXT_PROD"}
    mono_constraints = [-1 if c in ext_neg_names else 0 for c in manifolds["Xck"].columns]

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
        ("qf_shallow", manifolds["Xco"], manifolds["Xcot"],
         {"objective": "binary", "learning_rate": 0.05, "num_leaves": 7, "max_depth": 3,
          "min_child_samples": 150, "feature_fraction": 0.5, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "extra_trees": True, "n_jobs": -1, "verbose": -1, "seed": 42},
         400 if fast else 1300),

        ("qf_reg", manifolds["Xco"], manifolds["Xcot"],
         {"objective": "binary", "learning_rate": 0.03, "num_leaves": 11,
          "min_child_samples": 300, "feature_fraction": 0.4, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 40, "extra_trees": True, "n_jobs": -1, "verbose": -1, "seed": 43},
         500 if fast else 1939),

        ("v1_allfam_oh", manifolds["Xao"], manifolds["Xaot"],
         {"objective": "binary", "learning_rate": 0.04, "num_leaves": 15,
          "min_child_samples": 150, "feature_fraction": 0.3, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "extra_trees": True, "n_jobs": -1, "verbose": -1, "seed": 44},
         350 if fast else 881),

        ("v2_allfam_oh", manifolds["Xaok"], manifolds["Xaokt"],
         {"objective": "binary", "learning_rate": 0.04, "num_leaves": 15,
          "min_child_samples": 150, "feature_fraction": 0.3, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "extra_trees": True, "n_jobs": -1, "verbose": -1, "seed": 45},
         350 if fast else 881),

        ("v2_mono", manifolds["Xck"], manifolds["Xckt"],
         {"objective": "binary", "learning_rate": 0.02, "num_leaves": 15,
          "min_child_samples": 150, "feature_fraction": 0.5, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "monotone_constraints": mono_constraints,
          "extra_trees": True, "n_jobs": -1, "verbose": -1, "seed": 46},
         450 if fast else 1571),

        ("deep_et", manifolds["Xao"], manifolds["Xaot"],
         {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31,
          "min_child_samples": 80, "feature_fraction": 0.3, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 20, "extra_trees": True, "n_jobs": -1, "verbose": -1, "seed": 47},
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

        # Dual-seed full-data retraining
        full_preds = []
        for sd in [42, 7]:
            m_full = lgb.train({**params, "seed": sd}, lgb.Dataset(X_tr, y_train), num_boost_round=int(rounds * 1.1))
            full_preds.append(rk(m_full.predict(X_te)))

        # 50/50 blend of fold-average and full-data models + uniform rank projection
        final_test = rk(0.50 * rk(test_fold_pred) + 0.50 * np.mean(full_preds, axis=0))

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
    print(f"      -> {'cb_product':15s} Complete | OOF AUC: {auc_cb:.6f} ({time.time()-t_cb:.1f}s)")
    bundle_dict["oof_cb_product"] = oof_cb
    bundle_dict["test_cb_product"] = test_cb

    keys_order = [
        "oof_qf_shallow", "oof_qf_reg", "oof_v1_allfam_oh", "oof_v2_allfam_oh",
        "oof_v2_mono", "oof_deep_et", "oof_cb_product",
        "test_qf_shallow", "test_qf_reg", "test_v1_allfam_oh", "test_v2_allfam_oh",
        "test_v2_mono", "test_deep_et", "test_cb_product"
    ]
    ordered_dict = {k: bundle_dict[k] for k in keys_order}
    np.savez_compressed(output_path, **ordered_dict)
    print(f"\n[+] Saved complete V55 bundle to: {output_path} ({Path(output_path).stat().st_size:,} bytes)")
    print(f"    Total Runtime: {time.time()-t_start:.1f}s")
    return bundle_dict


def main():
    parser = argparse.ArgumentParser(description="Generate v55_members.npz strictly from raw data")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode")
    parser.add_argument("--output", type=str, default=None, help="Output path for v55_members.npz")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else OUTPUT_DIR / "v55_members.npz"
    train_v55_suite_from_scratch(fast=args.fast, output_path=out_p)


if __name__ == "__main__":
    main()
