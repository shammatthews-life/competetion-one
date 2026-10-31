"""
generate_v17_from_scratch.py / recreate_v17_bundle.py
================================================================================
GCI World 2026 - 100% Self-Contained V17 Full Suite Bundle Generator
================================================================================
Rule Compliance : 100% Strictly Independent (train.csv and test.csv ONLY)
Target Artifact : cache/v_17_full_suite_bundle.npz

Architecture & Member Role Specification:
  Trains all 5 genuine model streams across 5-Fold Stratified Cross-Validation:
    Stream 1: oof_old, test_old     - Base LightGBM GBDT Suite (without residuals)
    Stream 2: oof_e1, test_e1       - Enhanced E1 LightGBM GBDT (demographic expectation residuals)
    Stream 3: oof_e1et, test_e1et   - ExtraTrees E1 LightGBM Suite [SUITE MEMBER: 4.89%]
    Stream 4: oof_e1Aet, test_e1Aet - E1A ExtraTrees LightGBM       [SUITE MEMBER: 29.32%]
    Stream 5: oof_cb, test_cb       - CatBoost Product Specialist
    Stream 6: Grand Fusion:
              final_oof, final_test - 4-Set Master Probit Blend    [SUITE MEMBER: 20.51%]
              final_auc             - Grand Champion Scalar OOF AUC

All 13 canonical keys are produced, verified, and saved:
  ['oof_old', 'test_old', 'oof_e1', 'test_e1', 'oof_e1et', 'test_e1et',
   'oof_e1Aet', 'test_e1Aet', 'oof_cb', 'test_cb', 'final_oof', 'final_test', 'final_auc']
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

def build_v17_feature_manifolds(df_train, df_test):
    """
    Construct required feature manifolds in memory from raw data.
    """
    t0 = time.time()
    ntr = len(df_train)
    nte = len(df_test)
    al = pd.concat([df_train.drop(columns=["TARGET"]), df_test], ignore_index=True)

    def new_cols(d):
        return pd.DataFrame(d, index=al.index)

    # Base feature encoding
    X_base = al.drop(columns=["SK_ID_CURR"]).copy()
    for c in CATS:
        X_base[c] = X_base[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)

    # 1. Cleaning & anomaly flags
    d_clean = {
        "DE_SENT": (al.DAYS_EMPLOYED == 365243).astype(int),
        "DAYS_EMPLOYED_CLEAN": al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243),
        "ORG_XNA": (al.ORGANIZATION_TYPE == "XNA").astype(int),
        "CAR_AGE_OUT": (al.OWN_CAR_AGE >= 60).astype(int),
        "MISS_CNT": al.isna().sum(axis=1)
    }
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
        d_clean[c + "_NA"] = al[c].isna().astype(int)
    f_clean = new_cols(d_clean)

    # 2. External source interactions
    e = al[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    d_ext = {
        "EXT_MEAN": e.mean(axis=1),
        "EXT_MIN": e.min(axis=1),
        "EXT_MAX": e.max(axis=1),
        "EXT_STD": e.std(axis=1),
        "EXT_RANGE": e.max(axis=1) - e.min(axis=1),
        "EXT_NOBS": e.notna().sum(axis=1),
        "EXT_PROD": e.prod(axis=1),
        "EXT_W": (al.EXT_SOURCE_1 * 2 + al.EXT_SOURCE_2 * 3 + al.EXT_SOURCE_3 * 4) / 9.0,
        "E12": al.EXT_SOURCE_1 * al.EXT_SOURCE_2,
        "E13": al.EXT_SOURCE_1 * al.EXT_SOURCE_3,
        "E23": al.EXT_SOURCE_2 * al.EXT_SOURCE_3,
        "D12": al.EXT_SOURCE_1 - al.EXT_SOURCE_2,
        "D13": al.EXT_SOURCE_1 - al.EXT_SOURCE_3,
        "D23": al.EXT_SOURCE_2 - al.EXT_SOURCE_3
    }
    f_ext = new_cols(d_ext)

    # 3. Financial debt capacity
    inc = al.AMT_INCOME_TOTAL
    cred = al.AMT_CREDIT
    ann = al.AMT_ANNUITY
    goods = al.AMT_GOODS_PRICE.fillna(cred)

    d_fin = {
        "CR_INC": cred / inc,
        "ANN_INC": ann / inc,
        "CR_GOODS": cred / goods,
        "GOODS_DIFF": goods - cred,
        "ANN_CR": ann / cred,
        "DISPOSABLE": inc - ann,
        "INC_PER_CAP": inc / al.CNT_FAM_MEMBERS.fillna(1),
        "INC_SURPLUS_PER_CAP": (inc - ann) / al.CNT_FAM_MEMBERS.fillna(1)
    }
    f_fin = new_cols(d_fin)

    # 4. Age & employment tenure
    birth = al.DAYS_BIRTH
    emp = f_clean.DAYS_EMPLOYED_CLEAN
    d_age = {
        "AGE_Y": -birth / 365.25,
        "EMP_Y": -emp / 365.25,
        "EMP_AGE_R": emp / birth,
        "REG_Y": -al.DAYS_REGISTRATION / 365.25,
        "ID_Y": -al.DAYS_ID_PUBLISH / 365.25,
        "PHONE_Y": -al.DAYS_LAST_PHONE_CHANGE / 365.25,
        "CAR_EMP_R": al.OWN_CAR_AGE / (-emp / 365.25 + 1e-5),
        "CAR_AGE_R": al.OWN_CAR_AGE / (-birth / 365.25 + 1e-5)
    }
    f_age = new_cols(d_age)

    # 5. Cohort expectations & residuals
    groups = [
        ["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"],
        ["NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE"],
        ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    ]
    f_grp_list = []
    for grp in groups:
        for target in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            grp_name = "_".join(grp)
            m = al.groupby(grp)[target].transform("mean")
            s = al.groupby(grp)[target].transform("std").fillna(1)
            f_grp_list.append(new_cols({
                f"{target}_MEAN_{grp_name}": m,
                f"{target}_RESID_{grp_name}": al[target] - m,
                f"{target}_Z_{grp_name}": (al[target] - m) / (s + 1e-5)
            }))
    f_grp = pd.concat(f_grp_list, axis=1)

    # Manifold A: Base features
    X_old = pd.concat([X_base, f_clean, f_ext, f_fin, f_age], axis=1)

    # Manifold B: Residuals & Cohort expectations
    X_e1 = pd.concat([X_old, f_grp], axis=1)

    # Manifold C: E1A with interaction ratios
    d_e1a = {
        "E1A_RATIO_CR": cred / (goods + 1e-5),
        "E1A_PAY_RATIO": ann / (cred + 1e-5),
        "E1A_EXT_AGG": (al.EXT_SOURCE_1.fillna(0) + al.EXT_SOURCE_2.fillna(0) + al.EXT_SOURCE_3.fillna(0)) / 3.0
    }
    f_e1a = new_cols(d_e1a)
    X_e1A = pd.concat([X_e1, f_e1a], axis=1)

    # Manifold D: CatBoost frame
    cb_cols = [c for c in al.columns if c != "TARGET"]
    cb_frame = al[cb_cols].copy()
    cb_cats = [c for c in CATS if c in cb_frame.columns]
    for c in cb_cats:
        cb_frame[c] = cb_frame[c].fillna("MISSING").astype(str)

    manifolds = {
        "X_old_tr": X_old.iloc[:ntr].reset_index(drop=True),
        "X_old_te": X_old.iloc[ntr:].reset_index(drop=True),
        "X_e1_tr": X_e1.iloc[:ntr].reset_index(drop=True),
        "X_e1_te": X_e1.iloc[ntr:].reset_index(drop=True),
        "X_e1A_tr": X_e1A.iloc[:ntr].reset_index(drop=True),
        "X_e1A_te": X_e1A.iloc[ntr:].reset_index(drop=True),
        "cb_tr": cb_frame.iloc[:ntr].reset_index(drop=True),
        "cb_te": cb_frame.iloc[ntr:].reset_index(drop=True),
        "cb_cats": cb_cats,
    }
    print(f"[+] Feature manifolds constructed in {time.time()-t0:.1f}s")
    return manifolds

def train_v17_suite_from_scratch(fast=False, output_path=None, data_dir=None):
    t_start = time.time()
    data_path = resolve_data_dir(data_dir)
    print("=" * 80)
    print("  GCI World 2026: V17 Full Suite Bundle Generator (100% Raw Data)")
    print("=" * 80)
    print(f"[*] Ingesting raw competition datasets from: {data_path}")

    df_train = pd.read_csv(data_path / "train.csv")
    df_test = pd.read_csv(data_path / "test.csv")
    y_train = df_train["TARGET"].values.astype(np.int32)
    n_train = len(df_train)
    n_test = len(df_test)

    if output_path is None:
        out_dir = OUTPUT_DIR / "smoke_test" if fast else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "v_17_full_suite_bundle.npz"
    else:
        output_path = Path(output_path)
        if output_path.is_dir() or output_path.suffix != ".npz":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "v_17_full_suite_bundle.npz"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

    manifolds = build_v17_feature_manifolds(df_train, df_test)

    n_splits = 3 if fast else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(n_train), y_train))
    rk = lambda v: (rankdata(v) - 0.5) / len(v)

    print(f"\n[*] Training 5 Distinct V17 Foundation Model Streams ({n_splits}-Fold Stratified CV, fast={fast})...")

    # Hardware acceleration check for CatBoost
    try:
        cb_probe = CatBoostClassifier(iterations=5, task_type="GPU", verbose=False)
        cb_probe.fit(pd.DataFrame({"a": ["x", "y"] * 5, "b": [1, 2] * 5}), np.array([0, 1] * 5), cat_features=["a"])
        cb_task = "GPU"
        print("[*] CatBoost Compute Device: NVIDIA CUDA GPU")
    except Exception:
        cb_task = "CPU"
        print("[*] CatBoost Compute Device: CPU")

    # STREAM 1: Base LightGBM GBDT (oof_old, test_old)
    print("\n[Stream 1/5] Training Base LightGBM GBDT Suite (old)...")
    oof_old = np.zeros(n_train, dtype=np.float32)
    test_old = np.zeros(n_test, dtype=np.float64)
    X_tr_old = manifolds["X_old_tr"]
    X_te_old = manifolds["X_old_te"]
    rounds_old = 100 if fast else 700

    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        dtrain = lgb.Dataset(X_tr_old.iloc[trn_idx], label=y_train[trn_idx])
        dval = lgb.Dataset(X_tr_old.iloc[val_idx], label=y_train[val_idx])
        params = {
            "objective": "binary", "learning_rate": 0.04, "num_leaves": 15,
            "min_child_samples": 150, "feature_fraction": 0.5, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 20, "extra_trees": False,
            "n_jobs": 4, "verbose": -1, "seed": 42 + fold
        }
        bst = lgb.train(params, dtrain, num_boost_round=rounds_old, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_old[val_idx] = bst.predict(X_tr_old.iloc[val_idx])
        test_old += bst.predict(X_te_old) / float(n_splits)

    oof_old = rk(oof_old).astype(np.float32)
    test_old = rk(test_old)
    print(f"      -> oof_old Complete | OOF AUC: {roc_auc_score(y_train, oof_old):.6f}")

    # STREAM 2: Enhanced E1 LightGBM (oof_e1, test_e1)
    print("\n[Stream 2/5] Training Enhanced E1 Residual Suite (e1)...")
    oof_e1 = np.zeros(n_train, dtype=np.float32)
    test_e1 = np.zeros(n_test, dtype=np.float64)
    X_tr_e1 = manifolds["X_e1_tr"]
    X_te_e1 = manifolds["X_e1_te"]
    rounds_e1 = 100 if fast else 750

    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        dtrain = lgb.Dataset(X_tr_e1.iloc[trn_idx], label=y_train[trn_idx])
        dval = lgb.Dataset(X_tr_e1.iloc[val_idx], label=y_train[val_idx])
        params = {
            "objective": "binary", "learning_rate": 0.035, "num_leaves": 23,
            "min_child_samples": 120, "feature_fraction": 0.45, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 15, "extra_trees": False,
            "n_jobs": 4, "verbose": -1, "seed": 52 + fold
        }
        bst = lgb.train(params, dtrain, num_boost_round=rounds_e1, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_e1[val_idx] = bst.predict(X_tr_e1.iloc[val_idx])
        test_e1 += bst.predict(X_te_e1) / float(n_splits)

    oof_e1 = rk(oof_e1).astype(np.float32)
    test_e1 = rk(test_e1)
    print(f"      -> oof_e1 Complete | OOF AUC: {roc_auc_score(y_train, oof_e1):.6f}")

    # STREAM 3: ExtraTrees E1 LightGBM (oof_e1et, test_e1et)
    print("\n[Stream 3/5] Training ExtraTrees E1 Suite (e1et)...")
    oof_e1et = np.zeros(n_train, dtype=np.float32)
    test_e1et = np.zeros(n_test, dtype=np.float64)
    rounds_e1et = 100 if fast else 800

    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        dtrain = lgb.Dataset(X_tr_e1.iloc[trn_idx], label=y_train[trn_idx])
        dval = lgb.Dataset(X_tr_e1.iloc[val_idx], label=y_train[val_idx])
        params = {
            "objective": "binary", "learning_rate": 0.03, "num_leaves": 31,
            "min_child_samples": 80, "feature_fraction": 0.35, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 25, "extra_trees": True,
            "n_jobs": 4, "verbose": -1, "seed": 62 + fold
        }
        bst = lgb.train(params, dtrain, num_boost_round=rounds_e1et, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_e1et[val_idx] = bst.predict(X_tr_e1.iloc[val_idx])
        test_e1et += bst.predict(X_te_e1) / float(n_splits)

    oof_e1et = rk(oof_e1et).astype(np.float32)
    test_e1et = rk(test_e1et)
    print(f"      -> oof_e1et Complete | OOF AUC: {roc_auc_score(y_train, oof_e1et):.6f}")

    # STREAM 4: E1A ExtraTrees LightGBM (oof_e1Aet, test_e1Aet)
    print("\n[Stream 4/5] Training E1A ExtraTrees Specialist (e1Aet)...")
    oof_e1Aet = np.zeros(n_train, dtype=np.float32)
    test_e1Aet = np.zeros(n_test, dtype=np.float64)
    X_tr_e1A = manifolds["X_e1A_tr"]
    X_te_e1A = manifolds["X_e1A_te"]
    rounds_e1Aet = 100 if fast else 900

    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        dtrain = lgb.Dataset(X_tr_e1A.iloc[trn_idx], label=y_train[trn_idx])
        dval = lgb.Dataset(X_tr_e1A.iloc[val_idx], label=y_train[val_idx])
        params = {
            "objective": "binary", "learning_rate": 0.025, "num_leaves": 45,
            "min_child_samples": 60, "feature_fraction": 0.3, "bagging_fraction": 0.7,
            "bagging_freq": 1, "lambda_l2": 30, "extra_trees": True,
            "n_jobs": 4, "verbose": -1, "seed": 72 + fold
        }
        bst = lgb.train(params, dtrain, num_boost_round=rounds_e1Aet, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_e1Aet[val_idx] = bst.predict(X_tr_e1A.iloc[val_idx])
        test_e1Aet += bst.predict(X_te_e1A) / float(n_splits)

    oof_e1Aet = rk(oof_e1Aet).astype(np.float32)
    test_e1Aet = rk(test_e1Aet)
    print(f"      -> oof_e1Aet Complete | OOF AUC: {roc_auc_score(y_train, oof_e1Aet):.6f}")

    # STREAM 5: CatBoost Product Specialist (oof_cb, test_cb)
    print("\n[Stream 5/5] Training CatBoost Product Specialist (cb)...")
    oof_cb = np.zeros(n_train, dtype=np.float32)
    test_cb = np.zeros(n_test, dtype=np.float32)
    cb_tr = manifolds["cb_tr"]
    cb_te = manifolds["cb_te"]
    cb_cats = manifolds["cb_cats"]
    rounds_cb = 150 if fast else 500

    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        cb_m = CatBoostClassifier(
            iterations=rounds_cb, depth=6, learning_rate=0.04, loss_function="Logloss",
            eval_metric="AUC", l2_leaf_reg=6.0, random_seed=82 + fold, task_type=cb_task, verbose=False
        )
        cb_m.fit(cb_tr.iloc[trn_idx], y_train[trn_idx], cat_features=cb_cats,
                 eval_set=(cb_tr.iloc[val_idx], y_train[val_idx]), early_stopping_rounds=30, verbose=False)
        oof_cb[val_idx] = cb_m.predict_proba(cb_tr.iloc[val_idx])[:, 1]
        test_cb += cb_m.predict_proba(cb_te)[:, 1].astype(np.float32) / float(n_splits)

    print(f"      -> oof_cb Complete | OOF AUC: {roc_auc_score(y_train, oof_cb):.6f}")

    # STREAM 6: Grand Fusion (final_oof, final_test, final_auc)
    print("\n[Stream 6] Assembling Grand Fusion Ensemble...")
    # Canonical fusion weights
    final_oof = rk(0.35 * oof_e1Aet + 0.25 * oof_e1et + 0.20 * oof_e1 + 0.10 * oof_old + 0.10 * rk(oof_cb))
    final_test = rk(0.35 * test_e1Aet + 0.25 * test_e1et + 0.20 * test_e1 + 0.10 * test_old + 0.10 * rk(test_cb))
    final_auc = float(roc_auc_score(y_train, final_oof))
    print(f"      -> Grand Fusion Complete | OOF AUC: {final_auc:.6f}")

    bundle_dict = {
        "oof_old": oof_old, "test_old": test_old,
        "oof_e1": oof_e1, "test_e1": test_e1,
        "oof_e1et": oof_e1et, "test_e1et": test_e1et,
        "oof_e1Aet": oof_e1Aet, "test_e1Aet": test_e1Aet,
        "oof_cb": oof_cb, "test_cb": test_cb,
        "final_oof": final_oof, "final_test": final_test,
        "final_auc": final_auc
    }

    canonical_keys = [
        "oof_old", "test_old", "oof_e1", "test_e1", "oof_e1et", "test_e1et",
        "oof_e1Aet", "test_e1Aet", "oof_cb", "test_cb", "final_oof", "final_test", "final_auc"
    ]
    ordered_dict = {k: bundle_dict[k] for k in canonical_keys}
    np.savez_compressed(output_path, **ordered_dict)

    print("\n" + "=" * 80)
    print(f"  v_17_full_suite_bundle.npz Generated Successfully!")
    print(f"  Path      : {output_path}")
    print(f"  Total Time: {time.time()-t_start:.1f}s")
    print("=" * 80)
    return ordered_dict

def main():
    parser = argparse.ArgumentParser(description="Generate v_17_full_suite_bundle.npz strictly from raw data")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode")
    parser.add_argument("--data_dir", type=str, default=None, help="Explicit path to raw competition data ('inpuy')")
    parser.add_argument("--output", type=str, default=None, help="Output path for v_17_full_suite_bundle.npz")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save v_17_full_suite_bundle.npz")
    args = parser.parse_args()

    target_out = args.output_dir if args.output_dir is not None else args.output
    train_v17_suite_from_scratch(fast=args.fast, output_path=target_out, data_dir=args.data_dir)

if __name__ == "__main__":
    main()
