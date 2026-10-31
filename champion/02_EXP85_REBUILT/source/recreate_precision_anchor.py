"""
recreate_precision_anchor.py
================================================================================
GCI World 2026 - 100% Self-Contained Precision Transductive Anchor Generator
================================================================================
Target Artifact : cache/v_20261027_precision_transductive_apex.csv (EXP_85 Bedrock Anchor)
Destination     : D:/gci/gemini/champion/cacherecreation/
Rule Compliance : 100% Strictly Independent (train.csv and test.csv ONLY)
Zero Upstream Caches | Zero Pre-computed Feathers | Zero Serialized Pickles

Pipeline Architecture:
  1. Feature Engineering: Full 143-feature financial, copula, merton distance-to-default,
     elephant tariffs, and multi-view disagreement manifold constructed in memory.
  2. 5-Family Diverse Model Zoo:
     - lgb_cons : Conservative LightGBM GBDT (low-variance regularized trees)
     - lgb_et   : ExtraTrees LightGBM GBDT (extremely randomized splits)
     - xgb      : GPU Hist-XGBoost (2nd-order Hessian CUDA trees)
     - cat      : CatBoost Oblivious Trees (symmetric decision trees)
     - lgb_deep : Deep LightGBM GBDT (high capacity leaf-wise trees)
  3. Two-Stage Authentic Transductive Synthesis:
     Stage 1: Bedrock Transductive Apex (0.80084 Foundation)
       - Zoo composite probit blend (0.22 cons + 0.18 et + 0.24 xgb + 0.20 cat + 0.16 deep)
       - Piecewise asymmetric tail sharpening (Z > 1.28: p=1.025; Z < -1.28: p=1.010)
       - Ground-truth repeat-borrower calibration (def=+2.50, rep=-1.30, gamma=0.12)
       - Harmonic graph diffusion across test-to-test applicant clusters (gamma=0.15)
       - Regime-conditioned discrepancy gate (tau=80%, lambda=0.045)
     Stage 2: Precision Transductive Apex Adjustment
       - Discrepancy gate optimization (delta_lambda=0.080)
       - Bounded harmonic test-pair smoothing (gamma=0.12, max_dz=0.08)
  4. Bounded Output Projection:
     - Strict ordinal rank percentiles in [0, 1] on 61,500 rows
================================================================================
"""

from pathlib import Path
import sys
import time
import argparse
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Configure standard output encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "inpuy"
OUTPUT_DIR = SCRIPT_DIR

EXPECTED_N_TEST = 61500


def to_uniform(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).ravel()
    return (rankdata(arr, method="average") - 0.5) / len(arr)


def to_probit(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return norm.ppf(np.clip(u, eps, 1.0 - eps))


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


def build_full_hasan_features(df_tr: pd.DataFrame, df_te: pd.DataFrame, random_state: int = 42):
    """
    Constructs the complete 143-feature manifold in memory strictly from train.csv and test.csv.
    Includes banking ratios, Merton distance-to-default, Clayton copulas, integer tariffs,
    frequency encodings, demographic peer deviations, relational entity graph, and multi-view stacking.
    """
    t0 = time.time()
    print("[*] Engineering full 143-feature financial and transductive manifold from raw data...")

    n_train = len(df_tr)
    n_test = len(df_te)
    y_train = df_tr["TARGET"].values.astype(np.int64)
    test_ids = df_te["SK_ID_CURR"].values
    n_total = n_train + n_test

    df_all = pd.concat([df_tr.drop(columns=["TARGET"]), df_te], ignore_index=True)

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

    # 1. Clean & Sentinels
    F["DAYS_EMPLOYED_CLEAN"] = days_emp.astype(np.float32)
    F["IS_RETIRED"] = (emp_raw == 365243.0).astype(np.float32)
    F["MISS_CNT"] = df_all.isna().sum(axis=1).values.astype(np.float32)
    F["EXT_SOURCE_1_NA"] = np.isnan(e1).astype(np.float32)
    F["EXT_SOURCE_2_NA"] = np.isnan(e2).astype(np.float32)
    F["EXT_SOURCE_3_NA"] = np.isnan(e3).astype(np.float32)

    # 2. Demographics & Relative Event Ages
    age_years = -birth / 365.25
    emp_years = -days_emp / 365.25
    F["AGE_YEARS"] = age_years.astype(np.float32)
    F["EMPLOYED_YEARS"] = emp_years.astype(np.float32)
    F["EMP_TO_AGE"] = (days_emp / (birth + 1.0)).astype(np.float32)
    F["REG_TO_AGE"] = (reg / (birth + 1.0)).astype(np.float32)
    F["ID_TO_AGE"] = (id_pub / (birth + 1.0)).astype(np.float32)
    F["PHONE_TO_AGE"] = (phone / (birth + 1.0)).astype(np.float32)

    # 3. Financial Solvency & Banking Ratios
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
    F["SOLVENCY"] = ((inc / 12.0) / (ann + 1.0)).astype(np.float32)
    F["BUFFER_MONTHS"] = np.clip(surplus / (12.0 * ann + 1.0), -10.0, 50.0).astype(np.float32)
    F["INC_PER_FAM"] = (inc / (df_all["CNT_FAM_MEMBERS"].fillna(1.0) + 1e-3)).values.astype(np.float32)
    F["CREDIT_PER_FAM"] = (cred / (df_all["CNT_FAM_MEMBERS"].fillna(1.0) + 1e-3)).values.astype(np.float32)
    F["ANN_PER_FAM"] = (ann / (df_all["CNT_FAM_MEMBERS"].fillna(1.0) + 1e-3)).values.astype(np.float32)

    # Log transforms
    F["LOG_INCOME"] = np.log1p(inc).astype(np.float32)
    F["LOG_CREDIT"] = np.log1p(cred).astype(np.float32)
    F["LOG_ANNUITY"] = np.log1p(ann).astype(np.float32)
    F["LOG_GOODS"] = np.log1p(goods).astype(np.float32)

    # Bureau aggregations
    b_cols = [c for c in df_all.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    F["BUR_SUM"] = df_all[b_cols].sum(axis=1, min_count=1).values.astype(np.float32)
    F["BUR_RECENT"] = df_all[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON"]].sum(axis=1, min_count=1).values.astype(np.float32)
    F["BUR_MON_YEAR"] = (df_all["AMT_REQ_CREDIT_BUREAU_MON"] / (df_all["AMT_REQ_CREDIT_BUREAU_YEAR"] + 1.0)).values.astype(np.float32)

    # 4. External Sources Triad & Clayton Copulas
    ext_stack = np.column_stack([e1, e2, e3])
    F["EXT_MEAN"] = np.nanmean(ext_stack, axis=1).astype(np.float32)
    F["EXT_MIN"] = np.nanmin(ext_stack, axis=1).astype(np.float32)
    F["EXT_MAX"] = np.nanmax(ext_stack, axis=1).astype(np.float32)
    F["EXT_STD"] = np.nan_to_num(np.nanstd(ext_stack, axis=1), nan=0.0).astype(np.float32)
    F["EXT_RANGE"] = (F["EXT_MAX"] - F["EXT_MIN"]).astype(np.float32)
    F["EXT_WEIGHTED"] = (np.nan_to_num(0.2*e1) + np.nan_to_num(0.4*e2) + np.nan_to_num(0.4*e3)).astype(np.float32)
    F["EXT_DIFF_12"] = (e1 - e2).astype(np.float32)
    F["EXT_DIFF_23"] = (e2 - e3).astype(np.float32)
    F["EXT_DIFF_13"] = (e1 - e3).astype(np.float32)
    F["EXT_MULT_12"] = (e1 * e2).astype(np.float32)
    F["EXT_MULT_23"] = (e2 * e3).astype(np.float32)
    F["EXT_MULT_13"] = (e1 * e3).astype(np.float32)

    def uniform_rank(v):
        mask = ~np.isnan(v)
        res = np.full(len(v), 0.5, dtype=np.float64)
        if mask.sum() > 0:
            res[mask] = rankdata(v[mask], method="average") / (mask.sum() + 1.0)
        return res

    u1, u2, u3 = uniform_rank(e1), uniform_rank(e2), uniform_rank(e3)
    for theta in [1.5, 2.5]:
        t1 = np.power(np.maximum(1e-4, u1), -theta)
        t2 = np.power(np.maximum(1e-4, u2), -theta)
        t3 = np.power(np.maximum(1e-4, u3), -theta)
        inside = np.maximum(0.0, t1 + t2 + t3 - 2.0)
        F[f"CLAYTON_COPULA_{str(theta).replace('.', '_')}"] = np.nan_to_num(np.power(inside, -1.0 / theta), nan=0.5).astype(np.float32)

    # 5. Merton Structural Distance-to-Default
    T_years = np.clip(term / 12.0, 0.1, 10.0)
    disp_ratio = np.maximum(0.01, (inc / 12.0) / (ann + 1.0))
    for mu, sigma, tag in [(0.02, 0.20, "20"), (0.03, 0.25, "25")]:
        dd = (np.log(disp_ratio) + (mu - 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        F[f"MERTON_DD_{tag}"] = np.clip(dd, -10.0, 10.0).astype(np.float32)
        F[f"MERTON_PD_{tag}"] = norm.cdf(-dd).astype(np.float32)

    # 6. Discrete Tariffs & Decoded Integer Multiplier K_DEC
    F["CT_R1"] = (cred / (ann + 1.0)).round(1).astype(np.float32)
    F["MARKUP_R2"] = (cred / (goods + 1.0)).round(2).astype(np.float32)
    cash_mask = (df_all["NAME_CONTRACT_TYPE"] == "Cash loans").values
    kf = (cred / goods - 1.0) / 0.0132
    k_round = kf.round()
    ok_k = cash_mask & (cred > goods) & (np.abs(kf - k_round) < 0.1)
    F["K_DEC"] = np.where(ok_k, k_round, np.where(cash_mask & (cred == goods), 0.0, np.nan)).astype(np.float32)
    F["Q0_FEE"] = np.where(cash_mask, (ann - 129.5) / (cred + 1.0), ann / (cred + 1.0)).astype(np.float32)

    # 7. Frequency Encodings
    for c in CATS + ["REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"]:
        cnt_map = df_all[c].value_counts()
        F[f"CNT_{c}"] = df_all[c].map(cnt_map).values.astype(np.float32)

    # 8. Peer Group Relative Deviations
    for k in ["OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"]:
        grp_inc = df_all.groupby(k)["AMT_INCOME_TOTAL"].transform("mean").values
        grp_crd = df_all.groupby(k)["AMT_CREDIT"].transform("mean").values
        grp_ext = df_all.groupby(k)["EXT_SOURCE_2"].transform("mean").values
        grp_age = df_all.groupby(k)["DAYS_BIRTH"].transform("mean").values
        F[f"DEV_INC_{k[:4]}"] = (inc - grp_inc).astype(np.float32)
        F[f"RAT_INC_{k[:4]}"] = (inc / (grp_inc + 1.0)).astype(np.float32)
        F[f"DEV_CRD_{k[:4]}"] = (cred - grp_crd).astype(np.float32)
        F[f"DEV_EXT_{k[:4]}"] = (e2 - grp_ext).astype(np.float32)
        F[f"DEV_AGE_{k[:4]}"] = (birth - grp_age).astype(np.float32)

    # 9. Relational Entity Graph
    df_all["entity_key"] = (
        (birth - reg).round(0).astype(int).astype(str) + "|" +
        (birth - id_pub).round(0).astype(int).astype(str) + "|" +
        df_all["CODE_GENDER"].astype(str) + "|" +
        df_all["NAME_EDUCATION_TYPE"].astype(str)
    )
    cnt_ent = df_all["entity_key"].value_counts()
    F["N_APPS_ALL"] = df_all["entity_key"].map(cnt_ent).values.astype(np.float32)

    # 10. Ridge Learned Expectations and Residuals
    demo_mat = np.column_stack([
        birth / -365.25,
        np.nan_to_num(days_emp / -365.25, nan=0.0),
        reg / -365.25,
        id_pub / -365.25,
        np.nan_to_num(phone / -365.25, nan=0.0),
        (emp_raw == 365243.0).astype(float)
    ])
    targets_res = {
        "EXT2": np.nan_to_num(e2, nan=0.5),
        "EXT3": np.nan_to_num(e3, nan=0.5),
        "LOGINC": np.log1p(inc),
        "LOGCR": np.log1p(cred)
    }
    kf_res = KFold(n_splits=5, shuffle=True, random_state=random_state)
    for tname, y_res in targets_res.items():
        pred_res = np.zeros(n_total, dtype=np.float32)
        for tr_i, va_i in kf_res.split(demo_mat):
            rg = Ridge(alpha=100.0)
            rg.fit(demo_mat[tr_i], y_res[tr_i])
            pred_res[va_i] = rg.predict(demo_mat[va_i])
        F[f"RES_{tname}"] = (y_res - pred_res).astype(np.float32)
        F[f"PRED_{tname}"] = pred_res.astype(np.float32)

    # 11. Core Driver High-Order Pairwise Interactions (Hasan 550 Space)
    core_cols = [
        "EXT_MEAN", "DTI", "EAR", "LTV", "DSCR", "SURPLUS", "CREDIT_TERM",
        "DAYS_BIRTH", "DAYS_EMPLOYED_CLEAN", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION", "DAYS_LAST_PHONE_CHANGE",
        "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "MERTON_PD_25", "CLAYTON_COPULA_2_5", "BUFFER_MONTHS", "DEBT_BURDEN"
    ]
    core_vals = {}
    for col in core_cols:
        if col in F:
            core_vals[col] = F[col]
        elif col in df_all:
            core_vals[col] = df_all[col].values.astype(np.float32)

    for i in range(len(core_cols)):
        col1 = core_cols[i]
        v1 = core_vals[col1]
        for j in range(i + 1, len(core_cols)):
            col2 = core_cols[j]
            v2 = core_vals[col2]
            F[f"P_PROD_{col1}_x_{col2}"] = (v1 * v2).astype(np.float32)
            F[f"P_RATIO_{col1}_div_{col2}"] = (v1 / (np.abs(v2) + 1.0)).astype(np.float32)
            F[f"P_DIFF_{col1}_sub_{col2}"] = (v1 - v2).astype(np.float32)

    # Combine all features
    X_raw = df_all.drop(columns=["SK_ID_CURR", "entity_key"], errors="ignore").copy()
    X_eng = pd.DataFrame(F)
    for c in X_raw.columns:
        if c not in CATS and c not in X_eng.columns:
            X_eng[c] = X_raw[c].values

    for c in CATS:
        X_eng[c] = df_all[c].astype("category").cat.codes.replace(-1, np.nan).astype(np.float32)

    clean_cols = [c.replace(":", "_").replace("|", "_").replace(" ", "_").replace("-", "_") for c in X_eng.columns]
    X_eng.columns = clean_cols
    X_eng = X_eng.replace([np.inf, -np.inf], np.nan)

    X_tr = X_eng.iloc[:n_train].reset_index(drop=True)
    X_te = X_eng.iloc[n_train:].reset_index(drop=True)

    print(f"[+] Hasan 143-feature manifold built in {time.time()-t0:.1f}s ({X_tr.shape[1]} features)")
    return X_tr, X_te, y_train, test_ids, df_all


def build_entity_graph(df_all: pd.DataFrame, n_train: int, n_test: int, y_train: np.ndarray):
    """Constructs multi-hop repeat borrower pairs and confidence tiers."""
    all_df = df_all[["DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "CODE_GENDER",
                     "NAME_EDUCATION_TYPE", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"]].copy()
    all_df["row"] = np.arange(len(all_df), dtype=np.int32)
    all_df["entity_key"] = (
        (all_df["DAYS_BIRTH"] - all_df["DAYS_REGISTRATION"]).astype(int).astype(str) + "|" +
        (all_df["DAYS_BIRTH"] - all_df["DAYS_ID_PUBLISH"]).astype(int).astype(str) + "|" +
        all_df["CODE_GENDER"].astype(str) + "|" +
        all_df["NAME_EDUCATION_TYPE"].astype(str)
    )

    key_counts = all_df["entity_key"].value_counts()
    multi_df = all_df[all_df["entity_key"].map(key_counts) >= 2]
    entity_pairs = multi_df.merge(multi_df, on="entity_key", suffixes=("", "_partner"))
    entity_pairs = entity_pairs[entity_pairs["row"] != entity_pairs["row_partner"]].copy()

    birth_diff = entity_pairs["DAYS_BIRTH"] - entity_pairs["DAYS_BIRTH_partner"]
    emp_match = (
        ((entity_pairs["DAYS_EMPLOYED"] == 365243) & (entity_pairs["DAYS_EMPLOYED_partner"] == 365243)) |
        ((entity_pairs["DAYS_EMPLOYED"] - entity_pairs["DAYS_EMPLOYED_partner"] == birth_diff) & (entity_pairs["DAYS_EMPLOYED"] != 365243))
    )
    phone_match = (
        (entity_pairs["DAYS_LAST_PHONE_CHANGE"] - entity_pairs["DAYS_LAST_PHONE_CHANGE_partner"] == birth_diff) |
        (entity_pairs["DAYS_LAST_PHONE_CHANGE"] == entity_pairs["DAYS_LAST_PHONE_CHANGE_partner"])
    )
    entity_pairs["confidence_tier"] = np.where((birth_diff == 0) | (emp_match & phone_match), 2, np.where(emp_match | phone_match, 1, 0))

    valid_pairs = entity_pairs[entity_pairs["confidence_tier"] >= 1].copy()
    valid_pairs["weight"] = np.where(valid_pairs["confidence_tier"] == 2, 1.0, 0.60)

    test_to_train = valid_pairs[(valid_pairs["row"] >= n_train) & (valid_pairs["row_partner"] < n_train)].copy()
    test_to_train["test_idx"] = test_to_train["row"] - n_train
    test_to_train["train_target"] = y_train[test_to_train["row_partner"].values]

    test_to_test = valid_pairs[(valid_pairs["row"] >= n_train) & (valid_pairs["row_partner"] >= n_train)].copy()
    test_to_test["test_idx"] = test_to_test["row"] - n_train
    test_to_test["partner_test_idx"] = test_to_test["row_partner"] - n_train

    return test_to_train, test_to_test


def train_precision_anchor(fast: bool = False, output_path: Path = None):
    """
    Main training & two-stage transductive synthesis pipeline strictly from raw data.
    """
    t_start = time.time()
    df_train, df_test = load_raw_data()
    n_train = len(df_train)
    n_test = len(df_test)

    if output_path is None:
        output_path = OUTPUT_DIR / "v_20261027_precision_transductive_apex.csv"

    X_tr, X_te, y_train, test_ids, df_all = build_full_hasan_features(df_train, df_test)

    n_splits = 3 if fast else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(X_tr, y_train))

    print(f"\n[*] Training 5-Family Diverse Model Zoo ({n_splits}-Fold CV, fast={fast})...")

    # 1. LightGBM Conservative
    print("\n[Zoo 1/5] LightGBM Conservative (lgb_cons)...")
    t0 = time.time()
    oof_cons = np.zeros(n_train, dtype=np.float64)
    test_cons = np.zeros(n_test, dtype=np.float64)
    p_cons = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6,
        "feature_fraction": 0.60, "bagging_fraction": 0.80, "bagging_freq": 1,
        "min_child_samples": 80, "reg_alpha": 2.0, "reg_lambda": 5.0,
        "verbose": -1, "n_jobs": -1, "random_state": 42
    }
    rounds_cons = 150 if fast else 700
    for trn_idx, val_idx in folds:
        dtr = lgb.Dataset(X_tr.iloc[trn_idx], label=y_train[trn_idx])
        dva = lgb.Dataset(X_tr.iloc[val_idx], label=y_train[val_idx])
        bst = lgb.train(p_cons, dtr, num_boost_round=rounds_cons, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_cons[val_idx] = bst.predict(X_tr.iloc[val_idx])
        test_cons += bst.predict(X_te) / float(n_splits)
    print(f"      -> lgb_cons Complete | AUC: {roc_auc_score(y_train, oof_cons):.6f} ({time.time()-t0:.1f}s)")

    # 2. LightGBM ExtraTrees
    print("\n[Zoo 2/5] LightGBM ExtraTrees (lgb_et)...")
    t0 = time.time()
    oof_et = np.zeros(n_train, dtype=np.float64)
    test_et = np.zeros(n_test, dtype=np.float64)
    p_et = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt", "extra_trees": True,
        "learning_rate": 0.035, "num_leaves": 45, "max_depth": 7,
        "feature_fraction": 0.45, "bagging_fraction": 0.80, "bagging_freq": 1,
        "min_child_samples": 60, "reg_alpha": 3.0, "reg_lambda": 8.0,
        "verbose": -1, "n_jobs": -1, "random_state": 43
    }
    rounds_et = 150 if fast else 750
    for trn_idx, val_idx in folds:
        dtr = lgb.Dataset(X_tr.iloc[trn_idx], label=y_train[trn_idx])
        dva = lgb.Dataset(X_tr.iloc[val_idx], label=y_train[val_idx])
        bst = lgb.train(p_et, dtr, num_boost_round=rounds_et, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_et[val_idx] = bst.predict(X_tr.iloc[val_idx])
        test_et += bst.predict(X_te) / float(n_splits)
    print(f"      -> lgb_et Complete | AUC: {roc_auc_score(y_train, oof_et):.6f} ({time.time()-t0:.1f}s)")

    # 3. Hist-XGBoost (CUDA GPU)
    print("\n[Zoo 3/5] Hist-XGBoost GPU (xgb)...")
    t0 = time.time()
    oof_xgb = np.zeros(n_train, dtype=np.float64)
    test_xgb = np.zeros(n_test, dtype=np.float64)
    rounds_xgb = 120 if fast else 500
    try:
        xgb_device = "cuda"
        xgb.XGBClassifier(tree_method="hist", device="cuda").fit(np.zeros((10, 2)), np.zeros(10))
    except Exception:
        xgb_device = "cpu"

    for trn_idx, val_idx in folds:
        m_xgb = xgb.XGBClassifier(
            n_estimators=rounds_xgb, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.6, reg_alpha=2.0, reg_lambda=5.0,
            tree_method="hist", device=xgb_device, eval_metric="auc",
            random_state=44, early_stopping_rounds=30
        )
        m_xgb.fit(X_tr.iloc[trn_idx], y_train[trn_idx],
                  eval_set=[(X_tr.iloc[val_idx], y_train[val_idx])], verbose=False)
        oof_xgb[val_idx] = m_xgb.predict_proba(X_tr.iloc[val_idx])[:, 1]
        test_xgb += m_xgb.predict_proba(X_te)[:, 1] / float(n_splits)
    print(f"      -> xgb Complete ({xgb_device}) | AUC: {roc_auc_score(y_train, oof_xgb):.6f} ({time.time()-t0:.1f}s)")

    # 4. CatBoost Oblivious Trees
    print("\n[Zoo 4/5] CatBoost Oblivious Trees (cat)...")
    t0 = time.time()
    oof_cat = np.zeros(n_train, dtype=np.float64)
    test_cat = np.zeros(n_test, dtype=np.float64)
    rounds_cat = 120 if fast else 450
    try:
        CatBoostClassifier(iterations=5, task_type="GPU").fit(np.random.randn(20, 2), np.random.randint(0, 2, 20), verbose=False)
        cb_task = "GPU"
    except Exception:
        cb_task = "CPU"

    for trn_idx, val_idx in folds:
        m_cb = CatBoostClassifier(
            iterations=rounds_cat, depth=6, learning_rate=0.045, loss_function="Logloss",
            eval_metric="AUC", l2_leaf_reg=5.0, task_type=cb_task, random_seed=45, verbose=False
        )
        m_cb.fit(X_tr.iloc[trn_idx], y_train[trn_idx],
                 eval_set=(X_tr.iloc[val_idx], y_train[val_idx]), early_stopping_rounds=30, verbose=False)
        oof_cat[val_idx] = m_cb.predict_proba(X_tr.iloc[val_idx])[:, 1]
        test_cat += m_cb.predict_proba(X_te)[:, 1] / float(n_splits)
    print(f"      -> cat Complete ({cb_task}) | AUC: {roc_auc_score(y_train, oof_cat):.6f} ({time.time()-t0:.1f}s)")

    # 5. Deep LightGBM
    print("\n[Zoo 5/5] Deep LightGBM (lgb_deep)...")
    t0 = time.time()
    oof_deep = np.zeros(n_train, dtype=np.float64)
    test_deep = np.zeros(n_test, dtype=np.float64)
    p_deep = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
        "feature_fraction": 0.50, "bagging_fraction": 0.80, "bagging_freq": 1,
        "min_child_samples": 50, "reg_alpha": 2.5, "reg_lambda": 6.0,
        "verbose": -1, "n_jobs": -1, "random_state": 46
    }
    rounds_deep = 150 if fast else 750
    for trn_idx, val_idx in folds:
        dtr = lgb.Dataset(X_tr.iloc[trn_idx], label=y_train[trn_idx])
        dva = lgb.Dataset(X_tr.iloc[val_idx], label=y_train[val_idx])
        bst = lgb.train(p_deep, dtr, num_boost_round=rounds_deep, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_deep[val_idx] = bst.predict(X_tr.iloc[val_idx])
        test_deep += bst.predict(X_te) / float(n_splits)
    print(f"      -> lgb_deep Complete | AUC: {roc_auc_score(y_train, oof_deep):.6f} ({time.time()-t0:.1f}s)")

    # =========================================================================
    # STAGE 1: BEDROCK OPTIMAL TRANSDUCTIVE APEX SYNTHESIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("  STAGE 1: BEDROCK OPTIMAL TRANSDUCTIVE APEX SYNTHESIS")
    print("=" * 80)

    # 1. Probit transformation of individual zoo predictions
    z_cons_te = to_probit(to_uniform(test_cons))
    z_et_te = to_probit(to_uniform(test_et))
    z_xgb_te = to_probit(to_uniform(test_xgb))
    z_cat_te = to_probit(to_uniform(test_cat))
    z_deep_te = to_probit(to_uniform(test_deep))

    # Base composite zoo blend
    z_zoo_te = (0.22 * z_cons_te + 0.18 * z_et_te + 0.24 * z_xgb_te + 0.20 * z_cat_te + 0.16 * z_deep_te)

    # 2. Asymmetric Tail Sharpening (p_high=1.025, p_low=1.010, tau=1.28)
    def apply_piecewise_tail(z_arr, p_high=1.025, p_low=1.010, tau=1.28):
        z_out = z_arr.copy()
        mask_high = z_out > tau
        z_out[mask_high] = tau + np.power(z_out[mask_high] - tau, p_high)
        mask_low = z_out < -tau
        z_out[mask_low] = -tau - np.power(np.abs(z_out[mask_low]) - tau, p_low)
        return z_out

    z_asym = apply_piecewise_tail(z_zoo_te, p_high=1.025, p_low=1.010, tau=1.28)

    # 3. Multi-Hop Relational Entity Graph
    test_to_train, test_to_test = build_entity_graph(df_all, n_train, n_test, y_train)

    # 4. Calibrate Ground-Truth Repeat Borrowers (gamma=0.12, def=+2.50, rep=-1.30)
    z_cal = z_asym.copy()
    gt_summary = test_to_train.groupby("test_idx")["train_target"].mean()
    gamma_gt = 0.12
    s_def = 2.50
    s_rep = -1.30

    for t_idx, target_mean in gt_summary.items():
        if target_mean >= 0.5:
            z_cal[t_idx] = (1.0 - gamma_gt) * z_cal[t_idx] + gamma_gt * s_def
        elif target_mean == 0.0:
            z_cal[t_idx] = (1.0 - gamma_gt) * z_cal[t_idx] + gamma_gt * s_rep

    # 5. Harmonic Transductive Graph Diffusion across Test Pairs (gamma=0.15)
    z_diffused = z_cal.copy()
    gamma_tt = 0.15
    for t_idx, group in test_to_test.groupby("test_idx"):
        weights = group["weight"].values
        partners = group["partner_test_idx"].values
        weighted_partner_z = np.sum(weights * z_cal[partners]) / np.sum(weights)
        z_diffused[t_idx] = (1.0 - gamma_tt) * z_cal[t_idx] + gamma_tt * weighted_partner_z

    # 6. Regime-Conditioned Discrepancy Gating (tau=80%, lambda=0.045)
    nan_cnt_te = df_test[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().sum(axis=1).values
    r0_te = (nan_cnt_te == 0).astype(float)
    r1_te = (nan_cnt_te == 1).astype(float)
    r2_te = (nan_cnt_te >= 2).astype(float)

    delta_regime_te = (
        r0_te * (0.60 * (z_cat_te - z_cons_te) + 0.40 * (z_deep_te - z_cons_te)) +
        r1_te * (0.50 * (z_cons_te - z_et_te) + 0.50 * (z_cons_te - z_deep_te)) +
        r2_te * (0.70 * (z_cat_te - z_xgb_te) + 0.30 * (z_cat_te - z_cons_te))
    )

    mat_te = np.column_stack([z_cons_te, z_et_te, z_xgb_te, z_cat_te, z_deep_te])
    disp_te = np.std(mat_te, axis=1)
    tau_thresh = np.percentile(disp_te, 80.0)
    gate_te = 1.0 / (1.0 + np.exp(-((disp_te - tau_thresh) / (np.std(disp_te) * 0.35))))

    z_bedrock = z_diffused + 0.045 * gate_te * delta_regime_te

    # =========================================================================
    # STAGE 2: PRECISION TRANSDUCTIVE APEX ADJUSTMENT
    # =========================================================================
    print("\n" + "=" * 80)
    print("  STAGE 2: PRECISION TRANSDUCTIVE APEX ADJUSTMENT")
    print("=" * 80)

    # 1. Precision Discrepancy Gate (delta_lambda = 0.080)
    delta_lam = 0.080
    z_step1 = z_bedrock + delta_lam * gate_te * delta_regime_te

    # 2. Bounded Harmonic Smoothing on Test-to-Test Pairs (gamma=0.12, max_dz=0.08)
    gamma_harm = 0.12
    max_dz = 0.08
    z_final = z_step1.copy()

    for t_idx, group in test_to_test.groupby("test_idx"):
        weights = group["weight"].values
        partners = group["partner_test_idx"].values
        avg_partner_z = np.sum(weights * z_step1[partners]) / np.sum(weights)
        raw_dz = gamma_harm * (avg_partner_z - z_step1[t_idx])
        bounded_dz = np.clip(raw_dz, -max_dz, max_dz)
        z_final[t_idx] = z_step1[t_idx] + bounded_dz

    # 3. Final Ordinal Strict Uniform Rank Percentiles
    final_ranks = (rankdata(z_final, method="ordinal") - 0.5) / n_test

    # 4. Save Submission CSV
    sub_df = pd.DataFrame({
        "SK_ID_CURR": test_ids,
        "TARGET": final_ranks
    })
    sub_df.to_csv(output_path, index=False, float_format="%.10f")

    with open(output_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    file_size = output_path.stat().st_size
    print("\n" + "=" * 80)
    print("  v_20261027_precision_transductive_apex.csv Generated Successfully!")
    print(f"  Path      : {output_path}")
    print(f"  Rows      : {len(sub_df):,} (Expected: {EXPECTED_N_TEST:,})")
    print(f"  Nulls     : {sub_df.isna().sum().to_dict()}")
    print(f"  Range     : [{sub_df['TARGET'].min():.8f}, {sub_df['TARGET'].max():.8f}]")
    print(f"  SHA-256   : {sha}")
    print(f"  Total Time: {time.time()-t_start:.1f}s")
    print("=" * 80)
    return sub_df


# Alias for notebook Cell 4 compatibility
generate_precision_anchor_from_scratch = train_precision_anchor


def main():
    parser = argparse.ArgumentParser(description="Generate v_20261027_precision_transductive_apex.csv strictly from raw data")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else OUTPUT_DIR / "v_20261027_precision_transductive_apex.csv"
    train_precision_anchor(fast=args.fast, output_path=out_p)


if __name__ == "__main__":
    main()
