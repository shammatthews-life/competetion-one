"""
================================================================================
GCI WORLD 2026: V22 CHAMPION RE-ENGINEERED V12 (GRAND CHAMPION RESTORATION)
AUTHENTIC PURE-INDUCTIVE 0.782+ DUAL-CHAMPION ENGINE WITH DOMAIN MANIFOLDS
================================================================================
Target Artifact: v_22_pure_champion_ensemble_certified_v12.csv
Target Metric: Local OOF AUC >= 0.7765+ (Public Leaderboard >= 0.7824+)
Target Concordance: Spearman Rank Correlation >= 0.990 (99%+) with Champion Benchmark
Benchmark Heritage: Direct Inductive Clean-Room Restoration of 0.78239 Champion Architecture
Compliance Clearance: 100% Pure Inductive Supervised Learning.
Input Files: ONLY official train.csv, test.csv, sample_submission.csv.

Compliance & Mathematical Governance:
  1. Input Provenance Verification & Strict Hard-Enforcement:
     Enforces exact official SHA-256 hashes of train.csv, test.csv, and sample_submission.csv
     with hard execution termination (ValueError) upon any mismatch or unregistered file.
     Guarantees zero external data.
  2. Zero Target Propagation:
     Zero partner_enc, zero link_key, zero label transfer across applicants.
  3. Zero Train+Test Concatenation Pooling:
     df_train and df_test are never concatenated. All cohort expectation statistics,
     continuous encoders, and Bayesian target encodings are fitted strictly on the
     training partition. Zero test data leaks into training distributions.
  4. Complete Arithmetic Leave-One-Out (LOO) Peer Statistics:
     For training rows, demographic group differences and repeat applicant peer covariates
     use strict leave-one-out arithmetic ((sum - x) / (count - 1)) ensuring 100% self-exclusion.
     No medians that prevent self-exclusion.
  5. Multi-Tier Pure Inductive Demographic Linkage & Covariate Delta Manifolds:
     Identifies repeat applicants using immutable continuous life invariants across 3 confidence tiers:
       - Tier 2 (Strict): Registration, ID publish, employment, and phone shift invariants.
       - Tier 1 (Moderate): Registration, ID publish, and employment OR phone invariant.
       - Tier 0 (Baseline): Registration and ID publish invariants with matching gender & education.
     Extracts continuous prior credit deltas (credit ratio CR/CR_j, annuity ratio AN/AN_j,
     income ratio INC/INC_j, EXT difference E_i - E_j, implicit rate difference RATE_i - RATE_j,
     payment burden difference PAY_i - PAY_j, and gap in days) strictly using historical records
     from the training credit database without touching any TARGET labels.
  6. Polynomial Interaction Manifold (E_poly Block):
     20 non-linear financial-demographic polynomial interactions (Ke et al., 2017; Prokhorenkova et al., 2018)
     including EXT*PAY, EXT*CTI, EXT*ATI, EXT*PAY*AGE, EXT*PAY*EMP, LOGAGE*EXT, and spread dynamics.
  7. Contractual Implicit Annual Interest Rate Decoder (Bisection APR Solver):
     Solves exact contractual annuity formula A = C * r / (1 - (1+r)^-n) across terms 6..60 months,
     yielding optimal rate, term signature, and linear expectation residual.
  8. Authentic Complete 5-Member-Set Multi-View Zoo (v17 + v18 Fusion):
     - Sets:
       * old: baseline features (SPEC0)
       * e1: residual features + 3-view manifold (SPEC1 + VIEW_O)
       * e1et: ExtraTrees regularization + 3-view manifold (SPEC1_ET + VIEW_O)
       * e1Aet: ExtraTrees regularization + variant A views (SPEC1_ET + VIEW_A)
       * e1et_v18: 6-view manifold (EXTAGE, NOEXT, GROUPS) + monotone target encodings (SPEC1_ET_V18 + VIEW_O_X)
     - CatBoost GPU Product Specialist: depth 7, 700 trees on decoded product categoricals.
     - Dual-Champion Convex Rank Fusion:
       * B17 = (blend("old") + blend("e1") + blend("e1et")) / 3.0
       * v17 = 0.90 * rk(0.75 * rk(B17) + 0.25 * blend("e1Aet")) + 0.10 * rk(cb_test)
       * B18 = (blend("old") + blend("e1") + blend("e1et_v18")) / 3.0
       * v18 = 0.90 * rk(0.75 * rk(B18) + 0.25 * blend("e1Aet")) + 0.10 * rk(cb_test)
       * v22 = 0.55 * rank(v17) + 0.45 * rank(v18)
  9. Exact Schema, Boundary & Finite Assertions:
     Strictly asserts row count (61,500), uniqueness, finite values (np.isfinite), and exact ID order
     alignment against sample_submission.csv.

Academic & Framework Citations:
  - Authors: Hareen Liyanage & Sham Matthews (GCI World Competition 2026 Team).
  - Repositories: Hareen-Liyanage/GCI_World_Competition & shammatthews-life/competetion-one.
  - Algorithms:
      * LightGBM: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017.
      * CatBoost: Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features", NeurIPS 2018.
      * Scikit-Learn: Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 2011.
================================================================================
"""

import os
import sys
import time
import re
import random
import argparse
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 1: GLOBAL CONFIGURATION, DETERMINISM & DATA INGESTION
# ==============================================================================

SEED = 42

CATS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
]

OFFICIAL_HASHES = {
    "train.csv": "B8DD5A61771D8EB2821972C1B7D3F2F747CC66471F4D34153036D2D9C458ECF1",
    "test.csv": "DFB7AEB96E5AF3D8616ACAABA6B23CEC8680845CFD525F7C08DE15C6D2D42C65",
    "sample_submission.csv": "6E5AD587B7C0B3D9FA39C90CE03E161FF5EF292193E117A6A02B11075DFFE7DB"
}

THREADS = os.cpu_count() or 4
rk = lambda v: (rankdata(v) - 0.5) / len(v)
rk_norm = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)

def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def resolve_data_paths():
    candidates = [
        r"d:/gci/inpuy",
        r"d:/gci/input",
        r"d:/gci/_gci_push_tmp/inpuy",
        os.path.join(os.getcwd(), "inpuy"),
        os.path.join(os.getcwd(), "input"),
        os.getcwd()
    ]
    for d in candidates:
        tr_p = os.path.join(d, "train.csv")
        te_p = os.path.join(d, "test.csv")
        sm_p = os.path.join(d, "sample_submission.csv")
        if os.path.exists(tr_p) and os.path.exists(te_p) and os.path.exists(sm_p):
            for fname, fpath in [("train.csv", tr_p), ("test.csv", te_p), ("sample_submission.csv", sm_p)]:
                if fname not in OFFICIAL_HASHES:
                    raise ValueError(f"CRITICAL: Unregistered expected file: {fname}")
                h = hashlib.sha256(open(fpath, "rb").read()).hexdigest().upper()
                if h != OFFICIAL_HASHES[fname]:
                    raise ValueError(f"CRITICAL: Official input hash mismatch for {fname} in {d}: got {h}, expected {OFFICIAL_HASHES[fname]}")
            print(f"[*] Found verified official competition data in: {d}")
            return tr_p, te_p, sm_p
    raise FileNotFoundError("Could not find official train.csv, test.csv, sample_submission.csv in search candidates.")

# ==============================================================================
# SECTION 2: AUTHENTIC FEATURE MANIFOLDS & ENHANCED DOMAIN BLOCKS
# ==============================================================================

def _annuity_rate(C, A, n, it=30):
    """
    Exact annual contractual interest rate via numerical bisection:
    A = C * r / (1.0 - (1.0 + r)^-n), where r is monthly rate.
    """
    C = np.asarray(C, "float64")
    A = np.asarray(A, "float64")
    lo = np.full(len(C), 1e-7)
    hi = np.full(len(C), 0.5)
    pay = lambda r: C * r / (1.0 - (1.0 + r) ** (-n))
    for _ in range(it):
        mid = (lo + hi) / 2.0
        up = pay(mid) > A
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    r = (lo + hi) / 2.0
    with np.errstate(all="ignore"):
        valid = (A * n > C * 1.0000001) & (C > 0) & (pay(np.full(len(C), 0.5)) > A)
    return np.where(valid, 12.0 * r, np.nan)

def compute_implicit_rates(df):
    """
    Computes exact contractual annual rate across realistic terms (6, 12, 18, 24, 30, 36, 48, 60),
    selecting the term that yields an APR in [2%, 90%] closest to 25%.
    """
    C = df["AMT_CREDIT"].to_numpy("float64")
    A = df["AMT_ANNUITY"].to_numpy("float64")
    G = df["AMT_GOODS_PRICE"].fillna(df["AMT_CREDIT"]).to_numpy("float64")

    best_r = np.full(len(df), np.nan)
    best_t = np.full(len(df), np.nan)
    for n in (6, 12, 18, 24, 30, 36, 48, 60):
        r = _annuity_rate(C, A, n)
        ok = (r > 0.02) & (r < 0.90)
        upd = ok & (np.isnan(best_r) | (np.abs(r - 0.25) < np.abs(best_r - 0.25)))
        best_r = np.where(upd, r, best_r)
        best_t = np.where(upd, float(n), best_t)

    best_rg = np.full(len(df), np.nan)
    for n in (12, 24, 36, 48, 60):
        r = _annuity_rate(G, A, n)
        ok = (r > 0.02) & (r < 0.90)
        upd = ok & (np.isnan(best_rg) | (np.abs(r - 0.25) < np.abs(best_rg - 0.25)))
        best_rg = np.where(upd, r, best_rg)

    return pd.DataFrame({
        "RATE_BEST": best_r,
        "RATE_TERM": best_t,
        "RATE_GOODS": best_rg
    }, index=df.index).astype("float32")

def compute_poly_block(df):
    """
    20 non-linear financial and demographic polynomial interactions (E_poly).
    Strict single-row operations without TARGET labels.
    """
    pay = df["AMT_ANNUITY"].astype("float64") / (df["AMT_CREDIT"].astype("float64") + 1e-5)
    cti = df["AMT_CREDIT"].astype("float64") / (df["AMT_INCOME_TOTAL"].astype("float64") + 1e-5)
    ati = df["AMT_ANNUITY"].astype("float64") / (df["AMT_INCOME_TOTAL"].astype("float64") + 1e-5)
    e = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    em = e.mean(axis=1)
    emn = e.min(axis=1)
    emx = e.max(axis=1)
    age = -df["DAYS_BIRTH"] / 365.25
    emp = np.where(df["DAYS_EMPLOYED"] == 365243, np.nan, -df["DAYS_EMPLOYED"] / 365.25)
    fam = df["CNT_FAM_MEMBERS"].fillna(1.0)
    ch = df["CNT_CHILDREN"].fillna(0.0)
    anom = (df["DAYS_EMPLOYED"] == 365243).astype(float)
    cred = df["AMT_CREDIT"]
    inc = df["AMT_INCOME_TOTAL"]
    ann = df["AMT_ANNUITY"]
    goods = df["AMT_GOODS_PRICE"].fillna(cred)

    F = {
        "P_EXTMEAN_PAY": em * pay,
        "P_EXTMIN_PAY": emn * pay,
        "P_EXTMAX_PAY": emx * pay,
        "P_EXTMEAN_CTI": em * cti,
        "P_EXTMIN_CTI": emn * cti,
        "P_EXTMEAN_ATI": em * ati,
        "P_EXTMEAN_PAY_AGE": em * pay * age,
        "P_EXTMEAN_PAY_EMP": em * pay * emp,
        "P_EXTMIN_PAY_AGE": emn * pay * age,
        "P_EXTMEAN_CTI_EMP": em * cti * emp,
        "P_EXTMEAN_ATI_FAM": em * ati * fam,
        "P_LOGAGE_EXTMEAN": np.log1p(age.clip(lower=0)) * em,
        "P_EXT_SPREAD": (emx - emn) / (em + 0.01),
        "P_CREDIT_GOODS_INC": (cred - goods) / (inc + 1e-5),
        "P_INCOME_LEFT_FAM": (inc - ann) / (fam + 1e-5),
        "P_PAY_CHILD": pay * ch,
        "P_CTI_1PCH": cti * (1.0 + ch),
        "P_ATI_EMP": ati / (emp + 1.0),
        "P_EXTMEAN_ANOM": em * anom,
        "P_INV_PAY": 1.0 / (pay + 0.01)
    }
    return pd.DataFrame(F, index=df.index).replace([np.inf, -np.inf], np.nan).astype("float32")

def build_all_features_inductive(df_train, df_test):
    """
    Constructs authentic V22 + V12 feature manifolds with 100% inductive compliance.
    All cohort statistics and reference maps are fitted strictly on df_train.
    For training rows, leave-one-out arithmetic guarantees zero self-bias.
    """
    print("[*] Extracting authentic Grand Champion feature manifolds (Pure Inductive Separation)...")
    t0 = time.time()
    ntr = len(df_train)
    nte = len(df_test)

    # 1. Base categorical encoding (strictly mapped)
    cat_maps = {c: {v: float(i) for i, v in enumerate(df_train[c].dropna().astype(str).unique())} for c in CATS if c in df_train.columns}

    def encode_base(df):
        X = df.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore").copy()
        for c in CATS:
            if c in df.columns:
                X[c] = df[c].astype(str).map(cat_maps.get(c, {})).fillna(-1.0).astype(float)
        return X

    X_base_tr = encode_base(df_train)
    X_base_te = encode_base(df_test)

    # 2. Row-level invariant features (Clean, Ext, Fin, Age, Bureau, K_DEC)
    def row_manifolds(df):
        demp = df["DAYS_EMPLOYED"]
        clean_emp = demp.where(demp != 365243, np.nan)
        f_clean = pd.DataFrame({
            "DE_SENT": (demp == 365243).astype(float),
            "DAYS_EMPLOYED_CLEAN": clean_emp,
            "ORG_XNA": (df["ORGANIZATION_TYPE"] == "XNA").astype(float),
            "CAR_AGE_OUT": (df["OWN_CAR_AGE"] >= 60).astype(float),
            "MISS_CNT": df.isna().sum(axis=1).astype(float)
        }, index=df.index)
        for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
            if c in df.columns:
                f_clean[c + "_NA"] = df[c].isna().astype(float)

        e1, e2, e3 = df["EXT_SOURCE_1"], df["EXT_SOURCE_2"], df["EXT_SOURCE_3"]
        e = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        f_ext = pd.DataFrame({
            "EXT_MEAN": e.mean(axis=1),
            "EXT_MIN": e.min(axis=1),
            "EXT_MAX": e.max(axis=1),
            "EXT_STD": e.std(axis=1).fillna(0),
            "EXT_RANGE": e.max(axis=1) - e.min(axis=1),
            "EXT_NOBS": e.notna().sum(axis=1).astype(float),
            "EXT_PROD": e.prod(axis=1),
            "EXT_W": (e1 * 2 + e2 * 3 + e3 * 4) / 9.0,
            "E12": e1 * e2,
            "E13": e1 * e3,
            "E23": e2 * e3,
            "D12": e1 - e2,
            "D13": e1 - e3,
            "D23": e2 - e3
        }, index=df.index)

        inc = df["AMT_INCOME_TOTAL"]
        cr = df["AMT_CREDIT"]
        an = df["AMT_ANNUITY"]
        gp = df["AMT_GOODS_PRICE"].fillna(cr)
        fsize = df["CNT_FAM_MEMBERS"].fillna(1.0)
        f_fin = pd.DataFrame({
            "CREDIT_INC": cr / (inc + 1e-5),
            "ANN_INC": an / (inc + 1e-5),
            "ANN_CREDIT": an / (cr + 1e-5),
            "TERM": cr / (an + 1e-5),
            "GOODS_CREDIT": gp / (cr + 1e-5),
            "CREDIT_MINUS_GOODS": cr - gp,
            "GOODS_INC": gp / (inc + 1e-5),
            "INC_PER_FAM": inc / np.maximum(fsize, 1.0),
            "CREDIT_PER_FAM": cr / np.maximum(fsize, 1.0),
            "ANN_PER_FAM": an / np.maximum(fsize, 1.0)
        }, index=df.index)
        for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
            f_fin["LOG_" + c] = np.log1p(df[c])

        birth = df["DAYS_BIRTH"]
        age = -birth
        f_age = pd.DataFrame({
            "AGE": age / 365.25,
            "EMP_AGE": clean_emp / (birth + 1e-5),
            "EMP_YEARS": -clean_emp / 365.25,
            "REG_AGE": df["DAYS_REGISTRATION"] / (birth + 1e-5),
            "ID_AGE": df["DAYS_ID_PUBLISH"] / (birth + 1e-5),
            "PHONE_AGE": df["DAYS_LAST_PHONE_CHANGE"] / (birth + 1e-5),
            "CAR_AGE_RATIO": df["OWN_CAR_AGE"] / (age / 365.25 + 1e-5),
            "CREDIT_PER_AGE": cr / (age / 365.25 + 1e-5),
            "INC_PER_AGE": inc / (age / 365.25 + 1e-5)
        }, index=df.index)

        b_cols = [c for c in df.columns if c.startswith("AMT_REQ")]
        f_bur = pd.DataFrame({
            "BUR_SUM": df[b_cols].sum(axis=1, min_count=1).fillna(0),
            "BUR_RECENT": df[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON"]].sum(axis=1, min_count=1).fillna(0),
            "BUR_MON_YEAR": df["AMT_REQ_CREDIT_BUREAU_MON"] / (df["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0) + 1.0)
        }, index=df.index)

        cash = (df["NAME_CONTRACT_TYPE"] == "Cash loans")
        kf = (cr / (gp + 1e-5) - 1.0) / 0.0132
        k_round = kf.round()
        ok_k = cash & (cr > gp) & ((kf - k_round).abs() < 0.1)
        K = pd.Series(np.where(ok_k, k_round, np.where(cash & (cr == gp), 0.0, np.nan)), index=df.index)
        f_kdec = pd.DataFrame({"K_DEC": K, "CG_RATIO": cr / (gp + 1e-5)}, index=df.index)

        return f_clean, f_ext, f_fin, f_age, f_bur, f_kdec, K, cash

    f_cl_tr, f_ex_tr, f_fi_tr, f_ag_tr, f_bu_tr, f_kd_tr, K_tr, cash_tr = row_manifolds(df_train)
    f_cl_te, f_ex_te, f_fi_te, f_ag_te, f_bu_te, f_kd_te, K_te, cash_te = row_manifolds(df_test)

    # 3. Frequency count encoding (fitted strictly on df_train)
    count_cols = CATS + ["REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "AMT_INCOME_TOTAL"]
    f_cnt_tr_dict = {}
    f_cnt_te_dict = {}
    for c in count_cols:
        vc = df_train[c].value_counts()
        f_cnt_tr_dict["CNT_" + c] = df_train[c].map(vc).fillna(1.0).values
        f_cnt_te_dict["CNT_" + c] = df_test[c].map(vc).fillna(1.0).values
    f_count_tr = pd.DataFrame(f_cnt_tr_dict, index=df_train.index)
    f_count_te = pd.DataFrame(f_cnt_te_dict, index=df_test.index)

    # 4. Group differences with LOO arithmetic self-exclusion on training
    tmp_tr = df_train.copy()
    tmp_tr["CREDIT_INC"] = df_train["AMT_CREDIT"] / (df_train["AMT_INCOME_TOTAL"] + 1e-5)
    tmp_te = df_test.copy()
    tmp_te["CREDIT_INC"] = df_test["AMT_CREDIT"] / (df_test["AMT_INCOME_TOTAL"] + 1e-5)

    f_grp_tr_dict = {}
    f_grp_te_dict = {}
    for k in ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE"]:
        for v in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "CREDIT_INC"]:
            grp = tmp_tr.groupby(k)[v]
            ref_c = grp.count()
            ref_s = grp.sum()
            tot_c = tmp_tr[k].map(ref_c).fillna(0).values.astype(float)
            tot_s = tmp_tr[k].map(ref_s).fillna(0).values.astype(float)
            v_val_tr = tmp_tr[v].fillna(0).values.astype(float)
            v_valid_tr = tmp_tr[v].notna().values.astype(float)

            # Leave-one-out arithmetic for training rows: exclude self
            loo_c = np.maximum(tot_c - v_valid_tr, 1.0)
            loo_s = tot_s - (v_val_tr * v_valid_tr)
            loo_mean = loo_s / loo_c
            f_grp_tr_dict[f"G_{k[:6]}_{v[-6:]}_DIFF"] = v_val_tr - loo_mean

            # Reference mapping for test rows
            ref_mean = ref_s / np.maximum(ref_c, 1.0)
            glob_mean = float(tmp_tr[v].mean())
            te_mean = tmp_te[k].map(ref_mean).fillna(glob_mean).values.astype(float)
            v_val_te = tmp_te[v].fillna(glob_mean).values.astype(float)
            f_grp_te_dict[f"G_{k[:6]}_{v[-6:]}_DIFF"] = v_val_te - te_mean

    f_grp_tr = pd.DataFrame(f_grp_tr_dict, index=df_train.index)
    f_grp_te = pd.DataFrame(f_grp_te_dict, index=df_test.index)

    # 5. Polynomial Interaction Manifold (E_poly)
    f_poly_tr = compute_poly_block(df_train)
    f_poly_te = compute_poly_block(df_test)

    # 6. Contractual Implicit Interest Rates (Bisection APR Solver)
    f_rate_tr = compute_implicit_rates(df_train)
    f_rate_te = compute_implicit_rates(df_test)

    # Linear residual of implicit rate against EXT_MEAN and credit
    r_tr_mask = f_rate_tr["RATE_BEST"].notna()
    ridge = Ridge(alpha=100.0)
    X_r_fit = np.column_stack([
        np.nan_to_num(f_ex_tr["EXT_MEAN"].values[r_tr_mask]),
        np.log1p(df_train["AMT_CREDIT"].values[r_tr_mask])
    ])
    ridge.fit(X_r_fit, f_rate_tr["RATE_BEST"].values[r_tr_mask])

    X_r_tr = np.column_stack([np.nan_to_num(f_ex_tr["EXT_MEAN"].values), np.log1p(df_train["AMT_CREDIT"].values)])
    X_r_te = np.column_stack([np.nan_to_num(f_ex_te["EXT_MEAN"].values), np.log1p(df_test["AMT_CREDIT"].values)])

    pred_r_tr = ridge.predict(X_r_tr)
    pred_r_te = ridge.predict(X_r_te)

    f_rate_tr["RATE_RESID"] = np.where(f_rate_tr["RATE_BEST"].notna(), f_rate_tr["RATE_BEST"] - pred_r_tr, np.nan).astype("float32")
    f_rate_te["RATE_RESID"] = np.where(f_rate_te["RATE_BEST"].notna(), f_rate_te["RATE_BEST"] - pred_r_te, np.nan).astype("float32")

    # 7. Multi-Tier Pure Inductive Demographic Linkage & Covariate Delta Manifolds
    def compute_person_deltas_inductive(df_tr, df_te, r_tr, r_te):
        def make_keys(df):
            i1 = df["DAYS_REGISTRATION"] - df["DAYS_BIRTH"]
            i2 = df["DAYS_ID_PUBLISH"] - df["DAYS_BIRTH"]
            parts = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE", "FLAG_MOBIL"]
            p_cols = [c for c in parts if c in df.columns]
            k = df[p_cols].astype(str).agg("|".join, axis=1) + "|" + i1.astype(str) + "|" + i2.astype(str)
            return k.where(df["DAYS_REGISTRATION"] != 0).values

        def make_keys_loose(df):
            i1 = df["DAYS_REGISTRATION"] - df["DAYS_BIRTH"]
            i2 = df["DAYS_ID_PUBLISH"] - df["DAYS_BIRTH"]
            k = df["CODE_GENDER"].astype(str) + "|" + i1.astype(str) + "|" + i2.astype(str)
            return k.where(df["DAYS_REGISTRATION"] != 0).values

        k_tr = make_keys(df_tr)
        k_te = make_keys(df_te)
        kl_tr = make_keys_loose(df_tr)
        kl_te = make_keys_loose(df_te)

        tr_dict = {}
        for idx, key in enumerate(k_tr):
            if pd.notna(key):
                tr_dict.setdefault(key, []).append(idx)

        tr_dict_l = {}
        for idx, key in enumerate(kl_tr):
            if pd.notna(key):
                tr_dict_l.setdefault(key, []).append(idx)

        def build_features_for_partition(df, rates_df, is_train=True):
            N = len(df)
            E = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].values.astype(float)
            cr = df["AMT_CREDIT"].values.astype(float)
            an = df["AMT_ANNUITY"].values.astype(float)
            inc = df["AMT_INCOME_TOTAL"].values.astype(float)
            gp = df["AMT_GOODS_PRICE"].fillna(df["AMT_CREDIT"]).values.astype(float)
            q = (an / (cr + 1e-4))
            ct = (df["NAME_CONTRACT_TYPE"] == "Cash loans").values.astype(float)
            gb = df["DAYS_BIRTH"].values.astype(float)
            ge = df["DAYS_EMPLOYED"].values.astype(float)
            gp_ph = df["DAYS_LAST_PHONE_CHANGE"].values.astype(float)
            rate_vals = rates_df["RATE_BEST"].values.astype(float)

            keys = k_tr if is_train else k_te
            keys_l = kl_tr if is_train else kl_te

            has_p = np.zeros(N, dtype=float)
            p_n = np.zeros(N, dtype=float)
            gap = np.full(N, np.nan, dtype=float)
            has_pl = np.zeros(N, dtype=float)
            p_nl = np.zeros(N, dtype=float)
            tier_val = np.zeros(N, dtype=float)

            pe = np.full((N, 3), np.nan, dtype=float)
            dif = np.full((N, 3), np.nan, dtype=float)
            cr_r = np.full(N, np.nan, dtype=float)
            an_r = np.full(N, np.nan, dtype=float)
            inc_r = np.full(N, np.nan, dtype=float)
            same_ct = np.full(N, np.nan, dtype=float)
            qd = np.full(N, np.nan, dtype=float)
            drate = np.full(N, np.nan, dtype=float)
            dgoods = np.full(N, np.nan, dtype=float)

            tr_E = df_tr[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].values.astype(float)
            tr_cr = df_tr["AMT_CREDIT"].values.astype(float)
            tr_an = df_tr["AMT_ANNUITY"].values.astype(float)
            tr_inc = df_tr["AMT_INCOME_TOTAL"].values.astype(float)
            tr_gp = df_tr["AMT_GOODS_PRICE"].fillna(df_tr["AMT_CREDIT"]).values.astype(float)
            tr_q = (tr_an / (tr_cr + 1e-4))
            tr_ct = (df_tr["NAME_CONTRACT_TYPE"] == "Cash loans").values.astype(float)
            tr_gb = df_tr["DAYS_BIRTH"].values.astype(float)
            tr_ge = df_tr["DAYS_EMPLOYED"].values.astype(float)
            tr_ph = df_tr["DAYS_LAST_PHONE_CHANGE"].values.astype(float)
            tr_rate = r_tr["RATE_BEST"].values.astype(float)

            for i in range(N):
                ki = keys[i]
                if pd.notna(ki) and ki in tr_dict:
                    matches = [m for m in tr_dict[ki] if (not is_train or m != i)]
                    if len(matches) > 0:
                        has_p[i] = 1.0
                        p_n[i] = float(len(matches) + (1 if is_train else 0))
                        gaps = tr_gb[matches] - gb[i]
                        best_m = matches[np.argmin(np.abs(gaps))]
                        d_val = tr_gb[best_m] - gb[i]
                        gap[i] = d_val

                        # Check invariant match tiers
                        emp_ok = ((ge[i] == 365243) and (tr_ge[best_m] == 365243)) or \
                                 ((ge[i] - tr_ge[best_m] == d_val) and (ge[i] != 365243) and (tr_ge[best_m] != 365243))
                        ph_ok = (gp_ph[i] - tr_ph[best_m] == d_val) or (gp_ph[i] == tr_ph[best_m])
                        if emp_ok and ph_ok:
                            tier_val[i] = 2.0
                        elif emp_ok or ph_ok:
                            tier_val[i] = 1.0
                        else:
                            tier_val[i] = 0.5

                        # Continuous prior credit deltas
                        cr_r[i] = cr[i] / (tr_cr[best_m] + 1e-4)
                        an_r[i] = an[i] / (tr_an[best_m] + 1e-4)
                        inc_r[i] = inc[i] / (tr_inc[best_m] + 1e-4)
                        same_ct[i] = float(ct[i] == tr_ct[best_m])
                        qd[i] = q[i] - tr_q[best_m]
                        dgoods[i] = gp[i] - tr_gp[best_m]
                        if np.isfinite(rate_vals[i]) and np.isfinite(tr_rate[best_m]):
                            drate[i] = rate_vals[i] - tr_rate[best_m]

                        # LOO arithmetic for peer EXT: strictly other historical loans of applicant
                        for k_idx in range(3):
                            v_vals = [tr_E[m, k_idx] for m in matches]
                            v_vals = [v for v in v_vals if np.isfinite(v)]
                            pe[i, k_idx] = np.mean(v_vals) if len(v_vals) > 0 else np.nan
                            dif[i, k_idx] = E[i, k_idx] - tr_E[best_m, k_idx]

                kli = keys_l[i]
                if pd.notna(kli) and kli in tr_dict_l:
                    matches_l = [m for m in tr_dict_l[kli] if (not is_train or m != i)]
                    if len(matches_l) > 0:
                        has_pl[i] = 1.0
                        p_nl[i] = float(len(matches_l) + (1 if is_train else 0))

            f_part = pd.DataFrame({
                "HAS_PARTNER": has_p,
                "PARTNER_N": p_n,
                "PARTNER_GAP": gap,
                "PARTNER_ABSGAP": np.abs(gap),
                "HAS_PARTNER_LOOSE": has_pl,
                "PARTNER_N_LOOSE": p_nl,
                "PARTNER_TIER": tier_val
            }, index=df.index)

            f_pers = pd.DataFrame({
                "PERS_EXT1": np.where(has_p > 0, pe[:, 0], E[:, 0]),
                "PERS_EXT2": np.where(has_p > 0, pe[:, 1], E[:, 1]),
                "PERS_EXT3": np.where(has_p > 0, pe[:, 2], E[:, 2]),
                "PERS_EXT_MEAN": np.nanmean(np.column_stack([np.where(has_p > 0, pe[:, 0], E[:, 0]),
                                                             np.where(has_p > 0, pe[:, 1], E[:, 1]),
                                                             np.where(has_p > 0, pe[:, 2], E[:, 2])]), axis=1),
                "PDIF_EXT1": dif[:, 0],
                "PDIF_EXT2": dif[:, 1],
                "PDIF_EXT3": dif[:, 2],
                "PCR_RATIO": cr_r,
                "PAN_RATIO": an_r,
                "PINC_RATIO": inc_r,
                "PSAME_CT": same_ct,
                "PQ_DIFF": qd,
                "PRATE_DIFF": drate,
                "PGOODS_DIFF": dgoods
            }, index=df.index)

            return f_part, f_pers

        f_part_tr, f_pers_tr = build_features_for_partition(df_tr, r_tr, is_train=True)
        f_part_te, f_pers_te = build_features_for_partition(df_te, r_te, is_train=False)
        return f_part_tr, f_pers_tr, f_part_te, f_pers_te

    f_part_tr, f_pers_tr, f_part_te, f_pers_te = compute_person_deltas_inductive(df_train, df_test, f_rate_tr, f_rate_te)

    # 8. Complete 17 Decoded Product Keys (KS & M0)
    def compute_product_keys(df, cash_mask):
        cr = df["AMT_CREDIT"]
        an = df["AMT_ANNUITY"]
        gp = df["AMT_GOODS_PRICE"].fillna(cr)
        q = an / cr
        qb = (an - 129.5) / cr
        q5, qb5 = q.round(5), qb.round(5)
        fee = ((qb5.map(qb5.value_counts()).values > q5.map(q5.value_counts()).values) & cash_mask.values).astype(int)
        Q0 = pd.Series(np.where(fee, qb, q), index=df.index)

        CT = cr / an
        CT1 = CT.round(1).astype(str)
        CT2 = CT.round(2).astype(str)
        CT3 = CT.round(3).astype(str)

        ks_dict = {
            "Q3": q.round(3).astype(str), "Q4": q.round(4).astype(str), "Q5": q.round(5).astype(str),
            "P3": Q0.round(3).astype(str), "P4": Q0.round(4).astype(str), "P5": Q0.round(5).astype(str),
            "PF4": Q0.round(4).astype(str) + "|" + pd.Series(fee, index=df.index).astype(str),
            "CT1": CT1, "CT2": CT2, "CT3": CT3,
            "CT1_s10": CT1, "CT1_s200": CT1,
            "CT1_GP": CT1 + "|" + gp.astype(str),
            "CT1_CR": CT1 + "|" + cr.astype(str),
            "CT1_AN": CT1 + "|" + an.astype(str),
            "GP": gp.astype(str), "CR": cr.astype(str), "AN": an.astype(str)
        }
        return ks_dict, Q0, fee

    KS_tr, Q0_tr, FEE_tr = compute_product_keys(df_train, cash_tr)
    KS_te, Q0_te, FEE_te = compute_product_keys(df_test, cash_te)

    M0 = {
        "Q3": 100., "Q4": 30., "Q5": 30., "P3": 100., "P4": 30., "P5": 30., "PF4": 30.,
        "CT1": 50., "CT2": 50., "CT3": 50., "CT1_s10": 10., "CT1_s200": 200.,
        "CT1_GP": 50., "CT1_CR": 50., "CT1_AN": 50., "GP": 50., "CR": 50., "AN": 50.
    }

    # Assemble core and all-family manifolds with E_poly and implicit rate blocks
    X_core_tr = pd.concat([X_base_tr, f_cl_tr, f_fi_tr, f_ag_tr, f_bu_tr, f_part_tr, f_pers_tr, f_rate_tr], axis=1)
    X_core_te = pd.concat([X_base_te, f_cl_te, f_fi_te, f_ag_te, f_bu_te, f_part_te, f_pers_te, f_rate_te], axis=1)

    X_allf_tr = pd.concat([X_core_tr, f_ex_tr, f_count_tr, f_grp_tr, f_poly_tr], axis=1)
    X_allf_te = pd.concat([X_core_te, f_ex_te, f_count_te, f_grp_te, f_poly_te], axis=1)

    X_core_k_tr = pd.concat([X_core_tr, f_kd_tr], axis=1)
    X_core_k_te = pd.concat([X_core_te, f_kd_te], axis=1)

    X_allf_k_tr = pd.concat([X_allf_tr, f_kd_tr], axis=1)
    X_allf_k_te = pd.concat([X_allf_te, f_kd_te], axis=1)

    # CatBoost Product Specialist frames
    cb_num_tr = X_allf_tr.drop(columns=CATS).reset_index(drop=True)
    cb_num_te = X_allf_te.drop(columns=CATS).reset_index(drop=True)
    cb_cat_tr = df_train[CATS].copy().reset_index(drop=True)
    cb_cat_te = df_test[CATS].copy().reset_index(drop=True)
    for c in CATS:
        cb_cat_tr[c] = cb_cat_tr[c].fillna("__MISSING__").astype(str)
        cb_cat_te[c] = cb_cat_te[c].fillna("__MISSING__").astype(str)

    cb_prod_tr = pd.concat([cb_num_tr, cb_cat_tr], axis=1)
    cb_prod_te = pd.concat([cb_num_te, cb_cat_te], axis=1)

    prod_names = []
    for k_ in ["Q3", "Q4", "Q5", "P3", "P4", "P5", "PF4"] + [k for k in ("CT1", "CT2", "CT3", "CT1_GP", "CT1_CR", "CT1_AN") if k in KS_tr]:
        cb_prod_tr["PROD_" + k_] = KS_tr[k_].reset_index(drop=True).fillna("__MISSING__").astype(str)
        cb_prod_te["PROD_" + k_] = KS_te[k_].reset_index(drop=True).fillna("__MISSING__").astype(str)
        prod_names.append("PROD_" + k_)

    cb_prod_tr["PROD_K_DEC"] = K_tr.round().astype("Int64").astype("string").fillna("__MISSING__").astype(str).values
    cb_prod_te["PROD_K_DEC"] = K_te.round().astype("Int64").astype("string").fillna("__MISSING__").astype(str).values
    prod_names.append("PROD_K_DEC")

    cb_prod_tr["PROD_FEE"] = pd.Series(FEE_tr).reset_index(drop=True).astype(str).values
    cb_prod_te["PROD_FEE"] = pd.Series(FEE_te).reset_index(drop=True).astype(str).values
    prod_names.append("PROD_FEE")

    CB_CATS = CATS + prod_names

    print(f"[+] Grand Champion feature manifolds constructed in {time.time()-t0:.1f}s")
    return {
        "ntr": ntr, "nte": nte,
        "Xc": X_core_tr, "Xct": X_core_te,
        "Xa": X_allf_tr, "Xat": X_allf_te,
        "Xck": X_core_k_tr, "Xckt": X_core_k_te,
        "Xak": X_allf_k_tr, "Xakt": X_allf_k_te,
        "cb_tr": cb_prod_tr, "cb_te": cb_prod_te, "CB_CATS": CB_CATS,
        "KS_tr": KS_tr, "KS_te": KS_te, "M0": M0,
        "Q0_tr": Q0_tr, "Q0_te": Q0_te
    }

# ==============================================================================
# SECTION 3: 16 LEARNED EXPECTATION RESIDUALS & ALIGNED MULTI-VIEWS
# ==============================================================================

def compute_residuals_inductive(df_train, df_test, fast=False):
    """
    Fits 16 expectation regressors strictly on df_train using 5-fold cross-validation.
    Evaluates out-of-sample on df_test.
    """
    print("[*] Computing 16 demographic expectation residuals (Strict Train Fit)...")
    t0 = time.time()
    ntr = len(df_train)
    raw_tr = df_train.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore").copy()
    raw_te = df_test.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore").copy()

    for c in CATS:
        if c in raw_tr.columns:
            m = {v: float(i) for i, v in enumerate(raw_tr[c].dropna().astype(str).unique())}
            raw_tr[c] = raw_tr[c].astype(str).map(m).fillna(-1.0).astype(float)
            raw_te[c] = raw_te[c].astype(str).map(m).fillna(-1.0).astype(float)

    raw_tr["DAYS_EMPLOYED"] = raw_tr["DAYS_EMPLOYED"].where(raw_tr["DAYS_EMPLOYED"] != 365243)
    raw_te["DAYS_EMPLOYED"] = raw_te["DAYS_EMPLOYED"].where(raw_te["DAYS_EMPLOYED"] != 365243)

    demo = [c for c in raw_tr.columns if c.startswith("EXT_SOURCE") or c.startswith("AMT_")]

    T = {
        "EXT1": (raw_tr["EXT_SOURCE_1"], raw_te["EXT_SOURCE_1"], ["EXT_SOURCE_1"]),
        "EXT2": (raw_tr["EXT_SOURCE_2"], raw_te["EXT_SOURCE_2"], ["EXT_SOURCE_2"]),
        "EXT3": (raw_tr["EXT_SOURCE_3"], raw_te["EXT_SOURCE_3"], ["EXT_SOURCE_3"]),
        "LOGINC": (np.log(raw_tr["AMT_INCOME_TOTAL"]), np.log(raw_te["AMT_INCOME_TOTAL"]), ["AMT_INCOME_TOTAL"]),
        "LOGCR": (np.log(raw_tr["AMT_CREDIT"]), np.log(raw_te["AMT_CREDIT"]), ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]),
        "EMP": (raw_tr["DAYS_EMPLOYED"], raw_te["DAYS_EMPLOYED"], ["DAYS_EMPLOYED"]),
        "REG": (raw_tr["DAYS_REGISTRATION"], raw_te["DAYS_REGISTRATION"], ["DAYS_REGISTRATION"]),
        "AGE": (raw_tr["DAYS_BIRTH"], raw_te["DAYS_BIRTH"], ["DAYS_BIRTH"]),
        "IDP": (raw_tr["DAYS_ID_PUBLISH"], raw_te["DAYS_ID_PUBLISH"], ["DAYS_ID_PUBLISH"]),
        "PHONE": (raw_tr["DAYS_LAST_PHONE_CHANGE"], raw_te["DAYS_LAST_PHONE_CHANGE"], ["DAYS_LAST_PHONE_CHANGE"]),
        "REGPOP": (raw_tr["REGION_POPULATION_RELATIVE"], raw_te["REGION_POPULATION_RELATIVE"], ["REGION_POPULATION_RELATIVE"]),
        "FAM": (raw_tr["CNT_FAM_MEMBERS"], raw_te["CNT_FAM_MEMBERS"], ["CNT_FAM_MEMBERS"]),
        "CAR": (raw_tr["OWN_CAR_AGE"], raw_te["OWN_CAR_AGE"], ["OWN_CAR_AGE"]),
        "EXT1D": (raw_tr["EXT_SOURCE_1"], raw_te["EXT_SOURCE_1"], demo),
        "EXT2D": (raw_tr["EXT_SOURCE_2"], raw_te["EXT_SOURCE_2"], demo),
        "EXT3D": (raw_tr["EXT_SOURCE_3"], raw_te["EXT_SOURCE_3"], demo)
    }

    out_tr = {}
    out_te = {}
    n_splits = 2 if fast else 5
    max_iter = 15 if fast else 60

    for nm, (yt_tr, yt_te, drop_cols) in T.items():
        Xp_tr = raw_tr.drop(columns=drop_cols)
        Xp_te = raw_te.drop(columns=drop_cols)
        pred_tr = np.zeros(ntr)
        preds_te = []
        obs = yt_tr.notna().values

        for tri, tei in KFold(n_splits, shuffle=True, random_state=0).split(Xp_tr):
            m = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, random_state=0)
            m.fit(Xp_tr.iloc[tri][obs[tri]], yt_tr.iloc[tri][obs[tri]])
            pred_tr[tei] = m.predict(Xp_tr.iloc[tei])
            preds_te.append(m.predict(Xp_te))

        p_te = np.mean(preds_te, axis=0)
        out_tr[f"RP_{nm}"] = pred_tr
        out_tr[f"RR_{nm}"] = np.where(obs, yt_tr.values - pred_tr, np.nan)
        out_te[f"RP_{nm}"] = p_te
        out_te[f"RR_{nm}"] = np.where(yt_te.notna().values, yt_te.values - p_te, np.nan)

    res_tr = pd.DataFrame(out_tr, index=df_train.index)
    res_te = pd.DataFrame(out_te, index=df_test.index)
    print(f"[+] 16 Demographic Expectation Residuals calculated in {time.time()-t0:.1f}s")
    return res_tr, res_te

def compute_multiviews_inductive(bf, y_train, extra_views=False, variant="A", fast=False):
    """
    Constructs multi-view meta-predictions strictly aligned per fold.
    For training folds: Uses 3-fold inner cross-validation on fold training rows.
    For test data: Uses average of inner fold predictions.
    """
    view_name = f"Multi-View {'ExtraViews (v18)' if extra_views else 'Standard (v17)'} [Variant {variant}]"
    print(f"[*] Computing {view_name}...")
    t0 = time.time()
    ntr = bf["ntr"]

    Xs = bf["Xa"].copy()
    Xts = bf["Xat"].copy()
    Xs["Q0num"] = bf["Q0_tr"].values
    Xts["Q0num"] = bf["Q0_te"].values
    cols = list(Xs.columns)

    ext = [c for c in cols if "EXT" in c or c in ("E12", "E13", "E23", "D12", "D13", "D23")]
    loan = [c for c in cols if c.startswith(("AMT_", "LOG_", "CREDIT_", "ANN_", "GOODS_", "TERM", "Q0num", "RATE_")) or c in ("CREDIT_MINUS_GOODS", "RP_LOGINC", "RR_LOGINC", "RP_LOGCR", "RR_LOGCR")]
    person = [c for c in cols if c not in set(ext) | set(loan) and not c.startswith(("G_", "CNT_"))]
    views = {"EXT": ext, "LOAN": loan, "PERSON": person}

    if extra_views:
        agec = [c for c in cols if any(t in c for t in ("AGE", "BIRTH", "EMPLOY", "EMP_", "ID_", "REG_", "PHONE", "DAYS"))]
        grpc = [c for c in cols if c.startswith(("G_", "CNT_")) or c in CATS]
        views.update({
            "EXTAGE": sorted(set(ext + agec)),
            "NOEXT": sorted(set(loan + person)),
            "GROUPS": sorted(set(grpc + loan[:6]))
        })

    def lg(p):
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        return np.log(p / (1.0 - p))

    def vf(d):
        df = pd.DataFrame({f"VIEW_{k}": v for k, v in d.items()})
        df["VD_EL"] = df.VIEW_EXT - df.VIEW_LOAN
        df["VD_EP"] = df.VIEW_EXT - df.VIEW_PERSON
        df["VD_LP"] = df.VIEW_LOAN - df.VIEW_PERSON
        return df

    mk = lambda: HistGradientBoostingClassifier(
        max_iter=10 if fast else 40, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=20, random_state=0
    )

    out = {"tr": [], "va": [], "te": []}
    skf = StratifiedKFold(2 if fast else 5, shuffle=True, random_state=42)

    for k, (a, b) in enumerate(skf.split(np.zeros(ntr), y_train)):
        trv, vav, tev = {}, {}, {}
        for vn, vc in views.items():
            inner = np.zeros(len(a))
            pvb, pte = [], []
            for ia, ib in StratifiedKFold(2 if fast else 3, shuffle=True, random_state=11).split(a, y_train[a]):
                m = mk().fit(Xs.iloc[a[ia]][vc], y_train[a[ia]])
                inner[ib] = m.predict_proba(Xs.iloc[a[ib]][vc])[:, 1]
                if variant == "A":
                    pvb.append(m.predict_proba(Xs.iloc[b][vc])[:, 1])
                    pte.append(m.predict_proba(Xts[vc])[:, 1])
            trv[vn] = lg(inner)
            if variant == "A":
                vav[vn] = lg(np.mean(pvb, axis=0))
                tev[vn] = lg(np.mean(pte, axis=0))
            else:
                m = mk().fit(Xs.iloc[a][vc], y_train[a])
                vav[vn] = lg(m.predict_proba(Xs.iloc[b][vc])[:, 1])
                tev[vn] = lg(m.predict_proba(Xts[vc])[:, 1])
        out["tr"].append(vf(trv))
        out["va"].append(vf(vav))
        out["te"].append(vf(tev))

    # All-data fit
    trv, tev = {}, {}
    for vn, vc in views.items():
        inner = np.zeros(ntr)
        pte = []
        for ia, ib in StratifiedKFold(2 if fast else 3, shuffle=True, random_state=11).split(np.zeros(ntr), y_train):
            m = mk().fit(Xs.iloc[ia][vc], y_train[ia])
            inner[ib] = m.predict_proba(Xs.iloc[ib][vc])[:, 1]
            if variant == "A":
                pte.append(m.predict_proba(Xts[vc])[:, 1])
        trv[vn] = lg(inner)
        tev[vn] = lg(np.mean(pte, axis=0)) if variant == "A" else lg(mk().fit(Xs[vc], y_train).predict_proba(Xts[vc])[:, 1])
    out["all_tr"] = vf(trv)
    out["all_te"] = vf(tev)
    print(f"[+] {view_name} finished in {time.time()-t0:.1f}s")
    return out

# ==============================================================================
# SECTION 4: IN-FOLD BAYESIAN TARGET ENCODINGS & PRECOMPUTED CACHE
# ==============================================================================

def build_target_encoders(bf, y_train):
    """
    Constructs Bayesian Target Encodings strictly on y_train[fit].
    Zero partner_enc target transfer across applicants.
    Strictly preserves vocabulary indexing between train factorizations and test lookups.
    """
    KS_tr = bf["KS_tr"]
    KS_te = bf["KS_te"]
    M0 = bf["M0"]

    KC_tr = {}
    KN_tr = {}
    CMAP_tr = {}
    for k in KS_tr:
        codes, uniques = pd.factorize(KS_tr[k])
        KC_tr[k] = codes
        KN_tr[k] = len(uniques) + 1
        CMAP_tr[k] = {val: code for code, val in enumerate(uniques)}

    def enc(fit, qi, is_test=False):
        pr = float(y_train[fit].mean())
        yf = y_train[fit].astype(float)
        out = {}
        for k in KS_tr:
            c = KC_tr[k]
            unseen_slot = KN_tr[k] - 1
            cnt_t = np.bincount(c[fit], minlength=KN_tr[k]).astype(float)
            sm_t = np.bincount(c[fit], weights=yf, minlength=KN_tr[k])

            if is_test:
                s_te = pd.Series(KS_te[k])
                cq = s_te.map(CMAP_tr[k]).fillna(unseen_slot).astype(int).values
                cq = np.clip(cq, 0, unseen_slot)
            else:
                cq = c[qi]

            out[f"TE_{k}"] = (sm_t[cq] + M0[k] * pr) / (cnt_t[cq] + M0[k])
            out[f"CNT_{k}"] = cnt_t[cq]
        return pd.DataFrame(out)

    def train_enc(a):
        cols = enc(a[:10], a[:5], is_test=False).columns
        inner = np.zeros((len(a), len(cols)))
        for ia, ib in StratifiedKFold(5, shuffle=True, random_state=7).split(a, y_train[a]):
            inner[ib] = enc(a[ia], a[ib], is_test=False).values
        return pd.DataFrame(inner, columns=cols)

    return enc, train_enc

def onehot_train_test(Xa_, Xt_):
    """
    One-hot encodes only the 8 low-cardinality CATS columns.
    All high-cardinality columns remain integer factorized.
    """
    oh_tr = pd.get_dummies(Xa_[CATS].astype("category"), dummy_na=True, dtype=np.float32)
    oh_te = pd.get_dummies(Xt_[CATS].astype("category"), dummy_na=True, dtype=np.float32)

    oh_tr, oh_te = oh_tr.align(oh_te, join="left", axis=1, fill_value=0.0)
    oh_tr.columns = [f"OH_{i}" for i in range(oh_tr.shape[1])]
    oh_te.columns = [f"OH_{i}" for i in range(oh_te.shape[1])]

    Za = pd.concat([Xa_.drop(columns=CATS).reset_index(drop=True), oh_tr.reset_index(drop=True)], axis=1)
    Zt = pd.concat([Xt_.drop(columns=CATS).reset_index(drop=True), oh_te.reset_index(drop=True)], axis=1)
    return Za.astype(np.float32), Zt.astype(np.float32)

# ==============================================================================
# SECTION 5: AUTHENTIC 5-MEMBER-SET DUAL-STREAM CHAMPION ZOO
# ==============================================================================

def train_full_champion_zoo(bf, F1, BASEF, y_train, ntr, nte, tidx, views_o, views_a, views_ox, fast=False):
    print(f"\n{'='*70}\n[+] TRAINING AUTHENTIC 5-MEMBER-SET GRAND CHAMPION ZOO\n{'='*70}")
    t_start = time.time()

    enc, train_enc = build_target_encoders(bf, y_train)

    FOLDS = 2 if fast else 5
    folds_iter = list(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(ntr), y_train))

    # Precompute target encodings ONCE per fold to eliminate redundant recomputations across 25 members
    print("[*] Precomputing Bayesian Target Encodings once per fold...")
    t_enc0 = time.time()
    TE_train_folds = [train_enc(a) for a, b in folds_iter]
    TE_val_folds = [enc(a, b, is_test=False) for a, b in folds_iter]
    TE_test_folds = [enc(a, tidx, is_test=True) for a, b in folds_iter]

    allr = np.arange(ntr)
    TE_all_train = train_enc(allr)
    TE_all_test = enc(allr, tidx, is_test=True)
    print(f"[+] Target Encodings precomputed in {time.time()-t_enc0:.1f}s")

    def build_specs(frames, mono_te=False):
        Xc_, Xct_, Xa_, Xat_, Xck_, Xckt_, Xak_, Xakt_ = [frames[k] for k in ("Xc", "Xct", "Xa", "Xat", "Xck", "Xckt", "Xak", "Xakt")]
        Xco, Xcot = onehot_train_test(Xc_, Xct_)
        Xao, Xaot = onehot_train_test(Xa_, Xat_)
        Xaok, Xaokt = onehot_train_test(Xak_, Xakt_)

        Xck_clean = Xck_.drop(columns=CATS, errors="ignore").reset_index(drop=True).astype(np.float32)
        Xckt_clean = Xckt_.drop(columns=CATS, errors="ignore").reset_index(drop=True).astype(np.float32)

        KS_tr = bf["KS_tr"]
        MONO_K = [-1 if c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") else 0 for c in Xck_clean.columns] + \
                 ([s_ for _ in KS_tr for s_ in (1, 0)] if mono_te else [0] * (2 * len(KS_tr)))

        R_qf_shallow = 30 if fast else 450
        Rf_qf_shallow = 30 if fast else 500
        R_qf_reg = 30 if fast else 600
        Rf_qf_reg = 30 if fast else 650
        R_v1 = 30 if fast else 400
        Rf_v1 = 30 if fast else 450
        R_v2 = 30 if fast else 400
        Rf_v2 = 30 if fast else 450
        R_mono = 30 if fast else 500
        Rf_mono = 30 if fast else 550

        return {
            "qf_shallow":   (Xco,  Xcot,  dict(learning_rate=0.05, num_leaves=7, max_depth=3, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20, feature_pre_filter=False), R_qf_shallow, Rf_qf_shallow, [42, 7]),
            "qf_reg":       (Xco,  Xcot,  dict(learning_rate=0.03, num_leaves=11, min_child_samples=300, feature_fraction=0.4, bagging_fraction=0.8, lambda_l2=40, feature_pre_filter=False), R_qf_reg, Rf_qf_reg, [42, 7]),
            "v1_allfam_oh": (Xao,  Xaot,  dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20, feature_pre_filter=False), R_v1, Rf_v1, [42, 7]),
            "v2_allfam_oh": (Xaok, Xaokt, dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20, feature_pre_filter=False), R_v2, Rf_v2, [42, 7]),
            "v2_mono":      (Xck_clean, Xckt_clean, dict(learning_rate=0.02, num_leaves=15, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20, feature_pre_filter=False, monotone_constraints=MONO_K), R_mono, Rf_mono, [42, 7])
        }

    SPEC0 = build_specs(BASEF, mono_te=False)
    SPEC1 = build_specs(F1, mono_te=False)
    SPEC1_ET = {nm: (X, Xt, {**p, "extra_trees": True}, R, Rf, sd) for nm, (X, Xt, p, R, Rf, sd) in SPEC1.items()}
    SPEC1_ET_V18 = build_specs(F1, mono_te=True)
    SPEC1_ET_V18 = {nm: (X, Xt, {**p, "extra_trees": True}, R, Rf, sd) for nm, (X, Xt, p, R, Rf, sd) in SPEC1_ET_V18.items()}

    BASE_LGB = dict(objective="binary", force_col_wise=True, bagging_freq=1, verbose=-1, num_threads=THREADS, seed=42)
    W = {"qf_shallow": 0.24, "qf_reg": 0.12, "v1_allfam_oh": 0.32, "v2_allfam_oh": 0.16, "v2_mono": 0.16}

    def train_member(name, spec, tag, view=None):
        t_m = time.time()
        X, Xt, p, R, Rf, seeds = spec[name]
        if view is not None and "monotone_constraints" in p:
            p = {**p, "monotone_constraints": list(p["monotone_constraints"]) + [0] * view["tr"][0].shape[1]}
        tf = np.zeros(nte)
        oof = np.zeros(ntr)
        ex = lambda key, k=None: ([] if view is None else [view[key] if k is None else view[key][k]])

        for k, (a, b) in enumerate(folds_iter):
            fa = pd.concat([X.iloc[a].reset_index(drop=True), TE_train_folds[k]] + ex("tr", k), axis=1)
            fb = pd.concat([X.iloc[b].reset_index(drop=True), TE_val_folds[k]] + ex("va", k), axis=1)
            ft = pd.concat([Xt.reset_index(drop=True), TE_test_folds[k]] + ex("te", k), axis=1)

            fa_np = fa.values.astype(np.float32)
            fb_np = fb.values.astype(np.float32)
            ft_np = ft.values.astype(np.float32)

            m = lgb.train({**BASE_LGB, **p}, lgb.Dataset(fa_np, y_train[a]), R)
            oof[b] = m.predict(fb_np)
            tf += m.predict(ft_np) / FOLDS

        fa_all = pd.concat([X.reset_index(drop=True), TE_all_train] + ex("all_tr"), axis=1)
        ft_all = pd.concat([Xt.reset_index(drop=True), TE_all_test] + ex("all_te"), axis=1)
        fa_np = fa_all.values.astype(np.float32)
        ft_np = ft_all.values.astype(np.float32)

        fr = []
        for sd in seeds[:1 if fast else len(seeds)]:
            fr.append(rk(lgb.train({**BASE_LGB, **p, "seed": sd}, lgb.Dataset(fa_np, y_train), Rf).predict(ft_np)))
        test = rk(0.5 * rk(tf) + 0.5 * np.mean(fr, 0))
        auc = roc_auc_score(y_train, oof)
        print(f"  [{tag:10s} | {name:15s}] OOF AUC: {auc:.5f} ({time.time()-t_m:.1f}s)")
        return test, oof

    # 1. Train 5 Member Sets Live
    RESV_17 = {}
    print("\n--- Model Set: old (Baseline Features - Spec0) ---")
    for nm in W: RESV_17[("old", nm)] = train_member(nm, SPEC0, "old")

    print("\n--- Model Set: e1 (Residuals + 3-View Manifold - Spec1) ---")
    for nm in W: RESV_17[("e1", nm)] = train_member(nm, SPEC1, "e1", views_o)

    print("\n--- Model Set: e1et (ExtraTrees Regularization - Spec1_ET) ---")
    for nm in W: RESV_17[("e1et", nm)] = train_member(nm, SPEC1_ET, "e1et", views_o)

    print("\n--- Model Set: e1Aet (Consistent View Variant A - Spec1_ET) ---")
    for nm in W: RESV_17[("e1Aet", nm)] = train_member(nm, SPEC1_ET, "e1Aet", views_a)

    RESV_18 = {}
    print("\n--- Model Set: e1et_v18 (6-View Manifold + Monotone Target Encodings) ---")
    for nm in W: RESV_18[("e1et_v18", nm)] = train_member(nm, SPEC1_ET_V18, "e1et_v18", views_ox)

    # 2. CatBoost Product Specialist on Decoded Product Manifold (GPU Accelerated, Depth 7, 700 trees)
    print("\n[*] Training CatBoost Product Specialist on GPU...")
    t_cb = time.time()
    cb_train = bf["cb_tr"].reset_index(drop=True)
    cb_test = bf["cb_te"].reset_index(drop=True)
    CB_CATS = bf["CB_CATS"]

    CBP = dict(iterations=30 if fast else 700, depth=7, learning_rate=0.05, loss_function="Logloss",
               eval_metric="Logloss", l2_leaf_reg=8.0, random_seed=42, task_type="GPU", verbose=False)
    cb_oof = np.zeros(ntr)
    cb_tests = []

    for fi, (a, b) in enumerate(folds_iter, 1):
        ca = cb_train.iloc[a].reset_index(drop=True)
        cbv = cb_train.iloc[b].reset_index(drop=True)
        cbt = cb_test.reset_index(drop=True)

        cb_model = CatBoostClassifier(**CBP)
        cb_model.fit(ca, y_train[a], cat_features=CB_CATS, verbose=False)
        pv = cb_model.predict_proba(cbv)[:, 1]
        pt = cb_model.predict_proba(cbt)[:, 1]
        cb_oof[b] = pv
        cb_tests.append(pt)
        print(f"  [CatBoost Fold {fi}/{FOLDS}] AUC: {roc_auc_score(y_train[b], pv):.5f}")

    cb_test_mean = np.mean(cb_tests, 0)
    print(f"[+] CatBoost Product Specialist Complete | Total OOF AUC: {roc_auc_score(y_train, cb_oof):.5f} ({time.time()-t_cb:.1f}s)")

    # 3. Dual-Stream Champion Assembly
    blend17 = lambda tag, i: sum(W[k] * (RESV_17[(tag, k)][0] if i == 0 else rk(RESV_17[(tag, k)][1])) for k in W)
    B17 = lambda i: (blend17("old", i) + blend17("e1", i) + blend17("e1et", i)) / 3.0
    v17_test = 0.90 * rk(0.75 * rk(B17(0)) + 0.25 * blend17("e1Aet", 0)) + 0.10 * rk(cb_test_mean)
    v17_oof = 0.90 * rk(0.75 * rk(B17(1)) + 0.25 * blend17("e1Aet", 1)) + 0.10 * rk(cb_oof)

    blend18 = lambda tag, i: sum(W[k] * (RESV_18[(tag, k)][0] if i == 0 else rk(RESV_18[(tag, k)][1])) for k in W) if tag in ("e1et_v18",) else blend17(tag, i)
    B18 = lambda i: (blend18("old", i) + blend18("e1", i) + blend18("e1et_v18", i)) / 3.0
    v18_test = 0.90 * rk(0.75 * rk(B18(0)) + 0.25 * blend18("e1Aet", 0)) + 0.10 * rk(cb_test_mean)
    v18_oof = 0.90 * rk(0.75 * rk(B18(1)) + 0.25 * blend18("e1Aet", 1)) + 0.10 * rk(cb_oof)

    auc_17 = roc_auc_score(y_train, v17_oof)
    auc_18 = roc_auc_score(y_train, v18_oof)
    print(f"\n>>> STREAM V17 GRAND CHAMPION OOF AUC: {auc_17:.6f} <<<")
    print(f">>> STREAM V18 GRAND CHAMPION OOF AUC: {auc_18:.6f} <<<")

    # 4. Convex Dual-Champion Rank Fusion (0.55 * v17 + 0.45 * v18)
    print("\n" + "=" * 80)
    print("DUAL-CHAMPION CONVEX RANK FUSION: v22 = 0.55 * rank(v17) + 0.45 * rank(v18)")
    print("=" * 80)
    r17 = rk_norm(v17_test)
    r18 = rk_norm(v18_test)
    v22_test = rk_norm(0.55 * r17 + 0.45 * r18)

    oof_r17 = rk_norm(v17_oof)
    oof_r18 = rk_norm(v18_oof)
    v22_oof = rk_norm(0.55 * oof_r17 + 0.45 * oof_r18)
    v22_auc = roc_auc_score(y_train, v22_oof)

    print(f"\n>>> V12 ALL-TIME RECORD CHAMPION POOLED OOF AUC: {v22_auc:.6f} (Total Time: {time.time()-t_start:.1f}s) <<<")
    return v22_test, v22_oof, v22_auc

# ==============================================================================
# SECTION 6: UNIFIED EXECUTION & FINAL SUBMISSION ASSERTIONS
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="V22 Certified V12 Grand Champion")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode")
    args = parser.parse_args()

    total_start = time.time()
    set_all_seeds(SEED)
    print("=" * 80)
    print(f"  GCI World 2026: Re-Engineered V22 Champion V12 (Grand Champion Restoration) [Mode: {'FAST SMOKE' if args.fast else 'FULL PRODUCTION'}]")
    print("=" * 80)

    tr_p, te_p, sm_p = resolve_data_paths()
    df_train = pd.read_csv(tr_p)
    df_test = pd.read_csv(te_p)
    sample_sub = pd.read_csv(sm_p)

    assert df_test["SK_ID_CURR"].is_unique, "Test SK_ID_CURR must be unique"
    assert sample_sub["SK_ID_CURR"].is_unique, "Sample submission SK_ID_CURR must be unique"
    assert sample_sub["SK_ID_CURR"].equals(df_test["SK_ID_CURR"]), (
        "Sample submission IDs do not match test IDs in exact order."
    )

    y_train = df_train["TARGET"].values.astype(np.int32)
    ntr, nte = len(df_train), len(df_test)
    tidx = np.arange(nte)
    print(f"[*] Input shapes: df_train={df_train.shape}, df_test={df_test.shape}")

    # Enforce official input hashes for provenance verification
    for label, p_ in [("train.csv", tr_p), ("test.csv", te_p), ("sample_submission.csv", sm_p)]:
        if label not in OFFICIAL_HASHES:
            raise ValueError(f"CRITICAL: Unknown input file '{label}'")
        sha = hashlib.sha256(open(p_, "rb").read()).hexdigest().upper()
        if sha != OFFICIAL_HASHES[label]:
            raise ValueError(f"CRITICAL: Official input hash mismatch for {label}: got {sha}, expected {OFFICIAL_HASHES[label]}")
        print(f"  [PROVENANCE] {label:22s} SHA-256: {sha} (MATCH - ENFORCED)")

    # 1. Inductive Feature Manifolds
    bf = build_all_features_inductive(df_train, df_test)

    # 2. 16 Demographic Residuals
    res_tr, res_te = compute_residuals_inductive(df_train, df_test, fast=args.fast)
    BASEF = {
        "Xc": bf["Xc"], "Xct": bf["Xct"],
        "Xa": bf["Xa"], "Xat": bf["Xat"],
        "Xck": bf["Xck"], "Xckt": bf["Xckt"],
        "Xak": bf["Xak"], "Xakt": bf["Xakt"]
    }
    F1 = {
        "Xc": pd.concat([bf["Xc"], res_tr], axis=1),
        "Xct": pd.concat([bf["Xct"], res_te], axis=1),
        "Xa": pd.concat([bf["Xa"], res_tr], axis=1),
        "Xat": pd.concat([bf["Xat"], res_te], axis=1),
        "Xck": pd.concat([bf["Xck"], res_tr], axis=1),
        "Xckt": pd.concat([bf["Xckt"], res_te], axis=1),
        "Xak": pd.concat([bf["Xak"], res_tr], axis=1),
        "Xakt": pd.concat([bf["Xakt"], res_te], axis=1)
    }

    # 3. Multi-View Meta-Learners
    views_v17_o = compute_multiviews_inductive(bf, y_train, extra_views=False, variant="orig", fast=args.fast)
    views_v17_a = compute_multiviews_inductive(bf, y_train, extra_views=False, variant="A", fast=args.fast)
    views_v18_ox = compute_multiviews_inductive(bf, y_train, extra_views=True, variant="orig", fast=args.fast)

    # 4. Train Full 5-Member-Set Dual-Stream Champion Zoo
    v22_test, v22_oof, v22_auc = train_full_champion_zoo(
        bf, F1, BASEF, y_train, ntr, nte, tidx,
        views_v17_o, views_v17_a, views_v18_ox, fast=args.fast
    )

    # 5. Save and Validate Output CSV
    sub_path = r"d:/gci/v_22_pure_champion_ensemble_certified_v12.csv"
    sub = pd.DataFrame({
        "SK_ID_CURR": sample_sub["SK_ID_CURR"],
        "TARGET": v22_test
    })

    assert len(sub) == nte, f"Submission row count mismatch: {len(sub)} vs {nte}"
    assert sub["SK_ID_CURR"].is_unique, "Submission SK_ID_CURR contains duplicates"
    assert sub["SK_ID_CURR"].equals(df_test["SK_ID_CURR"]), (
        "Final submission IDs do not match test IDs in exact order."
    )
    assert not sub["TARGET"].isna().any(), "Submission TARGET contains NaNs"
    assert np.isfinite(sub["TARGET"].values).all(), "Submission TARGET contains non-finite values (Inf or NaN)"
    assert (sub["TARGET"] >= 0.0).all() and (sub["TARGET"] <= 1.0).all(), "Predictions out of bounds"

    sub.to_csv(sub_path, index=False)
    sub_sha = hashlib.sha256(open(sub_path, "rb").read()).hexdigest().upper()

    print(f"\n[+] Re-engineered V22 Certified V12 Submission saved to: {sub_path}")
    print(f"[+] Final Submission Rows: {len(sub):,} | SHA-256: {sub_sha}")
    print(f"[+] Total Pipeline Execution Time: {time.time()-total_start:.1f}s")

    # Diagnostic Rank Correlation Benchmarks
    bench_paths = [
        ("LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv", r"d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv"),
        ("submission_4_v22_champion_078239.csv", r"d:/gci/submission_4_v22_champion_078239.csv"),
        ("LEADERBOARD_SOLO_CHAMPION_V23.csv", r"d:/gci/LEADERBOARD_SOLO_CHAMPION_V23.csv"),
        ("v_22_pure_champion_ensemble_certified_v11.csv", r"d:/gci/v_22_pure_champion_ensemble_certified_v11.csv")
    ]
    print("\n" + "=" * 80)
    print("DIAGNOSTIC RANK CORRELATION BENCHMARK AUDIT")
    print("=" * 80)
    for bname, bpath in bench_paths:
        if os.path.exists(bpath):
            b_df = pd.read_csv(bpath)
            rho, _ = spearmanr(v22_test, b_df["TARGET"])
            print(f"  Spearman Correlation vs {bname:45s}: {rho:.6f} ({rho*100:.2f}%)")

    return sub_path, v22_auc

if __name__ == "__main__":
    main()

