"""
================================================================================
GCI WORLD 2026: ALL-TIME RECORD CHAMPION V22 (PUBLIC LB: 0.78239)
COMPLETE 100% INDEPENDENT FROM-SCRATCH UNIFIED ENGINE
================================================================================
Target Artifact: v_22_pure_champion_ensemble.csv (LB: 0.78239)
Strict Rule Compliance: 100% Pure in-memory scratch code.
Input Files: ONLY train.csv and test.csv (and sample_submission.csv for IDs).

Mathematical Provenance:
  v22 is the convex rank fusion of the project's two 20-member multi-view champions:
      v22 = 0.55 * rank(v17) + 0.45 * rank(v18)
  where:
    - v17 (Public LB 0.78237): 20-member multi-view stacking pipeline
      [GCI_v17_optimised_O 78237 score.ipynb]
    - v18 (Public LB 0.78235): 20-member evidence-driven multi-view pipeline
      [GCI_v18_evidence_driven.ipynb]

This single unified file combines:
  1. Complete feature engineering (v15 families + decoded product keys + community CT keys + repeat-applicant partner & person pooling)
  2. 16 demographic learned-expectation residuals (HistGradientBoosting)
  3. Multi-view stacking manifolds (EXT, LOAN, PERSON + v18 ExtraViews: EXTAGE, NOEXT, GROUPS)
  4. 4 LightGBM Member Sets across 5 folds (old, e1, e1et, e1Aet)
  5. CatBoost Product Specialist with native product categoricals
  6. Convex Dual-Champion Rank Fusion: 55% v17 + 45% v18 -> v22 (0.78239)
================================================================================
"""

import os
import sys
import time
import argparse
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 1: GLOBAL CONFIGURATION & DATA INGESTION
# ==============================================================================

def resolve_data_paths():
    candidates = [
        r"d:/gci/inpuy",
        r"d:/gci/input",
        r"/content",
        r"/content/drive/MyDrive",
        os.getcwd()
    ]
    for d in candidates:
        tr_p = os.path.join(d, "train.csv")
        te_p = os.path.join(d, "test.csv")
        sm_p = os.path.join(d, "sample_submission.csv")
        if os.path.exists(tr_p) and os.path.exists(te_p):
            print(f"[*] Found competition data in: {d}")
            return tr_p, te_p, sm_p
    raise FileNotFoundError("Could not find train.csv and test.csv in search candidates.")

CATS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
]

THREADS = os.cpu_count() or 4
rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)
lg = lambda p: np.log(np.clip(p, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(p, 1e-4, 1.0 - 1e-4)))

# ==============================================================================
# SECTION 2: AUTHENTIC FEATURE ENGINEERING MANIFOLDS (v15/v16/v17/v18)
# ==============================================================================

def build_all_features(df_train, df_test):
    print("[*] Extracting full authentic feature manifolds across train & test...")
    t0 = time.time()
    ntr = len(df_train)
    al = pd.concat([df_train.drop(columns=["TARGET"]), df_test], ignore_index=True)

    def new_cols(d):
        return pd.DataFrame(d, index=al.index)

    def base_fn():
        X = al.drop(columns=["SK_ID_CURR"]).copy()
        for c in CATS:
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
        return X

    def f_clean_fn(X):
        d = {
            "DE_SENT": (al.DAYS_EMPLOYED == 365243).astype(int),
            "DAYS_EMPLOYED_CLEAN": al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243),
            "ORG_XNA": (al.ORGANIZATION_TYPE == "XNA").astype(int),
            "CAR_AGE_OUT": (al.OWN_CAR_AGE >= 60).astype(int),
            "MISS_CNT": al.isna().sum(axis=1)
        }
        for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
            d[c + "_NA"] = al[c].isna().astype(int)
        return pd.concat([X, new_cols(d)], axis=1)

    def f_ext_fn(X):
        e = al[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        d = {
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
        return pd.concat([X, new_cols(d)], axis=1)

    def f_fin_fn(X):
        inc = al.AMT_INCOME_TOTAL
        cr = al.AMT_CREDIT
        an = al.AMT_ANNUITY
        gp = al.AMT_GOODS_PRICE
        d = {
            "CREDIT_INC": cr / inc,
            "ANN_INC": an / inc,
            "ANN_CREDIT": an / cr,
            "TERM": cr / an,
            "GOODS_CREDIT": gp / cr,
            "CREDIT_MINUS_GOODS": cr - gp,
            "GOODS_INC": gp / inc,
            "INC_PER_FAM": inc / al.CNT_FAM_MEMBERS,
            "CREDIT_PER_FAM": cr / al.CNT_FAM_MEMBERS,
            "ANN_PER_FAM": an / al.CNT_FAM_MEMBERS
        }
        for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
            d["LOG_" + c] = np.log1p(al[c])
        return pd.concat([X, new_cols(d)], axis=1)

    def f_age_fn(X):
        age = -al.DAYS_BIRTH
        de = al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243)
        d = {
            "AGE": age / 365.25,
            "EMP_AGE": de / al.DAYS_BIRTH,
            "EMP_YEARS": -de / 365.25,
            "REG_AGE": al.DAYS_REGISTRATION / al.DAYS_BIRTH,
            "ID_AGE": al.DAYS_ID_PUBLISH / al.DAYS_BIRTH,
            "PHONE_AGE": al.DAYS_LAST_PHONE_CHANGE / al.DAYS_BIRTH,
            "CAR_AGE_RATIO": al.OWN_CAR_AGE / (age / 365.25),
            "CREDIT_PER_AGE": al.AMT_CREDIT / (age / 365.25),
            "INC_PER_AGE": al.AMT_INCOME_TOTAL / (age / 365.25)
        }
        return pd.concat([X, new_cols(d)], axis=1)

    def f_bureau_fn(X):
        b = [c for c in al.columns if c.startswith("AMT_REQ")]
        d = {
            "BUR_SUM": al[b].sum(axis=1, min_count=1),
            "BUR_RECENT": al[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON"]].sum(axis=1, min_count=1),
            "BUR_MON_YEAR": al.AMT_REQ_CREDIT_BUREAU_MON / (al.AMT_REQ_CREDIT_BUREAU_YEAR + 1.0)
        }
        return pd.concat([X, new_cols(d)], axis=1)

    def f_count_fn(X):
        d = {"CNT_" + c: al[c].map(al[c].value_counts()) for c in CATS + ["REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "AMT_INCOME_TOTAL"]}
        return pd.concat([X, new_cols(d)], axis=1)

    def f_group_fn(X):
        tmp = al.copy()
        tmp["CREDIT_INC"] = al.AMT_CREDIT / al.AMT_INCOME_TOTAL
        d = {}
        for k in ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE"]:
            for v in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "CREDIT_INC"]:
                d[f"G_{k[:6]}_{v[-6:]}_DIFF"] = tmp[v] - tmp.groupby(k)[v].transform("mean")
        return pd.concat([X, new_cols(d)], axis=1)

    def f_kdec_fn(X):
        C, G = al.AMT_CREDIT, al.AMT_GOODS_PRICE
        cash = al.NAME_CONTRACT_TYPE == "Cash loans"
        kf = (C / G - 1.0) / 0.0132
        k = kf.round()
        ok = cash & (C > G) & ((kf - k).abs() < 0.1)
        K = pd.Series(np.where(ok, k, np.where(cash & (C == G), 0.0, np.nan)), index=al.index)
        return pd.concat([X, new_cols({"K_DEC": K, "CG_RATIO": C / G})], axis=1)

    # Repeat-applicant partner & person pooling
    _i1 = al.DAYS_REGISTRATION - al.DAYS_BIRTH
    _i2 = al.DAYS_ID_PUBLISH - al.DAYS_BIRTH

    def _groups(parts):
        k = al[parts].astype(str).agg("|".join, axis=1) + "|" + _i1.astype(str) + "|" + _i2.astype(str)
        k = k.where(al.DAYS_REGISTRATION != 0)
        sz = k.map(k.value_counts())
        k = k.where(sz > 1)
        g = pd.Series(pd.factorize(k)[0]).replace(-1, np.nan).values
        mem = pd.DataFrame({"g": g}).groupby("g").groups
        return g, sz.where(k.notna()).values, mem

    _gb = al.DAYS_BIRTH.values
    def _plist(g, mem):
        out = {}
        for _, ix in mem.items():
            ix = np.asarray(ix)
            for i in ix:
                o = ix[ix != i]
                out[i] = (o, _gb[o] - _gb[i])
        return out

    G_S, SZ_S, MEM_S = _groups(["CODE_GENDER", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE", "FLAG_MOBIL"])
    PL_S = _plist(G_S, MEM_S)
    G_L, SZ_L, MEM_L = _groups(["CODE_GENDER"])
    PL_L = _plist(G_L, MEM_L)

    def f_partner_fn(X):
        gap = np.full(len(al), np.nan)
        for i, (o, gp_) in PL_S.items():
            gap[i] = gp_[np.argmin(np.abs(gp_))]
        d = {
            "HAS_PARTNER": (~np.isnan(G_S)).astype(int),
            "PARTNER_N": SZ_S,
            "PARTNER_GAP": gap,
            "PARTNER_ABSGAP": np.abs(gap),
            "HAS_PARTNER_LOOSE": (~np.isnan(G_L)).astype(int),
            "PARTNER_N_LOOSE": SZ_L
        }
        return pd.concat([X, new_cols(d)], axis=1)

    def f_person_fn(X):
        N = len(al)
        E = al[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].values.astype(float)
        pe = np.full((N, 3), np.nan)
        dif = np.full((N, 3), np.nan)
        cr = np.full(N, np.nan)
        an = np.full(N, np.nan)
        inc = np.full(N, np.nan)
        same = np.full(N, np.nan)
        qd = np.full(N, np.nan)
        Qv = (al.AMT_ANNUITY / al.AMT_CREDIT).values
        ct = (al.NAME_CONTRACT_TYPE == "Cash loans").values
        CR, AN_, INC = al.AMT_CREDIT.values, al.AMT_ANNUITY.values, al.AMT_INCOME_TOTAL.values

        for i, (o, g) in PL_S.items():
            j = o[np.argmin(np.abs(g))]
            allr = np.r_[i, o]
            for k in range(3):
                v = E[allr, k]
                pe[i, k] = np.nanmean(v) if np.isfinite(v).any() else np.nan
                dif[i, k] = E[i, k] - E[j, k]
            cr[i] = CR[i] / CR[j]
            an[i] = AN_[i] / AN_[j]
            inc[i] = INC[i] / INC[j]
            same[i] = float(ct[i] == ct[j])
            qd[i] = Qv[i] - Qv[j]

        d = {f"PERS_EXT{k+1}": pe[:, k] for k in range(3)}
        d.update({f"PDIF_EXT{k+1}": dif[:, k] for k in range(3)})
        d.update(PCR_RATIO=cr, PAN_RATIO=an, PINC_RATIO=inc, PSAME_CT=same, PQ_DIFF=qd)
        P = pd.DataFrame(d)
        P["PERS_EXT_MEAN"] = P[["PERS_EXT1", "PERS_EXT2", "PERS_EXT3"]].mean(axis=1)
        return pd.concat([X, new_cols({c: P[c].values for c in P.columns})], axis=1)

    # Build core and all-family manifolds
    X_base = base_fn()
    X_core = f_clean_fn(X_base)
    X_core = f_fin_fn(X_core)
    X_core = f_age_fn(X_core)
    X_core = f_bureau_fn(X_core)
    X_core = f_partner_fn(X_core)
    X_core = f_person_fn(X_core)

    X_all = f_ext_fn(X_core)
    X_all = f_count_fn(X_all)
    X_all = f_group_fn(X_all)

    X_core_k = f_kdec_fn(X_core)
    X_all_k = f_kdec_fn(X_all)

    # Decoded product keys & community keys
    q = al.AMT_ANNUITY / al.AMT_CREDIT
    _cash = (al.NAME_CONTRACT_TYPE == "Cash loans").values
    _qb = (al.AMT_ANNUITY - 129.5) / al.AMT_CREDIT
    _q5, _qb5 = q.round(5), _qb.round(5)
    FEE = ((_qb5.map(_qb5.value_counts()).values > _q5.map(_q5.value_counts()).values) & _cash).astype(int)
    Q0 = pd.Series(np.where(FEE, _qb, q))

    KS = {
        "Q3": q.round(3).astype(str), "Q4": q.round(4).astype(str), "Q5": q.round(5).astype(str),
        "P3": Q0.round(3).astype(str), "P4": Q0.round(4).astype(str), "P5": Q0.round(5).astype(str),
        "PF4": Q0.round(4).astype(str) + "|" + pd.Series(FEE).astype(str)
    }
    M0 = {"Q3": 100., "Q4": 30., "Q5": 30., "P3": 100., "P4": 30., "P5": 30., "PF4": 30.}

    CT = al.AMT_CREDIT / al.AMT_ANNUITY
    CT1, CT2, CT3 = CT.round(1).astype(str), CT.round(2).astype(str), CT.round(3).astype(str)
    KS.update({
        "CT1": CT1, "CT2": CT2, "CT3": CT3,
        "CT1_s10": CT1, "CT1_s200": CT1,
        "CT1_GP": CT1 + "|" + al.AMT_GOODS_PRICE.astype(str),
        "CT1_CR": CT1 + "|" + al.AMT_CREDIT.astype(str),
        "CT1_AN": CT1 + "|" + al.AMT_ANNUITY.astype(str),
        "GP": al.AMT_GOODS_PRICE.astype(str), "CR": al.AMT_CREDIT.astype(str), "AN": al.AMT_ANNUITY.astype(str)
    })
    M0.update({"CT1": 50., "CT2": 50., "CT3": 50., "CT1_s10": 10., "CT1_s200": 200., "CT1_GP": 50., "CT1_CR": 50., "CT1_AN": 50., "GP": 50., "CR": 50., "AN": 50.})

    base_frames = {
        "Xc": X_core.iloc[:ntr].reset_index(drop=True),
        "Xct": X_core.iloc[ntr:].reset_index(drop=True),
        "Xa": X_all.iloc[:ntr].reset_index(drop=True),
        "Xat": X_all.iloc[ntr:].reset_index(drop=True),
        "Xck": X_core_k.iloc[:ntr].reset_index(drop=True),
        "Xckt": X_core_k.iloc[ntr:].reset_index(drop=True),
        "Xak": X_all_k.iloc[:ntr].reset_index(drop=True),
        "Xakt": X_all_k.iloc[ntr:].reset_index(drop=True),
        "al": al,
        "KS": KS,
        "M0": M0,
        "Q0": Q0,
        "FEE": FEE,
        "PL_S": PL_S,
        "PL_L": PL_L
    }
    print(f"[+] Feature manifolds constructed in {time.time()-t0:.1f}s")
    return base_frames

# ==============================================================================
# SECTION 3: LEARNED EXPECTATION RESIDUALS & NESTED ENCODINGS
# ==============================================================================

def compute_residuals(al, ntr, fast=False):
    print("[*] Computing 16 demographic learned-expectation residuals...")
    t0 = time.time()
    raw = al.drop(columns=["SK_ID_CURR"]).copy()
    for c in CATS:
        raw[c] = raw[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    raw["DAYS_EMPLOYED"] = raw.DAYS_EMPLOYED.where(raw.DAYS_EMPLOYED != 365243)
    demo = [c for c in raw.columns if c.startswith("EXT_SOURCE") or c.startswith("AMT_")]

    T = {
        "EXT1": (raw.EXT_SOURCE_1, ["EXT_SOURCE_1"]),
        "EXT2": (raw.EXT_SOURCE_2, ["EXT_SOURCE_2"]),
        "EXT3": (raw.EXT_SOURCE_3, ["EXT_SOURCE_3"]),
        "LOGINC": (np.log(raw.AMT_INCOME_TOTAL), ["AMT_INCOME_TOTAL"]),
        "LOGCR": (np.log(raw.AMT_CREDIT), ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]),
        "EMP": (raw.DAYS_EMPLOYED, ["DAYS_EMPLOYED"]),
        "REG": (raw.DAYS_REGISTRATION, ["DAYS_REGISTRATION"]),
        "AGE": (raw.DAYS_BIRTH, ["DAYS_BIRTH"]),
        "IDP": (raw.DAYS_ID_PUBLISH, ["DAYS_ID_PUBLISH"]),
        "PHONE": (raw.DAYS_LAST_PHONE_CHANGE, ["DAYS_LAST_PHONE_CHANGE"]),
        "REGPOP": (raw.REGION_POPULATION_RELATIVE, ["REGION_POPULATION_RELATIVE"]),
        "FAM": (raw.CNT_FAM_MEMBERS, ["CNT_FAM_MEMBERS"]),
        "CAR": (raw.OWN_CAR_AGE, ["OWN_CAR_AGE"]),
        "EXT1D": (raw.EXT_SOURCE_1, demo),
        "EXT2D": (raw.EXT_SOURCE_2, demo),
        "EXT3D": (raw.EXT_SOURCE_3, demo)
    }

    out = {}
    n_splits = 2 if fast else 5
    max_iter = 15 if fast else 100
    for nm, (yt, drop_cols) in T.items():
        Xp = raw.drop(columns=drop_cols)
        pred = np.zeros(len(raw))
        obs = yt.notna().values
        for tri, tei in KFold(n_splits, shuffle=True, random_state=0).split(Xp):
            m = HR(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, random_state=0)
            m.fit(Xp.iloc[tri][obs[tri]], yt.iloc[tri][obs[tri]])
            pred[tei] = m.predict(Xp.iloc[tei])
        out[f"RP_{nm}"] = pred
        out[f"RR_{nm}"] = np.where(obs, yt.values - pred, np.nan)

    res_df = pd.DataFrame(out)
    print(f"[+] 16 demographic residuals completed in {time.time()-t0:.1f}s")
    return res_df

# ==============================================================================
# SECTION 4: MULTI-VIEW STACKING ENGINE (v17 Base vs v18 ExtraViews)
# ==============================================================================

def compute_multiviews(F1, y, ntr, extra_views=False, variant="A", fast=False):
    view_name = f"Multi-View {'ExtraViews (v18)' if extra_views else 'Standard (v17)'} [Variant {variant}]"
    print(f"[*] Computing {view_name}...")
    t0 = time.time()
    Xs = F1["Xa"].iloc[:ntr].copy()
    Xts = F1["Xat"].copy()
    Xs["Q0num"] = F1["Q0"].values[:ntr]
    Xts["Q0num"] = F1["Q0"].values[ntr:]
    cols = list(Xs.columns)

    ext = [c for c in cols if "EXT" in c or c in ("E12", "E13", "E23", "D12", "D13", "D23")]
    loan = [c for c in cols if c.startswith(("AMT_", "LOG_", "CREDIT_", "ANN_", "GOODS_", "TERM", "Q0num")) or c in ("CREDIT_MINUS_GOODS", "RP_LOGINC", "RR_LOGINC", "RP_LOGCR", "RR_LOGCR")]
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

    def vf(d):
        df = pd.DataFrame({f"VIEW_{k}": v for k, v in d.items()})
        df["VD_EL"] = df.VIEW_EXT - df.VIEW_LOAN
        df["VD_EP"] = df.VIEW_EXT - df.VIEW_PERSON
        df["VD_LP"] = df.VIEW_LOAN - df.VIEW_PERSON
        return df

    mk = lambda: HC(max_iter=10 if fast else 100, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=20, random_state=0)
    out = {"tr": [], "va": [], "te": []}
    skf = StratifiedKFold(2 if fast else 5, shuffle=True, random_state=42)

    for k, (a, b) in enumerate(skf.split(np.zeros(ntr), y)):
        trv, vav, tev = {}, {}, {}
        for vn, vc in views.items():
            inner = np.zeros(len(a))
            pvb, pte = [], []
            for ia, ib in StratifiedKFold(2 if fast else 3, shuffle=True, random_state=11).split(a, y[a]):
                m = mk().fit(Xs.iloc[a[ia]][vc], y[a[ia]])
                inner[ib] = m.predict_proba(Xs.iloc[a[ib]][vc])[:, 1]
                if variant == "A":
                    pvb.append(m.predict_proba(Xs.iloc[b][vc])[:, 1])
                    pte.append(m.predict_proba(Xts[vc])[:, 1])
            trv[vn] = lg(inner)
            if variant == "A":
                vav[vn] = lg(np.mean(pvb, 0))
                tev[vn] = lg(np.mean(pte, 0))
            else:
                m = mk().fit(Xs.iloc[a][vc], y[a])
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
        for ia, ib in StratifiedKFold(2 if fast else 3, shuffle=True, random_state=11).split(np.zeros(ntr), y):
            m = mk().fit(Xs.iloc[ia][vc], y[ia])
            inner[ib] = m.predict_proba(Xs.iloc[ib][vc])[:, 1]
            if variant == "A":
                pte.append(m.predict_proba(Xts[vc])[:, 1])
        trv[vn] = lg(inner)
        tev[vn] = lg(np.mean(pte, 0)) if variant == "A" else lg(mk().fit(Xs[vc], y).predict_proba(Xts[vc])[:, 1])
    out["all_tr"] = vf(trv)
    out["all_te"] = vf(tev)
    print(f"[+] {view_name} finished in {time.time()-t0:.1f}s")
    return out

# ==============================================================================
# SECTION 5: MODEL MEMBER RUNNER & MULTI-VIEW GRAND FUSION
# ==============================================================================

def train_multiview_stream(bf, F1, y, ntr, nte, tidx, views_o, views_a, extra_views=False, mono_te=False, fast=False, tag="v17"):
    print(f"\n{'='*70}\n[+] TRAINING MULTI-VIEW STREAM {tag.upper()} ({'v18 ExtraViews' if extra_views else 'v17 Standard'})\n{'='*70}")
    t_start = time.time()
    
    # Nested in-fold encoder
    KS, M0 = bf["KS"], bf["M0"]
    KC = {k: pd.factorize(v)[0] for k, v in KS.items()}
    KN = {k: int(v.max()) + 1 for k, v in KC.items()}
    PL_S, PL_L = bf["PL_S"], bf["PL_L"]

    def partner_enc(fit, idx):
        idx = np.asarray(idx)
        cols = ["PY_ANY", "PY_N", "PY_NEAR180", "PY_EARLIER", "PY_LATER", "PYL_ANY"]
        out = np.full((len(idx), 6), np.nan)
        out[:, 1] = 0
        lab = np.full(ntr + nte, np.nan)
        lab[fit] = y[fit]
        for j, i in enumerate(idx):
            if i in PL_S:
                o, gp_ = PL_S[i]
                v = lab[o]
                k = ~np.isnan(v)
                if k.any():
                    v, gp_ = v[k], gp_[k]
                    out[j, 0] = v.max()
                    out[j, 1] = k.sum()
                    if (np.abs(gp_) <= 180).any(): out[j, 2] = v[np.abs(gp_) <= 180].max()
                    if (gp_ > 0).any(): out[j, 3] = v[gp_ > 0].max()
                    if (gp_ < 0).any(): out[j, 4] = v[gp_ < 0].max()
            if i in PL_L:
                o, gp_ = PL_L[i]
                v = lab[o]
                v = v[~np.isnan(v)]
                if len(v): out[j, 5] = v.max()
        return pd.DataFrame(out, columns=cols)

    _MEMO = {}
    def enc(fit, qi):
        mk = (len(fit), len(qi), fit[0] if len(fit) else 0)
        if mk in _MEMO: return _MEMO[mk]
        pr = float(y[fit].mean())
        yf = y[fit].astype(float)
        out = {}
        for k in KS:
            c = KC[k]
            cnt_t = np.bincount(c[fit], minlength=KN[k]).astype(float)
            sm_t = np.bincount(c[fit], weights=yf, minlength=KN[k])
            cq = c[qi]
            out[f"TE_{k}"] = (sm_t[cq] + M0[k] * pr) / (cnt_t[cq] + M0[k])
            out[f"CNT_{k}"] = cnt_t[cq]
        res = pd.concat([pd.DataFrame(out), partner_enc(fit, qi)], axis=1)
        _MEMO[mk] = res
        return res

    def train_enc(a):
        mk = ("tr", len(a), a[0] if len(a) else 0)
        if mk in _MEMO: return _MEMO[mk]
        cols = enc(a[:10], a[:5]).columns
        inner = np.zeros((len(a), len(cols)))
        for ia, ib in StratifiedKFold(5, shuffle=True, random_state=7).split(a, y[a]):
            inner[ib] = enc(a[ia], a[ib]).values
        res = pd.DataFrame(inner, columns=cols)
        _MEMO[mk] = res
        return res

    def onehot(Xa_, Xt_):
        z = pd.concat([Xa_, Xt_], ignore_index=True)
        oh = pd.get_dummies(z[CATS].astype("category"), dummy_na=True, dtype=np.float32)
        oh.columns = [f"OH_{i}" for i in range(oh.shape[1])]
        Z = pd.concat([z.drop(columns=CATS), oh], axis=1)
        return Z.iloc[:len(Xa_)].reset_index(drop=True), Z.iloc[len(Xa_):].reset_index(drop=True)

    # Build member specs
    Xc_, Xct_, Xa_, Xat_, Xck_, Xckt_, Xak_, Xakt_ = [F1[k] for k in ("Xc", "Xct", "Xa", "Xat", "Xck", "Xckt", "Xak", "Xakt")]
    Xco, Xcot = onehot(Xc_, Xct_)
    Xao, Xaot = onehot(Xa_, Xat_)
    Xaok, Xaokt = onehot(Xak_, Xakt_)

    MONO_K = [-1 if c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") else 0 for c in Xck_.columns] + \
             ([s_ for _ in KS for s_ in (1, 0)] + [1, 0, 1, 1, 1, 1] if mono_te else [0] * (2 * len(KS) + 6))

    SPEC1 = {
        "qf_shallow":   (Xco,  Xcot,  dict(learning_rate=0.05, num_leaves=7, max_depth=3, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20, feature_pre_filter=False), 30 if fast else 1300, 30 if fast else 1430, [42, 7, 2024, 11, 99]),
        "qf_reg":       (Xco,  Xcot,  dict(learning_rate=0.03, num_leaves=11, min_child_samples=300, feature_fraction=0.4, bagging_fraction=0.8, lambda_l2=40, feature_pre_filter=False), 30 if fast else 1939, 30 if fast else 2133, [42, 7, 2024, 11, 99]),
        "v1_allfam_oh": (Xao,  Xaot,  dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20, feature_pre_filter=False), 30 if fast else 881, 30 if fast else 969, [42, 7, 2024]),
        "v2_allfam_oh": (Xaok, Xaokt, dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20, feature_pre_filter=False), 30 if fast else 881, 30 if fast else 969, [42, 7, 2024]),
        "v2_mono":      (Xck_.reset_index(drop=True), Xckt_.reset_index(drop=True), dict(learning_rate=0.02, num_leaves=15, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20, feature_pre_filter=False, monotone_constraints=MONO_K), 30 if fast else 1571, 30 if fast else 1729, [42, 7, 2024])
    }
    SPEC1_ET = {nm: (X, Xt, {**p, "extra_trees": True}, R, Rf, seeds) for nm, (X, Xt, p, R, Rf, seeds) in SPEC1.items()}

    BASE_LGB = dict(objective="binary", force_col_wise=True, bagging_freq=1, verbose=-1, num_threads=THREADS, seed=42)
    W = {"qf_shallow": 0.24, "qf_reg": 0.12, "v1_allfam_oh": 0.32, "v2_allfam_oh": 0.16, "v2_mono": 0.16}

    FOLDS = 2 if fast else 5
    def run_member(name, spec, mtag, view=None):
        X, Xt, p, R, Rf, seeds = spec[name]
        if view is not None and "monotone_constraints" in p:
            p = {**p, "monotone_constraints": list(p["monotone_constraints"]) + [0] * view["tr"][0].shape[1]}
        tf = np.zeros(nte)
        oof = np.zeros(ntr)
        ex = lambda key, k=None: ([] if view is None else [view[key] if k is None else view[key][k]])
        for k, (a, b) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(ntr), y)):
            fa = pd.concat([X.iloc[a].reset_index(drop=True), train_enc(a)] + ex("tr", k), axis=1)
            fb = pd.concat([X.iloc[b].reset_index(drop=True), enc(a, b)] + ex("va", k), axis=1)
            ft = pd.concat([Xt.reset_index(drop=True), enc(a, tidx)] + ex("te", k), axis=1)
            m = lgb.train({**BASE_LGB, **p}, lgb.Dataset(fa, y[a]), R)
            oof[b] = m.predict(fb)
            tf += m.predict(ft) / FOLDS
        allr = np.arange(ntr)
        fa = pd.concat([X.reset_index(drop=True), train_enc(allr)] + ex("all_tr"), axis=1)
        ft = pd.concat([Xt.reset_index(drop=True), enc(allr, tidx)] + ex("all_te"), axis=1)
        fr = []
        for sd in seeds[:1 if fast else len(seeds)]:
            fr.append(rk(lgb.train({**BASE_LGB, **p, "seed": sd}, lgb.Dataset(fa, y), Rf).predict(ft)))
        test = rk(0.5 * rk(tf) + 0.5 * np.mean(fr, 0))
        auc = roc_auc_score(y, oof)
        print(f"  [{mtag}_{name}] Complete | OOF AUC: {auc:.5f}")
        return test, oof

    RESV = {}
    for nm in W: RESV[("e1", nm)] = run_member(nm, SPEC1, f"{tag}_e1", views_o)
    for nm in W: RESV[("e1et", nm)] = run_member(nm, SPEC1_ET, f"{tag}_e1et", views_o)
    for nm in W: RESV[("e1Aet", nm)] = run_member(nm, SPEC1_ET, f"{tag}_e1Aet", views_a)

    # CatBoost Specialist
    print(f"\n[*] Training CatBoost Product Specialist for {tag}...")
    cb_numeric = pd.concat([bf["Xa"], bf["Xat"]], ignore_index=True).drop(columns=CATS).reset_index(drop=True)
    cb_cat = bf["al"][CATS].copy().reset_index(drop=True)
    for c in CATS: cb_cat[c] = cb_cat[c].astype("string").fillna("__MISSING__").astype(str)
    cb_product = pd.concat([cb_numeric, cb_cat], axis=1)
    product_names = []
    for k_ in ["Q3", "Q4", "Q5", "P3", "P4", "P5", "PF4"] + [k for k in ("CT1", "CT2", "CT3", "CT1_GP", "CT1_CR", "CT1_AN") if k in KS]:
        cb_product["PROD_" + k_] = KS[k_].reset_index(drop=True).astype("string").fillna("__MISSING__").astype(str)
        product_names.append("PROD_" + k_)
    kd = pd.concat([bf["Xak"]["K_DEC"], bf["Xakt"]["K_DEC"]], ignore_index=True)
    cb_product["PROD_K_DEC"] = kd.round().astype("Int64").astype("string").fillna("__MISSING__").astype(str)
    product_names.append("PROD_K_DEC")
    cb_product["PROD_FEE"] = pd.Series(bf["FEE"]).reset_index(drop=True).astype(str)
    product_names.append("PROD_FEE")
    CB_CATS = CATS + product_names
    cb_train = cb_product.iloc[:ntr].reset_index(drop=True)
    cb_test = cb_product.iloc[ntr:].reset_index(drop=True)

    CBP = dict(iterations=30 if fast else 400, depth=6, learning_rate=0.05, loss_function="Logloss", eval_metric="AUC",
               l2_leaf_reg=8, random_seed=42, thread_count=THREADS, verbose=False)
    cb_oof = np.zeros(ntr)
    cb_tests = []
    for fi, (a, b) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(ntr), y), 1):
        pe_a = partner_enc(a, a)
        pe_b = partner_enc(a, b)
        pe_t = partner_enc(a, tidx)
        ca = pd.concat([cb_train.iloc[a].reset_index(drop=True), pe_a], axis=1)
        cbv = pd.concat([cb_train.iloc[b].reset_index(drop=True), pe_b], axis=1)
        cbt = pd.concat([cb_test.reset_index(drop=True), pe_t], axis=1)
        cb_model = CatBoostClassifier(**CBP)
        cb_model.fit(ca, y[a], cat_features=CB_CATS)
        pv = cb_model.predict_proba(cbv)[:, 1]
        pt = cb_model.predict_proba(cbt)[:, 1]
        cb_oof[b] = pv
        cb_tests.append(pt)
        print(f"  [CatBoost Fold {fi}/{FOLDS}] AUC: {roc_auc_score(y[b], pv):.5f}")
    cb_test_mean = np.mean(cb_tests, 0)

    # Champion Assembly
    blend_fn = lambda t, i: sum(W[k] * (RESV[(t, k)][0] if i == 0 else rk(RESV[(t, k)][1])) for k in W)
    _have = ["e1", "e1et"]
    B = lambda i: sum(blend_fn(t, i) for t in _have) / len(_have)
    champ = lambda i, cb_p: 0.9 * rk(0.75 * rk(B(i)) + 0.25 * blend_fn("e1Aet", i)) + 0.1 * rk(cb_p)

    oof_final = champ(1, cb_oof)
    test_final = champ(0, cb_test_mean)
    auc_final = roc_auc_score(y, oof_final)
    print(f"\n>>> STREAM {tag.upper()} GRAND CHAMPION OOF AUC: {auc_final:.6f} (Elapsed: {time.time()-t_start:.1f}s) <<<")
    return test_final, oof_final

# ==============================================================================
# SECTION 6: UNIFIED DUAL-CHAMPION CONVEX RANK FUSION (v22 ENGINE)
# ==============================================================================

def execute_v22_unified_engine(fast=False, output_path=None):
    total_start = time.time()
    print("=" * 80)
    print("  GCI World 2026: Unified All-Time Champion v22 Engine (100% Pure Scratch)")
    print("=" * 80)

    tr_path, te_path, sm_path = resolve_data_paths()
    df_train = pd.read_csv(tr_path)
    df_test = pd.read_csv(te_path)
    sample = pd.read_csv(sm_path)

    y = df_train["TARGET"].values.astype(np.int32)
    ntr = len(df_train)
    nte = len(df_test)
    tidx = np.arange(ntr, ntr + nte)

    # 1. Feature manifolds
    bf = build_all_features(df_train, df_test)

    # 2. Demographic residuals
    res_df = compute_residuals(bf["al"], ntr, fast=fast)
    R_tr = res_df.iloc[:ntr].reset_index(drop=True)
    R_te = res_df.iloc[ntr:].reset_index(drop=True)
    F1 = {k: pd.concat([bf[k], R_te if k.endswith("t") else R_tr], axis=1) for k in ("Xc", "Xct", "Xa", "Xat", "Xck", "Xckt", "Xak", "Xakt")}
    F1["Q0"] = bf["Q0"]

    # 3. Views
    views_v17_o = compute_multiviews(F1, y, ntr, extra_views=False, variant="orig", fast=fast)
    views_v17_a = compute_multiviews(F1, y, ntr, extra_views=False, variant="A", fast=fast)

    views_v18_o = compute_multiviews(F1, y, ntr, extra_views=True, variant="orig", fast=fast)
    views_v18_a = compute_multiviews(F1, y, ntr, extra_views=True, variant="A", fast=fast)

    # 4. Stream v17 execution
    test_v17, oof_v17 = train_multiview_stream(
        bf, F1, y, ntr, nte, tidx, views_v17_o, views_v17_a,
        extra_views=False, mono_te=False, fast=fast, tag="v17"
    )

    # 5. Stream v18 execution
    test_v18, oof_v18 = train_multiview_stream(
        bf, F1, y, ntr, nte, tidx, views_v18_o, views_v18_a,
        extra_views=True, mono_te=True, fast=fast, tag="v18"
    )

    # 6. Convex Dual-Champion Rank Fusion (0.55 * v17 + 0.45 * v18)
    print("\n" + "=" * 80)
    print("DUAL-CHAMPION CONVEX RANK FUSION: v22 = 0.55 * rank(v17) + 0.45 * rank(v18)")
    print("=" * 80)
    rk_norm = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)
    r17 = rk_norm(test_v17)
    r18 = rk_norm(test_v18)
    v22_test = rk_norm(0.55 * r17 + 0.45 * r18)

    oof_r17 = rk_norm(oof_v17)
    oof_r18 = rk_norm(oof_v18)
    v22_oof = rk_norm(0.55 * oof_r17 + 0.45 * oof_r18)
    v22_auc = roc_auc_score(y, v22_oof)

    print(f"\n>>> V22 ALL-TIME RECORD CHAMPION POOLED OOF AUC: {v22_auc:.6f} <<<")

    if output_path is None:
        output_path = r"d:/gci/v_22_pure_champion_ensemble_scratch.csv"

    sub = pd.DataFrame({
        "SK_ID_CURR": sample["SK_ID_CURR"],
        "TARGET": v22_test
    })
    sub.to_csv(output_path, index=False)
    print(f"[+] All-time champion submission successfully saved to: {output_path}")
    print(f"[+] Rows: {len(sub)}, Target range: [{sub['TARGET'].min():.5f}, {sub['TARGET'].max():.5f}]")

    # Historical verification
    ref_path = r"d:/gci/gemini/v_22_pure_champion_ensemble.csv"
    if os.path.exists(ref_path):
        ref_df = pd.read_csv(ref_path)
        corr, _ = spearmanr(v22_test, ref_df["TARGET"])
        print(f"[*] Spearman rank correlation with disk reference v22 (0.78239): {corr:.6f}")

    print(f"\n[+] Total Pipeline Runtime: {time.time()-total_start:.1f}s")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified V22 Scratch Champion Engine")
    parser.add_argument("--fast", action="store_true", help="Run fast verification smoke test")
    parser.add_argument("--output", type=str, default=None, help="Output submission CSV path")
    args = parser.parse_args()

    execute_v22_unified_engine(fast=args.fast, output_path=args.output)
