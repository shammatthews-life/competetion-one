import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

t0 = time.time()
df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y = df_train["TARGET"].values.astype(np.float64)
ntr = len(df_train)

CATS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]

def build_clean_x1(df):
    X = df.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore").copy()
    for c in CATS:
        X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    de = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != 365243)
    X["DE_SENT"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    X["DAYS_EMPLOYED_CLEAN"] = de
    X["ORG_XNA"] = (df["ORGANIZATION_TYPE"] == "XNA").astype(int)
    X["CAR_AGE_OUT"] = (df["OWN_CAR_AGE"] >= 60).astype(int)
    X["MISS_CNT"] = df.drop(columns=["TARGET"], errors="ignore").isna().sum(axis=1)
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
        X[c + "_NA"] = df[c].isna().astype(int)

    e1, e2, e3 = df["EXT_SOURCE_1"], df["EXT_SOURCE_2"], df["EXT_SOURCE_3"]
    e = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    X["EXT_MEAN"] = e.mean(axis=1); X["EXT_MIN"] = e.min(axis=1); X["EXT_MAX"] = e.max(axis=1)
    X["EXT_STD"] = e.std(axis=1); X["EXT_RANGE"] = X["EXT_MAX"] - X["EXT_MIN"]
    X["EXT_NOBS"] = e.notna().sum(axis=1); X["EXT_PROD"] = e.prod(axis=1)
    X["EXT_W"] = (e1 * 2 + e2 * 3 + e3 * 4) / 9.0
    X["E12"] = e1 * e2; X["E13"] = e1 * e3; X["E23"] = e2 * e3
    X["D12"] = e1 - e2; X["D13"] = e1 - e3; X["D23"] = e2 - e3

    inc, cr, an, gp = df["AMT_INCOME_TOTAL"], df["AMT_CREDIT"], df["AMT_ANNUITY"], df["AMT_GOODS_PRICE"]
    X["CREDIT_INC"] = cr / inc; X["ANN_INC"] = an / inc; X["ANN_CREDIT"] = an / cr
    X["TERM"] = cr / an; X["GOODS_CREDIT"] = gp / cr; X["CREDIT_MINUS_GOODS"] = cr - gp
    X["GOODS_INC"] = gp / inc; X["INC_PER_FAM"] = inc / df["CNT_FAM_MEMBERS"]
    X["CREDIT_PER_FAM"] = cr / df["CNT_FAM_MEMBERS"]; X["ANN_PER_FAM"] = an / df["CNT_FAM_MEMBERS"]
    for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        X["LOG_" + c] = np.log1p(df[c])

    age = -df["DAYS_BIRTH"]
    X["AGE"] = age / 365.25; X["EMP_AGE"] = de / df["DAYS_BIRTH"]; X["EMP_YEARS"] = -de / 365.25
    X["REG_AGE"] = df["DAYS_REGISTRATION"] / df["DAYS_BIRTH"]; X["ID_AGE"] = df["DAYS_ID_PUBLISH"] / df["DAYS_BIRTH"]
    X["PHONE_AGE"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
    X["CAR_AGE_RATIO"] = df["OWN_CAR_AGE"] / (age / 365.25)
    X["CREDIT_PER_AGE"] = cr / (age / 365.25); X["INC_PER_AGE"] = inc / (age / 365.25)

    b = [c for c in df.columns if c.startswith("AMT_REQ")]
    X["BUR_SUM"] = df[b].sum(axis=1, min_count=1)
    X["BUR_RECENT"] = df[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON"]].sum(axis=1, min_count=1)
    X["BUR_MON_YEAR"] = df["AMT_REQ_CREDIT_BUREAU_MON"] / (df["AMT_REQ_CREDIT_BUREAU_YEAR"] + 1.0)

    cash = df["NAME_CONTRACT_TYPE"] == "Cash loans"
    kf = (cr / gp - 1.0) / 0.0132
    kr = kf.round()
    ok = cash & (cr > gp) & ((kf - kr).abs() < 0.1)
    X["K_DEC"] = np.where(ok, kr, np.where(cash & (cr == gp), 0.0, np.nan))
    X["CG_RATIO"] = cr / gp

    # V27 verified upgrades
    X["FINANCED_MARKUP_RATIO"] = cr / (gp + 1e-4)
    X["ANNUITY_TO_GOODS"] = an / (gp + 1e-4)
    X["CREDIT_TERM_APPROX"] = cr / (an * 12.0 + 1e-4)
    X["APR_36M"] = (36.0 * an - cr) / (cr + 1e-4)
    X["NEG_DOWNPAYMENT"] = np.clip((cr - gp) / (gp + 1e-4), 0.0, None)
    X["INQ_ACCEL"] = df["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0.0) / (df["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0.0) / 4.0 + 0.25)
    X["CAR_AGE_TO_CAREER"] = np.where(df["OWN_CAR_AGE"].fillna(-1.0) >= 0, df["OWN_CAR_AGE"] / (X["EMP_YEARS"].fillna(0.0) + 1.0), -1.0)
    X["EXT_WEIGHTED_SUM_IMPUTED"] = (e1.fillna(0.5)*2.0 + e2.fillna(0.5)*3.0 + e3.fillna(0.5)*4.0) / 9.0
    X["EXT_MISSING_COUNT"] = e1.isna().astype(int) + e2.isna().astype(int) + e3.isna().astype(int)
    return X

X_base = build_clean_x1(df_train)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(df_train, y))

cr = df_train["AMT_CREDIT"]; an = df_train["AMT_ANNUITY"]; gp = df_train["AMT_GOODS_PRICE"]
cash = df_train["NAME_CONTRACT_TYPE"] == "Cash loans"
q_r = an / cr
q0_1285 = pd.Series(np.where(cash, (an - 128.5) / cr, q_r), index=df_train.index)
q0_1295 = pd.Series(np.where(cash, (an - 129.5) / cr, q_r), index=df_train.index)
ct = cr / an
kf066 = ((cr / gp - 1.0) / 0.0066).round()

# Pre-factorize each string key across train for fast integer indexing ONLY inside fold a -> b checks
# Strictly leak-free: counts & sums ONLY ever read train fold indices `a` (or inner `ia`), never validation/test targets!
all_keys_raw = {
    "Q3": (q_r.round(3).astype(str), 100.0),
    "Q4": (q_r.round(4).astype(str), 30.0),
    "P3": (q0_1295.round(3).astype(str), 100.0),
    "P4": (q0_1295.round(4).astype(str), 30.0),
    "CT1": (ct.round(1).astype(str), 50.0),
    "CT2": (ct.round(2).astype(str), 50.0),
    "K066": (pd.Series(kf066).astype(str), 30.0),
    "GENDER_EDU": (df_train["CODE_GENDER"].astype(str) + "|" + df_train["NAME_EDUCATION_TYPE"].astype(str), 30.0),
    "ORG": (df_train["ORGANIZATION_TYPE"].astype(str), 30.0),
    "OCC": (df_train["OCCUPATION_TYPE"].fillna("NA").astype(str), 30.0),
    "ORG_OCC": (df_train["ORGANIZATION_TYPE"].astype(str) + "|" + df_train["OCCUPATION_TYPE"].fillna("NA").astype(str), 30.0),
    "CT1_GP": (ct.round(1).astype(str) + "|" + gp.astype(str), 50.0),
    "CT1_CR": (ct.round(1).astype(str) + "|" + cr.astype(str), 50.0),
    "CT1_AN": (ct.round(1).astype(str) + "|" + an.astype(str), 50.0),
    "Q5": (q_r.round(5).astype(str), 30.0),
    "P5": (q0_1295.round(5).astype(str), 30.0),
    "P4_1285": (q0_1285.round(4).astype(str), 30.0),
    "P5_1285": (q0_1285.round(5).astype(str), 30.0),
    "CT3": (ct.round(3).astype(str), 50.0),
    "GP": (gp.astype(str), 50.0),
    "CR": (cr.astype(str), 50.0),
    "AN": (an.astype(str), 50.0),
    "CT1_s10": (ct.round(1).astype(str), 10.0),
    "CT1_s200": (ct.round(1).astype(str), 200.0),
    "K066_P4": (pd.Series(kf066).astype(str) + "|" + q0_1285.round(4).astype(str), 20.0),
    "REG_HOUS": (df_train["REGION_RATING_CLIENT"].astype(str) + "|" + df_train["NAME_HOUSING_TYPE"].astype(str), 30.0),
    "INC_EDU": (df_train["NAME_INCOME_TYPE"].astype(str) + "|" + df_train["NAME_EDUCATION_TYPE"].astype(str), 30.0),
}
all_codes = {k: (pd.factorize(v[0])[0], v[1]) for k, v in all_keys_raw.items()}
all_n = {k: int(c[0].max()) + 2 for k, c in all_codes.items()}

# Precompute exact fold TE tables once per key (~0.5s total!)
te_cache_tr = {fi: {} for fi in range(5)}
te_cache_va = {fi: {} for fi in range(5)}
for fi, (a, b) in enumerate(folds):
    pr_a = float(y[a].mean())
    in_splits = list(StratifiedKFold(5, shuffle=True, random_state=42+fi).split(a, y[a]))
    for kname, (codes, m0) in all_codes.items():
        n_cat = all_n[kname]
        ca = np.where(codes[a] >= 0, codes[a], n_cat - 1)
        cb = np.where(codes[b] >= 0, codes[b], n_cat - 1)
        enc_a = np.full(len(a), pr_a); cnt_a = np.zeros(len(a))
        for ia, ib in in_splits:
            cia = ca[ia]; cib = ca[ib]
            cnt = np.bincount(cia, minlength=n_cat).astype(float)
            sm = np.bincount(cia, weights=y[a[ia]], minlength=n_cat)
            mp = (sm + m0 * pr_a) / (cnt + m0)
            enc_a[ib] = mp[cib]
            cnt_a[ib] = cnt[cib]
        cnt_f = np.bincount(ca, minlength=n_cat).astype(float)
        sm_f = np.bincount(ca, weights=y[a], minlength=n_cat)
        mp_f = (sm_f + m0 * pr_a) / (cnt_f + m0)
        te_cache_tr[fi][kname] = (enc_a, cnt_a)
        te_cache_va[fi][kname] = (mp_f[cb], cnt_f[cb])

print(f"[+] Precomputed all 27 strict nested in-fold TE matrices in {time.time()-t0:.1f}s", flush=True)

age_at_id = (df_train["DAYS_ID_PUBLISH"] - df_train["DAYS_BIRTH"]) / 365.25
id_dist = np.minimum(np.abs(age_at_id - 20.0), np.abs(age_at_id - 45.0))
extra_row_bank = {
    "K066_AND_Q0_1285": {"K_0066": np.where(cash & (cr >= gp) & (((cr/gp-1)/0.0066 - kf066).abs() < 0.05), kf066, -1.0),
                         "Q0_1285": q0_1285.values},
    "ID_MILESTONE": {"AGE_AT_ID": age_at_id.values, "ID_DIST_20_45": id_dist.values},
    "HOUSEHOLD_EXACT": {"SINGLE_PARENT": ((df_train["CNT_CHILDREN"] > 0) & ((df_train["CNT_FAM_MEMBERS"] - df_train["CNT_CHILDREN"]) <= 1.0)).astype(int).values,
                        "INC_PER_ADULT": (df_train["AMT_INCOME_TOTAL"] / np.maximum(df_train["CNT_FAM_MEMBERS"] - df_train["CNT_CHILDREN"], 1.0)).values},
    "PH_ZERO": {"PH_ZERO": (df_train["DAYS_LAST_PHONE_CHANGE"] == 0).astype(int).values},
    "LOGIT_EXTS": {"LOGIT_EXT2": np.log(np.clip(df_train["EXT_SOURCE_2"],1e-4,1-1e-4)/(1-np.clip(df_train["EXT_SOURCE_2"],1e-4,1-1e-4))),
                   "LOGIT_EXT3": np.log(np.clip(df_train["EXT_SOURCE_3"],1e-4,1-1e-4)/(1-np.clip(df_train["EXT_SOURCE_3"],1e-4,1-1e-4)))}
}

def eval_config(te_keys_list, row_dict=None, label=""):
    X = X_base.copy()
    if row_dict:
        for k, v in row_dict.items(): X[k] = v
    oof = np.zeros(ntr)
    fold_aucs = []
    params = dict(objective="binary", learning_rate=0.04, num_leaves=15, min_child_samples=150,
                  feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, verbose=-1, n_jobs=8, seed=42)
    for fi, (a, b) in enumerate(folds):
        Xa = X.iloc[a].copy()
        Xb = X.iloc[b].copy()
        for kn in te_keys_list:
            ea, ca = te_cache_tr[fi][kn]
            eb, cb = te_cache_va[fi][kn]
            Xa[f"TE_{kn}"] = ea; Xa[f"CNT_{kn}"] = ca
            Xb[f"TE_{kn}"] = eb; Xb[f"CNT_{kn}"] = cb
        dtr = lgb.Dataset(Xa, label=y[a], free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=500)
        pv = m.predict(Xb)
        oof[b] = pv
        fold_aucs.append(roc_auc_score(y[b], pv))
    total = roc_auc_score(y, oof)
    print(f"{label:32s} | OOF={total:.6f} | folds={[round(x,5) for x in fold_aucs]}", flush=True)
    return total, fold_aucs

base_list = ["Q3","Q4","P3","P4","CT1","CT2","K066","GENDER_EDU","ORG","OCC","ORG_OCC","CT1_GP","CT1_CR","CT1_AN"]
b_tot, b_f = eval_config(base_list, None, "BASE (Clean V23+V27+14 TE)")

v22_list = base_list + ["Q5","P5","CT3","GP","CR","AN","CT1_s10","CT1_s200"]
t1, f1 = eval_config(v22_list, None, "+V22 Full Tariff TE Keys")
print(f"  Delta vs Base: {(t1-b_tot)*10000:+.2f} bps ({sum(x>y_ for x,y_ in zip(f1,b_f))}/5 wins)", flush=True)

l1285 = v22_list + ["P4_1285","P5_1285","K066_P4"]
t2, f2 = eval_config(l1285, None, "+Fee128.5 & K066_P4 TE Keys")
print(f"  Delta vs Prev: {(t2-t1)*10000:+.2f} bps ({sum(x>y_ for x,y_ in zip(f2,f1))}/5 wins)", flush=True)

ldemo = (l1285 if t2 > t1 else v22_list) + ["REG_HOUS","INC_EDU"]
ref_t, ref_f = (t2, f2) if t2 > t1 else (t1, f1)
t3, f3 = eval_config(ldemo, None, "+REG_HOUS & INC_EDU TE")
print(f"  Delta vs Prev: {(t3-ref_t)*10000:+.2f} bps ({sum(x>y_ for x,y_ in zip(f3,ref_f))}/5 wins)", flush=True)

best_te = ldemo if t3 > ref_t else (l1285 if t2 > t1 else v22_list)
best_ref, best_f = (t3, f3) if t3 > ref_t else (ref_t, ref_f)

print("\n--- Screening Individual Row Feature Blocks ---", flush=True)
for rname, rcols in extra_row_bank.items():
    rtot, rf = eval_config(best_te, rcols, f"+ROW: {rname}")
    print(f"  Delta: {(rtot-best_ref)*10000:+.2f} bps ({sum(x>y_ for x,y_ in zip(rf,best_f))}/5 wins)", flush=True)

