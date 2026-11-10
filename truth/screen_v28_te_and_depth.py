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

age_dec = ((-df_train["DAYS_BIRTH"] / 365.25) // 10).astype(int).astype(str)
occ = df_train["OCCUPATION_TYPE"].fillna("NA").astype(str)
org = df_train["ORGANIZATION_TYPE"].astype(str)
gen = df_train["CODE_GENDER"].astype(str)
edu = df_train["NAME_EDUCATION_TYPE"].astype(str)
inc_t = df_train["NAME_INCOME_TYPE"].astype(str)
hous = df_train["NAME_HOUSING_TYPE"].astype(str)
fam_s = df_train["NAME_FAMILY_STATUS"].astype(str)
con_t = df_train["NAME_CONTRACT_TYPE"].astype(str)
reg = df_train["REGION_RATING_CLIENT"].astype(str)
ext_nobs = df_train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notna().sum(axis=1).astype(str)
de_sent = (df_train["DAYS_EMPLOYED"] == 365243).astype(int).astype(str)
has_car = df_train["OWN_CAR_AGE"].notna().astype(int).astype(str)

# Established best 27 keys from previous run:
best27 = {
    "Q3": (q_r.round(3).astype(str), 100.0),
    "Q4": (q_r.round(4).astype(str), 30.0),
    "P3": (q0_1295.round(3).astype(str), 100.0),
    "P4": (q0_1295.round(4).astype(str), 30.0),
    "CT1": (ct.round(1).astype(str), 50.0),
    "CT2": (ct.round(2).astype(str), 50.0),
    "K066": (pd.Series(kf066).astype(str), 30.0),
    "GENDER_EDU": (gen + "|" + edu, 30.0),
    "ORG": (org, 30.0),
    "OCC": (occ, 30.0),
    "ORG_OCC": (org + "|" + occ, 30.0),
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
    "REG_HOUS": (reg + "|" + hous, 30.0),
    "INC_EDU": (inc_t + "|" + edu, 30.0),
}

new_candidates = {
    "GROUP_A_SOCIO": {
        "GENDER_OCC": (gen + "|" + occ, 30.0),
        "CONTRACT_EDU": (con_t + "|" + edu, 30.0),
        "HOUS_FAM": (hous + "|" + fam_s, 30.0),
        "REG_EDU": (reg + "|" + edu, 30.0),
    },
    "GROUP_B_LIFECYCLE": {
        "AGE_GEN_EDU": (age_dec + "|" + gen + "|" + edu, 30.0),
        "REG_OCC": (reg + "|" + occ, 30.0),
        "CAR_HOUS_FAM": (has_car + "|" + hous + "|" + fam_s, 30.0),
    },
    "GROUP_C_BUREAU_CONTACT": {
        "EXT_CON_REG": (ext_nobs + "|" + con_t + "|" + reg, 30.0),
        "PHONE_OCC": (df_train["FLAG_EMP_PHONE"].astype(str) + "|" + df_train["FLAG_WORK_PHONE"].astype(str) + "|" + occ, 30.0),
        "DE_INC_FAM": (de_sent + "|" + inc_t + "|" + fam_s, 30.0),
    }
}

all_bank = {**best27}
for g in new_candidates.values():
    all_bank.update(g)

all_codes = {k: (pd.factorize(v[0])[0], v[1]) for k, v in all_bank.items()}
all_n = {k: int(c[0].max()) + 2 for k, c in all_codes.items()}

te_tr = {fi: {} for fi in range(5)}
te_va = {fi: {} for fi in range(5)}
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
        te_tr[fi][kname] = (enc_a, cnt_a)
        te_va[fi][kname] = (mp_f[cb], cnt_f[cb])

print(f"[+] Precomputed {len(all_bank)} strict TE keys in {time.time()-t0:.1f}s", flush=True)

def run_lgb(key_list, lr=0.04, rounds=500, leaves=15, ff=0.4, label=""):
    oof = np.zeros(ntr)
    fold_aucs = []
    params = dict(objective="binary", learning_rate=lr, num_leaves=leaves, min_child_samples=150,
                  feature_fraction=ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, verbose=-1, n_jobs=8, seed=42)
    for fi, (a, b) in enumerate(folds):
        Xa = X_base.iloc[a].copy()
        Xb = X_base.iloc[b].copy()
        for kn in key_list:
            ea, ca = te_tr[fi][kn]; eb, cb = te_va[fi][kn]
            Xa[f"TE_{kn}"] = ea; Xa[f"CNT_{kn}"] = ca
            Xb[f"TE_{kn}"] = eb; Xb[f"CNT_{kn}"] = cb
        dtr = lgb.Dataset(Xa, label=y[a], free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=rounds)
        pv = m.predict(Xb)
        oof[b] = pv
        fold_aucs.append(roc_auc_score(y[b], pv))
    tot = roc_auc_score(y, oof)
    print(f"{label:34s} | OOF={tot:.6f} | folds={[round(x,5) for x in fold_aucs]}", flush=True)
    return tot, fold_aucs

cur_keys = list(best27.keys())
b_tot, b_f = run_lgb(cur_keys, label="REF: Best 27 TE Keys (500r)")

for gname, gdict in new_candidates.items():
    cand = cur_keys + list(gdict.keys())
    gtot, gf = run_lgb(cand, label=f"+{gname}")
    wins = sum(x > y_ for x, y_ in zip(gf, b_f))
    print(f"  Delta: {(gtot-b_tot)*10000:+.2f} bps ({wins}/5 wins)", flush=True)
    if wins >= 4 and gtot > b_tot:
        cur_keys, b_tot, b_f = cand, gtot, gf
        print(f"  >>> ACCEPTED {gname}! New Best={b_tot:.6f}", flush=True)

# Now test full convergence schedule matching V22 rounds!
print("\n--- Convergence Scaling Check (lr=0.025, rounds=1100 vs lr=0.04, 500r) ---", flush=True)
run_lgb(cur_keys, lr=0.025, rounds=1100, leaves=15, ff=0.35, label="CONV: BestKeys lr=0.025,1100r,L15")
run_lgb(cur_keys, lr=0.025, rounds=1100, leaves=23, ff=0.35, label="CONV: BestKeys lr=0.025,1100r,L23")
