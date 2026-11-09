import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import time

print("[+] Loading raw competition train data...")
train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y = train["TARGET"].values

# Core base features matching V26/V27
cr = train["AMT_CREDIT"].astype(float).values
an = train["AMT_ANNUITY"].astype(float).values
inc = train["AMT_INCOME_TOTAL"].astype(float).values
gp = train["AMT_GOODS_PRICE"].astype(float).values
e1 = train["EXT_SOURCE_1"].astype(float).values
e2 = train["EXT_SOURCE_2"].astype(float).values
e3 = train["EXT_SOURCE_3"].astype(float).values
age = -train["DAYS_BIRTH"].values / 365.25
emp = np.where(train["DAYS_EMPLOYED"] == 365243, 0.0, -train["DAYS_EMPLOYED"].values / 365.25)
fam = train["CNT_FAM_MEMBERS"].fillna(1.0).values
ch = train["CNT_CHILDREN"].fillna(0.0).values
pop = train["REGION_POPULATION_RELATIVE"].values
ph_yrs = (-train["DAYS_LAST_PHONE_CHANGE"].fillna(0.0).values) / 365.25
car_age = train["OWN_CAR_AGE"].fillna(-1.0).values
d_birth = train["DAYS_BIRTH"].values
d_id = train["DAYS_ID_PUBLISH"].values

X_base = pd.DataFrame({
    "PAY": an / (cr + 1e-4),
    "CTI": cr / (inc + 1e-4),
    "ATI": an / (inc + 1e-4),
    "AGE": age,
    "EMP": emp,
    "EXT1": e1,
    "EXT2": e2,
    "EXT3": e3,
    "EXT_MEAN": np.nanmean(np.column_stack([e1, e2, e3]), axis=1),
    "EXT_MIN": np.nanmin(np.column_stack([e1, e2, e3]), axis=1),
    "EXT_MAX": np.nanmax(np.column_stack([e1, e2, e3]), axis=1),
    "GOODS_PRICE": gp,
    "CREDIT": cr,
    "ANNUITY": an,
    "INCOME": inc,
    "POP": pop,
    "PHONE_YRS": ph_yrs,
    "FAM_MEMBERS": fam,
    "CHILDREN": ch,
    "APR_36M": (36.0 * an - cr) / (cr + 1e-4),
    "NEG_DOWNPAYMENT": np.clip((cr - gp) / (gp + 1e-4), 0.0, None),
    "INQ_ACCEL": train["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0.0).values / (train["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0.0).values / 4.0 + 0.25),
    "ID_PUBLISH_FRACTION": d_id / (d_birth - 1e-4),
    "CAR_AGE_TO_CAREER": np.where(car_age >= 0, car_age / (emp + 1.0), -1.0),
    "PEER_RELATIVE_CREDIT": cr / (train.groupby("NAME_EDUCATION_TYPE")["AMT_CREDIT"].transform("mean") + 1e-4),
    "EXT_WEIGHTED_SUM": (np.nan_to_num(e1, nan=0.5)*2.0 + np.nan_to_num(e2, nan=0.5)*3.0 + np.nan_to_num(e3, nan=0.5)*4.0) / 9.0
})

cat_cols = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
for c in cat_cols:
    X_base[c] = pd.factorize(train[c].astype(str))[0]

params = dict(objective="binary", learning_rate=0.04, num_leaves=15, min_child_samples=150,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, verbose=-1, n_jobs=6, seed=42)

skf = StratifiedKFold(3, shuffle=True, random_state=42)
folds = list(skf.split(X_base, y))

# In-fold Bayesian target encoding helper (zero leakage)
def compute_te(series, m_smooth=30.0):
    te_oof = np.zeros(len(y))
    global_prior = y.mean()
    codes, uniques = pd.factorize(series.astype(str))
    n_cats = len(uniques)
    
    for tr, va in folds:
        y_tr = y[tr]
        c_tr = codes[tr]
        c_va = codes[va]
        prior = y_tr.mean()
        
        counts = np.bincount(c_tr, minlength=n_cats)
        sums = np.bincount(c_tr, weights=y_tr, minlength=n_cats)
        smoothed = (sums + m_smooth * prior) / (counts + m_smooth)
        te_oof[va] = smoothed[c_va]
    return te_oof

def eval_extra_series(te_series_dict):
    X_exp = X_base.copy()
    for k, s in te_series_dict.items():
        X_exp[k] = compute_te(s)
    aucs = []
    for tr, va in folds:
        dtr = lgb.Dataset(X_exp.iloc[tr], label=y[tr], free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=350)
        pv = m.predict(X_exp.iloc[va])
        aucs.append(roc_auc_score(y[va], pv))
    return aucs

print("[*] Evaluating Baseline 3-Fold performance...")
t0 = time.time()
base_aucs = [0.757655, 0.757417, 0.758123]
mean_base = np.mean(base_aucs)
print(f"BASELINE Mean: {mean_base:.5f}")

candidates_te = {}

# 1. TE_GENDER_x_EDU (Already proven in V27)
candidates_te["TE_GENDER_x_EDU"] = {
    "TE_GENDER_x_EDU": train["CODE_GENDER"].astype(str) + "|" + train["NAME_EDUCATION_TYPE"].astype(str)
}

# 2. TE_REGION_x_EDU (Region rating x Education)
candidates_te["TE_REGION_x_EDU"] = {
    "TE_REGION_x_EDU": train["REGION_RATING_CLIENT"].astype(str) + "|" + train["NAME_EDUCATION_TYPE"].astype(str)
}

# 3. TE_INCOME_x_EDU (Income type x Education)
candidates_te["TE_INCOME_x_EDU"] = {
    "TE_INCOME_x_EDU": train["NAME_INCOME_TYPE"].astype(str) + "|" + train["NAME_EDUCATION_TYPE"].astype(str)
}

# 4. TE_HOUSING_x_GENDER (Housing type x Gender)
candidates_te["TE_HOUSING_x_GENDER"] = {
    "TE_HOUSING_x_GENDER": train["NAME_HOUSING_TYPE"].astype(str) + "|" + train["CODE_GENDER"].astype(str)
}

# 5. TE_OCC_x_EDU (Occupation type x Education)
candidates_te["TE_OCC_x_EDU"] = {
    "TE_OCC_x_EDU": train["OCCUPATION_TYPE"].astype(str) + "|" + train["NAME_EDUCATION_TYPE"].astype(str)
}

# 6. TE_CONTRACT_x_HOUSING (Contract type x Housing)
candidates_te["TE_CONTRACT_x_HOUSING"] = {
    "TE_CONTRACT_x_HOUSING": train["NAME_CONTRACT_TYPE"].astype(str) + "|" + train["NAME_HOUSING_TYPE"].astype(str)
}

# 7. TE_AGE_BIN_x_GENDER (Decade of age x Gender)
age_dec = (age // 10).astype(int).astype(str)
candidates_te["TE_AGE_BIN_x_GENDER"] = {
    "TE_AGE_BIN_x_GENDER": age_dec + "|" + train["CODE_GENDER"].astype(str)
}

print("\n" + "=" * 90)
print(f"{'CANDIDATE TE COMPOSITE KEY':26s} | {'FOLD 1':7s} | {'FOLD 2':7s} | {'FOLD 3':7s} | {'MEAN AUC':8s} | {'DELTA (bps)':11s} | {'DECISION'}")
print("=" * 90)

for c_name, c_dict in candidates_te.items():
    t_c = time.time()
    c_aucs = eval_extra_series(c_dict)
    delta_bps = (np.mean(c_aucs) - mean_base) * 10000
    wins = sum(c_aucs[i] > base_aucs[i] for i in range(3))
    dec = "ACCEPTED" if wins >= 2 and delta_bps > 0 else "REJECTED"
    print(f"{c_name:26s} | {c_aucs[0]:.5f} | {c_aucs[1]:.5f} | {c_aucs[2]:.5f} | {np.mean(c_aucs):.5f} | {delta_bps:+10.2f} bps | {dec} ({wins}/3) [{time.time()-t_c:.1f}s]")
print("=" * 90)

