import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import time

print("[+] Loading raw competition train data...")
train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y = train["TARGET"].values

# Core base features matching V27
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
    "FLAG_EMP_PHONE": train["FLAG_EMP_PHONE"].values,
    "FLAG_WORK_PHONE": train["FLAG_WORK_PHONE"].values,
    "APR_36M": (36.0 * an - cr) / (cr + 1e-4),
    "NEG_DOWNPAYMENT": np.clip((cr - gp) / (gp + 1e-4), 0.0, None),
    "INQ_ACCEL": train["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0.0).values / (train["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0.0).values / 4.0 + 0.25),
    "ID_PUBLISH_FRACTION": d_id / (d_birth - 1e-4),
    "CAR_AGE_TO_CAREER": np.where(car_age >= 0, car_age / (emp + 1.0), -1.0),
    "PEER_RELATIVE_CREDIT": cr / (train.groupby("NAME_EDUCATION_TYPE")["AMT_CREDIT"].transform("mean") + 1e-4),
    "EXT_WEIGHTED_SUM": (np.nan_to_num(e1, nan=0.5)*2.0 + np.nan_to_num(e2, nan=0.5)*3.0 + np.nan_to_num(e3, nan=0.5)*4.0) / 9.0,
    "EXT_MISSING_COUNT": np.isnan(e1).astype(float) + np.isnan(e2).astype(float) + np.isnan(e3).astype(float)
})

cat_cols = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
for c in cat_cols:
    X_base[c] = pd.factorize(train[c].astype(str))[0]

params = dict(objective="binary", learning_rate=0.04, num_leaves=15, min_child_samples=150,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, verbose=-1, n_jobs=8, seed=42)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X_base, y))

def eval_5fold(extra_cols=None):
    X = X_base.copy()
    if extra_cols:
        for k, v in extra_cols.items():
            X[k] = v
    oof = np.zeros(len(y))
    fold_aucs = []
    for k, (tr, va) in enumerate(folds):
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr], free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=450)
        pv = m.predict(X.iloc[va])
        oof[va] = pv
        auc_f = roc_auc_score(y[va], pv)
        fold_aucs.append(auc_f)
    total_auc = roc_auc_score(y, oof)
    return fold_aucs, total_auc

print("[*] Running 5-Fold Baseline...")
t0 = time.time()
base_folds, base_total = eval_5fold()
print(f"5-Fold Baseline Folds: {[round(x, 5) for x in base_folds]} | Total OOF: {base_total:.6f} ({time.time()-t0:.1f}s)")

# Combined new accepted features
combo_upgrades = {
    "CREDIT_TERM_APPROX": cr / (an * 12.0 + 1e-4),
    "ANNUITY_TO_GOODS": an / (gp + 1e-4),
    "FINANCED_MARKUP_RATIO": cr / (gp + 1e-4)
}

print("[*] Running 5-Fold with Combo Upgrades (Credit Term Approx + Annuity To Goods + Financed Markup)...")
t1 = time.time()
upg_folds, upg_total = eval_5fold(combo_upgrades)
print(f"Upgraded Folds:       {[round(x, 5) for x in upg_folds]} | Total OOF: {upg_total:.6f} ({time.time()-t1:.1f}s)")

delta_total_bps = (upg_total - base_total) * 10000
print(f"\nNet 5-Fold Lift: {delta_total_bps:+.2f} bps")
for i in range(5):
    d_f = (upg_folds[i] - base_folds[i]) * 10000
    print(f"  Fold {i+1}: {base_folds[i]:.5f} -> {upg_folds[i]:.5f} ({d_f:+.2f} bps)")

