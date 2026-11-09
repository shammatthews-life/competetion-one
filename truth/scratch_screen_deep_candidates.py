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
d_reg = train["DAYS_REGISTRATION"].values
d_ph = train["DAYS_LAST_PHONE_CHANGE"].values

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

skf = StratifiedKFold(3, shuffle=True, random_state=42)
folds = list(skf.split(X_base, y))

def eval_feature(extra_dict):
    X_exp = X_base.copy()
    for k, v in extra_dict.items():
        X_exp[k] = v
    aucs = []
    for tr, va in folds:
        dtr = lgb.Dataset(X_exp.iloc[tr], label=y[tr], free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=350)
        pv = m.predict(X_exp.iloc[va])
        aucs.append(roc_auc_score(y[va], pv))
    return aucs

print("[*] Evaluating Baseline 3-Fold performance...")
t0 = time.time()
base_aucs = eval_feature({})
print(f"BASELINE 3-Fold: {base_aucs} | Mean: {np.mean(base_aucs):.5f} ({time.time()-t0:.1f}s)")

candidates = {}

# 1. FINANCED_MARKUP_RATIO (cr / gp unclipped)
candidates["FINANCED_MARKUP_RATIO"] = {"FINANCED_MARKUP_RATIO": cr / (gp + 1e-4)}

# 2. IS_PARITY_LOAN (cr == gp binary flag)
candidates["IS_PARITY_LOAN"] = {"IS_PARITY_LOAN": (cr == gp).astype(float)}

# 3. PHONE_REG_SPAN (Time between address registration and phone change)
candidates["PHONE_REG_SPAN"] = {"PHONE_REG_SPAN": (d_ph - d_reg) / 365.25}

# 4. REG_ID_SPAN (Time between address registration and ID publish)
candidates["REG_ID_SPAN"] = {"REG_ID_SPAN": (d_reg - d_id) / 365.25}

# 5. CAREER_TO_AGE_RATIO (emp years / age years for working adults)
candidates["CAREER_TO_AGE_RATIO"] = {"CAREER_TO_AGE_RATIO": np.where(emp > 0, emp / (age + 1e-4), 0.0)}

# 6. ANNUITY_TO_GOODS (Payment-to-Goods ratio)
candidates["ANNUITY_TO_GOODS"] = {"ANNUITY_TO_GOODS": an / (gp + 1e-4)}

# 7. CONTRACT_CASH_FLAG (NAME_CONTRACT_TYPE == Cash loans)
candidates["CONTRACT_CASH_FLAG"] = {"CONTRACT_CASH_FLAG": (train["NAME_CONTRACT_TYPE"] == "Cash loans").astype(float)}

# 8. CREDIT_TERM_APPROX (cr / (an * 12) implied duration in years)
candidates["CREDIT_TERM_APPROX"] = {"CREDIT_TERM_APPROX": cr / (an * 12.0 + 1e-4)}

print("\n" + "=" * 90)
print(f"{'CANDIDATE HYPOTHESIS':26s} | {'FOLD 1':7s} | {'FOLD 2':7s} | {'FOLD 3':7s} | {'MEAN AUC':8s} | {'DELTA (bps)':11s} | {'DECISION'}")
print("=" * 90)

for c_name, c_dict in candidates.items():
    t_c = time.time()
    c_aucs = eval_feature(c_dict)
    delta_bps = (np.mean(c_aucs) - np.mean(base_aucs)) * 10000
    wins = sum(c_aucs[i] > base_aucs[i] for i in range(3))
    dec = "ACCEPTED" if wins >= 2 and delta_bps > 0 else "REJECTED"
    print(f"{c_name:26s} | {c_aucs[0]:.5f} | {c_aucs[1]:.5f} | {c_aucs[2]:.5f} | {np.mean(c_aucs):.5f} | {delta_bps:+10.2f} bps | {dec} ({wins}/3) [{time.time()-t_c:.1f}s]")
print("=" * 90)

