import time, os, re
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

t_start = time.time()
np.random.seed(42)

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y = df_train["TARGET"].values.astype(np.int32)
ntr = len(df_train)

CATS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]

def build_clean_train_features(df):
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

    # V27 verified upgrades (pure row transforms)
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

X_base = build_clean_train_features(df_train)

# Strictly Train-Only Full-Covariate Residuals (Zero test data, zero y_train!)
t_res = time.time()
raw_res = df_train.drop(columns=["SK_ID_CURR", "TARGET"]).copy()
for c in CATS:
    raw_res[c] = raw_res[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
raw_res["DAYS_EMPLOYED"] = raw_res["DAYS_EMPLOYED"].where(raw_res["DAYS_EMPLOYED"] != 365243)
demo_cols = [c for c in raw_res.columns if c.startswith("EXT_SOURCE") or c.startswith("AMT_")]

T_spec = {
    "EXT1": (raw_res["EXT_SOURCE_1"], ["EXT_SOURCE_1"]),
    "EXT2": (raw_res["EXT_SOURCE_2"], ["EXT_SOURCE_2"]),
    "EXT3": (raw_res["EXT_SOURCE_3"], ["EXT_SOURCE_3"]),
    "LOGINC": (np.log(raw_res["AMT_INCOME_TOTAL"]), ["AMT_INCOME_TOTAL"]),
    "LOGCR": (np.log(raw_res["AMT_CREDIT"]), ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]),
    "EMP": (raw_res["DAYS_EMPLOYED"], ["DAYS_EMPLOYED"]),
    "REG": (raw_res["DAYS_REGISTRATION"], ["DAYS_REGISTRATION"]),
    "AGE": (raw_res["DAYS_BIRTH"], ["DAYS_BIRTH"]),
    "IDP": (raw_res["DAYS_ID_PUBLISH"], ["DAYS_ID_PUBLISH"]),
    "PHONE": (raw_res["DAYS_LAST_PHONE_CHANGE"], ["DAYS_LAST_PHONE_CHANGE"]),
    "EXT1D": (raw_res["EXT_SOURCE_1"], demo_cols),
    "EXT2D": (raw_res["EXT_SOURCE_2"], demo_cols),
    "EXT3D": (raw_res["EXT_SOURCE_3"], demo_cols),
}

res_out = {}
for nm, (yt, drop_cols) in T_spec.items():
    Xp = raw_res.drop(columns=drop_cols)
    pred = np.zeros(ntr)
    obs = yt.notna().values
    for tri, tei in KFold(5, shuffle=True, random_state=42).split(Xp):
        m = HistGradientBoostingRegressor(max_iter=60, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, random_state=42)
        m.fit(Xp.iloc[tri][obs[tri]], yt.iloc[tri][obs[tri]])
        pred[tei] = m.predict(Xp.iloc[tei])
    res_out[f"RP_{nm}"] = pred
    res_out[f"RR_{nm}"] = np.where(obs, yt.values - pred, np.nan)
df_res = pd.DataFrame(res_out, index=df_train.index).astype(np.float32)
print(f"[+] Train-Only 13 Full-Covariate Residuals built in {time.time()-t_res:.1f}s ({df_res.shape[1]} cols)", flush=True)

# Strict Nested OOF Multi-View Stacking (Fixes BOTH in-sample leakage AND np.concatenate scrambling!)
t_mv = time.time()
Xs_mv = pd.concat([X_base, df_res], axis=1).copy()
cash_mask = df_train["NAME_CONTRACT_TYPE"] == "Cash loans"
Xs_mv["Q0_1285"] = np.where(cash_mask, (df_train["AMT_ANNUITY"] - 128.5) / df_train["AMT_CREDIT"], df_train["AMT_ANNUITY"] / df_train["AMT_CREDIT"])
cols_mv = list(Xs_mv.columns)
v_ext = [c for c in cols_mv if "EXT" in c or c in ("E12","E13","E23","D12","D13","D23")]
v_loan = [c for c in cols_mv if c.startswith(("AMT_","LOG_","CREDIT_","ANN_","GOODS_","TERM","Q0_","FINANCED","APR_36M","NEG_DOWN")) or "LOGCR" in c or "LOGINC" in c]
v_person = [c for c in cols_mv if c not in set(v_ext) | set(v_loan)]
views_map = {"EXT": v_ext, "LOAN": v_loan, "PERSON": v_person}

lg = lambda p: np.log(np.clip(p, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(p, 1e-4, 1.0 - 1e-4)))
mk_view = lambda: HistGradientBoostingClassifier(max_iter=80, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=20, random_state=42)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(df_train, y))

view_tr_folds, view_va_folds = [], []
for k, (a, b) in enumerate(folds):
    trv, vav = {}, {}
    for vn, vc in views_map.items():
        inner = np.zeros(len(a))
        pvb = []
        for ia, ib in StratifiedKFold(3, shuffle=True, random_state=11 + k).split(a, y[a]):
            m = mk_view().fit(Xs_mv.iloc[a[ia]][vc], y[a[ia]])
            inner[ib] = m.predict_proba(Xs_mv.iloc[a[ib]][vc])[:, 1]
            pvb.append(m.predict_proba(Xs_mv.iloc[b][vc])[:, 1])
        trv[f"VIEW_{vn}"] = lg(inner)
        vav[f"VIEW_{vn}"] = lg(np.mean(pvb, axis=0))
    df_trv = pd.DataFrame(trv)
    df_vav = pd.DataFrame(vav)
    for d in (df_trv, df_vav):
        d["VD_EL"] = d["VIEW_EXT"] - d["VIEW_LOAN"]
        d["VD_EP"] = d["VIEW_EXT"] - d["VIEW_PERSON"]
        d["VD_LP"] = d["VIEW_LOAN"] - d["VIEW_PERSON"]
    view_tr_folds.append(df_trv.astype(np.float32))
    view_va_folds.append(df_vav.astype(np.float32))
print(f"[+] Strictly Aligned Inner-OOF Multi-Views built in {time.time()-t_mv:.1f}s", flush=True)

# Precompute 27 Best TE Keys cleanly inside folds
cr = df_train["AMT_CREDIT"]; an = df_train["AMT_ANNUITY"]; gp = df_train["AMT_GOODS_PRICE"]
q_r = an / cr
q0_1285 = pd.Series(np.where(cash_mask, (an - 128.5) / cr, q_r), index=df_train.index)
q0_1295 = pd.Series(np.where(cash_mask, (an - 129.5) / cr, q_r), index=df_train.index)
ct = cr / an
kf066 = ((cr / gp - 1.0) / 0.0066).round()

best27 = {
    "Q3": (q_r.round(3).astype(str), 100.0), "Q4": (q_r.round(4).astype(str), 30.0),
    "P3": (q0_1295.round(3).astype(str), 100.0), "P4": (q0_1295.round(4).astype(str), 30.0),
    "CT1": (ct.round(1).astype(str), 50.0), "CT2": (ct.round(2).astype(str), 50.0),
    "K066": (pd.Series(kf066).astype(str), 30.0),
    "GENDER_EDU": (df_train["CODE_GENDER"].astype(str) + "|" + df_train["NAME_EDUCATION_TYPE"].astype(str), 30.0),
    "ORG": (df_train["ORGANIZATION_TYPE"].astype(str), 30.0),
    "OCC": (df_train["OCCUPATION_TYPE"].fillna("NA").astype(str), 30.0),
    "ORG_OCC": (df_train["ORGANIZATION_TYPE"].astype(str) + "|" + df_train["OCCUPATION_TYPE"].fillna("NA").astype(str), 30.0),
    "CT1_GP": (ct.round(1).astype(str) + "|" + gp.astype(str), 50.0),
    "CT1_CR": (ct.round(1).astype(str) + "|" + cr.astype(str), 50.0),
    "CT1_AN": (ct.round(1).astype(str) + "|" + an.astype(str), 50.0),
    "Q5": (q_r.round(5).astype(str), 30.0), "P5": (q0_1295.round(5).astype(str), 30.0),
    "P4_1285": (q0_1285.round(4).astype(str), 30.0), "P5_1285": (q0_1285.round(5).astype(str), 30.0),
    "CT3": (ct.round(3).astype(str), 50.0), "GP": (gp.astype(str), 50.0),
    "CR": (cr.astype(str), 50.0), "AN": (an.astype(str), 50.0),
    "CT1_s10": (ct.round(1).astype(str), 10.0), "CT1_s200": (ct.round(1).astype(str), 200.0),
    "K066_P4": (pd.Series(kf066).astype(str) + "|" + q0_1285.round(4).astype(str), 20.0),
    "REG_HOUS": (df_train["REGION_RATING_CLIENT"].astype(str) + "|" + df_train["NAME_HOUSING_TYPE"].astype(str), 30.0),
    "INC_EDU": (df_train["NAME_INCOME_TYPE"].astype(str) + "|" + df_train["NAME_EDUCATION_TYPE"].astype(str), 30.0),
}

all_codes = {k: (pd.factorize(v[0])[0], v[1]) for k, v in best27.items()}
all_n = {k: int(c[0].max()) + 2 for k, c in all_codes.items()}
te_tr_df, te_va_df = [], []
for fi, (a, b) in enumerate(folds):
    pr_a = float(y[a].mean())
    in_splits = list(StratifiedKFold(5, shuffle=True, random_state=42+fi).split(a, y[a]))
    d_tr, d_va = {}, {}
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
            enc_a[ib] = mp[cib]; cnt_a[ib] = cnt[cib]
        cnt_f = np.bincount(ca, minlength=n_cat).astype(float)
        sm_f = np.bincount(ca, weights=y[a], minlength=n_cat)
        mp_f = (sm_f + m0 * pr_a) / (cnt_f + m0)
        d_tr[f"TE_{kname}"] = enc_a; d_tr[f"CNT_{kname}"] = cnt_a
        d_va[f"TE_{kname}"] = mp_f[cb]; d_va[f"CNT_{kname}"] = cnt_f[cb]
    te_tr_df.append(pd.DataFrame(d_tr, dtype=np.float32))
    te_va_df.append(pd.DataFrame(d_va, dtype=np.float32))

# Check individual lifts of Residuals and Multi-Views + Diversity Zoo
X_res_full = pd.concat([X_base, df_res], axis=1)

def run_lgb_test(use_res, use_views, extra_trees=False, lr=0.04, rounds=500, leaves=15, label=""):
    oof = np.zeros(ntr)
    fold_aucs = []
    params = dict(objective="binary", learning_rate=lr, num_leaves=leaves, min_child_samples=150,
                  feature_fraction=0.35, bagging_fraction=0.8, bagging_freq=1, lambda_l2=25,
                  extra_trees=extra_trees, verbose=-1, n_jobs=8, seed=42)
    X_src = X_res_full if use_res else X_base
    for fi, (a, b) in enumerate(folds):
        parts_a = [X_src.iloc[a].reset_index(drop=True), te_tr_df[fi]]
        parts_b = [X_src.iloc[b].reset_index(drop=True), te_va_df[fi]]
        if use_views:
            parts_a.append(view_tr_folds[fi]); parts_b.append(view_va_folds[fi])
        Xa = pd.concat(parts_a, axis=1); Xb = pd.concat(parts_b, axis=1)
        m = lgb.train(params, lgb.Dataset(Xa, label=y[a], free_raw_data=False), num_boost_round=rounds)
        pv = m.predict(Xb)
        oof[b] = pv
        fold_aucs.append(roc_auc_score(y[b], pv))
    print(f"{label:34s} | OOF={roc_auc_score(y, oof):.6f} | folds={[round(x,5) for x in fold_aucs]}", flush=True)
    return oof

oof_lgb_base = run_lgb_test(False, False, False, 0.04, 550, 15, "LGB1: Base+27TE (NoRes,NoView)")
oof_lgb_res  = run_lgb_test(True,  False, False, 0.04, 550, 15, "LGB2: +TrainOnlyRes (NoView)")
oof_lgb_view = run_lgb_test(True,  True,  False, 0.04, 550, 15, "LGB3: +TrainOnlyRes+AlignedViews")
oof_lgb_et   = run_lgb_test(True,  True,  True,  0.035, 700, 23, "LGB4: Deep ExtraTrees (Views)")

# Train XGBoost CUDA GPU Specialist cleanly on Train-Only Res + Views + TE
oof_xgb = np.zeros(ntr)
xgb_f = []
for fi, (a, b) in enumerate(folds):
    Xa = pd.concat([X_res_full.iloc[a].reset_index(drop=True), te_tr_df[fi], view_tr_folds[fi]], axis=1).astype(np.float32)
    Xb = pd.concat([X_res_full.iloc[b].reset_index(drop=True), te_va_df[fi], view_va_folds[fi]], axis=1).astype(np.float32)
    mx = xgb.XGBClassifier(n_estimators=650, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.45,
                           reg_alpha=3.0, reg_lambda=10.0, min_child_weight=50, tree_method="hist", device="cuda",
                           eval_metric="auc", random_state=42 + fi)
    mx.fit(Xa, y[a], verbose=False)
    pv = mx.predict_proba(Xb)[:, 1]
    oof_xgb[b] = pv
    xgb_f.append(roc_auc_score(y[b], pv))
print(f"{'XGB_GPU: CUDA Hist D5':34s} | OOF={roc_auc_score(y, oof_xgb):.6f} | folds={[round(x,5) for x in xgb_f]}", flush=True)

# Train CatBoost GPU Specialist cleanly on Rich Numeric + Native High-Card String Tariff & Socio Keys (fixed 950 iters, zero b leakage)
cb_num = X_res_full.drop(columns=CATS).copy()
cb_str = pd.DataFrame({
    **{c: df_train[c].fillna("NA").astype(str) for c in CATS},
    **{f"K_{k}": best27[k][0].astype(str) for k in ["Q3","Q4","P3","P4","P4_1285","CT1","CT2","CT1_GP","CT1_CR","CT1_AN","K066","K066_P4","GENDER_EDU","REG_HOUS","INC_EDU"]}
})
cb_df = pd.concat([cb_num, cb_str], axis=1)
cb_cats = list(cb_str.columns)
oof_cb = np.zeros(ntr)
cb_f = []
for fi, (a, b) in enumerate(folds):
    ca = pd.concat([cb_df.iloc[a].reset_index(drop=True), view_tr_folds[fi]], axis=1)
    cbv = pd.concat([cb_df.iloc[b].reset_index(drop=True), view_va_folds[fi]], axis=1)
    mc = CatBoostClassifier(iterations=950, depth=7, learning_rate=0.04, l2_leaf_reg=10.0,
                            loss_function="Logloss", random_seed=42 + fi, task_type="GPU", verbose=False)
    mc.fit(ca, y[a], cat_features=cb_cats, verbose=False)
    pv = mc.predict_proba(cbv)[:, 1]
    oof_cb[b] = pv
    cb_f.append(roc_auc_score(y[b], pv))
print(f"{'CB_GPU: Oblivious Native Cats':34s} | OOF={roc_auc_score(y, oof_cb):.6f} | folds={[round(x,5) for x in cb_f]}", flush=True)

# STRICT NESTED OOF ENSEMBLE EVALUATION WITH TRAIN-ONLY QUANTILE PROBIT CALIBRATION
def fit_probit_map(p_fit):
    p_s = np.sort(p_fit)
    u_s = np.linspace(1e-5, 1.0 - 1e-5, len(p_s))
    return p_s, norm.ppf(u_s)

def apply_probit_map(p_target, p_s, z_s):
    return np.interp(p_target, p_s, z_s)

streams = [oof_lgb_res, oof_lgb_view, oof_lgb_et, oof_xgb, oof_cb]
nested_oof = np.zeros(ntr)
eq_oof = np.zeros(ntr)
nested_folds = []

for fi, (a, b) in enumerate(folds):
    Za_cols, Zb_cols = [], []
    for s in streams:
        p_s, z_s = fit_probit_map(s[a])
        Za_cols.append(apply_probit_map(s[a], p_s, z_s))
        Zb_cols.append(apply_probit_map(s[b], p_s, z_s))
    Za = np.column_stack(Za_cols)
    Zb = np.column_stack(Zb_cols)
    def loss_fn(w):
        w = np.maximum(w, 0); sm = w.sum()
        if sm == 0: return 1.0
        w = w / sm
        return -roc_auc_score(y[a], Za @ w)
    opt = minimize(loss_fn, np.ones(len(streams))/len(streams), method="Nelder-Mead", options={"maxiter": 300})
    w_k = np.maximum(opt.x, 0); w_k /= w_k.sum()
    nested_oof[b] = norm.cdf(Zb @ w_k)
    eq_oof[b] = norm.cdf(Zb.mean(axis=1))
    nested_folds.append(roc_auc_score(y[b], nested_oof[b]))
    print(f"  Nested Fold {fi+1} weights: {np.round(w_k, 3)} | Untouched Val AUC: {nested_folds[-1]:.6f}", flush=True)

print("\n" + "="*80, flush=True)
print(f"  STRICTLY UNBIASED NESTED OOF AUC (100% CLEAN-ROOM): {roc_auc_score(y, nested_oof):.6f}", flush=True)
print(f"  Equal-Weight Probit OOF AUC                       : {roc_auc_score(y, eq_oof):.6f}", flush=True)
print(f"  Total elapsed: {time.time()-t_start:.1f}s", flush=True)
print("="*80, flush=True)
