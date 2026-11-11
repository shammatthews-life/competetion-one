import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import sys
sys.path.insert(0, r"D:\gci")
import solo_champion_v28_certified as v28

v28.set_all_seeds(42)

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y_train = df_train["TARGET"].values.astype(np.int32)
outer_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(df_train)), y_train))
a, b = outer_folds[0]

df_a, df_b = df_train.iloc[a], df_train.iloc[b]
y_a, y_b = y_train[a], y_train[b]

print("=== BENCHMARK 1: Strict Inner-Fold Prior `prior_inner` vs Old `pr` in TE + Inner Preprocessing ===")
t0 = time.time()
te_old_a, te_old_b, _, _ = v28.compute_local_te(df_a, df_b, y_a, n_inner=5, seed=42)

def compute_local_te_strict(df_fit, df_val, y_fit, n_inner=4, seed=42, df_te=None):
    kf = v28.build_key_series(df_fit)
    kv = v28.build_key_series(df_val)
    kt = v28.build_key_series(df_te) if df_te is not None else None

    in_splits = list(StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(np.zeros(len(y_fit)), y_fit))
    d_fit, d_val = {}, {}
    d_te = {} if df_te is not None else None

    for kname, (s_fit, m0) in kf.items():
        uniq = pd.unique(s_fit)
        cmap = {v: i for i, v in enumerate(uniq)}
        unk = len(uniq); n_cat = unk + 1
        ca = s_fit.map(cmap).fillna(unk).astype(np.int32).values
        cb = kv[kname][0].map(cmap).fillna(unk).astype(np.int32).values

        enc_a = np.empty(len(y_fit), dtype=np.float64)
        cnt_a = np.empty(len(y_fit), dtype=np.float64)
        for ia, ib in in_splits:
            prior_inner = float(y_fit[ia].mean())
            cia, cib = ca[ia], ca[ib]
            cnt = np.bincount(cia, minlength=n_cat).astype(float)
            sm = np.bincount(cia, weights=y_fit[ia], minlength=n_cat)
            mp = (sm + m0 * prior_inner) / (cnt + m0)
            enc_a[ib] = mp[cib]
            cnt_a[ib] = cnt[cib]

        prior_full = float(y_fit.mean())
        cnt_f = np.bincount(ca, minlength=n_cat).astype(float)
        sm_f = np.bincount(ca, weights=y_fit, minlength=n_cat)
        mp_f = (sm_f + m0 * prior_full) / (cnt_f + m0)
        d_fit[f"TE_{kname}"] = enc_a
        d_fit[f"CNT_{kname}"] = cnt_a
        d_val[f"TE_{kname}"] = mp_f[cb]
        d_val[f"CNT_{kname}"] = cnt_f[cb]
        if df_te is not None:
            ct_arr = kt[kname][0].map(cmap).fillna(unk).astype(np.int32).values
            d_te[f"TE_{kname}"] = mp_f[ct_arr]
            d_te[f"CNT_{kname}"] = cnt_f[ct_arr]

    res = (pd.DataFrame(d_fit, dtype=np.float32), pd.DataFrame(d_val, dtype=np.float32), kf, kv)
    return res if df_te is None else (res[0], res[1], pd.DataFrame(d_te, dtype=np.float32), kf, kv, kt)

te_new_a, te_new_b, _, _ = compute_local_te_strict(df_a, df_b, y_a, n_inner=5, seed=42)
max_diff = float(np.max(np.abs(te_old_a.values - te_new_a.values)))
print(f"Max TE matrix diff (`prior_inner` vs full `pr`): {max_diff:.8f} ({time.time()-t0:.1f}s)")

cmaps = v28.fit_category_maps(df_a)
Xa_base = v28.build_row_features(df_a, cmaps)
Xb_base = v28.build_row_features(df_b, cmaps)
res_a, res_b, _ = v28.compute_fold_residuals(df_a, df_b, df_b.iloc[:5], cmaps, seed=42)
Xa_res = pd.concat([Xa_base, res_a], axis=1)
Xb_res = pd.concat([Xb_base, res_b], axis=1)
v_a, v_b = v28.compute_local_views(Xa_res, Xb_res, df_a, df_b, y_a, n_inner=3, seed=42)

# Compare LGB_Mono_ET_6View old TE vs strict TE
for label, (ta, tb) in [("Old TE", (te_old_a, te_old_b)), ("Strict prior_inner TE", (te_new_a, te_new_b))]:
    Xa_f = pd.concat([Xa_res, ta, v_a], axis=1)
    Xb_f = pd.concat([Xb_res, tb, v_b], axis=1)
    p = dict(objective="binary", force_col_wise=True, bagging_freq=1, bagging_fraction=0.8, verbose=-1, n_jobs=8,
             seed=49, learning_rate=0.028, num_leaves=15, min_child_samples=150, feature_fraction=0.40, lambda_l2=25, extra_trees=True)
    p["monotone_constraints"] = [-1 if c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_MEAN") else (1 if c.startswith("TE_") else 0) for c in Xa_f.columns]
    m = lgb.train(p, lgb.Dataset(Xa_f, label=y_a, free_raw_data=False), num_boost_round=850)
    auc = roc_auc_score(y_b, m.predict(Xb_f))
    print(f"  LGB_Mono_ET_6View [{label:22s}] Fold 1 AUC: {auc:.6f}")

print("\n=== BENCHMARK 2: TabResMLP Single Seed vs 2-Seed + 3-Seed + Richer Cosine ===")
Xa_f = pd.concat([Xa_res, te_new_a, v_a], axis=1)
Xb_f = pd.concat([Xb_res, te_new_b, v_b], axis=1)

t1 = time.time()
p_m1 = v28.fit_predict_mlp(Xa_f, Xb_f, y_a, Xte_df=None, seed=42)
print(f"  TabResMLP 1-seed AUC: {roc_auc_score(y_b, p_m1):.6f} ({time.time()-t1:.1f}s)")

t2 = time.time()
p_m2 = v28.fit_predict_mlp(Xa_f, Xb_f, y_a, Xte_df=None, seed=1042)
p_m3 = v28.fit_predict_mlp(Xa_f, Xb_f, y_a, Xte_df=None, seed=2042)
print(f"  TabResMLP 2-seed avg AUC: {roc_auc_score(y_b, 0.5*(p_m1+p_m2)):.6f}")
print(f"  TabResMLP 3-seed avg AUC: {roc_auc_score(y_b, (p_m1+p_m2+p_m3)/3.0):.6f} (+{time.time()-t2:.1f}s extra)")

print("\n=== BENCHMARK 3: Clean Row Feature Additions (`ANN_FEE1285_CR`, `EXT_RATIO`, `LOG_BUR`, etc.) ===")
def add_row_candidates(df_raw, X):
    X2 = X.copy()
    an, cr, gp = df_raw["AMT_ANNUITY"], df_raw["AMT_CREDIT"], df_raw["AMT_GOODS_PRICE"]
    cash = df_raw["NAME_CONTRACT_TYPE"] == "Cash loans"
    X2["Q0_1285_NUM"] = np.where(cash, (an - 128.5) / cr, an / cr)
    X2["EXT_HARMONIC"] = 3.0 / (1.0 / (df_raw["EXT_SOURCE_1"] + 0.05) + 1.0 / (df_raw["EXT_SOURCE_2"] + 0.05) + 1.0 / (df_raw["EXT_SOURCE_3"] + 0.05))
    X2["CREDIT_EMP_RATIO"] = cr / (-df_raw["DAYS_EMPLOYED"].where(df_raw["DAYS_EMPLOYED"] != 365243) / 365.25 + 1.0)
    return X2.reset_index(drop=True)

Xa_cand = pd.concat([add_row_candidates(df_a, Xa_res), te_new_a, v_a], axis=1)
Xb_cand = pd.concat([add_row_candidates(df_b, Xb_res), te_new_b, v_b], axis=1)
p["monotone_constraints"] = [-1 if c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_MEAN") else (1 if c.startswith("TE_") else 0) for c in Xa_cand.columns]
m_c = lgb.train(p, lgb.Dataset(Xa_cand, label=y_a, free_raw_data=False), num_boost_round=850)
print(f"  LGB_Mono_ET_6View [+3 Clean Row Feats] Fold 1 AUC: {roc_auc_score(y_b, m_c.predict(Xb_cand)):.6f}")

p_mc1 = v28.fit_predict_mlp(Xa_cand, Xb_cand, y_a, Xte_df=None, seed=42)
p_mc2 = v28.fit_predict_mlp(Xa_cand, Xb_cand, y_a, Xte_df=None, seed=1042)
print(f"  TabResMLP 2-seed [+3 Clean Row Feats] Fold 1 AUC: {roc_auc_score(y_b, 0.5*(p_mc1+p_mc2)):.6f}")

