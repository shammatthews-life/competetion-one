import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import solo_champion_v28_certified as v28

v28.set_all_seeds(42)
assert torch.cuda.is_available()
DEV = torch.device("cuda")
print(f"[*] 3-Fold Inner-Dev Scientific Ablation on {torch.cuda.get_device_name(0)}")

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y_train = df_train["TARGET"].values.astype(np.int32)
a_dev = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(df_train)), y_train))[0][0]
y_dev = y_train[a_dev]
dev_splits = list(StratifiedKFold(3, shuffle=True, random_state=777).split(np.zeros(len(a_dev)), y_dev))

names = [
    "LGB_Mono", "XGB_D5_B512", "CB_D7_NoTE", "CB_D6_B254_TE",
    "MLP_Old", "MLP_NaN", "Periodic_SwiGLU"
]
oof = {k: np.zeros(len(a_dev)) for k in names}

for fi, (ia, ib) in enumerate(dev_splits):
    t0 = time.time()
    df_ia, df_ib = df_train.iloc[a_dev[ia]], df_train.iloc[a_dev[ib]]
    y_ia, y_ib = y_dev[ia], y_dev[ib]

    cmaps = v28.fit_category_maps(df_ia)
    Xia_b = v28.build_row_features(df_ia, cmaps)
    Xib_b = v28.build_row_features(df_ib, cmaps)
    res_ia, res_ib = v28.compute_fold_residuals(df_ia, df_ib, None, cmaps, seed=42 + fi)
    Xia_r = pd.concat([Xia_b, res_ia], axis=1)
    Xib_r = pd.concat([Xib_b, res_ib], axis=1)
    te_ia, te_ib, k_ia, k_ib = v28.compute_local_te(df_ia, df_ib, y_ia, n_inner=4, seed=42 + fi)
    v_ia, v_ib = v28.compute_local_views(Xia_r, Xib_r, df_ia, df_ib, y_ia, n_inner=2, seed=42 + fi)

    Xia_f = pd.concat([Xia_r, te_ia, v_ia], axis=1)
    Xib_f = pd.concat([Xib_r, te_ib, v_ib], axis=1)

    # 1. LGB_Mono
    pm = dict(objective="binary", force_col_wise=True, bagging_freq=1, bagging_fraction=0.8, verbose=-1, n_jobs=8,
              seed=49 + fi, learning_rate=0.028, num_leaves=15, min_child_samples=150, feature_fraction=0.40, lambda_l2=25, extra_trees=True)
    pm["monotone_constraints"] = [-1 if c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_MEAN") else (1 if c.startswith("TE_") else 0) for c in Xia_f.columns]
    oof["LGB_Mono"][ib] = lgb.train(pm, lgb.Dataset(Xia_f, label=y_ia, free_raw_data=False), num_boost_round=850).predict(Xib_f)

    # 2. XGB_D5_B512
    mx = xgb.XGBClassifier(n_estimators=850, learning_rate=0.026, max_depth=5, subsample=0.8, colsample_bytree=0.38,
                           min_child_weight=65, reg_alpha=3.5, reg_lambda=15.0, max_bin=512, tree_method="hist", device="cuda", random_state=42 + fi)
    mx.fit(Xia_f.astype(np.float32).values, y_ia, verbose=False)
    mx.get_booster().set_param({"device": "cpu"})
    oof["XGB_D5_B512"][ib] = mx.get_booster().inplace_predict(Xib_f.astype(np.float32).values)

    # 3. Neural pair + Old MLP (GPU)
    med = Xia_f.median()
    ma = Xia_f.isna().values.astype(np.float32); mb = Xib_f.isna().values.astype(np.float32)
    xa_raw = np.clip(Xia_f.fillna(med).fillna(0.0).values, -1e6, 1e6).astype(np.float32)
    xb_raw = np.clip(Xib_f.fillna(med).fillna(0.0).values, -1e6, 1e6).astype(np.float32)
    lo = np.percentile(xa_raw, 0.5, axis=0); hi = np.percentile(xa_raw, 99.5, axis=0)
    sc = StandardScaler()
    Xa_s = sc.fit_transform(np.clip(xa_raw, lo, hi)).astype(np.float32)
    Xb_s = sc.transform(np.clip(xb_raw, lo, hi)).astype(np.float32)
    Xa_m = np.hstack([Xa_s, ma]); Xb_m = np.hstack([Xb_s, mb])

    def run_gpu(Xa_arr, Xb_arr, mk, ep, lr, wd):
        Xt = torch.from_numpy(Xa_arr).to(DEV); yt = torch.from_numpy(y_ia.astype(np.float32)).to(DEV)
        Xv = torch.from_numpy(Xb_arr).to(DEV)
        acc = np.zeros(len(Xb_arr))
        for sd in (42 + fi, 1042 + fi):
            torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
            net = mk().to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep); crit = nn.BCEWithLogitsLoss()
            for _ in range(ep):
                net.train(); perm = torch.randperm(len(Xa_arr), device=DEV)
                for i in range(0, len(Xa_arr), 4096):
                    idx = perm[i:i + 4096]; opt.zero_grad(set_to_none=True)
                    loss = crit(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
                sched.step()
            net.eval()
            with torch.no_grad(): acc += torch.sigmoid(net(Xv)).cpu().numpy() / 2.0
        del Xt, yt, Xv; torch.cuda.empty_cache()
        return acc

    oof["MLP_Old"][ib] = run_gpu(Xa_s, Xb_s, lambda: v28.TabResMLP(Xa_s.shape[1]), 13, 3e-3, 2e-2)
    oof["MLP_NaN"][ib] = run_gpu(Xa_m, Xb_m, lambda: v28.TabResMLP(Xa_m.shape[1]), 13, 3e-3, 2e-2)
    oof["Periodic_SwiGLU"][ib] = run_gpu(Xa_m, Xb_m, lambda: v28.TabPeriodicSwiGLU(Xa_s.shape[1], ma.shape[1], sigma=0.12), 15, 2e-3, 2.5e-2)

    # 4. CatBoost GPU: No-TE D7_B128 vs +TE D6_B254
    cb_base = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_CONTRACT_TYPE"]
    cb_keys = ["Q3", "Q4", "P4_1285", "CT1", "CT1_GP", "K066_P4", "GENDER_EDU", "REG_HOUS", "INC_EDU"]
    cb_cats = cb_base + [f"K_{k}" for k in cb_keys]
    def mkcb(Xr, df_s, kd, vd, td=None):
        cs = pd.DataFrame({**{c: df_s[c].fillna("NA").astype(str).reset_index(drop=True) for c in cb_base},
                           **{f"K_{k}": kd[k][0].astype(str).reset_index(drop=True) for k in cb_keys}})
        pts = [Xr.drop(columns=v28.CATS).reset_index(drop=True), vd.reset_index(drop=True)]
        if td is not None: pts.append(td.reset_index(drop=True))
        pts.append(cs)
        return pd.concat(pts, axis=1)

    mc1 = CatBoostClassifier(iterations=850, depth=7, learning_rate=0.04, l2_leaf_reg=10.0, border_count=128,
                             max_ctr_complexity=2, loss_function="Logloss", random_seed=42 + fi, task_type="GPU", verbose=False)
    mc1.fit(mkcb(Xia_r, df_ia, k_ia, v_ia), y_ia, cat_features=cb_cats, verbose=False)
    oof["CB_D7_NoTE"][ib] = mc1.predict_proba(mkcb(Xib_r, df_ib, k_ib, v_ib))[:, 1]

    mc2 = CatBoostClassifier(iterations=950, depth=6, learning_rate=0.035, l2_leaf_reg=15.0, border_count=254,
                             max_ctr_complexity=2, loss_function="Logloss", random_seed=42 + fi, task_type="GPU", verbose=False)
    mc2.fit(mkcb(Xia_r, df_ia, k_ia, v_ia, te_ia), y_ia, cat_features=cb_cats, verbose=False)
    oof["CB_D6_B254_TE"][ib] = mc2.predict_proba(mkcb(Xib_r, df_ib, k_ib, v_ib, te_ib))[:, 1]
    print(f"Dev Fold {fi+1}/3 done ({time.time()-t0:.1f}s)")

print("\n=== 3-FOLD POOLED INNER-DEV RESULTS (n=136,961) ===")
for k in names:
    corr = np.corrcoef(oof[k], oof["LGB_Mono"])[0, 1]
    print(f"  [{k:16s}] 3-Fold OOF AUC: {roc_auc_score(y_dev, oof[k]):.6f} | Corr w/ LGB_Mono: {corr:.4f}")

def eval_combo(label, keys):
    pm = {k: v28.fit_probit_map(oof[k]) for k in keys}
    Z = np.column_stack([np.interp(oof[k], *pm[k]) for k in keys])
    eq = roc_auc_score(y_dev, Z.mean(axis=1))
    # Smooth Logistic-on-Probit convex stacking (strictly convex OOF estimate here; inside outer CV would nest)
    lr = LogisticRegression(C=0.05, max_iter=500).fit(Z, y_dev)
    w_lr = np.maximum(lr.coef_[0], 0); w_lr /= w_lr.sum()
    auc_lr = roc_auc_score(y_dev, Z @ w_lr)
    print(f"  {label:38s}: EqualProbit={eq:.6f} | LogRegProbit={auc_lr:.6f} | w={np.round(w_lr, 3)}")

eval_combo("Old-Style (LGB+XGB+CB_NoTE+MLP_Old)", ["LGB_Mono", "XGB_D5_B512", "CB_D7_NoTE", "MLP_Old"])
eval_combo("Upgraded-NoTE-CB (CB_NoTE + 2NN)", ["LGB_Mono", "XGB_D5_B512", "CB_D7_NoTE", "MLP_NaN", "Periodic_SwiGLU"])
eval_combo("Upgraded-TE-CB   (CB_+TE   + 2NN)", ["LGB_Mono", "XGB_D5_B512", "CB_D6_B254_TE", "MLP_NaN", "Periodic_SwiGLU"])

