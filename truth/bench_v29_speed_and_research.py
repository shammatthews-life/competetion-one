import time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y = df_train["TARGET"].values.astype(np.int32)
CATS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]

folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(df_train)), y))
a, b = folds[0] # Test strictly on Outer Fold 1 (fit ONLY on `a`, evaluate untouched `b`)

from solo_champion_v28_certified import fit_category_maps, build_row_features, compute_strict_te_folds

# 1. Pure fold-local category maps & row features (fit ONLY on a!)
t0 = time.time()
cmaps = fit_category_maps(df_train.iloc[a])
Xa_base = build_row_features(df_train.iloc[a], cmaps).reset_index(drop=True)
Xb_base = build_row_features(df_train.iloc[b], cmaps).reset_index(drop=True)

# 2. Benchmark Fold-Local Residuals: HGBR vs LGBMRegressor (fit strictly inside a!)
def make_fold_res(use_lgb=False):
    tr = time.time()
    def prep(df):
        r = df.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore").copy()
        for c in CATS: r[c] = df[c].astype(str).map(cmaps[c]).astype(float)
        r["DAYS_EMPLOYED"] = r["DAYS_EMPLOYED"].where(r["DAYS_EMPLOYED"] != 365243)
        return r
    ra, rb = prep(df_train.iloc[a]).reset_index(drop=True), prep(df_train.iloc[b]).reset_index(drop=True)
    demo = [c for c in ra.columns if c.startswith("EXT_SOURCE") or c.startswith("AMT_")]
    T_spec = {
        "EXT1": (ra["EXT_SOURCE_1"], rb["EXT_SOURCE_1"], ["EXT_SOURCE_1"]),
        "EXT2": (ra["EXT_SOURCE_2"], rb["EXT_SOURCE_2"], ["EXT_SOURCE_2"]),
        "EXT3": (ra["EXT_SOURCE_3"], rb["EXT_SOURCE_3"], ["EXT_SOURCE_3"]),
        "LOGINC": (np.log(ra["AMT_INCOME_TOTAL"]), np.log(rb["AMT_INCOME_TOTAL"]), ["AMT_INCOME_TOTAL"]),
        "LOGCR": (np.log(ra["AMT_CREDIT"]), np.log(rb["AMT_CREDIT"]), ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]),
        "EMP": (ra["DAYS_EMPLOYED"], rb["DAYS_EMPLOYED"], ["DAYS_EMPLOYED"]),
        "REG": (ra["DAYS_REGISTRATION"], rb["DAYS_REGISTRATION"], ["DAYS_REGISTRATION"]),
        "AGE": (ra["DAYS_BIRTH"], rb["DAYS_BIRTH"], ["DAYS_BIRTH"]),
        "IDP": (ra["DAYS_ID_PUBLISH"], rb["DAYS_ID_PUBLISH"], ["DAYS_ID_PUBLISH"]),
        "PHONE": (ra["DAYS_LAST_PHONE_CHANGE"], rb["DAYS_LAST_PHONE_CHANGE"], ["DAYS_LAST_PHONE_CHANGE"]),
        "EXT1D": (ra["EXT_SOURCE_1"], rb["EXT_SOURCE_1"], demo),
        "EXT2D": (ra["EXT_SOURCE_2"], rb["EXT_SOURCE_2"], demo),
        "EXT3D": (ra["EXT_SOURCE_3"], rb["EXT_SOURCE_3"], demo),
    }
    res_a, res_b = {}, {}
    kf3 = list(KFold(3, shuffle=True, random_state=SEED).split(ra))
    for nm, (yta, ytb, dcols) in T_spec.items():
        Xpa, Xpb = ra.drop(columns=dcols), rb.drop(columns=dcols)
        pa = np.zeros(len(ra)); pbs = []
        obsa, obsb = yta.notna().values, ytb.notna().values
        for tri, vai in kf3:
            if use_lgb:
                m = lgb.LGBMRegressor(n_estimators=70, max_depth=5, num_leaves=23, learning_rate=0.08,
                                      subsample=0.8, colsample_bytree=0.7, min_child_samples=80, n_jobs=8, random_state=SEED, verbose=-1)
            else:
                m = HistGradientBoostingRegressor(max_iter=60, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, random_state=SEED)
            m.fit(Xpa.iloc[tri][obsa[tri]], yta.iloc[tri][obsa[tri]])
            pa[vai] = m.predict(Xpa.iloc[vai]); pbs.append(m.predict(Xpb))
        pb = np.mean(pbs, axis=0)
        res_a[f"RP_{nm}"] = pa; res_a[f"RR_{nm}"] = np.where(obsa, yta.values - pa, np.nan)
        res_b[f"RP_{nm}"] = pb; res_b[f"RR_{nm}"] = np.where(obsb, ytb.values - pb, np.nan)
    return pd.DataFrame(res_a, dtype=np.float32), pd.DataFrame(res_b, dtype=np.float32), time.time()-tr

ra_hgbr, rb_hgbr, dt_hgbr = make_fold_res(use_lgb=False)
ra_lgb,  rb_lgb,  dt_lgb  = make_fold_res(use_lgb=True)
print(f"Residuals timing (1 outer fold): HGBR={dt_hgbr:.1f}s vs LGBMRegressor={dt_lgb:.1f}s", flush=True)

# Precompute TE for fold 0 cleanly inside a!
te_tr, te_va, _, k_tr, _ = compute_strict_te_folds(df_train, df_train.iloc[:1], y, folds)
te_a, te_b = te_tr[0], te_va[0]

# Test downstream LightGBM AUC (HGBR res vs LGB res, and max_bin=255 vs 511)
for rname, (ra_d, rb_d) in [("HGBR_res", (ra_hgbr, rb_hgbr)), ("LGBM_res", (ra_lgb, rb_lgb))]:
    for mb in [255, 511]:
        t1 = time.time()
        Xa = pd.concat([Xa_base, ra_d, te_a], axis=1)
        Xb = pd.concat([Xb_base, rb_d, te_b], axis=1)
        p = dict(objective="binary", learning_rate=0.03, num_leaves=23, min_child_samples=150,
                 feature_fraction=0.35, bagging_fraction=0.8, bagging_freq=1, lambda_l2=30,
                 extra_trees=True, max_bin=mb, verbose=-1, n_jobs=8, seed=SEED)
        m = lgb.train(p, lgb.Dataset(Xa, label=y[a], free_raw_data=False), num_boost_round=800)
        auc = roc_auc_score(y[b], m.predict(Xb))
        print(f"LGB_ET ({rname}, max_bin={mb}): Fold1 Val AUC = {auc:.6f} ({time.time()-t1:.1f}s)", flush=True)

# 3. Benchmark Vectorized PyTorch TabResMLP: StandardScaler vs RankGauss QuantileTransformer + Missingness Mask!
class TabResMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.25))
        self.b1 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.25),
                                nn.Linear(256, 256), nn.LayerNorm(256))
        self.b2 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.25),
                                nn.Linear(256, 256), nn.LayerNorm(256))
        self.head = nn.Sequential(nn.SiLU(), nn.Dropout(0.20), nn.Linear(256, 1))
    def forward(self, x):
        h = self.inp(x)
        h = h + 0.5 * self.b1(h)
        h = h + 0.5 * self.b2(h)
        return self.head(h).squeeze(-1)

Xa_num = pd.concat([Xa_base, ra_lgb, te_a], axis=1)
Xb_num = pd.concat([Xb_base, rb_lgb, te_b], axis=1)

def run_fast_mlp(use_rank_gauss=False):
    t1 = time.time()
    med = Xa_num.median()
    if use_rank_gauss:
        # Missing indicators for columns with nulls + RankGauss strictly fit on a!
        na_cols = [c for c in Xa_num.columns if Xa_num[c].isna().any()]
        ma = Xa_num[na_cols].isna().astype(np.float32).values
        mb = Xb_num[na_cols].isna().astype(np.float32).values
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED)
        xa_g = qt.fit_transform(Xa_num.fillna(med).values).astype(np.float32)
        xb_g = qt.transform(Xb_num.fillna(med).values).astype(np.float32)
        Xa_s = np.hstack([xa_g, ma]); Xb_s = np.hstack([xb_g, mb])
    else:
        lo = np.percentile(np.clip(Xa_num.fillna(med).values, -1e6, 1e6), 0.5, axis=0)
        hi = np.percentile(np.clip(Xa_num.fillna(med).values, -1e6, 1e6), 99.5, axis=0)
        sc = StandardScaler()
        Xa_s = sc.fit_transform(np.clip(Xa_num.fillna(med).values, lo, hi)).astype(np.float32)
        Xb_s = sc.transform(np.clip(Xb_num.fillna(med).values, lo, hi)).astype(np.float32)

    X_t = torch.from_numpy(Xa_s); y_t = torch.from_numpy(y[a].astype(np.float32))
    X_v = torch.from_numpy(Xb_s)
    torch.manual_seed(SEED)
    net = TabResMLP(Xa_s.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=14)
    crit = nn.BCEWithLogitsLoss()
    bs = 2048; n = len(Xa_s)
    for _ in range(14):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad(); loss = crit(net(X_t[idx]), y_t[idx]); loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        pv = torch.sigmoid(net(X_v)).numpy()
    print(f"Vectorized MLP (RankGauss={use_rank_gauss}): Fold1 Val AUC = {roc_auc_score(y[b], pv):.6f} ({time.time()-t1:.1f}s)", flush=True)

run_fast_mlp(False)
run_fast_mlp(True)

# 4. Benchmark CatBoost GPU speed & AUC with max_ctr_complexity=1 vs 2
cb_keys = ["Q3", "Q4", "P3", "P4_1285", "CT1", "CT2", "CT1_GP", "K066", "K066_P4", "GENDER_EDU", "REG_HOUS", "INC_EDU"]
cb_df_a = pd.concat([Xa_base.drop(columns=CATS), ra_lgb, pd.DataFrame({
    **{c: df_train.iloc[a][c].fillna("NA").astype(str).reset_index(drop=True) for c in CATS},
    **{f"K_{k}": k_tr[k][0].iloc[a].astype(str).reset_index(drop=True) for k in cb_keys}
})], axis=1)
cb_df_b = pd.concat([Xb_base.drop(columns=CATS), rb_lgb, pd.DataFrame({
    **{c: df_train.iloc[b][c].fillna("NA").astype(str).reset_index(drop=True) for c in CATS},
    **{f"K_{k}": k_tr[k][0].iloc[b].astype(str).reset_index(drop=True) for k in cb_keys}
})], axis=1)
cb_cats = CATS + [f"K_{k}" for k in cb_keys]

for ctr in [1, 2]:
    t1 = time.time()
    mc = CatBoostClassifier(iterations=850, depth=7, learning_rate=0.04, l2_leaf_reg=10.0,
                            max_ctr_complexity=ctr, loss_function="Logloss", random_seed=SEED, task_type="GPU", verbose=False)
    mc.fit(cb_df_a, y[a], cat_features=cb_cats, verbose=False)
    pv = mc.predict_proba(cb_df_b)[:, 1]
    print(f"CB_GPU_D7 (ctr={ctr}, 850r): Fold1 Val AUC = {roc_auc_score(y[b], pv):.6f} ({time.time()-t1:.1f}s)", flush=True)

