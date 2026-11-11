import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import solo_champion_v28_certified as v28

v28.set_all_seeds(42)
assert torch.cuda.is_available()
DEV = torch.device("cuda")
print(f"[*] Phase-2 Inner-Dev GPU Research on {torch.cuda.get_device_name(0)}")

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y_train = df_train["TARGET"].values.astype(np.int32)
dev_outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(df_train)), y_train))[0][0]
ia, ib = list(StratifiedKFold(3, shuffle=True, random_state=777).split(np.zeros(len(dev_outer)), y_train[dev_outer]))[0]
df_ia, df_ib = df_train.iloc[dev_outer[ia]], df_train.iloc[dev_outer[ib]]
y_ia, y_ib = y_train[dev_outer[ia]], y_train[dev_outer[ib]]

cmaps = v28.fit_category_maps(df_ia)
Xia_res = pd.concat([v28.build_row_features(df_ia, cmaps), v28.compute_fold_residuals(df_ia, df_ib, None, cmaps, seed=42)[0]], axis=1)
Xib_res = pd.concat([v28.build_row_features(df_ib, cmaps), v28.compute_fold_residuals(df_ia, df_ib, None, cmaps, seed=42)[1]], axis=1)
# Reuse residual computation in 1 call cleanly:
res_ia, res_ib = v28.compute_fold_residuals(df_ia, df_ib, None, cmaps, seed=42)
Xia_res = pd.concat([v28.build_row_features(df_ia, cmaps), res_ia], axis=1)
Xib_res = pd.concat([v28.build_row_features(df_ib, cmaps), res_ib], axis=1)
te_ia, te_ib, k_ia, k_ib = v28.compute_local_te(df_ia, df_ib, y_ia, n_inner=4, seed=42)
v_ia, v_ib = v28.compute_local_views(Xia_res, Xib_res, df_ia, df_ib, y_ia, n_inner=2, seed=42)
Xia_f = pd.concat([Xia_res, te_ia, v_ia], axis=1)
Xib_f = pd.concat([Xib_res, te_ib, v_ib], axis=1)

# --- 1. Neural Architectures with NaN-Mask & Periodic Fourier Embeddings ---
med = Xia_f.median()
miss_ia = Xia_f.isna().values.astype(np.float32)
miss_ib = Xib_f.isna().values.astype(np.float32)
xa_raw = np.clip(Xia_f.fillna(med).fillna(0.0).values, -1e6, 1e6).astype(np.float32)
xb_raw = np.clip(Xib_f.fillna(med).fillna(0.0).values, -1e6, 1e6).astype(np.float32)
lo = np.percentile(xa_raw, 0.5, axis=0); hi = np.percentile(xa_raw, 99.5, axis=0)
sc = StandardScaler()
Xa_s = sc.fit_transform(np.clip(xa_raw, lo, hi)).astype(np.float32)
Xb_s = sc.transform(np.clip(xb_raw, lo, hi)).astype(np.float32)
Xa_m = np.hstack([Xa_s, miss_ia])
Xb_m = np.hstack([Xb_s, miss_ib])

class PeriodicEmbedding(nn.Module):
    def __init__(self, n_num, n_freq=8, sigma=0.05):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_num, n_freq) * sigma)
        self.proj = nn.Linear(n_num * 2 * n_freq, 128)
    def forward(self, x):
        v = 2.0 * np.pi * x.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return self.proj(torch.cat([torch.sin(v), torch.cos(v)], dim=-1).flatten(1))

class TabPeriodicSwiGLU(nn.Module):
    def __init__(self, n_num, n_miss, sigma=0.05, width=256):
        super().__init__()
        self.n_num = n_num
        self.per = PeriodicEmbedding(n_num, n_freq=8, sigma=sigma)
        self.lin = nn.Linear(n_num + n_miss, width - 128)
        self.norm0 = nn.LayerNorm(width)
        self.b1 = nn.ModuleDict({"norm": nn.LayerNorm(width), "w1": nn.Linear(width, width * 2), "w2": nn.Linear(width, width), "drop": nn.Dropout(0.25)})
        self.b2 = nn.ModuleDict({"norm": nn.LayerNorm(width), "w1": nn.Linear(width, width * 2), "w2": nn.Linear(width, width), "drop": nn.Dropout(0.25)})
        self.head = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Dropout(0.20), nn.Linear(width, 1))
    def _glu(self, blk, h):
        x = blk["norm"](h)
        u, v = blk["w1"](x).chunk(2, dim=-1)
        return h + 0.5 * blk["drop"](blk["w2"](torch.nn.functional.silu(u) * v))
    def forward(self, x_full):
        h = self.norm0(torch.cat([self.lin(x_full), self.per(x_full[:, :self.n_num])], dim=-1))
        h = self._glu(self.b1, h)
        h = self._glu(self.b2, h)
        return self.head(h).squeeze(-1)

def run_nn_2seed(cls, x_tr, x_va, epochs=13, lr=3e-3, wd=2e-2, **kw):
    X_t = torch.from_numpy(x_tr).to(DEV)
    y_t = torch.from_numpy(y_ia.astype(np.float32)).to(DEV)
    X_v = torch.from_numpy(x_va).to(DEV)
    acc = np.zeros(len(x_va), dtype=np.float64)
    for sd in (42, 1042):
        torch.manual_seed(sd)
        net = cls(**kw).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.BCEWithLogitsLoss()
        bs, n = 4096, len(x_tr)
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                loss = crit(net(X_t[idx]), y_t[idx])
                loss.backward()
                opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            acc += torch.sigmoid(net(X_v)).cpu().numpy() / 2.0
    del X_t, y_t, X_v; torch.cuda.empty_cache()
    return acc

p_res_old = run_nn_2seed(v28.TabResMLP, Xa_s, Xb_s, epochs=13, lr=3e-3, wd=2e-2, in_dim=Xa_s.shape[1])
p_res_nan = run_nn_2seed(v28.TabResMLP, Xa_m, Xb_m, epochs=13, lr=3e-3, wd=2e-2, in_dim=Xa_m.shape[1])
p_per_s05 = run_nn_2seed(TabPeriodicSwiGLU, Xa_m, Xb_m, epochs=15, lr=2e-3, wd=2.5e-2, n_num=Xa_s.shape[1], n_miss=miss_ia.shape[1], sigma=0.05)
p_per_s12 = run_nn_2seed(TabPeriodicSwiGLU, Xa_m, Xb_m, epochs=15, lr=2e-3, wd=2.5e-2, n_num=Xa_s.shape[1], n_miss=miss_ia.shape[1], sigma=0.12)
print(f"TabResMLP old(2seed): {roc_auc_score(y_ib, p_res_old):.6f} | +NaNMask: {roc_auc_score(y_ib, p_res_nan):.6f}")
print(f"TabPeriodicSwiGLU sigma=0.05: {roc_auc_score(y_ib, p_per_s05):.6f} | sigma=0.12: {roc_auc_score(y_ib, p_per_s12):.6f}")
print(f"50/50 Blend (+NaNMask + s05): {roc_auc_score(y_ib, 0.5*(p_res_nan + p_per_s05)):.6f} | (+NaNMask + s12): {roc_auc_score(y_ib, 0.5*(p_res_nan + p_per_s12)):.6f}")

# --- 2. CatBoost GPU: B128 + TE vs B254 + TE ---
cb_base_cats = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_CONTRACT_TYPE"]
cb_keys = ["Q3", "Q4", "P4_1285", "CT1", "CT1_GP", "K066_P4", "GENDER_EDU", "REG_HOUS", "INC_EDU"]
cb_cats = cb_base_cats + [f"K_{k}" for k in cb_keys]

def make_cb(X_base, df_src, k_dict, v_df, te_df=None):
    c_str = pd.DataFrame({
        **{c: df_src[c].fillna("NA").astype(str).reset_index(drop=True) for c in cb_base_cats},
        **{f"K_{k}": k_dict[k][0].astype(str).reset_index(drop=True) for k in cb_keys}
    })
    parts = [X_base.drop(columns=v28.CATS).reset_index(drop=True), v_df.reset_index(drop=True)]
    if te_df is not None:
        parts.append(te_df.reset_index(drop=True))
    parts.append(c_str)
    return pd.concat(parts, axis=1)

ca_te = make_cb(Xia_res, df_ia, k_ia, v_ia, te_df=te_ia)
cb_te = make_cb(Xib_res, df_ib, k_ib, v_ib, te_df=te_ib)
for nm, bc in [("CB_D6_B128_+TE", 128), ("CB_D6_B254_+TE", 254)]:
    tc = time.time()
    mc = CatBoostClassifier(iterations=950, depth=6, learning_rate=0.035, l2_leaf_reg=15.0, border_count=bc,
                            max_ctr_complexity=2, loss_function="Logloss", random_seed=42, task_type="GPU", verbose=False)
    mc.fit(ca_te, y_ia, cat_features=cb_cats, verbose=False)
    pc = mc.predict_proba(cb_te)[:, 1]
    print(f"[{nm:16s}] Dev AUC: {roc_auc_score(y_ib, pc):.6f} ({time.time()-tc:.1f}s)")

# --- 3. XGBoost CUDA GPU: max_bin=256 vs 512 vs Lossguide ---
Xa_np, Xb_np = Xia_f.astype(np.float32).values, Xib_f.astype(np.float32).values
for nm, xcfg in [
    ("XGB_D5_B256", dict(n_estimators=800, learning_rate=0.028, max_depth=5, subsample=0.8, colsample_bytree=0.40, min_child_weight=60, reg_alpha=3.0, reg_lambda=12.0, max_bin=256)),
    ("XGB_D5_B512", dict(n_estimators=850, learning_rate=0.026, max_depth=5, subsample=0.8, colsample_bytree=0.38, min_child_weight=65, reg_alpha=3.5, reg_lambda=15.0, max_bin=512)),
    ("XGB_Lossguide", dict(n_estimators=900, learning_rate=0.025, max_depth=0, max_leaves=23, grow_policy="lossguide", subsample=0.8, colsample_bytree=0.35, min_child_weight=100, reg_alpha=4.0, reg_lambda=20.0, max_bin=512)),
]:
    tx = time.time()
    mx = xgb.XGBClassifier(**xcfg, tree_method="hist", device="cuda", eval_metric="auc", random_state=42)
    mx.fit(Xa_np, y_ia, verbose=False)
    px = mx.get_booster().inplace_predict(Xb_np)
    print(f"[{nm:16s}] Dev AUC: {roc_auc_score(y_ib, px):.6f} ({time.time()-tx:.1f}s)")

