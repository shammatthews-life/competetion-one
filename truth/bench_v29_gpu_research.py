import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import solo_champion_v28_certified as v28

v28.set_all_seeds(42)
assert torch.cuda.is_available(), "CUDA PyTorch required for GPU research!"
DEV = torch.device("cuda")
print(f"[*] Running Strictly Inner-Dev Research on {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__})")

# Strict Dev Split: use ONLY `a_dev` (SEED_DEV=42 fold 0 train), split internally into (ia, ib).
# NEVER touches outer `b` or evaluation split `SEED_EVAL=2026`.
df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y_train = df_train["TARGET"].values.astype(np.int32)
dev_outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(df_train)), y_train))[0][0]
ia, ib = list(StratifiedKFold(3, shuffle=True, random_state=777).split(np.zeros(len(dev_outer)), y_train[dev_outer]))[0]
df_ia, df_ib = df_train.iloc[dev_outer[ia]], df_train.iloc[dev_outer[ib]]
y_ia, y_ib = y_train[dev_outer[ia]], y_train[dev_outer[ib]]
print(f"[*] Inner-Dev Split: Fit={len(df_ia):,}, Dev-Val={len(df_ib):,}")

t0 = time.time()
cmaps = v28.fit_category_maps(df_ia)
Xia_base = v28.build_row_features(df_ia, cmaps)
Xib_base = v28.build_row_features(df_ib, cmaps)
res_ia, res_ib = v28.compute_fold_residuals(df_ia, df_ib, None, cmaps, seed=42)
Xia_res = pd.concat([Xia_base, res_ia], axis=1)
Xib_res = pd.concat([Xib_base, res_ib], axis=1)
te_ia, te_ib, k_ia, k_ib = v28.compute_local_te(df_ia, df_ib, y_ia, n_inner=4, seed=42)
v_ia, v_ib = v28.compute_local_views(Xia_res, Xib_res, df_ia, df_ib, y_ia, n_inner=2, seed=42)
Xia_f = pd.concat([Xia_res, te_ia, v_ia], axis=1)
Xib_f = pd.concat([Xib_res, te_ib, v_ib], axis=1)
print(f"[*] Inner-Dev Feature Prep Complete: {Xia_f.shape} ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------------------------
# RESEARCH 1: CUDA TabResMLP vs Periodic Fourier + Missingness + SwiGLU ResNet
# ------------------------------------------------------------------------------
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
    def __init__(self, n_num, n_freq=8, sigma=0.15):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_num, n_freq) * sigma)
        self.proj = nn.Linear(n_num * 2 * n_freq, 128)
    def forward(self, x):
        v = 2.0 * np.pi * x.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return self.proj(torch.cat([torch.sin(v), torch.cos(v)], dim=-1).flatten(1))

class TabPeriodicSwiGLU(nn.Module):
    def __init__(self, n_num, n_miss, width=256):
        super().__init__()
        self.n_num = n_num
        self.per = PeriodicEmbedding(n_num, n_freq=8, sigma=0.12)
        self.lin = nn.Linear(n_num + n_miss, width - 128)
        self.norm0 = nn.LayerNorm(width)
        self.b1 = self._glu_block(width)
        self.b2 = self._glu_block(width)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Dropout(0.20), nn.Linear(width, 1))
    def _glu_block(self, d):
        return nn.ModuleDict({
            "norm": nn.LayerNorm(d),
            "w1": nn.Linear(d, d * 2),
            "w2": nn.Linear(d, d),
            "drop": nn.Dropout(0.25),
        })
    def _run_glu(self, blk, h):
        x = blk["norm"](h)
        u, v = blk["w1"](x).chunk(2, dim=-1)
        return h + 0.5 * blk["drop"](blk["w2"](torch.nn.functional.silu(u) * v))
    def forward(self, x_full):
        x_num = x_full[:, :self.n_num]
        h = self.norm0(torch.cat([self.lin(x_full), self.per(x_num)], dim=-1))
        h = self._run_glu(self.b1, h)
        h = self._run_glu(self.b2, h)
        return self.head(h).squeeze(-1)

def run_nn_gpu(cls, x_tr, x_va, epochs=15, lr=2.5e-3, wd=2e-2, seed=42, **kw):
    t1 = time.time()
    torch.manual_seed(seed)
    X_t = torch.from_numpy(x_tr).to(DEV)
    y_t = torch.from_numpy(y_ia.astype(np.float32)).to(DEV)
    X_v = torch.from_numpy(x_va).to(DEV)
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
        pv = torch.sigmoid(net(X_v)).cpu().numpy()
    return pv, time.time() - t1

p_mlp1, dt1 = run_nn_gpu(v28.TabResMLP, Xa_s, Xb_s, epochs=13, lr=3e-3, wd=2e-2, seed=42, in_dim=Xa_s.shape[1])
p_mlp2, dt2 = run_nn_gpu(v28.TabResMLP, Xa_s, Xb_s, epochs=13, lr=3e-3, wd=2e-2, seed=1042, in_dim=Xa_s.shape[1])
print(f"[NN-1] TabResMLP GPU 1-seed: {roc_auc_score(y_ib, p_mlp1):.6f} ({dt1:.2f}s!) | 2-seed: {roc_auc_score(y_ib, 0.5*(p_mlp1+p_mlp2)):.6f}")

p_mlp_m, dtm = run_nn_gpu(v28.TabResMLP, Xa_m, Xb_m, epochs=13, lr=3e-3, wd=2e-2, seed=42, in_dim=Xa_m.shape[1])
print(f"[NN-2] TabResMLP + NaN-Mask GPU 1-seed: {roc_auc_score(y_ib, p_mlp_m):.6f} ({dtm:.2f}s)")

p_per1, dtp1 = run_nn_gpu(TabPeriodicSwiGLU, Xa_m, Xb_m, epochs=15, lr=2e-3, wd=2.5e-2, seed=42, n_num=Xa_s.shape[1], n_miss=miss_ia.shape[1])
p_per2, dtp2 = run_nn_gpu(TabPeriodicSwiGLU, Xa_m, Xb_m, epochs=15, lr=2e-3, wd=2.5e-2, seed=1042, n_num=Xa_s.shape[1], n_miss=miss_ia.shape[1])
print(f"[NN-3] TabPeriodicSwiGLU GPU 1-seed: {roc_auc_score(y_ib, p_per1):.6f} ({dtp1:.2f}s) | 2-seed: {roc_auc_score(y_ib, 0.5*(p_per1+p_per2)):.6f}")
print(f"[NN-Blend] 50% TabResMLP + 50% TabPeriodicSwiGLU: {roc_auc_score(y_ib, 0.25*(p_mlp1+p_mlp2+p_per1+p_per2)):.6f}")

# ------------------------------------------------------------------------------
# RESEARCH 2: CatBoost GPU border_count=254 & depth=6 vs 7 + TE inclusion
# ------------------------------------------------------------------------------
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

ca_0 = make_cb(Xia_res, df_ia, k_ia, v_ia)
cb_0 = make_cb(Xib_res, df_ib, k_ib, v_ib)
for name, cfg in [
    ("CB_D7_B128_base", dict(iterations=800, depth=7, learning_rate=0.04, l2_leaf_reg=10.0, border_count=128)),
    ("CB_D7_B254", dict(iterations=850, depth=7, learning_rate=0.04, l2_leaf_reg=12.0, border_count=254)),
    ("CB_D6_B254_1100", dict(iterations=1100, depth=6, learning_rate=0.035, l2_leaf_reg=15.0, border_count=254)),
]:
    tc = time.time()
    mc = CatBoostClassifier(**cfg, max_ctr_complexity=2, loss_function="Logloss", random_seed=42, task_type="GPU", verbose=False)
    mc.fit(ca_0, y_ia, cat_features=cb_cats, verbose=False)
    pc = mc.predict_proba(cb_0)[:, 1]
    print(f"[{name:16s}] Dev AUC: {roc_auc_score(y_ib, pc):.6f} ({time.time()-tc:.1f}s)")

tc = time.time()
ca_te = make_cb(Xia_res, df_ia, k_ia, v_ia, te_df=te_ia)
cb_te = make_cb(Xib_res, df_ib, k_ib, v_ib, te_df=te_ib)
mc_te = CatBoostClassifier(iterations=950, depth=6, learning_rate=0.035, l2_leaf_reg=15.0, border_count=254,
                           max_ctr_complexity=2, loss_function="Logloss", random_seed=42, task_type="GPU", verbose=False)
mc_te.fit(ca_te, y_ia, cat_features=cb_cats, verbose=False)
pc_te = mc_te.predict_proba(cb_te)[:, 1]
print(f"[CB_D6_B254_+TE   ] Dev AUC: {roc_auc_score(y_ib, pc_te):.6f} ({time.time()-tc:.1f}s)")

# ------------------------------------------------------------------------------
# RESEARCH 3: XGBoost CUDA GPU max_bin=256 vs 512 + lossguide + inplace_predict
# ------------------------------------------------------------------------------
Xa_x = torch.from_numpy(Xia_f.astype(np.float32).values).to(DEV)
Xb_x = torch.from_numpy(Xib_f.astype(np.float32).values).to(DEV)
y_xg = torch.from_numpy(y_ia).to(DEV)
for name, xcfg in [
    ("XGB_D5_B256", dict(n_estimators=800, learning_rate=0.028, max_depth=5, subsample=0.8, colsample_bytree=0.40, min_child_weight=60, reg_alpha=3.0, reg_lambda=12.0, max_bin=256)),
    ("XGB_D5_B512", dict(n_estimators=850, learning_rate=0.026, max_depth=5, subsample=0.8, colsample_bytree=0.38, min_child_weight=65, reg_alpha=3.5, reg_lambda=15.0, max_bin=512)),
    ("XGB_Lossguide", dict(n_estimators=900, learning_rate=0.025, max_depth=0, max_leaves=23, grow_policy="lossguide", subsample=0.8, colsample_bytree=0.35, min_child_weight=100, reg_alpha=4.0, reg_lambda=20.0, max_bin=512)),
]:
    tx = time.time()
    mx = xgb.XGBClassifier(**xcfg, tree_method="hist", device="cuda", eval_metric="auc", random_state=42)
    mx.fit(Xa_x, y_xg, verbose=False)
    px = mx.get_booster().inplace_predict(xgb.DMatrix(Xib_f.astype(np.float32))) if False else mx.predict_proba(Xb_x)[:, 1]
    print(f"[{name:16s}] Dev AUC: {roc_auc_score(y_ib, px):.6f} ({time.time()-tx:.1f}s)")

