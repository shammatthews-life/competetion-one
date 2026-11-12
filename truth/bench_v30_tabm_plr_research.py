import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import solo_champion_v28_certified as v28

v28.set_all_seeds(42)
assert torch.cuda.is_available()
DEV = torch.device("cuda")
print(f"[*] V30 TabM + PLR + RankGauss Inner-Dev GPU Research on {torch.cuda.get_device_name(0)}")

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y_train = df_train["TARGET"].values.astype(np.int32)
dev_outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(df_train)), y_train))[0][0]
ia, ib = list(StratifiedKFold(3, shuffle=True, random_state=777).split(np.zeros(len(dev_outer)), y_train[dev_outer]))[0]
df_ia, df_ib = df_train.iloc[dev_outer[ia]], df_train.iloc[dev_outer[ib]]
y_ia, y_ib = y_train[dev_outer[ia]], y_train[dev_outer[ib]]

cmaps = v28.fit_category_maps(df_ia)
res_ia, res_ib = v28.compute_fold_residuals(df_ia, df_ib, None, cmaps, seed=42)
Xia_res = pd.concat([v28.build_row_features(df_ia, cmaps), res_ia], axis=1)
Xib_res = pd.concat([v28.build_row_features(df_ib, cmaps), res_ib], axis=1)
te_ia, te_ib, k_ia, k_ib = v28.compute_local_te(df_ia, df_ib, y_ia, n_inner=4, seed=42)
v_ia, v_ib = v28.compute_local_views(Xia_res, Xib_res, df_ia, df_ib, y_ia, n_inner=2, seed=42)
Xia_f = pd.concat([Xia_res, te_ia, v_ia], axis=1)
Xib_f = pd.concat([Xib_res, te_ib, v_ib], axis=1)

# --- 1. Reference LightGBM anchor on exact same split ---
p_lgb = lgb.LGBMClassifier(
    n_estimators=1250, learning_rate=0.022, num_leaves=23, min_child_samples=160,
    subsample_for_bin=100000, colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
    reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=42, verbose=-1
).fit(Xia_f, y_ia).predict_proba(Xib_f)[:, 1]
print(f"[Ref LGB_ET_L23_6View] AUC = {roc_auc_score(y_ib, p_lgb):.6f}")

# --- 2. Baseline V28 Neural Pair ---
p_res, p_per = v28.fit_predict_neural_pair(Xia_f, Xib_f, y_ia, Xte_df=None, seed=42)
print(f"[Base TabResMLP_NaNMask_GPU]  AUC = {roc_auc_score(y_ib, p_res):.6f}")
print(f"[Base TabPeriodicSwiGLU_GPU] AUC = {roc_auc_score(y_ib, p_per):.6f}")

# --- 3. Preprocessing: StandardScaler + RankGauss + Quantile PLR Edges (fit strictly on Xia_f) ---
med = Xia_f.median()
miss_ia = Xia_f.isna().values.astype(np.float32)
miss_ib = Xib_f.isna().values.astype(np.float32)
xa_raw = np.clip(Xia_f.fillna(med).fillna(0.0).values, -1e6, 1e6).astype(np.float32)
xb_raw = np.clip(Xib_f.fillna(med).fillna(0.0).values, -1e6, 1e6).astype(np.float32)
lo = np.percentile(xa_raw, 0.5, axis=0)
hi = np.percentile(xa_raw, 99.5, axis=0)
xa_c = np.clip(xa_raw, lo, hi)
xb_c = np.clip(xb_raw, lo, hi)

sc = StandardScaler()
Xa_s = sc.fit_transform(xa_c).astype(np.float32)
Xb_s = sc.transform(xb_c).astype(np.float32)

qt = QuantileTransformer(n_quantiles=512, output_distribution="normal", random_state=42, subsample=100000)
Xa_rg = qt.fit_transform(xa_c).astype(np.float32)
Xb_rg = qt.transform(xb_c).astype(np.float32)

# Quantile PLR bin edges (n_bins=24, strictly on xa_c)
N_BINS = 24
q_grid = np.linspace(0.0, 100.0, N_BINS + 1)
edges = np.percentile(xa_c, q_grid, axis=0).T  # (n_num, N_BINS+1)
widths = np.maximum(edges[:, 1:] - edges[:, :-1], 1e-6)  # (n_num, N_BINS)
edges_t = torch.tensor(edges[:, :-1], dtype=torch.float32, device=DEV)
widths_t = torch.tensor(widths, dtype=torch.float32, device=DEV)

# --- 4. ICLR 2025 TabM (BatchEnsemble k=16) + Piecewise Linear Embeddings (PLR) ---
class BatchEnsembleLinear(nn.Module):
    def __init__(self, d_in, d_out, k=16):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(d_in, d_out)
        self.r = nn.Parameter(torch.empty(k, d_in).normal_(1.0, 0.10))
        self.s = nn.Parameter(torch.empty(k, d_out).normal_(1.0, 0.10))

    def forward(self, x):
        # x: (B, k, d_in) -> out: (B, k, d_out)
        xr = x * self.r.unsqueeze(0)
        y = F.linear(xr, self.linear.weight, None)
        return y * self.s.unsqueeze(0) + self.linear.bias

class TabM_PLR(nn.Module):
    def __init__(self, n_num, n_miss, edges_lo, widths, d_plr=12, width=256, k=16):
        super().__init__()
        self.k = k
        self.register_buffer("edges_lo", edges_lo)
        self.register_buffer("widths", widths)
        n_bins = widths.shape[1]
        self.plr_w = nn.Parameter(torch.randn(n_num, n_bins, d_plr) * 0.08)
        self.plr_b = nn.Parameter(torch.zeros(n_num, d_plr))
        d_in = n_num * d_plr + n_miss
        self.in_proj = BatchEnsembleLinear(d_in, width, k=k)
        self.b1 = nn.LayerNorm(width)
        self.fc1 = BatchEnsembleLinear(width, width, k=k)
        self.b2 = nn.LayerNorm(width)
        self.fc2 = BatchEnsembleLinear(width, width, k=k)
        self.head = BatchEnsembleLinear(width, 1, k=k)

    def forward(self, x_num, x_miss):
        # PLR encoding: (B, n_num, n_bins)
        enc = torch.clamp((x_num.unsqueeze(-1) - self.edges_lo.unsqueeze(0)) / self.widths.unsqueeze(0), 0.0, 1.0)
        emb = F.silu(torch.einsum("bnc,ncd->bnd", enc, self.plr_w) + self.plr_b.unsqueeze(0)).flatten(1)
        x = torch.cat([emb, x_miss], dim=1)
        h = F.silu(self.in_proj(x.unsqueeze(1).expand(-1, self.k, -1)))
        h = h + F.dropout(F.silu(self.fc1(self.b1(h))), p=0.22, training=self.training)
        h = h + F.dropout(F.silu(self.fc2(self.b2(h))), p=0.22, training=self.training)
        return self.head(h).squeeze(-1)  # (B, k)

def train_tabm_plr(xa_num, xa_m, xb_num, xb_m, y_tr, seeds=(42, 1042), epochs=14):
    t0 = time.time()
    n_num, n_miss = xa_num.shape[1], xa_m.shape[1]
    xt_n = torch.tensor(xa_num, dtype=torch.float32)
    xt_m = torch.tensor(xa_m, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    xb_nt = torch.tensor(xb_num, dtype=torch.float32, device=DEV)
    xb_mt = torch.tensor(xb_m, dtype=torch.float32, device=DEV)
    ds = torch.utils.data.TensorDataset(xt_n, xt_m, yt)

    preds = np.zeros(len(xb_num))
    for s in seeds:
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(s))
        model = TabM_PLR(n_num, n_miss, edges_t, widths_t, d_plr=12, width=256, k=16).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            model.train()
            for bn, bm, by in dl:
                bn, bm, by = bn.to(DEV), bm.to(DEV), by.to(DEV)
                opt.zero_grad()
                logits = model(bn, bm)  # (B, k)
                loss = loss_fn(logits, by.unsqueeze(1).expand_as(logits))
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            p_k = []
            for st in range(0, len(xb_num), 4096):
                p_k.append(torch.sigmoid(model(xb_nt[st:st+4096], xb_mt[st:st+4096])).mean(dim=1).cpu().numpy())
            preds += np.concatenate(p_k) / float(len(seeds))
    print(f"  -> TabM_PLR_GPU finished in {time.time()-t0:.1f}s")
    return preds

p_tabm = train_tabm_plr(xa_c, miss_ia, xb_c, miss_ib, y_ia)
print(f"[New TabM_PLR_GPU (k=16)]    AUC = {roc_auc_score(y_ib, p_tabm):.6f}")

# --- 5. RankGauss + TabResMLP ---
Xa_rgm = np.hstack([Xa_rg, miss_ia])
Xb_rgm = np.hstack([Xb_rg, miss_ib])

def train_resmlp(xa_in, xb_in, y_tr, seeds=(42, 1042), epochs=14):
    t0 = time.time()
    xt = torch.tensor(xa_in, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    xbt = torch.tensor(xb_in, dtype=torch.float32, device=DEV)
    ds = torch.utils.data.TensorDataset(xt, yt)
    preds = np.zeros(len(xb_in))
    for s in seeds:
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(s))
        model = v28.TabResMLP(xa_in.shape[1]).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=3e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            model.train()
            for bx, by in dl:
                bx, by = bx.to(DEV), by.to(DEV)
                opt.zero_grad(); loss_fn(model(bx), by).backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            preds += np.concatenate([torch.sigmoid(model(xbt[st:st+4096])).cpu().numpy() for st in range(0, len(xb_in), 4096)]) / float(len(seeds))
    print(f"  -> RankGauss_TabResMLP finished in {time.time()-t0:.1f}s")
    return preds

p_rg = train_resmlp(Xa_rgm, Xb_rgm, y_ia)
print(f"[New RankGauss_TabResMLP]    AUC = {roc_auc_score(y_ib, p_rg):.6f}")

# --- 6. Probit Blend Comparison (LGB + Neural) ---
def to_z(p):
    r = pd.Series(p).rank(method="average").values / (len(p) + 1.0)
    return norm.ppf(np.clip(r, 1e-6, 1.0 - 1e-6))

z_lgb, z_res, z_per, z_tabm, z_rg = map(to_z, (p_lgb, p_res, p_per, p_tabm, p_rg))
print(f"\n[Blend: LGB + Base(ResMLP, Periodic)]          = {roc_auc_score(y_ib, 0.50*z_lgb + 0.25*z_res + 0.25*z_per):.6f}")
print(f"[Blend: LGB + Base + TabM_PLR]                 = {roc_auc_score(y_ib, 0.45*z_lgb + 0.15*z_res + 0.15*z_per + 0.25*z_tabm):.6f}")
print(f"[Blend: LGB + RankGauss_ResMLP + TabM_PLR]     = {roc_auc_score(y_ib, 0.45*z_lgb + 0.20*z_rg + 0.35*z_tabm):.6f}")
print(f"[Blend: LGB + RankGauss + Periodic + TabM_PLR] = {roc_auc_score(y_ib, 0.45*z_lgb + 0.15*z_rg + 0.15*z_per + 0.25*z_tabm):.6f}")

