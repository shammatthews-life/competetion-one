import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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

cmaps = v28.fit_category_maps(df_a)
Xa_base = v28.build_row_features(df_a, cmaps)
Xb_base = v28.build_row_features(df_b, cmaps)
res_a, res_b, _ = v28.compute_fold_residuals(df_a, df_b, df_b.iloc[:5], cmaps, seed=42)
Xa_res = pd.concat([Xa_base, res_a], axis=1)
Xb_res = pd.concat([Xb_base, res_b], axis=1)
te_a, te_b, _, _ = v28.compute_local_te(df_a, df_b, y_a, n_inner=5, seed=42)
v_a, v_b = v28.compute_local_views(Xa_res, Xb_res, df_a, df_b, y_a, n_inner=3, seed=42)

Xa_f = pd.concat([Xa_res, te_a, v_a], axis=1)
Xb_f = pd.concat([Xb_res, te_b, v_b], axis=1)

med = Xa_f.median()
xa_raw = np.clip(Xa_f.fillna(med).fillna(0.0).values, -1e6, 1e6).astype(np.float32)
xb_raw = np.clip(Xb_f.fillna(med).fillna(0.0).values, -1e6, 1e6).astype(np.float32)
lo = np.percentile(xa_raw, 0.5, axis=0); hi = np.percentile(xa_raw, 99.5, axis=0)
sc = StandardScaler()
Xa_s = sc.fit_transform(np.clip(xa_raw, lo, hi)).astype(np.float32)
Xb_s = sc.transform(np.clip(xb_raw, lo, hi)).astype(np.float32)

def run_mlp_snapshots(seed):
    torch.manual_seed(seed)
    X_t = torch.from_numpy(Xa_s); y_t = torch.from_numpy(y_a.astype(np.float32))
    net = v28.TabResMLP(Xa_s.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=13)
    crit = nn.BCEWithLogitsLoss()
    bs, n = 4096, len(Xa_s)
    snaps = []
    for ep in range(13):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = crit(net(X_t[idx]), y_t[idx]); loss.backward(); opt.step()
        sched.step()
        if ep >= 10:
            net.eval()
            with torch.no_grad():
                snaps.append(torch.sigmoid(net(torch.from_numpy(Xb_s))).numpy())
    return snaps[-1], np.mean(snaps, axis=0)

t0 = time.time()
p1_last, p1_snap = run_mlp_snapshots(42)
print(f"Seed1: last={roc_auc_score(y_b, p1_last):.6f} | last3_snap={roc_auc_score(y_b, p1_snap):.6f} ({time.time()-t0:.1f}s)")
t1 = time.time()
p2_last, p2_snap = run_mlp_snapshots(1042)
print(f"2-Seed avg last={roc_auc_score(y_b, 0.5*(p1_last+p2_last)):.6f} | 2-Seed avg last3_snap={roc_auc_score(y_b, 0.5*(p1_snap+p2_snap)):.6f} ({time.time()-t1:.1f}s)")

