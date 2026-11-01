import os, sys, time, gc, warnings, pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

warnings.filterwarnings('ignore')

WORK_DIR = 'd:/gci/gemini'
CACHE_DIR = os.path.join(WORK_DIR, 'cache')

print("==================================================")
print(" GCI World 2026 - EXP_13: Complete 4-Set LightGBM Ensemble (old + e1 + e1et + e1Aet)")
print("==================================================")
t0 = time.time()

TRAIN_PATH = 'd:/gci/inpuy/train.csv'
TEST_PATH = 'd:/gci/inpuy/test.csv'
SAMPLE_PATH = 'd:/gci/inpuy/sample_submission.csv'

tr = pd.read_csv(TRAIN_PATH)
te = pd.read_csv(TEST_PATH)
y = tr["TARGET"].values
ntr = len(tr)
nte = len(te)
n = ntr
FOLDS = 5
folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y))

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)

# 1. Load Base Features and Residuals
print("Loading base features, residuals, and view caches...")
Xc = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xc.feather'))
Xct = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xct.feather'))
Xa = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xa.feather'))
Xat = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xat.feather'))
Xck = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xck.feather'))
Xckt = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xckt.feather'))
Xak = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xak.feather'))
Xakt = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xakt.feather'))

resid = pd.read_pickle('D:/gci/codex/v_3_cache/resid16.pkl')
R_tr = resid.iloc[:ntr].reset_index(drop=True)
R_te = resid.iloc[ntr:].reset_index(drop=True)

# Frames WITHOUT residuals (for old)
BASEF = {
    "Xc": Xc, "Xct": Xct, "Xa": Xa, "Xat": Xat,
    "Xck": Xck, "Xckt": Xckt, "Xak": Xak, "Xakt": Xakt
}

# Frames WITH residuals (for e1 and e1et)
F1 = {
    "Xc": pd.concat([Xc, R_tr], axis=1),
    "Xct": pd.concat([Xct, R_te], axis=1),
    "Xa": pd.concat([Xa, R_tr], axis=1),
    "Xat": pd.concat([Xat, R_te], axis=1),
    "Xck": pd.concat([Xck, R_tr], axis=1),
    "Xckt": pd.concat([Xckt, R_te], axis=1),
    "Xak": pd.concat([Xak, R_tr], axis=1),
    "Xakt": pd.concat([Xakt, R_te], axis=1)
}

# Load original views for e1 and e1et
VIEW_O = pd.read_pickle(os.path.join(CACHE_DIR, 'e1_views.pkl'))

CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","ORGANIZATION_TYPE"]

def onehot(Xa_, Xt_):
    z = pd.concat([Xa_, Xt_], ignore_index=True)
    oh = pd.get_dummies(z[CATS].astype("category"), dummy_na=True, dtype=np.float32)
    oh.columns = [f"OH_{i}" for i in range(oh.shape[1])]
    Z = pd.concat([z.drop(columns=CATS), oh], axis=1)
    return Z.iloc[:len(Xa_)].reset_index(drop=True), Z.iloc[len(Xa_):].reset_index(drop=True)

def build_specs(frames, extra_trees=False):
    Xco, Xcot = onehot(frames["Xc"], frames["Xct"])
    Xao, Xaot = onehot(frames["Xa"], frames["Xat"])
    Xaok, Xaokt = onehot(frames["Xak"], frames["Xakt"])
    mono_k = [-1 if c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") else 0 for c in frames["Xck"].columns] + [0] * (2 * 18 + 6)
    
    return {
        "qf_shallow":   (Xco,  Xcot,  dict(learning_rate=0.05, num_leaves=7, max_depth=3, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20, extra_trees=extra_trees), 1300),
        "qf_reg":       (Xco,  Xcot,  dict(learning_rate=0.03, num_leaves=11, min_child_samples=300, feature_fraction=0.4, bagging_fraction=0.8, lambda_l2=40, extra_trees=extra_trees), 1939),
        "v1_allfam_oh": (Xao,  Xaot,  dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20, extra_trees=extra_trees), 881),
        "v2_allfam_oh": (Xaok, Xaokt, dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20, extra_trees=extra_trees), 881),
        "v2_mono":      (frames["Xck"].reset_index(drop=True), frames["Xckt"].reset_index(drop=True), dict(learning_rate=0.02, num_leaves=15, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20, monotone_constraints=mono_k, extra_trees=extra_trees), 1571)
    }

SPEC_OLD = build_specs(BASEF, extra_trees=False)
SPEC_E1 = build_specs(F1, extra_trees=False)
SPEC_E1ET = build_specs(F1, extra_trees=True)

W = {"qf_shallow": 0.24, "qf_reg": 0.12, "v1_allfam_oh": 0.32, "v2_allfam_oh": 0.16, "v2_mono": 0.16}
BASE = dict(objective="binary", bagging_freq=1, verbose=-1, num_threads=-1, seed=42)

# Load precomputed fold encodings
enc_folds = []
for k in range(1, 6):
    with open(os.path.join(CACHE_DIR, f'v16_enc_fold_{k}.pkl'), 'rb') as f_in:
        enc_folds.append(pickle.load(f_in))

with open(os.path.join(CACHE_DIR, 'v16_enc_full_data.pkl'), 'rb') as f_in:
    enc_full = pickle.load(f_in)

def run_set(spec, tag, view_obj=None):
    print(f"\n--- Running LightGBM Set: {tag} ---")
    members_oof = {}
    members_test = {}
    
    for name in W.keys():
        t_m0 = time.time()
        X, Xt, p, R = spec[name]
        p_run = dict(p)
        if view_obj is not None and "monotone_constraints" in p_run:
            p_run["monotone_constraints"] = list(p_run["monotone_constraints"]) + [0] * view_obj["tr"][0].shape[1]
        
        oof_m = np.zeros(n, dtype=np.float32)
        test_fold_m = np.zeros(nte, dtype=np.float32)
        
        for k, (a, b) in enumerate(folds):
            enc_data = enc_folds[k]
            extra_tr = [view_obj['tr'][k]] if view_obj else []
            extra_va = [view_obj['va'][k]] if view_obj else []
            extra_te = [view_obj['te'][k]] if view_obj else []
            
            fa = pd.concat([X.iloc[a].reset_index(drop=True), enc_data['fa_enc']] + extra_tr, axis=1)
            fb = pd.concat([X.iloc[b].reset_index(drop=True), enc_data['fb_enc']] + extra_va, axis=1)
            ft = pd.concat([Xt.reset_index(drop=True), enc_data['ft_enc']] + extra_te, axis=1)
            
            trn_ds = lgb.Dataset(fa, y[a])
            model = lgb.train({**BASE, **p_run}, trn_ds, num_boost_round=R)
            
            oof_m[b] = model.predict(fb)
            test_fold_m += model.predict(ft) / FOLDS
        
        # Fast seed bagging on full data
        extra_all_tr = [view_obj['all_tr']] if view_obj else []
        extra_all_te = [view_obj['all_te']] if view_obj else []
        all_fa = pd.concat([X.reset_index(drop=True), enc_full['all_fa_enc']] + extra_all_tr, axis=1)
        all_ft = pd.concat([Xt.reset_index(drop=True), enc_full['all_ft_enc']] + extra_all_te, axis=1)
        
        full_preds = []
        for sd in [42, 7]:
            m_full = lgb.train({**BASE, **p_run, "seed": sd}, lgb.Dataset(all_fa, y), num_boost_round=int(R * 1.1))
            full_preds.append(rk(m_full.predict(all_ft)))
        
        final_test_m = rk(0.5 * rk(test_fold_m) + 0.5 * np.mean(full_preds, 0))
        auc_m = roc_auc_score(y, oof_m)
        print(f"  {tag}_{name} Complete OOF AUC: {auc_m:.5f} ({time.time()-t_m0:.1f}s)")
        members_oof[name] = oof_m
        members_test[name] = final_test_m
    
    set_oof = sum(W[k] * rk(members_oof[k]) for k in W)
    set_test = sum(W[k] * members_test[k] for k in W)
    auc_set = roc_auc_score(y, set_oof)
    print(f"--> {tag} Weighted Blend OOF AUC: {auc_set:.6f} <--")
    return set_oof, set_test

# 2. Run the three missing sets
oof_old, test_old = run_set(SPEC_OLD, "old", view_obj=None)
oof_e1, test_e1 = run_set(SPEC_E1, "e1", view_obj=VIEW_O)
oof_e1et, test_e1et = run_set(SPEC_E1ET, "e1et", view_obj=VIEW_O)

# 3. Load e1Aet and CatBoost from v16 cache
v16_cache = np.load(os.path.join(CACHE_DIR, 'v_16_blend_bundle.npz'))
oof_e1Aet = v16_cache['oof_e1Aet']
test_e1Aet = v16_cache['test_e1Aet']
oof_cb = v16_cache['oof_cb']
test_cb = v16_cache['test_cb']

# 4. Master v16 Complete Formula
print("\n==================================================")
print(" COMPUTING FULL 4-SET MASTER CHAMPION ENSEMBLE")
print("==================================================")
B_oof = (rk(oof_old) + rk(oof_e1) + rk(oof_e1et)) / 3.0
B_test = (rk(test_old) + rk(test_e1) + rk(test_e1et)) / 3.0
print(f"Base Trio B = (old + e1 + e1et)/3 OOF AUC: {roc_auc_score(y, B_oof):.6f}")

v16_full_oof = 0.90 * rk(0.75 * rk(B_oof) + 0.25 * rk(oof_e1Aet)) + 0.10 * rk(oof_cb)
v16_full_test = 0.90 * rk(0.75 * rk(B_test) + 0.25 * rk(test_e1Aet)) + 0.10 * rk(test_cb)
auc_v16_full = roc_auc_score(y, v16_full_oof)
print(f"Full v16 Master Formula OOF AUC: {auc_v16_full:.6f}")

# 5. Grand Fusion with Multi-Seed Bagging (v6) and KNN Manifold (v8)
v6_cache = np.load(os.path.join(CACHE_DIR, 'v_6_blend_bundle.npz'))
v8_cache = np.load(os.path.join(CACHE_DIR, 'v_8_blend_bundle.npz'))
oof_v6, test_v6 = v6_cache['final_oof'], v6_cache['final_test_rank']
oof_v8, test_v8 = v8_cache['final_oof'], v8_cache['final_test_rank']

# Optimal multi-model blend search
best_auc = auc_v16_full
best_weights = (1.0, 0.0, 0.0)

for w_v16 in np.linspace(0.70, 0.95, 15):
    for w_v8 in np.linspace(0.02, 1.0 - w_v16, 12):
        w_v6 = max(0.0, 1.0 - w_v16 - w_v8)
        blend_cand = w_v16 * rk(v16_full_oof) + w_v8 * rk(oof_v8) + w_v6 * rk(oof_v6)
        cand_auc = roc_auc_score(y, blend_cand)
        if cand_auc > best_auc:
            best_auc = cand_auc
            best_weights = (w_v16, w_v8, w_v6)

w16, w8, w6 = best_weights
print(f"\n>>> FINAL GRAND CHAMPION BLEND <<<")
print(f"Weights: {100*w16:.1f}% v16_full + {100*w8:.1f}% v8_knn + {100*w6:.1f}% v6_multiseed")
print(f"ALL-TIME RECORD OOF AUC: {best_auc:.6f} (+{best_auc - 0.774041:+.6f} over baseline v15)")

final_champion_test = rk(w16 * rk(v16_full_test) + w8 * rk(test_v8) + w6 * rk(test_v6))

# Save submission
sub = pd.read_csv(SAMPLE_PATH)
sub['TARGET'] = final_champion_test
sub_file = os.path.join(WORK_DIR, 'v_17_full_suite_champion_submission.csv')
sub.to_csv(sub_file, index=False)
print(f"Wrote final champion submission to {sub_file}")

# Update cache bundle
np.savez_compressed(
    os.path.join(CACHE_DIR, 'v_17_full_suite_bundle.npz'),
    oof_old=oof_old, test_old=test_old,
    oof_e1=oof_e1, test_e1=test_e1,
    oof_e1et=oof_e1et, test_e1et=test_e1et,
    oof_e1Aet=oof_e1Aet, test_e1Aet=test_e1Aet,
    oof_cb=oof_cb, test_cb=test_cb,
    final_oof=w16 * rk(v16_full_oof) + w8 * rk(oof_v8) + w6 * rk(oof_v6),
    final_test=final_champion_test,
    final_auc=best_auc
)
print("EXP_13 completed in {:.1f}s.".format(time.time() - t0))
