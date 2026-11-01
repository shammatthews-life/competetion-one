"""
Stage 3: Hasan Master Multi-Family Zoo Training
Trains 4 structurally diverse gradient boosted decision tree architectures on the selected Hasan feature matrix:
1. LightGBM Conservative (Low-variance regularized trees)
2. LightGBM ExtraTrees (Extremely randomized tree splits)
3. GPU XGBoost Hist (NVIDIA RTX 4050 CUDA histogram trees)
4. GPU CatBoost Symmetric (GPU symmetric obliquis trees)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HASAN MASTER MULTI-FAMILY MODEL ZOO TRAINING")
print("=" * 80)

# Load selected features
X_tr = pd.read_feather('cache/hasan_selected_550_tr.feather')
X_te = pd.read_feather('cache/hasan_selected_550_te.feather')
y_tr = np.load('cache/y_train.npy')

# Replace any infinite values with NaN for XGBoost compatibility
X_tr.replace([np.inf, -np.inf], np.nan, inplace=True)
X_te.replace([np.inf, -np.inf], np.nan, inplace=True)

n_train = len(X_tr)
n_test = len(X_te)
n_features = X_tr.shape[1]
print(f"Loaded features: {n_features} features, Train={n_train}, Test={n_test}")

# 5-Fold Stratified Cross-Validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==============================================================================
# Pillar 1: LightGBM Conservative
# ==============================================================================
print("\n>>> PILLAR 1: LightGBM Conservative (5-Fold CV) <<<")
if os.path.exists('cache/oof_lgb_cons.npy') and os.path.exists('cache/test_lgb_cons.npy'):
    print("Loading cached Pillar 1 predictions...")
    oof_lgb_cons = np.load('cache/oof_lgb_cons.npy')
    test_lgb_cons = np.load('cache/test_lgb_cons.npy')
else:
    oof_lgb_cons = np.zeros(n_train, dtype=np.float64)
    test_lgb_cons = np.zeros(n_test, dtype=np.float64)

    params_lgb_cons = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.60,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'min_child_samples': 80,
        'reg_alpha': 2.0,
        'reg_lambda': 5.0,
        'verbose': -1,
        'n_jobs': 4,
        'random_state': 42
    }

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        dtrain = lgb.Dataset(X_tr.iloc[trn_idx], label=y_tr[trn_idx])
        dval = lgb.Dataset(X_tr.iloc[val_idx], label=y_tr[val_idx], reference=dtrain)
        
        model = lgb.train(
            params_lgb_cons,
            dtrain,
            num_boost_round=800,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        oof_lgb_cons[val_idx] = model.predict(X_tr.iloc[val_idx])
        test_lgb_cons += model.predict(X_te) / 5.0
        print(f"  Fold {fold+1} AUC: {roc_auc_score(y_tr[val_idx], oof_lgb_cons[val_idx]):.6f}")

    np.save('cache/oof_lgb_cons.npy', oof_lgb_cons)
    np.save('cache/test_lgb_cons.npy', test_lgb_cons)

auc_lgb_cons = roc_auc_score(y_true=y_tr, y_score=oof_lgb_cons)
print(f"Pillar 1 OOF AUC (LGBM Conservative): {auc_lgb_cons:.6f}")

# ==============================================================================
# Pillar 2: LightGBM ExtraTrees
# ==============================================================================
print("\n>>> PILLAR 2: LightGBM ExtraTrees (5-Fold CV) <<<")
if os.path.exists('cache/oof_lgb_et.npy') and os.path.exists('cache/test_lgb_et.npy'):
    print("Loading cached Pillar 2 predictions...")
    oof_lgb_et = np.load('cache/oof_lgb_et.npy')
    test_lgb_et = np.load('cache/test_lgb_et.npy')
else:
    oof_lgb_et = np.zeros(n_train, dtype=np.float64)
    test_lgb_et = np.zeros(n_test, dtype=np.float64)

    params_lgb_et = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'extra_trees': True,
        'learning_rate': 0.02,
        'num_leaves': 45,
        'max_depth': 7,
        'feature_fraction': 0.45,
        'bagging_fraction': 0.70,
        'bagging_freq': 1,
        'min_child_samples': 60,
        'reg_alpha': 5.0,
        'reg_lambda': 10.0,
        'verbose': -1,
        'n_jobs': 4,
        'random_state': 2026
    }

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        dtrain = lgb.Dataset(X_tr.iloc[trn_idx], label=y_tr[trn_idx])
        dval = lgb.Dataset(X_tr.iloc[val_idx], label=y_tr[val_idx], reference=dtrain)
        
        model = lgb.train(
            params_lgb_et,
            dtrain,
            num_boost_round=800,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        oof_lgb_et[val_idx] = model.predict(X_tr.iloc[val_idx])
        test_lgb_et += model.predict(X_te) / 5.0
        print(f"  Fold {fold+1} AUC: {roc_auc_score(y_tr[val_idx], oof_lgb_et[val_idx]):.6f}")

    np.save('cache/oof_lgb_et.npy', oof_lgb_et)
    np.save('cache/test_lgb_et.npy', test_lgb_et)

auc_lgb_et = roc_auc_score(y_true=y_tr, y_score=oof_lgb_et)
print(f"Pillar 2 OOF AUC (LGBM ExtraTrees): {auc_lgb_et:.6f}")

# ==============================================================================
# Pillar 3: GPU XGBoost Hist
# ==============================================================================
print("\n>>> PILLAR 3: GPU XGBoost Hist (5-Fold CV) <<<")
oof_xgb = np.zeros(n_train, dtype=np.float64)
test_xgb = np.zeros(n_test, dtype=np.float64)

params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.80,
    'colsample_bytree': 0.60,
    'reg_alpha': 3.0,
    'reg_lambda': 8.0,
    'random_state': 42
}

dtest_xgb = xgb.DMatrix(X_te)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
    dtrn = xgb.DMatrix(X_tr.iloc[trn_idx], label=y_tr[trn_idx])
    dval = xgb.DMatrix(X_tr.iloc[val_idx], label=y_tr[val_idx])
    
    bst = xgb.train(
        params_xgb,
        dtrn,
        num_boost_round=800,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    oof_xgb[val_idx] = bst.predict(dval)
    test_xgb += bst.predict(dtest_xgb) / 5.0
    print(f"  Fold {fold+1} AUC: {roc_auc_score(y_tr[val_idx], oof_xgb[val_idx]):.6f}")

auc_xgb = roc_auc_score(y_tr, oof_xgb)
print(f"Pillar 3 OOF AUC (GPU XGBoost Hist): {auc_xgb:.6f}")

# ==============================================================================
# Save Multi-Model Zoo Predictions
# ==============================================================================
print("\nSaving Multi-Model Zoo Predictions...")
np.savez_compressed(
    'cache/hasan_zoo_predictions.npz',
    oof_lgb_cons=oof_lgb_cons,
    test_lgb_cons=test_lgb_cons,
    oof_lgb_et=oof_lgb_et,
    test_lgb_et=test_lgb_et,
    oof_xgb=oof_xgb,
    test_xgb=test_xgb,
    y_true=y_tr
)

print(f"\nModel Zoo Summary:")
print(f"  LightGBM Conservative : {auc_lgb_cons:.6f}")
print(f"  LightGBM ExtraTrees   : {auc_lgb_et:.6f}")
print(f"  GPU XGBoost Hist      : {auc_xgb:.6f}")

# Probit Rank Blending
def to_probit(p):
    r = rankdata(p, method='average') / (len(p) + 1.0)
    r = np.clip(r, 1e-6, 1.0 - 1e-6)
    return norm.ppf(r)

z_lgb_c = to_probit(oof_lgb_cons)
z_lgb_e = to_probit(oof_lgb_et)
z_xgb_p = to_probit(oof_xgb)

# Equal-weighted simplex blend
z_blend = (z_lgb_c + z_lgb_e + z_xgb_p) / 3.0
blend_auc = roc_auc_score(y_tr, z_blend)
print(f"\nTri-Family Probit Simplex Blend OOF AUC: {blend_auc:.6f}")
print("Done!")
