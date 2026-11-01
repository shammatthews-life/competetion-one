"""
TRAIN CATBOOST GPU & DEEP LIGHTGBM ON 550 PRISTINE FEATURES
==========================================================
Expands the model zoo with:
1. CatBoost GPU (Oblivious / Symmetric decision trees) - 5 folds
2. LightGBM Deep (High capacity leaf-wise trees) - 5 folds
Combined with existing LightGBM Conservative, ExtraTrees, and XGBoost Hist.
Saves all 5 model families to cache/hasan_expanded_zoo.npz.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

print("=" * 80)
print("TRAINING EXPANDED ZOO: CATBOOST GPU + DEEP LIGHTGBM (5-FOLD)")
print("=" * 80)

# Load 550 features
cache_dir = "cache"
X_tr = pd.read_feather(os.path.join(cache_dir, "hasan_selected_550_tr.feather"))
X_te = pd.read_feather(os.path.join(cache_dir, "hasan_selected_550_te.feather"))
y_tr = pd.read_csv("d:/gci/inpuy/train.csv", usecols=["TARGET"])["TARGET"].values

n_train = len(y_tr)
n_test = len(X_te)
print(f"Features: Train {X_tr.shape}, Test {X_te.shape}")

# Load existing zoo
prev_zoo = np.load(os.path.join(cache_dir, "hasan_zoo_predictions.npz"))
oof_lgb_cons = prev_zoo["oof_lgb_cons"]
test_lgb_cons = prev_zoo["test_lgb_cons"]
oof_lgb_et = prev_zoo["oof_lgb_et"]
test_lgb_et = prev_zoo["test_lgb_et"]
oof_xgb = prev_zoo["oof_xgb"]
test_xgb = prev_zoo["test_xgb"]

print(f"Loaded existing zoo: LGBM Cons (AUC: {roc_auc_score(y_tr, oof_lgb_cons):.6f}), "
      f"LGBM ET (AUC: {roc_auc_score(y_tr, oof_lgb_et):.6f}), "
      f"XGB GPU (AUC: {roc_auc_score(y_tr, oof_xgb):.6f})")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 1. CatBoost GPU (Oblivious Trees)
# ==========================================
print("\n--- Training CatBoost GPU (Depth 6, 750 iterations) ---")
oof_cat = np.zeros(n_train, dtype=np.float64)
test_cat = np.zeros(n_test, dtype=np.float64)

t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
    print(f"Fold {fold+1}/5...", end="", flush=True)
    f_t0 = time.time()
    X_train, y_train = X_tr.iloc[train_idx], y_tr[train_idx]
    X_val, y_val = X_tr.iloc[val_idx], y_tr[val_idx]

    cb_model = cb.CatBoostClassifier(
        iterations=750,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=5.0,
        eval_metric="Logloss",
        task_type="GPU",
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=60, verbose=0)
    
    val_pred = cb_model.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = val_pred
    test_cat += cb_model.predict_proba(X_te)[:, 1] / 5.0

    fold_auc = roc_auc_score(y_val, val_pred)
    print(f" AUC: {fold_auc:.6f} ({time.time()-f_t0:.1f}s)")

cat_auc = roc_auc_score(y_tr, oof_cat)
print(f"==> CatBoost GPU Overall OOF AUC: {cat_auc:.6f} (Total: {time.time()-t0:.1f}s)")

# ==========================================
# 2. LightGBM Deep (High-Capacity Trees)
# ==========================================
print("\n--- Training LightGBM Deep (max_depth=8, num_leaves=127) ---")
oof_lgb_deep = np.zeros(n_train, dtype=np.float64)
test_lgb_deep = np.zeros(n_test, dtype=np.float64)

deep_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.025,
    "num_leaves": 127,
    "max_depth": 8,
    "min_child_samples": 80,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.65,
    "reg_alpha": 10.0,
    "reg_lambda": 30.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
    print(f"Fold {fold+1}/5...", end="", flush=True)
    f_t0 = time.time()
    dtrain = lgb.Dataset(X_tr.iloc[train_idx], label=y_tr[train_idx])
    dval = lgb.Dataset(X_tr.iloc[val_idx], label=y_tr[val_idx], reference=dtrain)

    booster = lgb.train(
        deep_params,
        dtrain,
        num_boost_round=800,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    val_pred = booster.predict(X_tr.iloc[val_idx])
    oof_lgb_deep[val_idx] = val_pred
    test_lgb_deep += booster.predict(X_te) / 5.0

    fold_auc = roc_auc_score(y_tr[val_idx], val_pred)
    print(f" AUC: {fold_auc:.6f} ({time.time()-f_t0:.1f}s)")

deep_auc = roc_auc_score(y_tr, oof_lgb_deep)
print(f"==> LightGBM Deep Overall OOF AUC: {deep_auc:.6f} (Total: {time.time()-t0:.1f}s)")

# Save expanded zoo
save_path = os.path.join(cache_dir, "hasan_expanded_zoo.npz")
np.savez_compressed(
    save_path,
    oof_lgb_cons=oof_lgb_cons,
    test_lgb_cons=test_lgb_cons,
    oof_lgb_et=oof_lgb_et,
    test_lgb_et=test_lgb_et,
    oof_xgb=oof_xgb,
    test_xgb=test_xgb,
    oof_cat=oof_cat,
    test_cat=test_cat,
    oof_lgb_deep=oof_lgb_deep,
    test_lgb_deep=test_lgb_deep,
    y_true=y_tr
)
print(f"\nSaved all 5 model families to {save_path}!")
print("=" * 80)
