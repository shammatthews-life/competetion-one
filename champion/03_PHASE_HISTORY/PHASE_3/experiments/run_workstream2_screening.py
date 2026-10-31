"""
EXP85 Phase 3 - Workstream 2 Step B: 550 Feature Selection Reconstruction
Runs screen_and_select_features.py using the independently regenerated 669 features
and saves selected 550 feature matrices into independent_rebuild.
"""

import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def main():
    t0 = time.time()
    print("=" * 80)
    print("WORKSTREAM 2 STEP B: SCREENING AND SELECTING TOP 550 FEATURES")
    print("=" * 80)

    in_dir = r"D:\gci\gemini\champion\phase3_reconstruction\independent_rebuild"
    out_dir = in_dir
    raw_train_path = r"D:\gci\inpuy\train.csv"

    print("[1] Loading regenerated 669 features and training labels...")
    X_tr = pd.read_feather(os.path.join(in_dir, "hasan_features_tr.feather"))
    X_te = pd.read_feather(os.path.join(in_dir, "hasan_features_te.feather"))
    y_tr = pd.read_csv(raw_train_path, usecols=["TARGET"])["TARGET"].values

    feature_names = list(X_tr.columns)
    print(f"Features: {len(feature_names)}, Train: {X_tr.shape}, Test: {X_te.shape}")

    # Fast 5-fold screening with LightGBM
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_importances = np.zeros(len(feature_names), dtype=np.float64)
    oof_preds = np.zeros(len(y_tr), dtype=np.float64)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 50,
        'verbose': -1,
        'n_jobs': 4,
        'random_state': 42
    }

    print("[2] Running 5-fold feature gain evaluation...")
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        dtrain = lgb.Dataset(X_tr.iloc[trn_idx], label=y_tr[trn_idx])
        dval = lgb.Dataset(X_tr.iloc[val_idx], label=y_tr[val_idx], reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )

        oof_preds[val_idx] = model.predict(X_tr.iloc[val_idx])
        feature_importances += model.feature_importance(importance_type='gain')
        fold_auc = roc_auc_score(y_tr[val_idx], oof_preds[val_idx])
        print(f"  Fold {fold+1} AUC: {fold_auc:.6f}")

    overall_auc = roc_auc_score(y_tr, oof_preds)
    print(f"All 669 Features 5-Fold OOF AUC: {overall_auc:.6f}")

    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': feature_importances
    }).sort_values('importance_gain', ascending=False).reset_index(drop=True)

    df_importance.to_csv(os.path.join(out_dir, 'hasan_feature_importance.csv'), index=False)

    selected_features = df_importance.head(550)['feature'].tolist()
    print(f"[3] Selected top {len(selected_features)} features.")

    X_tr_selected = X_tr[selected_features]
    X_te_selected = X_te[selected_features]

    X_tr_selected.to_feather(os.path.join(out_dir, 'hasan_selected_550_tr.feather'))
    X_te_selected.to_feather(os.path.join(out_dir, 'hasan_selected_550_te.feather'))
    np.save(os.path.join(out_dir, 'oof_hasan_lgb_raw669.npy'), oof_preds)

    print(f"[4] Successfully saved 550 features: Train {X_tr_selected.shape}, Test {X_te_selected.shape} ({time.time()-t0:.1f}s)")

    # Compare with historical 550 features
    hist_tr_path = r'D:\gci\gemini\cache\hasan_selected_550_tr.feather'
    hist_te_path = r'D:\gci\gemini\cache\hasan_selected_550_te.feather'

    df_hist_tr = pd.read_feather(hist_tr_path)
    df_hist_te = pd.read_feather(hist_te_path)

    cols_identical = list(X_tr_selected.columns) == list(df_hist_tr.columns)
    print(f"\nExact 550 Feature Names and Ordering Match: {cols_identical}")
    
    # Calculate feature overlap
    common_cols = set(X_tr_selected.columns).intersection(set(df_hist_tr.columns))
    print(f"Selected Feature Overlap: {len(common_cols)} / 550 ({len(common_cols)/550*100:.2f}%)")

    # If common columns exist, verify values on common columns
    val_diffs = []
    for c in list(common_cols)[:50]:
        v1 = X_tr_selected[c].values
        v2 = df_hist_tr[c].values
        valid = ~np.isnan(v1) & ~np.isnan(v2)
        if np.any(valid):
            d = np.max(np.abs(v1[valid] - v2[valid]))
            if d > 1e-4:
                val_diffs.append((c, d))
    print(f"Value differences across sampled common columns: {len(val_diffs)} mismatches")

if __name__ == '__main__':
    main()

