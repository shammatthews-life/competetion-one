"""
Stage 2: Fast Feature Screening & Importance Ranking Engine
Evaluates all 669 Hasan Mazi features using LightGBM feature gain on Stratified 5-Fold CV.
Filters out zero-gain and low-importance features to select the top ~530-580 most discriminative features.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print("Loading 669 Hasan Features...")
X_tr = pd.read_feather('cache/hasan_features_tr.feather')
X_te = pd.read_feather('cache/hasan_features_te.feather')
y_tr = np.load('cache/y_train.npy')

feature_names = list(X_tr.columns)
print(f"Features: {len(feature_names)}, Train shape: {X_tr.shape}, Test shape: {X_te.shape}")

# Fast 5-fold screening with LightGBM
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
feature_importances = np.zeros(len(feature_names), dtype=np.float64)
oof_preds = np.zeros(len(y_tr), dtype=np.float64)

# Use fold 1 & 2 for rapid feature gain estimation, then evaluate full 5-fold CV
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

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
    print(f"--- Screening Fold {fold+1}/5 ---")
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
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")

overall_auc = roc_auc_score(y_tr, oof_preds)
print(f"\n=======================================================")
print(f"All 669 Hasan Features 5-Fold OOF AUC: {overall_auc:.6f}")
print(f"=======================================================\n")

# Rank features by importance gain
df_importance = pd.DataFrame({
    'feature': feature_names,
    'importance_gain': feature_importances
}).sort_values('importance_gain', ascending=False).reset_index(drop=True)

df_importance.to_csv('cache/hasan_feature_importance.csv', index=False)

print("Top 25 Most Important Features:")
for idx, row in df_importance.head(25).iterrows():
    print(f"  {idx+1:2d}. {row['feature']:35s}: gain = {row['importance_gain']:10.1f}")

zero_imp = df_importance[df_importance['importance_gain'] == 0]
print(f"\nFeatures with exactly 0 gain: {len(zero_imp)}")

# Select top features matching Hasan's sweet spot (~530 to 580 features)
# Keep all features with non-trivial importance (gain > threshold or top 550)
selected_features = df_importance.head(550)['feature'].tolist()
print(f"Selected top {len(selected_features)} features for master multi-model zoo!")

# Save selected feature matrices
X_tr_selected = X_tr[selected_features]
X_te_selected = X_te[selected_features]

X_tr_selected.to_feather('cache/hasan_selected_550_tr.feather')
X_te_selected.to_feather('cache/hasan_selected_550_te.feather')

# Also save full OOF predictions as baseline
np.save('cache/oof_hasan_lgb_raw669.npy', oof_preds)

print("Successfully saved selected 550 feature matrices to cache!")
