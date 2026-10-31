"""
EXP85 Phase 3 - Workstream 3: Independent Foundation Model Zoo Training
Trains the five foundation models from the independently regenerated 550-feature matrix:
1. LightGBM Conservative
2. LightGBM ExtraTrees
3. GPU XGBoost Hist
4. GPU CatBoost Symmetric
5. LightGBM Deep

Saves regenerated predictions to:
independent_rebuild/hasan_expanded_zoo.npz
And compares against historical cache/hasan_expanded_zoo.npz.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

def evaluate_model(name, oof_rep, test_rep, oof_hist, test_hist, y_tr):
    sp_oof = spearmanr(oof_rep, oof_hist)[0]
    pe_oof = pearsonr(oof_rep, oof_hist)[0]
    sp_te = spearmanr(test_rep, test_hist)[0]
    pe_te = pearsonr(test_rep, test_hist)[0]
    mae_te = np.mean(np.abs(test_rep - test_hist))
    auc_rep = roc_auc_score(y_tr, oof_rep)
    auc_hist = roc_auc_score(y_tr, oof_hist)

    print(f"\n--- {name} Verification Telemetry ---")
    print(f"  OOF AUC: Rebuilt = {auc_rep:.6f} | Hist = {auc_hist:.6f} (Gap: {auc_rep - auc_hist:+.6f})")
    print(f"  Test Spearman Correlation: {sp_te:.7f}")
    print(f"  Test Pearson Correlation:  {pe_te:.7f}")
    print(f"  Test Mean Absolute Error:  {mae_te:.6e}")
    return {
        'model': name,
        'auc_rebuilt': auc_rep,
        'auc_hist': auc_hist,
        'sp_oof': sp_oof,
        'pe_oof': pe_oof,
        'sp_test': sp_te,
        'pe_test': pe_te,
        'mae_test': mae_te
    }

def main():
    t_start = time.time()
    print("=" * 80)
    print("WORKSTREAM 3: INDEPENDENT FIVE FOUNDATION MODEL TRAINING")
    print("=" * 80)

    in_dir = r"D:\gci\gemini\champion\phase3_reconstruction\independent_rebuild"
    out_dir = in_dir
    raw_train_path = r"D:\gci\inpuy\train.csv"
    hist_zoo_path = r"D:\gci\gemini\cache\hasan_expanded_zoo.npz"

    print("[1] Loading regenerated 550 features...")
    X_tr = pd.read_feather(os.path.join(in_dir, "hasan_selected_550_tr.feather"))
    X_te = pd.read_feather(os.path.join(in_dir, "hasan_selected_550_te.feather"))
    y_tr = pd.read_csv(raw_train_path, usecols=["TARGET"])["TARGET"].values

    # Clean infinities
    X_tr.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_te.replace([np.inf, -np.inf], np.nan, inplace=True)

    n_train = len(X_tr)
    n_test = len(X_te)
    print(f"Loaded: Train {X_tr.shape}, Test {X_te.shape}")

    hist_zoo = np.load(hist_zoo_path)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    telemetry = []

    # =========================================================================
    # 1. LightGBM Conservative
    # =========================================================================
    print("\n[Pillar 1/5] Training LightGBM Conservative (5 folds)...")
    t0 = time.time()
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
        model = lgb.train(params_lgb_cons, dtrain, num_boost_round=800, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb_cons[val_idx] = model.predict(X_tr.iloc[val_idx])
        test_lgb_cons += model.predict(X_te) / 5.0
        print(f"  Fold {fold+1}/5 AUC: {roc_auc_score(y_tr[val_idx], oof_lgb_cons[val_idx]):.6f}")

    results['oof_lgb_cons'] = oof_lgb_cons
    results['test_lgb_cons'] = test_lgb_cons
    print(f"Pillar 1 completed in {time.time()-t0:.1f}s")
    telemetry.append(evaluate_model('LGBM Conservative', oof_lgb_cons, test_lgb_cons, hist_zoo['oof_lgb_cons'], hist_zoo['test_lgb_cons'], y_tr))

    # =========================================================================
    # 2. LightGBM ExtraTrees
    # =========================================================================
    print("\n[Pillar 2/5] Training LightGBM ExtraTrees (5 folds)...")
    t0 = time.time()
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
        model = lgb.train(params_lgb_et, dtrain, num_boost_round=800, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb_et[val_idx] = model.predict(X_tr.iloc[val_idx])
        test_lgb_et += model.predict(X_te) / 5.0
        print(f"  Fold {fold+1}/5 AUC: {roc_auc_score(y_tr[val_idx], oof_lgb_et[val_idx]):.6f}")

    results['oof_lgb_et'] = oof_lgb_et
    results['test_lgb_et'] = test_lgb_et
    print(f"Pillar 2 completed in {time.time()-t0:.1f}s")
    telemetry.append(evaluate_model('LGBM ExtraTrees', oof_lgb_et, test_lgb_et, hist_zoo['oof_lgb_et'], hist_zoo['test_lgb_et'], y_tr))

    # =========================================================================
    # 3. GPU XGBoost Hist
    # =========================================================================
    print("\n[Pillar 3/5] Training GPU XGBoost Hist (5 folds on CUDA)...")
    t0 = time.time()
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
        bst = xgb.train(params_xgb, dtrn, num_boost_round=800, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        oof_xgb[val_idx] = bst.predict(dval)
        test_xgb += bst.predict(dtest_xgb) / 5.0
        print(f"  Fold {fold+1}/5 AUC: {roc_auc_score(y_tr[val_idx], oof_xgb[val_idx]):.6f}")

    results['oof_xgb'] = oof_xgb
    results['test_xgb'] = test_xgb
    print(f"Pillar 3 completed in {time.time()-t0:.1f}s")
    telemetry.append(evaluate_model('GPU XGBoost', oof_xgb, test_xgb, hist_zoo['oof_xgb'], hist_zoo['test_xgb'], y_tr))

    # =========================================================================
    # 4. GPU CatBoost Symmetric
    # =========================================================================
    print("\n[Pillar 4/5] Training GPU CatBoost Symmetric (5 folds on CUDA)...")
    t0 = time.time()
    oof_cat = np.zeros(n_train, dtype=np.float64)
    test_cat = np.zeros(n_test, dtype=np.float64)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        X_train, y_train = X_tr.iloc[trn_idx], y_tr[trn_idx]
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
        oof_cat[val_idx] = cb_model.predict_proba(X_val)[:, 1]
        test_cat += cb_model.predict_proba(X_te)[:, 1] / 5.0
        print(f"  Fold {fold+1}/5 AUC: {roc_auc_score(y_val, oof_cat[val_idx]):.6f}")

    results['oof_cat'] = oof_cat
    results['test_cat'] = test_cat
    print(f"Pillar 4 completed in {time.time()-t0:.1f}s")
    telemetry.append(evaluate_model('GPU CatBoost', oof_cat, test_cat, hist_zoo['oof_cat'], hist_zoo['test_cat'], y_tr))

    # =========================================================================
    # 5. LightGBM Deep
    # =========================================================================
    print("\n[Pillar 5/5] Training LightGBM Deep (5 folds)...")
    t0 = time.time()
    oof_lgb_deep = np.zeros(n_train, dtype=np.float64)
    test_lgb_deep = np.zeros(n_test, dtype=np.float64)

    deep_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.025,
        'num_leaves': 127,
        'max_depth': 8,
        'min_child_samples': 80,
        'subsample': 0.80,
        'subsample_freq': 1,
        'colsample_bytree': 0.65,
        'reg_alpha': 10.0,
        'reg_lambda': 30.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        dtrain = lgb.Dataset(X_tr.iloc[trn_idx], label=y_tr[trn_idx])
        dval = lgb.Dataset(X_tr.iloc[val_idx], label=y_tr[val_idx], reference=dtrain)
        booster = lgb.train(deep_params, dtrain, num_boost_round=800, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb_deep[val_idx] = booster.predict(X_tr.iloc[val_idx])
        test_lgb_deep += booster.predict(X_te) / 5.0
        print(f"  Fold {fold+1}/5 AUC: {roc_auc_score(y_tr[val_idx], oof_lgb_deep[val_idx]):.6f}")

    results['oof_lgb_deep'] = oof_lgb_deep
    results['test_lgb_deep'] = test_lgb_deep
    print(f"Pillar 5 completed in {time.time()-t0:.1f}s")
    telemetry.append(evaluate_model('LGBM Deep', oof_lgb_deep, test_lgb_deep, hist_zoo['oof_lgb_deep'], hist_zoo['test_lgb_deep'], y_tr))

    # Save to independent_rebuild
    out_npz = os.path.join(out_dir, "hasan_expanded_zoo.npz")
    np.savez_compressed(out_npz, **results, y_true=y_tr)
    print(f"\n[+] Successfully saved rebuilt zoo to: {out_npz}")
    print(f"Total Workstream 3 runtime: {time.time()-t_start:.1f}s")

    # Save summary telemetry table
    df_tel = pd.DataFrame(telemetry)
    df_tel.to_csv(os.path.join(out_dir, "model_zoo_reconstruction_telemetry.csv"), index=False)
    print("\n=== SUMMARY TELEMETRY TABLE ===")
    print(df_tel.to_string(index=False))

if __name__ == '__main__':
    main()

