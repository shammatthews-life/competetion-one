# 550 Feature Screening Methodology

The 550 features were selected from the 669 candidate features using 5-fold Stratified LightGBM gain ranking:
1. A baseline LightGBM model was fit across 5 folds with random seed 42.
2. The total feature importance gain was averaged across all 5 folds.
3. Features with zero gain or negative marginal validation impact were eliminated.
4. The top 550 features were exported to `hasan_selected_550_tr.feather` and `hasan_selected_550_te.feather`.
5. Feature ranking details are stored in `hasan_feature_importance.csv`.
