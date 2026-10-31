# V55 Suite Lineage & Architectural Decomposition

The V55 model bundle (`v55_members.npz`) provides non-linear categorical and deep tree diversity:
1. `oof_cb_product` / `test_cb_product`: CatBoost trained on product-conditional interaction terms (Weight in Suite: 8.74%).
2. `oof_deep_et` / `test_deep_et`: Deep ExtraTrees with 63 leaves (Weight in Suite: 4.42%).
3. `oof_v2_allfam_oh` / `test_v2_allfam_oh`: LightGBM one-hot encoded on demographic families (Weight in Suite: 3.83%).
4. `oof_v2_mono` / `test_v2_mono`: Monotonicity-constrained LightGBM on credit terms (Weight in Suite: 2.33%).

Combined V55 contribution to the 8-Engine Suite: **19.32%**.
