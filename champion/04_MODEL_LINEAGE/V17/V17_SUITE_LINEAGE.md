# V17 Suite Lineage & Architectural Decomposition

The V17 model bundle (`v_17_full_suite_bundle.npz`) provides three key high-capacity tree ensembles:
1. `oof_e1Aet` / `test_e1Aet`: LightGBM ExtraTrees trained with $EXT\_SOURCE\_1$ augmented features (Weight in Suite: 29.32%).
2. `final_oof` / `final_test`: 10-fold composite meta-prediction (Weight in Suite: 20.51%).
3. `oof_e1et` / `test_e1et`: LightGBM ExtraTrees with raw $EXT\_SOURCE\_1$ (Weight in Suite: 4.89%).

Combined V17 contribution to the 8-Engine Suite: **54.72%**.
