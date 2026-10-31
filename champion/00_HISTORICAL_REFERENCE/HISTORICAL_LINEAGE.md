# Historical EXP85 Lineage & Synthesis Architecture

```mermaid
graph TD
    A["Raw Competition Data<br>train.csv & test.csv"] --> B["Hasan 669 Feature System<br>Clayton Copula + Merton PD + Solvency"]
    B --> C["5-Fold Stratified LightGBM Gain Screening<br>Top 550 Pristine Features"]
    C --> D["5-Model Foundation Zoo<br>LGBM Cons, LGBM ET, Hist-XGB, CatBoost GPU, LGBM Deep"]
    D --> E["Transductive Calibration Pyramid<br>Tail Gating -> Harmonic Diffusion -> Discrepancy Calibration"]
    E --> F["Bedrock Anchor 74<br>v_20261027_precision_transductive_apex.csv"]
    
    G["V17 Full Suite Bundle<br>oof_e1Aet, oof_e1et, final_oof"] --> H["Upgraded 8-Engine Suite<br>Nelder-Mead Simplex Weights"]
    I["V55 Members Bundle<br>cbprod, deep_et, allfam, v2_mono"] --> H
    J["10-Fold Hist-XGBoost<br>oof_10f_xgb.npy, test_10f_xgb.npy"] --> H
    
    F --> K["Probit Macro-Synthesis<br>85% Anchor 74 + 15% 8-Engine Suite"]
    H --> K
    K --> L["EXP85 Champion Submission<br>v_20261027_optimal_macro_apex_v85.csv<br>LB: 0.801340 | SHA: 78c31bb4..."]
```

## The 85/15 Macro Probit Synthesis Formula
$$\Phi^{-1}(U_{	ext{macro}}) = 0.850 \cdot \Phi^{-1}(U_{	ext{Anchor 74}}) + 0.150 \cdot \Phi^{-1}(U_{	ext{8-Engine Suite}})$$

### 8-Engine Suite Member Weights:
1. `v17_e1Aet`   : 0.2932
2. `xgb10`       : 0.2596
3. `v17_final`   : 0.2051
4. `v55_cbprod`  : 0.0874
5. `v17_e1et`    : 0.0489
6. `v55_deep_et` : 0.0442
7. `v55_allfam`  : 0.0383
8. `v55_v2_mono` : 0.0233
