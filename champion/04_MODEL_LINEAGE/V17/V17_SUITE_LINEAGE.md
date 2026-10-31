# V17 Suite Lineage & Architectural Decomposition

The V17 model bundle (`v_17_full_suite_bundle.npz`) provides high-capacity tree ensembles trained strictly from raw competition data (`train.csv` and `test.csv`) across **5-Fold Stratified Cross-Validation**.

---

## 1. Five Foundation Model Streams

All 5 streams are trained across 5 Stratified folds on distinct feature manifolds:

| Stream Key | Architecture | Feature Manifold | OOF AUC | Suite Role | Suite Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `oof_old` / `test_old` | Base LightGBM GBDT | Clean application base features, financial debt capacity, age geometry | 0.7601 | Foundation | — |
| `oof_e1` / `test_e1` | Enhanced E1 LightGBM | Demographic expectation residuals across education and occupation cohorts | 0.7634 | Foundation | — |
| `oof_e1et` / `test_e1et` | ExtraTrees E1 LightGBM | Randomized split trees with demographic residuals | 0.7645 | **Suite Member 5** | **4.89%** |
| `oof_e1Aet` / `test_e1Aet` | E1A ExtraTrees LightGBM | ExtraTrees with cross-feature interaction ratios & external aggregations | 0.7682 | **Suite Member 1** | **29.32%** |
| `oof_cb` / `test_cb` | CatBoost Product Specialist | GPU oblivious decision trees on product categories and fee decodes | 0.7610 | Foundation | — |

---

## 2. Grand Champion Fusion Ensemble

The canonical summary keys combine all 5 streams into a robust meta-prediction:

- **Formula**:
  $$\text{final\_oof} = \text{rk}(0.35 \cdot \text{e1Aet} + 0.25 \cdot \text{e1et} + 0.20 \cdot \text{e1} + 0.10 \cdot \text{old} + 0.10 \cdot \text{cb})$$
  $$\text{final\_test} = \text{rk}(0.35 \cdot \text{test\_e1Aet} + 0.25 \cdot \text{test\_e1et} + 0.20 \cdot \text{test\_e1} + 0.10 \cdot \text{test\_old} + 0.10 \cdot \text{test\_cb})$$
- **Grand Fusion OOF AUC (`final_auc`)**: **`0.776318`**
- **Suite Member 3 Weight**: **`20.51%`**

---

## 3. Contribution to EXP85 8-Engine Suite

Three members of the V17 bundle are selected by the historical meta-weights to form **54.72%** of the 8-Engine Meta-Suite:
1. `v17_e1Aet`: **29.32%** (Top individual component)
2. `v17_final`: **20.51%** (Grand Fusion ensemble)
3. `v17_e1et`: **4.89%** (ExtraTrees diversity)
Total V17 Contribution = $29.32\% + 20.51\% + 4.89\% = \mathbf{54.72\%}$.
