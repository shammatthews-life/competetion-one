# V55 Multi-Architecture Suite Lineage & Decomposition

The V55 suite bundle (`v55_members.npz`) provides 7 distinct model architectures trained across **5-Fold Stratified Cross-Validation** to provide maximum structural diversity for meta-ensembling.

---

## 1. Complete Architecture Inventory

| Member Key | Algorithm | Structural Specification | OOF AUC | Suite Role | Fixed Suite Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `qf_shallow` | LightGBM | Shallow Reg (num_leaves=7, max_depth=3, extra_trees=True) | 0.7582 | Diagnostic | — |
| `qf_reg` | LightGBM | Conservative Reg (num_leaves=11, min_child_samples=300, L2=40) | 0.7595 | Diagnostic | — |
| `v1_allfam_oh` | LightGBM | Demographics One-Hot (num_leaves=15, feature_fraction=0.3) | 0.7612 | Diagnostic | — |
| `v2_allfam_oh` | LightGBM | V2 Demographics OH + Financial Ratios (num_leaves=15) | 0.7648 | **Suite Member 7** | **3.83%** |
| `v2_mono` | LightGBM | Monotone Constraints on External Sources (leaves=15, L2=20) | 0.7629 | **Suite Member 8** | **2.33%** |
| `deep_et` | LightGBM | Deep ExtraTrees (num_leaves=31, min_child_samples=80) | 0.7661 | **Suite Member 6** | **4.42%** |
| `cb_product` | CatBoost GPU | Oblivious Trees on Product Categories & Terms (depth=6, L2=8.0) | 0.7635 | **Suite Member 4** | **8.74%** |

---

## 2. Validation & Inference Symmetry

- **Symmetric Fold-Averaging**:
  Both OOF predictions and test predictions are generated strictly through held-out fold evaluation:
  $$P_{\text{OOF}}[val\_idx] = \mathcal{M}_k(X[val\_idx]), \quad P_{\text{test}} = \frac{1}{K} \sum_{k=1}^K \mathcal{M}_k(X_{\text{test}})$$
  This guarantees that test predictions share identical distributional properties with the OOF evaluation base.
- **Uniform Rank Percentile Projection**:
  Predictions are uniformly projected onto $(0, 1)$ via average ranking:
  $$\text{rk}(P) = \frac{\text{rank}(P) - 0.5}{N}$$

---

## 3. Contribution to EXP85 8-Engine Suite

The 4 selected V55 members contribute **19.32%** of the 8-Engine Meta-Suite:
1. `v55_cbprod`: **8.74%** (CatBoost categorical product specialist)
2. `v55_deep_et`: **4.42%** (Deep ExtraTrees high-capacity learner)
3. `v55_allfam`: **3.83%** (One-hot demographic feature specialist)
4. `v55_v2_mono`: **2.33%** (Monotone solvency regularization)
Total V55 Contribution = $8.74\% + 4.42\% + 3.83\% + 2.33\% = \mathbf{19.32\%}$.
