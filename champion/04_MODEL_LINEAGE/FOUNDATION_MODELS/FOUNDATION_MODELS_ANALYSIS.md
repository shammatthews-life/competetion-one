# Foundation Model Zoo Architectural Analysis

| Model Pillar | Boost Type | Objective | Leaves / Depth | Subsample / Colsample | Regularization | OOF AUC | Determinism Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Conservative** | GBDT | binary | 31 / 6 | 0.80 / 0.60 | $\alpha=2.0, \lambda=5.0$ | 0.754136 | **100% Bitwise Match (1.0000000)** |
| **LightGBM ExtraTrees**    | GBDT (ET) | binary | 45 / 7 | 0.70 / 0.45 | $\alpha=5.0, \lambda=10.0$| 0.750626 | **100% Bitwise Match (1.0000000)** |
| **GPU Hist-XGBoost**      | Hist (CUDA)| binary:logistic | - / 6 | 0.80 / 0.60 | $\alpha=3.0, \lambda=8.0$ | 0.753403 | **100% Bitwise Match (1.0000000)** |
| **GPU CatBoost Symmetric** | Symmetric | Logloss | - / 6 | 1.00 / 1.00 | $L_2=5.0$ | 0.752590 | **Numerically Close (0.9997356)** |
| **LightGBM Deep**          | GBDT | binary | 127 / 8 | 0.80 / 0.65 | $\alpha=10.0, \lambda=30.0$| 0.752316 | **100% Bitwise Match (1.0000000)** |
