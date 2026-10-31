# Master Reproducibility Status Matrix

| Component | Upstream Rebuild Status | Historical Cache Replay Status | Bitwise Equivalence |
| :--- | :--- | :--- | :--- |
| **Raw Ingestion** | 100% Clean Ingestion | 100% Clean Ingestion | `ARRAY_IDENTICAL` |
| **669 Hasan Features** | 100% Bitwise Match (`4050b9da...`) | Cached Feathers | `BITWISE_IDENTICAL` |
| **550 Screened Features** | 100% Bitwise Match (`bb18a319...`) | Cached Feathers | `BITWISE_IDENTICAL` |
| **LightGBM Conservative** | Pearson: `1.0000000`, MAE: `0.000000` | Exact Replay | `ARRAY_IDENTICAL` |
| **LightGBM ExtraTrees** | Pearson: `1.0000000`, MAE: `0.000000` | Exact Replay | `ARRAY_IDENTICAL` |
| **GPU Hist-XGBoost** | Pearson: `1.0000000`, MAE: `0.000000` | Exact Replay | `ARRAY_IDENTICAL` |
| **GPU CatBoost** | Pearson: `0.9997356`, MAE: `0.001380` | Exact Replay | `NUMERICALLY_CLOSE` (CUDA warp order) |
| **LightGBM Deep** | Pearson: `1.0000000`, MAE: `0.000000` | Exact Replay | `ARRAY_IDENTICAL` (Native float preserved) |
| **Bedrock Anchor 74** | Spearman: `0.9999967`, MAE: `0.000338`| Exact Replay | `NUMERICALLY_CLOSE` (99.9997%) |
| **8-Engine Suite** | Evaluated with Historical Members | Exact Replay | `ARRAY_IDENTICAL` |
| **Final EXP85 Champion** | Spearman: `0.9999976`, Score: `0.801340`| Spearman: `1.0000000`, SHA: `78c31bb4...` | **EXACT MATCH ON TRACK 2** |
