# Research Findings

## Established methods and implementation relevance

This project already has a strong, reproducible LightGBM/CatBoost baseline and nested target encodings. The following sources guide what is worth testing next; they do not imply an AUC gain on this dataset.

| Source | Established finding | Project consequence |
|---|---|---|
| [CatBoost paper](https://arxiv.org/abs/1706.09516) and [official GPU documentation](https://catboost.ai/docs/en/features/training-on-gpu) | Ordered boosting/categorical processing is designed to reduce prediction shift; GPU summation is nondeterministic. | Keep CatBoost as a fixed Phase 2 member; record GPU activity and compare by OOF, not exact GPU bit parity. |
| [LightGBM official parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#device_type) and [GPU tutorial](https://lightgbm.readthedocs.io/en/stable/GPU-Tutorial.html) | GPU support requires a backend-enabled build; a `device` flag cannot add missing backend support. | Keep the installed LightGBM path explicitly CPU-based. The current model blend's CatBoost is GPU-enabled. |
| [XGBoost official GPU guide](https://xgboost.readthedocs.io/en/stable/gpu/) | Current GPU training uses histogram trees and CUDA device selection, with memory-sensitive quantized data support. | Historical XGB10 parity is unresolved; another XGB sweep is lower value than validating fold-safe, complementary features. |
| [scikit-learn TargetEncoder documentation](https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder) and [common leakage pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) | Target-informed encoders/selection must be cross-fitted and fitted only inside each training partition. | Preserve Phase 2 nested OOF behavior. The new group-context features use no labels and are fit on outer-training covariates only. |
| Wolpert, [Stacked Generalization (1992)](https://doi.org/10.1016/S0893-6080(05)80023-1) | Meta-models need out-of-fold base predictions to avoid training on in-sample predictions. | Any model blend expansion must use nested/cross-fitted meta-features; do not fit weights to historical predictions. |
| Gorishniy et al., [TabM (ICLR 2025)](https://arxiv.org/abs/2410.24210), [author repository](https://github.com/yandex-research/tabm) | Parameter-efficient ensemble of MLP-like tabular predictors; broad benchmark evidence is not a guarantee of transfer. | Defer: PyTorch is absent in the active environment and a 6-GiB RTX 4050 makes feasibility/memory preflight necessary before installation/training. |
| Shimodaira, [covariate shift (2000)](https://doi.org/10.1016/S0378-3758(00)00115-4) | Importance weighting relies on covariate-shift assumptions. | Treat transductive/frequency adaptation as a separate protocol; report inductive and transductive evidence distinctly. |

## Experiment-specific hypothesis

The available data has no auxiliary bureau, installments, previous-application, POS_CASH, or credit-card tables. A relational-table feature study is therefore blocked by absent inputs, not by implementation time.

The current rowwise Phase 2 builder already supplies per-numeric-column missing indicators, an overall missing count, and EXT_SOURCE availability. A duplicate missing-indicator study was rejected before training.

Two controlled feature screens have now completed and both were rejected:

- **GC-01 category-peer context:** fold-local group means and individual deviations for income, education, occupation, and organization; own training-row measurements excluded. Matched OOF delta was `-0.0016302`, negative on all five folds.
- **TE-01 core categorical target encodings:** nested, smoothed encodings for organization, occupation, income type, and education; inner OOF maps exclude each training row's own target, and outer-validation/test encodings use outer-fit labels only. Matched OOF delta was `-0.0006403`, with 1/5 folds positive.

Both experiments recomputed feature matrices and candidate predictions from raw train/test CSVs; no historical predictions were used to generate candidate predictions. Existing baseline/Phase 1 OOF vectors are recorded as comparison-only inputs by the Phase 2 runner. Their output manifests, exact folds, OOF metrics, candidate integrity, and actual GPU telemetry were independently verified. Neither result justifies a second-seed replication or promotion.

The next experiment should not be another variation of these rejected categories or groupings. Require a distinct, predeclared hypothesis and a rationale for expected signal before spending another full five-fold run.

## Research limitations

- No TabM environment install or fit has been performed.
- No extra relational CSVs exist in the supplied raw-data directory.
- No historical submission or champion predictions are labels.
- No method is promoted without matched-fold OOF evidence and an independent-seed confirmation.
