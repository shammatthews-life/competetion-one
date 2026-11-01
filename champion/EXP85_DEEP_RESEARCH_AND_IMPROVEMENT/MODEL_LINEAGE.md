# Model Lineage and Reproducibility

| Component generation | Inputs / model family | Reported folds, seed, OOF | Current evidence | Classification / unresolved dependency |
|---|---|---|---|---|
| Hasan zoo: LGBM conservative | 550 screened features, LightGBM | 5-fold, seed 42; OOF ~0.7541 | Phase 3 source and run reports; raw reconstruction reported exact test array | Source recovered; historical feature selection and model/cache provenance must be pinned. |
| Hasan zoo: LGBM ExtraTrees | 550 screened features, LightGBM ExtraTrees | 5-fold, seed 42; OOF ~0.7506 | Phase 3 source and reports | Similar; historical array lacks ID sidecar. |
| Hasan zoo: XGBoost | 550 screened features, XGBoost CUDA | 5-fold, seed 42; OOF ~0.7534 | Raw reconstruction reported exact array | Distinct from XGB10. |
| Hasan zoo: CatBoost | 550 screened features, CatBoost GPU | 5-fold, seed 42; OOF ~0.7526 | Rebuild OOF ~0.75234, reports show small nonzero delta | GPU summation is nondeterministic per CatBoost docs; model parity is not exact. |
| Hasan zoo: Deep LGBM | 550 screened features, LightGBM | 5-fold, seed 42; OOF ~0.7523 | Rebuild was sensitive to preserving native infinities | Infinities vs NaNs are semantically different; exact run hash/config is required. |
| Anchor74 | Transductive postprocessing of prior apex + zoo + borrower covariates | No model CV at final postprocessing step | Historical artifact and reconstructed Phase 3 hash are distinct; raw production direct Anchor Spearman vs historical Anchor ~0.8691 | Successful V5 route uses specific Phase 3 Anchor + cached suite; full prior Level-2/optimal ancestors not certified raw-only. |
| V17 suite | V16 and E1/E1A views; LGBM/ExtraTrees/CatBoost and rank fusion | 5-fold, seed 42; lineage note reports member OOF ~0.760–0.768; final ~0.776 | Full raw generator executed; output bundle hash `41911ba0...` | Historical bundle has no row IDs; upstream V16 views and historical cache binding unresolved. |
| V55 suite | seven member views; LightGBM and CatBoost product model | 5-fold, seed 42; lineage note reports member OOF ~0.750–0.766 | Full raw generator executed; output bundle hash `bff5dd29...` | Historical bundle has no row IDs; original member caches/producer manifests not fully recovered. |
| XGB10 | 239 financial/demographic features; XGBoost hist | 10-fold, seed 42; historical report OOF 0.774331 | Full raw generator executed with CUDA; recovered OOF AUC 0.7609519 | Not exact parity; historical test array has no IDs. |
| Independent Phase 2 | Raw application, fee/product features, nested smoothed target encodings; 50/50 LightGBM/CatBoost | Outer five-fold seed 42 OOF AUC 0.7715614423; outer seed 20261101 repeats at pooled 0.7711413866 / fold-local 0.7709638204 | Frozen, verified, candidate SHA `2470ce...7411fa` | Reproducible raw-only development candidate; official score not evaluated. |

## Historical-cache status

### Confirmed cache dependencies

- V5's high-correlation route uses historical V17/V55/XGB10 arrays plus the exact Phase 3 Anchor74 artifact.
- Phase 3 final synthesis source directly reads those historical suite arrays (including fallback path resolution).
- The historical precision Anchor writer consumes an earlier optimal apex and `hasan_expanded_zoo.npz`; it is not a raw-to-final end-to-end trainer.
- Historical NPZ/NPY suite components do not carry `SK_ID_CURR` arrays or cryptographically bound row-order sidecars. Comparisons are positional and cannot qualify as ID-aligned until binding evidence is recovered.

### Raw scripts that execute are not proof of cache parity

The recovered XGB10, V55, V17, and direct Anchor74 scripts completed full raw executions in `EXP85_FINAL_RECONSTRUCTION`. Their outputs demonstrate executable raw pipelines. They do not reproduce the historical cached arrays: the final Track1B raw composition correlated only `0.8921288944` Spearman with the champion. LightGBM ran CPU because the installed build lacks GPU tree support; XGBoost/CatBoost GPU execution was reported and logged for the applicable runs.

## Missing lineage needed for full raw rebuild

1. Exact producer revisions/configurations and fold IDs for historical V17/V55/XGB10 bundles.
2. A trusted `SK_ID_CURR` binding for each historical cached member array.
3. Full ancestry and raw-only producer for the initial Level-2/base submission and optimal transductive apex consumed by Anchor74.
4. The historically selected 550-feature list and its exact selection protocol, separated from an unbiased nested feature-selection CV estimate.
5. Exact input/run identity that explains why same-name Anchor74 files differ (`315b...` historical route versus `04de...`/`6c32...` production routes).
6. Archived official platform evaluation record that binds the reported score to the uploaded artifact hash.

## Historical hardware findings

- Machine detected during this research: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB, driver 592.82.
- Current base Python has XGBoost, CatBoost, and LightGBM installed; PyTorch/TabM are absent.
- Local prior execution evidence reports CatBoost GPU and XGBoost CUDA for recovered full runs. Current LightGBM build lacks GPU tree learner. No model was trained in this stage.
- CatBoost GPU is explicitly documented as nondeterministic due to floating-point summation order. A GPU flag alone is insufficient; capture fit log/device telemetry per run.
