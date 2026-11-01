# Negative Findings and Rejected Approaches

- **Champion-guided fitting/blending:** rejected. Historical test predictions are not labels; optimizing their match would overfit a reference and cannot establish default-risk generalization.
- **Calling V5 raw-only because it accepts raw CSV paths:** rejected. Its high-correlation route reuses historical suite arrays and a specific recovered Anchor74.
- **Treating same-named Anchor74 outputs as interchangeable:** disproven by distinct hashes and fixed-input swap correlations; exact producer/path/hash must be pinned.
- **Assuming successful raw component execution proves historical parity:** rejected. Rebuilt Track1B had Spearman `0.8921288944`; historical suite arrays lack trusted ID sidecars.
- **Duplicating missing indicators as a new Phase 2 feature test:** rejected before training. Phase 2 already has per-numeric-column `NA_*`, `RAW_MISSING_COUNT`, and `EXT_COUNT`.
- **GC-01 category peer means/deviations:** rejected after a matched five-fold raw-data screen. OOF AUC fell by `0.0016302`, with negative deltas on 5/5 folds and declines in both LightGBM and CatBoost; the feature block was influential but not helpful on these folds. No second seed was spent.
- **TE-01 core categorical target encodings:** rejected after a matched five-fold screen. OOF AUC fell by `0.0006403`; only 1/5 folds improved. Nested target leakage and artifact integrity checks passed, so the negative result is a model outcome, not a validation failure. No second seed was spent.
- **Relational-table aggregations:** blocked by input availability. The raw folder contains only application train/test/sample files; auxiliary tables are absent.
- **Immediate TabM installation/training:** deferred. PyTorch is absent; the RTX 4050 has 6 GiB; no feasibility budget was yet established.
- **Interpreting the correlation gap as an AUC or score gap:** invalid. The champion is an unlabeled test prediction reference.
- **Treating the supplied Stage 9 notebook printout as a complete integrity/evaluation audit:** invalid. The shown cell compares prediction arrays positionally without first proving unique/equal IDs or canonical row alignment, and its “0.801340 confirmed” line is not bound to an official evaluation response. Later independent ID-aligned analysis confirms V5 correlation separately.
- **GC-01 experiment implementation:** initial synthetic tests exposed that a finite-value guard incorrectly rejected expected NaNs. It was corrected to reject infinities only; all targeted tests then passed. This is a fixed pre-run test failure, not a model result.

All model experiments that run and fail will be retained with their failure report; none will be silently discarded or relabeled as successful.
