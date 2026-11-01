# Verified Current State

## Snapshot

- Git baseline before this experiment: `7ba27a45d86a2ee8bd7b3a8403f9dbb4912edc60`; only this research workspace is changed.
- Frozen Phase 2 candidate SHA-256: `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`.
- Raw train/test/sample hashes match the frozen release.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, driver `592.82`, 6,141 MiB VRAM; baseline snapshot showed 5,920 MiB free and 0% utilization.
- Python 3.12.10; LightGBM 4.7.0, CatBoost, XGBoost, NumPy, pandas, SciPy, scikit-learn installed; PyTorch absent.
- Available raw competition data in `D:\gci\inpuy` is only `train.csv`, `test.csv`, and `sample_submission.csv`; no bureau, installments, POS_CASH, prior-application or credit-card relational tables are present.
- Phase 2's selected model is a 50/50 LightGBM/CatBoost blend with fee/product features and nested target encodings. Its existing rowwise features already include numeric missingness flags, a raw missing count, and an EXT_SOURCE availability count.
- Existing seed-42 OOF vectors and exact fold indices are preserved in `EXP85_PHASE2_FROZEN_RELEASE/EVIDENCE/original_runs/phase_02/`.

## Evidence classes

### Executed and verified

- Frozen Phase 2 integrity verifier passes.
- The complete research test suite passes (11/11); the frozen nested target-encoding tests pass (3/3).
- The group-context feature builder passed a real-data first-fold preflight: 171,202 training rows, 61,500 test rows, 76 features, no infinite values, 9.35 seconds.
- GC-01 and TE-01 each completed a full five-fold screen on the exact frozen Phase 2 seed-42 folds. Both scored below Phase 2 and were rejected. Detailed metrics and integrity evidence are in `EXPERIMENT_REGISTER.md` and each immutable run directory.

### Historical report claims, not new evaluations

- Historical official score reported as `0.801340`; no current official evaluation was performed.
- Phase 2's previously reported matched OOF AUC is `0.7715614423`; second-seed reports show repeatable lift over the raw baseline.
- V5's `0.9999975816` correlation is a hybrid replay using cached V17/V55/XGB10 predictions.

### Supported inference

- The large Phase 2/champion test-prediction difference cannot be explained by blend weights alone: a fixed-input swap report isolates a major divergence to Anchor74 lineage, especially the different same-named Anchor74 artifacts.
- Historical cache lineage and missing row-ID bindings prevent a certified raw-only reconstruction of every original ensemble member.

### Unknown

- Whether a future, materially different hypothesis can improve generalization beyond frozen Phase 2; the two completed feature screens did not.
- The exact cause of each V17/V55/XGB10 cached-member difference; historical member arrays lack trusted ID sidecars and producer manifests.
- Official score of any independent candidate.

Neither experiment used the historical champion as training data or a selection target. The frozen Phase 2 OOF was used only as a matched validation reference.
