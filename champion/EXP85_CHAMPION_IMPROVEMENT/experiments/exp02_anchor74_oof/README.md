# EXP02: Fold-safe raw Anchor74 surrogate OOF

This experiment does **not** claim to reproduce historical Anchor74. The
historical Anchor74 consumes a previously trained optimal-transductive base
prediction and a five-family zoo; that complete upstream stack has no
verified, leakage-safe OOF outputs or fold provenance. The available raw
retrainer uses a different feature/model pipeline and is therefore evaluated
only as an independent raw-data surrogate.

The runner reuses the tracked raw feature builder, which prints “143-feature”
but actually returns **695 columns**. This observed discrepancy is recorded
rather than silently assuming the header is correct. The runner trains five
base-model families on fixed outer folds, and never uses an outer validation label for
training, early stopping, feature selection, or graph calibration. The
builder's unsupervised applicant-wide transforms use covariates only. Repeat-
borrower calibration and applicant-pair smoothing are recomputed independently
inside each outer fold using only outer-fit labels.

The folds use `StratifiedGroupKFold` keyed by the exact entity key used by the
historical repeat-borrower graph. This intentionally differs from row-wise
folds in prior experiments because group isolation takes priority over exact
fold parity.

The suite OOF arrays used for the diagnostic 85:15 blend are historical
prediction caches. Their exact original fold sidecars are unavailable, so
the blend is classified as hybrid-cache-assisted validation. It is not a
validated EXP85 candidate, and this experiment does not emit a submission.

Run from the improvement workspace:

```powershell
python experiments\exp02_anchor74_oof\run_anchor74_oof.py
```

The runner requires the raw competition CSVs, the tracked source
`champion/02_EXP85_REBUILT/source/generate_anchor74_from_scratch.py`, the
frozen suite cache artifacts, and working CUDA support for XGBoost and
CatBoost. It fails instead of silently falling back to CPU.
