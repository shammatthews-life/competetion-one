# EXP04: GPU CatBoost Lossguide screen

This experiment tests one distinct model architecture: CatBoost's asymmetric
`Lossguide` tree growth. It uses raw competition application data and the
frozen Phase 2 fold-local feature pipeline, including its outer-fit-only
frequency features and nested target encodings.

The five folds and seed match the independently trained Phase 2 second-seed
reference. Its OOF comparator is used only for validation and never as a model
input or label. The frozen champion CSV is read only after training for a
clearly labeled similarity diagnostic.

Iteration selection uses a stratified inner split drawn only from each outer
fit partition. The selected iteration count is then refit on the complete
outer-fit partition; the outer validation labels are not used for early
stopping, feature generation, or fitting.

The frozen Phase 2 comparator's original training run selected stopping
iterations using each outer validation fold. Its OOF reference may therefore
be optimistic relative to this stricter candidate protocol; the reported
paired comparison is conservative, not a basis for promotion.

The RTX 4050 is required. The runner performs an explicit GPU/VRAM preflight,
configures CatBoost `task_type=GPU`, caps its requested GPU memory fraction,
samples `nvidia-smi` during every fit, and verifies the effective CatBoost
device and tree policy. It fails instead of falling back to CPU.

Run from the repository root:

```powershell
python champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp04_catboost_lossguide\run_experiment.py
```

The predeclared screen advances only if the pooled OOF AUC improves by at
least `0.001` over the matched Phase 2 reference and at least four of five
paired folds improve. Passing this screen would require a separate seed
replication before promotion. It cannot establish an official leaderboard
score.

## Completed run

The leakage-safe run is
`runs/exp04_lossguide_strict_seed20261101/`. It used 122 features, the RTX
4050 for all five folds, 205 GPU telemetry samples, and at most 4,041 MiB of
VRAM. Total runtime was 156.4 seconds.

Pooled OOF ROC-AUC was `0.7680327122`, versus `0.7709638204` for the
same-fold Phase 2 OOF vector, a delta of `-0.0029311082`. All five paired
folds declined, so the predeclared gate rejected the model. The first run,
`runs/exp04_lossguide_seed20261101/`, is retained but marked invalid because
its outer validation labels selected the stopping iteration.

The strict run produced a schema-valid 61,500-row diagnostic candidate at
`runs/exp04_lossguide_strict_seed20261101/submission_catboost_lossguide.csv`,
SHA-256
`a718e1b55e319a35ffe9945d1a7fa222e06eb1794e59aa26e7ce8579f4b31ef9`.
Its Pearson/Spearman correlations with the historical champion are
`0.780166`/`0.910521`; these are post-training diagnostics only. The
candidate is not promoted, and no official competition score was measured.
