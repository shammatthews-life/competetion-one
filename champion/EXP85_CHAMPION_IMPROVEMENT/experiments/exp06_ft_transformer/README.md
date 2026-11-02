# EXP06: FT-Transformer

EXP06 trains an FT-Transformer over the Phase 2 fold-local feature pipeline.
Each numerical feature becomes its own learned token; each categorical feature
has a dedicated embedding token; a CLS token is passed through a multi-head
Transformer encoder and its output is used for binary classification. This is
an attention-based feature-interaction architecture, not a residual MLP.

The five outer folds and row indices are verified against the frozen Phase 2
and EXP05 fold artifacts. Inner validation is drawn only from outer-fit rows.
Frequency features and nested target encodings are regenerated from each
inner-fit/outer-fit partition. Preprocessing and category vocabularies are
fit on the corresponding training partition only. Inner validation selects
the number of epochs; the selected epoch count is then used to refit on the
entire outer-fit fold before predicting the outer validation and test rows.
Outer validation labels are not available to model selection or refitting.

## Matched benchmark

The archived Phase 2 OOF result used outer-validation early stopping, so it
is reported only as a historical, protocol-mismatched reference. EXP06
re-estimates the Phase 2 LightGBM/CatBoost fold-local blend on the same outer
folds using inner-only early stopping. Each base model's best iteration is
selected on inner validation and refit on the complete outer-fit fold. The
fixed benchmark is the arithmetic mean of these two refit models' OOF
probabilities. The blend weights are not tuned on outer validation.

The decision gate is predeclared: the FT-Transformer must exceed the matched
benchmark pooled OOF AUC by at least 0.001 and improve at least four of five
paired outer-fold AUCs. Failure means `REJECTED`; it never promotes or alters
the historical champion. EXP04 and EXP05 are secondary comparators only.
The RTX 4050 run configuration uses 16-dimensional tokens, two attention
heads, two encoder layers, a 64-wide feed-forward block, and batches of 2,048
to keep attention training within the device's runtime and memory budget.
The run completed and was rejected by the predeclared promotion gate; see
[RESULTS.md](./RESULTS.md) for fold-level results and the candidate hash.

## Run

Use the isolated CUDA environment installed for EXP05:

```powershell
champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\.venv\Scripts\python.exe champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp06_ft_transformer\run_experiment.py
```

By default, the runner reads `D:\gci\inpuy` and creates a timestamped
`runs\` directory under this experiment. Use `--data-dir` or `--run-dir` to
override those paths. The runner requires CUDA-enabled PyTorch and an RTX 4050,
performs a GPU forward/backward smoke fit, caps PyTorch allocator usage, and
fails rather than falling back to CPU. FT training runs sequentially without
mixed precision. It also regenerates the matched LightGBM/CatBoost baseline;
LightGBM follows the frozen Phase 2 CPU configuration and CatBoost uses the
RTX 4050.

Runs preserve OOF/test predictions, matched baseline predictions, fold
indices and metrics, model checkpoints, submission, telemetry, source/input
hashes, and a complete output SHA-256 manifest. No official leaderboard
result is inferred from local validation.
