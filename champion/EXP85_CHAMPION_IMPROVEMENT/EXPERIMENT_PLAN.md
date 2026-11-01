# Improvement experiment plan

Experiments are run only in unique directories under `experiments/runs`.
Historical cache arrays are never used as training labels; test predictions
are not used as a target or validation criterion.

## Completed evidence reused to avoid duplicate work

The sibling `EXP85_DEEP_RESEARCH_AND_IMPROVEMENT` workspace completed and
rejected:

- category peer context (GC-01);
- nested category encodings (TE-01);
- raw rowwise XGBoost, including an independent seed replication (XGB-01/02);
- five EXT_SOURCE/affordability interactions (E-INT-01).

Those candidates were tested against frozen Phase 2, not the historical
EXP85 OOF baseline, so they are not described as EXP85 improvements. They are
not rerun here.

## Ranked remaining opportunities

| Rank | Test | Rationale | Gate / limitation |
|---:|---|---|---|
| 1 | Regularized meta-stacker on the eight historical OOF suite streams | Directly tests whether the existing fixed 8-stream suite leaves usable OOF signal without retraining base models | Five outer folds; scaler and logistic model fit only on each outer-fit partition; one predeclared `C=0.1`; require mean paired AUC lift >= 0.0002, at least 4/5 positive folds, and lift > 2 standard errors |
| 2 | Additional raw-data neural/TabM model | Potentially distinct signal | Existing Python 3.12 environment lacks CUDA-enabled PyTorch; current `gemini` virtual environment has CPU-only PyTorch. Defer a full run until an isolated CUDA environment is available and validated. |
| 3 | New relational features from auxiliary competition tables | Could add substantial signal | Required auxiliary tables are absent from the supplied data; no unverified download or fabricated input. |
| 4 | Retune full EXP85 weights against historical test predictions | Not valid | Explicitly prohibited: prediction similarity to the historical test output is not a generalization target. |

The first experiment uses cached OOF arrays supplied by the historical
pipeline. Their exact original fold sidecars and producer run manifest are
unavailable, so the result is classified as **historical-cache-assisted
meta-validation**, not certified independent raw reconstruction. If the
predeclared gate fails, no candidate submission is generated and no seed or
parameter sweep follows from this experiment.

OOF lifts from the eight-engine suite cannot be directly equated to the full
EXP85 champion OOF metric: the original 85%-weight Anchor74 has no verified
aligned OOF vector. Official score confirmation still requires a public
competition evaluation.

## Executed result

EXP01 completed with `REJECTED_SCREEN`. Fixed-suite pooled OOF AUC was
`0.7772070078`; cross-fit meta pooled OOF AUC was `0.7771115234`. The mean
paired fold delta was `-0.0000813250` (standard error `0.0000494806`), with
only one of five folds positive. The gate failed, so **no test candidate was
generated**. The report and fold-level outputs are preserved under
`experiments/runs/20261101T175035Z_exp01_oof_meta_weights/`.

The fixed-weight comparison is not fully nested because those historical
weights were selected using the complete OOF label set; this makes the fixed
comparator optimistic. The result is a conservative rejection, not proof that
every possible stacker is inferior.

No parameter or seed sweep will follow this result. The next viable evaluation
must either establish leakage-safe OOF predictions for the 85%-weight
Anchor74 component or introduce a genuinely new raw-data model whose
contribution can be assessed without using historical test predictions as a
tuning target. In the checked `D:\gci\gemini\.venv`, PyTorch is CPU-only
(`2.14.1+cpu`, CUDA unavailable); no GPU neural experiment was run.
