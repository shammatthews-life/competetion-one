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
| 1 | Fold-safe raw-data Anchor74 surrogate OOF (EXP02) | Tests five independently trained raw-data model families with group-isolated folds | Five-fold diagnostic completed; nonhistorical surrogate and not eligible for submission |
| 2 | Additional raw-data neural/TabM model | Potentially distinct signal | Existing Python 3.12 environment lacks CUDA-enabled PyTorch; current `gemini` virtual environment has CPU-only PyTorch. Defer a full run until an isolated CUDA environment is available and validated. |
| 3 | New relational features from auxiliary competition tables | Could add substantial signal | Required auxiliary tables are absent from the supplied data; no unverified download or fabricated input. |
| 4 | Retune full EXP85 weights against historical test predictions | Not valid | Explicitly prohibited: prediction similarity to the historical test output is not a generalization target. |

EXP01 used cached OOF arrays supplied by the historical pipeline. Their exact
original fold sidecars and producer run manifest are unavailable, so the
result is classified as **historical-cache-assisted meta-validation**, not
certified independent raw reconstruction. The predeclared gate failed; no
candidate submission was generated and no seed or parameter sweep followed.

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
historical Anchor74 component or use a genuinely new raw-data surrogate without
using historical test predictions as a tuning target. EXP02 implements the
latter diagnostic path; it does not claim to reconstruct Anchor74.

## EXP02 execution gate

The runner is implemented in
`experiments/exp02_anchor74_oof/run_anchor74_oof.py`. It uses five fixed
outer folds grouped by the exact historical repeat-borrower entity key,
disables early stopping, and recomputes graph calibration using only each
fold's training labels. The raw feature builder returns 695 columns despite
printing a 143-feature description. This is documented and validated rather
than silently reconciled.

The grouped-fold and calibration tests pass; actual raw-data split preflight
confirmed zero entity overlap in every fold. Two prior partial training runs
were invalidated and their logs/telemetry retained with abort reports. The
complete five-fold run completed as `20261101T183230Z`. Every fold had zero
group overlap and complete finite predictions. Pooled AUCs were `0.7597101`
for the graph-calibrated raw surrogate, `0.7643409` for the 85:15 surrogate/
cache blend, and `0.7771967` for the cross-fit two-stream logistic diagnostic.
The 85:15 result is not comparable to the true historical EXP85 pipeline
because suite OOF fold lineage is unavailable. The report, OOF arrays, fold
indices, and telemetry are preserved under
`experiments/runs/20261101T183230Z_exp02_anchor74_oof_surrogate/`.

The experiment remains diagnostic-only: there is no verified historical
Anchor74 OOF baseline, so no submission is permitted. The next valid full
85:15 evaluation requires leakage-safe predictions from the actual historical
Anchor74 upstream stack and verified fold provenance for the suite streams,
or independent retraining of those historical producers.

In the checked `D:\gci\gemini\.venv`, PyTorch is CPU-only
(`2.14.1+cpu`, CUDA unavailable); no GPU neural experiment was run.
