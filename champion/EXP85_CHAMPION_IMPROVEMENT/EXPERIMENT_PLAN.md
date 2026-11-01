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

## EXP03: reconstruct the five-family zoo dependency

The historical zoo file exists but has no verified producer manifest, so it
is treated as a parity reference only. EXP03 runs the hash-verified historical
669-feature builder, 550-feature LightGBM selector, three-family trainer, and
CatBoost/deep-LightGBM expansion from raw competition inputs in a fresh cache.
The runner records split indices, exact input/source/output hashes, CUDA
preflight and telemetry, schemas, fixed seeds, and per-array parity metrics.
It must not load the historical predictions into its training runtime.

The follow-on Anchor74-to-EXP85 execution is explicitly hybrid: it uses the
rebuilt zoo but begins from available historical 080041/080150 inputs and
reuses the existing V17, V55, XGB10, and Grand Master 79 artifacts. This
integration can test whether the original downstream code executes and can
measure parity; it cannot establish complete raw-to-final lineage while the
remote neural/synthesis artifacts referenced by `reproduce_final_submission.py`
remain unavailable.

### EXP03 result

The raw rebuild completed in 1,019.2 seconds; all four historical source
stages exited successfully. Generated 669-feature and selected-550 matrices
were exact against their historical references, including the original
`HOUSING_STABILITY` infinities. Eight of ten zoo prediction streams were
exact. CatBoost GPU was not bitwise reproducible under a fixed seed: fold 2
OOF AUC changed from `0.751470` to `0.751814`; Deep LightGBM repeated exactly.
The second CatBoost run was very close to the historical cache but not exact,
and was retained as a repeatability result rather than selected to tune parity.

The isolated downstream pipeline reproduced 080041/080150/080200 exactly and
generated all later Anchor74 artifacts with complete IDs, high but non-exact
correlation. The first run stopped at the original final guard (mean shift
149.4283; required 150–250). A separate integration replay on the second raw
zoo passed all seven downstream scripts with the source assertion/formula
unchanged and wrote its 85:15 output only under the ignored run runtime.
That output has 61,500 aligned IDs, hash
`13b24617a8fd5eafd51eefe6f1a33b290b4efc5be157125cdb15648d3f1763c0`,
Spearman `0.9999999872` and MAE `0.00003163` against the frozen champion.
It is a hybrid replay using historical anchors and suite bundles, not an
independent submission, and has not been evaluated on the official leaderboard.
