# Opportunity Ranking

Rank reflects expected value for independent validated performance, execution cost, and evidence quality—not novelty.

| Rank | Opportunity | Expected value | Cost | Main risk/blocker | Decision |
|---:|---|---|---|---|---|
| 1 | TabM feasibility pilot in an isolated Python environment | Potentially complementary; actual benefit unknown | Dependency/CUDA smoke test plus small-sample runtime/VRAM pilot before any five-fold run | PyTorch is absent; do not alter the validated environment or install into it | Exact next experiment: verify isolated CUDA-compatible PyTorch/TabM setup, then pilot on a fixed 20k-row sample; stop if GPU is unavailable, memory exceeds 5.5 GiB, or projected five-fold runtime exceeds 2 hours. |
| 2 | Obtain permitted auxiliary Home Credit tables | Potentially high information gain | Data provenance/license/hash and schema audit before feature work | Current supplied input directory contains no relational tables | Blocked until permitted source files are actually present; do not invent or silently download data. |
| 3 | Recover historical cache producer/ID sidecars | Very high for historical fidelity, none guaranteed for new OOF AUC | Medium forensic effort | Original manifests/IDs may not exist | Continue separately; do not fit against champion predictions. |
| 4 | Third-seed Phase 2 stability check | High for confidence, not direct improvement | One full validation-gate run | Additional model-selection look | Phase 2 is already independently validated; defer absent a concrete uncertainty. |
| 5 | Nested core categorical target encodings | Low after evidence | One five-fold screen already completed | Reduced AUC; only 1/5 folds improved | TE-01 rejected; do not tune smoothing/category definitions on the same OOF seed. |
| 6 | Category peer means/deviations | Low after evidence | One five-fold screen already completed | Reduced AUC on all five folds | GC-01 rejected; do not tune the groups on the same OOF seed. |
| 7 | Raw XGBoost as another ensemble member | Low after two seeds | Two 30.7-second screens completed | About 0.010 lower standalone AUC; blend effects tiny/inconsistent | XGB-01/02 not promoted; no weight sweep or third seed. |
| 8 | E_POLY interactions | Low after matched screen | One five-fold screen completed | OOF declined; only 2/5 folds improved | E-INT-01 rejected; no second seed or same-fold retuning. |
| 9 | Historical prediction-vector matching / champion-guided blend tuning | Not an acceptable generalization objective | — | Test-reference overfit; not labels | Reject. |

## Decision protocol

Use the frozen Phase 2 model as the reference. Screens use the exact original seed-42 folds. A positive same-seed result is only screening evidence; any candidate for promotion requires a predeclared independent seed with fold-local leakage checks and stable paired-fold deltas.
