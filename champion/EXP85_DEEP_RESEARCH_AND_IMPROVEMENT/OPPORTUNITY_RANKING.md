# Opportunity Ranking

Rank reflects expected value for independent validated performance, execution cost, and evidence quality—not novelty.

| Rank | Opportunity | Expected value | Cost | Main risk/blocker | Decision |
|---:|---|---|---|---|---|
| 1 | Recover historical cache producer/ID sidecars | Very high for historical fidelity, none guaranteed for new OOF AUC | Medium forensic effort | Original manifests/IDs may not exist | Continue separately; do not fit against champion predictions. |
| 2 | Third-seed Phase 2 stability check | High for confidence, not necessarily direct improvement | About one prior validation-gate run | Uses an additional model-selection look; predeclare seed before running | Defer until a specific uncertainty justifies it; Phase 2 remains frozen. |
| 3 | One-feature-block ablations within Phase 2 | Medium; improves attribution and may simplify model | One additional five-fold run per block | Avoid selecting among many ablations on one seed | Avoid broad ablation sweeps; existing controls are evidence. |
| 4 | Cross-fitted raw XGBoost complementarity | Low/medium; prior addition was small and inconsistent | GPU/memory cost, stacker complexity | Outer-fold meta leakage and seed instability | Defer until a component has stable OOF value. |
| 5 | Auxiliary-table aggregations | Potentially high for Home Credit | High | Current supplied input directory lacks all auxiliary tables | Blocked unless licensed/competition-permitted data is located. |
| 6 | Nested core categorical target encodings | Low after evidence | One five-fold screen already completed | Reduced AUC; only 1/5 folds improved | TE-01 rejected; do not tune smoothing/category definitions on the same OOF seed. |
| 7 | Category peer means/deviations | Low after evidence | One five-fold screen already completed | Reduced AUC on all five folds | GC-01 rejected; do not tune the groups on the same OOF seed. |
| 8 | TabM or FT-Transformer | Uncertain | High GPU/runtime and dependency cost | PyTorch absent; 6-GiB GPU; no feasibility budget was yet established | Defer pending feasibility and stronger evidence for complementarity. |
| 9 | Historical prediction-vector matching / champion-guided blend tuning | Not an acceptable generalization objective | — | Test-reference overfit; not labels | Reject. |

## Decision protocol

Use the frozen Phase 2 model as the reference. Screens use the exact original seed-42 folds. A positive same-seed result is only screening evidence; any candidate for promotion requires a predeclared independent seed with fold-local leakage checks and stable paired-fold deltas.
