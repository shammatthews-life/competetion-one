# Ranked Research Opportunity Matrix

Scores are qualitative priority judgments, not predicted AUC gains. “Potential” means usefulness for resolving a blocker or producing a robust independent model, not an invented leaderboard improvement.

| Rank | Opportunity | Potential usefulness | Evidence | Reproducibility | Complexity | GPU cost | Leakage risk | Priority rationale |
|---:|---|---|---|---|---|---|---|---|
| 1 | Bind historical cache arrays to IDs and producer/config hashes | Very high for historical lineage | Exact cached arrays/hashes exist; arrays lack embedded IDs | Low until producer evidence recovered | Medium | None | Medium if positional order assumed | Without a trusted ID sidecar, member-level comparisons cannot be certified or safely replaced. Recover manifests/source and establish ordering before modeling. |
| 2 | Resolve Anchor74 ancestry and same-name artifact mismatch | Very high for fidelity | Historical Anchor `315b...`; current raw Anchor `6c32...`; swap analysis isolates large delta | Medium if historical inputs exist | High | Medium | High for transductive replay | Anchor is 85% of final probit blend; current raw route is not an equivalent producer. |
| 3 | Nested reproduction of historical feature-screening protocol | High for fair validation and fidelity | 669/550 hash parity is reported; nested benchmark source exists | Medium | High | CPU-heavy in current build | High if global target-guided screen | Separate historical fidelity from unbiased CV estimates; audit selection fold scope. |
| 4 | Third-seed replication of frozen Phase 2 | High for stability | Two seeds and 5/5 paired-fold lifts already exist | High | Low–medium | CatBoost GPU; ~22 min per prior full gate | Low with same gated source | Most direct low-risk robustness test; no new model idea or champion fitting. |
| 5 | Feature-family ablations within Phase 2 | High for interpretability | Fee decoder ablation already +0.00319 on second seed; nested encodings have self-exclusion tests | High | Medium | CatBoost GPU; repeated folds | Low | Learn which block contributes; one block at a time, predeclare seeds. |
| 6 | Complementarity of raw XGBoost to Phase 2 | Medium | A historical Phase 3 blend gained only ~0.000134 on one reported seed and was inconsistent (3/5 folds) | High if raw Phase 2 outputs are reused as base OOF correctly | Medium | CUDA, memory-bound | Medium (stacking) | Re-evaluate only with proper OOF stacker and independent seed; not broad tuning. |
| 7 | Missingness-aware Phase 2 ablation | Medium | Historical Anchor uses EXT_SOURCE regimes; native missing handling is available | Medium | Medium | CPU/GPU varies | Low–medium | Could test one predeclared regime interaction feature block against Phase 2. |
| 8 | Strict inductive vs transductive fee-frequency test | Medium | Fold-local version retained nearly all Phase 2 lift | High | Low | None | Low | Establish deployment scope; the fold-local gate already makes pooled transductive use unnecessary for the next candidate. |
| 9 | Rebuild V17/V55 from their exact historical feature views | Medium for lineage; uncertain for parity | Sources and full raw runs recovered; historic arrays lack ID metadata | Low–medium | High | Mixed; LightGBM CPU, Cat/XGB GPU | Medium | First recover producer, feature schema, and row order; retraining without them risks another unrelated raw model. |
| 10 | TabM feasibility and complementary model | Uncertain | Published research exists; no local PyTorch and no prior EXP85 execution evidence | Low until dependencies/runtime tested | High | High, 6-GiB cap | Medium | Fashionable is not a reason to run; consider only after top lineage and ablation tasks. |
| 11 | Graph/borrower diffusion | Uncertain | Historical scripts describe it; exact identity-link quality is not certified | Low–medium | High | CPU feasible | High | Requires verified links, fold-local graph, and negative controls; current evidence insufficient. |
| 12 | Optimize blend to maximize champion correlation | Reject | Historical test reference exists but is not labels | Not acceptable | — | — | Prohibited | Would overfit to test predictions and would not establish better default-risk generalization. |

## Recommended immediate order

1. Run the predeclared ID-aligned test-prediction gap slices (no fit/tuning).
2. Recover exact Phase3/V17/V55/XGB10 source revision, producer manifests, and ID order sidecars.
3. If lineages are clean, run a third-seed independent Phase 2 stability check or a single-block Phase 2 ablation.
4. Only then consider one complementary raw XGBoost experiment; keep TabM and graph methods later.
