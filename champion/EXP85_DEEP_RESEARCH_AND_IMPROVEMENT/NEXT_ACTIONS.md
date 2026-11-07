# Next Actions

1. Keep Phase 2 frozen as best; GC-01, TE-01, XGB-01/02 and E-INT-01 do not qualify for promotion.
2. Do not spend another seed or tune the rejected XGBoost blend or E_POLY features on the screening folds.
3. Exact next experiment: conduct the isolated TabM feasibility pilot defined in `OPPORTUNITY_RANKING.md`. Do not modify the validated Python environment; only proceed to a small CUDA pilot if an isolated install is compatible.
4. Do not start a full five-fold TabM run unless the 20k-row pilot uses at most 5.5 GiB VRAM and projects to at most two hours for five folds.
5. Auxiliary relational feature work is blocked until permitted Home Credit tables are actually available and hash/provenance-verified.
6. Continue historical lineage recovery separately. Never select or tune using historical test-prediction similarity.
7. Preserve immutable runs, regenerate and verify the package manifest after code/docs settle, then commit/push only verified work in this research workspace.

No candidate submission or official-score claim is authorized by these actions.
