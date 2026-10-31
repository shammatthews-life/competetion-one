# Model Configuration and Parity Report

## Current-source configuration

The source snapshot executed is commit `48b66288dfb262b1a1572bace1e70e41da573ef6`.
The generated arrays are for 171,202 raw training rows and 61,500 raw test rows.
Commands omit `--fast`; all full-mode tree/iteration settings are used.

| Component | Family and full-mode configuration | Folds / seeds | Early stopping / calibration | Device |
|---|---|---|---|---|
| XGB10 | Hist XGBoost, 450 estimators, learning rate .035, depth 6, subsample .80, column sample .55, alpha 3, lambda 8, minimum child weight 30 | 10-fold stratified shuffled; splitter seed 42; model seed 42 + fold | 30-round early stopping; prediction probabilities; final formula rank/probit normalization | CUDA verified by GPU-only preflight and generator device log |
| V55 LightGBM members | Six full-mode members: `qf_shallow` 1300 rounds (`lr=.05`, leaves 7, depth 3); `qf_reg` 1939 (`.03`, leaves 11); `v1_allfam_oh` 1563 (`.04`, leaves 15); `v2_allfam_oh` 1500 (`.03`, leaves 15); `v2_mono` 1571 (`.02`, leaves 15); `deep_et` 1200 (`.03`, leaves 31). The per-member minimum child, feature/bagging fractions, L2, extra-trees, and monotonic constraint settings are source-defined. | 5-fold stratified shuffled; splitter seed 42; per-member LightGBM seeds 42–47, incremented by fold | 30-round early stopping; fold-averaged test estimates; rank normalization for suite outputs; `use_full_retrain=False` | CPU: installed LightGBM build reports `GPU Tree Learner was not enabled in this build`; the GPU test exited 1 rather than falling back |
| V55 CatBoost | Product specialist, 400 iterations, depth 6, learning rate .045, Logloss, AUC metric, L2 8 | Same 5 folds; CatBoost seed 42 + fold | 30-round early stopping; ranked test output | GPU confirmed by preflight and generator device log |
| V17 LightGBM streams | `old`: 700 rounds, `.04`, 15 leaves; `e1`: 750, `.035`, 23 leaves; `e1et`: 800, `.03`, 31 leaves with extra-trees; `e1Aet`: 900, `.025`, 45 leaves with extra-trees. Other per-stream leaf, feature/bagging, L2 and seed values are source-defined. | 5-fold stratified shuffled; splitter seed 42; stream seeds 42/52/62/72 plus fold | 30-round early stopping; member OOF/test rank transforms; final OOF/test fusion weights `.35/.25/.20/.10/.10` | CPU: installed LightGBM build has no GPU tree learner |
| V17 CatBoost | 500 iterations, depth 6, learning rate .04, Logloss, AUC metric, L2 6 | Same 5 folds; CatBoost seed 82 + fold | 30-round early stopping | GPU confirmed by preflight and generator device log |
| Anchor74 source snapshot | 143-feature Hasan financial/copula/interaction manifold; LightGBM conservative (700 rounds), LightGBM ExtraTrees (750), Hist XGBoost (500), CatBoost (450), and deep LightGBM (750) | 5-fold stratified shuffled seed 42; feature transform also uses 5-fold KFold seed 42; component model seeds 42–46 | 30-round early stopping where configured; combined prediction is ranked/transductive before output | XGBoost CUDA and CatBoost GPU confirmed; LightGBM CPU-only build; ExtraTrees is CPU |

These settings describe the source executed here, not the historical producer
configuration. Full per-member settings, code hashes, run commands, and actual
execution results are retained in the run logs and `execution.json`.

The current fold IDs were independently reconstructed from raw row order and
the source seeds. The 5-fold stratified assignment shared by V55, V17, and
Anchor74 has fold-ID SHA-256
`833b6a22ff4014b7f064eb2e3df2d8ef03d0db9420e1ed6c09ef0de038ef1de5`; XGB10's
10-fold assignment is
`83c086add0341aa489dd921ea5c042727a8a78917acd8342871056533ed3cf5a`. The
Anchor74 transductive Ridge-residual KFold spans all 232,702 train+test feature
rows and hashes to
`f2154c7257c713ebc60af75c7e0429951dfa8bf6158df2e6c03f93455b0782c3`. The
full counts and raw ID/label hashes are in `runs/raw_full_gpu_20261031_235900/reports/fold_assignments.json`.

## Historical parity assessment

| Question | Proven evidence | Remaining unknown / conclusion |
|---|---|---|
| Same raw source rows and labels? | Raw input hashes and exact generated `y_train.npy`/raw `TARGET` equality are recorded. The historical cached label vector is also checked after candidate generation. | Historical prediction arrays have no row-ID sidecars; positional comparison does not independently prove their test row identity. |
| Same feature definitions and order? | The Phase 3 669-feature matrices and selected-550 matrices were previously reported byte-identical to their historical counterparts. | The Git snapshot executed here builds a 143-feature Anchor74 matrix in memory and does not persist it. A separate local production-lineage report describes a 695-feature matrix and a first shared-feature divergence at `SOLVENCY`; that source is a different snapshot. No 143/669/550/695 feature matrices were compared cellwise in this execution. |
| Same folds and assignments? | Current source explicitly sets fold count, shuffle, and seed; the current labels and raw row order are verified. | Original historical fold membership files are not available for all current/historical suite members. Same seed text does not prove same feature rows or historical assignments. |
| Same model families/parameters? | Current model families and parameters are source-defined and hashed. The selected suite members and weights match the historical v85 formula. | Historical V17/V55/XGB model producer configurations are incomplete; saved predictions are not a model checkpoint. |
| CatBoost cause resolved? | Five same-seed GPU fold-5 repeats were stable and historical-like; that audit recorded the historical input matrix, settings and row map. | The earlier two-run discrepancy was not reproduced, and CPU/GPU settings were not an apples-to-apples control. Cause remains unknown. |
| Early stopping and calibration? | Current source uses 30-round early stopping for the listed boosting components; the final suite is rank/probit normalized. | Historical per-fold stopping rounds and any historical calibration beyond saved predictions are not fully known. |
| Transductive processing? | The current Anchor74 builder uses raw train and test data together for feature generation and repeat-borrower reasoning; its output IDs are validated against raw test order. | Historical transductive transforms and intermediate artifacts are not fully rebuilt from raw data in this run. |
| Existing-model inference comparison? | Historical caches are available for read-only post-run prediction comparison; no historical model checkpoint is available for inference-only control. | Prediction-array comparison is not equivalent to loading the original fitted model and proving parity. |
| Final rank/blend? | Historical source and the isolated implementation use the same eight weights, 85/15 anchor/suite probit blend, and ordinal projection; independent candidate recomputation checks the exact output formula. | Matching the formula cannot compensate for changed upstream predictions. |

## Full raw-run output comparisons

The four full-mode generators completed with exit code 0. The isolated candidate
was then synthesized only from their new outputs. Its formula was independently
recomputed to a maximum absolute error of `4.96e-11` after ten-decimal CSV
rounding. The candidate's Spearman correlation with the ID-aligned frozen
champion is `0.891570`; the prior raw candidate comparison is `0.902144`.
Neither correlation is an official leaderboard score.

All 27 fresh-vs-historical OOF/test member-array comparisons are in
`runs/raw_full_gpu_20261031_235900/reports/historical_output_comparison.json`.
None is an exact rank match. The reported member Spearman values span
`0.857506` to `0.938947`. The component arrays lack ID sidecars, so OOF
comparisons are positional after checking labels against raw `TARGET`; test
array comparisons are positional and cannot prove per-row identity.

One value-scale distinction matters: the fresh V55 `test_cb_product` is stored
as rank-uniform values, while the historical member is stored as raw
probabilities. Their direct-value MAE/RMSE are not like-for-like; the report
includes rank-percentile comparisons as well. Their Spearman is `0.900908`,
so even after allowing for the scale difference their ordering is not
historically identical.

Fresh Anchor74 was compared by exact test-ID alignment against the supplied
historical Anchor74 CSV: Spearman/Pearson `0.869363`, MAE `0.110323`, RMSE
`0.147555`, maximum absolute difference `0.814677`, mean rank displacement
`6,784.91`, and maximum rank displacement `50,103`. No fitted historical model
checkpoint was available for an inference-only control.

The Anchor74 log confirms XGBoost `cuda` and CatBoost `GPU`. XGBoost also
emitted a device-mismatch warning during `predict_proba` and explicitly fell
back to DMatrix prediction because its pandas input was on CPU while the
booster was on CUDA. This is a prediction-path warning, not a silent fallback:
the generator confirms XGBoost training completed on CUDA. LightGBM remains
CPU-only in this installed build, whose GPU preflight reports that GPU tree
support was not enabled.

## Decision

Model configuration parity is **not established**. A full, fresh raw-data
candidate can be produced, but historical feature/model parity and the cause of
the CatBoost discrepancy remain unresolved. The 0.801340 historical leaderboard
result is not a result of this candidate unless independently evaluated by the
competition platform; no submission is made by this audit.
