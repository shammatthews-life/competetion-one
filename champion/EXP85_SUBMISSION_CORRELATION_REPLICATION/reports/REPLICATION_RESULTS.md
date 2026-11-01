# Replication Results

## Primary question

**Was the reported approximately 0.99999 correlation reproduced against the exact same reference and candidate?**

**Yes.** The original notebook's Stage 9 result is reproduced to its reported precision using the exact candidate/reference files and confirmed ID order.

| Metric | Notebook-style replay (`notebook_stage9`) | ID-aligned run | Original isolated auditor |
|---|---:|---:|---:|
| Pearson | 0.9999975815867518 | 0.9999975815867518 | 0.9999975815867513 |
| Spearman | 0.9999975815867501 | 0.9999975815867501 | 0.9999975815867501 |
| MAE | 0.00030141741027642276 | 0.00030141741027642276 | 0.00030141741027642276 |
| RMSE | 0.000634877054701475 | 0.000634877054701475 | 0.000634877054701475 |
| Maximum absolute error | 0.012504065100000017 | 0.012504065100000017 | 0.012504065100000017 |
| Mean rank displacement | 18.537170731707317 | 18.537170731707317 | 18.537170731707317 |
| Median / p95 / maximum rank displacement | 7 / 80 / 769 | 7 / 80 / 769 | 7 / (not emitted) / 769 |
| Exact rank agreements | 3,093 | 3,093 | Not emitted |

The original auditor uses `np.corrcoef` for Pearson; its roughly `5e-16` Pearson difference is floating-point implementation noise and all methods round to the exact reported `0.9999975816`.

## Exact inputs and identity

- Reference `historical_champion.csv`: SHA-256 `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`.
- Candidate `track1_v5_rebuilt.csv`: SHA-256 `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`.
- Candidate copy under `02_EXP85_REBUILT\submissions` has the same SHA-256.
- Reference and candidate: 61,500 rows each, zero duplicate IDs, zero missing/extra IDs, same ID row order.
- Candidate target range: finite and within `[0, 1]`; min/max recorded in current run JSON.
- Notebook Stage 9 did not explicitly join by ID. The audit verifies the order assumption; the preferred ID-aligned method independently reaches the same result.

## Why the high result is not a raw-only rebuild

The notebook reads its Phase 3 Anchor74 from `D:\gci\champion\02_EXP85_REBUILT\artifacts\v_20261027_precision_transductive_apex.csv` (SHA-256 `315b0bd80e4698240a7a902648a157d84da1480dea77901cdb60892502a7b55b`) and suite predictions from historical cache files under `D:\gci\champion\01_EXP85_RECOVERY\cache\`:

- `v_17_full_suite_bundle.npz`: `c1dfacea535ddac5d46a6d69ffcbbddf05ccbe876fc0fd115742feb91a701223`
- `v55_members.npz`: `995b0e80162b706e49e2d4b5551a34244bd33f6e8ae4893de1fd1fad38cbbc0f`
- `test_10f_xgb.npy`: `5a8d8a23b2f36ac6f27703b1f8c4b0b29aa15fefe4292edc4a628702fff76195`

The 85/15 rank/probit synthesis is a real high-correlation numerical result, but the cached suite prediction dependencies make this a **hybrid historical-cache replay**. The output's former “independent rebuild” label must not be interpreted as every upstream model being newly trained from raw data.

## Why the other numbers differ

- `0.9999628751933789` Spearman belongs to `submission_exp85_rebuilt.csv` (SHA `8165ad2c...`), a separate Phase 3 candidate. Its reported Pearson is `0.9999592592916924`; MAE `0.0025133036257398`; max rank displacement 2,091. The root-cause report also identifies historical suite cache use in its lineage.
- `0.9982094981549220` Spearman belongs to `submission_exp85_full_rebuilt.csv` (SHA `6bfc518b...`), a different archived full-scratch candidate. Its maximum rank displacement is 9,878, not 2,500. Its raw-only provenance is not recertified by this metric run.
- `submission_final.csv` (SHA `8e8158ae...`) is another distinct file; it scores 0.9969658880270557 Spearman against the same champion.
- Current production Track1B (SHA `9b9f7dde...`) scores 0.8921288944402191 against the champion.

These are not competing calculations of one candidate. Their file identities differ.

## Historical-cache reconstruction replay

The reconstruction runner was executed with hash-pinned inputs and wrote a new candidate under `runs/20261101T151558Z_historical_cache_v5/`. The output is byte-for-byte and value-for-value identical to the pinned prior V5 candidate:

- Candidate SHA-256: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`.
- Prior V5 candidate SHA-256: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`.
- Pearson / Spearman against the historical champion: `0.9999975815867518` / `0.9999975815867501`.
- MAE / RMSE / maximum absolute difference: `0.00030141741027642276` / `0.000634877054701475` / `0.012504065100000017`.
- Mean / maximum rank displacement: `18.537170731707317` / `769`; exact prediction matches: `3,093`.
- 61,500 unique IDs in raw-test/sample order; expected schema and probability bounds verified.
- Run manifest records command exit code `0`, source hash, input hashes, and output hashes.

The first replay wrote a candidate but failed while rendering its report. It also exposed a numerical mismatch: the notebook's `float32` XGB10 and V55 `cb_product` values remain `float32` through rank normalization and probit, while the initial runner promoted them to `float64` too early. The corrected runner preserves those source dtypes and now reproduces the pinned candidate exactly. The failed run and its failure report remain preserved at `runs/20261101T151056Z_historical_cache_v5/`; no prior artifact was overwritten.

## Official evaluation and stale hash

Local correlation does not establish a competition score. The historical `SUBMISSION_RESULTS.md` claims score `0.801340` for the V5 filename but lists hash `e979a04a...`, not the on-disk file hash `91aa95c9...`. The evidence gathered here does not resolve which bytes were evaluated by the official platform. No new submission was made, and no official score is claimed for any replicated candidate.

## Run artifacts

- Exact historical-cache V5 reconstruction: `runs/20261101T151558Z_historical_cache_v5/`.
- Preserved initial failed reconstruction attempt: `runs/20261101T151056Z_historical_cache_v5/`.
- Current-code legacy Stage 9 revalidation: `runs/20261101T133203Z_notebook_stage9_v5_revalidated/`
- Current-code ID-aligned V5 comparison: `runs/20261101T133251Z_id_aligned_v5/`
- Current-code candidate comparisons: `runs/20261101T133254Z_phase3/`, `runs/20261101T133257Z_full_rebuilt/`, `runs/20261101T133259Z_submission_final/`, and `runs/20261101T133302Z_current_track1b/`.
- Initial legacy Stage 9 replay: `runs/20261101T184300_notebook_stage9_v5/`
- Initial ID-aligned V5 comparison: `runs/20261101T184400_id_aligned_v5/`
- Original isolated auditor: `runs/20261101T184200_original_auditor_v5_rebuilt/`
- Initial separate candidate comparisons: `runs/20261101T184500_phase3/` through `runs/20261101T184800_current_track1b/`.

Each runner result includes full-precision JSON, a readable Markdown report, paths, hashes, alignment results, prediction bounds, source/config hashes, command, Git commit, method, and lineage caveat. The current-code revalidation uses the runner version that enforces the 61,500-row count and records prediction bounds.
