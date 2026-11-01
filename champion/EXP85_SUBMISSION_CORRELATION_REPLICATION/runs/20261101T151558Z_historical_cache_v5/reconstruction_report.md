# Historical EXP85 V5 Cache Reconstruction

- Run ID: `20261101T151558Z_historical_cache_v5`
- Status: **REPRODUCED_EXACT_PRIOR_V5**
- Classification: **hybrid historical-cache reconstruction**.
- Official competition score: **NOT EVALUATED**.

## Reconstructed submission

- SHA-256: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`
- Previous V5 candidate SHA-256: `91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482`
- Byte-identical to previous V5 candidate: `True`
- Rows/schema: `61500` / `SK_ID_CURR,TARGET`
- IDs match raw test and sample order: `True`

## ID-aligned comparison to the historical champion

- Pearson: `0.9999975815867518`
- Spearman: `0.9999975815867501`
- MAE / RMSE: `0.0003014174102764` / `0.0006348770547015`
- Maximum absolute difference: `0.0125040651000000`
- Mean / maximum rank displacement: `18.537171` / `769`
- Exactly matching predictions: `3093`

## Artifact dependency graph

| Artifact | Exact path | SHA-256 | Producer / lineage status | Consumer |
|---|---|---|---|---|
| Rebuilt Anchor74 used by V5 | `D:\gci\champion\02_EXP85_REBUILT\artifacts\v_20261027_precision_transductive_apex.csv` | `315b0bd80e4698240a7a902648a157d84da1480dea77901cdb60892502a7b55b` | Exact notebook input identified; producer execution is not bound to this file by a run manifest. | 85% Anchor74 probit branch |
| V17 historical test bundle | `D:\gci\champion\01_EXP85_RECOVERY\cache\v_17_full_suite_bundle.npz` | `c1dfacea535ddac5d46a6d69ffcbbddf05ccbe876fc0fd115742feb91a701223` | Exact historical producer run unavailable; recovered generator output is not byte/array-identical. | Three explicitly named suite members |
| V55 historical test members | `D:\gci\champion\01_EXP85_RECOVERY\cache\v55_members.npz` | `995b0e80162b706e49e2d4b5551a34244bd33f6e8ae4893de1fd1fad38cbbc0f` | Exact historical producer run unavailable; recovered generator output is not byte/array-identical. | Four explicitly named suite members |
| XGB10 historical test vector | `D:\gci\champion\01_EXP85_RECOVERY\cache\test_10f_xgb.npy` | `5a8d8a23b2f36ac6f27703b1f8c4b0b29aa15fefe4292edc4a628702fff76195` | Exact historical producer run unavailable; recovered generator output is not byte/array-identical. | One explicitly named suite member |
| Raw test IDs and sample submission | `D:\gci\inpuy` | `test=dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65; sample=6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db` | Supplied competition inputs; hash-verified. | ID-order validation only; not model features. |

## Audit findings

- The original notebook's exact 85/15 probit formula and eight explicit suite weights were used.
- The correct V5 Anchor74 generation is pinned by its expected SHA-256. Alternate same-named Anchor74 outputs were not substituted.
- V17/V55/XGB10 cache producers are not fully provenance-bound. Recovered raw-data generators produce different arrays; their outputs were not substituted.
- Historical suite arrays contain no embedded SK_ID_CURR IDs or verified row-ID sidecars. Their positional order is retained as in the original notebook; the final submission's ID order is independently checked.
- This is not a full raw-data historical pipeline reconstruction and does not certify the official score.

## Incomplete dependencies

- **Original V17/V55/XGB10 training execution metadata and cache ID sidecars**: UNAVAILABLE. The historical vectors are exactly identified and reproduce the prior candidate, but their original per-row ID mapping and training environment cannot be independently certified.
- **Exact Anchor74 producer run manifest for SHA-256 315b0bd80e4698240a7a902648a157d84da1480dea77901cdb60892502a7b55b**: UNAVAILABLE. The exact V5 input is present and hash-pinned; its upstream training/cache ancestry is not fully proven.
- **Official platform record linking score 0.801340 to this candidate hash**: UNAVAILABLE. The local reconstruction proves numeric correlation only; the official score is not claimed.

