# Best Independent Model

## Current validated best

Frozen Phase 2 remains the best **validated independent raw-data candidate** until a new experiment completes all integrity checks and any promising result is independently replicated.

- Candidate SHA-256: `2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa`
- Seed-42 OOF ROC-AUC: `0.7715614423106346`
- Matched baseline OOF ROC-AUC: `0.7625457907047022`
- Seed-42 lift: `+0.0090156516059324`
- Independent seed-20261101 pooled OOF ROC-AUC: `0.7711413866258005`
- Independent seed fold-local OOF ROC-AUC: `0.7709638203765881`
- Official competition score: **NOT EVALUATED**

## New experiment status

GC-01 added fold-local category peer context. It scored `0.7699312062` pooled OOF AUC (`-0.0016302361` versus Phase 2), with negative paired deltas on all five folds and declines in both component models. It is rejected; no candidate is promoted.

TE-01 is screening cross-fitted target encodings for four core categorical fields. It is not a candidate until it passes the same OOF, fold, integrity, and GPU evidence checks. Any positive screen still requires an independent-seed replication.

TE-01 completed and was rejected: pooled OOF AUC `0.7709211321` (`-0.0006403102` versus Phase 2), with only 1/5 paired folds positive. Integrity, exact-fold, OOF recomputation, and GPU checks passed. No candidate was promoted; no second seed was run.
