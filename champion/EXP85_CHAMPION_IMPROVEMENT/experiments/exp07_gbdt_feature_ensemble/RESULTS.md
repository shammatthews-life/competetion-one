# EXP07 results — application-signature feature screen

## Decision: REJECTED

The application-signature GBDT screen passed its initial seed-`20261101`
gate, but the independent seed-42 replication did not reproduce the required
minimum pooled lift. The seed-42 nested blend gained `0.0006971` pooled AUC
against its freshly refit, inner-stopped Phase 2 baseline, below the
predeclared `0.001` gate. The candidate is not a promotion candidate. Do not
tune this signature hypothesis further or replace the historical champion.

No official leaderboard evaluation was performed. The historical champion
and frozen Phase 2 release remain unchanged.

## Method

EXP07 added two fold-local application-signature features to the frozen
Phase 2 pipeline:

- `EXP07_SIG_COUNT_LOG`: log frequency of a covariate signature, calculated
  from the model-fit partition and self-excluded for training rows.
- `EXP07_SIG_TARGET_RATE`: smoothed target rate for the same signature,
  cross-fitted within the model-fit partition; apply-row rates use only
  fit-partition labels.

The key uses truncated birth-registration and birth-ID-publish day
differences, gender, and education. It is not an applicant ID or verified
borrower identity. LightGBM and CatBoost predictions were averaged 50:50.
For each outer fold, the candidate-vs-Phase 2 weight was selected from a
fixed grid using inner validation only; outer validation labels were not used
for model or blend selection.

Seed `20261101` used the verified strict Phase 2 reference. Seed `42` used
the same archived five-fold partition but refit Phase 2 in this experiment,
selecting stopping iterations using only inner validation. Archived seed-42
predictions were not used as reference predictions.

## OOF results

### Seed 20261101 — initial screen

| Fold | Strict Phase 2 | Signature model | Nested blend | Blend delta |
|---:|---:|---:|---:|---:|
| 1 | 0.7660692 | 0.7669761 | 0.7669268 | +0.0008576 |
| 2 | 0.7726370 | 0.7739172 | 0.7738859 | +0.0012489 |
| 3 | 0.7714246 | 0.7726990 | 0.7727047 | +0.0012801 |
| 4 | 0.7685362 | 0.7684104 | 0.7684104 | -0.0001258 |
| 5 | 0.7733029 | 0.7750886 | 0.7750886 | +0.0017857 |
| **Pooled OOF** | **0.7703496** | **0.7713843** | **0.7713667** | **+0.0010171** |

The nested blend improved four of five folds. Mean paired fold gain was
`0.0010093` (standard error `0.0003198`), so the screen passed its
predeclared threshold and required an independent-seed replication.

### Seed 42 — independent replication

| Fold | Fresh strict Phase 2 | Signature model | Nested blend | Blend delta |
|---:|---:|---:|---:|---:|
| 1 | 0.7678841 | 0.7697499 | 0.7696796 | +0.0017955 |
| 2 | 0.7731767 | 0.7740357 | 0.7740833 | +0.0009066 |
| 3 | 0.7762336 | 0.7762895 | 0.7767096 | +0.0004760 |
| 4 | 0.7693846 | 0.7696279 | 0.7697867 | +0.0004021 |
| 5 | 0.7697764 | 0.7697566 | 0.7697566 | -0.0000198 |
| **Pooled OOF** | **0.7712486** | **0.7718294** | **0.7719457** | **+0.0006971** |

The blend improved four of five folds; mean paired gain was `0.0007121`
(standard error `0.0003081`). It missed the required `+0.001` pooled gain, so
the independent replication failed the advancement gate.

## Architecture comparisons

On seed `20261101`, the nested blend scored `0.7713667` versus EXP04
CatBoost Lossguide `0.7680327`, EXP05 residual MLP `0.7637273`, and EXP06
FT-Transformer `0.7640223`. These are secondary same-seed diagnostics, not
the advancement baseline. On seed 42, those earlier-run scores are
cross-seed pooled diagnostics only; no fold-paired comparison is claimed.

The signature model and Phase 2 OOF predictions were highly correlated:

| Seed | OOF Pearson | OOF Spearman | Error correlation | Test Pearson |
|---:|---:|---:|---:|---:|
| 20261101 | 0.9912614 | 0.9947136 | 0.9991063 | 0.9961003 |
| 42 | 0.9908374 | 0.9944867 | 0.9990565 | 0.9962861 |

This indicates limited prediction diversity despite the small lift. The
test-score distributions and full diagnostics are in each run summary.

## GPU, artifacts, and validation

| Run | GPU | Peak utilization | Peak VRAM | Runtime |
|---|---|---:|---:|---:|
| Seed 20261101 | RTX 4050 Laptop GPU | 99% | 5,092 MiB | 732.4 s |
| Seed 42 | RTX 4050 Laptop GPU | 98% | 5,146 MiB | 1,145.5 s |

CatBoost GPU training was verified for every fold; LightGBM used the frozen
CPU build. Peak VRAM stayed below the 6,141 MiB device capacity. Each run
contains fold metrics and indices, OOF/test predictions, submissions,
checkpoints, telemetry, source/input/output hashes, and an experiment
summary.

- Five focused unit tests passed; the runner and verifier passed
  `py_compile`.
- Independent verification passed for both runs: input/source/output hashes,
  all five outer/inner partitions, saved fold AUCs, RTX 4050 training
  evidence, submission schemas, and exact ordered test IDs.
- Both submissions contain 61,500 rows with schema `SK_ID_CURR,TARGET`.
- Seed `20261101` nested-blend submission SHA-256:
  `45291e04dc75c9a230af3c9cd392b046249b83a2dd717cdf4db3efe49a2a2f79`
- Seed 42 nested-blend submission SHA-256:
  `e9b224373b05d04f4af5452690a37b64bfe5ce5d0703972b37368c2b6901b486`

Run directories:

- `runs/exp07_signature_seed20261101_v1/`
- `runs/exp07_signature_seed42_v2/`
