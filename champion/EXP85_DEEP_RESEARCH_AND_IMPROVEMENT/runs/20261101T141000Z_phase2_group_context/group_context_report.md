# Phase 2 Group-Context Feature Experiment

- Run: `20261101T141000Z_phase2_group_context`
- Status: `COMPLETED_DIAGNOSTIC_VALIDATION`
- Decision: `REJECT_ON_SCREENING`
- Same five seed-42 folds as frozen Phase 2; validation labels were used only to score OOF predictions.
- Fold group statistics use only outer-training rows; each training row's own measurement is left out.
- No historical champion or test labels were read.

| Model | Full OOF AUC |
|---|---:|
| Frozen Phase 2 | 0.7715614423 |
| Phase 2 + group context | 0.7699312062 |

Full OOF delta: `-0.0016302361`
Mean paired fold delta: `-0.0016195316` (SD `0.0008504625`; 0/5 positive folds).

| Fold | Phase 2 AUC | Group-context AUC | Paired delta |
|---:|---:|---:|---:|
| 1 | 0.76827281 | 0.76760059 | -0.00067222 |
| 2 | 0.77412739 | 0.77258258 | -0.00154480 |
| 3 | 0.77616376 | 0.77320565 | -0.00295811 |
| 4 | 0.76956077 | 0.76782722 | -0.00173355 |
| 5 | 0.77001386 | 0.76882489 | -0.00118898 |

The historical champion was not involved in feature design or selection. This is a screening result on the already-used seed-42 folds; do not promote unless a predeclared independent seed confirms it.

Candidate SHA-256: `9c80b3a277927c68b98756fccbb9095b79aa6698cff4c8a4146eef966013ad0a`
Official score: **NOT EVALUATED**.
