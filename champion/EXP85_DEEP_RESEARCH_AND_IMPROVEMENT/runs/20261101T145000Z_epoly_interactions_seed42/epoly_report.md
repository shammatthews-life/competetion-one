# E_POLY Interaction Screen

- Run: `20261101T145000Z_epoly_interactions_seed42`
- Seed: `42`
- Decision: `REJECT_ON_SCREENING`
- Same exact saved outer folds as the Phase 2 reference.
- Interaction definitions are rowwise and label-free; no target or applicant ID is used.

Frozen Phase 2 OOF AUC: `0.7715614423`
E_POLY OOF AUC: `0.7712424145`
Pooled OOF delta: `-0.0003190278`
Mean paired fold delta: `-0.0003192742`
Paired-fold standard error: `0.0002908994`
Positive folds: `2/5`

| Fold | Phase 2 AUC | E_POLY AUC | Delta |
|---:|---:|---:|---:|
| 1 | 0.76827281 | 0.76864570 | +0.00037289 |
| 2 | 0.77412739 | 0.77321266 | -0.00091473 |
| 3 | 0.77616376 | 0.77632358 | +0.00015982 |
| 4 | 0.76956077 | 0.76847219 | -0.00108857 |
| 5 | 0.77001386 | 0.76988809 | -0.00012578 |

- Runtime: `284.0` seconds.
- GPU samples: `56`; max utilization `44%`; peak VRAM `5036` MiB.
- Candidate SHA-256: `6f2290bbff346d14997df9134c748c81c6f7bab41ee2f783d05e978dd4caae81`
- Official score: **NOT EVALUATED**.
