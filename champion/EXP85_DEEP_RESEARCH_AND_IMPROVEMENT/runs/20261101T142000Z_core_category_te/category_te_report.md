# Core-Category Nested Target-Encoding Experiment

- Run: `20261101T142000Z_core_category_te`
- Decision: `REJECT_ON_SCREENING`
- Same exact five seed-42 folds as frozen Phase 2.
- The target encoder uses nested OOF maps for outer-training rows; validation/test encodings use only outer-training labels.
- No historical champion or test labels were read.

Frozen Phase 2 OOF AUC: `0.7715614423`
Core-category TE OOF AUC: `0.7709211321`
Paired delta: `-0.0006403102`
Positive folds: `1/5`

| Fold | Phase 2 AUC | Category-TE AUC | Delta |
|---:|---:|---:|---:|
| 1 | 0.76827281 | 0.76804005 | -0.00023276 |
| 2 | 0.77412739 | 0.77321044 | -0.00091695 |
| 3 | 0.77616376 | 0.77671942 | +0.00055566 |
| 4 | 0.76956077 | 0.76780060 | -0.00176017 |
| 5 | 0.77001386 | 0.76911611 | -0.00089776 |

This is a screening run only. A positive result requires a predeclared independent-seed replication before promotion.
Candidate SHA-256: `0e28cd837b516d60811ffe3abcb2fd9972bca0a67b68b3f76edad18f9f046f99`
Official score: **NOT EVALUATED**.
