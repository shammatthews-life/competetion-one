# EXP10 — Independent Replication of the EXP09 Best Pair

- Decision: **REJECTED**.
- Outer seed: `20261102`; folds: `5`.
- Prediction streams were trained fresh; EXP09 OOF arrays were not reused.
- All AUC values below are OOF metrics, not public leaderboard scores.

## Exact predeclared pair

- EXP09 strategy: `equal_exp07_signature__exp08_lgbcat`.
- Components: EXP07 application-signature model and EXP08 fold-local LightGBM/CatBoost component.
- Pair rule: fixed arithmetic mean, 50:50; no weights were fit or tuned.
- EXP09 discovery reference (seed 20261101): pooled AUC 0.77135704, delta +0.00100746, positive folds 5/5.

| Stream | Pooled OOF AUC | Delta vs fresh Phase 2 | Positive folds |
|---|---:|---:|---:|
| Fresh matched Phase 2 | 0.77076772 | +0.00000000 | 0/5 |
| EXP07 signature | 0.77136190 | +0.00059418 | 5/5 |
| EXP08 LGB/Cat | 0.77093880 | +0.00017108 | 3/5 |
| Fixed pair | 0.77156130 | +0.00079358 | 5/5 |

### Paired fold results

| Fold | Phase 2 AUC | Pair AUC | Delta |
|---:|---:|---:|---:|
| 1 | 0.76970934 | 0.77022525 | +0.00051591 |
| 2 | 0.77795252 | 0.77864737 | +0.00069485 |
| 3 | 0.76660063 | 0.76717883 | +0.00057820 |
| 4 | 0.77349758 | 0.77430121 | +0.00080363 |
| 5 | 0.76645277 | 0.76784523 | +0.00139246 |

## Complementarity

- Component Pearson correlation: 0.989926.
- Component Spearman correlation: 0.993943.
- Component residual correlation: 0.998971.
- Component top-prevalence overlap: 0.922871.
- Component binary-error Jaccard: 0.895454.

## GPU and runtime evidence

- EXP07: NVIDIA GeForce RTX 4050 Laptop GPU; peak utilization 99%; peak VRAM 4903 MiB; training 1008.8s.
- EXP08: COMPLETED; NVIDIA GeForce RTX 4050 Laptop GPU; peak utilization 100%; peak VRAM 4901 MiB; training 1759.4s.
- LightGBM ran on CPU because the installed build is CPU-only; CatBoost GPU training was verified. CUDA XGBoost preflight reported no CPU fallback.

## Integrity and runtime

- Every reported comparison uses one authoritative Phase 2 OOF/test stream from the fresh EXP07 seed-20261102 run, saved with IDs, labels, and fold membership in `shared_baseline_contract.npz`.
- EXP08 independently produced a different local Phase 2 diagnostic stream (max OOF difference 0.102994; max test difference 0.030009); it was not used for pair scoring.
- Fold membership was independently reconstructed from StratifiedKFold(seed=20261102), and each component's saved fit/validation indices were compared exactly.
- OOF rows and test IDs were aligned to the hashed raw train/test inputs. See `SHA256_MANIFEST.json`.
- Test prediction rows: 61500; ordered test-ID SHA-256: `d7fd39f08d114162f875cfd643a5952c74e0599192010884dfa675f54e85c645`.
- Test pair prediction values SHA-256: `b3ca98121fff570890c8eefe1bd0aab5dbc022eb2881bfde979ceab60e5623fc`; compressed prediction archive SHA-256: `b859bb5213cc0f6264f99fed3214904d5662ad682a81b4e1c29a2c01fe375d30`.
- GPU-supported model training was required to pass the source CUDA preflights; LightGBM used CPU because the installed build is CPU-only.
- The only source snapshot changes add this new seed/fold evidence path and suppress component submission files; model configurations and feature logic are preserved.

## Advancement decision

- Required pooled improvement: +0.001.
- Observed pooled improvement: +0.00079358.
- Required positive folds: 4/5; observed 5/5.
- Candidate submission created: `False`.
- Decision: **REJECTED**.

EXP10 does not establish an official public leaderboard score or prove the 0.80314 target.
