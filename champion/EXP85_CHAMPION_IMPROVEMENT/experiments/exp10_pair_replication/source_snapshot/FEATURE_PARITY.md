# EXP08 historical feature parity and lineage

## Executable recovery

The read-only `build_hasan_600_features.py` executable produces 669 columns.
EXP08 reconstructs the executable formulas for 660 model-eligible features.
The bounded parity test runs that original executable in memory against a
15-row synthetic fixture, suppresses its cache writes, and compares every
eligible feature by name and value. The test matched all 660 columns exactly.
It writes no historical feature cache.

The 660 reconstructed columns are grouped and counted as:

| Family | Features |
|---|---:|
| A_FINANCIAL | 34 |
| B_RISK | 11 |
| C_TEMPORAL | 11 |
| D_DEMOGRAPHIC | 8 |
| E_INQUIRY | 10 |
| F_EXTERNAL | 16 |
| H_INTERACTIONS | 570 |

`HOUSING_STABILITY` is one of the 11 temporal features. Its exact historical
ratio is retained for parity and distribution reporting, but it is excluded
from model matrices because `DAYS_ID_PUBLISH + 1` can be zero. No infinity is
converted to a missing value. The run inventory records observed missing and
infinite counts; other unexpected infinite features fail preflight.

## Explicit exclusions

Nine columns are inventoried but not modeled:

- `N_APPS_ALL`, `APP_ORDER`, `DAYS_SINCE_PREV_APP`, and `DAYS_TO_NEXT_APP`
  reuse the rejected EXP07 entity signature and/or depend on a pooled
  train/test cohort.
- `UIDY_N`, `UIDY_SUM`, `UIDY_NEAR_Y`, `UIDY_NEAR_D`, and `UIDY_ANYBAD` depend
  on a global seed-42 label-memory split that is not nested inside EXP08's
  outer folds.

## Specification disagreement and unavailable lineage

The historical prose specification describes bivariate Clayton copulas and
a Merton model with regional occupational volatility. The executable instead
implements a trivariate Clayton transform over `EXT_SOURCE_1/2/3` and
fixed-parameter disposable-income Merton-like transforms. EXP08 follows the
executable, not the conflicting prose.

The historical parity audit reports full-matrix hashes, but the corresponding
historical cached matrices were not available in the inspected lineage. Full
historical matrix/hash parity therefore remains unverified. The bounded
executable parity test and the run's source/input/output hashes are the
available reproducible evidence.

## Validation distinction

The parity test fits ranks on the complete small fixture only to reproduce
the historical pooled executable exactly. Model evaluation does not use this
pooled transform: each EXP08 inner/outer fit partition estimates its own
empirical ranks, which are then applied to that partition's validation and
test rows. This intentional difference prevents validation/test distribution
information from influencing model-fit rank transforms.
