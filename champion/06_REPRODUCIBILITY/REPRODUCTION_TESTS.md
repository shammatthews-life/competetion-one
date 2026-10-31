# Automated Reproduction Tests & Verification Protocols

```bash
# 1. Run Hash Validation
python -c "
import hashlib
with open('08_SUBMISSIONS/submission_exp85_recovery_v5.csv', 'rb') as f:
    h = hashlib.sha256(f.read()).hexdigest()
assert h == '78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947'
print('Test 1 Passed: SHA-256 Bitwise Match!')
"

# 2. Run Rank Correlation Validation
python -c "
import pandas as pd
from scipy.stats import spearmanr
df1 = pd.read_csv('08_SUBMISSIONS/submission_exp85_recovery_v5.csv')
df2 = pd.read_csv('00_HISTORICAL_REFERENCE/EXP85_HISTORICAL_CHAMPION.csv')
sp = spearmanr(df1['TARGET'], df2['TARGET']).correlation
assert sp == 1.0
print(f'Test 2 Passed: Spearman = {sp:.10f}')
"
```
