# Hasan 669 Feature Engineering Mathematical Specification

## 1. Clayton Extreme Value Copulas
Measures asymmetric joint default coupling between primary asset quality and applicant liquidity:
$$C_{\theta}(u, v) = \max\left(\left(u^{-\theta} + v^{-\theta} - 1\right)^{-1/\theta}, 0\right)$$
Evaluated across grid $\theta \in \{1.5, 2.5, 4.0\}$ on pairs $(EXT\_SOURCE\_1, EXT\_SOURCE\_2)$, $(EXT\_SOURCE\_2, EXT\_SOURCE\_3)$, and $(EXT\_SOURCE\_1, EXT\_SOURCE\_3)$.

## 2. Merton Structural Distance-to-Default
Adopts the Black-Scholes-Merton structural credit risk framework:
$$DD = \frac{\ln(V_A / D) + (\mu - 0.5 \sigma_A^2) T}{\sigma_A \sqrt{T}}$$
where $V_A = AMT\_INCOME\_TOTAL$, $D = AMT\_CREDIT$, and $\sigma_A$ is estimated from regional occupational volatility.

## 3. Financial Solvency & Annuity Leverage
- Payment-to-Income: $AMT\_ANNUITY / AMT\_INCOME\_TOTAL$
- Credit-to-Goods Ratio: $AMT\_CREDIT / AMT\_GOODS\_PRICE$
- Net Financial Burden: $(AMT\_CREDIT - AMT\_GOODS\_PRICE) / AMT\_INCOME\_TOTAL$
- Employment-to-Age Ratio: $DAYS\_EMPLOYED / DAYS\_BIRTH$
