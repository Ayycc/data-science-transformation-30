# Day 27 report: dynamic pricing with contextual bandits (SYNTHETIC)

Data path: `synthetic_simulator`. SYNTHETIC. The prompt specifies a market simulator; no public dataset is used.

## Logged year and elasticity estimation

1095 (night, booking window) cells over 365 nights; 471 randomised price-test cells. Occupancy 81.7%, sell-out nights 38.9%. On manager-priced cells, corr(log price multiplier, log demand) = +0.294 (price follows demand).

| Estimator | Semi-elasticity theta | 95% CI | Covers true mean | Rows |
|---|---|---|---|---|
| naive_ols | -0.290 | [-0.51, -0.07] | False | 1095 |
| randomised_test_ols | -1.845 | [-2.16, -1.53] | True | 471 |
| iv_2sls | -1.814 | [-2.12, -1.51] | True | 1095 |
| truth (simulator, mean over cells at the static rate) | -1.756 | p10-p90 [-2.81, -0.78] | | |

2SLS first-stage F = 997.

## Policy study

20 seeds x 1095 nights x 3 booking windows; guardrails floor $79, ceiling $449, max 20% change per night per window. Bandits warm-started from randomised cells only: True.

| Policy | Uplift vs static [95% t-CI] | Bootstrap CI | Seeds beating static | RevPAR $ | Occupancy | ADR $ | Cum. regret $k | Regret exponent | Regret $/night by year | Volatility | Max nightly change | Guardrail violations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static | +0.00% [+0.00%, +0.00%] | [+0.00%, +0.00%] | 0/20 | 126.57 | 81.1% | 156.15 | 1,341 | 1.00 | 1225 -> 1217 -> 1231 | 2.9% | 20.0% | 0 |
| greedy_naive | -46.93% [-47.38%, -46.48%] | [-47.32%, -46.51%] | 0/20 | 67.17 | 24.2% | 277.13 | 3,674 | 1.00 | 3354 -> 3360 -> 3351 | 3.0% | 20.0% | 0 |
| greedy_iv | +4.91% [+4.76%, +5.06%] | [+4.77%, +5.05%] | 20/20 | 132.79 | 89.0% | 149.25 | 226 | 0.99 | 206 -> 209 -> 205 | 11.0% | 20.0% | 0 |
| eps_greedy | +4.53% [+4.34%, +4.72%] | [+4.35%, +4.69%] | 20/20 | 132.30 | 82.3% | 160.73 | 402 | 0.73 | 497 -> 334 -> 270 | 11.5% | 20.0% | 0 |
| linucb | +5.73% [+5.60%, +5.86%] | [+5.62%, +5.85%] | 20/20 | 133.83 | 80.7% | 165.87 | 260 | 0.66 | 349 -> 209 -> 154 | 11.0% | 20.0% | 0 |
| thompson | +5.79% [+5.65%, +5.92%] | [+5.66%, +5.91%] | 20/20 | 133.89 | 81.9% | 163.53 | 242 | 0.67 | 321 -> 195 -> 145 | 11.7% | 20.0% | 0 |
| linucb_no_rm | -6.69% [-6.88%, -6.50%] | [-6.87%, -6.52%] | 0/20 | 118.10 | 91.6% | 128.99 | 1,523 | 1.01 | 1358 -> 1405 -> 1409 | 11.6% | 20.0% | 0 |
| oracle | +6.01% [+5.88%, +6.13%] | [+5.89%, +6.12%] | 20/20 | 134.17 | 89.5% | 149.86 | n/a | n/a | n/a | 12.3% | 20.0% | 0 |

Regret exponent = slope of log mean cumulative regret on log night over the last 75% of the horizon (1 = linear).

## Stretch: competitor reaction (strength 0.5, 10 seeds)

| Policy | Uplift vs static [95% t-CI] | Bootstrap CI | Seeds beating static | RevPAR $ | Occupancy | ADR $ | Cum. regret $k | Regret exponent | Regret $/night by year | Volatility | Max nightly change | Guardrail violations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static | +0.00% [+0.00%, +0.00%] | [+0.00%, +0.00%] | 0/10 | 126.45 | 81.0% | 156.15 | 1,344 | 1.00 | 1229 -> 1199 -> 1253 | 2.9% | 20.0% | 0 |
| greedy_iv | +5.04% [+4.81%, +5.27%] | [+4.87%, +5.24%] | 10/10 | 132.83 | 88.9% | 149.37 | 213 | 0.99 | 195 -> 197 -> 191 | 10.9% | 20.0% | 0 |
| linucb | +5.51% [+5.27%, +5.74%] | [+5.33%, +5.72%] | 10/10 | 133.42 | 80.5% | 165.65 | 255 | 0.65 | 345 -> 206 -> 147 | 10.7% | 20.0% | 0 |
| thompson | +5.53% [+5.26%, +5.80%] | [+5.32%, +5.76%] | 10/10 | 133.44 | 81.5% | 163.79 | 239 | 0.68 | 314 -> 196 -> 144 | 11.6% | 20.0% | 0 |

## Stretch: off-policy evaluation from 471 randomised log cells

| Target policy | Truth $/cell | IPS [95% CI] | SNIPS | DM | DR [95% CI] | IPS rel. err | DM rel. err | DR rel. err | Matched cells |
|---|---|---|---|---|---|---|---|---|---|
| static_rate (m=1.0) | 6349 | 6370 [4990, 7851] | 6061 | 6456 | 6319 [5759, 6915] | +0.3% | +1.7% | -0.5% | 99 |
| premium (m=1.1) | 5791 | 5845 [4527, 7285] | 5985 | 5992 | 5938 [5366, 6495] | +0.9% | +3.5% | +2.5% | 92 |
| discount (m=0.9) | 6774 | 8069 [6328, 9966] | 7104 | 6709 | 6888 [6179, 7653] | +19.1% | -1.0% | +1.7% | 107 |
| late-window premium (m=1.2 in 0-3d, else 1.0) | 6327 | 6564 [5185, 8069] | 6183 | 6424 | 6239 [5658, 6878] | +3.7% | +1.5% | -1.4% | 100 |
| greedy_iv myopic | 6946 | 5730 [4458, 7176] | 6664 | 6702 | 6860 [6307, 7484] | -17.5% | -3.5% | -1.2% | 81 |

Run time: 102 s.
