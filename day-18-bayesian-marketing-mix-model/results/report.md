# Day 18 run report (SYNTHETIC data)

Data path: `synthetic_generator`; 156 weeks x 10 regions; seed 20261028.
Sampler: NUTS, 4 chains x 1000 draws (1000 tune), 246.4 s.

## Convergence

- Max R-hat over all 785 posterior quantities: 1.004
- Quantities with R-hat >= 1.01: 0
- Min bulk ESS: 1370.2405; min tail ESS: 1321.5275; divergences: 0

| variable | size | max_rhat | min_ess_bulk | min_ess_tail |
|---|---|---|---|---|
| mu_log_effect | 4 | 1.0016 | 1370.2405 | 2012.3101 |
| z_effect | 40 | 1.0040 | 1786.5840 | 2187.9518 |
| intercept_mu | 1 | 1.0014 | 2694.8540 | 2579.5126 |
| intercept | 10 | 1.0021 | 1453.8129 | 1784.1869 |
| trend | 1 | 1.0014 | 5760.5721 | 2924.7845 |
| gamma_fourier | 4 | 1.0019 | 1623.7982 | 2417.9704 |
| gamma_holiday | 1 | 0.9999 | 6104.4535 | 3068.3553 |
| gamma_price | 1 | 1.0011 | 5781.6265 | 2981.3080 |
| gamma_comp | 1 | 1.0000 | 5231.0312 | 2868.0399 |
| alpha | 4 | 1.0007 | 2965.7901 | 1860.8647 |
| lam | 4 | 1.0020 | 1793.2701 | 2064.7528 |
| sigma_log_effect | 4 | 1.0023 | 1860.9083 | 1321.5275 |
| intercept_sigma | 1 | 1.0007 | 4185.4305 | 2682.8575 |
| sigma | 1 | 1.0000 | 5689.3343 | 2730.7767 |
| effect_at_mean | 40 | 1.0023 | 1424.0751 | 1885.7388 |
| beta | 40 | 1.0039 | 2342.4523 | 2101.7173 |
| channel_contribution | 624 | 1.0030 | 1371.6611 | 2014.2375 |
| roas | 4 | 1.0019 | 1391.0882 | 2198.2441 |

## Predictive checks

- Prior predictive 94% interval covers 1.0 of region-weeks; median interval width / observed = 2.0012
- Posterior predictive 94% interval covers 0.9455 of region-weeks and 0.9359 of national weeks
- Posterior mean R^2 (region-week) 0.9983; MAPE 0.0154 (region-week), 0.0047 (national)

## ROAS

| channel | spend_k | true_roas | roas_mean | roas_hdi_low | roas_hdi_high | true_in_hdi | last_click_roas | contribution_mean_k | contribution_hdi_low_k | contribution_hdi_high_k | true_contribution_k | contribution_share_of_sales |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tv | 7532.177 | 1.600 | 1.560 | 1.493 | 1.630 | True | 0.160 | 11751.529 | 11248.471 | 12273.912 | 12051.483 | 0.043 |
| search | 6509.444 | 1.100 | 1.220 | 0.797 | 1.667 | True | 4.846 | 7943.350 | 5187.260 | 10850.846 | 7160.389 | 0.029 |
| social | 5131.896 | 2.200 | 2.333 | 2.133 | 2.561 | True | 1.100 | 11973.216 | 10945.894 | 13141.217 | 11290.172 | 0.044 |
| retail_media | 6240.935 | 1.400 | 1.623 | 1.197 | 2.115 | True | 1.120 | 10130.830 | 7473.052 | 13202.649 | 8737.308 | 0.037 |

## Budget (annual $k)

| channel | current | expected_optimal | risk_adjusted | last_click_guided | oracle_true_optimal |
|---|---|---|---|---|---|
| tv | 2274.8 | 2964.9 | 3412.1 | 1137.4 | 3412.1 |
| search | 2162.6 | 1081.3 | 1081.3 | 3243.9 | 1081.3 |
| social | 1677.4 | 2516.2 | 2516.2 | 838.7 | 2516.2 |
| retail_media | 2082.3 | 1634.7 | 1187.5 | 2977.1 | 1187.5 |

- **expected_optimal**: posterior lift 1028.4 $k (94% HDI 516.0 to 1482.3), P(lift>0) 1.000; true lift 1125.0 $k
- **risk_adjusted**: posterior lift 993.5 $k (94% HDI 426.3 to 1504.3), P(lift>0) 0.999; true lift 1184.2 $k
- **last_click_guided**: posterior lift -1619.7 $k (94% HDI -2000.8 to -1268.0), P(lift>0) 0.000; true lift -1613.5 $k
- **oracle_true_optimal**: posterior lift 993.5 $k (94% HDI 426.3 to 1504.3), P(lift>0) 0.999; true lift 1184.2 $k
