# Day 16 run report (SYNTHETIC data)

Data path: **synthetic**; 600,000 policies over 24 months; runtime 98.9 s.

## Initial training and gated promotion

| family | note | version | action | val_tweedie_deviance | null_deviance | val_gini | val_ae_ratio | deviance_gain_vs_reference | gate_reference | checks_passed |
|---|---|---|---|---|---|---|---|---|---|---|
| glm | bootstrap champion | 1 | promoted | 83.5 | 90.53 | 0.3556 | 1.041 | 0.07772 | null model | deviance_gain=Y, gini_floor=Y, ae_band=Y |
| gbm | challenger | 2 | promoted | 82.18 | 90.53 | 0.3693 | 1.038 | 0.01581 | champion v1 | deviance_gain=Y, gini_floor=Y, ae_band=Y |
| glm-weak | negative control: under-specified challenger | 3 | rejected | 86.72 | 90.53 | 0.2712 | 1.038 | -0.05534 | champion v2 | deviance_gain=N, gini_floor=Y, ae_band=Y |

## Monthly production monitoring

| month | champion_version | deviance | static_deviance | ae_ratio | static_ae_ratio | gini | drift_share | drifted_columns | trigger | suppressed_by_cooldown | api_rows_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 2 | 84.27 | 84.27 | 1.083 | 1.083 | 0.3472 | 0 |  | False | False | 2.372e+04 |
| 14 | 2 | 83.01 | 83.01 | 1.008 | 1.008 | 0.3209 | 0 |  | False | False | 4.691e+04 |
| 15 | 2 | 82.36 | 82.36 | 1.043 | 1.043 | 0.3792 | 0 |  | False | False | 3.74e+04 |
| 16 | 2 | 83.19 | 83.19 | 1.055 | 1.055 | 0.3734 | 0 |  | False | False | 2.689e+04 |
| 17 | 2 | 84.42 | 84.42 | 1.103 | 1.103 | 0.3778 | 0 |  | True | False | 3.804e+04 |
| 18 | 4 | 84.64 | 84.53 | 1.06 | 1.094 | 0.3716 | 0 |  | False | False | 2.889e+04 |
| 19 | 4 | 94.77 | 94.8 | 1.093 | 1.132 | 0.593 | 0.4444 | DrivAge;BonusMalus;Density;Area | True | False | 2.205e+04 |
| 20 | 5 | 95.9 | 96.42 | 1.021 | 1.097 | 0.5666 | 0 |  | False | False | 2.267e+04 |
| 21 | 5 | 94.34 | 94.36 | 1.017 | 1.097 | 0.5816 | 0 |  | False | False | 2.961e+04 |
| 22 | 5 | 95.83 | 96.06 | 1.012 | 1.092 | 0.5601 | 0 |  | False | False | 2.859e+04 |
| 23 | 5 | 96.5 | 96.67 | 1.029 | 1.108 | 0.5554 | 0 |  | False | False | 2.31e+04 |
| 24 | 5 | 96.75 | 97.17 | 1.068 | 1.154 | 0.5797 | 0 |  | False | False | 2.318e+04 |

## Retraining triggers

| month | version | action | candidate_deviance | champion_deviance | deviance_gain | candidate_ae | champion_ae |
|---|---|---|---|---|---|---|---|
| 17 | 4 | promoted | 84.95 | 85.08 | 0.001474 | 1.074 | 1.11 |
| 19 | 5 | promoted | 95.62 | 95.75 | 0.001304 | 1.049 | 1.09 |

## Shadow / canary comparison (previous champion shadows the new one)

| month | champion_version | shadow_version | champion_deviance | shadow_deviance | champion_ae | shadow_ae | api_shadow_rows | api_shadow_mean_champion | api_shadow_mean_shadow | api_mean_ratio_shadow_to_champion | api_rank_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 18 | 4 | 2 | 84.64 | 84.53 | 1.06 | 1.094 | 25000 | 240.6 | 233.1 | 0.9688 | 0.9586 |
| 19 | 4 | 2 | 94.77 | 94.8 | 1.093 | 1.132 | 25000 | 471.8 | 455.5 | 0.9654 | 0.9776 |
| 20 | 5 | 4 | 95.9 | 96.06 | 1.021 | 1.059 | 25000 | 487.5 | 469.8 | 0.9638 | 0.9891 |
| 21 | 5 | 4 | 94.34 | 94.25 | 1.017 | 1.058 | 25000 | 494.1 | 474.8 | 0.9609 | 0.9889 |
| 22 | 5 | 4 | 95.83 | 96.05 | 1.012 | 1.049 | 25000 | 487.8 | 470.8 | 0.9651 | 0.9889 |
| 23 | 5 | 4 | 96.5 | 96.6 | 1.029 | 1.065 | 25000 | 484.1 | 467.9 | 0.9666 | 0.989 |
| 24 | 5 | 4 | 96.75 | 97 | 1.068 | 1.111 | 25000 | 490 | 471.4 | 0.962 | 0.9891 |

## Model registry

| version | stage | aliases | family | gate | train_months | trigger | val_tweedie_deviance | val_gini | val_ae_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Archived | [] | glm | passed | 1-10 |  | 83.5 | 0.3556 | 1.041 |
| 2 | Archived | [] | gbm | passed | 1-10 |  | 82.18 | 0.3693 | 1.038 |
| 3 | Archived | [] | glm | failed | 1-10 |  | 86.72 | 0.2712 | 1.038 |
| 4 | Archived | [] | gbm | passed | 6-17 | month 17: \|A/E-1\| 0.103 > 0.10 | 84.95 | 0.3717 | 1.074 |
| 5 | Production | ['champion'] | gbm | passed | 8-19 | month 19: drift_share 0.44 >= 0.20 (DrivAge, BonusMalus, Density, Area) \| deviance ratio 1.116 > 1.10 | 95.62 | 0.58 | 1.049 |

## Exported champion API smoke test

- ready_status: 200
- ready_version: v5
- score_status: 200
- example_pure_premium: 152.85467616058176
- example_expected_cost: 76.42733808029088
- invalid_request_status: 422
- invalid_request_errors: 2
