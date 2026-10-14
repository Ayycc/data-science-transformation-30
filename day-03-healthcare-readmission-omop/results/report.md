# Day 03 results (SYNTHETIC data)

## Cohort attrition

| Step | n | Unplanned 30-day readmits |
|---|---|---|
| 01 discharge inside 365-day lookback | 6,053 | 745 |
| 02 under 30 days of follow-up left | 370 | 26 |
| 03 age under 18 | 22 | 2 |
| 04 died in hospital | 388 | 0 |
| 05 transferred to another acute hospital | 338 | 8 |
| 06 left against medical advice | 135 | 38 |
| 07 planned admission for chemotherapy | 1,123 | 143 |
| 08 died within 30 days before any readmission | 252 | 0 |
| 99 eligible index admission | 9,301 | 1,563 |

## Test-period performance

| Model | AUROC (95% CI) | Brier | Scaled Brier | Calibration slope (95% CI) | CITL | O/E |
|---|---|---|---|---|---|---|
| logistic_l2 | 0.782 (0.756-0.806) | 0.1200 | 0.176 | 1.018 (0.928-1.113) | 0.031 | 1.021 |
| gbm_calibrated | 0.783 (0.757-0.807) | 0.1191 | 0.183 | 1.104 (1.005-1.217) | 0.040 | 1.028 |
| lace | 0.707 (0.678-0.734) | 0.1329 | 0.088 | 1.006 (0.859-1.165) | 0.057 | 1.044 |

Primary model (lowest training out-of-fold log loss): **logistic_l2**

## Subgroups (primary model, test)

| Dimension | Level | n | Events | Obs | Pred | O/E | AUROC | Brier | Slope |
|---|---|---|---|---|---|---|---|---|---|
| age_band | 18-49 | 275 | 13 | 0.047 | 0.053 | 0.89 | 0.799 | 0.0412 | n/a |
| age_band | 50-64 | 529 | 55 | 0.104 | 0.101 | 1.03 | 0.789 | 0.0806 | 1.24 |
| age_band | 65-74 | 532 | 70 | 0.132 | 0.143 | 0.92 | 0.690 | 0.1056 | 0.84 |
| age_band | 75-84 | 973 | 221 | 0.227 | 0.206 | 1.10 | 0.789 | 0.1397 | 1.16 |
| age_band | 85+ | 600 | 156 | 0.260 | 0.267 | 0.97 | 0.712 | 0.1718 | 0.77 |
| sex | female | 1,505 | 285 | 0.189 | 0.182 | 1.04 | 0.787 | 0.1243 | 1.02 |
| sex | male | 1,395 | 228 | 0.163 | 0.164 | 1.00 | 0.775 | 0.1154 | 1.00 |
| sex | unknown | 9 | 2 | 0.222 | 0.222 | 1.00 | 0.857 | 0.1120 | n/a |
