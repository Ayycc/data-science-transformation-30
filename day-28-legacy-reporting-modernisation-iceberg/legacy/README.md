# Legacy source of record (not executed)

These files represent the legacy quarterly regulatory return (QRT) chain: four
SAS programs run in sequence by the month-end scheduler, followed by an Excel
workbook macro that builds the final return. They are written for this SYNTHETIC
case study. No SAS licence or Excel is needed: the logic is re-implemented
line by line in `src/legacy28/legacy.py`, and the modern pipeline is
reconciled against that re-implementation.

| Order | File | Rules |
|---|---|---|
| 1 | `sas/01_policy_prep.sas` | R01, R02 |
| 2 | `sas/02_premium_earn.sas` | R03, R04, R11 |
| 3 | `sas/03_claims.sas` | R06, R07 |
| 4 | `sas/04_summarise.sas` | R05, R08 |
| 5 | `excel/QRT_build.bas` | R09, R10, R11 |
