# Model card: `reweighing_a0.5`

**Trained and evaluated on SYNTHETIC data only.**

- **Task:** probability that an instalment-loan applicant defaults (`bad`); approve if p_bad <= 0.1930.
- **Model:** LightGBM with monotone constraints; inputs: `credit_utilization`, `debt_to_income`, `num_delinquencies`, `num_inquiries_6m`, `credit_history_months`, `annual_income`, `employment_years`, `loan_amount`, `months_at_address`. Protected attributes and geography are not inputs.
- **Intended users:** credit-decisioning engine with human review of declines near the cutoff.
- **Out of scope:** pricing, line assignment, marketing, any population unlike the development sample.

## Performance (out-of-time test)

| AUC | KS | Brier | Approval rate | Bad rate among approved |
|---|---|---|---|---|
| 0.763 | 0.396 | 0.1252 | 66.4% | 9.2% |

## Fairness (out-of-time test, 95% bootstrap CI)

| Model | Approval A | Approval B | Approval C | AIR B [95% CI] | AIR C [95% CI] |
|---|---|---|---|---|---|
| `reweighing_a0.5` | 71.7% | 55.6% | 61.3% | 0.776 [0.747, 0.807] | 0.855 [0.815, 0.892] |

Demographic-parity difference 0.161 [0.138, 0.183]; equalised-odds difference 0.097 [0.082, 0.146].

## Explanations

Four ranked SHAP reason codes for each declined applicant (3,314 of 3,314); counterfactual found for 99.9%.

## Limitations

See section 8 of `model_development_document.md`. Key items: synthetic data, fairness measured at a single cutoff, demographics assumed known, reweighing uses the protected attribute in training, no reject inference.

## Monitoring

Monthly score and feature PSI, monthly AIR per group with bootstrap CI, quarterly AUC on matured vintages, reason-code drift (thresholds in section 9 of the development document).
