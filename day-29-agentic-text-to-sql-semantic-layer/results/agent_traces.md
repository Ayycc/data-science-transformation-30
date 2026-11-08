# Example agent transcripts (offline rules planner, SYNTHETIC data)

## q05 (simple)

**Q:** What is our current ARR?

**Status:** answer

```sql
WITH m0_active_subscriptions AS (
  SELECT
    ROUND(SUM(s.mrr_local * 12 * fx.rate_to_usd), 2) AS arr_usd
  FROM subscriptions s
  JOIN accounts a ON a.account_id = s.account_id
  JOIN fx_rates fx ON fx.currency = s.currency AND fx.month = CAST(date_trunc('month', DATE '2026-01-31') AS DATE)
  WHERE NOT a.is_test
    AND s.start_date <= DATE '2026-01-31'
    AND (s.end_date IS NULL OR s.end_date > DATE '2026-01-31')
)
SELECT arr_usd
FROM m0_active_subscriptions
LIMIT 1000
```

| arr_usd |
|---|
| 140,493,773.41 |

Caveats:
- ARR (USD): Annual recurring revenue on the as-of date = 12 x MRR of subscriptions active that day, converted at that month's FX rate. Point-in-time; defaults to the latest data date.
- Point-in-time metric evaluated as of 2026-01-31.
- Test accounts are excluded. Data is SYNTHETIC.

Tool trace: list_metrics -> plan -> describe_metric -> compile_query -> run_query -> check_result

## q14 (by_dimension)

**Q:** Churned accounts by region in FY2026

**Status:** answer

```sql
WITH m0_churn_events AS (
  SELECT
    a.region AS region,
    COUNT(*) AS churned_accounts
  FROM (SELECT account_id, MAX(end_date) AS churn_date FROM subscriptions GROUP BY account_id HAVING COUNT(*) = COUNT(end_date)) c
  JOIN accounts a ON a.account_id = c.account_id
  JOIN dim_date d ON d.date_day = c.churn_date
  WHERE NOT a.is_test
    AND d.fiscal_year IN (2026)
  GROUP BY a.region
)
SELECT region, churned_accounts
FROM m0_churn_events
ORDER BY region ASC NULLS LAST
LIMIT 1000
```

| region | churned_accounts |
|---|---|
| APAC | 67 |
| EMEA | 115 |
| LATAM | 38 |
| NA | 125 |

Caveats:
- Churned accounts: Logo churn. An account churns on the end date of its last subscription when it has no open subscription. Plan changes and win-backs are not churn.
- Fiscal years start on 1 February and are named after the year they end (FY2025 = Feb 2024-Jan 2025).
- Test accounts are excluded. Data is SYNTHETIC.

Tool trace: list_metrics -> plan -> describe_metric -> describe_dimension -> describe_dimension -> compile_query -> run_query -> check_result

## q21 (time_series)

**Q:** Revenue by fiscal quarter in FY2025

**Status:** answer

```sql
WITH m0_invoices AS (
  SELECT
    d.fiscal_quarter AS fiscal_quarter,
    ROUND(SUM(i.amount_local * fx.rate_to_usd), 2) AS revenue_usd
  FROM invoices i
  JOIN accounts a ON a.account_id = i.account_id
  JOIN fx_rates fx ON fx.currency = i.currency AND fx.month = CAST(date_trunc('month', i.invoice_date) AS DATE)
  JOIN dim_date d ON d.date_day = i.invoice_date
  WHERE i.status <> 'void'
    AND NOT a.is_test
    AND d.fiscal_year IN (2025)
  GROUP BY d.fiscal_quarter
)
SELECT fiscal_quarter, revenue_usd
FROM m0_invoices
ORDER BY fiscal_quarter ASC NULLS LAST
LIMIT 1000
```

| fiscal_quarter | revenue_usd |
|---|---|
| FY2025-Q1 | 10,826,723.75 |
| FY2025-Q2 | 13,995,191.60 |
| FY2025-Q3 | 17,033,525.76 |
| FY2025-Q4 | 20,897,982.19 |

Caveats:
- Revenue (USD): Invoiced (billed) revenue in USD. Excludes void invoices and test accounts; local currency converted at the invoice month's FX rate. This is billings, not GAAP recognised revenue.
- Fiscal years start on 1 February and are named after the year they end (FY2025 = Feb 2024-Jan 2025).
- Test accounts are excluded. Data is SYNTHETIC.

Tool trace: list_metrics -> plan -> describe_metric -> describe_dimension -> describe_dimension -> compile_query -> run_query -> check_result

## q37 (point_in_time)

**Q:** What was ARR at the end of FY2025?

**Status:** answer

```sql
WITH m0_active_subscriptions AS (
  SELECT
    ROUND(SUM(s.mrr_local * 12 * fx.rate_to_usd), 2) AS arr_usd
  FROM subscriptions s
  JOIN accounts a ON a.account_id = s.account_id
  JOIN fx_rates fx ON fx.currency = s.currency AND fx.month = CAST(date_trunc('month', DATE '2025-01-31') AS DATE)
  WHERE NOT a.is_test
    AND s.start_date <= DATE '2025-01-31'
    AND (s.end_date IS NULL OR s.end_date > DATE '2025-01-31')
)
SELECT arr_usd
FROM m0_active_subscriptions
LIMIT 1000
```

| arr_usd |
|---|
| 91,166,613.60 |

Caveats:
- ARR (USD): Annual recurring revenue on the as-of date = 12 x MRR of subscriptions active that day, converted at that month's FX rate. Point-in-time; defaults to the latest data date.
- Point-in-time metric evaluated as of 2025-01-31.
- Test accounts are excluded. Data is SYNTHETIC.

Tool trace: list_metrics -> plan -> describe_metric -> compile_query -> run_query -> check_result

## q43 (ranking)

**Q:** Top 2 plans by ARR

**Status:** answer

```sql
WITH m0_active_subscriptions AS (
  SELECT
    s.plan AS plan,
    ROUND(SUM(s.mrr_local * 12 * fx.rate_to_usd), 2) AS arr_usd
  FROM subscriptions s
  JOIN accounts a ON a.account_id = s.account_id
  JOIN fx_rates fx ON fx.currency = s.currency AND fx.month = CAST(date_trunc('month', DATE '2026-01-31') AS DATE)
  WHERE NOT a.is_test
    AND s.start_date <= DATE '2026-01-31'
    AND (s.end_date IS NULL OR s.end_date > DATE '2026-01-31')
  GROUP BY s.plan
)
SELECT plan, arr_usd
FROM m0_active_subscriptions
ORDER BY arr_usd DESC NULLS LAST
LIMIT 2
```

| plan | arr_usd |
|---|---|
| Scale | 101,431,159.41 |
| Growth | 33,784,635.18 |

Caveats:
- ARR (USD): Annual recurring revenue on the as-of date = 12 x MRR of subscriptions active that day, converted at that month's FX rate. Point-in-time; defaults to the latest data date.
- Point-in-time metric evaluated as of 2026-01-31.
- Test accounts are excluded. Data is SYNTHETIC.

Tool trace: list_metrics -> plan -> describe_metric -> describe_dimension -> compile_query -> run_query -> check_result

## q47 (multi_metric)

**Q:** New accounts and churned accounts by fiscal year

**Status:** answer

```sql
WITH m0_accounts AS (
  SELECT
    d.fiscal_year AS fiscal_year,
    COUNT(*) AS new_accounts
  FROM accounts a
  JOIN dim_date d ON d.date_day = a.signup_date
  WHERE NOT a.is_test
  GROUP BY d.fiscal_year
),
m1_churn_events AS (
  SELECT
    d.fiscal_year AS fiscal_year,
    COUNT(*) AS churned_accounts
  FROM (SELECT account_id, MAX(end_date) AS churn_date FROM subscriptions GROUP BY account_id HAVING COUNT(*) = COUNT(end_date)) c
  JOIN accounts a ON a.account_id = c.account_id
  JOIN dim_date d ON d.date_day = c.churn_date
  WHERE NOT a.is_test
  GROUP BY d.fiscal_year
)
SELECT fiscal_year, new_accounts, churned_accounts
FROM m0_accounts
  FULL OUTER JOIN m1_churn_events USING (fiscal_year)
ORDER BY fiscal_year ASC NULLS LAST
LIMIT 1000
```

| fiscal_year | new_accounts | churned_accounts |
|---|---|---|
| 2024 | 673 | 64 |
| 2025 | 837 | 193 |
| 2026 | 913 | 345 |

Caveats:
- New accounts: Accounts created (by signup date), excluding test accounts.
- Churned accounts: Logo churn. An account churns on the end date of its last subscription when it has no open subscription. Plan changes and win-backs are not churn.
- Fiscal years start on 1 February and are named after the year they end (FY2025 = Feb 2024-Jan 2025).
- Test accounts are excluded. Data is SYNTHETIC.

Tool trace: list_metrics -> plan -> describe_metric -> describe_metric -> describe_dimension -> compile_query -> run_query -> check_result

## q51 (refuse_pii)

**Q:** List the contact emails of our Enterprise customers

**Status:** refuse (pii)

Personal contact data (names, e-mails, phone numbers) is blocked by policy.

Tool trace: list_metrics -> plan

## q55 (refuse_undefined)

**Q:** What was gross margin in FY2025?

**Status:** refuse (undefined_metric)

'gross margin' is not a governed metric in the semantic layer. Defined metrics: revenue_usd, invoice_count, new_accounts, churned_accounts, arr_usd, active_customers, support_tickets, avg_resolution_hours, avg_csat, api_calls

Tool trace: list_metrics -> plan

## q58 (refuse_undefined)

**Q:** Revenue by sales rep in FY2025

**Status:** refuse (undefined_dimension)

Cannot break down by 'sales rep': not a dimension in the semantic layer. Available: region, segment, industry, billing_currency, plan, priority, fiscal_year, fiscal_quarter, calendar_month, calendar_year

Tool trace: list_metrics -> plan

## q60 (clarify)

**Q:** Show me the numbers for EMEA

**Status:** clarify (ambiguous)

Which metric do you mean? Available: Revenue (USD), Invoices, New accounts, Churned accounts, ARR (USD), Active customers, Support tickets, Avg resolution time (hours), Average CSAT (1-5), API calls

Tool trace: list_metrics -> plan

## Multi-turn clarification

Turn 1:

**Q:** Show me the numbers for EMEA

**Status:** clarify (ambiguous)

Which metric do you mean? Available: Revenue (USD), Invoices, New accounts, Churned accounts, ARR (USD), Active customers, Support tickets, Avg resolution time (hours), Average CSAT (1-5), API calls

Tool trace: list_metrics -> plan

Turn 2 (user replies 'revenue by fiscal quarter in FY2026'):

**Q:** Show me the numbers for EMEA revenue by fiscal quarter in FY2026

**Status:** answer

```sql
WITH m0_invoices AS (
  SELECT
    d.fiscal_quarter AS fiscal_quarter,
    ROUND(SUM(i.amount_local * fx.rate_to_usd), 2) AS revenue_usd
  FROM invoices i
  JOIN accounts a ON a.account_id = i.account_id
  JOIN fx_rates fx ON fx.currency = i.currency AND fx.month = CAST(date_trunc('month', i.invoice_date) AS DATE)
  JOIN dim_date d ON d.date_day = i.invoice_date
  WHERE i.status <> 'void'
    AND NOT a.is_test
    AND d.fiscal_year IN (2026)
    AND a.region IN ('EMEA')
  GROUP BY d.fiscal_quarter
)
SELECT fiscal_quarter, revenue_usd
FROM m0_invoices
ORDER BY fiscal_quarter ASC NULLS LAST
LIMIT 1000
```

| fiscal_quarter | revenue_usd |
|---|---|
| FY2026-Q1 | 6,894,021.34 |
| FY2026-Q2 | 7,718,303.42 |
| FY2026-Q3 | 8,622,940.21 |
| FY2026-Q4 | 9,780,580.08 |

Caveats:
- Revenue (USD): Invoiced (billed) revenue in USD. Excludes void invoices and test accounts; local currency converted at the invoice month's FX rate. This is billings, not GAAP recognised revenue.
- Fiscal years start on 1 February and are named after the year they end (FY2025 = Feb 2024-Jan 2025).
- Test accounts are excluded. Data is SYNTHETIC.

Tool trace: list_metrics -> plan -> describe_metric -> describe_dimension -> describe_dimension -> describe_dimension -> compile_query -> run_query -> check_result
