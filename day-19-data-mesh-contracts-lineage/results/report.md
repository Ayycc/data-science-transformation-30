# Day 19 run report (SYNTHETIC data)

As of 2026-09-29 06:00:00; seed 20261029; runtime 32.2 s.

## SLO status

| product | version | freshness h | completeness % | uniqueness % | rows | status |
|---|---|---|---|---|---|---|
| fct_orders_v2 | 2.1.0 | 5.0 (target 12) PASS | 100.0 (target 99.5) PASS | 100.0 (target 100) PASS | 46665 (target 30000) PASS | **PASS** |
| fct_payments_v1 | 1.2.0 | 5.0 (target 12) PASS | 100.0 (target 99.5) PASS | 100.0 (target 100) PASS | 51379 (target 30000) PASS | **PASS** |
| fct_shipments_v1 | 1.1.0 | 76.0 (target 48) FAIL | 100.0 (target 99.5) PASS | 100.0 (target 100) PASS | 42167 (target 30000) PASS | **FAIL** |

## Contract release history through the gate

| domain | version | required | actual | passed |
|---|---|---|---|---|
| orders | 1.0.0 -> 1.1.0 | minor | minor | True |
| orders | 1.1.0 -> 2.0.0 | major | major | True |
| orders | 2.0.0 -> 2.1.0 | minor | minor | True |
| payments | 1.0.0 -> 1.1.0 | minor | minor | True |
| payments | 1.1.0 -> 1.2.0 | minor | minor | True |
| logistics | 1.0.0 -> 1.0.1 | patch | patch | True |
| logistics | 1.0.1 -> 1.1.0 | minor | minor | True |

## Labelled change scenarios

| scenario | version | expected | gate | correct | first error |
|---|---|---|---|---|---|
| rename_amount_minor | 1.2.0 -> 1.3.0 | fail | fail | True | major change(s) need at least a major bump; got minor (1.2.0 -> 1.3.0) |
| rename_amount_patch | 1.2.0 -> 1.2.1 | fail | fail | True | major change(s) need at least a major bump; got patch (1.2.0 -> 1.2.1) |
| rename_amount_major_dual | 1.2.0 -> 2.0.0 | pass | pass | True |  |
| rename_amount_major_no_dual | 1.2.0 -> 2.0.0 | fail | fail | True | major release must dual-publish fct_payments_v1 (customProperties/dualPublish) for a deprecation window |
| drop_sales_channel_patch | 2.1.0 -> 2.1.1 | fail | fail | True | major change(s) need at least a major bump; got patch (2.1.0 -> 2.1.1) |
| narrow_type_minor | 2.1.0 -> 2.2.0 | fail | fail | True | major change(s) need at least a major bump; got minor (2.1.0 -> 2.2.0) |
| widen_decimal_minor | 2.1.0 -> 2.2.0 | pass | pass | True |  |
| widen_decimal_patch | 2.1.0 -> 2.1.1 | fail | fail | True | minor change(s) need at least a minor bump; got patch (2.1.0 -> 2.1.1) |
| relax_required_minor | 1.1.0 -> 1.2.0 | fail | fail | True | major change(s) need at least a major bump; got minor (1.1.0 -> 1.2.0) |
| tighten_required_minor | 1.1.0 -> 1.2.0 | pass | pass | True |  |
| loosen_freshness_patch | 1.1.0 -> 1.1.1 | fail | fail | True | major change(s) need at least a major bump; got patch (1.1.0 -> 1.1.1) |
| tighten_freshness_minor | 1.1.0 -> 1.2.0 | pass | pass | True |  |
| add_enum_value_minor | 2.1.0 -> 2.2.0 | pass | pass | True |  |
| remove_enum_value_minor | 2.1.0 -> 2.2.0 | fail | fail | True | major change(s) need at least a major bump; got minor (2.1.0 -> 2.2.0) |
| add_optional_column_minor | 1.2.0 -> 1.3.0 | pass | pass | True |  |
| add_optional_column_patch | 1.2.0 -> 1.2.1 | fail | fail | True | minor change(s) need at least a minor bump; got patch (1.2.0 -> 1.2.1) |
| change_primary_key_major_dual | 2.1.0 -> 3.0.0 | pass | pass | True |  |
| change_primary_key_minor | 2.1.0 -> 2.2.0 | fail | fail | True | major change(s) need at least a major bump; got minor (2.1.0 -> 2.2.0) |
| description_patch | 1.1.0 -> 1.1.1 | pass | pass | True |  |
| owner_handover_patch | 1.2.0 -> 1.2.1 | pass | pass | True |  |
| no_version_bump | 1.2.0 -> 1.2.0 | fail | fail | True | version must increase: 1.2.0 -> 1.2.0 |
| retire_v1_early | 2.1.0 -> 2.2.0 | fail | fail | True | major change(s) need at least a major bump; got minor (2.1.0 -> 2.2.0) |
| major_without_physical_rename | 1.1.0 -> 2.0.0 | fail | fail | True | schema/fct_shipments: physicalName fct_shipments_v1 must end with _v2 (major version) |
| invalid_contract_no_owner | 1.1.0 -> 1.1.1 | fail | fail | True | team/members: [{'username': 'x', 'role': 'engineer'}] does not contain items matching the given schema |

## dbt enforced-contract demo

- change: fct_orders_v2.sql: order_amount -> amount (contract unchanged)
- dbt build succeeded: False
- message: `Compilation Error in model fct_orders (models/marts/fct_orders_v2.sql)
  This model has an enforced contract that failed.
  Please ensure the name, data_type, and number of columns in your contract match the columns in your model's definition.
  
  | column_name  | definition_type | contract_type | `

## Consumer breakage replay

| dashboard | domain | live days | broken days (raw) | first break | broken days (mesh) | interface |
|---|---|---|---|---|---|---|
| exec_revenue | orders | 90 | 41 | 2026-08-19 | 0 | fct_orders_v1 |
| channel_mix | orders | 65 | 0 | - | 0 | fct_orders_v1 |
| basket_value | orders | 35 | 0 | - | 0 | fct_orders_v2 |
| payments_recon | payments | 90 | 55 | 2026-08-05 | 0 | fct_payments_v1 |
| carrier_scorecard | logistics | 90 | 34 | 2026-08-26 | 0 | fct_shipments_v1 |

## Lineage

- 34 OpenLineage events, 17 jobs, 128 column nodes, 107 direct edges
- `analytics.main.mart_daily_kpis.net_revenue` <- payments/payments_*.csv::amount, payments/payments_*.csv::amount_cents, payments/payments_*.csv::status (5 hops, 5 jobs)
- incl. indirect (join/filter/group-by): 8 source columns from logistics, orders, payments

## Gold KPIs (SYNTHETIC)

- 46665 orders over 90 days; GMV 2,685,405.53; net revenue 2,357,328.44 (87.8% of GMV)
- on-time delivery 82.0% of 42167 shipped orders
