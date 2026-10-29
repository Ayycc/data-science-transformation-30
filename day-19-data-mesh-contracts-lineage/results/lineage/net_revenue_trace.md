# Column lineage: `analytics.main.mart_daily_kpis.net_revenue`

Reconstructed only from `openlineage_events.jsonl` (direct column lineage). SYNTHETIC data.

Source columns: `payments/payments_*.csv::amount`, `payments/payments_*.csv::amount_cents`, `payments/payments_*.csv::status`

```mermaid
flowchart LR
    n0["analytics.int_order_payments.captured_amount"]
    n1["analytics.int_order_payments.refunded_amount"]
    n2["analytics.mart_daily_kpis.net_revenue"]
    n3["payments.fct_payments_v1.amount"]
    n4["payments.fct_payments_v1.status"]
    n5["payments.stg_payments.amount"]
    n6["payments.stg_payments.status"]
    n7["payments.raw.payments.amount"]
    n8["payments.raw.payments.amount_cents"]
    n9["payments.raw.payments.status"]
    n10["file:payments/*.csv.amount"]
    n11["file:payments/*.csv.amount_cents"]
    n12["file:payments/*.csv.status"]
    n0 -->|AGGREGATION| n2
    n1 -->|AGGREGATION| n2
    n3 -->|AGGREGATION| n0
    n3 -->|AGGREGATION| n1
    n4 -->|AGGREGATION| n0
    n4 -->|AGGREGATION| n1
    n5 -->|IDENTITY| n3
    n6 -->|IDENTITY| n4
    n7 -->|TRANSFORMATION| n5
    n8 -->|TRANSFORMATION| n5
    n9 -->|IDENTITY| n6
    n10 -->|IDENTITY| n7
    n11 -->|IDENTITY| n8
    n12 -->|IDENTITY| n9
```
