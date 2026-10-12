{{ config(materialized='view') }}
-- Both channels in one shape, still including duplicates, training-lane and unmapped rows.
select
    order_line_id, order_id, line_number, 'store' as channel, store_id, customer_id, sku_code,
    quantity, unit_price, discount_amount, net_amount, order_ts_utc, store_timezone,
    was_exported_in_utc, sku_needed_cleaning, ingested_at, _load_id
from {{ ref('int_pos_order_lines__utc') }}
union all
select
    w.order_line_id, w.order_id, w.line_number, 'web' as channel, w.store_id, w.customer_id, w.sku_code,
    w.quantity, w.unit_price, w.discount_amount, w.net_amount, w.order_ts_utc, s.timezone as store_timezone,
    false as was_exported_in_utc, false as sku_needed_cleaning, w.ingested_at, w._load_id
from {{ ref('stg_web__order_lines') }} as w
left join {{ ref('stg_erp__stores') }} as s on s.store_id = w.store_id
