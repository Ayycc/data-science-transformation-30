-- Grain: one in-scope order line. product_sk is the SCD2 version valid at the moment of
-- sale (UTC), so historical lines keep the name, SKU code and list price they sold under.
with lines as (
    select * from {{ ref('int_order_lines__classified') }}
    where exclusion_reason is null
)

select
    l.order_line_id,
    l.order_id,
    l.line_number,
    l.channel,
    l.store_id,
    coalesce(l.customer_id, 0) as customer_id,
    p.product_sk,
    l.product_id,
    cast(strftime(l.order_date_local, '%Y%m%d') as integer) as date_key,
    l.order_date_local,
    l.order_ts_utc,
    l.sku_code as sku_code_at_sale,
    l.quantity,
    l.unit_price,
    cast(l.quantity * l.unit_price as decimal(18, 2)) as gross_amount,
    l.discount_amount,
    l.net_amount,
    l.first_ingested_at,
    l.is_late_arrival,
    l.was_exported_in_utc
from lines as l
left join {{ ref('dim_product') }} as p
    on p.product_id = l.product_id
    and l.order_ts_utc >= p.valid_from
    and l.order_ts_utc < p.valid_to
