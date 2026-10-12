-- Typed web lines: cents -> DECIMAL(18,2), epoch milliseconds -> UTC timestamp.
select
    order_line_id,
    order_id,
    cast(line_number as integer) as line_number,
    0 as store_id,
    cast(customer_id as integer) as customer_id,
    nullif(upper(trim(sku_code)), '') as sku_code,
    cast(quantity as integer) as quantity,
    {{ cents_to_amount('unit_price_cents') }} as unit_price,
    {{ cents_to_amount('discount_cents') }} as discount_amount,
    {{ cents_to_amount('unit_price_cents * quantity - discount_cents') }} as net_amount,
    epoch_ms(event_time_ms) as order_ts_utc,
    cast(_ingested_at as timestamp) as ingested_at,
    _load_id,
    _source_file
from {{ source('bronze', 'web_order_lines') }}
