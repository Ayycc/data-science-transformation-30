-- Typed POS lines. No filtering here: dedup, timezone fixes and exclusions happen downstream.
select
    order_line_id,
    receipt_no as order_id,
    cast(line_no as integer) as line_number,
    cast(store_id as integer) as store_id,
    case when loyalty_card is not null then cast(substr(loyalty_card, 2) as integer) end as customer_id,
    nullif(upper(trim(sku)), '') as sku_code,
    sku <> upper(trim(sku)) as sku_needed_cleaning,
    cast(qty as integer) as quantity,
    cast(unit_price as decimal(18, 2)) as unit_price,
    cast(discount as decimal(18, 2)) as discount_amount,
    cast(line_total as decimal(18, 2)) as net_amount,
    strptime(txn_ts, '%Y-%m-%d %H:%M:%S') as txn_ts_raw,
    cast(_ingested_at as timestamp) as ingested_at,
    _load_id,
    _source_file
from {{ source('bronze', 'pos_order_lines') }}
