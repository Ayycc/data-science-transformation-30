-- Grain: one return. Only returns whose order line is in fct_order_lines; the rest wait
-- in int_returns__unmatched until their (late) line lands.
select
    r.return_id,
    r.order_line_id,
    f.order_id,
    f.channel,
    f.store_id,
    f.customer_id,
    f.product_sk,
    f.product_id,
    r.return_channel,
    r.returned_qty,
    r.refund_amount,
    r.reason_code,
    r.return_ts_utc,
    cast({{ utc_to_local('r.return_ts_utc', 's.timezone') }} as date) as return_date_local,
    cast(strftime(cast({{ utc_to_local('r.return_ts_utc', 's.timezone') }} as date), '%Y%m%d') as integer) as date_key,
    date_diff('day', f.order_ts_utc, r.return_ts_utc) as days_to_return
from {{ ref('stg_ops__order_returns') }} as r
inner join {{ ref('fct_order_lines') }} as f on f.order_line_id = r.order_line_id
left join {{ ref('dim_store') }} as s on s.store_id = f.store_id
