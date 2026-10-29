-- Latest major (contract 2.x): order_total renamed to order_amount, status includes 'returned'.
select
    order_id,
    customer_id,
    order_ts,
    cast(order_ts as date) as order_date,
    order_amount,
    discount_amount,
    currency,
    status,
    sales_channel,
    loaded_at
from {{ ref('stg_orders') }}
