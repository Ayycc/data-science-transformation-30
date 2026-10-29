-- Absorbs raw schema evolution: order_total (r1-r2) was renamed order_amount (r3+); retried rows are de-duplicated.
with ranked as (
    select
        *,
        row_number() over (partition by order_id order by _loaded_at desc) as rn
    from {{ source('orders_raw', 'orders') }}
)

select
    cast(order_id as varchar) as order_id,
    cast(customer_id as varchar) as customer_id,
    cast(order_ts as timestamp) as order_ts,
    cast(coalesce(order_amount, order_total) as decimal(12, 2)) as order_amount,
    cast(discount_amount as decimal(12, 2)) as discount_amount,
    cast(currency as varchar) as currency,
    cast(status as varchar) as status,
    cast(sales_channel as varchar) as sales_channel,
    cast(_loaded_at as timestamp) as loaded_at
from ranked
where rn = 1
