-- Gold cross-domain mart: one row per order date.
select
    o.order_date,
    count(*) as orders,
    cast(sum(o.order_amount) as decimal(14, 2)) as gmv,
    cast(sum(coalesce(p.captured_amount, 0) - coalesce(p.refunded_amount, 0)) as decimal(14, 2)) as net_revenue,
    avg(case when f.delivered_on_time then 1.0 when not f.delivered_on_time then 0.0 end) as on_time_delivery_rate
from {{ source('orders', 'fct_orders_v2') }} as o
left join {{ ref('int_order_payments') }} as p on o.order_id = p.order_id
left join {{ ref('int_order_fulfilment') }} as f on o.order_id = f.order_id
group by o.order_date
