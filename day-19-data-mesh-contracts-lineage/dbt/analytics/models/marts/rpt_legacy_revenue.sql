-- A legacy dashboard that still reads the deprecated v1 interface (it uses order_total).
-- The deprecation report in results/ finds it from OpenLineage events, before v1 is retired.
select
    o1.order_date,
    sum(o1.order_total) as completed_revenue
from {{ source('orders', 'fct_orders_v1') }} as o1
where o1.status = 'completed'
group by o1.order_date
