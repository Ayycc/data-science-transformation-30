-- Dual-published v1 (contract 1.1.0) derived from v2 until its deprecation date.
-- Keeps the old column name and the old status domain so v1 consumers see no change.
select
    order_id,
    customer_id,
    order_ts,
    order_date,
    order_amount as order_total,
    currency,
    case when status = 'returned' then 'completed' else status end as status,
    sales_channel,
    loaded_at
from {{ ref('fct_orders', v=2) }}
