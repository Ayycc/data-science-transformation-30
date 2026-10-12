-- Grain: business date (store-local; web uses America/New_York) x channel x store.
-- Refunds are attributed to the date the return happened.
with sales as (
    select
        order_date_local as date_day,
        channel,
        store_id,
        count(distinct order_id) as orders,
        count(*) as order_lines,
        sum(quantity) as units,
        sum(gross_amount) as gross_sales,
        sum(discount_amount) as discounts,
        sum(net_amount) as net_sales,
        count(distinct nullif(customer_id, 0)) as identified_customers
    from {{ ref('fct_order_lines') }}
    group by all
),

refunds as (
    select
        return_date_local as date_day,
        channel,
        store_id,
        count(*) as returns,
        sum(returned_qty) as returned_units,
        sum(refund_amount) as refunds
    from {{ ref('fct_returns') }}
    group by all
),

joined as (
    select
        coalesce(s.date_day, r.date_day) as date_day,
        coalesce(s.channel, r.channel) as channel,
        coalesce(s.store_id, r.store_id) as store_id,
        coalesce(s.orders, 0) as orders,
        coalesce(s.order_lines, 0) as order_lines,
        coalesce(s.units, 0) as units,
        coalesce(s.gross_sales, 0) as gross_sales,
        coalesce(s.discounts, 0) as discounts,
        coalesce(s.net_sales, 0) as net_sales,
        coalesce(s.identified_customers, 0) as identified_customers,
        coalesce(r.returns, 0) as returns,
        coalesce(r.returned_units, 0) as returned_units,
        coalesce(r.refunds, 0) as refunds
    from sales as s
    full outer join refunds as r
        on s.date_day = r.date_day and s.channel = r.channel and s.store_id = r.store_id
)

select
    cast(strftime(date_day, '%Y%m%d') as integer) as date_key,
    *,
    net_sales - refunds as net_sales_after_returns
from joined
