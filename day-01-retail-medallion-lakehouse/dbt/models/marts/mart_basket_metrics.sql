-- Basket (order) metrics by month and channel.
with baskets as (
    select
        f.order_id,
        f.channel,
        cast(date_trunc('month', min(f.order_date_local)) as date) as order_month,
        count(*) as order_lines,
        sum(f.quantity) as units,
        sum(f.net_amount) as basket_value,
        count(distinct p.category) as categories
    from {{ ref('fct_order_lines') }} as f
    inner join {{ ref('dim_product') }} as p on p.product_sk = f.product_sk
    group by f.order_id, f.channel
)

select
    order_month,
    channel,
    count(*) as orders,
    sum(basket_value) as net_sales,
    avg(basket_value) as avg_basket_value,
    quantile_cont(basket_value, 0.5) as median_basket_value,
    avg(units) as avg_units_per_basket,
    avg(order_lines) as avg_lines_per_basket,
    avg(case when categories > 1 then 1.0 else 0.0 end) as share_multi_category
from baskets
group by order_month, channel
