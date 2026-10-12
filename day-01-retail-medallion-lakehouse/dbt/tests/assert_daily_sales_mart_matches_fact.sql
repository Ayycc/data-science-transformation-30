-- The daily sales mart must add up to the fact table (sales and refunds).
with m as (select sum(net_sales) as net, sum(refunds) as refunds from {{ ref('mart_daily_sales') }}),
f as (select sum(net_amount) as net from {{ ref('fct_order_lines') }}),
r as (select coalesce(sum(refund_amount), 0) as refunds from {{ ref('fct_returns') }})
select * from m, f, r
where m.net <> f.net or m.refunds <> r.refunds
