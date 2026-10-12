-- Customer dimension (Type 1, latest CRM record) plus the anonymous member 0 used for
-- POS sales without a loyalty card. First-purchase attributes come from silver lines.
with first_orders as (
    select
        customer_id,
        min(order_ts_utc) as first_order_ts_utc,
        min(order_date_local) as first_order_date
    from {{ ref('int_order_lines__classified') }}
    where exclusion_reason is null and customer_id is not null
    group by customer_id
)

select
    c.customer_id,
    c.customer_ref,
    c.first_name,
    c.last_name,
    c.email,
    c.city,
    c.state,
    c.loyalty_tier,
    c.signup_ts,
    f.first_order_ts_utc,
    f.first_order_date,
    cast(date_trunc('month', f.first_order_date) as date) as first_order_month,
    c.signup_ts >= cast('{{ var("window_start") }}' as timestamp) as signed_up_in_window
from {{ ref('stg_crm__customers') }} as c
left join first_orders as f on f.customer_id = c.customer_id
union all
select
    0, 'ANON', null, null, null, null, null, 'None', null, null, null, null, false
