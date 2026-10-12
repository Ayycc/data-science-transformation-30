-- Monthly acquisition cohorts: customers who signed up inside the data window, by month
-- of first purchase. Pre-existing customers are left-censored and excluded.
with activity as (
    select
        f.customer_id,
        cast(date_trunc('month', f.order_date_local) as date) as activity_month,
        sum(f.net_amount) as net_sales
    from {{ ref('fct_order_lines') }} as f
    inner join {{ ref('dim_customer') }} as c on c.customer_id = f.customer_id
    where f.customer_id <> 0 and c.signed_up_in_window
    group by all
),

cohorts as (
    select customer_id, min(activity_month) as cohort_month
    from activity
    group by customer_id
),

sizes as (
    select cohort_month, count(*) as cohort_size from cohorts group by cohort_month
)

select
    c.cohort_month,
    date_diff('month', c.cohort_month, a.activity_month) as months_since_first_order,
    s.cohort_size,
    count(distinct a.customer_id) as active_customers,
    count(distinct a.customer_id) / s.cohort_size as retention_rate,
    sum(a.net_sales) as net_sales,
    sum(a.net_sales) / s.cohort_size as net_sales_per_cohort_customer
from activity as a
inner join cohorts as c on c.customer_id = a.customer_id
inner join sizes as s on s.cohort_month = c.cohort_month
group by c.cohort_month, months_since_first_order, s.cohort_size
