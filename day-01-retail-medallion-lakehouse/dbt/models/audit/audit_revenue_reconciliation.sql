-- Independent bronze -> gold revenue reconciliation (net line revenue, before returns).
-- Bronze is re-read raw and every exclusion bucket is recomputed from bronze, then
-- compared with the sum of fct_order_lines.net_amount.
with bronze_lines as (
    select
        order_line_id, store_id, nullif(upper(trim(sku)), '') as sku_code,
        cast(line_total as decimal(18, 2)) as amount, _ingested_at, _load_id
    from {{ source('bronze', 'pos_order_lines') }}
    union all
    select
        order_line_id, 0 as store_id, nullif(upper(trim(sku_code)), '') as sku_code,
        {{ cents_to_amount('unit_price_cents * quantity - discount_cents') }} as amount, _ingested_at, _load_id
    from {{ source('bronze', 'web_order_lines') }}
),

ranked as (
    select
        *,
        row_number() over (partition by order_line_id order by _ingested_at desc, _load_id desc) as copy_rank
    from bronze_lines
),

classified as (
    select
        r.*,
        case
            when r.copy_rank > 1 then 'duplicate_resend'
            when r.store_id = 999 then 'training_lane'
            when m.sku_code is null then 'unmapped_sku'
        end as bucket
    from ranked as r
    left join {{ ref('int_sku_product_map') }} as m on m.sku_code = r.sku_code
),

b as (
    select
        count(*) as bronze_rows,
        sum(amount) as bronze_revenue,
        count(*) filter (where bucket = 'duplicate_resend') as excluded_duplicate_rows,
        coalesce(sum(amount) filter (where bucket = 'duplicate_resend'), 0) as excluded_duplicate_revenue,
        count(*) filter (where bucket = 'training_lane') as excluded_training_rows,
        coalesce(sum(amount) filter (where bucket = 'training_lane'), 0) as excluded_training_revenue,
        count(*) filter (where bucket = 'unmapped_sku') as excluded_unmapped_rows,
        coalesce(sum(amount) filter (where bucket = 'unmapped_sku'), 0) as excluded_unmapped_revenue
    from classified
),

g as (
    select count(*) as gold_rows, sum(net_amount) as gold_revenue
    from {{ ref('fct_order_lines') }}
)

select
    b.*,
    b.bronze_revenue - b.excluded_duplicate_revenue - b.excluded_training_revenue
        - b.excluded_unmapped_revenue as expected_gold_revenue,
    g.gold_rows,
    g.gold_revenue,
    g.gold_revenue - (b.bronze_revenue - b.excluded_duplicate_revenue - b.excluded_training_revenue
        - b.excluded_unmapped_revenue) as difference,
    100.0 * abs(g.gold_revenue - (b.bronze_revenue - b.excluded_duplicate_revenue
        - b.excluded_training_revenue - b.excluded_unmapped_revenue)) / b.bronze_revenue as abs_difference_pct,
    {{ var('revenue_recon_tolerance_pct') }} as tolerance_pct
from b cross join g
