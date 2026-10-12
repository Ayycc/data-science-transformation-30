-- Data-quality scorecard: one row per check, with observed value, threshold and pass flag.
with lines as (select * from {{ ref('int_order_lines__classified') }}),
recon as (select * from {{ ref('audit_revenue_reconciliation') }}),
pos as (select * from {{ ref('stg_pos__order_lines') }}),
versions as (select * from {{ ref('dim_product') }}),
version_overlaps as (
    select count(*) as n
    from versions as a
    inner join versions as b
        on a.product_id = b.product_id and a.product_sk < b.product_sk
        and a.valid_from < b.valid_to and b.valid_from < a.valid_to
),
checks as (
    select 'uniqueness' as dimension, 'bronze_duplicate_row_rate' as check_name,
        recon.excluded_duplicate_rows / recon.bronze_rows as observed, 0.05 as threshold, '<=' as op
    from recon
    union all
    select 'uniqueness', 'gold_duplicate_order_lines',
        (select count(*) - count(distinct order_line_id) from {{ ref('fct_order_lines') }}), 0, '<='
    union all
    select 'validity', 'unmapped_sku_rate',
        avg(case when exclusion_reason = 'unmapped_sku' then 1.0 else 0.0 end), 0.005, '<='
    from lines
    union all
    select 'validity', 'pos_line_total_mismatches',
        count(*) filter (where quantity * unit_price - discount_amount <> net_amount), 0, '<='
    from pos
    union all
    select 'timeliness', 'late_arrival_rate',
        avg(case when is_late_arrival then 1.0 else 0.0 end), 0.05, '<='
    from lines
    union all
    select 'consistency', 'pos_rows_clock_corrected_share',
        avg(case when was_exported_in_utc then 1.0 else 0.0 end), 1.0, '<='
    from lines where channel = 'store'
    union all
    select 'consistency', 'revenue_recon_abs_diff_pct', abs_difference_pct, tolerance_pct, '<='
    from recon
    union all
    select 'consistency', 'scd2_overlapping_versions', n, 0, '<=' from version_overlaps
    union all
    select 'completeness', 'lines_without_product_version',
        (select count(*) from {{ ref('fct_order_lines') }} where product_sk is null), 0, '<='
    union all
    select 'completeness', 'returns_awaiting_order_line',
        (select count(*) from {{ ref('int_returns__unmatched') }}), 0, '<='
)

select
    dimension,
    check_name,
    cast(observed as double) as observed_value,
    cast(threshold as double) as threshold,
    op as comparison,
    cast(observed as double) <= cast(threshold as double) as passed
from checks
