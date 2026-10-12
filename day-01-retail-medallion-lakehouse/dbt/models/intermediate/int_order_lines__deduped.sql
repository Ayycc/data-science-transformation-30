{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key='order_line_id',
        on_schema_change='fail',
    )
}}
-- One row per order_line_id. Incremental on the bronze load watermark (_load_id), NOT on
-- order time, so late-arriving lines (uploaded weeks after the sale) are still picked up.
-- A re-sent line replaces its earlier copy but keeps the first ingestion time, so
-- lateness is measured against when the line was first seen.
with new_rows as (
    select *
    from {{ ref('int_order_lines__unioned') }}
    {% if is_incremental() %}
    where _load_id > (select coalesce(max(last_load_id), 0) from {{ this }})
    {% endif %}
),

ranked as (
    select
        *,
        count(*) over (partition by order_line_id) as copies_in_batch,
        min(ingested_at) over (partition by order_line_id) as first_ingested_in_batch
    from new_rows
    qualify row_number() over (partition by order_line_id order by ingested_at desc, _load_id desc) = 1
)

select
    r.order_line_id,
    r.order_id,
    r.line_number,
    r.channel,
    r.store_id,
    r.customer_id,
    r.sku_code,
    r.quantity,
    r.unit_price,
    r.discount_amount,
    r.net_amount,
    r.order_ts_utc,
    cast({{ utc_to_local('r.order_ts_utc', 'r.store_timezone') }} as date) as order_date_local,
    r.store_timezone,
    r.was_exported_in_utc,
    r.sku_needed_cleaning,
    {% if is_incremental() %}
    least(r.first_ingested_in_batch, coalesce(t.first_ingested_at, r.first_ingested_in_batch)) as first_ingested_at,
    r.copies_in_batch + coalesce(t.copies_received, 0) as copies_received,
    {% else %}
    r.first_ingested_in_batch as first_ingested_at,
    r.copies_in_batch as copies_received,
    {% endif %}
    r.ingested_at as last_ingested_at,
    r._load_id as last_load_id
from ranked as r
{% if is_incremental() %}
left join {{ this }} as t on t.order_line_id = r.order_line_id
{% endif %}
