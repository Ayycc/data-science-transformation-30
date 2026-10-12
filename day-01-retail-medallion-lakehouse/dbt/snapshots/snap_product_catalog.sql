{#-
  SCD Type 2 history of the product catalog. The pipeline replays each monthly extract
  in order with --vars '{catalog_extract_date: YYYY-MM-DD}'. The timestamp strategy uses
  the source system's updated_at, so dbt_valid_from is the true change time, not the
  extract date. Without the var, the latest extract is used (a no-op once replayed).
-#}
{% snapshot snap_product_catalog %}
{{
    config(
        schema='snapshots',
        unique_key='product_id',
        strategy='timestamp',
        updated_at='updated_at',
        dbt_valid_to_current="cast('9999-12-31 00:00:00' as timestamp)",
    )
}}
select
    cast(product_id as integer) as product_id,
    sku_code,
    product_name,
    category,
    brand,
    cast(list_price as decimal(18, 2)) as list_price,
    strptime(updated_at, '%Y-%m-%d %H:%M:%S') as updated_at,
    cast(extract_date as date) as captured_from_extract
from {{ source('bronze', 'product_catalog') }}
where cast(extract_date as date) =
{%- if var('catalog_extract_date') %} cast('{{ var("catalog_extract_date") }}' as date)
{%- else %} (select max(cast(extract_date as date)) from {{ source('bronze', 'product_catalog') }})
{%- endif %}
{% endsnapshot %}
