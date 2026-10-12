-- Product dimension, SCD Type 2, straight from the dbt snapshot. One row per version.
-- product_sk = product_id * 1000 + version_number (deterministic, readable).
with versions as (
    select
        *,
        row_number() over (partition by product_id order by dbt_valid_from) as version_number
    from {{ ref('snap_product_catalog') }}
)

select
    product_id * 1000 + version_number as product_sk,
    product_id,
    version_number,
    sku_code,
    product_name,
    category,
    brand,
    list_price,
    dbt_valid_from as valid_from,
    dbt_valid_to as valid_to,
    dbt_valid_to = cast('9999-12-31 00:00:00' as timestamp) as is_current,
    captured_from_extract,
    dbt_scd_id
from versions
