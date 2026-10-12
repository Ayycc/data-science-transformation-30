-- All monthly catalog extracts, typed. The SCD2 history itself lives in snap_product_catalog.
select
    cast(extract_date as date) as extract_date,
    cast(product_id as integer) as product_id,
    sku_code,
    product_name,
    category,
    brand,
    cast(list_price as decimal(18, 2)) as list_price,
    strptime(updated_at, '%Y-%m-%d %H:%M:%S') as updated_at
from {{ source('bronze', 'product_catalog') }}
