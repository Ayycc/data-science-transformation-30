-- Every SKU code ever issued, mapped to its stable product_id (codes are never reused).
-- Built from SCD2 history so re-coded SKUs keep resolving for old receipts.
select
    sku_code,
    min(product_id) as product_id,
    count(distinct product_id) as n_products,
    min(dbt_valid_from) as first_valid_from,
    max(dbt_valid_to) as last_valid_to
from {{ ref('snap_product_catalog') }}
group by sku_code
