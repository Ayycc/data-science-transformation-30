-- No two versions of the same product may be valid at the same instant, and every
-- version's window must be non-empty.
select a.product_id, a.product_sk, b.product_sk as overlapping_sk
from {{ ref('dim_product') }} as a
inner join {{ ref('dim_product') }} as b
    on a.product_id = b.product_id
    and a.product_sk < b.product_sk
    and a.valid_from < b.valid_to
    and b.valid_from < a.valid_to
union all
select product_id, product_sk, null from {{ ref('dim_product') }} where valid_from >= valid_to
