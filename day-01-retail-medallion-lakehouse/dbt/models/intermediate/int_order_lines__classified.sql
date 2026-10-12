-- Resolve SKU -> product and classify every deduplicated line as in-scope or excluded.
-- Exclusions are the documented reconciliation buckets: training lane and unmapped SKU.
select
    l.*,
    m.product_id,
    l.first_ingested_at > l.order_ts_utc + interval {{ var('late_arrival_hours') }} hour as is_late_arrival,
    case
        when l.store_id = 999 then 'training_lane'
        when m.product_id is null then 'unmapped_sku'
    end as exclusion_reason
from {{ ref('int_order_lines__deduped') }} as l
left join {{ ref('int_sku_product_map') }} as m on m.sku_code = l.sku_code
