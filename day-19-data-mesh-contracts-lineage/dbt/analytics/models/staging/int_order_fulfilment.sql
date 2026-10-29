select
    shp.order_id,
    bool_and(shp.is_on_time) as delivered_on_time
from {{ source('logistics', 'fct_shipments_v1') }} as shp
group by shp.order_id
