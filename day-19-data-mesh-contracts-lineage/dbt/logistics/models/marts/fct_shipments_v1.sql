select
    shipment_id,
    order_id,
    carrier,
    carrier_service,
    shipped_at,
    delivered_at,
    promised_date,
    cast(delivered_at as date) <= promised_date as is_on_time,
    loaded_at
from {{ ref('stg_shipments') }}
