-- Absorbs raw schema evolution: `carrier` (r1) was renamed `carrier_code` (r2).
select
    cast(shipment_id as varchar) as shipment_id,
    cast(order_id as varchar) as order_id,
    cast(coalesce(carrier_code, carrier) as varchar) as carrier,
    cast(carrier_service as varchar) as carrier_service,
    cast(shipped_at as timestamp) as shipped_at,
    cast(delivered_at as timestamp) as delivered_at,
    cast(promised_date as date) as promised_date,
    cast(_loaded_at as timestamp) as loaded_at
from {{ source('logistics_raw', 'shipments') }}
