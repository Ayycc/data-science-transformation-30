-- Absorbs raw schema evolution: dollars `amount` (r1) became integer `amount_cents` (r2).
select
    cast(payment_id as varchar) as payment_id,
    cast(order_id as varchar) as order_id,
    cast(paid_at as timestamp) as paid_at,
    cast(coalesce(amount_cents / 100.0, amount) as decimal(12, 2)) as amount,
    cast(currency as varchar) as currency,
    cast(status as varchar) as status,
    cast(payment_method as varchar) as payment_method,
    cast(_loaded_at as timestamp) as loaded_at
from {{ source('payments_raw', 'payments') }}
