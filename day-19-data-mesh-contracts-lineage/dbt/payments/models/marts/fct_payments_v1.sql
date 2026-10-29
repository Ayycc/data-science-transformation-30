select
    payment_id,
    order_id,
    paid_at,
    amount,
    currency,
    status,
    payment_method,
    loaded_at
from {{ ref('stg_payments') }}
