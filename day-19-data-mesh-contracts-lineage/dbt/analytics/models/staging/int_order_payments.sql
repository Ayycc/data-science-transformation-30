select
    pay.order_id,
    sum(case when pay.status = 'captured' then pay.amount else 0 end) as captured_amount,
    sum(case when pay.status = 'refunded' then pay.amount else 0 end) as refunded_amount
from {{ source('payments', 'fct_payments_v1') }} as pay
where pay.order_id is not null
group by pay.order_id
