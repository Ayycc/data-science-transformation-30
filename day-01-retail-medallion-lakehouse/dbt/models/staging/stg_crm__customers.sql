-- CRM customers, keeping only the most recently updated record per customer_id.
select
    cast(customer_id as integer) as customer_id,
    customer_ref,
    first_name,
    last_name,
    lower(email) as email,
    city,
    state,
    loyalty_tier,
    strptime(signup_ts, '%Y-%m-%d %H:%M:%S') as signup_ts,
    strptime(updated_at, '%Y-%m-%d %H:%M:%S') as updated_at
from {{ source('bronze', 'customers') }}
qualify row_number() over (partition by customer_id order by updated_at desc) = 1
