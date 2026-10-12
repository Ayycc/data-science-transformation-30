select
    cast(store_id as integer) as store_id,
    store_name,
    case when store_id = 0 then 'web' else 'store' end as channel,
    store_id = 999 as is_training_lane,
    city,
    state,
    timezone,
    store_format,
    cast(opened_on as date) as opened_on
from {{ source('bronze', 'stores') }}
