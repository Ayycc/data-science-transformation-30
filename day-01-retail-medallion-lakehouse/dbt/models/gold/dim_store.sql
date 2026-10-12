-- Store dimension (120 stores + web store 0). The POS training lane (999) is excluded.
select store_id, store_name, channel, city, state, timezone, store_format, opened_on
from {{ ref('stg_erp__stores') }}
where not is_training_lane
