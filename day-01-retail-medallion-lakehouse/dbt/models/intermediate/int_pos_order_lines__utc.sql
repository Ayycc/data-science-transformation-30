{{ config(materialized='view') }}
-- Timezone normalisation for POS lines. Stores export local wall-clock time, except
-- stores in the vendor change log (seed pos_clock_changes), which export UTC from their
-- firmware change date onwards. Stores open 08:00-22:00 local, so comparing the raw
-- timestamp with the change date is unambiguous.
select
    l.*,
    s.timezone as store_timezone,
    coalesce(c.store_id is not null and l.txn_ts_raw >= c.change_date, false) as was_exported_in_utc,
    case
        when c.store_id is not null and l.txn_ts_raw >= c.change_date then l.txn_ts_raw
        else {{ local_to_utc('l.txn_ts_raw', 's.timezone') }}
    end as order_ts_utc
from {{ ref('stg_pos__order_lines') }} as l
left join {{ ref('stg_erp__stores') }} as s on s.store_id = l.store_id
left join {{ ref('pos_clock_changes') }} as c on c.store_id = l.store_id
