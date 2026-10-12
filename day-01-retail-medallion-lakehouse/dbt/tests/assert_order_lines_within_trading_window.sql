-- The generator's trading window is defined on local business days (store-local for POS),
-- so UTC timestamps may spill up to the largest US offset (10h for Hawaii; we use 12h)
-- past the UTC window edges. Anything beyond that is a timestamp-parsing bug.
select order_line_id, order_ts_utc
from {{ ref('fct_order_lines') }}
where order_ts_utc < cast('{{ var("window_start") }}' as timestamp) - interval 12 hour
   or order_ts_utc >= cast('{{ var("window_start") }}' as timestamp) + interval 2 year + interval 12 hour
