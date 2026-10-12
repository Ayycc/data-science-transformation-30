{{ config(materialized='view') }}
-- Returns whose order line has not landed yet (late-arriving lines). Held back from
-- fct_returns until the line arrives; should be empty after a complete load.
select r.*
from {{ ref('stg_ops__order_returns') }} as r
left join {{ ref('fct_order_lines') }} as f on f.order_line_id = r.order_line_id
where f.order_line_id is null
