-- Returns, deduplicated on return_id (re-sent rows keep the latest ingestion).
select
    return_id,
    order_line_id,
    return_channel,
    cast(returned_qty as integer) as returned_qty,
    {{ cents_to_amount('refund_cents') }} as refund_amount,
    reason_code,
    strptime(return_ts, '%Y-%m-%dT%H:%M:%SZ') as return_ts_utc,
    cast(_ingested_at as timestamp) as ingested_at,
    _load_id
from {{ source('bronze', 'order_returns') }}
qualify row_number() over (partition by return_id order by _ingested_at desc, _load_id desc) = 1
