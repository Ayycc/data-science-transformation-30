select r.order_line_id, sum(r.returned_qty) as returned, max(f.quantity) as sold
from {{ ref('fct_returns') }} as r
inner join {{ ref('fct_order_lines') }} as f on f.order_line_id = r.order_line_id
group by r.order_line_id
having sum(r.returned_qty) > max(f.quantity) or sum(r.refund_amount) > max(f.net_amount)
