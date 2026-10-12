-- SCD2 correctness: the version joined to each order line must carry exactly the SKU
-- code and list price printed on the receipt. A renamed / re-coded SKU sold before the
-- change must resolve to the old version, and after it to the new one.
select f.order_line_id, f.sku_code_at_sale, p.sku_code, f.unit_price, p.list_price
from {{ ref('fct_order_lines') }} as f
inner join {{ ref('dim_product') }} as p on p.product_sk = f.product_sk
where f.sku_code_at_sale <> p.sku_code or f.unit_price <> p.list_price
