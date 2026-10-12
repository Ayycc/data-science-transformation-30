-- Each product has exactly one current version and no gaps between consecutive versions.
with v as (
    select
        product_id, product_sk, valid_from, valid_to, is_current,
        lead(valid_from) over (partition by product_id order by valid_from) as next_from
    from {{ ref('dim_product') }}
)
select product_id, 'gap' as problem from v where next_from is not null and next_from <> valid_to
union all
select product_id, 'current_count' from v group by product_id having sum(case when is_current then 1 else 0 end) <> 1
