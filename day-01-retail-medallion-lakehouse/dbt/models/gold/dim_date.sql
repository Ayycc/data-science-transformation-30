select
    cast(strftime(d, '%Y%m%d') as integer) as date_key,
    cast(d as date) as date_day,
    year(d) as year,
    quarter(d) as quarter,
    month(d) as month,
    monthname(d) as month_name,
    cast(date_trunc('month', d) as date) as month_start,
    cast(date_trunc('week', d) as date) as week_start,
    isodow(d) as iso_day_of_week,
    dayname(d) as day_name,
    isodow(d) in (6, 7) as is_weekend
from range(date '2024-01-01', date '2027-01-01', interval 1 day) as t(d)
