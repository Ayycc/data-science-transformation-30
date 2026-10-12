{#- Interpret a naive wall-clock timestamp in `tz` and return the naive UTC timestamp. -#}
{% macro local_to_utc(ts, tz) -%}
    timezone('UTC', timezone({{ tz }}, {{ ts }}))
{%- endmacro %}

{#- Convert a naive UTC timestamp to naive wall-clock time in `tz`. -#}
{% macro utc_to_local(ts, tz) -%}
    timezone({{ tz }}, timezone('UTC', {{ ts }}))
{%- endmacro %}

{#- Integer cents (or an expression in cents) to an exact DECIMAL(18,2) amount. -#}
{% macro cents_to_amount(expr) -%}
    cast(cast({{ expr }} as decimal(18, 0)) * 0.01 as decimal(18, 2))
{%- endmacro %}
