-- Gold net revenue must reconcile to bronze within tolerance after documented exclusions.
select *
from {{ ref('audit_revenue_reconciliation') }}
where abs_difference_pct > tolerance_pct
