"""Naive raw-schema baseline (offline): same parsed intent, SQL guessed from the raw schema.

This simulates the failure mode in the business brief: a text-to-SQL system that sees only
table and column names. It is "schema-literate" (it filters the obviously named `is_test`
flag and `void` status) but it cannot know the business definitions that live only in the
semantic layer:

* fiscal calendar  -> uses calendar years/quarters for "FY2025" / "fiscal quarter";
* currency         -> sums local-currency amounts as if they were USD;
* churn            -> counts subscriptions with status 'cancelled' (includes plan changes);
* point-in-time    -> ARR / active customers from status = 'active' (today), ignoring as-of.

It is a heuristic stand-in, not an LLM. The LLM raw-schema comparison runs only with a key.
"""

from __future__ import annotations

from .semantic import QuerySpec, _lit

RAW_METRICS = {
    "revenue_usd": ("inv", "SUM(i.amount_local)"),
    "invoice_count": ("inv", "COUNT(*)"),
    "new_accounts": ("acc", "COUNT(*)"),
    "churned_accounts": ("churn", "COUNT(*)"),
    "arr_usd": ("subs", "SUM(s.mrr_local * 12)"),
    "active_customers": ("subs", "COUNT(DISTINCT s.account_id)"),
    "support_tickets": ("tkt", "COUNT(*)"),
    "avg_resolution_hours": (
        "tkt",
        "AVG(date_diff('second', t.opened_at, t.resolved_at) / 3600.0)",
    ),
    "avg_csat": ("tkt", "AVG(t.csat_score)"),
    "api_calls": ("use", "SUM(u.api_calls)"),
}
RAW_MODELS = {
    "inv": {
        "from": "invoices i JOIN accounts a ON a.account_id = i.account_id",
        "time": "i.invoice_date",
        "where": ["i.status <> 'void'", "NOT a.is_test"],
        "plan_join": "JOIN subscriptions s ON s.subscription_id = i.subscription_id",
    },
    "acc": {"from": "accounts a", "time": "a.signup_date", "where": ["NOT a.is_test"]},
    "churn": {
        "from": "subscriptions s JOIN accounts a ON a.account_id = s.account_id",
        "time": "s.end_date",
        "where": ["s.status = 'cancelled'", "NOT a.is_test"],
    },
    "subs": {
        "from": "subscriptions s JOIN accounts a ON a.account_id = s.account_id",
        "time": "s.start_date",
        "where": ["s.status = 'active'", "NOT a.is_test"],
    },
    "tkt": {
        "from": "support_tickets t JOIN accounts a ON a.account_id = t.account_id",
        "time": "CAST(t.opened_at AS DATE)",
        "where": ["NOT a.is_test"],
    },
    "use": {
        "from": "usage_monthly u JOIN accounts a ON a.account_id = u.account_id",
        "time": "u.usage_month",
        "where": ["NOT a.is_test"],
    },
}
RAW_DIMS = {
    "region": "a.region",
    "segment": "a.segment",
    "industry": "a.industry",
    "billing_currency": "a.billing_currency",
    "plan": "s.plan",
    "priority": "t.priority",
}


def _dim_sql(dim: str, time_col: str) -> str:
    if dim in RAW_DIMS:
        return RAW_DIMS[dim]
    if dim in ("fiscal_year", "calendar_year"):
        return f"year({time_col})"
    if dim == "fiscal_quarter":
        return f"'FY' || year({time_col}) || '-Q' || quarter({time_col})"
    if dim == "calendar_month":
        return f"strftime({time_col}, '%Y-%m')"
    raise KeyError(dim)


def naive_raw_sql(spec: QuerySpec) -> str:
    groups: dict[str, list[str]] = {}
    for m in spec.metrics:
        groups.setdefault(RAW_METRICS[m][0], []).append(m)
    ctes = []
    for k, (model_key, metrics) in enumerate(groups.items()):
        mdl = RAW_MODELS[model_key]
        tc = mdl["time"]
        frm = mdl["from"]
        if "plan" in [*spec.dimensions, *spec.filters] and "plan_join" in mdl:
            frm += " " + mdl["plan_join"]
        where = list(mdl["where"])
        for d, vals in spec.filters.items():
            if d == "fiscal_quarter":
                conds = []
                for v in vals:
                    fy, qn = v[2:6], v[-1]
                    conds.append(f"(year({tc}) = {fy} AND quarter({tc}) = {qn})")
                where.append("(" + " OR ".join(conds) + ")")
            else:
                where.append(f"{_dim_sql(d, tc)} IN ({', '.join(_lit(v) for v in vals)})")
        sel = [f"{_dim_sql(d, tc)} AS {d}" for d in spec.dimensions]
        sel += [f"{RAW_METRICS[m][1]} AS {m}" for m in metrics]
        body = f"SELECT {', '.join(sel)} FROM {frm} WHERE {' AND '.join(where)}"
        if spec.dimensions:
            body += " GROUP BY " + ", ".join(str(i + 1) for i in range(len(spec.dimensions)))
        ctes.append((f"r{k}", body))
    sql = (
        "WITH "
        + ", ".join(f"{n} AS ({b})" for n, b in ctes)
        + f" SELECT {', '.join([*spec.dimensions, *spec.metrics])} FROM {ctes[0][0]}"
    )
    for n, _ in ctes[1:]:
        sql += f" FULL OUTER JOIN {n} USING ({', '.join(spec.dimensions)})" if spec.dimensions else f" CROSS JOIN {n}"
    order = spec.order_by or [[d, "asc"] for d in spec.dimensions]
    if order:
        sql += " ORDER BY " + ", ".join(f"{c} {d.upper()}" for c, d in order)
    if spec.limit:
        sql += f" LIMIT {spec.limit}"
    return sql
