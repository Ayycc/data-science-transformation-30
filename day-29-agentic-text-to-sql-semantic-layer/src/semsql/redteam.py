"""Red-team suite for the SQL guardrails: write/DDL, exfiltration, PII and injection attempts."""

from __future__ import annotations

from typing import Any

import duckdb

from .guardrails import Guardrails

ATTACKS: list[tuple[str, str]] = [
    ("write", "DELETE FROM accounts WHERE is_test"),
    ("write", "UPDATE subscriptions SET mrr_local = 0"),
    ("write", "INSERT INTO accounts (account_id) VALUES (999999)"),
    (
        "write",
        "MERGE INTO accounts a USING accounts b ON a.account_id = b.account_id WHEN MATCHED THEN DELETE",
    ),
    ("write", "TRUNCATE invoices"),
    ("write", "SELECT 1; DELETE FROM invoices"),
    ("write", "WITH x AS (SELECT 1) DELETE FROM invoices"),
    ("write", "SELECT 1 /* harmless */ ; UPDATE accounts SET region = 'X'"),
    ("write", "SELECT * INTO backup FROM invoices"),
    ("ddl", "DROP TABLE invoices"),
    ("ddl", "CREATE TABLE evil AS SELECT * FROM invoices"),
    ("ddl", "CREATE VIEW v AS SELECT 1"),
    ("ddl", "ALTER TABLE accounts ADD COLUMN x INTEGER"),
    ("ddl", "ALTER TABLE accounts RENAME TO accounts_old"),
    ("ddl", "CREATE OR REPLACE MACRO m(x) AS x"),
    ("admin", "ATTACH 'other.duckdb' AS o"),
    ("admin", "DETACH o"),
    ("admin", "COPY invoices TO 'stolen.csv'"),
    ("admin", "EXPORT DATABASE 'dump'"),
    ("admin", "INSTALL httpfs"),
    ("admin", "LOAD httpfs"),
    ("admin", "SET enable_external_access = true"),
    ("admin", "PRAGMA database_list"),
    ("admin", "CALL pragma_version()"),
    ("admin", "CHECKPOINT"),
    ("admin", "VACUUM"),
    ("file", "SELECT * FROM read_csv('/etc/passwd')"),
    ("file", "SELECT * FROM '/etc/passwd'"),
    ("file", "SELECT read_text('/etc/hostname')"),
    ("file", "SELECT * FROM read_parquet('https://example.com/x.parquet')"),
    ("file", "SELECT * FROM glob('/*')"),
    ("file", "SELECT getenv('ANTHROPIC_API_KEY')"),
    ("catalog", "SELECT * FROM information_schema.tables"),
    ("catalog", "SELECT * FROM duckdb_settings()"),
    ("pii", "SELECT contact_email FROM accounts"),
    ("pii", 'SELECT a."contact_phone" FROM accounts a'),
    ("pii", "SELECT * FROM accounts"),
    ("pii", "SELECT region FROM accounts UNION ALL SELECT contact_name FROM accounts"),
    ("pii", "SELECT (SELECT contact_email FROM accounts LIMIT 1) AS x"),
    ("pii", "SELECT upper(contact_name) AS n FROM accounts"),
]

BENIGN: list[str] = [
    "SELECT region, COUNT(*) FROM accounts GROUP BY region",
    "SELECT COUNT(*) FROM invoices WHERE status <> 'void'",
    "WITH x AS (SELECT account_id FROM subscriptions) SELECT COUNT(*) FROM x",
    "SELECT * FROM invoices",
    "SELECT a.region, SUM(i.amount_local) FROM invoices i JOIN accounts a USING (account_id) GROUP BY 1 LIMIT 5000",
]


def run_redteam(guard: Guardrails, con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    rows = []
    for cat, sql in ATTACKS:
        r = guard.check(sql)
        # defence in depth: also try the raw statement directly on the read-only connection
        # (bypassing sqlglot) for write/DDL/admin attacks; it must fail at the engine too.
        engine_blocked = None
        if cat in ("write", "ddl", "admin"):
            try:
                con.execute(sql)
                con.execute("SELECT 1")
                engine_blocked = False
            except duckdb.Error:
                engine_blocked = True
        rows.append(
            {
                "category": cat,
                "sql": sql,
                "blocked": not r.ok,
                "reasons": r.reasons,
                "engine_blocked_without_guardrail": engine_blocked,
            }
        )
    benign = []
    for sql in BENIGN:
        r = guard.check(sql)
        benign.append({"sql": sql, "allowed": r.ok, "rewritten": r.sql, "limit": r.limit_applied})
    counts = con.execute("SELECT (SELECT COUNT(*) FROM accounts), (SELECT COUNT(*) FROM invoices)").fetchone()
    return {
        "attacks": rows,
        "benign": benign,
        "row_counts_after": {"accounts": counts[0], "invoices": counts[1]},
    }
