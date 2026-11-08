import duckdb
import pytest

from semsql.guardrails import GuardrailViolation, guarded_execute
from semsql.redteam import ATTACKS, BENIGN, run_redteam

WRITES = [sql for cat, sql in ATTACKS if cat in ("write", "ddl")]


@pytest.mark.parametrize("cat,sql", ATTACKS)
def test_every_attack_is_blocked(guard, cat, sql):
    assert not guard.check(sql).ok, sql


@pytest.mark.parametrize("sql", WRITES)
def test_write_and_ddl_never_execute_through_guarded_path(con, guard, sql):
    with pytest.raises(GuardrailViolation):
        guarded_execute(con, guard, sql)


@pytest.mark.parametrize("sql", WRITES)
def test_read_only_engine_rejects_writes_even_without_guardrails(con, sql):
    with pytest.raises(duckdb.Error):
        con.execute(sql)


def test_toolbox_run_query_blocks_and_data_unchanged(toolbox, con):
    before = con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    for sql in WRITES:
        res = toolbox.run_query(sql=sql)
        assert not res["ok"] and res.get("blocked"), sql
    assert con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == before


def test_settings_are_locked(con):
    with pytest.raises(duckdb.Error):
        con.execute("SET enable_external_access = true")


@pytest.mark.parametrize("sql", BENIGN)
def test_benign_queries_allowed(guard, sql):
    assert guard.check(sql).ok


def test_row_limit_added_and_capped(guard, con):
    r = guard.check("SELECT * FROM invoices")
    assert r.ok and r.limit_applied == guard.max_rows and "LIMIT 1000" in r.sql
    r = guard.check("SELECT * FROM invoices LIMIT 50000")
    assert r.limit_applied == guard.max_rows
    r = guard.check("SELECT * FROM invoices LIMIT 5")
    assert r.limit_applied == 5
    res = guarded_execute(con, guard, "SELECT * FROM invoices")
    assert len(res.rows) == guard.max_rows and res.truncated
    res = guarded_execute(con, guard, "SELECT region FROM accounts UNION SELECT plan FROM subscriptions")
    assert len(res.rows) <= guard.max_rows


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT contact_email FROM accounts",
        "SELECT lower(CONTACT_NAME) FROM accounts",
        "SELECT x FROM (SELECT contact_phone AS x FROM accounts) t",
        "SELECT accounts.* FROM accounts",
        "SELECT * FROM invoices JOIN accounts USING (account_id)",
    ],
)
def test_pii_blocked(guard, sql):
    r = guard.check(sql)
    assert not r.ok and any("PII" in x for x in r.reasons)


def test_star_allowed_without_pii_tables(guard):
    assert guard.check("SELECT * FROM invoices").ok


def test_unparseable_and_multi_statement(guard):
    assert not guard.check("SELEC region FROM").ok
    assert not guard.check("SELECT 1; SELECT 2").ok


def test_redteam_summary(guard, con):
    rt = run_redteam(guard, con)
    assert all(a["blocked"] for a in rt["attacks"])
    assert all(b["allowed"] for b in rt["benign"])
