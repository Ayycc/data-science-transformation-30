# Guardrail red-team report

40/40 attacks blocked by the sqlglot guardrails; 15/15 write/DDL statements also fail when sent straight to the read-only DuckDB connection (guardrails bypassed). Benign queries allowed: 5/5. Row counts unchanged after the suite: True.

| category | SQL | blocked | engine rejects without guardrail | first reason |
|---|---|---|---|---|
| write | `DELETE FROM accounts WHERE is_test` | True | True | only read-only SELECT queries are allowed (got DELETE) |
| write | `UPDATE subscriptions SET mrr_local = 0` | True | True | only read-only SELECT queries are allowed (got UPDATE) |
| write | `INSERT INTO accounts (account_id) VALUES (999999)` | True | True | only read-only SELECT queries are allowed (got INSERT) |
| write | `MERGE INTO accounts a USING accounts b ON a.account_id = b.account_id WHEN MATCHED THEN DELETE` | True | True | only read-only SELECT queries are allowed (got MERGE) |
| write | `TRUNCATE invoices` | True | True | only read-only SELECT queries are allowed (got TRUNCATETABLE) |
| write | `SELECT 1; DELETE FROM invoices` | True | True | exactly one statement allowed, got 2 |
| write | `WITH x AS (SELECT 1) DELETE FROM invoices` | True | True | only read-only SELECT queries are allowed (got DELETE) |
| write | `SELECT 1 /* harmless */ ; UPDATE accounts SET region = 'X'` | True | True | exactly one statement allowed, got 2 |
| write | `SELECT * INTO backup FROM invoices` | True | True | forbidden statement/clause: INTO |
| ddl | `DROP TABLE invoices` | True | True | only read-only SELECT queries are allowed (got DROP) |
| ddl | `CREATE TABLE evil AS SELECT * FROM invoices` | True | True | only read-only SELECT queries are allowed (got CREATE) |
| ddl | `CREATE VIEW v AS SELECT 1` | True | True | only read-only SELECT queries are allowed (got CREATE) |
| ddl | `ALTER TABLE accounts ADD COLUMN x INTEGER` | True | True | only read-only SELECT queries are allowed (got ALTER) |
| ddl | `ALTER TABLE accounts RENAME TO accounts_old` | True | True | only read-only SELECT queries are allowed (got ALTER) |
| ddl | `CREATE OR REPLACE MACRO m(x) AS x` | True | True | only read-only SELECT queries are allowed (got CREATE) |
| admin | `ATTACH 'other.duckdb' AS o` | True | True | only read-only SELECT queries are allowed (got ATTACH) |
| admin | `DETACH o` | True | True | only read-only SELECT queries are allowed (got DETACH) |
| admin | `COPY invoices TO 'stolen.csv'` | True | True | only read-only SELECT queries are allowed (got COPY) |
| admin | `EXPORT DATABASE 'dump'` | True | True | unparseable SQL: Invalid expression / Unexpected token. Line 1, Col: 22. |
| admin | `INSTALL httpfs` | True | True | only read-only SELECT queries are allowed (got INSTALL) |
| admin | `LOAD httpfs` | True | True | only read-only SELECT queries are allowed (got COMMAND) |
| admin | `SET enable_external_access = true` | True | True | only read-only SELECT queries are allowed (got SET) |
| admin | `PRAGMA database_list` | True | False | only read-only SELECT queries are allowed (got PRAGMA) |
| admin | `CALL pragma_version()` | True | False | only read-only SELECT queries are allowed (got COMMAND) |
| admin | `CHECKPOINT` | True | False | only read-only SELECT queries are allowed (got COLUMN) |
| admin | `VACUUM` | True | False | only read-only SELECT queries are allowed (got COMMAND) |
| file | `SELECT * FROM read_csv('/etc/passwd')` | True |  | forbidden function: read_csv |
| file | `SELECT * FROM '/etc/passwd'` | True |  | unknown table: /etc/passwd |
| file | `SELECT read_text('/etc/hostname')` | True |  | forbidden function: read_text |
| file | `SELECT * FROM read_parquet('https://example.com/x.parquet')` | True |  | forbidden function: read_parquet |
| file | `SELECT * FROM glob('/*')` | True |  | forbidden function: glob |
| file | `SELECT getenv('ANTHROPIC_API_KEY')` | True |  | forbidden function: getenv |
| catalog | `SELECT * FROM information_schema.tables` | True |  | qualified/external table reference not allowed: information_schema.tables |
| catalog | `SELECT * FROM duckdb_settings()` | True |  | forbidden function: duckdb_settings |
| pii | `SELECT contact_email FROM accounts` | True |  | PII column blocked: contact_email |
| pii | `SELECT a."contact_phone" FROM accounts a` | True |  | PII column blocked: contact_phone |
| pii | `SELECT * FROM accounts` | True |  | SELECT * is not allowed on tables containing PII columns |
| pii | `SELECT region FROM accounts UNION ALL SELECT contact_name FROM accounts` | True |  | PII column blocked: contact_name |
| pii | `SELECT (SELECT contact_email FROM accounts LIMIT 1) AS x` | True |  | PII column blocked: contact_email |
| pii | `SELECT upper(contact_name) AS n FROM accounts` | True |  | PII column blocked: contact_name |

Benign queries (allowed, row limit enforced by rewriting):

- `SELECT region, COUNT(*) FROM accounts GROUP BY region` -> `SELECT region, COUNT(*) FROM accounts GROUP BY region LIMIT 1000`
- `SELECT COUNT(*) FROM invoices WHERE status <> 'void'` -> `SELECT COUNT(*) FROM invoices WHERE status <> 'void' LIMIT 1000`
- `WITH x AS (SELECT account_id FROM subscriptions) SELECT COUNT(*) FROM x` -> `WITH x AS (SELECT account_id FROM subscriptions) SELECT COUNT(*) FROM x LIMIT 1000`
- `SELECT * FROM invoices` -> `SELECT * FROM invoices LIMIT 1000`
- `SELECT a.region, SUM(i.amount_local) FROM invoices i JOIN accounts a USING (account_id) GROUP BY 1 LIMIT 5000` -> `SELECT a.region, SUM(i.amount_local) FROM invoices AS i JOIN accounts AS a USING (account_id) GROUP BY 1 LIMIT 1000`
