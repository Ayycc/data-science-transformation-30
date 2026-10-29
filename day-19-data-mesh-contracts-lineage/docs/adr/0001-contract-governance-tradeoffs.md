# ADR 0001: Governing domain data products with contracts, semantic versions and lineage

- **Status:** accepted (portfolio project, SYNTHETIC data)
- **Date:** 2026-10-29
- **Deciders:** platform team, the orders, payments and logistics domain owners, and analytics

## Context

Ownership of data moved to three domains (orders, payments, logistics). Each domain ships source-system changes on its own schedule. Downstream dashboards read domain tables directly, so a column rename breaks them, often weekly. We need a way to let domains move fast without breaking consumers, and to know who is affected when a change is necessary.

## Decision

1. **The contract is the interface.** Each data product has one ODCS-style contract (`contracts/<domain>/*.odcs.yaml`). The contract records the schema, SLAs, owner and a semantic version, and it is validated with JSON Schema. dbt model-contract YAML is *generated* from it, and dbt `contract: enforced` fails the build if the SQL output drifts.
2. **Semantic versioning is checked from the consumer's side.** A CI gate (`scripts/check_contracts.py`) diffs every contract against the PR base. Removals, renames, narrowing type changes, relaxed nullability, removed enum values, weaker SLAs and primary-key changes all need a **major** bump. Additive changes need a **minor** bump, and metadata changes a **patch**. Released contract files in `history/` are immutable.
3. **Majors are published side by side, and old majors are deprecated with a date.** A major release must keep the previous major published (`fct_orders_v1` next to `fct_orders_v2`) until a `deprecationDate`. Both versions come from one dbt versioned model. Consumers pin a major through the physical name.
4. **Lineage is emitted, not inferred later.** Every ingest, dbt model and SLO check emits OpenLineage events with column-level lineage. Impact analysis and the deprecation report ("who still reads v1?") are computed from those events.
5. **SLAs are part of the contract and are checked on every run.** Each product reports PASS or FAIL per SLA (freshness, completeness, uniqueness, volume). An SLA breach is reported to the owner but **does not block** publishing.

## Options considered

| Option | Why not (alone) |
|---|---|
| Central data team owns all models | It brings back the bottleneck the mesh was meant to remove, and domain knowledge is lost. |
| Consumers read raw domain tables (status quo) | Every source change reaches dashboards. In the replay, 3 of 5 dashboards broke for 130 of 370 dashboard-days. |
| Schema registry only (e.g. Avro or Protobuf compatibility) | It covers structure but not SLAs, ownership, semantics (enum meaning) or the deprecation lifecycle. |
| dbt model contracts only | They catch drift between SQL and YAML. They do not catch a domain editing *both* the SQL and the YAML in a breaking way. The semver gate does. |
| Block publishing on any SLA breach | A late logistics batch would then also stop correct orders and payments data from publishing. Alerting the owner and labelling the product was preferred. |

## Consequences and trade-offs

- **Producer friction vs consumer safety.** A rename now costs a major release, a dual-publish window and a deprecation date. In return, consumers get a stable interface. Domains can still change their *raw* feeds freely, because staging models absorb those changes. In this project, 5 raw changes reached 0 consumers.
- **Storage and compute for dual-publishing.** v1 is a second table derived from v2. That cost is acceptable for a bounded window, and the deprecation report shows when the window can close safely.
- **The rules are opinionated.** For example, adding an enum value is treated as minor, although strict consumers that handle every value explicitly may disagree. Rules live in one module (`mesh/diff.py`) with a labelled test suite, so changing a rule is a reviewed code change.
- **Column lineage comes from static SQL analysis** (sqlglot over dbt compiled SQL). Columns reached only through CTE aliases (for example the `rn = 1` dedup filter) are not resolved as indirect lineage. Python transformations need hand-written facets.
- **No central runtime is required.** Events are files, and a lineage server (Marquez) is optional. The cost is that nothing enforces that every producer emits events. That rule belongs in the platform's paved road, meaning its templates and CI.

## Follow-ups

- Make the contract gate a required status check on every domain repository (see `ci/contract-gate.yml`).
- Fail CI when a deprecated interface passes its `deprecationDate` while lineage still shows readers.
- Add a contract for the consumer mart itself, so the analytics team becomes a producer too.
