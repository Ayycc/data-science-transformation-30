"""make run: build SYNTHETIC warehouse -> benchmark agents -> guardrail red team -> results/."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logging  # noqa: E402

import duckdb  # noqa: E402
import sqlglot  # noqa: E402

from semsql import generator  # noqa: E402
from semsql.agent import (  # noqa: E402
    DEFAULT_MODEL,
    ClaudeAgent,
    Conversation,
    RawRulesAgent,
    RulesAgent,
    llm_available,
)
from semsql.benchmark import evaluate, gold_answers, load_questions, summarise  # noqa: E402
from semsql.guardrails import Guardrails, open_readonly  # noqa: E402
from semsql.redteam import run_redteam  # noqa: E402
from semsql.report import CAT_ORDER, bar_compare, failure_md  # noqa: E402
from semsql.semantic import SemanticLayer  # noqa: E402
from semsql.tools import Toolbox  # noqa: E402

logging.getLogger("sqlglot").setLevel(logging.ERROR)


def write_csv(path: Path, outs: list, agent: str) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "agent",
                    "id",
                    "category",
                    "expected",
                    "status",
                    "correct",
                    "reason",
                    "traps",
                    "latency_ms",
                    "code",
                    "sql",
                ]
            )
        for o in outs:
            w.writerow(
                [
                    agent,
                    o.id,
                    o.category,
                    o.expected,
                    o.status,
                    o.correct,
                    o.reason,
                    " ".join(o.traps),
                    round(o.latency_ms, 2),
                    o.code or "",
                    (o.sql or "").replace("\n", " "),
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", type=int, default=2500)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--db", default=str(ROOT / "data" / "warehouse.duckdb"))
    args = ap.parse_args()
    t_start = time.perf_counter()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for p in ("benchmark_results.csv", "heldout_results.csv"):
        (out / p).unlink(missing_ok=True)

    # 1. data -------------------------------------------------------------------------
    cfg = generator.GenConfig(n_accounts=args.accounts)
    tables = generator.generate(cfg)
    counts = generator.write_duckdb(tables, Path(args.db))
    sample_bytes = generator.write_sample(tables, ROOT / "data" / "sample")
    print(f"[data] SYNTHETIC warehouse: {counts}; sample {sample_bytes:,} bytes")

    layer = SemanticLayer()
    con = open_readonly(args.db)
    tb = Toolbox(layer, con)
    questions = load_questions(ROOT / "benchmark" / "questions.yaml")
    heldout = load_questions(ROOT / "benchmark" / "heldout_paraphrases.yaml")
    gold = gold_answers(con, questions)
    gold_h = gold_answers(con, heldout)

    # 2. offline benchmark ------------------------------------------------------------
    results: dict[str, dict] = {}
    outcomes: dict[str, list] = {}
    for agent in (RulesAgent(tb), RawRulesAgent(tb)):
        o = evaluate(agent, questions, gold)
        outcomes[agent.name] = o
        results[agent.name] = summarise(o)
        write_csv(out / "benchmark_results.csv", o, agent.name)
        print(
            f"[bench] {agent.name}: exec acc {results[agent.name]['execution_accuracy']:.3f}, "
            f"refusal {results[agent.name]['refusal_correctness']:.3f}"
        )
    held: dict[str, dict] = {}
    held_out: dict[str, list] = {}
    for agent in (RulesAgent(Toolbox(layer, con)), RawRulesAgent(Toolbox(layer, con))):
        o = evaluate(agent, heldout, gold_h)
        held_out[agent.name] = o
        held[agent.name] = summarise(o)
        write_csv(out / "heldout_results.csv", o, agent.name)
        print(f"[heldout] {agent.name}: exec acc {held[agent.name]['execution_accuracy']:.3f}")

    # 3. live LLM (optional) ----------------------------------------------------------
    llm: dict = {"used": False, "model": os.environ.get("SEMSQL_MODEL", DEFAULT_MODEL)}
    if llm_available():
        llm["used"] = True
        for mode in ("semantic", "raw"):
            agent = ClaudeAgent(Toolbox(layer, con), mode=mode)
            o = evaluate(agent, questions, gold)
            outcomes[agent.name] = o
            results[agent.name] = summarise(o)
            write_csv(out / "benchmark_results.csv", o, agent.name)
            print(f"[llm] {agent.name}: exec acc {results[agent.name]['execution_accuracy']:.3f}")
        llm["semantic_minus_raw_execution_accuracy"] = round(
            results["claude_semantic"]["execution_accuracy"] - results["claude_raw"]["execution_accuracy"],
            4,
        )
    else:
        llm["status"] = (
            "pending: ANTHROPIC_API_KEY not set, so the Claude semantic-layer vs raw-schema "
            "comparison was not run; only offline (rules-based) results are reported"
        )
        print("[llm] skipped: ANTHROPIC_API_KEY not set (offline fallback only)")

    # 4. answer cache (stretch) -------------------------------------------------------
    t0 = time.perf_counter()
    cold_tb = Toolbox(layer, con)
    cold = RulesAgent(cold_tb)
    for q in questions:
        cold.ask(q["question"])
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    for q in questions:
        cold.ask(q["question"])
    t_warm = time.perf_counter() - t0
    cache = {
        "distinct_compiled_queries": len(cold_tb.cache),
        "warm_pass_cache_hits": cold_tb.cache_hits,
        "cold_pass_seconds": round(t_cold, 3),
        "warm_pass_seconds": round(t_warm, 3),
    }

    # 5. multi-turn clarification (stretch) -------------------------------------------
    conv = Conversation(RulesAgent(Toolbox(layer, con)))
    a1 = conv.say("Show me the numbers for EMEA")
    a2 = conv.say("revenue by fiscal quarter in FY2026")
    clar = {
        "turn1_status": a1.status,
        "turn2_status": a2.status,
        "turn2_rows": len(a2.rows),
    }

    # 6. guardrail red team -----------------------------------------------------------
    guard = Guardrails(layer.pii_columns, layer.max_rows)
    rt = run_redteam(guard, con)
    attacks = rt["attacks"]
    write_like = [a for a in attacks if a["category"] in ("write", "ddl")]
    rt_summary = {
        "attacks": len(attacks),
        "blocked_by_guardrails": sum(a["blocked"] for a in attacks),
        "write_ddl_attempts": len(write_like),
        "write_ddl_blocked_by_engine_without_guardrails": sum(bool(a["engine_blocked_without_guardrail"]) for a in write_like),
        "benign_allowed": sum(b["allowed"] for b in rt["benign"]),
        "benign_total": len(rt["benign"]),
        "row_counts_unchanged": rt["row_counts_after"] == {"accounts": counts["accounts"], "invoices": counts["invoices"]},
        "by_category": {},
    }
    for a in attacks:
        c = rt_summary["by_category"].setdefault(a["category"], {"n": 0, "blocked": 0})
        c["n"] += 1
        c["blocked"] += int(a["blocked"])

    # 7. write results ----------------------------------------------------------------
    sem, raw = results["rules_semantic"], results["rules_raw_schema"]
    metrics = {
        "data": {
            "path_used": "synthetic_generator",
            "synthetic": True,
            "seed": cfg.seed,
            "n_accounts": cfg.n_accounts,
            "row_counts": counts,
            "sample_bytes": sample_bytes,
            "period": f"{generator.DATA_START} to {generator.DATA_END} (FY2024-FY2026)",
        },
        "benchmark": {"main_60": results, "heldout_paraphrases_20": held},
        "headline": {
            "semantic_rules_execution_accuracy": sem["execution_accuracy"],
            "raw_schema_rules_execution_accuracy": raw["execution_accuracy"],
            "semantic_rules_refusal_correctness": sem["refusal_correctness"],
            "heldout_semantic_rules_execution_accuracy": held["rules_semantic"]["execution_accuracy"],
            "heldout_semantic_rules_overall_accuracy": held["rules_semantic"]["overall_accuracy"],
            "guardrail_attacks_blocked": f"{rt_summary['blocked_by_guardrails']}/{rt_summary['attacks']}",
        },
        "llm": llm,
        "guardrails": rt_summary,
        "answer_cache": cache,
        "clarification_demo": clar,
        "environment": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "sqlglot": sqlglot.__version__,
            "runtime_seconds": None,
        },
    }
    bar_compare(
        {
            "semantic layer (rules)": sem["by_category"],
            "raw schema (naive rules)": raw["by_category"],
        },
        CAT_ORDER,
        "Accuracy by question category, 60-question benchmark (SYNTHETIC)",
        out / "figures" / "accuracy_by_category.png",
    )
    # test/void are visible in the raw schema and the naive baseline handles them, so the chart
    # shows only the four semantics that live solely in the semantic layer.
    bar_compare(
        {
            "semantic layer (rules)": sem["by_trap"],
            "raw schema (naive rules)": raw["by_trap"],
        },
        ["fiscal", "fx", "churn", "point_in_time"],
        "Execution accuracy on questions that involve each business definition",
        out / "figures" / "accuracy_by_trap.png",
    )

    fa = [
        "# Failure analysis (SYNTHETIC data)",
        "",
        "Generated by `make run` from `benchmark/questions.yaml` and `benchmark/heldout_paraphrases.yaml`.",
        "",
        "## Main 60-question benchmark",
        "",
    ]
    for name, o in outcomes.items():
        fa.append(failure_md(name, o))
    fa += [
        "## Raw-schema baseline: which trap breaks it",
        "",
        "Traps co-occur (most questions carry a fiscal period), so the counts below overlap. The naive "
        "baseline filters `is_test` and `void` correctly because those are visible in the raw schema; its "
        "failures on questions tagged `test`/`void` come from a co-occurring fiscal, FX, churn or "
        "point-in-time trap.",
        "",
        "| trap | answerable questions with trap | semantic correct | raw-schema correct |",
        "|---|---|---|---|",
    ]
    for t, d in sem["by_trap"].items():
        fa.append(f"| {t} | {d['n']} | {d['correct']} | {raw['by_trap'][t]['correct']} |")
    fa += ["", "## Held-out paraphrases (20, not used for planner development)", ""]
    for name, o in held_out.items():
        fa.append(failure_md(name, o))
    (out / "failure_analysis.md").write_text("\n".join(fa), encoding="utf-8")

    gr = [
        "# Guardrail red-team report",
        "",
        f"{rt_summary['blocked_by_guardrails']}/{rt_summary['attacks']} attacks blocked by the sqlglot guardrails; "
        f"{rt_summary['write_ddl_blocked_by_engine_without_guardrails']}/{rt_summary['write_ddl_attempts']} write/DDL "
        "statements also fail when sent straight to the read-only DuckDB connection (guardrails bypassed). "
        f"Benign queries allowed: {rt_summary['benign_allowed']}/{rt_summary['benign_total']}. "
        f"Row counts unchanged after the suite: {rt_summary['row_counts_unchanged']}.",
        "",
        "| category | SQL | blocked | engine rejects without guardrail | first reason |",
        "|---|---|---|---|---|",
    ]
    for a in attacks:
        eb = "" if a["engine_blocked_without_guardrail"] is None else str(a["engine_blocked_without_guardrail"])
        gr.append(
            f"| {a['category']} | `{a['sql'].replace('|', '/')}` | {a['blocked']} | {eb} | {(a['reasons'] or [''])[0].replace('|', '/')} |"
        )
    gr += ["", "Benign queries (allowed, row limit enforced by rewriting):", ""]
    for b in rt["benign"]:
        gr.append(f"- `{b['sql']}` -> `{b['rewritten']}`")
    (out / "guardrail_report.md").write_text("\n".join(gr) + "\n", encoding="utf-8")

    ex_ids = ["q05", "q14", "q21", "q37", "q43", "q47", "q51", "q55", "q58", "q60"]
    demo = RulesAgent(Toolbox(layer, con))
    tr = ["# Example agent transcripts (offline rules planner, SYNTHETIC data)", ""]
    for q in questions:
        if q["id"] in ex_ids:
            tr += [
                f"## {q['id']} ({q['category']})",
                "",
                demo.ask(q["question"]).to_markdown(),
                "",
            ]
    tr += [
        "## Multi-turn clarification",
        "",
        "Turn 1:",
        "",
        a1.to_markdown(),
        "",
        "Turn 2 (user replies 'revenue by fiscal quarter in FY2026'):",
        "",
        a2.to_markdown(),
        "",
    ]
    (out / "agent_traces.md").write_text("\n".join(tr), encoding="utf-8")

    metrics["environment"]["runtime_seconds"] = round(time.perf_counter() - t_start, 2)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(metrics["headline"], indent=2))
    print(f"[done] {metrics['environment']['runtime_seconds']} s")
    con.close()


if __name__ == "__main__":
    main()
