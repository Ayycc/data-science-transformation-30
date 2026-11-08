"""60-question execution-accuracy benchmark with refusal correctness and failure analysis."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import yaml

from .tools import _jsonable

DEFAULT_QUESTIONS = Path(__file__).resolve().parents[2] / "benchmark" / "questions.yaml"


def load_questions(path: Path | str = DEFAULT_QUESTIONS) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def gold_answers(con: duckdb.DuckDBPyConnection, questions: list[dict[str, Any]]) -> dict[str, list[list[Any]]]:
    out = {}
    for q in questions:
        if q.get("gold_sql"):
            out[q["id"]] = [[_jsonable(v) for v in r] for r in con.execute(q["gold_sql"]).fetchall()]
    return out


def _eq(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=0.01)
    return str(a) == str(b)


def _key(row: list[Any]) -> tuple:
    return tuple((0, round(float(v), 1)) if isinstance(v, (int, float)) and not isinstance(v, bool) else (1, str(v)) for v in row)


def results_match(gold: list[list[Any]], pred: list[list[Any]], ordered: bool = False) -> tuple[bool, str]:
    """Execution match: same multiset of rows (or same sequence if ordered), columns matched
    up to a permutation, numbers within abs 0.01 / rel 1e-6."""
    if len(gold) != len(pred):
        return False, f"row count {len(pred)} != gold {len(gold)}"
    if not gold:
        return True, "both empty"
    if len(gold[0]) != len(pred[0]):
        return False, f"column count {len(pred[0])} != gold {len(gold[0])}"
    ncol = len(gold[0])
    g_rows = gold if ordered else sorted(gold, key=_key)
    for perm in itertools.permutations(range(ncol)):
        p = [[r[i] for i in perm] for r in pred]
        p_rows = p if ordered else sorted(p, key=_key)
        if all(_eq(a, b) for gr, pr in zip(g_rows, p_rows) for a, b in zip(gr, pr)):
            return True, "match"
    return False, "values differ"


@dataclass
class Outcome:
    id: str
    category: str
    question: str
    expected: str
    status: str
    correct: bool
    reason: str
    traps: list[str]
    latency_ms: float
    sql: str | None
    code: str | None
    pred_rows: int
    gold_rows: int


def evaluate(agent: Any, questions: list[dict[str, Any]], gold: dict[str, list[list[Any]]]) -> list[Outcome]:
    outs = []
    for q in questions:
        t0 = time.perf_counter()
        ans = agent.ask(q["question"])
        ms = (time.perf_counter() - t0) * 1000
        exp = q["expected"]
        if exp == "answer":
            if ans.status != "answer":
                ok, why = (
                    False,
                    f"expected an answer, got {ans.status}: {ans.message[:120]}",
                )
            else:
                ok, why = results_match(gold[q["id"]], ans.rows, bool(q.get("ordered")))
        else:
            ok = ans.status == exp
            why = "correct non-answer" if ok else f"expected {exp}, got {ans.status}"
        outs.append(
            Outcome(
                q["id"],
                q["category"],
                q["question"],
                exp,
                ans.status,
                ok,
                why,
                list(q.get("traps", [])),
                ms,
                ans.sql,
                ans.code,
                len(ans.rows),
                len(gold.get(q["id"], [])),
            )
        )
    return outs


def summarise(outs: list[Outcome]) -> dict[str, Any]:
    ans = [o for o in outs if o.expected == "answer"]
    non = [o for o in outs if o.expected != "answer"]
    cats: dict[str, list[Outcome]] = {}
    for o in outs:
        cats.setdefault(o.category, []).append(o)
    traps: dict[str, list[Outcome]] = {}
    for o in ans:
        for t in o.traps:
            traps.setdefault(t, []).append(o)
    lat = sorted(o.latency_ms for o in outs)
    return {
        "n_questions": len(outs),
        "n_answerable": len(ans),
        "n_should_not_answer": len(non),
        "overall_accuracy": round(sum(o.correct for o in outs) / len(outs), 4),
        "execution_accuracy": round(sum(o.correct for o in ans) / len(ans), 4),
        "execution_correct": sum(o.correct for o in ans),
        "refusal_correctness": round(sum(o.correct for o in non) / len(non), 4) if non else None,
        "refusal_correct": sum(o.correct for o in non),
        "non_answer_recall": round(sum(o.status in ("refuse", "clarify") for o in non) / len(non), 4) if non else None,
        "false_refusal_rate": round(sum(o.status in ("refuse", "clarify") for o in ans) / len(ans), 4),
        "wrong_answer_rate": round(sum(o.status == "answer" and not o.correct for o in ans) / len(ans), 4),
        "unsafe_answers": sum(o.status == "answer" for o in non),
        "by_category": {
            c: {
                "n": len(v),
                "correct": sum(o.correct for o in v),
                "accuracy": round(sum(o.correct for o in v) / len(v), 4),
            }
            for c, v in cats.items()
        },
        "by_trap": {
            t: {
                "n": len(v),
                "correct": sum(o.correct for o in v),
                "accuracy": round(sum(o.correct for o in v) / len(v), 4),
            }
            for t, v in sorted(traps.items())
        },
        "latency_ms_p50": round(lat[len(lat) // 2], 2),
        "latency_ms_max": round(lat[-1], 2),
    }
