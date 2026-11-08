"""Ask the agent one question: python scripts/ask.py "Revenue by region in FY2025" [--llm]."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semsql import generator  # noqa: E402
from semsql.agent import ClaudeAgent, RulesAgent, llm_available  # noqa: E402
from semsql.guardrails import open_readonly  # noqa: E402
from semsql.semantic import SemanticLayer  # noqa: E402
from semsql.tools import Toolbox  # noqa: E402

logging.getLogger("sqlglot").setLevel(logging.ERROR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--llm", action="store_true", help="use Claude (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()
    db = ROOT / "data" / "warehouse.duckdb"
    if not db.exists():
        generator.build_warehouse(db)
    tb = Toolbox(SemanticLayer(), open_readonly(db))
    if args.llm and not llm_available():
        print(
            "ANTHROPIC_API_KEY is not set; using the offline rules planner.",
            file=sys.stderr,
        )
    agent = ClaudeAgent(tb) if args.llm and llm_available() else RulesAgent(tb)
    print(agent.ask(args.question).to_markdown())


if __name__ == "__main__":
    main()
