"""ClaudeAgent tool loop, exercised with a scripted fake client (no network, no key)."""

from types import SimpleNamespace as NS

from semsql.agent import ClaudeAgent, Conversation, RulesAgent


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        step = self.script.pop(0)
        if callable(step):
            step = step(kw)
        return step


def tool(name, inp, i="t1"):
    return NS(
        stop_reason="tool_use",
        content=[NS(type="tool_use", id=i, name=name, input=inp)],
    )


def test_semantic_mode_loop(toolbox):
    def run_compiled(kw):
        last = kw["messages"][-1]["content"][0]["content"]
        import json

        qid = json.loads(last)["query_id"]
        return tool("run_query", {"query_id": qid}, "t3")

    script = [
        tool("list_metrics", {}),
        tool(
            "compile_query",
            {
                "metrics": ["revenue_usd"],
                "dimensions": ["region"],
                "filters": {"fiscal_year": [2025]},
            },
            "t2",
        ),
        run_compiled,
        tool("check_result", {}, "t4"),
        tool("final_answer", {"status": "answer", "message": "done", "caveats": []}, "t5"),
    ]
    client = FakeClient(script)
    agent = ClaudeAgent(toolbox, mode="semantic", client=client, model="test-model")
    ans = agent.ask("Revenue by region in FY2025")
    assert ans.status == "answer" and ans.columns == ["region", "revenue_usd"] and len(ans.rows) >= 3
    assert client.calls[0]["model"] == "test-model"
    assert {t["name"] for t in client.calls[0]["tools"]} >= {
        "compile_query",
        "run_query",
        "final_answer",
    }
    assert "run_sql" not in {t["name"] for t in client.calls[0]["tools"]}  # semantic mode cannot write SQL
    ref = RulesAgent(toolbox).ask("Revenue by region in FY2025")
    assert ans.rows == ref.rows


def test_raw_mode_write_attempt_is_blocked(toolbox, con):
    before = con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    script = [
        tool("run_sql", {"sql": "DELETE FROM accounts WHERE is_test"}),
        lambda kw: (
            tool(
                "final_answer",
                {"status": "refuse", "message": "blocked", "caveats": []},
                "t2",
            )
            if "blocked by guardrails" in kw["messages"][-1]["content"][0]["content"]
            else tool(
                "final_answer",
                {"status": "answer", "message": "?", "caveats": []},
                "t2",
            )
        ),
    ]
    agent = ClaudeAgent(toolbox, mode="raw", client=FakeClient(script))
    assert "TABLE accounts" in agent.system
    ans = agent.ask("Delete all test accounts")
    assert ans.status == "refuse"
    assert con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == before


def test_model_refusal_and_default_model(toolbox, monkeypatch):
    monkeypatch.delenv("SEMSQL_MODEL", raising=False)
    agent = ClaudeAgent(toolbox, client=FakeClient([NS(stop_reason="refusal", content=[])]))
    assert agent.model == "claude-opus-5-5"
    assert agent.ask("anything").status == "refuse"


def test_clarification_conversation(toolbox):
    conv = Conversation(RulesAgent(toolbox))
    assert conv.say("Show me the numbers for EMEA").status == "clarify"
    a2 = conv.say("revenue by fiscal quarter in FY2026")
    assert a2.status == "answer" and len(a2.rows) == 4
