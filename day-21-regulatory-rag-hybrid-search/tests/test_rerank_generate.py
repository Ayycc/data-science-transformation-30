from types import SimpleNamespace

from regrag.chunking import Chunk
from regrag.generate import CITATION_RE, ClaudeGenerator, ExtractiveGenerator
from regrag.rerank import Scored, passage_types, question_type


def test_question_types():
    assert question_type("Within how many days must a firm notify the regulator?") == "DAYS"
    assert question_type("How long does a firm need to keep its client files?") == "YEARS"
    assert question_type("How large a fine can a firm face?") == "MONEY"
    assert question_type("Whose sign-off do vendor records need?") == "ROLE"
    assert question_type("Are small firms excused from the rules?") == "COUNT"
    assert passage_types("liable to a civil penalty of up to $50,000 within 10 business days") == {"MONEY", "DAYS"}


def _ctx(corpus):
    q = next(q for q in corpus["questions"] if q.in_scope and q.clause_type == "deadline")
    cid = q.gold_clause_ids[0]
    c = corpus["by_id"][cid]
    chunks = {cid: Chunk(cid, c.text, [cid], *corpus["offsets"][cid])}
    return q, cid, chunks


def test_extractive_answer_cites_and_refuses(corpus):
    q, cid, chunks = _ctx(corpus)
    gen = ExtractiveGenerator(tau=0.5)
    a = gen.answer(q.qid, q.question, [Scored(cid, 0.9, 0)], chunks, corpus["by_id"], corpus["offsets"], corpus["doc"])
    assert not a.refused and a.citations == [cid]
    assert CITATION_RE.findall(a.text) and q.answer in a.text
    r = gen.answer(q.qid, q.question, [Scored(cid, 0.1, 0)], chunks, corpus["by_id"], corpus["offsets"], corpus["doc"])
    assert r.refused and r.citations == []


class FakeMessages:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["model"] == "claude-opus-5-5" and "output_config" in kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def _fake_client(text):
    m = FakeMessages(text)
    return SimpleNamespace(messages=m, beta=SimpleNamespace(messages=m)), m


def test_claude_generator_with_fake_client(corpus, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    q, cid, chunks = _ctx(corpus)
    client, m = _fake_client(f"The deadline is {q.answer} [{cid}].")
    gen = ClaudeGenerator(tau=0.5, client=client)
    a = gen.answer(q.qid, q.question, [Scored(cid, 0.9, 0)], chunks, corpus["by_id"], corpus["offsets"], corpus["doc"])
    assert a.citations == [cid] and not a.refused and a.input_tokens == 100
    client2, m2 = _fake_client("INSUFFICIENT_CONTEXT")
    a2 = ClaudeGenerator(tau=0.5, client=client2).answer(
        q.qid, q.question, [Scored(cid, 0.9, 0)], chunks, corpus["by_id"], corpus["offsets"], corpus["doc"]
    )
    assert a2.refused
    a3 = gen.answer(q.qid, q.question, [Scored(cid, 0.1, 0)], chunks, corpus["by_id"], corpus["offsets"], corpus["doc"])
    assert a3.refused and m.calls == 1  # guardrail refuses before any API call


def test_model_is_configurable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "some-other-model")
    client, _ = _fake_client("x")
    assert ClaudeGenerator(tau=0.5, client=client).model == "some-other-model"
