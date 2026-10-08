"""Day 29: a profile reaches the real wire requests (answer, rewrite, rerank), not just a config."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from advent_core import config as config_module
from advent_core import journal as journal_module
from advent_core import openai_compat as oc
from advent_core.errors import AdventError
from advent_core.openai_compat import LocalResult
from advent_core.rag import RagSettings
from advent_core.telemetry import Usage
from tests.test_w06d27_rag import _fake_embed, _no_cloud, _write
from week_05 import rag as rag_module
from week_05 import rag_cli
from week_06 import profiles
from week_06 import ragbench as rb

LOCAL_URL = "http://127.0.0.1:1234"
QUESTION = rag_module.ControlQuestion(1, "что такое alpha?", (("alpha",),), ("a.md",))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(journal_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.delenv("ADVENT_LOCAL_URL", raising=False)


def _schema_name(payload: dict) -> str:
    return ((payload.get("response_format") or {}).get("json_schema") or {}).get("name") or ""


def fragments(payload: dict) -> int:
    """How many numbered fragments the rerank prompt of this request carries."""
    return len(re.findall(r"^\[\d+\] ", payload["messages"][0]["content"], re.M))


class Wire:
    """Replaces the HTTP chat call: records every payload and answers by the stage it sees."""

    def __init__(self, monkeypatch, tmp_path, chunks: int = 25):
        self.payloads: list[dict] = []
        self.settings: list[RagSettings] = []
        self.rerank_replies: list[str] = []  # consumed first, then the well-formed default
        db = tmp_path / "local.sqlite3"
        _write(
            db,
            endpoint="local",
            texts=tuple(f"chunk {i} aaa bbb ccc" for i in range(chunks)),
            doc_prefix="search_document: ",
            query_prefix="search_query: ",
        )
        monkeypatch.setattr(oc, "embed", _fake_embed())
        _no_cloud(monkeypatch)
        monkeypatch.setattr(oc, "chat_complete", self._chat)
        real_make = rag_cli.rag_module.make_retriever

        def spying_make(*args, **kwargs):
            inner = real_make(*args, **kwargs)

            def retrieve(question, settings):
                self.settings.append(settings)
                return inner(question, settings)

            return retrieve

        monkeypatch.setattr(rag_cli.rag_module, "make_retriever", spying_make)
        self.db = db

    def _chat(self, url, payload):
        self.payloads.append(payload)
        schema = ((payload.get("response_format") or {}).get("json_schema") or {}).get("name")
        if schema in ("rerank_positional", "rerank"):
            n = fragments(payload)
            if self.rerank_replies:
                text = self.rerank_replies.pop(0)
            elif schema == "rerank_positional":
                text = json.dumps({"scores": [9] * n})
            else:
                text = json.dumps({"scores": [{"id": i, "score": 9} for i in range(1, n + 1)]})
        elif schema == "cite_answer":
            text = json.dumps({"status": "unknown", "answer": "", "sources": [], "quotes": []})
        else:
            text = QUESTION.question  # the same query: the fused list is exactly n long
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        return LocalResult(
            text=text,
            reasoning_text="",
            usage=usage,
            latency_ms=5.0,
            ttft_ms=1.0,
            content_ms=2.0,
            finish_reason="stop",
            model=payload.get("model", "ornith"),
        )

    def by_stage(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for p in self.payloads:
            name = ((p.get("response_format") or {}).get("json_schema") or {}).get("name")
            stage = {
                "rerank": "rerank",
                "rerank_positional": "rerank",
                "cite_answer": "answer",
            }.get(name, "rewrite")
            out[stage] = p
        return out

    def run(self, name: str | None) -> None:
        profile = profiles.get_profile(name) if name else None
        backend = rb.Backend("local", rb.local_config(LOCAL_URL), self.db, profile=profile)
        rb.default_ask(backend, QUESTION, 1, lambda entry: None)


@pytest.fixture
def wire(monkeypatch, tmp_path):
    return Wire(monkeypatch, tmp_path)


def test_baseline_sends_no_sampling_and_keeps_reasoning_on(wire):
    wire.run("baseline")
    answer = wire.by_stage()["answer"]
    for key in ("temperature", "top_p", "top_k", "reasoning_effort"):
        assert key not in answer
    assert answer["max_tokens"] == 4096


def test_sampling_profile_reaches_the_answer_request_but_not_the_aux_calls(wire):
    wire.run("sampling")
    stages = wire.by_stage()
    answer = stages["answer"]
    assert (answer["temperature"], answer["top_p"], answer["top_k"]) == (0.6, 0.95, 20)
    for aux in (stages["rewrite"], stages["rerank"]):
        assert aux["temperature"] == 0
        assert "top_p" not in aux and "top_k" not in aux


def test_cap_profile_sets_the_answer_cap(wire):
    wire.run("cap")
    assert wire.by_stage()["answer"]["max_tokens"] == 1536


def test_k12_profile_sends_twelve_fragments_to_the_reranker_not_twenty(wire):
    wire.run("k12")
    assert [s.k_before for s in wire.settings] == [12]
    assert fragments(wire.by_stage()["rerank"]) == 12


def test_baseline_sends_twenty_fragments_to_the_reranker(wire):
    wire.run("baseline")
    assert [s.k_before for s in wire.settings] == [20]
    assert fragments(wire.by_stage()["rerank"]) == 20


def test_positional_grammar_length_equals_the_fragment_count_it_asks_about(wire):
    wire.run("positional")
    rerank = wire.by_stage()["rerank"]
    scores = rerank["response_format"]["json_schema"]["schema"]["properties"]["scores"]
    assert fragments(rerank) == 20
    assert scores["minItems"] == scores["maxItems"] == 20


def test_a_bad_positional_reply_is_retried_once_then_accepted(wire):
    wire.rerank_replies = ['{"scores": [7, {"id": 2, "score": 9}, 10]}']
    wire.run("positional")
    reranks = [p for p in wire.payloads if _schema_name(p) == "rerank_positional"]
    assert len(reranks) == 2


def test_two_bad_positional_replies_fail_the_cell_instead_of_guessing(wire):
    wire.rerank_replies = ['{"scores": [1, 2]}', '{"scores": [true, 2, 3]}']
    with pytest.raises(AdventError):
        wire.run("positional")
    reranks = [p for p in wire.payloads if _schema_name(p).startswith("rerank")]
    assert len(reranks) == 2


def test_the_object_format_does_not_retry_on_partial_scores(wire):
    wire.rerank_replies = ['{"scores": [{"id": 2, "score": 9}]}']
    wire.run("baseline")
    reranks = [p for p in wire.payloads if _schema_name(p) == "rerank"]
    assert len(reranks) == 1


def test_noreason_sends_reasoning_effort_none_on_the_answer_only_added_by_the_profile(wire):
    wire.run("noreason")
    stages = wire.by_stage()
    assert stages["answer"]["reasoning_effort"] == "none"
    # aux calls were already reasoning-off in Day 28 and stay so
    assert stages["rewrite"]["reasoning_effort"] == "none"
    assert stages["rerank"]["reasoning_effort"] == "none"


def test_with_reasoning_the_answer_carries_no_reasoning_effort(wire):
    wire.run("cap")
    assert "reasoning_effort" not in wire.by_stage()["answer"]


def test_positional_profile_sends_the_positional_grammar_with_exact_length(wire):
    wire.run("positional")
    rerank = wire.by_stage()["rerank"]
    schema = rerank["response_format"]["json_schema"]
    assert schema["name"] == "rerank_positional"
    scores = schema["schema"]["properties"]["scores"]
    assert scores["minItems"] == scores["maxItems"] == fragments(rerank)
    prompt = rerank["messages"][0]["content"]
    assert f"ровно {fragments(rerank)} целых чисел по порядку фрагментов" in prompt
    assert "[7, 0, 10, ...]" in prompt


def test_baseline_rerank_stays_in_the_object_format(wire):
    wire.run("baseline")
    rerank = wire.by_stage()["rerank"]
    assert rerank["response_format"]["json_schema"]["name"] == "rerank"
    assert '"id": 1, "score": 0' in rerank["messages"][0]["content"]


def test_citelocal_changes_only_the_instruction_text_not_the_grammar(wire):
    wire.run("baseline")
    default = wire.by_stage()["answer"]
    wire.payloads.clear()
    wire.run("citelocal")
    local = wire.by_stage()["answer"]
    assert default["response_format"] == local["response_format"]
    assert "единственный фрагмент" in local["messages"][-1]["content"]
    assert "единственный фрагмент" not in default["messages"][-1]["content"]
    assert "Цитата — точная копия" in default["messages"][-1]["content"]
    assert "Цитата — точная копия" not in local["messages"][-1]["content"]


def test_a_run_without_a_profile_is_the_day_28_request(wire):
    wire.run(None)
    answer = wire.by_stage()["answer"]
    assert answer["max_tokens"] == 4096
    for key in ("temperature", "top_p", "top_k", "reasoning_effort"):
        assert key not in answer


def test_effective_settings_come_from_the_config_the_request_uses(wire):
    backend = rb.Backend(
        "local",
        rb.local_config(LOCAL_URL),
        Path("x.sqlite3"),
        profile=profiles.PROFILES["sampling"],
    )
    settings = rb.effective_settings(backend)
    assert settings["temperature"] == 0.6 and settings["top_p"] == 0.95
    assert settings["top_k"] == 20 and settings["profile"] == "sampling"
    assert settings["answer_reasoning"] is True and settings["rerank_format"] == "objects"
    assert settings["profile_fields"]["answer_top_k"] == 20
    plain = rb.effective_settings(rb.Backend("local", rb.local_config(LOCAL_URL), Path("x")))
    assert plain["temperature"] is None and "profile" not in plain
    assert plain["k_before"] == 20 and plain["answer_max_tokens"] == 4096


def test_journal_day_follows_the_profile(wire, tmp_path):
    wire.run("cap")
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows and {r["day"] for r in rows} == {29}


def test_journal_day_without_a_profile_stays_28(wire, tmp_path):
    wire.run(None)
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows and {r["day"] for r in rows} == {28}


def test_the_unprofiled_answer_request_has_exactly_the_day_28_shape(wire):
    wire.run(None)
    answer = wire.by_stage()["answer"]
    assert set(answer) == {"max_tokens", "messages", "model", "response_format"}
    assert answer["model"] == "ornith" and answer["max_tokens"] == 4096
    assert [m["role"] for m in answer["messages"]] == ["system", "user"]
    assert answer["response_format"]["json_schema"]["name"] == "cite_answer"
    rerank = wire.by_stage()["rerank"]
    assert set(rerank) == {
        "max_tokens",
        "messages",
        "model",
        "reasoning_effort",
        "response_format",
        "temperature",
    }
    assert rerank["response_format"]["json_schema"]["name"] == "rerank"
