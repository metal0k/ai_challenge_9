"""Day 25: `adventrag chat-eval` on fakes — no network, no API credits.

The agent is a real `Agent` (memory strategy + cite) with only the model calls faked:
the main complete() and the memory extractor. Expected values are literals.
"""

from __future__ import annotations

import io
import json
import re
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from rich.console import Console
from typer.testing import CliRunner

from advent_core import config as config_module
from advent_core import console
from advent_core import scenario as scenario_core
from advent_core.agent import Agent, AgentReply
from advent_core.config import Config, ConfigError
from advent_core.errors import AdventError, NetworkError
from advent_core.memory import MemoryDelta, MemorySnapshot, MemoryUpdate, StructuredMemory
from advent_core.params import GenerationParams
from advent_core.rag import RagContext, RagHit
from advent_core.telemetry import CallResult, Usage
from week_02 import cli as agent_cli
from week_05 import chat_eval, rag_cli
from week_05 import cli as week05_cli
from week_05 import rag as rag_module

_ANSI = re.compile(r"\[[0-9;]*m")

HIT_TEXT = "verify_capture проверяет кадр до записи. Максимум temperature равен 1.5."
QUOTE_CAPTURE = "verify_capture проверяет кадр до записи"
QUOTE_TEMPERATURE = "Максимум temperature равен 1.5"
CLOSING = "Метрики цели и факты считает код, без модели-судьи."

M1 = chat_eval.ChatMessage(
    "Хочу записать демо через advent record. С чего начать?",
    "goal",
    facts=(("verify_capture",),),
    expect_state={"goal": ("демо",)},
)
M2 = chat_eval.ChatMessage(
    "Я на Windows Terminal. Когда ловится чёрный кадр?",
    "clarify",
    facts=(("до записи",),),
    expect_state={"clarified": ("windows terminal",)},
)
M3 = chat_eval.ChatMessage("Кстати, какая температура максимальная?", "switch", facts=(("1.5",),))
SCENARIO = chat_eval.Scenario("obs", "Запись демо", (M1, M2, M3), placeholder=True)


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _cite(answer: str, quote: str) -> str:
    return json.dumps(
        {
            "status": "answer",
            "answer": answer,
            "sources": [1],
            "quotes": [{"id": 1, "text": quote}],
        },
        ensure_ascii=False,
    )


def _delta(*sets: dict) -> CallResult:
    raw = {"working": {"set": list(sets), "delete": []}, "long_term": {"set": [], "delete": []}}
    return CallResult(
        text=json.dumps(raw, ensure_ascii=False),
        model_requested="m",
        usage=Usage(prompt_tokens=50, completion_tokens=10),
    )


def _op(field: str, key: str, value: str, evidence: str) -> dict:
    return {"field": field, "key": key, "value": value, "evidence": evidence}


GOAL_DELTA = _delta(_op("goal", "main", "записать демо", "Хочу записать демо"))
ENV_DELTA = _delta(
    _op("clarified", "env", "Windows Terminal", "Я на Windows Terminal"),
    _op("goal", "other", "ловить чёрный кадр", "Когда ловится чёрный кадр"),
)


def _main(text: str, prompt: int | None = 100, completion: int | None = 10) -> CallResult:
    return CallResult(
        text=text,
        model_requested="m",
        finish_reason="stop",
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion),
        stream=False,
    )


class Harness:
    """Fakes for one run: main model, extractor, retriever, journal."""

    def __init__(self, monkeypatch) -> None:
        self.main: list[CallResult | Exception] = []
        self.deltas: list[CallResult] = []
        self.journal: list[dict] = []
        self.aux_calls: list[dict] = []
        self.segments: list[list] = []
        self.pauses: list[float] = []
        self.retrieve_empty = False
        self.main_calls = 0
        monkeypatch.setattr(chat_eval, "log_call", self._log)
        monkeypatch.setattr(rag_module, "_aux_call", self._aux)
        monkeypatch.setattr(rag_cli, "_pause", self.pauses.append)

    def _log(self, result, messages, *, week, day, error=None, extra=None, path=None) -> None:
        self.journal.append({"week": week, "day": day, "error": error, "extra": dict(extra or {})})

    def _aux(self, prompt, *, command, max_tokens, json_mode, day):
        self.aux_calls.append({"prompt": prompt, "command": command, "day": day, "json": json_mode})
        return CallResult(
            text="", model_requested="m", usage=Usage(prompt_tokens=20, completion_tokens=5)
        )

    def complete(self, config, messages, capabilities=None):
        self.main_calls += 1
        item = self.main.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def retrieve(self, question, settings) -> RagContext:
        hit = RagHit("c1", "CLAUDE.md", "OBS", 0.9, HIT_TEXT)
        if self.retrieve_empty:
            return RagContext((), "structure", 5, "mistral-embed", 7, "abc", passed=0, candidates=3)
        return RagContext(
            (hit,),
            "structure",
            5,
            "mistral-embed",
            7,
            "abc",
            candidates=3,
            passed=1,
            aux_prompt_tokens=30,
            aux_completion_tokens=4,
        )

    def agent(self) -> Agent:
        params = GenerationParams(
            rag=True,
            rag_cite=True,
            rag_rewrite=True,
            rag_rerank=True,
            context_strategy="memory",
        )
        agent = Agent(
            Config(api_key="ключ", model="ministral-14b-latest", params=params),
            complete=self.complete,
            stream=self.complete,
            memory_prompt="извлеки",
            retrieve=self.retrieve,
            on_warning=lambda text: None,
        )

        def memory_call(snapshot, segment):
            self.segments.append(list(segment))
            return self.deltas.pop(0) if self.deltas else _delta()

        agent._memory_call = memory_call  # type: ignore[method-assign]
        return agent


@pytest.fixture
def h(monkeypatch):
    return Harness(monkeypatch)


@pytest.fixture
def out(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=80, no_color=True))
    return buffer


@pytest.fixture
def err(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "err", Console(file=buffer, width=80, no_color=True))
    return buffer


def _happy(h: Harness) -> None:
    h.main = [
        _main(_cite("Сначала verify_capture проверяет кадр.", QUOTE_CAPTURE)),
        _main(_cite("Кадр проверяют до записи.", QUOTE_CAPTURE)),
        _main(_cite("Максимум — 2.0.", QUOTE_TEMPERATURE)),
    ]
    h.deltas = [GOAL_DELTA, ENV_DELTA, _delta()]


def _run(h: Harness, scenario=SCENARIO, detail: bool = True) -> chat_eval.ScenarioResult:
    return chat_eval.run_scenario_chat(scenario, h.agent(), detail=detail)


# --- metrics ------------------------------------------------------------------------------


def test_three_message_run_scores_sources_facts_state_and_goal(h, out, err):
    _happy(h)

    result = _run(h)
    m = chat_eval.scenario_metrics(result)

    assert (m.planned, m.done, m.errors) == (3, 3, 0)
    assert m.with_source == 3
    assert m.refusals == {}
    # The third answer says 2.0: "1.5" sits only in its quote, which does not count.
    assert (m.facts_found, m.facts_total) == (2, 3)
    assert (m.state_ok, m.state_total) == (2, 2)
    assert (m.goal.pinned_from, m.goal.stable) == (1, True)
    assert m.goal_blocked == 1  # ENV_DELTA tries goal.other after the pin
    assert m.after_switch == ()  # the switch message is the last one: no next turn


def test_goal_stays_when_the_extractor_tries_to_replace_it(h, out, err):
    _happy(h)
    h.deltas[2] = _delta(_op("goal", "main", "другая цель", "Кстати, какая температура"))

    result = _run(h)

    assert result.memory.working.entries["goal.main"] == "записать демо"
    assert result.memory.working.pinned == frozenset({"goal.main"})
    m = chat_eval.scenario_metrics(result)
    assert m.goal.stable is True
    assert m.goal_blocked == 2  # turn 2 (goal.other) and turn 3 (goal.main), both pinned out
    assert "!goal.other" in _plain(out.getvalue())
    assert "!goal.main" in _plain(out.getvalue())


def _row(n: int, goal: tuple, pinned: bool) -> chat_eval.TurnRow:
    return chat_eval.TurnRow(n=n, message=M3, goal=goal, goal_pinned=pinned)


def test_goal_stability_is_false_when_the_text_changes_after_the_pin():
    old, new = (("goal.main", "a"),), (("goal.main", "b"),)

    unstable = chat_eval.goal_stability(
        [_row(1, old, True), _row(2, old, True), _row(3, new, True)]
    )
    unpinned = chat_eval.goal_stability([_row(1, old, True), _row(2, old, False)])
    never = chat_eval.goal_stability([_row(1, (), False), _row(2, old, False)])
    late = chat_eval.goal_stability([_row(1, (), False), _row(2, old, True), _row(3, old, True)])

    assert (unstable.pinned_from, unstable.stable) == (1, False)
    assert (unpinned.pinned_from, unpinned.stable) == (1, False)
    assert (never.pinned_from, never.stable) == (None, False)
    assert (late.pinned_from, late.stable) == (2, True)


def _after_row(n: int, kind: str, *, answered=True, quotes=(1, 1), facts=(True,), error=None):
    message = chat_eval.ChatMessage("q", kind)
    return chat_eval.TurnRow(
        n=n, message=message, error=error, answered=answered, quotes=quotes, facts=facts
    )


def test_after_switch_needs_a_sourced_answer_with_a_fact_hit_on_the_next_turn():
    ok = [_after_row(1, "switch"), _after_row(2, "question")]
    no_fact = [_after_row(1, "switch"), _after_row(2, "question", facts=(False,))]
    no_facts_defined = [_after_row(1, "switch"), _after_row(2, "question", facts=())]
    refused = [_after_row(1, "switch"), _after_row(2, "question", answered=False, quotes=(0, 0))]
    unverified = [_after_row(1, "switch"), _after_row(2, "question", quotes=(0, 1))]
    failed = [_after_row(1, "switch"), _after_row(2, "question", error="x")]
    two = [*ok, _after_row(3, "switch"), _after_row(4, "question", facts=(False,))]
    last = [_after_row(1, "question"), _after_row(2, "switch")]
    check = chat_eval.after_switch

    assert check(ok) == (True,)
    assert check(no_fact) == (False,)
    assert check(no_facts_defined) == (False,)
    assert check(refused) == (False,)
    assert check(unverified) == (False,)
    assert check(failed) == (False,)
    assert check(two) == (True, False)
    assert check(last) == ()  # a trailing switch has no next turn to judge


def test_switch_in_the_middle_is_scored_and_shown_in_the_summary(h, out, err):
    _happy(h)
    scenario = chat_eval.Scenario("obs", "t", (M1, M3, M2))
    h.main[1], h.main[2] = (
        _main(_cite("Максимум — 1.5.", QUOTE_TEMPERATURE)),
        _main(_cite("Кадр проверяют до записи.", QUOTE_CAPTURE)),
    )

    result = _run(h, scenario)
    out.truncate(0)
    out.seek(0)
    chat_eval.print_summary([result])

    assert chat_eval.scenario_metrics(result).after_switch == (True,)
    flat = re.sub(r"\s+", " ", _plain(out.getvalue()))
    assert "│ obs │ 3/3 │ 3/3 │ 0 │ 3/3 │ ✓ │ ✓ ход 1 │" in flat


def test_wrong_expect_state_value_fails_the_metric(h, out, err):
    _happy(h)
    wrong = chat_eval.ChatMessage(
        M2.text, "clarify", facts=M2.facts, expect_state={"clarified": ("linux",)}
    )
    scenario = chat_eval.Scenario("obs", "t", (M1, wrong, M3))

    result = _run(h, scenario)
    m = chat_eval.scenario_metrics(result)

    # The field is filled (Windows Terminal), but not with what the message expects.
    assert result.memory.working.entries["clarified.env"] == "Windows Terminal"
    assert (m.state_ok, m.state_total) == (1, 2)
    assert "state ✗" in _plain(out.getvalue())


def test_contradictory_value_under_a_matching_key_does_not_meet_expect_state():
    working = StructuredMemory({"clarified.windows": "Linux"})

    assert not chat_eval.expect_state_met({"clarified": ("windows",)}, working)
    assert chat_eval.expect_state_met({"clarified": ("linux",)}, working)


def test_expect_state_matches_values_case_insensitively_and_by_field():
    working = StructuredMemory({"terms.дубль": "Одна запись advent record", "goal.main": "демо"})
    met = chat_eval.expect_state_met

    assert met({"terms": ("ОДНА ЗАПИСЬ",)}, working)
    assert not met({"terms": ("дубль",)}, working)  # the key name is not evidence
    assert met({"terms": ("advent record",)}, working)
    assert not met({"terms": ("демо",)}, working)  # "демо" lives in goal, not terms
    assert not met({"constraints": ("коротк",)}, working)  # the field is empty
    assert met({"goal": ("демо",), "terms": ("одна запись",)}, working)
    assert not met({"goal": ("демо",), "terms": ("нет такого",)}, working)


def test_fact_present_only_in_a_quote_is_not_counted(h, out, err):
    h.main = [_main(_cite("Максимум — 2.0.", QUOTE_TEMPERATURE))]
    scenario = chat_eval.Scenario("obs", "t", (M3,))

    result = _run(h, scenario)

    row = result.rows[0]
    assert row.answered is True
    assert row.quotes == (1, 1)
    assert row.facts == (False,)
    assert chat_eval.scenario_metrics(result).facts_found == 0


def test_all_refusals_run_makes_no_model_call_and_counts_refusals(h, out, err):
    h.retrieve_empty = True
    h.deltas = [GOAL_DELTA, ENV_DELTA, _delta()]

    result = _run(h)
    m = chat_eval.scenario_metrics(result)

    assert h.main_calls == 0
    assert h.aux_calls == []
    assert (m.done, m.with_source) == (3, 0)
    assert m.refusals == {"порог": 3}
    assert (m.facts_found, m.facts_total) == (0, 3)
    # The extractor still ran on refused turns: the goal reached working memory.
    assert (m.goal.pinned_from, m.goal.stable) == (1, True)
    # Nothing was sent to the answer model: zero is a known cost, not an unknown.
    assert [r.tokens.answer for r in result.rows] == [(0, 0)] * 3
    text = _plain(out.getvalue())
    assert "Не знаю: в документации проекта не нашлось фрагментов" in text


# --- journal and tokens ---------------------------------------------------------------------


def test_chat_eval_journals_its_own_rows_as_week_5_day_25(h, out, err):
    _happy(h)

    _run(h)

    kinds = [(r["week"], r["day"], r["extra"].get("kind"), r["extra"]["turn"]) for r in h.journal]
    assert kinds == [
        (5, 25, "memory", 1),
        (5, 25, None, 1),
        (5, 25, "memory", 2),
        (5, 25, None, 2),
        (5, 25, "memory", 3),
        (5, 25, None, 3),
    ]
    assert [r["extra"]["command"] for r in h.journal] == ["chat_eval"] * 6
    assert [r["extra"].get("mode") for r in h.journal][1] == "cite"
    assert h.aux_calls == []


def test_every_paid_call_reaches_the_real_journal_as_week_5_day_25(monkeypatch, tmp_path, out, err):
    """Real retriever, _aux_call, embeddings, Agent and journal; only the network is faked."""
    from advent_core import config as config_module
    from advent_core import journal as journal_module

    monkeypatch.setattr(journal_module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setattr(rag_module, "mistral_client", lambda config: _FakeEmbedClient())
    monkeypatch.setattr(
        rag_module.index_module,
        "load_runs",
        lambda db_path: {
            "structure": SimpleNamespace(model="mistral-embed", corpus_rev="abc", n_chunks=1)
        },
    )
    chunk = SimpleNamespace(chunk_id="c1", source="CLAUDE.md", section="OBS", text=HIT_TEXT)
    monkeypatch.setattr(
        rag_module.index_module,
        "search",
        lambda db_path, strategy, vec, k: [SimpleNamespace(chunk=chunk, score=0.9)],
    )

    def network(config, messages, capabilities=None, **kwargs):
        first, last = messages[0]["content"], messages[-1]["content"]
        if first == "извлеки":
            text = GOAL_DELTA.text
        elif last.startswith("Перепиши"):
            text = "verify_capture проверка кадра"
        elif last.startswith("Оцени, насколько"):
            text = json.dumps({"scores": [{"id": 1, "score": 9}]})
        else:
            text = _cite("Сначала verify_capture проверяет кадр.", QUOTE_CAPTURE)
        return CallResult(
            text=text,
            model_requested="m",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=2),
            stream=False,
        )

    monkeypatch.setattr(rag_module.chat_core, "complete", network)
    params = GenerationParams(
        rag=True, rag_cite=True, rag_rewrite=True, rag_rerank=True, context_strategy="memory"
    )
    agent = Agent(
        Config(api_key="ключ", model="ministral-14b-latest", params=params),
        complete=network,
        stream=network,
        memory_prompt="извлеки",
        retrieve=rag_module.make_retriever(
            None, day=chat_eval.CHAT_DAY, aux_day=chat_eval.CHAT_DAY
        ),
        on_warning=lambda text: None,
    )

    chat_eval.run_scenario_chat(chat_eval.Scenario("obs", "t", (M1,)), agent, detail=False)

    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text("utf-8").splitlines()]
    assert [(r["week"], r["day"], r.get("kind") or r["command"]) for r in rows] == [
        (5, 25, "rag_rewrite"),
        (5, 25, "embed"),
        (5, 25, "rag_rerank"),
        (5, 25, "memory"),
        (5, 25, "chat_eval"),
    ]


class _FakeEmbedClient:
    """The network boundary of embed_texts: `client.embeddings.create`."""

    def __init__(self) -> None:
        self.embeddings = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def create(self, *, model, inputs):
        data = [
            SimpleNamespace(index=i, embedding=[1.0, 0.0, 0.0, 0.0]) for i in range(len(inputs))
        ]
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=0, total_tokens=7)
        return SimpleNamespace(data=data, model=model, usage=usage)


def test_paid_retrieval_survives_a_failed_answer_and_a_retry_counts_both(h, out, err):
    h.main = [
        NetworkError("обрыв"),
        _main(_cite("Сначала verify_capture проверяет кадр.", QUOTE_CAPTURE)),
    ]
    h.deltas = [GOAL_DELTA, _delta()]

    result = _run(h, chat_eval.Scenario("obs", "t", (M1,)))

    # Two attempts, each: rewrite+rerank 30/4 and embed 7.
    assert result.rows[0].tokens.aux == (60, 8)
    assert result.rows[0].tokens.embed == 14


def test_paid_retrieval_of_a_turn_whose_answer_failed_is_still_counted(h, out, err):
    h.main = [AdventError("сервер отказал")]
    h.deltas = [GOAL_DELTA]

    result = _run(h, chat_eval.Scenario("obs", "t", (M1,)))

    row = result.rows[0]
    assert row.error == "сервер отказал"
    assert row.tokens.aux == (30, 4)
    assert row.tokens.embed == 7
    assert chat_eval.total_tokens([result])["embed"] == 7


def test_agent_retriever_is_restored_after_the_turn(h, out, err):
    _happy(h)
    agent = h.agent()
    before = agent.retrieve

    chat_eval.run_scenario_chat(SCENARIO, agent, detail=False)

    assert agent.retrieve is before


def test_recovery_extraction_gets_the_messages_a_real_budget_trim_omitted(h, out, err):
    """Extractor failed on turn 2, trim dropped turn 1 from the working copy; turn 3 re-reads it."""
    from advent_core.tokens import EstimateCounter

    pad = " слово" * 40
    questions = [f"Вопрос {n}: что делает verify_capture?{pad}" for n in (1, 2, 3)]
    messages = tuple(chat_eval.ChatMessage(q, "question") for q in questions)
    h.main = [
        _main(_cite(f"Ответ {n}: проверяет кадр.", QUOTE_CAPTURE), prompt=None, completion=None)
        for n in (1, 2, 3)
    ]
    calls = {"n": 0}
    agent = Agent(
        Config(
            api_key="ключ",
            model="ministral-14b-latest",
            params=GenerationParams(
                rag=True,
                rag_cite=True,
                rag_rewrite=True,
                rag_rerank=True,
                context_strategy="memory",
                max_tokens=100,
            ),
        ),
        complete=h.complete,
        stream=h.complete,
        counter=EstimateCounter(),
        context_limit=520,
        memory_prompt="извлеки",
        retrieve=h.retrieve,
        on_warning=lambda text: None,
    )

    def memory_call(snapshot, segment):
        calls["n"] += 1
        h.segments.append(list(segment))
        if calls["n"] == 2:
            return CallResult(model_requested="m", finish_reason="error")
        return _delta()

    agent._memory_call = memory_call  # type: ignore[method-assign]
    state = scenario_core.ScenarioState(memory=MemorySnapshot())
    rows = []
    for n, message in enumerate(messages, start=1):
        if n == 3:
            # Trim really fired on turn 2: the working copy is shorter than the transcript.
            assert len(state.history) < len(state.transcript) == 4
        row, _reply = chat_eval.run_chat_turn(agent, state, "s", n, message)
        rows.append(row)

    assert all(r.done for r in rows)
    third = h.segments[2]
    assert [m["role"] for m in third] == ["user", "assistant", "user"]
    assert third[0]["content"] == questions[1]
    assert third[2]["content"] == questions[2]


def test_token_totals_are_split_by_kind_and_unknown_usage_prints_a_dash(h, out, err):
    _happy(h)
    h.main[1] = _main(
        _cite("Кадр проверяют до записи.", QUOTE_CAPTURE), prompt=None, completion=None
    )

    result = chat_eval.run_scenario_chat(SCENARIO, h.agent(), detail=False)
    totals = chat_eval.total_tokens([result])

    assert totals["answer"] == (None, None)
    assert totals["aux"] == (90, 12)
    assert totals["extractor"] == (150, 30)
    assert totals["embed"] == 21
    out.truncate(0)
    chat_eval.print_summary([result])
    flat = re.sub(r"\s+", " ", _plain(out.getvalue()))
    assert "ответ —/— · rewrite+rerank 90/12" in flat
    assert "extractor 150/30 · embed 21" in flat


# --- errors, retry, state carried between turns ---------------------------------------------


def test_failed_turn_keeps_its_paid_extractor_and_the_scenario_goes_on(h, out, err):
    _happy(h)
    h.main[1] = AdventError("сервер отказал")
    h.main.append(_main(_cite("Максимум — 1.5.", QUOTE_TEMPERATURE)))

    result = _run(h)
    m = chat_eval.scenario_metrics(result)

    assert (m.done, m.errors) == (2, 1)
    row = result.rows[1]
    assert row.error == "сервер отказал"
    assert row.expect_ok is False
    # The extractor ran before the failing call: its cost and its update are salvaged.
    assert row.tokens.extractor == (50, 10)
    assert result.memory.working.entries["clarified.env"] == "Windows Terminal"
    assert [r["extra"].get("kind") for r in h.journal if r["extra"]["turn"] == 2] == [
        "memory",
        None,
    ]
    assert [r["error"] for r in h.journal if r["extra"]["turn"] == 2] == [None, "сервер отказал"]
    assert h.pauses == []  # not transient: no retry
    text = _plain(out.getvalue())
    assert "ошибка: сервер отказал" in text


def test_transient_error_is_retried_once_and_pays_for_both_extractor_calls(h, out, err):
    _happy(h)
    h.main.insert(0, NetworkError("обрыв"))
    h.deltas.insert(1, _delta())  # the retry's extraction

    result = _run(h)

    assert h.pauses == [rag_cli.RETRY_PAUSE_S]
    assert chat_eval.scenario_metrics(result).errors == 0
    assert result.rows[0].tokens.extractor == (100, 20)
    assert "обрыв — повтор" in _plain(err.getvalue())


def _fake_reply(question: str, history: list) -> AgentReply:
    reply_history = [*history[-0:], {"role": "user", "content": question}]
    reply_history = reply_history[-1:] + [{"role": "assistant", "content": "ответ " + question}]
    return AgentReply(
        text="ответ " + question,
        history=reply_history,
        result=CallResult(),
        memory=MemorySnapshot(),
        memory_upto=0,
        memory_update=MemoryUpdate(MemorySnapshot(), MemoryDelta()),
    )


def test_trimmed_history_and_full_transcript_travel_separately():
    calls: list[dict] = []

    class FakeAgent:
        def ask(self, question, history, **kwargs):
            snapshot = {k: list(v) if k == "memory_history" else v for k, v in kwargs.items()}
            calls.append({"history": list(history), **snapshot})
            return _fake_reply(question, history)

    state = scenario_core.ScenarioState(memory=MemorySnapshot())
    for question in ("q1", "q2", "q3"):
        scenario_core.run_turn(FakeAgent(), state, question)

    # The agent's own history was trimmed to the last pair every time...
    assert [len(c["history"]) for c in calls] == [0, 2, 2]
    # ...while the extractor is fed the full transcript, and the cursor covers it.
    assert [len(c["memory_history"]) for c in calls] == [0, 2, 4]
    assert [c["memory_upto"] for c in calls] == [0, 2, 4]
    assert state.transcript == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "ответ q1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "ответ q2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "ответ q3"},
    ]
    assert state.memory_upto == 6
    assert calls[1]["memory"].short_term.messages == tuple(calls[1]["history"])


def test_untracked_state_sends_no_memory_arguments():
    seen: list[dict] = []

    class FakeAgent:
        def ask(self, question, history, **kwargs):
            seen.append(kwargs)
            return _fake_reply(question, history)

    state = scenario_core.ScenarioState()
    scenario_core.run_turn(FakeAgent(), state, "q")

    assert set(seen[0]) == {"summary", "facts", "facts_pinned", "facts_upto"}
    assert state.transcript == [] and state.memory is None


def test_real_agent_extractor_sees_the_whole_transcript_after_each_turn(h, out, err):
    _happy(h)

    _run(h)

    # Turn 2 extracts from the user message only (turn 1 is covered); the history
    # kept by the REPL-like carry is the full transcript, not the trimmed working copy.
    assert [len(s) for s in h.segments] == [1, 1, 1]
    assert h.segments[1] == [{"role": "user", "content": M2.text}]


# --- output ----------------------------------------------------------------------------------


def test_detail_prints_question_answer_sources_delta_task_and_state(h, out, err):
    _happy(h)

    _run(h)

    text = _plain(out.getvalue())
    assert text.count("── obs · ход 1/3 · goal ──") == 1
    assert f"Вопрос: {M1.text}" in text
    assert "ответ: «Сначала verify_capture проверяет кадр.»" in text
    assert "[1] ✓ CLAUDE.md «verify_capture проверяет кадр до записи»" in text
    assert "memory: working +goal.main" in text
    assert "memory: working +clarified.env, !goal.other" in text
    assert ("задача: цель 📌 «записать демо» · уточнено 0 · ограничений 0 · терминов 0") in text
    assert "факты ✓ · state ✓" in text
    assert max(cell_len(line) for line in text.splitlines()) <= 80


def test_no_detail_prints_one_line_per_turn(h, out, err):
    _happy(h)

    _run(h, detail=False)

    lines = [line for line in _plain(out.getvalue()).splitlines() if line.strip()]
    assert len(lines) == 3
    assert lines[0].startswith("1/3 Хочу записать демо")
    assert lines[0].endswith(" · источник ✓ · факты 1/1")
    assert lines[2].endswith(" · источник ✓ · факты 0/1")
    assert all(cell_len(line) <= 80 for line in lines)


def _second_scenario() -> chat_eval.Scenario:
    return chat_eval.Scenario("budget", "Бюджет", (M1, M2, M3))


def test_summary_table_at_width_80_has_one_header_and_no_ellipsis(h, out, err):
    _happy(h)
    first = _run(h)
    out.truncate(0)
    out.seek(0)
    _happy(h)
    second = _run(h, _second_scenario())
    out.truncate(0)
    out.seek(0)

    chat_eval.print_summary([first, second])

    text = _plain(out.getvalue())
    flat = re.sub(r"\s+", " ", text)
    assert text.count("│ сценарий") == 1
    assert text.count("после побочного") == 1
    assert text.count("Мини-чат: сценарии") == 1
    assert "│ obs │ 3/3 │ 3/3 │ 0 │ 2/3 │ — │ ✓ ход 1 │" in flat
    assert "│ budget │ 3/3 │ 3/3 │ 0 │ 2/3 │ — │ ✓ ход 1 │" in flat
    assert "…" not in text
    assert max(cell_len(line) for line in text.splitlines()) <= 80


def test_summary_is_last_and_lists_final_state_tokens_and_the_caveat(h, out, err):
    _happy(h)
    result = _run(h)
    out.truncate(0)
    out.seek(0)

    chat_eval.print_summary([result])

    text = _plain(out.getvalue())
    flat = re.sub(r"\s+", " ", text)
    order = [
        text.index("Мини-чат: сценарии"),
        text.index("obs: цель закреплена с хода 1, не менялась"),
        text.index("Итоговый task state — obs:"),
        text.index("Токены prompt/completion по вызовам:"),
        text.index("Сценарии с placeholder"),
        text.index("Один прогон на сценарий"),
    ]
    assert order == sorted(order)
    assert "*goal.main = записать демо" in text
    assert "clarified.env = Windows Terminal" in text
    assert "state по expect_state 2/2 · попыток переписать цель: 1" in flat
    assert flat.strip().endswith(CLOSING)


def test_all_refusal_summary_shows_the_reason_and_still_ends_with_the_caveat(h, out, err):
    h.retrieve_empty = True
    h.deltas = [GOAL_DELTA, ENV_DELTA, _delta()]
    result = _run(h)
    out.truncate(0)
    out.seek(0)

    chat_eval.print_summary([result])

    flat = re.sub(r"\s+", " ", _plain(out.getvalue()))
    assert "│ obs │ 3/3 │ 0/3 │ 3 │ 0/3 │ — │ ✓ ход 1 │" in flat
    assert "не знаю: порог 3" in flat
    assert flat.strip().endswith(CLOSING)


# --- scenarios file, command, wiring ---------------------------------------------------------


def test_shipped_scenarios_load_and_cover_the_required_message_kinds():
    scenarios = chat_eval.load_scenarios()

    assert [s.id for s in scenarios] == ["obs", "budget"]
    for scenario in scenarios:
        assert len(scenario.messages) == 12
        assert scenario.placeholder is False
        kinds = [m.kind for m in scenario.messages]
        assert {"goal", "clarify", "constraint", "term", "switch"} <= set(kinds)
        assert kinds.count("question") >= 3
        for i, message in enumerate(scenario.messages):
            if message.kind == "switch":
                msg = f"switch at {i} has no following message"
                assert i + 1 < len(scenario.messages), msg
                next_kind = scenario.messages[i + 1].kind
                msg = f"after switch at {i}: {next_kind} != 'question'"
                assert next_kind == "question", msg
            for field, needles in message.expect_state.items():
                assert field in chat_eval.STATE_FIELDS and needles
    obs = scenarios[0]
    assert obs.messages[3].expect_state == {"terms": ("дубль", "advent record")}
    assert obs.messages[0].facts[0] == (
        "verify_capture",
        "чёрн",
        "black",
        "программн",
        "program scene",
        "заголов",
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"kind": "chitchat"},
        {"text": " "},
        {"facts": [[]]},
        {"expect_state": ["goal"]},  # the draft's list shape is not {field: [substring]}
        {"expect_state": {"decisions": ["x"]}},
        {"expect_state": {"goal": []}},
        {"sources": "CLAUDE.md"},
    ],
)
def test_bad_scenario_files_are_adventerrors(tmp_path, patch):
    message = {"text": "вопрос", "kind": "question"}
    message.update(patch)
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps({"scenarios": [{"id": "a", "title": "t", "messages": [message]}]}),
        encoding="utf-8",
    )

    with pytest.raises(AdventError):
        chat_eval.load_scenarios(path)


def test_missing_broken_and_duplicate_scenario_files_are_adventerrors(tmp_path):
    with pytest.raises(AdventError):
        chat_eval.load_scenarios(tmp_path / "нет.json")
    broken = tmp_path / "b.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(AdventError):
        chat_eval.load_scenarios(broken)
    dup = tmp_path / "d.json"
    one = {"id": "a", "title": "t", "messages": [{"text": "q", "kind": "goal"}]}
    dup.write_text(json.dumps({"scenarios": [one, one]}), encoding="utf-8")
    with pytest.raises(AdventError):
        chat_eval.load_scenarios(dup)


def _scenario_file(tmp_path):
    def encode(message: chat_eval.ChatMessage) -> dict:
        return {
            "text": message.text,
            "kind": message.kind,
            "facts": [list(alts) for alts in message.facts],
            "expect_state": {k: list(v) for k, v in message.expect_state.items()},
        }

    items = [
        {
            "id": sid,
            "title": sid,
            "placeholder": True,
            "messages": [encode(M) for M in (M1, M2, M3)],
        }
        for sid in ("obs", "budget")
    ]
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": items}, ensure_ascii=False), encoding="utf-8")
    return path


def test_run_chat_eval_runs_every_scenario_on_a_fresh_agent_and_prints_the_summary_last(
    h, out, err, tmp_path, monkeypatch
):
    built: list[Agent] = []

    def fake_build(model, db_path=None, **kwargs):
        _happy(h)
        agent = h.agent()
        built.append(agent)
        return agent

    checked: list[tuple] = []
    monkeypatch.setattr(chat_eval, "build_chat_agent", fake_build)
    monkeypatch.setattr(
        rag_module, "check_index", lambda db_path, strategy: checked.append((db_path, strategy))
    )

    results = chat_eval.run_chat_eval(_scenario_file(tmp_path), detail=False)

    assert [r.scenario.id for r in results] == ["obs", "budget"]
    assert len(built) == 2 and built[0] is not built[1]
    assert checked == [(None, "structure")]
    text = _plain(out.getvalue())
    assert text.count("Мини-чат: сценарии") == 1
    assert re.sub(r"\s+", " ", text).strip().endswith(CLOSING)
    assert "сценарий obs: факты не сверены" in _plain(err.getvalue())


def test_unknown_scenario_id_is_refused_before_any_paid_call(h, out, err, tmp_path, monkeypatch):
    monkeypatch.setattr(
        chat_eval, "build_chat_agent", lambda *a, **k: pytest.fail("agent must not be built")
    )
    monkeypatch.setattr(rag_module, "check_index", lambda db_path, strategy: None)

    with pytest.raises(AdventError) as caught:
        chat_eval.run_chat_eval(_scenario_file(tmp_path), scenario_id="нет")

    assert "Нет сценария 'нет'" in caught.value.message
    assert "obs, budget" in caught.value.hint


def test_single_scenario_flag_runs_only_that_one(h, out, err, tmp_path, monkeypatch):
    monkeypatch.setattr(chat_eval, "build_chat_agent", lambda *a, **k: (_happy(h), h.agent())[1])
    monkeypatch.setattr(rag_module, "check_index", lambda db_path, strategy: None)

    results = chat_eval.run_chat_eval(_scenario_file(tmp_path), scenario_id="budget", detail=False)

    assert [r.scenario.id for r in results] == ["budget"]


def test_chat_eval_is_registered_on_the_adventrag_app():
    result = CliRunner().invoke(week05_cli.app, ["chat-eval", "--help"])

    assert result.exit_code == 0
    for flag in ("--scenario", "--detail", "--no-detail", "--model", "--db"):
        assert flag in _plain(result.output)


MODEL_CARD = {
    "id": "ministral-14b-latest",
    "aliases": [],
    "max_context_length": 262144,
    "capabilities": {"function_calling": True, "reasoning": False},
}


@pytest.fixture
def wiring(monkeypatch, err):
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    counter = SimpleNamespace(exact=True, name="тест")
    retriever = lambda question, settings: None  # noqa: E731 - identity is what is checked
    made: list[dict] = []

    def make_retriever(db_path=None, **kwargs):
        made.append({"db_path": db_path, **kwargs})
        return retriever

    models = {"value": [MODEL_CARD]}

    def list_models(config):
        if isinstance(models["value"], Exception):
            raise models["value"]
        return models["value"]

    monkeypatch.setattr(chat_eval, "list_models", list_models)
    monkeypatch.setattr(chat_eval, "counter_for", lambda model, on_notice=None: (counter, None))
    monkeypatch.setattr(rag_module, "make_retriever", make_retriever)
    return SimpleNamespace(counter=counter, retriever=retriever, made=made, models=models)


def test_agent_is_built_the_way_the_repl_builds_it(wiring, tmp_path):
    agent = chat_eval.build_chat_agent(None, tmp_path / "index.sqlite3")

    config = Config.resolve(stream=False)
    assert agent.config.model == config.model
    assert agent.memory_prompt == agent_cli._memory_prompt()
    assert agent.memory_prompt and "goal" in agent.memory_prompt
    assert agent.summary_prompt == agent_cli._summary_prompt()
    assert agent.persona == agent_cli._persona(agent.config)
    assert agent.persona
    assert agent.counter is wiring.counter
    assert agent.capabilities == {"function_calling": True, "reasoning": False}
    assert agent.context_limit == 262144
    assert agent.context_strategy == "memory"
    assert agent.rag_enabled is True
    params = agent.config.params
    assert (params.rag_cite, params.rag_rewrite, params.rag_rerank) == (True, True, True)
    assert (params.rag_strategy, params.rag_k, params.rag_k_before, params.rag_threshold) == (
        "structure",
        5,
        20,
        5.0,
    )
    assert agent.config.stream is False
    # Retrieval is journaled as day 25 end to end: embed and the rewrite/rerank rows alike.
    assert agent._retrieve is wiring.retriever
    assert wiring.made == [{"db_path": tmp_path / "index.sqlite3", "day": 25, "aux_day": 25}]


def test_agent_survives_an_unreadable_model_list_but_refuses_an_unknown_model(wiring):
    wiring.models["value"] = AdventError("нет сети")
    agent = chat_eval.build_chat_agent(None)
    assert (agent.capabilities, agent.context_limit) == (None, None)

    wiring.models["value"] = [MODEL_CARD]
    with pytest.raises(ConfigError):
        chat_eval.build_chat_agent("нет-такой-модели")


def test_memory_strategy_stays_on_with_the_built_agent_wiring(wiring):
    agent = chat_eval.build_chat_agent(None)
    warnings: list[str] = []
    agent._on_warning = warnings.append

    # The memory extractor prompt is present, so the strategy does not degrade.
    assert agent.memory_prompt
    assert agent.context_strategy == "memory"
    assert warnings == []


def test_goal_cell_is_short_and_unambiguous():
    stable = chat_eval.GoalStability(pinned_from=1, stable=True)
    broken = chat_eval.GoalStability(pinned_from=2, stable=False)
    assert chat_eval._goal_cell(stable) == "✓ ход 1"
    assert chat_eval._goal_cell(broken) == "✗"
    assert chat_eval._goal_cell(chat_eval.GoalStability(pinned_from=None, stable=False)) == "—"
