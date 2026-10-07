"""Day 28 bench: order, config isolation, ledger, outcomes, aggregates, tables, command wiring."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from rich.console import Console

from advent_cli import record
from advent_core import config as config_module
from advent_core import console, offline
from advent_core.config import LOCAL_API_KEY, Config, ConfigError
from advent_core.errors import AdventError, NetworkError
from advent_core.params import GenerationParams
from advent_core.rag import CitedAnswer, LedgerEntry, Quote, RagContext, RagTimings
from week_01.models_bench import ModelPrice
from week_05 import rag as rag_module
from week_05 import rag_cli
from week_06 import ragbench as rb

LOCAL_URL = "http://127.0.0.1:1234"
Q1 = rag_module.ControlQuestion(1, "что такое alpha?", (("alpha",),), ("a.md",))
Q2 = rag_module.ControlQuestion(2, "что такое beta?", (("beta",), ("gamma",)), ("a.md",))
UN = rag_module.UnanswerableQuestion(101, "как варить борщ?")
GOOD_QUOTE = Quote(1, "цитата из контекста", True)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.delenv("ADVENT_LOCAL_URL", raising=False)
    monkeypatch.setattr(rb, "_pause", lambda seconds: None)


def _config(local: bool) -> Config:
    return Config(
        api_key=LOCAL_API_KEY if local else "k" * 12,
        model="ornith" if local else "ministral-14b-latest",
        params=GenerationParams.build(max_tokens=4096),
        stream=False,
        base_url=LOCAL_URL if local else None,
        offline=local,
    )


def _backend(name: str) -> rb.Backend:
    return rb.Backend(name, _config(name == "local"), Path(f"{name}.sqlite3"))


def _cited(answer="alpha", quotes=(GOOD_QUOTE,), status="answer", reason=""):
    return CitedAnswer(status, answer if status == "answer" else "", (), (), tuple(quotes), reason)


def _mode_run(cited, *, ctx=None, called=True) -> rag_cli.ModeRun:
    return rag_cli.ModeRun(
        text=cited.answer or "не знаю",
        ctx=ctx,
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=100,
        mode="cite",
        cited=cited,
        model_called=called,
    )


def _entry(stage="answer", endpoint="cloud", prompt=10, completion=5, latency=1000, model="m"):
    return LedgerEntry(stage, model, endpoint, prompt, completion, latency)


def _plan(*, questions=(Q1, Q2), unanswerable=()) -> rb.Plan:
    return rb.Plan(tuple(questions), tuple(unanswerable), "rev1")


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _FakeOffline:
    def __init__(self, events=None, *, blocked=0, enabled=False):
        self.events = events if events is not None else []
        self.blocked = blocked
        self.enabled = enabled

    def is_enabled(self):
        return self.enabled

    def enable(self):
        self.enabled = True
        self.events.append("enable")

    def reset_counters(self):
        self.events.append("reset")

    def counters(self):
        return offline.Counters(
            attempted={"local": 3, "cloud": self.blocked},
            blocked={"local": 0, "cloud": self.blocked},
            completed={"local": 3, "cloud": 0},
        )


def _cell(backend="local", qid=1, run=1, *, kind="answerable", outcome="answered", **kw):
    defaults = dict(
        correct=outcome == "answered",
        facts=(True,),
        facts_total=1,
        wall_ms=1000,
    )
    defaults.update(kw)
    return rb.BenchRun(backend, kind, qid, run, outcome, **defaults)


def _render(table_or_text, width=80) -> str:
    buf = io.StringIO()
    Console(file=buf, width=width, no_color=True, force_terminal=False).print(table_or_text)
    return buf.getvalue()


def _ok_ask(events=None):
    def ask(backend, question, run, on_call):
        if events is not None:
            events.append(("ask", backend.name, question.id, run, backend.config.is_local))
        on_call(_entry(endpoint=backend.endpoint))
        return _mode_run(_cited("alpha beta gamma"))

    return ask


# --- configs ------------------------------------------------------------------------------


def test_cloud_config_ignores_hostile_env(monkeypatch):
    monkeypatch.setenv("ADVENT_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("ADVENT_OFFLINE", "1")
    monkeypatch.setenv("MISTRAL_API_KEY", "realkey123456")
    monkeypatch.setenv("MISTRAL_TEMPERATURE", "0.9")
    cfg = rb.cloud_config()
    assert cfg is not None
    assert cfg.base_url is None
    assert cfg.offline is False
    assert cfg.is_local is False
    assert cfg.api_key == "realkey123456"
    assert cfg.model == "ministral-14b-latest"
    assert cfg.params.temperature is None
    assert cfg.params.max_tokens == 4096


def test_cloud_config_is_none_without_a_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    assert rb.cloud_config() is None


def test_local_config_is_loopback_offline_and_never_reads_the_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "realkey123456")
    cfg = rb.local_config(LOCAL_URL)
    assert (cfg.api_key, cfg.model, cfg.offline, cfg.is_local) == (
        LOCAL_API_KEY,
        "ornith",
        True,
        True,
    )
    with pytest.raises(ConfigError):
        rb.local_config("http://example.com:1234")


def test_apply_rag_settings_is_explicit_and_isolated_per_copy():
    base = _config(False)
    first, second = rb.isolated_config(base), rb.isolated_config(base)
    rb.apply_rag_settings(first.params)
    assert base.params.rag_k is None and base.params.rag_cite is None
    assert second.params.rag_k is None
    p = first.params
    assert (p.rag_strategy, p.rag_k, p.rag_k_before, p.rag_threshold) == ("structure", 5, 20, 5.0)
    assert (p.rag, p.rag_rewrite, p.rag_rerank, p.rag_cite) == (True, True, True, True)
    assert p.rag_aux_reasoning is False
    assert (p.max_tokens, p.temperature, p.top_p) == (4096, None, None)


def test_default_ask_builds_an_isolated_agent_with_week_day_and_identity(monkeypatch):
    seen: dict = {}

    def fake_build(config, db_path, day, **kw):
        seen["build"] = (config, db_path, day, kw)
        return "agent"

    def fake_run_mode(agent, question, **kw):
        seen["run_mode"] = (agent, question, kw)
        return "mode-run"

    monkeypatch.setattr(rag_cli, "build_agent", fake_build)
    monkeypatch.setattr(rag_cli, "run_mode", fake_run_mode)
    backend = _backend("local")
    sink = []
    assert rb.default_ask(backend, Q2, 3, sink.append) == "mode-run"
    config, db, day, kw = seen["build"]
    identity = {"command": "ragbench", "backend": "local", "run": 3, "question_id": 2}
    assert config is not backend.config and config.params is not backend.config.params
    assert backend.config.params.rag_k is None  # the input config stays untouched
    assert kw["aux_config"] is config
    assert (db, day, kw["week"], kw["extra"], kw["on_call"]) == (
        backend.db_path,
        28,
        6,
        identity,
        sink.append,
    )
    _, question, rm = seen["run_mode"]
    assert question == Q2.question
    assert (rm["mode"], rm["command"], rm["day"], rm["week"], rm["extra"]) == (
        "cite",
        "ragbench",
        28,
        6,
        identity,
    )
    assert config.params.rag_aux_reasoning is False


# --- order ----------------------------------------------------------------------------------


def test_cloud_runs_first_then_guard_then_local_is_built_and_run():
    events: list = []
    fake = _FakeOffline(events)

    def cloud_factory():
        events.append(("factory", "cloud", fake.enabled))
        return _backend("cloud")

    def local_factory():
        events.append(("factory", "local", fake.enabled))
        return _backend("local")

    result = rb.run_bench(
        {"local": local_factory, "cloud": cloud_factory},
        _plan(questions=(Q1,)),
        runs=1,
        ask=_ok_ask(events),
        offline_mod=fake,
        on_run=lambda cell, n: None,
    )
    assert events == [
        ("factory", "cloud", False),
        ("ask", "cloud", 1, 1, False),
        "enable",
        "reset",
        ("factory", "local", True),
        ("ask", "local", 1, 1, True),
    ]
    assert result.backends == ["cloud", "local"]
    assert result.offline is not None and result.offline.ok


def test_local_only_enables_the_guard_before_its_factory():
    events: list = []
    fake = _FakeOffline(events)
    rb.run_bench(
        {"local": lambda: events.append("factory") or _backend("local")},
        _plan(questions=(Q1,)),
        runs=1,
        ask=_ok_ask(),
        offline_mod=fake,
        on_run=lambda cell, n: None,
    )
    assert events == ["enable", "reset", "factory"]


def test_cloud_is_refused_when_the_guard_is_already_on():
    with pytest.raises(AdventError) as caught:
        rb.run_bench(
            {"cloud": lambda: _backend("cloud")},
            _plan(),
            runs=1,
            ask=_ok_ask(),
            offline_mod=_FakeOffline(enabled=True),
        )
    assert "Offline-режим уже включён" in caught.value.message


def test_runs_are_run_major_inside_a_backend():
    events: list = []
    rb.run_bench(
        {"cloud": lambda: _backend("cloud")},
        _plan(questions=(Q1, Q2), unanswerable=(UN,)),
        runs=2,
        ask=_ok_ask(events),
        offline_mod=_FakeOffline(),
        on_run=lambda cell, n: None,
    )
    assert [(e[2], e[3]) for e in events] == [
        (1, 1),
        (2, 1),
        (101, 1),
        (1, 2),
        (2, 2),
        (101, 2),
    ]


def test_a_failed_cell_is_an_error_cell_and_the_bench_goes_on():
    def ask(backend, question, run, on_call):
        if (question.id, run) == (1, 1):
            on_call(_entry("rewrite"))
            raise AdventError("сервер упал")
        return _mode_run(_cited("alpha beta gamma"))

    result = rb.run_bench(
        {"cloud": lambda: _backend("cloud")},
        _plan(),
        runs=2,
        ask=ask,
        offline_mod=_FakeOffline(),
        on_run=lambda cell, n: None,
    )
    assert len(result.runs) == 4
    bad = result.runs[0]
    assert (bad.outcome, bad.error, bad.ok) == ("error", "сервер упал", False)
    assert [c.stage for c in bad.ledger] == ["rewrite"]  # the paid call survives the failure
    found, total = rb.facts_totals(result.runs)
    assert (found, total) == (1 + 2 + 2, 2 + 4)  # the failed cell stays in the total
    assert "1/4" in _render(rb.quality_table(result.runs, ["cloud"]))


def test_transient_error_is_retried_and_keeps_every_completed_call():
    clock = _Clock()
    calls = {"n": 0}

    def ask(backend, question, run, on_call):
        calls["n"] += 1
        on_call(_entry("embed", completion=None, latency=50))
        clock.t += 1.0
        if calls["n"] == 1:
            raise NetworkError("обрыв")
        on_call(_entry("answer"))
        return _mode_run(_cited("alpha beta gamma"))

    result = rb.run_bench(
        {"cloud": lambda: _backend("cloud")},
        _plan(questions=(Q1,)),
        runs=1,
        ask=ask,
        offline_mod=_FakeOffline(),
        clock=clock,
        pause=lambda s: setattr(clock, "t", clock.t + s),
        on_run=lambda cell, n: None,
    )
    cell = result.runs[0]
    assert cell.retries == 1 and cell.ok
    assert [c.stage for c in cell.ledger] == ["embed", "embed", "answer"]
    assert cell.wall_ms == int((1.0 + rb.RETRY_PAUSE_S + 1.0) * 1000)  # the pause is in the wall


def test_a_non_transient_error_is_not_retried():
    calls = {"n": 0}

    def ask(backend, question, run, on_call):
        calls["n"] += 1
        raise AdventError("плохой запрос")

    result = rb.run_bench(
        {"cloud": lambda: _backend("cloud")},
        _plan(questions=(Q1,)),
        runs=1,
        ask=ask,
        offline_mod=_FakeOffline(),
        on_run=lambda cell, n: None,
    )
    assert calls["n"] == 1 and result.runs[0].retries == 0


def test_local_setup_failure_keeps_the_cloud_results():
    def broken_local():
        raise AdventError("ornith не загружена", hint="запусти сервер")

    result = rb.run_bench(
        {"cloud": lambda: _backend("cloud"), "local": broken_local},
        _plan(questions=(Q1,)),
        runs=1,
        ask=_ok_ask(),
        offline_mod=_FakeOffline(),
        on_run=lambda cell, n: None,
    )
    assert len(result.runs) == 1 and result.backends == ["cloud"]
    assert "local: ornith не загружена" in (result.failed or "")
    assert rb.exit_code(result) == 1


def test_a_setup_failure_with_nothing_run_raises():
    def broken():
        raise AdventError("нет индекса")

    with pytest.raises(AdventError):
        rb.run_bench({"cloud": broken}, _plan(), runs=1, ask=_ok_ask(), offline_mod=_FakeOffline())


# --- offline acceptance ---------------------------------------------------------------------


def test_blocked_cloud_attempt_in_the_local_phase_fails_acceptance_after_the_tables(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(
        console, "out", Console(file=buf, width=80, no_color=True, force_terminal=False)
    )
    result = rb.run_bench(
        {"local": lambda: _backend("local")},
        _plan(questions=(Q1,)),
        runs=1,
        ask=_ok_ask(),
        offline_mod=_FakeOffline(blocked=2),
        on_run=lambda cell, n: None,
    )
    assert result.offline is not None and not result.offline.ok
    assert rb.exit_code(result) == 1
    rb.print_report(result)
    text = buf.getvalue()
    assert "заблокировано 2" in text
    assert text.index("offline-приёмка НЕ пройдена") > text.index("Стабильность")


def test_a_cloud_model_call_in_the_local_phase_fails_acceptance():
    def ask(backend, question, run, on_call):
        on_call(_entry(endpoint="cloud"))
        return _mode_run(_cited("alpha beta gamma"))

    result = rb.run_bench(
        {"local": lambda: _backend("local")},
        _plan(questions=(Q1,)),
        runs=1,
        ask=ask,
        offline_mod=_FakeOffline(),
        on_run=lambda cell, n: None,
    )
    assert result.offline.ledger_cloud == 1 and not result.offline.ok


def test_clean_local_phase_is_accepted_and_counts_local_calls():
    result = rb.run_bench(
        {"local": lambda: _backend("local")},
        _plan(questions=(Q1,)),
        runs=1,
        ask=_ok_ask(),
        offline_mod=_FakeOffline(),
        on_run=lambda cell, n: None,
    )
    assert (result.offline.ledger_local, result.offline.ledger_cloud) == (1, 0)
    assert result.offline.ok and rb.exit_code(result) == 0


def test_offline_line_has_http_counters():
    report = rb.OfflineReport(
        attempted={"local": 7, "cloud": 0},
        blocked={"local": 0, "cloud": 0},
        completed={"local": 7, "cloud": 0},
        ledger_local=5,
        ledger_cloud=0,
    )
    line = rb.offline_line(report)
    assert "модельных вызовов локальных 5, облачных 0 (заблокировано 0)" in line
    assert "попыток локальных 7, облачных 0" in line
    assert "завершено локальных 7, облачных 0" in line


# --- outcomes, quotes -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cited", "outcome", "correct_answerable", "correct_unanswerable"),
    [
        (_cited("alpha"), "answered", True, False),
        (_cited(status="unknown", reason="empty_context", quotes=()), "empty_context", False, True),
        (_cited(status="unknown", reason="model_unknown", quotes=()), "model_unknown", False, True),
        (_cited(status="unknown", reason="unverified", quotes=()), "unverified", False, False),
        (_cited(status="unknown", reason="bad_json", quotes=()), "bad_json", False, False),
        (_cited(status="unknown", reason="truncated", quotes=()), "truncated", False, False),
        (_cited(status="unknown", reason="no_index", quotes=()), "no_index", False, False),
    ],
)
def test_outcome_categories_and_what_counts_as_correct(
    cited, outcome, correct_answerable, correct_unanswerable
):
    mode_run = _mode_run(cited)
    assert rb.outcome_of(mode_run) == outcome
    a = rb.build_cell("local", Q1, 1, mode_run, [], 10, 0)
    u = rb.build_cell("local", UN, 1, mode_run, [], 10, 0)
    assert (a.outcome, a.correct) == (outcome, correct_answerable)
    assert (u.outcome, u.correct) == (outcome, correct_unanswerable)


def test_an_answer_without_a_verbatim_quote_scores_no_facts():
    cited = _cited("alpha", quotes=(Quote(1, "не дословно", False),), status="unknown")
    cell = rb.build_cell("local", Q1, 1, _mode_run(cited), [], 10, 0)
    assert cell.facts == (False,) and not cell.all_facts and not cell.correct


def test_quote_accounting_is_split_into_four_counters():
    quotes = (
        Quote(1, "a", True),
        Quote(2, "b", True, claimed=5),
        Quote(3, "c", False),
        Quote(1, "d", False, claimed=4),
    )
    cell = rb.build_cell("cloud", Q1, 1, _mode_run(_cited("alpha", quotes)), [], 10, 0)
    assert (
        cell.quotes_emitted,
        cell.quotes_reattributed,
        cell.quotes_dropped,
        cell.quotes_verbatim,
    ) == (4, 2, 2, 2)
    stats = rb.quote_stats([cell, _cell(outcome="empty_context")])
    assert stats["answers_with_quotes"] == 1 and stats["all_verbatim"] == 0


def test_all_verbatim_needs_a_non_empty_set_of_quotes():
    refusal = _cell(outcome="model_unknown")
    full = _cell(quotes_emitted=2, quotes_verbatim=2)
    stats = rb.quote_stats([refusal, full])
    assert (stats["answers_with_quotes"], stats["all_verbatim"]) == (1, 1)


def test_stage_times_are_ledger_sums_over_every_attempt_not_the_final_attempt_timings():
    ledger = [
        rb.CallRecord("rewrite", "m", "cloud", 1, 1, 300, None),
        rb.CallRecord("embed", "e", "cloud", 4, None, 800, None),
        rb.CallRecord("search", "e", "cloud", None, None, 4, None),
        rb.CallRecord("rerank", "m", "cloud", 1, 1, 400, None),
        # transient retry: the whole retrieval ran again
        rb.CallRecord("embed", "e", "cloud", 4, None, 300, None),
        rb.CallRecord("search", "e", "cloud", None, None, 5, None),
        rb.CallRecord("rerank", "m", "cloud", 1, 1, 100, None),
        rb.CallRecord("answer", "m", "cloud", 10, 50, 2000, None),
    ]
    # the final attempt's own timings disagree on purpose and must be ignored
    timings = RagTimings(300, 9999, 9999, 500)
    ctx = RagContext((), "structure", 5, "emb", None, "rev", timings=timings)
    cell = rb.build_cell("cloud", Q1, 1, _mode_run(_cited("alpha"), ctx=ctx), ledger, 3000, 1)
    assert (cell.rewrite_ms, cell.rerank_ms, cell.answer_ms) == (300, 500, 2000)
    assert (cell.embed_ms, cell.search_ms) == (1100, 9)
    assert cell.embed_search_ms is None
    assert cell.tok_s == 25.0


def test_a_missing_search_entry_is_unknown_not_zero():
    ledger = [rb.CallRecord("embed", "e", "cloud", 4, None, 800, None)]
    cell = rb.build_cell("cloud", Q1, 1, _mode_run(_cited("alpha")), ledger, 3000, 0)
    assert cell.embed_ms == 800 and cell.search_ms is None


def test_search_entry_costs_nothing_and_is_not_a_model_call_in_the_offline_report():
    search = LedgerEntry("search", "emb", "cloud", None, None, 5)
    assert rb.ledger_cost(search, None) == 0.0

    def ask(backend, question, run, on_call):
        on_call(_entry(stage="embed", endpoint="local", completion=None))
        on_call(_entry(stage="search", endpoint="local", prompt=None, completion=None))
        on_call(_entry(endpoint="local"))
        return _mode_run(_cited("alpha"))

    result = rb.run_bench(
        {"local": lambda: _backend("local")},
        _plan(questions=(Q1,)),
        runs=1,
        ask=ask,
        offline_mod=_FakeOffline(),
        on_run=lambda cell, n: None,
    )
    assert (result.offline.ledger_local, result.offline.ledger_cloud) == (2, 0)


def test_sources_are_alternatives_one_cited_source_is_enough():
    question = rag_module.ControlQuestion(7, "что такое alpha?", (("alpha",),), ("a.md", "b.md"))
    hit = rag_module.RagHit("c1", "a.md", "S", 0.5, "цитата из контекста")
    cited = CitedAnswer("answer", "alpha", (hit,), (hit,), (GOOD_QUOTE,), "")
    cell = rb.build_cell("cloud", question, 1, _mode_run(cited), [], 10, 0)
    assert cell.sources_cited is True


# --- stability ------------------------------------------------------------------------------


def test_stability_patterns_are_literal():
    cells = [
        _cell(qid=1, run=1),
        _cell(qid=1, run=2, facts=(False,), correct=False),
        _cell(qid=1, run=3),
        _cell(qid=2, run=1, outcome="model_unknown", correct=False, facts=(False,)),
        _cell(qid=2, run=2, outcome="model_unknown", correct=False, facts=(False,)),
        _cell(qid=2, run=3, outcome="model_unknown", correct=False, facts=(False,)),
        _cell(qid=3, run=1, outcome="error", error="x", correct=False, facts=()),
        _cell(qid=3, run=2),
    ]
    st = rb.stability(cells)
    assert st[1]["pattern"] == "✓✗✓"
    assert st[2]["pattern"] == "ббб"
    assert st[3]["pattern"] == "!✓"
    assert (st[1]["same_verdict"], st[1]["all_correct"]) == (False, False)
    assert st[2]["same_verdict"] is True and st[3]["same_verdict"] is False


def test_unverified_and_format_cells_have_their_own_glyphs_and_legend():
    cells = [
        _cell(qid=1, run=1, outcome="unverified", correct=False, facts=(False,)),
        _cell(qid=1, run=2, outcome="bad_json", correct=False, facts=(False,)),
        _cell(qid=1, run=3, outcome="truncated", correct=False, facts=(False,)),
        _cell(qid=1, run=4, facts=(False,), correct=False),
    ]
    assert rb.stability(cells)[1]["pattern"] == "нф" + "ф✗"
    text = _render(rb.stability_table(cells, ["local"]))
    assert "нфф✗" in text
    flat = " ".join(text.split())
    assert "н неподтверждено" in flat and "ф формат" in flat


def test_consistent_failure_is_never_reported_as_stable_correct():
    cells = [_cell(run=r, facts=(False,), correct=False) for r in (1, 2, 3)]
    st = rb.stability(cells)[1]
    assert st["pattern"] == "✗✗✗"
    assert st["same_verdict"] is True and st["all_correct"] is False
    assert rb.stability_counts(cells) == (1, 0, 1)
    lines = rb.stability_lines(cells)
    assert "local 1/1" in lines[0]
    assert "все прогоны верны: local 0/1" in lines[1]


def test_same_outcome_with_a_different_correctness_is_not_stable():
    cells = [_cell(run=1), _cell(run=2, facts=(False,), correct=False), _cell(run=3)]
    assert {c.outcome for c in cells} == {"answered"}
    assert rb.stability_counts(cells) == (0, 0, 1)
    refused = _cell("local", 101, 1, kind="unanswerable", outcome="empty_context", facts=())
    refused.correct = True
    answered = _cell("local", 101, 2, kind="unanswerable", outcome="answered", facts=())
    answered.correct = False
    assert rb.stability_counts([refused, answered]) == (0, 0, 1)


def test_all_errors_are_the_same_outcome_but_not_correct():
    cells = [_cell(run=r, outcome="error", error="x", correct=False, facts=()) for r in (1, 2)]
    assert rb.stability_counts(cells) == (1, 0, 1)


# --- ties and single backend ---------------------------------------------------------------


def _both(local_ok, cloud_ok, *, local_ms=1000, cloud_ms=1000):
    cells = []
    for name, ok, ms in (("local", local_ok, local_ms), ("cloud", cloud_ok, cloud_ms)):
        for r in (1, 2):
            cells.append(_cell(name, 1, r, facts=(ok,), correct=ok, wall_ms=ms))
    return cells


def test_equal_quality_and_speed_name_no_leader():
    runs = _both(True, True)
    assert rb.quality_line(runs) == "качество: ничья (local 2/2, cloud 2/2)"
    assert rb.speed_line(runs).startswith("скорость: ничья")
    assert "ничья" in rb.stability_lines(runs)[-1]


def test_quality_leader_and_small_difference_note():
    runs = _both(True, False)
    line = rb.quality_line(runs)
    assert line.startswith("качество: выше local (local 2/2, cloud 0/2)")
    one_off = [
        _cell("local", 1, 1),
        _cell("local", 1, 2),
        _cell("cloud", 1, 1),
        _cell("cloud", 1, 2, facts=(False,), correct=False),
    ]
    assert "разница 1, малая выборка" in rb.quality_line(one_off)


def test_speed_ratio_and_direction():
    assert "local медленнее в 3.0×" in rb.speed_line(
        _both(True, True, local_ms=9000, cloud_ms=3000)
    )
    assert "local быстрее в 2.0×" in rb.speed_line(_both(True, True, local_ms=1000, cloud_ms=2000))


def test_stability_leader_uses_all_runs_correct_only():
    runs = _both(True, False)
    assert rb.stability_lines(runs)[-1] == "стабильность: чаще верен во всех прогонах local"


def test_single_backend_prints_no_comparison():
    result = rb.BenchResult(
        runs=[_cell("local", 1, 1), _cell("local", 1, 2)],
        backends=["local"],
        n_runs=2,
        plan=_plan(questions=(Q1,)),
    )
    lines = rb.conclusion_lines(result)
    assert lines == ["2 прогона × 1 вопрос — малая выборка"]
    both = rb.BenchResult(runs=_both(True, True), backends=["cloud", "local"], n_runs=2)
    both.plan = _plan(questions=(Q1,))
    assert any(line.startswith("качество:") for line in rb.conclusion_lines(both))


def test_comparison_without_answers_says_so_instead_of_picking():
    runs = [_cell("local", 1, 1, outcome="error", error="x", correct=False, facts=())]
    assert "сравнить нечем" in rb.speed_line(runs)
    assert "сравнить нечем" in rb.quality_line([])


# --- cost -----------------------------------------------------------------------------------


class _Prices:
    def price_of(self, model):
        return ModelPrice(input=0.2, cached_input=0.02, output=0.6) if model == "m" else None


def test_ledger_cost_local_is_zero_cloud_uses_the_price_table_unknown_is_none():
    result = SimpleNamespace(usage=SimpleNamespace(cached_tokens=None))
    chat = LedgerEntry("answer", "m", "cloud", 1_000_000, 1_000_000, 10, result)
    assert rb.ledger_cost(chat, _Prices()) == pytest.approx(0.8)
    assert rb.ledger_cost(chat, None) is None
    assert rb.ledger_cost(LedgerEntry("answer", "x", "cloud", 1, 1, 10, result), _Prices()) is None
    assert (
        rb.ledger_cost(LedgerEntry("answer", "ornith", "local", None, None, 10, result), None) == 0
    )
    embed = LedgerEntry("embed", "mistral-embed", "cloud", 1_000_000, None, 10)
    assert rb.ledger_cost(embed, None) == pytest.approx(0.1)


def test_unknown_cost_is_not_zero_in_the_total():
    known = _cell(ledger=(rb.CallRecord("answer", "m", "cloud", 1, 1, 10, 0.5),))
    unknown = _cell(ledger=(rb.CallRecord("answer", "m", "cloud", 1, 1, 10, None),))
    assert rb.cost_total([known]) == 0.5
    assert rb.cost_total([known, unknown]) is None
    assert rb.cost_total([_cell()]) == 0.0


# --- tables ---------------------------------------------------------------------------------


def _table_runs():
    cells = []
    for name, ok in (("local", False), ("cloud", True)):
        for r in (1, 2, 3):
            cells.append(
                _cell(
                    name,
                    1,
                    r,
                    facts=(ok,),
                    correct=ok,
                    wall_ms=12345 + r * 1000,
                    quotes_emitted=2,
                    quotes_verbatim=2,
                    ledger=(rb.CallRecord("answer", "m", name, 1234, 567, 900, 0.001),),
                )
            )
        cells.append(_cell(name, 101, 1, kind="unanswerable", outcome="empty_context", facts=()))
        cells[-1].correct = True
    return cells


def test_tables_render_at_80_columns_without_ellipsis_and_with_one_header():
    runs = _table_runs()
    backends = ["cloud", "local"]
    quality = _render(rb.quality_table(runs, backends))
    speed = _render(rb.speed_table(runs, backends))
    stab = _render(rb.stability_table(runs, backends))
    for text in (quality, speed, stab):
        assert "…" not in text
        assert max(cell_len(line) for line in text.splitlines()) <= 80
    assert quality.count("Качество") == 1 and speed.count("Скорость") == 1
    assert stab.count("Стабильность") == 1
    assert quality.count("факты найдено") == 1
    row = next(line for line in quality.splitlines() if "факты найдено" in line)
    assert "0/3" in row and "3/3" in row
    assert "неотвечаемые: верный отказ" in quality
    assert "1/1" in next(line for line in quality.splitlines() if "верный отказ" in line)
    assert "стена на вопрос, медиана" in speed and "13.8 s" in speed
    assert "✗✗✗" in stab and "✓✓✓" in stab


def test_speed_table_shows_unknown_tokens_as_a_question_mark():
    cell = _cell(ledger=(rb.CallRecord("answer", "m", "local", None, None, 10, 0.0),))
    text = _render(rb.speed_table([cell], ["local"]))
    assert "?/?" in text


def _answer_call(prompt, completion):
    return rb.CallRecord("answer", "m", "local", prompt, completion, 10, 0.0)


def test_token_totals_with_a_partly_unknown_usage_are_a_lower_bound_not_a_total():
    cells = [
        _cell(run=1, ledger=(_answer_call(100, 10),)),
        _cell(run=2, ledger=(_answer_call(None, None),)),
    ]
    totals = rb.token_totals(cells)
    assert str(totals["answer_prompt"]) == "≥100" and totals["answer_prompt"].partial
    assert str(totals["answer_completion"]) == "≥10"
    full = rb.token_totals([cells[0]])
    assert str(full["answer_prompt"]) == "100" and not full["answer_prompt"].partial
    assert str(rb.token_totals([cells[1]])["answer_prompt"]) == "?"
    text = _render(rb.speed_table(cells, ["local"]))
    row = next(line for line in text.splitlines() if "токены ответа" in line)
    assert "≥100/≥10" in row and "…" not in text
    assert "≥ — у части вызовов сервер не вернул usage" in text
    clean = _render(rb.speed_table([cells[0]], ["local"]))
    assert "≥" not in clean


def test_speed_table_has_separate_embed_and_search_rows():
    ledger = (
        rb.CallRecord("embed", "e", "local", 1, None, 800, 0.0),
        rb.CallRecord("search", "e", "local", None, None, 12, 0.0),
    )
    text = _render(rb.speed_table([_cell(ledger=ledger, embed_ms=800, search_ms=12)], ["local"]))
    embed = next(line for line in text.splitlines() if "embed, медиана" in line)
    search = next(line for line in text.splitlines() if "search, медиана" in line)
    assert "0.8 s" in embed and "0.0 s" in search
    assert "embed+search" not in text and "…" not in text


def test_empty_report_does_not_crash(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(
        console, "out", Console(file=buf, width=80, no_color=True, force_terminal=False)
    )
    rb.print_report(rb.BenchResult(n_runs=1, plan=_plan()))
    assert "Качество" not in buf.getvalue()
    assert rb.exit_code(rb.BenchResult()) == 1


# --- per-run line ---------------------------------------------------------------------------


def test_run_line_fits_the_width_in_cells_and_names_the_verdict():
    cell = _cell(
        "local", 4, 2, wall_ms=14200, quotes_emitted=3, quotes_verbatim=3, answer="😀" * 80
    )
    cell.facts = (True,)
    line = rb.run_line(cell, 3, 80)
    assert line.startswith("local  #4 прогон 2/3  ✓ 1/1  цитаты 3/3  14.2 s  «")
    assert cell_len(line) <= 80


def test_run_line_for_refusals_errors_and_unanswerable():
    refusal = _cell(outcome="empty_context", correct=False, facts=(False,))
    assert "не знаю (порог)" in rb.run_line(refusal, 1, 80)
    err = _cell(outcome="error", error="сервер упал", correct=False, facts=())
    assert "ошибка: сервер упал" in rb.run_line(err, 1, 80)
    right = _cell(kind="unanswerable", outcome="model_unknown", facts=())
    right.correct = True
    assert "не знаю ✓ (модель)" in rb.run_line(right, 1, 80)
    wrong = _cell(kind="unanswerable", outcome="answered", facts=(), correct=False)
    assert "ответил ✗" in rb.run_line(wrong, 1, 80)


# --- pre-flight, command wiring -------------------------------------------------------------


def _patch_preflight(monkeypatch, revs, *, valid=None):
    monkeypatch.setattr(
        rb,
        "_index_run",
        lambda db, backend: SimpleNamespace(corpus_rev=revs[backend], endpoint=backend),
    )
    monkeypatch.setattr(rb.index_module, "load_chunks", lambda db, strategy: [db.name])

    def fake_validate(questions, chunks):
        name = chunks[0]
        ok_ids = valid[name] if valid else None
        good = [q for q in questions if ok_ids is None or q.id in ok_ids]
        bad = [(q, "нет") for q in questions if q not in good]
        return good, bad

    monkeypatch.setattr(rb.rag_module, "validate_questions", fake_validate)
    monkeypatch.setattr(rb.rag_module, "load_questions", lambda path: [Q1, Q2])


def test_different_corpus_revisions_are_refused_with_a_hint(monkeypatch):
    _patch_preflight(monkeypatch, {"cloud": "aaaaaaaaaaaaaa", "local": "bbbbbbbbbbbbbb"})
    with pytest.raises(AdventError) as caught:
        rb.prepare_plan(["local", "cloud"], unanswerable=False)
    assert "разных снимков" in caught.value.message
    assert "adventrag index --local" in (caught.value.hint or "")


def test_question_set_is_the_intersection_valid_for_both_indexes(monkeypatch):
    _patch_preflight(
        monkeypatch,
        {"cloud": "r", "local": "r"},
        valid={"index.sqlite3": {1, 2}, "index.local.sqlite3": {2}},
    )
    plan = rb.prepare_plan(
        ["local", "cloud"],
        cloud_db=Path("index.sqlite3"),
        local_db=Path("index.local.sqlite3"),
        unanswerable=False,
    )
    assert [q.id for q in plan.questions] == [2]
    assert any("#1" in note for note in plan.notes)


def test_unknown_question_id_and_empty_intersection_are_errors(monkeypatch):
    _patch_preflight(monkeypatch, {"cloud": "r"})
    with pytest.raises(AdventError):
        rb.prepare_plan(["cloud"], ids=[99], unanswerable=False)
    _patch_preflight(monkeypatch, {"cloud": "r"}, valid={"index.sqlite3": set()})
    with pytest.raises(AdventError):
        rb.prepare_plan(["cloud"], cloud_db=Path("index.sqlite3"), unanswerable=False)


def test_unanswerable_questions_follow_the_flag(monkeypatch):
    _patch_preflight(monkeypatch, {"cloud": "r"})
    monkeypatch.setattr(rb.rag_module, "load_unanswerable", lambda path: [UN])
    assert rb.prepare_plan(["cloud"], unanswerable=True).unanswerable == (UN,)
    assert rb.prepare_plan(["cloud"], unanswerable=False).unanswerable == ()


def test_parse_backends_and_ids():
    assert rb.parse_backends("local,cloud") == ["cloud", "local"]
    assert rb.parse_backends("local") == ["local"]
    with pytest.raises(AdventError):
        rb.parse_backends("gpu")
    assert rb.parse_ids("1, 4,7") == [1, 4, 7] and rb.parse_ids(None) is None
    with pytest.raises(AdventError):
        rb.parse_ids("1,x")


def _wire(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        rb,
        "prepare_plan",
        lambda wanted, **kw: captured.setdefault("wanted", list(wanted)) and _plan(),
    )
    monkeypatch.setattr(
        rb,
        "run_bench",
        lambda factories, plan, **kw: (
            captured.update(factories=set(factories))
            or rb.BenchResult(runs=[_cell()], backends=["local"], n_runs=1, plan=plan)
        ),
    )
    monkeypatch.setattr(rb, "print_report", lambda result: None)
    monkeypatch.setattr(rb.oc, "default_url", lambda: LOCAL_URL)
    return captured


def test_missing_key_skips_cloud_with_a_warning(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    captured = _wire(monkeypatch)
    assert rb.run_rag_command(backends="local,cloud") == 0
    assert captured["wanted"] == ["local"] and captured["factories"] == {"local"}


def test_cloud_only_without_a_key_is_a_config_error(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    _wire(monkeypatch)
    with pytest.raises(ConfigError):
        rb.run_rag_command(backends="cloud")


def test_command_refuses_cloud_when_the_guard_is_already_on(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "realkey123456")
    _wire(monkeypatch)
    offline.enable()
    with pytest.raises(AdventError):
        rb.run_rag_command(backends="cloud")


def test_command_refuses_cloud_when_offline_is_only_in_the_environment(monkeypatch):
    # no offline.enable(): the guard starts at the first HTTP call, which is too late
    monkeypatch.setenv("MISTRAL_API_KEY", "realkey123456")
    monkeypatch.setenv("ADVENT_OFFLINE", "1")
    assert not offline.is_enabled()
    captured = _wire(monkeypatch)
    with pytest.raises(AdventError) as info:
        rb.run_rag_command(backends="local,cloud")
    assert "ADVENT_OFFLINE" in info.value.message
    assert "wanted" not in captured and "factories" not in captured


def test_command_loads_dotenv_before_the_offline_check(monkeypatch):
    loaded: list = []
    monkeypatch.setattr(config_module, "load_env", lambda: loaded.append(1))
    monkeypatch.delenv("ADVENT_OFFLINE", raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    _wire(monkeypatch)
    rb.run_rag_command(backends="local,cloud")
    assert loaded


def test_command_rejects_a_non_loopback_url_before_any_cloud_spend(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "realkey123456")
    captured = _wire(monkeypatch)
    with pytest.raises(ConfigError):
        rb.run_rag_command(backends="local,cloud", url="http://example.com:1234")
    assert "factories" not in captured


def test_local_backend_is_not_built_by_the_command_before_run_bench(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    built: list = []
    monkeypatch.setattr(rb, "make_local_backend", lambda cfg, db=None: built.append(cfg))
    captured = _wire(monkeypatch)
    rb.run_rag_command(backends="local")
    assert built == [] and captured["factories"] == {"local"}


def test_save_writes_every_run_and_the_effective_settings(tmp_path):
    result = rb.run_bench(
        {"cloud": lambda: _backend("cloud")},
        _plan(questions=(Q1,)),
        runs=2,
        ask=_ok_ask(),
        offline_mod=_FakeOffline(),
        on_run=lambda cell, n: None,
    )
    path = tmp_path / "out" / "bench.json"
    rb.save_json(result, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["week"], data["day"], data["runs"]) == (6, 28, 2)
    assert len(data["results"]) == 2
    settings = data["settings"]["cloud"]
    assert settings["model"] == "ministral-14b-latest" and settings["aux_reasoning"] is False
    assert settings["answer_max_tokens"] == 4096 and settings["temperature"] is None
    assert "JSON-инструкция" in settings["protocol"]
    assert data["results"][0]["ledger"][0]["stage"] == "answer"


# --- demo -----------------------------------------------------------------------------------


def test_demo_steps_w06d28_shape():
    steps = record.demo_steps(6, 28)
    assert [s.module for s in steps] == ["week_06.cli", "week_05.cli", "week_06.cli"]
    assert steps[0].args == ["status"] and steps[1].args == ["check"]
    last = steps[-1]
    assert last.args[0] == "rag" and "--no-unanswerable" in last.args
    assert last.args[last.args.index("--runs") + 1] == "2"
    assert last.args[last.args.index("--questions") + 1] == "1,7,9,10"
    assert last.timeout is not None and last.timeout >= 1200


def test_rehearsal_steps_w06d28_are_the_cheap_gates_only():
    steps = record.rehearsal_steps(6, 28)
    assert [s.args for s in steps] == [["status"], ["check"]]
