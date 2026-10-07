"""Day 28 plumbing: stage timings, journal identity, per-call ledger, aux config and week."""

from __future__ import annotations

import json

import pytest

from advent_core import config as config_module
from advent_core import journal as journal_module
from advent_core import openai_compat as oc
from advent_core.config import Config
from advent_core.errors import AdventError
from advent_core.rag import LedgerEntry, RagSettings, RagTimings
from advent_core.telemetry import CallResult, Usage
from tests.test_w06d27_rag import NOMIC, _fake_embed, _local_db, _no_cloud
from week_05 import rag
from week_05 import rag_cli as cli

_REAL_EMBED = oc.embed
LOCAL_URL = "http://127.0.0.1:1234"
IDENTITY = {"command": "ragbench", "backend": "local", "run": 2, "question_id": 4}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(journal_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.delenv("ADVENT_LOCAL_URL", raising=False)


def _aux_config() -> Config:
    return Config.resolve(offline=True, base_url=LOCAL_URL, model="ornith")


def _result(text, *, prompt=5, completion=3, model="ornith", latency=7):
    usage = Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=None)
    return CallResult(
        text=text,
        model_requested=model,
        model_actual=model,
        usage=usage,
        latency_ms=latency,
        stream=False,
    )


def _scores(n=3, score=9):
    return json.dumps({"scores": [{"id": i, "score": score} for i in range(1, n + 1)]})


def _journal(tmp_path) -> list[dict]:
    path = tmp_path / "logs" / "calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _reporting_embed(*, ms=3, tokens=5, clock=None):
    """`_fake_embed` that also honours on_request, advancing a shared clock by `ms`."""
    inner = _fake_embed()

    def fake(url, model, texts, **kw):
        out = inner(url, model, texts, **kw)
        if clock is not None:
            clock.advance(ms)
        if kw.get("on_request"):
            kw["on_request"](tokens, ms)
        return out

    return fake


def _setup(monkeypatch, tmp_path, replies):
    """Local db, faked embed and chat; `replies` is consumed one chat call at a time."""
    db = _local_db(tmp_path)
    monkeypatch.setattr(oc, "embed", _reporting_embed())
    _no_cloud(monkeypatch)
    seen: list = []

    def fake_complete(config, messages):
        seen.append(config)
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(rag.chat_core, "complete", fake_complete)
    return db, seen


FULL = RagSettings("structure", 2, 3, True, True, 5.0)


class _Clock:
    """Each call advances by 0.25 s (exact in binary) so every stage measures 250 ms."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.25
        return self.t


class _SharedClock:
    """Reading is free; the fake operations advance it by their own distinct durations."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms / 1000


def test_each_stage_is_timed_by_its_own_operation_not_by_clock_ticks(monkeypatch, tmp_path):
    clock = _SharedClock()
    db, _ = _setup(monkeypatch, tmp_path, [])
    monkeypatch.setattr(oc, "embed", _reporting_embed(ms=30, clock=clock))
    real_search = rag.index_module.search

    def slow_search(*args, **kwargs):
        clock.advance(7)
        return real_search(*args, **kwargs)

    monkeypatch.setattr(rag.index_module, "search", slow_search)
    replies = [(_result("xyz xyz"), 100), (_result(_scores()), 500)]

    def complete(config, messages):
        result, ms = replies.pop(0)
        clock.advance(ms)
        return result

    monkeypatch.setattr(rag.chat_core, "complete", complete)
    entries: list[LedgerEntry] = []
    retrieve = rag.make_retriever(db, aux_config=_aux_config(), clock=clock, on_call=entries.append)
    ctx = retrieve("xyz", FULL)
    # distinct durations: a swapped or shifted stage boundary changes at least one number
    assert ctx.timings == RagTimings(
        rewrite_ms=100, embed_ms=30, search_ms=14, rerank_ms=500
    )  # 7 ms x 2 query texts
    by_stage = {e.stage: e.latency_ms for e in entries if e.stage in ("embed", "search")}
    assert by_stage == {"embed": 30, "search": 14}


def test_a_good_embed_request_is_reported_even_when_a_later_batch_fails(monkeypatch, tmp_path):
    import functools

    import httpx

    db, _ = _setup(monkeypatch, tmp_path, [_result("xyz xyz")])
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500, json={"error": "boom"})
        body = {
            "data": [{"index": 0, "embedding": [1.0] * 768}],
            "model": NOMIC,
            "usage": {"prompt_tokens": 4},
        }
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        oc,
        "_make_client",
        lambda timeout, transport=None: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    # real batching code, one text per request: question + rewrite = two requests
    monkeypatch.setattr(oc, "embed", functools.partial(_REAL_EMBED, batch=1))
    entries: list[LedgerEntry] = []
    retrieve = rag.make_retriever(db, aux_config=_aux_config(), on_call=entries.append)
    with pytest.raises(AdventError):
        retrieve("xyz", FULL)
    assert [e.stage for e in entries] == ["rewrite", "embed"]
    assert entries[1].prompt_tokens == 4 and entries[1].endpoint == "local"


def test_timings_are_none_for_stages_that_did_not_run(monkeypatch, tmp_path):
    db, _ = _setup(monkeypatch, tmp_path, [])
    retrieve = rag.make_retriever(db, aux_config=_aux_config(), clock=_Clock())
    ctx = retrieve("xyz", RagSettings("structure", 2, 2, False, False, 0.0))
    assert ctx.timings is not None
    assert ctx.timings.rewrite_ms is None
    assert ctx.timings.rerank_ms is None
    assert ctx.timings.embed_ms == 250
    assert ctx.timings.search_ms == 250


def test_default_clock_is_a_real_one(monkeypatch, tmp_path):
    db, _ = _setup(monkeypatch, tmp_path, [])
    ctx = rag.make_retriever(db, aux_config=_aux_config())(
        "xyz", RagSettings("structure", 2, 2, False, False, 0.0)
    )
    assert isinstance(ctx.timings.embed_ms, int)
    assert ctx.timings.embed_ms >= 0


def test_extra_reaches_every_record_type_and_stage_stays_distinguishable(monkeypatch, tmp_path):
    db, _ = _setup(monkeypatch, tmp_path, [_result("xyz xyz"), _result(_scores())])
    retrieve = rag.make_retriever(db, aux_config=_aux_config(), week=6, extra=IDENTITY)
    retrieve("xyz", FULL)
    rows = _journal(tmp_path)
    assert [r["stage"] for r in rows] == ["rewrite", "embed", "rerank"]
    for row in rows:
        assert row["week"] == 6
        for key, value in IDENTITY.items():
            assert row[key] == value


def test_extra_reaches_the_answer_record(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli.chat_core, "complete", lambda config, messages, caps=None: _result("ответ")
    )
    agent = cli.build_agent(_aux_config(), None)
    run = cli.run_mode(
        agent, "вопрос", mode="off", command="ignored", week=6, extra=IDENTITY, question_id=4
    )
    assert run.result is not None
    (row,) = _journal(tmp_path)
    assert row["week"] == 6
    assert row["stage"] == "answer"
    assert row["command"] == "ragbench"
    assert row["backend"] == "local"
    assert row["run"] == 2


def test_without_extra_records_keep_the_old_shape(monkeypatch, tmp_path):
    db, _ = _setup(monkeypatch, tmp_path, [_result("xyz xyz"), _result(_scores())])
    rag.make_retriever(db, aux_config=_aux_config())("xyz", FULL)
    rows = _journal(tmp_path)
    assert all("stage" not in r for r in rows)
    assert all(r["week"] == rag.RAG_WEEK for r in rows)
    assert [r["command"] for r in rows] == ["rag_rewrite", "rag", "rag_rerank"]


def test_on_call_receives_every_completed_call_in_order(monkeypatch, tmp_path):
    db, _ = _setup(
        monkeypatch,
        tmp_path,
        [_result("xyz xyz", prompt=11, completion=2), _result(_scores(), prompt=40, completion=9)],
    )
    entries: list[LedgerEntry] = []
    rag.make_retriever(db, aux_config=_aux_config(), on_call=entries.append)("xyz", FULL)
    assert [e.stage for e in entries] == ["rewrite", "embed", "search", "rerank"]
    rewrite, embed, search, rerank = entries
    assert (search.prompt_tokens, search.completion_tokens) == (None, None)
    assert (rewrite.model, rewrite.endpoint) == ("ornith", "local")
    assert (rewrite.prompt_tokens, rewrite.completion_tokens, rewrite.latency_ms) == (11, 2, 7)
    assert (embed.model, embed.endpoint) == (NOMIC, "local")
    assert embed.completion_tokens is None
    assert (rerank.prompt_tokens, rerank.completion_tokens) == (40, 9)


def test_unknown_usage_stays_none_not_zero(monkeypatch, tmp_path):
    rewrite = _result("xyz xyz")
    rewrite.usage = Usage()
    db, _ = _setup(monkeypatch, tmp_path, [rewrite, _result(_scores())])
    entries: list[LedgerEntry] = []
    rag.make_retriever(db, aux_config=_aux_config(), on_call=entries.append)("xyz", FULL)
    assert entries[0].stage == "rewrite"
    assert entries[0].prompt_tokens is None
    assert entries[0].completion_tokens is None


def test_completed_rewrite_is_reported_when_the_rerank_then_fails(monkeypatch, tmp_path):
    db, _ = _setup(monkeypatch, tmp_path, [])
    entries: list[LedgerEntry] = []
    retrieve = rag.make_retriever(db, aux_config=_aux_config(), on_call=entries.append)
    replies = [_result("xyz xyz"), _result("garbage"), _result("garbage again")]
    monkeypatch.setattr(rag.chat_core, "complete", lambda c, m: replies.pop(0))
    with pytest.raises(AdventError):
        retrieve("xyz", FULL)
    assert [e.stage for e in entries] == ["rewrite", "embed", "search", "rerank", "rerank"]


def test_local_rerank_retry_attempt_is_reported_and_succeeds(monkeypatch, tmp_path):
    db, seen = _setup(
        monkeypatch,
        tmp_path,
        [_result("xyz xyz"), _result("prose, no json", latency=30), _result(_scores(), latency=5)],
    )
    entries: list[LedgerEntry] = []
    ctx = rag.make_retriever(db, aux_config=_aux_config(), on_call=entries.append)("xyz", FULL)
    assert [e.stage for e in entries] == ["rewrite", "embed", "search", "rerank", "rerank"]
    assert [e.latency_ms for e in entries if e.stage == "rerank"] == [30, 5]
    assert len(seen) == 3
    assert ctx.passed == 3


def test_failed_rewrite_call_is_not_reported_as_completed(monkeypatch, tmp_path):
    db, _ = _setup(monkeypatch, tmp_path, [AdventError("boom")])
    entries: list[LedgerEntry] = []
    retrieve = rag.make_retriever(db, aux_config=_aux_config(), on_call=entries.append)
    with pytest.raises(AdventError):
        retrieve("xyz", FULL)
    assert entries == []


def test_aux_calls_go_to_the_aux_config_model_and_base_url(monkeypatch, tmp_path):
    db, seen = _setup(monkeypatch, tmp_path, [_result("xyz xyz"), _result(_scores())])
    aux = Config.resolve(offline=True, base_url=LOCAL_URL, model="my-local-model")
    rag.make_retriever(db, aux_config=aux)("xyz", FULL)
    assert [c.model for c in seen] == ["my-local-model", "my-local-model"]
    assert all(c.base_url == LOCAL_URL for c in seen)
    assert rag.RAG_AUX_MODEL not in {c.model for c in seen}


def test_week_reaches_the_aux_and_embed_records(monkeypatch, tmp_path):
    db, _ = _setup(monkeypatch, tmp_path, [_result("xyz xyz"), _result(_scores())])
    rag.make_retriever(db, aux_config=_aux_config(), week=6)("xyz", FULL)
    assert {r["week"] for r in _journal(tmp_path)} == {6}


def test_aux_call_uses_its_own_week_not_the_constant(monkeypatch):
    seen: list = []
    monkeypatch.setattr(rag.chat_core, "complete", lambda c, m: _result("x"))
    monkeypatch.setattr(rag, "log_call", lambda *a, **kw: seen.append(kw))
    rag._aux_call(
        "p",
        command="rag_rewrite",
        max_tokens=10,
        json_mode=False,
        day=23,
        aux_config=_aux_config(),
        week=6,
        extra={"backend": "local"},
    )
    assert seen[0]["week"] == 6
    assert seen[0]["extra"]["stage"] == "rewrite"
    assert seen[0]["extra"]["backend"] == "local"


def test_build_agent_propagates_aux_config_week_extra_and_on_call(monkeypatch):
    got: dict = {}

    def fake_make(db_path=None, **kwargs):
        got.update(kwargs, db_path=db_path)
        return lambda question, settings: None

    monkeypatch.setattr(cli.rag_module, "make_retriever", fake_make)
    aux = _aux_config()
    sink = [].append
    cli.build_agent(aux, None, 28, aux_config=aux, week=6, extra=IDENTITY, on_call=sink)
    assert got["aux_config"] is aux
    assert got["week"] == 6
    assert got["extra"] == IDENTITY
    assert got["on_call"] is sink
    assert got["day"] == 28 and got["aux_day"] == 28


def test_build_agent_passes_nothing_extra_by_default(monkeypatch):
    got: dict = {}

    def fake_make(db_path=None, **kwargs):
        got.update(kwargs)
        return lambda question, settings: None

    monkeypatch.setattr(cli.rag_module, "make_retriever", fake_make)
    cli.build_agent(_aux_config(), None)
    assert set(got) == {"day", "aux_day"}


def test_run_mode_reports_the_answer_call_to_on_call(monkeypatch):
    monkeypatch.setattr(
        cli.chat_core,
        "complete",
        lambda config, messages, caps=None: _result(
            "ответ", prompt=100, completion=20, latency=900
        ),
    )
    entries: list[LedgerEntry] = []
    agent = cli.build_agent(_aux_config(), None)
    run = cli.run_mode(agent, "вопрос", mode="off", command="x", on_call=entries.append)
    (entry,) = entries
    assert entry.stage == "answer"
    assert (entry.model, entry.endpoint) == ("ornith", "local")
    assert (entry.prompt_tokens, entry.completion_tokens, entry.latency_ms) == (100, 20, 900)
    assert entry.result is run.result
