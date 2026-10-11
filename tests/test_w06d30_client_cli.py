"""`adventlocal service ...`: stdout/stderr contract, stress table at width 80, overflow, keygen.

The gateway is a MockTransport double injected through `cli._make_client`; nothing here
touches a network or the real journal.
"""

from __future__ import annotations

import hashlib
import io
import json
import zlib

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

import week_06.cli as cli
from advent_core import console, journal, offline
from week_06 import service_client as svc

ORIGIN = "https://advent-llm.duckdns.org"
KEY = "adv_" + "Aa1" * 14
DEMO_KEY = "adv_" + "Bb2" * 14
runner = CliRunner()

# literal copy of the gateway's /v1/status body (week_06/gateway/app.py: status)
STATUS = {
    "version": "abc123",
    "model": "ready",
    "model_id": "qwen2.5-0.5b-instruct",
    "rss_mb": 601.2,
    "model_uptime_s": 42.5,
    "idle_unload_in_s": 17.5,
    "gateway_uptime_s": 3600.0,
    "last_exit_code": None,
    "queue": {"active": 0, "waiting": 0, "max": 2},
    "limits": {
        "n_ctx": 2048,
        "max_tokens_default": 256,
        "max_tokens_ceiling": 512,
        "body_max_bytes": 65536,
        "idle_seconds": 60,
        "global_per_minute": 120,
    },
    "key": {"name": "owner", "rpm": 30, "burst": 6, "tokens_left": 6.0},
    "counters": {"200": 5, "401": 1, "413": 0, "429": 2, "503": 0, "5xx": 0, "other": 0},
    "outcomes": {"ok": 5, "client_abort": 1},
}
HEADERS = {
    "content-type": "text/event-stream",
    "x-advent-queue-wait-ms": "340",
    "x-advent-cold-start-ms": "2100",
}


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def sse(obj) -> bytes:
    data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


def answer(text: str, *, finish=True) -> httpx.Response:
    chunks = [sse({"choices": [{"delta": {"content": text}}]})]
    if finish:
        chunks += [
            sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            sse({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}}),
            sse("[DONE]"),
        ]
    return httpx.Response(200, headers=HEADERS, stream=Chunks(chunks))


def refusal(status: int, kind: str, **extra) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": "отказ", "type": kind, **extra}})


class Gateway:
    """Records requests and answers chat calls by a replaceable policy."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status = dict(STATUS)
        self.status_seq: list[dict] = []
        self.chat_policy = lambda n, body: answer("Привет, мир")
        self.chats = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/":
            return httpx.Response(200, text="<html></html>")
        if request.url.path == "/v1/status":
            if len(self.status_seq) > 1:
                return httpx.Response(200, json=self.status_seq.pop(0))
            return httpx.Response(200, json=self.status_seq[0] if self.status_seq else self.status)
        self.chats += 1
        return self.chat_policy(self.chats, json.loads(request.content))

    def chat_bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]


@pytest.fixture
def gw(monkeypatch):
    monkeypatch.setenv("ADVENT_PRIVATE_ORIGINS", ORIGIN)
    monkeypatch.setenv("ADVENT_SERVICE_URL", ORIGIN)
    monkeypatch.setenv("ADVENT_SERVICE_KEY", KEY)
    monkeypatch.setenv("ADVENT_SERVICE_DEMO_KEY", DEMO_KEY)
    monkeypatch.setattr(svc, "load_env", lambda: None)
    gateway = Gateway()
    monkeypatch.setattr(
        cli,
        "_make_client",
        lambda target: svc.ServiceClient(target, transport=httpx.MockTransport(gateway), timeout=5),
    )
    monkeypatch.setattr(cli, "STATUS_POLL_SECONDS", 0.0)
    return gateway


@pytest.fixture
def wide(monkeypatch):
    """console.out rendered at width 80 into a buffer."""
    buf = io.StringIO()
    monkeypatch.setattr(
        console, "out", Console(file=buf, width=80, color_system=None, soft_wrap=True)
    )
    return buf


def journal_rows() -> list[dict]:
    path = journal.LOG_DIR / "calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- chat ---------------------------------------------------------------------------------


def test_one_shot_chat_answer_on_stdout_everything_else_on_stderr(gw):
    result = runner.invoke(cli.app, ["service", "chat", "привет"])
    assert result.exit_code == 0
    assert result.stdout == "Привет, мир\n"
    for part in ("cold start 2.1 с", "очередь 340 мс", "tokens 12/5"):
        assert part in result.stderr
    assert "finish" not in result.stderr
    assert "Привет, мир" not in result.stderr
    assert gw.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert gw.chat_bodies()[0]["max_tokens"] == 256


def test_repl_keeps_history_and_status_goes_to_stderr(gw):
    result = runner.invoke(cli.app, ["service", "chat"], input="один\n/status\n/new\nдва\n/exit\n")
    assert result.exit_code == 0
    assert result.stdout == "Привет, мир\nПривет, мир\n"
    assert "Сервис" in result.stderr and "загружена" in result.stderr
    assert "загружена" not in result.stdout
    assert [m["role"] for m in gw.chat_bodies()[1]["messages"]] == ["user"]  # after /new


def test_repl_second_turn_sends_the_first_exchange(gw):
    runner.invoke(cli.app, ["service", "chat"], input="один\nдва\n")
    roles = [m["role"] for m in gw.chat_bodies()[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_repl_survives_a_refusal(gw):
    gw.chat_policy = lambda n, body: (
        refusal(429, "rate_limited") if n == 1 else answer("после отказа")
    )
    result = runner.invoke(cli.app, ["service", "chat"], input="a\nb\n")
    assert result.exit_code == 0
    assert result.stdout == "после отказа\n"
    assert "429" in result.stderr


def test_incomplete_answer_is_flagged_and_dropped_from_history(gw):
    gw.chat_policy = lambda n, body: answer("обрезано", finish=False) if n == 1 else answer("ok")
    result = runner.invoke(cli.app, ["service", "chat"], input="a\nb\n")
    assert "обрыв: stream_cut" in result.stderr
    assert [m["role"] for m in gw.chat_bodies()[1]["messages"]] == ["user"]


def test_chat_writes_a_private_journal_row(gw):
    runner.invoke(cli.app, ["service", "chat", "привет"])
    (row,) = journal_rows()
    assert row["endpoint"] == "private"
    assert (row["outcome"], row["http_status"], row["cold_start_ms"]) == ("complete", 200, 2100)
    assert KEY not in json.dumps(row)


def test_failed_call_is_journaled_as_error(gw):
    gw.chat_policy = lambda n, body: refusal(503, "queue_full")
    result = runner.invoke(cli.app, ["service", "chat", "привет"])
    assert result.exit_code != 0 and isinstance(result.exception, svc.ServiceError)
    (row,) = journal_rows()
    assert (row["outcome"], row["http_status"], row["error_type"]) == ("error", 503, "queue_full")


def test_chat_under_the_offline_guard_counts_a_private_attempt(gw):
    offline.enable()
    result = runner.invoke(cli.app, ["service", "chat", "привет"])
    assert result.exit_code == 0
    assert offline.counters().attempted["private"] == 1
    assert offline.counters().blocked["cloud"] == 0
    assert journal_rows()[0]["endpoint"] == "private"


# --- key handling -------------------------------------------------------------------------


def test_there_is_no_flag_that_takes_a_key_value(gw):
    result = runner.invoke(cli.app, ["service", "status", "--key", KEY])
    assert result.exit_code == 2
    assert gw.requests == []


def test_key_env_rejects_a_pasted_key_without_echoing_it(gw):
    result = runner.invoke(cli.app, ["service", "status", "--key-env", KEY])
    assert result.exit_code != 0
    assert KEY not in result.output and KEY not in str(result.exception)
    assert gw.requests == []


def test_unlisted_service_url_never_sees_the_key(gw):
    result = runner.invoke(cli.app, ["service", "status", "--url", "https://example.org"])
    assert result.exit_code != 0
    assert gw.requests == []


def test_keygen_prints_key_and_matching_hash_on_stdout(gw):
    result = runner.invoke(cli.app, ["service", "keygen"])
    key_line, hash_line = result.stdout.splitlines()
    key = key_line.removeprefix("ключ: ")
    assert key.startswith("adv_") and len(key) == 47
    assert hash_line == f"sha256: {hashlib.sha256(key.encode()).hexdigest()}"
    assert key not in result.stderr


# --- status -------------------------------------------------------------------------------


def test_status_table_on_stdout_nothing_on_stderr(gw, wide):
    result = runner.invoke(cli.app, ["service", "status"])
    assert result.exit_code == 0 and result.stderr == ""
    out = wide.getvalue()
    assert "…" not in out
    for line in (
        "загружена",
        "qwen2.5-0.5b-instruct",
        "rpm 30 · burst 6 · в bucket сейчас 6",
        "ждут 0 · активных 0 · максимум ожидающих 2",
        "контекст",
        "2048",
        "65536",
        "до выгрузки (idle)",
        "17.5 с",
        "42.5 с",
        "owner",
    ):
        assert out.count(line) == 1, line
    # no raw field dump: neither dotted names nor values shown twice
    for raw in ("key.", "counters.", "limits.", "tokens_left", "model_uptime_s", "n_ctx"):
        assert raw not in out, raw
    assert out.count("rpm") == 1 and out.count("burst") == 1


def test_wait_stopped_polls_until_stopped(gw, wide):
    ready = {**STATUS}
    stopped = {**STATUS, "model": "stopped"}
    gw.status_seq = [ready, ready, stopped]
    result = runner.invoke(cli.app, ["service", "status", "--wait-stopped", "30"])
    assert result.exit_code == 0
    assert result.stderr.count("жду выгрузки модели: сейчас ready") == 2
    assert "выгружена" in wide.getvalue()


def test_wait_stopped_gives_up_with_exit_1(gw, wide):
    result = runner.invoke(cli.app, ["service", "status", "--wait-stopped", "0"])
    assert result.exit_code == 1
    assert "модель не остановилась за 0 с (состояние: ready)" in result.stderr


# --- stress -------------------------------------------------------------------------------


def test_stress_table_counts_and_barrier(gw, wide):
    gw.chat_policy = lambda n, body: (
        answer("длинный ответ") if n <= 3 else refusal(503, "queue_full")
    )
    result = runner.invoke(cli.app, ["service", "stress", "-n", "6", "--settle", "0"])
    assert result.exit_code == 0
    out = wide.getvalue()
    assert "…" not in out
    assert out.count("Итого: 200: 3, 503: 3") == 1
    assert out.count("queue_full") == 3 and out.count("stop") == 3
    assert out.count("wall успешных: p50") == 1
    assert out.count("предусловия: очередь пуста, bucket полон") == 1
    assert result.stdout == ""  # the tables went to the patched console.out
    kinds = [(r.method, r.url.path) for r in gw.requests]
    expected = [("GET", "/v1/status")] + [("GET", "/")] * 6
    expected += [("POST", "/v1/chat/completions")] * 6
    assert kinds == expected
    assert gw.requests[1].headers.get("authorization") is None
    assert "открыто соединений: 6/6" in result.stderr
    assert [b["max_tokens"] for b in gw.chat_bodies()] == [128] * 6


def test_stress_with_demo_key_shows_429(gw, wide):
    gw.chat_policy = lambda n, body: answer("ok") if n <= 2 else refusal(429, "rate_limited")
    args = ["service", "stress", "-n", "4", "--key-env", "ADVENT_SERVICE_DEMO_KEY", "--settle", "0"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    assert wide.getvalue().count("Итого: 200: 2, 429: 2") == 1
    sent = {r.headers["authorization"] for r in gw.requests if r.url.path != "/"}
    assert sent == {f"Bearer {DEMO_KEY}"}


def test_stress_reports_unmet_preconditions_honestly(gw, wide):
    gw.status = {**STATUS, "queue": {"active": 1, "waiting": 1, "max": 2}}
    result = runner.invoke(cli.app, ["service", "stress", "-n", "2", "--settle", "0"])
    assert "предусловия: не выполнены (очередь не пуста)" in result.stderr
    assert wide.getvalue().count("предусловия: не выполнены (очередь не пуста)") == 1


def test_stress_counts_incomplete_answers_separately(gw, wide):
    gw.chat_policy = lambda n, body: answer("x") if n == 1 else answer("y", finish=False)
    runner.invoke(cli.app, ["service", "stress", "-n", "2", "--settle", "0"])
    out = wide.getvalue()
    assert out.count("Итого: 200: 1, обрыв: 1") == 1
    assert out.count("обрыв: stream_cut") == 1


def test_percentiles_are_nearest_rank():
    assert cli._percentile([], 0.5) is None
    assert cli._percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert cli._percentile(list(range(1, 11)), 0.9) == 9


# --- overflow -----------------------------------------------------------------------------


def test_overflow_text_is_varied_and_under_the_body_limit():
    text = cli.overflow_text()
    assert 14_000 <= len(text) < 15_000
    body = json.dumps({"messages": [{"role": "user", "content": text}]}).encode()
    assert len(body) < 65_536
    assert len(zlib.compress(text.encode())) / len(text.encode()) > 0.2  # monotone text would not


def test_overflow_prints_numbers_from_a_verified_413(gw, wide):
    gw.chat_policy = lambda n, body: refusal(
        413, "context_length_exceeded", input_tokens=3100, max_tokens=16, n_ctx=2048
    )
    result = runner.invoke(cli.app, ["service", "overflow"])
    assert result.exit_code == 0
    out = wide.getvalue()
    assert out.count("413 context_length_exceeded") == 1
    assert "3116 > 2048" in out and "…" not in out
    assert gw.chat_bodies()[0]["stream"] is False


def test_overflow_refuses_to_print_numbers_for_the_wrong_413(gw, wide):
    gw.chat_policy = lambda n, body: refusal(413, "body_too_large")
    result = runner.invoke(cli.app, ["service", "overflow"])
    assert result.exit_code == 1
    assert "ожидался 413 context_length_exceeded, пришло 413 body_too_large" in (
        result.exception.message
    )
    assert wide.getvalue() == ""


def test_overflow_fails_when_the_service_accepts_the_input(gw, wide):
    gw.chat_policy = lambda n, body: httpx.Response(
        200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    )
    result = runner.invoke(cli.app, ["service", "overflow"])
    assert result.exit_code == 1 and wide.getvalue() == ""


def _footer_text(monkeypatch, **kw) -> str:
    buf = io.StringIO()
    monkeypatch.setattr(
        console, "err", Console(file=buf, width=200, color_system=None, soft_wrap=True)
    )
    res = svc.ServiceResult(
        usage=svc.Usage(prompt_tokens=43, completion_tokens=112),
        cold_start_ms=2400,
        queue_wait_ms=0,
        wall_ms=14000.0,
        ttft_ms=3500.0,
        **kw,
    )
    cli._chat_footer(res)
    return buf.getvalue()


def test_footer_one_line_without_finish_on_stop(monkeypatch):
    out = _footer_text(monkeypatch, finish_reason="stop")
    assert out == (
        "· cold start 2.4 с  ·  очередь 0 мс  ·  14.0 с  ·  tokens 43/112  ·  tok/s 10.6\n"
    )


def test_footer_length_appends_truncation_note(monkeypatch):
    out = _footer_text(monkeypatch, finish_reason="length")
    assert out == (
        "· cold start 2.4 с  ·  очередь 0 мс  ·  14.0 с  ·  tokens 43/112  ·  tok/s 10.6"
        "  ·  обрезано по max_tokens\n"
    )


@pytest.mark.parametrize(
    ("state", "text"),
    [
        ("stopped", "выгружена"),
        ("starting", "загружается"),
        ("ready", "загружена"),
        ("failed", "ошибка"),
        ("weird", "неизвестно (weird)"),
    ],
)
def test_model_state_in_russian(state, text):
    assert cli._model_state(state) == text


def test_counters_row_single_line_at_width_80(gw, wide):
    gw.status = {
        **STATUS,
        "counters": {"200": 15, "401": 4, "413": 0, "429": 0, "503": 0, "5xx": 0, "other": 5},
    }
    result = runner.invoke(cli.app, ["service", "status"])
    assert result.exit_code == 0
    out = wide.getvalue()
    assert "…" not in out
    lines = [ln for ln in out.splitlines() if "ответы по кодам" in ln]
    assert len(lines) == 1
    assert "200 15 · 401 4 · прочие (не 5xx, напр. 499) 5" in lines[0]
    assert "413 0" not in out and "5xx 0" not in out
