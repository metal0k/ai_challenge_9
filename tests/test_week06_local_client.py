"""week_06/local_client.py: SSE contract, HTTP errors, readiness (httpx MockTransport)."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from advent_core import openai_compat as oc
from advent_core.errors import (
    AdventError,
    AuthError,
    ConfigurationError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from week_06 import local_client as lc


def sse(*events, done=True) -> bytes:
    lines = []
    for ev in events:
        data = ev if isinstance(ev, str) else json.dumps(ev)
        lines.append(f"data: {data}\n\n")
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def delta(reasoning=None, content=None, finish=None):
    d = {}
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    if content is not None:
        d["content"] = content
    return {"choices": [{"delta": d, "finish_reason": finish}]}


USAGE = {
    "choices": [],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 64,
        "total_tokens": 84,
        "completion_tokens_details": {"reasoning_tokens": 57},
    },
}


@pytest.fixture
def server(monkeypatch):
    """Install a handler; returns the list of captured requests."""
    seen: list[httpx.Request] = []
    state = {"handler": None}

    def factory(timeout):
        def handler(request):
            seen.append(request)
            return state["handler"](request)

        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(oc, "_make_client", factory)

    def install(handler):
        state["handler"] = handler
        return seen

    return install


def stream_response(body: bytes, status=200):
    return lambda request: httpx.Response(status, content=body)


def run_chat(**kw):
    return lc.chat("http://x", "ornith", [{"role": "user", "content": "hi"}], **kw)


def test_stream_reasoning_then_content_then_usage(server):
    seen = server(
        stream_response(
            sse(
                delta(content=""),
                delta(reasoning="дум"),
                delta(reasoning="аю"),
                delta(content="Канб"),
                delta(content="ерра", finish="stop"),
                USAGE,
            )
        )
    )
    reasoning, content = [], []
    result = run_chat(on_reasoning=reasoning.append, on_content=content.append)
    assert reasoning == ["дум", "аю"] and content == ["Канб", "ерра"]
    assert result.text == "Канберра" and result.reasoning_text == "думаю"
    assert result.usage.completion_tokens == 64 and result.usage.reasoning_tokens == 57
    assert result.finish_reason == "stop" and not result.truncated
    assert result.ttft_ms is not None and result.content_ms is not None
    payload = json.loads(seen[0].content)
    assert payload["stream"] is True and payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 0.6 and payload["top_p"] == 0.95
    assert payload["max_tokens"] == 8192
    assert seen[0].url.path == "/v1/chat/completions"


def test_ttft_is_first_reasoning_not_content(server, monkeypatch):
    """Controlled clock: reasoning at t=2 s, content at t=5 s, end at t=9 s."""
    server(stream_response(sse(delta(reasoning="a"), delta(content="b", finish="stop"))))
    state = {"t": 2.0, "calls": 0}

    def clock():
        state["calls"] += 1
        return 0.0 if state["calls"] == 1 else state["t"]

    monkeypatch.setattr(oc, "_clock", clock)

    def on_reasoning(_):
        state["t"] = 5.0

    def on_content(_):
        state["t"] = 9.0

    result = run_chat(on_reasoning=on_reasoning, on_content=on_content)
    assert result.ttft_ms == 2000.0
    assert result.content_ms == 5000.0
    assert result.latency_ms == 9000.0


def test_missing_usage_is_none_not_zero(server):
    server(stream_response(sse(delta(content="x", finish="stop"))))
    assert run_chat().usage is None


def test_finish_then_eof_without_done_is_complete(server):
    server(stream_response(sse(delta(content="x", finish="stop"), USAGE, done=False)))
    result = run_chat()
    assert not result.truncated and result.usage is not None


def test_eof_without_finish_is_truncated_and_keeps_partial(server):
    server(stream_response(sse(delta(reasoning="r"), delta(content="par"), done=False)))
    result = run_chat()
    assert result.truncated and result.text == "par" and result.cutoff_note
    assert result.finish_reason is None


def test_malformed_json_is_protocol_error(server):
    server(stream_response(sse(delta(content="a"), "{not json", done=False)))
    with pytest.raises(lc.ProtocolError) as info:
        run_chat()
    assert info.value.exit_code == 5


def test_error_event_is_server_error(server):
    server(stream_response(sse({"error": {"message": "Model unloaded."}})))
    with pytest.raises(ServerError, match="Model unloaded"):
        run_chat()


def test_empty_choices_chunk_is_fine(server):
    server(stream_response(sse({"choices": []}, delta(content="a", finish="stop"))))
    assert run_chat().text == "a"


def test_length_with_empty_content_has_honest_note(server):
    server(stream_response(sse(delta(reasoning="долго"), delta(finish="length"), USAGE)))
    result = run_chat()
    assert result.text == "" and result.finish_reason == "length"
    assert result.cutoff_note == "обрыв: reasoning съел max_tokens"


def test_deadline_gives_partial_truncated(server):
    server(stream_response(sse(delta(content="a"), delta(content="b", finish="stop"))))
    result = run_chat(deadline=-1)
    assert result.truncated


class SlowTail(httpx.SyncByteStream):
    """Sends `head`, then stalls for `stall` seconds before the stream ends."""

    def __init__(self, head: bytes, stall: float) -> None:
        self.head, self.stall = head, stall

    def __iter__(self):
        yield self.head
        time.sleep(self.stall)
        yield b""


def test_deadline_cuts_a_stalled_stream_after_partial_text(server):
    head = sse(delta(reasoning="r"), delta(content="par"), done=False)
    server(lambda request: httpx.Response(200, stream=SlowTail(head, 3.0)))
    began = time.monotonic()
    result = run_chat(deadline=0.3)
    assert time.monotonic() - began < 2.0
    assert result.truncated and result.text == "par" and result.reasoning_text == "r"
    assert result.cutoff_note


def test_deadline_after_finish_keeps_complete_result(server):
    head = sse(delta(content="ok", finish="stop"), USAGE, done=False)
    server(lambda request: httpx.Response(200, stream=SlowTail(head, 3.0)))
    began = time.monotonic()
    result = run_chat(deadline=0.3)
    assert time.monotonic() - began < 2.0
    assert not result.truncated and result.text == "ok" and result.usage is not None


def test_deadline_bounds_waiting_for_headers(server):
    def handler(request):
        time.sleep(3.0)
        return httpx.Response(200, content=sse(delta(content="x", finish="stop")))

    server(handler)
    began = time.monotonic()
    result = run_chat(deadline=0.3)
    assert time.monotonic() - began < 2.0
    assert result.truncated and result.text == ""


# --- shape validation (review #3) ------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        {"choices": [1]},
        {"choices": "x"},
        {"choices": [{"delta": 5}]},
        {"choices": [{"delta": {"content": 5}}]},
        {"choices": [{"delta": {"reasoning_content": ["a"]}}]},
        {"choices": [], "usage": "many"},
    ],
)
def test_wrong_shape_chunk_is_protocol_error(server, event):
    server(stream_response(sse(event)))
    with pytest.raises(lc.ProtocolError):
        run_chat()


@pytest.mark.parametrize("body", [{"data": None}, {"data": {"id": "x"}}, [1, 2], "text"])
def test_wrong_shape_models_is_protocol_error(server, body):
    server(lambda r: httpx.Response(200, json=body))
    with pytest.raises(lc.ProtocolError):
        lc.server_status("http://x")


def test_tokens_per_second_uses_ttft_and_guards_denominator():
    from advent_core.telemetry import Usage

    base = dict(text="", reasoning_text="", content_ms=None, finish_reason="stop", model="m")
    ok = lc.LocalResult(usage=Usage(completion_tokens=100), latency_ms=3000, ttft_ms=1000, **base)
    assert ok.tokens_per_second == pytest.approx(50)
    zero = lc.LocalResult(usage=Usage(completion_tokens=100), latency_ms=1000, ttft_ms=1000, **base)
    assert zero.tokens_per_second is None
    unknown = lc.LocalResult(usage=None, latency_ms=3000, ttft_ms=1000, **base)
    assert unknown.tokens_per_second is None
    no_ttft = lc.LocalResult(
        usage=Usage(completion_tokens=1), latency_ms=3000, ttft_ms=None, **base
    )
    assert no_ttft.tokens_per_second is None


# --- HTTP errors in chat ---------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "exc"),
    [
        (404, "nope", ConfigurationError),
        (400, '{"error": "model not loaded"}', ConfigurationError),
        (400, '{"error": "bad param"}', AdventError),
        (401, "", AuthError),
        (403, "", AuthError),
        (429, "", RateLimitError),
        (500, "boom", ServerError),
        (503, "boom", ServerError),
    ],
)
def test_chat_http_errors(server, status, body, exc):
    server(lambda request: httpx.Response(status, content=body.encode()))
    with pytest.raises(exc) as info:
        run_chat()
    assert type(info.value) is exc or (exc is AdventError and type(info.value) is AdventError)
    if exc is ConfigurationError:
        assert "не загружена" in info.value.message


def test_connect_error_is_network_error_with_local_hint(server):
    def refuse(request):
        raise httpx.ConnectError("refused")

    server(refuse)
    with pytest.raises(NetworkError) as info:
        run_chat()
    assert "start-local-llm.ps1 -Context 40960" in info.value.hint


def test_timeout_is_network_error(server):
    def slow(request):
        raise httpx.ReadTimeout("slow")

    server(slow)
    with pytest.raises(NetworkError, match="не ответил вовремя"):
        run_chat()


# --- server_status / readiness ---------------------------------------------

V0 = {
    "data": [
        {"id": "text-embed", "type": "embeddings", "state": "not-loaded"},
        {
            "id": "ornith",
            "type": "llm",
            "state": "loaded",
            "loaded_context_length": 57344,
            "quantization": "Q4_K_M",
            "arch": "qwen35",
        },
    ]
}


def test_server_status_lm_studio_loaded_first(server):
    server(lambda r: httpx.Response(200, json=V0))
    models = lc.server_status("http://x")
    assert [m.id for m in models] == ["ornith", "text-embed"]
    first = models[0]
    assert first.loaded and first.loaded_context_length == 57344
    assert first.quantization == "Q4_K_M" and first.arch == "qwen35"


def test_fallback_to_v1_only_on_404(server):
    def handler(request):
        if request.url.path == "/api/v0/models":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "llama"}]})

    seen = server(handler)
    models = lc.server_status("http://x")
    assert models == [lc.LocalModel(id="llama")] and models[0].state is None
    assert [r.url.path for r in seen] == ["/api/v0/models", "/v1/models"]


@pytest.mark.parametrize("status", [500, 401])
def test_no_fallback_on_other_statuses(server, status):
    seen = server(lambda r: httpx.Response(status))
    with pytest.raises((ServerError, AuthError)):
        lc.server_status("http://x")
    assert len(seen) == 1


def test_status_connect_error(server):
    def refuse(request):
        raise httpx.ConnectError("refused")

    server(refuse)
    with pytest.raises(NetworkError) as info:
        lc.server_status("http://x")
    assert info.value.exit_code == 6


def test_ensure_ready_ok(server):
    server(lambda r: httpx.Response(200, json=V0))
    assert lc.ensure_ready("ornith", "http://x").loaded


def test_ensure_ready_not_loaded_exit_2_with_hint(server):
    server(lambda r: httpx.Response(200, json=V0))
    with pytest.raises(ConfigurationError) as info:
        lc.ensure_ready("text-embed", "http://x")
    assert info.value.exit_code == 2
    assert "-Context 40960" in info.value.hint


def test_ensure_ready_missing_model(server):
    server(lambda r: httpx.Response(200, json=V0))
    with pytest.raises(ConfigurationError, match="нет в списке"):
        lc.ensure_ready("ghost", "http://x")


def test_ensure_ready_fallback_requires_presence_only(server):
    def handler(request):
        if request.url.path == "/api/v0/models":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "llama"}]})

    server(handler)
    assert lc.ensure_ready("llama", "http://x").state is None
    with pytest.raises(ConfigurationError):
        lc.ensure_ready("ornith", "http://x")


def test_default_url_env(monkeypatch):
    monkeypatch.delenv("ADVENT_LOCAL_URL", raising=False)
    assert lc.default_url() == "http://127.0.0.1:1234"
    monkeypatch.setenv("ADVENT_LOCAL_URL", "http://h:9/")
    assert lc.default_url() == "http://h:9"
