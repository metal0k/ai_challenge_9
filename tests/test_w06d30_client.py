"""week_06/service_client.py: origin gate, real Authorization/URL/body at the transport seam,
SSE parser edge cases, gateway error mapping. No network: httpx.MockTransport only."""

from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import pytest

from advent_core.config import ConfigError, redact
from advent_core.errors import NetworkError
from week_06 import service_client as svc

ORIGIN = "https://advent-llm.duckdns.org"
KEY = "adv_" + "Aa1" * 14
DEMO_KEY = "adv_" + "Bb2" * 14


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ADVENT_PRIVATE_ORIGINS", ORIGIN)
    monkeypatch.setenv("ADVENT_SERVICE_URL", ORIGIN)
    monkeypatch.setenv("ADVENT_SERVICE_KEY", KEY)
    monkeypatch.setenv("ADVENT_SERVICE_DEMO_KEY", DEMO_KEY)
    monkeypatch.setattr(svc, "load_env", lambda: None)


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], fail: Exception | None = None) -> None:
        self.chunks, self.fail = chunks, fail

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise self.fail


def sse(obj: object) -> bytes:
    data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


def delta(text: str) -> dict:
    return {"model": "qwen", "choices": [{"index": 0, "delta": {"content": text}}]}


FINISH = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
USAGE = {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}}
HEADERS = {
    "content-type": "text/event-stream",
    "x-advent-queue-wait-ms": "340",
    "x-advent-cold-start-ms": "2100",
    "x-advent-input-tokens": "12",
}


def make_client(handler) -> svc.ServiceClient:
    return svc.ServiceClient(
        svc.make_target(ORIGIN, KEY), transport=httpx.MockTransport(handler), timeout=5
    )


def stream_response(chunks, fail=None, status=200, headers=None) -> httpx.Response:
    return httpx.Response(status, headers=headers or HEADERS, stream=ChunkStream(chunks, fail))


def chat(handler, **kw) -> svc.ServiceResult:
    async def go():
        async with make_client(handler) as client:
            return await client.chat([{"role": "user", "content": "привет"}], **kw)

    return asyncio.run(go())


# --- origin gate -------------------------------------------------------------------------


def test_request_carries_real_bearer_url_and_body():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return stream_response([sse(delta("ок")), sse(FINISH), sse(USAGE), sse("[DONE]")])

    chat(handler)
    (request,) = seen
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert str(request.url) == "https://advent-llm.duckdns.org/v1/chat/completions"
    assert json.loads(request.content) == {
        "messages": [{"role": "user", "content": "привет"}],
        "max_tokens": 256,
        "stream": True,
    }


def test_unlisted_origin_is_refused_without_leaking_the_key():
    with pytest.raises(ConfigError) as info:
        svc.make_target("https://example.org", KEY)
    assert "ADVENT_PRIVATE_ORIGINS" in str(info.value)
    assert KEY not in str(info.value)


def test_mistral_host_refused_even_when_listed(monkeypatch):
    monkeypatch.setenv("ADVENT_PRIVATE_ORIGINS", f"https://api.mistral.ai,{ORIGIN}")
    with pytest.raises(ConfigError):
        svc.make_target("https://api.mistral.ai", KEY)


@pytest.mark.parametrize(
    "url",
    [
        "http://advent-llm.duckdns.org",
        "https://user:pw@advent-llm.duckdns.org",
        "https://advent-llm.duckdns.org:8443",
        "https://advent-llm.duckdns.org/x",
        "https://advent-llm.duckdns.org?a=1",
        "ftp://advent-llm.duckdns.org",
        "https://advent-llm.duckdns.org:bad",
    ],
)
def test_malformed_or_different_origin_is_rejected(url):
    with pytest.raises(ConfigError) as info:
        svc.make_target(url, KEY)
    assert "pw" not in str(info.value)


@pytest.mark.parametrize(
    "url",
    [ORIGIN, ORIGIN + "/", "https://ADVENT-LLM.duckdns.org:443", "https://advent-llm.duckdns.org."],
)
def test_equivalent_spellings_of_the_listed_origin_pass(url):
    assert svc.make_target(url, KEY).origin == ORIGIN


def test_missing_url_and_empty_key_are_config_errors():
    with pytest.raises(ConfigError):
        svc.make_target(None, KEY)
    with pytest.raises(ConfigError):
        svc.make_target(ORIGIN, "")


def test_key_env_takes_a_name_never_a_value():
    assert svc.resolve_target(None, "ADVENT_SERVICE_DEMO_KEY").key == DEMO_KEY
    for bad in (KEY, "has space", "1abc", ""):
        with pytest.raises(ConfigError) as info:
            svc.resolve_target(None, bad)
        assert bad == "" or bad not in str(info.value)


def test_empty_key_variable_names_the_variable(monkeypatch):
    monkeypatch.delenv("ADVENT_SERVICE_KEY")
    with pytest.raises(ConfigError) as info:
        svc.resolve_target()
    assert "ADVENT_SERVICE_KEY" in str(info.value)


def test_generated_key_shape_and_hash():
    key, digest = svc.generate_key()
    assert key.startswith("adv_") and len(key) == 4 + 43
    assert digest == hashlib.sha256(key.encode()).hexdigest()


def test_redact_masks_both_service_keys():
    assert redact(f"a {KEY} b {DEMO_KEY} c") == "a *** b *** c"


def test_redirect_is_not_followed_and_key_stays_home():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"location": "https://evil.example/"})

    with pytest.raises(svc.ServiceError) as info:
        chat(handler)
    assert info.value.status == 302
    assert [str(r.url.host) for r in seen] == ["advent-llm.duckdns.org"]


# --- SSE parser ---------------------------------------------------------------------------


def test_parser_keeps_a_utf8_character_torn_across_chunks():
    raw = sse(delta("привет"))
    cut = raw.index("р".encode()) + 1  # inside the two-byte "р"
    parser = svc.SSEParser()
    assert parser.feed(raw[:cut]) == []
    (event,) = parser.feed(raw[cut:])
    assert json.loads(event.data)["choices"][0]["delta"]["content"] == "привет"


def test_parser_keeps_a_torn_data_line_and_splits_several_events():
    parser = svc.SSEParser()
    assert parser.feed(b"da") == []
    assert parser.feed(b"ta: one\n") == []
    events = parser.feed(b"\ndata: two\n\ndata: thr")
    assert [e.data for e in events] == ["one", "two"]
    assert [e.data for e in parser.feed(b"ee\n\n")] == ["three"]


def test_parser_handles_crlf_split_between_chunks_comments_and_event_names():
    parser = svc.SSEParser()
    out = parser.feed(b": keep-alive\r\nevent: error\r\ndata: x\r")
    out += parser.feed(b"\n\r\n")
    assert [(e.event, e.data) for e in out] == [("error", "x")]


def test_parser_reports_a_trailing_incomplete_event_on_close():
    parser = svc.SSEParser()
    assert parser.feed(b"data: {}\n\ndata: half") == [svc.SSEEvent(None, "{}")]
    parser.close()
    assert parser.pending is True


# --- streaming ----------------------------------------------------------------------------


def test_full_stream_with_usage_only_chunk_and_done():
    def handler(request):
        return stream_response(
            [sse(delta("при")), sse(delta("вет")), sse(FINISH), sse(USAGE), sse("[DONE]")]
        )

    got: list[str] = []
    res = chat(handler, on_content=got.append)
    assert (res.text, got) == ("привет", ["при", "вет"])
    assert res.finish_reason == "stop" and res.incomplete is False
    assert (res.usage.prompt_tokens, res.usage.completion_tokens) == (12, 5)
    assert (res.queue_wait_ms, res.cold_start_ms, res.input_tokens) == (340, 2100, 12)
    assert res.model == "qwen"


def test_absent_headers_are_unknown_but_zero_is_a_value():
    def handler(request):
        return stream_response(
            [sse(delta("a")), sse(FINISH), sse("[DONE]")],
            headers={"x-advent-cold-start-ms": "0"},
        )

    res = chat(handler)
    assert res.cold_start_ms == 0
    assert res.queue_wait_ms is None


def test_several_events_in_one_network_chunk():
    blob = sse(delta("a")) + sse(delta("b")) + sse(FINISH) + sse("[DONE]")
    res = chat(lambda r: stream_response([blob]))
    assert res.text == "ab" and not res.incomplete


def test_premature_eof_marks_the_answer_incomplete():
    res = chat(lambda r: stream_response([sse(delta("часть"))]))
    assert res.text == "часть"
    assert (res.incomplete, res.incomplete_reason) == (True, "stream_cut")


def test_error_event_after_headers_is_an_incomplete_answer_with_its_type():
    err = {"error": {"message": "модель упала", "type": "upstream_failed"}}
    res = chat(lambda r: stream_response([sse(delta("часть")), sse(err)]))
    assert res.status == 200 and res.text == "часть"
    assert (res.incomplete, res.incomplete_reason) == (True, "upstream_failed")


def test_connection_drop_mid_stream_is_incomplete_not_an_exception():
    res = chat(lambda r: stream_response([sse(delta("x"))], fail=httpx.ReadError("boom")))
    assert res.incomplete and res.incomplete_reason == "network: ReadError"


def test_done_without_finish_reason_is_flagged():
    res = chat(lambda r: stream_response([sse(delta("x")), sse("[DONE]")]))
    assert res.incomplete_reason == "no_finish_reason"


def test_non_stream_json_answer():
    body = {
        "model": "qwen",
        "choices": [{"message": {"content": "ответ"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    res = chat(lambda r: httpx.Response(200, json=body, headers=HEADERS), stream=False)
    assert (res.text, res.finish_reason, res.usage.completion_tokens) == ("ответ", "length", 4)
    assert res.stream is False


def test_connection_failure_before_headers_is_a_network_error():
    def handler(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(NetworkError) as info:
        chat(handler)
    assert KEY not in info.value.message


# --- errors -------------------------------------------------------------------------------


def err_body(**fields) -> dict:
    return {"error": {"message": "текст", **fields}}


@pytest.mark.parametrize(
    ("status", "exit_code"), [(401, 3), (429, 4), (503, 5), (502, 5), (413, 1), (400, 1)]
)
def test_status_maps_to_exit_code(status, exit_code):
    with pytest.raises(svc.ServiceError) as info:
        chat(lambda r: httpx.Response(status, json=err_body(type="t")))
    assert (info.value.status, info.value.exit_code) == (status, exit_code)


def test_429_carries_retry_after():
    headers = {"retry-after": "3"}
    with pytest.raises(svc.ServiceError) as info:
        chat(lambda r: httpx.Response(429, json=err_body(type="rate_limited"), headers=headers))
    assert info.value.retry_after == 3.0
    assert "3" in (info.value.hint or "")


def test_context_error_exposes_the_three_numbers():
    body = err_body(type="context_length_exceeded", input_tokens=3100, max_tokens=16, n_ctx=2048)
    with pytest.raises(svc.ServiceError) as info:
        chat(lambda r: httpx.Response(413, json=body), stream=False)
    assert info.value.error_type == "context_length_exceeded"
    assert info.value.numbers == {"input_tokens": 3100, "max_tokens": 16, "n_ctx": 2048}


def test_gateway_error_id_comes_from_code_not_from_the_generic_type():
    body = err_body(
        type="invalid_request_error",
        code="context_length_exceeded",
        input_tokens=3100,
        max_tokens=16,
        n_ctx=2048,
    )
    with pytest.raises(svc.ServiceError) as info:
        chat(lambda r: httpx.Response(413, json=body), stream=False)
    assert info.value.error_type == "context_length_exceeded"


def test_status_reads_the_gateways_real_key_names():
    data = {
        "model": "ready",
        "model_uptime_s": 12.5,
        "queue": {"active": 0, "waiting": 0, "max": 2},
        "key": {"name": "owner", "rpm": 6000.0, "burst": 100, "tokens_left": 100.0},
    }
    st = svc.ServiceStatus.from_json(data)
    assert (st.uptime_s, st.bucket_tokens, st.bucket_full) == (12.5, 100.0, True)


def test_body_too_large_has_no_numbers():
    with pytest.raises(svc.ServiceError) as info:
        chat(lambda r: httpx.Response(413, json=err_body(type="body_too_large")))
    assert info.value.numbers == {}


def test_non_json_error_body_and_echoed_key_are_handled():
    with pytest.raises(svc.ServiceError) as info:
        chat(lambda r: httpx.Response(502, text="<html>Bad gateway</html>"))
    assert "Bad gateway" in info.value.message
    with pytest.raises(svc.ServiceError) as info:
        chat(lambda r: httpx.Response(401, json=err_body(message=f"плохой ключ {KEY}")))
    assert KEY not in info.value.message


# --- status and warm ----------------------------------------------------------------------

STATUS = {  # real gateway shape: week_06/gateway/app.py status()
    "version": "abc123",
    "model": "ready",
    "model_id": "qwen2.5-0.5b-instruct",
    "rss_mb": 601.2,
    "model_uptime_s": 42.5,
    "idle_unload_in_s": 40,
    "queue": {"active": 1, "waiting": 2, "max": 2},
    "limits": {"n_ctx": 2048, "idle_seconds": 60},
    "key": {"name": "owner", "rpm": 30, "burst": 6, "tokens_left": 6.0},
    "counters": {"200": 5},
}


def test_status_is_fetched_with_auth_and_parsed():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=STATUS)

    async def go():
        async with make_client(handler) as client:
            return await client.status()

    st = asyncio.run(go())
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"
    assert str(seen[0].url) == f"{ORIGIN}/v1/status"
    assert (st.state, st.queue_waiting, st.queue_active, st.bucket_full) == ("ready", 2, 1, True)
    assert st.queue_empty is False


def test_unknown_status_shape_is_unknown_not_ok():
    st = svc.ServiceStatus.from_json({"model": "stopped"})
    assert st.state == "stopped"
    assert st.bucket_full is None and st.queue_empty is None


def test_warm_uses_anonymous_requests_only():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text="<html></html>")

    async def go():
        async with make_client(handler) as client:
            return await client.warm(4)

    assert asyncio.run(go()) == 4
    assert len(seen) == 4
    assert all("authorization" not in r.headers and r.url.path == "/" for r in seen)
