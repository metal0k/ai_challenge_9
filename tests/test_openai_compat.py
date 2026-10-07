"""advent_core/openai_compat.py: request bodies, non-stream parsing, embeddings adapter, errors.

The SSE contract itself is pinned by test_week06_local_client.py; this file covers
what the move added (payload pass-through, chat_complete, embed, transport flags).
"""

from __future__ import annotations

import json
import time

import httpx
import numpy as np
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


@pytest.fixture
def server(monkeypatch):
    """Install a handler; returns the list of captured requests."""
    seen: list[httpx.Request] = []
    state = {"handler": None}

    def factory(timeout, transport=None):
        def handler(request):
            seen.append(request)
            return state["handler"](request)

        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(oc, "_make_client", factory)

    def install(handler):
        state["handler"] = handler
        return seen

    return install


def sse(*events, done=True) -> bytes:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def completion(message=None, finish="stop", usage=None, model="ornith", **extra):
    body = {
        "model": model,
        "choices": [{"message": message or {"content": "ok"}, "finish_reason": finish}],
        **extra,
    }
    if usage is not None:
        body["usage"] = usage
    return body


OK_STREAM = (
    b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
)
PAYLOAD = {"model": "ornith", "messages": [{"role": "user", "content": "hi"}]}


def respond(body, status=200):
    return lambda request: httpx.Response(status, json=body)


# --- transport -------------------------------------------------------------


def test_make_client_has_no_env_proxy_no_redirects_and_a_stub_authorization(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    client = oc._make_client(5.0)
    assert client.follow_redirects is False
    assert client._trust_env is False
    assert client.headers["authorization"] == "Bearer lm-studio-local"
    assert "k" * 32 not in str(dict(client.headers))


def test_stub_authorization_reaches_the_wire():
    got = {}

    def handler(request):
        got["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"data": []})

    client = oc._make_client(5.0, transport=httpx.MockTransport(handler))
    client.get("http://127.0.0.1:1234/api/v0/models")
    assert got["auth"] == "Bearer lm-studio-local"


# --- chat_stream ----------------------------------------------------------


def test_stream_sends_the_callers_payload_plus_stream_flags_without_mutating_it(server):
    seen = server(lambda r: httpx.Response(200, content=OK_STREAM))
    payload = {**PAYLOAD, "max_tokens": 50, "temperature": 0.2}
    oc.chat_stream("http://x", payload)
    body = json.loads(seen[0].content)
    assert body == {
        **payload,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert "stream" not in payload  # the caller's dict is untouched
    assert seen[0].url.path == "/v1/chat/completions"


def test_json_object_is_dropped_on_the_wire_but_not_from_the_callers_payload(server):
    seen = server(respond(completion()))
    payload = {**PAYLOAD, "response_format": {"type": "json_object"}}
    oc.chat_complete("http://x", payload)
    assert "response_format" not in json.loads(seen[0].content)
    assert payload["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("stream", [False, True])
def test_schema_definition_is_renamed_to_schema_on_the_wire(server, stream):
    seen = server(
        (lambda r: httpx.Response(200, content=OK_STREAM)) if stream else respond(completion())
    )
    sch = {"type": "object", "title": "T"}
    payload = {
        **PAYLOAD,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "T", "schema_definition": sch, "strict": True},
        },
    }
    (oc.chat_stream if stream else oc.chat_complete)("http://x", payload)
    sent = json.loads(seen[0].content)["response_format"]["json_schema"]
    assert sent == {"name": "T", "schema": sch, "strict": True}
    assert "schema_definition" in payload["response_format"]["json_schema"]


def test_stream_unset_params_are_not_invented(server):
    seen = server(lambda r: httpx.Response(200, content=OK_STREAM))
    oc.chat_stream("http://x", PAYLOAD)
    assert set(json.loads(seen[0].content)) == {"model", "messages", "stream", "stream_options"}


def test_ctrl_c_returns_the_partial_text_as_truncated(server):
    server(
        lambda r: httpx.Response(
            200, content=sse({"choices": [{"delta": {"content": "часть"}}]}, done=False)
        )
    )

    def interrupt(_chunk):
        raise KeyboardInterrupt

    res = oc.chat_stream("http://x", PAYLOAD, on_content=interrupt)
    assert res.text == "часть" and res.truncated is True


class _DropAfter(httpx.SyncByteStream):
    def __init__(self, first: bytes):
        self.first = first

    def __iter__(self):
        if self.first:
            yield self.first
        raise httpx.ReadError("connection reset")


def test_disconnect_after_partial_output_keeps_it_as_truncated(server):
    first = sse({"choices": [{"delta": {"content": "кусок"}}]}, done=False)
    server(lambda r: httpx.Response(200, stream=_DropAfter(first)))
    res = oc.chat_stream("http://x", PAYLOAD)
    assert res.text == "кусок" and res.truncated is True


def test_disconnect_before_any_output_is_a_network_error(server):
    server(lambda r: httpx.Response(200, stream=_DropAfter(b"")))
    with pytest.raises(NetworkError):
        oc.chat_stream("http://x", PAYLOAD)


def test_stream_timeout_is_a_network_error(server):
    def boom(request):
        raise httpx.ReadTimeout("slow")

    server(boom)
    with pytest.raises(NetworkError) as info:
        oc.chat_stream("http://x", PAYLOAD)
    assert "вовремя" in info.value.message


# --- chat_complete ----------------------------------------------------------


def test_complete_parses_text_reasoning_usage_finish_and_model(server):
    seen = server(
        respond(
            completion(
                {"content": "ответ", "reasoning_content": "думаю"},
                usage={
                    "prompt_tokens": 7,
                    "completion_tokens": 9,
                    "total_tokens": 16,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
                model="ornith-q4",
            )
        )
    )
    res = oc.chat_complete("http://x", PAYLOAD)
    assert (res.text, res.reasoning_text, res.finish_reason, res.model) == (
        "ответ",
        "думаю",
        "stop",
        "ornith-q4",
    )
    assert (res.usage.prompt_tokens, res.usage.completion_tokens) == (7, 9)
    assert res.usage.reasoning_tokens == 5
    assert res.tool_calls == ()
    body = json.loads(seen[0].content)
    assert body["stream"] is False and "stream_options" not in body


def test_complete_forwards_tools_and_parses_tool_calls(server):
    tools = [{"type": "function", "function": {"name": "git_log", "parameters": {}}}]
    seen = server(
        respond(
            completion(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "git_log", "arguments": '{"n": 3}'},
                        },
                        {"id": "c2", "function": {"name": "other", "arguments": {"a": "б"}}},
                    ],
                },
                finish="tool_calls",
            )
        )
    )
    res = oc.chat_complete("http://x", {**PAYLOAD, "tools": tools, "tool_choice": "auto"})
    assert json.loads(seen[0].content)["tools"] == tools
    assert [(c.id, c.name, c.arguments) for c in res.tool_calls] == [
        ("c1", "git_log", '{"n": 3}'),
        ("c2", "other", '{"a": "б"}'),  # dict arguments become JSON text, not a Python repr
    ]
    assert res.text == "" and res.finish_reason == "tool_calls"


def test_complete_without_usage_is_none_not_zero(server):
    server(respond(completion()))
    assert oc.chat_complete("http://x", PAYLOAD).usage is None


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"choices": None},
        {"choices": [1]},
        {"choices": [{"message": "text"}]},
        {"choices": [{"message": {"content": 5}}]},
        {"choices": [{"message": {"content": "a", "reasoning_content": 3}}]},
        {"choices": [{"message": {"content": "a"}, "finish_reason": 7}]},
        {"choices": [{"message": {"content": "a"}}], "usage": "many"},
        {"choices": [{"message": {"tool_calls": "x"}}]},
        {"choices": [{"message": {"tool_calls": [1]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": "{}"}}]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"name": "f", "arguments": 3}}]}}]},
        {"choices": [{"message": {"tool_calls": [{"id": 1, "function": {"name": "f"}}]}}]},
        [],
    ],
)
def test_complete_malformed_bodies_are_protocol_errors(server, body):
    server(respond(body))
    with pytest.raises(oc.ProtocolError) as info:
        oc.chat_complete("http://x", PAYLOAD)
    assert isinstance(info.value, ServerError) and info.value.exit_code == 5


def test_complete_non_json_body_is_a_protocol_error(server):
    server(lambda r: httpx.Response(200, content=b"<html>"))
    with pytest.raises(oc.ProtocolError):
        oc.chat_complete("http://x", PAYLOAD)


def test_complete_error_object_in_a_200_body_is_a_server_error(server):
    server(respond({"error": {"message": "model crashed"}}))
    with pytest.raises(ServerError) as info:
        oc.chat_complete("http://x", PAYLOAD)
    assert "model crashed" in info.value.message


@pytest.mark.parametrize(
    ("status", "body", "exc"),
    [
        (404, "no such model", ConfigurationError),
        (400, "model not loaded", ConfigurationError),
        (401, "", AuthError),
        (429, "", RateLimitError),
        (503, "overloaded", ServerError),
        (418, "teapot", AdventError),
    ],
)
def test_complete_http_errors_are_mapped(server, status, body, exc):
    server(lambda r: httpx.Response(status, text=body))
    with pytest.raises(exc) as info:
        oc.chat_complete("http://x", PAYLOAD)
    assert type(info.value) is exc


def test_complete_timeout_and_disconnect_are_network_errors(server):
    def timeout(request):
        raise httpx.ReadTimeout("slow")

    def reset(request):
        raise httpx.RemoteProtocolError("Server disconnected without sending a response")

    for handler in (timeout, reset):
        server(handler)
        with pytest.raises(NetworkError):
            oc.chat_complete("http://x", PAYLOAD)


def test_complete_absolute_deadline_is_a_network_error(server):
    def slow(request):
        time.sleep(0.4)
        return httpx.Response(200, json=completion())

    server(slow)
    with pytest.raises(NetworkError) as info:
        oc.chat_complete("http://x", PAYLOAD, deadline=0.05)
    assert "не ответил за" in info.value.message


def test_complete_does_not_mutate_the_callers_payload(server):
    server(respond(completion()))
    payload = dict(PAYLOAD)
    oc.chat_complete("http://x", payload)
    assert payload == PAYLOAD


# --- check_ready -------------------------------------------------------------


def test_check_ready_require_state_refuses_a_server_without_state():
    models = [oc.LocalModel("ornith")]
    assert oc.check_ready(models, "ornith", "http://x").state is None  # default: lenient
    with pytest.raises(ConfigurationError) as info:
        oc.check_ready(models, "ornith", "http://x", require_state=True)
    assert "state" in info.value.message


def test_ensure_ready_require_state_refuses_the_v1_models_fallback(server):
    def handler(request):
        if request.url.path == "/api/v0/models":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "llama"}]})

    server(handler)
    assert oc.ensure_ready("llama", "http://x").state is None
    with pytest.raises(ConfigurationError):
        oc.ensure_ready("llama", "http://x", require_state=True)


# --- embed -------------------------------------------------------------------


def emb(vectors, *, order=None, usage=7, model="nomic"):
    items = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]
    if order is not None:
        items = [items[i] for i in order]
    body = {"object": "list", "data": items, "model": model}
    if usage is not None:
        body["usage"] = {"prompt_tokens": usage, "total_tokens": usage}
    return body


def test_embed_returns_normalized_float32_rows_in_input_order(server):
    seen = server(respond(emb([[3.0, 4.0], [0.0, 2.0]], usage=11)))
    res = oc.embed("http://x", "nomic", ["a", "b"])
    assert res.vectors.dtype == np.float32 and res.vectors.shape == (2, 2)
    np.testing.assert_allclose(res.vectors, [[0.6, 0.8], [0.0, 1.0]], atol=1e-6)
    assert (res.model, res.prompt_tokens, res.requests, res.dim) == ("nomic", 11, 1, 2)
    assert json.loads(seen[0].content) == {"model": "nomic", "input": ["a", "b"]}
    assert seen[0].url.path == "/v1/embeddings"


def test_embed_reorders_a_shuffled_response_by_index(server):
    server(respond(emb([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], order=[2, 0, 1])))
    res = oc.embed("http://x", "m", ["a", "b", "c"])
    np.testing.assert_allclose(res.vectors[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(res.vectors[1], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(res.vectors[2], [0.70710678, 0.70710678], atol=1e-6)


def test_embed_batches_and_sums_usage(server):
    inputs = []

    def handler(request):
        batch = json.loads(request.content)["input"]
        inputs.append(batch)
        return httpx.Response(200, json=emb([[1.0, float(len(t))] for t in batch], usage=3))

    server(handler)
    progress = []
    res = oc.embed(
        "http://x",
        "m",
        ["a", "bb", "ccc", "dddd", "e"],
        batch=2,
        on_batch=lambda *a: progress.append(a),
    )
    assert inputs == [["a", "bb"], ["ccc", "dddd"], ["e"]]
    assert res.requests == 3 and res.prompt_tokens == 9 and res.vectors.shape == (5, 2)
    assert progress == [(2, 5), (4, 5), (5, 5)]


def test_embed_missing_usage_makes_prompt_tokens_unknown(server):
    server(respond(emb([[1.0, 0.0]], usage=None)))
    assert oc.embed("http://x", "m", ["a"]).prompt_tokens is None


def test_embed_empty_input_makes_no_request(server):
    seen = server(respond(emb([])))
    res = oc.embed("http://x", "m", [])
    assert seen == [] and res.vectors.shape[0] == 0 and res.requests == 0


@pytest.mark.parametrize(
    "body",
    [
        {"data": None},
        {},
        {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},  # one vector for two texts
        {"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]},  # no index
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}]},  # dup
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 2, "embedding": [1.0]}]},  # hole
        {"data": [{"index": True, "embedding": [1.0]}, {"index": 1, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 1, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": "ab"}, {"index": 1, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": ["0.5", 1.0]}, {"index": 1, "embedding": [1, 0]}]},
        {"data": [{"index": 0, "embedding": [True, 1.0]}, {"index": 1, "embedding": [1, 0]}]},
        {"data": [{"index": 0, "embedding": [1e40, 1.0]}, {"index": 1, "embedding": [1, 0]}]},
        {"data": [{"index": 0, "embedding": [0.0, 0.0]}, {"index": 1, "embedding": [1, 0]}]},
        {"data": [1, 2]},
    ],
)
def test_embed_rejects_malformed_raw_bodies(server, body):
    server(respond(body))
    with pytest.raises(oc.ProtocolError):
        oc.embed("http://x", "m", ["a", "b"])


def test_embed_rejects_nan_literals(server):
    server(
        lambda r: httpx.Response(
            200,
            content=b'{"data":[{"index":0,"embedding":[NaN,1.0]},{"index":1,"embedding":[1,0]}]}',
        )
    )
    with pytest.raises(oc.ProtocolError):
        oc.embed("http://x", "m", ["a", "b"])


def test_embed_enforces_the_expected_dimension(server):
    server(respond(emb([[1.0, 0.0, 0.0]])))
    assert oc.embed("http://x", "m", ["a"], expected_dim=3).dim == 3
    with pytest.raises(oc.ProtocolError) as info:
        oc.embed("http://x", "m", ["a"], expected_dim=768)
    assert "768" in info.value.message


def test_embed_dimension_must_not_change_between_batches(server):
    dims = iter([2, 3])
    server(lambda r: httpx.Response(200, json=emb([[1.0] * next(dims)])))
    with pytest.raises(oc.ProtocolError):
        oc.embed("http://x", "m", ["a", "b"], batch=1)


@pytest.mark.parametrize(
    ("status", "exc"), [(404, ConfigurationError), (401, AuthError), (500, ServerError)]
)
def test_embed_http_errors_follow_the_chat_contract(server, status, exc):
    server(lambda r: httpx.Response(status, text="model not found"))
    with pytest.raises(exc):
        oc.embed("http://x", "m", ["a"])


def test_embed_connection_failure_is_a_network_error(server):
    def boom(request):
        raise httpx.ConnectError("refused")

    server(boom)
    with pytest.raises(NetworkError):
        oc.embed("http://x", "m", ["a"])


def test_embed_whole_call_deadline_spans_batches(server, monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(oc, "_clock", lambda: now["t"])

    def handler(request):
        now["t"] += 60.0  # every request "takes" a minute
        return httpx.Response(200, json=emb([[1.0, 0.0]]))

    server(handler)
    with pytest.raises(NetworkError):
        oc.embed("http://x", "m", ["a", "b"], batch=1, deadline=30.0)


# --- day 27 review: response validation ---------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"choices": [{"message": {"content": [{"text": 42}]}}]},
        {"choices": [{"message": {"content": ["raw string"]}}]},
        {"choices": [{"message": {"content": "a"}}], "usage": {"prompt_tokens": "oops"}},
        {"choices": [{"message": {"content": "a"}}], "usage": {"completion_tokens": 1.5}},
        {"choices": [{"message": {"content": "a"}}], "usage": {"total_tokens": True}},
        {"choices": [{"message": {"content": "a"}}], "usage": {"total_tokens": -1}},
        {
            "choices": [{"message": {"content": "a"}}],
            "usage": {"prompt_tokens_details": {"cached_tokens": "x"}},
        },
        {"choices": [{"message": {"content": "a"}}], "usage": {"completion_tokens_details": 5}},
    ],
)
def test_complete_content_blocks_and_usage_numbers_are_validated(server, body):
    server(respond(body))
    with pytest.raises(oc.ProtocolError):  # never a raw TypeError, never a str in token counts
        oc.chat_complete("http://x", PAYLOAD)


def test_complete_valid_content_blocks_still_join(server):
    body = {"choices": [{"message": {"content": [{"text": "a"}, {"text": None}, {"text": "b"}]}}]}
    server(respond(body))
    assert oc.chat_complete("http://x", PAYLOAD).text == "ab"


@pytest.mark.parametrize(
    "usage",
    [{"prompt_tokens": "7"}, {"completion_tokens": 2.5}, {"total_tokens": [1]}, "many"],
)
def test_stream_usage_numbers_are_validated(server, usage):
    stream = sse({"choices": [{"delta": {"content": "x"}}]}, {"choices": [], "usage": usage})
    server(lambda r: httpx.Response(200, content=stream))
    with pytest.raises(oc.ProtocolError):
        oc.chat_stream("http://x", PAYLOAD)


def test_stream_with_only_usage_events_is_an_empty_completion(server):
    usage = {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3}
    for events in ([{"choices": []}], [{"choices": [], "usage": usage}], [{"choices": None}]):
        server(lambda r, e=events: httpx.Response(200, content=sse(*e)))
        with pytest.raises(oc.ProtocolError):
            oc.chat_stream("http://x", PAYLOAD)


def test_stream_usage_only_event_next_to_real_ones_is_fine(server):
    usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    stream = sse(
        {"choices": []},
        {"choices": [{"delta": {"content": "x"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": usage},
    )
    server(lambda r: httpx.Response(200, content=stream))
    res = oc.chat_stream("http://x", PAYLOAD)
    assert res.text == "x" and res.usage.total_tokens == 4 and res.truncated is False


def test_embed_normalizes_in_float64_so_huge_vectors_do_not_become_zero(server):
    server(respond(emb([[1e30, 1e30], [1e-30, 0.0]])))
    res = oc.embed("http://x", "m", ["a", "b"])
    assert np.isfinite(res.vectors).all()
    np.testing.assert_allclose(res.vectors, [[2**-0.5, 2**-0.5], [1.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(res.vectors, axis=1), [1.0, 1.0], atol=1e-6)


def test_embed_on_request_reports_each_good_request_before_a_later_failure(server):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 2:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=emb([[1.0, 0.0]], usage=9))

    server(handler)
    seen: list[tuple[int | None, int]] = []
    with pytest.raises(ServerError):
        oc.embed(
            "http://x",
            "m",
            ["a", "b"],
            batch=1,
            on_request=lambda tokens, ms: seen.append((tokens, ms)),
        )
    assert len(seen) == 1 and seen[0][0] == 9 and seen[0][1] >= 0


def test_embed_on_request_reports_unknown_usage_as_none(server):
    server(respond(emb([[1.0, 0.0]], usage=None)))
    seen: list[int | None] = []
    oc.embed("http://x", "m", ["a"], on_request=lambda tokens, ms: seen.append(tokens))
    assert seen == [None]
