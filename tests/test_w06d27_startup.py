"""Day 27: the real `adventagent --local` startup and one turn, faking only the socket layer.

Unlike test_w06d27_agent.py (which replaces readiness, model listing and the tokenizer
with doubles), nothing above `httpx.HTTPTransport.handle_request` is replaced here: the
guard, Config.resolve, readiness, list_models, the chat client and the counters are all real.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import httpx
import pytest

import week_02.cli as cli
from advent_core import journal as journal_module
from advent_core import offline
from advent_core import session as session_module
from advent_core.config import LOOPBACK_HOSTS

LM_MODELS = {
    "data": [
        {
            "id": "ornith",
            "type": "llm",
            "state": "loaded",
            "loaded_context_length": 40960,
            "arch": "qwen3",
            "quantization": "Q4",
        },
        {"id": "text-embedding-bge-m3", "type": "embeddings", "state": "loaded"},
    ]
}
V1_MODELS = {"data": [{"id": "ornith"}, {"id": "text-embedding-bge-m3"}]}
STREAM = (
    'data: {"choices":[{"delta":{"reasoning_content":"думаю"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Привет!"},"finish_reason":"stop"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}\n\n'
    "data: [DONE]\n\n"
).encode()


@pytest.fixture
def wire(monkeypatch, tmp_path):
    """Capture every request that reaches the (faked) socket layer."""
    sent: list[httpx.Request] = []

    def handle(self, request):
        sent.append(request)
        path = request.url.path
        if path == "/api/v0/models":
            return httpx.Response(200, json=LM_MODELS, request=request)
        if path == "/v1/models":
            return httpx.Response(200, json=V1_MODELS, request=request)
        if path == "/v1/chat/completions":
            return httpx.Response(200, content=STREAM, request=request)
        return httpx.Response(404, json={"error": "no such route"}, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    monkeypatch.setattr(journal_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(cli, "MCP_REGISTRY_PATH", Path("no-such-registry.json"))
    # the demo blanks the cloud key in the child; .env must not bring it back
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    monkeypatch.setenv("ADVENT_LOCAL_URL", "http://127.0.0.1:1234")
    for name in ("ADVENT_BASE_URL", "MISTRAL_MODEL", "HTTP_PROXY", "http_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    return sent


def _run(monkeypatch, lines: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["adventagent", "--local", "--session", "startup-test"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    try:
        cli.main()
    except SystemExit as exit_:
        assert exit_.code in (0, None)


def test_real_local_startup_and_one_turn_never_leave_loopback(monkeypatch, wire, capsys):
    _run(monkeypatch, ["Привет", "/local", "/exit"])

    hosts = {request.url.host for request in wire}
    assert hosts and hosts <= LOOPBACK_HOSTS, hosts
    paths = {request.url.path for request in wire}
    assert {"/api/v0/models", "/v1/chat/completions"} <= paths
    # the stub key only, on every request
    assert {r.headers["authorization"] for r in wire} == {"Bearer lm-studio-local"}
    # asserted BEFORE the fixture teardown wipes the counters
    snap = offline.counters()
    assert snap.attempted["cloud"] == 0 and snap.blocked["cloud"] == 0
    assert snap.attempted["local"] == len(wire) and snap.completed["local"] == len(wire)
    assert "Привет!" in capsys.readouterr().out  # the answer is the product: stdout


def test_the_turn_reached_the_model_and_was_journaled(monkeypatch, wire, tmp_path):
    _run(monkeypatch, ["Привет", "/exit"])

    chat = [r for r in wire if r.url.path == "/v1/chat/completions"]
    assert len(chat) == 1
    body = json.loads(chat[0].content)
    assert body["model"] == "ornith" and body["stream"] is True
    assert body["messages"][-1] == {"role": "user", "content": "Привет"}
    saved = session_module.Session.load("startup-test", directory=tmp_path / "sessions")
    assert [t.role for t in saved.turns][-2:] == ["user", "assistant"]


@pytest.mark.allow_cloud_attempts
def test_a_stray_cloud_request_in_the_real_path_is_counted_and_blocked(wire):
    """Opts out of the suite-wide teardown check, which fails on any cloud attempt."""
    offline.enable()
    with httpx.Client() as client, pytest.raises(offline.CloudBlockedError):
        client.get("https://api.mistral.ai/v1/models")
    assert wire == []  # blocked before the socket layer
    assert offline.counters().blocked["cloud"] == 1
