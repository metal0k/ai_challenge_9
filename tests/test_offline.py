"""advent_core/offline.py: the httpx transport guard, counters, env inheritance."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx
import pytest

from advent_core import chat as chat_core
from advent_core import client as client_mod
from advent_core import offline
from advent_core.config import PROJECT_ROOT, Config
from advent_core.errors import AdventError, CloudBlockedError

ORIGINAL_SEND = httpx.Client._send_single_request


def mock_client(hits: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(200, json={"data": []})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_guard_is_off_until_enabled():
    hits: list[httpx.Request] = []
    assert httpx.Client._send_single_request is ORIGINAL_SEND
    assert mock_client(hits).get("https://api.mistral.ai/v1/models").status_code == 200
    assert offline.counters().attempted == {"local": 0, "cloud": 0}


@pytest.mark.allow_cloud_attempts
def test_cloud_request_is_blocked_before_send_and_names_the_url():
    hits: list[httpx.Request] = []
    offline.enable()
    with pytest.raises(CloudBlockedError) as info:
        mock_client(hits).get("https://api.mistral.ai/v1/models")
    assert hits == []  # never reached the transport
    assert "https://api.mistral.ai/v1/models" in info.value.message
    assert "GET api.mistral.ai" in info.value.message
    assert isinstance(info.value, AdventError) and info.value.exit_code == 2
    snap = offline.counters()
    assert snap.attempted == {"local": 0, "cloud": 1}
    assert snap.blocked == {"local": 0, "cloud": 1}
    assert snap.completed == {"local": 0, "cloud": 0}


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:1234/v1/models", "http://localhost:1234/x", "http://[::1]:1234/x"]
)
def test_loopback_passes_and_is_counted_as_local(url):
    hits: list[httpx.Request] = []
    offline.enable()
    assert mock_client(hits).get(url).status_code == 200
    assert len(hits) == 1
    snap = offline.counters()
    assert snap.attempted == {"local": 1, "cloud": 0}
    assert snap.completed == {"local": 1, "cloud": 0}
    assert snap.blocked == {"local": 0, "cloud": 0}


@pytest.mark.allow_cloud_attempts
@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/x/resolve/main/tekken.json",
        "http://127.0.0.1.evil.example/x",
        "http://10.0.0.5:1234/x",
        "http://localhost.example.com/x",
    ],
)
def test_non_loopback_hosts_are_blocked(url):
    offline.enable()
    with pytest.raises(CloudBlockedError):
        mock_client([]).get(url)


@pytest.mark.allow_cloud_attempts
def test_module_level_httpx_get_is_blocked_without_touching_the_network():
    offline.enable()
    with pytest.raises(CloudBlockedError):
        httpx.get("https://api.mistral.ai/v1/models")
    with pytest.raises(CloudBlockedError), httpx.stream("GET", "https://huggingface.co/x"):
        pass


@pytest.mark.allow_cloud_attempts
def test_async_client_is_guarded_too():
    offline.enable()

    async def go():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200))
        ) as client:
            await client.get("https://api.mistral.ai/v1/models")

    with pytest.raises(CloudBlockedError):
        asyncio.run(go())
    assert offline.counters().blocked["cloud"] == 1


def test_enable_is_idempotent_and_disable_restores_httpx():
    offline.enable()
    patched = httpx.Client._send_single_request
    offline.enable()
    assert httpx.Client._send_single_request is patched and patched is not ORIGINAL_SEND
    offline.disable()
    assert httpx.Client._send_single_request is ORIGINAL_SEND
    assert not offline.is_enabled()


def test_enable_exports_the_flag_for_children(monkeypatch):
    import os

    offline.enable()
    assert os.environ["ADVENT_OFFLINE"] == "1"
    offline.disable()
    assert "ADVENT_OFFLINE" not in os.environ


def test_init_from_env_enables_only_when_the_flag_is_set(monkeypatch):
    assert offline.init_from_env() is False
    monkeypatch.setenv("ADVENT_OFFLINE", "1")
    assert offline.init_from_env() is True
    assert offline.is_enabled()


@pytest.mark.allow_cloud_attempts
def test_assert_cloud_allowed_is_a_noop_online_and_raises_offline():
    offline.assert_cloud_allowed("RAG rewrite")
    offline.enable()
    with pytest.raises(CloudBlockedError) as info:
        offline.assert_cloud_allowed("RAG rewrite: ministral-14b-latest")
    assert "RAG rewrite: ministral-14b-latest" in info.value.message
    assert offline.counters().blocked["cloud"] == 1


def test_reset_counters():
    offline.enable()
    with pytest.raises(CloudBlockedError):
        mock_client([]).get("https://api.mistral.ai/x")
    offline.reset_counters()
    assert offline.counters().attempted == {"local": 0, "cloud": 0}


@pytest.mark.allow_cloud_attempts
def test_list_models_without_base_url_is_blocked_in_offline():
    # Real httpx.get inside list_models: the guard refuses before any socket.
    offline.enable()
    with pytest.raises(CloudBlockedError) as info:
        client_mod.list_models(Config(api_key="stub", offline=True))
    assert client_mod.MODELS_URL in info.value.message


def test_list_models_with_loopback_base_url_passes_the_guard(monkeypatch):
    offline.enable()
    seen: list[str] = []

    def fake_send(self, request, *a, **k):  # replaces the single-request step below the guard
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "ornith"}]}, request=request)

    # The guard wraps whatever _send_single_request is when enabled; re-enable over a fake.
    offline.disable()
    monkeypatch.setattr(httpx.Client, "_send_single_request", fake_send)
    offline.enable()
    config = Config(api_key="stub", base_url="http://127.0.0.1:1234", offline=True)
    assert client_mod.list_models(config) == [{"id": "ornith"}]
    assert seen == ["http://127.0.0.1:1234/v1/models"]
    assert offline.counters().completed["local"] == 1


@pytest.mark.allow_cloud_attempts
def test_sdk_chat_without_base_url_is_blocked_in_offline():
    offline.enable()
    with pytest.raises(CloudBlockedError):
        chat_core.complete(Config(api_key="k" * 32), [{"role": "user", "content": "hi"}])
    assert offline.counters().blocked["cloud"] >= 1


def test_project_does_not_use_urllib_request_or_requests():
    """The guard covers httpx only, so nothing else may open a socket to the web."""
    needle = re.compile(
        r"^\s*(import\s+(urllib\.request|requests)\b|from\s+(urllib\.request|requests)\b)"
        r"|\burllib\.request\.urlopen\b|\burlopen\(",
        re.MULTILINE,
    )
    skip = {"tests", "node_modules", "__pycache__", "data", "logs"}
    offenders = []
    candidates = [PROJECT_ROOT] + [
        d for d in PROJECT_ROOT.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    for base in candidates:
        files = base.glob("*.py") if base is PROJECT_ROOT else base.rglob("*.py")
        try:
            for path in files:
                if skip & set(path.relative_to(PROJECT_ROOT).parts):
                    continue
                if needle.search(path.read_text(encoding="utf-8", errors="replace")):
                    offenders.append(str(Path(path).relative_to(PROJECT_ROOT)))
        except PermissionError:
            continue
    assert offenders == []


def _loopback_only_transport_recorder(monkeypatch, seen: list[tuple[str, str]]):
    """Replace the real socket transports; record (pool class, url) per request."""

    def sync(self, request):
        seen.append((type(self._pool).__name__, str(request.url)))
        return httpx.Response(200, json={"data": [{"id": "m"}]}, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", sync)


def test_list_models_ignores_an_env_proxy_for_local_and_offline(monkeypatch):
    """HTTP_PROXY must not re-route a loopback call to an external host (finding 1)."""
    monkeypatch.setenv("HTTP_PROXY", "http://external-proxy.example:8080")
    monkeypatch.setenv("http_proxy", "http://external-proxy.example:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    seen: list[tuple[str, str]] = []
    _loopback_only_transport_recorder(monkeypatch, seen)
    offline.enable()
    config = Config(api_key="stub", base_url="http://127.0.0.1:1234", offline=True)
    assert client_mod.list_models(config) == [{"id": "m"}]
    assert seen == [("ConnectionPool", "http://127.0.0.1:1234/v1/models")]
    assert offline.counters().attempted == {"local": 1, "cloud": 0}


def test_list_models_does_not_follow_redirects_when_local(monkeypatch):
    def redirect(self, request):
        return httpx.Response(
            302, headers={"location": "https://api.mistral.ai/x"}, request=request
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", redirect)
    offline.enable()
    config = Config(api_key="stub", base_url="http://127.0.0.1:1234", offline=True)
    with pytest.raises(AdventError):  # the 302 is an error, never followed
        client_mod.list_models(config)
    assert offline.counters().attempted == {"local": 1, "cloud": 0}


def _redirecting_client(hits: list[str], **kw) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        if request.url.host == "127.0.0.1":
            return httpx.Response(302, headers={"location": "https://api.mistral.ai/v1/models"})
        return httpx.Response(200, json={})

    return httpx.Client(transport=httpx.MockTransport(handler), **kw)


@pytest.mark.allow_cloud_attempts
def test_redirect_to_the_cloud_is_blocked_for_a_client_created_before_enable():
    hits: list[str] = []
    client = _redirecting_client(hits, follow_redirects=True)  # built BEFORE enable()
    offline.enable()
    with pytest.raises(CloudBlockedError):
        client.get("http://127.0.0.1:1234/start")
    assert hits == ["http://127.0.0.1:1234/start"]  # the cloud hop never reached the transport
    snap = offline.counters()
    assert snap.attempted == {"local": 1, "cloud": 1}
    assert snap.blocked == {"local": 0, "cloud": 1}


@pytest.mark.allow_cloud_attempts
def test_async_redirect_to_the_cloud_is_blocked_for_a_client_created_before_enable():
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://api.mistral.ai/x"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    offline.enable()

    async def go():
        async with client:
            await client.get("http://localhost:1234/start")

    with pytest.raises(CloudBlockedError):
        asyncio.run(go())
    assert hits == ["http://localhost:1234/start"]
    assert offline.counters().blocked["cloud"] == 1
