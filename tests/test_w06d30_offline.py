"""offline guard: the `private` endpoint class (ADVENT_PRIVATE_ORIGINS allowlist)."""

from __future__ import annotations

import httpx
import pytest

from advent_core import offline
from advent_core.errors import CloudBlockedError

ORIGIN = "https://advent-llm.duckdns.org"


@pytest.fixture(autouse=True)
def _origins(monkeypatch):
    monkeypatch.setenv("ADVENT_PRIVATE_ORIGINS", ORIGIN)


def client(hits: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request)
        return httpx.Response(200, json={})

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.allow_cloud_attempts
def test_loopback_private_and_blocked_cloud_in_one_process():
    hits: list[httpx.Request] = []
    offline.enable()
    http = client(hits)
    assert http.get("http://127.0.0.1:1234/v1/models").status_code == 200
    assert http.get(f"{ORIGIN}/health").status_code == 200
    with pytest.raises(CloudBlockedError):
        http.get("https://api.mistral.ai/v1/models")
    assert len(hits) == 2
    snap = offline.counters()
    assert snap.attempted == {"local": 1, "private": 1, "cloud": 1}
    assert snap.blocked == {"local": 0, "private": 0, "cloud": 1}
    assert snap.completed == {"local": 1, "private": 1, "cloud": 0}


@pytest.mark.allow_cloud_attempts
@pytest.mark.parametrize(
    "url",
    [
        "https://api.mistral.ai/v1/chat",
        "https://eu.api.mistral.ai/v1/chat",
        "https://example.org/",
        "http://advent-llm.duckdns.org/health",
        "https://user@advent-llm.duckdns.org/health",
        "https://advent-llm.duckdns.org:8443/health",
        "https://advent-llm.duckdns.org.evil.example/health",
    ],
)
def test_everything_outside_the_allowlist_is_blocked(url):
    hits: list[httpx.Request] = []
    offline.enable()
    with pytest.raises(CloudBlockedError):
        client(hits).get(url)
    assert hits == []
    assert offline.counters().blocked["cloud"] == 1


@pytest.mark.allow_cloud_attempts
def test_mistral_host_stays_blocked_even_when_listed(monkeypatch):
    monkeypatch.setenv("ADVENT_PRIVATE_ORIGINS", f"https://api.mistral.ai {ORIGIN}")
    offline.enable()
    with pytest.raises(CloudBlockedError):
        client([]).get("https://api.mistral.ai/v1/models")
    assert offline.endpoint_of("https://api.mistral.ai/x") == "cloud"


@pytest.mark.allow_cloud_attempts
def test_without_the_env_a_private_host_is_cloud(monkeypatch):
    monkeypatch.delenv("ADVENT_PRIVATE_ORIGINS")
    offline.enable()
    with pytest.raises(CloudBlockedError):
        client([]).get(f"{ORIGIN}/health")


def test_default_port_and_case_normalize():
    assert offline.endpoint_of("https://ADVENT-LLM.duckdns.org:443/x") == "private"
    assert offline.endpoint_of(httpx.URL(f"{ORIGIN}/v1/status")) == "private"


def test_bad_allowlist_entries_authorize_nothing(monkeypatch):
    monkeypatch.setenv(
        "ADVENT_PRIVATE_ORIGINS",
        "http://a.example, https://b.example/path, https://u@c.example, https://ok.example, junk",
    )
    assert offline.private_origins() == frozenset({"https://ok.example"})


@pytest.mark.allow_cloud_attempts
def test_redirect_hop_to_another_host_is_blocked():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://evil.example/"})

    offline.enable()
    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    with pytest.raises(CloudBlockedError):
        http.get(f"{ORIGIN}/x")
    assert calls == ["advent-llm.duckdns.org"]


def test_reset_and_snapshot_cover_private():
    offline.enable()
    client([]).get(f"{ORIGIN}/x")
    snap = offline.counters()
    offline.reset_counters()
    assert snap.attempted["private"] == 1
    assert offline.counters().attempted == {"local": 0, "private": 0, "cloud": 0}
