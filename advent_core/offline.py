"""Offline guard: in offline mode the process may talk to loopback only.

The guard sits on `Client._send_single_request` / `AsyncClient._send_single_request`,
the one place every request passes through (each redirect hop and auth retry included,
and for clients created before `enable()` too), not on a client factory: the Mistral
SDK, `list_models`, `openai_compat`, the tokenizer download and anything added later
all go through httpx, so a forgotten call site fails loudly instead of leaking.
`urllib`/`requests` are not used in the project; a test greps for them.

State is process-wide and inherited through `ADVENT_OFFLINE=1`, so MCP children
switch their own guard on with `init_from_env()`. Counters are per process only.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from advent_core.config import (
    LOOPBACK_HOSTS,
    is_loopback_url,
    offline_env,
    url_host,
    validate_loopback_url,
)
from advent_core.errors import CloudBlockedError

__all__ = [
    "LOOPBACK_HOSTS",
    "CloudBlockedError",
    "Counters",
    "assert_cloud_allowed",
    "counters",
    "disable",
    "enable",
    "endpoint_of",
    "init_from_env",
    "is_enabled",
    "is_loopback_url",
    "normalize_origin",
    "private_origins",
    "reset_counters",
    "validate_loopback_url",
]

ENV_FLAG = "ADVENT_OFFLINE"
PRIVATE_ORIGINS_ENV = "ADVENT_PRIVATE_ORIGINS"
_ENDPOINTS = ("local", "private", "cloud")

_lock = threading.Lock()
_enabled = False
_originals: dict[str, object] = {}


def _zeros() -> dict[str, int]:
    return dict.fromkeys(_ENDPOINTS, 0)


@dataclass(slots=True)
class Counters:
    """attempted/blocked/completed per endpoint ("local" | "private" | "cloud")."""

    attempted: dict[str, int] = field(default_factory=_zeros)
    blocked: dict[str, int] = field(default_factory=_zeros)
    completed: dict[str, int] = field(default_factory=_zeros)

    def snapshot(self) -> Counters:
        return Counters(dict(self.attempted), dict(self.blocked), dict(self.completed))


_counters = Counters()


def _is_mistral_host(host: str) -> bool:
    return host == "mistral.ai" or host.endswith(".mistral.ai")


def normalize_origin(url: object, *, origin_only: bool = False) -> str | None:
    """`https://host[:port]` (443 dropped) for a well-formed https URL, else None.

    Rejects other schemes, userinfo, a bad port and a Mistral host. With
    `origin_only` also a path, query or fragment: an allowlist entry is an origin,
    not a prefix, so `https://x/../` cannot widen what it authorizes.
    """
    try:
        parts = urlsplit(str(url).strip())
        port = parts.port
    except ValueError:
        return None
    host = (parts.hostname or "").lower().rstrip(".")
    if parts.scheme != "https" or not host or "@" in parts.netloc or _is_mistral_host(host):
        return None
    if origin_only and (parts.path not in ("", "/") or parts.query or parts.fragment):
        return None
    if port == 0:
        return None
    host_part = f"[{host}]" if ":" in host else host
    return f"https://{host_part}" if port in (None, 443) else f"https://{host_part}:{port}"


def private_origins() -> frozenset[str]:
    """Normalized origins from ADVENT_PRIVATE_ORIGINS (comma/space separated).

    Read on every call, not at import: tests and `.env` loading change it later.
    A bad entry is dropped rather than failing the process, and it authorizes nothing.
    """
    raw = os.environ.get(PRIVATE_ORIGINS_ENV) or ""
    entries = (normalize_origin(item, origin_only=True) for item in raw.replace(",", " ").split())
    return frozenset(entry for entry in entries if entry)


def endpoint_of(url: object) -> str:
    """Endpoint class: "local" (loopback), "private" (allowlisted https origin) or "cloud".

    The URL never authorizes itself: "private" needs an explicit ADVENT_PRIVATE_ORIGINS
    entry, and an unknown or unparsable host is "cloud".
    """
    host = getattr(url, "host", None)
    if host is None:
        host = url_host(str(url))
    if host and str(host).lower() in LOOPBACK_HOSTS:
        return "local"
    origin = normalize_origin(url)
    return "private" if origin is not None and origin in private_origins() else "cloud"


_ALLOWED_HINT = (
    "разрешены loopback (127.0.0.1, localhost, ::1) и origin'ы из ADVENT_PRIVATE_ORIGINS"
)


def _blocked(url: object, what: str | None) -> CloudBlockedError:
    shown = str(url)
    detail = f": {what}" if what else ""
    return CloudBlockedError(
        f"облако отключено (offline): запрос на {shown} заблокирован{detail}",
        hint=_ALLOWED_HINT,
    )


def _guard_request(request: httpx.Request) -> str:
    endpoint = endpoint_of(request.url)
    with _lock:
        _counters.attempted[endpoint] += 1
        if endpoint == "cloud":
            _counters.blocked[endpoint] += 1
    if endpoint == "cloud":
        raise _blocked(request.url, f"{request.method} {request.url.host}")
    return endpoint


def _complete(endpoint: str) -> None:
    with _lock:
        _counters.completed[endpoint] += 1


def _wrap_sync(original):
    def send_single(self, request, *args, **kwargs):
        endpoint = _guard_request(request)
        response = original(self, request, *args, **kwargs)
        _complete(endpoint)
        return response

    send_single._advent_guard = True  # type: ignore[attr-defined]
    return send_single


def _wrap_async(original):
    async def send_single(self, request, *args, **kwargs):
        endpoint = _guard_request(request)
        response = await original(self, request, *args, **kwargs)
        _complete(endpoint)
        return response

    send_single._advent_guard = True  # type: ignore[attr-defined]
    return send_single


def enable() -> None:
    """Turn the guard on for this process and for children (ADVENT_OFFLINE=1). Idempotent."""
    global _enabled
    with _lock:
        os.environ[ENV_FLAG] = "1"
        if _enabled:
            return
        _originals["sync"] = httpx.Client._send_single_request
        _originals["async"] = httpx.AsyncClient._send_single_request
        httpx.Client._send_single_request = _wrap_sync(_originals["sync"])  # type: ignore[method-assign]
        httpx.AsyncClient._send_single_request = _wrap_async(_originals["async"])  # type: ignore[method-assign]
        _enabled = True


def disable() -> None:
    """Restore httpx and clear the env flag (tests, and nothing else)."""
    global _enabled
    with _lock:
        os.environ.pop(ENV_FLAG, None)
        if not _enabled:
            return
        httpx.Client._send_single_request = _originals.pop("sync")  # type: ignore[method-assign]
        httpx.AsyncClient._send_single_request = _originals.pop("async")  # type: ignore[method-assign]
        _enabled = False


def is_enabled() -> bool:
    return _enabled


def init_from_env() -> bool:
    """Enable the guard when ADVENT_OFFLINE is set (child processes, late cloud paths)."""
    if offline_env():
        enable()
    return _enabled


def assert_cloud_allowed(what: str) -> None:
    """Explicit check for a cloud path that is not an httpx call (e.g. before a download)."""
    if _enabled or offline_env():
        with _lock:
            _counters.attempted["cloud"] += 1
            _counters.blocked["cloud"] += 1
        raise CloudBlockedError(
            f"облако отключено (offline): {what}",
            hint=_ALLOWED_HINT,
        )


def counters() -> Counters:
    """Snapshot of the per-endpoint counters of this process."""
    with _lock:
        return _counters.snapshot()


def reset_counters() -> None:
    with _lock:
        for table in (_counters.attempted, _counters.blocked, _counters.completed):
            table.update(_zeros())
