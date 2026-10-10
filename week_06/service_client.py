"""Client for the private LLM service (the day-30 gateway): httpx, explicit Bearer, SSE.

Separate from `local_client`/`openai_compat` on purpose: those talk to a loopback
LM Studio with a stub key. Here a real key leaves the machine, so the origin is
checked against the explicit ADVENT_PRIVATE_ORIGINS allowlist *before* the key is
attached to anything, the client ignores proxy env and never follows a redirect
(a redirect would carry the Authorization header to another host).

Gateway contract: specs/SPEC-w06d30.md §2.3-2.5, §10. Before the headers the
answer is a plain HTTP status; after HTTP 200 a stream can still end with an
`error` event or just stop, and that is reported as an incomplete result, not
as a success.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit

import httpx

from advent_core import offline
from advent_core.config import LOOPBACK_HOSTS, ConfigError, load_env, redact, url_host
from advent_core.errors import AdventError, NetworkError
from advent_core.telemetry import Usage

DEFAULT_MAX_TOKENS = 256
MAX_TOKENS_CEILING = 512
SERVICE_URL_ENV = "ADVENT_SERVICE_URL"
SERVICE_KEY_ENV = "ADVENT_SERVICE_KEY"
DEMO_KEY_ENV = "ADVENT_SERVICE_DEMO_KEY"
KEY_PREFIX = "adv_"
# Queue wait (<=150 s) + cold start + generation (<=150 s) all pass before the
# first byte of a non-stream answer; headers of a stream arrive after admission.
READ_TIMEOUT = 330.0
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# keys and target
# ---------------------------------------------------------------------------


def generate_key() -> tuple[str, str]:
    """(key, sha256 hex): the key is shown once, only the hash goes into keys.json."""
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    return key, sha256(key.encode("utf-8")).hexdigest()


def validate_key_env_name(name: str) -> str:
    """`--key-env` takes a variable NAME; a pasted key must not be echoed back."""
    # A real key is a valid identifier too, so the prefix is checked as well.
    if not _ENV_NAME.match(name or "") or name.startswith(KEY_PREFIX) or len(name) > 64:
        raise ConfigError(
            "--key-env принимает имя переменной окружения, а не сам ключ "
            "(значение не показано; ключ держи в .env)."
        )
    return name


def _origin_allowed(origin: str) -> bool:
    if origin in offline.private_origins():
        return True
    # A loopback gateway (local dev, tests) never sends the key off the machine.
    return (url_host(origin) or "") in LOOPBACK_HOSTS


def _loopback_origin(url: str) -> str | None:
    """`http(s)://127.0.0.1[:port]` for a well-formed loopback URL, else None."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if host not in LOOPBACK_HOSTS or parts.scheme not in ("http", "https") or "@" in parts.netloc:
        return None
    shown = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{shown}" + (f":{port}" if port else "")


@dataclass(frozen=True, slots=True)
class ServiceTarget:
    """A validated origin plus the key for it. Cannot be built from an unlisted origin."""

    origin: str
    key: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.key:
            raise ConfigError("ключ сервиса пуст.")


def make_target(
    url: str | None, key: str | None, *, url_hint: str = SERVICE_URL_ENV
) -> ServiceTarget:
    """Validate the URL against ADVENT_PRIVATE_ORIGINS, then (only then) pair it with the key."""
    if not url or not url.strip():
        raise ConfigError(f"Не задан адрес сервиса: {url_hint} в .env или --url.")
    origin = offline.normalize_origin(url, origin_only=True) or _loopback_origin(url)
    if origin is None:
        raise ConfigError(
            "Адрес сервиса некорректен: нужен https://host[:порт] без логина, пути и "
            "параметров (хосты Mistral недопустимы)."
        )
    if not _origin_allowed(origin):
        raise ConfigError(
            f"Origin {origin} не разрешён: добавь его в {offline.PRIVATE_ORIGINS_ENV} "
            "(адрес сам себя не разрешает, ключ туда не отправлен)."
        )
    if not key:
        raise ConfigError("Не задан ключ сервиса (переменная из --key-env пуста или отсутствует).")
    return ServiceTarget(origin, key)


def resolve_target(url: str | None = None, key_env: str = SERVICE_KEY_ENV) -> ServiceTarget:
    """Address from --url or ADVENT_SERVICE_URL, key from the env var NAMED by `key_env`."""
    load_env()
    validate_key_env_name(key_env)
    value = (os.environ.get(key_env) or "").strip()
    if not value:
        raise ConfigError(f"Не задан ключ сервиса: переменная {key_env} пуста (см. .env.example).")
    return make_target(url or os.environ.get(SERVICE_URL_ENV), value)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class ServiceError(AdventError):
    """A gateway refusal (HTTP status before the stream) with its machine-readable parts."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        error_type: str | None = None,
        numbers: dict[str, int] | None = None,
        retry_after: float | None = None,
        hint: str | None = None,
        exit_code: int = 1,
    ) -> None:
        super().__init__(message, hint=hint)
        self.status = status
        self.error_type = error_type
        self.numbers = numbers or {}
        self.retry_after = retry_after
        self.exit_code = exit_code


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_error(response: httpx.Response, body: bytes, secret: str = "") -> ServiceError:
    """Map a non-200 gateway answer to ServiceError (OpenAI-style `{"error": {...}}`)."""
    status = response.status_code
    err: dict[str, Any] = {}
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            err = data["error"]
    except ValueError:
        pass
    raw_message = err.get("message") if isinstance(err.get("message"), str) else None
    if raw_message is None:
        raw_message = body.decode("utf-8", errors="replace").strip()[:200] or f"HTTP {status}"
    # The gateway puts the specific id in `code`; its `type` is only the OpenAI class.
    error_type = next((err[k] for k in ("code", "type") if isinstance(err.get(k), str)), None)
    numbers = {
        k: v
        for k in ("input_tokens", "max_tokens", "n_ctx")
        if (v := _int_or_none(err.get(k))) is not None
    }
    retry_after = None
    with contextlib.suppress(ValueError):
        retry_after = float(response.headers.get("retry-after", ""))
    message = redact(raw_message)
    if secret:
        message = message.replace(secret, "***")
    label = f"{status}" + (f" {error_type}" if error_type else "")
    if status in (401, 403):
        code, hint = 3, "проверь ключ в .env (переменная из --key-env)"
    elif status == 429:
        code, hint = 4, None
    elif status >= 500:
        code, hint = 5, None
    else:
        code, hint = 1, None
    if retry_after is not None and status in (429, 503):
        hint = f"повтор через {retry_after:g} с (Retry-After)"
    return ServiceError(
        f"сервис ответил {label}: {message}",
        status=status,
        error_type=error_type,
        numbers=numbers,
        retry_after=retry_after,
        hint=hint,
        exit_code=code,
    )


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SSEEvent:
    event: str | None
    data: str


class SSEParser:
    """Incremental SSE: bytes in (any split), complete events out.

    A UTF-8 decoder with state keeps a character torn across chunks whole; the text
    buffer keeps a torn `data:` line; events end at a blank line, so several events
    in one chunk come out together. A trailing event without its blank line is not
    an event (per the SSE spec) and shows up as `pending` after `close()`.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buf = ""
        self.pending = False

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        text = self._decoder.decode(chunk)
        # A lone trailing "\r" may be the first half of "\r\n": hold it back.
        self._buf += text
        hold = ""
        if self._buf.endswith("\r"):
            self._buf, hold = self._buf[:-1], "\r"
        self._buf = self._buf.replace("\r\n", "\n").replace("\r", "\n") + hold
        events: list[SSEEvent] = []
        while True:
            head, sep, tail = self._buf.partition("\n\n")
            if not sep:
                break
            self._buf = tail
            event = self._event(head)
            if event is not None:
                events.append(event)
        return events

    def close(self) -> None:
        self._buf += self._decoder.decode(b"", final=True)
        self.pending = bool(self._buf.strip())

    @staticmethod
    def _event(block: str) -> SSEEvent | None:
        name: str | None = None
        data: list[str] = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            field_name, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field_name == "data":
                data.append(value)
            elif field_name == "event":
                name = value
        return SSEEvent(name, "\n".join(data)) if data else None


# ---------------------------------------------------------------------------
# result and status
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ServiceResult:
    text: str = ""
    status: int = 200
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    queue_wait_ms: int | None = None
    cold_start_ms: int | None = None
    input_tokens: int | None = None
    wall_ms: float = 0.0
    ttft_ms: float | None = None
    stream: bool = True
    incomplete: bool = False
    incomplete_reason: str | None = None

    @property
    def tokens_per_second(self) -> float | None:
        tokens = self.usage.completion_tokens
        if tokens is None or tokens < 2 or self.ttft_ms is None:
            return None
        window = (self.wall_ms - self.ttft_ms) / 1000
        return (tokens - 1) / window if window > 0 else None


def _header_ms(response: httpx.Response, name: str) -> int | None:
    """Absent header -> None (unknown); a present one, 0 included, is a measurement."""
    raw = response.headers.get(name)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _pick(source: Any, *paths: str) -> Any:
    """First present value among dotted `paths` of nested dicts; None when none is."""
    for path in paths:
        node = source
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
    return None


@dataclass(slots=True)
class ServiceStatus:
    """`/v1/status` as the client reads it. Field names are tolerant: the spec fixes
    the content (§2.5), not the JSON keys, so each value is looked up under a few names."""

    raw: dict[str, Any]
    state: str | None = None
    version: str | None = None
    rss_mb: float | None = None
    uptime_s: float | None = None
    idle_unload_s: float | None = None
    queue_waiting: int | None = None
    queue_active: int | None = None
    queue_max: int | None = None
    rpm: float | None = None
    burst: float | None = None
    bucket_tokens: float | None = None

    @classmethod
    def from_json(cls, data: Any) -> ServiceStatus:
        if not isinstance(data, dict):
            raise ServiceError("сервис вернул статус не объектом", status=200, exit_code=5)
        model = data.get("model")
        state = (
            model if isinstance(model, str) else _pick(data, "model.state", "model_state", "state")
        )
        rss = _pick(data, "model.rss_mb", "rss_mb", "model.rss.mb")
        if rss is None and (raw_bytes := _pick(data, "model.rss_bytes", "rss_bytes")) is not None:
            rss = raw_bytes / 1_048_576
        queue = data.get("queue")
        waiting = (
            queue
            if isinstance(queue, int)
            else _pick(data, "queue.waiting", "queue.queued", "queue.length")
        )
        return cls(
            raw=data,
            state=state if isinstance(state, str) else None,
            version=_pick(data, "version"),
            rss_mb=rss,
            uptime_s=_pick(
                data, "model.uptime_s", "model_uptime_s", "uptime_s", "model.uptime_seconds"
            ),
            idle_unload_s=_pick(
                data,
                "model.idle_unload_in_s",
                "idle_unload_in_s",
                "model.idle_remaining_s",
                "idle_remaining_s",
            ),
            queue_waiting=waiting,
            queue_active=_pick(data, "queue.active", "queue.running", "active"),
            queue_max=_pick(data, "queue.max", "queue.queue_max", "queue_max"),
            rpm=_pick(data, "limits.rpm", "key.rpm", "rate_limit.rpm"),
            burst=_pick(data, "limits.burst", "key.burst", "rate_limit.burst"),
            bucket_tokens=_pick(
                data,
                "limits.bucket_tokens",
                "limits.bucket",
                "limits.tokens",
                "key.tokens_left",
                "key.bucket_tokens",
                "rate_limit.bucket_tokens",
                "rate_limit.tokens",
            ),
        )

    @property
    def bucket_full(self) -> bool | None:
        """None = the status does not say; the stress precondition then cannot be proven."""
        if self.bucket_tokens is None or self.burst is None:
            return None
        return self.bucket_tokens >= self.burst - 0.01

    @property
    def queue_empty(self) -> bool | None:
        if self.queue_waiting is None and self.queue_active is None:
            return None
        return not (self.queue_waiting or 0) and not (self.queue_active or 0)


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


class ServiceClient:
    """Async client; `transport` is the test seam (httpx.MockTransport)."""

    def __init__(
        self,
        target: ServiceTarget,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = READ_TIMEOUT,
        max_connections: int = 32,
    ) -> None:
        self.target = target
        self._http = httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout, connect=15.0),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
                keepalive_expiry=60.0,
            ),
        )

    async def __aenter__(self) -> ServiceClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._http.aclose()

    def _url(self, path: str) -> str:
        return f"{self.target.origin}{path}"

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.target.key}"}

    def _network_error(self, exc: Exception) -> NetworkError:
        text = redact(f"{type(exc).__name__}: {exc}").replace(self.target.key, "***")
        return NetworkError(f"сервис недоступен: {text}", hint="проверь сеть и адрес сервиса")

    async def warm(self, count: int) -> int:
        """Open `count` idle keep-alive connections without touching a key's bucket.

        Plain `GET /` (the anonymous chat page, no Authorization) from `count` tasks at
        once: with no idle connection yet, each task opens its own. Returns how many
        answered; the rest simply connect later.
        """

        async def one() -> bool:
            try:
                response = await self._http.get(self._url("/"))
                await response.aread()
                return True
            except httpx.HTTPError:
                return False

        return sum(await asyncio.gather(*(one() for _ in range(count))))

    async def status(self) -> ServiceStatus:
        try:
            response = await self._http.get(self._url("/v1/status"), headers=self._auth())
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from None
        if response.status_code != 200:
            raise parse_error(response, response.content, self.target.key)
        try:
            return ServiceStatus.from_json(response.json())
        except ValueError:
            raise ServiceError("сервис вернул нечитаемый статус", status=200, exit_code=5) from None

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        stream: bool = True,
        on_content: Callable[[str], None] | None = None,
    ) -> ServiceResult:
        """One completion. Pre-header refusals raise ServiceError/NetworkError; a stream that
        dies after HTTP 200 returns `incomplete=True` with whatever text had arrived."""
        body: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        started = time.monotonic()
        result = ServiceResult(stream=stream)
        try:
            request = self._http.build_request(
                "POST", self._url("/v1/chat/completions"), json=body, headers=self._auth()
            )
            response = await self._http.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from None
        try:
            if response.status_code != 200:
                try:
                    payload = await response.aread()
                except httpx.HTTPError:
                    payload = b""
                raise parse_error(response, payload, self.target.key)
            result.status = 200
            result.queue_wait_ms = _header_ms(response, "x-advent-queue-wait-ms")
            result.cold_start_ms = _header_ms(response, "x-advent-cold-start-ms")
            result.input_tokens = _header_ms(response, "x-advent-input-tokens")
            if stream:
                await self._read_stream(response, result, on_content, started)
            else:
                await self._read_json(response, result, on_content, started)
        finally:
            await response.aclose()
        result.wall_ms = (time.monotonic() - started) * 1000
        return result

    async def _read_json(
        self,
        response: httpx.Response,
        result: ServiceResult,
        on_content: Callable[[str], None] | None,
        started: float,
    ) -> None:
        try:
            data = json.loads(await response.aread())
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except httpx.HTTPError as exc:
            result.incomplete, result.incomplete_reason = True, f"network: {type(exc).__name__}"
            return
        except (ValueError, KeyError, IndexError, TypeError):
            result.incomplete, result.incomplete_reason = True, "bad_response"
            return
        result.ttft_ms = (time.monotonic() - started) * 1000
        result.text, result.finish_reason = text, choice.get("finish_reason")
        result.model = data.get("model")
        result.usage = Usage.from_raw(data.get("usage"))
        if on_content and text:
            on_content(text)

    async def _read_stream(
        self,
        response: httpx.Response,
        result: ServiceResult,
        on_content: Callable[[str], None] | None,
        started: float,
    ) -> None:
        parser = SSEParser()
        parts: list[str] = []
        done = False

        def handle(event: SSEEvent) -> bool:
            """True when the stream is over (DONE or an error event)."""
            if event.data.strip() == "[DONE]":
                return True
            try:
                obj = json.loads(event.data)
            except ValueError:
                result.incomplete, result.incomplete_reason = True, "bad_event"
                return True
            if not isinstance(obj, dict):
                return False
            if isinstance(obj.get("error"), dict):
                err = obj["error"]
                kind = next(
                    (err[k] for k in ("code", "type") if isinstance(err.get(k), str)), "error"
                )
                result.incomplete, result.incomplete_reason = True, kind
                return True
            if isinstance(obj.get("model"), str):
                result.model = obj["model"]
            if obj.get("usage"):
                result.usage = Usage.from_raw(obj["usage"])
            for choice in obj.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if result.ttft_ms is None:
                        result.ttft_ms = (time.monotonic() - started) * 1000
                    parts.append(piece)
                    result.text = "".join(parts)
                    if on_content:
                        on_content(piece)
                if choice.get("finish_reason"):
                    result.finish_reason = choice["finish_reason"]
            return False

        try:
            async for chunk in response.aiter_bytes():
                for event in parser.feed(chunk):
                    if handle(event):
                        done = True
                        break
                if done:
                    break
        except httpx.HTTPError as exc:
            result.incomplete, result.incomplete_reason = True, f"network: {type(exc).__name__}"
            return
        if not done:
            parser.close()
            result.incomplete = True
            result.incomplete_reason = "stream_cut"
        elif result.incomplete:
            return
        elif result.finish_reason is None:
            result.incomplete, result.incomplete_reason = True, "no_finish_reason"


def run(coro):
    """Sync entry for the CLI (one `asyncio.run` per command or REPL turn)."""
    return asyncio.run(coro)
