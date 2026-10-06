"""Thin httpx client for an OpenAI-compatible local server (LM Studio).

Not the Mistral SDK: in streaming it drops usage, and the day is about the HTTP
API being visible in code.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from advent_core.errors import (
    AdventError,
    AuthError,
    ConfigurationError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from advent_core.telemetry import Usage

DEFAULT_URL = "http://127.0.0.1:1234"
DEFAULT_MODEL = "ornith"
# Model card "precise" mode; repeat_penalty is left to the server.
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 8192
READ_TIMEOUT = 300.0
DEADLINE = 600.0
START_HINT = "запусти LM Studio: start-local-llm.ps1 -Context 57344"

Callback = Callable[[str], None]

# Seam for tests: a controlled clock gives exact ttft/content timestamps.
_clock = time.monotonic


class ProtocolError(ServerError):
    """Malformed SSE stream: the server spoke, but not the protocol."""


@dataclass(slots=True, frozen=True)
class LocalModel:
    id: str
    type: str | None = None
    # None = unknown (fallback to /v1/models: not LM Studio).
    state: str | None = None
    loaded_context_length: int | None = None
    quantization: str | None = None
    arch: str | None = None

    @property
    def loaded(self) -> bool:
        return self.state == "loaded"


@dataclass(slots=True)
class LocalResult:
    text: str
    reasoning_text: str
    usage: Usage | None
    latency_ms: float
    # to the first generated delta (reasoning or content) / to the first content.
    ttft_ms: float | None
    content_ms: float | None
    finish_reason: str | None
    model: str
    truncated: bool = False
    chunks: int = field(default=0, repr=False)

    @property
    def cutoff_note(self) -> str | None:
        if self.finish_reason == "length" and not self.text.strip():
            return "обрыв: reasoning съел max_tokens"
        if self.truncated:
            return "обрыв: поток закончился раньше времени"
        return None

    @property
    def tokens_per_second(self) -> float | None:
        """completion / (latency - ttft); None when unknown or denominator <= 0."""
        if self.usage is None or self.usage.completion_tokens is None or self.ttft_ms is None:
            return None
        window = (self.latency_ms - self.ttft_ms) / 1000
        if window <= 0:
            return None
        return self.usage.completion_tokens / window


def default_url() -> str:
    return (os.environ.get("ADVENT_LOCAL_URL") or DEFAULT_URL).rstrip("/")


def _make_client(timeout: float) -> httpx.Client:
    # Seam for tests: they replace this with a MockTransport-backed client.
    return httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))


def _network_error(exc: Exception, url: str) -> NetworkError:
    if isinstance(exc, httpx.TimeoutException):
        return NetworkError(f"Локальный сервер {url} не ответил вовремя.", hint=START_HINT)
    return NetworkError(f"Локальный сервер {url} не отвечает.", hint=START_HINT)


def server_status(url: str | None = None, *, timeout: float = 10.0) -> list[LocalModel]:
    """Models from LM Studio /api/v0/models; fallback to /v1/models only on 404."""
    base = (url or default_url()).rstrip("/")
    client = _make_client(timeout)
    try:
        resp = client.get(f"{base}/api/v0/models")
        if resp.status_code == 404:
            resp = client.get(f"{base}/v1/models")
            return _parse_models(resp, base, lm_studio=False)
        return _parse_models(resp, base, lm_studio=True)
    except httpx.HTTPError as exc:
        raise _network_error(exc, base) from exc
    finally:
        client.close()


def _parse_models(resp: httpx.Response, base: str, *, lm_studio: bool) -> list[LocalModel]:
    if resp.status_code in (401, 403):
        raise AuthError(f"Сервер {base} отклонил запрос ({resp.status_code}).")
    if resp.status_code != 200:
        raise ServerError(f"Сервер {base} вернул {resp.status_code} на список моделей.")
    try:
        body = resp.json()
    except ValueError as exc:
        raise ProtocolError(f"Сервер {base} вернул не JSON на список моделей.") from exc
    data = body.get("data", []) if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise ProtocolError(f"Сервер {base} вернул список моделей неожиданной формы.")
    models = []
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        models.append(
            LocalModel(
                id=str(item["id"]),
                type=item.get("type"),
                state=item.get("state") if lm_studio else None,
                loaded_context_length=item.get("loaded_context_length") if lm_studio else None,
                quantization=item.get("quantization") if lm_studio else None,
                arch=item.get("arch") if lm_studio else None,
            )
        )
    # loaded first, stable otherwise
    return sorted(models, key=lambda m: not m.loaded)


def check_ready(models: list[LocalModel], model: str, base: str) -> LocalModel:
    """The model must be listed and (when the server reports state) loaded."""
    found = next((m for m in models if m.id == model), None)
    if found is None:
        raise ConfigurationError(f"Модели {model!r} нет в списке сервера {base}.", hint=START_HINT)
    if found.state is not None and not found.loaded:
        raise ConfigurationError(
            f"Модель {model!r} не загружена (state: {found.state}).", hint=START_HINT
        )
    return found


def ensure_ready(model: str, url: str | None = None) -> LocalModel:
    base = (url or default_url()).rstrip("/")
    return check_ready(server_status(base), model, base)


def _http_error(resp: httpx.Response, model: str) -> AdventError:
    status = resp.status_code
    try:
        body = resp.read().decode("utf-8", errors="replace")[:500]
    except httpx.HTTPError:
        body = ""
    if status in (400, 404) and ("model" in body.lower() or status == 404):
        return ConfigurationError(
            f"модель не загружена: {model!r} ({status}).",
            hint="загрузи её в LM Studio: " + START_HINT,
        )
    if status in (401, 403):
        return AuthError(f"Локальный сервер отклонил запрос ({status}).")
    if status == 429:
        return RateLimitError("Локальный сервер перегружен (429).")
    if 500 <= status < 600:
        return ServerError(f"Локальный сервер вернул {status}: {body}")
    return AdventError(f"Локальный сервер ответил {status}: {body}")


def chat(
    url: str | None,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = MAX_TOKENS,
    temperature: float | None = TEMPERATURE,
    top_p: float | None = TOP_P,
    on_reasoning: Callback | None = None,
    on_content: Callback | None = None,
    timeout: float = READ_TIMEOUT,
    deadline: float = DEADLINE,
) -> LocalResult:
    """POST /v1/chat/completions with stream + include_usage."""
    base = (url or default_url()).rstrip("/")
    payload: dict = {
        "model": model,
        "messages": list(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    text: list[str] = []
    reasoning: list[str] = []
    usage: Usage | None = None
    finish: str | None = None
    done = False
    chunks = 0
    ttft: float | None = None
    content_at: float | None = None
    start = _clock()

    def elapsed_ms() -> float:
        return (_clock() - start) * 1000

    client = _make_client(timeout)
    # The read runs in a worker so the absolute deadline also bounds waiting for
    # headers, the next chunk and EOF, not just the gap between two lines.
    inbox: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            with client.stream("POST", f"{base}/v1/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    inbox.put(("exc", _http_error(resp, model)))
                    return
                for raw in resp.iter_lines():
                    inbox.put(("line", raw))
            inbox.put(("end", None))
        except BaseException as exc:  # forwarded to the caller thread
            inbox.put(("exc", exc))

    worker = threading.Thread(target=pump, daemon=True)
    worker.start()
    try:
        while True:
            remaining = deadline - (_clock() - start)
            if remaining <= 0:
                break
            try:
                kind, raw = inbox.get(timeout=remaining)
            except queue.Empty:
                break
            if kind == "exc":
                raise raw
            if kind == "end":
                break
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            try:
                event = json.loads(data)
            except ValueError as exc:
                raise ProtocolError(f"Битая строка в потоке сервера: {data[:80]!r}") from exc
            if not isinstance(event, dict):
                raise ProtocolError(f"Неожиданный chunk в потоке: {data[:80]!r}")
            if "error" in event:
                err = event["error"]
                msg = err.get("message") if isinstance(err, dict) else err
                raise ServerError(f"Сервер прислал ошибку в потоке: {msg}")
            chunks += 1
            raw_usage = event.get("usage")
            if raw_usage:
                if not isinstance(raw_usage, dict):
                    raise ProtocolError(f"usage неожиданной формы: {str(raw_usage)[:80]!r}")
                usage = Usage.from_raw(raw_usage)
            choices = event.get("choices")
            if choices is None:
                choices = []
            if not isinstance(choices, list):
                raise ProtocolError(f"choices неожиданной формы: {str(choices)[:80]!r}")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ProtocolError(f"choice неожиданной формы: {str(choice)[:80]!r}")
                delta = choice.get("delta")
                if delta is None:
                    delta = {}
                if not isinstance(delta, dict):
                    raise ProtocolError(f"delta неожиданной формы: {str(delta)[:80]!r}")
                r = delta.get("reasoning_content")
                c = delta.get("content")
                for part in (r, c):
                    if part is not None and not isinstance(part, str):
                        raise ProtocolError(f"текст delta не строка: {str(part)[:80]!r}")
                if r:
                    if ttft is None:
                        ttft = elapsed_ms()
                    reasoning.append(r)
                    if on_reasoning:
                        on_reasoning(r)
                if c:
                    if ttft is None:
                        ttft = elapsed_ms()
                    if content_at is None:
                        content_at = elapsed_ms()
                    text.append(c)
                    if on_content:
                        on_content(c)
                reason = choice.get("finish_reason")
                if reason:
                    if not isinstance(reason, str):
                        raise ProtocolError(f"finish_reason не строка: {str(reason)[:80]!r}")
                    finish = reason
        # Deadline, EOF or a dropped stream without a finish marker all end up here.
        truncated = not done and finish is None
    except httpx.HTTPError as exc:
        raise _network_error(exc, base) from exc
    finally:
        # The worker may still hold the response (deadline case): closing can refuse.
        with contextlib.suppress(Exception):
            client.close()

    return LocalResult(
        text="".join(text),
        reasoning_text="".join(reasoning),
        usage=usage,
        latency_ms=elapsed_ms(),
        ttft_ms=ttft,
        content_ms=content_at,
        finish_reason=finish,
        model=model,
        truncated=truncated,
        chunks=chunks,
    )
