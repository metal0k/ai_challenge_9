"""Thin httpx client for an OpenAI-compatible local server (LM Studio).

Not the Mistral SDK: in streaming it sends no `stream_options` (no usage), and it
cannot parse LM Studio's embeddings response. The payload comes from the caller
(chat._payload() for the agent), this module only speaks HTTP and validates what
comes back. Error contract: SPEC-w06d26 section 9a (7, 8), SPEC-w06d27 section 9a (12, 16).
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
from typing import TYPE_CHECKING

import httpx
import numpy as np

from advent_core import config as config_module
from advent_core import offline
from advent_core.config import LOCAL_API_KEY, normalize_base_url
from advent_core.errors import (
    AdventError,
    AuthError,
    ConfigurationError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from advent_core.telemetry import RawToolCall, Usage

DEFAULT_URL = "http://127.0.0.1:1234"
READ_TIMEOUT = 300.0
DEADLINE = 600.0
EMBED_BATCH = 16
START_HINT = "запусти LM Studio: start-local-llm.ps1 -Context 57344"

if TYPE_CHECKING:
    from advent_core.embeddings import EmbedResult

Callback = Callable[[str], None]

# Seam for tests: a controlled clock gives exact ttft/content timestamps.
_clock = time.monotonic


class ProtocolError(ServerError):
    """Malformed response: the server spoke, but not the protocol."""


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
    tool_calls: tuple[RawToolCall, ...] = ()

    @property
    def cutoff_note(self) -> str | None:
        if self.finish_reason == "length" and not self.text.strip() and not self.tool_calls:
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
    """ADVENT_LOCAL_URL (also from .env, loaded here) or the LM Studio default, normalized."""
    config_module.load_env()  # attribute access: tests patch it
    return normalize_base_url(os.environ.get("ADVENT_LOCAL_URL")) or DEFAULT_URL


def _make_client(timeout: float, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    # Seam for tests: they replace this with a MockTransport-backed client.
    # No proxy from the environment and no redirects: a local call must not be
    # re-routed anywhere. The Authorization value is a stub, never a real key.
    offline.init_from_env()
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=10.0),
        trust_env=False,
        follow_redirects=False,
        headers={"Authorization": f"Bearer {LOCAL_API_KEY}"},
        transport=transport,
    )


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


def check_ready(
    models: list[LocalModel], model: str, base: str, *, require_state: bool = False
) -> LocalModel:
    """The model must be listed and (when the server reports state) loaded.

    require_state: a server without `state` (the /v1/models fallback, i.e. not
    LM Studio) is refused instead of passing with "unknown".
    """
    found = next((m for m in models if m.id == model), None)
    if found is None:
        raise ConfigurationError(f"Модели {model!r} нет в списке сервера {base}.", hint=START_HINT)
    if found.state is None and require_state:
        raise ConfigurationError(
            f"Сервер {base} не отдаёт state моделей (не LM Studio?) — готовность не проверить.",
            hint=START_HINT,
        )
    if found.state is not None and not found.loaded:
        raise ConfigurationError(
            f"Модель {model!r} не загружена (state: {found.state}).", hint=START_HINT
        )
    return found


def ensure_ready(model: str, url: str | None = None, *, require_state: bool = False) -> LocalModel:
    base = (url or default_url()).rstrip("/")
    return check_ready(server_status(base), model, base, require_state=require_state)


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


def _shape(value: object, what: str) -> ProtocolError:
    return ProtocolError(f"{what} неожиданной формы: {str(value)[:80]!r}")


def _count(value: object, what: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{what} не неотрицательное целое: {str(value)[:40]!r}")
    return value


def _parse_usage(raw: object) -> Usage:
    """Usage with every numeric field validated: a string must not reach token accounting."""
    if not isinstance(raw, dict):
        raise _shape(raw, "usage")
    for key in ("prompt_tokens_details", "completion_tokens_details"):
        if raw.get(key) is not None and not isinstance(raw[key], dict):
            raise _shape(raw[key], key)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        _count(raw.get(key), f"usage.{key}")
    for container, key in (
        ("prompt_tokens_details", "cached_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    ):
        _count((raw.get(container) or {}).get(key), f"usage.{container}.{key}")
    return Usage.from_raw(raw)


def _wire_payload(payload: dict) -> dict:
    """Mistral-shaped response_format -> what LM Studio accepts (json_schema/text only).

    json_object is dropped (the prompt already demands JSON); the Mistral-only
    `schema_definition` key is renamed to `schema`. The caller's dict is not mutated.
    """
    rf = payload.get("response_format")
    if not isinstance(rf, dict):
        return payload
    out = dict(payload)
    if rf.get("type") == "json_object":
        del out["response_format"]
    elif rf.get("type") == "json_schema" and isinstance(rf.get("json_schema"), dict):
        js = dict(rf["json_schema"])
        if "schema_definition" in js:
            js["schema"] = js.pop("schema_definition")
        out["response_format"] = {**rf, "json_schema": js}
    return out


def chat_stream(
    url: str | None,
    payload: dict,
    *,
    on_reasoning: Callback | None = None,
    on_content: Callback | None = None,
    timeout: float = READ_TIMEOUT,
    deadline: float = DEADLINE,
) -> LocalResult:
    """POST /v1/chat/completions with stream + include_usage.

    `payload` is the caller's request body (model, messages, params); stream and
    stream_options are added here. Deadline, EOF without a finish marker, a drop
    after partial output and Ctrl+C all return what arrived with truncated=True.
    """
    base = (url or default_url()).rstrip("/")
    model = payload.get("model", "")
    body = {**_wire_payload(payload), "stream": True, "stream_options": {"include_usage": True}}

    text: list[str] = []
    reasoning: list[str] = []
    usage: Usage | None = None
    finish: str | None = None
    done = False
    truncated = False
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
            with client.stream("POST", f"{base}/v1/chat/completions", json=body) as resp:
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
                usage = _parse_usage(raw_usage)
            choices = event.get("choices")
            if choices is None:
                choices = []
            if not isinstance(choices, list):
                raise _shape(choices, "choices")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise _shape(choice, "choice")
                delta = choice.get("delta")
                if delta is None:
                    delta = {}
                if not isinstance(delta, dict):
                    raise _shape(delta, "delta")
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
        if done and finish is None and not (text or reasoning):
            # [DONE] with nothing generated (only usage events / empty choices): the
            # non-stream path refuses an empty completion too.
            raise ProtocolError("Сервер закончил поток без ответа (пустые choices).")
        # Deadline, EOF or a dropped stream without a finish marker all end up here.
        truncated = not done and finish is None
    except KeyboardInterrupt:
        truncated = True
    except httpx.HTTPError as exc:
        if not (text or reasoning):
            raise _network_error(exc, base) from exc
        # Part of the answer is already on screen: hand it over, do not lose it.
        truncated = True
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


def _post_json(base: str, path: str, body: dict, *, timeout: float, deadline: float, model: str):
    """POST in a worker with an absolute deadline; returns the finished httpx.Response."""
    client = _make_client(timeout)
    inbox: queue.Queue = queue.Queue(maxsize=1)

    def pump() -> None:
        try:
            inbox.put(("resp", client.post(f"{base}{path}", json=body)))
        except BaseException as exc:  # forwarded to the caller thread
            inbox.put(("exc", exc))

    threading.Thread(target=pump, daemon=True).start()
    try:
        try:
            kind, value = inbox.get(timeout=deadline)
        except queue.Empty:
            raise NetworkError(
                f"Локальный сервер {base} не ответил за {deadline:.0f} с.", hint=START_HINT
            ) from None
        if kind == "exc":
            raise value
        if value.status_code != 200:
            raise _http_error(value, model)
        return value
    except httpx.HTTPError as exc:
        raise _network_error(exc, base) from exc
    finally:
        with contextlib.suppress(Exception):
            client.close()


def _json_body(resp: httpx.Response, what: str) -> dict:
    try:
        body = resp.json()
    except ValueError as exc:
        raise ProtocolError(f"Сервер вернул не JSON ({what}).") from exc
    if not isinstance(body, dict):
        raise _shape(body, what)
    if "error" in body:
        err = body["error"]
        msg = err.get("message") if isinstance(err, dict) else err
        raise ServerError(f"Сервер прислал ошибку: {msg}")
    return body


def _parse_tool_calls(raw: object) -> tuple[RawToolCall, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _shape(raw, "tool_calls")
    calls = []
    for item in raw:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            raise _shape(item, "tool_call")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError(f"tool_call без имени функции: {str(item)[:80]!r}")
        arguments = function.get("arguments", "")
        if arguments is None:
            arguments = ""
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            raise _shape(item, "arguments tool_call")
        call_id = item.get("id")
        if call_id is not None and not isinstance(call_id, str):
            raise _shape(item, "id tool_call")
        calls.append(RawToolCall(id=call_id or "", name=name, arguments=arguments))
    return tuple(calls)


def chat_complete(
    url: str | None,
    payload: dict,
    *,
    timeout: float = READ_TIMEOUT,
    deadline: float = DEADLINE,
) -> LocalResult:
    """Non-stream POST /v1/chat/completions; `payload` may carry tools/tool_choice."""
    base = (url or default_url()).rstrip("/")
    model = payload.get("model", "")
    body = {**_wire_payload(payload), "stream": False}
    start = _clock()
    resp = _post_json(
        base, "/v1/chat/completions", body, timeout=timeout, deadline=deadline, model=model
    )
    latency = (_clock() - start) * 1000
    data = _json_body(resp, "chat completion")
    choices = data.get("choices")
    if not isinstance(choices, list):
        raise _shape(choices, "choices")
    if not choices:
        raise ProtocolError("Сервер вернул ответ без вариантов (пустые choices).")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        raise _shape(choice, "choice")
    content = message.get("content")
    if content is None:
        content = ""
    elif isinstance(content, list):
        pieces = []
        for part in content:
            if not isinstance(part, dict):
                raise _shape(part, "блок content")
            text_part = part.get("text")
            if text_part is not None and not isinstance(text_part, str):
                raise ProtocolError(f"text блока content не строка: {str(text_part)[:40]!r}")
            pieces.append(text_part or "")
        content = "".join(pieces)
    elif not isinstance(content, str):
        raise _shape(content, "content")
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise _shape(reasoning, "reasoning_content")
    finish = choice.get("finish_reason")
    if finish is not None and not isinstance(finish, str):
        raise _shape(finish, "finish_reason")
    raw_usage = data.get("usage")
    usage = _parse_usage(raw_usage) if raw_usage is not None else None
    if raw_usage == {}:
        usage = None
    actual = data.get("model")
    return LocalResult(
        text=content,
        reasoning_text=reasoning or "",
        usage=usage,
        latency_ms=latency,
        ttft_ms=None,
        content_ms=None,
        finish_reason=finish,
        model=actual if isinstance(actual, str) and actual else model,
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
    )


def _validate_embeddings(
    data: object, n_inputs: int, expected_dim: int | None, dim_box: list[int | None]
) -> np.ndarray:
    """Strict adapter for a raw /v1/embeddings body: (n_inputs, dim) float32 in input order."""
    if not isinstance(data, list):
        raise _shape(data, "data эмбеддингов")
    if len(data) != n_inputs:
        raise ProtocolError(f"Эмбеддинг вернул {len(data)} векторов вместо {n_inputs}.")
    rows: dict[int, list] = {}
    for item in data:
        if not isinstance(item, dict):
            raise _shape(item, "элемент эмбеддинга")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ProtocolError(f"У вектора нет целого index: {str(index)[:40]!r}.")
        vector = item.get("embedding")
        if not isinstance(vector, list):
            raise _shape(vector, "embedding")
        if index in rows:
            raise ProtocolError(f"Повторный index эмбеддинга: {index}.")
        rows[index] = vector
    if sorted(rows) != list(range(n_inputs)):
        raise ProtocolError(f"Индексы эмбеддинга не образуют 0..{n_inputs - 1}: {sorted(rows)}.")
    dims = {len(v) for v in rows.values()}
    if len(dims) != 1:
        raise ProtocolError("Векторы одного батча имеют разную размерность.")
    (dim,) = dims
    for want in (expected_dim, dim_box[0]):
        if want is not None and want != dim:
            raise ProtocolError(f"Размерность эмбеддинга {dim}, ожидалась {want}.")
    dim_box[0] = dim
    for vector in rows.values():
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProtocolError(f"В векторе не число: {str(value)[:40]!r}.")
    # Finiteness is checked AFTER the cast: 1e40 is finite as a float, Inf as float32.
    try:
        with np.errstate(over="ignore"):
            matrix = np.array([rows[i] for i in range(n_inputs)], dtype=np.float32)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ProtocolError(f"Вектор не приводится к float32: {exc}") from exc
    if not np.isfinite(matrix).all():
        raise ProtocolError("Эмбеддинг содержит нечисловое значение (NaN/Inf).")
    # float64: a float32 norm overflows at 1e30 and underflows at 1e-30.
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ProtocolError("Эмбеддинг нулевой или бесконечной длины.")
    return matrix


def embed(
    url: str | None,
    model: str,
    texts: list[str],
    *,
    batch: int = EMBED_BATCH,
    expected_dim: int | None = None,
    on_batch: Callable[[int, int], None] | None = None,
    timeout: float = READ_TIMEOUT,
    deadline: float = DEADLINE,
) -> EmbedResult:
    """Embed `texts` through /v1/embeddings; row i of the result is texts[i], L2-normalized.

    Raw HTTP, strictly validated (`_validate_embeddings`); the Mistral SDK cannot
    parse LM Studio's response. `deadline` bounds the whole call across batches.
    """
    # Lazy: embeddings -> journal -> chat -> this module would be a cycle.
    from advent_core.embeddings import EmbedResult

    base = (url or default_url()).rstrip("/")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    start = _clock()
    parts: list[np.ndarray] = []
    dim_box: list[int | None] = [expected_dim]
    prompt_tokens: int | None = 0
    requests = 0
    actual = model
    for lo in range(0, len(texts), batch):
        remaining = deadline - (_clock() - start)
        if remaining <= 0:
            raise NetworkError(
                f"Локальный сервер {base} не уложился в {deadline:.0f} с на эмбеддинги.",
                hint=START_HINT,
            )
        chunk = texts[lo : lo + batch]
        resp = _post_json(
            base,
            "/v1/embeddings",
            {"model": model, "input": chunk},
            timeout=timeout,
            deadline=remaining,
            model=model,
        )
        requests += 1
        data = _json_body(resp, "embeddings")
        parts.append(_validate_embeddings(data.get("data"), len(chunk), expected_dim, dim_box))
        raw_usage = data.get("usage")
        used = raw_usage.get("prompt_tokens") if isinstance(raw_usage, dict) else None
        if isinstance(used, int) and not isinstance(used, bool) and prompt_tokens is not None:
            prompt_tokens += used
        else:
            prompt_tokens = None
        if isinstance(data.get("model"), str) and data["model"]:
            actual = data["model"]
        if on_batch:
            on_batch(min(lo + batch, len(texts)), len(texts))
    if parts:
        matrix = np.vstack(parts).astype(np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        vectors = (matrix / norms).astype(np.float32)
        if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
            raise ProtocolError("Эмбеддинг после нормализации нулевой или не конечный.")
    else:
        vectors = np.zeros((0, expected_dim or 0), dtype=np.float32)
        prompt_tokens = None
    return EmbedResult(
        vectors=vectors,
        model=actual,
        prompt_tokens=prompt_tokens,
        requests=requests,
        latency_ms=int((_clock() - start) * 1000),
    )
