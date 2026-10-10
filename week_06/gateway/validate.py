"""Strict request validation (spec 10-3..5): everything is checked before the model starts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .errors import ApiError
from .settings import Settings

ALLOWED = {
    "messages", "max_tokens", "temperature", "top_p", "stop", "stream", "stream_options",
    "seed", "model",
}  # fmt: skip
ROLES = {"system", "user", "assistant"}


@dataclass
class ChatRequest:
    messages: list[dict]
    max_tokens: int
    stream: bool
    temperature: float
    top_p: float | None
    stop: list[str] | None
    seed: int | None

    def upstream_payload(self) -> dict:
        """Built field by field: nothing the client sent leaks through unchecked."""
        payload: dict = {
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.stop is not None:
            payload["stop"] = self.stop
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.stream:
            # Client stream_options are never read: usage in the last chunk is always wanted.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def counting_payload(self) -> dict:
        return {"messages": self.messages}


def _bad(code: str, message: str) -> ApiError:
    return ApiError(400, code, message)


def _is_number(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def parse_chat_request(raw: bytes, settings: Settings) -> ChatRequest:
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise _bad("invalid_json", "Тело запроса — не корректный JSON.") from None
    if not isinstance(data, dict):
        raise _bad("invalid_json", "Тело запроса должно быть JSON-объектом.")

    extra = sorted(set(data) - ALLOWED)
    if extra:
        raise _bad("unsupported_field", f"Неподдерживаемые поля: {', '.join(extra)}.")

    messages = _messages(data.get("messages"))

    mt = data.get("max_tokens")
    if mt is None:
        max_tokens = settings.default_max_tokens
    elif (
        isinstance(mt, int) and not isinstance(mt, bool) and 1 <= mt <= settings.max_tokens_ceiling
    ):
        max_tokens = mt
    else:
        raise _bad(
            "invalid_max_tokens",
            f"max_tokens должно быть целым числом от 1 до {settings.max_tokens_ceiling}.",
        )

    temperature = settings.default_temperature
    if "temperature" in data:
        t = data["temperature"]
        if not _is_number(t) or not 0 <= t <= 1.5:
            raise _bad("invalid_temperature", "temperature должна быть числом от 0 до 1.5.")
        temperature = float(t)

    top_p = None
    if "top_p" in data:
        p = data["top_p"]
        if not _is_number(p) or not 0 <= p <= 1:
            raise _bad("invalid_top_p", "top_p должно быть числом от 0 до 1.")
        top_p = float(p)

    stop = None
    if "stop" in data:
        s = data["stop"]
        items = [s] if isinstance(s, str) else s
        if (
            not isinstance(items, list)
            or len(items) > 4
            or not all(isinstance(i, str) and len(i) <= 32 for i in items)
        ):
            raise _bad("invalid_stop", "stop — строка или список до 4 строк по ≤ 32 символов.")
        stop = items

    seed = None
    if "seed" in data:
        sd = data["seed"]
        if not isinstance(sd, int) or isinstance(sd, bool):
            raise _bad("invalid_seed", "seed должно быть целым числом.")
        seed = sd

    stream = data.get("stream", False)
    if not isinstance(stream, bool):
        raise _bad("invalid_stream", "stream должно быть true или false.")

    return ChatRequest(messages, max_tokens, stream, temperature, top_p, stop, seed)


def _messages(msgs) -> list[dict]:
    if not isinstance(msgs, list) or not 1 <= len(msgs) <= 32:
        raise _bad("invalid_messages", "messages — список из 1..32 сообщений.")
    out = []
    for m in msgs:
        if (
            not isinstance(m, dict)
            or set(m) != {"role", "content"}
            or not isinstance(m["role"], str)
            or m["role"] not in ROLES
            or not isinstance(m["content"], str)
        ):
            raise _bad(
                "invalid_messages",
                "Каждое сообщение — ровно {role, content}; role: system|user|assistant, "
                "content — строка.",
            )
        out.append({"role": m["role"], "content": m["content"]})
    return out
