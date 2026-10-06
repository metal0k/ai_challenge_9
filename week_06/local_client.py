"""adventlocal: day-26 sampling defaults on top of advent_core.openai_compat.

The HTTP, SSE and error contract live in `advent_core.openai_compat`; this module
keeps the names `week_06` imports and the sampling defaults of the day.
"""

from __future__ import annotations

from advent_core.openai_compat import (
    DEADLINE,
    DEFAULT_URL,
    READ_TIMEOUT,
    START_HINT,
    Callback,
    LocalModel,
    LocalResult,
    ProtocolError,
    chat_stream,
    check_ready,
    default_url,
    server_status,
)

__all__ = [
    "DEADLINE",
    "DEFAULT_MODEL",
    "DEFAULT_URL",
    "MAX_TOKENS",
    "READ_TIMEOUT",
    "START_HINT",
    "TEMPERATURE",
    "TOP_P",
    "Callback",
    "LocalModel",
    "LocalResult",
    "ProtocolError",
    "chat",
    "check_ready",
    "default_url",
    "ensure_ready",
    "server_status",
]

DEFAULT_MODEL = "ornith"
# Model card "precise" mode; repeat_penalty is left to the server.
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 8192


def ensure_ready(model: str, url: str | None = None, *, require_state: bool = False) -> LocalModel:
    # Defined here, not re-exported: it must call THIS module's server_status,
    # which is the seam week_06 tests replace.
    base = (url or default_url()).rstrip("/")
    return check_ready(server_status(base), model, base, require_state=require_state)


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
    """Streaming chat with adventlocal's sampling defaults."""
    payload: dict = {"model": model, "messages": list(messages), "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    return chat_stream(
        url,
        payload,
        on_reasoning=on_reasoning,
        on_content=on_content,
        timeout=timeout,
        deadline=deadline,
    )
