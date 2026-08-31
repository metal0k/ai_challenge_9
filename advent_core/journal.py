"""JSONL-лог вызовов LLM.

Файл лежит в logs/ и в .gitignore: он содержит промпты и ответы целиком,
а значит не место в публичном репо.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from advent_core.chat import Message
from advent_core.config import LOG_DIR, redact
from advent_core.telemetry import CallResult


def log_call(
    result: CallResult,
    messages: list[Message],
    *,
    week: int,
    day: int | None = None,
    error: str | None = None,
    path: Path | None = None,
) -> None:
    """Дописывает одну строку в JSONL. Никогда не роняет вызывающий код.

    Логирование — побочный эффект демо; упавшая запись в лог не должна
    убивать уже полученный ответ на экране пользователя.
    """
    record = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "week": week,
        "day": day,
        "model_requested": result.model_requested,
        "model_actual": result.model_actual,
        "messages": [{"role": m["role"], "content": redact(m["content"])} for m in messages],
        "response": redact(result.text),
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "latency_ms": result.latency_ms,
        "stream": result.stream,
        "truncated": result.truncated,
        "error": redact(error) if error else None,
    }

    target = path or (LOG_DIR / "calls.jsonl")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
