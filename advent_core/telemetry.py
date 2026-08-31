"""Результат одного вызова LLM: текст, usage, latency."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_raw(cls, raw: object) -> Usage:
        """Собирает usage и из dict, и из pydantic-модели SDK."""
        if raw is None:
            return cls()
        get = raw.get if isinstance(raw, dict) else lambda k: getattr(raw, k, None)
        return cls(
            prompt_tokens=get("prompt_tokens"),
            completion_tokens=get("completion_tokens"),
            total_tokens=get("total_tokens"),
        )

    def is_empty(self) -> bool:
        return self.total_tokens is None and self.prompt_tokens is None


@dataclass(slots=True)
class CallResult:
    text: str = ""
    model_requested: str = ""
    model_actual: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    stream: bool = True
    truncated: bool = False
    # Параметры, не отправленные из-за отсутствующей capability модели.
    skipped_params: list[str] = field(default_factory=list)
