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
    # stop | length | model_length | error | tool_calls (complete()); в
    # stream() набор без model_length — SDK его в чанках не присылает
    # (SPEC-w01d02.md §2). None — SDK не вернул значение вовсе.
    finish_reason: str | None = None
    # Вердикт по формату из formats.verify(): True/False — проверено,
    # None — для text/yaml/md вердикт принципиально не выносится (нет ни
    # API-гарантии, ни дешёвой верификации).
    format_ok: bool | None = None
    # Человекочитаемая расшифровка вердикта для footer, например
    # "JSON ✓ · схема ✓ · items: 3" или "—" для форматов без вердикта.
    format_detail: str | None = None
    # Сообщения, фактически ушедшие в Mistral — то есть messages ДО этой
    # функции плюс инструкция пресета формата, дописанная chat._payload().
    # Журнал (week_01/cli.py → log_call) исторически писал в logs/calls.jsonl
    # тот messages, что был построен ДО дописывания инструкции формата —
    # source of truth для недель 2/5 расходился с тем, что реально видела
    # модель. None — только когда вызов не дошёл до _payload() вовсе
    # (например, CallResult(model_requested=...) для error-веток в cli.py,
    # где messages для лога и так есть отдельно).
    sent_messages: list[dict[str, str]] | None = None
