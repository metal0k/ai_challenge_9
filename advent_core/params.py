"""Параметры генерации: разбор, валидация и отсев неподдерживаемых моделью.

Отдельный модуль, потому что один и тот же набор нужен трём потребителям:
флагам CLI, команде `/set` в REPL и сборке payload запроса.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


class ParamError(Exception):
    """Некорректное значение параметра — показывается пользователю текстом."""


@dataclass(slots=True)
class Spec:
    """Описание одного параметра: как разобрать и чем ограничен."""

    name: str
    kind: str
    help: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None
    # Возможность из capabilities модели, без которой параметр не отправляем.
    requires: str | None = None


SPECS: tuple[Spec, ...] = (
    Spec(
        "temperature",
        "float",
        "Разброс ответа: 0 — детерминированно, выше — креативнее.",
        0.0,
        2.0,
    ),
    Spec("top_p", "float", "Nucleus sampling: доля вероятностной массы.", 0.0, 1.0),
    Spec("max_tokens", "int", "Потолок длины ответа в токенах.", 1, None),
    Spec("random_seed", "int", "Seed для воспроизводимости ответа.", 0, None),
    Spec("stop", "list", "Стоп-последовательности через запятую."),
    Spec(
        "reasoning_effort",
        "choice",
        "Глубина рассуждения. Только для моделей с capability reasoning.",
        choices=REASONING_EFFORTS,
        requires="reasoning",
    ),
)

BY_NAME = {spec.name: spec for spec in SPECS}


def _parse(spec: Spec, raw: Any) -> Any:
    if raw is None:
        return None

    if spec.kind == "float":
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ParamError(f"{spec.name} должен быть числом, а не {raw!r}") from exc
    elif spec.kind == "int":
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ParamError(f"{spec.name} должен быть целым числом, а не {raw!r}") from exc
    elif spec.kind == "choice":
        value = str(raw).strip().lower()
        if value not in (spec.choices or ()):
            allowed = ", ".join(spec.choices or ())
            raise ParamError(f"{spec.name} принимает только: {allowed}")
        return value
    else:  # list
        if isinstance(raw, list):
            items = [str(x).strip() for x in raw]
        else:
            items = [part.strip() for part in str(raw).split(",")]
        items = [item for item in items if item]
        if not items:
            return None
        return items

    if spec.minimum is not None and value < spec.minimum:
        raise ParamError(f"{spec.name} не может быть меньше {spec.minimum}")
    if spec.maximum is not None and value > spec.maximum:
        raise ParamError(f"{spec.name} не может быть больше {spec.maximum}")
    return value


@dataclass(slots=True)
class GenerationParams:
    """Значения, заданные пользователем. None означает «не передавать».

    Незаданный параметр сознательно не подставляется дефолтом: Mistral сам
    применит рекомендованное для модели значение, и оно чаще лучше, чем
    зашитая в клиент константа.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    random_seed: int | None = None
    stop: list[str] | None = None
    reasoning_effort: str | None = None

    @classmethod
    def build(cls, **raw: Any) -> GenerationParams:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ParamError(f"Неизвестные параметры: {', '.join(sorted(unknown))}")
        return cls(**{name: _parse(BY_NAME[name], value) for name, value in raw.items()})

    def set(self, name: str, raw: Any) -> Any:
        """Меняет один параметр по имени. Используется командой `/set`."""
        spec = BY_NAME.get(name)
        if spec is None:
            allowed = ", ".join(BY_NAME)
            raise ParamError(f"Неизвестный параметр {name!r}. Доступны: {allowed}")

        value = None if str(raw).strip().lower() in ("", "none", "default") else _parse(spec, raw)
        setattr(self, name, value)
        return value

    def as_payload(self, capabilities: dict | None = None) -> tuple[dict, list[str]]:
        """Payload для API и список параметров, отсеянных по capabilities.

        Отправлять reasoning_effort модели без reasoning бессмысленно: ответ
        либо проигнорирует его, либо прилетит 422 посреди демо.
        """
        payload: dict[str, Any] = {}
        skipped: list[str] = []

        for spec in SPECS:
            value = getattr(self, spec.name)
            if value is None:
                continue
            if spec.requires and capabilities is not None and not capabilities.get(spec.requires):
                skipped.append(spec.name)
                continue
            payload[spec.name] = value

        return payload, skipped

    def describe(self) -> list[tuple[str, str, str]]:
        """Строки для `/params`: имя, значение, пояснение."""
        rows = []
        for spec in SPECS:
            value = getattr(self, spec.name)
            if value is None:
                shown = "—"
            elif isinstance(value, list):
                shown = ",".join(value)
            else:
                shown = str(value)
            rows.append((spec.name, shown, spec.help))
        return rows
