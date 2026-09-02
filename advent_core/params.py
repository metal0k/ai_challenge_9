"""Параметры генерации: разбор, валидация и отсев неподдерживаемых моделью.

Отдельный модуль, потому что один и тот же набор нужен трём потребителям:
флагам CLI, команде `/set` в REPL и сборке payload запроса.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
FORMAT_CHOICES = ("text", "json", "schema", "yaml", "md")
MODE_CHOICES = ("chat", "dialog")

# Названия стратегий рассуждения дня 03. Живут здесь, а не в
# week_01/strategies.py, по той же причине, что FORMAT_CHOICES и MODE_CHOICES
# дня 02: реестр параметров обязан валидировать значение сам, а импортировать
# неделю в core нельзя — зависимость идёт только в обратную сторону.
# strategies.py импортирует этот кортеж, чтобы список стратегий не пришлось
# править в двух файлах.
STRATEGIES = ("direct", "steps", "meta", "panel")
STRATEGY_CHOICES = (*STRATEGIES, "all")

# Команды CLI, у которых есть собственные значения по умолчанию (Spec.defaults).
CHAT_COMMAND = "chat"
SOLVE_COMMAND = "solve"

# Слова, которыми задаётся булев параметр. Оба языка: `/set judge выкл` на
# видео читается, `--no-judge` в командной строке — тоже, и обе формы должны
# работать одинаково.
BOOL_TRUE = ("1", "true", "yes", "on", "y", "да", "вкл")
BOOL_FALSE = ("0", "false", "no", "off", "n", "нет", "выкл")


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
    # Локальный параметр (формат, диалог, …): в payload API не попадает вообще,
    # это не то же самое, что «срезан по capabilities» — см. as_payload().
    local: bool = False
    # Значение по умолчанию, своё для каждой команды CLI: `strategy` — direct в
    # chat и all в solve, `judge` — выкл в chat и вкл в solve (SPEC-w01d03.md
    # §4). Держим здесь, а не в сигнатурах typer: иначе одно и то же правило
    # оказалось бы записано в двух командах и разъехалось при первой же правке.
    # Пустой словарь означает «умолчания нет, None и есть значение».
    defaults: dict[str, Any] = field(default_factory=dict)


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
    Spec(
        "format",
        "choice",
        "Пресет формата ответа: text (без ограничений) / json / schema / yaml / md.",
        choices=FORMAT_CHOICES,
        local=True,
    ),
    Spec(
        "schema_file",
        "path",
        "Путь к .json со схемой ответа — нужен для format=schema.",
        local=True,
    ),
    Spec(
        "done",
        "string",
        'Условие завершения диалога: "text:<строка>" или "json:<поле>".',
        local=True,
    ),
    Spec(
        "mode",
        "choice",
        "Режим REPL: chat (обычный) или dialog (вопросы до готовности).",
        choices=MODE_CHOICES,
        local=True,
    ),
    Spec(
        "max_turns",
        "int",
        "Потолок ходов диалога в mode=dialog.",
        1,
        None,
        local=True,
    ),
    Spec(
        "strategy",
        "choice",
        "Способ рассуждения: direct / steps / meta / panel / all.",
        choices=STRATEGY_CHOICES,
        local=True,
        defaults={CHAT_COMMAND: "direct", SOLVE_COMMAND: "all"},
    ),
    Spec(
        "problem",
        "string",
        'id задачи из банка week_01/problems (по умолчанию — с флагом "default": true).',
        local=True,
    ),
    Spec(
        "runs",
        "int",
        "Прогонов на стратегию: >1 показывает разброс вместо одного броска.",
        1,
        None,
        local=True,
        defaults={SOLVE_COMMAND: 1},
    ),
    Spec(
        "judge",
        "bool",
        "Оценивать ли ответы LLM-судьёй (эталона судья не видит).",
        local=True,
        defaults={CHAT_COMMAND: False, SOLVE_COMMAND: True},
    ),
    Spec(
        "judge_model",
        "string",
        "Модель судьи. Не задана — судит та же модель, что решала.",
        local=True,
    ),
)

BY_NAME = {spec.name: spec for spec in SPECS}


def defaults_for(command: str) -> dict[str, Any]:
    """Умолчания параметров для команды CLI: `chat` и `solve` различаются."""
    return {spec.name: spec.defaults[command] for spec in SPECS if command in spec.defaults}


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
    elif spec.kind == "bool":
        # bool раньше str(): булев флаг typer приходит уже разобранным, а
        # `/set judge вкл` — строкой, и обе формы обязаны дать один результат.
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in BOOL_TRUE:
            return True
        if text in BOOL_FALSE:
            return False
        raise ParamError(f"{spec.name} принимает вкл/выкл: {', '.join((*BOOL_TRUE, *BOOL_FALSE))}")
    elif spec.kind in ("string", "path"):
        # Разбор содержимого (префикс done, существование файла схемы и т.п.)
        # сознательно не здесь — Spec валидирует только форму значения,
        # семантику знает advent_core/formats.py (parse_done, load_schema).
        text = str(raw).strip()
        return text or None
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

    # Локальные параметры (Spec.local=True) — управляют CLI/REPL, на сервер
    # не уходят. max_turns хранит содержательный дефолт (10), а не None:
    # это не «дай серверу решить», это дефолт цикла диалога. `/set max_turns
    # default` всё равно сбросит его в None по общей логике set() ниже —
    # потребитель (цикл mode=dialog) обязан трактовать None как 10.
    format: str | None = None
    schema_file: str | None = None
    done: str | None = None
    mode: str | None = None
    max_turns: int | None = 10

    # Параметры дня 03. Значения по умолчанию не проставляются здесь: они
    # разные у chat и solve, и живут в Spec.defaults — см. apply_defaults().
    strategy: str | None = None
    problem: str | None = None
    runs: int | None = None
    judge: bool | None = None
    judge_model: str | None = None

    @classmethod
    def build(cls, **raw: Any) -> GenerationParams:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ParamError(f"Неизвестные параметры: {', '.join(sorted(unknown))}")
        return cls(**{name: _parse(BY_NAME[name], value) for name, value in raw.items()})

    def apply_defaults(self, command: str) -> None:
        """Проставляет умолчания команды в параметры, которых пользователь не задал.

        Зовётся и на старте команды, и после каждого `/set … default` в REPL:
        «default» означает «умолчание этой команды», а не «пусто навсегда».
        Уже заданное значение (в том числе False у judge) не трогается —
        именно поэтому проверка на None, а не на ложность.
        """
        for name, value in defaults_for(command).items():
            if getattr(self, name) is None:
                setattr(self, name, value)

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
            if spec.local:
                # Срезаются штатно и молча: skipped — предупреждение об уже
                # заданном параметре, который сервер бы отклонил или
                # проигнорировал, а не про параметр, который туда и не должен
                # был идти.
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
            elif isinstance(value, bool):
                # Раньше isinstance(value, list): bool — не список и не строка,
                # и без этой ветки judge печатался бы как "False", что читается
                # как значение, а не как выключенный флаг.
                shown = "вкл" if value else "выкл"
            elif isinstance(value, list):
                shown = ",".join(value)
            else:
                shown = str(value)
            rows.append((spec.name, shown, spec.help))
        return rows
