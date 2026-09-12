"""Параметры генерации: разбор, валидация и отсев неподдерживаемых моделью.

Отдельный модуль, потому что один и тот же набор нужен трём потребителям:
флагам CLI, команде `/set` в REPL и сборке payload запроса.
"""

from __future__ import annotations

import math
from collections.abc import Collection
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

# Context-assembly strategies of week 02 day 10 (SPEC-w02d10.md §3). One axis,
# four values — "window+facts both on" would mean nothing (facts already
# implies "facts + last N"), so this is a choice, not a set of flags.
CONTEXT_STRATEGY_CHOICES = ("window", "facts", "branch", "summary")

# Команды CLI, у которых есть собственные значения по умолчанию (Spec.defaults).
CHAT_COMMAND = "chat"
SOLVE_COMMAND = "solve"
TEMP_COMMAND = "temp"
BENCH_COMMAND = "bench"
# Агент недели 02 — не подкоманда недели, а отдельная точка входа
# (`adventagent`), поэтому и контекст умолчаний у него свой
# (SPEC-w02d06.md §3, §16).
AGENT_COMMAND = "agent"

# Параметры, которые агент действительно читает. Локальные параметры недели 01
# (problem, runs, judge, judge_model, temps, models) сюда НЕ входят: `/params`
# у агента не должен показывать то, на что он не смотрит — выставленный
# параметр, ни на что не влияющий, выглядит как поломка (SPEC-w02d06.md §16).
# Реестр общий на весь проект, поэтому фильтр — это список имён, а не
# отдельный набор Spec: второй реестр разъехался бы с первым на первой же
# правке.
AGENT_PARAMS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "max_tokens",
    "random_seed",
    "stop",
    "reasoning_effort",
    "format",
    "schema_file",
    "session",
    "mode",
    "done",
    "max_turns",
    # Override лимита окна (день 08): агент читает его при trim — без имени
    # в этом списке /set context_limit в агенте отвергся бы как «не читается».
    "context_limit",
    # History compaction (day 09): agent reads it before building the request.
    "compact",
    "keep_last",
    "compact_every",
    # Context strategies (day 10): which of the four assemblies to use, and
    # the size cap on the facts block the "facts" strategy maintains.
    "context_strategy",
    "facts_max_tokens",
)

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
    # maximum=1.5, не 2.0: живой замер 2026-09-03 (SPEC-w01d04.md §2) дал 422
    # "Input should be less than or equal to 1.5" уже на 1.51 — одинаково у
    # mistral-small-latest и magistral-medium-latest. 0..2 — диапазон OpenAI,
    # у Mistral его нет; баг ехал с Day 01.
    Spec(
        "temperature",
        "float",
        "Разброс ответа: ниже — стабильнее, выше — разнообразнее. Потолок API 1.5.",
        0.0,
        1.5,
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
        # У агента маркер задан по умолчанию: `/set mode dialog` обязан
        # работать сразу, а не требовать второй командой то, без чего диалог
        # не закончится никогда. Значение НЕ должно пересекаться со stop —
        # API вырезает stop из ответа (CLAUDE.md), и agent.check_done() ловит
        # это, если пользователь задаст stop сам.
        defaults={AGENT_COMMAND: "text:ГОТОВО"},
    ),
    Spec(
        "mode",
        "choice",
        "Режим REPL: chat (обычный) или dialog (вопросы до готовности).",
        choices=MODE_CHOICES,
        local=True,
        defaults={AGENT_COMMAND: "chat"},
    ),
    Spec(
        "max_turns",
        "int",
        "Потолок ходов диалога в mode=dialog.",
        1,
        None,
        local=True,
        # Дублирует дефолт поля GenerationParams.max_turns намеренно: поле
        # держит 10 для недели 01, где Spec.defaults для него не было вовсе, а
        # `/set max_turns default` сбрасывает поле в None — и вернуть туда
        # десятку умеет только apply_defaults(). Без этой строки у агента
        # «default» означало бы «неизвестно».
        defaults={AGENT_COMMAND: 10},
    ),
    # День 08 (SPEC-w02d08.md §4): override лимита окна для trim. None —
    # «из карточки модели». Локальный, потому что сервер знает только своё
    # окно: клиентский override физически не может дать серверный 400
    # (PROBE-w02d08-overflow.md §5), его читает один лишь агент.
    Spec(
        "context_limit",
        "int",
        "Лимит окна контекста в токенах (override карточки модели). Не задан — из карточки.",
        1,
        None,
        local=True,
    ),
    # Day 09 (SPEC-w02d09.md §9): history compaction. All three are local —
    # the server never sees them, the agent decides before sending the request.
    Spec(
        "compact",
        "bool",
        "Сжимать старую часть истории в пересказ вместо выбрасывания.",
        local=True,
        # On by default: the day's result is framed as "an agent that
        # compresses". Turning it off is needed for comparison and the demo.
        defaults={AGENT_COMMAND: True},
    ),
    Spec(
        "keep_last",
        "int",
        "Сколько последних сообщений остаётся в контексте как есть.",
        # Minimum 2, not 1: a question-answer pair is indivisible; a
        # one-message tail would mean an answer without its question or vice versa.
        2,
        None,
        local=True,
        defaults={AGENT_COMMAND: 6},
    ),
    Spec(
        "compact_every",
        "int",
        "Порог сжатия: столько несжатых старых сообщений запускают пересказ.",
        2,
        None,
        local=True,
        defaults={AGENT_COMMAND: 10},
    ),
    # Day 10 (SPEC-w02d10.md §3, §12): which of the four request assemblies to
    # use. Default "summary" — day 09 keeps behaving exactly as before for
    # every caller that never sets this. Local: the server never sees it, the
    # agent decides before building the request, same as compact/keep_last.
    Spec(
        "context_strategy",
        "choice",
        "Стратегия сборки контекста: window (окно) / facts (память фактов) / "
        "branch (вся история ветки) / summary (пересказ, как в дне 09).",
        choices=CONTEXT_STRATEGY_CHOICES,
        local=True,
        defaults={AGENT_COMMAND: "summary"},
    ),
    # Day 10 (SPEC-w02d10.md §5.5): soft cap on the facts block. Not enforced
    # by code — the extractor gets a "уплотняй формулировки" nudge past this
    # size; only the extractor or a human ever deletes a fact.
    Spec(
        "facts_max_tokens",
        "int",
        "Потолок размера блока facts в токенах (сигнал экстрактору, не жёсткий срез).",
        100,
        None,
        local=True,
        defaults={AGENT_COMMAND: 400},
    ),
    Spec(
        "session",
        "string",
        "Имя сессии агента: logs/sessions/<имя>.json.",
        local=True,
        defaults={AGENT_COMMAND: "default"},
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
        defaults={TEMP_COMMAND: "all"},
    ),
    Spec(
        "runs",
        "int",
        "Прогонов на стратегию: >1 показывает разброс вместо одного броска.",
        1,
        None,
        local=True,
        defaults={SOLVE_COMMAND: 1, TEMP_COMMAND: 3, BENCH_COMMAND: 3},
    ),
    Spec(
        "judge",
        "bool",
        "Оценивать ли ответы LLM-судьёй (эталона судья не видит).",
        local=True,
        # TEMP_COMMAND сознательно не здесь: LLM-судья дня 04 убран целиком
        # (пользовательское решение — креативность open-задач оценивает
        # человек, week_01/temperature.py.print_human_review(), а не API).
        # `judge`/`judge_model` остаются в реестре ради `solve` — команда
        # `temp` их больше не читает вовсе.
        defaults={CHAT_COMMAND: False, SOLVE_COMMAND: True},
    ),
    Spec(
        "judge_model",
        "string",
        "Модель судьи. Не задана — судит та же модель, что решала.",
        local=True,
    ),
    Spec(
        "temps",
        "floats",
        "Список температур для развёртки (day 04).",
        0.0,
        1.5,
        local=True,
        defaults={TEMP_COMMAND: [0.0, 0.7, 1.2]},
    ),
    Spec(
        "models",
        "models",
        "Список моделей для развёртки (day 05), через запятую.",
        local=True,
        defaults={
            BENCH_COMMAND: [
                "ministral-3b-latest",
                "ministral-8b-latest",
                "ministral-14b-latest",
            ]
        },
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
        # math.isnan() отдельно от minimum/maximum ниже: NaN не меньше и не
        # больше ничего (сравнение по IEEE754 всегда ложно), обе проверки
        # границ молча пропускают его — `--temperature nan` доехал бы до
        # payload и до Mistral как нестрогий JSON-литерал NaN.
        if math.isnan(value):
            raise ParamError(f"{spec.name} не может быть NaN")
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
    elif spec.kind == "floats":
        # Разбор — как у "list" (строка через запятую или готовый список), но
        # каждый элемент приводится к float и сверяется с minimum/maximum сразу
        # здесь: ветка возвращает список и до общей проверки границ ниже
        # (она рассчитана на одиночное float/int в value) не доходит. Порядок
        # и дубликаты сохраняются — пользователь мог намеренно попросить одну
        # температуру дважды.
        if isinstance(raw, list):
            items = [str(x).strip() for x in raw]
        else:
            items = [part.strip() for part in str(raw).split(",")]
        items = [item for item in items if item]
        if not items:
            return None
        values: list[float] = []
        for item in items:
            try:
                item_value = float(item)
            except (TypeError, ValueError) as exc:
                raise ParamError(f"{spec.name}: элемент {item!r} должен быть числом") from exc
            # Та же дыра, что у одиночного kind="float" выше: NaN не меньше
            # minimum и не больше maximum, поэтому без явной проверки проезжает
            # в список температур молча (`--temps 0,nan,1.2`).
            if math.isnan(item_value):
                raise ParamError(f"{spec.name}: элемент {item!r} не может быть NaN")
            if spec.minimum is not None and item_value < spec.minimum:
                raise ParamError(f"{spec.name}: элемент {item!r} меньше {spec.minimum}")
            if spec.maximum is not None and item_value > spec.maximum:
                raise ParamError(f"{spec.name}: элемент {item!r} больше {spec.maximum}")
            values.append(item_value)
        return values
    elif spec.kind == "models":
        # Тот же разбор через запятую, что у "list", но пустой список — это
        # ParamError, а не None: у "list" (stop) пустое значение осмысленно
        # («стоп-строк нет»), а у "models" пустая лестница означает, что
        # bench нечего перебирать — молчаливое None утонуло бы в
        # apply_defaults() и подменилось бы дефолтом там, где пользователь
        # явно (пусть и неудачно) задал --models "" или "--models , ,".
        if isinstance(raw, list):
            items = [str(x).strip() for x in raw]
        else:
            items = [part.strip() for part in str(raw).split(",")]
        items = [item for item in items if item]
        if not items:
            raise ParamError(f"{spec.name}: список моделей не может быть пустым")
        return items
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

    # Override лимита окна (день 08, SPEC-w02d08.md §4). None — «из карточки
    # модели»: у агента свой контракт на None, чем у max_turns выше — там
    # потребитель трактует None как 10, а здесь None означает «карточка».
    # Умолчание команды не заводим: `/set context_limit default` обязан
    # вернуть именно None, а не подменить его дефолтом через apply_defaults().
    context_limit: int | None = None

    # Day 09 params (SPEC-w02d09.md §9): history compaction. None here means
    # "command default" (apply_defaults fills it in), not "off" — the agent's
    # compact defaults to on.
    compact: bool | None = None
    keep_last: int | None = None
    compact_every: int | None = None

    # Day 10 params (SPEC-w02d10.md §12): context strategy + facts size cap.
    # None means "command default" (apply_defaults fills it in), matching
    # compact/keep_last/compact_every above.
    context_strategy: str | None = None
    facts_max_tokens: int | None = None

    # Параметр агента (день 06): имя сессии на диске. Тоже локальный — в
    # payload не идёт, но живёт в общем реестре, потому что `/set session
    # <имя>` обязан валидироваться тем же кодом, что и остальные параметры.
    session: str | None = None

    # Параметры дня 03. Значения по умолчанию не проставляются здесь: они
    # разные у chat и solve, и живут в Spec.defaults — см. apply_defaults().
    strategy: str | None = None
    problem: str | None = None
    runs: int | None = None
    judge: bool | None = None
    judge_model: str | None = None

    # Параметр дня 04 (SPEC-w01d04.md §5). Тоже локальный: развёртка по
    # температурам делает week_01/temperature.py отдельными вызовами chat_core,
    # список сам по себе в payload не идёт.
    temps: list[float] | None = None

    # Параметр дня 05 (SPEC-w01d05.md §8): лестница моделей для `bench` —
    # список имён, а не Config.model, потому что bench перебирает несколько
    # моделей за один прогон, а не работает с одной.
    models: list[str] | None = None

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
                # list(value), а не value как есть: Spec.defaults[TEMP_COMMAND]
                # для temps — один list-объект, общий на весь модуль (SPECS
                # строится один раз при импорте). Присвоить его напрямую
                # значило бы, что все инстансы GenerationParams, получившие
                # дефолт temps, делят один и тот же список — мутация в одном
                # (например .append() где-то ниже по стеку) была бы видна
                # всем остальным, включая следующий build() без temps.
                setattr(self, name, list(value) if isinstance(value, list) else value)

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

    def describe(self, names: Collection[str] | None = None) -> list[tuple[str, str, str]]:
        """Строки для `/params`: имя, значение, пояснение.

        `names` ограничивает вывод теми параметрами, которые команда реально
        читает (AGENT_PARAMS у агента). None — показать все, как было в
        неделе 01: там `/params` показывает весь реестр и предупреждает про
        чужие параметры отдельно (NON_CHAT_PARAMS в week_01/cli.py).
        """
        rows = []
        for spec in SPECS:
            if names is not None and spec.name not in names:
                continue
            value = getattr(self, spec.name)
            if value is None:
                shown = "—"
            elif isinstance(value, bool):
                # Раньше isinstance(value, list): bool — не список и не строка,
                # и без этой ветки judge печатался бы как "False", что читается
                # как значение, а не как выключенный флаг.
                shown = "вкл" if value else "выкл"
            elif spec.kind == "floats":
                # ":g", а не str(): str(0.0) даёт "0.0", а не "0" — читается
                # как отдельное число из другого разбора, хотя ввод и вывод
                # должны совпадать буквально ("0,0.7,1.2").
                shown = ",".join(f"{v:g}" for v in value)
            elif isinstance(value, list):
                shown = ",".join(value)
            else:
                shown = str(value)
            rows.append((spec.name, shown, spec.help))
        return rows
