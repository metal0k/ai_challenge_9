"""Неделя 01 — команды `advent w01 ...`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.markup import escape as rich_escape

from advent_core import chat as chat_core
from advent_core import console, formats
from advent_core.client import (
    capabilities_of,
    chat_models,
    find_model,
    list_models,
    model_names,
    resolve_alias,
)
from advent_core.config import DEFAULT_SYSTEM_PROMPT, LOG_DIR, Config, ConfigError
from advent_core.errors import AdventError, ConfigurationError
from advent_core.journal import log_call
from advent_core.params import (
    BENCH_COMMAND,
    BY_NAME,
    CHAT_COMMAND,
    FORMAT_CHOICES,
    MODE_CHOICES,
    REASONING_EFFORTS,
    SOLVE_COMMAND,
    STRATEGY_CHOICES,
    TEMP_COMMAND,
    ParamError,
    defaults_for,
)
from advent_core.telemetry import CallResult
from week_01 import models_bench, strategies, temperature

WEEK = 1
# Поднимать вместе с номером дня, который последним трогал ЭТОТ код —
# единственное место, которое это знает. Раньше log_call() звался с зашитым
# day=1 буквально везде, включая код дня 02 (_handle_again, _run_dialog):
# `advent record --day 2` писал в logs/calls.jsonl день 1 (НАХОДКА 4 code
# review). Протаскивать day через всю цепочку вызовов ради одной константы —
# избыточно; альтернатива проще и заведомо не разъедется, пока не забыть её
# поднять.
#
# День 04 добавил `temp` — код, которого chat/dialog/solve не касаются и
# который последним меняли не в день 03. Общая константа на весь модуль здесь
# уже не работает: подняв DAY до 4 бездумно, получили бы день 4 и у вызовов
# chat/solve, хотя эти команды в день 04 не менялись — их последняя правка
# по-прежнему день 03 (см. tests/test_cli_dialog.py, где cli.DAY==3 закреплён
# отдельной проверкой). Поэтому у chat/solve остаётся DAY, а у temp —
# отдельная TEMP_DAY ниже; при следующем дне, который тронет chat или solve,
# поднимать нужно будет именно DAY, а не эту константу.
DAY = 3
# Только для temp_command — единственной команды, написанной в день 04.
TEMP_DAY = 4
# Только для bench_command — единственной команды, написанной в день 05. Тест
# сверяет этот литерал напрямую (5), а не cli.BENCH_DAY: ожидание, взятое из
# того же источника, что и код под тестом, покраснеть не может (CLAUDE.md,
# «A test whose expected value comes from the same source…»).
BENCH_DAY = 5
HISTORY_PATH = LOG_DIR / "repl_history_w01.json"
DIALOG_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dialog.md"

# Параметры дней 03-04, которые REPL (всегда контекст `chat`) не читает — их
# читают только команды `solve`/`temp`. В REPL про них надо сказать вслух:
# выставленный параметр, ни на что не влияющий, выглядит как поломка — ровно
# та же логика, что у предупреждения про capabilities в /set. `temps` —
# единственный параметр здесь, который решает только `temp`; `problem`/`runs`
# общие для `solve` и `temp`; `judge`/`judge_model` читает только `solve` —
# LLM-судья дня 04 убран целиком (пользовательское решение), `temp` эти два
# параметра больше не читает вовсе.
# `session` — параметр агента недели 02: реестр параметров общий на проект, и
# новая запись в нём немедленно видна в `/params` недели 01 и принимается её
# `/set`. Без этой строки `/set session рецепты` печатал бы «session =
# рецепты» без единого предупреждения, ни на что при этом не влияя, — ровно
# тот отказ, ради которого NON_CHAT_PARAMS и заведён, да ещё и в уже сданном
# дне, который виден по тегу.
NON_CHAT_PARAMS = {
    "problem": "solve, temp и bench",
    "runs": "solve, temp и bench",
    "judge": "solve",
    "judge_model": "solve",
    "temps": "temp",
    "models": "bench",
    "session": "adventagent (неделя 02)",
    # Override лимита окна (день 08, SPEC-w02d08.md §4): решает только trim
    # у агента, REPL недели 01 его вообще не читает — без этой строки /set
    # context_limit молча «принимался» бы, как раньше молчал session.
    "context_limit": "adventagent (неделя 02)",
    # History compaction (day 09, SPEC-w02d09.md §9): decided by the agent
    # before building the request. Week 01's REPL has no token-based history
    # at all — it's trimmed by chat_core.trim_history()'s character budget,
    # nothing there to summarize. Same reason as the three lines above.
    "compact": "adventagent (неделя 02)",
    "keep_last": "adventagent (неделя 02)",
    "compact_every": "adventagent (неделя 02)",
    # Context strategies (day 10, SPEC-w02d10.md §3, §12): they pick how the
    # agent assembles a request — facts block, window, branch or summary.
    # Week 01 assembles nothing of the sort, so the same rule as above.
    "context_strategy": "adventagent (неделя 02)",
    "facts_max_tokens": "adventagent (неделя 02)",
    "working_max_tokens": "adventagent (неделя 03)",
    "long_term_max_tokens": "adventagent (неделя 03)",
    "memory_max_tokens": "adventagent (неделя 03)",
}

REPL_COMMANDS = [
    ("/help", "показать этот список"),
    ("/model", "текущая модель"),
    ("/model <имя>", "переключить модель"),
    ("/model list", "таблица chat-моделей (list all — вообще все)"),
    ("/model info", "карточка текущей модели: контекст, возможности, t°"),
    ("/params", "текущие параметры генерации (и итоговый system prompt)"),
    (
        "/set <параметр> <значение>",
        "изменить параметр, включая format/schema_file/done/mode/max_turns/strategy "
        "(default — вернуть умолчание команды)",
    ),
    ("/again", "повторить последний вопрос с текущими настройками, без истории"),
    ("/reset", "очистить историю диалога"),
    ("/exit", "выход"),
]


def _default_hint(command: str, param: str) -> str:
    """«(по умолчанию X)» для help флага, взятое из реестра параметров.

    Дефолт у каждой команды свой и живёт в Spec.defaults; напечатанный рядом
    литералом, он становится вторым источником истины и однажды разъедется с
    настоящим — ровно то, из-за чего в `--temperature` полгода висел
    несуществующий потолок 2.0 (CLAUDE.md).
    """
    value = defaults_for(command).get(param)
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return f"(по умолчанию {value})"


app = typer.Typer(help="Неделя 01: первый запрос к LLM через API.", no_args_is_help=True)


@app.command("chat")
def chat_command(
    question: str | None = typer.Argument(None, help="Вопрос. Без аргумента открывается REPL."),
    model: str | None = typer.Option(None, "--model", "-m", help="Имя модели Mistral."),
    system: Path | None = typer.Option(None, "--system", help="Файл с system prompt."),
    # help берётся из реестра, а не пишется здесь второй раз: тот же текст уже
    # печатает /params через Spec.describe(), и «0..2.» здесь однажды разошлось
    # с потолком API 1.5 в Spec (SPEC-w01d04.md §2) — одно место истины не даёт
    # этому повториться.
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help=BY_NAME["temperature"].help
    ),
    top_p: float | None = typer.Option(None, "--top-p", help="Nucleus sampling, 0..1."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Потолок длины ответа."),
    seed: int | None = typer.Option(None, "--seed", help="Seed для воспроизводимости."),
    stop: str | None = typer.Option(None, "--stop", help="Стоп-строки через запятую."),
    reasoning_effort: str | None = typer.Option(
        None,
        "--reasoning-effort",
        help=f"Глубина рассуждения: {', '.join(REASONING_EFFORTS)}. Нужна capability reasoning.",
    ),
    format_: str | None = typer.Option(
        None,
        "--format",
        help=f"Пресет формата ответа: {', '.join(FORMAT_CHOICES)}.",
    ),
    schema_file: str | None = typer.Option(
        None, "--schema-file", help="Путь к .json со схемой ответа — для --format schema."
    ),
    done: str | None = typer.Option(
        None,
        "--done",
        help='Условие завершения диалога: "text:<строка>" или "json:<поле>".',
    ),
    mode: str | None = typer.Option(None, "--mode", help=f"Режим REPL: {', '.join(MODE_CHOICES)}."),
    max_turns: int | None = typer.Option(
        None, "--max-turns", help="Потолок ходов в --mode dialog (по умолчанию 10)."
    ),
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        help=(
            f"Способ рассуждения над вопросом: {', '.join(STRATEGY_CHOICES)}. "
            "direct — обычный чат без добавок."
        ),
    ),
    no_stream: bool = typer.Option(False, "--no-stream", help="Получить ответ одним куском."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Детали запроса в stderr."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Базовый URL OpenAI-совместимого endpoint "
        "(например http://127.0.0.1:1234 для LM Studio).",
    ),
) -> None:
    """Задать вопрос модели: один ответ или интерактивный диалог."""
    config = Config.resolve(
        model=model,
        system=system,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        random_seed=seed,
        stop=stop,
        reasoning_effort=reasoning_effort,
        stream=not no_stream,
        verbose=verbose,
        base_url=base_url,
    )
    # Config.resolve() не знает про пять локальных параметров дня 02 (вне моей
    # зоны — не трогаю config.py), поэтому кладём их той же машинерией, что и
    # /set: GenerationParams.set() с той же валидацией и теми же ParamError.
    _apply_local_flags(
        config,
        format=format_,
        schema_file=schema_file,
        done=done,
        mode=mode,
        max_turns=max_turns,
        strategy=strategy,
    )
    # Умолчания у chat и solve разные (strategy=direct против all), и знает об
    # этом реестр параметров, а не сигнатура typer: иначе одно правило было бы
    # записано в двух командах.
    config.params.apply_defaults(CHAT_COMMAND)
    _guard_one_shot_dialog(question, config)

    if question:
        # direct — это отсутствие каких-либо добавок к промпту, то есть ровно
        # обычный путь дней 01–02 со стримом и всем прочим. Поэтому стратегия
        # уводит вопрос в машинерию дня 03 только когда она НЕ direct: иначе
        # каждый обычный `chat "привет"` начал бы требовать строку ОТВЕТ: и
        # потерял бы стрим.
        if config.params.strategy == "direct":
            _ask_once(config, question)
        else:
            _ask_with_strategy(Session(config), question)
    else:
        _repl(config)


def _guard_one_shot_dialog(question: str | None, config: Config) -> None:
    """--mode dialog вместе с позиционным вопросом раньше молча одношотился.

    `_repl()`/`_run_dialog()` — единственное место, которое читает mode/done,
    а одношотовый `_ask_once()` про них ничего не знает. В результате
    `advent w01 chat "..." --mode dialog` тихо отвечал обычным одиночным
    ответом, как будто флага не было (НАХОДКА 2 code review). Диалог по
    природе многоходовый — молча игнорировать флаг нельзя, поэтому падаем
    понятной ошибкой конфигурации вместо этого. ConfigurationError — та же
    машинерия и тот же exit_code=2, что и у остальных ошибок конфигурации
    (advent_core/errors.py).
    """
    if question is not None and config.params.mode == "dialog":
        raise ConfigurationError(
            "--mode dialog работает только в интерактивном режиме — "
            "убери позиционный вопрос и открой REPL: advent w01 chat"
        )


def _apply_local_flags(config: Config, *, allow_all_problem: bool = False, **flags: object) -> None:
    """Кладёт локальные флаги команды (--format, --strategy, …) в config.params.

    Именованные аргументы, а не фиксированная сигнатура: у `chat`, `solve` и
    `temp` наборы флагов разные, а обрабатываются они одинаково. Неизвестное
    имя не проглатывается — `GenerationParams.set()` поднимет ParamError, как
    на опечатку в `/set`.

    Валидация та же, что у `/set`: GenerationParams.set() проверяет форму
    значения (choices, непустая строка, …), а `_check_local_param()` следом
    проверяет смысл (файл схемы существует, задача есть в банке) — на старте
    CLI ошибка должна остановить программу, а не всплыть только на первом
    запросе к модели.

    `allow_all_problem` — только для `temp`: там "all" у `--problem» значит
    «обе задачи дня» (week_01.temperature.TEMP_PROBLEMS), а не id из банка,
    так что искать его через strategies.load_problem() нельзя — этот же
    вопрос уже решён для strategy=all на Day 03, только там magic-значение
    ловится валидацией choices, а problem — свободная строка (SPEC-w01d04.md
    §5).
    """
    try:
        for name, raw in flags.items():
            if raw is None:
                continue
            config.params.set(name, raw)
            _check_local_param(name, raw, allow_all_problem=allow_all_problem)
    except ParamError as exc:
        raise ConfigError(str(exc)) from exc


def _check_local_param(name: str, value: object, *, allow_all_problem: bool = False) -> None:
    """Ранняя проверка смысла (не только формы) для schema_file/done/problem.

    Spec в params.py валидирует только форму значения (см. её комментарий);
    смысл — существование и валидность файла схемы, распознаваемый префикс
    done, наличие задачи в банке — знают formats.load_schema(),
    formats.parse_done() и strategies.load_problem(). Здесь их зовут ради
    самой дешёвой точки сообщить об ошибке: сразу, а не на первом запросе к
    модели или посреди демо. Для `problem` это особенно важно: опечатка в id
    иначе всплыла бы после девяти (а у `temp` — после девятнадцати) оплаченных
    вызовов.
    """
    if name == "schema_file":
        formats.load_schema(str(value))
    elif name == "done":
        formats.parse_done(str(value))
    elif name == "problem":
        if allow_all_problem and str(value) == "all":
            return
        strategies.load_problem(str(value))


@app.command("models")
def models_command(
    model: str | None = typer.Option(None, "--model", "-m", help="Подсветить эту модель."),
    all_models: bool = typer.Option(False, "--all", help="Показать и не-chat модели."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Базовый URL OpenAI-совместимого endpoint "
        "(например http://127.0.0.1:1234 для LM Studio).",
    ),
) -> None:
    """Список моделей аккаунта прямо из API."""
    config = Config.resolve(model=model, base_url=base_url)
    models = list_models(config)
    shown = models if all_models else chat_models(models)

    console.models_table(shown, highlight=config.model)

    if actual := resolve_alias(models, config.model):
        console.note(f"{config.model} сейчас разрешается в {actual}")
    console.note(f"показано {len(shown)} из {len(models)}; t° — рекомендованная моделью")


@app.command("solve")
def solve_command(
    problem: str | None = typer.Option(None, "--problem", help="id задачи из банка problems/."),
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        help=f"Способ рассуждения: {', '.join(STRATEGY_CHOICES)}. all — все четыре подряд.",
    ),
    runs: int | None = typer.Option(
        None,
        "--runs",
        help=(
            f"Прогонов на стратегию: >1 показывает разброс {_default_hint(SOLVE_COMMAND, 'runs')}."
        ),
    ),
    judge: bool | None = typer.Option(
        None, "--judge/--no-judge", help="Оценка ответов LLM-судьёй (по умолчанию включена)."
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Модель судьи. По умолчанию судит та же, что решала."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Имя модели Mistral."),
    system: Path | None = typer.Option(None, "--system", help="Файл с system prompt."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Детали запуска в stderr."),
) -> None:
    """Решить одну задачу разными способами рассуждения и сравнить результаты."""
    config = Config.resolve(model=model, system=system, verbose=verbose)
    _apply_local_flags(
        config,
        problem=problem,
        strategy=strategy,
        runs=runs,
        judge=judge,
        judge_model=judge_model,
    )
    config.params.apply_defaults(SOLVE_COMMAND)
    params = config.params

    task = strategies.load_problem(params.problem)
    names = _strategy_names(params.strategy)
    # None здесь означает «сброшено через /set runs default» — по контракту
    # params.py это умолчание команды, то есть 1, а не «ноль прогонов».
    count = params.runs or 1

    session = Session(config)
    _check_judge_model(session)

    if config.verbose:
        console.note(
            f"model={config.model} problem={task.id} strategies={', '.join(names)} "
            f"runs={count} judge={'вкл' if params.judge else 'выкл'}"
        )

    strategies.print_problem(task)

    warned: set[str] = set()

    def on_step(step: strategies.Step) -> None:
        """Вызов состоялся — печатаем и пишем в журнал немедленно.

        Единица и печати, и записи — вызов, а не прогон и не стратегия.
        Раньше и то и другое шло пачкой после законченной стратегии: 429 на
        четвёртом вызове панели уносил три уже оплаченных ответа и с экрана,
        и из logs/calls.jsonl — а журнал единственное, что этот день
        сохраняет. Заодно исчезает минута молчания на записи: ответ виден
        сразу, а не после девятого вызова.
        """
        _adopt_results([step.result], session, warned)
        strategies.print_step(step)
        strategies.log_step(step, week=WEEK, day=DAY, problem=task.id)

    outcomes = strategies.solve(
        task,
        config,
        strategies=names,
        runs=count,
        capabilities=session.capabilities,
        # complete передаётся явно, хотя у solve() ровно такой дефолт:
        # значение по умолчанию связывается в момент импорта strategies, и
        # подмена chat_core.complete (так тесты убирают сеть — см.
        # tests/test_cli_dialog.py) до него уже не достаёт.
        complete=chat_core.complete,
        on_step=on_step,
        # Вердикт по эталону существует только после последнего вызова
        # прогона, поэтому он отдельным колбэком, а не внутри on_step.
        on_run=strategies.print_verdict,
    )

    verdict = None
    if params.judge:
        # capabilities судьи считаются по ЕГО модели, а не по решавшей: при
        # --judge-model отсев параметров шёл по чужой карточке и, например,
        # reasoning_effort уезжал модели без такой возможности — 400 вместо
        # оценки, колонка «судья» превращалась в «—» после девяти оплаченных
        # вызовов. Свой warned-набор по той же причине: предупреждение о
        # срезанном параметре относится к другой модели.
        judge_name = params.judge_model or config.model
        judge_warned: set[str] = set()
        verdict = strategies.judge(
            task,
            outcomes,
            config,
            capabilities=capabilities_of(session.models, judge_name) if session.models else None,
            model=params.judge_model,
            complete=chat_core.complete,  # см. комментарий у solve() выше
        )
        if verdict.result is not None:
            _adopt_results([verdict.result], session, judge_warned, model=params.judge_model)
        strategies.log_judge(verdict, week=WEEK, day=DAY, problem=task.id)

    strategies.print_comparison(task, outcomes, verdict)


def _strategy_names(strategy: str | None) -> tuple[str, ...]:
    """`all` разворачивается в четыре стратегии в фиксированном порядке.

    Порядок задаёт и таблицу, и метки судьи A–D, поэтому берётся как есть из
    реестра и не сортируется.
    """
    if strategy is None or strategy == "all":
        return strategies.STRATEGIES
    return (strategy,)


def _check_judge_model(session: Session) -> None:
    """Проверяет модель судьи до первого вызова, а не после девяти.

    strategies.judge() переживает недоступную модель штатно — ошибка попадает
    в JudgeVerdict.error и таблица всё равно печатается. Но узнать про
    опечатку в имени после всех вызовов решения — значит оплатить прогон
    впустую, поэтому проверяем по уже загруженному списку моделей аккаунта.
    """
    name = session.config.params.judge_model
    if not name or not session.models:
        return
    if name not in model_names(session.models):
        raise ConfigError(
            f"Модель судьи {name!r} недоступна аккаунту.\nПосмотри список: advent w01 models"
        )


@app.command("temp")
def temp_command(
    problem: str | None = typer.Option(
        None,
        "--problem",
        help="id задачи из банка problems/, all — обе задачи дня (по умолчанию).",
    ),
    temps: str | None = typer.Option(
        None,
        "--temps",
        help="Температуры через запятую, потолок API 1.5 (по умолчанию 0,0.7,1.2).",
    ),
    runs: int | None = typer.Option(
        None,
        "--runs",
        help=f"Прогонов на ячейку задача×температура {_default_hint(TEMP_COMMAND, 'runs')}.",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Имя модели Mistral."),
    system: Path | None = typer.Option(None, "--system", help="Файл с system prompt."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Детали запуска в stderr."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Базовый URL OpenAI-совместимого endpoint "
        "(например http://127.0.0.1:1234 для LM Studio).",
    ),
) -> None:
    """Один и тот же запрос при разных температурах: точность, формат, разнообразие.

    Креативность open-задач эта команда не оценивает сама — печатает все
    ответы рядом (week_01.temperature.print_human_review), а балл ставит
    человек, глядя на экран. Прежний LLM-судья дня 04 убран целиком
    (пользовательское решение); `--judge`/`--judge-model` у этой команды
    больше нет — они остаются только у `solve` (Day 03).
    """
    config = Config.resolve(model=model, system=system, verbose=verbose, base_url=base_url)
    _apply_local_flags(
        config,
        problem=problem,
        temps=temps,
        runs=runs,
        # "all" у --problem здесь не id из банка, а «обе задачи дня» — см.
        # комментарий у _apply_local_flags().
        allow_all_problem=True,
    )
    config.params.apply_defaults(TEMP_COMMAND)
    params = config.params

    # load_temp_problems() — та же самая дешёвая точка отказа, что и
    # strategies.load_problem() у solve: опечатка в id обязана остановить
    # программу здесь, до Session (а значит и до первого сетевого вызова), а
    # не после части из девятнадцати оплаченных прогонов развёртки.
    problems = temperature.load_temp_problems(params.problem)

    # _check_judge_model(session) здесь намеренно НЕТ (было в дне 04 с
    # судьёй): `temp` больше не читает judge_model вовсе — LLM-судья дня 04
    # убран целиком, креативность open-задач оценивает человек
    # (temperature.print_human_review). У `solve` (Day 03) эта проверка
    # остаётся — там судья по-прежнему есть.
    session = Session(config)

    if config.verbose:
        console.note(
            f"model={config.model} problems={', '.join(p.id for p in problems)} "
            f"temps={','.join(f'{t:g}' for t in params.temps)} runs={params.runs}"
        )

    for task in problems:
        temperature.print_problem_header(task)

    warned: set[str] = set()

    def on_step(step: temperature.HeatStep) -> None:
        """Вызов состоялся — печатаем и пишем в журнал немедленно.

        Тот же принцип, что у solve_command.on_step: единица и печати, и
        записи — вызов, а не ячейка и не задача целиком. Обрыв посреди
        развёртки не должен унести с экрана и из logs/calls.jsonl уже
        оплаченные прогоны.
        """
        _adopt_results([step.result], session, warned)
        temperature.print_heat_step(step)
        temperature.log_heat_step(step, week=WEEK, day=TEMP_DAY)

    sweeps = temperature.run_temperature_sweep(
        config,
        problems,
        params.temps,
        params.runs or 1,
        capabilities=session.capabilities,
        # complete передаётся явно, хотя у run_temperature_sweep() ровно такой
        # дефолт: значение по умолчанию связывается в момент импорта
        # week_01.temperature, и подмена chat_core.complete (так тесты убирают
        # сеть) до него уже не достаёт — та же ловушка Day 03 (CLAUDE.md, «A
        # default argument binds at import time»).
        complete=chat_core.complete,
        on_step=on_step,
    )

    for sweep in sweeps:
        temperature.print_cell_table(sweep.problem, sweep.cells)

    temperature.print_conclusions(sweeps)


def _check_bench_models(models: list[str], account_models: list[dict]) -> None:
    """Проверяет имена --models до первого вызова, а не посреди развёртки.

    Тот же принцип, что у _check_judge_model: опечатка в имени модели иначе
    всплыла бы только после части оплаченных прогонов развёртки
    (SPEC-w01d05.md §8). account_models пустой — значит список моделей
    аккаунта недоступен вовсе (см. bench_command) — тогда проверять нечем, и
    вызов молча пропускается, ошибка (если она есть) всплывёт на первом
    запросе, как и у остальных команд при недоступном списке.
    """
    if not account_models:
        return
    available = model_names(account_models)
    missing = [name for name in models if name not in available]
    if missing:
        raise ConfigError(
            f"Модели недоступны аккаунту: {', '.join(missing)}.\nПосмотри список: advent w01 models"
        )


@app.command("bench")
def bench_command(
    problem: str | None = typer.Option(
        None,
        "--problem",
        help="id задачи из банка problems/, all — обе задачи дня (по умолчанию).",
    ),
    models: str | None = typer.Option(
        None,
        "--models",
        help=f"Модели через запятую {_default_hint(BENCH_COMMAND, 'models')}.",
    ),
    runs: int | None = typer.Option(
        None,
        "--runs",
        help=f"Прогонов на ячейку модель×задача {_default_hint(BENCH_COMMAND, 'runs')}.",
    ),
    system: Path | None = typer.Option(None, "--system", help="Файл с system prompt."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Детали запуска в stderr."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Базовый URL OpenAI-совместимого endpoint "
        "(например http://127.0.0.1:1234 для LM Studio).",
    ),
) -> None:
    """Одна и та же задача на разных версиях модели: точность, latency, токены, цена.

    Флага `--model` здесь нет (SPEC-w01d05.md §8): команда перебирает
    лестницу моделей, а не работает с одной — список задаёт `--models`.
    temperature=0 фиксируется на всю развёртку безусловно (week_01/
    models_bench._bench_hygiene) — здесь нет флага `--temperature`, как и у
    `--model`, по той же причине: это не то, что варьирует этот день.
    """
    config = Config.resolve(system=system, verbose=verbose, base_url=base_url)
    _apply_local_flags(
        config,
        problem=problem,
        models=models,
        runs=runs,
        # "all" у --problem здесь значит «обе задачи дня» (BENCH_PROBLEMS), не
        # id из банка — тот же приём, что и у `temp` (см. _apply_local_flags).
        allow_all_problem=True,
    )
    config.params.apply_defaults(BENCH_COMMAND)
    params = config.params

    # Дешёвые точки отказа до первого сетевого вызова — тот же принцип, что у
    # solve/temp: опечатка в id задачи или в файле цен не должна всплывать
    # после части из уже оплаченных прогонов развёртки.
    problems = models_bench.load_bench_problems(params.problem)
    prices = models_bench.load_prices()
    models_bench.warn_if_prices_stale(prices)

    try:
        account_models = list_models(config)
    except AdventError:
        # Список моделей — удобство (отсев параметров по capabilities, ранняя
        # проверка имён), а не обязательное условие для самого прогона: та же
        # снисходительность, что у Session.refresh() — ошибка тогда всплывёт
        # на первом же запросе к API, а не здесь.
        account_models = []

    _check_bench_models(params.models, account_models)

    if config.verbose:
        console.note(
            f"models={', '.join(params.models)} problems={', '.join(p.id for p in problems)} "
            f"runs={params.runs}"
        )

    for task in problems:
        temperature.print_problem_header(task)

    # Ключ по модели, а не общий набор, как у solve/temp: bench перебирает
    # НЕСКОЛЬКО моделей за один прогон, и предупреждение про параметр,
    # срезанный у одной модели (нет нужной capability), не должно молча
    # скрыть то же предупреждение у другой — у них разные карточки.
    warned: dict[str, set[str]] = {}

    def on_step(step: models_bench.BenchStep) -> None:
        """Вызов состоялся — печатаем и пишем в журнал немедленно.

        Тот же принцип, что у solve_command.on_step/temp_command.on_step:
        единица и печати, и записи — вызов, а не ячейка и не задача целиком.
        _adopt_results здесь не переиспользуется напрямую: она читает
        capabilities и имя запрошенной модели из одного Session, а bench на
        каждом шаге зовёт свою модель — эквивалент её двух действий (подставить
        реальную версию, предупредить о срезанных параметрах) выписан явно.
        """
        resolved = resolve_alias(account_models, step.model) if account_models else None
        if resolved:
            step.result.model_actual = resolved
        step_warned = warned.setdefault(step.model, set())
        for name in step.result.skipped_params:
            if name not in step_warned:
                step_warned.add(name)
                console.warn(f"{step.model} не поддерживает {name} — параметр не отправлен")

        models_bench.print_bench_step(step)
        models_bench.log_bench_step(step, week=WEEK, day=BENCH_DAY)

    sweeps = models_bench.run_bench(
        config,
        problems,
        params.models,
        # Умолчание (3) приходит из реестра — Spec("runs").defaults[BENCH_COMMAND],
        # применяется apply_defaults(BENCH_COMMAND). `or 3` здесь завело бы
        # второй источник истины для одного и того же числа (CLAUDE.md).
        params.runs,
        account_models=account_models,
        # complete передаётся явно, хотя у run_bench() ровно такой дефолт:
        # значение по умолчанию связывается в момент импорта
        # week_01.models_bench, и подмена chat_core.complete (так тесты убирают
        # сеть) до него уже не достаёт — та же ловушка Day 03 (CLAUDE.md, «A
        # default argument binds at import time»).
        complete=chat_core.complete,
        on_step=on_step,
    )

    for sweep in sweeps:
        models_bench.print_bench_table(sweep.problem, sweep.cells, prices)

    models_bench.print_conclusions(sweeps, prices)
    models_bench.print_links(params.models, prices)


class Session:
    """Состояние одного запуска чата: модель, её метаданные и параметры.

    Держим карточку модели рядом с конфигом: она нужна и для отсева
    параметров по capabilities, и для `/model info`, а тянуть список моделей
    на каждый запрос — лишний round trip.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.models: list[dict] = []
        self.card: dict | None = None
        # Последний вопрос обычного (не dialog) режима — только для /again.
        # None, пока в этой сессии не было ни одного вопроса.
        self.last_question: str | None = None
        self.refresh()

    def refresh(self) -> None:
        """Перечитывает список моделей и проверяет, что текущая существует."""
        try:
            self.models = list_models(self.config)
        except AdventError:
            # Список — удобство, а не обязательное условие: ошибка всплывёт
            # на самом запросе и будет переведена в человеческий текст.
            self.models = []
            self.card = None
            return

        names = model_names(self.models)
        if names and self.config.model not in names:
            raise ConfigError(
                f"Модель {self.config.model!r} недоступна аккаунту.\n"
                f"Посмотри список: advent w01 models"
            )
        self.card = find_model(self.models, self.config.model)

    @property
    def capabilities(self) -> dict | None:
        return capabilities_of(self.models, self.config.model) if self.models else None

    @property
    def resolved(self) -> str | None:
        return resolve_alias(self.models, self.config.model) if self.models else None


def _ask_once(config: Config, question: str) -> None:
    """Один вопрос — один ответ, без истории."""
    session = Session(config)
    messages = chat_core.build_messages(question, system=config.system_prompt())

    if config.verbose:
        # should_stream() — единая точка решения (см. _run), не сырой
        # config.stream: format=json/schema всегда без стрима, и verbose
        # печатал stream=True в этом случае, хотя реально ушла ветка без
        # стрима (НАХОДКА 6 code review).
        console.note(
            f"model={config.model} stream={chat_core.should_stream(config)} "
            f"messages={len(messages)}"
        )

    result = _run(session, messages)
    console.footer(result)
    # result.sent_messages — то, что реально дописала chat._payload() поверх
    # messages (инструкция пресета формата); fallback на messages нужен
    # только теоретически (sent_messages=None означало бы, что _run() упал до
    # _payload(), а тогда мы бы сюда не дошли) — журнал не должен разъезжаться
    # с тем, что реально ушло в API (НАХОДКА 5 code review).
    log_call(result, result.sent_messages or messages, week=WEEK, day=DAY)


def _repl(config: Config) -> None:
    """Интерактивный диалог с историей, командами и обрезкой контекста."""
    session = Session(config)
    system = config.system_prompt()
    history = _load_history()

    console.note("REPL. /help — список команд. Ctrl+C прерывает генерацию.")
    if history:
        console.note(f"подхвачена история прошлого запуска: {len(history)} сообщений")

    while True:
        try:
            line = typer.prompt("\nты", prompt_suffix=" › ", err=True).strip()
        except (EOFError, typer.Abort):
            console.note("пока")
            break

        # При запуске из скрипта записи stdin — это pipe, и терминал не
        # отображает то, что в него подали. Без эха в кадре видно приглашение
        # без вопроса.
        if line and not sys.stdin.isatty():
            console.echo_input(line)

        if not line:
            continue
        if line.startswith("/"):
            if _handle_command(line, session, history):
                break
            continue

        # Стратегия дня 03 применяется к каждому вопросу REPL (SPEC-w01d03.md
        # §3). direct — это отсутствие добавок, то есть обычный путь ниже, а
        # не отдельная ветка: иначе `/set strategy direct` вёл бы себя иначе,
        # чем чат по умолчанию.
        if session.config.params.strategy not in (None, "direct"):
            if session.config.params.mode == "dialog":
                # Диалог — многоходовый разговор с историей, стратегия — один
                # независимый прогон. Совместить их нельзя, и молча выбрать
                # одно из двух хуже, чем сказать, что именно выполняется.
                console.warn("strategy не применяется в mode=dialog — идёт диалог")
            else:
                session.last_question = line
                _strategy_question(session, line)
                continue

        # mode=dialog уводит обычную реплику в отдельный цикл «вопрос за
        # вопросом» (см. _run_dialog) — история обычного REPL здесь не
        # трогается: это независимый разговор со своим system-пресетом.
        #
        # Проверка done — здесь, а не только внутри _run_dialog: раньше
        # _run_dialog() при отсутствии done печатала warning и делала return
        # ДО обращения к API, без записи в session.last_question и без
        # журнала — введённая строка просто исчезала (НАХОДКА 3 code
        # review). Если done не настроен, диалог технически не может
        # работать (не с чем сравнивать условие завершения) — реплика падает
        # в обычную одноходовую обработку ниже, а не теряется молча.
        if session.config.params.mode == "dialog":
            if config.params.done:
                pending_command = _run_dialog(session, line)
                # _run_dialog() возвращает отложенную команду (строку с "/"),
                # только если диалог прервался именно на слэш-команде — не на
                # завершении по done, не на max_turns, не на EOF (НАХОДКА 7
                # code review). Она ещё не была обработана нигде, поэтому
                # исполняем её здесь же, как будто пользователь только что её
                # ввёл — иначе "/exit" в конце demo-шага 5 молча терялся бы.
                if pending_command is not None and _handle_command(
                    pending_command, session, history
                ):
                    break
                continue
            console.warn(
                "mode=dialog требует done — /set done text:<строка> или "
                "json:<поле>; реплика обработана как обычный вопрос"
            )

        session.last_question = line
        history, dropped = chat_core.trim_history(history)
        if dropped:
            console.warn(f"история обрезана: выброшено {dropped} старых сообщений")

        messages = chat_core.build_messages(line, system=system, history=history)

        try:
            result = _run(session, messages)
        except AdventError as error:
            # В REPL ошибка не должна убивать сессию — печатаем и живём дальше.
            console.fail(error)
            # Ошибка случилась до/внутри _payload() — result.sent_messages
            # тут нет (ошибка сконструирована вручную), messages и так то,
            # что реально пытались отправить.
            log_call(
                chat_core.CallResult(model_requested=session.config.model),
                messages,
                week=WEEK,
                day=DAY,
                error=error.message,
            )
            continue

        console.footer(result)
        log_call(result, result.sent_messages or messages, week=WEEK, day=DAY)

        history.append({"role": "user", "content": line})
        if result.text:
            history.append({"role": "assistant", "content": result.text})
        _save_history(history)


def _handle_command(line: str, session: Session, history: list[chat_core.Message]) -> bool:
    """Обрабатывает команду REPL. True означает «выходим»."""
    parts = line.split()
    command, args = parts[0], parts[1:]

    if command in ("/exit", "/quit"):
        console.note("пока")
        return True

    if command == "/help":
        console.commands_help(REPL_COMMANDS)
        return False

    if command == "/reset":
        history.clear()
        _save_history(history)
        console.note("история очищена")
        return False

    if command == "/params":
        _show_params(session)
        return False

    if command == "/set":
        _handle_set(args, session)
        return False

    if command == "/model":
        _handle_model(args, session)
        return False

    if command == "/again":
        _handle_again(session)
        return False

    console.warn(f"неизвестная команда {command}. /help — список")
    return False


def _show_params(session: Session) -> None:
    """`/params`: таблица параметров (локальные помечены) + итоговый system prompt."""
    rows = []
    for name, value, help_text in session.config.params.describe():
        if BY_NAME[name].local:
            # Скобки — круглые, не квадратные: console.params_table() кладёт
            # эту строку в ячейку Rich Table, а Rich-markup молча съедает
            # квадратные скобки как незакрытый тег (проверено вручную).
            help_text = f"{help_text} (local — не уходит на сервер)"
        rows.append((name, value, help_text))
    console.params_table(rows)

    system = _final_system_prompt(session)
    if system:
        # rich_escape: system prompt содержит инструкцию формата и, при
        # format=schema, дамп JSON Schema — квадратные скобки массивов иначе
        # console.note() (Rich-markup) молча съедает как незакрытый тег, и
        # часть текста пропадает с экрана без единой ошибки.
        console.note(f"итоговый system prompt:\n{rich_escape(system)}")
    else:
        console.note("system prompt не задан")

    # Показать «итоговый» system при активной стратегии нельзя: у panel их
    # четыре разных, у meta второй вообще пишет сама модель. Молчать тоже
    # нельзя — /params врал бы ровно так же, как врал на mode=dialog (НАХОДКА
    # 1 code review), просто в другую сторону.
    strategy = session.config.params.strategy
    if strategy in (None, "direct"):
        return

    if session.config.params.mode == "dialog":
        # В mode=dialog REPL стратегию не применяет вовсе (см. _repl): диалог
        # многоходовый и с историей, стратегия — один независимый прогон.
        # Обещать здесь промпт стратегии значило бы врать про запрос, которого
        # не будет, — тем же способом, каким /params врал про dialog раньше.
        console.note(f"strategy={strategy} не применяется в mode=dialog — идёт диалог")
        return

    # Штатный system-пресет проекта («лаконичный ассистент, без воды») в
    # стратегию не подмешивается вовсе (strategies.build_strategy_system): он
    # противоречит инструкции рассуждать пошагово и измеримо её обнуляет.
    # Значит показанный выше «итоговый system prompt» к вызовам стратегии
    # отношения не имеет, и говорить «поверх этого system» — ложь.
    explicit = session.config.system_prompt_path != DEFAULT_SYSTEM_PROMPT
    fate = (
        "явно заданный system при этом сохраняется"
        if explicit
        else "показанный выше system prompt в вызовы стратегии НЕ уходит"
    )
    console.note(
        f"strategy={strategy}: штатный system-пресет проекта в стратегию не подмешивается "
        f"({fate}). Каждый вызов собирает system заново: промпт из "
        f"week_01/prompts/reason_*.md плюс общая добавка про маркер "
        f"{strategies.ANSWER_MARKER}"
    )


def _final_system_prompt(session: Session) -> str | None:
    """system пользователя + дописанная инструкция пресета формата/диалога.

    Раньше ветка mode=dialog делала ранний return сразу после
    _dialog_system_prompt() и НИКОГДА не доходила до formats.build_system()
    — а реально уходящий запрос (chat._payload()) безусловно дописывает слой
    формата поверх ЛЮБОГО system, dialog-пресет не исключение. /params врал:
    показывал dialog-промпт без инструкции формата, хотя она реально уйдёт
    (НАХОДКА 1 code review). Теперь обе ветки собирают base_system и в конце
    ОБЕ проходят через formats.build_system() — ту же функцию, что и
    chat._payload()/_with_format_instruction() зовут перед отправкой, так что
    правило дописывания формата живёт в одном месте и не может разъехаться
    между /params и реальным запросом.
    """
    config = session.config
    base_system = config.system_prompt()

    if config.params.mode == "dialog" and config.params.done:
        try:
            kind, needle = formats.parse_done(config.params.done)
        except ConfigError:
            pass  # done ещё не поправлен — покажем обычный (не dialog) system
        else:
            # _dialog_system_prompt() уже сама берёт config.system_prompt() и
            # дописывает пресет диалога — дальше форматный слой ложится
            # поверх результата, как и на запросе.
            base_system = _dialog_system_prompt(config, kind, needle)

    format_name = config.params.format or "text"
    schema = None
    if format_name == "schema" and config.params.schema_file:
        try:
            schema = formats.load_schema(config.params.schema_file)
        except ConfigError:
            schema = None  # покажем то, что есть — ошибка всплывёт на запросе
    return formats.build_system(format_name, base_system, schema)


def _handle_set(args: list[str], session: Session) -> None:
    if len(args) < 2:
        console.warn("нужно: /set <параметр> <значение>. /params — что есть")
        return

    name, raw = args[0], " ".join(args[1:])
    try:
        value = session.config.params.set(name, raw)
    except ParamError as error:
        console.warn(str(error))
        return

    # `/set strategy default` означает «верни умолчание команды» (для REPL это
    # всегда chat), а не «оставь пусто»: пустая strategy иначе читалась бы
    # ниже как direct случайно, а не по правилу из реестра параметров.
    session.config.params.apply_defaults(CHAT_COMMAND)
    value = getattr(session.config.params, name)

    if name in NON_CHAT_PARAMS:
        console.warn(
            f"{name} читает только команда {NON_CHAT_PARAMS[name]} — "
            "на вопросы в этом REPL не влияет"
        )

    # rich_escape: значения вроде done=text:[ГОТОВО] (пример из спеки дня)
    # содержат квадратные скобки — без экранирования console.note() (Rich
    # markup) тихо съедает их как незакрытый тег, и маркер пропадает с экрана.
    if value is None:
        shown = "сброшен"
    elif isinstance(value, bool):
        # Раньше строки: bool попал бы в ветку else и напечатался как "False",
        # что читается как значение параметра, а не как выключенный флаг.
        shown = "вкл" if value else "выкл"
    elif isinstance(value, str):
        shown = rich_escape(value)
    else:
        shown = value
    console.note(f"{name} = {shown}")

    if value is not None:
        # Смысловая проверка (файл схемы существует, префикс done понятен) —
        # сразу, а не молча на первом запросе к модели. Параметр остаётся
        # выставленным даже при ошибке: тот же принцип, что и с capabilities
        # ниже — пользователь видит предупреждение и правит следующей командой.
        try:
            _check_local_param(name, value)
        except ConfigError as error:
            console.warn(str(error))

    # Предупреждаем сразу, а не молча на запросе: иначе непонятно, почему
    # выставленный параметр ни на что не влияет.
    capabilities = session.capabilities
    _, skipped = session.config.params.as_payload(capabilities)
    if name in skipped:
        console.warn(f"{session.config.model} не поддерживает {name} — параметр не отправляется")


def _handle_model(args: list[str], session: Session) -> None:
    if not args:
        console.note(f"текущая модель: {session.config.model}")
        return

    sub = args[0]

    if sub == "list":
        if not session.models:
            console.warn("список моделей недоступен — не отвечает API")
            return
        show_all = len(args) > 1 and args[1] == "all"
        shown = session.models if show_all else chat_models(session.models)
        console.models_table(shown, highlight=session.config.model)
        console.note(f"показано {len(shown)} из {len(session.models)}")
        if not show_all:
            console.note("/model list all — включая embedding, OCR и аудио")
        return

    if sub == "info":
        if session.card is None:
            console.warn("карточка модели недоступна — не отвечает API")
            return
        console.model_card(session.card, session.config.model)
        return

    previous = session.config.model
    session.config.model = sub
    try:
        session.refresh()
    except (ConfigError, AdventError) as error:
        # Опечатка в имени модели не должна ронять сессию: чинится следующей
        # же командой.
        session.config.model = previous
        session.refresh()
        console.warn(str(getattr(error, "message", error)))
        return

    resolved = session.resolved
    console.note(f"модель переключена на {sub}" + (f" → {resolved}" if resolved else ""))

    _, skipped = session.config.params.as_payload(session.capabilities)
    if skipped:
        console.warn(f"{sub} не поддерживает: {', '.join(skipped)} — параметры не отправляются")


def _handle_again(session: Session) -> None:
    """`/again`: тот же последний вопрос, текущие настройки, БЕЗ истории.

    Это и есть суть задания дня («один и тот же запрос») и защита от подмены:
    при обычном повторе через историю модель видит собственный прошлый ответ
    и отвечает «как я говорил выше» вместо нового независимого прогона —
    тогда сравнение стабильности format=schema теряет смысл.
    """
    if session.config.params.mode == "dialog":
        console.warn("/again недоступна в mode=dialog — там история и есть суть диалога")
        return
    if session.last_question is None:
        console.warn("нечего повторять — сначала задай вопрос")
        return

    if session.config.params.strategy not in (None, "direct"):
        # Повтор должен быть повтором того же самого: если вопросы идут через
        # стратегию, /again обязан идти через неё же, иначе сравнивались бы
        # разные механики.
        _strategy_question(session, session.last_question)
        return

    messages = chat_core.build_messages(
        session.last_question, system=session.config.system_prompt()
    )
    try:
        result = _run(session, messages)
    except AdventError as error:
        console.fail(error)
        log_call(
            chat_core.CallResult(model_requested=session.config.model),
            messages,
            week=WEEK,
            day=DAY,
            error=error.message,
        )
        return

    console.footer(result)
    log_call(result, result.sent_messages or messages, week=WEEK, day=DAY)
    # Сознательно не трогаем историю REPL: /again — независимый прогон того
    # же запроса, а не ещё один ход обычного диалога.


def _dialog_marker_instruction(kind: str, needle: str) -> str:
    """Инструкция про признак завершения — вставляется в пресет dialog.md.

    Формат признака задаётся опцией `done`, а не зашит: text — подстрока в
    свободном тексте, json — булево поле в структурированном ответе.
    """
    if kind == "text":
        return (
            f'Когда результат готов, включи в ответ строку ровно так: "{needle}" — '
            "это и есть признак завершения. Пока диалог не закончен, эту строку "
            "не используй."
        )
    # parse_done() уже проверил kind — тут он либо "text", либо "json".
    return (
        "Отвечай только в формате JSON, без пояснений и без обёртки в ```. Пока "
        f'идёт уточнение — верни объект вида {{"{needle}": false, "question": '
        f'"..."}}. Когда результат готов — верни {{"{needle}": true, "result": '
        '{...}} с итоговыми данными вместо "...".'
    )


def _dialog_system_prompt(config: Config, kind: str, needle: str) -> str:
    """system для mode=dialog: пользовательский + пресет из prompts/dialog.md.

    Пресет — шаблон с плейсхолдером %%MARKER%%, наполняемым конкретным
    условием завершения (см. _dialog_marker_instruction). Дальше слой
    формата (format=json/schema/…) дописывается уже внутри chat_core поверх
    этого system — ровно тот же порядок наслоения, что и в обычном чате
    (SPEC-w01d02.md §4).
    """
    template = DIALOG_PROMPT_PATH.read_text(encoding="utf-8").strip()
    preset = template.replace("%%MARKER%%", _dialog_marker_instruction(kind, needle))
    parts = [part for part in (config.system_prompt(), preset) if part]
    return "\n\n".join(parts)


def _run_dialog(session: Session, task: str) -> str | None:
    """Цикл mode=dialog: по одному уточняющему вопросу за ход, до готовности.

    Отдельная история от обычного REPL — независимый разговор со своим
    system-пресетом (SPEC-w01d02.md §5). Маркер завершения из вывода не
    вырезается: он виден на экране, это и есть предмет демонстрации.

    Возвращает отложенную команду REPL (строку с ведущим "/"), если диалог
    закончился из-за того, что пользователь ввёл команду вместо реплики —
    иначе None. Раньше ЛЮБАЯ строка, прочитанная внутри цикла (включая
    "/exit"), безусловно уходила в модель как очередная реплика: слэш-команды
    не распознавались вообще (НАХОДКА 7 code review, воспроизведена живым
    прогоном demo-шага 5). Если диалог не сходится за отведённые ходы модели
    — а он по природе не гарантированно сходится, ходов сколько потребуется
    — "/exit", которым шаг 5 в advent_cli/record.py завершает демо, был бы
    отправлен в API как бессмысленный ответ про салат, а сама команда выхода
    потеряна: сессия осталась бы висеть в диалоге, дубль сорван перед
    камерой. Возврат строки наверх — единственный вариант, не теряющий
    команду молча и не глотающий её как реплику: _run_dialog не знает, как
    исполняются команды REPL (history, /model, /again — состояние в _repl),
    поэтому решение отдаётся вызывающему коду.
    """
    config = session.config
    if not config.params.done:
        # Страховка на случай прямого вызова в обход _repl(): штатный путь
        # (_repl) сам проверяет done ДО вызова _run_dialog() и в его
        # отсутствие обрабатывает реплику как обычный вопрос вместо того,
        # чтобы звать эту функцию (НАХОДКА 3 code review — раньше именно
        # здесь введённая строка молча терялась). Сюда мы дойти не должны.
        console.warn("mode=dialog требует done — /set done text:<строка> или json:<поле>")
        return None
    try:
        kind, needle = formats.parse_done(config.params.done)
    except ConfigError as error:
        console.warn(str(error))
        return None

    system = _dialog_system_prompt(config, kind, needle)
    # None здесь означает «сброшено через /set max_turns default» — по
    # контракту params.py это трактуется как дефолт 10, а не «без потолка»:
    # без потолка demo-шаг advent record завис бы до таймаута в 180 секунд
    # на первом же вопросе модели, которого нет в фиксированном списке строк.
    max_turns = config.params.max_turns or 10
    dialog_history: list[chat_core.Message] = []
    line = task
    turns = 0

    while True:
        turns += 1
        messages = chat_core.build_messages(line, system=system, history=dialog_history)
        try:
            result = _run(session, messages)
        except AdventError as error:
            console.fail(error)
            log_call(
                chat_core.CallResult(model_requested=config.model),
                messages,
                week=WEEK,
                day=DAY,
                error=error.message,
            )
            return None

        console.footer(result)
        log_call(result, result.sent_messages or messages, week=WEEK, day=DAY)

        dialog_history.append({"role": "user", "content": line})
        if result.text:
            dialog_history.append({"role": "assistant", "content": result.text})

        if formats.is_done(result.text, kind, needle):
            console.note(f"диалог: условие завершения выполнено, ходов: {turns}")
            return None

        if turns >= max_turns:
            # НЕ ошибка и НЕ выход из сессии — просто возврат в обычный REPL,
            # диалог остаётся в логе как незавершённый.
            console.warn(
                f"диалог: достигнут потолок ходов ({max_turns}), условие не "
                "выполнено — возврат в обычный REPL"
            )
            return None

        try:
            line = typer.prompt("\nдиалог", prompt_suffix=" › ", err=True).strip()
        except (EOFError, typer.Abort):
            console.note("диалог прерван")
            return None

        if line and not sys.stdin.isatty():
            console.echo_input(line)
        if not line:
            console.note("пустой ввод — диалог прерван, возврат в обычный REPL")
            return None

        if line.startswith("/"):
            # Слэш-команда — не реплика диалога. "/exit" здесь раньше уходил
            # в API как ответ про салат (НАХОДКА 7): демо-шаг 5
            # (advent_cli/record.py) заканчивается ровно "/exit" внутри
            # диалога, и если модель не сошлась за отведённые ходы (она не
            # обязана сходиться быстро — это не гарантия, а наблюдение),
            # выход срывался перед камерой. Прерываем диалог и отдаём строку
            # наверх — _repl() исполнит её как только что введённую команду,
            # ничего не потеряется и ничего не уйдёт в модель.
            console.note("диалог: получена команда — выходим в обычный REPL")
            return line


def _ask_with_strategy(session: Session, question: str) -> None:
    """Вопрос через стратегию рассуждения дня 03 — вторая поверхность дня.

    Обе поверхности (`solve` и `chat --strategy`) ходят в один и тот же
    week_01/strategies.py, поэтому разъехаться в поведении не могут
    (SPEC-w01d03.md §3). Отличие ровно одно: эталона у вопроса из чата нет,
    поэтому печатаются только сами вызовы (print_step), без вердикта.

    История REPL сознательно не трогается: стратегия строит свои messages с
    нуля (панель экспертов независима по определению), и дописать её ответ в
    историю значило бы показать модели разговор, которого она не видела.
    Это тот же принцип, что у `/again`.
    """
    problem = strategies.question_as_problem(question)
    warned: set[str] = set()

    def on_step(step: strategies.Step) -> None:
        # Печать и журнал по факту вызова — та же причина, что в solve_command.
        _adopt_results([step.result], session, warned)
        strategies.print_step(step)
        strategies.log_step(step, week=WEEK, day=DAY, problem=problem.id)

    for position, name in enumerate(_strategy_names(session.config.params.strategy)):
        # Тот же прогресс в stderr, что печатает strategies.solve(): на записи
        # между шагами панели проходят десятки секунд, и молчащий экран
        # выглядит зависшим.
        console.note(name)
        strategies.run_strategy(
            name,
            problem,
            session.config,
            capabilities=session.capabilities,
            complete=chat_core.complete,  # см. комментарий у solve() выше
            # Предупреждения про stop/format зависят от конфига, а не от
            # стратегии: на `chat --strategy all` они иначе печатаются четыре
            # раза подряд. Сама правка конфига при этом делается каждый раз.
            warn_unsafe_params=position == 0,
            on_step=on_step,
        )


def _strategy_question(session: Session, question: str) -> None:
    """То же, что _ask_with_strategy(), но переживает ошибку — версия для REPL.

    В REPL ошибка не должна убивать сессию (тот же принцип, что и в обычной
    ветке). В журнал при этом уходит сам вопрос, а не реальные messages
    стратегии: их построил и потерял упавший вызов внутри run_strategy(), и
    придумывать их здесь значило бы записать в лог то, чего не отправляли.
    """
    try:
        _ask_with_strategy(session, question)
    except AdventError as error:
        console.fail(error)
        log_call(
            CallResult(model_requested=session.config.model),
            [{"role": "user", "content": question}],
            week=WEEK,
            day=DAY,
            error=error.message,
            extra={"strategy": session.config.params.strategy},
        )


def _adopt_results(
    results: list[CallResult],
    session: Session,
    warned: set[str],
    model: str | None = None,
) -> None:
    """Достраивает результаты, полученные в обход `_run()`.

    Стратегии зовут chat.complete() напрямую, поэтому две вещи, которые
    обычно делает `_run()`, приходится сделать здесь: подставить конкретную
    версию модели (API возвращает присланное имя — `-latest` так и остаётся
    `-latest`, см. CLAUDE.md) и предупредить о параметрах, срезанных по
    capabilities. `warned` копит уже сказанное: девять вызовов подряд иначе
    напечатали бы одно и то же предупреждение девять раз.
    """
    requested = model or session.config.model
    resolved = resolve_alias(session.models, requested) if session.models else None
    for result in results:
        if resolved:
            result.model_actual = resolved
        for name in result.skipped_params:
            if name not in warned:
                warned.add(name)
                console.warn(f"{requested} не поддерживает {name} — параметр не отправлен")


def _run(session: Session, messages: list[chat_core.Message]) -> chat_core.CallResult:
    config = session.config
    capabilities = session.capabilities

    # should_stream() — единая точка решения: format=json/schema всегда без
    # стрима (вердикт и поиск маркера завершения возможны только по целому
    # ответу), иначе — как задано --no-stream/config.stream.
    if chat_core.should_stream(config):
        result = chat_core.stream(config, messages, console.write_chunk, capabilities)
        console.finish_answer()
    else:
        result = chat_core.complete(config, messages, capabilities)
        # print_answer сама решает: json/schema с валидным ответом — pretty
        # print с отступами, иначе — сырой текст как есть.
        console.print_answer(result, config.params.format or "text")

    # API возвращает то же имя, что мы прислали, поэтому конкретную версию
    # подставляем из списка моделей.
    if resolved := session.resolved:
        result.model_actual = resolved
    if result.skipped_params:
        console.warn(
            f"{config.model} не поддерживает: {', '.join(result.skipped_params)} — "
            "параметры не отправлены"
        )
    return result


def _load_history() -> list[chat_core.Message]:
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [m for m in data if isinstance(m, dict) and m.get("role") and m.get("content")]


def _save_history(history: list[chat_core.Message]) -> None:
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
