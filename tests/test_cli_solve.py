"""Обе поверхности дня 03 на уровне CLI: `advent w01 solve` и `chat --strategy`.

До этого файла §14 спеки («четыре стратегии работают из обеих поверхностей»)
не проверялся ничем: `tests/test_cli_dialog.py` целиком про день 02, а
`tests/test_strategies.py` не импортирует cli вовсе. Не покрыто было всё
склеивающее: разворот `all` в четыре имени, порядок и число вызовов у команды
`solve`, пошаговая запись в журнал, capabilities судьи при `--judge-model`,
ранние проверки задачи и модели судьи, ветка стратегии в REPL и `/params`.
Стоимость дыры измеримая: если `_strategy_names(None)` начнёт возвращать
`('direct',)`, `advent w01 solve` сделает один вызов вместо восьми, таблица
покажет одну строку — и до этого файла ни один тест этого не заметил бы.

Сеть не трогается: `cli.list_models` подменён фикстурой, `chat_core.complete` —
в каждом тесте. Журнал тоже подменён — иначе тесты дописывали бы строки в
настоящий logs/calls.jsonl.
"""

from __future__ import annotations

import io
import json
import re

import pytest
from rich.console import Console

import week_01.cli as cli
from advent_core import config as config_module
from advent_core import console
from advent_core.config import ConfigError
from advent_core.errors import AdventError
from advent_core.params import GenerationParams
from advent_core.telemetry import CallResult, Usage
from week_01 import strategies

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    """Снимает ANSI-подсветку и схлопывает перенос Rich в пробел.

    Rich переносит длинную строку по ширине консоли и подсвечивает «key=value»
    отдельными escape-последовательностями: без этого подстрочная проверка
    рвётся посреди честного и неизменного текста.
    """
    return re.sub(r"\s+", " ", _ANSI.sub("", text)).strip()


# Две модели с РАЗНЫМИ capabilities: на этом держится проверка того, что
# судья фильтрует параметры по своей карточке, а не по карточке решателя.
MODELS = [
    {
        "id": "mistral-small-2603",
        "aliases": ["mistral-small-latest"],
        "capabilities": {"completion_chat": True},
    },
    {
        "id": "magistral-medium-2509",
        "aliases": ["magistral-medium-latest"],
        "capabilities": {"completion_chat": True, "reasoning": True},
    },
    # DEFAULT_MODEL сменился на ministral-14b-latest (SPEC-w01d05.md §15) —
    # Session.refresh() валидирует config.model против этого списка, и без
    # записи любой вызов без явного --model падает ConfigError'ом ещё до
    # первого complete() (найдено по красным test_cli_solve.py, отчёт фазы 1).
    {
        "id": "ministral-14b-2512",
        "aliases": ["ministral-14b-latest"],
        "capabilities": {"completion_chat": True},
    },
]

# Ключи .env, которые иначе просочились бы в Config.resolve() с машины
# разработчика и сделали бы результат теста зависящим от чужого файла.
_ENV_KEYS = (
    "MISTRAL_MODEL",
    "MISTRAL_TEMPERATURE",
    "MISTRAL_TOP_P",
    "MISTRAL_MAX_TOKENS",
    "MISTRAL_SEED",
    "MISTRAL_STOP",
    "MISTRAL_REASONING_EFFORT",
    "ADVENT_SYSTEM_PROMPT",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """`Config.resolve()` читает .env — в тесте он не должен решать ничего.

    load_env() глушится, ключ подставляется свой, остальные переменные
    снимаются: иначе выставленный у разработчика MISTRAL_REASONING_EFFORT
    менял бы payload и тест краснел бы на чужой машине.
    """
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(cli, "list_models", lambda config: MODELS)
    monkeypatch.setattr(cli.chat_core, "should_stream", lambda config: False)


@pytest.fixture
def journal(monkeypatch):
    """Перехватывает обе точки записи в JSONL — из strategies и из cli."""
    records: list[dict] = []

    def capture(result, messages, **kwargs):
        records.append({"messages": messages, **kwargs})

    monkeypatch.setattr(strategies, "log_call", capture)
    monkeypatch.setattr(cli, "log_call", capture)
    return records


@pytest.fixture
def stdout_capture(monkeypatch):
    """Широкая консоль в буфер: Rich иначе режет таблицу под 80 колонок."""
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=200))
    return buffer


class _Complete:
    """Замена chat.complete: копит вызовы и отдаёт заготовленные тексты.

    Судейский JSON подставляется по умолчанию последним: иначе каждый тест
    команды `solve` был бы обязан помнить, что девятый вызов — судья.
    """

    def __init__(self, *texts: str, default: str = "ОТВЕТ: 42"):
        self.texts = list(texts)
        self.default = default
        self.calls: list[tuple] = []
        self.before_call: list[int] = []
        self.watch: list | None = None

    def __call__(self, config, messages, capabilities=None) -> CallResult:
        if self.watch is not None:
            self.before_call.append(len(self.watch))
        self.calls.append((config, messages, capabilities))
        text = self.texts.pop(0) if self.texts else self.default
        return CallResult(
            text=text,
            model_requested=config.model,
            usage=Usage(10, 5, 15),
            latency_ms=100,
            stream=False,
            finish_reason="stop",
            sent_messages=messages,
        )

    def system_of(self, index: int) -> str:
        messages = self.calls[index][1]
        return next((m["content"] for m in messages if m["role"] == "system"), "")

    def user_of(self, index: int) -> str:
        return self.calls[index][1][-1]["content"]


JUDGE_JSON = json.dumps(
    {
        "scores": [
            {"label": label, "score": 5, "comment": "комментарий"}
            for label in strategies.JUDGE_LABELS
        ],
        "ranking": list(strategies.JUDGE_LABELS),
    },
    ensure_ascii=False,
)


def _solve(**overrides):
    """Зовёт команду `solve` с её штатными умолчаниями CLI.

    Аргументы перечислены полностью: значения по умолчанию в сигнатуре —
    объекты typer.Option, а не None, и вызвать функцию «как из терминала» без
    этого нельзя.
    """
    kwargs = {
        "problem": None,
        "strategy": None,
        "runs": None,
        "judge": None,
        "judge_model": None,
        "model": None,
        "system": None,
        "verbose": False,
    }
    kwargs.update(overrides)
    return cli.solve_command(**kwargs)


def _session(**param_kwargs) -> cli.Session:
    return cli.Session(
        config_module.Config(
            api_key="k" * 32,
            model="mistral-small-latest",
            params=GenerationParams.build(**param_kwargs),
        )
    )


# --------------------------------------------------------------------------
# Разворот `all` — от него зависят и бюджет вызовов, и метки судьи
# --------------------------------------------------------------------------


def test_all_expands_into_every_strategy_in_the_registry_order():
    """Если бы это вернуло ('direct',), solve сделал бы 1 вызов вместо 8 — молча."""
    assert cli._strategy_names(None) == strategies.STRATEGIES
    assert cli._strategy_names("all") == strategies.STRATEGIES
    assert list(strategies.STRATEGIES) == ["direct", "steps", "meta", "panel"]


def test_a_single_strategy_stays_single():
    assert cli._strategy_names("panel") == ("panel",)


# --------------------------------------------------------------------------
# `advent w01 solve` целиком
# --------------------------------------------------------------------------


def test_solve_spends_the_budget_of_the_day_and_prints_the_table(
    monkeypatch, journal, stdout_capture
):
    """Восемь вызовов на решение плюс один на судью (SPEC §2), таблица — итог."""
    complete = _Complete(default="ОТВЕТ: 3")
    complete.texts = ["ОТВЕТ: 2", "по шагам\nОТВЕТ: 3", "сгенерированный промпт", "ОТВЕТ: 3"]
    complete.texts += ["ОТВЕТ: 3"] * 4 + [JUDGE_JSON]
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve()

    assert len(complete.calls) == 9
    printed = stdout_capture.getvalue()
    for column in ("стратегия", "ответ", "эталон", "судья", "вызовов", "токены", "время"):
        assert column in printed
    for name in strategies.STRATEGIES:
        assert name in printed
    assert "ранжирование судьи" in printed


def test_solve_answers_reach_stdout_and_progress_stays_in_stderr(monkeypatch, journal, capsys):
    """Контракт потоков дня: ответ модели — в stdout, обстановка — в stderr."""
    complete = _Complete("уникальный текст ответа\nОТВЕТ: 3", default=JUDGE_JSON)
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve(strategy="direct")

    captured = capsys.readouterr()
    assert "уникальный текст ответа" in captured.out
    assert "уникальный текст ответа" not in captured.err
    stderr = _flat(captured.err)
    assert "эталон:" in stderr, "условие и эталон — обстановка, им место в stderr"


def test_solve_writes_one_journal_line_per_call_labelled_by_strategy(monkeypatch, journal):
    """Единица записи — вызов (SPEC §2): по нему неделя 2 будет считать токены."""
    complete = _Complete(default=JUDGE_JSON)
    complete.texts = ["ОТВЕТ: 3"] * 8 + [JUDGE_JSON]
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve()

    assert len(journal) == 9
    assert [record["extra"]["strategy"] for record in journal] == [
        "direct",
        "steps",
        "meta",
        "meta",
        "panel",
        "panel",
        "panel",
        "panel",
        "judge",
    ]
    assert {record["extra"]["problem"] for record in journal} == {strategies.load_problem().id}
    assert {record["day"] for record in journal} == {cli.DAY}


def test_solve_journals_each_call_before_making_the_next_one(monkeypatch, journal):
    """Запись идёт по факту вызова, а не пачкой в конце.

    Проверяется именно момент: перед N-м вызовом в журнале обязано лежать
    ровно N−1 строк. Пострановая (и тем более постратегийная) запись дала бы
    здесь нули до самого конца — и упавший вызов уносил бы всё предыдущее.
    """
    complete = _Complete(default="ОТВЕТ: 3")
    complete.watch = journal
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve(strategy="panel", judge=False)

    assert complete.before_call == [0, 1, 2, 3]


def test_a_mid_strategy_failure_keeps_the_paid_calls_in_the_journal(
    monkeypatch, journal, stdout_capture
):
    """429 на четвёртом вызове панели не должен уносить три оплаченных.

    Журнал — единственное, что этот день сохраняет: отдельного `--out` нет.
    Таблица при обрыве не печатается сознательно — она сводит то, чего не
    досчитали.
    """
    complete = _Complete(default="ОТВЕТ: 3")

    def failing(config, messages, capabilities=None):
        if len(complete.calls) == 3:
            raise AdventError("rate limit")
        return complete(config, messages, capabilities)

    monkeypatch.setattr(cli.chat_core, "complete", failing)

    with pytest.raises(AdventError):
        _solve(strategy="panel")

    assert [record["extra"]["step"] for record in journal] == ["analyst", "engineer", "critic"]
    assert "стратегия" not in stdout_capture.getvalue(), "таблицы недосчитанного быть не должно"


def test_solve_without_a_judge_makes_no_judge_call(monkeypatch, journal, stdout_capture):
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve(strategy="direct", judge=False)

    assert len(complete.calls) == 1
    assert "ранжирование судьи" not in stdout_capture.getvalue()


def test_runs_option_repeats_every_strategy(monkeypatch, journal, stdout_capture):
    complete = _Complete(default="ОТВЕТ: 3")
    complete.texts = ["ОТВЕТ: 3", "ОТВЕТ: 2", JUDGE_JSON]
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve(strategy="direct", runs=2)

    assert len(complete.calls) == 3, "два прогона плюс судья"
    assert [record["extra"]["run"] for record in journal[:2]] == [1, 2]
    assert "1/2" in stdout_capture.getvalue(), "колонка эталона показывает k/N"


def test_solve_does_not_mix_the_project_persona_into_the_strategies(monkeypatch, journal):
    """Замер дня, который легко потерять правкой: персона обнуляет эффект steps.

    Штатный system-пресет («лаконичный ассистент, без воды») противоречит
    инструкции рассуждать пошагово, и модель слушается его: с персоной steps
    даёт верный ответ 2 раза из 5 при 679 символах, без неё — 4 из 5 при 1232.
    Проверяется на пути CLI, потому что именно здесь Config.resolve()
    подставляет DEFAULT_SYSTEM_PROMPT — в юнит-тестах strategies его нет вовсе.
    """
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve(strategy="direct", judge=False)

    persona = config_module.DEFAULT_SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    system = complete.system_of(0)
    assert persona not in system
    assert strategies.marker_instruction() in system, "общая добавка про маркер уходить обязана"


def test_an_explicit_system_prompt_survives_into_the_strategy(monkeypatch, journal, tmp_path):
    """Явно заданный --system выбран осознанно — молча его игнорировать нельзя."""
    path = tmp_path / "system.md"
    path.write_text("ОСОБЫЙ SYSTEM ПОЛЬЗОВАТЕЛЯ", encoding="utf-8")
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve(strategy="direct", judge=False, system=path)

    assert "ОСОБЫЙ SYSTEM ПОЛЬЗОВАТЕЛЯ" in complete.system_of(0)


# --------------------------------------------------------------------------
# Судья: capabilities считаются по ЕГО модели (НАХОДКИ 6 и 12)
# --------------------------------------------------------------------------


def test_judge_gets_the_capabilities_of_its_own_model(monkeypatch, journal, stdout_capture):
    """Иначе reasoning_effort от magistral уезжает в mistral-small — 400 вместо оценки.

    Прогон не падает: ошибка уходит в JudgeVerdict.error и колонка «судья»
    молча становится «—» уже после девяти оплаченных вызовов.
    """
    complete = _Complete(default=JUDGE_JSON)
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve(strategy="direct", model="magistral-medium-latest", judge_model="mistral-small-latest")

    solving_caps = complete.calls[0][2]
    judge_caps = complete.calls[-1][2]
    assert solving_caps == {"completion_chat": True, "reasoning": True}
    assert judge_caps == {"completion_chat": True}, "карточка судьи, а не решателя"
    assert complete.calls[-1][0].model == "mistral-small-latest"


def test_judge_capabilities_match_the_solver_when_no_judge_model_is_given(
    monkeypatch, journal, stdout_capture
):
    """Обратная сторона той же правки: без --judge-model ничего не меняется."""
    complete = _Complete(default=JUDGE_JSON)
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _solve(strategy="direct", model="magistral-medium-latest")

    assert complete.calls[-1][2] == complete.calls[0][2]


def test_an_unavailable_judge_model_is_refused_before_the_first_call(monkeypatch, journal):
    """Узнать про опечатку после девяти вызовов — значит оплатить прогон впустую."""
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    with pytest.raises(ConfigError):
        _solve(judge_model="нет-такой-модели")

    assert complete.calls == []


def test_an_unknown_problem_is_refused_before_the_first_call(monkeypatch, journal):
    """`_check_local_param('problem')` — самая дешёвая точка сообщить об ошибке."""
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    with pytest.raises(ConfigError) as excinfo:
        _solve(problem="нет-такой-задачи")

    assert complete.calls == []
    assert strategies.load_problem().id in str(excinfo.value), "ошибка перечисляет доступные id"


# --------------------------------------------------------------------------
# Вторая поверхность: chat --strategy и /set strategy в REPL
# --------------------------------------------------------------------------


def test_chat_strategy_runs_the_strategy_and_prints_no_verdict(monkeypatch, journal, capsys):
    """У вопроса из чата эталона нет — «✗» там был бы прямой ложью."""
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)
    session = _session(strategy="panel")

    cli._ask_with_strategy(session, "Сколько сестёр у брата Алисы?")

    assert len(complete.calls) == 4
    assert [record["extra"]["step"] for record in journal] == [
        "analyst",
        "engineer",
        "critic",
        "synthesis",
    ]
    assert {record["extra"]["problem"] for record in journal} == {strategies.CHAT_PROBLEM_ID}
    stderr = _flat(capsys.readouterr().err)
    assert "извлечено:" not in stderr
    assert "эталон:" not in stderr


def test_chat_strategy_all_covers_every_strategy_from_the_second_surface(
    monkeypatch, journal, capsys
):
    """§14: четыре стратегии обязаны работать из ОБЕИХ поверхностей."""
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    cli._ask_with_strategy(_session(strategy="all"), "вопрос")

    assert len(complete.calls) == 8
    assert [record["extra"]["strategy"] for record in journal] == (
        ["direct", "steps", "meta", "meta"] + ["panel"] * 4
    )


def test_chat_strategy_warns_about_an_unsafe_stop_only_once(monkeypatch, journal, capsys):
    """Четыре стратегии подряд печатали одно и то же предупреждение четыре раза."""
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    cli._ask_with_strategy(_session(strategy="all", stop="ОТВЕТ:"), "вопрос")

    assert _flat(capsys.readouterr().err).count("пересекается с маркером") == 1
    assert all(call[0].params.stop is None for call in complete.calls)


def test_repl_routes_a_question_through_the_active_strategy(monkeypatch, journal, tmp_path, capsys):
    """`/set strategy steps` + вопрос — это стратегия, а не обычный чат."""
    config = config_module.Config(
        api_key="k" * 32, model="mistral-small-latest", params=GenerationParams.build()
    )
    monkeypatch.setattr(cli, "HISTORY_PATH", tmp_path / "history.json")
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    lines = iter(["/set strategy steps", "Сколько сестёр у брата Алисы?"])

    def fake_prompt(*args, **kwargs):
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr(cli.typer, "prompt", fake_prompt)

    cli._repl(config)

    assert len(complete.calls) == 1
    assert strategies.load_prompt("steps") in complete.system_of(0)
    assert not (tmp_path / "history.json").exists(), (
        "стратегия строит messages с нуля — дописывать её ответ в историю значило бы "
        "показать модели разговор, которого она не видела"
    )


def test_repl_says_out_loud_that_a_strategy_is_ignored_in_dialog_mode(
    monkeypatch, journal, tmp_path, capsys
):
    """Совместить многоходовый диалог и один независимый прогон нельзя.

    Молча выбрать одно из двух хуже, чем сказать, что именно выполняется.
    """
    config = config_module.Config(
        api_key="k" * 32,
        model="mistral-small-latest",
        params=GenerationParams.build(mode="dialog", done="text:ГОТОВО", strategy="panel"),
    )
    monkeypatch.setattr(cli, "HISTORY_PATH", tmp_path / "history.json")
    complete = _Complete(default="ГОТОВО")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    lines = iter(["вопрос"])

    def fake_prompt(*args, **kwargs):
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr(cli.typer, "prompt", fake_prompt)

    cli._repl(config)

    stderr = _flat(capsys.readouterr().err)
    assert "не применяется в mode=dialog" in stderr
    assert len(complete.calls) == 1, "ушёл обычный диалоговый вызов, а не четыре вызова панели"
    assert strategies.load_prompt("panel_analyst") not in complete.system_of(0)


def test_again_repeats_the_question_through_the_same_strategy(monkeypatch, journal, capsys):
    """Повтор обязан быть повтором того же самого, иначе сравнивались бы разные механики."""
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)
    session = _session(strategy="steps")
    session.last_question = "Сколько сестёр у брата Алисы?"

    cli._handle_again(session)

    assert len(complete.calls) == 1
    assert strategies.load_prompt("steps") in complete.system_of(0)


def test_a_failed_strategy_question_does_not_kill_the_repl(monkeypatch, journal, capsys):
    """В REPL ошибка печатается и жизнь продолжается — сессия не должна падать."""

    def failing(config, messages, capabilities=None):
        raise AdventError("сервис недоступен")

    monkeypatch.setattr(cli.chat_core, "complete", failing)

    cli._strategy_question(_session(strategy="direct"), "вопрос")  # не должно бросить

    assert "недоступен" in _flat(capsys.readouterr().err)
    assert journal[0]["error"] == "сервис недоступен"


def test_direct_stays_the_ordinary_chat_path(monkeypatch, journal, capsys):
    """direct — это отсутствие добавок, а не пятая стратегия.

    Иначе каждый обычный `chat "привет"` начал бы требовать строку ОТВЕТ: и
    потерял бы стрим.
    """
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)
    config = config_module.Config(
        api_key="k" * 32,
        model="mistral-small-latest",
        system_prompt_path=None,
        params=GenerationParams.build(strategy="direct"),
    )

    cli._ask_once(config, "привет")

    assert strategies.ANSWER_MARKER not in complete.system_of(0)


# --------------------------------------------------------------------------
# /params и /set: то, что читает пользователь между шагами демо
# --------------------------------------------------------------------------


def test_params_does_not_promise_the_shown_system_goes_into_the_strategy(capsys):
    """НАХОДКА 23: «поверх этого system» — ложь, пресет проекта туда не уходит."""
    session = _session(strategy="steps")
    session.config.system_prompt_path = config_module.DEFAULT_SYSTEM_PROMPT

    cli._show_params(session)

    stderr = _flat(capsys.readouterr().err)
    assert "поверх этого system" not in stderr
    assert "в стратегию не подмешивается" in stderr
    assert "НЕ уходит" in stderr


def test_params_says_an_explicit_system_is_kept(tmp_path, capsys):
    session = _session(strategy="steps")
    path = tmp_path / "system.md"
    path.write_text("мой system", encoding="utf-8")
    session.config.system_prompt_path = path

    cli._show_params(session)

    assert "явно заданный system при этом сохраняется" in _flat(capsys.readouterr().err)


def test_params_does_not_promise_a_strategy_prompt_in_dialog_mode(capsys):
    """НАХОДКА 13: в mode=dialog REPL стратегию не применяет вовсе."""
    session = _session(strategy="panel", mode="dialog", done="text:ГОТОВО")

    cli._show_params(session)

    stderr = _flat(capsys.readouterr().err)
    assert "не применяется в mode=dialog" in stderr
    assert "reason_*.md" not in stderr


def test_params_stays_quiet_about_strategy_when_there_is_none(capsys):
    """Ветка обязана уметь молчать — иначе соседние тесты ловили бы что угодно."""
    cli._show_params(_session(strategy="direct"))

    assert "стратег" not in _flat(capsys.readouterr().err).lower()


def test_set_of_a_solve_only_param_says_it_does_nothing_here(capsys):
    """Выставленный параметр, ни на что не влияющий, читается как поломка."""
    session = _session()

    cli._handle_set(["runs", "3"], session)

    assert "читает только команда solve" in _flat(capsys.readouterr().err)


def test_set_problem_checks_the_bank_immediately(capsys):
    """Опечатка в id иначе всплыла бы после девяти оплаченных вызовов."""
    session = _session()

    cli._handle_set(["problem", "нет-такой-задачи"], session)

    assert "не найдена" in _flat(capsys.readouterr().err)
