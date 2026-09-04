"""Day 04 на уровне CLI: `advent w01 temp` и потолок `--temperature` у `chat`.

До этого файла ничего не проверяло §15 SPEC-w01d04.md на уровне команды:
разворот `--problem all`/одиночного id, ранние (досетевые) отказы на
неизвестной задаче и недоступной модели судьи, переопределение умолчаний
`--temps`/`--runs`. Сеть не трогается: `cli.list_models` подменён фикстурой,
`chat_core.complete` — в каждом тесте. Журнал тоже подменён.
"""

from __future__ import annotations

import io
import re

import pytest
from rich.console import Console

import week_01.cli as cli
from advent_core import config as config_module
from advent_core import console
from advent_core.config import ConfigError
from advent_core.errors import AdventError
from advent_core.telemetry import CallResult, Usage
from week_01 import strategies, temperature

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", _ANSI.sub("", text)).strip()


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
    # первого complete() (найдено по красным test_cli_temp.py, отчёт фазы 1).
    {
        "id": "ministral-14b-2512",
        "aliases": ["ministral-14b-latest"],
        "capabilities": {"completion_chat": True},
    },
]

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
    """Перехватывает обе точки записи в JSONL — из week_01.temperature и из cli.

    temperature.py импортирует `log_call` напрямую (`from advent_core.journal
    import log_call`), поэтому подмены одного только cli.log_call недостаточно
    — та же ловушка, что у tests/test_cli_solve.py со strategies.log_call.
    """
    records: list[dict] = []

    def capture(result, messages, **kwargs):
        records.append({"messages": messages, **kwargs})

    monkeypatch.setattr(cli, "log_call", capture)
    monkeypatch.setattr(temperature, "log_call", capture)
    return records


@pytest.fixture
def stdout_capture(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=200))
    return buffer


class _Complete:
    """Замена chat.complete: копит вызовы, отдаёт заготовленные тексты по очереди."""

    def __init__(self, *texts: str, default: str = "ОТВЕТ: 3"):
        self.texts = list(texts)
        self.default = default
        self.calls: list[tuple] = []

    def __call__(self, config, messages, capabilities=None) -> CallResult:
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


def _temp(**overrides):
    """Зовёт команду `temp` с её штатными умолчаниями CLI (typer.Option объекты).

    Без judge/judge_model: `temp` их больше не читает вовсе — LLM-судья дня 04
    убран целиком (пользовательское решение), эти два флага остались только у
    `solve` (Day 03), см. week_01/cli.py.temp_command().
    """
    kwargs = {
        "problem": None,
        "temps": None,
        "runs": None,
        "model": None,
        "system": None,
        "verbose": False,
    }
    kwargs.update(overrides)
    return cli.temp_command(**kwargs)


def _chat(**overrides):
    kwargs = {
        "question": None,
        "model": None,
        "system": None,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "seed": None,
        "stop": None,
        "reasoning_effort": None,
        "format_": None,
        "schema_file": None,
        "done": None,
        "mode": None,
        "max_turns": None,
        "strategy": None,
        "no_stream": False,
        "verbose": False,
    }
    kwargs.update(overrides)
    return cli.chat_command(**kwargs)


# --------------------------------------------------------------------------
# --temperature 2 у `chat`: потолок отвергается локально (SPEC §12, п.3)
# --------------------------------------------------------------------------


def test_temperature_above_the_ceiling_is_refused_before_any_network_call(monkeypatch, journal):
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    with pytest.raises(ConfigError):
        _chat(question="привет", temperature=2)

    assert complete.calls == [], "отказ обязан быть локальным, без единого вызова API"


def test_temperature_at_the_ceiling_is_accepted(monkeypatch, journal):
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _chat(question="привет", temperature=1.5)

    assert len(complete.calls) == 1
    assert complete.calls[0][0].params.temperature == 1.5


# --------------------------------------------------------------------------
# --problem: all разворачивается в обе задачи, одиночный id — в одну
# --------------------------------------------------------------------------


def test_problem_all_expands_into_all_three_tasks_of_the_day(monkeypatch, journal, stdout_capture):
    complete = _Complete(default="ОТВЕТ: 3")
    # Порядок вызовов следует week_01.temperature.TEMP_PROBLEMS: digits5,
    # alice, coffee — ровно по одному вызову на задачу, без добавки на судью
    # (LLM-судья дня 04 убран целиком, пользовательское решение).
    complete.texts = ["ОТВЕТ: 225", "ОТВЕТ: 3", "Морской бриз"]
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="all", temps="0", runs=1)

    # digits5 + alice + coffee = 3, ни одного вызова сверх задач дня.
    assert len(complete.calls) == 3
    printed = stdout_capture.getvalue()
    assert "Сёстры брата Алисы" in printed or "alice" in printed.lower()
    assert "digits5" in printed.lower() or "трёхзначных" in printed.lower()


def test_problem_alice_runs_only_one_task(monkeypatch, journal, stdout_capture):
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0", runs=1)

    assert len(complete.calls) == 1


def test_problem_coffee_runs_only_one_task_with_no_judge_call(monkeypatch, journal):
    """coffee — open-задача, но LLM-судья дня 04 убран целиком: один прогон,
    ни одного вызова сверх него (пользовательское решение)."""
    complete = _Complete(default="Морской бриз")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="coffee", temps="0", runs=1)

    assert len(complete.calls) == 1


# --------------------------------------------------------------------------
# Ранние (досетевые) отказы
# --------------------------------------------------------------------------


def test_an_unknown_problem_id_is_refused_before_the_first_call(monkeypatch, journal):
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    with pytest.raises(ConfigError):
        _temp(problem="нет-такой-задачи")

    assert complete.calls == []


def test_an_invalid_temps_element_is_refused_before_the_first_call(monkeypatch, journal):
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    with pytest.raises(ConfigError):
        _temp(temps="0,1.51")

    assert complete.calls == []


# --------------------------------------------------------------------------
# --temps и --runs переопределяют умолчания команды
# --------------------------------------------------------------------------


def test_temps_flag_overrides_the_command_default(monkeypatch, journal):
    """Умолчание — 0,0.7,1.2 (три); --temps 0,0.7 обязан урезать развёртку до двух."""
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0,0.7", runs=1)

    assert len(complete.calls) == 2


def test_runs_flag_overrides_the_command_default(monkeypatch, journal):
    """Умолчание команды temp — 3 прогона; --runs 1 обязан урезать развёртку."""
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0", runs=1)

    assert len(complete.calls) == 1


def test_default_runs_and_temps_give_the_full_budget(monkeypatch, journal):
    """Без флагов — умолчания команды: 3 температуры × 3 прогона на alice = 9."""
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice")

    assert len(complete.calls) == 9


# --------------------------------------------------------------------------
# Журнал и печать целиком через CLI
# --------------------------------------------------------------------------


def test_temp_command_journals_one_line_per_call(monkeypatch, journal):
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0,0.7", runs=1)

    assert len(journal) == 2
    assert {record["extra"]["problem"] for record in journal} == {"alice"}
    # Литерал 4, а не {cli.DAY}: сверка с той же константой, из которой
    # журнал заполняется, тавтологична и проходит при любом значении DAY —
    # именно так дефект «DAY=3 в дне 04» доехал до review незамеченным
    # (находка tests #12, review-w01d04.md).
    assert {record["day"] for record in journal} == {4}


def test_temp_command_answers_reach_stdout_progress_stays_in_stderr(monkeypatch, journal, capsys):
    complete = _Complete("уникальный ответ модели\nОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0", runs=1)

    captured = capsys.readouterr()
    assert "уникальный ответ модели" in captured.out
    assert "уникальный ответ модели" not in captured.err


def test_temp_command_prints_the_human_review_block_for_an_open_problem(
    monkeypatch, journal, stdout_capture
):
    """coffee — open-задача: вместо вердикта судьи (убран целиком) команда
    печатает блок ручной оценки со всеми ответами."""
    complete = _Complete(default="Морской бриз")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="coffee", temps="0", runs=1)

    printed = stdout_capture.getvalue()
    assert "оценка креативности за человеком" in printed
    assert "ранжирование судьи" not in printed
    assert "судья" not in printed


def test_temp_command_does_not_print_the_human_review_block_for_a_non_open_problem(
    monkeypatch, journal, stdout_capture
):
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0", runs=1)

    assert "оценка креативности" not in stdout_capture.getvalue()


def test_a_mid_sweep_failure_does_not_kill_the_already_journaled_calls(monkeypatch, journal):
    complete = _Complete(default="ОТВЕТ: 3")

    def failing(config, messages, capabilities=None):
        if len(complete.calls) == 1:
            raise AdventError("rate limit")
        return complete(config, messages, capabilities)

    monkeypatch.setattr(cli.chat_core, "complete", failing)

    with pytest.raises(AdventError):
        _temp(problem="alice", temps="0,0.7,1.2", runs=1)

    assert len(journal) == 1


# --------------------------------------------------------------------------
# Персона и top_p на полном пути CLI (Config.resolve подставляет DEFAULT_SYSTEM_PROMPT)
# --------------------------------------------------------------------------


def test_temp_command_strips_the_project_persona_by_default(monkeypatch, journal):
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0", runs=1)

    persona = config_module.DEFAULT_SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    system = next((m["content"] for m in complete.calls[0][1] if m["role"] == "system"), "")
    assert persona not in system


def test_temp_command_keeps_an_explicit_system_prompt(monkeypatch, journal, tmp_path):
    path = tmp_path / "system.md"
    path.write_text("ОСОБЫЙ SYSTEM ПОЛЬЗОВАТЕЛЯ", encoding="utf-8")
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0", runs=1, system=path)

    system = next((m["content"] for m in complete.calls[0][1] if m["role"] == "system"), "")
    assert "ОСОБЫЙ SYSTEM ПОЛЬЗОВАТЕЛЯ" in system


def test_temp_command_warns_about_top_p_from_the_environment(monkeypatch, journal, capsys):
    monkeypatch.setenv("MISTRAL_TOP_P", "0.9")
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0", runs=1)

    assert complete.calls[0][0].params.top_p is None
    assert "top_p" in _flat(capsys.readouterr().err)


# --------------------------------------------------------------------------
# strategies.load_problem() и load_temp_problems используются, не переизобретаются
# --------------------------------------------------------------------------


def test_temp_command_uses_the_shared_problem_bank(monkeypatch, journal):
    """Стык с week_01/strategies.py: команда обязана читать тот же банк."""
    complete = _Complete(default="ОТВЕТ: 3")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _temp(problem="alice", temps="0", runs=1)

    assert complete.calls[0][1][-1]["content"] == strategies.load_problem("alice").statement
