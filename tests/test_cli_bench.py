"""Day 05 на уровне CLI: `advent w01 bench`.

Тот же приём, что в tests/test_cli_temp.py: сеть не трогается (`cli.list_models`
подменён фикстурой, `chat_core.complete` — в каждом тесте), журнал перехвачен
через `models_bench.log_call` (bench пишет в журнал через
`models_bench.log_bench_step`, а не напрямую через `cli.log_call` — другая
точка перехвата, чем у chat/solve/temp).
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
from advent_core.telemetry import CallResult, Usage
from week_01 import models_bench

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", _ANSI.sub("", text)).strip()


MODELS = [
    {
        "id": "ministral-3b-2512",
        "aliases": ["ministral-3b-latest"],
        "capabilities": {"completion_chat": True},
    },
    {
        "id": "ministral-8b-2512",
        "aliases": ["ministral-8b-latest"],
        "capabilities": {"completion_chat": True},
    },
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


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """ministral-14b-latest несёт снимок лимита 30 req/min (SPEC §13) — без
    подмены _maybe_pause реально засыпает между прогонами одной модели, и
    полный набор тестов этого файла превращается в секунды настоящего sleep()
    вместо мокнутого прогона (models_bench.py docstring прямо предупреждает
    об этом приёме)."""
    monkeypatch.setattr(models_bench.time, "sleep", lambda _seconds: None)


@pytest.fixture
def journal(monkeypatch):
    """bench пишет через models_bench.log_bench_step -> advent_core.journal.log_call,
    импортированный в week_01.models_bench напрямую — подмена только cli.log_call
    (как у chat/solve) сюда бы не достала (та же ловушка, что в tests/test_cli_temp.py
    у week_01.temperature)."""
    records: list[dict] = []

    def capture(result, messages, **kwargs):
        records.append({"messages": messages, **kwargs})

    monkeypatch.setattr(models_bench, "log_call", capture)
    return records


@pytest.fixture
def stdout_capture(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=200))
    return buffer


class _Complete:
    """Замена chat.complete: копит вызовы, отдаёт заготовленные тексты по очереди."""

    def __init__(self, *texts: str, default: str = "ОТВЕТ: 7"):
        self.texts = list(texts)
        self.default = default
        self.calls: list[tuple] = []

    def __call__(self, config, messages, capabilities=None) -> CallResult:
        self.calls.append((config, messages, capabilities))
        text = self.texts.pop(0) if self.texts else self.default
        return CallResult(
            text=text,
            model_requested=config.model,
            usage=Usage(100, 10),
            latency_ms=100,
            stream=False,
            finish_reason="stop",
            sent_messages=messages,
        )


def _bench(**overrides):
    kwargs = {
        "problem": None,
        "models": None,
        "runs": None,
        "system": None,
        "verbose": False,
    }
    kwargs.update(overrides)
    return cli.bench_command(**kwargs)


# --------------------------------------------------------------------------
# Журнал: day=5 литералом (CLAUDE.md — тест не должен брать ожидание из
# cli.BENCH_DAY, того же источника, что и код под тестом).
# --------------------------------------------------------------------------


def test_bench_command_journals_with_day_5_as_a_literal(monkeypatch, journal):
    complete = _Complete(default="ОТВЕТ: 7")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _bench(problem="children", models="ministral-3b-latest", runs=1)

    assert len(journal) == 1
    assert journal[0]["day"] == 5
    assert journal[0]["week"] == 1
    assert journal[0]["extra"]["problem"] == "children"
    assert journal[0]["extra"]["model"] == "ministral-3b-latest"


def test_bench_journal_has_one_line_per_call_of_the_full_sweep(monkeypatch, journal):
    complete = _Complete(default="ОТВЕТ: 7")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _bench(problem="children", models="ministral-3b-latest,ministral-8b-latest", runs=2)

    # 1 задача × 2 модели × 2 прогона = 4.
    assert len(journal) == 4
    assert {record["extra"]["model"] for record in journal} == {
        "ministral-3b-latest",
        "ministral-8b-latest",
    }


# --------------------------------------------------------------------------
# --models переопределяет лестницу; --problem/--runs режут развёртку
# --------------------------------------------------------------------------


def test_models_flag_overrides_the_default_ladder_to_a_subset(monkeypatch, journal):
    complete = _Complete(default="ОТВЕТ: 7")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _bench(problem="children", models="ministral-3b-latest", runs=1)

    assert len(complete.calls) == 1
    assert complete.calls[0][0].model == "ministral-3b-latest"


def test_default_ladder_and_default_runs_give_the_full_budget_on_one_problem(monkeypatch, journal):
    """Без --models/--runs: лестница из трёх моделей × 3 прогона из реестра."""
    complete = _Complete(default="ОТВЕТ: 7")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _bench(problem="children")

    assert len(complete.calls) == 9


def test_problem_children_runs_only_one_task(monkeypatch, journal):
    complete = _Complete(default="ОТВЕТ: 7")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _bench(problem="children", models="ministral-3b-latest", runs=1)

    assert len(complete.calls) == 1


def test_problem_all_runs_both_tasks_of_the_day(monkeypatch, journal):
    complete = _Complete(default="ОТВЕТ: 7")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _bench(problem="all", models="ministral-3b-latest", runs=1)

    # children + sky = 2 вызова, ни одного сверх задач дня.
    assert len(complete.calls) == 2


# --------------------------------------------------------------------------
# Ранние (досетевые) отказы
# --------------------------------------------------------------------------


def test_an_unknown_problem_id_is_refused_before_the_first_call(monkeypatch, journal):
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    with pytest.raises(ConfigError):
        _bench(problem="нет-такой-задачи")

    assert complete.calls == []


def test_an_unknown_model_name_is_refused_before_the_first_call(monkeypatch, journal):
    complete = _Complete()
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    with pytest.raises(ConfigError):
        _bench(problem="children", models="модель-которой-нет-в-аккаунте")

    assert complete.calls == [], "опечатка в имени модели не должна стоить ни одного вызова"


# --------------------------------------------------------------------------
# stdout/stderr контракт
# --------------------------------------------------------------------------


def test_bench_answers_reach_stdout_progress_stays_in_stderr(monkeypatch, journal, capsys):
    complete = _Complete("уникальный ответ модели\nОТВЕТ: 7")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _bench(problem="children", models="ministral-3b-latest", runs=1)

    captured = capsys.readouterr()
    assert "уникальный ответ модели" in captured.out
    assert "уникальный ответ модели" not in captured.err


# --------------------------------------------------------------------------
# temperature=0 зафиксирована на всю развёртку (SPEC §12) — на полном пути CLI
# --------------------------------------------------------------------------


def test_bench_pins_temperature_to_zero_even_when_the_environment_sets_another(
    monkeypatch, journal, capsys
):
    monkeypatch.setenv("MISTRAL_TEMPERATURE", "0.9")
    complete = _Complete(default="ОТВЕТ: 7")
    monkeypatch.setattr(cli.chat_core, "complete", complete)

    _bench(problem="children", models="ministral-3b-latest", runs=1)

    assert complete.calls[0][0].params.temperature == 0.0
    assert "temperature" in _flat(capsys.readouterr().err)
