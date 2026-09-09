"""Интерфейс агента (week_02/cli.py) — сеть не трогается, кредиты не тратятся.

`chat_core.complete` подменяется ДО создания AgentShell: агент забирает шов
вызова в конструкторе (advent_core/agent.py), и подмена после создания до него
уже не дотягивается — та же ловушка «дефолт связывается на импорте», за
которую заплатили в дне 03 (CLAUDE.md).

`counter_for` подменяется тоже: настоящий счётчик качает 16.7 МБ токенизатора,
а тестам нужен счёт, а не сеть.
"""

from __future__ import annotations

import inspect
import re
import sys

import pytest

import week_02.cli as cli
from advent_core import console
from advent_core.agent import INTERRUPT_NOTE
from advent_core.config import Config, ConfigError
from advent_core.errors import AdventError
from advent_core.params import AGENT_COMMAND, defaults_for
from advent_core.session import Session
from advent_core.telemetry import CallResult, Usage
from advent_core.tokens import EstimateCounter

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Снимает ANSI-подсветку Rich: без неё substring-проверка рвётся посреди
    честного и неизменного текста."""
    return _ANSI.sub("", text)


def _flat(text: str) -> str:
    """То же плюс схлопнутые переносы.

    Rich переносит длинную строку по ширине консоли (80 в тестах), и проверка
    вроде «счёт токенов теперь другие» рвётся ровно посередине честного и
    неизменного предупреждения. Схлопывание возвращает пробел на место
    переноса — сравнивается смысл, а не раскладка терминала.
    """
    return re.sub(r"\s+", " ", _plain(text))


MODELS = [
    {
        "id": "ministral-14b-2512",
        "aliases": ["ministral-14b-latest"],
        "capabilities": {"completion_chat": True},
        "max_context_length": 128_000,
    },
    {
        "id": "ministral-3b-2512",
        "aliases": ["ministral-3b-latest"],
        "capabilities": {"completion_chat": True},
        "max_context_length": 32_000,
    },
]


@pytest.fixture(autouse=True)
def no_journal(monkeypatch):
    """Тесты не дописывают строки в настоящий logs/calls.jsonl — по нему
    считаются токены недели, и выдуманные диалоги в нём означают испорченный
    подсчёт. Тесты, которым журнал нужен по существу, подменяют log_call сами."""
    monkeypatch.setattr(cli, "log_call", lambda result, messages, **kwargs: None)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Ни списка моделей по сети, ни скачивания токенизатора."""
    monkeypatch.setattr(cli, "list_models", lambda config: MODELS)
    monkeypatch.setattr(cli, "counter_for", lambda model, **kwargs: (EstimateCounter(), None))


def _config(**params) -> Config:
    from advent_core.params import GenerationParams

    config = Config(
        api_key="k" * 32,
        model="ministral-14b-latest",
        # Персона читается только когда system prompt остался проектным
        # умолчанием; здесь его нет вовсе, чтобы тесты не зависели от текста
        # week_02/prompts/agent.md.
        system_prompt_path=None,
        params=GenerationParams.build(**params),
    )
    config.params.apply_defaults(AGENT_COMMAND)
    # Стрим выключен: тесты подменяют complete, а не stream.
    config.stream = False
    return config


def _reply(text: str = "ответ", **kwargs) -> CallResult:
    usage = kwargs.pop("usage", Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    return CallResult(
        text=text,
        model_requested="ministral-14b-latest",
        usage=usage,
        finish_reason="stop",
        stream=False,
        **kwargs,
    )


def _complete(replies=None, seen=None, **result_kwargs):
    """Двойник chat_core.complete: отдаёт заготовленные ответы по очереди."""
    queue = list(replies or [])

    def complete(config, messages, capabilities=None):
        if seen is not None:
            seen.append(messages)
        text = queue.pop(0) if queue else "ответ"
        result = _reply(text, sent_messages=messages, **result_kwargs)
        result.model_requested = config.model
        return result

    return complete


def _shell(monkeypatch, tmp_path, complete=None, **params) -> cli.AgentShell:
    monkeypatch.setattr(cli.chat_core, "complete", complete or _complete())
    return cli.AgentShell(_config(**params), directory=tmp_path)


# --- контракт stdout/stderr -------------------------------------------------


def test_answer_goes_to_stdout_and_everything_else_to_stderr(monkeypatch, tmp_path, capsys):
    """Продукт агента — ответ модели. Панель токенов, footer и приглашение —
    служебное, и редирект `adventagent > answer.txt` обязан остаться честным."""
    shell = _shell(monkeypatch, tmp_path, complete=_complete(["Париж"]))

    cli._turn(shell, "столица Франции?")

    written = capsys.readouterr()
    assert _plain(written.out).strip() == "Париж"
    stderr = _flat(written.err)
    assert "токены" in stderr
    assert "Париж" not in stderr


def test_token_panel_and_footer_never_reach_stdout(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path, complete=_complete(["ок"]))

    cli._turn(shell, "вопрос")

    written = capsys.readouterr()
    assert "токены" not in _plain(written.out)
    assert "model ministral" not in _plain(written.out)


def test_sessions_list_goes_to_stderr(monkeypatch, tmp_path, capsys):
    """SPEC §14 называет список сессий отдельно: это сводка о памяти агента,
    а не продукт."""
    shell = _shell(monkeypatch, tmp_path)
    cli._turn(shell, "вопрос")
    capsys.readouterr()

    cli._dispatch("/sessions", shell)

    written = capsys.readouterr()
    assert "default" in _flat(written.err)
    assert "default" not in _flat(written.out)


# --- панель токенов: неизвестное не превращается в ноль ---------------------


def test_unknown_usage_prints_dash_never_zero(monkeypatch, tmp_path, capsys):
    """Самый частый класс бага во всей разведке (goose #8479, pydantic-ai
    #7808): отсутствующий usage превращается в честно выглядящий ноль."""
    shell = _shell(monkeypatch, tmp_path, complete=_complete(["ок"], usage=Usage()))

    cli._turn(shell, "вопрос")

    stderr = _flat(capsys.readouterr().err)
    assert "токены" in stderr, "панель токенов не напечаталась"
    assert "ход —/—" in stderr
    assert "сессия —" in stderr
    # Ноль не подставляется вместо неизвестного: проверяются те две цифры, а не
    # размер окна модели (в котором ноль стоит законно — 128000).
    turn_and_session = stderr.split("токены", 1)[1].split("контекст", 1)[0]
    assert "0" not in turn_and_session


def test_unknown_usage_is_not_written_to_the_session_as_zero(monkeypatch, tmp_path):
    shell = _shell(monkeypatch, tmp_path, complete=_complete(["ок"], usage=Usage()))

    cli._turn(shell, "вопрос")

    saved = Session.load("default", directory=tmp_path)
    assistant = [turn for turn in saved.turns if turn.role == "assistant"]
    assert assistant and assistant[0].usage is None
    assert saved.token_total() is None
    assert saved.missing_usage() == 1


def test_context_share_is_marked_as_an_estimate_and_counted_against_the_window(
    monkeypatch, tmp_path, capsys
):
    """Оценка обязана быть помечена `~` везде, где показывается, а окно берётся
    из карточки модели (max_context_length), а не выдумывается."""
    shell = _shell(monkeypatch, tmp_path)

    cli._turn(shell, "вопрос")

    stderr = _flat(capsys.readouterr().err)
    assert "контекст ~" in stderr
    assert "/128000" in stderr


def test_panel_counts_the_exchange_that_just_happened(monkeypatch, tmp_path, capsys):
    """Панель обещает СЛЕДУЮЩИЙ запрос и сумму сессии, поэтому печатается после
    того, как ход попал и в рабочий контекст, и в файл сессии. Пока она стояла
    внутри _ask(), оба числа отставали на ход: контекст показывал «~1», как
    будто разговора ещё нет, а сумма сессии — предыдущую."""
    shell = _shell(monkeypatch, tmp_path, complete=_complete(["ответ на три десятка символов"]))

    cli._turn(shell, "вопрос подлиннее одного слова")

    stderr = _flat(capsys.readouterr().err)
    assert "сессия 15" in stderr
    used = int(re.search(r"контекст ~(\d+)/", stderr).group(1))
    assert used > 1, "контекст посчитан по истории ДО хода"


def test_context_share_says_unknown_when_the_window_is_not_in_the_card(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "list_models", lambda config: [{"id": "ministral-14b-latest"}])
    shell = _shell(monkeypatch, tmp_path)

    cli._turn(shell, "вопрос")

    stderr = _flat(capsys.readouterr().err)
    assert "окно модели неизвестно" in stderr
    assert "%" not in stderr


def test_trimming_says_how_many_tokens_it_freed(monkeypatch, tmp_path, capsys):
    """SPEC §8: выброшенное называется вслух и в ходах, и в токенах. «Выброшено
    6 сообщений» не отличает освобождённые 200 токенов от 20 000, а ради этой
    цифры день и перешёл с символьного порога недели 01 на токенный."""
    shell = _shell(monkeypatch, tmp_path)
    shell.agent.context_limit = 2_000  # окно, в которое история заведомо не влезет
    shell.history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "с" * 500} for i in range(10)
    ]

    cli._turn(shell, "вопрос")

    stderr = _flat(capsys.readouterr().err)
    assert "контекст обрезан" in stderr
    freed = re.search(r"освобождено токенов (\d+)", stderr)
    assert freed and int(freed.group(1)) > 0, "объём выброшенного не назван"


# --- /tokens ----------------------------------------------------------------


def _tokens_table(shell, capsys) -> str:
    capsys.readouterr()
    cli._dispatch("/tokens", shell)
    return _flat(capsys.readouterr().err)


def test_tokens_never_turns_partial_usage_into_a_zero(monkeypatch, tmp_path, capsys):
    """Локальный сервер (`--base-url`, SPEC §7.2) присылает {"total_tokens": 51}
    без prompt/completion. `usage.is_empty()` для такого usage — False, и
    сложение через `or 0` печатало «prompt/completion 0/0», то есть выдавало
    неизвестное за точный ноль в той самой команде, ради которой заведён день."""
    shell = _shell(monkeypatch, tmp_path, complete=_complete(["ок"], usage=Usage(total_tokens=51)))
    cli._turn(shell, "вопрос")

    table = _tokens_table(shell, capsys)

    assert "prompt/completion —/—" in table
    assert "0/0" not in table
    # Сумма при этом известна: total сервер прислал.
    assert "51" in table


def test_tokens_marks_how_many_turns_are_unknown_in_the_breakdown(monkeypatch, tmp_path, capsys):
    """Частично известная сумма не должна выглядеть полной — как и «(без
    usage: N)» у суммы за сессию."""
    shell = _shell(
        monkeypatch,
        tmp_path,
        complete=_complete(["раз", "два"], usage=Usage(prompt_tokens=10, completion_tokens=5)),
    )
    cli._turn(shell, "первый")
    # Второй ход приходит с одним лишь total — прежний код молча прибавил бы 0.
    shell.agent._complete = _complete(["два"], usage=Usage(total_tokens=99))
    cli._turn(shell, "второй")

    table = _tokens_table(shell, capsys)

    assert "10 (+1 неизвестно)/5 (+1 неизвестно)" in table


def test_tokens_does_not_pass_an_estimate_off_as_an_exact_count(monkeypatch, tmp_path, capsys):
    """Вердикт «сошлось/разошлось» относится к ТОЧНОМУ счёту: расхождение у него
    означает «таблица токенизаторов разъехалась с моделью» (SPEC §7.1). У оценки
    расхождение — норма и повод откалиброваться, ровно поэтому молчит и
    TokenCheck.warning(); `/tokens` обязана применять то же правило."""
    shell = _shell(monkeypatch, tmp_path)
    cli._turn(shell, "вопрос")

    table = _tokens_table(shell, capsys)

    assert "сошлось" not in table and "разошлось" not in table
    assert "оценка ~" in table
    assert "калибруется" in table
    # И строка про сам счётчик тоже помечена как оценка.
    assert "(~)" in table


def test_tokens_keeps_the_verdict_for_an_exact_counter(monkeypatch, tmp_path, capsys):
    """Обратная сторона: у точного счётчика вердикт остаётся — иначе пропал бы
    единственный сигнал о разъехавшейся таблице токенизаторов."""

    class _Exact:
        """Двойник точного счётчика: символ за токен, ни сети, ни токенизатора."""

        exact = True
        name = "точный"

        def count(self, messages):
            return sum(len(m["content"]) for m in messages)

        def calibrate(self, messages, prompt_tokens):
            return None

    monkeypatch.setattr(cli, "counter_for", lambda model, **kwargs: (_Exact(), None))
    shell = _shell(monkeypatch, tmp_path)
    cli._turn(shell, "вопрос")

    table = _tokens_table(shell, capsys)

    assert "сошлось" in table or "разошлось" in table


def test_tokens_says_there_was_nothing_to_reconcile_yet(monkeypatch, tmp_path, capsys):
    """«Сверять не с чем» — отдельный исход, а не «сошлось»."""
    shell = _shell(monkeypatch, tmp_path)

    table = _tokens_table(shell, capsys)

    assert "ходов ещё не было" in table
    assert "сошлось" not in table


def test_tokens_goes_to_stderr(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path)
    cli._turn(shell, "вопрос")
    capsys.readouterr()

    cli._dispatch("/tokens", shell)

    written = capsys.readouterr()
    assert "окно модели" in _flat(written.err)
    assert "окно модели" not in _flat(written.out)


# --- служебные таблицы не идут в stdout -------------------------------------


@pytest.mark.parametrize(
    ("line", "needle"),
    [
        ("/help", "/sessions"),
        ("/params", "temperature"),
        ("/model list", "ministral-3b-2512"),
        ("/model info", "реальная версия"),
    ],
)
def test_service_tables_never_reach_stdout(monkeypatch, tmp_path, capsys, line, needle):
    """SPEC §14: продукт агента — ответ модели. Всё остальное в stdout ломает
    `adventagent -s demo > answer.txt` — в файле оказывается таблица команд, а
    ответ модели где-то под ней. В неделе 01 те же таблицы идут в stdout
    законно: там `advent w01 models` печатает таблицу как продукт команды."""
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch(line, shell)

    written = capsys.readouterr()
    assert needle in _flat(written.err), f"{line}: вывод пропал совсем"
    assert _plain(written.out).strip() == "", f"{line}: служебная таблица ушла в stdout"


def test_params_shows_only_what_the_agent_reads(monkeypatch, tmp_path, capsys):
    """`/params` у агента не показывает problem/runs/judge/temps/models: агент
    их не читает, а выставленный параметр, ни на что не влияющий, выглядит как
    поломка (SPEC §16)."""
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/params", shell)

    table = _flat(capsys.readouterr().err)
    for name in ("problem", "runs", "judge", "temps", "models", "strategy"):
        assert name not in table, f"{name} не читается агентом, но показан в /params"
    for name in ("temperature", "session", "mode", "done", "max_turns"):
        assert name in table


# --- журнал -----------------------------------------------------------------


def test_journal_is_written_with_week_2_and_day_7(monkeypatch, tmp_path):
    """Литералы 2 и 7, а не cli.WEEK/cli.DAY: ожидание, взятое из того же
    источника, что и код под тестом, покраснеть не может (CLAUDE.md)."""
    logged: dict = {}
    monkeypatch.setattr(cli, "log_call", lambda result, messages, **kwargs: logged.update(kwargs))
    shell = _shell(monkeypatch, tmp_path)

    cli._turn(shell, "вопрос")

    assert logged["week"] == 2
    assert logged["day"] == 7


def test_journal_logs_what_actually_went_to_the_api(monkeypatch, tmp_path):
    logged: dict = {}
    monkeypatch.setattr(
        cli, "log_call", lambda result, messages, **kwargs: logged.update(messages=messages)
    )
    shell = _shell(monkeypatch, tmp_path)

    cli._turn(shell, "вопрос")

    assert logged["messages"][-1]["content"] == "вопрос"


# --- реестр слэш-команд -----------------------------------------------------


def test_slash_commands_live_in_a_registry_not_an_if_chain():
    """SPEC §10: набор фиксирован, и каждая команда — запись реестра с
    обработчиком, а не ветка цепочки."""
    expected = {
        "/help",
        "/model",
        "/params",
        "/set",
        "/again",
        "/reset",
        "/new",
        "/sessions",
        "/tokens",
        "/mode",
        "/exit",
    }

    assert expected <= set(cli.COMMANDS)
    for name in expected:
        assert callable(cli.COMMANDS[name].handler)
    # Алиас ведёт на ту же запись, а не на копию.
    assert cli.COMMANDS["/quit"] is cli.COMMANDS["/exit"]
    # /help показывает каждую команду один раз, а не по разу на алиас.
    names = [entry.name for entry in cli.unique_commands()]
    assert len(names) == len(set(names))


def test_unknown_command_warns_and_keeps_the_session(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path)

    assert cli._dispatch("/такой-команды-нет", shell) is False

    assert "неизвестная команда" in _flat(capsys.readouterr().err)


def test_exit_command_ends_the_loop(monkeypatch, tmp_path):
    shell = _shell(monkeypatch, tmp_path)

    assert cli._dispatch("/exit", shell) is True
    assert cli._dispatch("/quit", shell) is True


def test_set_refuses_parameters_the_agent_does_not_read(monkeypatch, tmp_path, capsys):
    """`/params` у агента не показывает problem/runs/judge/temps/models, и
    выставить их тоже нельзя: параметр, ни на что не влияющий, выглядит как
    поломка (SPEC §16)."""
    shell = _shell(monkeypatch, tmp_path)

    cli._dispatch("/set strategy panel", shell)

    assert "не читает параметр" in _flat(capsys.readouterr().err)
    assert shell.config.params.strategy is None


def test_set_rejects_a_session_name_that_escapes_the_folder(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path)

    cli._dispatch("/set session ../../.env", shell)

    assert "Недопустимое имя сессии" in _flat(capsys.readouterr().err)
    assert shell.session.name == "default"


# --- умолчания приходят из реестра ------------------------------------------


def test_defaults_come_from_the_registry_not_from_the_typer_signature():
    """Литерал рядом с флагом становится вторым источником истины и однажды
    разъезжается с настоящим умолчанием (CLAUDE.md про потолок 2.0)."""
    assert defaults_for(AGENT_COMMAND) == {
        "done": "text:ГОТОВО",
        "mode": "chat",
        "max_turns": 10,
        "session": "default",
    }

    signature = inspect.signature(cli.agent)
    for name in ("session", "mode", "done", "max_turns"):
        assert signature.parameters[name].default.default is None, (
            f"{name}: умолчание записано литералом в сигнатуре typer"
        )


# --- смена модели внутри сессии ---------------------------------------------


def test_model_change_inside_a_session_warns_instead_of_silently_swapping(
    monkeypatch, tmp_path, capsys
):
    """Прямой пробел разведки: у simonw/llm модель при продолжении сессии
    подменяется молча (issue #1140 открыт годами)."""
    session = Session.new("default", directory=tmp_path)
    session.add_turn("assistant", "прошлый ответ", model="ministral-3b-latest")
    session.save()

    _shell(monkeypatch, tmp_path)

    stderr = _flat(capsys.readouterr().err)
    assert "ministral-3b-latest" in stderr
    assert "счёт токенов теперь другие" in stderr


def test_model_switch_retargets_the_context_window(monkeypatch, tmp_path):
    """У другой модели другое окно: оставить прежнее значило бы показывать
    заполненность, посчитанную не для той модели (goose #6185)."""
    shell = _shell(monkeypatch, tmp_path)
    assert shell.agent.context_limit == 128_000

    cli._dispatch("/model ministral-3b-latest", shell)

    assert shell.config.model == "ministral-3b-latest"
    assert shell.agent.context_limit == 32_000


# --- прерывание, ошибки -----------------------------------------------------


def test_interrupted_answer_is_stored_with_the_interrupt_note(monkeypatch, tmp_path):
    """Пометка об обрыве обязана доехать до файла сессии: иначе следующий
    запуск подсунет модели её же оборванный ответ как законченный."""
    shell = _shell(monkeypatch, tmp_path, complete=_complete(["полов"], truncated=True))

    cli._turn(shell, "вопрос")

    saved = Session.load("default", directory=tmp_path)
    assert INTERRUPT_NOTE in saved.turns[-1].content


def test_api_error_prints_and_does_not_kill_the_session(monkeypatch, tmp_path, capsys):
    def boom(config, messages, capabilities=None):
        raise AdventError("модель недоступна")

    shell = _shell(monkeypatch, tmp_path, complete=boom)

    cli._turn(shell, "вопрос")  # не должно бросить

    assert "модель недоступна" in _flat(capsys.readouterr().err)
    # Провалившийся ход не попадает ни в память, ни в рабочий контекст.
    assert shell.session.turns == []
    assert shell.history == []


def test_session_write_failure_warns_instead_of_crashing(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path)

    def fail(self) -> None:
        raise OSError("диск только для чтения")

    # Session — dataclass(slots=True): атрибут инстанса не подменить, поэтому
    # подменяется метод класса.
    monkeypatch.setattr(cli.Session, "save", fail)

    cli._turn(shell, "вопрос")

    assert "сессия не сохранена" in _flat(capsys.readouterr().err)


# --- ввод -------------------------------------------------------------------


def test_triple_quote_sentinel_collects_a_multiline_question(monkeypatch):
    """Sentinel, а не Shift+Enter: портируемого способа поймать Shift+Enter в
    терминалах нет (документация aider, SPEC §13)."""
    lines = iter(['"""', "первая строка", "", "вторая строка", '"""'])
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(lines))

    assert cli._read_input() == "первая строка\n\nвторая строка"


def test_end_of_input_is_distinct_from_an_empty_line(monkeypatch):
    """None означает «ввод закончился», пустая строка — «ничего не набрали»:
    склеить их значило бы выходить из разговора по случайному Enter."""

    def eof(*args, **kwargs):
        raise EOFError

    monkeypatch.setattr(cli.typer, "prompt", eof)
    assert cli._read_input() is None

    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "   ")
    assert cli._read_input() == ""


def test_input_is_echoed_when_stdin_is_not_a_tty(monkeypatch, capsys):
    """При записи демо stdin — это pipe, и без эха в кадре видно приглашение
    без вопроса."""
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "вопрос из пайпа")

    cli._read_input()

    assert "вопрос из пайпа" in _flat(capsys.readouterr().err)


# --- режимы и память --------------------------------------------------------


def test_dialog_runs_in_the_shared_session_memory(monkeypatch, tmp_path):
    """SPEC §9: у агента одна память, а не две параллельные — целевой диалог
    видит всё, что обсуждали до него, и остаётся в сессии после."""
    seen: list[list[dict]] = []
    shell = _shell(monkeypatch, tmp_path, complete=_complete(["первый", "второй"], seen=seen))

    cli._turn(shell, "запомни число 7")
    cli._dispatch("/mode dialog", shell)
    cli._turn(shell, "хочу приготовить салат")

    assert shell.agent.mode == "dialog"
    # Второй запрос несёт предыдущий обмен: история общая.
    assert [m["content"] for m in seen[1] if m["role"] != "system"][:3] == [
        "запомни число 7",
        "первый",
        "хочу приготовить салат",
    ]
    assert len(shell.session.turns) == 4


def test_dialog_stops_at_the_turn_limit_without_killing_the_session(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path, mode="dialog", done="text:ГОТОВО", max_turns=2)

    cli._turn(shell, "хочу салат")
    cli._turn(shell, "греческий")

    assert "потолок ходов (2)" in _flat(capsys.readouterr().err)
    assert shell.dialog_turns == 0


def test_dialog_marker_ends_the_dialog(monkeypatch, tmp_path, capsys):
    shell = _shell(
        monkeypatch,
        tmp_path,
        complete=_complete(["салат готов, ГОТОВО"]),
        mode="dialog",
        done="text:ГОТОВО",
    )

    cli._turn(shell, "хочу салат")

    assert "условие завершения выполнено, ходов: 1" in _flat(capsys.readouterr().err)


def test_dialog_marker_returns_the_mode_to_chat(monkeypatch, tmp_path, capsys):
    """Достигнутая цель заканчивает эпизод — режим возвращается в chat.

    Пойман живым dry-run 2026-09-07: после выданного рецепта агент оставался
    в режиме dialog и трактовал следующую реплику как НОВУЮ цель, начиная
    допрос заново. Это ещё и регрессия относительно недели 01, где
    `_run_dialog()` после маркера возвращал управление в обычный REPL.

    Переключение обязано быть названным вслух: молча менять настройку
    пользователя нельзя, поэтому проверяется и режим, и сообщение.
    """
    shell = _shell(
        monkeypatch,
        tmp_path,
        complete=_complete(["салат готов, ГОТОВО", "уточняющий вопрос?"]),
        mode="dialog",
        done="text:ГОТОВО",
    )

    cli._turn(shell, "хочу салат")

    assert shell.config.params.mode == "chat"
    assert "режим вернулся в chat" in _flat(capsys.readouterr().err)

    # Следующая реплика идёт обычным ходом, а не новым допросом: счётчик
    # ходов диалога остаётся нулевым.
    cli._turn(shell, "спасибо")
    assert shell.dialog_turns == 0


def test_done_conflicting_with_stop_is_reported_at_startup(monkeypatch, tmp_path, capsys):
    """Ловушка из CLAUDE.md: API вырезает stop из ответа, и совпадающий с ним
    маркер завершения не доедет до нас никогда — диалог не закончится молча."""
    _shell(monkeypatch, tmp_path, mode="dialog", done="text:ГОТОВО", stop="ГОТОВО")

    assert "пересекается с маркером завершения" in _flat(capsys.readouterr().err)


def test_reset_clears_the_working_context_but_keeps_the_session_file(monkeypatch, tmp_path):
    shell = _shell(monkeypatch, tmp_path)
    cli._turn(shell, "вопрос")

    cli._dispatch("/reset", shell)

    assert shell.history == []
    assert Session.load("default", directory=tmp_path).turns  # файл на месте


def test_new_starts_an_empty_session_and_keeps_topics_apart(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path)
    cli._turn(shell, "про работу")

    cli._dispatch("/new recipes", shell)
    cli._turn(shell, "про салат")

    assert shell.session.name == "recipes"
    assert len(shell.session.turns) == 2
    assert len(Session.load("default", directory=tmp_path).turns) == 2

    capsys.readouterr()
    cli._dispatch("/sessions", shell)
    listed = _flat(capsys.readouterr().err)
    assert "recipes" in listed
    assert "default" in listed


def test_sessions_shows_a_broken_session_and_does_not_lose_it(monkeypatch, tmp_path, capsys):
    """`/sessions` — команда «покажи, что у меня есть», а не «переложи файлы».

    Пока карантин делался и при листинге, первый вызов уносил битую сессию в
    .bak, а второй не показывал ни строки, ни предупреждения — пользователь
    решал, что разговора не было вовсе.
    """
    (tmp_path / "битая.json").write_text("{это не json", encoding="utf-8")
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/sessions", shell)
    first = _flat(capsys.readouterr().err)
    cli._dispatch("/sessions", shell)
    second = _flat(capsys.readouterr().err)

    assert (tmp_path / "битая.json").is_file(), "листинг переименовал файл сессии"
    for listing in (first, second):
        assert "битая" in listing
        # «Ходов 0» — утверждение о разговоре, которого мы не видели.
        assert "нечитаема" in listing


def test_persona_failure_does_not_quietly_bring_back_the_project_persona(
    monkeypatch, tmp_path, capsys
):
    """None означает не «без персоны», а «взять персону проекта» — ту самую
    «лаконичный ассистент, без воды», которая гасит встречные вопросы агента.
    Предупреждение при этом обещало «работаем без неё»."""
    from advent_core.config import DEFAULT_SYSTEM_PROMPT

    config = _config()
    config.system_prompt_path = DEFAULT_SYSTEM_PROMPT
    monkeypatch.setattr(cli, "AGENT_PROMPT_PATH", tmp_path / "нет-такого-файла.md")

    persona = cli._persona(config)

    assert persona == "", "None вернуло бы в запрос персону проекта"
    agent = cli.Agent(config, complete=_complete(), stream=None, persona=persona)
    assert agent.system_prompt() is None
    assert "персона агента не прочиталась" in _flat(capsys.readouterr().err)


def test_again_repeats_without_history_and_without_touching_the_session(monkeypatch, tmp_path):
    """`/again` — независимый прогон того же вопроса, а не ещё один ход
    разговора: при повторе через историю модель видит собственный прошлый
    ответ и отвечает «как я говорил выше»."""
    seen: list[list[dict]] = []
    shell = _shell(monkeypatch, tmp_path, complete=_complete(seen=seen))
    cli._turn(shell, "вопрос")
    turns_before = len(shell.session.turns)

    cli._dispatch("/again", shell)

    assert [m["role"] for m in seen[1]] == ["user"]
    assert len(shell.session.turns) == turns_before


# --- персистентность состояния диалога (день 07) -----------------------------
#
# «Продолжать как будто агент не выключался» из задания дня обязано покрывать
# не только ходы, но и режим: выход посреди dialog раньше терял mode, done и
# счётчик, и перезапуск начинал всё заново.


def test_restart_keeps_the_history_and_resends_it(monkeypatch, tmp_path, capsys):
    """Основное требование дня: перезапуск на том же каталоге продолжает разговор.

    Проверяются обе половины: история второго shell'а равна истории первого, и
    ПЕРВЫЙ же запрос после перезапуска несёт прошлые user/assistant сообщения —
    иначе агент «помнит», но модели об этом не говорит.
    """
    seen: list[list[dict]] = []
    first = _shell(monkeypatch, tmp_path, complete=_complete(["первый ответ"], seen=seen))
    cli._turn(first, "вопрос")

    second = _shell(monkeypatch, tmp_path, complete=_complete(seen=seen))
    assert second.history == first.history

    cli._turn(second, "продолжаем")
    exchanged = [m for m in seen[1] if m["role"] != "system"]
    assert [m["role"] for m in exchanged] == ["user", "assistant", "user"]
    assert [m["content"] for m in exchanged] == ["вопрос", "первый ответ", "продолжаем"]


def test_dialog_state_survives_a_restart(monkeypatch, tmp_path, capsys):
    """Выход посреди dialog и запуск заново: режим, счётчик и маркер на месте.

    Раньше всё это жило только в памяти запуска, и «как будто не выключался»
    нарушалось ровно на середине целевого диалога. Подхват анонсируется —
    без анонса в кадре не видно, что диалог продолжен, а не начат.
    """
    first = _shell(
        monkeypatch,
        tmp_path,
        complete=_complete(["уточняющий вопрос?"]),
        mode="dialog",
        done="text:ГОТОВО",
    )
    cli._turn(first, "хочу салат")
    assert first.dialog_turns == 1
    capsys.readouterr()

    second = _shell(monkeypatch, tmp_path, complete=_complete(["ответ"]))

    assert second.agent.mode == "dialog"
    assert second.dialog_turns == 1
    assert second.config.params.done == "text:ГОТОВО"
    assert "подхвачен целевой диалог: ход 1 из" in _flat(capsys.readouterr().err)


def test_explicit_flag_beats_the_mode_from_the_file(monkeypatch, tmp_path, capsys):
    """Приоритет «флаг > файл > дефолт»: явный --mode chat при старте не даёт
    файлу сессии вернуть dialog — явный выбор пользователя не наш, чтобы его
    игнорировать (то же правило, что у --system против персоны проекта)."""
    first = _shell(monkeypatch, tmp_path, complete=_complete(["уточняющий?"]), mode="dialog")
    cli._turn(first, "цель")
    assert Session.load("default", directory=tmp_path).state["mode"] == "dialog"
    capsys.readouterr()

    monkeypatch.setattr(cli.chat_core, "complete", _complete())
    second = cli.AgentShell(_config(mode="chat"), directory=tmp_path, explicit=frozenset({"mode"}))

    assert second.agent.mode == "chat"
    assert second.config.params.mode == "chat"
    assert "подхвачен целевой диалог" not in _flat(capsys.readouterr().err)


def test_done_flip_is_persisted_to_the_session_file(monkeypatch, tmp_path):
    """Завершённый эпизод не воскресает: save в _dialog_progress стоит ПОСЛЕ
    flip'а mode→chat, иначе файл при выходе сразу после done хранил бы
    протухший dialog, и перезапуск начинал бы допрос заново."""
    shell = _shell(
        monkeypatch,
        tmp_path,
        complete=_complete(["салат готов, ГОТОВО"]),
        mode="dialog",
        done="text:ГОТОВО",
    )

    cli._turn(shell, "хочу салат")
    assert shell.config.params.mode == "chat"

    saved = Session.load("default", directory=tmp_path)
    assert saved.state["mode"] == "chat"
    assert saved.state["dialog_turns"] == 0


def test_mode_change_is_saved_even_without_a_turn(monkeypatch, tmp_path, capsys):
    """`/mode dialog` → `/exit` без единого хода: режим обязан уже быть в файле.

    Save после хода здесь не случится никогда — единственная точка записи это
    сама смена режима."""
    shell = _shell(monkeypatch, tmp_path)

    cli._dispatch("/mode dialog", shell)
    assert "mode = dialog" in _flat(capsys.readouterr().err)

    saved = Session.load("default", directory=tmp_path)
    assert saved.state["mode"] == "dialog"
    assert saved.state["dialog_turns"] == 0


def test_reset_zeroes_the_dialog_counter_on_disk_too(monkeypatch, tmp_path, capsys):
    """`/reset` обнуляет счётчик и в памяти, и в слепке: иначе перезапуск
    воскресил бы ходы диалога из протухшего state."""
    shell = _shell(monkeypatch, tmp_path, complete=_complete(["уточняющий?"]), mode="dialog")
    cli._turn(shell, "хочу салат")
    assert Session.load("default", directory=tmp_path).state["dialog_turns"] == 1
    capsys.readouterr()

    cli._dispatch("/reset", shell)

    assert shell.dialog_turns == 0
    assert Session.load("default", directory=tmp_path).state["dialog_turns"] == 0


def test_state_that_is_not_a_dict_warns_and_is_ignored(monkeypatch, tmp_path, capsys):
    """Файлу не доверяем вслепую: state — dict или его нет. Чужая запись —
    предупреждение и дефолты, а не падение и не молчаливая подмена."""
    session = Session.new("default", directory=tmp_path)
    session.state = "dialog"  # намеренно не dict — чужая или битая запись
    session.save()

    shell = _shell(monkeypatch, tmp_path)

    assert shell.agent.mode == "chat"
    stderr = _flat(capsys.readouterr().err)
    assert "state не похож на объект" in stderr


def test_unknown_mode_in_the_file_warns_and_falls_back_to_default(monkeypatch, tmp_path, capsys):
    """Невалидное значение из файла предупреждает и пропускается: mode
    остаётся дефолтом реестра, а сессия грузится дальше — это разговор, а не
    конфиг, из-за одной опечатки его не выбрасываем."""
    session = Session.new("default", directory=tmp_path)
    session.state = {"mode": "неттакого"}
    session.save()

    shell = _shell(monkeypatch, tmp_path)

    assert shell.agent.mode == "chat"
    stderr = _flat(capsys.readouterr().err)
    assert "негодное mode='неттакого'" in stderr
    assert "default" in stderr, "сессия загрузилась, несмотря на битый state"


# --- пикеры голых /model и /set (день 07) ------------------------------------
#
# Только при isatty: запись демо гоняет REPL через pipe, и промпт пикера съел
# бы следующую строку сценария как «ответ». Поэтому у каждого теста есть и
# не-tty двойник, ловящий регрессию «пикер звался без tty».


def test_bare_model_command_picks_interactively_and_retargets(monkeypatch, tmp_path, capsys):
    """Голая /model на tty: выбор из списка переключает модель вместе со всем,
    что от неё зависит (карточка, окно контекста) — тот же путь, что у
    `/model <имя>`, а не параллельная реализация."""
    _patch_tty(monkeypatch)
    asked_current: list[str] = []

    def fake_choose_model(models, *, current, base_url):
        asked_current.append(current)
        return "ministral-3b-latest"

    monkeypatch.setattr(console, "choose_model", fake_choose_model)
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/model", shell)

    assert shell.config.model == "ministral-3b-latest"
    assert shell.card and shell.card["id"] == "ministral-3b-2512"
    assert shell.agent.context_limit == 32_000
    assert asked_current == ["ministral-14b-latest"], "пикеру показана текущая модель"
    assert "модель переключена на ministral-3b-latest" in _flat(capsys.readouterr().err)


def test_bare_model_command_refusal_keeps_the_current_model(monkeypatch, tmp_path, capsys):
    """Enter на приглашении пикера — «оставить как есть»: модель и её окно не
    тронуты."""
    _patch_tty(monkeypatch)
    monkeypatch.setattr(console, "choose_model", lambda models, **kwargs: None)
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/model", shell)

    assert shell.config.model == "ministral-14b-latest"
    assert shell.agent.context_limit == 128_000
    assert "не изменена" in _flat(capsys.readouterr().err)


def test_bare_model_command_without_a_tty_prints_and_never_prompts(monkeypatch, tmp_path, capsys):
    """Пайп (запись демо, CI): прежнее поведение — печать текущей модели, и
    пикер не звался вовсе, иначе промпт съел бы строку сценария."""

    def boom(*args, **kwargs):
        raise AssertionError("пикер зваться не должен без tty")

    monkeypatch.setattr(console, "choose_model", boom)
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/model", shell)

    stderr = _flat(capsys.readouterr().err)
    assert "текущая модель: ministral-14b-latest" in stderr


def test_bare_set_picks_param_then_value_through_two_menus(monkeypatch, tmp_path, capsys):
    """Голая /set на tty: два choose — параметр, затем значение из choices.
    Дальше тот же хвост, что у `/set mode dialog`: значение применилось и
    слепок state ушёл в файл."""
    _patch_tty(monkeypatch)
    menus: list[tuple[str, list[str]]] = []

    def fake_choose(prompt, options):
        menus.append((prompt, list(options)))
        return 9 if len(menus) == 1 else 1  # индекс mode, затем индекс dialog

    monkeypatch.setattr(console, "choose", fake_choose)
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/set", shell)

    assert shell.config.params.mode == "dialog"
    assert menus[0][0] == "параметр"
    assert any(option.startswith("mode =") for option in menus[0][1]), (
        "в меню параметра показано текущее значение"
    )
    assert menus[1][0] == "mode"
    assert menus[1][1] == ["chat", "dialog", "default"]
    assert Session.load("default", directory=tmp_path).state["mode"] == "dialog"
    assert "mode = dialog" in _flat(capsys.readouterr().err)


def test_bare_set_cancel_at_the_value_step_changes_nothing(monkeypatch, tmp_path, capsys):
    """Отмена на шаге значения (Enter) — выход без изменений: ни параметр, ни
    файл сессии не тронуты."""
    _patch_tty(monkeypatch)
    answers = iter([9, None])  # параметр выбрали, значение отменили
    monkeypatch.setattr(console, "choose", lambda prompt, options: next(answers))
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/set", shell)

    assert shell.config.params.mode == "chat"
    assert not (tmp_path / "default.json").exists(), "отмена записала файл сессии"
    assert "отменено" in _flat(capsys.readouterr().err)


def test_bare_set_without_a_tty_warns_as_before(monkeypatch, tmp_path, capsys):
    """Пайп: прежнее предупреждение про форму команды, пикер не звался."""

    def boom(prompt, options):
        raise AssertionError("пикер зваться не должен без tty")

    monkeypatch.setattr(console, "choose", boom)
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/set", shell)

    assert "нужно: /set <параметр> <значение>" in _flat(capsys.readouterr().err)
    assert shell.config.params.mode == "chat"


# --- старт с моделью, которой нет на сервере --------------------------------
#
# Живой пример: LM Studio на 127.0.0.1:1234 отдаёт две ornith-модели, а конфиг
# просит ministral. Раньше это было фатально С НЕВЕРНЫМ текстом («недоступна
# аккаунту» и совет без --base-url показал бы облачный список). Теперь: не-tty
# — ConfigError с честным текстом; tty — интерактивный выбор на старте.

LOCAL_MODELS = [
    {
        "id": "ornith-1.5-35b-a3b",
        "capabilities": {"completion_chat": True},
        "max_context_length": 40_000,
    },
    {
        "id": "ornith-1.5-9b",
        "capabilities": {"completion_chat": True},
        "max_context_length": 32_000,
    },
]


def _patch_tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)


def test_choose_model_accepts_a_number(monkeypatch, capsys):
    _patch_tty(monkeypatch)
    monkeypatch.setattr(console.typer, "prompt", lambda *a, **k: "2")

    chosen = console.choose_model(LOCAL_MODELS, current="ministral-14b-latest", base_url=None)

    assert chosen == "ornith-1.5-9b"
    # Весь выбор — в stderr (контракт stdout/stderr): продукт команды не стартует.
    # Rich подсвечивает цифры в именах моделей — сравнивается текст без ANSI.
    written = capsys.readouterr()
    assert written.out == ""
    assert "ornith-1.5-9b" in _flat(written.err)
    assert "1. ornith-1.5-35b-a3b" in _flat(written.err)


def test_choose_model_accepts_an_exact_name(monkeypatch):
    _patch_tty(monkeypatch)
    monkeypatch.setattr(console.typer, "prompt", lambda *a, **k: "ornith-1.5-35b-a3b")

    chosen = console.choose_model(LOCAL_MODELS, current="x", base_url=None)

    assert chosen == "ornith-1.5-35b-a3b"


def test_choose_model_accepts_an_alias_and_returns_the_canonical_id(monkeypatch):
    """Конфиг живёт алиасами (-latest): принять их и вернуть канонический id,
    чтобы find_model()/capabilities дальше по __post_init__ сошлись."""
    nine_b = dict(LOCAL_MODELS[1], aliases=["ornith-9b-latest"])
    models = [LOCAL_MODELS[0], nine_b]
    _patch_tty(monkeypatch)
    monkeypatch.setattr(console.typer, "prompt", lambda *a, **k: "ornith-9b-latest")

    chosen = console.choose_model(models, current="x", base_url=None)

    assert chosen == "ornith-1.5-9b"


def test_choose_model_repeats_the_question_on_garbage_input(monkeypatch, capsys):
    """Неверный ввод не выходит: вопрос повторяется, пока не придёт вариант."""
    answers = iter(["99", "нет-такой-модели", "1"])
    _patch_tty(monkeypatch)
    monkeypatch.setattr(console.typer, "prompt", lambda *a, **k: next(answers))

    chosen = console.choose_model(LOCAL_MODELS, current="x", base_url=None)

    assert chosen == "ornith-1.5-35b-a3b"
    assert "нет варианта" in _flat(capsys.readouterr().err)


def test_choose_model_empty_input_is_a_refusal(monkeypatch):
    """Enter на приглашении — отказ, а не первая модель из списка."""
    _patch_tty(monkeypatch)
    monkeypatch.setattr(console.typer, "prompt", lambda *a, **k: "")

    assert console.choose_model(LOCAL_MODELS, current="x", base_url=None) is None


def test_choose_model_eof_is_a_refusal(monkeypatch):
    def eof(*args, **kwargs):
        raise EOFError

    _patch_tty(monkeypatch)
    monkeypatch.setattr(console.typer, "prompt", eof)

    assert console.choose_model(LOCAL_MODELS, current="x", base_url=None) is None


def test_choose_model_with_no_candidates_is_a_refusal_without_asking(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("спрашивать не о чем — prompt зваться не должен")

    monkeypatch.setattr(console.typer, "prompt", boom)

    assert console.choose_model([], current="x", base_url=None) is None


def test_start_with_an_unknown_model_and_no_tty_raises_with_the_server_text(monkeypatch, tmp_path):
    """Не-tty (пайп, запись демо, CI): падаем по-прежнему, но текст обязан
    называть СЕРВЕР и доступные модели — «аккаунту» тут ни при чём, а совет
    `advent w01 models` без --base-url показал бы облачный список."""
    monkeypatch.setattr(cli, "list_models", lambda config: LOCAL_MODELS)
    monkeypatch.setattr(cli.chat_core, "complete", _complete())
    config = _config()
    config.model = "ministral-14b-latest"  # в списке сервера такого нет
    config.base_url = "http://127.0.0.1:1234"

    with pytest.raises(ConfigError) as excinfo:
        cli.AgentShell(config, directory=tmp_path)

    message = str(excinfo.value)
    assert "не найдена на сервере http://127.0.0.1:1234" in message
    assert "ornith-1.5-35b-a3b" in message
    assert "ornith-1.5-9b" in message
    assert "аккаунту" not in message
    assert "advent w01 models" not in message


def test_start_with_an_unknown_model_cloud_text_still_names_the_account(monkeypatch, tmp_path):
    """Обратная сторона: облачный случай по-прежнему про аккаунт и про
    `advent w01 models` — там этот совет верен."""
    monkeypatch.setattr(cli.chat_core, "complete", _complete())
    config = _config()
    config.model = "magistral-medium-latest"  # нет в списке фикстуры

    with pytest.raises(ConfigError) as excinfo:
        cli.AgentShell(config, directory=tmp_path)

    message = str(excinfo.value)
    assert "недоступна аккаунту" in message
    assert "advent w01 models" in message
    assert "ministral-3b-2512" in message, "доступные модели перечислены"


def test_start_offers_a_choice_and_retargets_everything(monkeypatch, tmp_path, capsys):
    """tty: агент спрашивает, и выбранное имя пересчитывает ВСЁ, что от него
    зависит — card, окно контекста, capabilities и счётчик токенов. Счётчик
    создаётся ниже refresh() по __post_init__, поэтому достаточно, чтобы
    config.model был новым к моменту его вызова; тест ловит именно это:
    counter_for обязан получить ВЫБРАННУЮ модель, а не ту, что упала."""
    _patch_tty(monkeypatch)
    monkeypatch.setattr(cli, "list_models", lambda config: LOCAL_MODELS)
    counter_models: list[str] = []

    def fake_counter_for(model, **kwargs):
        counter_models.append(model)
        return EstimateCounter(), None

    monkeypatch.setattr(cli, "counter_for", fake_counter_for)
    answers = iter(["2"])  # ornith-1.5-9b
    monkeypatch.setattr(console.typer, "prompt", lambda *a, **k: next(answers))
    monkeypatch.setattr(cli.chat_core, "complete", _complete())

    config = _config()
    config.model = "ministral-14b-latest"
    config.base_url = "http://127.0.0.1:1234"
    shell = cli.AgentShell(config, directory=tmp_path)

    assert config.model == "ornith-1.5-9b"
    assert shell.card and shell.card["id"] == "ornith-1.5-9b"
    assert shell.capabilities == {"completion_chat": True}
    assert shell.agent.context_limit == 32_000
    assert counter_models == ["ornith-1.5-9b"], "счётчик создан для выбранной модели"
    assert "модель выбрана: ornith-1.5-9b" in _flat(capsys.readouterr().err)


def test_start_offers_models_when_the_server_has_no_capabilities(monkeypatch, tmp_path):
    """LM Studio отдаёт карточки без `capabilities` вообще: chat_models()
    фильтрует всё, и старый вариант предложил бы пусто — то есть по факту не
    предложил бы ничего. Fallback: предлагаем всё, что сервер отдал."""
    _patch_tty(monkeypatch)
    bare = [{"id": "ornith-1.5-9b"}, {"id": "text-embedding-nomic-embed-text-v1.5"}]
    monkeypatch.setattr(cli, "list_models", lambda config: bare)
    monkeypatch.setattr(cli.chat_core, "complete", _complete())
    answers = iter(["1"])
    monkeypatch.setattr(console.typer, "prompt", lambda *a, **k: next(answers))

    config = _config()
    config.model = "ministral-14b-latest"
    config.base_url = "http://127.0.0.1:1234"
    shell = cli.AgentShell(config, directory=tmp_path)

    assert config.model == "ornith-1.5-9b"
    assert shell.card == {"id": "ornith-1.5-9b"}


def test_start_refusal_does_not_continue_the_startup(monkeypatch, tmp_path):
    """Enter на приглашении — отказ: запуск не продолжается, ConfigError."""
    _patch_tty(monkeypatch)
    monkeypatch.setattr(cli, "list_models", lambda config: LOCAL_MODELS)
    monkeypatch.setattr(console.typer, "prompt", lambda *a, **k: "")
    config = _config()
    config.model = "ministral-14b-latest"

    with pytest.raises(ConfigError):
        cli.AgentShell(config, directory=tmp_path)


def test_model_command_with_a_typo_rolls_back_and_never_prompts(monkeypatch, tmp_path, capsys):
    """`/model <опечатка>` внутри сессии — НЕ место для интерактивного выбора:
    откат к прежней модели и warning, даже на tty. Спрашивать можно только на
    старте; регрессия ловится дублем prompt'а, который обязан не зваться."""
    _patch_tty(monkeypatch)

    def boom(*args, **kwargs):
        raise AssertionError("выбор предлагается только на старте, не посреди /model")

    monkeypatch.setattr(console.typer, "prompt", boom)
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/model нет-такой-модели", shell)

    assert shell.config.model == "ministral-14b-latest"
    assert shell.card and shell.card["id"] == "ministral-14b-2512"
    assert "недоступна аккаунту" in _flat(capsys.readouterr().err)
