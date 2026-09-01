"""`/again` и цикл mode=dialog в week_01/cli.py — сеть не трогается.

Session.refresh() обычно тянет живой список моделей — здесь week_01.cli.list_models
подменён фикстурой; chat_core.complete подменяется в каждом тесте отдельно, чтобы
проверить именно то, какие messages в него ушли и сколько раз он был вызван.
"""

from __future__ import annotations

import re

import pytest

import week_01.cli as cli
from advent_core import chat as chat_core
from advent_core import formats
from advent_core.config import Config
from advent_core.errors import ConfigurationError
from advent_core.params import GenerationParams

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Снимает ANSI-коды Rich-подсветки.

    Console.warn()/note() автоматически подсвечивают "key=value" и числа
    внутри сообщения отдельными escape-последовательностями — без снятия их
    substring-проверка вроде "mode=dialog" рвётся посреди строки на честном
    и неизменном тексте.
    """
    return _ANSI.sub("", text)


MODELS = [
    {
        "id": "mistral-small-2603",
        "aliases": ["mistral-small-latest"],
        "capabilities": {"completion_chat": True},
    }
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Session.refresh() не должна стучаться в сеть, а _run() — стримить."""
    monkeypatch.setattr(cli, "list_models", lambda config: MODELS)
    # should_stream() отдельно юнит-тестируется в test_chat.py — здесь важно
    # только не пойти по ветке chat_core.stream(), которую эти тесты не мокают.
    monkeypatch.setattr(cli.chat_core, "should_stream", lambda config: False)


def _config(**parm_kwargs) -> Config:
    return Config(
        api_key="k" * 32,
        model="mistral-small-latest",
        params=GenerationParams.build(**parm_kwargs),
    )


def _session(**parm_kwargs) -> cli.Session:
    return cli.Session(_config(**parm_kwargs))


# --- /again: понятная ошибка до первого вопроса и в mode=dialog ---


def test_again_before_first_question_warns_not_crashes(capsys):
    session = _session()
    assert session.last_question is None

    cli._handle_again(session)  # не должно бросить исключение

    assert "нечего повторять" in _plain(capsys.readouterr().err)


def test_again_in_dialog_mode_is_refused(capsys):
    session = _session(mode="dialog")
    session.last_question = "предыдущий вопрос"

    cli._handle_again(session)

    assert "mode=dialog" in _plain(capsys.readouterr().err)


def test_again_repeats_last_question_without_history(monkeypatch, capsys):
    """Суть /again: тот же вопрос уходит один раз, без истории — ровно messages=[user]."""
    session = _session()
    session.last_question = "вопрос"
    seen_messages = []

    def fake_complete(config, messages, capabilities=None):
        seen_messages.append(messages)
        return chat_core.CallResult(
            text="ответ", model_requested=config.model, finish_reason="stop"
        )

    monkeypatch.setattr(cli.chat_core, "complete", fake_complete)

    cli._handle_again(session)

    assert len(seen_messages) == 1
    assert [m["role"] for m in seen_messages[0]] == ["user"]
    assert seen_messages[0][0]["content"] == "вопрос"
    assert "ответ" in _plain(capsys.readouterr().out)


# --- max_turns: останавливает цикл диалога и возвращает управление ---


def test_max_turns_stops_dialog_and_returns_control(monkeypatch, capsys):
    session = _session(mode="dialog", done="json:done", max_turns=2)
    call_count = 0

    def fake_complete(config, messages, capabilities=None):
        nonlocal call_count
        call_count += 1
        return chat_core.CallResult(
            text='{"done": false, "question": "уточнение"}',
            model_requested=config.model,
            finish_reason="stop",
        )

    monkeypatch.setattr(cli.chat_core, "complete", fake_complete)
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "ответ пользователя")

    cli._run_dialog(session, "хочу приготовить салат")  # не должно зависнуть/упасть

    assert call_count == 2  # ровно max_turns — цикл не ушёл дальше потолка
    assert "потолок ходов" in _plain(capsys.readouterr().err)


def test_dialog_stops_on_done_condition_before_hitting_max_turns(monkeypatch, capsys):
    """Срабатывает по полю done=json:done раньше потолка — max_turns не при чём тут."""
    session = _session(mode="dialog", done="json:done", max_turns=10)
    responses = [
        '{"done": false, "question": "какой салат?"}',
        '{"done": true, "result": {"name": "греческий"}}',
    ]

    def fake_complete(config, messages, capabilities=None):
        return chat_core.CallResult(
            text=responses.pop(0), model_requested=config.model, finish_reason="stop"
        )

    monkeypatch.setattr(cli.chat_core, "complete", fake_complete)
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "греческий, с фетой")

    cli._run_dialog(session, "хочу приготовить салат")

    stderr = _plain(capsys.readouterr().err)
    assert "ходов: 2" in stderr
    assert "потолок ходов" not in stderr


# --- НАХОДКА 1: /params не должен врать в mode=dialog ---


def test_final_system_prompt_includes_format_instruction_in_dialog_mode():
    """Раньше dialog-ветка делала ранний return и никогда не доходила до
    formats.build_system() — /params показывал dialog-промпт БЕЗ инструкции
    формата, хотя chat._payload() безусловно дописывает её поверх любого
    system, dialog-пресет не исключение."""
    session = _session(mode="dialog", done="json:done", format="json")

    shown = cli._final_system_prompt(session)

    assert "JSON" in shown  # инструкция пресета формата дошла
    assert "done" in shown  # инструкция про признак завершения тоже на месте


def test_final_system_prompt_matches_what_payload_actually_sends():
    """Регрессия на рассинхронизацию: /params должен показать РОВНО то, что
    chat._payload() реально положит в system-сообщение запроса."""
    session = _session(mode="dialog", done="json:done", format="json")
    config = session.config

    shown = cli._final_system_prompt(session)

    kind, needle = formats.parse_done(config.params.done)
    dialog_system = cli._dialog_system_prompt(config, kind, needle)
    messages = chat_core.build_messages("вопрос", system=dialog_system)
    payload, *_ = chat_core._payload(config, messages)

    assert shown == payload["messages"][0]["content"]


# --- НАХОДКА 2: одношот с --mode dialog не должен молча игнорировать флаг ---


def test_guard_rejects_one_shot_question_with_dialog_mode():
    config = _config(mode="dialog")

    with pytest.raises(ConfigurationError) as excinfo:
        cli._guard_one_shot_dialog("Хочу приготовить салат", config)

    assert excinfo.value.exit_code == 2


def test_guard_allows_repl_with_dialog_mode():
    config = _config(mode="dialog")
    cli._guard_one_shot_dialog(None, config)  # не должно бросить — вопроса нет, это REPL


def test_guard_allows_one_shot_without_dialog_mode():
    config = _config()
    cli._guard_one_shot_dialog("вопрос", config)  # не должно бросить — mode=chat


# --- НАХОДКА 3: mode=dialog без done не должен терять введённую строку ---


def test_dialog_without_done_does_not_swallow_input(monkeypatch, capsys, tmp_path):
    """Раньше _run_dialog() при отсутствии done печатала warning и делала
    return ДО обращения к API — введённый текст просто исчезал. Реплика
    обязана дойти до модели обычным одноходовым путём."""
    config = _config(mode="dialog")  # done НЕ задан
    monkeypatch.setattr(cli, "HISTORY_PATH", tmp_path / "history.json")

    prompts = iter(["хочу приготовить салат"])

    def fake_prompt(*args, **kwargs):
        try:
            return next(prompts)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr(cli.typer, "prompt", fake_prompt)

    seen_messages = []

    def fake_complete(config, messages, capabilities=None):
        seen_messages.append(messages)
        return chat_core.CallResult(
            text="ответ", model_requested=config.model, finish_reason="stop"
        )

    monkeypatch.setattr(cli.chat_core, "complete", fake_complete)

    cli._repl(config)  # не должно зависнуть/упасть

    stderr = _plain(capsys.readouterr().err)
    assert "требует done" in stderr
    assert len(seen_messages) == 1  # реплика дошла до API, а не потерялась
    assert seen_messages[0][-1]["content"] == "хочу приготовить салат"


# --- НАХОДКА 4: журнал должен писать текущий день, а не зашитый 1 ---


def test_ask_once_logs_current_day_not_hardcoded_one(monkeypatch):
    config = _config()
    monkeypatch.setattr(
        cli.chat_core,
        "complete",
        lambda config, messages, capabilities=None: chat_core.CallResult(
            text="ok", model_requested=config.model, finish_reason="stop"
        ),
    )
    logged = {}
    monkeypatch.setattr(cli, "log_call", lambda result, messages, **kwargs: logged.update(kwargs))

    cli._ask_once(config, "вопрос")

    assert cli.DAY == 2
    assert logged["day"] == cli.DAY


def test_run_dialog_logs_current_day_not_hardcoded_one(monkeypatch):
    session = _session(mode="dialog", done="json:done", max_turns=1)
    monkeypatch.setattr(
        cli.chat_core,
        "complete",
        lambda config, messages, capabilities=None: chat_core.CallResult(
            text='{"done": false, "question": "?"}',
            model_requested=config.model,
            finish_reason="stop",
        ),
    )
    logged = {}
    monkeypatch.setattr(cli, "log_call", lambda result, messages, **kwargs: logged.update(kwargs))

    cli._run_dialog(session, "хочу приготовить салат")

    assert logged["day"] == cli.DAY


# --- НАХОДКА 5: журнал должен писать то, что реально ушло в API ---


def test_ask_once_logs_sent_messages_not_pre_format_messages(monkeypatch):
    """result.sent_messages — фактически ушедшие messages (с дописанной
    инструкцией формата); журнал не должен писать messages ДО дописывания."""
    config = _config(format="json")
    sent = [
        {"role": "system", "content": "дописанная инструкция формата"},
        {"role": "user", "content": "вопрос"},
    ]

    monkeypatch.setattr(
        cli.chat_core,
        "complete",
        lambda config, messages, capabilities=None: chat_core.CallResult(
            text="{}", model_requested=config.model, finish_reason="stop", sent_messages=sent
        ),
    )
    logged = {}
    monkeypatch.setattr(
        cli,
        "log_call",
        lambda result, messages, **kwargs: logged.update(messages=messages),
    )

    cli._ask_once(config, "вопрос")

    assert logged["messages"] == sent


def test_ask_once_falls_back_to_raw_messages_when_sent_messages_missing(monkeypatch):
    """sent_messages=None (вызов не дошёл до _payload()) — fallback на messages."""
    config = _config()

    monkeypatch.setattr(
        cli.chat_core,
        "complete",
        lambda config, messages, capabilities=None: chat_core.CallResult(
            text="ok", model_requested=config.model, finish_reason="stop", sent_messages=None
        ),
    )
    logged = {}
    monkeypatch.setattr(
        cli,
        "log_call",
        lambda result, messages, **kwargs: logged.update(messages=messages),
    )

    cli._ask_once(config, "вопрос")

    assert logged["messages"][-1]["content"] == "вопрос"


# --- НАХОДКА 6: --verbose должен печатать реальное решение о стриме ---


def test_ask_once_verbose_prints_should_stream_result_not_raw_config_stream(monkeypatch, capsys):
    """config.stream=True по умолчанию, но should_stream() (замоканный fixture'ой
    no_network этого файла на False) — то, что реально решает _run(). verbose
    обязан печатать её результат, а не сырой config.stream."""
    config = _config()
    config.verbose = True
    assert config.stream is True

    monkeypatch.setattr(
        cli.chat_core,
        "complete",
        lambda config, messages, capabilities=None: chat_core.CallResult(
            text="ok", model_requested=config.model, finish_reason="stop"
        ),
    )

    cli._ask_once(config, "вопрос")

    stderr = _plain(capsys.readouterr().err)
    assert "stream=False" in stderr


def test_max_turns_none_is_treated_as_ten(monkeypatch):
    """`/set max_turns default` сбрасывает в None — цикл обязан читать это как 10."""
    session = _session(mode="dialog", done="json:done")
    session.config.params.max_turns = None  # эмулирует "/set max_turns default"
    call_count = 0

    def fake_complete(config, messages, capabilities=None):
        nonlocal call_count
        call_count += 1
        return chat_core.CallResult(
            text='{"done": false, "question": "ещё"}',
            model_requested=config.model,
            finish_reason="stop",
        )

    monkeypatch.setattr(cli.chat_core, "complete", fake_complete)
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "дальше")

    cli._run_dialog(session, "хочу приготовить салат")

    assert call_count == 10


# --- НАХОДКА 7: слэш-команды внутри диалога не должны уходить в API ---


def test_run_dialog_returns_slash_command_without_sending_it_to_api(monkeypatch):
    """Живой прогон demo-шага 5: диалог не сошёлся за отведённые ходы,
    "/exit" попал в typer.prompt() внутри цикла — раньше он безусловно
    уходил в модель как очередная реплика. _run_dialog обязан распознать
    строку с "/" и вернуть её наверх, не потратив на неё вызов API."""
    session = _session(mode="dialog", done="json:done", max_turns=10)
    call_count = 0

    def fake_complete(config, messages, capabilities=None):
        nonlocal call_count
        call_count += 1
        return chat_core.CallResult(
            text='{"done": false, "question": "какой салат?"}',
            model_requested=config.model,
            finish_reason="stop",
        )

    monkeypatch.setattr(cli.chat_core, "complete", fake_complete)
    # Первая (и единственная) реплика после исходной задачи — команда выхода.
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "/exit")

    pending = cli._run_dialog(session, "хочу приготовить салат")

    assert pending == "/exit"
    assert call_count == 1  # только исходная задача ушла в API, "/exit" — нет


def test_repl_dialog_exit_command_ends_session_without_extra_api_call(monkeypatch, tmp_path):
    """Интеграционный сценарий шага 5: REPL → mode=dialog → диалог не
    сходится → "/exit" завершает сессию, а не уходит в модель как реплика."""
    config = _config(mode="dialog", done="json:done", max_turns=10)
    monkeypatch.setattr(cli, "HISTORY_PATH", tmp_path / "history.json")

    lines = iter(["хочу приготовить салат", "/exit"])

    def fake_prompt(*args, **kwargs):
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr(cli.typer, "prompt", fake_prompt)

    call_count = 0

    def fake_complete(config, messages, capabilities=None):
        nonlocal call_count
        call_count += 1
        return chat_core.CallResult(
            text='{"done": false, "question": "какой салат?"}',
            model_requested=config.model,
            finish_reason="stop",
        )

    monkeypatch.setattr(cli.chat_core, "complete", fake_complete)

    cli._repl(config)  # не должно зависнуть — "/exit" обязан завершить REPL

    assert call_count == 1  # ровно один ход диалога, "/exit" в API не ушёл


def test_repl_dialog_other_command_is_executed_not_lost(monkeypatch, tmp_path, capsys):
    """Не только "/exit" — любая команда внутри диалога обязана исполниться
    в обычном REPL, а не потеряться и не уйти в модель как реплика."""
    config = _config(mode="dialog", done="json:done", max_turns=10)
    monkeypatch.setattr(cli, "HISTORY_PATH", tmp_path / "history.json")

    lines = iter(["хочу приготовить салат", "/reset"])

    def fake_prompt(*args, **kwargs):
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr(cli.typer, "prompt", fake_prompt)

    call_count = 0

    def fake_complete(config, messages, capabilities=None):
        nonlocal call_count
        call_count += 1
        return chat_core.CallResult(
            text='{"done": false, "question": "какой салат?"}',
            model_requested=config.model,
            finish_reason="stop",
        )

    monkeypatch.setattr(cli.chat_core, "complete", fake_complete)

    cli._repl(config)  # второй EOFError на следующей внешней подсказке завершает цикл

    stderr = _plain(capsys.readouterr().err)
    assert "история очищена" in stderr  # /reset реально исполнилась
    assert call_count == 1  # /reset не ушла в API как реплика диалога


def test_dialog_convergence_still_returns_control_to_repl(monkeypatch, tmp_path, capsys):
    """Регрессия штатного пути: диалог, который СХОДИТСЯ (done:true),
    по-прежнему просто возвращает управление в обычный REPL, а следующий
    "/exit" обрабатывается там как всегда — НАХОДКА 7 не должна была задеть
    случай, когда пользователь ничего не отправлял командой."""
    config = _config(mode="dialog", done="json:done", max_turns=10)
    monkeypatch.setattr(cli, "HISTORY_PATH", tmp_path / "history.json")

    responses = iter(
        [
            '{"done": false, "question": "какой салат?"}',
            '{"done": true, "result": {"name": "греческий"}}',
        ]
    )

    def fake_complete(config, messages, capabilities=None):
        return chat_core.CallResult(
            text=next(responses), model_requested=config.model, finish_reason="stop"
        )

    monkeypatch.setattr(cli.chat_core, "complete", fake_complete)

    lines = iter(["хочу приготовить салат", "греческий, с фетой", "/exit"])

    def fake_prompt(*args, **kwargs):
        try:
            return next(lines)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr(cli.typer, "prompt", fake_prompt)

    cli._repl(config)  # диалог сходится на втором ходу, "/exit" — уже обычный REPL

    stderr = _plain(capsys.readouterr().err)
    assert "ходов: 2" in stderr
    assert "пока" in stderr  # /exit штатно завершил сессию
