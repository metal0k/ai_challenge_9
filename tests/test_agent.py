"""Агент: явная история, обрезка в токенах, детект done, прерывание.

Сети нет: complete/stream подменяются через ЯВНЫЙ конструкторный шов — тот
самый, из-за отсутствия которого в дне 03 monkeypatch не дотягивался до
дефолта аргумента, связанного на импорте (CLAUDE.md).
"""

from __future__ import annotations

import pytest

from advent_core import chat as chat_core
from advent_core.agent import (
    DEFAULT_COMPACT_EVERY,
    DEFAULT_KEEP_LAST,
    DEFAULT_MAX_TURNS,
    INTERRUPT_NOTE,
    SUMMARY_MAX_TOKENS,
    Agent,
    AgentReply,
    done_conflicts_with_stop,
    marker_instruction,
)
from advent_core.compact import SUMMARY_ACK, SUMMARY_PREFIX
from advent_core.config import Config
from advent_core.errors import AdventError
from advent_core.params import AGENT_COMMAND, GenerationParams, defaults_for
from advent_core.telemetry import CallResult, Usage


def make_config(**params) -> Config:
    return Config(api_key="ключ", model="ministral-14b-latest", params=GenerationParams(**params))


class _Recorder:
    """Двойник complete/stream: запоминает вызов, отдаёт заготовленный ответ."""

    def __init__(self, *results: CallResult) -> None:
        self.results = list(results) or [CallResult(text="ответ", model_requested="m")]
        self.calls: list[list[dict]] = []

    def complete(self, config, messages, capabilities=None) -> CallResult:
        self.calls.append(messages)
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]

    def stream(self, config, messages, on_chunk, capabilities=None) -> CallResult:
        self.calls.append(messages)
        result = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        for piece in result.text.split(" "):
            on_chunk(piece)
        return result


class _CharCounter:
    """Счётчик «символ = токен»: числа предсказуемы, сеть не нужна."""

    def __init__(self, *, exact: bool = True) -> None:
        self.exact = exact
        self.name = "символы"
        self.calibrations: list[int | None] = []

    def count(self, messages) -> int | None:
        return sum(len(m["content"]) for m in messages)

    def calibrate(self, messages, prompt_tokens) -> None:
        self.calibrations.append(prompt_tokens)


def build_agent(recorder: _Recorder, config: Config | None = None, **kwargs) -> Agent:
    warnings: list[str] = []
    agent = Agent(
        config or make_config(),
        complete=recorder.complete,
        stream=recorder.stream,
        on_warning=kwargs.pop("on_warning", warnings.append),
        **kwargs,
    )
    # Список предупреждений доступен тесту прямо на агенте — печатать их
    # агент не имеет права, а проверять надо.
    agent.warnings = warnings  # type: ignore[attr-defined]
    return agent


# --- явная история --------------------------------------------------------


def test_history_goes_in_and_comes_back_with_the_exchange():
    recorder = _Recorder(CallResult(text="здравствуй"))
    agent = build_agent(recorder)
    history = [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
    ]

    reply = agent.ask("привет", history)

    assert isinstance(reply, AgentReply)
    assert reply.text == "здравствуй"
    assert reply.history[-2:] == [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "здравствуй"},
    ]
    assert len(reply.history) == 4


def test_caller_history_is_not_mutated():
    agent = build_agent(_Recorder())
    history = [{"role": "user", "content": "было"}]
    agent.ask("вопрос", history)
    assert history == [{"role": "user", "content": "было"}]


def test_agent_does_not_hold_history_between_calls():
    recorder = _Recorder()
    agent = build_agent(recorder)
    agent.ask("первый", [])
    agent.ask("второй", [])
    # Второй вызов ушёл БЕЗ первого обмена: память живёт снаружи агента.
    second = recorder.calls[1]
    assert [m["content"] for m in second] == ["второй"]


def test_agent_prints_nothing(capsys):
    agent = build_agent(_Recorder())
    agent.ask("вопрос", [])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --- шов вызова -----------------------------------------------------------


def test_constructor_seam_wins_over_module_level_function(monkeypatch):
    """Дефолт аргумента связался бы на импорте — этот тест закрепляет шов."""

    def _explode(*args, **kwargs):
        raise AssertionError("агент обязан звать функцию из конструктора")

    monkeypatch.setattr(chat_core, "complete", _explode)
    monkeypatch.setattr(chat_core, "stream", _explode)

    recorder = _Recorder(CallResult(text="ок"))
    reply = build_agent(recorder).ask("вопрос", [])
    assert reply.text == "ок"
    assert len(recorder.calls) == 1


def test_stream_is_used_only_when_a_sink_is_given():
    recorder = _Recorder(CallResult(text="раз два три"))
    agent = build_agent(recorder)
    chunks: list[str] = []
    agent.ask("вопрос", [], on_chunk=chunks.append)
    assert chunks == ["раз", "два", "три"]

    # Без on_chunk печатать некуда — идём через complete.
    chunks.clear()
    agent.ask("ещё", [])
    assert chunks == []


def test_json_format_never_streams():
    # should_stream() выключает стрим для json/schema: вердикт по формату и
    # маркер завершения ищутся по целому ответу.
    recorder = _Recorder(CallResult(text='{"a": 1}'))
    agent = build_agent(recorder, make_config(format="json"))
    chunks: list[str] = []
    agent.ask("вопрос", [], on_chunk=chunks.append)
    assert chunks == []


# --- usage и счёт токенов -------------------------------------------------


def test_unknown_usage_does_not_become_zero():
    recorder = _Recorder(CallResult(text="ответ", usage=Usage()))
    reply = build_agent(recorder).ask("вопрос", [])
    assert reply.result.usage.prompt_tokens is None
    assert reply.result.usage.total_tokens is None
    # Без счётчика заполненность контекста неизвестна — это None, не ноль.
    assert reply.context_tokens is None
    assert reply.context_exact is False


def test_context_tokens_come_from_the_counter():
    recorder = _Recorder(CallResult(text="ответ", usage=Usage(7, 1, 8)))
    counter = _CharCounter()
    reply = build_agent(recorder, counter=counter, context_limit=100_000).ask("вопрос", [])
    assert reply.context_tokens == len("вопрос")
    assert reply.context_exact is True


def test_estimate_is_never_reported_as_exact():
    recorder = _Recorder(CallResult(text="ответ"))
    counter = _CharCounter(exact=False)
    reply = build_agent(recorder, counter=counter, context_limit=100_000).ask("вопрос", [])
    assert reply.context_tokens is not None
    assert reply.context_exact is False


def test_mismatch_with_server_count_is_reported_upwards():
    recorder = _Recorder(CallResult(text="ответ", usage=Usage(999, 1, 1000)))
    counter = _CharCounter()
    agent = build_agent(recorder, counter=counter, context_limit=100_000)
    agent.ask("вопрос", [])
    assert any("разошёлся" in w for w in agent.warnings)
    # Сверка идёт по фактически отправленным сообщениям, и калибровка тоже.
    assert counter.calibrations == [999]


def test_no_mismatch_warning_when_counts_agree():
    recorder = _Recorder(CallResult(text="ответ", usage=Usage(len("вопрос"), 1, 7)))
    agent = build_agent(recorder, counter=_CharCounter(), context_limit=100_000)
    agent.ask("вопрос", [])
    assert not any("разошёлся" in w for w in agent.warnings)


# --- обрезка контекста ----------------------------------------------------


def _pair_history(pairs: int, size: int) -> list[dict]:
    history: list[dict] = []
    for _ in range(pairs):
        history.append({"role": "user", "content": "x" * size})
        history.append({"role": "assistant", "content": "y" * size})
    return history


def test_trimming_drops_whole_pairs_and_reports_the_amount():
    recorder = _Recorder(CallResult(text="ответ"))
    # Окно 100, запас под ответ 10 — порог 90.
    agent = build_agent(
        recorder, make_config(max_tokens=10), counter=_CharCounter(), context_limit=100
    )
    reply = agent.ask("вопрос", _pair_history(3, 30))

    assert reply.dropped == 4, "выброшены две пары целиком"
    assert reply.dropped % 2 == 0
    # В запрос уехало ровно то, что осталось после обрезки.
    sent = recorder.calls[0]
    assert len(sent) == 3  # одна уцелевшая пара + новый вопрос
    assert reply.context_tokens is not None and reply.context_tokens <= 90


def test_reserve_for_the_answer_is_taken_from_max_tokens():
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(
        recorder, make_config(max_tokens=50), counter=_CharCounter(), context_limit=100
    )
    reply = agent.ask("вопрос", _pair_history(3, 30))
    # Порог теперь 100-50=50 — выброшено больше, чем при запасе 10 выше.
    assert reply.dropped == 6
    assert reply.context_tokens is not None and reply.context_tokens <= 50


def test_short_history_is_not_trimmed():
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, counter=_CharCounter(), context_limit=100_000)
    reply = agent.ask("вопрос", _pair_history(1, 10))
    assert reply.dropped == 0


def test_trimming_says_how_many_tokens_it_freed():
    """SPEC §8 требует называть выброшенное и в ходах, и в токенах: «выброшено
    6 сообщений» не отличает освобождённые 200 токенов от 20 000, а именно это
    число и было смыслом перехода с символьного порога недели 01 на токенный."""
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(
        recorder, make_config(max_tokens=10), counter=_CharCounter(), context_limit=100
    )

    reply = agent.ask("вопрос", _pair_history(3, 30))

    assert reply.dropped == 4
    # Две выброшенные пары по 30 символов каждая — счётчик считает символ за
    # токен, поэтому число проверяется точно, а не «больше нуля».
    assert reply.dropped_tokens == 120


def test_dropped_tokens_are_unknown_not_zero_when_counting_fails():
    """Обрезка случилась, а объём назвать нечем — это «—», а не ноль."""

    class _NoCount(_CharCounter):
        def count(self, messages) -> int | None:
            return None

    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, counter=_NoCount(), context_limit=128_000)

    reply = agent.ask("вопрос", _pair_history(30, 1000))

    assert reply.dropped > 0, "символьный бюджет обязан был сработать"
    assert reply.dropped_tokens is None


def test_char_budget_fallback_is_announced_even_when_the_window_is_known():
    """Порог молча уезжает с окна модели на 24 000 символов, когда счёт сорвался.

    Предупреждение стояло под условием «окно неизвестно», поэтому при
    известном окне история резалась раньше времени, а пользователю сообщали
    только «выброшено сообщений N» — по какому правилу, не сказано нигде.
    """

    class _NoCount(_CharCounter):
        def count(self, messages) -> int | None:
            return None

    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, counter=_NoCount(), context_limit=128_000)

    agent.ask("вопрос", _pair_history(1, 10))

    assert any("символьному бюджету" in w for w in agent.warnings)


def test_char_budget_fallback_is_announced_when_there_is_no_counter():
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, counter=None, context_limit=128_000)

    agent.ask("вопрос", [])

    assert any("счётчика токенов нет" in w for w in agent.warnings)


def test_unknown_window_falls_back_to_char_budget_with_a_warning():
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, counter=_CharCounter(), context_limit=None)
    reply = agent.ask("вопрос", _pair_history(1, 10))
    assert reply.dropped == 0
    assert any("max_context_length" in w for w in agent.warnings)


def test_unknown_window_warning_is_said_once():
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, context_limit=None)
    agent.ask("раз", [])
    agent.ask("два", [])
    assert sum("max_context_length" in w for w in agent.warnings) == 1


def test_trimming_stops_when_there_is_nothing_left_to_drop():
    # Вопрос сам по себе не влезает в окно: выбрасывать нечего, крутиться в
    # цикле нельзя — уходит как есть, сервер скажет своё.
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, counter=_CharCounter(), context_limit=1030)
    reply = agent.ask("q" * 500, _pair_history(2, 10))
    assert reply.dropped <= 4
    assert recorder.calls


# --- целевой диалог -------------------------------------------------------


def _dialog_config(**extra):
    return make_config(mode="dialog", done="text:ГОТОВО", **extra)


def test_done_marker_is_detected_on_the_whole_answer():
    recorder = _Recorder(CallResult(text="вот рецепт. ГОТОВО"))
    reply = build_agent(recorder, _dialog_config()).ask("салат", [])
    assert reply.done is True


def test_dialog_without_the_marker_is_not_done():
    recorder = _Recorder(CallResult(text="а какой салат?"))
    reply = build_agent(recorder, _dialog_config()).ask("салат", [])
    assert reply.done is False


def test_chat_mode_never_reports_done():
    # В обычном чате маркера нет вовсе — даже если слово попало в ответ.
    recorder = _Recorder(CallResult(text="ГОТОВО"))
    reply = build_agent(recorder, make_config(done="text:ГОТОВО")).ask("вопрос", [])
    assert reply.done is False


def test_json_done_condition_reads_the_field():
    recorder = _Recorder(CallResult(text='{"finished": true, "result": {}}'))
    config = make_config(mode="dialog", done="json:finished", format="json")
    reply = build_agent(recorder, config).ask("салат", [])
    assert reply.done is True


def test_done_marker_colliding_with_stop_is_caught():
    # API вырезает stop из ответа — совпавший маркер не доедет никогда.
    assert done_conflicts_with_stop("text:ГОТОВО", ["ГОТОВО"])
    assert done_conflicts_with_stop("text:ГОТОВО", ["ГОТО"]), "подстрока тоже режет ответ"
    assert done_conflicts_with_stop("text:ГОТОВО", ["СТОП"]) is None
    assert done_conflicts_with_stop(None, ["ГОТОВО"]) is None
    assert done_conflicts_with_stop("text:ГОТОВО", None) is None


def test_json_done_marker_conflicts_with_any_stop():
    """Для done=json:<поле> опасна любая непустая стоп-строка, а не только
    пересекающаяся с именем поля: API вырезает stop, до formats.is_done()
    доезжает обрубок JSON, разбор падает в JSONDecodeError, и тот штатно
    читается как «ещё не готово» — диалог не закончится ни разу и молча."""
    assert done_conflicts_with_stop("json:ready", ["}"])
    assert done_conflicts_with_stop("json:ready", ["СТОП"]), "имя поля тут ни при чём"
    assert done_conflicts_with_stop("json:ready", None) is None
    assert done_conflicts_with_stop("json:ready", [""]) is None


def test_agent_reports_the_json_stop_collision_upwards():
    recorder = _Recorder(CallResult(text='{"ready": false}'))
    config = make_config(mode="dialog", done="json:ready", format="json", stop=["}"])
    agent = build_agent(recorder, config)

    assert agent.check_done()
    agent.ask("салат", [])
    assert any("не закончится ни разу" in w for w in agent.warnings)


def test_agent_reports_the_stop_collision_upwards():
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, _dialog_config(stop=["ГОТОВО"]))
    assert agent.check_done()
    agent.ask("салат", [])
    assert any("вырезает stop" in w for w in agent.warnings)


def test_dialog_without_done_is_reported():
    agent = build_agent(_Recorder(), make_config(mode="dialog"))
    assert "требует done" in (agent.check_done() or "")


def test_max_turns_default_and_ceiling():
    agent = build_agent(_Recorder(), make_config(mode="dialog", done="text:ГОТОВО"))
    assert agent.max_turns == DEFAULT_MAX_TURNS
    assert agent.turn_limit_reached(DEFAULT_MAX_TURNS) is True
    assert agent.turn_limit_reached(DEFAULT_MAX_TURNS - 1) is False

    limited = build_agent(_Recorder(), _dialog_config(max_turns=3))
    assert limited.max_turns == 3


# --- system prompt --------------------------------------------------------


def test_system_is_persona_plus_dialog_preset():
    agent = build_agent(
        _Recorder(),
        _dialog_config(),
        persona="Ты агент.",
        dialog_preset="Веди диалог.\n\n%%MARKER%%",
    )
    system = agent.system_prompt()
    assert system.startswith("Ты агент.")
    assert "Веди диалог." in system
    assert "ГОТОВО" in system
    assert "%%MARKER%%" not in system


def test_chat_mode_system_is_persona_only():
    agent = build_agent(_Recorder(), make_config(), persona="Ты агент.")
    assert agent.system_prompt() == "Ты агент."


def test_format_layer_is_not_duplicated_in_system():
    # Инструкцию формата дописывает chat._payload() — агент её не трогает.
    agent = build_agent(_Recorder(), make_config(format="json"), persona="Ты агент.")
    assert agent.system_prompt() == "Ты агент."


def test_marker_instruction_mentions_the_needle():
    assert "ГОТОВО" in marker_instruction("text", "ГОТОВО")
    assert "finished" in marker_instruction("json", "finished")


# --- прерывание -----------------------------------------------------------


def test_interrupted_stream_keeps_the_partial_answer_and_marks_it():
    recorder = _Recorder(CallResult(text="начал отвечать", truncated=True))
    agent = build_agent(recorder)
    reply = agent.ask("вопрос", [], on_chunk=lambda _: None)

    # Сохранённая часть остаётся продуктом хода...
    assert reply.text == "начал отвечать"
    # ...а в историю едет она же с явной пометкой: следующий ход модели
    # должен видеть обрыв, а не считать обрубок законченной мыслью.
    assert reply.history[-1]["content"] == f"начал отвечать\n\n{INTERRUPT_NOTE}"


def test_interrupted_empty_answer_still_leaves_a_trace():
    recorder = _Recorder(CallResult(text="", truncated=True))
    reply = build_agent(recorder).ask("вопрос", [], on_chunk=lambda _: None)
    assert reply.history[-1] == {"role": "assistant", "content": INTERRUPT_NOTE}


def test_empty_answer_without_interruption_adds_no_assistant_turn():
    recorder = _Recorder(CallResult(text=""))
    reply = build_agent(recorder).ask("вопрос", [])
    assert reply.history == [{"role": "user", "content": "вопрос"}]


@pytest.mark.parametrize("mode", ["chat", "dialog"])
def test_mode_property_defaults_to_chat(mode):
    agent = build_agent(_Recorder(), make_config(mode=mode, done="text:ГОТОВО"))
    assert agent.mode == mode
    assert build_agent(_Recorder(), make_config()).mode == "chat"


# --- history compaction (day 09) -------------------------------------------

HISTORY_6 = [
    {"role": "user", "content": "в1"},
    {"role": "assistant", "content": "о1"},
    {"role": "user", "content": "в2"},
    {"role": "assistant", "content": "о2"},
    {"role": "user", "content": "в3"},
    {"role": "assistant", "content": "о3"},
]


def compact_config(**params) -> Config:
    """A config with compaction and small thresholds: six messages already trigger it.

    The registry defaults (keep_last=6, compact_every=10) need sixteen
    messages, which would read as a fixture wall instead of a rule.
    """
    params.setdefault("compact", True)
    params.setdefault("keep_last", 2)
    params.setdefault("compact_every", 4)
    return make_config(**params)


class _ConfigRecorder(_Recorder):
    """The same double, but also records each call's config."""

    def __init__(self, *results: CallResult) -> None:
        super().__init__(*results)
        self.configs: list[Config] = []

    def complete(self, config, messages, capabilities=None) -> CallResult:
        self.configs.append(config)
        return super().complete(config, messages, capabilities)


def test_compaction_replaces_old_messages_with_the_pseudo_pair():
    recorder = _Recorder(CallResult(text="пересказ"), CallResult(text="ответ"))
    agent = build_agent(recorder, compact_config(), summary_prompt="сожми")

    agent.ask("вопрос", list(HISTORY_6))

    # First call is the summarizer: its own system, old turns in the body.
    summarizer = recorder.calls[0]
    assert summarizer[0] == {"role": "system", "content": "сожми"}
    assert "в1" in summarizer[1]["content"]

    # Second is the actual turn: pseudo-pair up front, no old turns at all.
    request = recorder.calls[1]
    assert [m["content"] for m in request] == [
        f"{SUMMARY_PREFIX}\n\nпересказ",
        SUMMARY_ACK,
        "в3",
        "о3",
        "вопрос",
    ]
    assert [m["role"] for m in request] == ["user", "assistant", "user", "assistant", "user"]


def test_summary_travels_in_the_reply_and_the_compaction_call_is_not_a_turn():
    recorder = _Recorder(CallResult(text="пересказ"), CallResult(text="ответ"))
    agent = build_agent(recorder, compact_config(), summary_prompt="сожми")

    reply = agent.ask("вопрос", list(HISTORY_6))

    assert reply.summary == "пересказ"
    assert reply.compaction is not None
    assert reply.compaction.covered == 4
    assert reply.compaction.tail == HISTORY_6[4:]
    # The turn's product is the model's answer, not the summary: mixing them
    # up would print internal text to the user instead of the answer.
    assert reply.text == "ответ"
    # The compaction call does not become a conversation turn — only actual turns are in history.
    assert reply.history == [
        *HISTORY_6[4:],
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"},
    ]


def test_compact_off_leaves_the_request_exactly_as_it_was_before_day_09():
    """Days 06-08 are tagged and must keep behaving as before (SPEC §6)."""
    off = _Recorder(CallResult(text="ответ"))
    build_agent(off, compact_config(compact=False), summary_prompt="сожми").ask(
        "вопрос", list(HISTORY_6)
    )
    before = _Recorder(CallResult(text="ответ"))
    build_agent(before, make_config()).ask("вопрос", list(HISTORY_6))

    assert len(off.calls) == 1
    assert off.calls[0] == before.calls[0]


def test_compact_without_a_summarizer_prompt_falls_back_to_trim_with_a_warning():
    """Silent "compaction" that's actually trimming is worse than compaction turned off."""
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, compact_config())

    reply = agent.ask("вопрос", list(HISTORY_6))

    assert len(recorder.calls) == 1
    assert reply.summary is None
    assert reply.compaction is None
    assert any("промпт суммаризатора" in text for text in agent.warnings)


def test_empty_summary_is_not_substituted():
    """An empty summary is not a summary: substituting it would erase the conversation."""
    recorder = _Recorder(CallResult(text="   "), CallResult(text="ответ"))
    agent = build_agent(recorder, compact_config(), summary_prompt="сожми")

    reply = agent.ask("вопрос", list(HISTORY_6))

    assert reply.summary is None
    assert reply.compaction is None
    # The whole history went through untouched: compaction didn't happen, nothing to lose.
    assert [m["content"] for m in recorder.calls[1]][:2] == ["в1", "о1"]
    assert any("пустой пересказ" in text for text in agent.warnings)


def test_trim_after_compaction_never_drops_the_summary_pair():
    """The pseudo-pair IS the compacted form of everything dropped (SPEC §6)."""
    history = [
        *HISTORY_6[:4],
        {"role": "user", "content": "в3" * 60},
        {"role": "assistant", "content": "о3" * 60},
    ]
    recorder = _Recorder(CallResult(text="пересказ"), CallResult(text="ответ"))
    agent = build_agent(
        recorder,
        compact_config(max_tokens=20),
        summary_prompt="сожми",
        counter=_CharCounter(),
        context_limit=140,
    )

    reply = agent.ask("вопрос", history)

    assert reply.dropped == 2
    assert [m["content"] for m in recorder.calls[1]] == [
        f"{SUMMARY_PREFIX}\n\nпересказ",
        SUMMARY_ACK,
        "вопрос",
    ]


def test_budget_trigger_fires_before_the_message_threshold():
    """One long message overflows the window without reaching ten turns."""
    recorder = _Recorder(CallResult(text="пересказ"), CallResult(text="ответ"))
    agent = build_agent(
        recorder,
        compact_config(compact_every=100, max_tokens=10),
        summary_prompt="сожми",
        counter=_CharCounter(),
        context_limit=60,
    )
    history = [
        {"role": "user", "content": "в" * 40},
        {"role": "assistant", "content": "о" * 40},
        {"role": "user", "content": "в2"},
        {"role": "assistant", "content": "о2"},
    ]

    reply = agent.ask("вопрос", history)

    assert reply.compaction is not None
    assert reply.compaction.covered == 2


def test_summarizer_gets_its_own_params_and_the_session_config_is_untouched():
    recorder = _ConfigRecorder(CallResult(text="пересказ"), CallResult(text="ответ"))
    config = compact_config(format="json", stop=["СТОП"], max_tokens=50)
    agent = build_agent(recorder, config, summary_prompt="сожми")

    agent.ask("вопрос", list(HISTORY_6))

    summarizer = recorder.configs[0].params
    assert summarizer.max_tokens == SUMMARY_MAX_TOKENS
    # format would force the summary to be an object, stop would cut it off
    # mid-word — session params don't fit a service call.
    assert summarizer.format is None
    assert summarizer.stop is None
    # The actual turn runs with the user's params: a copy, not an in-place edit.
    assert recorder.configs[1].params.max_tokens == 50
    assert recorder.configs[1].params.format == "json"
    assert agent.config.params.max_tokens == 50


def test_previous_summary_is_folded_into_the_next_one():
    """A second compaction round must inherit the facts of the first (SPEC §3)."""
    recorder = _Recorder(CallResult(text="новый пересказ"), CallResult(text="ответ"))
    agent = build_agent(recorder, compact_config(), summary_prompt="сожми")

    reply = agent.ask("вопрос", list(HISTORY_6), summary="кодовое слово КАРАКУМЫ")

    assert "КАРАКУМЫ" in recorder.calls[0][1]["content"]
    assert reply.summary == "новый пересказ"


def test_summary_goes_into_the_request_even_when_nothing_is_compacted():
    """A summary from a previous turn keeps applying afterward."""
    recorder = _Recorder(CallResult(text="ответ"))
    agent = build_agent(recorder, compact_config(compact_every=100), summary_prompt="сожми")

    reply = agent.ask("вопрос", list(HISTORY_6[:2]), summary="было условлено X")

    assert len(recorder.calls) == 1
    assert recorder.calls[0][0]["content"] == f"{SUMMARY_PREFIX}\n\nбыло условлено X"
    assert reply.summary == "было условлено X"
    assert reply.compaction is None


def test_compact_now_compresses_without_waiting_for_a_trigger():
    recorder = _Recorder(CallResult(text="пересказ"))
    agent = build_agent(recorder, compact_config(compact_every=100), summary_prompt="сожми")

    compaction = agent.compact_now(None, HISTORY_6)

    assert compaction is not None
    assert compaction.summary == "пересказ"
    assert compaction.covered == 4
    assert compaction.tail == HISTORY_6[4:]
    assert len(recorder.calls) == 1


def test_summarizer_failure_falls_back_to_trim_and_does_not_lose_the_turn():
    """Compaction is an optimization, not a precondition: the turn must still happen."""

    class _FailingSummarizer(_Recorder):
        def complete(self, config, messages, capabilities=None) -> CallResult:
            if not self.calls:
                self.calls.append(messages)
                raise AdventError("сеть отвалилась")
            return super().complete(config, messages, capabilities)

    recorder = _FailingSummarizer(CallResult(text="ответ"))
    agent = build_agent(recorder, compact_config(), summary_prompt="сожми")

    reply = agent.ask("вопрос", list(HISTORY_6))

    assert reply.text == "ответ"
    assert reply.summary is None
    assert reply.compaction is None
    # The whole history went through untouched — no compaction, nothing to trim by (no window).
    assert [m["content"] for m in recorder.calls[1]][:2] == ["в1", "о1"]
    assert any("сжатие истории не удалось" in text for text in agent.warnings)


def test_a_paid_compaction_is_parked_when_the_turn_itself_fails():
    """A successful summarizer call must survive the turn that failed after it.

    It is already paid for: losing it means no journal row, nothing in
    /tokens, and — since the history it was built from is unchanged — a second
    summarizer call on the next attempt.
    """

    class _FailingTurn(_Recorder):
        def complete(self, config, messages, capabilities=None) -> CallResult:
            self.calls.append(messages)
            if len(self.calls) == 1:
                return CallResult(text="пересказ", model_requested="m")
            raise AdventError("сеть отвалилась")

    agent = build_agent(_FailingTurn(), compact_config(), summary_prompt="сожми")

    with pytest.raises(AdventError):
        agent.ask("вопрос", list(HISTORY_6))

    pending = agent.take_pending_compaction()
    assert pending is not None
    assert pending.summary == "пересказ"
    # Handed over exactly once: a second caller would bill the same call twice.
    assert agent.take_pending_compaction() is None


def test_a_successful_turn_leaves_nothing_parked():
    """The compaction travels in the reply — a copy left behind would be reported twice."""
    recorder = _Recorder(CallResult(text="пересказ"), CallResult(text="ответ"))
    agent = build_agent(recorder, compact_config(), summary_prompt="сожми")

    reply = agent.ask("вопрос", list(HISTORY_6))

    assert reply.compaction is not None
    assert agent.pending_compaction is None


def test_thresholds_unset_fall_back_to_the_same_numbers_the_registry_hands_out():
    """`None` means "not set", and the fallback must equal the registry default.

    Two constants for one rule (Spec.defaults in params.py and DEFAULT_* in
    agent.py) — this is the only place that compares them, and every other
    compaction test sets both values explicitly.
    """
    agent = build_agent(_Recorder(), compact_config(keep_last=None, compact_every=None))

    assert agent.keep_last == DEFAULT_KEEP_LAST
    assert agent.compact_every == DEFAULT_COMPACT_EVERY
    registry = defaults_for(AGENT_COMMAND)
    assert (registry["keep_last"], registry["compact_every"]) == (6, 10)
    assert (DEFAULT_KEEP_LAST, DEFAULT_COMPACT_EVERY) == (6, 10)


def test_compact_now_without_a_summarizer_names_the_prompt_not_the_toggle():
    """compact=on plus no summarizer prompt is not "compaction is off".

    `/set compact on` is a remedy for a setting that is already on — the user
    re-toggles it and `/compact` keeps doing nothing.
    """
    agent = build_agent(_Recorder(), compact_config())

    assert agent.compact_now(None, list(HISTORY_6)) is None
    assert any("суммаризатора" in text for text in agent.warnings)
    assert not any("/set compact on" in text for text in agent.warnings)


def test_compact_now_off_by_setting_still_points_at_the_toggle():
    agent = build_agent(_Recorder(), make_config(compact=False), summary_prompt="сожми")

    assert agent.compact_now(None, list(HISTORY_6)) is None
    assert any("/set compact on" in text for text in agent.warnings)


def test_compact_now_on_a_short_history_makes_no_call():
    """`/compact` on a short history is a clear message, not an LLM call (§10)."""
    recorder = _Recorder()
    agent = build_agent(recorder, compact_config(), summary_prompt="сожми")

    assert agent.compact_now(None, HISTORY_6[:2]) is None
    assert recorder.calls == []
    assert any("сжимать нечего" in text for text in agent.warnings)
