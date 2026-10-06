"""Сборка сообщений, обрезка истории и разбор чанков стрима — без сети.

complete()/stream() тоже тестируются здесь, но клиент Mistral полностью
замокан (_FakeMistral ниже) — сеть не трогается ни разу, как и во всём проекте.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from advent_core import chat as chat_core
from advent_core import client as client_mod
from advent_core import formats
from advent_core.chat import _extract_delta, build_messages, should_stream, trim_history
from advent_core.client import model_names, resolve_alias
from advent_core.config import Config
from advent_core.errors import AdventError
from advent_core.params import GenerationParams
from advent_core.telemetry import RawToolCall, Usage


def test_messages_order_is_system_history_question():
    history = [
        {"role": "user", "content": "первый"},
        {"role": "assistant", "content": "ответ"},
    ]
    messages = build_messages("второй", system="ты бот", history=history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == "ты бот"
    assert messages[-1]["content"] == "второй"


def test_no_system_message_when_prompt_is_empty():
    messages = build_messages("вопрос", system=None)
    assert [m["role"] for m in messages] == ["user"]


def test_history_is_not_mutated_by_build():
    history = [{"role": "user", "content": "старое"}]
    build_messages("новое", history=history)
    assert len(history) == 1


def test_short_history_survives_trimming():
    history = [{"role": "user", "content": "коротко"}]
    trimmed, dropped = trim_history(history, budget=1000)
    assert dropped == 0
    assert len(trimmed) == 1


def test_long_history_is_trimmed_in_pairs():
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 100} for i in range(10)
    ]
    trimmed, dropped = trim_history(history, budget=250)
    assert dropped > 0
    assert dropped % 2 == 0, "пары user+assistant выкидываются целиком"
    assert sum(len(m["content"]) for m in trimmed) <= 250


def test_trimming_terminates_when_single_message_exceeds_budget():
    """Один гигантский вопрос не должен зациклить обрезку."""
    history = [{"role": "user", "content": "x" * 5000}]
    trimmed, dropped = trim_history(history, budget=10)
    assert trimmed == []
    assert dropped == 1


def _chunk(content, *, model=None, usage=None, finish_reason=None, reasoning_content=None):
    delta_kwargs = {"content": content}
    if reasoning_content is not None:
        delta_kwargs["reasoning_content"] = reasoning_content
    delta = SimpleNamespace(**delta_kwargs)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    data = SimpleNamespace(choices=[choice], model=model, usage=usage)
    return SimpleNamespace(data=data)


def test_extract_delta_reads_text():
    text, usage, model = _extract_delta(_chunk("привет", model="mistral-small-2603"))
    assert text == "привет"
    assert model == "mistral-small-2603"
    assert usage is None


def test_extract_delta_handles_empty_final_chunk():
    """Последний чанк приходит без текста, но с usage."""
    usage_obj = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    text, usage, _ = _extract_delta(_chunk(None, usage=usage_obj))
    assert text == ""
    assert Usage.from_raw(usage).total_tokens == 15


def test_extract_delta_joins_multimodal_blocks():
    blocks = [SimpleNamespace(text="раз "), SimpleNamespace(text="два")]
    text, _, _ = _extract_delta(_chunk(blocks))
    assert text == "раз два"


def test_usage_from_dict_and_object():
    assert Usage.from_raw({"total_tokens": 7}).total_tokens == 7
    assert Usage.from_raw(SimpleNamespace(total_tokens=7)).total_tokens == 7
    assert Usage.from_raw(None).is_empty()


MODELS = [
    {"id": "mistral-small-2603", "aliases": ["mistral-small-latest"]},
    {"id": "ministral-8b-2512", "aliases": ["ministral-8b-latest"]},
]


def test_alias_resolves_to_concrete_version():
    assert resolve_alias(MODELS, "mistral-small-latest") == "mistral-small-2603"


def test_unknown_alias_returns_none_instead_of_guessing():
    assert resolve_alias(MODELS, "mistral-small-2603") is None
    assert resolve_alias(MODELS, "выдуманная-модель") is None


def test_model_names_include_ids_and_aliases():
    names = model_names(MODELS)
    assert "mistral-small-2603" in names
    assert "mistral-small-latest" in names
    assert "нет-такой" not in names


# --- complete()/stream() против фейкового клиента Mistral — сеть не трогается ---


def _config(**parm_kwargs) -> Config:
    return Config(
        api_key="k" * 32,
        model="mistral-small-latest",
        params=GenerationParams.build(**parm_kwargs),
    )


class _FakeChat:
    """Замена mistral.chat: помнит payload, отдаёт заранее заданный ответ."""

    def __init__(self, complete_response=None, stream_events=None):
        self._complete_response = complete_response
        self._stream_events = stream_events or []
        self.complete_kwargs: dict | None = None
        self.stream_kwargs: dict | None = None

    def complete(self, **kwargs):
        self.complete_kwargs = kwargs
        return self._complete_response

    def stream(self, **kwargs):
        self.stream_kwargs = kwargs
        return _FakeStreamCtx(self._stream_events)


class _FakeStreamCtx:
    """response.stream(...) в SDK — контекстный менеджер, отдающий события."""

    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *exc_info):
        return False


class _FakeMistral:
    def __init__(self, complete_response=None, stream_events=None):
        self.chat = _FakeChat(complete_response, stream_events)


def _patch_client(monkeypatch, fake: _FakeMistral) -> None:
    """Подменяет advent_core.chat.mistral_client на фейковый контекст-менеджер."""

    @contextmanager
    def _fake_client(config):
        yield fake

    monkeypatch.setattr(chat_core, "mistral_client", _fake_client)


def _complete_response(
    text: str,
    finish_reason: str | None,
    model: str = "mistral-small-2603",
    reasoning_content: str | None = None,
):
    # reasoning_content не передаётся вовсе, когда параметр не задан — так же,
    # как SimpleNamespace обычного ответа Mistral его не несёт: getattr в
    # chat.complete() тогда должен вернуть None, а не упасть.
    message_kwargs = {"content": text}
    if reasoning_content is not None:
        message_kwargs["reasoning_content"] = reasoning_content
    message = SimpleNamespace(**message_kwargs)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


# Регрессия дня: раньше finish_reason не читался ВООБЩЕ ни в complete(), ни в
# stream() (SPEC-w01d02.md §6.3, §9) — это главная дыра, которую день закрывает.


def test_complete_reads_finish_reason_stop(monkeypatch):
    fake = _FakeMistral(complete_response=_complete_response("привет", "stop"))
    _patch_client(monkeypatch, fake)

    result = chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert result.finish_reason == "stop"
    assert result.text == "привет"


def test_complete_reads_length_finish_reason(monkeypatch):
    fake = _FakeMistral(complete_response=_complete_response("обрезан", "length"))
    _patch_client(monkeypatch, fake)

    result = chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert result.finish_reason == "length"


def test_stream_reads_finish_reason_from_last_chunk(monkeypatch):
    """В стриме finish_reason приходит только в последнем чанке, там же, где usage."""
    events = [
        _chunk("привет", model="mistral-small-2603"),
        _chunk(" мир"),
        _chunk(None, finish_reason="stop"),
    ]
    fake = _FakeMistral(stream_events=events)
    _patch_client(monkeypatch, fake)

    seen = []
    result = chat_core.stream(_config(), [{"role": "user", "content": "q"}], seen.append)

    assert result.finish_reason == "stop"
    assert "".join(seen) == "привет мир"


def test_stream_reads_length_finish_reason(monkeypatch):
    """В стриме нет model_length — только stop/length/error/tool_calls (SPEC §2)."""
    events = [_chunk("текст"), _chunk(None, finish_reason="length")]
    fake = _FakeMistral(stream_events=events)
    _patch_client(monkeypatch, fake)

    result = chat_core.stream(_config(), [{"role": "user", "content": "q"}], lambda c: None)

    assert result.finish_reason == "length"


def test_stream_finish_reason_is_none_when_missing(monkeypatch):
    events = [_chunk("текст")]
    fake = _FakeMistral(stream_events=events)
    _patch_client(monkeypatch, fake)

    result = chat_core.stream(_config(), [{"role": "user", "content": "q"}], lambda c: None)

    assert result.finish_reason is None


# --- should_stream(): единая точка решения «стримить или нет» ---


def test_should_stream_true_by_default():
    assert should_stream(_config()) is True


def test_should_stream_respects_config_stream_flag():
    config = _config()
    config.stream = False
    assert should_stream(config) is False


@pytest.mark.parametrize("fmt", ["json", "schema"])
def test_should_stream_disabled_for_structured_formats(fmt):
    """Вердикт по формату и поиск маркера завершения возможны только по целому ответу."""
    assert should_stream(_config(format=fmt)) is False


@pytest.mark.parametrize("fmt", ["text", "yaml", "md"])
def test_should_stream_enabled_for_free_form_formats(fmt):
    assert should_stream(_config(format=fmt)) is True


# --- _payload(): response_format для json/schema, инструкция дописана к system ---


def test_payload_has_no_response_format_for_plain_text():
    payload, _, format_name, schema = chat_core._payload(
        _config(), [{"role": "user", "content": "q"}]
    )
    assert "response_format" not in payload
    assert format_name == "text"
    assert schema is None


def test_payload_includes_response_format_for_json():
    payload, _, format_name, _ = chat_core._payload(
        _config(format="json"), [{"role": "user", "content": "q"}]
    )
    assert payload["response_format"] == {"type": "json_object"}
    assert format_name == "json"


def test_payload_includes_response_format_for_schema(tmp_path):
    schema_dict = {"title": "recipe", "type": "object"}
    schema_path = tmp_path / "s.json"
    schema_path.write_text(json.dumps(schema_dict), encoding="utf-8")

    payload, _, format_name, schema = chat_core._payload(
        _config(format="schema", schema_file=str(schema_path)),
        [{"role": "user", "content": "q"}],
    )

    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema_definition"] == schema_dict
    assert format_name == "schema"
    assert schema == schema_dict


def test_payload_schema_without_file_raises_advent_error_not_config_error():
    """REPL ловит только AdventError — ConfigError из formats.py обязан быть переведён."""
    with pytest.raises(AdventError):
        chat_core._payload(_config(format="schema"), [{"role": "user", "content": "q"}])


def test_payload_appends_format_instruction_without_overwriting_user_system():
    messages = build_messages("вопрос", system="ты дружелюбный ассистент")

    payload, *_ = chat_core._payload(_config(format="json"), messages)

    system_message = payload["messages"][0]
    assert system_message["role"] == "system"
    assert system_message["content"].startswith("ты дружелюбный ассистент")
    assert "JSON" in system_message["content"]


def test_payload_local_params_never_reach_the_request():
    """format/schema_file/done/mode/max_turns управляют клиентом, не уходят в API."""
    payload, _, _, _ = chat_core._payload(
        _config(format="text", mode="dialog", done="text:[ГОТОВО]", max_turns=3),
        [{"role": "user", "content": "q"}],
    )
    for name in ("format", "schema_file", "done", "mode", "max_turns"):
        assert name not in payload


# --------------------------------------------------------------------------
# Лимит частоты из заголовка ответа (Day 05, SPEC-w01d05.md §13)
# --------------------------------------------------------------------------


class _FakeHeaders:
    """httpx-подобный ответ: важна только нечувствительность ключей к регистру,
    которую _RateLimitProbe обязан снять при сохранении."""

    def __init__(self, headers):
        self.headers = headers


def test_probe_lowercases_header_names():
    """Заголовки httpx регистронезависимы, обычный dict — уже нет. Если не
    привести ключи при сохранении, чтение по 'x-ratelimit-...' промахнётся на
    ответе, где сервер прислал 'X-RateLimit-...'."""
    probe = client_mod._RateLimitProbe()

    probe.after_success(None, _FakeHeaders({"X-RateLimit-Limit-Req-Minute": "30"}))

    assert probe.headers["x-ratelimit-limit-req-minute"] == "30"


def test_probe_returns_the_response_unchanged():
    """SDKHooks присваивает возвращённое значение обратно в цепочку хуков:
    вернуть None здесь значило бы сломать разбор ответа всем следующим."""
    probe = client_mod._RateLimitProbe()
    response = _FakeHeaders({"x-ratelimit-limit-req-minute": "30"})

    assert probe.after_success(None, response) is response


def test_requests_per_minute_reads_zero_as_zero_not_as_unknown():
    """Ноль — законный ответ Mistral для модели, недоступной на тарифе
    (проверено 2026-09-04), и путать его с None нельзя: None означает
    «заголовка не было», а ноль — «звать эту модель бесполезно»."""
    fake = _FakeMistral()
    fake._advent_rate_limit_probe = client_mod._RateLimitProbe()
    fake._advent_rate_limit_probe.headers = {"x-ratelimit-limit-req-minute": "0"}

    assert client_mod.requests_per_minute(fake) == 0


def test_requests_per_minute_is_none_without_a_probe():
    """Шов регистрации хука приватный и может исчезнуть в апдейте SDK. Тогда
    честный ответ — «неизвестно», а не подставленное умолчание."""
    assert client_mod.requests_per_minute(_FakeMistral()) is None


def test_requests_per_minute_is_none_on_garbage():
    fake = _FakeMistral()
    fake._advent_rate_limit_probe = client_mod._RateLimitProbe()
    fake._advent_rate_limit_probe.headers = {"x-ratelimit-limit-req-minute": "много"}

    assert client_mod.requests_per_minute(fake) is None


def test_complete_carries_the_rate_limit_into_the_result(monkeypatch):
    """Развёртка Day 05 считает по нему паузу — без этого поля она вернулась бы
    к зашитой константе."""
    fake = _FakeMistral(complete_response=_complete_response("привет", "stop"))
    fake._advent_rate_limit_probe = client_mod._RateLimitProbe()
    fake._advent_rate_limit_probe.headers = {"x-ratelimit-limit-req-minute": "30"}
    _patch_client(monkeypatch, fake)

    result = chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert result.rate_limit_rpm == 30


# --------------------------------------------------------------------------
# Локальный OpenAI-совместимый endpoint (LM Studio): reasoning_content
# --------------------------------------------------------------------------


def test_usage_reasoning_tokens_from_dict():
    usage = Usage.from_raw({"completion_tokens_details": {"reasoning_tokens": 41}})
    assert usage.reasoning_tokens == 41


def test_usage_reasoning_tokens_from_object():
    details = SimpleNamespace(reasoning_tokens=41)
    usage = Usage.from_raw(SimpleNamespace(completion_tokens_details=details))
    assert usage.reasoning_tokens == 41


def test_usage_reasoning_tokens_missing_is_none():
    """Обычная модель Mistral не присылает эту деталь вовсе — это не ошибка."""
    usage = Usage.from_raw({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    assert usage.reasoning_tokens is None


def test_missing_reasoning_tokens_does_not_affect_is_empty():
    """reasoning_tokens — опциональная деталь учёта, не признак прихода usage."""
    usage = Usage.from_raw({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    assert usage.is_empty() is False


def test_complete_puts_reasoning_content_into_reasoning_text(monkeypatch):
    fake = _FakeMistral(
        complete_response=_complete_response("42", "stop", reasoning_content="думаю: 6*7=42")
    )
    _patch_client(monkeypatch, fake)

    result = chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert result.text == "42"
    assert result.reasoning_text == "думаю: 6*7=42"


def test_complete_reasoning_text_is_none_when_field_absent(monkeypatch):
    """Обычная модель Mistral не несёт reasoning_content — поле остаётся None, не ''."""
    fake = _FakeMistral(complete_response=_complete_response("привет", "stop"))
    _patch_client(monkeypatch, fake)

    result = chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert result.reasoning_text is None


def test_complete_empty_content_stays_empty_even_with_reasoning(monkeypatch):
    """Пустой content при малом max_tokens — законный результат, а не транспортная
    ошибка: подменять text рассуждением нельзя, иначе метрики точности (день 04)
    начнут мерить не то, что реально ответила модель."""
    fake = _FakeMistral(
        complete_response=_complete_response(
            "", "length", reasoning_content="длинная цепочка рассуждения, до ответа не дошла"
        )
    )
    _patch_client(monkeypatch, fake)

    result = chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert result.text == ""
    assert result.reasoning_text == "длинная цепочка рассуждения, до ответа не дошла"


def test_stream_accumulates_reasoning_separately_from_on_chunk(monkeypatch):
    """Дельты reasoning_content не должны ни падать, ни попасть в on_chunk как ответ."""
    events = [
        _chunk(None, reasoning_content="дум"),
        _chunk(None, reasoning_content="аю"),
        _chunk("от"),
        _chunk("вет"),
        _chunk(None, finish_reason="stop"),
    ]
    fake = _FakeMistral(stream_events=events)
    _patch_client(monkeypatch, fake)

    seen = []
    result = chat_core.stream(_config(), [{"role": "user", "content": "q"}], seen.append)

    assert "".join(seen) == "ответ"
    assert result.text == "ответ"
    assert result.reasoning_text == "думаю"


def test_stream_reasoning_text_is_none_without_reasoning_deltas(monkeypatch):
    events = [_chunk("привет"), _chunk(None, finish_reason="stop")]
    fake = _FakeMistral(stream_events=events)
    _patch_client(monkeypatch, fake)

    result = chat_core.stream(_config(), [{"role": "user", "content": "q"}], lambda c: None)

    assert result.reasoning_text is None


# --------------------------------------------------------------------------
# tools/tool_choice (W04D17 §3): function-calling plumbing on complete()
# --------------------------------------------------------------------------


def test_complete_without_tools_sends_no_tools_keys(monkeypatch):
    """Backward compat: tools=None/tool_choice=None must not reach the SDK."""
    fake = _FakeMistral(complete_response=_complete_response("привет", "stop"))
    _patch_client(monkeypatch, fake)

    chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert "tools" not in fake.chat.complete_kwargs
    assert "tool_choice" not in fake.chat.complete_kwargs


def test_complete_sends_tools_and_tool_choice_verbatim(monkeypatch):
    fake = _FakeMistral(complete_response=_complete_response("", "tool_calls"))
    _patch_client(monkeypatch, fake)
    tools = [
        {
            "type": "function",
            "function": {"name": "git_log", "description": "…", "parameters": {}},
        }
    ]

    chat_core.complete(
        _config(),
        [{"role": "user", "content": "q"}],
        tools=tools,
        tool_choice="auto",
    )

    assert fake.chat.complete_kwargs["tools"] == tools
    assert fake.chat.complete_kwargs["tool_choice"] == "auto"


def test_complete_parses_tool_calls_from_response(monkeypatch):
    """Literal shape from a live SDK probe (2026-09-22), not an invented one:
    choice.message.tool_calls == [ToolCall(function=FunctionCall(name='git_log',
    arguments='{"n": 3}'), id='BpfW4tBmq', type='function', index=0)],
    finish_reason == 'tool_calls', content == ''."""
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="git_log", arguments='{"n": 3}'),
        id="BpfW4tBmq",
        type="function",
        index=0,
    )
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response = SimpleNamespace(choices=[choice], model="mistral-small-2603", usage=usage)
    fake = _FakeMistral(complete_response=response)
    _patch_client(monkeypatch, fake)

    result = chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert result.finish_reason == "tool_calls"
    assert result.text == ""
    assert result.tool_calls == (RawToolCall(id="BpfW4tBmq", name="git_log", arguments='{"n": 3}'),)


def test_complete_encodes_dict_arguments_as_json_not_python_repr(monkeypatch):
    """SDK's FunctionCall.arguments type allows str OR dict (SPEC §4 Risks:
    'тест должен покрыть именно ветвление, а не один случай'). Not observed
    live (probe only showed str) — covers the type branch defensively.

    json.dumps, not str(): agent.py json.loads() this downstream, and a
    Python repr (single quotes) never parses — the dict branch used to turn
    a well-formed call into a paid "bad arguments" retry round. The spec asks
    only that both types be accepted; producing valid JSON satisfies it."""
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="git_log", arguments={"n": 3}),
        id="xyz",
        type="function",
        index=0,
    )
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response = SimpleNamespace(choices=[choice], model="mistral-small-2603", usage=usage)
    fake = _FakeMistral(complete_response=response)
    _patch_client(monkeypatch, fake)

    result = chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert result.tool_calls[0].arguments == '{"n": 3}'
    assert isinstance(result.tool_calls[0].arguments, str)
    assert json.loads(result.tool_calls[0].arguments) == {"n": 3}


def test_sent_messages_is_a_snapshot_not_an_alias_of_the_caller_list(monkeypatch):
    """`_payload()` при format=text отдаёт ТОТ ЖЕ список по identity.

    Tool-loop в agent.py дописывает в него эфемерные сообщения между
    раундами, и запаркованный `CallResult` раунда 1 дорастал до сообщений,
    которых в его запросе не было и быть не могло — а `week_02/cli.py`
    пишет их в журнал строкой `mcp_tool_round`.
    """
    fake = _FakeMistral(complete_response=_complete_response("привет", "stop"))
    _patch_client(monkeypatch, fake)
    messages = [{"role": "user", "content": "q"}]

    result = chat_core.complete(_config(), messages)

    assert result.sent_messages == messages
    assert result.sent_messages is not messages
    messages.append({"role": "assistant", "content": "позже"})
    assert len(result.sent_messages) == 1, "снимок вырос вслед за списком вызывающего кода"


def test_complete_tool_calls_empty_when_response_has_none(monkeypatch):
    """Ordinary response (no tool_calls attribute at all) — getattr fallback."""
    fake = _FakeMistral(complete_response=_complete_response("привет", "stop"))
    _patch_client(monkeypatch, fake)

    result = chat_core.complete(_config(), [{"role": "user", "content": "q"}])

    assert result.tool_calls == ()


def test_list_models_uses_base_url_when_set(monkeypatch):
    """С заданным base_url список моделей идёт на локальный сервер, не в облако."""
    calls = []

    def _handle(self, request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "ornith"}]}, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _handle)

    config = Config(api_key="lm-studio-local", base_url="http://127.0.0.1:1234")
    models = client_mod.list_models(config)

    assert calls == ["http://127.0.0.1:1234/v1/models"]
    assert models == [{"id": "ornith"}]


def test_list_models_uses_cloud_url_without_base_url(monkeypatch):
    calls = []

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    def _fake_get(url, **kwargs):
        calls.append(url)
        return _FakeResponse()

    monkeypatch.setattr(client_mod.httpx, "get", _fake_get)

    client_mod.list_models(Config(api_key="k" * 32))

    assert calls == [client_mod.MODELS_URL]


# --- base_url: complete()/stream() идут через advent_core.openai_compat ---------

import httpx  # noqa: E402

from advent_core import openai_compat as oc_mod  # noqa: E402
from advent_core.errors import StreamTruncated  # noqa: E402


def _local_config(**parm_kwargs) -> Config:
    return Config(
        api_key="lm-studio-local",
        model="ornith",
        params=GenerationParams.build(**parm_kwargs),
        base_url="http://127.0.0.1:1234",
    )


def _local_server(monkeypatch, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def factory(timeout, transport=None):
        def wrapped(request):
            seen.append(request)
            return handler(request)

        return httpx.Client(transport=httpx.MockTransport(wrapped))

    monkeypatch.setattr(oc_mod, "_make_client", factory)

    @contextmanager
    def _no_sdk(config):
        raise AssertionError("с base_url SDK Mistral не используется")
        yield

    monkeypatch.setattr(chat_core, "mistral_client", _no_sdk)
    return seen


def _sse(*events, done=True) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


_LOCAL_USAGE = {
    "choices": [],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 30,
        "total_tokens": 42,
        "completion_tokens_details": {"reasoning_tokens": 25},
    },
}


def _completion_json(content="ответ", reasoning=None, finish="stop", tool_calls=None):
    message = {"content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": "ornith",
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }


def test_local_stream_returns_exact_usage_reasoning_and_finish(monkeypatch):
    events = [
        {"choices": [{"delta": {"reasoning_content": "дума"}}]},
        {"choices": [{"delta": {"reasoning_content": "ю"}}]},
        {"choices": [{"delta": {"content": "при"}}]},
        {"choices": [{"delta": {"content": "вет"}, "finish_reason": "stop"}]},
        _LOCAL_USAGE,
    ]
    seen = _local_server(monkeypatch, lambda r: httpx.Response(200, content=_sse(*events)))
    chunks, thoughts = [], []

    result = chat_core.stream(
        _local_config(),
        [{"role": "user", "content": "q"}],
        chunks.append,
        on_reasoning=thoughts.append,
    )

    assert "".join(chunks) == "привет" and result.text == "привет"
    assert thoughts == ["дума", "ю"] and result.reasoning_text == "думаю"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (12, 30)
    assert result.usage.reasoning_tokens == 25
    assert result.finish_reason == "stop" and result.truncated is False
    assert result.stream is True and result.model_requested == "ornith"
    assert seen[0].url.path == "/v1/chat/completions"


def test_local_request_body_is_exactly_this_literal(monkeypatch):
    """A literal expected body, not _payload(): the test must not share code with the SUT."""
    ok = {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}
    seen = _local_server(monkeypatch, lambda r: httpx.Response(200, content=_sse(ok, _LOCAL_USAGE)))
    config = _local_config(temperature=0.3, max_tokens=77, random_seed=5, format="json")

    chat_core.stream(config, [{"role": "user", "content": "q"}], lambda c: None)

    # LM Studio refuses json_object, so the wire body has no response_format; the JSON
    # demand travels as the system message the format preset adds.
    assert json.loads(seen[0].content) == {
        "model": "ornith",
        "messages": [
            {
                "role": "system",
                "content": "Верни ответ строго в виде одного валидного JSON-объекта — "
                "без пояснений и без обёртки в ```.",
            },
            {"role": "user", "content": "q"},
        ],
        "temperature": 0.3,
        "max_tokens": 77,
        "random_seed": 5,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_cloud_sdk_payload_keeps_response_format_literal(monkeypatch):
    fake = _FakeMistral(complete_response=_complete_response("{}", "stop"))
    _patch_client(monkeypatch, fake)

    chat_core.complete(_config(temperature=0.3, format="json"), [{"role": "user", "content": "q"}])

    assert fake.chat.complete_kwargs == {
        "model": "mistral-small-latest",
        "messages": [
            {
                "role": "system",
                "content": "Верни ответ строго в виде одного валидного JSON-объекта — "
                "без пояснений и без обёртки в ```.",
            },
            {"role": "user", "content": "q"},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }


def _cloud_base_url_config(**parm_kwargs) -> Config:
    # An explicit non-loopback base_url: previous SDK path, real key, no LM Studio rewrite.
    return Config(
        api_key="k" * 32,
        model="mistral-small-latest",
        params=GenerationParams.build(**parm_kwargs),
        base_url="https://api.mistral.ai",
    )


def test_non_loopback_base_url_goes_through_the_sdk_with_the_real_key(monkeypatch):
    fake = _FakeMistral(complete_response=_complete_response("привет", "stop"))
    configs = []

    @contextmanager
    def _client(config):
        configs.append(config)
        yield fake

    monkeypatch.setattr(chat_core, "mistral_client", _client)

    def _no_local(*a, **k):
        raise AssertionError("non-loopback base_url must not use openai_compat")

    monkeypatch.setattr(oc_mod, "chat_complete", _no_local)
    monkeypatch.setattr(oc_mod, "chat_stream", _no_local)

    config = _cloud_base_url_config(format="json")
    chat_core.complete(config, [{"role": "user", "content": "q"}])

    assert configs[0].api_key == "k" * 32
    # json_object stays: the LM Studio wire rewrite is for a loopback server only
    assert fake.chat.complete_kwargs["response_format"] == {"type": "json_object"}


def test_loopback_base_url_still_drops_json_object_on_the_wire(monkeypatch):
    seen = _local_server(monkeypatch, lambda r: httpx.Response(200, json=_completion_json()))
    chat_core.complete(_local_config(format="json"), [{"role": "user", "content": "q"}])
    body = json.loads(seen[0].content)
    assert "response_format" not in body
    assert body["model"] == "ornith" and body["stream"] is False


def test_local_unset_params_are_not_sent(monkeypatch):
    seen = _local_server(monkeypatch, lambda r: httpx.Response(200, json=_completion_json()))
    chat_core.complete(_local_config(), [{"role": "user", "content": "q"}])
    assert set(json.loads(seen[0].content)) == {"model", "messages", "stream"}


def test_local_skipped_params_are_reported_like_the_sdk_path(monkeypatch):
    _local_server(monkeypatch, lambda r: httpx.Response(200, json=_completion_json()))
    config = _local_config(reasoning_effort="high")
    result = chat_core.complete(
        config, [{"role": "user", "content": "q"}], capabilities={"reasoning": False}
    )
    assert result.skipped_params == ["reasoning_effort"]


def test_local_complete_returns_usage_reasoning_tool_calls_and_snapshot(monkeypatch):
    tool_calls = [{"id": "t1", "function": {"name": "git_log", "arguments": '{"n": 2}'}}]
    answer = _completion_json(
        "", reasoning="надо вызвать", finish="tool_calls", tool_calls=tool_calls
    )
    seen = _local_server(monkeypatch, lambda r: httpx.Response(200, json=answer))
    tools = [{"type": "function", "function": {"name": "git_log"}}]
    messages = [{"role": "user", "content": "q"}]

    result = chat_core.complete(_local_config(), messages, tools=tools, tool_choice="auto")
    messages.append({"role": "assistant", "content": "позже"})  # the agent's tool loop does this

    body = json.loads(seen[0].content)
    assert body["tools"] == tools and body["tool_choice"] == "auto" and body["stream"] is False
    assert result.finish_reason == "tool_calls" and result.stream is False
    assert result.tool_calls == (RawToolCall(id="t1", name="git_log", arguments='{"n": 2}'),)
    assert result.reasoning_text == "надо вызвать"
    assert (result.usage.prompt_tokens, result.usage.total_tokens) == (5, 11)
    assert result.sent_messages == [{"role": "user", "content": "q"}]  # a snapshot, not the list


def test_local_complete_without_usage_gives_an_empty_usage(monkeypatch):
    body = _completion_json()
    del body["usage"]
    _local_server(monkeypatch, lambda r: httpx.Response(200, json=body))
    result = chat_core.complete(_local_config(), [{"role": "user", "content": "q"}])
    assert result.usage.is_empty()


def test_local_format_verdict_is_filled(monkeypatch):
    _local_server(monkeypatch, lambda r: httpx.Response(200, json=_completion_json('{"a": 1}')))
    result = chat_core.complete(_local_config(format="json"), [{"role": "user", "content": "q"}])
    assert result.format_ok is True


def test_local_stream_works_without_an_on_reasoning_callback(monkeypatch):
    events = [
        {"choices": [{"delta": {"reasoning_content": "x"}}]},
        {"choices": [{"delta": {"content": "y"}, "finish_reason": "stop"}]},
    ]
    _local_server(monkeypatch, lambda r: httpx.Response(200, content=_sse(*events)))
    result = chat_core.stream(_local_config(), [{"role": "user", "content": "q"}], lambda c: None)
    assert result.text == "y" and result.reasoning_text == "x"


def test_local_stream_reasoning_only_with_length_is_an_error(monkeypatch):
    events = [
        {"choices": [{"delta": {"reasoning_content": "думаю и думаю"}, "finish_reason": "length"}]},
        _LOCAL_USAGE,
    ]
    _local_server(monkeypatch, lambda r: httpx.Response(200, content=_sse(*events)))
    with pytest.raises(StreamTruncated) as info:
        chat_core.stream(_local_config(), [{"role": "user", "content": "q"}], lambda c: None)
    assert "reasoning съел max_tokens" in info.value.message


def test_local_complete_reasoning_only_with_length_is_an_error(monkeypatch):
    answer = _completion_json("", reasoning="долго", finish="length")
    _local_server(monkeypatch, lambda r: httpx.Response(200, json=answer))
    with pytest.raises(StreamTruncated):
        chat_core.complete(_local_config(), [{"role": "user", "content": "q"}])


def test_local_length_with_an_answer_is_not_an_error(monkeypatch):
    answer = _completion_json("обрезано", reasoning="r", finish="length")
    _local_server(monkeypatch, lambda r: httpx.Response(200, json=answer))
    result = chat_core.complete(_local_config(), [{"role": "user", "content": "q"}])
    assert result.text == "обрезано" and result.finish_reason == "length"


def test_local_ctrl_c_keeps_partial_output(monkeypatch):
    events = [{"choices": [{"delta": {"content": "часть"}}]}]
    _local_server(monkeypatch, lambda r: httpx.Response(200, content=_sse(*events, done=False)))

    def interrupt(_chunk):
        raise KeyboardInterrupt

    result = chat_core.stream(_local_config(), [{"role": "user", "content": "q"}], interrupt)
    assert result.text == "часть" and result.truncated is True


def test_sdk_stream_on_reasoning_is_called_only_when_given(monkeypatch):
    events = [_chunk("", reasoning_content="думаю"), _chunk("ответ", finish_reason="stop")]
    _patch_client(monkeypatch, _FakeMistral(stream_events=events))
    thoughts: list[str] = []
    result = chat_core.stream(
        _config(), [{"role": "user", "content": "q"}], lambda c: None, on_reasoning=thoughts.append
    )
    assert thoughts == ["думаю"] and result.reasoning_text == "думаю"
    # Without the callback nothing changes.
    result = chat_core.stream(_config(), [{"role": "user", "content": "q"}], lambda c: None)
    assert result.reasoning_text == "думаю"


def test_without_base_url_the_sdk_path_is_still_used(monkeypatch):
    fake = _FakeMistral(complete_response=_complete_response("sdk", "stop"))
    _patch_client(monkeypatch, fake)

    def _no_http(*a, **k):
        raise AssertionError("без base_url openai_compat не используется")

    monkeypatch.setattr(oc_mod, "chat_complete", _no_http)
    assert chat_core.complete(_config(), [{"role": "user", "content": "q"}]).text == "sdk"


# --- Day 27: fenced JSON verdict and local response_format precedence -------------


def _with_rf(config, rf):
    import dataclasses

    return dataclasses.replace(
        config, params=dataclasses.replace(config.params, response_format=rf)
    )


_FENCED = '```json\n{"a": 1}\n```'


def test_local_fenced_json_is_a_valid_format_verdict(monkeypatch):
    _local_server(monkeypatch, lambda r: httpx.Response(200, json=_completion_json(_FENCED)))
    res = chat_core.complete(_local_config(format="json"), [{"role": "user", "content": "q"}])
    assert res.format_ok is True
    assert res.text == _FENCED


def test_local_unfenced_garbage_is_still_invalid(monkeypatch):
    _local_server(
        monkeypatch, lambda r: httpx.Response(200, json=_completion_json("```json\n{x\n```"))
    )
    res = chat_core.complete(_local_config(format="json"), [{"role": "user", "content": "q"}])
    assert res.format_ok is False


def test_cloud_verdict_on_fenced_json_stays_false():
    assert formats.verify("json", _FENCED).ok is False
    assert formats.verify("json", _FENCED, strip_fence=True).ok is True
    assert formats.verify("json", "```json\n```json\n{}\n```\n```", strip_fence=True).ok is False


def test_local_explicit_response_format_beats_the_json_preset(monkeypatch):
    rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    payload, *_ = chat_core._payload(
        _with_rf(_local_config(format="json"), rf), [{"role": "user", "content": "q"}]
    )
    assert payload["response_format"] == rf


def test_cloud_payload_keeps_the_json_preset_response_format():
    rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    config = _with_rf(
        Config(api_key="k", model="m", params=GenerationParams.build(format="json")), rf
    )
    payload, *_ = chat_core._payload(config, [{"role": "user", "content": "q"}])
    assert payload["response_format"] == {"type": "json_object"}
