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
from advent_core.chat import _extract_delta, build_messages, should_stream, trim_history
from advent_core.client import model_names, resolve_alias
from advent_core.config import Config
from advent_core.errors import AdventError
from advent_core.params import GenerationParams
from advent_core.telemetry import Usage


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


def _chunk(content, *, model=None, usage=None, finish_reason=None):
    delta = SimpleNamespace(content=content)
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


def _complete_response(text: str, finish_reason: str | None, model: str = "mistral-small-2603"):
    message = SimpleNamespace(content=text)
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
