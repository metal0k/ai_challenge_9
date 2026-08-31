"""Сборка сообщений, обрезка истории и разбор чанков стрима — без сети."""

from __future__ import annotations

from types import SimpleNamespace

from advent_core.chat import _extract_delta, build_messages, trim_history
from advent_core.client import model_names, resolve_alias
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
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 100}
        for i in range(10)
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


def _chunk(content, *, model=None, usage=None):
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
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
