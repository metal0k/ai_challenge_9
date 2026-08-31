"""Параметры генерации: разбор, границы, отсев по capabilities модели."""

from __future__ import annotations

import pytest

from advent_core.client import capabilities_of, chat_models, find_model
from advent_core.params import GenerationParams, ParamError

REASONING = {"completion_chat": True, "reasoning": True, "vision": True}
NO_REASONING = {"completion_chat": True, "reasoning": False, "completion_fim": True}

MODELS = [
    {
        "id": "mistral-small-2603",
        "aliases": ["mistral-small-latest"],
        "capabilities": REASONING,
        "default_model_temperature": 0.3,
    },
    {"id": "codestral-2508", "aliases": ["codestral-latest"], "capabilities": NO_REASONING},
    {"id": "mistral-embed", "aliases": [], "capabilities": {"completion_chat": False}},
]


def test_unset_params_are_not_sent():
    payload, skipped = GenerationParams().as_payload(REASONING)
    assert payload == {}
    assert skipped == []


def test_set_params_reach_payload():
    params = GenerationParams.build(temperature=0.2, max_tokens=100, top_p=0.9)
    payload, _ = params.as_payload(REASONING)
    assert payload == {"temperature": 0.2, "top_p": 0.9, "max_tokens": 100}


def test_zero_temperature_is_sent_not_dropped():
    """0.0 — осмысленное значение, а не «не задано»."""
    payload, _ = GenerationParams.build(temperature=0).as_payload(REASONING)
    assert payload["temperature"] == 0.0


def test_stop_is_split_on_commas():
    params = GenerationParams.build(stop="КОНЕЦ, ###, ")
    assert params.stop == ["КОНЕЦ", "###"]


@pytest.mark.parametrize(
    ("name", "value"),
    [("temperature", 5), ("temperature", -1), ("top_p", 1.5), ("max_tokens", 0)],
)
def test_out_of_range_is_rejected(name, value):
    with pytest.raises(ParamError):
        GenerationParams.build(**{name: value})


def test_non_numeric_is_rejected():
    with pytest.raises(ParamError):
        GenerationParams.build(max_tokens="много")


def test_unknown_reasoning_effort_is_rejected():
    with pytest.raises(ParamError):
        GenerationParams.build(reasoning_effort="turbo")


def test_reasoning_effort_reaches_capable_model():
    payload, skipped = GenerationParams.build(reasoning_effort="high").as_payload(REASONING)
    assert payload == {"reasoning_effort": "high"}
    assert skipped == []


def test_reasoning_effort_is_skipped_for_incapable_model():
    """codestral не умеет reasoning — параметр не должен уходить в запрос."""
    params = GenerationParams.build(reasoning_effort="high", temperature=0.5)
    payload, skipped = params.as_payload(NO_REASONING)
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0.5
    assert skipped == ["reasoning_effort"]


def test_unknown_capabilities_send_everything():
    """Список моделей мог не загрузиться — тогда не отсеиваем ничего сами."""
    payload, skipped = GenerationParams.build(reasoning_effort="low").as_payload(None)
    assert payload == {"reasoning_effort": "low"}
    assert skipped == []


def test_set_changes_one_param():
    params = GenerationParams()
    assert params.set("temperature", "0.7") == 0.7
    assert params.temperature == 0.7


def test_set_default_resets_param():
    params = GenerationParams.build(temperature=0.7)
    assert params.set("temperature", "default") is None
    assert params.temperature is None


def test_set_rejects_unknown_name():
    with pytest.raises(ParamError):
        GenerationParams().set("magic", "1")


def test_set_keeps_old_value_on_bad_input():
    params = GenerationParams.build(temperature=0.7)
    with pytest.raises(ParamError):
        params.set("temperature", "жарко")
    assert params.temperature == 0.7


def test_describe_lists_every_param():
    rows = GenerationParams.build(temperature=0.2).describe()
    names = [name for name, _, _ in rows]
    assert "temperature" in names and "reasoning_effort" in names
    values = dict((name, value) for name, value, _ in rows)
    assert values["temperature"] == "0.2"
    assert values["top_p"] == "—"


def test_find_model_by_id_and_alias():
    assert find_model(MODELS, "mistral-small-latest")["id"] == "mistral-small-2603"
    assert find_model(MODELS, "mistral-small-2603")["id"] == "mistral-small-2603"
    assert find_model(MODELS, "нет-такой") is None


def test_capabilities_lookup_follows_alias():
    assert capabilities_of(MODELS, "mistral-small-latest")["reasoning"] is True
    assert capabilities_of(MODELS, "codestral-latest")["reasoning"] is False
    assert capabilities_of(MODELS, "нет-такой") is None


def test_chat_models_filter_drops_non_chat():
    ids = [m["id"] for m in chat_models(MODELS)]
    assert "mistral-embed" not in ids
    assert "mistral-small-2603" in ids
