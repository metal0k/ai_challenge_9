"""Параметры генерации: разбор, границы, отсев по capabilities модели."""

from __future__ import annotations

import pytest

from advent_core.client import capabilities_of, chat_models, find_model
from advent_core.params import (
    CHAT_COMMAND,
    SOLVE_COMMAND,
    STRATEGIES,
    STRATEGY_CHOICES,
    GenerationParams,
    ParamError,
    defaults_for,
)

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


def test_as_payload_never_sends_local_params():
    """format/schema_file/done/mode/max_turns управляют CLI, не уходят на сервер.

    Локальный параметр не должен попасть ни в payload, ни в skipped —
    skipped значит «сервер бы отклонил», а тут сервер вообще не видит имени.
    """
    params = GenerationParams.build(
        temperature=0.5,
        format="json",
        schema_file="week_01/schemas/ingredients.json",
        done="text:[ГОТОВО]",
        mode="dialog",
        max_turns=5,
    )
    payload, skipped = params.as_payload(REASONING)
    assert payload == {"temperature": 0.5}
    assert skipped == []


def test_max_turns_defaults_to_ten_not_none():
    """Дефолт диалога — содержательное значение 10, а не «сервер решит»."""
    assert GenerationParams().max_turns == 10


def test_set_max_turns_default_resets_to_none():
    """Потребитель (цикл mode=dialog) обязан трактовать None как 10 — это его дело."""
    params = GenerationParams()
    assert params.set("max_turns", "default") is None
    assert params.max_turns is None


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


# --- Локальные параметры дня 03: strategy/problem/runs/judge/judge_model ---


def test_day03_params_never_reach_the_request():
    """Пять параметров дня 03 управляют CLI, на сервер не уходят вовсе.

    Как и у дня 02: локальный параметр не должен попасть ни в payload, ни в
    skipped — skipped значит «сервер бы отклонил», а тут сервер его не видит.
    """
    params = GenerationParams.build(
        temperature=0.5,
        strategy="panel",
        problem="children",
        runs=3,
        judge=True,
        judge_model="ministral-8b-latest",
    )
    payload, skipped = params.as_payload(REASONING)
    assert payload == {"temperature": 0.5}
    assert skipped == []


def test_unknown_strategy_is_rejected():
    with pytest.raises(ParamError):
        GenerationParams.build(strategy="телепатия")


def test_all_is_a_valid_strategy_but_not_a_strategy_name():
    """`all` — значение флага, а не пятый способ рассуждения."""
    assert GenerationParams.build(strategy="all").strategy == "all"
    assert "all" not in STRATEGIES
    assert list(STRATEGY_CHOICES) == [*STRATEGIES, "all"]


def test_runs_below_one_is_rejected():
    with pytest.raises(ParamError):
        GenerationParams.build(runs=0)


@pytest.mark.parametrize("raw", ["вкл", "да", "true", "1", "yes", "on", True])
def test_bool_param_accepts_both_languages_true(raw):
    assert GenerationParams.build(judge=raw).judge is True


@pytest.mark.parametrize("raw", ["выкл", "нет", "false", "0", "no", "off", False])
def test_bool_param_accepts_both_languages_false(raw):
    assert GenerationParams.build(judge=raw).judge is False


def test_bool_param_rejects_nonsense():
    with pytest.raises(ParamError):
        GenerationParams.build(judge="наверное")


def test_bool_param_is_shown_as_on_off_not_as_python_literal():
    """`/params` с "False" читалось бы как значение, а не как выключенный флаг."""

    def shown(value):
        rows = GenerationParams(judge=value).describe()
        return dict((name, cell) for name, cell, _ in rows)["judge"]

    assert shown(False) == "выкл"
    assert shown(True) == "вкл"
    assert shown(None) == "—"


# --- Умолчания различаются у chat и solve (SPEC-w01d03.md §4) ---


def test_defaults_differ_between_chat_and_solve():
    assert defaults_for(CHAT_COMMAND)["strategy"] == "direct"
    assert defaults_for(SOLVE_COMMAND)["strategy"] == "all"
    assert defaults_for(CHAT_COMMAND)["judge"] is False
    assert defaults_for(SOLVE_COMMAND)["judge"] is True
    assert defaults_for(SOLVE_COMMAND)["runs"] == 1


def test_apply_defaults_fills_only_unset_values():
    params = GenerationParams.build(strategy="panel")
    params.apply_defaults(SOLVE_COMMAND)
    assert params.strategy == "panel", "заданное пользователем не перетирается"
    assert params.runs == 1
    assert params.judge is True


def test_apply_defaults_does_not_flip_an_explicit_false():
    """--no-judge даёт False, а не None: проверка на None, а не на ложность."""
    params = GenerationParams.build(judge=False)
    params.apply_defaults(SOLVE_COMMAND)
    assert params.judge is False


def test_apply_defaults_for_chat_keeps_solve_only_params_unset():
    """problem/runs/judge_model читает только solve — в chat им браться неоткуда."""
    params = GenerationParams()
    params.apply_defaults(CHAT_COMMAND)
    assert params.strategy == "direct"
    assert params.judge is False
    assert params.runs is None
    assert params.problem is None
    assert params.judge_model is None


def test_set_strategy_default_returns_none_and_command_default_is_reapplied():
    """`/set strategy default` — «верни умолчание команды», а не «пусто навсегда»."""
    params = GenerationParams.build(strategy="panel")
    assert params.set("strategy", "default") is None

    params.apply_defaults(CHAT_COMMAND)
    assert params.strategy == "direct"
