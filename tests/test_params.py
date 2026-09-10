"""Параметры генерации: разбор, границы, отсев по capabilities модели."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from advent_core.client import capabilities_of, chat_models, find_model
from advent_core.params import (
    AGENT_PARAMS,
    BENCH_COMMAND,
    CHAT_COMMAND,
    SOLVE_COMMAND,
    SPECS,
    STRATEGIES,
    STRATEGY_CHOICES,
    GenerationParams,
    ParamError,
    defaults_for,
)
from week_01 import cli as w01_cli

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


# --------------------------------------------------------------------------
# kind="models" — лестница моделей для bench (Day 05)
# --------------------------------------------------------------------------


def test_models_param_splits_on_commas_and_trims_spaces():
    params = GenerationParams.build(models=" ministral-3b-latest , ministral-14b-latest ")

    assert params.models == ["ministral-3b-latest", "ministral-14b-latest"]


def test_models_param_drops_empty_items_between_commas():
    """`--models a,,b` — опечатка, а не запрос пустой модели: пустое имя
    улетело бы в API как model="" и вернулось 400 посреди развёртки."""
    params = GenerationParams.build(models="ministral-3b-latest,,ministral-8b-latest,")

    assert params.models == ["ministral-3b-latest", "ministral-8b-latest"]


@pytest.mark.parametrize("raw", ["", "   ", ",", " , , "])
def test_models_param_rejects_an_empty_ladder(raw):
    """Пустой список — ParamError, а НЕ None. None утонул бы в apply_defaults()
    и молча подменился дефолтной лестницей там, где пользователь явно задал
    --models: тихая подмена явного выбора хуже отказа."""
    with pytest.raises(ParamError):
        GenerationParams.build(models=raw)


def test_models_param_accepts_a_ready_list():
    """Дефолт реестра приходит уже списком, и он обязан проходить тем же путём."""
    params = GenerationParams.build(models=["ministral-3b-latest"])

    assert params.models == ["ministral-3b-latest"]


def test_bench_default_ladder_comes_from_the_registry_not_from_the_cli():
    """Умолчание --models и --runs живёт в реестре: иначе одно и то же число
    записано дважды и однажды разъедется (CLAUDE.md)."""
    defaults = defaults_for(BENCH_COMMAND)

    assert defaults["models"] == [
        "ministral-3b-latest",
        "ministral-8b-latest",
        "ministral-14b-latest",
    ]
    assert defaults["runs"] == 3


# --------------------------------------------------------------------------
# kind="int" context_limit — override лимита окна (Day 08, SPEC-w02d08.md §4)
# --------------------------------------------------------------------------


def test_context_limit_parses_and_validates():
    params = GenerationParams.build(context_limit="2500")
    assert params.context_limit == 2500


@pytest.mark.parametrize("raw", [0, -5, "0", "-5"])
def test_context_limit_out_of_range_is_rejected(raw):
    """Окно меньше одного токена — опечатка, а не «окно ноль»."""
    with pytest.raises(ParamError):
        GenerationParams.build(context_limit=raw)


def test_context_limit_defaults_to_none_and_set_default_restores_none():
    """None — «из карточки модели»: дефолт команды не подменяет его."""
    params = GenerationParams()
    assert params.context_limit is None
    params.set("context_limit", "2500")
    assert params.context_limit == 2500
    assert params.set("context_limit", "default") is None
    assert params.context_limit is None


def test_context_limit_is_local_and_never_reaches_payload():
    """Клиентский override окна серверу не нужен: сервер знает только своё."""
    params = GenerationParams.build(context_limit=2500, temperature=0.5)
    payload, skipped = params.as_payload(REASONING)
    assert payload == {"temperature": 0.5}
    assert skipped == []


def test_context_limit_is_among_agent_params():
    """Без имени в AGENT_PARAMS /set context_limit в агенте отвергся бы
    как «не читается» — а агент читает его при trim."""
    assert "context_limit" in AGENT_PARAMS


def test_bench_default_ladder_is_copied_not_shared():
    """Общий мутируемый дефолт: правка списка в одном прогоне не имеет права
    просочиться в следующий. Именно поэтому apply_defaults копирует list."""
    first = GenerationParams.build()
    first.apply_defaults(BENCH_COMMAND)
    first.models.append("codestral-latest")

    second = GenerationParams.build()
    second.apply_defaults(BENCH_COMMAND)

    assert "codestral-latest" not in second.models


# --------------------------------------------------------------------------
# NON_CHAT_PARAMS (week_01/cli.py) — параметр реестра, который REPL недели 01
# молча принимает и ни на что не тратит, должен либо реально читаться кодом
# недели 01, либо быть в этом списке (находка ревью, повторившаяся трижды:
# problem/runs, потом session, потом context_limit).
# --------------------------------------------------------------------------


def test_non_chat_params_names_are_known_local_specs():
    """Запись в NON_CHAT_PARAMS про несуществующее или НЕ-локальное имя —
    мёртвый груз: API-параметры (temperature и т.п.) уходят в payload целиком
    и не нуждаются в этом предупреждении вовсе."""
    local_names = {spec.name for spec in SPECS if spec.local}
    unknown = set(w01_cli.NON_CHAT_PARAMS) - local_names
    assert unknown == set()


def test_every_local_param_is_read_by_week01_or_listed_in_non_chat_params():
    """Замыкает дыру, из-за которой context_limit (день 08) тихо принимался
    REPL'ом недели 01 без предупреждения — та же дыра, что раньше была у
    session и до него у problem/runs.

    «Читается» проверяется не по второй ручной копии NON_CHAT_PARAMS (тогда
    тест был бы тавтологией и не мог покраснеть), а по исходнику
    week_01/cli.py: ищем обращение вида `params.<имя>`, которым REPL реально
    трогает session.config.params.<имя>. Не покрывает не-локальные параметры
    (temperature, top_p, …) — они уходят в payload целиком через
    GenerationParams.as_payload(), а не по отдельному имени, так что для них
    этой проверки не нужно и NON_CHAT_PARAMS про них не заводится.
    """
    source = Path(w01_cli.__file__).read_text(encoding="utf-8")
    missing = [
        spec.name
        for spec in SPECS
        if spec.local
        and spec.name not in w01_cli.NON_CHAT_PARAMS
        and re.search(rf"\bparams\.{re.escape(spec.name)}\b", source) is None
    ]
    assert missing == []


# --- history compaction: three Day 09 params -------------------------------


def test_compaction_params_never_reach_the_payload():
    """compact/keep_last/compact_every control the agent, the server never sees them.

    Not in payload, not in skipped: skipped means "the model would reject it",
    but here the name never reaches the server at all.
    """
    params = GenerationParams.build(compact=False, keep_last=4, compact_every=8)
    payload, skipped = params.as_payload(REASONING)
    assert payload == {}
    assert skipped == []


def test_compaction_params_are_parsed_into_their_types():
    params = GenerationParams.build(compact=True, keep_last=4, compact_every=8)
    assert params.compact is True
    assert params.keep_last == 4
    assert params.compact_every == 8


@pytest.mark.parametrize("bad", [{"keep_last": 1}, {"keep_last": 0}, {"compact_every": 1}])
def test_compaction_bounds_are_enforced(bad):
    """A user+assistant pair is indivisible: keep_last=1 would leave half an
    exchange, and compact_every=1 would compact after every single turn."""
    with pytest.raises(ParamError):
        GenerationParams.build(**bad)
