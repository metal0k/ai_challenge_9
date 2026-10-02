"""Day 25 review fixes: list answers in cite JSON, key normalisation, rewrite referent, prompts."""

from __future__ import annotations

import json

import pytest

from advent_core.memory import (
    MemoryDelta,
    MemoryOperation,
    MemorySnapshot,
    apply_delta,
    manual_set,
)
from advent_core.rag import (
    RAG_LAST_ANSWER_MAX,
    RagHit,
    RagSettings,
    build_rewrite_prompt,
    parse_cited,
)
from advent_core.scenario import ScenarioState, run_turn, salvage_pending_memory
from tools import bench_core
from week_05 import chat_eval
from week_05 import rag as rag_module

HITS = [RagHit("c1", "CLAUDE.md", "OBS", 0.9, "verify_capture проверяет кадр до записи.")]
QUOTE = "verify_capture проверяет кадр"


def _cited(answer: object) -> str:
    return json.dumps(
        {
            "status": "answer",
            "answer": answer,
            "sources": [1],
            "quotes": [{"id": 1, "text": QUOTE}],
        },
        ensure_ascii=False,
    )


# --- 6: a list of strings is an answer ----------------------------------------------------


def test_list_answer_is_joined_as_numbered_lines():
    c = parse_cited(_cited(["Открой OBS", " Запусти advent record "]), HITS)

    assert c.status == "answer"
    assert c.answer == "1. Открой OBS\n2. Запусти advent record"
    assert c.quoted is True


@pytest.mark.parametrize(
    "answer",
    [
        [],
        [""],
        ["шаг", "  "],
        ["шаг", 2],
        ["шаг", None],
        [["вложенный"]],
        [{"a": 1}],
        7,
        {},
        {"a": [1]},
        {"a": {"b": 1}},
        {"a": True},
        {"a": ""},
        {" ": "x"},
        {"a": None},
    ],
)
def test_other_answer_shapes_stay_a_format_error(answer):
    assert parse_cited(_cited(answer), HITS).reason == "bad_json"


def test_answer_object_with_scalar_values_becomes_key_value_lines():
    c = parse_cited(
        _cited({"keep_last": 6, "compact_every": 10, "ratio": 0.5, "mode": " x "}), HITS
    )

    assert c.status == "answer"
    assert c.answer == "keep_last: 6\ncompact_every: 10\nratio: 0.5\nmode: x"
    assert c.quoted is True


def test_cite_instruction_says_answer_is_always_one_string():
    from advent_core.rag import RAG_CITE_INSTRUCTION

    assert "«answer» — всегда ОДНА строка" in RAG_CITE_INSTRUCTION
    assert "шаги пиши строками внутри этой строки" in RAG_CITE_INSTRUCTION


# --- 7: automatic keys are normalised, manual ones are not --------------------------------


def test_automatic_key_with_whitespace_is_normalised_to_underscores():
    op = MemoryOperation("set", "terms", "мёртвое  время", "паузы в записи", "мёртвое время")
    msgs = ({"role": "user", "content": "мёртвое время — это паузы"},)

    update = apply_delta(MemorySnapshot(), MemoryDelta(working=(op,)), user_messages=msgs)

    assert update.rejected == ()
    assert update.snapshot.working.entries == {"terms.мёртвое_время": "паузы в записи"}


def test_manual_key_with_whitespace_is_still_refused():
    with pytest.raises(ValueError, match="without whitespace"):
        manual_set(MemorySnapshot(), "working", "terms.мёртвое время", "паузы")


# --- 8: the previous answer travels to the rewrite prompt ---------------------------------


def test_last_answer_is_capped_and_blank_becomes_none():
    long = RagSettings("s", 5, 20, True, True, 5.0, last_answer="я" * 900)
    assert long.last_answer == "я" * RAG_LAST_ANSWER_MAX == "я" * 300
    assert RagSettings("s", 5, 20, True, True, 5.0, last_answer="  ").last_answer is None
    assert RagSettings("s", 5, 20, True, True, 5.0).last_answer is None


def test_rewrite_prompt_shows_the_previous_answer_after_the_previous_question():
    prompt = build_rewrite_prompt(
        "А какой --min-freeze тогда брать?",
        task="Цель main: g",
        last_question="А почему нельзя через select?",
        last_answer="Потому что trim/concat режут обе дорожки.",
    )

    assert prompt.endswith(
        "Предыдущий вопрос: А почему нельзя через select?"
        "\n\nОтвет на него (начало): Потому что trim/concat режут обе дорожки."
        "\n\nТекущий вопрос: А какой --min-freeze тогда брать?"
    )


def test_rewrite_prompt_without_an_answer_has_no_answer_block():
    prompt = build_rewrite_prompt("q", last_question="L")
    assert "Ответ на него" not in prompt


def test_rewrite_instruction_keeps_context_out_of_the_query():
    prompt = build_rewrite_prompt("q", task="Цель main: g")

    assert "ТОЛЬКО чтобы понять, к чему относится текущий вопрос" in prompt
    assert "Не копируй в запрос окружение, ограничения, термины и цель" in prompt


def test_prompt_without_any_context_is_the_day_23_prompt():
    assert rag_module.rewrite_prompt("вопрос", None) == rag_module.RAG_REWRITE_PROMPT.replace(
        "{question}", "вопрос"
    )


# --- 9 / 10: prompt wording ---------------------------------------------------------------


def test_extractor_prompt_forbids_storing_questions():
    from advent_core.config import PROJECT_ROOT

    text = (PROJECT_ROOT / "advent_core" / "prompts" / "memory.md").read_text(encoding="utf-8")

    assert "Никогда не сохраняй сам вопрос пользователя" in text
    assert "open_items — только то, что пользователь прямо назвал нерешённым" in text
    assert "Если сообщение — просто вопрос, верни пустые set и delete" in text


# --- 1: the carry lives in advent_core, benches re-use it ---------------------------------


def test_bench_core_reuses_the_advent_core_scenario_pieces():
    assert bench_core.ScenarioState is ScenarioState
    assert bench_core.run_turn is run_turn
    assert bench_core.salvage_pending_memory is salvage_pending_memory


def test_chat_eval_does_not_import_the_unpackaged_tools_package():
    import inspect

    source = inspect.getsource(chat_eval)

    assert "from tools" not in source and "import tools" not in source
