"""Day 29: profile registry, payload gating of top_k, rerank positional contract, cite text."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console

from advent_core import chat as chat_core
from advent_core import config as config_module
from advent_core.config import LOCAL_API_KEY, Config, ConfigError
from advent_core.errors import AdventError
from advent_core.params import GenerationParams, ParamError
from advent_core.rag import (
    RAG_CITE_INSTRUCTION,
    RAG_CITE_INSTRUCTION_LOCAL,
    RagHit,
    build_cite_prompt,
    build_rerank_prompt,
    parse_rerank,
)
from week_05.rag import LOCAL_RERANK_RESPONSE_FORMAT, local_rerank_positional_format
from week_06 import profiles

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.setattr(config_module, "load_env", lambda: None)


# --- registry -------------------------------------------------------------------------------


def test_baseline_is_the_day_28_setup_as_literals():
    b = profiles.PROFILES["baseline"]
    assert (b.quant, b.context) == ("Q4_K_M", 40960)
    assert (b.answer_temperature, b.answer_top_p, b.answer_top_k) == (None, None, None)
    assert (b.answer_max_tokens, b.answer_reasoning, b.k_before) == (4096, True, 20)
    assert (b.rerank_format, b.cite_prompt) == ("objects", "default")
    assert (b.gguf, b.publisher, b.file_gb) == ("Ornith-1.5-9B-Q4_K_M.gguf", "ornith-ai", 5.78)


def test_registry_holds_baseline_and_ten_screening_profiles_and_no_tuned():
    assert list(profiles.PROFILES) == [
        "baseline",
        "sampling",
        "cap",
        "noreason",
        "ctx24k",
        "positional",
        "k12",
        "citelocal",
        "q4b",
        "q3",
        "q5",
    ]
    assert "tuned" not in profiles.PROFILES


@pytest.mark.parametrize(
    ("name", "changed"),
    [
        ("sampling", {"answer_temperature", "answer_top_p", "answer_top_k"}),
        ("cap", {"answer_max_tokens"}),
        ("noreason", {"answer_reasoning"}),
        ("ctx24k", {"context"}),
        ("positional", {"rerank_format"}),
        ("k12", {"k_before"}),
        ("citelocal", {"cite_prompt"}),
        ("q4b", {"publisher", "file_gb"}),
        ("q3", {"quant", "gguf", "publisher", "file_gb"}),
        ("q5", {"quant", "gguf", "publisher", "file_gb"}),
    ],
)
def test_each_screening_profile_differs_from_baseline_in_exactly_the_declared_fields(name, changed):
    assert set(profiles.changed_fields(profiles.PROFILES[name])) == changed


def test_screening_values_are_the_declared_literals():
    p = profiles.PROFILES
    assert (p["sampling"].answer_temperature, p["sampling"].answer_top_p) == (0.6, 0.95)
    assert p["sampling"].answer_top_k == 20
    assert p["cap"].answer_max_tokens == 1536 and p["noreason"].answer_reasoning is False
    assert p["ctx24k"].context == 24576 and p["k12"].k_before == 12
    assert p["positional"].rerank_format == "positional" and p["citelocal"].cite_prompt == "local"
    assert (p["q4b"].publisher, p["q4b"].file_gb, p["q4b"].quant) == ("bartowski", 5.91, "Q4_K_M")
    assert (p["q3"].quant, p["q3"].gguf, p["q3"].file_gb) == (
        "Q3_K_M",
        "Ornith-1.5-9B-Q3_K_M.gguf",
        4.92,
    )
    assert (p["q5"].quant, p["q5"].gguf, p["q5"].file_gb) == (
        "Q5_K_M",
        "Ornith-1.5-9B-Q5_K_M.gguf",
        6.85,
    )


def test_profiles_are_frozen():
    with pytest.raises(AttributeError):
        profiles.BASELINE.k_before = 1  # type: ignore[misc]


def test_unknown_profile_is_a_config_error_naming_the_known_ones():
    with pytest.raises(ConfigError) as info:
        profiles.get_profile("tuned")
    assert "baseline" in str(info.value) and "citelocal" in str(info.value)


def test_quant_row_that_moved_the_context_is_labelled_quant_plus_ctx():
    import dataclasses

    assert profiles.axis_label(profiles.PROFILES["q5"]) == "квант"
    moved = dataclasses.replace(profiles.PROFILES["q5"], context=32768)
    assert profiles.axis_label(moved) == "квант+ctx"
    assert profiles.axis_label(profiles.PROFILES["cap"]) == "—"


# --- profiles table -------------------------------------------------------------------------


def _render(tables, width=80) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=width, no_color=True, force_terminal=False)
    for table in tables:
        console.print(table)
    return buf.getvalue()


def test_profiles_table_rows_are_fields_columns_are_profiles_and_differences_are_marked():
    text = _render(profiles.profiles_tables(["baseline", "sampling", "cap"]))
    assert text.count("Профили локального RAG") == 1
    header = next(line for line in text.splitlines() if "поле" in line)
    assert "baseline" in header and "sampling" in header and "cap" in header
    top_k = next(line for line in text.splitlines() if line.lstrip("│ ").startswith("top_k"))
    assert "» 20" in top_k and "сервер" in top_k
    cap = next(line for line in text.splitlines() if "max_tokens ответа" in line)
    assert "» 1536" in cap and "4096" in cap
    assert text.count("» 4096") == 0  # the baseline column is never marked
    assert "…" not in text and max(cell_len(line) for line in text.splitlines()) <= 80


def test_all_profiles_split_into_tables_that_fit_80_columns():
    tables = profiles.profiles_tables(None)
    assert len(tables) == 4  # ten non-baseline profiles, three per table
    text = _render(tables)
    assert text.count("Профили локального RAG") == 4
    assert max(cell_len(line) for line in text.splitlines()) <= 80 and "…" not in text


def test_profiles_table_for_an_unknown_name_is_a_config_error():
    with pytest.raises(ConfigError):
        profiles.profiles_tables(["baseline", "nope"])


# --- top_k only goes to a local server --------------------------------------------------------


def _config(local: bool, **params) -> Config:
    return Config(
        api_key=LOCAL_API_KEY if local else "k" * 12,
        model="ornith" if local else "ministral-14b-latest",
        params=GenerationParams.build(**params),
        stream=False,
        base_url="http://127.0.0.1:1234" if local else None,
        offline=local,
    )


def test_top_k_reaches_a_local_payload():
    payload, _, _, _ = chat_core._payload(
        _config(True, top_k=20, temperature=0.6), [{"role": "user", "content": "x"}]
    )
    assert payload["top_k"] == 20 and payload["temperature"] == 0.6


def test_top_k_never_reaches_a_cloud_payload():
    payload, skipped, _, _ = chat_core._payload(
        _config(False, top_k=20, temperature=0.6), [{"role": "user", "content": "x"}]
    )
    assert "top_k" not in payload and payload["temperature"] == 0.6
    assert "top_k" not in skipped  # dropped silently, it is not a capability gap


def test_as_payload_gates_top_k_by_the_local_server_flag():
    params = GenerationParams.build(top_k=5)
    assert params.as_payload(None, local_server=True)[0] == {"top_k": 5}
    assert params.as_payload(None, local_server=False)[0] == {}
    assert params.as_payload(None)[0] == {}


def test_top_k_is_validated_as_a_positive_int():
    with pytest.raises(ParamError):
        GenerationParams.build(top_k=0)


def test_cite_prompt_and_rerank_format_are_local_params_never_sent():
    params = GenerationParams.build(rag_cite_prompt="local", rag_rerank_format="positional")
    assert params.as_payload(None, local_server=True)[0] == {}
    with pytest.raises(ParamError):
        GenerationParams.build(rag_cite_prompt="shiny")


# --- rerank positional ----------------------------------------------------------------------


def test_positional_schema_pins_the_length_to_n_and_scores_to_0_10():
    schema = local_rerank_positional_format(7)["json_schema"]["schema"]["properties"]["scores"]
    assert schema["minItems"] == 7 and schema["maxItems"] == 7
    assert schema["items"] == {"type": "integer", "minimum": 0, "maximum": 10}
    assert local_rerank_positional_format(3)["json_schema"]["strict"] is True


def test_object_schema_is_unchanged():
    items = LOCAL_RERANK_RESPONSE_FORMAT["json_schema"]["schema"]["properties"]["scores"]["items"]
    assert set(items["properties"]) == {"id", "score"}


def _hit(n):
    return RagHit(f"c{n}", "a.md", "S", 0.5, f"text {n}")


def test_positional_prompt_has_its_own_instruction_and_tail():
    hits = [_hit(1), _hit(2), _hit(3)]
    text = build_rerank_prompt("q?", hits, local=True, positional=True)
    assert "оценка фрагмента 1, оценка фрагмента 2" in text
    assert text.rstrip().endswith("без пояснений и без текста вокруг.")
    assert "ровно 3 целых чисел" in text
    assert '{"id": 1, "score": 0}' not in text and '"id": <номер>' not in text


# Literals rendered by the Day 28 code (HEAD before Day 29) for the same two hits.
HEAD_RERANK_OBJECTS = 'Оцени, насколько каждый фрагмент помогает ответить на вопрос. Шкала 0–10: 10 — фрагмент содержит прямой ответ, 0 — не относится. Верни JSON {"scores": [{"id": <номер>, "score": <0-10>}, ...]} для всех 2 фрагментов без пропусков.\n\nВопрос: вопрос?\n\n[1] a.md — Раздел\nтекст один\n\n[2] b.md\nтекст два\n\nВопрос: вопрос?'  # noqa: E501 - verbatim Day 28 output
HEAD_RERANK_OBJECTS_LOCAL = 'Оцени, насколько каждый фрагмент помогает ответить на вопрос. Шкала 0–10: 10 — фрагмент содержит прямой ответ, 0 — не относится. Верни JSON {"scores": [{"id": <номер>, "score": <0-10>}, ...]} для всех 2 фрагментов без пропусков.\n\nВопрос: вопрос?\n\n[1] a.md — Раздел\nтекст один\n\n[2] b.md\nтекст два\n\nВопрос: вопрос?\n\nОтветь ТОЛЬКО JSON-объектом вида {"scores": [{"id": 1, "score": 0}, ...]} — по одной записи на каждый из 2 фрагментов, без пояснений и без текста вокруг.'  # noqa: E501 - verbatim Day 28 output
HEAD_CITE = 'Ответь только по фрагментам документации ниже. Верни JSON {"status": "answer" или "unknown", "answer": "...", "sources": [номера фрагментов], "quotes": [{"id": номер фрагмента, "text": "дословная цитата"}]}. Цитата — точная копия куска фрагмента, 1–2 предложения, без пересказа и без форматирования. Поле «answer» — всегда ОДНА строка, даже если пользователь просит шаги или список: шаги пиши строками внутри этой строки, не списком и не объектом. Каждое утверждение ответа подтверждается хотя бы одной цитатой. Если во фрагментах ответа нет — "status": "unknown", остальные поля пустые.\n\nВопрос: вопрос?\n\n[1] a.md — Раздел\nтекст один\n\n[2] b.md\nтекст два\n\nВопрос: вопрос?'  # noqa: E501 - verbatim Day 28 output


def _two_hits():
    return [
        RagHit("c1", "a.md", "Раздел", 0.5, "текст один"),
        RagHit("c2", "b.md", "", 0.4, "текст два"),
    ]


def test_unprofiled_rerank_prompt_is_the_exact_day_28_text():
    assert build_rerank_prompt("вопрос?", _two_hits()) == HEAD_RERANK_OBJECTS
    assert build_rerank_prompt("вопрос?", _two_hits(), local=True) == HEAD_RERANK_OBJECTS_LOCAL


def test_unprofiled_cite_prompt_is_the_exact_day_28_text():
    assert build_cite_prompt("вопрос?", _two_hits()) == HEAD_CITE


def test_parse_positional_list_maps_position_to_fragment():
    assert parse_rerank('{"scores": [7, 0, 10]}', 3) == {1: 7.0, 2: 0.0, 3: 10.0}


@pytest.mark.parametrize(
    "raw",
    [
        '{"scores": [7, 0]}',  # too short
        '{"scores": [7, 0, 10, 1]}',  # too long
        '{"scores": [7, true, 10]}',  # bool is not an int
        '{"scores": [7, "0", 10]}',  # mixed types
        '{"scores": [7, 11, 10]}',  # out of range
        '{"scores": [7, -1, 10]}',
        '{"scores": [7, 0.5, 10]}',  # float
        '{"scores": [7.0, 0, 10]}',
    ],
)
def test_positional_list_with_a_wrong_length_or_value_is_a_format_error(raw):
    with pytest.raises(AdventError):
        parse_rerank(raw, 3)


def test_object_format_still_accepts_partial_scores():
    assert parse_rerank('{"scores": [{"id": 2, "score": 6}]}', 3) == {2: 6.0}


def test_empty_list_is_still_an_error():
    with pytest.raises(AdventError):
        parse_rerank('{"scores": []}', 3)


# --- cite instruction -----------------------------------------------------------------------


def test_cite_prompt_uses_the_given_instruction_and_defaults_to_the_shared_one():
    hits = [_hit(1)]
    assert build_cite_prompt("q?", hits).startswith(RAG_CITE_INSTRUCTION)
    local = build_cite_prompt("q?", hits, instruction=RAG_CITE_INSTRUCTION_LOCAL)
    assert local.startswith(RAG_CITE_INSTRUCTION_LOCAL) and RAG_CITE_INSTRUCTION not in local


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-zа-яё0-9_.]{4,}", text.lower()))


def test_local_cite_instruction_borrows_no_words_from_the_control_questions():
    texts: list[str] = []
    for name in ("rag_questions.json", "rag_unanswerable.json"):
        for q in json.loads((ROOT / "week_05" / name).read_text(encoding="utf-8")):
            texts.append(q["question"])
            texts += [alt for alternatives in q.get("expect") or [] for alt in alternatives]
    json_keys = {"answer"}  # the JSON key the grammar fixes; it cannot be reworded
    shared = _words(RAG_CITE_INSTRUCTION_LOCAL) & set().union(*(_words(t) for t in texts))
    assert shared - json_keys == set()


def test_local_cite_instruction_keeps_the_contract_of_the_grammar():
    text = RAG_CITE_INSTRUCTION_LOCAL
    for needle in ('"status"', '"answer"', '"sources"', '"quotes"', '"unknown"', "не больше двух"):
        assert needle in text
    assert len(text) < len(RAG_CITE_INSTRUCTION)


# --- review fixes: format-aware parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"scores": [7, {"id": 2, "score": 9}, 10]}',  # mixed array
        '{"scores": [{"id": 1, "score": 7}, {"id": 2, "score": 9}, {"id": 3, "score": 1}]}',
        '{"scores": []}',
        '{"scores": [7, 8, 9, 10]}',
        '{"scores": [7, 8, null]}',
    ],
)
def test_when_positional_is_expected_nothing_but_n_ints_is_accepted(raw):
    with pytest.raises(AdventError):
        parse_rerank(raw, 3, positional=True)


def test_the_same_mixed_array_stays_tolerant_when_the_object_format_is_expected():
    assert parse_rerank('{"scores": [7, {"id": 2, "score": 9}, 10]}', 3) == {2: 9.0}


def test_positional_expected_accepts_the_exact_contract():
    assert parse_rerank('{"scores": [7, 0, 10]}', 3, positional=True) == {1: 7.0, 2: 0.0, 3: 10.0}
