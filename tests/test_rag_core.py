import pytest

from advent_core.errors import AdventError
from advent_core.rag import (
    RAG_EMPTY_INSTRUCTION,
    RAG_INSTRUCTION,
    RAG_REWRITE_PROMPT,
    RagContext,
    RagHit,
    apply_rerank,
    build_rag_prompt,
    build_rerank_prompt,
    check_facts,
    cited_sources,
    clean_rewrite,
    fact_span,
    fit_hits,
    normalize,
    parse_rerank,
    rrf_merge,
)


def test_fact_span_covers_the_digit_gap_in_the_original():
    text = "Окно 262 144 токена"
    span = fact_span(text, ["262144"])
    assert span == (5, 12)
    assert text[span[0] : span[1]] == "262 144"


def test_fact_span_accepts_nbsp_and_narrow_nbsp_between_digits():
    assert fact_span("окно 262 144 и 8 192", ["262144"]) == (5, 12)
    assert fact_span("окно 8 192", ["8192"]) == (5, 10)


def test_fact_span_is_case_insensitive():
    text = "см. Alice тут"
    assert fact_span(text, ["alice"]) == (4, 9)


def test_fact_span_refuses_a_longer_number():
    assert fact_span("максимум 11.5", ["1.5"]) is None
    assert fact_span("размерность 10240", ["1024"]) is None
    assert fact_span("Потолок — 1.5.", ["1.5"]) == (10, 13)


def test_fact_span_refuses_a_number_glued_across_whitespace():
    assert fact_span("1 262 144", ["262144"]) is None
    assert fact_span("262144 5", ["262144"]) is None


def test_fact_span_takes_the_first_alternative_in_order_not_the_first_in_text():
    text = "сначала beta, потом alpha"
    assert fact_span(text, ["alpha", "beta"]) == (20, 25)
    assert fact_span(text, ["beta", "alpha"]) == (8, 12)


def test_fact_span_none_when_absent_or_empty():
    assert fact_span("ничего", ["262144", "w01d01"]) is None
    assert fact_span("ничего", ["", "ни"]) == (0, 2)
    assert fact_span("ничего", [""]) is None
    assert fact_span("ничего", []) is None


def test_fact_span_agrees_with_check_facts():
    cases = [
        ("окно 262 144 токена", ["262144"]),
        ("окно 262144", ["262 144"]),
        ("максимум 11.5", ["1.5"]),
        ("потолок 1.5.", ["1.5"]),
        ("версия 1.55", ["1.5"]),
        ("1 262 144", ["262144"]),
        ("262144 5", ["262144"]),
        ("ТЕГ W01D01", ["w01d01"]),
        ("ничего", ["w01d01", "x"]),
        ("(1024) и 8192!", ["8192", "1024"]),
        ("размерность 10240", ["1024"]),
        ("окно 262 144", ["262144"]),
    ]
    for answer, alternatives in cases:
        assert (fact_span(answer, alternatives) is not None) == check_facts(answer, [alternatives])[
            0
        ], (answer, alternatives)


def _hit(
    text: str = "текст",
    *,
    source: str = "CLAUDE.md",
    section: str = "Раздел",
    chunk_id: str = "structure:CLAUDE.md#1",
    score: float = 0.5,
) -> RagHit:
    return RagHit(chunk_id=chunk_id, source=source, section=section, score=score, text=text)


def test_prompt_without_hits_is_the_question():
    assert build_rag_prompt("Что это?", []) == "Что это?"


def test_prompt_exact_text():
    hits = [
        _hit("  первый текст \n", source="a.md", section="Раздел A"),
        _hit("второй", source="b.py", section="  "),
    ]
    result = build_rag_prompt("Сколько?", hits)
    assert result.startswith(RAG_INSTRUCTION)
    assert result.endswith(
        "\n\n[1] a.md — Раздел A\nпервый текст\n\n[2] b.py\nвторой\n\nВопрос: Сколько?"
    )
    assert result.count("Вопрос:") == 1


def test_prompt_instruction_literal():
    result = build_rag_prompt("Что?", [_hit("текст")])
    assert result.startswith("Ответь на вопрос по фрагментам документации проекта ниже.")


def test_normalize_joins_digit_groups():
    assert normalize("Окно 262 144 токена") == "окно 262144 токена"
    assert normalize("262 144") == "262144"
    assert normalize("262\u00a0144") == "262144"
    assert normalize("a 1 b") == "a 1 b"


def test_check_facts_alternatives():
    assert check_facts(
        "Тег W01D01, окно 262 144.", [["w01d01", "wNNdDD"], ["262144"], ["alice"]]
    ) == (True, True, False)


def test_check_facts_empty_alternative_never_matches():
    assert check_facts("что угодно", [[""], []]) == (False, False)


def test_cited_sources_full_path_and_basename():
    assert cited_sources(
        "См. week_02/README.md и mcp_client.py",
        ["week_02/README.md", "advent_core/mcp_client.py", "CLAUDE.md"],
    ) == (True, True, False)


def test_fit_hits_empty():
    assert fit_hits([]) == ()


def test_fit_hits_stops_at_limit():
    hits = [_hit("a" * 6), _hit("b" * 4), _hit("c" * 1), _hit("d" * 1)]
    assert fit_hits(hits, limit=10) == tuple(hits[:2])
    assert fit_hits(hits, limit=9) == (hits[0],)


def test_fit_hits_truncates_oversized_first():
    result = fit_hits([_hit("x" * 50), _hit("y")], limit=20)
    assert len(result) == 1
    assert result[0].text == "x" * 20
    assert result[0].source == "CLAUDE.md"


def test_check_facts_does_not_match_inside_a_longer_number():
    assert check_facts("максимум 11.5", [["1.5"]]) == (False,)
    assert check_facts("размерность 10240", [["1024"]]) == (False,)
    assert check_facts("версия 1.55", [["1.5"]]) == (False,)
    assert check_facts("это 21.5", [["1.5"]]) == (False,)


def test_check_facts_matches_a_number_with_punctuation_around():
    assert check_facts("Потолок — 1.5.", [["1.5"]]) == (True,)
    assert check_facts("(1024) измерения, 8192!", [["1024"], ["8192"]]) == (True, True)
    assert check_facts("окно 262 144.", [["262144"]]) == (True,)
    assert check_facts("t=1.5", [["1.5"]]) == (True,)


def test_check_facts_identifier_alternatives_still_substring():
    assert check_facts("функция call_tool_once()", [["call_tool_once"]]) == (True,)
    assert check_facts("кодировка cp1252", [["cp1252"]]) == (True,)


def test_cited_sources_other_path_with_same_basename_is_not_a_citation():
    assert cited_sources("см. other/README.md", ["week_05/README.md"]) == (False,)
    assert cited_sources("см. other/README.md", ["README.md"]) == (False,)


def test_cited_sources_bare_basename_counts_only_when_unambiguous():
    assert cited_sources("см. README.md", ["week_05/README.md"]) == (True,)
    assert cited_sources("см. README.md", ["README.md", "week_01/README.md"]) == (True, False)
    assert cited_sources("см. week_01/README.md", ["README.md", "week_01/README.md"]) == (
        False,
        True,
    )


def test_cited_sources_in_backticks_and_brackets():
    assert cited_sources(
        "Источники: `CLAUDE.md`, [advent_core/tokens.py]",
        ["CLAUDE.md", "advent_core/tokens.py"],
    ) == (True, True)


# --- Day 23: rewrite, RRF, rerank ------------------------------------------


def _h(cid: str, score: float = 0.8, text: str = "т") -> RagHit:
    return RagHit(cid, "CLAUDE.md", "Раздел", score, text)


def test_hit_and_context_new_fields_default_to_none_or_empty():
    hit = _h("a")
    assert (hit.rerank, hit.rank, hit.fused) == (None, None, None)
    ctx = RagContext((), "structure", 5, "m", None, "rev")
    assert (ctx.rewritten, ctx.candidates, ctx.passed, ctx.threshold) == (None, 0, None, None)
    assert (ctx.rerank_unrated, ctx.aux_prompt_tokens, ctx.aux_completion_tokens) == (0, None, None)
    assert ctx.warnings == ()
    assert ctx.trace is None


def test_rewrite_prompt_literal():
    assert RAG_REWRITE_PROMPT.endswith("одной строкой.\n\nВопрос: {question}")
    assert RAG_REWRITE_PROMPT.startswith("Перепиши вопрос пользователя в поисковый запрос")


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ('**"max_tokens limit"**', "max_tokens limit"),
        ("`query`", "query"),
        ("«запрос»", "запрос"),
        ("\n\n  первая строка\nвторая строка", "первая строка"),
        ("", None),
        ("  \n **  ** \n", None),
    ],
)
def test_clean_rewrite(raw, want):
    assert clean_rewrite(raw) == want


def test_rrf_merge_numbers_and_ranks():
    a, b, c, d = (_h(x) for x in "abcd")
    merged = rrf_merge([[a, b, c], [d, a]])
    by_id = {h.chunk_id: h for h in merged}
    assert by_id["a"].fused == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["b"].fused == pytest.approx(1 / 62)
    assert by_id["d"].fused == pytest.approx(1 / 61)
    assert [h.chunk_id for h in merged] == ["a", "d", "b", "c"]
    assert [h.rank for h in merged] == [1, 2, 3, 4]


def test_rrf_duplicate_keeps_cosine_of_the_first_list():
    original = _h("a", score=0.81)
    rewrite = _h("a", score=0.77)
    (only,) = rrf_merge([[original], [rewrite]])
    assert only.score == 0.81
    assert only.fused == pytest.approx(2 / 61)


def test_rrf_ties_break_by_chunk_id():
    merged = rrf_merge([[_h("b")], [_h("a")]])
    assert [h.chunk_id for h in merged] == ["a", "b"]
    assert merged[0].fused == merged[1].fused


def test_rerank_prompt_literal_parts():
    hits = [RagHit("a", "CLAUDE.md", "Раздел", 0.8, " тело А "), RagHit("b", "x.md", "", 0.7, "Б")]
    prompt = build_rerank_prompt("что?", hits)
    assert "для всех 2 фрагментов без пропусков." in prompt
    assert '{"scores": [{"id": <номер>, "score": <0-10>}, ...]}' in prompt
    assert "\n\n[1] CLAUDE.md — Раздел\nтело А\n\n[2] x.md\nБ\n\nВопрос: что?" in prompt
    # the question also precedes the fragments, right after the instruction
    assert prompt.count("Вопрос: что?") == 2
    assert "без пропусков.\n\nВопрос: что?\n\n[1] CLAUDE.md" in prompt


def test_parse_rerank_huge_integer_score_is_clamped_not_overflow():
    assert parse_rerank('{"scores": [{"id": 1, "score": 1' + "0" * 400 + "}]}", 1) == {1: 10.0}
    assert parse_rerank('{"scores": [{"id": 1, "score": -1' + "0" * 400 + "}]}", 1) == {1: 0.0}


def test_parse_rerank_happy_path():
    raw = '{"scores": [{"id": 1, "score": 9}, {"id": 2, "score": 3.5}]}'
    assert parse_rerank(raw, 2) == {1: 9.0, 2: 3.5}


@pytest.mark.parametrize(
    "raw",
    [
        "не json",
        "[1, 2]",
        '{"results": []}',
        '{"scores": "x"}',
        '{"scores": []}',
        '{"scores": [{"id": true, "score": 5}]}',
        '{"scores": [{"id": 1, "score": true}]}',
        '{"scores": [{"id": 1, "score": NaN}]}',
        '{"scores": [{"id": 9, "score": 5}]}',
        '{"scores": [{"id": 0, "score": 5}]}',
    ],
)
def test_parse_rerank_unusable_output_raises(raw):
    with pytest.raises(AdventError):
        parse_rerank(raw, 3)


def test_parse_rerank_clamps_dedups_and_reports_only_rated():
    raw = (
        '{"scores": [{"id": 1, "score": 15}, {"id": 1, "score": 2},'
        ' {"id": 2, "score": -4}, {"id": "3", "score": 5}, {"id": 7, "score": 5}, 5]}'
    )
    assert parse_rerank(raw, 3) == {1: 10.0, 2: 0.0}


def test_apply_rerank_threshold_order_and_k():
    hits = [_h("a"), _h("b"), _h("c"), _h("d")]
    scores = {1: 5.0, 2: 9.0, 3: 5.0, 4: 2.0}
    top, passed = apply_rerank(hits, scores, 5.0, 2)
    assert passed == 3
    assert [h.chunk_id for h in top] == ["b", "a"]  # ties keep input order
    assert [h.rerank for h in top] == [9.0, 5.0]
    assert [h.rank for h in top] == [2, 1]


def test_apply_rerank_keeps_existing_rank():
    hits = [RagHit("a", "s", "", 0.5, "t", rank=14)]
    (top,), passed = apply_rerank(hits, {1: 8.0}, 5.0, 5)
    assert (top.rank, passed) == (14, 1)


def test_apply_rerank_unrated_never_pass_even_at_threshold_zero():
    hits = [_h("a"), _h("b"), _h("c")]
    top, passed = apply_rerank(hits, {2: 0.0}, 0.0, 5)
    assert [h.chunk_id for h in top] == ["b"]
    assert passed == 1


def test_apply_rerank_nothing_passes():
    top, passed = apply_rerank([_h("a")], {1: 1.0}, 5.0, 5)
    assert (top, passed) == ((), 0)


def test_filtered_out_prompt_is_the_empty_instruction():
    assert build_rag_prompt("борщ?", [], filtered_out=True) == (
        "В документации проекта не нашлось фрагментов, относящихся к вопросу. "
        "Скажи об этом прямо и не отвечай по памяти.\n\nВопрос: борщ?"
    )
    assert RAG_EMPTY_INSTRUCTION in build_rag_prompt("q", [], filtered_out=True)


def test_empty_hits_without_filtered_out_stay_the_bare_question():
    assert build_rag_prompt("q", []) == "q"
    assert build_rag_prompt("q", [], filtered_out=False) == "q"
