from advent_core.rag import (
    RAG_INSTRUCTION,
    RagHit,
    build_rag_prompt,
    check_facts,
    cited_sources,
    fact_span,
    fit_hits,
    normalize,
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
