from advent_core.rag import (
    RAG_INSTRUCTION,
    RagHit,
    build_rag_prompt,
    check_facts,
    cited_sources,
    fit_hits,
    normalize,
)


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
