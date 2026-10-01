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


# --- Day 24: cited answers ---

CITE_HITS = (
    RagHit(
        "s:a#1", "CLAUDE.md", "OBS recording", 0.8, "Run `rag_cite` mode. Window is 262144 tokens."
    ),
    RagHit("s:b#2", "README.md", "", 0.7, "Tag w01d01 must exist on the remote before submit."),
    RagHit(
        "s:c#3", "CLAUDE.md", "Secrets", 0.6, "The env file is gitignored from the first commit."
    ),
    RagHit("s:d#4", "x.md", "Other", 0.5, "Fourth fragment text goes right here."),
)


def _cited_json(**over):
    import json

    data = {
        "status": "answer",
        "answer": "Тег должен быть на remote.",
        "sources": [2],
        "quotes": [{"id": 2, "text": "Tag w01d01 must exist on the remote"}],
    }
    data.update(over)
    return json.dumps(data, ensure_ascii=False)


def test_quote_norm_strips_markdown_edge_quotes_and_casefolds():
    from advent_core.rag import quote_norm

    assert quote_norm("**«Hello   World.»**") == "hello world"
    assert quote_norm("  `rag_cite` mode… ") == "rag_cite mode"
    assert quote_norm("a_b *c*") == "a_b *c*"


def test_quote_in_accepts_markdown_wrapped_verbatim_quote():
    from advent_core.rag import quote_in

    assert quote_in("**«Window is 262144 tokens.»**", CITE_HITS[0].text)


def test_quote_in_rejects_identifier_glued_together():
    from advent_core.rag import quote_in

    assert not quote_in("Run ragcite mode", CITE_HITS[0].text)


def test_quote_in_rejects_short_quote_and_paraphrase():
    from advent_core.rag import quote_in

    assert not quote_in("262144", CITE_HITS[0].text)
    assert not quote_in("The window holds 262144 tokens", CITE_HITS[0].text)


def test_build_cite_prompt_puts_the_question_at_both_ends_with_numbering():
    from advent_core.rag import RAG_CITE_INSTRUCTION, build_cite_prompt

    prompt = build_cite_prompt("Как?", CITE_HITS[:2])
    parts = prompt.split("\n\n")
    assert parts[0] == RAG_CITE_INSTRUCTION
    assert parts[1] == "Вопрос: Как?"
    assert parts[-1] == "Вопрос: Как?"
    assert "[1] CLAUDE.md — OBS recording\nRun `rag_cite` mode." in prompt
    assert "[2] README.md\nTag w01d01" in prompt
    assert prompt.count("Вопрос: Как?") == 2
    assert "JSON" in RAG_CITE_INSTRUCTION


def test_parse_cited_accepts_a_verbatim_answer():
    from advent_core.rag import parse_cited

    c = parse_cited(_cited_json(), CITE_HITS)
    assert c.status == "answer" and c.reason == ""
    assert c.answer == "Тег должен быть на remote."
    assert [h.chunk_id for h in c.sources] == ["s:b#2"]
    assert [h.chunk_id for h in c.quoted_sources] == ["s:b#2"]
    assert c.quotes[0].verified is True and c.quotes[0].n == 2
    assert c.verified_quotes == 1 and c.quoted is True


def test_parse_cited_sources_are_declared_quoted_sources_are_verbatim_only():
    from advent_core.rag import parse_cited

    raw = _cited_json(
        sources=[3, 1],
        quotes=[
            {"id": 1, "text": "Window is 262144 tokens."},
            {"id": 3, "text": "something the model made up here"},
        ],
    )
    c = parse_cited(raw, CITE_HITS)
    assert [h.chunk_id for h in c.sources] == ["s:a#1", "s:c#3"]
    assert [h.chunk_id for h in c.quoted_sources] == ["s:a#1"]
    assert [q.verified for q in c.quotes] == [True, False]


def test_parse_cited_without_verbatim_quote_is_unverified_and_keeps_the_draft():
    from advent_core.rag import parse_cited

    raw = _cited_json(quotes=[{"id": 2, "text": "Tag must be somewhere maybe"}])
    c = parse_cited(raw, CITE_HITS)
    assert (c.status, c.reason, c.answer) == ("unknown", "unverified", "")
    assert c.draft == "Тег должен быть на remote."
    assert c.quoted is False and c.quoted_sources == ()
    assert len(c.quotes) == 1 and c.quotes[0].verified is False


def test_parse_cited_model_unknown_clarifies_from_hits():
    from advent_core.rag import parse_cited

    c = parse_cited('{"status": "unknown", "answer": "", "sources": [], "quotes": []}', CITE_HITS)
    assert (c.status, c.reason) == ("unknown", "model_unknown")
    assert c.clarify == ("CLAUDE.md — OBS recording", "README.md", "CLAUDE.md — Secrets")


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "[1, 2]",
        "null",
        "42",
        '{"status": "maybe"}',
        '{"status": 1}',
        '{"status": "answer"}',
        '{"status": "answer", "answer": 5}',
        None,
        b"bytes",
    ],
)
def test_parse_cited_is_total_on_garbage(raw):
    from advent_core.rag import parse_cited

    c = parse_cited(raw, CITE_HITS)
    assert (c.status, c.reason, c.answer) == ("unknown", "bad_json", "")


def test_parse_cited_survives_pathological_nesting():
    from advent_core.rag import parse_cited

    c = parse_cited("[" * 100000, CITE_HITS)
    assert (c.status, c.reason) == ("unknown", "bad_json")


def test_parse_cited_empty_answer_with_status_answer_is_bad_json():
    from advent_core.rag import parse_cited

    assert parse_cited(_cited_json(answer="   "), CITE_HITS).reason == "bad_json"


def test_parse_cited_quotes_null_and_sources_not_a_list_are_tolerated():
    from advent_core.rag import parse_cited

    c = parse_cited(_cited_json(quotes=None, sources="2"), CITE_HITS)
    assert (c.status, c.reason) == ("unknown", "unverified")
    assert c.sources == () and c.quotes == ()


def test_parse_cited_skips_bool_out_of_range_and_garbage_ids_and_entries():
    from advent_core.rag import parse_cited

    raw = _cited_json(
        sources=[True, 0, 5, "2", 2.0, 4, 4],
        quotes=[
            {"id": True, "text": "Tag w01d01 must exist on the remote"},
            {"id": 9, "text": "a sentence nobody wrote anywhere"},
            {"id": 2, "text": 7},
            {"id": 2, "text": "  "},
            "junk",
            {"id": 2, "text": "Tag w01d01 must exist on the remote"},
        ],
    )
    c = parse_cited(raw, CITE_HITS)
    assert [h.chunk_id for h in c.sources] == ["s:d#4"]
    assert len(c.quotes) == 1 and c.quotes[0].n == 2 and c.quoted is True


def test_parse_cited_truncated_wins_even_over_valid_json():
    from advent_core.rag import parse_cited

    c = parse_cited(_cited_json(), CITE_HITS, True)
    assert (c.status, c.reason) == ("unknown", "truncated")


def test_parse_cited_accepts_a_json_code_fence():
    from advent_core.rag import parse_cited

    assert parse_cited("```json\n" + _cited_json() + "\n```", CITE_HITS).quoted is True


def test_unknown_answer_dedupes_sections_and_caps_at_three():
    from advent_core.rag import unknown_answer

    near = [CITE_HITS[0], CITE_HITS[0], CITE_HITS[1], CITE_HITS[2], CITE_HITS[3]]
    c = unknown_answer("empty_context", near)
    assert c.clarify == ("CLAUDE.md — OBS recording", "README.md", "CLAUDE.md — Secrets")
    assert (c.status, c.reason, c.answer, c.sources, c.quotes) == (
        "unknown",
        "empty_context",
        "",
        (),
        (),
    )
    assert unknown_answer("no_index").clarify == ()


def test_render_cited_answer_literal_with_context_numbers_and_marks():
    from advent_core.rag import parse_cited, render_cited

    raw = _cited_json(
        sources=[2, 3],
        quotes=[
            {"id": 2, "text": "Tag w01d01 must exist on the remote"},
            {"id": 3, "text": "an invented sentence for fragment three"},
        ],
    )
    c = parse_cited(raw, CITE_HITS)
    expected = (
        "Тег должен быть на remote.\n"
        "\n"
        "Источники:\n"
        "[2] README.md · s:b#2 · цитата ✓\n"
        "[3] CLAUDE.md — Secrets · s:c#3\n"
        "Цитаты:\n"
        "[2] ✓ «Tag w01d01 must exist on the remote»\n"
        "[3] ✗ «an invented sentence for fragment three»"
    )
    assert render_cited(c, CITE_HITS, for_history=False) == expected
    assert render_cited(c, CITE_HITS, for_history=True) == expected
    assert "подтвержд" not in expected


def test_parse_cited_reattributes_a_quote_to_the_fragment_that_holds_it():
    from advent_core.rag import parse_cited

    raw = _cited_json(
        sources=[3],
        quotes=[{"id": 3, "text": "Tag w01d01 must exist on the remote"}],
    )
    c = parse_cited(raw, CITE_HITS)
    assert c.status == "answer" and c.reason == ""
    q = c.quotes[0]
    assert (q.n, q.claimed, q.verified) == (2, 3, True)
    assert [h.chunk_id for h in c.quoted_sources] == ["s:b#2"]


def test_parse_cited_reattribution_picks_the_lowest_fragment():
    from advent_core.rag import parse_cited

    hits = (*CITE_HITS, CITE_HITS[1])
    raw = _cited_json(quotes=[{"id": 4, "text": "Tag w01d01 must exist on the remote"}])
    q = parse_cited(raw, hits).quotes[0]
    assert (q.n, q.claimed) == (2, 4)


def test_parse_cited_quote_found_nowhere_stays_unverified_with_its_claimed_id():
    from advent_core.rag import parse_cited

    raw = _cited_json(quotes=[{"id": 3, "text": "a sentence nobody wrote anywhere"}])
    c = parse_cited(raw, CITE_HITS)
    q = c.quotes[0]
    assert (q.n, q.claimed, q.verified) == (3, None, False)
    assert c.reason == "unverified"


def test_parse_cited_out_of_range_id_is_reattributed_when_the_quote_is_verbatim():
    from advent_core.rag import parse_cited

    raw = _cited_json(
        sources=[5], quotes=[{"id": 5, "text": "Tag w01d01 must exist on the remote"}]
    )
    c = parse_cited(raw, CITE_HITS)
    q = c.quotes[0]
    assert (q.n, q.claimed, q.verified) == (2, 5, True)
    assert c.status == "answer" and c.quoted is True
    assert c.sources == ()  # declared out-of-range source stays dropped


def test_parse_cited_out_of_range_id_found_nowhere_is_dropped():
    from advent_core.rag import parse_cited

    raw = _cited_json(quotes=[{"id": 5, "text": "a sentence nobody wrote anywhere"}])
    c = parse_cited(raw, CITE_HITS)
    assert c.quotes == ()
    assert (c.status, c.reason) == ("unknown", "unverified")


def test_parse_cited_bool_id_is_still_rejected_even_with_a_verbatim_quote():
    from advent_core.rag import parse_cited

    raw = _cited_json(quotes=[{"id": True, "text": "Tag w01d01 must exist on the remote"}])
    assert parse_cited(raw, CITE_HITS).quotes == ()


def test_parse_cited_quote_in_its_claimed_fragment_is_not_marked_corrected():
    from advent_core.rag import parse_cited

    q = parse_cited(_cited_json(), CITE_HITS).quotes[0]
    assert (q.n, q.claimed, q.verified) == (2, None, True)


def test_render_cited_shows_the_correction_literal():
    from advent_core.rag import parse_cited, render_cited

    raw = _cited_json(
        sources=[3],
        quotes=[{"id": 3, "text": "Tag w01d01 must exist on the remote"}],
    )
    out = render_cited(parse_cited(raw, CITE_HITS), CITE_HITS, for_history=False)
    assert "[2] ✓ (модель указала [3]) «Tag w01d01 must exist on the remote»" in out
    assert out.endswith("«Tag w01d01 must exist on the remote»")


def test_render_cited_unknown_literal():
    from advent_core.rag import render_cited, unknown_answer

    c = unknown_answer("empty_context", CITE_HITS[:2])
    expected = (
        "Не знаю: в документации проекта не нашлось фрагментов, относящихся к вопросу.\n"
        "Уточните вопрос — возможно, вы про:\n"
        "· CLAUDE.md — OBS recording\n"
        "· README.md"
    )
    assert render_cited(c, CITE_HITS, for_history=True) == expected
    assert render_cited(c, CITE_HITS, for_history=False) == expected


def test_render_cited_unverified_draft_is_screen_only():
    from advent_core.rag import RAG_UNKNOWN_PREFIX, parse_cited, render_cited

    raw = _cited_json(quotes=[{"id": 2, "text": "Tag must be somewhere maybe"}])
    c = parse_cited(raw, CITE_HITS)
    screen = render_cited(c, CITE_HITS, for_history=False)
    history = render_cited(c, CITE_HITS, for_history=True)
    assert screen.startswith("Не знаю: ни одна цитата модели не нашлась дословно")
    assert "неподтверждённый ответ модели: «Тег должен быть на remote.»" in screen
    assert "Тег должен быть на remote." not in history
    assert "неподтверждённый" not in history
    assert history.startswith("Не знаю:")
    assert RAG_UNKNOWN_PREFIX == "Не знаю:"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Не знаю: в документации нет этого.", True),
        ("К сожалению, в документации проекта нет информации об этом.", True),
        ("Этот факт не нашлось где подтвердить.", True),
        ("Окно контекста — 262144 токена.", False),
        ("", False),
        ("Не знаю. " + "очень длинное пояснение " * 30, False),
        ("Длинный ответ " * 20 + "не знаю", False),
    ],
)
def test_says_unknown_heuristic(text, expected):
    from advent_core.rag import says_unknown

    assert says_unknown(text) is expected


def test_rag_settings_cite_defaults_to_false():
    from advent_core.rag import RagSettings

    assert RagSettings("structure", 5, 20, False, False, 5.0).cite is False
    assert RagSettings("structure", 5, 20, True, True, 5.0, cite=True).cite is True
