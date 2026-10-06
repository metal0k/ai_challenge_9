"""week_05/index.py — SQLite storage, search, chunk stats, eval (SPEC-w05d21.md §5, §6, §9).

No network: EmbedResult is built by hand everywhere a real embed_texts() call
would otherwise be needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from advent_core.embeddings import EmbedResult
from advent_core.errors import AdventError
from week_05.chunking import Chunk, Document
from week_05.index import (
    BrokenQuestion,
    EvalQuestion,
    ExpectedAnswer,
    Hit,
    RunInfo,
    _is_cut_md,
    _metrics_from_ranks,
    _py_def_spans,
    _rank_of,
    _valid_expected,
    check_comparable,
    chunk_stats,
    evaluate,
    find_chunk_by_line,
    get_chunk,
    leader,
    load_chunks,
    load_eval,
    load_runs,
    search,
    write_index,
)


def _chunk(
    chunk_id: str,
    *,
    strategy: str = "fixed",
    source: str = "CLAUDE.md",
    section: str = "(preamble)",
    ordinal: int = 0,
    char_start: int = 0,
    char_end: int = 10,
    line_start: int = 1,
    line_end: int = 1,
    text: str = "hello.",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        strategy=strategy,
        source=source,
        title="CLAUDE",
        section=section,
        ordinal=ordinal,
        char_start=char_start,
        char_end=char_end,
        line_start=line_start,
        line_end=line_end,
        text=text,
    )


def _embed_result(n: int, dim: int, *, model: str = "mistral-embed") -> EmbedResult:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms
    return EmbedResult(vectors=matrix, model=model, prompt_tokens=100, requests=1, latency_ms=5)


def _write(db_path, chunks: list[Chunk], embed_result: EmbedResult, **kw) -> None:
    strategy = chunks[0].strategy
    corpus_rev = kw.pop("corpus_rev", "deadbeef")
    corpus_files = kw.pop("corpus_files", 1)
    corpus_chars = kw.pop("corpus_chars", 1000)
    write_index(
        db_path,
        {strategy: (chunks, embed_result)},
        corpus_rev=corpus_rev,
        corpus_files=corpus_files,
        corpus_chars=corpus_chars,
    )


# ---------------------------------------------------------------------------
# write_index / load_runs / get_chunk roundtrip
# ---------------------------------------------------------------------------


def test_write_and_load_roundtrip(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    chunks = [_chunk("fixed:CLAUDE.md#0", ordinal=0), _chunk("fixed:CLAUDE.md#1", ordinal=1)]
    embed_result = _embed_result(2, dim=4)

    _write(db_path, chunks, embed_result, corpus_rev="abc123", corpus_files=1, corpus_chars=20)

    runs = load_runs(db_path)
    assert set(runs) == {"fixed"}
    info = runs["fixed"]
    assert isinstance(info, RunInfo)
    assert info.model == "mistral-embed"
    assert info.dim == 4
    assert info.n_chunks == 2
    assert info.corpus_rev == "abc123"
    assert info.corpus_files == 1
    assert info.corpus_chars == 20
    assert info.cost_usd == pytest.approx(100 / 1_000_000 * 0.10)

    fetched = get_chunk(db_path, "fixed:CLAUDE.md#1")
    assert fetched is not None
    assert fetched.chunk_id == "fixed:CLAUDE.md#1"
    assert fetched.ordinal == 1


def test_get_chunk_missing_returns_none(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    _write(db_path, [_chunk("fixed:CLAUDE.md#0")], _embed_result(1, dim=3))
    assert get_chunk(db_path, "fixed:CLAUDE.md#999") is None


def test_load_runs_missing_db_raises_advent_error(tmp_path):
    with pytest.raises(AdventError):
        load_runs(tmp_path / "nope.sqlite3")


def test_load_runs_corrupt_db_raises_advent_error_with_recovery_hint(tmp_path):
    db_path = tmp_path / "garbage.sqlite3"
    db_path.write_bytes(b"not a sqlite database at all")

    with pytest.raises(AdventError) as excinfo:
        load_runs(db_path)
    assert excinfo.value.hint is not None
    assert "adventrag index" in excinfo.value.hint
    assert "Traceback" not in str(excinfo.value)


def test_load_chunks_returns_stored_chunks_for_strategy(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    chunks = [_chunk("fixed:CLAUDE.md#0", ordinal=0), _chunk("fixed:CLAUDE.md#1", ordinal=1)]
    _write(db_path, chunks, _embed_result(2, dim=3))

    loaded = load_chunks(db_path, "fixed")
    assert [c.chunk_id for c in loaded] == ["fixed:CLAUDE.md#0", "fixed:CLAUDE.md#1"]

    assert load_chunks(db_path, "structure") == []


def test_default_db_path_used_when_none_given(tmp_path, monkeypatch):
    """path=None is resolved INSIDE the body, not bound at import time (CLAUDE.md trap)."""
    import week_05.index as index_module

    monkeypatch.setattr(index_module, "DEFAULT_DB", tmp_path / "default.sqlite3")
    _write(None, [_chunk("fixed:CLAUDE.md#0")], _embed_result(1, dim=3))
    assert (tmp_path / "default.sqlite3").exists()
    assert set(load_runs(None)) == {"fixed"}


# ---------------------------------------------------------------------------
# write_index: one transaction across strategies, rollback on failure
# ---------------------------------------------------------------------------


def test_write_index_replaces_only_given_strategies(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    _write(db_path, [_chunk("fixed:CLAUDE.md#0")], _embed_result(1, dim=3))

    struct_chunk = _chunk("structure:CLAUDE.md#0", strategy="structure")
    write_index(
        db_path,
        {"structure": ([struct_chunk], _embed_result(1, dim=3))},
        corpus_rev="deadbeef",
        corpus_files=1,
        corpus_chars=1000,
    )

    runs = load_runs(db_path)
    assert set(runs) == {"fixed", "structure"}
    assert get_chunk(db_path, "fixed:CLAUDE.md#0") is not None


def test_write_index_replaces_existing_rows_of_same_strategy(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    _write(db_path, [_chunk("fixed:CLAUDE.md#0", ordinal=0)], _embed_result(1, dim=3))
    _write(
        db_path,
        [_chunk("fixed:CLAUDE.md#0", ordinal=0, text="new text")],
        _embed_result(1, dim=3),
    )

    runs = load_runs(db_path)
    assert runs["fixed"].n_chunks == 1
    fetched = get_chunk(db_path, "fixed:CLAUDE.md#0")
    assert fetched.text == "new text"


def test_write_index_rolls_back_on_mid_write_failure(tmp_path, monkeypatch):
    """Inject a failure between strategies' inserts and check the WHOLE write rolls back.

    Can't monkeypatch sqlite3.Connection.executemany itself (it's an
    immutable C-extension type), so the injection point is
    week_05.index.embed_cost_usd — a plain Python function called once per
    strategy, after that strategy's chunk rows are already INSERTed but
    before its `runs` row is. That still proves the transaction, not just
    the per-statement call, is what gets rolled back.
    """
    db_path = tmp_path / "index.sqlite3"
    old_chunk = _chunk("fixed:CLAUDE.md#0", text="old text")
    _write(db_path, [old_chunk], _embed_result(1, dim=3), corpus_rev="rev1")

    import week_05.index as index_module

    real_cost = index_module.embed_cost_usd
    calls = {"n": 0}

    def failing_cost(model, tokens):
        calls["n"] += 1
        if calls["n"] == 2:  # let "fixed" (1st strategy) succeed, fail on "structure" (2nd)
            raise RuntimeError("injected failure")
        return real_cost(model, tokens)

    monkeypatch.setattr(index_module, "embed_cost_usd", failing_cost)

    with pytest.raises(RuntimeError):
        write_index(
            db_path,
            {
                "fixed": ([_chunk("fixed:CLAUDE.md#0", text="new text")], _embed_result(1, dim=3)),
                "structure": (
                    [_chunk("structure:CLAUDE.md#0", strategy="structure")],
                    _embed_result(1, dim=3),
                ),
            },
            corpus_rev="rev2",
            corpus_files=1,
            corpus_chars=1000,
        )

    # Whole transaction rolled back: even "fixed", which fully succeeded
    # before the injected failure, is still the OLD row from the first call.
    runs = load_runs(db_path)
    assert set(runs) == {"fixed"}
    assert runs["fixed"].corpus_rev == "rev1"
    fetched = get_chunk(db_path, "fixed:CLAUDE.md#0")
    assert fetched.text == "old text"


# ---------------------------------------------------------------------------
# check_comparable
# ---------------------------------------------------------------------------


def _run_info(**overrides) -> RunInfo:
    base = dict(
        strategy="fixed",
        model="mistral-embed",
        dim=4,
        n_chunks=1,
        prompt_tokens=10,
        requests=1,
        latency_ms=5,
        cost_usd=0.001,
        corpus_rev="deadbeef",
        corpus_files=1,
        corpus_chars=100,
        created_at="2026-09-28T00:00:00+00:00",
    )
    base.update(overrides)
    return RunInfo(**base)


def test_check_comparable_passes_when_model_and_rev_match():
    runs = {
        "fixed": _run_info(strategy="fixed"),
        "structure": _run_info(strategy="structure"),
    }
    check_comparable(runs, ["fixed", "structure"])  # no raise


def test_check_comparable_rejects_different_models():
    runs = {
        "fixed": _run_info(strategy="fixed", model="mistral-embed"),
        "structure": _run_info(strategy="structure", model="codestral-embed"),
    }
    with pytest.raises(AdventError):
        check_comparable(runs, ["fixed", "structure"])


def test_check_comparable_rejects_different_corpus_rev():
    runs = {
        "fixed": _run_info(strategy="fixed", corpus_rev="rev1"),
        "structure": _run_info(strategy="structure", corpus_rev="rev2"),
    }
    with pytest.raises(AdventError):
        check_comparable(runs, ["fixed", "structure"])


@pytest.mark.parametrize(
    ("field", "left", "right"),
    [
        ("endpoint", "cloud", "local"),
        ("dim", 4, 8),
        ("doc_prefix", "search_document: ", ""),
        ("query_prefix", "search_query: ", "query: "),
    ],
)
def test_check_comparable_rejects_a_different_vector_space_field(field, left, right):
    runs = {
        "fixed": _run_info(strategy="fixed", **{field: left}),
        "structure": _run_info(strategy="structure", **{field: right}),
    }
    with pytest.raises(AdventError) as info:
        check_comparable(runs, ["fixed", "structure"])
    assert "нечестным" in info.value.message


def test_check_comparable_rejects_missing_strategy():
    runs = {"fixed": _run_info(strategy="fixed")}
    with pytest.raises(AdventError):
        check_comparable(runs, ["fixed", "structure"])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_ranks_by_cosine_and_respects_k(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]],
        dtype=np.float32,
    )
    embed_result = EmbedResult(
        vectors=vectors, model="m", prompt_tokens=1, requests=1, latency_ms=1
    )
    chunks = [
        _chunk("fixed:a.md#0", source="a.md", ordinal=0),
        _chunk("fixed:b.md#0", source="b.md", ordinal=0),
        _chunk("fixed:c.md#0", source="c.md", ordinal=0),
    ]
    _write(db_path, chunks, embed_result)

    query = np.array([1.0, 0.0], dtype=np.float32)
    hits = search(db_path, "fixed", query, k=2)

    assert [h.chunk.source for h in hits] == ["a.md", "c.md"]
    assert isinstance(hits[0], Hit)
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.7071, abs=1e-3)


def test_search_unknown_strategy_returns_empty(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    _write(db_path, [_chunk("fixed:a.md#0")], _embed_result(1, dim=3))
    assert search(db_path, "structure", np.zeros(3, dtype=np.float32)) == []


def test_search_dim_mismatch_raises(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    _write(db_path, [_chunk("fixed:a.md#0")], _embed_result(1, dim=4))
    with pytest.raises(AdventError):
        search(db_path, "fixed", np.zeros(3, dtype=np.float32))


def test_search_breaks_score_ties_by_chunk_id(tmp_path):
    """np.argsort's tie order is not guaranteed stable — pin the tie-break (finding 7)."""
    db_path = tmp_path / "index.sqlite3"
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # all tied
    embed_result = EmbedResult(
        vectors=vectors, model="m", prompt_tokens=1, requests=1, latency_ms=1
    )
    chunks = [
        _chunk("fixed:z.md#0", source="z.md", ordinal=0),
        _chunk("fixed:a.md#0", source="a.md", ordinal=0),
        _chunk("fixed:m.md#0", source="m.md", ordinal=0),
    ]
    _write(db_path, chunks, embed_result)

    query = np.array([1.0, 0.0], dtype=np.float32)
    hits = search(db_path, "fixed", query, k=3)
    assert [h.chunk.chunk_id for h in hits] == ["fixed:a.md#0", "fixed:m.md#0", "fixed:z.md#0"]


def test_search_accepts_2d_query_vector(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    embed_result = EmbedResult(
        vectors=vectors, model="m", prompt_tokens=1, requests=1, latency_ms=1
    )
    _write(
        db_path,
        [_chunk("fixed:a.md#0", source="a.md"), _chunk("fixed:b.md#0", source="b.md", ordinal=1)],
        embed_result,
    )
    query = np.array([[1.0, 0.0]], dtype=np.float32)  # shape (1, dim), as embed_texts returns
    hits = search(db_path, "fixed", query, k=1)
    assert hits[0].chunk.source == "a.md"


# ---------------------------------------------------------------------------
# find_chunk_by_line
# ---------------------------------------------------------------------------


def test_find_chunk_by_line(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    chunks = [
        _chunk("structure:a.md#0", strategy="structure", source="a.md", line_start=1, line_end=5),
        _chunk(
            "structure:a.md#1",
            strategy="structure",
            source="a.md",
            ordinal=1,
            line_start=6,
            line_end=10,
        ),
    ]
    _write(db_path, chunks, _embed_result(2, dim=3))

    found = find_chunk_by_line(db_path, "structure", "a.md", 7)
    assert found is not None
    assert found.chunk_id == "structure:a.md#1"
    assert find_chunk_by_line(db_path, "structure", "a.md", 100) is None


# ---------------------------------------------------------------------------
# chunk_stats
# ---------------------------------------------------------------------------


def test_chunk_stats_empty():
    stats = chunk_stats([], [])
    assert stats.n_chunks == 0
    assert stats.small_share == 0.0
    assert stats.cut_share == 0.0


def test_chunk_stats_sizes_and_small_share():
    chunks = [
        _chunk("c0", text="x" * 100, char_end=100),  # small
        _chunk("c1", ordinal=1, text="x" * 300, char_end=300),
        _chunk("c2", ordinal=2, text="x" * 500, char_end=500),
    ]
    stats = chunk_stats(chunks, [Document(source="CLAUDE.md", text="x" * 500)])
    assert stats.n_chunks == 3
    assert stats.min_chars == 100
    assert stats.max_chars == 500
    assert stats.median_chars == 300.0
    assert stats.small_share == pytest.approx(1 / 3)


def test_is_cut_md_ends_mid_sentence():
    chunk = _chunk("c0", text="this sentence never ends")
    assert _is_cut_md(chunk) is True


def test_is_cut_md_ends_at_punctuation():
    chunk = _chunk("c0", text="a complete sentence.")
    assert _is_cut_md(chunk) is False


def test_is_cut_md_odd_fence_count_is_cut():
    chunk = _chunk("c0", text="text\n```python\ncode continues past the chunk end")
    assert _is_cut_md(chunk) is True


def test_is_cut_md_even_fence_count_is_not_cut():
    chunk = _chunk("c0", text="text\n```python\ncode\n```\ndone.")
    assert _is_cut_md(chunk) is False


def test_py_def_spans_and_cut_share():
    py_text = "\n".join(
        [
            "import os",  # line 1
            "",  # 2
            "def foo():",  # 3
            "    return os.getcwd()",  # 4 (end of foo)
            "",  # 5
            "def bar():",  # 6
            "    return 1",  # 7 (end of bar)
        ]
    )
    doc = Document(source="advent_core/x.py", text=py_text)
    spans = _py_def_spans(doc)
    assert spans == [(3, 4), (6, 7)]

    # chunk covering lines 1-4: ends exactly at foo's end_lineno -> not a cut
    foo_text = py_text[: py_text.index("\n\ndef bar")]
    whole_foo = _chunk("c0", source="advent_core/x.py", line_start=1, line_end=4, text=foo_text)
    # chunk starting mid-body of foo (line 4 only) -> cut
    mid_foo = _chunk("c1", ordinal=1, source="advent_core/x.py", line_start=4, line_end=4, text="x")

    stats = chunk_stats([whole_foo, mid_foo], [doc])
    assert stats.n_chunks == 2
    assert stats.cut_share == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# eval: load_eval, _valid_expected, _metrics_from_ranks, evaluate, leader
# ---------------------------------------------------------------------------


def test_load_eval_parses_format(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        '[{"id": 1, "question": "q1?", "expected": [{"source": "CLAUDE.md", "anchor": "foo"}], '
        '"placeholder": true}]',
        encoding="utf-8",
    )
    questions = load_eval(path)
    assert len(questions) == 1
    q = questions[0]
    assert isinstance(q, EvalQuestion)
    assert q.id == 1
    assert q.question == "q1?"
    assert q.expected == [ExpectedAnswer(source="CLAUDE.md", anchor="foo")]
    assert q.placeholder is True


def test_load_eval_placeholder_defaults_false(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        '[{"id": 1, "question": "q?", "expected": [{"source": "CLAUDE.md", "anchor": "x"}]}]',
        encoding="utf-8",
    )
    assert load_eval(path)[0].placeholder is False


def test_load_eval_missing_file_raises_advent_error(tmp_path):
    with pytest.raises(AdventError):
        load_eval(tmp_path / "nope.json")


def test_load_eval_malformed_json_raises_advent_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AdventError):
        load_eval(path)


def test_valid_expected_drops_anchor_missing_from_doc():
    q = EvalQuestion(
        id=1,
        question="q?",
        expected=[
            ExpectedAnswer(source="CLAUDE.md", anchor="present text"),
            ExpectedAnswer(source="CLAUDE.md", anchor="absent text"),
        ],
    )
    docs = {"CLAUDE.md": Document(source="CLAUDE.md", text="this has present text in it")}
    valid, reasons = _valid_expected(q, docs)
    assert valid == [ExpectedAnswer(source="CLAUDE.md", anchor="present text")]
    assert len(reasons) == 1
    assert "absent text" in reasons[0]


def test_valid_expected_drops_source_missing_from_snapshot():
    q = EvalQuestion(id=1, question="q?", expected=[ExpectedAnswer(source="gone.md", anchor="x")])
    valid, reasons = _valid_expected(q, {})
    assert valid == []
    assert "gone.md" in reasons[0]


def test_rank_of_miss_when_source_matches_but_chunk_text_lacks_the_anchor():
    """Same source/section as the expected answer, but the anchor itself isn't in THIS chunk."""
    expected = [ExpectedAnswer(source="CLAUDE.md", anchor="ceiling is 1.5")]
    hits = [
        Hit(
            chunk=_chunk(
                "fixed:CLAUDE.md#0",
                source="CLAUDE.md",
                section="temperature",
                text="temperature has an upper bound but this chunk never states the number",
            ),
            score=1.0,
        )
    ]
    assert _rank_of(hits, expected) is None


def test_rank_of_miss_when_anchor_is_split_across_two_chunk_texts():
    """The anchor straddles a chunk boundary: neither chunk contains it whole."""
    expected = [ExpectedAnswer(source="CLAUDE.md", anchor="ceiling is 1.5")]
    hits = [
        Hit(
            chunk=_chunk(
                "fixed:CLAUDE.md#0", source="CLAUDE.md", ordinal=0, text="temperature ceiling is"
            ),
            score=1.0,
        ),
        Hit(
            chunk=_chunk("fixed:CLAUDE.md#1", source="CLAUDE.md", ordinal=1, text=" 1.5, not 2.0"),
            score=0.9,
        ),
    ]
    assert _rank_of(hits, expected) is None


def test_metrics_from_ranks_literal():
    metrics = _metrics_from_ranks([1, None, 3])
    assert metrics.n_questions == 3
    assert metrics.hit_at_1 == pytest.approx(1 / 3)
    assert metrics.hit_at_3 == pytest.approx(2 / 3)
    assert metrics.hit_at_5 == pytest.approx(2 / 3)
    assert metrics.mrr_at_5 == pytest.approx((1 + 0 + 1 / 3) / 3)


def test_metrics_from_ranks_all_misses():
    metrics = _metrics_from_ranks([None, None])
    assert metrics.hit_at_1 == 0.0
    assert metrics.hit_at_5 == 0.0
    assert metrics.mrr_at_5 == 0.0


def test_metrics_from_ranks_empty():
    metrics = _metrics_from_ranks([])
    assert metrics.n_questions == 0
    assert metrics.hit_at_1 == 0.0


def test_evaluate_end_to_end_with_broken_question():
    docs_by_source = {
        "CLAUDE.md": Document(source="CLAUDE.md", text="the ceiling is 1.5 for temperature"),
    }
    questions = [
        EvalQuestion(
            id=1,
            question="max temperature?",
            expected=[ExpectedAnswer(source="CLAUDE.md", anchor="ceiling is 1.5")],
        ),
        EvalQuestion(
            id=2,
            question="broken question",
            expected=[ExpectedAnswer(source="CLAUDE.md", anchor="not in the doc anywhere")],
        ),
    ]

    def hit_for(source: str, text: str) -> Hit:
        return Hit(chunk=_chunk(f"fixed:{source}#0", source=source, text=text), score=1.0)

    hits_by_question = {
        "fixed": {
            1: [hit_for("CLAUDE.md", "the ceiling is 1.5 for temperature")],
            2: [hit_for("CLAUDE.md", "irrelevant text")],
        },
        "structure": {
            1: [hit_for("other.md", "wrong source entirely")],
            2: [],
        },
    }

    report = evaluate(hits_by_question, questions, docs_by_source)

    assert len(report.broken) == 1
    assert report.broken[0].id == 2

    fixed_metrics = report.per_strategy["fixed"]
    assert fixed_metrics.n_questions == 1  # only question 1 survives (question 2 is broken)
    assert fixed_metrics.hit_at_1 == 1.0
    assert fixed_metrics.mrr_at_5 == 1.0

    structure_metrics = report.per_strategy["structure"]
    assert structure_metrics.n_questions == 1
    assert structure_metrics.hit_at_1 == 0.0
    assert structure_metrics.mrr_at_5 == 0.0


def test_evaluate_question_with_one_invalid_anchor_of_several_goes_entirely_broken():
    """Finding 3: one bad location among several is a dataset error for the WHOLE question,
    not a silent partial-scoring on the surviving location."""
    docs_by_source = {
        "CLAUDE.md": Document(source="CLAUDE.md", text="the ceiling is 1.5 for temperature"),
    }
    questions = [
        EvalQuestion(
            id=1,
            question="q?",
            expected=[
                ExpectedAnswer(source="CLAUDE.md", anchor="ceiling is 1.5"),  # valid
                ExpectedAnswer(source="CLAUDE.md", anchor="not anywhere in the doc"),  # invalid
            ],
        ),
    ]
    hits_by_question = {
        "fixed": {
            1: [Hit(chunk=_chunk("fixed:CLAUDE.md#0", text="the ceiling is 1.5"), score=1.0)]
        },
    }

    report = evaluate(hits_by_question, questions, docs_by_source)

    assert len(report.broken) == 1
    assert report.broken[0].id == 1
    assert isinstance(report.broken[0], BrokenQuestion)
    assert "not anywhere in the doc" in report.broken[0].reason
    assert report.per_strategy["fixed"].n_questions == 0  # not partially scored


def test_all_eval_anchors_are_shorter_than_fixed_overlap():
    """SPEC §6 guarantee: every anchor fits inside a single `fixed`-strategy overlap window."""
    from advent_core.config import PROJECT_ROOT
    from week_05.chunking import FIXED_OVERLAP

    questions = load_eval(PROJECT_ROOT / "week_05" / "eval_questions.json")
    for q in questions:
        for e in q.expected:
            assert len(e.anchor) < FIXED_OVERLAP, (q.id, e.anchor)


def test_leader_strict_max():
    assert leader({"fixed": 0.5, "structure": 0.8}) == "structure"


def test_leader_tie_returns_none():
    assert leader({"fixed": 0.5, "structure": 0.5}) is None


def test_leader_empty_returns_none():
    assert leader({}) is None
