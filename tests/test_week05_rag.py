from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from advent_core import config as config_module
from advent_core.errors import AdventError
from advent_core.rag import RagContext, RagHit, RagSettings
from advent_core.telemetry import CallResult, Usage
from week_05 import rag
from week_05.chunking import Chunk

ROOT = Path(__file__).resolve().parents[1]


def _chunk(text, *, source="CLAUDE.md", section="Раздел", n=1):
    return Chunk(
        chunk_id=f"structure:{source}#{n}",
        strategy="structure",
        source=source,
        title=source,
        section=section,
        ordinal=n,
        char_start=0,
        char_end=len(text),
        line_start=1,
        line_end=1,
        text=text,
    )


def _question(**kw):
    base = {
        "id": 1,
        "question": "Сколько?",
        "expect": (("262144",),),
        "sources": ("CLAUDE.md",),
    }
    base.update(kw)
    return rag.ControlQuestion(**base)


def _write(tmp_path, data):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _run(model="mistral-embed", rev="abc123", n=3):
    return SimpleNamespace(model=model, corpus_rev=rev, n_chunks=n)


class _Ctx:
    def __enter__(self):
        return object()

    def __exit__(self, *exc):
        return False


def _sr(text, n=1, score=0.5, source="CLAUDE.md"):
    return SimpleNamespace(chunk=_chunk(text, source=source, n=n), score=score)


@pytest.fixture
def seams(monkeypatch):
    calls = {"embed": [], "search": [], "chat": [], "journal": []}
    calls["rewrite_text"] = "  **окно контекста context window**\nвторая строка"
    calls["rerank_text"] = json.dumps({"scores": []})
    calls["search_fn"] = lambda row, k: [
        SimpleNamespace(chunk=_chunk("первый", n=1), score=0.9),
        SimpleNamespace(chunk=_chunk("второй", source="week_02/README.md", n=2), score=0.4),
    ]

    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setattr(rag.index_module, "load_runs", lambda db_path: {"structure": _run()})
    monkeypatch.setattr(rag, "mistral_client", lambda config: _Ctx())

    def fake_embed(client, model, texts, *, week, day, journal_extra):
        calls["embed"].append(
            {
                "model": model,
                "texts": list(texts),
                "week": week,
                "day": day,
                "journal_extra": journal_extra,
            }
        )
        vectors = np.stack([np.full(4, float(i + 1), dtype=np.float32) for i in range(len(texts))])
        return SimpleNamespace(vectors=vectors, prompt_tokens=7)

    monkeypatch.setattr(rag, "embed_texts", fake_embed)

    def fake_search(db_path, strategy, vec, k):
        calls["search"].append({"strategy": strategy, "k": k})
        return calls["search_fn"](int(vec[0]), k)

    monkeypatch.setattr(rag.index_module, "search", fake_search)

    def fake_complete(config, messages, capabilities=None, **kwargs):
        calls["chat"].append({"config": config, "messages": list(messages)})
        sent = list(messages)
        if config.params.format == "json":  # chat._payload adds this system message
            sent = [{"role": "system", "content": "JSON-ИНСТРУКЦИЯ"}, *sent]
        is_rewrite = messages[-1]["content"].startswith("Перепиши")
        text = calls["rewrite_text"] if is_rewrite else calls["rerank_text"]
        return CallResult(
            text=text,
            usage=Usage(prompt_tokens=1000 if not is_rewrite else 50, completion_tokens=20),
            latency_ms=10,
            stream=False,
            sent_messages=sent,
        )

    monkeypatch.setattr(rag.chat_core, "complete", fake_complete)
    monkeypatch.setattr(
        rag,
        "log_call",
        lambda result, messages, *, week, day, extra=None: calls["journal"].append(
            {"result": result, "week": week, "day": day, "extra": extra, "messages": messages}
        ),
    )

    return calls


def test_check_index_missing_strategy(seams):
    with pytest.raises(AdventError) as exc:
        rag.check_index(None, "fixed")
    assert "fixed" in exc.value.message


def test_check_index_returns_run(seams):
    run = rag.check_index(None, "structure")
    assert run.model == "mistral-embed"


def _settings(
    strategy: str = "structure",
    k: int = 5,
    *,
    rewrite: bool = False,
    rerank: bool = False,
    k_before: int = 20,
    threshold: float = 5.0,
) -> RagSettings:
    return RagSettings(strategy, k, k_before, rewrite, rerank, threshold)


def test_make_retriever_does_no_io(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError

    monkeypatch.setattr(rag.index_module, "load_runs", boom)
    rag.make_retriever()


def test_retriever_builds_context(seams):
    ctx = rag.make_retriever()("Сколько?", _settings("structure", 5))

    assert isinstance(ctx, RagContext)
    assert ctx.strategy == "structure"
    assert ctx.k == 5
    assert ctx.embed_model == "mistral-embed"
    assert ctx.embed_tokens == 7
    assert ctx.corpus_rev == "abc123"
    assert ctx.dropped == 0
    assert [h.source for h in ctx.hits] == ["CLAUDE.md", "week_02/README.md"]
    assert ctx.hits[0].score == 0.9
    assert ctx.hits[0].text == "первый"
    assert ctx.hits[0].chunk_id == "structure:CLAUDE.md#1"


def test_retriever_embeds_with_index_model_and_day_22(seams):
    rag.make_retriever()("Сколько?", _settings("structure", 5))

    assert seams["embed"] == [
        {
            "model": "mistral-embed",
            "texts": ["Сколько?"],
            "week": 5,
            "day": 22,
            "journal_extra": {"strategy": "structure", "command": "rag"},
        }
    ]
    assert seams["search"] == [{"strategy": "structure", "k": 5}]


def test_retriever_counts_dropped_hits(seams, monkeypatch):
    monkeypatch.setattr(
        rag.index_module,
        "search",
        lambda db_path, strategy, vec, k: [
            SimpleNamespace(chunk=_chunk("a" * 20000, n=1), score=0.5),
            SimpleNamespace(chunk=_chunk("a" * 20000, source="week_02/README.md", n=2), score=0.5),
        ],
    )

    ctx = rag.make_retriever()("q", _settings("structure", 5))

    assert len(ctx.hits) == 1
    assert ctx.dropped == 1


def test_retriever_refuses_unknown_strategy_before_embedding(seams):
    with pytest.raises(AdventError):
        rag.make_retriever()("q", _settings("fixed", 5))

    assert seams["embed"] == []


def test_load_questions_roundtrip(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": 3,
                "question": "Сколько?",
                "expect": [["262144"], ["a", "b"]],
                "sources": ["CLAUDE.md"],
                "note": "заметка",
            }
        ],
    )

    result = rag.load_questions(path)

    assert result == [
        rag.ControlQuestion(
            id=3,
            question="Сколько?",
            expect=(("262144",), ("a", "b")),
            sources=("CLAUDE.md",),
            note="заметка",
        )
    ]


def test_load_questions_missing_file(tmp_path):
    with pytest.raises(AdventError) as exc:
        rag.load_questions(tmp_path / "nope.json")

    assert "не найден" in exc.value.message


def test_load_questions_bad_json(tmp_path):
    path = tmp_path / "q.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(AdventError) as exc:
        rag.load_questions(path)

    assert "повреждён" in exc.value.message


_BASE = {
    "id": 1,
    "question": "q",
    "expect": [["x"]],
    "sources": ["CLAUDE.md"],
}


@pytest.mark.parametrize(
    "item, needle",
    [
        ({**_BASE, "id": True}, "id должен"),
        ({**_BASE, "id": "1"}, "id должен"),
        ({**_BASE, "question": "  "}, "question должен"),
        ({**_BASE, "expect": []}, "expect должен"),
        ({**_BASE, "expect": [[]]}, "expect должен"),
        ({**_BASE, "expect": ["x"]}, "expect должен"),
        ({**_BASE, "expect": [[" "]]}, "expect должен"),
        ({**_BASE, "sources": []}, "sources должен"),
        ({**_BASE, "sources": [""]}, "sources должен"),
        ({**_BASE, "note": 5}, "note должен"),
    ],
)
def test_load_questions_rejects_bad_shape(tmp_path, item, needle):
    path = _write(tmp_path, [item])

    with pytest.raises(AdventError) as exc:
        rag.load_questions(path)

    assert needle in exc.value.message
    assert "запись 1" in exc.value.message


def test_load_questions_rejects_non_list_and_duplicate_ids(tmp_path):
    with pytest.raises(AdventError) as exc:
        rag.load_questions(_write(tmp_path, {"id": 1}))

    assert "ожидался список" in exc.value.message

    with pytest.raises(AdventError) as exc:
        rag.load_questions(_write(tmp_path, [_BASE, _BASE]))

    assert "повторяются" in exc.value.message


def test_validate_questions_splits_valid_and_broken():
    chunks = [
        _chunk("Окно 262 144 токена"),
        _chunk("другое", source="week_02/README.md", n=2),
    ]

    q_ok = _question()
    q_no_source = _question(id=2, sources=("missing.md",))
    q_no_fact = _question(id=3, expect=(("262144",), ("alice", "алиса")))
    q_other_source = _question(id=4, sources=("week_02/README.md",))

    valid, broken = rag.validate_questions([q_ok, q_no_source, q_no_fact, q_other_source], chunks)

    assert valid == [q_ok]
    assert [q.id for q, _ in broken] == [2, 3, 4]
    assert broken[0][1] == "источника missing.md нет в индексе"
    assert broken[1][1] == "факт alice | алиса не найден в источниках"
    assert broken[2][1] == "факт 262144 не найден в источниках"


def test_score_answer_without_rag():
    score = rag.score_answer(
        _question(expect=(("262144",), ("alice",))),
        "Окно 262 144, см. CLAUDE.md",
        None,
    )

    assert score.facts == (True, False)
    assert score.facts_found == 1
    assert score.complete is False
    assert score.sources_retrieved is None
    assert score.sources_cited == (True,)


def test_score_answer_with_rag_context():
    ctx = RagContext(
        hits=(
            RagHit(
                chunk_id="c",
                source="week_02/README.md",
                section="",
                score=0.5,
                text="t",
            ),
        ),
        strategy="structure",
        k=5,
        embed_model="m",
        embed_tokens=1,
        corpus_rev="r",
    )

    q = _question(sources=("CLAUDE.md", "week_02/README.md"))
    score = rag.score_answer(q, "262144", ctx)

    assert score.sources_retrieved == (False, True)
    assert score.sources_cited == (False, False)
    assert score.complete is True


def test_shipped_question_set_has_ten_valid_shaped_questions():
    qs = rag.load_questions(ROOT / "week_05" / "rag_questions.json")

    assert len(qs) == 10
    assert sorted(q.id for q in qs) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert all(q.note for q in qs)


def test_shipped_question_set_is_supported_by_the_real_index():
    db = ROOT / "data" / "rag" / "index.sqlite3"
    if not db.exists():
        pytest.skip("no local index")

    from week_05 import index as index_module

    chunks = index_module.load_chunks(db, "structure")
    valid, broken = rag.validate_questions(
        rag.load_questions(ROOT / "week_05" / "rag_questions.json"),
        chunks,
    )

    assert broken == []
    assert len(valid) == 10


# --- Day 23: rewrite, RRF, rerank, threshold -------------------------------------------


def _scores(*pairs):
    return json.dumps({"scores": [{"id": i, "score": s} for i, s in pairs]})


def _by_vector(first, second=None):
    """search_fn: vector row 1 (the original question) gets `first`, row 2 gets `second`."""

    def search(row, k):
        return list(first if row == 1 else (second if second is not None else first))

    return search


def test_plain_settings_search_k_and_never_call_chat(seams):
    ctx = rag.make_retriever()("Сколько?", _settings(k=3))

    assert seams["search"] == [{"strategy": "structure", "k": 3}]
    assert seams["chat"] == []
    assert seams["journal"] == []
    assert ctx.trace is None
    assert ctx.rewritten is None
    assert ctx.candidates == 0
    assert ctx.passed is None
    assert ctx.aux_prompt_tokens is None
    assert ctx.warnings == ()


def test_rewrite_only_embeds_both_in_one_call_and_searches_k_before(seams):
    seams["search_fn"] = _by_vector(
        [_sr("a", 1, 0.9), _sr("b", 2, 0.8)], [_sr("b", 2, 0.7), _sr("c", 3, 0.6)]
    )
    ctx = rag.make_retriever()("Сколько?", _settings(k=2, rewrite=True, k_before=7))

    assert [e["texts"] for e in seams["embed"]] == [["Сколько?", "окно контекста context window"]]
    assert seams["search"] == [{"strategy": "structure", "k": 7}] * 2
    assert len(seams["chat"]) == 1
    assert ctx.rewritten == "окно контекста context window"
    assert ctx.candidates == 3
    assert ctx.passed is None
    assert ctx.threshold is None
    # RRF: b is rank 2 + rank 1 -> first; a (1/61) beats c (1/62)
    assert [h.chunk_id for h in ctx.hits] == [
        "structure:CLAUDE.md#2",
        "structure:CLAUDE.md#1",
    ]
    assert ctx.trace is not None
    assert [h.chunk_id for h in ctx.trace.original] == [
        "structure:CLAUDE.md#1",
        "structure:CLAUDE.md#2",
    ]
    assert len(ctx.trace.fused) == 3
    assert ctx.trace.reranked == ()


def test_duplicate_chunk_keeps_the_original_cosine_and_sums_rrf(seams):
    seams["search_fn"] = _by_vector([_sr("b", 2, 0.81)], [_sr("b", 2, 0.55)])
    ctx = rag.make_retriever()("q", _settings(rewrite=True))

    (hit,) = ctx.hits
    assert hit.score == 0.81
    assert hit.fused == pytest.approx(2 / 61)


def test_rerank_only_searches_k_before_once_and_filters_by_threshold(seams):
    seams["search_fn"] = _by_vector([_sr("a", 1, 0.9), _sr("b", 2, 0.8), _sr("c", 3, 0.7)])
    seams["rerank_text"] = _scores((1, 3), (2, 9), (3, 5))
    ctx = rag.make_retriever()("q", _settings(k=5, rerank=True, k_before=12, threshold=5))

    assert [e["texts"] for e in seams["embed"]] == [["q"]]
    assert seams["search"] == [{"strategy": "structure", "k": 12}]
    assert len(seams["chat"]) == 1
    assert [(h.chunk_id[-1], h.rerank, h.rank) for h in ctx.hits] == [("2", 9.0, 2), ("3", 5.0, 3)]
    assert ctx.candidates == 3
    assert ctx.passed == 2
    assert ctx.threshold == 5.0
    assert ctx.rewritten is None
    assert [h.rerank for h in ctx.trace.reranked] == [9.0, 5.0, 3.0]
    assert [h.chunk_id[-1] for h in ctx.trace.fused] == ["1", "2", "3"]


def test_top_k_is_applied_after_the_threshold(seams):
    seams["search_fn"] = _by_vector([_sr(c, i, 0.9) for i, c in enumerate("abcd", 1)])
    seams["rerank_text"] = _scores((1, 9), (2, 8), (3, 7), (4, 6))
    ctx = rag.make_retriever()("q", _settings(k=2, rerank=True, threshold=0))

    assert [h.chunk_id[-1] for h in ctx.hits] == ["1", "2"]
    assert ctx.passed == 4


def test_rewrite_and_rerank_send_the_whole_union_to_the_reranker(seams):
    seams["search_fn"] = _by_vector(
        [_sr("alpha", 1), _sr("beta", 2)], [_sr("beta", 2), _sr("gamma", 3)]
    )
    seams["rerank_text"] = _scores((1, 9), (2, 8), (3, 7))
    ctx = rag.make_retriever()("q", _settings(rewrite=True, rerank=True))

    assert len(seams["chat"]) == 2
    prompt = seams["chat"][1]["messages"][-1]["content"]
    assert "[3] CLAUDE.md" in prompt and "[4]" not in prompt
    for text in ("alpha", "beta", "gamma"):
        assert text in prompt
    assert prompt.rstrip().endswith("Вопрос: q")
    assert ctx.candidates == 3
    assert ctx.passed == 3
    assert [e["texts"] for e in seams["embed"]] == [["q", "окно контекста context window"]]


def test_rerank_prompt_carries_the_chunk_whole(seams):
    long_text = "x" * 5000 + "ФАКТ-В-КОНЦЕ"
    seams["search_fn"] = _by_vector([_sr(long_text, 1)])
    seams["rerank_text"] = _scores((1, 9))
    rag.make_retriever()("q", _settings(rerank=True))

    assert "ФАКТ-В-КОНЦЕ" in seams["chat"][0]["messages"][-1]["content"]


def test_aux_calls_are_isolated_from_env_params(seams, monkeypatch):
    from advent_core import chat as chat_core

    monkeypatch.setenv("MISTRAL_MAX_TOKENS", "5")
    monkeypatch.setenv("MISTRAL_STOP", "}")
    monkeypatch.setenv("MISTRAL_TOP_P", "0.1")
    monkeypatch.setenv("MISTRAL_TEMPERATURE", "1.4")
    seams["search_fn"] = _by_vector([_sr("a", 1)])
    seams["rerank_text"] = _scores((1, 9))
    rag.make_retriever()("q", _settings(rewrite=True, rerank=True))

    rewrite_cfg, rerank_cfg = (c["config"] for c in seams["chat"])
    rewrite_payload = chat_core._payload(rewrite_cfg, [{"role": "user", "content": "x"}])[0]
    rerank_payload = chat_core._payload(rerank_cfg, [{"role": "user", "content": "x"}])[0]
    assert rewrite_payload["model"] == rag.RAG_AUX_MODEL == "ministral-14b-latest"
    assert rewrite_payload["temperature"] == 0
    assert rewrite_payload["max_tokens"] == 200
    assert rerank_payload["temperature"] == 0
    assert rerank_payload["max_tokens"] == 2000
    assert rerank_payload["response_format"] == {"type": "json_object"}
    assert "response_format" not in rewrite_payload
    for payload in (rewrite_payload, rerank_payload):
        assert not {"stop", "top_p", "random_seed", "reasoning_effort"} & payload.keys()
    assert rewrite_cfg.stream is False and rerank_cfg.stream is False


def test_rerank_prompt_asks_for_json_in_words_not_only_in_the_format_flag(seams):
    seams["search_fn"] = _by_vector([_sr("a", 1)])
    seams["rerank_text"] = _scores((1, 9))
    rag.make_retriever()("q", _settings(rerank=True))

    first = seams["chat"][0]["messages"][-1]["content"]
    assert 'Верни JSON {"scores": [{"id": <номер>, "score": <0-10>}, ...]}' in first
    assert "для всех 1 фрагментов без пропусков." in first


def test_each_aux_call_is_journaled_as_day_23(seams):
    seams["search_fn"] = _by_vector([_sr("a", 1)])
    seams["rerank_text"] = _scores((1, 9))
    rag.make_retriever()("q", _settings(rewrite=True, rerank=True))

    rows = seams["journal"]
    assert [(r["week"], r["day"], r["extra"]) for r in rows] == [
        (5, 23, {"command": "rag_rewrite"}),
        (5, 23, {"command": "rag_rerank"}),
    ]


def test_rerank_journal_records_the_json_system_instruction_actually_sent(seams):
    seams["search_fn"] = _by_vector([_sr("a", 1)])
    seams["rerank_text"] = _scores((1, 9))
    rag.make_retriever()("q", _settings(rerank=True))

    (row,) = seams["journal"]
    assert row["messages"][0] == {"role": "system", "content": "JSON-ИНСТРУКЦИЯ"}


def test_validate_questions_fact_is_not_supported_inside_a_longer_number():
    chunks = [_chunk("лимит 11.5 токенов")]
    valid, broken = rag.validate_questions([_question(expect=(("1.5",),))], chunks)
    assert valid == []
    assert broken[0][1] == "факт 1.5 не найден в источниках"
    ok, _ = rag.validate_questions([_question(expect=(("11.5",),))], chunks)
    assert len(ok) == 1


def test_rerank_is_journaled_even_when_parsing_fails(seams):
    seams["search_fn"] = _by_vector([_sr("a", 1)])
    seams["rerank_text"] = "это не JSON"

    with pytest.raises(AdventError):
        rag.make_retriever()("q", _settings(rerank=True))

    (row,) = seams["journal"]
    assert row["extra"] == {"command": "rag_rerank"}
    assert row["result"].text == "это не JSON"
    assert row["day"] == 23


def test_unrated_chunks_never_pass_even_at_threshold_zero_and_warn(seams):
    seams["search_fn"] = _by_vector([_sr("a", 1), _sr("b", 2), _sr("c", 3)])
    seams["rerank_text"] = _scores((2, 0), (99, 9))
    ctx = rag.make_retriever()("q", _settings(rerank=True, threshold=0))

    assert [h.chunk_id[-1] for h in ctx.hits] == ["2"]
    assert ctx.passed == 1
    assert ctx.rerank_unrated == 2
    assert ctx.warnings == ("reranker не оценил 2 из 3 чанков",)
    assert [h.rerank for h in ctx.trace.reranked] == [0.0, None, None]


def test_everything_below_the_threshold_gives_passed_zero(seams):
    seams["search_fn"] = _by_vector([_sr("a", 1), _sr("b", 2)])
    seams["rerank_text"] = _scores((1, 2), (2, 1))
    ctx = rag.make_retriever()("q", _settings(rerank=True))

    assert ctx.hits == ()
    assert ctx.passed == 0
    assert ctx.candidates == 2


def test_empty_rewrite_warns_and_falls_back_to_the_original_question(seams):
    seams["rewrite_text"] = "  \n ** ** \n"
    seams["search_fn"] = _by_vector([_sr("a", 1)])
    ctx = rag.make_retriever()("q", _settings(rewrite=True))

    assert ctx.rewritten is None
    assert [e["texts"] for e in seams["embed"]] == [["q"]]
    assert seams["search"] == [{"strategy": "structure", "k": 20}]
    assert len(ctx.warnings) == 1 and "rewrite" in ctx.warnings[0]
    assert [h.chunk_id[-1] for h in ctx.hits] == ["1"]


def test_aux_tokens_are_summed_over_both_calls(seams):
    seams["search_fn"] = _by_vector([_sr("a", 1)])
    seams["rerank_text"] = _scores((1, 9))
    ctx = rag.make_retriever()("q", _settings(rewrite=True, rerank=True))

    assert ctx.aux_prompt_tokens == 1050
    assert ctx.aux_completion_tokens == 40
    assert ctx.embed_tokens == 7


def test_aux_tokens_are_unknown_when_usage_is_missing(seams, monkeypatch):
    seams["search_fn"] = _by_vector([_sr("a", 1)])

    def no_usage(config, messages, capabilities=None, **kwargs):
        return CallResult(text="запрос", usage=Usage(), latency_ms=1, stream=False)

    monkeypatch.setattr(rag.chat_core, "complete", no_usage)
    ctx = rag.make_retriever()("q", _settings(rewrite=True))

    assert ctx.aux_prompt_tokens is None
    assert ctx.aux_completion_tokens is None


def test_context_cap_applies_after_rerank(seams):
    seams["search_fn"] = _by_vector([_sr("a" * 20000, 1), _sr("b" * 20000, 2)])
    seams["rerank_text"] = _scores((1, 9), (2, 8))
    ctx = rag.make_retriever()("q", _settings(rerank=True))

    assert len(ctx.hits) == 1
    assert ctx.dropped == 1
    assert ctx.passed == 2


def test_relevant_chunk_ids_needs_source_and_every_fact():
    q = _question(expect=(("262144",), ("w01d01",)), sources=("CLAUDE.md",))
    both = _chunk("окно 262 144, тег w01d01", n=1)
    one = _chunk("только 262144", n=2)
    other_source = _chunk("окно 262144 и w01d01", source="README.md", n=3)

    assert rag.relevant_chunk_ids(q, [both, one, other_source]) == {both.chunk_id}


def test_relevant_chunk_ids_respects_number_boundaries():
    q = _question(expect=(("1.5",),))
    inside = _chunk("потолок 11.5 секунд", n=1)
    exact = _chunk("потолок 1.5 секунды", n=2)

    assert rag.relevant_chunk_ids(q, [inside, exact]) == {exact.chunk_id}


def test_relevant_chunk_ids_is_empty_when_facts_are_spread_over_chunks():
    q = _question(expect=(("альфа",), ("бета",)))

    assert rag.relevant_chunk_ids(q, [_chunk("альфа", n=1), _chunk("бета", n=2)]) == frozenset()


# --- Day 24: journal day, judge, cite scoring, unanswerable set ------------------------------


def _cited(answer="окно 262144 токена", quote="окно 262144 токена", verified=True, hits=None):
    from advent_core.rag import CitedAnswer, Quote

    hit = RagHit("structure:CLAUDE.md#1", "CLAUDE.md", "Раздел", 0.9, "окно 262144 токена")
    hits = hits if hits is not None else (hit,)
    if not verified:
        return CitedAnswer("unknown", "", (hit,), (), (Quote(1, quote, False),), "unverified",
                           (), answer)  # fmt: skip
    return CitedAnswer("answer", answer, (hit,), (hit,), (Quote(1, quote, True),), "")


def test_aux_calls_follow_the_aux_day_argument_and_embedding_keeps_day(seams):
    seams["search_fn"] = _by_vector([_sr("a", 1, 0.9)])
    seams["rerank_text"] = _scores((1, 9))
    rag.make_retriever(day=24, aux_day=24)("q", _settings(rewrite=True, rerank=True))
    assert [(r["week"], r["day"], r["extra"]["command"]) for r in seams["journal"]] == [
        (5, 24, "rag_rewrite"),
        (5, 24, "rag_rerank"),
    ]
    assert seams["embed"][0]["day"] == 24


def test_aux_day_defaults_to_23_for_day_23_callers(seams):
    seams["rerank_text"] = _scores((1, 9))
    rag.make_retriever()("q", _settings(rerank=True))
    assert [r["day"] for r in seams["journal"]] == [23]


def test_judge_journals_before_parsing_and_uses_the_given_day(seams):
    seams["rerank_text"] = json.dumps({"verdict": "частично", "reason": " часть  ответа "})
    cited = _cited()
    g = rag.judge_grounding(cited, cited.sources, "Сколько?", day=24)
    assert (g.verdict, g.reason) == ("частично", "часть ответа")
    (row,) = seams["journal"]
    assert (row["week"], row["day"], row["extra"]) == (5, 24, {"command": "rag_judge"})
    assert seams["chat"][0]["config"].params.format == "json"
    assert seams["chat"][0]["config"].params.temperature == 0


def test_judge_prompt_shows_only_verified_quotes_and_no_expected_facts(seams):
    from advent_core.rag import CitedAnswer, Quote

    hit = RagHit("c#1", "CLAUDE.md", "Раздел", 0.9, "ПОЛНЫЙ-ТЕКСТ-ЧАНКА окно 262144 токена")
    cited = CitedAnswer(
        "answer", "ответ", (hit,), (hit,),
        (Quote(1, "окно 262144 токена", True), Quote(1, "выдуманная цитата", False)), "",
    )  # fmt: skip
    seams["rerank_text"] = json.dumps({"verdict": "да", "reason": "ок"})
    rag.judge_grounding(cited, (hit,), "Вопрос?", day=24)
    prompt = seams["chat"][0]["messages"][-1]["content"]
    assert "[1] «окно 262144 токена»" in prompt
    assert "выдуманная" not in prompt
    assert "ПОЛНЫЙ-ТЕКСТ-ЧАНКА" not in prompt
    assert "Вопрос: Вопрос?" in prompt
    assert "Ответ: ответ" in prompt


@pytest.mark.parametrize(
    "raw",
    ["не JSON", "[]", '{"verdict": "возможно"}', '{"verdict": 3}', '{"reason": "x"}', "null"],
)
def test_judge_garbage_gives_none_not_no(seams, raw):
    seams["rerank_text"] = raw
    cited = _cited()
    g = rag.judge_grounding(cited, cited.sources, "q", day=24)
    assert g.verdict is None
    assert len(seams["journal"]) == 1  # the paid call is on record even when unreadable


def test_judge_accepts_a_fenced_verdict_and_normalises_case(seams):
    seams["rerank_text"] = '```json\n{"verdict": " Да ", "reason": "ok"}\n```'
    cited = _cited()
    assert rag.judge_grounding(cited, cited.sources, "q", day=24).verdict == "да"


def test_judge_is_not_called_for_a_refusal_or_an_unverified_answer(seams):
    from advent_core.rag import unknown_answer

    assert rag.judge_grounding(unknown_answer("empty_context"), (), "q", day=24).verdict is None
    unverified = _cited(verified=False)
    assert rag.judge_grounding(unverified, unverified.sources, "q", day=24).verdict is None
    assert seams["chat"] == []
    assert seams["journal"] == []


def test_cite_score_uses_the_answer_only_and_quoted_sources_only():
    q = _question(expect=(("262144",),), sources=("CLAUDE.md", "README.md"))
    ctx = RagContext(
        hits=(RagHit("c#1", "README.md", "Р", 0.9, "t"),),
        strategy="structure",
        k=5,
        embed_model="m",
        embed_tokens=1,
        corpus_rev="r",
    )
    score = rag.score_answer(q, "ТЕКСТ ИЗ СТАРОГО ОТВЕТА 262144", ctx, _cited())
    assert score.facts == (True,)
    assert score.sources_cited == (True, False)  # CLAUDE.md is the quoted fragment
    assert score.sources_retrieved == (False, True)


def test_cite_score_of_refusal_and_unverified_draft_is_zero():
    from advent_core.rag import unknown_answer

    q = _question()
    for cited in (unknown_answer("model_unknown"), _cited(answer="262144", verified=False)):
        score = rag.score_answer(q, "262144 есть в тексте", None, cited)
        assert score.facts == (False,)
        assert score.sources_cited == (False,)


def test_score_without_cite_is_unchanged():
    score = rag.score_answer(_question(), "окно 262144, см. CLAUDE.md", None)
    assert score.facts == (True,)
    assert score.sources_cited == (True,)


def test_load_unanswerable_reads_the_shipped_file():
    items = rag.load_unanswerable(ROOT / "week_05" / "rag_unanswerable.json")
    assert [i.id for i in items] == [101, 102, 103]
    assert items[0].question == "Как приготовить борщ?"
    assert items[1].question == (
        "Какая база данных PostgreSQL используется для хранения сессий агента?"
    )
    assert items[2].question == (
        "Какой тариф Mistral оплачен на аккаунте проекта и сколько он стоит в месяц?"
    )
    assert all(i.note for i in items)


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1},
        [1],
        [{"question": "q"}],
        [{"id": True, "question": "q"}],
        [{"id": 1, "question": "  "}],
        [{"id": 1, "question": "q", "note": 3}],
        [{"id": 1, "question": "q"}, {"id": 1, "question": "r"}],
    ],
)
def test_load_unanswerable_rejects_bad_files(tmp_path, data):
    with pytest.raises(AdventError):
        rag.load_unanswerable(_write(tmp_path, data))


def test_load_unanswerable_missing_and_corrupt(tmp_path):
    with pytest.raises(AdventError):
        rag.load_unanswerable(tmp_path / "none.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(AdventError):
        rag.load_unanswerable(bad)


@pytest.mark.parametrize(
    ("text", "answer", "quote", "expected"),
    [
        ("рендер без факта", "ответ без факта", "окно 262144 токена", False),  # only in the quote
        ("Окно 262144 токена (рендер)", "ответ без факта", "цитата без факта", False),  # rendered
        ("рендер без факта", "окно 262144 токена", "цитата без факта", True),  # only in answer
    ],
)
def test_cite_score_counts_facts_from_cited_answer_alone(text, answer, quote, expected):
    cited = _cited(answer=answer, quote=quote)
    score = rag.score_answer(_question(expect=(("262144",),)), text, None, cited)
    assert score.facts == (expected,)


def _followup_agent():
    from advent_core import chat as chat_core
    from advent_core.agent import Agent
    from advent_core.config import Config

    main_calls: list[list[dict]] = []
    answer = json.dumps(
        {
            "status": "answer",
            "answer": "Порог равен 4242.",
            "sources": [1],
            "quotes": [{"id": 1, "text": "порог compaction равен 4242 токена"}],
        },
        ensure_ascii=False,
    )

    def main_complete(config, messages, capabilities=None, **kwargs):
        main_calls.append(list(messages))
        return CallResult(text=answer, usage=Usage(10, 5), stream=False, sent_messages=messages)

    config = Config.resolve(model="ministral-14b-latest", stream=False)
    config.params.rag = True
    config.params.rag_cite = True
    agent = Agent(
        config,
        complete=main_complete,
        stream=chat_core.stream,
        retrieve=rag.make_retriever(),
    )
    return agent, main_calls


def test_cite_follow_up_embeds_the_bare_clarification_and_only_rewrite_sees_both(seams):
    seams["search_fn"] = lambda row, k: [_sr("порог compaction равен 4242 токена", 1, 0.9)]
    agent, main_calls = _followup_agent()

    seams["rerank_text"] = _scores((1, 0))  # turn 1: below the threshold -> refusal
    first = agent.ask("как оно устроено?", [])
    assert first.cited is not None and first.cited.reason == "empty_context"
    assert main_calls == []

    seams["rerank_text"] = _scores((1, 9))
    second = agent.ask("про порог compaction", first.history)

    assert second.cited is not None and second.cited.quoted
    embed_texts = [e["texts"] for e in seams["embed"]]
    assert embed_texts[0][0] == "как оно устроено?"
    assert embed_texts[1][0] == "про порог compaction"  # bare clarification, no concatenation
    rewrite_prompts = [
        c["messages"][-1]["content"]
        for c in seams["chat"]
        if c["messages"][-1]["content"].startswith("Перепиши")
    ]
    assert "Предыдущий вопрос" not in rewrite_prompts[0]
    assert "Предыдущий вопрос: как оно устроено?" in rewrite_prompts[1]
    assert "Уточнение: про порог compaction" in rewrite_prompts[1]
    assert second.rag_previous == "как оно устроено?"
