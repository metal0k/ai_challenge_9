from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from advent_core.errors import AdventError
from advent_core.rag import RagContext, RagHit
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


@pytest.fixture
def seams(monkeypatch):
    calls = {"embed": [], "search": []}

    monkeypatch.setattr(rag.index_module, "load_runs", lambda db_path: {"structure": _run()})
    monkeypatch.setattr(rag.Config, "resolve", classmethod(lambda cls: object()))
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
        return SimpleNamespace(
            vectors=np.ones((1, 4), dtype=np.float32),
            prompt_tokens=7,
        )

    monkeypatch.setattr(rag, "embed_texts", fake_embed)

    def fake_search(db_path, strategy, vec, k):
        calls["search"].append({"strategy": strategy, "k": k})
        return [
            SimpleNamespace(chunk=_chunk("первый", n=1), score=0.9),
            SimpleNamespace(chunk=_chunk("второй", source="week_02/README.md", n=2), score=0.4),
        ]

    monkeypatch.setattr(rag.index_module, "search", fake_search)

    return calls


def test_check_index_missing_strategy(seams):
    with pytest.raises(AdventError) as exc:
        rag.check_index(None, "fixed")
    assert "fixed" in exc.value.message


def test_check_index_returns_run(seams):
    run = rag.check_index(None, "structure")
    assert run.model == "mistral-embed"


def test_make_retriever_does_no_io(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError

    monkeypatch.setattr(rag.index_module, "load_runs", boom)
    rag.make_retriever()


def test_retriever_builds_context(seams):
    ctx = rag.make_retriever()("Сколько?", "structure", 5)

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
    rag.make_retriever()("Сколько?", "structure", 5)

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

    ctx = rag.make_retriever()("q", "structure", 5)

    assert len(ctx.hits) == 1
    assert ctx.dropped == 1


def test_retriever_refuses_unknown_strategy_before_embedding(seams):
    with pytest.raises(AdventError):
        rag.make_retriever()("q", "fixed", 5)

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
