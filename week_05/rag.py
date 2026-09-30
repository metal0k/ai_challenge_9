"""Day 22: retriever over the SQLite index and the control-question set."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from advent_core.client import mistral_client
from advent_core.config import Config
from advent_core.embeddings import embed_texts
from advent_core.errors import AdventError
from advent_core.rag import (
    RagContext,
    RagHit,
    RetrieveFn,
    check_facts,
    cited_sources,
    fit_hits,
    normalize,
)
from week_05 import index as index_module
from week_05.chunking import Chunk

RAG_WEEK = 5
RAG_DAY = 22


def check_index(db_path: Path | None, strategy: str) -> index_module.RunInfo:
    """Check that the requested strategy exists in the index."""
    runs = index_module.load_runs(db_path)
    if strategy not in runs:
        raise AdventError(
            f"В индексе нет стратегии {strategy!r}.",
            hint=f"Сначала `adventrag index --strategy {strategy}`.",
        )
    return runs[strategy]


def make_retriever(
    db_path: Path | None = None,
    *,
    week: int = RAG_WEEK,
    day: int = RAG_DAY,
) -> RetrieveFn:
    """Create a retriever bound to a database path and journal coordinates."""

    def retrieve(question: str, strategy: str, k: int) -> RagContext:
        """Retrieve the top-k chunks for a question using the configured strategy."""
        run = check_index(db_path, strategy)
        config = Config.resolve()
        with mistral_client(config) as client:
            result = embed_texts(
                client,
                run.model,
                [question],
                week=week,
                day=day,
                journal_extra={"strategy": strategy, "command": "rag"},
            )
        found = index_module.search(db_path, strategy, result.vectors[0], k=k)
        raw = tuple(
            RagHit(
                chunk_id=h.chunk.chunk_id,
                source=h.chunk.source,
                section=h.chunk.section,
                score=h.score,
                text=h.chunk.text,
            )
            for h in found
        )
        hits = fit_hits(raw)
        return RagContext(
            hits=hits,
            strategy=strategy,
            k=k,
            embed_model=run.model,
            embed_tokens=result.prompt_tokens,
            corpus_rev=run.corpus_rev,
            dropped=len(raw) - len(hits),
        )

    return retrieve


@dataclass(frozen=True, slots=True)
class ControlQuestion:
    """A control question with expected facts and required sources."""

    id: int
    question: str
    expect: tuple[tuple[str, ...], ...]
    sources: tuple[str, ...]
    note: str = ""


def load_questions(path: Path) -> list[ControlQuestion]:
    """Load and validate control questions from a JSON file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AdventError(f"Файл контрольных вопросов не найден: {path.name}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdventError(f"Файл контрольных вопросов повреждён ({path.name}): {exc}") from exc

    if not isinstance(data, list):
        raise AdventError(f"Файл контрольных вопросов ({path.name}): ожидался список.")

    questions: list[ControlQuestion] = []
    seen_ids: set[int] = set()

    for n, item in enumerate(data, start=1):
        where = f"Файл контрольных вопросов ({path.name}), запись {n}"
        if not isinstance(item, dict):
            raise AdventError(f"{where}: ожидался объект")

        if "id" not in item or not isinstance(item["id"], int) or isinstance(item["id"], bool):
            raise AdventError(f"{where}: id должен быть целым числом")

        if (
            "question" not in item
            or not isinstance(item["question"], str)
            or not item["question"].strip()
        ):
            raise AdventError(f"{where}: question должен быть непустой строкой")

        expect = item.get("expect")
        if (
            expect is None
            or not isinstance(expect, list)
            or not expect
            or any(not isinstance(alt, list) or not alt for alt in expect)
            or any(not isinstance(a, str) or not a.strip() for alt in expect for a in alt)
        ):
            raise AdventError(
                f"{where}: expect должен быть непустым списком непустых списков строк"
            )

        sources = item.get("sources")
        if (
            sources is None
            or not isinstance(sources, list)
            or not sources
            or any(not isinstance(source, str) or not source.strip() for source in sources)
        ):
            raise AdventError(f"{where}: sources должен быть непустым списком строк")

        note = item.get("note", "")
        if not isinstance(note, str):
            raise AdventError(f"{where}: note должен быть строкой")

        question_id = item["id"]
        if question_id in seen_ids:
            raise AdventError(f"Файл контрольных вопросов ({path.name}): id повторяются.")
        seen_ids.add(question_id)

        questions.append(
            ControlQuestion(
                id=question_id,
                question=item["question"],
                expect=tuple(tuple(alts) for alts in expect),
                sources=tuple(sources),
                note=note,
            )
        )

    return questions


def validate_questions(
    questions: Sequence[ControlQuestion],
    chunks: Sequence[Chunk],
) -> tuple[list[ControlQuestion], list[tuple[ControlQuestion, str]]]:
    """Split questions into valid and broken ones against the indexed chunks."""
    text_by_source: dict[str, str] = {}
    for chunk in chunks:
        text_by_source[chunk.source] = (
            text_by_source.get(chunk.source, "") + normalize(chunk.text) + "\n"
        )

    valid: list[ControlQuestion] = []
    broken: list[tuple[ControlQuestion, str]] = []

    for question in questions:
        reason = None

        for source in question.sources:
            if source not in text_by_source:
                reason = f"источника {source} нет в индексе"
                break

        if reason is None:
            for fact in question.expect:
                supported = any(
                    normalize(alt) in text_by_source[source]
                    for source in question.sources
                    for alt in fact
                )
                if not supported:
                    reason = f"факт {' | '.join(fact)} не найден в источниках"
                    break

        if reason is None:
            valid.append(question)
        else:
            broken.append((question, reason))

    return valid, broken


@dataclass(frozen=True, slots=True)
class AnswerScore:
    """Score for an answer against a control question."""

    facts: tuple[bool, ...]
    sources_retrieved: tuple[bool, ...] | None
    sources_cited: tuple[bool, ...]

    @property
    def facts_found(self) -> int:
        """Return the number of facts found in the answer."""
        return sum(self.facts)

    @property
    def complete(self) -> bool:
        """Return whether all facts were found in the answer."""
        return all(self.facts)


def score_answer(question: ControlQuestion, answer: str, ctx: RagContext | None) -> AnswerScore:
    """Score an answer using facts, retrieved sources, and cited sources."""
    facts = check_facts(answer, question.expect)

    if ctx is None:
        retrieved: tuple[bool, ...] | None = None
    else:
        found = {hit.source for hit in ctx.hits}
        retrieved = tuple(source in found for source in question.sources)

    cited = cited_sources(answer, question.sources)

    return AnswerScore(
        facts=facts,
        sources_retrieved=retrieved,
        sources_cited=cited,
    )
