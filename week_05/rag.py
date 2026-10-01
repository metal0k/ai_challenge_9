"""Day 22: retriever over the SQLite index and the control-question set."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from advent_core import chat as chat_core
from advent_core.chat import Message
from advent_core.client import mistral_client
from advent_core.config import Config
from advent_core.embeddings import embed_texts
from advent_core.errors import AdventError
from advent_core.journal import log_call
from advent_core.params import GenerationParams
from advent_core.rag import (
    RAG_REWRITE_FOLLOWUP_PROMPT,
    RAG_REWRITE_PROMPT,
    CitedAnswer,
    RagContext,
    RagHit,
    RagSettings,
    RetrievalTrace,
    RetrieveFn,
    apply_rerank,
    build_rerank_prompt,
    check_facts,
    cited_sources,
    clean_rewrite,
    fit_hits,
    normalize,
    order_by_rerank,
    parse_rerank,
    rrf_merge,
    says_unknown,
)
from advent_core.telemetry import CallResult
from week_05 import index as index_module
from week_05.chunking import Chunk

RAG_WEEK = 5
RAG_DAY = 22
RAG_AUX_DAY = 23
RAG_CITE_DAY = 24
JUDGE_MAX_TOKENS = 300
RAG_AUX_MODEL = "ministral-14b-latest"
REWRITE_MAX_TOKENS = 200
RERANK_MAX_TOKENS = 2000


def check_index(db_path: Path | None, strategy: str) -> index_module.RunInfo:
    """Check that the requested strategy exists in the index."""
    runs = index_module.load_runs(db_path)
    if strategy not in runs:
        raise AdventError(
            f"В индексе нет стратегии {strategy!r}.",
            hint=f"Сначала `adventrag index --strategy {strategy}`.",
        )
    return runs[strategy]


def _aux_call(
    prompt: str, *, command: str, max_tokens: int, json_mode: bool, day: int
) -> CallResult:
    """One isolated helper call: nothing from .env params, no system persona.

    Journaled here, before the caller parses the reply, so a paid call is on record
    even when parsing then fails.
    """
    config = dataclasses.replace(
        Config.resolve(model=RAG_AUX_MODEL, stream=False),
        params=GenerationParams(
            temperature=0, max_tokens=max_tokens, format="json" if json_mode else None
        ),
    )
    messages: list[Message] = [{"role": "user", "content": prompt}]
    result = chat_core.complete(config, messages)
    log_call(
        result,
        list(result.sent_messages or messages),
        week=RAG_WEEK,
        day=day,
        extra={"command": command},
    )
    return result


def _to_hits(found: Sequence[object]) -> tuple[RagHit, ...]:
    return tuple(
        RagHit(
            chunk_id=h.chunk.chunk_id,
            source=h.chunk.source,
            section=h.chunk.section,
            score=h.score,
            text=h.chunk.text,
            rank=position,
        )
        for position, h in enumerate(found, start=1)
    )


def _add_tokens(total: int | None, value: int | None, first: bool) -> int | None:
    """Sum token counts; one unknown value makes the whole sum unknown."""
    if first:
        return value
    if total is None or value is None:
        return None
    return total + value


def rewrite_prompt(question: str, previous: str | None) -> str:
    """Rewrite prompt; a follow-up (previous question known) uses the follow-up wording."""
    if previous:
        values = {"{previous}": previous, "{question}": question}
        # Single pass: user text containing a placeholder must not be substituted.
        return re.sub(
            r"\{previous\}|\{question\}",
            lambda m: values[m.group(0)],
            RAG_REWRITE_FOLLOWUP_PROMPT,
        )
    return RAG_REWRITE_PROMPT.replace("{question}", question)


def make_retriever(
    db_path: Path | None = None,
    *,
    week: int = RAG_WEEK,
    day: int = RAG_DAY,
    aux_day: int = RAG_AUX_DAY,
) -> RetrieveFn:
    """Create a retriever bound to a database path and journal coordinates.

    `day` is the embedding row's day, `aux_day` the rewrite/rerank rows' day.
    """

    def retrieve(question: str, settings: RagSettings) -> RagContext:
        """Rewrite -> embed -> search -> RRF -> rerank -> threshold -> top-k."""
        strategy, k = settings.strategy, settings.k
        staged = settings.rewrite or settings.rerank
        n = settings.k_before if staged else k
        run = check_index(db_path, strategy)
        warnings: list[str] = []
        aux_calls: list[CallResult] = []

        rewritten: str | None = None
        if settings.rewrite:
            result = _aux_call(
                rewrite_prompt(question, settings.previous),
                command="rag_rewrite",
                max_tokens=REWRITE_MAX_TOKENS,
                json_mode=False,
                day=aux_day,
            )
            aux_calls.append(result)
            rewritten = clean_rewrite(result.text)
            if rewritten is None:
                warnings.append("rewrite вернул пустую строку — поиск только по исходному вопросу")

        texts = [question] if rewritten is None else [question, rewritten]
        config = Config.resolve()
        with mistral_client(config) as client:
            embedded = embed_texts(
                client,
                run.model,
                texts,
                week=week,
                day=day,
                journal_extra={"strategy": strategy, "command": "rag"},
            )
        lists = [
            _to_hits(index_module.search(db_path, strategy, embedded.vectors[i], k=n))
            for i in range(len(texts))
        ]
        original = lists[0]
        fused = rrf_merge(lists) if rewritten is not None else original

        reranked: tuple[RagHit, ...] = ()
        passed: int | None = None
        unrated = 0
        if settings.rerank:
            result = _aux_call(
                build_rerank_prompt(question, fused),
                command="rag_rerank",
                max_tokens=RERANK_MAX_TOKENS,
                json_mode=True,
                day=aux_day,
            )
            aux_calls.append(result)
            scores = parse_rerank(result.text, len(fused))
            unrated = len(fused) - len(scores)
            if unrated:
                warnings.append(f"reranker не оценил {unrated} из {len(fused)} чанков")
            reranked = order_by_rerank(fused, scores)
            raw, passed = apply_rerank(fused, scores, settings.threshold, k)
        else:
            raw = fused[:k]

        hits = fit_hits(raw)
        aux_prompt: int | None = None
        aux_completion: int | None = None
        for i, call in enumerate(aux_calls):
            aux_prompt = _add_tokens(aux_prompt, call.usage.prompt_tokens, i == 0)
            aux_completion = _add_tokens(aux_completion, call.usage.completion_tokens, i == 0)
        return RagContext(
            hits=hits,
            strategy=strategy,
            k=k,
            embed_model=run.model,
            embed_tokens=embedded.prompt_tokens,
            corpus_rev=run.corpus_rev,
            dropped=len(raw) - len(hits),
            rewritten=rewritten,
            candidates=len(fused) if staged else 0,
            passed=passed,
            threshold=settings.threshold if settings.rerank else None,
            rerank_unrated=unrated,
            aux_prompt_tokens=aux_prompt,
            aux_completion_tokens=aux_completion,
            warnings=tuple(warnings),
            trace=RetrievalTrace(original, fused, reranked) if staged else None,
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
                    check_facts(text_by_source[source], [fact])[0] for source in question.sources
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


def score_answer(
    question: ControlQuestion,
    answer: str,
    ctx: RagContext | None,
    cited: CitedAnswer | None = None,
) -> AnswerScore:
    """Score an answer using facts, retrieved sources, and cited sources.

    Cite mode scores only `cited.answer` of an accepted answer (a refusal and an
    unverified draft score zero facts) and takes sources from the verbatim quotes.
    """
    if cited is not None:
        answer = cited.answer if cited.quoted else ""
    facts = check_facts(answer, question.expect)

    if ctx is None:
        retrieved: tuple[bool, ...] | None = None
    else:
        found = {hit.source for hit in ctx.hits}
        retrieved = tuple(source in found for source in question.sources)

    if cited is not None:
        quoted = {hit.source for hit in cited.quoted_sources}
        cited_flags = tuple(source in quoted for source in question.sources)
    else:
        cited_flags = cited_sources(answer, question.sources)

    return AnswerScore(
        facts=facts,
        sources_retrieved=retrieved,
        sources_cited=cited_flags,
    )


def relevant_chunk_ids(q: ControlQuestion, chunks: Sequence[Chunk]) -> frozenset[str]:
    """Chunks from the question's sources that hold every expected fact."""
    return frozenset(
        c.chunk_id for c in chunks if c.source in q.sources and all(check_facts(c.text, q.expect))
    )


# --- Day 24: grounding judge, questions without an answer ---------------------------------

JUDGE_VERDICTS = ("да", "частично", "нет")
JUDGE_INSTRUCTION = (
    "Ты проверяешь, подтверждают ли цитаты ответ. Ниже вопрос, ответ и цитаты из "
    "документации. Подтверждают ли цитаты каждое утверждение ответа? Оценивай только "
    "по цитатам, свои знания не используй. Верни JSON "
    '{"verdict": "да" | "частично" | "нет", "reason": "одна короткая фраза"}.'
)


@dataclass(frozen=True, slots=True)
class Grounding:
    """Judge verdict on answer-vs-quotes; `verdict` None means unreadable or not asked."""

    verdict: str | None
    reason: str
    result: CallResult | None


def build_judge_prompt(question: str, cited: CitedAnswer, hits: Sequence[RagHit]) -> str:
    """Question, answer and verified quotes only: no expected facts, no whole fragments."""
    quotes = [
        f"[{q.n}] «{q.text.strip()}»" for q in cited.quotes if q.verified and 1 <= q.n <= len(hits)
    ]
    return "\n\n".join(
        [
            JUDGE_INSTRUCTION,
            f"Вопрос: {question}",
            f"Ответ: {cited.answer}",
            "Цитаты:\n" + "\n".join(quotes),
        ]
    )


def parse_judge(raw: object) -> tuple[str | None, str]:
    """Total: anything but a clean verdict is (None, reason-if-any)."""
    try:
        if not isinstance(raw, str):
            return None, ""
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            return None, ""
        reason = data.get("reason")
        reason = " ".join(reason.split()) if isinstance(reason, str) else ""
        verdict = data.get("verdict")
        if isinstance(verdict, str) and verdict.strip().casefold() in JUDGE_VERDICTS:
            return verdict.strip().casefold(), reason
        return None, reason
    except Exception:  # noqa: BLE001 - garbage from the model must never raise
        return None, ""


def judge_grounding(
    cited: CitedAnswer,
    hits: Sequence[RagHit],
    question: str,
    *,
    day: int = RAG_CITE_DAY,
) -> Grounding:
    """Ask the isolated judge whether the verbatim quotes support the answer.

    Only a `quoted` answer is judged; a refusal makes no call. The call is journaled
    inside `_aux_call` before the reply is parsed.
    """
    if not cited.quoted:
        return Grounding(None, "", None)
    result = _aux_call(
        build_judge_prompt(question, cited, hits),
        command="rag_judge",
        max_tokens=JUDGE_MAX_TOKENS,
        json_mode=True,
        day=day,
    )
    verdict, reason = parse_judge(result.text)
    return Grounding(verdict, reason, result)


@dataclass(frozen=True, slots=True)
class UnanswerableQuestion:
    """A question the repo cannot answer: the expected outcome is a refusal."""

    id: int
    question: str
    note: str = ""


def load_unanswerable(path: Path) -> list[UnanswerableQuestion]:
    """Load and validate the questions without an answer."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AdventError(f"Файл вопросов без ответа не найден: {path.name}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdventError(f"Файл вопросов без ответа повреждён ({path.name}): {exc}") from exc
    if not isinstance(data, list):
        raise AdventError(f"Файл вопросов без ответа ({path.name}): ожидался список.")
    out: list[UnanswerableQuestion] = []
    seen: set[int] = set()
    for n, item in enumerate(data, start=1):
        where = f"Файл вопросов без ответа ({path.name}), запись {n}"
        if not isinstance(item, dict):
            raise AdventError(f"{where}: ожидался объект")
        ident = item.get("id")
        if not isinstance(ident, int) or isinstance(ident, bool):
            raise AdventError(f"{where}: id должен быть целым числом")
        text = item.get("question")
        if not isinstance(text, str) or not text.strip():
            raise AdventError(f"{where}: question должен быть непустой строкой")
        note = item.get("note", "")
        if not isinstance(note, str):
            raise AdventError(f"{where}: note должен быть строкой")
        if ident in seen:
            raise AdventError(f"Файл вопросов без ответа ({path.name}): id повторяются.")
        seen.add(ident)
        out.append(UnanswerableQuestion(ident, text, note))
    return out


REFUSAL_REASONS = ("empty_context", "model_unknown", "no_index")


def unanswerable_outcome(text: str, cited: CitedAnswer | None) -> str:
    """`refused` | `unverified` | `format` | `answered`; only `refused` is the right outcome.

    Cite mode reads the structured result; other modes use the `says_unknown` heuristic.
    """
    if cited is None:
        return "refused" if says_unknown(text) else "answered"
    if cited.status == "answer":
        return "answered"
    if cited.reason in REFUSAL_REASONS:
        return "refused"
    if cited.reason == "unverified":
        return "unverified"
    return "format"
