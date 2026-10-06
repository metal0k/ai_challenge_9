"""Day 22: retriever over the SQLite index and the control-question set."""

from __future__ import annotations

import dataclasses
import json
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
    RAG_REWRITE_PROMPT,
    CitedAnswer,
    RagContext,
    RagHit,
    RagSettings,
    RetrievalTrace,
    RetrieveFn,
    apply_rerank,
    build_rerank_prompt,
    build_rewrite_prompt,
    check_facts,
    cited_sources,
    clean_rewrite,
    fit_hits,
    has_dialog_context,
    normalize,
    order_by_rerank,
    parse_rerank,
    resolved_question,
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
RAG_TASK_DAY = 25  # rewrite/rerank that saw dialog context
JUDGE_MAX_TOKENS = 300
RAG_AUX_MODEL = "ministral-14b-latest"
REWRITE_MAX_TOKENS = 200
RERANK_MAX_TOKENS = 2000
# A reasoning local model spends max_tokens on its chain of thought first: at 200 the
# rewrite returned nothing ("reasoning съел max_tokens"), so a local aux call gets a floor.
LOCAL_AUX_MAX_TOKENS = 4096


LOCAL_RERANK_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "rerank",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        },
                        "required": ["id", "score"],
                    },
                }
            },
            "required": ["scores"],
        },
    },
}


def local_rerank_cap(n: int) -> int:
    """Output budget for a reasoning-off local rerank: the JSON only, so a runaway stops early."""
    return max(512, 64 + 24 * n)


def check_index(db_path: Path | None, strategy: str) -> index_module.RunInfo:
    """Check that the strategy exists and that the index endpoint fits the mode.

    Offline (`--local`) accepts only a local index: a cloud one has another
    dimension and would die on search with a 1024-vs-768 error.
    """
    runs = index_module.load_runs(db_path)
    if strategy not in runs:
        raise AdventError(
            f"В индексе нет стратегии {strategy!r}.",
            hint=f"Сначала `adventrag index --strategy {strategy}`.",
        )
    run = runs[strategy]
    if index_module.index_endpoint_mode() == index_module.ENDPOINT_LOCAL and (
        getattr(run, "endpoint", index_module.ENDPOINT_CLOUD) != index_module.ENDPOINT_LOCAL
    ):
        raise AdventError(
            f"Индекс {index_module.display_path(db_path)} собран через облако "
            f"(endpoint={run.endpoint}, модель {run.model}), а режим локальный.",
            hint="Собери локальный индекс: `adventrag index --local` "
            "(по умолчанию data/rag/index.local.sqlite3).",
        )
    return run


def _aux_call(
    prompt: str,
    *,
    command: str,
    max_tokens: int,
    json_mode: bool,
    day: int,
    aux_config: Config | None = None,
    reasoning: bool = True,
    local_cap: int | None = None,
    rerank_schema: bool = False,
) -> CallResult:
    """One isolated helper call: nothing from .env params, no system persona.

    Journaled here, before the caller parses the reply, so a paid call is on record
    even when parsing then fails.
    """
    base = (
        dataclasses.replace(aux_config, stream=False)
        if aux_config is not None
        else Config.resolve(model=RAG_AUX_MODEL, stream=False)
    )
    # LM Studio rejects response_format json_object (accepts only json_schema/text):
    # on a local server the prompt alone asks for JSON and parse_rerank copes with fences.
    use_json = json_mode and not base.is_local
    # Local rerank: grammar-enforced json_schema (LM Studio) stops the prose-before-JSON runaway.
    wire_format = LOCAL_RERANK_RESPONSE_FORMAT if base.is_local and rerank_schema else None
    if base.is_local:
        if local_cap is not None and not reasoning:
            max_tokens = local_cap  # no chain of thought to pay for: cap at what the JSON needs
        else:
            max_tokens = max(max_tokens, LOCAL_AUX_MAX_TOKENS)
    # Top-level reasoning_effort="none" is what switches a local reasoning model off
    # ("low" does not); a cloud aux call never gets it.
    effort = "none" if base.is_local and not reasoning else None
    config = dataclasses.replace(
        base,
        params=GenerationParams(
            temperature=0,
            max_tokens=max_tokens,
            format="json" if use_json else None,
            reasoning_effort=effort,
            response_format=wire_format,
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


def rewrite_prompt(
    question: str,
    previous: str | None,
    task: str | None = None,
    last_question: str | None = None,
    last_answer: str | None = None,
) -> str:
    """Rewrite prompt: day 23/24 wording without dialog context, the task one with any."""
    if has_dialog_context(task, last_question, previous):
        return build_rewrite_prompt(
            question,
            task=task,
            last_question=last_question,
            previous=previous,
            last_answer=last_answer,
        )
    return RAG_REWRITE_PROMPT.replace("{question}", question)


def make_retriever(
    db_path: Path | None = None,
    *,
    week: int = RAG_WEEK,
    day: int = RAG_DAY,
    aux_day: int = RAG_AUX_DAY,
    aux_config: Config | None = None,
) -> RetrieveFn:
    """Create a retriever bound to a database path and journal coordinates.

    `day` is the embedding row's day, `aux_day` the rewrite/rerank rows' day.
    `aux_config` replaces the cloud `RAG_AUX_MODEL` config for rewrite/rerank
    (the agent's own model/base_url in `--local`); the query embedding follows
    the index run's endpoint, model and prefix, not the config.
    """

    def retrieve(question: str, settings: RagSettings) -> RagContext:
        """Rewrite -> embed -> search -> RRF -> rerank -> threshold -> top-k."""
        strategy, k = settings.strategy, settings.k
        staged = settings.rewrite or settings.rerank
        n = settings.k_before if staged else k
        run = check_index(db_path, strategy)
        warnings: list[str] = []
        aux_calls: list[CallResult] = []

        contextual = has_dialog_context(settings.task, settings.last_question, settings.previous)
        stage_day = RAG_TASK_DAY if contextual else aux_day

        def rewrite_once(prompt: str, rewrite_day: int, label: str) -> str | None:
            # With dialog context a failed rewrite only narrows the search; without it
            # the single rewrite keeps its old contract and the error propagates.
            try:
                result = _aux_call(
                    prompt,
                    command="rag_rewrite",
                    max_tokens=REWRITE_MAX_TOKENS,
                    json_mode=False,
                    day=rewrite_day,
                    aux_config=aux_config,
                    reasoning=settings.aux_reasoning,
                )
            except AdventError as exc:
                if not contextual:
                    raise
                warnings.append(f"{label}rewrite не удался ({exc}) — поиск без него")
                return None
            aux_calls.append(result)
            cleaned = clean_rewrite(result.text)
            if cleaned is None:
                tail = "поиск без него" if contextual else "поиск только по исходному вопросу"
                warnings.append(f"{label}rewrite вернул пустую строку — {tail}")
            return cleaned

        # Context-free rewrite (day 23 prompt) is a search text of its own with context,
        # so a polluted context-aware rewrite cannot drown the result.
        rewritten: str | None = None
        texts = [question]
        if settings.rewrite:
            if contextual:
                plain = rewrite_once(
                    RAG_REWRITE_PROMPT.replace("{question}", question), aux_day, "plain "
                )
                aware = rewrite_once(
                    rewrite_prompt(
                        question,
                        settings.previous,
                        settings.task,
                        settings.last_question,
                        settings.last_answer,
                    ),
                    stage_day,
                    "context ",
                )
                texts += [t for t in (plain, aware) if t is not None]
                rewritten = aware or plain
            else:
                rewritten = rewrite_once(rewrite_prompt(question, None), aux_day, "")
                if rewritten is not None:
                    texts.append(rewritten)
        journal_extra = {"strategy": strategy, "command": "rag"}
        if getattr(run, "endpoint", index_module.ENDPOINT_CLOUD) == index_module.ENDPOINT_LOCAL:
            embedded = index_module.embed_queries_local(
                run,
                texts,
                base_url=aux_config.base_url if aux_config is not None else None,
                week=week,
                day=day,
                journal_extra=journal_extra,
            )
        else:
            with mistral_client(Config.resolve()) as client:
                embedded = embed_texts(
                    client,
                    run.model,
                    texts,
                    week=week,
                    day=day,
                    journal_extra=journal_extra,
                )
        lists = [
            _to_hits(index_module.search(db_path, strategy, embedded.vectors[i], k=n))
            for i in range(len(texts))
        ]
        original = lists[0]
        fused = rrf_merge(lists) if len(lists) > 1 else original

        reranked: tuple[RagHit, ...] = ()
        passed: int | None = None
        unrated = 0
        if settings.rerank:
            resolved = resolved_question(
                question,
                task=settings.task,
                last_question=settings.last_question,
                previous=settings.previous,
            )
            local = aux_config is not None and aux_config.is_local
            # Local path: one retry on a non-JSON/ambiguous answer, same settings; cloud: none.
            for attempt in range(2 if local else 1):
                result = _aux_call(
                    build_rerank_prompt(resolved, fused, local=local),
                    command="rag_rerank",
                    max_tokens=RERANK_MAX_TOKENS,
                    json_mode=True,
                    day=stage_day,
                    aux_config=aux_config,
                    reasoning=settings.aux_reasoning,
                    local_cap=local_rerank_cap(len(fused)) if local else None,
                    rerank_schema=True,
                )
                aux_calls.append(result)
                try:
                    scores = parse_rerank(result.text, len(fused), tolerant=local)
                    break
                except AdventError:
                    if attempt == 1 or not local:
                        raise
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
            task_used=bool(settings.task),
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
