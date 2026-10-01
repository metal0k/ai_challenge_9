"""Week-agnostic RAG pieces: hit/context types, prompt assembly, fact checks."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from advent_core.errors import AdventError

RAG_MAX_CONTEXT_CHARS = 24_000
RAG_INSTRUCTION = (
    "Ответь на вопрос по фрагментам документации проекта ниже. "
    "Если ответа в них нет — скажи об этом прямо и не выдумывай. "
    "В конце назови источники, на которые опирался."
)
RAG_QUESTION_LABEL = "Вопрос:"
RAG_EMPTY_INSTRUCTION = (
    "В документации проекта не нашлось фрагментов, относящихся к вопросу. "
    "Скажи об этом прямо и не отвечай по памяти."
)
RAG_REWRITE_PROMPT = (
    "Перепиши вопрос пользователя в поисковый запрос по документации и коду "
    "Python-проекта: раскрой подразумеваемые термины, добавь вероятные "
    "идентификаторы и английские эквиваленты. Верни только запрос одной "
    "строкой.\n\nВопрос: {question}"
)
# "{n}" is substituted with str.replace: the JSON braces rule out str.format.
RAG_RERANK_INSTRUCTION = (
    "Оцени, насколько каждый фрагмент помогает ответить на вопрос. "
    "Шкала 0–10: 10 — фрагмент содержит прямой ответ, 0 — не относится. "
    'Верни JSON {"scores": [{"id": <номер>, "score": <0-10>}, ...]} '
    "для всех {n} фрагментов без пропусков."
)
RRF_K0 = 60
_REWRITE_STRIP = ' \t*`"«»'
_DIGIT_GAP_RE = re.compile(r"(?<=\d)\s+(?=\d)")


@dataclass(frozen=True, slots=True)
class RagHit:
    chunk_id: str
    source: str
    section: str
    score: float  # always cosine; never replaced by RRF or rerank values
    text: str
    rerank: float | None = None
    rank: int | None = None  # 1-based position in the pre-filter candidate list
    fused: float | None = None  # RRF sum, set only by rrf_merge


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    original: tuple[RagHit, ...]  # cosine top-k_before of the original question
    fused: tuple[RagHit, ...]  # candidate list before rerank (== original without rewrite)
    reranked: tuple[RagHit, ...]  # every candidate with scores, before the threshold


@dataclass(frozen=True, slots=True)
class RagSettings:
    strategy: str
    k: int
    k_before: int
    rewrite: bool
    rerank: bool
    threshold: float


@dataclass(frozen=True, slots=True)
class RagContext:
    hits: tuple[RagHit, ...]
    strategy: str
    k: int
    embed_model: str
    embed_tokens: int | None
    corpus_rev: str
    dropped: int = 0
    rewritten: str | None = None
    candidates: int = 0
    passed: int | None = None  # None without rerank
    threshold: float | None = None
    rerank_unrated: int = 0
    aux_prompt_tokens: int | None = None
    aux_completion_tokens: int | None = None
    warnings: tuple[str, ...] = ()
    trace: RetrievalTrace | None = None


RetrieveFn = Callable[[str, RagSettings], RagContext]


def fit_hits(hits: Sequence[RagHit], limit: int = RAG_MAX_CONTEXT_CHARS) -> tuple[RagHit, ...]:
    """Fit hits into a character limit."""
    if not hits:
        return ()

    first = hits[0]
    if len(first.text) > limit:
        first = replace(first, text=first.text[:limit])

    kept: list[RagHit] = [first]
    total = len(first.text)

    for hit in hits[1:]:
        if total + len(hit.text) > limit:
            break
        kept.append(hit)
        total += len(hit.text)

    return tuple(kept)


def clean_rewrite(text: str) -> str | None:
    """First non-empty line of a rewrite reply, markdown/quote wrapping removed."""
    for line in text.splitlines():
        cleaned = line.strip(_REWRITE_STRIP)
        if cleaned:
            return cleaned
    return None


def rrf_merge(lists: Sequence[Sequence[RagHit]], k0: int = RRF_K0) -> tuple[RagHit, ...]:
    """Reciprocal rank fusion over chunk_id; the first list's hit (its cosine) wins."""
    fused: dict[str, float] = {}
    first: dict[str, RagHit] = {}
    for hits in lists:
        for position, hit in enumerate(hits, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k0 + position)
            first.setdefault(hit.chunk_id, hit)
    order = sorted(fused, key=lambda cid: (-fused[cid], cid))
    return tuple(
        replace(first[cid], fused=fused[cid], rank=position)
        for position, cid in enumerate(order, start=1)
    )


def build_rerank_prompt(question: str, hits: Sequence[RagHit]) -> str:
    """Rerank request: instruction, question, numbered whole chunks, question again.

    The question before the fragments is measured: with it only at the end the reranker
    scored a relevant chunk 0-3 on some questions, with it in both places 7-10.
    """
    blocks: list[str] = []
    for n, hit in enumerate(hits, start=1):
        header = f"[{n}] {hit.source}"
        if hit.section.strip():
            header += f" — {hit.section}"
        blocks.append(header + "\n" + hit.text.strip())
    instruction = RAG_RERANK_INSTRUCTION.replace("{n}", str(len(hits)))
    line = f"{RAG_QUESTION_LABEL} {question}"
    return "\n\n".join([instruction, line, *blocks, line])


def parse_rerank(raw: str, n: int) -> dict[int, float]:
    """Scores by 1-based id for the chunks the reranker rated; AdventError if unusable."""
    hint = "Повторите вопрос или выключите rag_rerank."
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise AdventError("Reranker вернул не JSON.", hint=hint) from exc
    entries = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise AdventError("В ответе reranker'а нет списка scores.", hint=hint)
    scores: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ident, score = entry.get("id"), entry.get("score")
        if not isinstance(ident, int) or isinstance(ident, bool) or not 1 <= ident <= n:
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if isinstance(score, int):
            score = min(10, max(0, score))  # clamp before float(): 10**400 overflows
        if not math.isfinite(score):
            continue
        scores.setdefault(ident, min(10.0, max(0.0, float(score))))
    if not scores:
        raise AdventError("Reranker не оценил ни одного фрагмента.", hint=hint)
    return scores


def order_by_rerank(hits: Sequence[RagHit], scores: dict[int, float]) -> tuple[RagHit, ...]:
    """All hits: rated by score desc then position, unrated last with rerank=None."""
    rated: list[tuple[float, int, RagHit]] = []
    unrated: list[RagHit] = []
    for position, hit in enumerate(hits, start=1):
        shown = replace(hit, rank=hit.rank if hit.rank is not None else position)
        if position in scores:
            rated.append((scores[position], position, replace(shown, rerank=scores[position])))
        else:
            unrated.append(shown)
    rated.sort(key=lambda item: (-item[0], item[1]))
    return (*(item[2] for item in rated), *unrated)


def apply_rerank(
    hits: Sequence[RagHit], scores: dict[int, float], threshold: float, k: int
) -> tuple[tuple[RagHit, ...], int]:
    """Threshold then top-k, plus how many passed; unrated hits never pass."""
    passing = [
        hit
        for hit in order_by_rerank(hits, scores)
        if hit.rerank is not None and hit.rerank >= threshold
    ]
    return tuple(passing[:k]), len(passing)


def build_rag_prompt(question: str, hits: Sequence[RagHit], *, filtered_out: bool = False) -> str:
    """Build a RAG prompt from a question and hits."""
    if not hits:
        if filtered_out:
            return f"{RAG_EMPTY_INSTRUCTION}\n\n{RAG_QUESTION_LABEL} {question}"
        return question

    blocks: list[str] = []
    for n, hit in enumerate(hits, start=1):
        if hit.section.strip():
            header = f"[{n}] {hit.source} — {hit.section}"
        else:
            header = f"[{n}] {hit.source}"
        blocks.append(header + "\n" + hit.text.strip())

    return "\n\n".join([RAG_INSTRUCTION, *blocks, f"{RAG_QUESTION_LABEL} {question}"])


def normalize(text: str) -> str:
    """Normalize text for matching."""
    return _DIGIT_GAP_RE.sub("", text.casefold())


def _has_fact(haystack: str, needle: str) -> bool:
    """Substring match that refuses to land inside a longer number."""
    if not needle:
        return False
    pattern = re.escape(needle)
    if needle[0].isdigit():
        pattern = r"(?<![\d.,])" + pattern
    if needle[-1].isdigit():
        pattern = pattern + r"(?![.,]?\d)"
    return re.search(pattern, haystack) is not None


def fact_span(answer: str, alternatives: Sequence[str]) -> tuple[int, int] | None:
    """Span in the original answer of the first alternative `check_facts` would accept.

    Same rules as `_has_fact` on `normalize`d text: case-insensitive, whitespace allowed
    between digits, no landing inside a longer number (also across a digit gap).
    IGNORECASE vs casefold can disagree ("Straße"/"strasse"): scoring always goes through
    `check_facts`, so a disagreement only makes the snippet fall back to the answer start.
    """
    for alt in alternatives:
        needle = normalize(alt)
        if not needle:
            continue
        parts: list[str] = []
        for i, ch in enumerate(needle):
            if i and ch.isdigit() and needle[i - 1].isdigit():
                parts.append(r"\s*")
            parts.append(re.escape(ch))
        pattern = "".join(parts)
        if needle[0].isdigit():
            pattern = r"(?<![\d.,])" + pattern
        if needle[-1].isdigit():
            pattern = pattern + r"(?![.,]?\d)"
        for match in re.finditer(pattern, answer, re.IGNORECASE):
            start, end = match.span()
            # normalize() glues digits across whitespace, so "1 " + "262 144" is one number.
            if needle[0].isdigit() and re.search(r"\d\s+\Z", answer[:start]):
                continue
            if needle[-1].isdigit() and re.match(r"\s+\d", answer[end:]):
                continue
            return start, end
    return None


def _has_path(haystack: str, path: str) -> bool:
    """Match a file path that is not the tail of a longer path."""
    if not path:
        return False
    pattern = r"(?<![\w/.-])" + re.escape(path) + r"(?![\w/-])"
    return re.search(pattern, haystack) is not None


def check_facts(answer: str, facts: Sequence[Sequence[str]]) -> tuple[bool, ...]:
    """Check whether each fact is present in the answer."""
    haystack = normalize(answer)
    results: list[bool] = []
    for fact in facts:
        results.append(any(_has_fact(haystack, normalize(alt)) for alt in fact))
    return tuple(results)


def cited_sources(answer: str, sources: Sequence[str]) -> tuple[bool, ...]:
    """Check which sources are cited in the answer."""
    haystack = normalize(answer)
    results: list[bool] = []
    for source in sources:
        base = source.rsplit("/", 1)[-1]
        if _has_path(haystack, normalize(source)):
            results.append(True)
            continue
        shared = sum(1 for other in sources if other.rsplit("/", 1)[-1] == base)
        results.append("/" in source and shared == 1 and _has_path(haystack, normalize(base)))
    return tuple(results)
