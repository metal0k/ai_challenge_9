"""Week-agnostic RAG pieces: hit/context types, prompt assembly, fact checks."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

RAG_MAX_CONTEXT_CHARS = 24_000
RAG_INSTRUCTION = (
    "Ответь на вопрос по фрагментам документации проекта ниже. "
    "Если ответа в них нет — скажи об этом прямо и не выдумывай. "
    "В конце назови источники, на которые опирался."
)
RAG_QUESTION_LABEL = "Вопрос:"
_DIGIT_GAP_RE = re.compile(r"(?<=\d)\s+(?=\d)")


@dataclass(frozen=True, slots=True)
class RagHit:
    chunk_id: str
    source: str
    section: str
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class RagContext:
    hits: tuple[RagHit, ...]
    strategy: str
    k: int
    embed_model: str
    embed_tokens: int | None
    corpus_rev: str
    dropped: int = 0


RetrieveFn = Callable[[str, str, int], RagContext]  # (question, strategy, k)


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


def build_rag_prompt(question: str, hits: Sequence[RagHit]) -> str:
    """Build a RAG prompt from a question and hits."""
    if not hits:
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
