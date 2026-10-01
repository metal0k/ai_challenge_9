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
# Follow-up after a cite refusal: only the rewrite sees the previous question,
# retrieval's own query stays the bare clarification (a premise the clarification
# cancels would otherwise be searched for again).
RAG_REWRITE_FOLLOWUP_PROMPT = (
    "Перепиши уточнение пользователя в поисковый запрос по документации и коду "
    "Python-проекта: раскрой подразумеваемые термины, добавь вероятные "
    "идентификаторы и английские эквиваленты. Сначала пользователь задал "
    "предыдущий вопрос и получил отказ, затем уточнил. Уточнение важнее "
    "предыдущего вопроса: посылки предыдущего вопроса, которые уточнение "
    "отменяет, отбрось. Верни только запрос одной строкой.\n\n"
    "Предыдущий вопрос: {previous}\n\nУточнение: {question}"
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
    cite: bool = False  # day 24: JSON answer with verbatim quotes, "don't know" refusal
    # Day 24: question a cite refusal answered; only the rewrite step reads it.
    previous: str | None = None


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


# --- Day 24: cited answers (SPEC-w05d24.md §2 + §9a) ---

RAG_MIN_QUOTE_CHARS = 12  # after normalisation; shorter is not a quote ("1.5")
RAG_CITE_INSTRUCTION = (
    "Ответь только по фрагментам документации ниже. Верни JSON "
    '{"status": "answer" или "unknown", "answer": "...", "sources": [номера фрагментов], '
    '"quotes": [{"id": номер фрагмента, "text": "дословная цитата"}]}. '
    "Цитата — точная копия куска фрагмента, 1–2 предложения, без пересказа и без "
    "форматирования. Каждое утверждение ответа подтверждается хотя бы одной цитатой. "
    'Если во фрагментах ответа нет — "status": "unknown", остальные поля пустые.'
)
RAG_UNKNOWN_PREFIX = "Не знаю:"
RAG_UNKNOWN_TEXTS = {
    "empty_context": "в документации проекта не нашлось фрагментов, относящихся к вопросу",
    "model_unknown": "во фрагментах документации нет ответа на вопрос",
    "unverified": "ни одна цитата модели не нашлась дословно во фрагментах",
    "bad_json": "ответ модели не удалось разобрать",
    "truncated": "ответ модели оборван по длине",
    "no_index": "индекс документации недоступен",
}
RAG_CLARIFY_HEADER = "Уточните вопрос — возможно, вы про:"
RAG_CLARIFY_MAX = 3
RAG_UNKNOWN_PHRASES = (
    "не знаю",
    "нет информации",
    "не нашлось",
    "не нашёл",
    "не нашел",
    "нет в документации",
    "в документации нет",
    "нет данных",
    "не упоминается",
    "не содержится",
)
_SAYS_UNKNOWN_HEAD = 200
_SAYS_UNKNOWN_MAX_LEN = 400
_QUOTE_EDGE = "«»“”„\"' .…"
_SPACES_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Quote:
    n: int  # 1-based fragment number in the context
    text: str  # as the model returned it
    verified: bool  # verbatim in the text of fragment n
    claimed: int | None = None  # the model's original id when n was re-attributed


@dataclass(frozen=True, slots=True)
class CitedAnswer:
    status: str  # "answer" | "unknown"
    answer: str  # "" unless status == "answer"
    sources: tuple[RagHit, ...]  # declared ids in range, ordered by fragment number
    quoted_sources: tuple[RagHit, ...]  # fragments of verbatim quotes, ordered by number
    quotes: tuple[Quote, ...]
    reason: str  # "" | empty_context | model_unknown | unverified | bad_json | truncated | no_index
    clarify: tuple[str, ...] = ()  # "source — section" of the nearest candidates
    draft: str = ""  # unverified only: the model's rejected answer, screen-only

    @property
    def verified_quotes(self) -> int:
        return sum(1 for q in self.quotes if q.verified)

    @property
    def quoted(self) -> bool:
        """An answer with at least one verbatim quote; says nothing about meaning."""
        return self.status == "answer" and self.verified_quotes > 0


def quote_norm(text: str) -> str:
    """Strip bold/code markers and edge quotes, collapse spaces, casefold.

    Underscores and single asterisks inside stay: `rag_cite` must not match `ragcite`.
    """
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = _SPACES_RE.sub(" ", text).strip()
    return text.strip(_QUOTE_EDGE).casefold()


def quote_in(quote: str, chunk_text: str) -> bool:
    """Verbatim (after quote_norm on both sides) and at least RAG_MIN_QUOTE_CHARS long."""
    needle = quote_norm(quote)
    return len(needle) >= RAG_MIN_QUOTE_CHARS and needle in quote_norm(chunk_text)


def build_cite_prompt(question: str, hits: Sequence[RagHit]) -> str:
    """Cite request: instruction, question, numbered whole chunks, question again."""
    blocks: list[str] = []
    for n, hit in enumerate(hits, start=1):
        header = f"[{n}] {hit.source}"
        if hit.section.strip():
            header += f" — {hit.section}"
        blocks.append(header + "\n" + hit.text.strip())
    line = f"{RAG_QUESTION_LABEL} {question}"
    return "\n\n".join([RAG_CITE_INSTRUCTION, line, *blocks, line])


def _label(hit: RagHit) -> str:
    return f"{hit.source} — {hit.section}" if hit.section.strip() else hit.source


def _clarify(near: Sequence[RagHit]) -> tuple[str, ...]:
    seen: list[str] = []
    for hit in near:
        label = _label(hit)
        if label not in seen:
            seen.append(label)
        if len(seen) == RAG_CLARIFY_MAX:
            break
    return tuple(seen)


def unknown_answer(reason: str, near: Sequence[RagHit] = ()) -> CitedAnswer:
    """Refusal; clarify lists up to three distinct nearest `source — section`."""
    return CitedAnswer("unknown", "", (), (), (), reason, _clarify(near))


def _valid_id(value: object, n: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= n


def _parse_cited(raw: str, hits: Sequence[RagHit]) -> CitedAnswer:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        return unknown_answer("bad_json")
    status = data.get("status")
    if not isinstance(status, str) or status not in ("answer", "unknown"):
        return unknown_answer("bad_json")
    if status == "unknown":
        return unknown_answer("model_unknown", hits)
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return unknown_answer("bad_json")
    n = len(hits)
    declared = data.get("sources")
    ids = {i for i in declared if _valid_id(i, n)} if isinstance(declared, list) else set()
    quotes: list[Quote] = []
    entries = data.get("quotes")
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, dict):
            continue
        ident, body = entry.get("id"), entry.get("text")
        if (
            not isinstance(ident, int)
            or isinstance(ident, bool)
            or not isinstance(body, str)
            or not body.strip()
        ):
            continue
        in_range = _valid_id(ident, n)
        if in_range and quote_in(body, hits[ident - 1].text):
            quotes.append(Quote(ident, body, True))
            continue
        # Right sentence, wrong label (incl. out-of-range id): lowest fragment that holds it.
        found = next((i for i, h in enumerate(hits, start=1) if quote_in(body, h.text)), None)
        if found is None:
            # Out-of-range and found nowhere: no fragment to show it against, drop.
            if in_range:
                quotes.append(Quote(ident, body, False))
        else:
            quotes.append(Quote(found, body, True, claimed=ident))
    quoted_ids = {q.n for q in quotes if q.verified}
    sources = tuple(hits[i - 1] for i in sorted(ids))
    if not quoted_ids:
        return CitedAnswer(
            "unknown", "", sources, (), tuple(quotes), "unverified", _clarify(hits), answer.strip()
        )
    quoted_sources = tuple(hits[i - 1] for i in sorted(quoted_ids))
    return CitedAnswer("answer", answer.strip(), sources, quoted_sources, tuple(quotes), "")


def parse_cited(raw: object, hits: Sequence[RagHit], truncated: bool = False) -> CitedAnswer:
    """Total: any input yields a CitedAnswer, never an exception (the call is already paid)."""
    if truncated:
        return unknown_answer("truncated")
    try:
        if not isinstance(raw, str):
            return unknown_answer("bad_json")
        return _parse_cited(raw, hits)
    except Exception:  # noqa: BLE001 - invalid JSON, recursion depth, anything in the payload
        return unknown_answer("bad_json")


def render_cited(c: CitedAnswer, hits: Sequence[RagHit], *, for_history: bool) -> str:
    """Screen/history text; numbers are context numbers. History omits the unverified draft."""
    if not c.quoted:
        reason = RAG_UNKNOWN_TEXTS.get(c.reason, c.reason)
        lines = [f"{RAG_UNKNOWN_PREFIX} {reason}."]
        if c.reason == "unverified" and c.draft and not for_history:
            lines.append(f"неподтверждённый ответ модели: «{c.draft}»")
        if c.clarify:
            lines.append(RAG_CLARIFY_HEADER)
            lines.extend(f"· {label}" for label in c.clarify)
        return "\n".join(lines)

    number = {hit.chunk_id: i for i, hit in enumerate(hits, start=1)}
    quoted_ids = {number.get(h.chunk_id) for h in c.quoted_sources}
    shown = {number[h.chunk_id]: h for h in (*c.sources, *c.quoted_sources) if h.chunk_id in number}
    lines = [c.answer, "", "Источники:"]
    for n in sorted(shown):
        mark = " · цитата ✓" if n in quoted_ids else ""
        lines.append(f"[{n}] {_label(shown[n])} · {shown[n].chunk_id}{mark}")
    if c.quotes:
        lines.append("Цитаты:")
        for q in c.quotes:
            note = f" (модель указала [{q.claimed}])" if q.claimed is not None else ""
            lines.append(f"[{q.n}] {'✓' if q.verified else '✗'}{note} «{q.text.strip()}»")
    return "\n".join(lines)


def says_unknown(text: str) -> bool:
    """Heuristic for non-cite modes: a refusal phrase up front in a short answer."""
    stripped = text.strip()
    if not stripped or len(stripped) > _SAYS_UNKNOWN_MAX_LEN:
        return False
    head = stripped[:_SAYS_UNKNOWN_HEAD].casefold()
    return any(phrase in head for phrase in RAG_UNKNOWN_PHRASES)
