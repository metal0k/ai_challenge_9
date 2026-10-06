"""Embedding batches: char-budget batching, adaptive split on server overflow.

Mirrors advent_core/client.py's retry story: the SDK client already carries a
RetryConfig for 429/5xx (SPEC-w05d21.md §11), so this module does not retry —
it only turns the two documented 400 shapes into recoverable batching moves
and everything else into an AdventError via errors.translate().
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from advent_core import openai_compat
from advent_core.config import redact
from advent_core.errors import AdventError, translate
from advent_core.journal import log_internal_call
from advent_core.telemetry import CallResult, Usage

DEFAULT_EMBED_MODEL = "mistral-embed"
NOMIC_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
BGE_M3_EMBED_MODEL = "text-embedding-bge-m3"
LOCAL_EMBED_MODEL = BGE_M3_EMBED_MODEL


@dataclass(frozen=True, slots=True)
class EmbedModelSpec:
    dim: int
    max_tokens: int
    local: bool
    doc_prefix: str = ""
    query_prefix: str = ""
    chunk_cap: int = 0  # local only: max chunk chars the model embeds whole (0 = no cap)


# Prefixes from the nomic-embed-text-v1.5 model card (checked 2026-10-06):
# `search_document: <text>` for corpus texts, `search_query: <text>` for queries.
EMBED_MODELS: dict[str, EmbedModelSpec] = {
    "mistral-embed": EmbedModelSpec(dim=1024, max_tokens=8192, local=False),
    NOMIC_EMBED_MODEL: EmbedModelSpec(
        dim=768,
        max_tokens=2048,
        local=True,
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
        chunk_cap=1600,
    ),
    # bge-m3: dense retrieval takes no prefixes (model card); cap measured 2026-10-06.
    BGE_M3_EMBED_MODEL: EmbedModelSpec(dim=1024, max_tokens=2048, local=True, chunk_cap=4000),
}

# Verified 2026-09-28 on docs.mistral.ai/models/mistral-embed-23-12.
PRICE_PER_M_TOKENS: dict[str, float] = {"mistral-embed": 0.10}

# Live 2026-09-04 measurement on mistral-embed: a batch totalling 47,744
# tokens succeeds, 71,616 returns 400 "Too many tokens overall, split into
# more batches." — the exact boundary is undocumented, so batching stays
# adaptive (halve-and-retry on that 400) rather than pinned to a token count.
# 40,000 chars is a conservative budget under either bound even for Cyrillic
# text, which runs roughly 2.5 chars/token.
BATCH_CHAR_BUDGET = 40_000
# Cap inputs per batch independently of the char budget: many short texts
# (e.g. eval questions) could stay well under the char budget while still
# hitting some other per-request shape limit.
BATCH_MAX_INPUTS = 64

_TOO_MANY_TOKENS_MARKER = "Too many tokens overall"
_INPUT_ID_RE = re.compile(r"Input id (\d+) has (\d+) tokens")


def embed_cost_usd(model: str, tokens: int | None) -> float | None:
    """USD cost for `tokens` prompt tokens on `model`; None when either is unknown."""
    spec = EMBED_MODELS.get(model)
    if spec is not None and spec.local:
        return 0.0
    price = PRICE_PER_M_TOKENS.get(model)
    if price is None or tokens is None:
        return None
    return tokens / 1_000_000 * price


@dataclass(slots=True)
class EmbedResult:
    vectors: np.ndarray  # (n, dim) float32, L2-normalized, row i == texts[i]
    model: str
    prompt_tokens: int | None
    requests: int
    latency_ms: int

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / norms).astype(np.float32)


def _batches(texts: list[str]) -> list[tuple[int, int]]:
    """Split `texts` into (start, end) ranges by char budget and input count.

    A single text that alone exceeds the budget still gets its own
    one-item batch — the check only blocks GROWING a non-empty batch.
    """
    batches: list[tuple[int, int]] = []
    n = len(texts)
    start = 0
    while start < n:
        end = start
        chars = 0
        while end < n:
            grown_chars = chars + len(texts[end])
            grown_count = end - start + 1
            if end > start and (grown_chars > BATCH_CHAR_BUDGET or grown_count > BATCH_MAX_INPUTS):
                break
            chars = grown_chars
            end += 1
        batches.append((start, end))
        start = end
    return batches


def _validate_batch_response(data: list, n_inputs: int, dim_box: list[int | None]) -> None:
    """Guard a batch response before it is trusted to fill `vectors` (SPEC-w05d21.md finding).

    A missing/duplicate/out-of-range index used to silently become (or
    overwrite) a zero vector; a NaN would have been stored and only found at
    search time. All four are dataset-corruption shapes, not user errors, so
    they raise AdventError rather than degrade the index quietly.
    """
    if len(data) != n_inputs:
        raise AdventError(
            f"Эмбеддинг вернул {len(data)} векторов вместо {n_inputs} — батч повреждён."
        )
    raw = [item.index for item in data]
    none_count = sum(1 for i in raw if i is None)
    if none_count == 0:
        if sorted(raw) != list(range(n_inputs)):
            raise AdventError(
                "Индексы эмбеддинга в батче не образуют 0.."
                f"{n_inputs - 1} без повторов: {sorted(raw)}."
            )
    elif none_count != len(raw):
        raise AdventError("Индексы эмбеддинга в батче смешаны: часть без index, часть с ним.")
    dims = {len(item.embedding) for item in data}
    if len(dims) != 1:
        raise AdventError("Векторы одного батча эмбеддинга имеют разную размерность.")
    (dim,) = dims
    if dim_box[0] is None:
        dim_box[0] = dim
    elif dim_box[0] != dim:
        raise AdventError(
            f"Размерность эмбеддинга изменилась между батчами: {dim_box[0]} -> {dim}."
        )
    for item in data:
        if not all(math.isfinite(v) for v in item.embedding):
            raise AdventError("Эмбеддинг содержит нечисловое значение (NaN/Inf).")


def _status_of(exc: Exception) -> int | None:
    """Duck-typed HTTP status, mirroring advent_core.errors._status_of.

    Not imported from there: that helper is private to errors.py, and this
    module needs the status BEFORE falling back to errors.translate() so it
    can recognize the two batching-specific 400 shapes first.
    """
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _journal(
    *,
    model: str,
    model_actual: str | None,
    usage: Usage,
    latency_ms: int,
    status: str,
    n_inputs: int,
    chars: int,
    week: int,
    day: int | None,
    journal_path: Path | None,
    journal_extra: dict | None,
) -> None:
    result = CallResult(
        text="",
        model_requested=model,
        model_actual=model_actual,
        usage=usage,
        latency_ms=latency_ms,
        stream=False,
    )
    log_internal_call(
        result,
        week=week,
        day=day,
        kind="embed",
        status=status,
        path=journal_path,
        extra={"inputs": n_inputs, "chars": chars, **(journal_extra or {})},
    )


def _embed_batch(
    client: object,
    model: str,
    texts: list[str],
    ids: list[str] | None,
    start: int,
    end: int,
    vectors: dict[int, list[float]],
    dim_box: list[int | None],
    *,
    week: int,
    day: int | None,
    journal_path: Path | None,
    journal_extra: dict | None,
) -> tuple[int | None, int, int]:
    """Embed texts[start:end]; returns (prompt_tokens|None, requests, latency_ms).

    On the "Too many tokens overall" 400, halves the range and retries each
    half recursively — a single-input range that still fails is unrecoverable.
    On "Input id N has X tokens, exceeding max 8192 tokens", raises naming the
    offending chunk_id rather than re-chunking (that would break the 1:1
    chunk<->vector mapping the caller relies on).
    """
    batch_texts = texts[start:end]
    n_inputs = len(batch_texts)
    chars = sum(len(t) for t in batch_texts)
    t0 = time.monotonic()
    try:
        resp = client.embeddings.create(model=model, inputs=batch_texts)
    except Exception as exc:  # noqa: BLE001 — translated below, never re-raised bare
        latency_ms = int((time.monotonic() - t0) * 1000)
        status = _status_of(exc)
        detail = redact(str(exc))[:500]

        if status == 400 and _TOO_MANY_TOKENS_MARKER in detail:
            _journal(
                model=model,
                model_actual=None,
                usage=Usage(),
                latency_ms=latency_ms,
                status="error",
                n_inputs=n_inputs,
                chars=chars,
                week=week,
                day=day,
                journal_path=journal_path,
                journal_extra=journal_extra,
            )
            if n_inputs <= 1:
                raise AdventError(
                    "Батч эмбеддинга не делится дальше: один текст сам по себе "
                    f"превышает лимит токенов на запрос ({detail})."
                ) from exc
            mid = start + n_inputs // 2
            left = _embed_batch(
                client,
                model,
                texts,
                ids,
                start,
                mid,
                vectors,
                dim_box,
                week=week,
                day=day,
                journal_path=journal_path,
                journal_extra=journal_extra,
            )
            right = _embed_batch(
                client,
                model,
                texts,
                ids,
                mid,
                end,
                vectors,
                dim_box,
                week=week,
                day=day,
                journal_path=journal_path,
                journal_extra=journal_extra,
            )
            tokens = None if left[0] is None or right[0] is None else left[0] + right[0]
            requests = 1 + left[1] + right[1]
            latency = latency_ms + left[2] + right[2]
            return tokens, requests, latency

        if status == 400:
            match = _INPUT_ID_RE.search(detail)
            if match is not None:
                local_idx, token_count = int(match.group(1)), match.group(2)
                global_idx = start + local_idx
                have_id = ids is not None and global_idx < len(ids)
                chunk_id = ids[global_idx] if have_id else str(global_idx)
                _journal(
                    model=model,
                    model_actual=None,
                    usage=Usage(),
                    latency_ms=latency_ms,
                    status="error",
                    n_inputs=n_inputs,
                    chars=chars,
                    week=week,
                    day=day,
                    journal_path=journal_path,
                    journal_extra=journal_extra,
                )
                raise AdventError(
                    f"Чанк {chunk_id!r} превышает лимит эмбеддинга в 8192 токена "
                    f"({token_count} токенов). Пере-разбить чанк здесь нельзя — "
                    "это сломало бы соответствие чанк:вектор. Проверь инвариант "
                    "MAX_CHUNK_CHARS в week_05/chunking.py."
                ) from exc

        _journal(
            model=model,
            model_actual=None,
            usage=Usage(),
            latency_ms=latency_ms,
            status="error",
            n_inputs=n_inputs,
            chars=chars,
            week=week,
            day=day,
            journal_path=journal_path,
            journal_extra=journal_extra,
        )
        raise translate(exc) from exc

    latency_ms = int((time.monotonic() - t0) * 1000)
    try:
        _validate_batch_response(resp.data, n_inputs, dim_box)
    except AdventError:
        _journal(
            model=model,
            model_actual=resp.model,
            usage=Usage.from_raw(resp.usage),
            latency_ms=latency_ms,
            status="error",
            n_inputs=n_inputs,
            chars=chars,
            week=week,
            day=day,
            journal_path=journal_path,
            journal_extra=journal_extra,
        )
        raise
    for pos, item in enumerate(resp.data):
        # Validated above: indices are all None (positional) or all present.
        idx = item.index if item.index is not None else pos
        vectors[start + idx] = item.embedding
    usage = Usage.from_raw(resp.usage)
    _journal(
        model=model,
        model_actual=resp.model,
        usage=usage,
        latency_ms=latency_ms,
        status="ok",
        n_inputs=n_inputs,
        chars=chars,
        week=week,
        day=day,
        journal_path=journal_path,
        journal_extra=journal_extra,
    )
    return usage.prompt_tokens, 1, latency_ms


def embed_texts(
    client: object,
    model: str,
    texts: list[str],
    *,
    ids: list[str] | None = None,
    on_batch: Callable[[int, int], None] | None = None,
    week: int = 5,
    day: int | None = 21,
    journal_path: Path | None = None,
    journal_extra: dict | None = None,
) -> EmbedResult:
    """Embed `texts` in char-budgeted batches; vector i corresponds to texts[i].

    `client` is a mistralai Mistral (or any object with
    .embeddings.create(model=, inputs=)). `ids` (parallel to `texts`) names
    chunks in the "exceeds max 8192 tokens" error; without it the error
    names the input's position instead. `on_batch(done_inputs, total_inputs)`
    fires after each successful batch (progress, stderr only — never stdout).
    """
    n = len(texts)
    vectors: dict[int, list[float]] = {}
    dim_box: list[int | None] = [None]  # shared across batches: catches a dim drift mid-run
    total_prompt_tokens = 0
    any_missing_tokens = False
    total_requests = 0
    total_latency_ms = 0
    done = 0

    for start, end in _batches(texts):
        prompt_tokens, requests, latency_ms = _embed_batch(
            client,
            model,
            texts,
            ids,
            start,
            end,
            vectors,
            dim_box,
            week=week,
            day=day,
            journal_path=journal_path,
            journal_extra=journal_extra,
        )
        total_requests += requests
        total_latency_ms += latency_ms
        if prompt_tokens is None:
            any_missing_tokens = True
        else:
            total_prompt_tokens += prompt_tokens
        done += end - start
        if on_batch is not None:
            on_batch(done, n)

    dim = len(next(iter(vectors.values()))) if vectors else 0
    matrix = np.zeros((n, dim), dtype=np.float32)
    with np.errstate(over="ignore"):  # 1e40 -> Inf is caught by the isfinite check below
        for idx, vector in vectors.items():
            matrix[idx, :] = vector
    # Checked on the float32 matrix: 1e40 is a finite float but Inf in float32.
    if not np.isfinite(matrix).all():
        raise AdventError("Эмбеддинг содержит нечисловое значение (NaN/Inf).")
    if matrix.size and not (np.linalg.norm(matrix, axis=1) > 0).all():
        raise AdventError("Эмбеддинг содержит нулевой вектор — нормализовать нечего.")
    matrix = _l2_normalize(matrix)

    return EmbedResult(
        vectors=matrix,
        model=model,
        prompt_tokens=None if any_missing_tokens else total_prompt_tokens,
        requests=total_requests,
        latency_ms=total_latency_ms,
    )


def embed_local(
    url: str | None,
    model: str,
    texts: list[str],
    *,
    kind: str = "doc",
    prefix: str | None = None,
    on_batch: Callable[[int, int], None] | None = None,
    week: int = 5,
    day: int | None = 21,
    journal_path: Path | None = None,
    journal_extra: dict | None = None,
) -> EmbedResult:
    """Embed `texts` on a local OpenAI-compatible server, adding the model's prefix.

    `kind` is "doc" or "query" and picks the table prefix; an explicit `prefix`
    (e.g. the one stored in an index run) wins. Dimension is checked against
    the table. Row i of the result is texts[i].
    """
    if kind not in ("doc", "query"):
        raise ValueError(f"kind must be 'doc' or 'query', got {kind!r}")
    spec = EMBED_MODELS.get(model)
    if prefix is None:
        prefix = (spec.doc_prefix if kind == "doc" else spec.query_prefix) if spec else ""
    t0 = time.monotonic()
    try:
        result = openai_compat.embed(
            url,
            model,
            [prefix + t for t in texts],
            expected_dim=spec.dim if spec else None,
            on_batch=on_batch,
        )
    except AdventError:
        _journal(
            model=model,
            model_actual=None,
            usage=Usage(),
            latency_ms=int((time.monotonic() - t0) * 1000),
            status="error",
            n_inputs=len(texts),
            chars=sum(len(t) for t in texts),
            week=week,
            day=day,
            journal_path=journal_path,
            journal_extra={"endpoint": "local", **(journal_extra or {})},
        )
        raise
    _journal(
        model=model,
        model_actual=result.model,
        usage=Usage(prompt_tokens=result.prompt_tokens),
        latency_ms=result.latency_ms,
        status="ok",
        n_inputs=len(texts),
        chars=sum(len(t) for t in texts),
        week=week,
        day=day,
        journal_path=journal_path,
        journal_extra={"endpoint": "local", **(journal_extra or {})},
    )
    return result
