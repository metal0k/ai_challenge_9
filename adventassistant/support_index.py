"""Versioned, isolated vector index for the fixed support FAQ corpus."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import numpy as np

from advent_core import console
from advent_core.client import mistral_client
from advent_core.config import PROJECT_ROOT
from advent_core.embeddings import DEFAULT_EMBED_MODEL, EmbedResult, embed_cost_usd
from advent_core.embeddings import embed_texts as embed_texts_fn
from advent_core.errors import AdventError
from adventassistant.corpus import cloud_config, format_index_summary
from week_05.chunking import Chunk, Document, chunk_structure
from week_05.index import (
    ENDPOINT_CLOUD,
    check_endpoint_free,
    display_path,
    load_chunks,
    load_meta,
    load_runs,
    write_index,
)

SUPPORT_INDEX_PATH = PROJECT_ROOT / "data" / "rag" / "index.w07d33.sqlite3"
PURPOSE = "w07d33"
STRATEGY = "support"
WEEK = 7
DAY = 33
CORPUS_VERSION = "1"
INDEX_HINT = "Сначала выполни `adventassistant index-support`."
CORPUS_DIR = Path(__file__).resolve().parent
CORPUS_FILES = ("support_faq.md",)
OLD_INDEX_PATHS = (
    PROJECT_ROOT / "data" / "rag" / "index.w07d31.sqlite3",
    PROJECT_ROOT / "data" / "rag" / "index.w07d32.sqlite3",
)
EmbedFn = Callable[[list[str]], EmbedResult]


def collect_corpus() -> tuple[list[Document], str]:
    docs: list[Document] = []
    digest = hashlib.sha256()
    digest.update(f"corpus-version:{CORPUS_VERSION}\0".encode())
    for name in sorted(CORPUS_FILES):
        path = CORPUS_DIR / name
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise AdventError(f"Не удалось прочитать корпус поддержки {name}: {exc}") from None
        digest.update(name.encode("utf-8") + b"\0" + raw + b"\0")
        docs.append(Document(source=f"adventassistant/{name}", text=text.replace("\r\n", "\n")))
    return docs, digest.hexdigest()


def build_chunks(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for chunk in chunk_structure(doc):
            prefix = "structure:"
            if chunk.strategy != "structure" or not chunk.chunk_id.startswith(prefix):
                raise AdventError(f"Чанк {chunk.chunk_id!r} не помечен стратегией structure.")
            chunks.append(
                replace(
                    chunk,
                    strategy=STRATEGY,
                    chunk_id=STRATEGY + ":" + chunk.chunk_id[len(prefix) :],
                )
            )
    if not chunks:
        raise AdventError("Корпус поддержки не дал чанков — индекс не построен.")
    if any(c.strategy != STRATEGY or not c.chunk_id.startswith(f"{STRATEGY}:") for c in chunks):
        raise AdventError("В чанках корпуса поддержки неверная стратегия.")
    return chunks


def _default_embed(texts: list[str]) -> EmbedResult:
    config = cloud_config(DEFAULT_EMBED_MODEL)
    with mistral_client(config) as client:
        return embed_texts_fn(
            client,
            DEFAULT_EMBED_MODEL,
            texts,
            ids=[f"{STRATEGY}:{i}" for i in range(len(texts))],
            week=WEEK,
            day=DAY,
            journal_extra={"command": "assistant_support_index"},
        )


def _validate_embeddings(chunks: list[Chunk], result: EmbedResult) -> None:
    try:
        matrix = np.asarray(result.vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise AdventError(f"Эмбеддинг поддержки не является матрицей float32: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] != 1024:
        raise AdventError(
            f"Эмбеддинг поддержки вернул форму {matrix.shape} на {len(chunks)} чанков."
        )
    if not np.isfinite(matrix).all():
        raise AdventError("Эмбеддинг поддержки содержит NaN/Inf.")
    if result.model != DEFAULT_EMBED_MODEL:
        raise AdventError(f"Эмбеддинг поддержки вернул модель {result.model!r}.")


def _refuse_destination(path: Path) -> None:
    resolved = path.expanduser().resolve()
    old_paths = {item.expanduser().resolve() for item in OLD_INDEX_PATHS}
    if resolved in old_paths:
        raise AdventError(
            f"Индекс поддержки не может использовать индекс другого дня: {display_path(path)}."
        )
    if not path.exists():
        return
    try:
        meta = load_meta(path)
    except AdventError as exc:
        raise AdventError(
            f"Нельзя перезаписать непроверенную базу {display_path(path)}: {exc.message}"
        ) from exc
    if meta.get("purpose") != PURPOSE:
        raise AdventError(
            f"База {display_path(path)} уже существует и не является индексом поддержки "
            f"(purpose={meta.get('purpose')!r}); файл сохранён без изменений."
        )


def build_support_index(*, db_path: Path | None = None, embed_fn: EmbedFn | None = None) -> str:
    """Build beside the target, then atomically replace only a valid support index."""
    path = db_path or SUPPORT_INDEX_PATH
    _refuse_destination(path)
    docs, fingerprint = collect_corpus()
    chunks = build_chunks(docs)
    check_endpoint_free(path, ENDPOINT_CLOUD)
    console.note(f"эмбеддинг support: {len(chunks)} чанков, модель {DEFAULT_EMBED_MODEL}")
    result = (embed_fn or _default_embed)([chunk.text for chunk in chunks])
    _validate_embeddings(chunks, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.unlink()
        write_index(
            temp_path,
            {STRATEGY: (chunks, result)},
            corpus_rev=fingerprint,
            corpus_files=len(docs),
            corpus_chars=sum(len(doc.text) for doc in docs),
            endpoint=ENDPOINT_CLOUD,
            meta={
                "purpose": PURPOSE,
                "support_version": CORPUS_VERSION,
                "support_fingerprint": fingerprint,
            },
        )
        # Inspect the complete temp DB before replacing the last working copy.
        _require_index(temp_path)
        os.replace(temp_path, path)
    finally:
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
    console.note(f"индекс записан: {display_path(path)}")
    return format_index_summary(
        fingerprint,
        len(docs),
        len(chunks),
        result.prompt_tokens,
        embed_cost_usd(result.model, result.prompt_tokens),
    )


def _require_index(path: Path) -> Path:
    if not path.exists():
        raise AdventError(f"Индекс не найден: {display_path(path)}. {INDEX_HINT}")
    try:
        meta = load_meta(path)
        runs = load_runs(path)
    except AdventError as exc:
        raise AdventError(f"Индекс поддержки повреждён: {exc.message} {INDEX_HINT}") from exc
    docs, fingerprint = collect_corpus()
    expected_chunks = build_chunks(docs)
    if meta.get("purpose") != PURPOSE:
        raise AdventError(
            f"Индекс не для поддержки (purpose={meta.get('purpose')!r}). {INDEX_HINT}"
        )
    if meta.get("support_version") != CORPUS_VERSION:
        raise AdventError(f"Версия корпуса индекса поддержки устарела. {INDEX_HINT}")
    if meta.get("support_fingerprint") != fingerprint:
        raise AdventError(f"Отпечаток корпуса индекса поддержки устарел. {INDEX_HINT}")
    run = runs.get(STRATEGY)
    if run is None:
        raise AdventError(f"В индексе нет run {STRATEGY}. {INDEX_HINT}")
    if run.model != DEFAULT_EMBED_MODEL or run.endpoint != ENDPOINT_CLOUD:
        raise AdventError(
            f"Индекс поддержки требует облачную модель {DEFAULT_EMBED_MODEL}. {INDEX_HINT}"
        )
    if run.doc_prefix or run.query_prefix:
        raise AdventError(f"Индекс поддержки содержит неожиданные префиксы. {INDEX_HINT}")
    if run.corpus_rev != fingerprint or run.dim != 1024 or run.n_chunks != len(expected_chunks):
        raise AdventError(
            f"Метаданные run индекса поддержки не соответствуют корпусу. {INDEX_HINT}"
        )
    try:
        stored = load_chunks(path, STRATEGY)
    except AdventError as exc:
        raise AdventError(
            f"Чанки индекса поддержки повреждены: {exc.message}. {INDEX_HINT}"
        ) from exc
    if len(stored) != len(expected_chunks) or any(
        (actual.chunk_id, actual.source, actual.section, actual.text)
        != (expected.chunk_id, expected.source, expected.section, expected.text)
        for actual, expected in zip(stored, expected_chunks, strict=True)
    ):
        raise AdventError(f"Содержимое индекса поддержки не соответствует корпусу. {INDEX_HINT}")
    try:
        conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            blobs = conn.execute(
                "SELECT embedding FROM chunks WHERE strategy = ? ORDER BY source, ordinal",
                (STRATEGY,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise AdventError(f"Векторы индекса поддержки повреждены: {exc}. {INDEX_HINT}") from None
    if len(blobs) != len(expected_chunks):
        raise AdventError(f"Число векторов индекса поддержки не совпадает с корпусом. {INDEX_HINT}")
    for (blob,) in blobs:
        if not isinstance(blob, bytes) or len(blob) != 1024 * np.dtype(np.float32).itemsize:
            raise AdventError(f"Размер вектора индекса поддержки повреждён. {INDEX_HINT}")
        vector = np.frombuffer(blob, dtype=np.float32)
        if not np.isfinite(vector).all():
            raise AdventError(f"Вектор индекса поддержки содержит NaN/Inf. {INDEX_HINT}")
    return path


def require_support_index(db_path: Path | None = None) -> Path:
    return _require_index(db_path or SUPPORT_INDEX_PATH)
