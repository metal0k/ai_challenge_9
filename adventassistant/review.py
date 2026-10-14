"""One review of `base...head`: diff, docs-and-code RAG, three sections, optional PR comment.

The model does not choose tools. An empty range prints `изменений нет` and
stops before the index, the key and the model. `--diff-only` prints the diff
and stops the same way. A paid call that already happened is journaled; a
journal failure does not drop the review.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from advent_core import console
from advent_core.client import mistral_client
from advent_core.config import load_env, redact
from advent_core.embeddings import DEFAULT_EMBED_MODEL
from advent_core.embeddings import embed_texts as embed_texts_fn
from advent_core.errors import AdventError, translate
from advent_core.journal import log_call
from advent_core.params import GenerationParams
from advent_core.rag import (
    RagHit,
    apply_rerank,
    build_rerank_prompt,
    order_by_rerank,
    parse_rerank,
)
from advent_core.telemetry import CallResult
from adventassistant.corpus import cloud_config
from adventassistant.review_index import (
    DAY,
    PURPOSE,
    REVIEW_INDEX_PATH,
    SHORT_SHA,
    STRATEGY,
    WEEK,
    git_stdout,
    require_review_index,
    resolve_rev,
)
from week_05.index import load_meta, load_runs, search
from week_05.rag import RAG_AUX_MODEL, RERANK_MAX_TOKENS

ANSWER_MODEL = "ministral-14b-latest"
SEARCH_K = 16
DOC_SEARCH_K = SEARCH_K // 2
DOC_SUFFIXES = (".md", ".json")
CODE_SUFFIXES = (".py",)
ANSWER_K = 4
THRESHOLD = 5
DIFF_CHAR_CAP = 24000
QUERY_CHAR_CAP = 6000
DEMO_BRANCH = "w07d32-demo"
DEMO_FILE = "adventassistant/demo_bug.py"
_MARKER = "<!-- adventassistant-review -->"
_EMPTY = "изменений нет"

_REVIEW_PROMPT = (
    (Path(__file__).resolve().parent / "prompts" / "review.md").read_text(encoding="utf-8").strip()
)
_SCHEMA = Path(__file__).resolve().parent / "schemas" / "review.json"

CompleteFn = Callable[..., CallResult]


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One name-status record. `new_path` is set for rename and copy."""

    status: str
    path: str
    new_path: str | None = None


@dataclass(frozen=True, slots=True)
class DemoFacts:
    """Git and index facts for the day-32 recording gate. No I/O of its own."""

    branch: str
    name_only_z: str
    target_diff: str
    purpose: str | None
    corpus_rev: str | None
    base_sha: str


def parse_name_status(raw: str) -> list[DiffEntry]:
    """NUL records from `git diff -z --name-status`. Rename and copy carry two paths."""
    parts = raw.split("\0")
    if parts and parts[-1] == "":
        parts.pop()
    entries: list[DiffEntry] = []
    index = 0
    while index < len(parts):
        status = parts[index]
        if status[:1] in {"R", "C"}:
            if index + 2 >= len(parts):
                raise AdventError("git diff -z вернул обрезанную запись rename/copy.")
            entries.append(
                DiffEntry(status=status, path=parts[index + 1], new_path=parts[index + 2])
            )
            index += 3
            continue
        if index + 1 >= len(parts):
            raise AdventError("git diff -z вернул обрезанную запись.")
        entries.append(DiffEntry(status=status, path=parts[index + 1]))
        index += 2
    return entries


def format_file_list(entries: Sequence[DiffEntry]) -> str:
    lines: list[str] = []
    for entry in entries:
        if entry.new_path is not None:
            lines.append(f"{entry.status} {entry.path} -> {entry.new_path}")
        else:
            lines.append(f"{entry.status} {entry.path}")
    return "\n".join(lines)


def allowed_paths(entries: Sequence[DiffEntry]) -> set[str]:
    found: set[str] = set()
    for entry in entries:
        found.add(entry.path)
        if entry.new_path is not None:
            found.add(entry.new_path)
    return found


def cap_diff(text: str) -> tuple[str, str | None]:
    """Keep the head of a long diff and append the truncation line. Short text is unchanged."""
    if len(text) <= DIFF_CHAR_CAP:
        return text, None
    note = f"… diff обрезан: показано {DIFF_CHAR_CAP} из {len(text)} символов"
    head = text[:DIFF_CHAR_CAP]
    if not head.endswith("\n"):
        head += "\n"
    return head + note + "\n", note


def _load_diff(root: Path, base_sha: str, head_sha: str) -> tuple[list[DiffEntry], str]:
    spec = f"{base_sha}...{head_sha}"
    status = git_stdout(root, ["diff", "-z", "--name-status", "--find-renames", spec])
    body = git_stdout(root, ["diff", "--find-renames", "--unified=3", spec])
    return parse_name_status(status), body


def _emit(text: str) -> None:
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def _stderr(lines: list[str]) -> None:
    for line in lines:
        console.err.print(line, markup=False, highlight=False)


def _to_hits(found: Sequence[object]) -> tuple[RagHit, ...]:
    return tuple(
        RagHit(
            chunk_id=hit.chunk.chunk_id,
            source=hit.chunk.source,
            section=hit.chunk.section,
            score=hit.score,
            text=hit.chunk.text,
            rank=position,
        )
        for position, hit in enumerate(found, start=1)
    )


def _search_review_candidates(db_path: Path, query_vec: np.ndarray) -> tuple[RagHit, ...]:
    """Reserve half the rerank pool for docs, then rank the combined candidates together."""
    found = [
        *search(
            db_path,
            STRATEGY,
            query_vec,
            k=DOC_SEARCH_K,
            source_suffixes=DOC_SUFFIXES,
        ),
        *search(
            db_path,
            STRATEGY,
            query_vec,
            k=SEARCH_K - DOC_SEARCH_K,
            source_suffixes=CODE_SUFFIXES,
        ),
    ]
    # Keep the strongest match if overlapping retrieval branches ever return one chunk twice.
    unique = {hit.chunk.chunk_id: hit for hit in found}
    combined = sorted(unique.values(), key=lambda hit: (-hit.score, hit.chunk.chunk_id))[:SEARCH_K]
    return _to_hits(combined)


def _retain_relevant_doc(
    hits: Sequence[RagHit], scores: dict[int, float], kept: tuple[RagHit, ...]
) -> tuple[RagHit, ...]:
    """Keep the strongest passing documentation hit in the answer context."""
    if not kept or any(hit.source.endswith(DOC_SUFFIXES) for hit in kept):
        return kept
    passing_docs = [
        hit
        for hit in order_by_rerank(hits, scores)
        if hit.source.endswith(DOC_SUFFIXES) and hit.rerank is not None and hit.rerank >= THRESHOLD
    ]
    if not passing_docs:
        return kept
    selected = [*kept[:-1], passing_docs[0]]
    selected.sort(key=lambda hit: (-(hit.rerank or 0), hit.rank or 0))
    return tuple(selected)


def embed_query(text: str) -> np.ndarray:
    """Query vector. `journal_path` is omitted so tests can redirect the journal."""
    config = cloud_config(DEFAULT_EMBED_MODEL)
    with mistral_client(config) as client:
        result = embed_texts_fn(
            client,
            DEFAULT_EMBED_MODEL,
            [text],
            week=WEEK,
            day=DAY,
            journal_extra={"command": "review_embed"},
        )
    if result.vectors.ndim != 2 or result.vectors.shape[0] != 1:
        raise AdventError(f"Эмбеддинг запроса вернул форму {result.vectors.shape}.")
    return np.asarray(result.vectors[0], dtype=np.float32)


def _query_text(file_list: str, diff_body: str) -> str:
    return f"ревью изменений\n{file_list}\n{diff_body}"[:QUERY_CHAR_CAP]


def _complete(complete: CompleteFn, config: object, messages: list[dict[str, str]]) -> CallResult:
    try:
        return complete(config, messages)
    except AdventError:
        raise
    except Exception as exc:
        raise translate(exc) from exc


def _log(result: CallResult, messages: list[dict[str, str]], command: str) -> None:
    try:
        log_call(
            result,
            list(result.sent_messages or messages),
            week=WEEK,
            day=DAY,
            extra={"command": command},
        )
    except Exception:
        return


def _rerank_config():
    base = cloud_config(RAG_AUX_MODEL)
    return replace(
        base,
        params=GenerationParams(
            temperature=0,
            max_tokens=RERANK_MAX_TOKENS,
            format="json",
        ),
    )


def _answer_config():
    base = cloud_config(ANSWER_MODEL)
    return replace(
        base,
        system_prompt_path=None,
        params=GenerationParams(
            temperature=0,
            max_tokens=1200,
            format="schema",
            schema_file=str(_SCHEMA),
        ),
    )


def _answer_messages(
    file_list: str, diff_body: str, hits: Sequence[RagHit]
) -> list[dict[str, str]]:
    blocks = [
        f"[{number}] {hit.source} — {hit.section}\n{hit.text.strip()}"
        for number, hit in enumerate(hits, start=1)
    ]
    parts = ["Файлы изменений:", file_list, "Diff:", diff_body]
    if blocks:
        parts.extend(["Фрагменты:", *blocks])
    parts.append("Сделай ревью этих изменений.")
    return [
        {"role": "system", "content": _REVIEW_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _require_token() -> None:
    if _has_token():
        return
    # A fresh CLI process has not read .env yet; override=False keeps the Action env.
    load_env()
    if _has_token():
        return
    raise AdventError(
        "Для --pr нужен GH_TOKEN или GITHUB_TOKEN. Комментарий не записан, модель не вызвана."
    )


def _has_token() -> bool:
    return any(os.environ.get(key, "").strip() for key in ("GH_TOKEN", "GITHUB_TOKEN"))


def parse_review(raw: str) -> dict[str, list]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdventError("Ревью вернуло не JSON. Комментарий не записан.") from exc
    if not isinstance(data, dict):
        raise AdventError("Ревью вернуло не объект JSON. Комментарий не записан.")
    for key in ("bugs", "architecture", "recommendations"):
        if not isinstance(data.get(key), list):
            raise AdventError(f"В ревью нет массива {key}. Комментарий не записан.")
    return data


def _item(item: object, *, need_file: bool) -> tuple[str | None, str]:
    if not isinstance(item, dict):
        raise AdventError("Пункт ревью не объект. Комментарий не записан.")
    text = item.get("text")
    if not isinstance(text, str):
        raise AdventError("Пункт ревью: text должен быть строкой. Комментарий не записан.")
    if not need_file:
        return None, text
    file = item.get("file")
    if not isinstance(file, str):
        raise AdventError("Пункт ревью: file должен быть строкой. Комментарий не записан.")
    return file, text


def render_review(data: dict[str, list], allowed: set[str]) -> tuple[str, int]:
    """Three headers always. A file outside the diff is printed, not dropped."""
    sections = (
        ("bugs", "Потенциальные баги", True),
        ("architecture", "Архитектурные проблемы", True),
        ("recommendations", "Рекомендации", False),
    )
    lines: list[str] = []
    outside = 0
    for key, header, has_file in sections:
        lines.append(header)
        items = data[key]
        if not items:
            lines.append("- нет")
            continue
        for item in items:
            path, text = _item(item, need_file=has_file)
            if has_file:
                assert path is not None
                if path not in allowed:
                    outside += 1
                    lines.append(f"- вне diff: {path}: {text}")
                else:
                    lines.append(f"- {path}: {text}")
            else:
                lines.append(f"- {text}")
    return "\n".join(lines), outside


def _comment(root: Path, pr: int, review_text: str, base_sha: str, head_sha: str) -> None:
    body = (
        f"{_MARKER}\n"
        f"{review_text.rstrip()}\n"
        f"ревью: {base_sha[:SHORT_SHA]}..{head_sha[:SHORT_SHA]}\n"
    )
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", suffix=".md", delete=False
        ) as handle:
            path = Path(handle.name)
            handle.write(body)
        command = ["gh", "pr", "comment", str(pr), "--body-file", str(path)]
        try:
            result = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
        except OSError as exc:
            raise AdventError(f"Не удалось запустить gh: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise AdventError(
                redact(f"gh pr comment завершился с кодом {result.returncode}: {detail}")
            )
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def run_review(
    root: Path,
    *,
    base: str,
    head: str,
    db_path: Path,
    complete: CompleteFn,
    pr: int | None = None,
    diff_only: bool = False,
) -> str:
    """Review one range. stdout is emitted here, before a PR comment."""
    base_sha = resolve_rev(root, base)
    head_sha = resolve_rev(root, head)
    entries, diff_body = _load_diff(root, base_sha, head_sha)
    if not entries:
        _emit(_EMPTY)
        return _EMPTY
    shown, note = cap_diff(diff_body)
    if diff_only:
        if note:
            _stderr([note])
        _emit(shown)
        return shown
    # Token, then the index. Both fail before the query embedding.
    if pr is not None:
        _require_token()
    path = require_review_index(db_path, base_sha)
    file_list = format_file_list(entries)
    query = _query_text(file_list, shown)
    hits = _search_review_candidates(path, embed_query(query))
    candidates = len(hits)
    passed = 0
    kept: tuple[RagHit, ...] = ()
    if hits:
        prompt = build_rerank_prompt(query, hits)
        messages = [{"role": "user", "content": prompt}]
        reranked = _complete(complete, _rerank_config(), messages)
        _log(reranked, messages, "review_rerank")
        scores = parse_rerank(reranked.text, len(hits))
        kept, passed = apply_rerank(hits, scores, THRESHOLD, ANSWER_K)
        kept = _retain_relevant_doc(hits, scores, kept)
    answer_messages = _answer_messages(file_list, shown, kept)
    answered = _complete(complete, _answer_config(), answer_messages)
    _log(answered, answer_messages, "review_answer")
    if answered.format_ok is not True:
        raise AdventError("Ответ ревью не соответствует схеме. Комментарий не записан.")
    data = parse_review(answered.text)
    rendered, outside = render_review(data, allowed_paths(entries))
    status = [
        f"база: {base_sha[:SHORT_SHA]}",
        f"голова: {head_sha[:SHORT_SHA]}",
        f"файлы: {len(entries)}",
        f"порог: {passed}/{candidates}",
    ]
    if passed:
        status.extend(f"источник: {hit.source} — {hit.section}" for hit in kept)
    if note:
        status.append(note)
    if outside:
        status.append(f"вне diff: {outside}")
    _stderr(status)
    _emit(rendered)
    if pr is not None:
        _comment(root, pr, rendered, base_sha, head_sha)
    return rendered


def parse_name_only_z(raw: str) -> list[str]:
    parts = raw.split("\0")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def check_demo_range(facts: DemoFacts) -> None:
    """Recording gate. Callers inject git output; this function does not run git."""
    if facts.branch != DEMO_BRANCH:
        raise AdventError(
            f"Демо дня 32 записывается с ветки {DEMO_BRANCH}, сейчас {facts.branch!r}."
        )
    paths = parse_name_only_z(facts.name_only_z)
    if paths != [DEMO_FILE]:
        raise AdventError(
            f"Диапазон main...HEAD для демо должен содержать ровно {DEMO_FILE}, а не {paths!r}."
        )
    if facts.target_diff != "":
        raise AdventError(
            f"{DEMO_FILE} на диске отличается от HEAD. В кадр попадёт не commit ветки."
        )
    if facts.purpose != PURPOSE:
        raise AdventError(
            f"Индекс демо не purpose={PURPOSE} (purpose={facts.purpose!r}). "
            "Сначала выполни `adventassistant index-review`."
        )
    if not facts.base_sha or facts.corpus_rev != facts.base_sha:
        raise AdventError(
            "corpus_rev индекса не равен SHA main "
            f"({facts.corpus_rev!r} != {facts.base_sha!r}). "
            "Сначала выполни `adventassistant index-review --rev main`."
        )


def collect_demo_facts(root: Path) -> DemoFacts:
    """Read the live repo. Tests use `check_demo_range` and do not call this."""
    branch = git_stdout(root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    name_only = git_stdout(root, ["diff", "-z", "--name-only", "main...HEAD"])
    target = git_stdout(root, ["diff", "HEAD", "--", DEMO_FILE])
    base_sha = resolve_rev(root, "main")
    if not REVIEW_INDEX_PATH.exists():
        raise AdventError(
            f"Индекс не найден: {REVIEW_INDEX_PATH.name}. "
            "Сначала выполни `adventassistant index-review`."
        )
    meta = load_meta(REVIEW_INDEX_PATH)
    run = load_runs(REVIEW_INDEX_PATH).get(STRATEGY)
    return DemoFacts(
        branch=branch,
        name_only_z=name_only,
        target_diff=target,
        purpose=meta.get("purpose"),
        corpus_rev=None if run is None else run.corpus_rev,
        base_sha=base_sha,
    )
