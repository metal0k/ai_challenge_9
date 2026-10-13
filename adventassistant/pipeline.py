"""One `/help` question: classify, read git, retrieve docs, answer or refuse.

The model does not choose tools. Branch and diff questions print the tool
text and do not call the answer model — a paraphrase would hide the tool.
A doc question with nothing over the threshold prints `не знаю` for the
same reason: an empty RAG prompt still invents an answer.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from advent_core import console
from advent_core.client import mistral_client
from advent_core.embeddings import DEFAULT_EMBED_MODEL
from advent_core.embeddings import embed_texts as embed_texts_fn
from advent_core.errors import AdventError, translate
from advent_core.journal import log_call
from advent_core.mcp_client import ToolCallOutcome, call_tool_once
from advent_core.params import GenerationParams
from advent_core.rag import RagHit, apply_rerank, build_rerank_prompt, parse_rerank
from advent_core.telemetry import CallResult
from adventassistant.corpus import DAY, STRATEGY, WEEK, cloud_config, require_index
from week_05.index import search
from week_05.rag import RAG_AUX_MODEL, RERANK_MAX_TOKENS

ANSWER_MODEL = "ministral-14b-latest"
SEARCH_K = 8
ANSWER_K = 4
THRESHOLD = 5
TOOL_TIMEOUT = 15.0
SERVER_CMD = [sys.executable, "-m", "adventassistant.server"]

_HELP_PROMPT = (
    (Path(__file__).resolve().parent / "prompts" / "help.md").read_text(encoding="utf-8").strip()
)

# Letters only: a digit or underscore is a boundary, not part of the word.
_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
_DIFF_PREFIXES = ("измен", "отличи", "коммит", "незакоммич")
_DIFF_WORDS = frozenset({"diff", "staged", "unstaged", "status"})
_BRANCH_PREFIXES = ("ветк", "branch")
_BRANCH_WORDS = frozenset({"head"})

CompleteFn = Callable[..., CallResult]


def words(text: str) -> list[str]:
    return [word.casefold() for word in _WORDS.findall(text)]


def classify(question: str) -> str:
    """`diff`, `branch` or `docs`. Diff wins when both match."""
    found = words(question)
    pairs = zip(found, found[1:], strict=False)
    phrase = any(left == "что" and right == "нового" for left, right in pairs)
    prefixed = any(word.startswith(_DIFF_PREFIXES) for word in found)
    whole = any(word in _DIFF_WORDS for word in found)
    if phrase or prefixed or whole:
        return "diff"
    if any(word.startswith(_BRANCH_PREFIXES) for word in found) or any(
        word in _BRANCH_WORDS for word in found
    ):
        return "branch"
    return "docs"


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


def embed_question(text: str) -> np.ndarray:
    """Query vector. `journal_path` is omitted so tests can redirect the journal."""
    config = cloud_config(DEFAULT_EMBED_MODEL)
    with mistral_client(config) as client:
        result = embed_texts_fn(
            client,
            DEFAULT_EMBED_MODEL,
            [text],
            week=WEEK,
            day=DAY,
            journal_extra={"command": "assistant_embed"},
        )
    if result.vectors.ndim != 2 or result.vectors.shape[0] != 1:
        raise AdventError(f"Эмбеддинг вопроса вернул форму {result.vectors.shape}.")
    return np.asarray(result.vectors[0], dtype=np.float32)


def _call_tool(name: str) -> str:
    try:
        outcome: ToolCallOutcome = call_tool_once(
            SERVER_CMD,
            name,
            {},
            timeout=TOOL_TIMEOUT,
            quiet=True,
        )
    except AdventError:
        raise
    except Exception as exc:
        raise translate(exc) from exc
    if outcome.is_error:
        detail = outcome.text.strip() or "MCP вернул ошибку."
        raise AdventError(detail)
    return outcome.text.strip()


def _complete(complete: CompleteFn, config: object, messages: list[dict[str, str]]) -> CallResult:
    try:
        return complete(config, messages)
    except AdventError:
        raise
    except Exception as exc:
        raise translate(exc) from exc


def _rerank_config():
    # Cloud JSON mode, copied from the non-local branch of week_05.rag._aux_call.
    # That private function is not imported.
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
    # temperature and every other generation param stay unset. No default persona.
    return replace(base, system_prompt_path=None, params=GenerationParams())


def _answer_messages(question: str, hits: Sequence[RagHit], branch: str) -> list[dict[str, str]]:
    blocks = [
        f"[{number}] {hit.source} — {hit.section}\n{hit.text.strip()}"
        for number, hit in enumerate(hits, start=1)
    ]
    user = "\n\n".join([*blocks, branch, f"Вопрос: {question}"])
    return [
        {"role": "system", "content": _HELP_PROMPT},
        {"role": "user", "content": user},
    ]


def _stderr(lines: list[str]) -> None:
    for line in lines:
        console.err.print(line, markup=False, highlight=False)


def run_help(question: str, *, db_path: Path, complete: CompleteFn) -> str:
    """Answer one question. stdout text is the return value; status goes to stderr."""
    text = question.strip()
    if not text:
        raise AdventError("Пустой вопрос.")
    # Missing index is a precondition: fail before git and before any paid call.
    path = require_index(db_path)
    kind = classify(text)

    branch = _call_tool("git_branch")
    diff_text = _call_tool("git_diff_stat") if kind == "diff" else None

    query = embed_question(text)
    found = search(path, STRATEGY, query, k=SEARCH_K)
    hits = _to_hits(found)
    candidates = len(hits)

    passed = 0
    kept: tuple[RagHit, ...] = ()
    if hits:
        prompt = build_rerank_prompt(text, hits)
        messages = [{"role": "user", "content": prompt}]
        reranked = _complete(complete, _rerank_config(), messages)
        log_call(
            reranked,
            list(reranked.sent_messages or messages),
            week=WEEK,
            day=DAY,
            extra={"command": "assistant_rerank"},
        )
        scores = parse_rerank(reranked.text, len(hits))
        kept, passed = apply_rerank(hits, scores, THRESHOLD, ANSWER_K)

    lines = [branch, "mcp: git_branch"]
    if diff_text is not None:
        lines.append("mcp: git_diff_stat")
    if passed:
        lines.extend(f"источник: {hit.source} — {hit.section}" for hit in kept)
    lines.append(f"порог: {passed}/{candidates}")

    if kind == "branch":
        _stderr(lines)
        return branch
    if kind == "diff":
        _stderr(lines)
        return diff_text or ""
    if not kept:
        _stderr(lines)
        return "не знаю"

    answer_messages = _answer_messages(text, kept, branch)
    answered = _complete(complete, _answer_config(), answer_messages)
    log_call(
        answered,
        list(answered.sent_messages or answer_messages),
        week=WEEK,
        day=DAY,
        extra={"command": "assistant_help"},
    )
    _stderr(lines)
    return answered.text.strip()
