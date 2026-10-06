"""Chunking strategies: pure `Document -> list[Chunk]`, no network (SPEC-w05d21.md §2, §3).

Two strategies, deliberately naive vs structure-aware, so the day-21 comparison
is honest: `fixed` never looks at content, `structure` never snaps to size.
"""

from __future__ import annotations

import ast
import bisect
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from advent_core.errors import AdventError

MAX_CHUNK_CHARS = 4000
FIXED_SIZE = 1000
FIXED_OVERLAP = 150

_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_IMPORT_TYPES = (ast.Import, ast.ImportFrom)


@dataclass(frozen=True, slots=True)
class Document:
    """One corpus file's content, already normalized (week_05/corpus.py)."""

    source: str  # repo-relative POSIX
    text: str  # LF-normalized


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit, with metadata comparable across strategies (SPEC §3)."""

    chunk_id: str
    strategy: str
    source: str
    title: str
    section: str
    ordinal: int
    char_start: int
    char_end: int
    line_start: int  # 1-based inclusive
    line_end: int  # 1-based inclusive
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Char/line offset helpers, shared by both strategies.
# ---------------------------------------------------------------------------


def _line_start_offsets(text: str) -> list[int]:
    """Char offset where each 0-based line starts (as `text.split("\\n")` would enumerate them)."""
    offsets = [0]
    start = 0
    while True:
        idx = text.find("\n", start)
        if idx == -1:
            break
        offsets.append(idx + 1)
        start = idx + 1
    return offsets


def _line_number(offset: int, line_starts: list[int]) -> int:
    """1-based line number containing `offset`."""
    idx = bisect.bisect_right(line_starts, offset) - 1
    return idx + 1


def _sliding_windows(start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Fixed-size, fixed-overlap windows over [start, end).

    Stops as soon as a window reaches `end` — this is what keeps a trailing
    remainder shorter than `overlap` from producing a duplicate tail window
    (see chunk_fixed's docstring for the failure mode this avoids).
    """
    windows: list[tuple[int, int]] = []
    step = size - overlap
    pos = start
    while True:
        window_end = min(pos + size, end)
        windows.append((pos, window_end))
        if window_end >= end:
            break
        pos += step
    return windows


# ---------------------------------------------------------------------------
# Markdown structure: heading path, fence-aware.
# ---------------------------------------------------------------------------


def _md_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(0-based line index, level, title) for each ATX heading not inside a fence."""
    fence_char: str | None = None
    fence_len = 0
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        fence = _FENCE_RE.match(line.strip())
        if fence:
            ch = fence.group(1)[0]
            length = len(fence.group(1))
            if fence_char is None:
                fence_char, fence_len = ch, length
            elif fence_char == ch and length >= fence_len:
                fence_char = None  # only a same-char, >=-length fence closes it
            continue
        if fence_char is not None:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            headings.append((i, len(heading.group(1)), heading.group(2)))
    return headings


def _md_title(text: str, source: str) -> str:
    for _, level, title in _md_headings(text.split("\n")):
        if level == 1:
            return title
    return Path(source).name


def _md_sections(text: str) -> list[tuple[int, int, str]]:
    """(char_start, char_end, heading-path) covering the whole document, in order."""
    lines = text.split("\n")
    offsets = _line_start_offsets(text)
    end_char = len(text)

    def char_at(line_idx: int) -> int:
        return offsets[line_idx] if line_idx < len(offsets) else end_char

    headings = _md_headings(lines)
    if not headings:
        return [(0, end_char, "(preamble)")]

    sections: list[tuple[int, int, str]] = []
    if headings[0][0] > 0:
        sections.append((0, char_at(headings[0][0]), "(preamble)"))

    stack: list[tuple[int, str]] = []
    for idx, (line_idx, level, title) in enumerate(headings):
        end_line = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        while stack and stack[-1][0] >= level:  # reset deeper/equal levels on a new heading
            stack.pop()
        stack.append((level, title))
        path = " > ".join(t for _, t in stack)
        sections.append((char_at(line_idx), char_at(end_line), path))
    return sections


# ---------------------------------------------------------------------------
# Python structure: top-level ast nodes.
# ---------------------------------------------------------------------------


def _module_docstring_node(body: list[ast.stmt]) -> ast.stmt | None:
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first
    return None


def _py_category(node: ast.stmt, docstring_node: ast.stmt | None) -> str:
    if isinstance(node, _DEF_TYPES):
        return "def"
    if isinstance(node, _IMPORT_TYPES) or node is docstring_node:
        return "mod"
    return "other"


def _py_label(node: ast.stmt) -> str:
    kind = "class" if isinstance(node, ast.ClassDef) else "def"
    return f"{kind} {node.name}"


def _extend_over_comments(lines: list[str], start_line: int, lower_bound_line: int) -> int:
    """Walk a 1-based def/class start upward over contiguous `#` comment lines.

    Stops at the first blank/non-comment line, and never crosses into the
    previous top-level node's own content (`lower_bound_line`).
    """
    result = start_line
    cand = start_line - 1
    while cand > lower_bound_line:
        candidate_line = lines[cand - 1].strip()
        if candidate_line.startswith("#") and candidate_line != "":
            result = cand
            cand -= 1
        else:
            break
    return result


def _py_sections(text: str) -> list[tuple[int, int, str]]:
    """(char_start, char_end, section label), covering the whole file.

    Consecutive top-level import/docstring nodes merge into one "(module)"
    run; a def/class is always its own run and steals contiguous comment
    lines directly above it; anything else merges into "(module-level)".
    Boundary between two runs = the (possibly comment-extended) start of the
    later one — this is what lets a def's comment block "win" a shared gap.
    """
    tree = ast.parse(text)
    body = tree.body
    end_char = len(text)
    if not body:
        return [(0, end_char, "(module)")]

    offsets = _line_start_offsets(text)

    def char_at(line: int) -> int:
        idx0 = line - 1
        return offsets[idx0] if 0 <= idx0 < len(offsets) else end_char

    docstring_node = _module_docstring_node(body)
    lines = text.split("\n")

    runs: list[tuple[str, ast.stmt]] = []  # (category, first-node-of-run); def/class never merge
    for node in body:
        cat = _py_category(node, docstring_node)
        if cat != "def" and runs and runs[-1][0] == cat:
            continue  # already covered by the run's first node
        runs.append((cat, node))

    def run_start(cat: str, node: ast.stmt) -> int:
        decorators = getattr(node, "decorator_list", None)
        raw = decorators[0].lineno if decorators else node.lineno
        if cat != "def":
            return raw
        idx = body.index(node)
        lower_bound = body[idx - 1].end_lineno if idx > 0 else 0
        return _extend_over_comments(lines, raw, lower_bound)

    starts = [run_start(cat, node) for cat, node in runs]
    starts_char = [char_at(s) for s in starts]

    sections: list[tuple[int, int, str]] = []
    for i, (cat, node) in enumerate(runs):
        char_start = 0 if i == 0 else starts_char[i]
        char_end = starts_char[i + 1] if i + 1 < len(runs) else end_char
        if cat == "mod":
            label = "(module)"
        elif cat == "other":
            label = "(module-level)"
        else:
            label = _py_label(node)
        sections.append((char_start, char_end, label))
    return sections


# ---------------------------------------------------------------------------
# Dispatch + title.
# ---------------------------------------------------------------------------


def _is_python(source: str) -> bool:
    return source.endswith(".py")


def _title_for(doc: Document) -> str:
    if _is_python(doc.source):
        return doc.source[: -len(".py")].replace("/", ".")
    return _md_title(doc.text, doc.source)


def _document_sections(doc: Document) -> tuple[list[tuple[int, int, str]], bool]:
    """Sections + whether parsing failed (Python only; caller decides whether to warn)."""
    if _is_python(doc.source):
        try:
            return _py_sections(doc.text), False
        except SyntaxError:
            return [(0, len(doc.text), "(unparsed)")], True
    return _md_sections(doc.text), False


def _nearest_section(sections: list[tuple[int, int, str]], pos: int) -> str:
    for start, end, label in sections:
        if start <= pos < end:
            return label
    return sections[-1][2] if sections else "(preamble)"


def _make_chunk(
    strategy: str,
    doc: Document,
    title: str,
    section: str,
    ordinal: int,
    char_start: int,
    char_end: int,
    line_starts: list[int],
) -> Chunk:
    text = doc.text[char_start:char_end]
    return Chunk(
        chunk_id=f"{strategy}:{doc.source}#{ordinal}",
        strategy=strategy,
        source=doc.source,
        title=title,
        section=section,
        ordinal=ordinal,
        char_start=char_start,
        char_end=char_end,
        line_start=_line_number(char_start, line_starts),
        line_end=_line_number(max(char_start, char_end - 1), line_starts),
        text=text,
    )


# ---------------------------------------------------------------------------
# Public strategies.
# ---------------------------------------------------------------------------


def chunk_fixed(doc: Document, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[Chunk]:
    """Pure sliding-window chunking, no sentence snapping (SPEC §2 — the honest baseline).

    `section`/`title` are still filled in, from the same structural map
    `chunk_structure` uses, so the two strategies' metadata is comparable —
    but the window boundaries themselves never look at content.
    """
    text = doc.text
    n = len(text)
    if n == 0:
        return []
    # Parse failures are chunk_structure's business to warn about; fixed just
    # wants *a* section label, so the silent (unparsed) fallback is enough.
    sections, _parse_failed = _document_sections(doc)
    title = _title_for(doc)
    line_starts = _line_start_offsets(text)
    chunks: list[Chunk] = []
    ordinal = 0
    for start, end in _sliding_windows(0, n, size, overlap):
        piece = text[start:end]
        if not piece.strip():
            continue
        section = _nearest_section(sections, start)
        chunks.append(_make_chunk("fixed", doc, title, section, ordinal, start, end, line_starts))
        ordinal += 1
    return chunks


def chunk_structure(doc: Document, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Heading-path (md) / top-level def-class (py) chunking (SPEC §2).

    Small sections are never glued together — their share is exactly the
    thing the day-21 comparison is meant to show. A section over `max_chars`
    is cut with fixed windows (overlap FIXED_OVERLAP) and gets a
    ` [part i/n]` suffix, so every chunk of every strategy, fallback
    included, stays under the invariant.
    """
    sections, parse_failed = _document_sections(doc)
    if parse_failed:
        print(f"week_05: {doc.source} не парсится как Python — секция (unparsed)", file=sys.stderr)
    title = _title_for(doc)
    line_starts = _line_start_offsets(doc.text)
    chunks: list[Chunk] = []
    ordinal = 0
    for char_start, char_end, section in sections:
        if char_end - char_start <= max_chars:
            piece = doc.text[char_start:char_end]
            if not piece.strip():
                continue
            chunks.append(
                _make_chunk(
                    "structure", doc, title, section, ordinal, char_start, char_end, line_starts
                )
            )
            ordinal += 1
            continue
        parts = _sliding_windows(char_start, char_end, max_chars, FIXED_OVERLAP)
        n_parts = len(parts)
        for i, (part_start, part_end) in enumerate(parts, start=1):
            piece = doc.text[part_start:part_end]
            if not piece.strip():
                continue
            label = f"{section} [part {i}/{n_parts}]"
            chunks.append(
                _make_chunk(
                    "structure", doc, title, label, ordinal, part_start, part_end, line_starts
                )
            )
            ordinal += 1
    assert all(c.n_chars <= max_chars for c in chunks)  # invariant guard, SPEC §2
    return chunks


STRATEGIES: dict[str, Callable[[Document], list[Chunk]]] = {
    "fixed": chunk_fixed,
    "structure": chunk_structure,
}


def chunk_corpus(docs: list[Document], strategy: str, max_chars: int | None = None) -> list[Chunk]:
    """Chunk every doc; `max_chars` caps a chunk (structure re-cuts, fixed shrinks its window)."""
    try:
        fn = STRATEGIES[strategy]
    except KeyError as exc:
        known = ", ".join(sorted(STRATEGIES))
        message = f"Неизвестная стратегия chunking: {strategy!r}. Доступны: {known}."
        raise AdventError(message) from exc
    chunks: list[Chunk] = []
    for doc in docs:
        if max_chars is None:
            chunks.extend(fn(doc))
        elif strategy == "structure":
            chunks.extend(chunk_structure(doc, max_chars))
        else:
            chunks.extend(chunk_fixed(doc, size=min(FIXED_SIZE, max_chars)))
    return chunks
