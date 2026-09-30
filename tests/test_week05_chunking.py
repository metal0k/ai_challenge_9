"""week_05/chunking.py — fixed vs structure, both pure and offline (SPEC-w05d21.md §2, §3, §9).

Every boundary asserted here is a literal computed by hand (or via
`str.index`/plain slicing on the fixture text, independent of the code under
test), never a value read back from the module itself.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from advent_core.errors import AdventError
from week_05.chunking import (
    FIXED_OVERLAP,
    FIXED_SIZE,
    MAX_CHUNK_CHARS,
    STRATEGIES,
    Chunk,
    Document,
    chunk_corpus,
    chunk_fixed,
    chunk_structure,
)


def _digits(n: int) -> str:
    """n chars of a repeating 0-9 pattern, single line (no "\\n") — a content-blind fixture."""
    return ("0123456789" * ((n // 10) + 1))[:n]


# ---------------------------------------------------------------------------
# chunk_fixed
# ---------------------------------------------------------------------------


def test_fixed_defaults_match_module_constants() -> None:
    assert FIXED_SIZE == 1000
    assert FIXED_OVERLAP == 150
    assert MAX_CHUNK_CHARS == 4000


def test_fixed_exact_boundaries_on_2300_chars() -> None:
    text = _digits(2300)
    doc = Document(source="fixture/long.md", text=text)
    chunks = chunk_fixed(doc)

    assert [(c.char_start, c.char_end) for c in chunks] == [
        (0, 1000),
        (850, 1850),
        (1700, 2300),
    ]
    assert [c.n_chars for c in chunks] == [1000, 1000, 600]
    # text == doc.text[char_start:char_end], independent of the code under test.
    assert chunks[0].text == text[0:1000]
    assert chunks[1].text == text[850:1850]
    assert chunks[2].text == text[1700:2300]
    assert [c.chunk_id for c in chunks] == [
        "fixed:fixture/long.md#0",
        "fixed:fixture/long.md#1",
        "fixed:fixture/long.md#2",
    ]
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert all(c.strategy == "fixed" for c in chunks)
    assert all(c.source == "fixture/long.md" for c in chunks)
    # single-line fixture: every chunk lives on line 1.
    assert all((c.line_start, c.line_end) == (1, 1) for c in chunks)
    # no headings anywhere -> whole-doc fallback section/title.
    assert all(c.section == "(preamble)" for c in chunks)
    assert all(c.title == "long.md" for c in chunks)


def test_fixed_overlap_is_150_chars_between_consecutive_windows() -> None:
    text = _digits(2300)
    doc = Document(source="fixture/long.md", text=text)
    chunks = chunk_fixed(doc)
    assert chunks[0].text[-150:] == chunks[1].text[:150]
    assert chunks[1].text[-150:] == chunks[2].text[:150]


def test_fixed_trailing_remainder_shorter_than_overlap_is_not_a_duplicate_chunk() -> None:
    # n=1701: a naive "start += step while start < n" loop would add a 3rd,
    # 1-char chunk [1700:1701) that duplicates content already covered by
    # the 2nd chunk's tail — see _sliding_windows' docstring.
    text = _digits(1701)
    doc = Document(source="fixture/tail.md", text=text)
    chunks = chunk_fixed(doc)
    assert [(c.char_start, c.char_end) for c in chunks] == [(0, 1000), (850, 1701)]
    assert chunks[-1].char_end == len(text)


def test_fixed_exact_single_window_is_one_chunk() -> None:
    text = _digits(1000)
    doc = Document(source="fixture/exact.md", text=text)
    chunks = chunk_fixed(doc)
    assert [(c.char_start, c.char_end) for c in chunks] == [(0, 1000)]


def test_fixed_empty_document_is_zero_chunks() -> None:
    doc = Document(source="fixture/empty.md", text="")
    assert chunk_fixed(doc) == []


def test_fixed_custom_size_and_overlap() -> None:
    text = _digits(250)
    doc = Document(source="fixture/small.md", text=text)
    chunks = chunk_fixed(doc, size=100, overlap=20)
    # step=80: starts 0, 80, 160; last window (160,250) len 90 reaches end -> stop.
    assert [(c.char_start, c.char_end) for c in chunks] == [
        (0, 100),
        (80, 180),
        (160, 250),
    ]


def test_fixed_section_is_nearest_heading_containing_char_start() -> None:
    text = "# Top\nintro text here that is fairly short\n## Sub\n" + _digits(1200)
    doc = Document(source="fixture/headed.md", text=text)
    chunks = chunk_fixed(doc, size=100, overlap=20)
    # every window's start lands inside the "Top > Sub" section (all content
    # after the two headings), so every chunk should report that path.
    tail_start = text.index(_digits(1200))
    tail_chunks = [c for c in chunks if c.char_start >= tail_start]
    assert tail_chunks  # sanity: fixture actually produced windows in the tail
    assert all(c.section == "Top > Sub" for c in tail_chunks)
    assert all(c.title == "Top" for c in tail_chunks)


def test_fixed_never_crosses_a_file_boundary(tmp_path=None) -> None:
    doc_a = Document(source="a.md", text=_digits(1500))
    doc_b = Document(source="b.md", text=_digits(1500))
    chunks = chunk_corpus([doc_a, doc_b], "fixed")
    a_chunks = [c for c in chunks if c.source == "a.md"]
    b_chunks = [c for c in chunks if c.source == "b.md"]
    assert [c.ordinal for c in a_chunks] == [0, 1]
    assert [c.ordinal for c in b_chunks] == [0, 1]  # ordinal resets per source
    assert [c.chunk_id for c in a_chunks] == ["fixed:a.md#0", "fixed:a.md#1"]
    assert [c.chunk_id for c in b_chunks] == ["fixed:b.md#0", "fixed:b.md#1"]
    assert len(chunks) == len(a_chunks) + len(b_chunks)


# ---------------------------------------------------------------------------
# chunk_structure — markdown
# ---------------------------------------------------------------------------


def test_structure_md_heading_path_with_preamble_and_level_reset() -> None:
    text = (
        "preamble text here\n"
        "\n"
        "# Title One\n"
        "content1\n"
        "## Sub A\n"
        "content2\n"
        "### Sub A1\n"
        "content3\n"
        "## Sub B\n"
        "content4\n"
        "# Title Two\n"
        "content5\n"
    )
    doc = Document(source="fixture/headings.md", text=text)
    chunks = chunk_structure(doc)

    assert [c.section for c in chunks] == [
        "(preamble)",
        "Title One",
        "Title One > Sub A",
        "Title One > Sub A > Sub A1",
        "Title One > Sub B",
        "Title Two",
    ]
    assert [c.ordinal for c in chunks] == list(range(6))
    assert all(c.title == "Title One" for c in chunks)  # first H1 in the file
    assert all(c.strategy == "structure" for c in chunks)
    # invariant: text is exactly the slice it claims to be.
    for c in chunks:
        assert c.text == text[c.char_start : c.char_end]
    # sections partition the whole document, contiguously.
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)
    for prev, nxt in pairwise(chunks):
        assert prev.char_end == nxt.char_start


def test_structure_md_fence_hash_is_not_a_heading_and_fence_char_must_match() -> None:
    text = "# H\n~~~\n```\n# inside\n~~~\n## After\n"
    doc = Document(source="fixture/fence.md", text=text)
    chunks = chunk_structure(doc)

    assert [c.section for c in chunks] == ["H", "H > After"]
    # the fake "# inside" heading and the mismatched ``` fence stay inside "H".
    assert "# inside" in chunks[0].text
    assert "```" in chunks[0].text
    assert chunks[1].text == "## After\n"


def test_structure_md_shorter_closing_fence_does_not_close_a_longer_opening_fence() -> None:
    # 4-backtick opener; a 3-backtick line is too short to close it, so the
    # "heading" inside stays fenced-out — only a >=4-backtick line closes it.
    text = "````\n```\n# not heading\n````\n## After\n"
    doc = Document(source="fixture/nested_fence.md", text=text)
    chunks = chunk_structure(doc)
    assert [c.section for c in chunks] == ["(preamble)", "After"]
    assert "# not heading" in chunks[0].text


def test_structure_md_no_headings_is_one_preamble_section() -> None:
    text = "just some prose\nwith two lines\n"
    doc = Document(source="fixture/plain.md", text=text)
    chunks = chunk_structure(doc)
    assert len(chunks) == 1
    assert chunks[0].section == "(preamble)"
    assert chunks[0].text == text
    assert chunks[0].title == "plain.md"  # no H1 -> file name


def test_structure_md_long_section_is_split_into_parts_with_suffix() -> None:
    text = "# Big\n" + "A" * 700
    doc = Document(source="fixture/big.md", text=text)
    chunks = chunk_structure(doc, max_chars=500)

    # section spans the whole file (0, 706); sliding_windows(0,706,500,150):
    # start=0 -> end=500 (<706); next start=350 -> end=706 (==706), stop.
    assert [(c.char_start, c.char_end) for c in chunks] == [(0, 500), (350, 706)]
    assert [c.section for c in chunks] == ["Big [part 1/2]", "Big [part 2/2]"]
    assert [c.n_chars for c in chunks] == [500, 356]
    assert all(c.n_chars <= 500 for c in chunks)


def test_structure_md_whitespace_only_preamble_is_skipped() -> None:
    text = "\n\n# A\ncontent\n"  # 2 blank lines before the first heading
    doc = Document(source="fixture/skip.md", text=text)
    chunks = chunk_structure(doc)
    assert [c.section for c in chunks] == ["A"]  # no "(preamble)" entry


# ---------------------------------------------------------------------------
# chunk_structure — python
# ---------------------------------------------------------------------------


_PY_LINES = [
    '"""Module docstring."""',
    "",
    "import os",
    "import sys",
    "",
    "X = 1",
    "",
    "",
    "# leading comment",
    "# second comment line",
    "@decorator",
    "def foo():",
    "    return 1",
    "",
    "",
    "class Bar:",
    '    """Bar docstring."""',
    "",
    "    def method(self):",
    "        pass",
    "",
    "",
    'if __name__ == "__main__":',
    "    foo()",
]
_PY_TEXT = "\n".join(_PY_LINES) + "\n"


def test_structure_py_module_defs_and_module_level_sections() -> None:
    doc = Document(source="advent_core/fixture.py", text=_PY_TEXT)
    chunks = chunk_structure(doc)

    assert [c.section for c in chunks] == [
        "(module)",
        "(module-level)",
        "def foo",
        "class Bar",
        "(module-level)",
    ]
    assert all(c.title == "advent_core.fixture" for c in chunks)
    assert [c.line_start for c in chunks] == [1, 6, 9, 16, 23]
    assert [c.line_end for c in chunks] == [5, 8, 15, 22, 24]
    for c in chunks:
        assert c.text == _PY_TEXT[c.char_start : c.char_end]


def test_structure_py_decorator_and_contiguous_comments_are_part_of_the_def_section() -> None:
    doc = Document(source="advent_core/fixture.py", text=_PY_TEXT)
    chunks = chunk_structure(doc)
    foo_section = next(c for c in chunks if c.section == "def foo")
    assert foo_section.char_start == _PY_TEXT.index("# leading comment")
    assert "# second comment line" in foo_section.text
    assert "@decorator" in foo_section.text
    assert "def foo():" in foo_section.text
    assert "return 1" in foo_section.text
    assert "class Bar" not in foo_section.text


def test_structure_py_module_section_holds_docstring_and_imports_not_the_constant() -> None:
    doc = Document(source="advent_core/fixture.py", text=_PY_TEXT)
    chunks = chunk_structure(doc)
    module_section = next(c for c in chunks if c.section == "(module)")
    assert "Module docstring" in module_section.text
    assert "import os" in module_section.text
    assert "import sys" in module_section.text
    assert "X = 1" not in module_section.text

    module_level = [c for c in chunks if c.section == "(module-level)"]
    assert "X = 1" in module_level[0].text
    assert "__name__" in module_level[1].text


def test_structure_py_nested_method_is_not_its_own_top_level_section() -> None:
    doc = Document(source="advent_core/fixture.py", text=_PY_TEXT)
    chunks = chunk_structure(doc)
    class_section = next(c for c in chunks if c.section == "class Bar")
    assert "def method" in class_section.text
    assert "pass" in class_section.text
    assert not any(c.section == "def method" for c in chunks)


def test_structure_py_unparsable_file_is_unparsed_section_with_stderr_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = "def broken(:\n    pass\n"
    doc = Document(source="advent_core/broken.py", text=text)
    chunks = chunk_structure(doc)

    assert [(c.section, c.char_start, c.char_end) for c in chunks] == [("(unparsed)", 0, len(text))]
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout stays clean — this is a warning, not the product
    assert "advent_core/broken.py" in captured.err
    assert "не парсится" in captured.err


def test_fixed_on_unparsable_python_file_does_not_warn() -> None:
    # chunk_fixed still needs a section label, but the warning belongs to
    # chunk_structure (SPEC §2) — no stderr side effect here.
    text = "def broken(:\n    pass\n" * 60  # long enough to span >1 window
    doc = Document(source="advent_core/broken.py", text=text)
    chunks = chunk_fixed(doc)
    assert all(c.section == "(unparsed)" for c in chunks)


def test_structure_py_dotted_module_title() -> None:
    doc = Document(source="advent_core/config.py", text='"""Doc."""\n')
    chunks = chunk_structure(doc)
    assert chunks[0].title == "advent_core.config"


# ---------------------------------------------------------------------------
# Metadata / registry / chunk_corpus
# ---------------------------------------------------------------------------


def test_chunk_id_is_stable_across_repeated_calls() -> None:
    doc = Document(source="fixture/stable.md", text=_digits(2300))
    first = [c.chunk_id for c in chunk_fixed(doc)]
    second = [c.chunk_id for c in chunk_fixed(doc)]
    assert first == second


def test_strategies_registry_matches_module_functions() -> None:
    assert {"fixed": chunk_fixed, "structure": chunk_structure} == STRATEGIES


def test_chunk_corpus_concatenates_all_documents() -> None:
    docs = [
        Document(source="a.md", text="# A\nfoo\n"),
        Document(source="b.md", text="# B\nbar\n"),
    ]
    chunks = chunk_corpus(docs, "structure")
    assert [c.source for c in chunks] == ["a.md", "b.md"]
    assert [c.chunk_id for c in chunks] == ["structure:a.md#0", "structure:b.md#0"]


def test_chunk_corpus_unknown_strategy_raises_advent_error() -> None:
    with pytest.raises(AdventError):
        chunk_corpus([Document(source="a.md", text="x")], "bogus")


def test_chunk_dataclass_is_frozen_and_n_chars_matches_text() -> None:
    doc = Document(source="a.md", text="# A\nhello world\n")
    chunk = chunk_structure(doc)[0]
    assert isinstance(chunk, Chunk)
    assert chunk.n_chars == len(chunk.text)
    with pytest.raises(AttributeError):
        chunk.text = "mutated"  # type: ignore[misc]


def test_no_chunk_of_any_strategy_exceeds_max_chunk_chars() -> None:
    text = "# Big\n" + _digits(9000)
    doc = Document(source="fixture/huge.md", text=text)
    for strategy_chunks in (chunk_fixed(doc), chunk_structure(doc)):
        assert all(c.n_chars <= MAX_CHUNK_CHARS for c in strategy_chunks)
