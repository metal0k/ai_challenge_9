"""Day 29 `rag --compare`: offline table of saved runs, deltas against the first file, gates."""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console
from typer.testing import CliRunner

from advent_core import console
from advent_core.errors import AdventError
from week_06 import cli, compare
from week_06 import ragbench as rb

QUESTIONS = [1, 2, 3]
UNANSWERABLE = [101]


@pytest.fixture
def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(
        console, "out", Console(file=buf, width=80, no_color=True, force_terminal=False)
    )
    monkeypatch.setattr(
        console, "err", Console(file=io.StringIO(), width=80, no_color=True, force_terminal=False)
    )
    return buf


def _cell(qid, run=1, *, kind="answerable", outcome="answered", facts=(True,), wall=10_000, **kw):
    answerable = kind == "answerable"
    ledger = (
        rb.CallRecord("answer", "ornith", "local", 100, kw.pop("completion", 800), 8000, 0.0),
    )
    defaults = dict(
        correct=(all(facts) and outcome == "answered") if answerable else outcome != "answered",
        facts=facts if answerable else (),
        facts_total=len(facts) if answerable else 0,
        wall_ms=wall,
        rerank_ms=4000,
        answer_ms=5000,
        tok_s=40.0,
        sources_cited=True if answerable else None,
        ledger=ledger,
    )
    defaults.update(kw)
    return rb.BenchRun("local", kind, qid, run, outcome, **defaults)


def _cells(runs=1, overrides=None):
    overrides = overrides or {}
    """Three answerable questions + one unanswerable, `runs` runs each."""
    cells = []
    for run in range(1, runs + 1):
        for qid in QUESTIONS:
            cells.append(_cell(qid, run, **overrides.get(qid, {})))
        cells.append(
            _cell(
                101,
                run,
                kind="unanswerable",
                outcome="empty_context",
                correct=True,
                **overrides.get(101, {}),
            )
        )
    return cells


def _write(tmp_path, name, cells, *, profile="baseline", runs=1, rev="rev1", **extra):
    data = {
        "week": 6,
        "day": 29,
        "runs": runs,
        "backends": ["local"],
        "skipped": [],
        "failed": None,
        "corpus_rev": rev,
        "questions": QUESTIONS,
        "unanswerable": UNANSWERABLE,
        "settings": {},
        "offline": None,
        "started_at": "2026-10-08T12:30:00+03:00",
        "finished_at": "2026-10-08T12:50:00+03:00",
        "profile": profile,
        "profile_fields": {"file_gb": 5.78, "quant": "Q4_K_M", "context": 40960},
        "model_identity": {
            "id": "ornith",
            "quantization": "Q4_K_M",
            "loaded_context_length": 40960,
            "publisher": "ornith-ai",
        },
        "vram": {"vram_used_start": 6000, "vram_used_peak": 6474, "vram_total": 8192},
        "results": [dataclasses.asdict(c) for c in cells],
    }
    data.update(extra)
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _render(files) -> str:
    buf = io.StringIO()
    c = Console(file=buf, width=80, no_color=True, force_terminal=False)
    c.print(compare.dates_line(files))
    for table in compare.build_tables(files):
        c.print(table)
    for line in compare.gate_lines(files):
        c.print(line, markup=False, highlight=False)
    return buf.getvalue()


def _pair(tmp_path, other_cells, *, other="cap", runs=1, ref_cells=None, **other_extra):
    ref = _write(tmp_path, "ref.json", ref_cells or _cells(runs), runs=runs)
    alt = _write(tmp_path, "alt.json", other_cells, profile=other, runs=runs, **other_extra)
    return compare.load_files([ref, alt])


# --- rendering ------------------------------------------------------------------------------


def test_renders_in_80_columns_without_ellipsis_and_each_title_once(tmp_path):
    slower = [dataclasses.replace(c, wall_ms=12_000) for c in _cells()]
    text = _render(_pair(tmp_path, slower))
    assert max(cell_len(line) for line in text.splitlines()) <= 80
    assert "…" not in text
    for title in ("Качество", "Цитаты и сбои", "Скорость, на вопрос", "Ответ и ресурсы", "Сборка"):
        assert text.count(title) == 1


def test_dates_line_comes_from_the_files_not_from_mtime(tmp_path):
    files = _pair(tmp_path, _cells())
    assert compare.dates_line(files) == (
        "из сохранённых замеров: baseline 2026-10-08 12:30; cap 2026-10-08 12:30"
    )


def test_legacy_files_without_dates_show_unknown(tmp_path):
    ref = _write(tmp_path, "old.json", _cells(), profile=None, started_at=None, finished_at=None)
    alt = _write(tmp_path, "alt.json", _cells(), profile="cap")
    files = compare.load_files([ref, alt])
    assert compare.dates_line(files) == (
        "из сохранённых замеров: день 28 неизвестно; cap 2026-10-08 12:30"
    )


def test_deltas_are_against_the_first_file_in_the_same_cell(tmp_path):
    ref = _cells()
    fast = [dataclasses.replace(c, wall_ms=8_300) for c in ref]
    text = _render(_pair(tmp_path, fast, ref_cells=ref))
    assert "10.0 s" in text  # the reference value carries no delta
    assert "8.3 s (−17%)" in text
    assert "3/3 (±0)" in text  # equal counts still show the delta explicitly


def test_count_cells_show_signed_differences(tmp_path):
    worse = _cells(overrides={1: {"facts": (False,), "correct": False}})
    text = _render(_pair(tmp_path, worse))
    row = next(line for line in text.splitlines() if line.lstrip("│ ").startswith("cap"))
    assert "2/3 (−1)" in row  # facts and all-runs-correct both lost one
    assert row.count("(−1)") == 2 and "4/4 (±0)" in row  # a single run is always stable


def test_reference_row_has_no_delta_at_all(tmp_path):
    text = _render(_pair(tmp_path, _cells()))
    row = next(line for line in text.splitlines() if line.lstrip("│ ").startswith("baseline"))
    assert "(" not in row


def test_timing_counts_every_attempt_failures_included(tmp_path):
    cells = _cells(runs=1)
    cells[0] = dataclasses.replace(cells[0], outcome="error", error="boom", wall_ms=90_000)
    m = compare.compute(cells)
    # 4 attempts: 90 s, 10 s, 10 s, 10 s -> median 10 s, p90 90 s; none dropped as an error
    assert m.median_ms == 10_000.0 and m.p90_ms == 90_000.0
    assert m.errors == 1


def test_p90_and_median_are_per_attempt_over_all_runs():
    cells = _cells(runs=3)
    assert len(cells) == 12
    assert compare.compute(cells).median_ms == 10_000.0


def test_failures_column_counts_truncated_bad_json_error_separately(tmp_path):
    cells = _cells()
    cells[0] = dataclasses.replace(cells[0], outcome="truncated", facts=(False,), correct=False)
    cells[1] = dataclasses.replace(cells[1], outcome="bad_json", facts=(False,), correct=False)
    m = compare.compute(cells)
    assert (m.truncated, m.bad_json, m.errors) == (1, 1, 0)
    text = _render(_pair(tmp_path, cells))
    assert "1/1/0" in text


def test_refusals_on_unanswerable_are_counted_separately_from_wrong_refusals():
    cells = _cells()
    cells[0] = dataclasses.replace(cells[0], outcome="empty_context", facts=(False,), correct=False)
    m = compare.compute(cells)
    assert (m.refused_ok, m.n_unanswerable, m.wrong_refusals) == (1, 1, 1)


def test_expected_source_share_and_reattributions():
    cells = _cells()
    cells[0] = dataclasses.replace(cells[0], sources_cited=False, quotes_reattributed=2)
    m = compare.compute(cells)
    assert (m.sourced, m.answered, m.reattributed) == (2, 3, 2)
    assert m.share == pytest.approx(2 / 3)


def test_vram_cell_reports_no_data_and_a_delta(tmp_path):
    files = _pair(tmp_path, _cells(), vram={"vram_used_peak": 6500, "vram_total": 8192})
    text = _render(files)
    assert "6500 MiB (+26)" in text
    files = _pair(tmp_path, _cells(), vram=None)
    assert "нет данных" in _render(files)


# --- paired median --------------------------------------------------------------------------


def test_paired_median_uses_only_questions_correct_in_both():
    ref = _cells()
    other = [dataclasses.replace(c, wall_ms=5_000) for c in _cells()]
    other[0] = dataclasses.replace(other[0], facts=(False,), correct=False, wall_ms=1)
    ref[1] = dataclasses.replace(ref[1], wall_ms=30_000)
    ref_med, med, n = compare.paired_median(ref, other)
    # question 1 is wrong in `other`, so only 2 and 3 count (and the unanswerable one)
    assert n == 3
    assert ref_med == 10_000.0  # median of 30 s, 10 s, 10 s; question 1 is left out
    assert med == 5_000.0


def test_paired_median_is_none_without_common_correct_questions():
    wrong = [dataclasses.replace(c, facts=(False,), correct=False) for c in _cells()[:3]]
    assert compare.paired_median(_cells(), wrong)[:2] == (None, None)


# --- refusals to compare --------------------------------------------------------------------


def test_one_file_is_not_a_comparison(tmp_path):
    ref = _write(tmp_path, "ref.json", _cells())
    with pytest.raises(AdventError, match="минимум два"):
        compare.load_files([ref])


def test_different_corpus_rev_is_refused(tmp_path):
    a = _write(tmp_path, "a.json", _cells(), rev="aaaaaaaaaaaaaaa")
    b = _write(tmp_path, "b.json", _cells(), profile="cap", rev="bbbbbbbbbbbbbbb")
    with pytest.raises(AdventError, match="разных снимков"):
        compare.load_files([a, b])


def test_different_question_sets_are_refused(tmp_path):
    a = _write(tmp_path, "a.json", _cells())
    b = _write(tmp_path, "b.json", _cells(), profile="cap", questions=[1, 2])
    with pytest.raises(AdventError, match="Наборы вопросов"):
        compare.load_files([a, b])


def test_different_run_counts_are_refused(tmp_path):
    a = _write(tmp_path, "a.json", _cells(1), runs=1)
    b = _write(tmp_path, "b.json", _cells(3), profile="cap", runs=3)
    with pytest.raises(AdventError, match="10×1 и 13×3"):
        compare.load_files([a, b])


def test_incomplete_coverage_is_refused(tmp_path):
    a = _write(tmp_path, "a.json", _cells())
    b = _write(tmp_path, "b.json", _cells()[:-1], profile="cap")
    with pytest.raises(AdventError, match="покрытие"):
        compare.load_files([a, b])


def test_duplicate_cells_are_refused(tmp_path):
    dup = _cells()
    dup.append(dup[0])
    a = _write(tmp_path, "a.json", _cells())
    b = _write(tmp_path, "b.json", dup, profile="cap")
    with pytest.raises(AdventError, match="дублями"):
        compare.load_files([a, b])


def test_a_file_without_local_runs_is_refused(tmp_path):
    cloud = [dataclasses.replace(c, backend="cloud") for c in _cells()]
    a = _write(tmp_path, "a.json", _cells())
    b = _write(tmp_path, "b.json", cloud, profile="cap")
    with pytest.raises(AdventError, match="нет локальных"):
        compare.load_files([a, b])


def test_same_profile_twice_gets_distinct_labels(tmp_path):
    a = _write(tmp_path, "a.json", _cells())
    b = _write(tmp_path, "b.json", _cells())
    assert [f.label for f in compare.load_files([a, b])] == ["baseline", "baseline#2"]


# --- gates ----------------------------------------------------------------------------------


def test_screening_gate_passes_when_facts_hold_and_nothing_new_fails(tmp_path):
    lines = compare.gate_lines(_pair(tmp_path, _cells()))
    assert lines[0] == "Ворота скрининга:"
    assert "cap: пройден" in lines
    assert not any(line.strip().startswith("✗") for line in lines)
    assert rb.SMALL_SAMPLE in lines[-1]


def test_screening_gate_fails_on_two_facts_lost(tmp_path):
    two = [(True, True)] * 3
    ref = [dataclasses.replace(c, facts=(True, True), facts_total=2) for c in _cells()[:3]]
    ref.append(_cells()[3])
    worse = list(ref)
    worse[0] = dataclasses.replace(worse[0], facts=(False, False), correct=False)
    worse[1] = dataclasses.replace(worse[1], facts=(False, True), correct=False)
    worse[2] = dataclasses.replace(worse[2], facts=(False, True), correct=False)
    assert len(two) == 3
    lines = compare.gate_lines(_pair(tmp_path, worse, ref_cells=ref))
    assert "cap: не пройден" in lines
    assert any(line.strip().startswith("✗ факты 2") for line in lines)


def test_screening_gate_tolerates_one_lost_fact(tmp_path):
    worse = _cells(overrides={1: {"facts": (False,), "correct": False}})
    assert "cap: пройден" in compare.gate_lines(_pair(tmp_path, worse))


def test_screening_gate_fails_on_a_new_truncation(tmp_path):
    worse = _cells(overrides={1: {"outcome": "truncated", "facts": (False,), "correct": False}})
    lines = compare.gate_lines(_pair(tmp_path, worse))
    assert "cap: не пройден" in lines and any("новые сбои: обрывы" in line for line in lines)


def test_vram_lever_needs_700_mib_free_and_no_data_fails(tmp_path):
    tight = {"vram_used_peak": 7600, "vram_total": 8192}
    lines = compare.gate_lines(_pair(tmp_path, _cells(), other="ctx24k", vram=tight))
    assert "ctx24k: не пройден" in lines and any("592 MiB" in line for line in lines)
    roomy = {"vram_used_peak": 7000, "vram_total": 8192}
    assert "ctx24k: пройден" in compare.gate_lines(
        _pair(tmp_path, _cells(), other="ctx24k", vram=roomy)
    )
    assert "q3: не пройден" in compare.gate_lines(_pair(tmp_path, _cells(), other="q3", vram=None))
    # a lever outside the VRAM axis is not gated by memory at all
    assert "cap: пройден" in compare.gate_lines(_pair(tmp_path, _cells(), other="cap", vram=None))


def _final(tmp_path, tuned_cells, ref_cells=None):
    return _pair(tmp_path, tuned_cells, other="tuned", runs=3, ref_cells=ref_cells)


def test_final_gate_succeeds_when_all_six_conditions_hold_and_it_is_faster(tmp_path):
    fast = [dataclasses.replace(c, wall_ms=6_000) for c in _cells(runs=3)]
    lines = compare.gate_lines(_final(tmp_path, fast))
    assert lines[0] == "Ворота финала:" and "tuned: успех" in lines
    assert sum(1 for line in lines if line.strip().startswith("✓")) == 6
    assert rb.SMALL_SAMPLE not in lines[-1]


def test_final_gate_not_met_is_named_as_the_result_of_the_day(tmp_path):
    same_speed = _cells(runs=3)
    lines = compare.gate_lines(_final(tmp_path, same_speed))
    assert "tuned: не выполнено — это результат дня" in lines
    assert any(line.strip().startswith("✗ парная медиана") for line in lines)


def test_final_gate_checks_unanswerable_refusals_separately(tmp_path):
    fast = [dataclasses.replace(c, wall_ms=6_000) for c in _cells(runs=3)]
    fast = [
        dataclasses.replace(c, outcome="answered", correct=False) if c.question_id == 101 else c
        for c in fast
    ]
    lines = compare.gate_lines(_final(tmp_path, fast))
    assert "tuned: не выполнено — это результат дня" in lines
    assert any(line.strip().startswith("✗ верные отказы на неотвечаемых 0/3") for line in lines)


def test_final_gate_source_share_may_drop_ten_points_but_not_more(tmp_path):
    fast = [dataclasses.replace(c, wall_ms=6_000) for c in _cells(runs=10)]
    ref = _cells(runs=10)
    # 9 answerable answered per question-run... use 30 answered cells: 3 unsourced = 10 p.p.
    unsourced = [i for i, c in enumerate(fast) if c.kind == "answerable"][:3]
    ok = list(fast)
    for i in unsourced:
        ok[i] = dataclasses.replace(ok[i], sources_cited=False)
    files = _pair(tmp_path, ok, other="tuned", runs=10, ref_cells=ref)
    assert any(
        line.strip().startswith("✓ источник в цитатах") for line in compare.gate_lines(files)
    )
    bad = list(fast)
    for i in [i for i, c in enumerate(fast) if c.kind == "answerable"][:4]:
        bad[i] = dataclasses.replace(bad[i], sources_cited=False)
    files = _pair(tmp_path, bad, other="tuned", runs=10, ref_cells=ref)
    assert any(
        line.strip().startswith("✗ источник в цитатах") for line in compare.gate_lines(files)
    )


def test_final_gate_reattributions_may_not_more_than_double(tmp_path):
    ref = _cells(runs=3)
    ref[0] = dataclasses.replace(ref[0], quotes_reattributed=1)
    fast = [dataclasses.replace(c, wall_ms=6_000) for c in _cells(runs=3)]
    fast[0] = dataclasses.replace(fast[0], quotes_reattributed=2)
    assert any(
        line.strip().startswith("✓ исправленных источников 2")
        for line in compare.gate_lines(_final(tmp_path, fast, ref))
    )
    fast[1] = dataclasses.replace(fast[1], quotes_reattributed=1)
    assert any(
        line.strip().startswith("✗ исправленных источников 3")
        for line in compare.gate_lines(_final(tmp_path, fast, ref))
    )


# --- the command ----------------------------------------------------------------------------


def test_cli_compare_prints_dates_tables_and_gates_without_network(tmp_path, out, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("compare must not touch LM Studio or the network")

    monkeypatch.setattr(rb.oc, "ensure_ready", boom)
    monkeypatch.setattr(rb.oc, "server_status", boom)
    a = _write(tmp_path, "a.json", _cells())
    b = _write(tmp_path, "b.json", _cells(), profile="cap")
    result = CliRunner().invoke(cli.app, ["rag", "--compare", str(a), str(b)])
    assert result.exit_code == 0, result.output
    text = out.getvalue()
    assert text.count("из сохранённых замеров:") == 1
    assert text.count("Ворота скрининга:") == 1


def test_cli_compare_rejects_extra_modes(tmp_path):
    a = _write(tmp_path, "a.json", _cells())
    result = CliRunner().invoke(cli.app, ["rag", "--compare", str(a), "--save", str(a)])
    assert result.exit_code != 0
    result = CliRunner().invoke(cli.app, ["rag", "--compare"])
    assert result.exit_code != 0


def test_old_report_command_still_loads_a_new_format_file(tmp_path, out):
    a = _write(tmp_path, "a.json", _cells())
    assert rb.run_report_command([Path(a)]) == 0
    assert "Качество" in out.getvalue()


# --- review fixes ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _control_set(monkeypatch):
    """The tests' three questions + one unanswerable stand in for the 10 + 3 control set."""
    monkeypatch.setattr(
        compare, "control_set", lambda: (frozenset(QUESTIONS), frozenset(UNANSWERABLE))
    )


def _two_facts(cells, found):
    """Every answerable cell gets two facts, `found` of them true."""
    facts = tuple(i < found for i in range(2))
    return [
        dataclasses.replace(c, facts=facts, facts_total=2, correct=found == 2)
        if c.kind == "answerable"
        else c
        for c in cells
    ]


def _quant_files(tmp_path, *, base=1, q4b=2, q3=1, q3_context=40960):
    paths = [
        _write(tmp_path, "base.json", _two_facts(_cells(), base), profile="baseline"),
        _write(tmp_path, "q4b.json", _two_facts(_cells(), q4b), profile="q4b"),
        _write(
            tmp_path,
            "q3.json",
            _two_facts(_cells(), q3),
            profile="q3",
            model_identity={
                "id": "ornith",
                "quantization": "Q3_K_M",
                "loaded_context_length": q3_context,
                "publisher": "bartowski",
            },
        ),
    ]
    return compare.load_files(paths)


def _row(text, name, needle):
    return next(
        line for line in text.splitlines() if line.lstrip("│ ").startswith(name) and needle in line
    )


def test_quant_rows_are_measured_against_q4b_not_the_official_baseline(tmp_path):
    files = _quant_files(tmp_path)  # facts: baseline 3, q4b 6, q3 3
    base, q4b, q3 = files
    assert compare.reference_for(q3, files) is q4b
    assert compare.reference_for(q4b, files) is base
    assert compare.reference_for(base, files) is base
    lines = compare.gate_lines(files)
    # 3 facts would pass against the baseline's 3, but the same build's Q4 found 6
    assert "q3 (опорный q4b): не пройден" in lines
    assert any(line.strip().startswith("✗ факты 3") and "опорный 6" in line for line in lines)
    assert "q4b: пройден" in lines  # q4b itself is judged against the first file
    assert "3/6 (−3)" in _render(files)  # the q3 row's delta is against q4b


def test_without_a_q4b_file_the_quant_rows_fall_back_to_the_first_file(tmp_path):
    base = _write(tmp_path, "b.json", _two_facts(_cells(), 2), profile="baseline")
    q3 = _write(tmp_path, "q3.json", _two_facts(_cells(), 2), profile="q3")
    files = compare.load_files([base, q3])
    assert compare.reference_for(files[1], files) is files[0]
    assert compare.reference_lines(files) == []


def test_the_quant_reference_is_named_in_the_output(tmp_path):
    files = _quant_files(tmp_path)
    assert compare.reference_lines(files) == [
        "опорный для квантов: q4b (для q3); для остальных — baseline"
    ]


def test_unknown_publisher_is_shown_as_unknown_never_taken_from_the_profile(tmp_path):
    ident = {"id": "ornith", "quantization": "Q4_K_M", "loaded_context_length": 40960}
    a = _write(tmp_path, "a.json", _cells(), model_identity=ident)
    b = _write(tmp_path, "b.json", _cells(), profile="cap", model_identity=ident)
    files = compare.load_files([a, b])
    assert [f.publisher for f in files] == [None, None]
    text = _render(files)
    assert "неизвестно" in _row(text, "cap", "Q4_K_M")
    assert "ornith-ai" not in text and "bartowski" not in text


def test_axis_is_quant_or_quant_plus_ctx_from_the_saved_contexts(tmp_path):
    same = _render(_quant_files(tmp_path, q3_context=40960))
    row = _row(same, "q3", "Q3_K_M")
    assert "квант+ctx" not in row and "квант" in row
    moved = _render(_quant_files(tmp_path, q3_context=24576))
    row = _row(moved, "q3", "Q3_K_M")
    assert "квант+ctx" in row and "24576" in row
    assert "опора" in _row(moved, "q4b", "Q4_K_M")


def test_final_verdict_needs_three_runs_over_the_whole_control_set(tmp_path):
    fast = [dataclasses.replace(c, wall_ms=6_000) for c in _cells(runs=2)]
    lines = compare.gate_lines(_pair(tmp_path, fast, other="tuned", runs=2, ref_cells=_cells(2)))
    assert lines[0].startswith("Пробное сравнение") and "без вердикта" in lines[0]
    assert "tuned: без вердикта" in lines
    assert not any("успех" in line or "не выполнено" in line for line in lines)


def test_a_three_run_file_over_a_partial_question_set_is_still_exploratory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compare, "control_set", lambda: (frozenset({1, 2, 3, 4}), frozenset(UNANSWERABLE))
    )
    fast = [dataclasses.replace(c, wall_ms=6_000) for c in _cells(runs=3)]
    lines = compare.gate_lines(_final(tmp_path, fast))
    assert "tuned: без вердикта" in lines and not any("успех" in line for line in lines)


def test_a_gate_with_a_zero_denominator_is_no_data_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "control_set", lambda: (frozenset(QUESTIONS), frozenset()))
    ref = [c for c in _cells(runs=3) if c.kind == "answerable"]
    fast = [dataclasses.replace(c, wall_ms=6_000) for c in ref]
    ref_path = _write(tmp_path, "r.json", ref, runs=3, unanswerable=[])
    alt_path = _write(tmp_path, "a.json", fast, profile="tuned", runs=3, unanswerable=[])
    lines = compare.gate_lines(compare.load_files([ref_path, alt_path]))
    assert any(line.strip() == "? верные отказы на неотвечаемых: нет данных" for line in lines)
    assert "tuned: не выполнено — это результат дня" in lines


def test_sampling_caveat_appears_for_different_loads_with_the_server_default(tmp_path):
    caveat = compare.sampling_caveat(_quant_files(tmp_path))
    assert caveat is not None and "разных загрузках" in caveat and "неизвестны" in caveat


def test_no_caveat_when_the_loads_are_identical_or_sampling_was_set(tmp_path):
    a = _write(tmp_path, "a.json", _cells())
    b = _write(tmp_path, "b.json", _cells(), profile="cap")
    assert compare.sampling_caveat(compare.load_files([a, b])) is None
    set_sampling = {"local": {"sampling_source": {"temperature": "profile", "top_p": "profile"}}}
    c = _write(tmp_path, "c.json", _cells(), settings=set_sampling)
    d = _write(
        tmp_path,
        "d.json",
        _cells(),
        profile="sampling",
        settings=set_sampling,
        model_identity={"id": "ornith", "quantization": "Q3_K_M"},
    )
    assert compare.sampling_caveat(compare.load_files([c, d])) is None


def test_the_caveat_is_printed_by_the_command(tmp_path, out):
    files = _quant_files(tmp_path)
    assert compare.run_compare_command([f.path for f in files]) == 0
    assert out.getvalue().count("оговорка: профили измерены в разных загрузках") == 1


def test_rates_table_labels_prefill_and_rerank_as_a_lower_bound(tmp_path):
    text = _render(_pair(tmp_path, _cells()))
    flat = " ".join(text.split())
    assert text.count("Токены в секунду") == 1
    assert "ток/с ответа (вкл. prefill)" in flat
    assert "rerank, ток/с (prompt+gen)" in flat
    assert "prompt_tokens / время rerank: нижняя оценка" in flat
    assert "completion / время всего ответа (prefill включён)" in flat
    assert "tok/s" not in text
    assert "…" not in text and max(cell_len(line) for line in text.splitlines()) <= 80


def test_rerank_tokens_per_second_is_prompt_tokens_over_rerank_time():
    cell = _cell(1, rerank_ms=4000)
    rerank_call = rb.CallRecord("rerank", "ornith", "local", 8000, 100, 4000, 0.0)
    cell = dataclasses.replace(cell, ledger=(*cell.ledger, rerank_call))
    assert compare.compute([cell]).rerank_tps == pytest.approx(2000.0)


def test_rerank_rate_is_unknown_without_prompt_tokens():
    assert compare.compute([_cell(1)]).rerank_tps is None


# --- --tables --------------------------------------------------------------------------------

TITLES = {
    "quality": "Качество",
    "citations": "Цитаты и сбои",
    "speed": "Скорость, на вопрос",
    "resources": "Ответ и ресурсы",
    "tps": "Токены в секунду",
    "build": "Сборка модели",
}


def _run(out, *args):
    result = CliRunner().invoke(cli.app, ["rag", "--compare", *map(str, args)])
    assert result.exit_code == 0, result.output
    text = out.getvalue()
    out.truncate(0)
    out.seek(0)
    return text


def _screening_files(tmp_path):
    a = _write(tmp_path, "a.json", _cells())
    b = _write(tmp_path, "b.json", _cells(), profile="cap")
    return a, b


def test_tables_option_is_parsed_and_unknown_key_lists_valid_ones(tmp_path, out):
    a, b = _screening_files(tmp_path)
    assert compare.parse_tables(None) is None
    assert compare.parse_tables("speed, gates") == frozenset({"speed", "gates"})
    with pytest.raises(cli.ConfigError) as err:
        compare.parse_tables("speeed")
    for key in compare.ALL_KEYS:
        assert key in str(err.value)
    result = CliRunner().invoke(cli.app, ["rag", "--compare", "--tables", "speeed", str(a), str(b)])
    assert result.exit_code != 0
    result = CliRunner().invoke(cli.app, ["rag", "--tables", "speed"])
    assert result.exit_code != 0


@pytest.mark.parametrize("key", list(TITLES))
def test_each_table_key_prints_only_its_block(tmp_path, out, key):
    a, b = _screening_files(tmp_path)
    text = _run(out, "--tables", key, a, b)
    for other, title in TITLES.items():
        assert text.count(title) == (1 if other == key else 0), (key, title)
    assert "Ворота" not in text
    assert text.count("из сохранённых замеров:") == 1


def test_speed_only_has_no_quality_title_and_one_speed_title(tmp_path, out):
    a, b = _screening_files(tmp_path)
    text = _run(out, "--tables", "speed", a, b)
    assert text.count("Качество") == 0 and text.count("Скорость") == 1


def test_gates_and_verdicts_keys_print_only_their_block(tmp_path, out):
    a, b = _screening_files(tmp_path)
    gates = _run(out, "--tables", "gates", a, b)
    assert gates.count("Ворота скрининга:") == 1 and "Качество" not in gates
    verdicts = _run(out, "--tables", "verdicts", a, b)
    assert "Ворота скрининга:" not in verdicts and "Качество" not in verdicts
    assert verdicts.count("cap: пройден") == 1


def test_footnotes_travel_with_their_table(tmp_path, out):
    a, b = _screening_files(tmp_path)
    tps = " ".join(_run(out, "--tables", "tps", a, b).split())
    assert compare.TPS_NOTE in tps
    speed = " ".join(_run(out, "--tables", "speed", a, b).split())
    assert compare.TPS_NOTE not in speed
    assert "медиана и p90" in speed


def test_default_output_is_the_old_full_concatenation(tmp_path, out):
    a, b = _screening_files(tmp_path)
    files = compare.load_files([a, b])
    buf = io.StringIO()
    c = Console(file=buf, width=80, no_color=True, force_terminal=False)
    for line in (compare.dates_line(files), *compare.reference_lines(files)):
        c.print(line, markup=False, highlight=False)
    caveat = compare.sampling_caveat(files)
    if caveat:
        c.print(caveat, markup=False, highlight=False)
    for f in files:
        if not f.vram.known:
            c.print(f"{f.label}: {compare.vram.NO_DATA}", markup=False, highlight=False)
    for table in compare.build_tables(files):
        c.print(table)
    for line in compare.gate_lines(files):
        c.print(line, markup=False, highlight=False)
    assert _run(out, a, b) == buf.getvalue()
    keys = ",".join([*TITLES, "gates"])
    assert _run(out, "--tables", keys, a, b) == buf.getvalue()


def test_screening_verdicts_name_the_failed_check_with_the_real_gate_text(tmp_path):
    ref = [dataclasses.replace(c, facts=(True, True), facts_total=2) for c in _cells()[:3]]
    ref.append(_cells()[3])
    worse = list(ref)
    worse[0] = dataclasses.replace(worse[0], facts=(False, False), correct=False)
    worse[1] = dataclasses.replace(worse[1], facts=(False, True), correct=False)
    worse[2] = dataclasses.replace(worse[2], facts=(False, True), correct=False)
    files = _pair(tmp_path, worse, ref_cells=ref)
    (line,) = compare.verdict_lines(files)
    assert line.startswith("cap: не пройден — ")
    failed = [ln.strip()[2:] for ln in compare.gate_lines(files) if ln.strip().startswith("✗")]
    assert failed and line == "cap: не пройден — " + "; ".join(failed)
    assert compare.verdict_lines(_pair(tmp_path, _cells())) == ["cap: пройден"]


def test_final_verdicts_use_the_final_gate(tmp_path):
    fast = [dataclasses.replace(c, wall_ms=6_000) for c in _cells(runs=3)]
    assert compare.verdict_lines(_final(tmp_path, fast)) == ["tuned: выполнено"]
    files = _final(tmp_path, _cells(runs=3))
    (line,) = compare.verdict_lines(files)
    failed = [ln.strip()[2:] for ln in compare.gate_lines(files) if ln.strip().startswith("✗")]
    assert failed and line == "tuned: не выполнено — " + "; ".join(failed)
    assert "парная медиана" in line


def test_selected_blocks_render_in_80_columns_without_ellipsis(tmp_path, out):
    a, b = _screening_files(tmp_path)
    text = _run(out, "--tables", "quality,speed,resources,verdicts", a, b)
    assert "…" not in text
    assert max(cell_len(line) for line in text.splitlines()) <= 80
