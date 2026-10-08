"""Day 29 demo and rehearsal scenarios: the exact step lines must survive the real CLI parser."""

from __future__ import annotations

import io

import pytest
import typer.main
from rich.cells import cell_len
from rich.console import Console

from advent_cli import record
from week_06 import cli, compare, profiles


def _parse(args: list[str]):
    """Parse a step's args with the real typer command; an unknown option or missing value fails."""
    group = typer.main.get_command(cli.app)
    sub = group.get_command(None, args[0])
    assert sub is not None, f"no such command: {args[0]}"
    return sub.make_context(args[0], list(args[1:]))


def test_demo_steps_w06d29_shape():
    steps = record.demo_steps(6, 29)
    assert [s.module for s in steps] == ["week_06.cli"] * 5
    assert steps[0].args == ["status", "--vram"]
    assert steps[1].args == ["profiles", "baseline", "tuned"]
    assert steps[2].args[:2] == ["rag", "--compare"]
    assert steps[3].args[:2] == ["rag", "--compare"]
    assert steps[4].args[0] == "rag"


def test_screening_table_lists_baseline_first_and_all_ten_levers():
    args = record.demo_steps(6, 29)[2].args
    files = args[2:]
    assert len(files) == 11
    assert files[0] == "logs/ragbench/w06d29_baseline_10x1.json"
    names = [f.removeprefix("logs/ragbench/w06d29_").removesuffix("_10x1.json") for f in files]
    assert names == [
        "baseline",
        "sampling",
        "cap",
        "noreason",
        "positional",
        "k12",
        "citelocal",
        "ctx24k",
        "q4b",
        "q3",
        "q5",
    ]
    assert {*names, "tuned"} == set(profiles.PROFILES)


def test_final_table_compares_baseline_with_tuned_13x3():
    args = record.demo_steps(6, 29)[3].args
    assert args[2:] == [
        "logs/ragbench/w06d29_baseline_13x3.json",
        "logs/ragbench/w06d29_tuned_13x3.json",
    ]


def test_the_live_step_names_the_profile_and_the_local_backend_together():
    live = record.demo_steps(6, 29)[4]
    args = live.args
    assert args[args.index("--profile") + 1] == "tuned"
    assert args[args.index("--backends") + 1] == "local"
    assert args[args.index("--runs") + 1] == "1" and "--no-unanswerable" in args
    assert live.timeout is not None and live.timeout >= 900


def _every_step_args():
    for builder in (record.demo_steps, record.rehearsal_steps):
        for step in builder(6, 29):
            if step.module == "week_06.cli":
                yield step.title, step.args


@pytest.mark.parametrize(("title", "args"), list(_every_step_args()))
def test_each_step_line_is_accepted_by_the_cli_parser(title, args):
    ctx = _parse(args)
    if args[0] == "rag" and "--profile" in args:
        assert ctx.params["profile"] == "tuned" and ctx.params["backends"] == "local"
    if args[0] == "rag" and "--compare" in args:
        assert ctx.params["compare"] is True and len(ctx.params["files"]) >= 2


def test_every_profile_step_pairs_a_non_baseline_profile_with_backends_local():
    for _title, args in _every_step_args():
        if "--profile" in args:
            assert "--backends" in args and args[args.index("--backends") + 1] == "local"


def test_rehearsal_steps_are_gates_plus_the_two_offline_tables():
    steps = record.rehearsal_steps(6, 29)
    assert [s.args[0] for s in steps] == ["status", "check", "profiles", "rag", "rag"]
    assert steps[0].args == ["status", "--vram"]
    assert [s.args[:2] for s in steps[3:]] == [["rag", "--compare"]] * 2
    assert steps[1].module == "week_05.cli"


def test_day_28_scenarios_are_unchanged():
    assert [s.args for s in record.rehearsal_steps(6, 28)] == [["status"], ["check"]]
    assert record.demo_steps(6, 28)[-1].args[0] == "rag"


def test_eleven_profile_screening_table_fits_80_columns_without_ellipsis(tmp_path):
    """The screening table of the demo: 11 rows, every table narrower than 80 cells."""
    import dataclasses
    import json

    from tests.test_w06d29_compare import QUESTIONS, UNANSWERABLE, _cells, _write

    paths = []
    for name in profiles.PROFILES:
        cells = [dataclasses.replace(c, wall_ms=9_000) for c in _cells()]
        paths.append(_write(tmp_path, f"{name}.json", cells, profile=name))
    assert QUESTIONS and UNANSWERABLE and json
    files = compare.load_files(paths)
    buf = io.StringIO()
    console = Console(file=buf, width=80, no_color=True, force_terminal=False)
    for table in compare.build_tables(files):
        console.print(table)
    text = buf.getvalue()
    assert "…" not in text
    assert max(cell_len(line) for line in text.splitlines()) <= 80
    rows = [line for line in text.splitlines() if line.lstrip("│ ").startswith("positional")]
    assert text.count("Качество") == 1 and len(rows) == 6  # one row in each of the six tables
