"""Day 28 --report: re-render and merge --save files without network, guard or LM Studio."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console
from typer.testing import CliRunner

from advent_core import console, offline
from advent_core.errors import AdventError
from week_06 import cli
from week_06 import ragbench as rb

FIX = Path(__file__).parent / "fixtures" / "w06d28"
LOCAL = FIX / "smoke_local.json"
CLOUD = FIX / "smoke.json"


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


def _variant(src: Path, dst: Path, **changes) -> Path:
    data = json.loads(src.read_text(encoding="utf-8"))
    data.update(changes)
    dst.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return dst


def test_two_current_format_files_render_a_two_backend_comparison(out):
    code = rb.run_report_command([LOCAL, CLOUD])
    text = out.getvalue()
    assert code == 0
    assert "Качество" in text and "Скорость" in text and "Стабильность" in text
    header = next(line for line in text.splitlines() if "метрика" in line)
    assert "cloud" in header and "local" in header
    assert "скорость: local медленнее в 1.6×" in text
    assert "качество: ничья (local 1/1, cloud 1/1)" in text
    assert "локальный прогон: модельных вызовов локальных 4, облачных 0" in text
    assert "1 прогон × 1 вопрос" in text
    # the local failure noted in the cloud-only file is covered by the local file
    assert "не отвечает" not in text and "пропущено" not in text
    assert "…" not in text
    assert max(cell_len(line) for line in text.splitlines()) <= 80


def test_old_format_degrades_gracefully_and_says_so(out):
    rb.run_report_command([LOCAL, CLOUD])
    text = out.getvalue()
    embed = next(line for line in text.splitlines() if "embed, медиана" in line)
    search = next(line for line in text.splitlines() if "search, медиана" in line)
    assert "0.8 s" in embed and "0.5 s" in embed  # from the ledger: 796 and 546 ms
    assert "—" in search
    assert "search: нет в файле старого формата" in text


def test_new_fields_are_read_and_the_legacy_note_disappears(tmp_path, out):
    data = json.loads(LOCAL.read_text(encoding="utf-8"))
    cell = data["results"][0]
    cell.pop("embed_search_ms")
    cell.update(embed_ms=546, search_ms=11)
    path = tmp_path / "new.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    rb.run_report_command([path])
    text = out.getvalue()
    assert "11" not in next(line for line in text.splitlines() if "embed, медиана" in line)
    assert "0.0 s" in next(line for line in text.splitlines() if "search, медиана" in line)
    assert "нет в файле старого формата" not in text


def test_different_corpus_rev_is_refused_and_names_both(tmp_path):
    other = _variant(CLOUD, tmp_path / "c.json", corpus_rev="deadbeefdeadbeef")
    with pytest.raises(AdventError) as info:
        rb.merge_result_files([LOCAL, other])
    assert "corpus_rev" in info.value.message
    assert "smoke_local.json: 6ee4ac4155ce" in info.value.message
    assert "c.json: deadbeefdead" in info.value.message


def test_different_question_sets_are_refused_and_say_which(tmp_path):
    other = _variant(CLOUD, tmp_path / "c.json", questions=[10, 11])
    with pytest.raises(AdventError) as info:
        rb.merge_result_files([LOCAL, other])
    assert "Наборы вопросов" in info.value.message
    assert "c.json: вопросы [10, 11]" in info.value.message


def test_the_same_backend_twice_is_refused(tmp_path):
    twin = tmp_path / "again.json"
    shutil.copy(LOCAL, twin)
    with pytest.raises(AdventError) as info:
        rb.merge_result_files([LOCAL, twin])
    assert "Backend local есть и в smoke_local.json, и в again.json" in info.value.message


def test_not_a_save_file_and_broken_json_are_adventerrors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(AdventError):
        rb.merge_result_files([bad])
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(AdventError):
        rb.merge_result_files([bad])


def test_a_failed_backend_nobody_supplies_is_still_reported(out):
    rb.run_report_command([CLOUD])
    text = out.getvalue()
    assert "пропущено: local" in text and "не отвечает" in text
    assert "качество:" not in text  # one backend prints no comparison


def test_report_touches_no_guard_no_env_and_no_network(monkeypatch, out):
    def boom(*a, **k):
        raise AssertionError("must not be called in --report mode")

    monkeypatch.setattr(offline, "enable", boom)
    monkeypatch.setattr(rb.config_module, "load_env", boom)
    monkeypatch.setattr(rb.oc, "ensure_ready", boom)
    monkeypatch.setattr(rb, "make_local_backend", boom)
    monkeypatch.setattr(rb, "make_cloud_backend", boom)
    assert rb.run_report_command([LOCAL, CLOUD]) == 0


def test_cli_report_option_is_repeatable_and_renders_to_stdout():
    result = CliRunner().invoke(
        cli.app, ["rag", "--report", str(LOCAL), "--report", str(CLOUD)], terminal_width=100
    )
    assert result.exit_code == 0, result.output
    assert "Скорость" in result.stdout and "Стабильность" in result.stdout


def test_cli_report_with_save_is_refused(tmp_path):
    result = CliRunner().invoke(
        cli.app, ["rag", "--report", str(LOCAL), "--save", str(tmp_path / "x.json")]
    )
    assert result.exit_code != 0


def test_what_save_writes_now_loads_back_with_split_stage_times(tmp_path, out):
    cell = rb.BenchRun(
        "local",
        "answerable",
        1,
        1,
        "answered",
        correct=True,
        facts=(True,),
        facts_total=1,
        wall_ms=1000,
        embed_ms=800,
        search_ms=12,
        ledger=(rb.CallRecord("embed", "e", "local", 4, None, 800, 0.0),),
    )
    result = rb.BenchResult(runs=[cell], backends=["local"], n_runs=1, question_count=1)
    path = tmp_path / "now.json"
    rb.save_json(result, path)
    merged = rb.merge_result_files([path])
    assert (merged.runs[0].embed_ms, merged.runs[0].search_ms) == (800, 12)
    assert merged.runs[0].embed_search_ms is None
