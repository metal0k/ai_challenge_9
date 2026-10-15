"""Day 34: run store, tool operations over the overlay, diff and files-apply."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from advent_core.errors import AdventError
from adventassistant import files_scope as scope
from adventassistant import files_store as store
from adventassistant.files_store import (
    ApplyConflictError,
    ApplyRolledBackError,
    FilesError,
    Workspace,
)
from tests.w07d34_repo import git, make_dir_link, make_repo, write


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


@pytest.fixture
def ws(repo: Path) -> Workspace:
    run_id, _ = store.create_run("цель", "model-x", repo)
    return Workspace(run_id, repo)


def _disk(repo: Path, rel: str) -> bytes:
    return (repo / rel).read_bytes()


# --- run ids and directories ---


def test_run_id_format_and_validation() -> None:
    run_id = store.new_run_id(datetime(2026, 10, 15, 12, 30, 5))
    assert store.RUN_ID_RE.fullmatch(run_id)
    assert run_id.startswith("20261015-123005-") and len(run_id) == len("20261015-123005-") + 8
    assert store.new_run_id() != store.new_run_id()
    assert store.validate_run_id(run_id) == run_id
    for bad in ("", "x", "20261015-123005-ABCDEF12", "20261015-123005-abc", "../x", None, 5):
        with pytest.raises(FilesError):
            store.validate_run_id(bad)


def test_create_run_writes_manifest_and_refuses_repeat(repo: Path, monkeypatch) -> None:
    run_id, path = store.create_run("найти вызовы", "ministral-14b-latest", repo)
    assert path == repo / "logs" / "files" / run_id
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "version": 1,
        "run_id": run_id,
        "goal": "найти вызовы",
        "model": "ministral-14b-latest",
        "files": {},
    }
    assert (path / "changes.diff").read_text(encoding="utf-8") == ""
    monkeypatch.setattr(store, "new_run_id", lambda now=None: run_id)
    with pytest.raises(FilesError, match="уже существует"):
        store.create_run("x", "m", repo)


def test_run_dir_rejects_missing_plain_file_and_linked_dir(repo: Path, tmp_path: Path) -> None:
    missing = "20261015-123005-deadbeef"
    with pytest.raises(FilesError, match="не найден"):
        store.run_dir_path(missing, repo)
    runs = repo / "logs" / "files"
    runs.mkdir(parents=True)
    (runs / missing).write_text("not a dir", encoding="utf-8")
    with pytest.raises(FilesError, match="обычным каталогом"):
        store.run_dir_path(missing, repo)
    (runs / missing).unlink()

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if not make_dir_link(runs / missing, elsewhere):
        pytest.skip("OS refuses to create symlinks/junctions")
    with pytest.raises(FilesError, match="обычным каталогом"):
        store.run_dir_path(missing, repo)
    with pytest.raises(FilesError):
        Workspace(missing, repo)


def test_logs_files_replaced_by_a_link_is_refused(repo: Path, tmp_path: Path) -> None:
    (repo / "logs").mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if not make_dir_link(repo / "logs" / "files", elsewhere):
        pytest.skip("OS refuses to create symlinks/junctions")
    with pytest.raises(FilesError, match="ссылкой"):
        store.create_run("g", "m", repo)


# --- list ---


def test_list_files_policy_and_sorting(ws: Workspace) -> None:
    out = ws.list_files()
    lines = out.splitlines()
    assert lines[-1].startswith("всего: ")
    listed = set(lines[:-1])
    assert {"docs/a.md", "src/app.py", "README.md"} <= listed
    for hidden in (".env", "certs/server.pem", "uv.lock", "linked.md"):
        assert hidden not in listed
    assert lines[:-1] == sorted(lines[:-1], key=str.casefold)
    assert ws.list_files("DOCS/*").splitlines() == ["docs/a.md", "docs/b.txt", "всего: 2"]
    assert ws.list_files("*.nomatch") == "всего: 0"
    with pytest.raises(FilesError):
        ws.list_files("")


def test_list_files_limit_200_reports_the_total(tmp_path: Path) -> None:
    root = make_repo(tmp_path / "many")
    for i in range(230):
        write(root, f"bulk/f{i:03}.txt", "x\n")
    git(root, "add", "-A")
    run_id, _ = store.create_run("g", "m", root)
    out = Workspace(run_id, root).list_files("bulk/*")
    lines = out.splitlines()
    assert len(lines) == 201
    assert lines[-1] == "всего: 230, показано 200"


# --- search ---


def test_search_fixed_case_insensitive_with_counts(ws: Workspace) -> None:
    out = ws.search_text("CALL_TOOL_ONCE")
    assert "src/app.py:1: def call_tool_once():" in out
    assert "src/other.py:2: value = call_tool_once()" in out
    assert out.splitlines()[-1] == "совпадений: 4 в файлах: 2"


def test_search_regex_glob_and_errors(ws: Workspace) -> None:
    out = ws.search_text(r"^def \w+", glob="src/*.py", regex=True)
    assert out.splitlines() == ["src/app.py:1: def call_tool_once():", "совпадений: 1 в файлах: 1"]
    with pytest.raises(FilesError, match="Невалидный regex"):
        ws.search_text("(", regex=True)
    with pytest.raises(FilesError, match="200"):
        ws.search_text("a" * 201, regex=True)
    with pytest.raises(FilesError, match="Пустой"):
        ws.search_text("")
    # a fixed string with regex metacharacters is literal
    assert ws.search_text("call_tool_once(").endswith("совпадений: 3 в файлах: 2")


def test_search_never_reads_denied_binary_or_symlink_files(ws: Workspace) -> None:
    for query in ("supersecretvalue", "BEGIN", "abc", "broken"):
        assert ws.search_text(query).splitlines() == ["совпадений: 0 в файлах: 0"]


def test_search_caps_hits_and_line_length(tmp_path: Path) -> None:
    root = make_repo(tmp_path / "hits")
    write(root, "docs/hits.md", "needle " + "y" * 300 + "\n" + "needle\n" * 149)
    git(root, "add", "-A")
    run_id, _ = store.create_run("g", "m", root)
    out = Workspace(run_id, root).search_text("needle")
    lines = out.splitlines()
    assert len(lines) == 101
    assert lines[-1] == "совпадений: 150 в файлах: 1, показано 100"
    first = lines[0].split(": ", 1)[1]
    assert len(first) == store.SEARCH_LINE_CHARS and first.startswith("needle yyy")


def test_search_scan_cap_by_files_and_bytes(tmp_path: Path, monkeypatch) -> None:
    root = make_repo(tmp_path / "scan")
    for i in range(410):
        write(root, f"bulk/f{i:03}.txt", "nothing\n")
    git(root, "add", "-A")
    run_id, _ = store.create_run("g", "m", root)
    out = Workspace(run_id, root).search_text("zzz-not-found")
    assert "скан обрезан" in out and "просканировано файлов: 400" in out
    small = Workspace(run_id, root).search_text("zzz-not-found", glob="docs/*")
    assert small == "совпадений: 0 в файлах: 0"

    monkeypatch.setattr(store, "SEARCH_BYTES", 40)
    out = Workspace(run_id, root).search_text("zzz-not-found", glob="bulk/*")
    assert "скан обрезан" in out


def test_search_overlay_sees_staged_files(ws: Workspace) -> None:
    ws.write_file("docs/new.md", "unique-marker-line\n")
    assert "docs/new.md:1: unique-marker-line" in ws.search_text("unique-marker")


# --- read ---


def test_read_numbers_lines_and_reports_total(ws: Workspace) -> None:
    assert (
        ws.read_file("docs/a.md") == "docs/a.md: строки 1-3 из 3\n1: alpha\n2: Beta line\n3: gamma"
    )
    assert ws.read_file("docs/a.md", 2, 2) == "docs/a.md: строки 2-2 из 3\n2: Beta line"
    assert ws.read_file("DOCS\\A.MD", 3).splitlines()[0] == "docs/a.md: строки 3-3 из 3"
    with pytest.raises(FilesError, match="за пределами"):
        ws.read_file("docs/a.md", 4)
    with pytest.raises(FilesError):
        ws.read_file("docs/a.md", 0)
    with pytest.raises(FilesError):
        ws.read_file("docs/a.md", 3, 2)


def test_read_caps_at_400_lines_and_40000_chars(tmp_path: Path) -> None:
    root = make_repo(tmp_path / "caps")
    write(root, "docs/long.md", "".join(f"line {i}\n" for i in range(1, 1001)))
    write(root, "docs/wide.md", "".join("w" * 900 + "\n" for _ in range(100)))
    git(root, "add", "-A")
    run_id, _ = store.create_run("g", "m", root)
    ws = Workspace(run_id, root)
    out = ws.read_file("docs/long.md")
    lines = out.splitlines()
    assert lines[0].startswith("docs/long.md: строки 1-400 из 1000")
    assert "продолжай со строки 401" in lines[0]
    assert len(lines) == 401
    wide = ws.read_file("docs/wide.md")
    assert len(wide) <= 40_000 + 200
    assert "из 100" in wide.splitlines()[0] and "продолжай со строки" in wide.splitlines()[0]


@pytest.mark.parametrize(
    "path,needle",
    [
        ("nope.md", "не найден"),
        (".env", "не найден|запрещ"),
        ("certs/server.pem", "запрещ"),
        ("uv.lock", "запрещ"),
        ("blob.bin", "Бинарный"),
        ("latin.md", "UTF-8"),
        ("big.txt", "1 MiB"),
        ("linked.md", "не найден|ссылк"),
        ("../outside.md", "'..'"),
        ("C:/Windows/win.ini", "Двоеточие"),
        (".GIT/config", "запрещ|не найден|'.git'|Компонент"),
        ("con.md", "Зарезервированное"),
    ],
)
def test_read_refusals(ws: Workspace, path: str, needle: str) -> None:
    with pytest.raises(AdventError, match=needle):
        ws.read_file(path)


def test_read_untracked_file_on_disk_is_refused(ws: Workspace, repo: Path) -> None:
    write(repo, "docs/untracked.md", "private\n")
    with pytest.raises(FilesError, match="не tracked"):
        ws.read_file("docs/untracked.md")
    assert "untracked" not in ws.list_files()


def test_read_through_symlinked_parent_is_refused(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.md").write_text("leak", encoding="utf-8")
    run_id, _ = store.create_run("g", "m", repo)
    # track a path under a directory that is then replaced by a link
    write(repo, "swap/x.md", "orig\n")
    git(repo, "add", "swap/x.md")
    ws = Workspace(run_id, repo)
    assert ws.read_file("swap/x.md").endswith("1: orig")
    (repo / "swap" / "x.md").unlink()
    (repo / "swap").rmdir()
    if not make_dir_link(repo / "swap", outside):
        pytest.skip("OS refuses to create symlinks/junctions")
    with pytest.raises(AdventError, match="ссылк"):
        ws.read_file("swap/x.md")


def test_tool_output_is_redacted(ws: Workspace, repo: Path, monkeypatch) -> None:
    write(repo, "docs/leak.md", "token = supersecretvalue\n")
    git(repo, "add", "docs/leak.md")
    monkeypatch.setenv("MISTRAL_API_KEY", "supersecretvalue")
    fresh = Workspace(ws.run_id, repo)
    assert "supersecretvalue" not in fresh.read_file("docs/leak.md")
    assert "supersecretvalue" not in fresh.search_text("token")
    assert "token = ***" in fresh.read_file("docs/leak.md")


# --- edit / write / overlay ---


def test_edit_requires_exactly_one_occurrence(ws: Workspace, repo: Path) -> None:
    with pytest.raises(FilesError, match="найдено вхождений: 0"):
        ws.edit_file("docs/a.md", "absent", "x")
    with pytest.raises(FilesError, match="найдено вхождений: 2"):
        ws.edit_file("src/app.py", "call_tool_once", "x")
    with pytest.raises(FilesError, match="пустым"):
        ws.edit_file("docs/a.md", "", "x")
    # .py is not writable at all, even with a single occurrence
    with pytest.raises(AdventError, match="только файлы .md и .txt"):
        ws.edit_file("src/other.py", "value", "v")
    assert ws.edit_file("docs/a.md", "Beta", "BETA") == "staged docs/a.md (+1 -1)"
    assert _disk(repo, "docs/a.md") == b"alpha\nBeta line\ngamma\n"
    assert ws.read_file("docs/a.md").splitlines()[2] == "2: BETA line"
    # edit of the staged copy, not of the disk one
    assert ws.edit_file("docs/a.md", "BETA line", "beta") == "staged docs/a.md (+1 -1)"
    assert "2: beta" in ws.read_file("docs/a.md")


def test_edit_keeps_crlf_files_consistent(ws: Workspace) -> None:
    ws.edit_file("crlf.md", "one\ntwo", "uno\ndos")
    manifest = ws.manifest()
    staged = (ws.run_dir / "files" / "crlf.md").read_bytes()
    assert staged == b"uno\r\ndos\r\n"
    assert manifest["files"]["crlf.md"]["base"] == store.sha256_hex(b"one\r\ntwo\r\n")


@pytest.mark.parametrize(
    "path",
    [
        "src/app.py",
        ".github/new.md",
        "specs/new.md",
        "README.md",
        "CLAUDE.md",
        "docs/CLAUDE.md",
        "logs/x.txt",
        "data/new.md",
        "docs/.env.md",
        "docs/a.md:ads",
        "docs/../README2.md",
        "con.md",
    ],
)
def test_write_policy_refusals(ws: Workspace, path: str) -> None:
    with pytest.raises(AdventError):
        ws.write_file(path, "x\n")
    assert ws.manifest()["files"] == {}


def test_write_new_file_rules(ws: Workspace, repo: Path) -> None:
    assert ws.write_file("docs/new.md", "a\nb\n") == "staged docs/new.md (+2)"
    with pytest.raises(FilesError, match="каталоге с tracked"):
        ws.write_file("brand_new_dir/x.md", "x\n")
    write(repo, "docs/untracked.md", "private\n")
    with pytest.raises(FilesError, match="не отслеживается"):
        ws.write_file("docs/untracked.md", "x\n")
    with pytest.raises(AdventError, match="не в UTF-8"):
        ws.write_file("latin.md", "x\n")
    assert not (repo / "docs" / "new.md").exists()


def test_write_size_and_file_count_limits(ws: Workspace) -> None:
    with pytest.raises(AdventError, match="64 KiB"):
        ws.write_file("docs/huge.md", "x" * (64 * 1024 + 1))
    ws.write_file("docs/ok64.md", "x" * (64 * 1024))
    for i in range(4):
        ws.write_file(f"docs/f{i}.md", "x\n")
    with pytest.raises(FilesError, match="не более 5"):
        ws.write_file("docs/f9.md", "x\n")
    # rewriting an already staged file does not count again
    assert ws.write_file("DOCS/F0.md", "y\n").startswith("staged docs/f0.md")
    assert len(ws.manifest()["files"]) == 5


def test_overlay_reads_search_and_list_but_disk_is_untouched(ws: Workspace, repo: Path) -> None:
    before = git(repo, "status", "--short")
    ws.write_file("docs/a.md", "replaced\n")
    ws.write_file("docs/new.md", "fresh\n")
    assert ws.read_file("docs/a.md").endswith("1: replaced")
    assert ws.read_file("docs/new.md").endswith("1: fresh")
    assert "docs/new.md" in ws.list_files("docs/*").splitlines()
    assert _disk(repo, "docs/a.md") == b"alpha\nBeta line\ngamma\n"
    assert not (repo / "docs" / "new.md").exists()
    assert git(repo, "status", "--short") == before


def test_manifest_records_bases_and_new_hashes(ws: Workspace) -> None:
    ws.write_file("docs/a.md", "replaced\n")
    ws.write_file("docs/new.md", "fresh\n")
    files = ws.manifest()["files"]
    assert files["docs/a.md"] == {
        "base": store.sha256_hex(b"alpha\nBeta line\ngamma\n"),
        "sha256": store.sha256_hex(b"replaced\n"),
    }
    assert files["docs/new.md"] == {"base": None, "sha256": store.sha256_hex(b"fresh\n")}
    # a second write of the same file keeps the original base
    ws.write_file("docs/a.md", "again\n")
    assert ws.manifest()["files"]["docs/a.md"]["base"] == files["docs/a.md"]["base"]


# --- diff ---


def test_diff_for_new_and_changed_files(ws: Workspace) -> None:
    ws.write_file("docs/new.md", "one\ntwo\n")
    ws.edit_file("docs/a.md", "Beta line", "Beta changed")
    diff = store.read_diff(ws.run_id, ws.root)
    assert diff == ws.diff()
    head = "--- a/docs/a.md\n+++ b/docs/a.md\n@@ -1,3 +1,3 @@\n"
    assert head + " alpha\n-Beta line\n+Beta changed\n gamma\n" in diff
    assert "--- /dev/null\n+++ b/docs/new.md\n@@ -0,0 +1,2 @@\n+one\n+two\n" in diff
    assert diff.index("docs/a.md") < diff.index("docs/new.md")


def test_diff_unit_cases() -> None:
    assert store.unified_diff("x.md", "same\n", "same\n") == ""
    assert store.unified_diff("x.md", None, "") == "--- /dev/null\n+++ b/x.md\n"
    assert "\\ No newline at end of file" in store.unified_diff("x.md", "a\n", "a\nb")


# --- apply ---


def _two_file_run(ws: Workspace) -> None:
    ws.write_file("docs/0new.md", "brand new\n")
    ws.edit_file("docs/a.md", "gamma", "GAMMA")


def test_apply_writes_exactly_the_saved_changeset(ws: Workspace, repo: Path) -> None:
    _two_file_run(ws)
    result = store.apply_run(ws.run_id, repo)
    assert result.written == ["docs/0new.md", "docs/a.md"]
    assert _disk(repo, "docs/0new.md") == b"brand new\n"
    assert _disk(repo, "docs/a.md") == b"alpha\nBeta line\nGAMMA\n"
    assert (ws.run_dir / "backup" / "docs" / "a.md").read_bytes() == b"alpha\nBeta line\ngamma\n"
    assert not list((repo / "docs").glob(".tmp-*"))
    status = store.git_status_short(result.written, repo)
    assert "?? docs/0new.md" in status and "M docs/a.md" in status


def test_apply_conflict_writes_nothing(ws: Workspace, repo: Path) -> None:
    _two_file_run(ws)
    write(repo, "docs/a.md", "someone else changed this\n")
    with pytest.raises(ApplyConflictError, match="изменился") as info:
        store.apply_run(ws.run_id, repo)
    assert info.value.exit_code == 12 and not info.value.already_applied
    assert not (repo / "docs" / "0new.md").exists()
    assert _disk(repo, "docs/a.md") == b"someone else changed this\n"


def test_apply_refuses_to_overwrite_a_new_file_created_meanwhile(ws: Workspace, repo: Path) -> None:
    _two_file_run(ws)
    write(repo, "docs/0new.md", "appeared\n")
    with pytest.raises(ApplyConflictError, match="уже существует"):
        store.apply_run(ws.run_id, repo)
    assert _disk(repo, "docs/a.md") == b"alpha\nBeta line\ngamma\n"


def test_reapply_reports_already_applied(ws: Workspace, repo: Path) -> None:
    _two_file_run(ws)
    store.apply_run(ws.run_id, repo)
    with pytest.raises(ApplyConflictError, match="уже применён") as info:
        store.apply_run(ws.run_id, repo)
    assert info.value.exit_code == 12 and info.value.already_applied


def test_failure_on_second_file_rolls_back_the_first(ws: Workspace, repo: Path) -> None:
    _two_file_run(ws)
    calls: list[int] = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk exploded")
        os.replace(src, dst)

    with pytest.raises(ApplyRolledBackError, match="disk exploded") as info:
        store.apply_run(ws.run_id, repo, replace=flaky)
    assert info.value.exit_code == 13
    assert info.value.rolled_back == ["docs/0new.md"]
    assert not (repo / "docs" / "0new.md").exists()
    assert _disk(repo, "docs/a.md") == b"alpha\nBeta line\ngamma\n"
    assert not list((repo / "docs").glob(".tmp-*"))


def test_failure_after_modifying_an_existing_file_restores_it(ws: Workspace, repo: Path) -> None:
    ws.edit_file("crlf.md", "one", "uno")
    ws.edit_file("docs/a.md", "gamma", "GAMMA")
    calls: list[int] = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("boom")
        os.replace(src, dst)

    with pytest.raises(ApplyRolledBackError) as info:
        store.apply_run(ws.run_id, repo, replace=flaky)
    assert info.value.rolled_back == ["crlf.md"]
    assert _disk(repo, "crlf.md") == b"one\r\ntwo\r\n"


def test_sha_recheck_before_each_replace_catches_a_change_during_apply(
    ws: Workspace, repo: Path
) -> None:
    _two_file_run(ws)
    calls: list[int] = []

    def racing(src, dst):
        calls.append(1)
        os.replace(src, dst)
        if len(calls) == 1:
            write(repo, "docs/a.md", "changed during apply\n")

    with pytest.raises(ApplyRolledBackError, match="изменился") as info:
        store.apply_run(ws.run_id, repo, replace=racing)
    assert info.value.rolled_back == ["docs/0new.md"]
    assert not (repo / "docs" / "0new.md").exists()
    assert _disk(repo, "docs/a.md") == b"changed during apply\n"


def test_apply_validates_manifest_and_files_before_writing(ws: Workspace, repo: Path) -> None:
    ws.write_file("docs/new.md", "fresh\n")
    staged = ws.run_dir / "files" / "docs" / "new.md"
    staged.write_bytes(b"tampered\n")
    with pytest.raises(ApplyConflictError, match="не совпадает с манифестом"):
        store.apply_run(ws.run_id, repo)
    assert not (repo / "docs" / "new.md").exists()

    staged.write_bytes(b"fresh\n")
    (ws.run_dir / "changes.diff").write_text("--- tampered\n", encoding="utf-8")
    with pytest.raises(ApplyConflictError, match="changes.diff"):
        store.apply_run(ws.run_id, repo)


def test_apply_rechecks_write_policy_from_the_manifest(ws: Workspace, repo: Path) -> None:
    ws.write_file("docs/new.md", "fresh\n")
    manifest = json.loads((ws.run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["src/evil.py"] = manifest["files"].pop("docs/new.md")
    (ws.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ApplyConflictError, match="Валидация"):
        store.apply_run(ws.run_id, repo)
    assert not (repo / "src" / "evil.py").exists()


def test_apply_empty_and_bad_manifest_versions(ws: Workspace, repo: Path) -> None:
    with pytest.raises(ApplyConflictError, match="применять нечего"):
        store.apply_run(ws.run_id, repo)
    manifest = json.loads((ws.run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = 99
    (ws.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FilesError, match="версия"):
        store.apply_run(ws.run_id, repo)


def test_error_classes_have_distinct_free_exit_codes() -> None:
    assert ApplyConflictError("x").exit_code == 12
    assert ApplyRolledBackError("x", rolled_back=[]).exit_code == 13
    assert scope.ScopeError("x").exit_code == 1


# --- link swapped between check and write ---


def _fail_from(monkeypatch, nth: int, *, only: bool) -> None:
    """Real check, but ScopeError from the nth call on (only=False) or on the nth alone."""
    real = scope.assert_inside_root
    count = {"n": 0}

    def fake(root, canon):
        count["n"] += 1
        if count["n"] == nth or (count["n"] > nth and not only):
            raise scope.ScopeError(f"Символическая ссылка или reparse point в пути: {canon}")
        return real(root, canon)

    monkeypatch.setattr(scope, "assert_inside_root", fake)


# two-file run: validate = calls 1-2, then per key (pre-check, before mkstemp, before replace):
# key 1 = calls 3-5, key 2 = calls 6-8, rollback of key 1 = call 9


def test_recheck_before_mkstemp_stops_the_second_write(ws: Workspace, repo: Path, monkeypatch):
    _two_file_run(ws)
    _fail_from(monkeypatch, 7, only=True)
    with pytest.raises(ApplyRolledBackError, match="Символическая ссылка") as info:
        store.apply_run(ws.run_id, repo)
    assert info.value.exit_code == 13
    assert info.value.rolled_back == ["docs/0new.md"]
    assert info.value.rollback_failed == []
    assert not (repo / "docs" / "0new.md").exists()
    assert _disk(repo, "docs/a.md") == b"alpha\nBeta line\ngamma\n"
    assert not list((repo / "docs").glob(".tmp-*"))


def test_recheck_before_replace_removes_the_tmp_and_leaves_the_target(
    ws: Workspace, repo: Path, monkeypatch
):
    _two_file_run(ws)
    _fail_from(monkeypatch, 8, only=True)
    with pytest.raises(ApplyRolledBackError, match="Символическая ссылка") as info:
        store.apply_run(ws.run_id, repo)
    assert info.value.rolled_back == ["docs/0new.md"]
    assert _disk(repo, "docs/a.md") == b"alpha\nBeta line\ngamma\n"
    assert not list((repo / "docs").glob(".tmp-*"))


def test_failed_recheck_in_rollback_counts_as_rollback_failed(
    ws: Workspace, repo: Path, monkeypatch
):
    _two_file_run(ws)
    _fail_from(monkeypatch, 7, only=False)
    with pytest.raises(ApplyRolledBackError) as info:
        store.apply_run(ws.run_id, repo)
    assert info.value.rolled_back == []
    assert info.value.rollback_failed == ["docs/0new.md"]
    assert "Откат не удался для: docs/0new.md." in str(info.value)
    # the refused rollback unlinked nothing through the suspect path
    assert (repo / "docs" / "0new.md").exists()


def test_real_junction_swapped_mid_apply_writes_nothing_outside(
    ws: Workspace, repo: Path, tmp_path: Path
):
    _two_file_run(ws)
    outside = tmp_path / "outside"
    outside.mkdir()
    swapped: list[bool] = []

    def swap_after_first(src, dst):
        os.replace(src, dst)
        if not swapped:
            (repo / "docs").rename(repo / "docs_real")
            swapped.append(make_dir_link(repo / "docs", outside))

    with pytest.raises(ApplyRolledBackError) as info:
        store.apply_run(ws.run_id, repo, replace=swap_after_first)
    if not swapped[0]:
        pytest.skip("OS refused to create a directory link")
    assert info.value.exit_code == 13
    assert list(outside.iterdir()) == []
    assert info.value.rollback_failed == ["docs/0new.md"]


def test_displayed_diff_redacts_context_but_apply_writes_exact_bytes(
    repo: Path, monkeypatch
) -> None:
    secret = "ctx-secret-value-not-a-key"
    write(repo, "docs/leak.md", f"head\ntoken = {secret}\ntail\nchange me\n")
    git(repo, "add", "-f", "docs/leak.md")
    monkeypatch.setenv("MISTRAL_API_KEY", secret)
    run_id, _ = store.create_run("цель", "model-x", repo)
    wsx = Workspace(run_id, repo)
    wsx.edit_file("docs/leak.md", "change me", "changed")

    assert secret not in wsx.diff()
    assert "token = ***" in wsx.diff()
    on_disk = (wsx.run_dir / "changes.diff").read_text(encoding="utf-8")
    assert secret not in on_disk
    assert "-change me" in on_disk and "+changed" in on_disk
    assert secret.encode() in (wsx.run_dir / "files" / "docs" / "leak.md").read_bytes()

    store.apply_run(run_id, repo)  # validation compares redacted diffs and still passes
    assert secret.encode() in _disk(repo, "docs/leak.md")
    assert b"changed" in _disk(repo, "docs/leak.md")


def test_tampered_redacted_diff_is_still_caught(repo: Path, monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "ctx-secret-value-not-a-key")
    run_id, _ = store.create_run("цель", "model-x", repo)
    wsx = Workspace(run_id, repo)
    wsx.write_file("docs/new.md", "fresh\n")
    (wsx.run_dir / "changes.diff").write_text("--- tampered\n", encoding="utf-8")
    with pytest.raises(ApplyConflictError, match="changes.diff"):
        store.apply_run(run_id, repo)
