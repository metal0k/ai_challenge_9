"""Recording helper for Day 34: carries the run id between steps and guards expected failures.

`advent record` runs each step as a separate process, so the run id printed by the
README scenario has to travel through a file. The second `files-apply` is expected
to fail with exit 12; here that is an outcome, not a take-killing error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from advent_core import console
from advent_core.config import PROJECT_ROOT
from advent_core.errors import AdventError
from adventassistant import files_store

README_PATH = "week_07/README.md"
RUN_ID_FILE = PROJECT_ROOT / "logs" / "demo" / "w07d34" / "run_id.txt"
GOAL_SEARCH = "Найди все места в коде, где вызывается `call_tool_once`, и объясни, зачем каждое"
GOAL_README = "Напиши README недели 7: какие команды есть у adventassistant и что делает каждая"
ALREADY_APPLIED_CODE = 12
RUN_TIMEOUT = 600
_RUN_LINE = re.compile(r"^run: (\d{8}-\d{6}-[0-9a-f]{8})\s*$", re.MULTILINE)


def check_readme_absent(root: Path | None = None) -> None:
    """Refuse a take over a rehearsal leftover; the file is never deleted here."""
    if (Path(root) if root is not None else PROJECT_ROOT).joinpath(README_PATH).exists():
        raise AdventError(
            f"{README_PATH} уже существует (остаток репетиции).",
            hint=f"Удали {README_PATH} вручную перед записью: демо создаёт его через files-apply.",
        )


def extract_run_id(stdout: str) -> str | None:
    found = _RUN_LINE.findall(stdout)
    return found[-1] if found else None


def _cli(args: list[str], *, capture_stdout: bool, capture_stderr: bool):
    return subprocess.run(
        [sys.executable, "-m", "adventassistant.cli", *args],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE if capture_stderr else None,
        text=True,
        encoding="utf-8",
        timeout=RUN_TIMEOUT,
        check=False,
    )


def run_readme(goal: str = GOAL_README) -> str:
    """Run the scenario, show its stdout, remember the run id; no diff means no run to apply."""
    check_readme_absent()
    proc = _cli(["files", goal], capture_stdout=True, capture_stderr=False)
    sys.stdout.write(proc.stdout or "")
    sys.stdout.flush()
    if proc.returncode != 0:
        raise AdventError(f"files завершился с кодом {proc.returncode}.")
    run_id = extract_run_id(proc.stdout or "")
    if run_id is None:
        raise AdventError("Сценарий не дал diff: строки `run: <id>` нет, применять нечего.")
    RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(run_id, encoding="utf-8")
    return run_id


def _saved_run_id() -> str:
    try:
        run_id = RUN_ID_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        run_id = ""
    if not run_id:
        raise AdventError("run_id предыдущего шага не найден: сначала шаг с README.")
    return files_store.validate_run_id(run_id)


def apply_saved() -> None:
    run_id = _saved_run_id()
    proc = _cli(["files-apply", run_id], capture_stdout=False, capture_stderr=False)
    if proc.returncode != 0:
        raise AdventError(f"files-apply завершился с кодом {proc.returncode}.")
    status = files_store.git_status_short([README_PATH], PROJECT_ROOT)
    console.out.print(f"$ git status --short {README_PATH}", markup=False, highlight=False)
    console.out.print(status or "(пусто)", markup=False, highlight=False)


def apply_again() -> None:
    """The repeat must be refused with exit 12 'уже применён'; anything else is a failure."""
    run_id = _saved_run_id()
    proc = _cli(["files-apply", run_id], capture_stdout=False, capture_stderr=True)
    sys.stderr.write(proc.stderr or "")
    sys.stderr.flush()
    if proc.returncode != ALREADY_APPLIED_CODE or "уже применён" not in (proc.stderr or ""):
        raise AdventError(
            f"Повторный files-apply: ждали код {ALREADY_APPLIED_CODE} «уже применён», "
            f"получили {proc.returncode}."
        )
    console.out.print(
        f"Ожидаемый отказ: exit {ALREADY_APPLIED_CODE}, прогон уже применён — "
        "повторная запись защищена.",
        markup=False,
        highlight=False,
    )


def main(argv: list[str] | None = None) -> None:
    console.force_utf8()
    parser = argparse.ArgumentParser(description="Проверки и шаги демо Day 34.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", metavar="GOAL")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--apply-again", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            check_readme_absent()
            console.out.print(f"{README_PATH} отсутствует: можно записывать.", markup=False)
        elif args.run is not None:
            run_readme(args.run)
        elif args.apply:
            apply_saved()
        else:
            apply_again()
    except AdventError as exc:
        console.fail(exc)
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    main()
