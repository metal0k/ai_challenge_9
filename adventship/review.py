"""Stage `review`: external Codex review of the day's commit range; the verdict is read by code."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from adventship.ctx import (
    EXIT_CONFIG,
    EXIT_REVIEW_FAIL,
    EXIT_STAGE_FAILED,
    ShipContext,
    ShipError,
    StageResult,
)
from adventship.gitutil import day_changes
from adventship.llm import load_prompt, scrub

REVIEW_TIMEOUT = 1500.0
_VERDICT = re.compile(r"^\s*\**VERDICT:\s*(PASS|FAIL)\**\s*$", re.IGNORECASE | re.MULTILINE)
_BLOCKER = re.compile(r"^\s*[-*]?\s*\**BLOCKER\**:", re.IGNORECASE | re.MULTILINE)


def parse_verdict(text: str) -> tuple[str, int]:
    """(`PASS`|`FAIL`, number of BLOCKER lines). The last VERDICT line wins; none is an error."""
    verdicts = _VERDICT.findall(text)
    if not verdicts:
        raise ShipError(
            "в ответе ревью нет строки VERDICT: PASS или VERDICT: FAIL",
            exit_code=EXIT_STAGE_FAILED,
            hint="Смотри logs/ship/<id>/review.md и повтори стадию.",
        )
    return verdicts[-1].upper(), len(_BLOCKER.findall(text))


def build_prompt(ctx: ShipContext, rng: str) -> str:
    task = ctx.task.text.strip() if ctx.task is not None else "(задание не прочитано)"
    return f"{load_prompt('review')}\n{scrub(task, ctx.env)}\n\nДИАПАЗОН: {rng}\n"


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _codex_bin(ctx: ShipContext) -> str:
    return ctx.env.get("CODEX_BIN") or shutil.which("codex") or "codex"


def run_review(ctx: ShipContext) -> StageResult:
    changes = day_changes(ctx)
    if not changes.commits and not changes.dirty_tracked:
        raise ShipError(
            f"в диапазоне {changes.range} нет коммитов и нет правок отслеживаемых файлов",
            exit_code=EXIT_STAGE_FAILED,
            hint="Сначала стадия commit.",
        )
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    out_file = ctx.workdir / "review.md"
    last_message = ctx.workdir / "review.last.md"
    last_message.unlink(missing_ok=True)
    # The prompt goes through stdin (`-`): a long multi-line argument is mangled by the
    # cmd.exe shim that npm installs for `codex` on Windows. Stdin is closed after it, so
    # codex never waits for more input.
    argv = [
        _codex_bin(ctx),
        "exec",
        "-s",
        "read-only",
        "--color",
        "never",
        "-o",
        str(last_message),
        "-",
    ]
    try:
        result = ctx.run(argv, timeout=REVIEW_TIMEOUT, input=build_prompt(ctx, changes.range))
    except FileNotFoundError as error:
        raise ShipError(
            "codex не найден",
            exit_code=EXIT_CONFIG,
            hint="Установи Codex CLI или задай CODEX_BIN.",
        ) from error
    text = (
        last_message.read_text(encoding="utf-8", errors="replace")
        if last_message.exists()
        else (result.stdout or "")
    )
    text = scrub(text, ctx.env)
    out_file.write_text(text, encoding="utf-8")
    relative = _relative(ctx.root, out_file)
    if result.returncode != 0 and not _VERDICT.search(text):
        tail = scrub("\n".join((result.stderr or "").strip().splitlines()[-8:]), ctx.env)
        raise ShipError(
            f"codex завершился с кодом {result.returncode}",
            exit_code=EXIT_STAGE_FAILED,
            hint=tail or None,
        )
    verdict, blockers = parse_verdict(text)
    if verdict == "FAIL" or blockers:
        raise ShipError(
            f"ревью не пройдено: VERDICT {verdict}, замечаний BLOCKER: {blockers}",
            exit_code=EXIT_REVIEW_FAIL,
            hint=f"Полный текст: {relative}",
        )
    data: dict = {
        "verdict": verdict,
        "blockers": blockers,
        "review_file": relative,
        "range": changes.range,
    }
    if changes.dirty_tracked:
        data["warning"] = "незакоммиченные отслеживаемые файлы: " + ", ".join(
            changes.dirty_tracked[:10]
        )
    return StageResult(
        data=data,
        summary=f"VERDICT {verdict}",
    )
