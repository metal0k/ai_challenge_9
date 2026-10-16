"""Stage `sheet`: one idempotent, human-approved, read-back-verified cell comment."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path

from advent_core.config import redact
from adventship.ctx import EXIT_CONFIG, ShipContext, ShipError, StageResult
from adventship.gates import confirm
from adventship.sheet_driver import CELL_RE, ProbeResult, SheetDriver

PROBE_MAX_AGE = timedelta(hours=24)
DESCRIPTION_MAX = 200
_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/(?:home|Users)/")


def normalize(text: str) -> str:
    return " ".join(text.split())


def _digest(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]


def _description(ctx: ShipContext) -> str:
    data = ctx.state.data("describe")
    raw = next(
        (data[k] for k in ("comment_line", "line", "description", "task", "text") if data.get(k)),
        "",
    )
    if not raw and ctx.task is not None:
        raw = ctx.task.title
    line = normalize(str(raw))
    if len(line) > DESCRIPTION_MAX:
        line = line[: DESCRIPTION_MAX - 3].rstrip() + "..."
    return line


def _guard(ctx: ShipContext, text: str) -> None:
    """Nothing private goes into a shared sheet."""
    sheet_url = ctx.env.get("COURSE_SHEET_URL", "")
    if redact(text) != text or (sheet_url and sheet_url in text) or _LOCAL_PATH.search(text):
        raise ShipError("в тексте комментария личные данные (путь, токен или ссылка таблицы)")


def build_comment(ctx: ShipContext) -> str:
    override = ctx.env.get("SHIP_COMMENT_TEXT_FILE")
    if override:
        text = Path(override).read_text(encoding="utf-8").strip()
    else:
        submit = ctx.state.data("submit")
        code_url, video_url = submit.get("code_url"), submit.get("video_url")
        if not code_url or not video_url:
            raise ShipError(
                "нет ссылок из стадии submit", exit_code=EXIT_CONFIG, hint="Сначала стадия submit."
            )
        text = (
            f"Week {ctx.week:02d}, Day {ctx.day:02d}\n"
            f"Код: {code_url}\nВидео: {video_url}\nЗадача: {_description(ctx)}"
        )
    _guard(ctx, text)
    return text


# --- probe ---------------------------------------------------------------


def _url_hash(ctx: ShipContext) -> str:
    return hashlib.sha256(ctx.env.get("COURSE_SHEET_URL", "").encode("utf-8")).hexdigest()[:12]


def _fresh_probe(ctx: ShipContext) -> ProbeResult:
    saved = ctx.state.doc.get("sheet_probe")
    manual = f"Добавь комментарий руками, текст в {ctx.workdir.name}/comment.txt"
    if not saved:
        raise ShipError(
            "нет результата sheet-probe",
            exit_code=EXIT_CONFIG,
            hint=f"Запусти: adventship sheet-probe. {manual}",
        )
    try:
        age = datetime.fromisoformat(ctx.now()) - datetime.fromisoformat(saved["at"])
    except (KeyError, ValueError):
        age = PROBE_MAX_AGE + timedelta(seconds=1)
    result = ProbeResult.from_dict(saved.get("result", {}))
    if age > PROBE_MAX_AGE:
        raise ShipError(
            "sheet-probe старше 24 ч", exit_code=EXIT_CONFIG, hint="adventship sheet-probe"
        )
    if saved.get("url_hash") != _url_hash(ctx) or result.slot != (
        ctx.env.get("SHEET_SLOT") or None
    ):
        raise ShipError(
            "sheet-probe сделан для другой таблицы или слота",
            exit_code=EXIT_CONFIG,
            hint="adventship sheet-probe",
        )
    if not result.ok or not result.cell or not CELL_RE.match(result.cell):
        raise ShipError(
            "sheet-probe не нашёл слот или контролы комментария",
            hint=f"{manual}. Потом adventship sheet-probe.",
        )
    return result


def run_probe(ctx: ShipContext) -> StageResult:
    result = ctx.driver_factory().probe()
    ctx.state.doc["sheet_probe"] = {
        "at": ctx.now(),
        "url_hash": _url_hash(ctx),
        "result": result.to_dict(),
    }
    ctx.state.save()
    if not result.ok:
        raise ShipError("sheet-probe: слот не готов к записи", hint=result.summary())
    return StageResult(data={"cell": result.cell}, summary=result.summary())


def run_login(ctx: ShipContext) -> None:
    driver = ctx.driver_factory()
    login = getattr(driver, "login", None)
    if login is None:
        raise ShipError("драйвер не умеет login", exit_code=EXIT_CONFIG)
    login()


# --- stage ---------------------------------------------------------------


def _has(comments: list[str], wanted: str) -> bool:
    return any(normalize(c) == normalize(wanted) for c in comments)


def _same_day(ctx: ShipContext, comments: list[str]) -> list[str]:
    head = f"Week {ctx.week:02d}, Day {ctx.day:02d}"
    return [c for c in comments if head in c or f"/tree/{ctx.tag}" in c]


def _shot(ctx: ShipContext, driver: SheetDriver) -> None:
    shoot = getattr(driver, "screenshot", None)
    if shoot and ctx.env.get("SHIP_DEBUG_SHOTS"):
        shoot(ctx.workdir / "sheet-failed.png")


def run_sheet(ctx: ShipContext) -> StageResult:
    text = build_comment(ctx)
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    (ctx.workdir / "comment.txt").write_text(text + "\n", encoding="utf-8")
    probe = _fresh_probe(ctx)
    cell = probe.cell
    assert cell is not None  # _fresh_probe guarantees it
    driver = ctx.driver_factory()
    base = {"cell": cell, "comment_sha256": _digest(text)}

    pending = ctx.state.doc.get("sheet_pending")
    existing = driver.read_comments(cell)
    if _has(existing, text):
        ctx.state.doc.pop("sheet_pending", None)
        ctx.state.save()
        data = {**base, "skipped": "reuse"}
        if pending:
            data["reconciled"] = True
        return StageResult(data=data, summary=f"комментарий уже есть в {cell}, пропуск")
    if _same_day(ctx, existing):
        raise ShipError(
            f"в {cell} уже есть комментарий этого дня с другим текстом",
            hint="Ничего не записано. Реши руками; ожидаемый текст в comment.txt.",
        )

    confirm(ctx, "Отправить комментарий в таблицу?", f"Ячейка: {cell}\n\n{text}")

    ctx.state.doc["sheet_pending"] = {"cell": cell, "comment_sha256": base["comment_sha256"]}
    ctx.state.save()
    driver.add_comment(cell, text)
    if not _has(driver.read_comments(cell), text):
        _shot(ctx, driver)
        raise ShipError(
            f"комментарий не найден в {cell} после отправки",
            hint="Проверь таблицу глазами: повторный run сверит и не задвоит.",
        )
    ctx.state.doc.pop("sheet_pending", None)
    ctx.state.save()
    return StageResult(data=base, summary=f"комментарий в {cell} добавлен и прочитан назад")


def recheck_sheet(ctx: ShipContext) -> bool:
    """Done stays done only while the comment is still in the cell."""
    cell = ctx.state.data("sheet").get("cell")
    if not cell:
        return False
    try:
        return _has(ctx.driver_factory().read_comments(cell), build_comment(ctx))
    except ShipError:
        return False
