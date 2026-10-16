"""Stage registry and runner: resume, `--from`/`--only`, postcondition re-check, journal."""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable, Mapping

from advent_core import console, journal
from advent_core.config import redact
from adventship.ctx import EXIT_CONFIG, ShipContext, ShipError, StageResult
from adventship.state import STAGES, ShipState

StageFn = Callable[[ShipContext], StageResult]
RecheckFn = Callable[[ShipContext], bool]

# Stage -> (candidate modules, function name). Resolved lazily so a missing module only
# breaks the stages that need it. `recheck_<id>` in the same module is optional.
IMPLEMENTATIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "intake": (("adventship.intake",), "run_intake"),
    "checks": (("adventship.checks",), "run_checks"),
    "describe": (("adventship.llm",), "run_describe"),
    "review": (("adventship.review",), "run_review"),
    "commit": (("adventship.gitops",), "run_commit"),
    "tag": (("adventship.gitops",), "run_tag"),
    "record": (("adventship.submit_stage",), "run_record"),
    "submit": (("adventship.submit_stage",), "run_submit"),
    "sheet": (("adventship.sheet",), "run_sheet"),
    "report": (("adventship.llm",), "run_report"),
}
# Stages that really execute under --dry-run (the rest only print a plan).
LIVE_IN_DRY_RUN = {"intake", "checks", "describe", "review"}
PLANS: dict[str, str] = {
    "commit": "LLM-сообщение, git add явного списка, commit (gate)",
    "tag": "локальный тег, push main + тега, проверка через GitHub API (gate)",
    "record": "advent record с тега",
    "submit": "публикация видео на Диск, проверка public_url",
    "sheet": "cell comment в таблице, чтение обратно (gate)",
    "report": "LLM-черновик specs/reports",
}


def _load(stage: str, attr: str) -> Callable | None:
    for name in IMPLEMENTATIONS[stage][0]:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            continue
        fn = getattr(module, attr, None)
        if fn is not None:
            return fn
    return None


def resolve_stage(stage: str) -> StageFn:
    fn = _load(stage, IMPLEMENTATIONS[stage][1])
    if fn is None:
        raise ShipError(
            f"стадия {stage} не реализована ({IMPLEMENTATIONS[stage][0][0]})",
            exit_code=EXIT_CONFIG,
        )
    return fn


def resolve_recheck(stage: str) -> RecheckFn | None:
    return _load(stage, f"recheck_{stage}")


def _journal(ctx: ShipContext, stage: str, status: str, started: float) -> None:
    row = {
        "ts": ctx.now(),
        "week": ctx.week,
        "day": ctx.day,
        "kind": "ship_stage",
        "id": stage,
        "tag": ctx.tag,
        "status": status,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
    target = ctx.journal_path or (journal.LOG_DIR / "calls.jsonl")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass  # the journal is a side effect; it must not kill a stage


def _validate(stage: str | None, label: str) -> None:
    if stage is not None and stage not in STAGES:
        raise ShipError(
            f"неизвестная стадия для {label}: {stage}",
            exit_code=EXIT_CONFIG,
            hint="Стадии: " + ", ".join(STAGES),
        )


def _execute(ctx: ShipContext, stage: str, fn: StageFn) -> StageResult:
    started = time.monotonic()
    try:
        result = fn(ctx)
    except Exception as error:
        ctx.state.mark(stage, "failed", at=ctx.now(), data={"error": redact(str(error))[:500]})
        _journal(ctx, stage, "failed", started)
        raise
    ctx.state.mark(stage, "done", at=ctx.now(), data=result.data)
    _journal(ctx, stage, "done", started)
    return result


def _run_intake(ctx: ShipContext, registry: Mapping[str, StageFn] | None) -> None:
    """Always executed: it is cheap, and it is how a changed task file is noticed."""
    fn = registry["intake"] if registry is not None else resolve_stage("intake")
    started = time.monotonic()
    result = fn(ctx)
    if ctx.state.set_task_hash(result.data.get("task_hash")):
        console.warn("задание изменилось: результаты прошлых стадий сброшены")
    ctx.state.mark("intake", "done", at=ctx.now(), data=result.data)
    _journal(ctx, "intake", "done", started)
    console.note(f"intake: {result.summary}")


def require_prerequisites(ctx: ShipContext, stage: str) -> None:
    for earlier in STAGES[: STAGES.index(stage)]:
        if not ctx.state.is_valid(earlier):
            raise ShipError(
                f"--only {stage}: стадия {earlier} не выполнена или её входы изменились",
                exit_code=EXIT_CONFIG,
                hint=f"Запусти без --only или с --from {earlier}.",
            )


def run_pipeline(
    ctx: ShipContext,
    *,
    from_stage: str | None = None,
    only: str | None = None,
    skip: set[str] | None = None,
    registry: Mapping[str, StageFn] | None = None,
    rechecks: Mapping[str, RecheckFn] | None = None,
) -> list[str]:
    """Run stages in order; returns the ids executed this call. Stops on the first failure.

    `registry`/`rechecks` replace lazy module resolution entirely (tests inject fakes).
    """
    _validate(from_stage, "--from")
    _validate(only, "--only")
    if from_stage and only:
        raise ShipError("--from и --only взаимоисключающие", exit_code=EXIT_CONFIG)
    skip = skip or set()
    executed: list[str] = []

    _run_intake(ctx, registry)
    executed.append("intake")
    if only == "intake":
        return executed

    if only:
        require_prerequisites(ctx, only)
        order = [only]
    else:
        if from_stage:
            ctx.state.invalidate_from(from_stage)
        order = STAGES[1:]

    for stage in order:
        if stage in skip:
            ctx.state.mark(stage, "skipped", at=ctx.now(), data={"reason": "пропущено флагом"})
            console.note(f"{stage}: пропущено")
            continue
        if ctx.state.status(stage) == "done" and ctx.state.is_valid(stage):
            check = rechecks.get(stage) if rechecks is not None else resolve_recheck(stage)
            if check is None or check(ctx):
                console.note(f"{stage}: уже сделано")
                continue
            console.warn(f"{stage}: внешнее состояние разошлось со state, повтор")
        if ctx.dry_run and stage not in LIVE_IN_DRY_RUN:
            console.note(f"[dry-run] {stage}: {PLANS.get(stage, 'план')}")
            continue
        if ctx.state.entry(stage) is not None:
            ctx.state.invalidate(stage)
        fn = registry[stage] if registry is not None else resolve_stage(stage)
        console.note(f"{stage}: ...")
        result = _execute(ctx, stage, fn)
        executed.append(stage)
        console.note(f"{stage}: готово. {result.summary}")
    return executed


def status_rows(state: ShipState, stages: list[str] | None = None) -> list[tuple[str, str, str]]:
    """(stage, status, at); a `done` whose inputs changed is shown as `stale`."""
    rows = []
    for stage in stages or STAGES:
        entry = state.entry(stage)
        if entry is None:
            rows.append((stage, "pending", ""))
            continue
        status = entry["status"]
        if status in ("done", "skipped") and not state.is_valid(stage):
            status = "stale"
        rows.append((stage, status, entry.get("at", "")))
    return rows
