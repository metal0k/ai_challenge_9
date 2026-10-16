"""Runner: resume, invalidation, --from/--only, re-check hook, dry-run, journal, checks stage."""

import json
import subprocess

import pytest

from adventship.checks import run_checks
from adventship.ctx import ShipContext, ShipError, StageResult
from adventship.stages import resolve_stage, run_pipeline, status_rows
from adventship.state import STAGES, ShipState


class Fakes:
    """Counts calls per stage; behaviour is tweakable per test."""

    def __init__(self):
        self.calls: list[str] = []
        self.task_hash = "task-1"
        self.data: dict[str, dict] = {}
        self.fail: set[str] = set()

    def registry(self):
        reg = {}
        for stage in STAGES:
            reg[stage] = self._make(stage)
        return reg

    def _make(self, stage):
        def fn(ctx):
            self.calls.append(stage)
            if stage in self.fail:
                raise ShipError(f"{stage} broke", exit_code=10)
            data = {"v": stage, **self.data.get(stage, {})}
            if stage == "intake":
                data["task_hash"] = self.task_hash
            return StageResult(data=data, summary=f"{stage} ok")

        return fn


def make_ctx(tmp_path, **overrides):
    workdir = tmp_path / "ship"
    values = dict(
        week=7,
        day=35,
        tag="w07d35",
        root=tmp_path,
        workdir=workdir,
        env={},
        state=ShipState.load(workdir / "state.json"),
        run=lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", ""),
        complete=lambda *a, **k: None,
        gate=lambda q, d: True,
        driver_factory=lambda: None,
        now=lambda: "2026-10-16T10:00:00+03:00",
        journal_path=tmp_path / "journal.jsonl",
    )
    values.update(overrides)
    return ShipContext(**values)


@pytest.fixture
def fakes():
    return Fakes()


def test_full_run_executes_every_stage_in_order(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    executed = run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    assert executed == STAGES
    assert fakes.calls == STAGES
    assert all(status == "done" for _, status, _ in status_rows(ctx.state))


def test_resume_continues_from_failed_stage_and_skips_done(tmp_path, fakes):
    fakes.fail = {"review"}
    ctx = make_ctx(tmp_path)
    with pytest.raises(ShipError):
        run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    assert ctx.state.status("review") == "failed"
    assert "error" in ctx.state.data("review")

    fakes.calls.clear()
    fakes.fail = set()
    ctx2 = make_ctx(tmp_path)  # fresh process: state is re-read from disk
    run_pipeline(ctx2, registry=fakes.registry(), rechecks={})
    assert fakes.calls == ["intake"] + STAGES[STAGES.index("review") :]


def test_failure_stops_pipeline_at_first_failed_stage(tmp_path, fakes):
    fakes.fail = {"describe"}
    ctx = make_ctx(tmp_path)
    with pytest.raises(ShipError):
        run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    assert fakes.calls == ["intake", "checks", "describe"]
    assert ctx.state.status("review") == "pending"


def test_task_hash_change_reruns_everything(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    fakes.calls.clear()
    fakes.task_hash = "task-2"
    run_pipeline(make_ctx(tmp_path), registry=fakes.registry(), rechecks={})
    assert fakes.calls == STAGES


def test_changed_commit_output_invalidates_downstream_chain(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    fakes.calls.clear()
    fakes.data["commit"] = {"sha": "new"}
    run_pipeline(make_ctx(tmp_path), from_stage="commit", registry=fakes.registry(), rechecks={})
    assert fakes.calls == ["intake", *STAGES[STAGES.index("commit") :]]


def test_from_invalidates_stage_and_everything_after(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    fakes.calls.clear()
    run_pipeline(make_ctx(tmp_path), from_stage="review", registry=fakes.registry(), rechecks={})
    assert fakes.calls == ["intake", *STAGES[STAGES.index("review") :]]


def test_only_refuses_when_prerequisite_missing(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    with pytest.raises(ShipError) as caught:
        run_pipeline(ctx, only="tag", registry=fakes.registry(), rechecks={})
    assert caught.value.exit_code == 2
    assert fakes.calls == ["intake"]  # nothing past intake ran


def test_only_refuses_when_prerequisite_inputs_changed(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    # an upstream output changed behind the state's back: record's recorded hash is stale
    ctx.state.doc["stages"]["tag"]["data"]["v"] = "tampered"
    ctx.state.save()
    with pytest.raises(ShipError):
        run_pipeline(make_ctx(tmp_path), only="submit", registry=fakes.registry(), rechecks={})


def test_only_runs_single_stage_when_prerequisites_valid(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    fakes.calls.clear()
    ctx2 = make_ctx(tmp_path)
    ctx2.state.invalidate("sheet")
    executed = run_pipeline(ctx2, only="sheet", registry=fakes.registry(), rechecks={})
    assert fakes.calls == ["intake", "sheet"]
    assert executed == ["intake", "sheet"]


def test_from_and_only_are_exclusive_and_names_validated(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    with pytest.raises(ShipError):
        run_pipeline(ctx, from_stage="tag", only="tag", registry=fakes.registry())
    with pytest.raises(ShipError):
        run_pipeline(ctx, only="nope", registry=fakes.registry())
    assert fakes.calls == []


def test_recheck_failure_reruns_stage_and_dependents(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    fakes.calls.clear()
    seen = []

    def tag_gone(c):
        seen.append("tag")
        return False

    run_pipeline(make_ctx(tmp_path), registry=fakes.registry(), rechecks={"tag": tag_gone})
    assert seen == ["tag"]
    assert fakes.calls == ["intake", "tag", "record", "submit", "sheet", "report"]


def test_recheck_pass_reuses_stage(tmp_path, fakes):
    run_pipeline(make_ctx(tmp_path), registry=fakes.registry(), rechecks={})
    fakes.calls.clear()
    run_pipeline(
        make_ctx(tmp_path),
        registry=fakes.registry(),
        rechecks={"tag": lambda c: True, "submit": lambda c: True, "sheet": lambda c: True},
    )
    assert fakes.calls == ["intake"]


def test_skip_marks_stage_skipped_and_it_counts_as_satisfied(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx, skip={"record"}, registry=fakes.registry(), rechecks={})
    assert "record" not in fakes.calls
    assert ctx.state.status("record") == "skipped"
    fakes.calls.clear()
    run_pipeline(make_ctx(tmp_path), skip={"record"}, registry=fakes.registry(), rechecks={})
    assert fakes.calls == ["intake"]
    # running again without the flag executes the stage and what depends on it
    run_pipeline(make_ctx(tmp_path), registry=fakes.registry(), rechecks={})
    assert "record" in fakes.calls


def test_dry_run_executes_only_live_stages_and_writes_no_plan_state(tmp_path, fakes):
    ctx = make_ctx(tmp_path, dry_run=True)
    executed = run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    assert executed == ["intake", "checks", "describe", "review"]
    assert fakes.calls == executed
    assert ctx.state.status("commit") == "pending"
    assert ctx.state.status("sheet") == "pending"


def test_journal_rows_for_stage_states(tmp_path, fakes):
    fakes.fail = {"checks"}
    ctx = make_ctx(tmp_path)
    with pytest.raises(ShipError):
        run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    rows = [
        json.loads(line) for line in (tmp_path / "journal.jsonl").read_text("utf-8").splitlines()
    ]
    assert [(r["id"], r["status"]) for r in rows] == [("intake", "done"), ("checks", "failed")]
    assert all(r["kind"] == "ship_stage" and r["day"] == 35 for r in rows)
    assert all("duration_ms" in r for r in rows)


def test_error_text_is_redacted_in_state(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr("adventship.stages.redact", lambda text: text.replace("SECRET", "***"))

    def leaky(ctx):
        raise ShipError("token SECRET leaked")

    reg = fakes.registry()
    reg["checks"] = leaky
    ctx = make_ctx(tmp_path)
    with pytest.raises(ShipError):
        run_pipeline(ctx, registry=reg, rechecks={})
    assert "SECRET" not in ctx.state.data("checks")["error"]


def test_unexpected_exception_is_recorded_as_failed(tmp_path, fakes):
    def crash(ctx):
        raise RuntimeError("boom")

    reg = fakes.registry()
    reg["review"] = crash
    ctx = make_ctx(tmp_path)
    with pytest.raises(RuntimeError):
        run_pipeline(ctx, registry=reg, rechecks={})
    assert ctx.state.status("review") == "failed"


def test_status_rows_flag_stale_done(tmp_path, fakes):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx, registry=fakes.registry(), rechecks={})
    ctx.state.doc["stages"]["describe"]["data"]["v"] = "changed"
    rows = dict((s, st) for s, st, _ in status_rows(ctx.state))
    assert rows["describe"] == "done"
    assert rows["commit"] == "stale"


def test_missing_stage_module_is_a_config_error():
    # nothing registers "record" in this repo state unless another module exists; either
    # it resolves to a callable or it raises exit 2 -- never ImportError
    try:
        assert callable(resolve_stage("record"))
    except ShipError as error:
        assert error.exit_code == 2


# --- checks stage -------------------------------------------------------------------


def test_checks_runs_all_commands_and_passes(tmp_path):
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    result = run_checks(make_ctx(tmp_path, run=run))
    assert len(seen) == 4
    assert seen[0][-2:] == ["check", "."]
    assert "pytest" in " ".join(seen[2])
    assert any("check_staged" in part for part in seen[3])
    assert len(result.data["checks"]) == 4


def test_checks_failure_exit_12_stops_at_first_failure(tmp_path):
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        code = 1 if "pytest" in argv else 0
        return subprocess.CompletedProcess(argv, code, "1 failed\n", "trace")

    with pytest.raises(ShipError) as caught:
        run_checks(make_ctx(tmp_path, run=run))
    assert caught.value.exit_code == 12
    assert "pytest" in caught.value.message
    assert "1 failed" in caught.value.hint
    assert len(seen) == 3


def test_stage_order_is_the_spec_order():
    assert STAGES == [
        "intake", "checks", "describe", "review", "commit",
        "tag", "record", "submit", "sheet", "report",
    ]  # fmt: skip


def test_dry_run_keeps_failed_entry_of_planned_stage(tmp_path, fakes):
    fakes.fail = {"commit"}
    with pytest.raises(ShipError):
        run_pipeline(make_ctx(tmp_path), registry=fakes.registry(), rechecks={})
    fakes.calls.clear()
    fakes.fail = set()
    dry = make_ctx(tmp_path, dry_run=True)
    run_pipeline(dry, registry=fakes.registry(), rechecks={})
    assert fakes.calls == ["intake"]
    assert dry.state.status("commit") == "failed"
