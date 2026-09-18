from __future__ import annotations

import pytest

from advent_core.task_state import TaskState, TaskStateError, is_task_message, task_messages


def task() -> TaskState:
    return TaskState.start("Release API", "Write plan", "Approve risks")


def test_round_trip_ignores_unknown_keys():
    raw = {**task().to_json(), "future": {"safe": True}}
    assert TaskState.from_json(raw) == task()


def test_constructor_rejects_boolean_version():
    with pytest.raises(TaskStateError, match="версия"):
        TaskState("goal", "planning", "step", "expected", version=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("phase", "review"),
        ("goal", ""),
        ("current_step", "x\ny"),
        ("plan_approved", 1),
        ("paused", 1),
    ],
)
def test_invalid_persisted_fields_are_rejected(field, value):
    raw = task().to_json()
    raw[field] = value
    with pytest.raises(TaskStateError):
        TaskState.from_json(raw)


def test_lengths_and_control_characters_are_rejected():
    with pytest.raises(TaskStateError, match="1000"):
        TaskState.start("x" * 1001, "step", "expected")
    with pytest.raises(TaskStateError, match="control"):
        TaskState.start("goal\x00", "step", "expected")


@pytest.mark.parametrize(
    "text",
    [
        "Bearer " + "a" * 20,
        "sk-" + "a" * 17,
        "-----BEGIN PRIVATE KEY-----",
        "api_key=" + "a" * 16,
    ],
)
def test_credential_shapes_are_rejected(text):
    with pytest.raises(TaskStateError, match="credential"):
        TaskState.start(text, "step", "expected")


def test_configured_secret_has_absolute_priority_but_placeholders_are_allowed():
    TaskState.start("rotate API key", "use <token>", "confirm placeholder")
    with pytest.raises(TaskStateError, match="configured secret"):
        TaskState.start(
            "release secret-value-123", "step", "expected", configured_secrets=["secret-value-123"]
        )


def test_legal_forward_and_retry_transitions():
    planning = task().update("Plan canary", "Approve plan").approve()
    execution = planning.advance("execution", "Deploy canary", "Read metrics")
    validation = execution.advance("validation", "Check metrics", "Decide rollback")
    retry = validation.advance("execution", "Rollback", "Confirm recovery")
    done = retry.advance("validation", "Recheck", "Record result").complete("stable")
    assert done.to_json() == {
        "version": 1,
        "goal": "Release API",
        "phase": "done",
        "current_step": "Task completed",
        "expected_action": "none",
        "plan_approved": True,
        "paused": False,
        "pause_reason": None,
        "result": "stable",
    }


def test_legacy_task_without_plan_approval_requires_explicit_approval():
    raw = task().to_json()
    raw.pop("plan_approved")
    restored = TaskState.from_json(raw)
    assert restored.plan_approved is False
    with pytest.raises(TaskStateError, match="/task approve"):
        restored.advance("execution", "deploy", "observe")


def test_approval_requires_active_planning_and_resets_after_plan_update():
    approved = task().approve()
    assert approved.plan_approved is True
    with pytest.raises(TaskStateError, match="уже approved"):
        approved.approve()
    changed = approved.update("Revised plan", "Approve revised plan")
    assert changed.plan_approved is False
    with pytest.raises(TaskStateError, match="/task approve"):
        changed.advance("execution", "deploy", "observe")
    execution = changed.approve().advance("execution", "deploy", "observe")
    with pytest.raises(TaskStateError, match="только в planning"):
        execution.approve()
    with pytest.raises(TaskStateError, match="paused"):
        task().pause().approve()


def test_retry_plan_approval_is_preserved():
    approved = task().approve()
    execution = approved.advance("execution", "deploy", "observe")
    validation = execution.advance("validation", "check", "decide")
    retry = validation.advance("execution", "rollback", "confirm")
    assert retry.plan_approved is True


@pytest.mark.parametrize("phase", ["planning", "validation", "done"])
def test_illegal_execution_transitions_are_rejected(phase):
    with pytest.raises(TaskStateError, match="transition"):
        task().advance(phase, "step", "expected")


def test_pause_preserves_work_and_allows_only_resume_semantically():
    before = task()
    paused = before.pause("waiting")
    assert (paused.phase, paused.current_step, paused.expected_action) == (
        before.phase,
        before.current_step,
        before.expected_action,
    )
    with pytest.raises(TaskStateError, match="paused"):
        paused.update("other", "other")
    assert paused.resume() == before


def test_done_is_terminal_and_not_injected():
    done = (
        task()
        .approve()
        .advance("execution", "ship", "observe")
        .advance("validation", "check", "decide")
        .complete("ok")
    )
    assert task_messages(done) == []
    with pytest.raises(TaskStateError, match="terminal"):
        done.pause()


def test_task_messages_name_priority_and_are_matched_structurally():
    state = task()
    messages = task_messages(state)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert "Current user request" in messages[1]["content"]
    assert "не меняет formal state" in messages[1]["content"]
    assert all(is_task_message(message, state) for message in messages)
    lookalike = {"role": "user", "content": messages[0]["content"] + " extra"}
    assert not is_task_message(lookalike, state)


def test_task_messages_commands_present():
    state = task()
    messages = task_messages(state)
    assert messages[1]["content"].find("/task approve") != -1
    approved_state = state.approve()
    approved_messages = task_messages(approved_state)
    assert approved_messages[1]["content"].find("/task advance execution") != -1
