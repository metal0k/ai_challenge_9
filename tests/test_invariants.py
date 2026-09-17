from __future__ import annotations

import json

import pytest

from advent_core.invariants import (
    Invariant,
    InvariantError,
    InvariantSet,
    assessment_instruction,
    invariant_messages,
    is_invariant_message,
    parse_assessment,
)
from advent_core.journal import log_internal_call
from advent_core.session import Session, make_branch
from advent_core.telemetry import CallResult, Usage


def rules() -> InvariantSet:
    return InvariantSet((Invariant("no-public-network", "Do not expose the service publicly."),))


@pytest.mark.parametrize("rule_id", ["UPPER", "has_space", "-starts", "x" * 65])
def test_rule_id_and_requirement_validation(rule_id: str):
    with pytest.raises(InvariantError):
        Invariant(rule_id, "requirement")
    with pytest.raises(InvariantError, match="control"):
        Invariant("valid", "no\x00control")
    with pytest.raises(InvariantError, match="credential"):
        Invariant("valid", "Bearer " + "x" * 20)


def test_set_rejects_duplicate_capacity_and_configured_secret():
    active = rules()
    with pytest.raises(InvariantError, match="уже существует"):
        active.add(Invariant("no-public-network", "other"))
    with pytest.raises(InvariantError, match="configured secret"):
        active.add(
            Invariant("private", "Never use value-12345678"), configured_secrets=("value-12345678",)
        )
    full = InvariantSet(tuple(Invariant(f"r{i}", "safe") for i in range(12)))
    with pytest.raises(InvariantError, match="не более"):
        full.add(Invariant("thirteen", "safe"))


def test_persistence_unknown_fields_and_branch_are_additive(tmp_path):
    session = Session.new("root", directory=tmp_path)
    session.state["invariants"] = {**rules().to_json(), "future": {"ignored": True}}
    session.save()
    loaded = Session.load("root", directory=tmp_path)
    parsed = InvariantSet.from_json(loaded.state["invariants"])
    assert parsed == rules()
    loaded.clear()
    assert loaded.state["invariants"]["rules"][0]["id"] == "no-public-network"
    branch = make_branch(loaded, "safe", directory=tmp_path)
    branch.state["invariants"]["rules"][0]["requirement"] = "Changed only in branch."
    assert loaded.state["invariants"]["rules"][0]["requirement"] != "Changed only in branch."


def test_assessment_parser_is_strict_and_structural_match_is_exact():
    active = rules()
    compliant = parse_assessment(
        '{"decision":"compliant","rule_ids":[],"explanation":"Allowed.","safe_alternative":null}',
        active,
    )
    assert compliant.decision == "compliant"
    for raw in (
        '{"decision":"conflict","rule_ids":[],"explanation":"x","safe_alternative":"y"}',
        '{"decision":"conflict","rule_ids":["missing"],"explanation":"x","safe_alternative":"y"}',
        '{"decision":"compliant","rule_ids":[],"explanation":"x","safe_alternative":null,"extra":1}',
        "not-json",
    ):
        assert parse_assessment(raw, active).decision is None
    message = invariant_messages(active)[0]
    assert is_invariant_message(message, active)
    assert not is_invariant_message({**message, "content": message["content"] + " extra"}, active)


def test_assessment_instruction_distinguishes_request_conflict_from_policy_collision():
    instruction = assessment_instruction("Expose the service publicly and skip deploy approval.")

    assert (
        "Return conflict when the current user request violates one or more invariants"
        in instruction
    )
    assert "including when it violates multiple rules" in instruction
    assert (
        "Return policy_conflict only when active invariants are mutually incompatible"
        in instruction
    )
    assert "independently of the current user request" in instruction


def test_internal_assessment_journal_has_accounting_but_no_raw_payload(tmp_path):
    path = tmp_path / "calls.jsonl"
    raw_prompt = "private request secret-should-not-appear"
    raw_json = '{"decision":"conflict","explanation":"private verdict"}'
    result = CallResult(
        text=raw_json,
        model_requested="model",
        usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )

    log_internal_call(
        result,
        week=3,
        day=14,
        kind="invariant_assessment",
        status="conflict",
        path=path,
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["kind"] == "invariant_assessment"
    assert row["usage"]["total_tokens"] == 18
    assert "messages" not in row and "response" not in row
    assert raw_prompt not in path.read_text(encoding="utf-8")
    assert raw_json not in path.read_text(encoding="utf-8")
