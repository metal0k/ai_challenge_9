"""Day 33 recording plan and the exact stdio-MCP proof guard; no live calls."""

from __future__ import annotations

import pytest

from advent_core.errors import AdventError
from adventassistant import support_demo


def test_day_33_scenario_has_six_steps_and_reuses_the_auth_question() -> None:
    from advent_cli import record

    steps = record.demo_steps(7, 33)
    assert len(steps) == 6
    assert steps[0].note is not None
    assert steps[0].args == []
    assert steps[-1].note is not None
    assert steps[-1].line_pause == 8.0

    first, second, unknown, owner_error = steps[1:5]
    assert first.module == second.module == unknown.module == "adventassistant.cli"
    assert first.args == [
        "support",
        "Почему не работает авторизация?",
        "--user",
        "demo-a",
        "--ticket",
        "AUTH-101",
    ]
    assert second.args == [
        "support",
        "Почему не работает авторизация?",
        "--user",
        "demo-b",
        "--ticket",
        "AUTH-102",
    ]
    assert unknown.args[:2] == [
        "support",
        "Где находится ближайший офис поддержки в Казани?",
    ]
    assert owner_error.module == "adventassistant.support_demo"
    assert owner_error.args == ["--owner-error"]
    assert owner_error.expect_failure is False


def test_day_33_rehearsal_uses_the_real_mcp_preflight() -> None:
    from advent_cli import record

    steps = record.rehearsal_steps(7, 33)
    assert len(steps) == 1
    assert steps[0].module == "adventassistant.support_demo"
    assert steps[0].args == ["--preflight"]
    assert steps[0].timeout == 60


def test_day_33_index_preparation_is_lazy_and_day_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    from advent_cli import record
    from adventassistant import support_index

    required: list[object] = []
    monkeypatch.setattr(
        support_index,
        "require_support_index",
        lambda path: required.append(path),
    )
    record._prepare_w07d33(7, 32)
    record._prepare_w07d33(1, 1)
    assert required == []

    record._prepare_w07d33(7, 33)
    assert required == [support_index.SUPPORT_INDEX_PATH]


def _context(user_id: str, ticket_id: str, cause: str) -> dict[str, object]:
    return {
        "user": {"id": user_id},
        "ticket": {"id": ticket_id, "facts": {"cause": cause}},
    }


def test_support_demo_preflight_checks_both_scopes_and_exact_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str | None]] = []

    def fake_mcp(user_id: str, ticket_id: str | None) -> dict[str, object]:
        seen.append((user_id, ticket_id))
        if (user_id, ticket_id) == ("demo-a", "AUTH-101"):
            return _context("demo-a", "AUTH-101", "expired_magic_link")
        if (user_id, ticket_id) == ("demo-b", "AUTH-102"):
            return _context("demo-b", "AUTH-102", "locked_after_failed_attempts")
        raise AdventError("Тикет 'AUTH-102' не принадлежит пользователю 'demo-a'.")

    monkeypatch.setattr(support_demo, "_mcp_context", fake_mcp)
    support_demo.preflight()
    assert seen == [
        ("demo-a", "AUTH-101"),
        ("demo-b", "AUTH-102"),
        ("demo-a", "AUTH-102"),
    ]


@pytest.mark.parametrize(
    "error",
    [
        "MCP не запустился.",
        "Тикет 'AUTH-101' не принадлежит пользователю 'demo-b'.",
    ],
)
def test_owner_guard_rejects_unexpected_errors(monkeypatch: pytest.MonkeyPatch, error: str) -> None:
    monkeypatch.setattr(
        support_demo,
        "_mcp_context",
        lambda *_args: (_ for _ in ()).throw(AdventError(error)),
    )
    with pytest.raises(AdventError, match="неожиданный отказ"):
        support_demo.owner_error()


def test_owner_guard_rejects_a_successful_foreign_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support_demo,
        "_mcp_context",
        lambda *_args: _context("demo-a", "AUTH-102", "locked_after_failed_attempts"),
    )
    with pytest.raises(AdventError, match="вернул чужой тикет"):
        support_demo.owner_error()
