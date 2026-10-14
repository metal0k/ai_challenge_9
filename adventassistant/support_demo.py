"""Deterministic checks for the Day 33 live recording scenario.

The preflight and owner-error commands exercise the real stdio MCP client via
the support pipeline helper. They never call embeddings, reranking or answer
generation, so the recording rehearsal is safe to repeat.
"""

from __future__ import annotations

import argparse

from advent_core import console
from advent_core.errors import AdventError

_EXPECTED_OWNER_ERROR = "Тикет 'AUTH-102' не принадлежит пользователю 'demo-a'."


def _mcp_context(user_id: str, ticket_id: str | None) -> dict[str, object]:
    # Use the pipeline's exact MCP boundary so rehearsal and the live command
    # cannot drift into separate implementations of subprocess handling.
    from adventassistant.support_pipeline import _mcp_context as fetch_context

    return fetch_context(user_id, ticket_id)


def _assert_context(
    context: dict[str, object], *, user_id: str, ticket_id: str, cause: str
) -> None:
    user = context.get("user")
    ticket = context.get("ticket")
    if not isinstance(user, dict) or user.get("id") != user_id:
        raise AdventError(f"MCP preflight вернул неверный профиль для {user_id}.")
    if not isinstance(ticket, dict) or ticket.get("id") != ticket_id:
        raise AdventError(f"MCP preflight вернул неверный тикет для {user_id}.")
    facts = ticket.get("facts")
    if not isinstance(facts, dict) or facts.get("cause") != cause:
        raise AdventError(f"MCP preflight не подтвердил причину тикета {ticket_id}.")


def _assert_owner_denied() -> str:
    try:
        _mcp_context("demo-a", "AUTH-102")
    except AdventError as exc:
        # MCP SDK wraps an error result with "Error executing tool <name>:";
        # verify the server's exact denial while allowing that transport prefix.
        if _EXPECTED_OWNER_ERROR not in exc.message:
            raise AdventError(
                f"MCP вернул неожиданный отказ для чужого тикета: {exc.message}"
            ) from exc
        return _EXPECTED_OWNER_ERROR
    raise AdventError("MCP вернул чужой тикет вместо ожидаемого отказа.")


def preflight() -> None:
    """Verify both real fixture scopes and the exact denial without paid calls."""
    first = _mcp_context("demo-a", "AUTH-101")
    _assert_context(first, user_id="demo-a", ticket_id="AUTH-101", cause="expired_magic_link")
    second = _mcp_context("demo-b", "AUTH-102")
    _assert_context(
        second,
        user_id="demo-b",
        ticket_id="AUTH-102",
        cause="locked_after_failed_attempts",
    )
    denial = _assert_owner_denied()
    console.out.print(
        "stdio MCP preflight: demo-a/AUTH-101 и demo-b/AUTH-102 получены в своих "
        "областях; чужой тикет отклонён точным ожидаемым отказом.",
        markup=False,
        highlight=False,
    )
    console.out.print(denial, markup=False, highlight=False)


def owner_error() -> None:
    """Show the specific expected denial; unexpected errors keep a nonzero exit."""
    message = _assert_owner_denied()
    console.out.print("Ожидаемый отказ stdio MCP:", markup=False, highlight=False)
    console.out.print(message, markup=False, highlight=False)


def main() -> None:
    console.force_utf8()
    parser = argparse.ArgumentParser(description="Проверки демо поддержки Day 33.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--owner-error", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            preflight()
        else:
            owner_error()
    except AdventError as exc:
        console.fail(exc)
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    main()
