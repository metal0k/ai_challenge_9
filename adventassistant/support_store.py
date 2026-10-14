"""Strict reader for the fictional, read-only support fixture."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from advent_core.errors import AdventError

DATA_PATH = Path(__file__).resolve().parent / "support_data.json"
MAX_FILE_BYTES = 128 * 1024
MAX_USERS = 100
MAX_TICKETS = 100
MAX_STRING = 1000
MAX_MESSAGES = 10
ID_PATTERN = re.compile(r"[A-Za-z0-9-]{1,64}\Z", re.ASCII)
USER_FIELDS = {"id", "display_name", "plan", "account_status", "locale"}
TICKET_FIELDS = {"id", "user_id", "subject", "status", "category", "facts", "messages"}
FACT_KEYS = {"cause", "link_age_minutes", "failed_attempts"}


def _invalid(detail: str) -> AdventError:
    return AdventError(f"Файл поддержки повреждён: {detail}.")


def _bounded_string(value: Any, name: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or len(value) > (64 if identifier else MAX_STRING):
        raise _invalid(f"поле {name!r} должно быть непустой строкой допустимой длины")
    if identifier and ID_PATTERN.fullmatch(value) is None:
        raise _invalid(f"поле {name!r} содержит недопустимый ID")
    return value


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise _invalid(f"{name} содержит неверный набор полей")
    return value


def load_support_data(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the whole fixture before returning any selected record."""
    try:
        with (path or DATA_PATH).open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise _invalid(f"не удалось прочитать ({exc})") from None
    if len(raw) > MAX_FILE_BYTES:
        raise _invalid(f"размер больше {MAX_FILE_BYTES} байт")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _invalid("ожидался UTF-8 JSON") from None
    if not isinstance(data, dict) or set(data) != {"version", "users", "tickets"}:
        raise _invalid("корневой объект содержит неверный набор полей")
    if type(data["version"]) is not int or data["version"] != 1:
        raise _invalid("version должен быть целым числом 1")
    users = data["users"]
    tickets = data["tickets"]
    if not isinstance(users, list) or not 1 <= len(users) <= MAX_USERS:
        raise _invalid(f"users должен содержать от 1 до {MAX_USERS} записей")
    if not isinstance(tickets, list) or len(tickets) > MAX_TICKETS:
        raise _invalid(f"tickets должен содержать не более {MAX_TICKETS} записей")

    user_ids: set[str] = set()
    for index, raw_user in enumerate(users):
        user = _object(raw_user, USER_FIELDS, f"users[{index}]")
        user_id = _bounded_string(user["id"], "user.id", identifier=True)
        if user_id in user_ids:
            raise _invalid(f"повторяется user ID {user_id!r}")
        user_ids.add(user_id)
        _bounded_string(user["display_name"], "user.display_name")
        if not isinstance(user["plan"], str) or user["plan"] not in {"free", "pro"}:
            raise _invalid("user.plan должен быть free или pro")
        if not isinstance(user["account_status"], str) or user["account_status"] not in {
            "active",
            "locked",
        }:
            raise _invalid("user.account_status должен быть active или locked")
        if not isinstance(user["locale"], str) or user["locale"] not in {"ru", "en"}:
            raise _invalid("user.locale должен быть ru или en")

    ticket_ids: set[str] = set()
    for index, raw_ticket in enumerate(tickets):
        ticket = _object(raw_ticket, TICKET_FIELDS, f"tickets[{index}]")
        ticket_id = _bounded_string(ticket["id"], "ticket.id", identifier=True)
        if ticket_id in ticket_ids:
            raise _invalid(f"повторяется ticket ID {ticket_id!r}")
        ticket_ids.add(ticket_id)
        owner = _bounded_string(ticket["user_id"], "ticket.user_id", identifier=True)
        if owner not in user_ids:
            raise _invalid(f"тикет {ticket_id!r} ссылается на неизвестного владельца")
        _bounded_string(ticket["subject"], "ticket.subject")
        if not isinstance(ticket["status"], str) or ticket["status"] not in {"open", "closed"}:
            raise _invalid("ticket.status должен быть open или closed")
        if not isinstance(ticket["category"], str) or ticket["category"] not in {
            "auth",
            "billing",
            "other",
        }:
            raise _invalid("ticket.category должен быть auth, billing или other")
        facts = ticket["facts"]
        if not isinstance(facts, dict) or not set(facts) <= FACT_KEYS:
            raise _invalid("ticket.facts содержит недопустимые ключи")
        for key, value in facts.items():
            _bounded_string(value, f"ticket.facts.{key}")
        messages = ticket["messages"]
        if not isinstance(messages, list) or len(messages) > MAX_MESSAGES:
            raise _invalid(f"ticket.messages должен содержать не более {MAX_MESSAGES} строк")
        for message in messages:
            _bounded_string(message, "ticket.messages")
    return data


def get_support_context(
    user_id: str, ticket_id: str | None, *, path: Path | None = None
) -> dict[str, Any]:
    """Return just one user and optionally their owned ticket."""
    if not isinstance(user_id, str) or ID_PATTERN.fullmatch(user_id) is None:
        raise AdventError("Некорректный USER: ожидается ID из ASCII-букв, цифр и дефиса.")
    if ticket_id is not None and (
        not isinstance(ticket_id, str) or ID_PATTERN.fullmatch(ticket_id) is None
    ):
        raise AdventError("Некорректный TICKET: ожидается ID из ASCII-букв, цифр и дефиса.")
    data = load_support_data(path)
    user = next((item for item in data["users"] if item["id"] == user_id), None)
    if user is None:
        raise AdventError(f"Пользователь {user_id!r} не найден.")
    ticket = None
    if ticket_id is not None:
        ticket = next((item for item in data["tickets"] if item["id"] == ticket_id), None)
        if ticket is None:
            raise AdventError(f"Тикет {ticket_id!r} не найден.")
        if ticket["user_id"] != user_id:
            raise AdventError(f"Тикет {ticket_id!r} не принадлежит пользователю {user_id!r}.")
    return {"user": user, "ticket": ticket}
