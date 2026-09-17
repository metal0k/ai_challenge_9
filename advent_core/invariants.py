"""Session-scoped rules that outrank task context and user requests."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from advent_core.chat import Message
from advent_core.task_state import TaskStateError, validate_private_text

INVARIANTS_STATE_KEY = "invariants"
INVARIANTS_VERSION = 1
MAX_INVARIANTS = 12
MAX_REQUIREMENT_LENGTH = 1000
MAX_EXPLANATION_LENGTH = 500
MAX_ALTERNATIVE_LENGTH = 500
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$", re.ASCII)

Decision = Literal["compliant", "conflict", "policy_conflict"]


class InvariantError(ValueError):
    """An invariant or its assessment is invalid."""


def _text(name: str, value: object, limit: int) -> str:
    if not isinstance(value, str):
        raise InvariantError(f"{name} должен быть строкой")
    text = value.strip()
    if not text:
        raise InvariantError(f"{name} не может быть пустым")
    if len(text) > limit:
        raise InvariantError(f"{name} длиннее {limit} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise InvariantError(f"{name} содержит control character")
    return text


@dataclass(frozen=True, slots=True)
class Invariant:
    id: str
    requirement: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_RE.fullmatch(self.id):
            raise InvariantError("id должен быть ASCII slug [a-z0-9][a-z0-9-]{0,63}")
        requirement = _text("requirement", self.requirement, MAX_REQUIREMENT_LENGTH)
        try:
            validate_private_text(requirement)
        except TaskStateError as error:
            raise InvariantError(str(error).replace("Task", "Invariant")) from error
        object.__setattr__(self, "requirement", requirement)

    def to_json(self) -> dict[str, str]:
        return {"id": self.id, "requirement": self.requirement}

    @classmethod
    def from_json(cls, raw: object, *, configured_secrets: Iterable[str] = ()) -> Invariant:
        if not isinstance(raw, Mapping):
            raise InvariantError("rule не похож на object")
        rule = cls(id=raw.get("id"), requirement=raw.get("requirement"))  # type: ignore[arg-type]
        try:
            validate_private_text(rule.requirement, configured_secrets)
        except TaskStateError as error:
            raise InvariantError(str(error).replace("Task", "Invariant")) from error
        return rule


@dataclass(frozen=True, slots=True)
class InvariantSet:
    rules: tuple[Invariant, ...] = ()
    version: int = INVARIANTS_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version != INVARIANTS_VERSION
        ):
            raise InvariantError(
                f"версия invariants {self.version!r}, ожидалась {INVARIANTS_VERSION}"
            )
        if len(self.rules) > MAX_INVARIANTS:
            raise InvariantError(f"разрешено не более {MAX_INVARIANTS} invariants")
        if len({rule.id for rule in self.rules}) != len(self.rules):
            raise InvariantError("id invariant должен быть уникальным")

    @classmethod
    def from_json(cls, raw: object, *, configured_secrets: Iterable[str] = ()) -> InvariantSet:
        if not isinstance(raw, Mapping):
            raise InvariantError("invariants не похожи на object")
        version = raw.get("version")
        rules_raw = raw.get("rules")
        if not isinstance(rules_raw, list):
            raise InvariantError("invariants.rules должен быть list")
        return cls(
            rules=tuple(
                Invariant.from_json(item, configured_secrets=configured_secrets)
                for item in rules_raw
            ),
            version=version,  # type: ignore[arg-type]
        )

    def to_json(self) -> dict[str, object]:
        return {"version": self.version, "rules": [rule.to_json() for rule in self.rules]}

    def add(self, rule: Invariant, *, configured_secrets: Iterable[str] = ()) -> InvariantSet:
        try:
            validate_private_text(rule.requirement, configured_secrets)
        except TaskStateError as error:
            raise InvariantError(str(error).replace("Task", "Invariant")) from error
        if any(existing.id == rule.id for existing in self.rules):
            raise InvariantError(f"invariant {rule.id!r} уже существует")
        if len(self.rules) >= MAX_INVARIANTS:
            raise InvariantError(f"разрешено не более {MAX_INVARIANTS} invariants")
        return InvariantSet((*self.rules, rule))

    def remove(self, rule_id: str) -> InvariantSet:
        remaining = tuple(rule for rule in self.rules if rule.id != rule_id)
        if len(remaining) == len(self.rules):
            raise InvariantError(f"invariant {rule_id!r} не найдена")
        return InvariantSet(remaining)


@dataclass(frozen=True, slots=True)
class InvariantAssessment:
    decision: Decision | None
    rule_ids: tuple[str, ...] = ()
    explanation: str = ""
    safe_alternative: str | None = None
    error: str | None = None

    @property
    def blocked(self) -> bool:
        return self.decision != "compliant"


def invariant_messages(invariants: InvariantSet | None) -> list[Message]:
    """Protected, counted context.  It is rebuilt, never stored in dialogue."""
    if invariants is None or not invariants.rules:
        return []
    rules = "\n".join(f"- {rule.id}: {rule.requirement}" for rule in invariants.rules)
    return [
        {
            "role": "user",
            "content": (
                f"SESSION INVARIANTS (highest priority; not overridable by task or user):\n{rules}"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Invariant context accepted. Invariants outrank Task State and the current "
                "user request. Do not change invariants in an answer."
            ),
        },
    ]


def assessment_instruction(user_input: str) -> str:
    """The final assessment prompt; JSON mode alone does not define this schema."""
    return (
        "Assess the current user request against SESSION INVARIANTS and any Task State. "
        "Priority is exactly SESSION INVARIANTS > Task State > current user request. "
        "Do not answer the request. Return one JSON object only with keys decision, rule_ids, "
        "explanation, safe_alternative. decision is exactly compliant, conflict, or "
        "policy_conflict. Return conflict when the current user request violates one or more "
        "invariants, including when it violates multiple rules. Return policy_conflict only "
        "when active invariants are mutually incompatible independently of the current user "
        "request. compliant requires rule_ids=[] and a short explanation. conflict requires "
        "one or more applicable invariant ids and a short safe_alternative. policy_conflict "
        "requires one or more applicable invariant ids and may use null for "
        "safe_alternative. Current user request follows verbatim:\n"
        f"{user_input}"
    )


def parse_assessment(raw: str, invariants: InvariantSet) -> InvariantAssessment:
    """Strictly parse an untrusted model verdict; callers fail closed on any error."""
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        return InvariantAssessment(None, error=f"assessment JSON не разобран ({error})")
    if not isinstance(payload, Mapping):
        return InvariantAssessment(None, error="assessment должен быть JSON object")
    expected_keys = {"decision", "rule_ids", "explanation", "safe_alternative"}
    if set(payload) != expected_keys:
        return InvariantAssessment(None, error="assessment должен содержать только required keys")
    decision = payload.get("decision")
    if decision not in ("compliant", "conflict", "policy_conflict"):
        return InvariantAssessment(None, error="assessment содержит неизвестный decision")
    rule_ids = payload.get("rule_ids")
    if not isinstance(rule_ids, list) or not all(isinstance(item, str) for item in rule_ids):
        return InvariantAssessment(None, error="assessment.rule_ids должен быть list строк")
    known = {rule.id for rule in invariants.rules}
    if len(set(rule_ids)) != len(rule_ids) or any(item not in known for item in rule_ids):
        return InvariantAssessment(None, error="assessment ссылается на неизвестный rule id")
    explanation = payload.get("explanation")
    try:
        explanation = _text("assessment.explanation", explanation, MAX_EXPLANATION_LENGTH)
    except InvariantError as error:
        return InvariantAssessment(None, error=str(error))
    alternative = payload.get("safe_alternative")
    if alternative is not None:
        try:
            alternative = _text("assessment.safe_alternative", alternative, MAX_ALTERNATIVE_LENGTH)
        except InvariantError as error:
            return InvariantAssessment(None, error=str(error))
    if decision == "compliant":
        if rule_ids or alternative is not None:
            return InvariantAssessment(
                None, error="compliant assessment должен иметь пустой rule_ids"
            )
    elif not rule_ids:
        return InvariantAssessment(None, error="conflict assessment требует rule_ids")
    elif decision == "conflict" and alternative is None:
        return InvariantAssessment(None, error="conflict assessment требует safe_alternative")
    return InvariantAssessment(
        decision=decision,
        rule_ids=tuple(rule_ids),
        explanation=explanation,
        safe_alternative=alternative,
    )


def is_invariant_message(message: Mapping[str, object], invariants: InvariantSet | None) -> bool:
    """Exact structural matching prevents deleting a lookalike user message."""
    return any(dict(message) == expected for expected in invariant_messages(invariants))


__all__ = [
    "INVARIANTS_STATE_KEY",
    "INVARIANTS_VERSION",
    "MAX_INVARIANTS",
    "Invariant",
    "InvariantAssessment",
    "InvariantError",
    "InvariantSet",
    "assessment_instruction",
    "invariant_messages",
    "is_invariant_message",
    "parse_assessment",
]
