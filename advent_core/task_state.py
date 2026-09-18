"""Persistent finite-state task context for the agent."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from advent_core.chat import Message

TASK_VERSION = 1
TASK_STATE_KEY = "task"
TASK_PHASES = ("planning", "execution", "validation", "done")
TaskPhase = Literal["planning", "execution", "validation", "done"]

_ASSIGNMENT_RE = re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*([^\s,;]+)")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")
_SK_RE = re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{17,}")
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")


class TaskStateError(ValueError):
    """Invalid task data or transition."""


def _text(name: str, value: object, limit: int) -> str:
    if not isinstance(value, str):
        raise TaskStateError(f"{name} должен быть строкой")
    cleaned = value.strip()
    if not cleaned:
        raise TaskStateError(f"{name} не может быть пустым")
    if len(cleaned) > limit:
        raise TaskStateError(f"{name} длиннее {limit} characters")
    if any(unicodedata.category(char) == "Cc" for char in cleaned):
        raise TaskStateError(f"{name} содержит control character")
    return cleaned


def validate_private_text(text: str, configured_secrets: Iterable[str] = ()) -> None:
    """Reject concrete credentials before task data reaches disk or a request."""
    folded = text.casefold()
    for secret in configured_secrets:
        if isinstance(secret, str) and len(secret) >= 8 and secret.casefold() in folded:
            raise TaskStateError("Task не может содержать configured secret")
    if _BEARER_RE.search(text) or _SK_RE.search(text) or _PRIVATE_KEY_RE.search(text):
        raise TaskStateError("Task не может содержать credential-like значение")
    assignment = _ASSIGNMENT_RE.search(text)
    if assignment and len(assignment.group(1)) >= 16:
        raise TaskStateError("Task не может содержать credential-like assignment")


@dataclass(frozen=True, slots=True)
class TaskState:
    goal: str
    phase: TaskPhase
    current_step: str
    expected_action: str
    plan_approved: bool = False
    paused: bool = False
    pause_reason: str | None = None
    result: str | None = None
    version: int = TASK_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version != TASK_VERSION
        ):
            raise TaskStateError(f"версия task {self.version!r}, ожидалась {TASK_VERSION}")
        if self.phase not in TASK_PHASES:
            raise TaskStateError(f"неизвестная task phase {self.phase!r}")
        object.__setattr__(self, "goal", _text("goal", self.goal, 1000))
        object.__setattr__(self, "current_step", _text("current_step", self.current_step, 1000))
        object.__setattr__(
            self, "expected_action", _text("expected_action", self.expected_action, 1000)
        )
        if not isinstance(self.plan_approved, bool):
            raise TaskStateError("plan_approved должен быть boolean")
        if not isinstance(self.paused, bool):
            raise TaskStateError("paused должен быть boolean")
        if self.pause_reason is not None:
            object.__setattr__(self, "pause_reason", _text("pause_reason", self.pause_reason, 500))
        if self.result is not None:
            object.__setattr__(self, "result", _text("result", self.result, 2000))
        if self.phase == "done":
            if self.paused or self.result is None or self.expected_action != "none":
                raise TaskStateError(
                    "done task требует paused=false, result и expected_action='none'"
                )
        elif self.result is not None:
            raise TaskStateError("result разрешён только для done task")
        if not self.paused and self.pause_reason is not None:
            raise TaskStateError("pause_reason разрешён только для paused task")

    @classmethod
    def start(
        cls,
        goal: str,
        current_step: str,
        expected_action: str,
        *,
        configured_secrets: Iterable[str] = (),
    ) -> TaskState:
        state = cls(goal, "planning", current_step, expected_action)
        state.validate_privacy(configured_secrets)
        return state

    @classmethod
    def from_json(cls, raw: object) -> TaskState:
        if not isinstance(raw, Mapping):
            raise TaskStateError("task state не похож на object")
        version = raw.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise TaskStateError("task version должен быть integer")
        phase = raw.get("phase")
        return cls(
            version=version,
            goal=raw.get("goal"),  # type: ignore[arg-type]
            phase=phase,  # type: ignore[arg-type]
            current_step=raw.get("current_step"),  # type: ignore[arg-type]
            expected_action=raw.get("expected_action"),  # type: ignore[arg-type]
            plan_approved=raw.get("plan_approved", False),  # type: ignore[arg-type]
            paused=raw.get("paused"),  # type: ignore[arg-type]
            pause_reason=raw.get("pause_reason"),  # type: ignore[arg-type]
            result=raw.get("result"),  # type: ignore[arg-type]
        )

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "goal": self.goal,
            "phase": self.phase,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
            "plan_approved": self.plan_approved,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "result": self.result,
        }

    def validate_privacy(self, configured_secrets: Iterable[str] = ()) -> None:
        validate_private_text(
            "\n".join(
                value
                for value in (
                    self.goal,
                    self.current_step,
                    self.expected_action,
                    self.pause_reason,
                    self.result,
                )
                if value
            ),
            configured_secrets,
        )

    def update(
        self,
        current_step: str,
        expected_action: str,
        *,
        configured_secrets: Iterable[str] = (),
    ) -> TaskState:
        self._require_mutable()
        state = replace(
            self,
            current_step=current_step,
            expected_action=expected_action,
            plan_approved=False if self.phase == "planning" else self.plan_approved,
        )
        state.validate_privacy(configured_secrets)
        return state

    def approve(self, *, configured_secrets: Iterable[str] = ()) -> TaskState:
        self._require_mutable()
        if self.phase != "planning":
            raise TaskStateError("Plan можно approve только в planning")
        if self.plan_approved:
            raise TaskStateError("Plan уже approved")
        state = replace(self, plan_approved=True)
        state.validate_privacy(configured_secrets)
        return state

    def advance(
        self,
        phase: str,
        current_step: str,
        expected_action: str,
        *,
        configured_secrets: Iterable[str] = (),
    ) -> TaskState:
        self._require_mutable()
        allowed = {
            "planning": {"execution"},
            "execution": {"validation"},
            "validation": {"execution"},
        }
        if phase not in allowed.get(self.phase, set()):
            raise TaskStateError(f"transition {self.phase} → {phase} запрещён")
        if self.phase == "planning" and not self.plan_approved:
            raise TaskStateError("Для transition в execution нужен /task approve")
        state = replace(
            self,
            phase=phase,  # type: ignore[arg-type]
            current_step=current_step,
            expected_action=expected_action,
        )
        state.validate_privacy(configured_secrets)
        return state

    def pause(
        self, reason: str | None = None, *, configured_secrets: Iterable[str] = ()
    ) -> TaskState:
        self._require_active()
        if self.paused:
            raise TaskStateError("Task уже paused")
        state = replace(self, paused=True, pause_reason=reason.strip() if reason else None)
        state.validate_privacy(configured_secrets)
        return state

    def resume(self, *, configured_secrets: Iterable[str] = ()) -> TaskState:
        self._require_active()
        if not self.paused:
            raise TaskStateError("Task не paused")
        state = replace(self, paused=False, pause_reason=None)
        state.validate_privacy(configured_secrets)
        return state

    def complete(self, result: str, *, configured_secrets: Iterable[str] = ()) -> TaskState:
        self._require_mutable()
        if self.phase != "validation":
            raise TaskStateError("Task можно завершить только из validation")
        state = replace(
            self,
            phase="done",
            current_step="Task completed",
            expected_action="none",
            paused=False,
            pause_reason=None,
            result=result,
        )
        state.validate_privacy(configured_secrets)
        return state

    def _require_active(self) -> None:
        if self.phase == "done":
            raise TaskStateError("Done task terminal; сначала /task clear")

    def _require_mutable(self) -> None:
        self._require_active()
        if self.paused:
            raise TaskStateError("Task paused; разрешены только /task resume и /task clear")


def task_messages(task: TaskState | None) -> list[Message]:
    """Render an active task as a protected counted pseudo exchange."""
    if task is None or task.paused or task.phase == "done":
        return []
    recommendations = {
        "planning": (
            "/task update … или /task approve …"
            if not task.plan_approved
            else "/task update … или /task advance execution …"
        ),
        "execution": "/task update … или /task advance validation …",
        "validation": "/task update …, /task advance execution … или /task complete …",
    }
    return [
        {
            "role": "user",
            "content": (
                "FORMAL TASK STATE (учитывай как context):\n"
                f"goal: {task.goal}\nphase: {task.phase}\n"
                f"current step: {task.current_step}\n"
                f"expected action: {task.expected_action}"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Task context принят. Session invariants при конфликте приоритетнее; "
                "Task State при конфликте приоритетнее Current user request. "
                "Ответ не меняет formal state. Если step действительно завершён, можно "
                f"дать одну concise recommendation: {recommendations[task.phase]}"
            ),
        },
    ]


def is_task_message(message: Mapping[str, object], task: TaskState | None) -> bool:
    """Match only a complete generated pseudo-message, never a text prefix."""
    return any(dict(message) == expected for expected in task_messages(task))


__all__ = [
    "TASK_PHASES",
    "TASK_STATE_KEY",
    "TASK_VERSION",
    "TaskState",
    "TaskStateError",
    "is_task_message",
    "task_messages",
    "validate_private_text",
]
