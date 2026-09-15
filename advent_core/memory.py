"""Explicit short-term, working and long-term memory for the agent.

The module deliberately contains no model calls.  It provides the value
objects, routing validation and the small JSON store used by the Week 03
agent.  Keeping these operations pure makes the privacy boundary testable:
only text with exact user evidence can be promoted to structured memory.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NamedTuple

from advent_core.chat import Message

MemoryLayer = Literal["short_term", "working", "long_term"]
WORKING_FIELDS = ("goal", "constraints", "decisions", "open_items")
LONG_TERM_FIELDS = ("profile", "preferences", "knowledge")
MEMORY_VERSION = 1
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*\.[^\s.].*$")
_CREDENTIAL_RE = re.compile(
    r"(?:password|passwd|token|api[_ -]?key|secret|private[_ -]?key|recovery[_ -]?code|"
    r"credential|card[_ -]?(?:number|cvv)|ssn|паспорт|парол|токен|секрет|ключ)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StructuredMemory:
    entries: dict[str, str] = field(default_factory=dict)
    pinned: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        entries = dict(self.entries)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "pinned", frozenset(self.pinned) & entries.keys())


@dataclass(frozen=True, slots=True)
class ShortTermMemory:
    messages: tuple[Message, ...] = ()
    from_turn: int = 0
    through_turn: int = 0


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)
    working: StructuredMemory = field(default_factory=StructuredMemory)
    long_term: StructuredMemory = field(default_factory=StructuredMemory)


@dataclass(frozen=True, slots=True)
class MemoryOperation:
    action: Literal["set", "delete"]
    field: str
    key: str
    value: str | None = None
    evidence: str = ""

    @property
    def canonical_key(self) -> str:
        return f"{self.field}.{self.key}"


@dataclass(frozen=True, slots=True)
class MemoryDelta:
    working: tuple[MemoryOperation, ...] = ()
    long_term: tuple[MemoryOperation, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryUpdate:
    snapshot: MemorySnapshot
    delta: MemoryDelta
    applied: tuple[tuple[MemoryLayer, MemoryOperation], ...] = ()
    blocked: tuple[tuple[MemoryLayer, MemoryOperation, str], ...] = ()
    rejected: tuple[tuple[MemoryLayer, MemoryOperation, str], ...] = ()
    memory_upto: int = 0
    call_result: object | None = None
    # Rendering needs the prior state to distinguish an added key from an
    # updated one. Keep it out of equality/repr for positional compatibility.
    previous: MemorySnapshot | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class MemoryFailure:
    call_result: object | None
    reason: Literal["invalid", "truncated", "request_failed"]
    detail: str = ""


class LoadResult(NamedTuple):
    value: StructuredMemory
    warnings: tuple[str, ...] = ()
    upto: int = 0
    exists: bool = False


def fields_for(layer: MemoryLayer) -> tuple[str, ...]:
    if layer == "working":
        return WORKING_FIELDS
    if layer == "long_term":
        return LONG_TERM_FIELDS
    return ()


def validate_key(layer: MemoryLayer, field: str, key: str) -> str:
    """Return canonical ``field.key`` or raise ``ValueError``."""
    if not isinstance(field, str) or not isinstance(key, str):
        raise ValueError("memory field and key must be strings")
    field, key = str(field).strip(), str(key).strip()
    if layer not in ("working", "long_term"):
        raise ValueError("structured operations cannot target short_term")
    if field not in fields_for(layer):
        raise ValueError(f"field {field!r} is not allowed for {layer}")
    if not key or any(ch.isspace() for ch in key) or key.startswith("."):
        raise ValueError("memory key must be a non-empty token without whitespace")
    return f"{field}.{key}"


def split_key(key: str) -> tuple[str, str]:
    if not isinstance(key, str):
        raise ValueError("memory key must be a string")
    field, sep, name = str(key).partition(".")
    if not sep or not name:
        raise ValueError("memory key must have the form <field>.<name>")
    return field, name


def is_credential_like(key: str, value: str | None = None) -> bool:
    return bool(_CREDENTIAL_RE.search(f"{key} {value or ''}"))


def _normalise_operation(layer: MemoryLayer, op: MemoryOperation) -> MemoryOperation:
    if not isinstance(op, MemoryOperation):
        raise ValueError("memory operation must be a MemoryOperation")
    if not isinstance(op.action, str):
        raise ValueError("memory operation action must be a string")
    if not isinstance(op.field, str) or not isinstance(op.key, str):
        raise ValueError("memory operation field and key must be strings")
    field, key = op.field, op.key
    validate_key(layer, field, key)
    action = op.action
    if action not in ("set", "delete"):
        raise ValueError(f"unknown memory operation: {action!r}")
    if not isinstance(op.evidence, str):
        raise ValueError("memory operation evidence must be a string")
    evidence = op.evidence.strip()
    if not evidence:
        raise ValueError("memory operation evidence cannot be empty")
    if action == "set" and not isinstance(op.value, str):
        raise ValueError("memory value must be a string")
    value = None if action == "delete" else op.value.strip()
    if action == "set" and not value:
        raise ValueError("memory value cannot be empty")
    return MemoryOperation(action, field, key, value, evidence)


def _evidence_ok(op: MemoryOperation, user_messages: tuple[Message, ...]) -> bool:
    return any(
        isinstance(message, Mapping)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and op.evidence in message["content"]
        for message in user_messages
    )


def apply_delta(
    snapshot: MemorySnapshot,
    delta: MemoryDelta,
    *,
    user_messages: tuple[Message, ...] = (),
    memory_upto: int | None = None,
    call_result: object | None = None,
    require_evidence: bool = True,
) -> MemoryUpdate:
    """Validate and apply a delta without partially applying invalid sections.

    Invalid operations reject the complete delta with ``ValueError``.  This
    keeps the extractor cursor unchanged: callers only advance it after this
    function returns successfully.  A pinned automatic change is different;
    it is ``blocked`` while unrelated valid operations continue.
    """
    applied: list[tuple[MemoryLayer, MemoryOperation]] = []
    blocked: list[tuple[MemoryLayer, MemoryOperation, str]] = []
    rejected: list[tuple[MemoryLayer, MemoryOperation, str]] = []
    result = {
        "working": dict(snapshot.working.entries),
        "long_term": dict(snapshot.long_term.entries),
    }
    pins = {"working": set(snapshot.working.pinned), "long_term": set(snapshot.long_term.pinned)}
    normalised_sections: dict[str, list[MemoryOperation]] = {"working": [], "long_term": []}

    for layer, operations in (("working", delta.working), ("long_term", delta.long_term)):
        if isinstance(operations, (str, bytes)) or not isinstance(operations, Iterable):
            raise ValueError(f"{layer} operations must be a sequence")
        normalised: list[MemoryOperation] = []
        seen: set[str] = set()
        for raw in operations:
            try:
                op = _normalise_operation(layer, raw)
                if op.canonical_key in seen:
                    raise ValueError("duplicate/conflicting operation")
                seen.add(op.canonical_key)
                if require_evidence and not _evidence_ok(op, user_messages):
                    raise ValueError("evidence is not an exact substring of a user message")
                if layer == "long_term" and is_credential_like(op.canonical_key, op.value):
                    raise ValueError("credential-like data cannot enter long_term memory")
                normalised.append(op)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid {layer} memory operation: {exc}") from exc
        normalised_sections[layer] = normalised

    # Both sections have now passed validation.  Only this second phase is
    # allowed to touch the candidate dictionaries, so one bad section can
    # never partially apply another one.
    for layer in ("working", "long_term"):
        normalised = normalised_sections[layer]
        for op in normalised:
            canonical = op.canonical_key
            if canonical in pins[layer]:
                blocked.append((layer, op, "pinned"))
                continue
            if op.action == "set":
                result[layer][canonical] = op.value or ""
            else:
                result[layer].pop(canonical, None)
                pins[layer].discard(canonical)
            applied.append((layer, op))

    working = StructuredMemory(result["working"], frozenset(pins["working"]))
    long_term = StructuredMemory(result["long_term"], frozenset(pins["long_term"]))
    return MemoryUpdate(
        MemorySnapshot(snapshot.short_term, working, long_term),
        delta,
        tuple(applied),
        tuple(blocked),
        tuple(rejected),
        snapshot.short_term.through_turn if memory_upto is None else memory_upto,
        call_result,
        snapshot,
    )


def manual_set(
    snapshot: MemorySnapshot, layer: MemoryLayer, key: str, value: str
) -> MemorySnapshot:
    field, name = split_key(key)
    canonical = validate_key(layer, field, name)
    if not value.strip():
        raise ValueError("memory value cannot be empty")
    if layer == "working":
        entries, pins = dict(snapshot.working.entries), set(snapshot.working.pinned)
        entries[canonical], _ = value.strip(), pins.add(canonical)
        return MemorySnapshot(
            snapshot.short_term, StructuredMemory(entries, frozenset(pins)), snapshot.long_term
        )
    entries, pins = dict(snapshot.long_term.entries), set(snapshot.long_term.pinned)
    entries[canonical], _ = value.strip(), pins.add(canonical)
    return MemorySnapshot(
        snapshot.short_term, snapshot.working, StructuredMemory(entries, frozenset(pins))
    )


def render_memory(snapshot: MemorySnapshot, *, layer: str | None = None) -> str:
    """Stable, compact human-facing representation used by `/memory`."""
    rows: list[str] = []
    names = ("short_term", "working", "long_term") if layer is None else (layer,)
    for name in names:
        if name == "short_term":
            rows.append(f"short_term: {len(snapshot.short_term.messages)} messages")
            continue
        value = getattr(snapshot, name)
        rows.append(f"{name}: {len(value.entries)} entries, {len(value.pinned)} pinned")
        for key in sorted(value.entries):
            mark = "*" if key in value.pinned else " "
            rows.append(f"  {mark}{key} = {value.entries[key]}")
    return "\n".join(rows)


def memory_messages(snapshot: MemorySnapshot, *, summary: str | None = None) -> list[Message]:
    """Build protected pseudo-pairs in the required request order."""
    messages: list[Message] = []
    for title, memory in (("long-term", snapshot.long_term), ("working", snapshot.working)):
        if memory.entries:
            body = "\n".join(f"{k}: {v}" for k, v in sorted(memory.entries.items()))
            messages.extend(
                (
                    {"role": "user", "content": f"[{title} memory]\n{body}"},
                    {"role": "assistant", "content": f"{title} memory loaded."},
                )
            )
    if summary:
        messages.extend(
            (
                {"role": "user", "content": "[summary]\n" + summary},
                {"role": "assistant", "content": "Summary loaded."},
            )
        )
    messages.extend(snapshot.short_term.messages)
    return messages


def render_delta(update: MemoryUpdate) -> str | None:
    parts: dict[str, list[str]] = {"working": [], "long-term": []}
    previous = update.previous
    for layer, op in update.applied:
        name = "long-term" if layer == "long_term" else layer
        if op.action == "delete":
            marker = "−"
        elif previous is None:
            marker = "~"
        else:
            prior = previous.long_term if layer == "long_term" else previous.working
            marker = "~" if op.canonical_key in prior.entries else "+"
        parts[name].append(marker + op.canonical_key)
    for layer, op, _ in update.blocked:
        parts["long-term" if layer == "long_term" else layer].append("!" + op.canonical_key)
    for layer, op, _ in update.rejected:
        parts["long-term" if layer == "long_term" else layer].append("×" + op.canonical_key)
    shown = [f"{name} {', '.join(values)}" for name, values in parts.items() if values]
    return "memory: " + "; ".join(shown) if shown else None


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(temp_name)


@dataclass(slots=True)
class MemoryStore:
    root: Path

    @property
    def working_root(self) -> Path:
        return self.root / "memory" / "working"

    @property
    def long_term_path(self) -> Path:
        return self.root / "memory" / "long_term.json"

    def working_path(self, session_name: str) -> Path:
        from advent_core.session import validate_name

        return self.working_root / f"{validate_name(session_name)}.json"

    @staticmethod
    def _validate_value(layer: MemoryLayer, value: StructuredMemory) -> None:
        if not isinstance(value, StructuredMemory):
            raise ValueError("memory value must be a StructuredMemory")
        for raw_key, raw_value in value.entries.items():
            if not isinstance(raw_key, str) or not isinstance(raw_value, str):
                raise ValueError("memory keys and values must be strings")
            field, name = split_key(raw_key)
            if validate_key(layer, field, name) != raw_key:
                raise ValueError("memory keys must be canonical")
            if not raw_value.strip():
                raise ValueError("memory values must be non-empty strings")
        for pinned in value.pinned:
            if not isinstance(pinned, str) or pinned not in value.entries:
                raise ValueError("pinned keys must exist in entries")

    @staticmethod
    def _fallback_upto(turns_count: int | None) -> int:
        return (
            turns_count
            if isinstance(turns_count, int)
            and not isinstance(turns_count, bool)
            and turns_count >= 0
            else 0
        )

    @staticmethod
    def _load(
        path: Path,
        layer: MemoryLayer,
        *,
        session: str | None = None,
        turns_count: int | None = None,
    ) -> LoadResult:
        if not path.is_file():
            return LoadResult(StructuredMemory(), (), 0, False)
        warnings: list[str] = []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("root must be an object")
            version = raw.get("version")
            if (
                isinstance(version, bool)
                or not isinstance(version, int)
                or version != MEMORY_VERSION
            ):
                raise ValueError("wrong version/layer")
            if raw.get("layer") != layer:
                raise ValueError("wrong version/layer")
            if layer == "working" and raw.get("session") != session:
                raise ValueError("session mismatch")
            raw_entries = raw.get("entries")
            raw_pinned = raw.get("pinned")
            if not isinstance(raw_entries, dict):
                raise ValueError("entries must be an object")
            if not isinstance(raw_pinned, list):
                raise ValueError("pinned must be a list")
            entries: dict[str, str] = {}
            for key, value in raw_entries.items():
                try:
                    if not isinstance(key, str):
                        raise ValueError("key is not a string")
                    field, name = split_key(key)
                    canonical = validate_key(layer, field, name)
                    if canonical != key:
                        raise ValueError("key is not canonical")
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("value is not a non-empty string")
                    entries[canonical] = value
                except (TypeError, ValueError) as exc:
                    warnings.append(f"{path}: ignored {key!r}: {exc}")
            pinned_values: set[str] = set()
            for raw_key in raw_pinned:
                try:
                    if not isinstance(raw_key, str):
                        raise ValueError("pinned key is not a string")
                    field, name = split_key(raw_key)
                    canonical = validate_key(layer, field, name)
                    if canonical != raw_key:
                        raise ValueError("pinned key is not canonical")
                    if canonical not in entries:
                        raise ValueError("pinned key is missing from entries")
                    pinned_values.add(canonical)
                except (TypeError, ValueError) as exc:
                    warnings.append(f"{path}: ignored pinned key {raw_key!r}: {exc}")
            pinned = frozenset(pinned_values)
            upto = 0
            if layer == "working":
                raw_upto = raw.get("upto")
                max_upto = (
                    turns_count
                    if isinstance(turns_count, int) and not isinstance(turns_count, bool)
                    else None
                )
                valid_upto = (
                    isinstance(raw_upto, int)
                    and not isinstance(raw_upto, bool)
                    and raw_upto >= 0
                    and (max_upto is None or raw_upto <= max_upto)
                )
                if valid_upto:
                    upto = raw_upto
                else:
                    upto = MemoryStore._fallback_upto(turns_count)
                    warnings.append(f"{path}: invalid upto; using {upto}")
            return LoadResult(StructuredMemory(entries, pinned), tuple(warnings), upto, True)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return LoadResult(
                StructuredMemory(), (f"{path}: ignored corrupted memory ({exc})",), 0, True
            )

    def load_working(self, session_name: str, turns_count: int | None = None) -> LoadResult:
        return self._load(
            self.working_path(session_name),
            "working",
            session=session_name,
            turns_count=turns_count,
        )

    def load_long_term(self) -> LoadResult:
        return self._load(self.long_term_path, "long_term")

    def save_working(self, session_name: str, value: StructuredMemory, upto: int) -> Path:
        self._validate_value("working", value)
        from advent_core.session import validate_name

        safe_session = validate_name(session_name)
        if not isinstance(upto, int) or isinstance(upto, bool) or upto < 0:
            raise ValueError("working cursor must be a non-negative integer")
        path = self.working_path(safe_session)
        payload = {
            "version": MEMORY_VERSION,
            "layer": "working",
            "session": safe_session,
            "updated": datetime.now(UTC).isoformat(),
            "entries": dict(sorted(value.entries.items())),
            "pinned": sorted(value.pinned),
            "upto": upto,
        }
        _atomic_json(path, payload)
        return path

    def save_long_term(self, value: StructuredMemory) -> Path:
        self._validate_value("long_term", value)
        path = self.long_term_path
        payload = {
            "version": MEMORY_VERSION,
            "layer": "long_term",
            "updated": datetime.now(UTC).isoformat(),
            "entries": dict(sorted(value.entries.items())),
            "pinned": sorted(value.pinned),
        }
        _atomic_json(path, payload)
        return path

    def copy_working(self, source: str, target: str) -> Path:
        loaded = self.load_working(source)
        return self.save_working(target, loaded.value, loaded.upto)


__all__ = [
    "LONG_TERM_FIELDS",
    "MEMORY_VERSION",
    "MemoryDelta",
    "MemoryFailure",
    "MemoryLayer",
    "MemoryOperation",
    "MemorySnapshot",
    "MemoryStore",
    "MemoryUpdate",
    "ShortTermMemory",
    "StructuredMemory",
    "WORKING_FIELDS",
    "apply_delta",
    "fields_for",
    "is_credential_like",
    "manual_set",
    "memory_messages",
    "render_delta",
    "render_memory",
    "split_key",
    "validate_key",
]
