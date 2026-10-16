"""Atomic `state.json`: stage statuses, input hashes and dependent invalidation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from adventship.ctx import EXIT_STAGE_FAILED, ShipError

STATE_VERSION = 1
STAGES = [
    "intake",
    "checks",
    "describe",
    "review",
    "commit",
    "tag",
    "record",
    "submit",
    "sheet",
    "report",
]
STATUSES = ("pending", "done", "failed", "skipped")

# Stage -> stages that read its output; changing it invalidates all of them.
DEPENDENTS: dict[str, list[str]] = {
    "intake": STAGES[1:],
    "checks": [],
    "describe": ["commit", "tag", "record", "submit", "sheet", "report"],
    "review": [],
    "commit": ["tag", "record", "submit", "sheet", "report"],
    "tag": ["record", "submit", "sheet", "report"],
    "record": ["submit", "sheet", "report"],
    "submit": ["sheet", "report"],
    "sheet": ["report"],
    "report": [],
}
# Pre-commit stages that read the working tree / commit range: their inputs include the code.
FINGERPRINTED = ("checks", "describe", "review")
UPSTREAM: dict[str, list[str]] = {
    stage: [other for other in STAGES if stage in DEPENDENTS[other]] for stage in STAGES
}

_READ_RETRIES = 5
_RETRY_DELAY = 0.05


def _retry(action):
    """Windows: a read/replace racing another process's os.replace raises PermissionError."""
    for attempt in range(_READ_RETRIES):
        try:
            return action()
        except PermissionError:
            if attempt == _READ_RETRIES - 1:
                raise
            time.sleep(_RETRY_DELAY)


def data_digest(data: dict[str, Any]) -> str:
    """Digest of public stage data; keys starting with `_` are volatile and ignored."""
    public = {k: v for k, v in data.items() if not k.startswith("_")}
    blob = json.dumps(public, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ShipState:
    def __init__(self, path: Path, doc: dict[str, Any] | None = None) -> None:
        self.path = path
        # stage -> code fingerprint mixed into inputs_hash; set by the runner per run.
        self.extra_inputs: dict[str, str] = {}
        self.doc: dict[str, Any] = doc or {
            "version": STATE_VERSION,
            "task_hash": None,
            "stages": {},
        }

    @classmethod
    def load(cls, path: Path) -> ShipState:
        try:
            raw = _retry(lambda: path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(path)
        try:
            doc = json.loads(raw)
            if (
                not isinstance(doc, dict)
                or doc.get("version") != STATE_VERSION
                or not isinstance(doc.get("stages"), dict)
            ):
                raise ValueError("неожиданная структура или версия")
        except ValueError as error:
            backup = path.with_name(path.name + ".bak")
            _retry(lambda: os.replace(path, backup))
            raise ShipError(
                f"state.json повреждён ({error}); сохранён как {backup.name}",
                exit_code=EXIT_STAGE_FAILED,
                hint="Удали .bak и запусти заново — состояние начнётся с нуля.",
            ) from None
        return cls(path, doc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(self.doc, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            _retry(lambda: os.replace(tmp, self.path))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    # --- reading -----------------------------------------------------------

    @property
    def task_hash(self) -> str | None:
        return self.doc.get("task_hash")

    def entry(self, stage: str) -> dict[str, Any] | None:
        return self.doc["stages"].get(stage)

    def status(self, stage: str) -> str:
        entry = self.entry(stage)
        return entry["status"] if entry else "pending"

    def data(self, stage: str) -> dict[str, Any]:
        entry = self.entry(stage)
        return dict(entry.get("data", {})) if entry else {}

    def inputs_hash(self, stage: str) -> str:
        """Hash of what the stage reads: task hash plus the data of every upstream stage."""
        upstream = [
            (name, data_digest(self.data(name)))
            for name in UPSTREAM[stage]
            if self.entry(name) is not None
        ]
        parts: list[Any] = [self.task_hash, stage, upstream]
        if stage in self.extra_inputs:
            parts.append(self.extra_inputs[stage])
        blob = json.dumps(parts, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def code_changed(self, stage: str) -> bool:
        """A done stage whose recorded code fingerprint differs from the current one."""
        entry = self.entry(stage)
        return bool(
            entry
            and stage in self.extra_inputs
            and entry["status"] in ("done", "skipped")
            and entry.get("fp") != self.extra_inputs[stage]
        )

    def is_valid(self, stage: str) -> bool:
        """`done` (or `skipped`) with inputs that still match."""
        entry = self.entry(stage)
        return bool(
            entry
            and entry["status"] in ("done", "skipped")
            and entry.get("inputs_hash") == self.inputs_hash(stage)
        )

    # --- writing -----------------------------------------------------------

    def set_task_hash(self, task_hash: str | None) -> bool:
        """Store the hash; a change wipes every stage. Returns True when it changed."""
        old = self.task_hash
        changed = old is not None and task_hash is not None and old != task_hash
        if changed:
            self.doc["stages"] = {}
        self.doc["task_hash"] = task_hash
        self.save()
        return changed

    def mark(self, stage: str, status: str, *, at: str, data: dict[str, Any] | None = None) -> None:
        if stage not in STAGES or status not in STATUSES:
            raise ValueError(f"unknown stage/status: {stage}/{status}")
        new_data = dict(data or {})
        previous = self.entry(stage)
        output_changed = previous is not None and data_digest(previous.get("data", {})) != (
            data_digest(new_data)
        )
        self.doc["stages"][stage] = {
            "status": status,
            "at": at,
            "data": new_data,
            "inputs_hash": self.inputs_hash(stage),
        }
        if stage in self.extra_inputs:
            self.doc["stages"][stage]["fp"] = self.extra_inputs[stage]
        if output_changed:
            for dependent in DEPENDENTS[stage]:
                self.doc["stages"].pop(dependent, None)
        self.save()

    def invalidate(self, stage: str, *, include_self: bool = True) -> list[str]:
        """Drop the stage and its dependents; returns the names actually removed."""
        names = ([stage] if include_self else []) + DEPENDENTS[stage]
        removed = [n for n in names if self.doc["stages"].pop(n, None) is not None]
        self.save()
        return removed

    def invalidate_from(self, stage: str) -> list[str]:
        """`--from`: the stage and everything after it in pipeline order."""
        names = STAGES[max(STAGES.index(stage), 1) :]  # intake is always re-run anyway
        removed = [n for n in names if self.doc["stages"].pop(n, None) is not None]
        self.save()
        return removed
