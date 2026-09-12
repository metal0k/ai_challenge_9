"""Sticky facts: fixed categories, a delta applied over a dict, stable rendering.

Pure functions over a facts dict — no I/O, no network, no state between
calls. The extractor call itself (schema, prompt, model) lives in the agent
(advent_core/agent.py), mirroring how advent_core/compact.py holds only the
rules the summarizer call decides by.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from advent_core.chat import Message

# Fixed set — an extractor key outside these lands in DeltaResult.rejected,
# never silently in facts. Declaration order is the category order used by
# format_facts/render_note, so the block's token size stays comparable
# between runs.
FACT_CATEGORIES = ("цель", "ограничения", "предпочтения", "решения", "договорённости")

# Same pattern as compact.SUMMARY_ACK/SUMMARY_PREFIX, kept local: facts.py has
# no dependency on compact.py, and the two blocks differ in wording on purpose
# (one is a summary, the other a fact sheet).
FACTS_PREFIX = "Важные факты из нашего разговора:"
FACTS_ACK = "Принято."

# U+2212 MINUS SIGN, not a hyphen — matches the SPEC's own example literally
# ("facts: +бюджет, ~срок, −платформа"), verified byte-for-byte against it.
_REMOVED_MARK = "−"
_PIN_MARK = " \U0001f4cc"  # 📌


@dataclass(frozen=True, slots=True)
class DeltaResult:
    """Outcome of applying one delta over a facts dict."""

    facts: dict[str, str]
    added: tuple[str, ...]
    updated: tuple[str, ...]
    removed: tuple[str, ...]
    blocked: tuple[str, ...]  # operations skipped: key is pinned
    rejected: tuple[str, ...]  # operations skipped: key not in FACT_CATEGORIES


def validate_key(key: str) -> str:
    """Checks "категория.имя" form and a known category. Raises ValueError otherwise."""
    key = key.strip()
    category, sep, name = key.partition(".")
    if not sep or not category or not name:
        raise ValueError(f'ключ факта должен быть вида "категория.имя", получено {key!r}')
    if category not in FACT_CATEGORIES:
        allowed = ", ".join(FACT_CATEGORIES)
        raise ValueError(
            f"неизвестная категория {category!r} в ключе {key!r}, ожидалось: {allowed}"
        )
    return key


def _sort_key(key: str) -> tuple[int, str]:
    category, _, name = key.partition(".")
    try:
        index = FACT_CATEGORIES.index(category)
    except ValueError:
        index = len(FACT_CATEGORIES)  # defensive: only validated keys should reach here
    return (index, name)


def apply_delta(facts: dict[str, str], pinned: Iterable[str], delta: dict) -> DeltaResult:
    """Applies {"set": [{"key", "value"}], "delete": [...]} over facts.

    Unmentioned keys are never touched. A pinned key's set/delete is skipped
    and reported in `blocked`, not applied and not silently dropped. A key
    outside FACT_CATEGORIES is skipped and reported in `rejected`.
    """
    pinned_set = frozenset(pinned)
    result = dict(facts)
    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    blocked: list[str] = []
    rejected: list[str] = []

    for item in delta.get("set") or []:
        # The schema constrains the shape, the model still writes it: a reply
        # of {"set": ["цель.имя"]} is valid JSON and used to die on .get() with
        # an AttributeError, which no caller catches — a traceback instead of
        # the turn. A malformed item is rejected like a bad key: visible in the
        # note, never fatal.
        if not isinstance(item, dict):
            rejected.append(str(item))
            continue
        raw_key = item.get("key", "")
        value = item.get("value", "")
        if not isinstance(raw_key, str):
            rejected.append(str(raw_key))
            continue
        if not isinstance(value, str):
            value = str(value)
        try:
            key = validate_key(raw_key)
        except ValueError:
            rejected.append(raw_key)
            continue
        if key in pinned_set:
            blocked.append(key)
            continue
        if key in result:
            if result[key] != value:
                result[key] = value
                updated.append(key)
            # same value again — not a change, not reported
        else:
            result[key] = value
            added.append(key)

    for raw_key in delta.get("delete") or []:
        if not isinstance(raw_key, str):
            rejected.append(str(raw_key))
            continue
        try:
            key = validate_key(raw_key)
        except ValueError:
            rejected.append(raw_key)
            continue
        if key in pinned_set:
            blocked.append(key)
            continue
        if key in result:
            del result[key]
            removed.append(key)
        # not present — nothing to delete, not an error

    return DeltaResult(
        facts=result,
        added=tuple(added),
        updated=tuple(updated),
        removed=tuple(removed),
        blocked=tuple(blocked),
        rejected=tuple(rejected),
    )


def format_facts(facts: dict[str, str], pinned: Iterable[str]) -> str:
    """Renders the facts block: category headers, alphabetical inside, pin marks.

    Used both for the pseudo-pair sent to the model and for `/facts` — the
    same ordering keeps the block's token size comparable between runs.
    """
    if not facts:
        return ""
    pinned_set = frozenset(pinned)
    lines: list[str] = []
    current_category: str | None = None
    for key in sorted(facts, key=_sort_key):
        category, _, name = key.partition(".")
        if category != current_category:
            lines.append(f"{category}:")
            current_category = category
        mark = _PIN_MARK if key in pinned_set else ""
        lines.append(f"  {name}: {facts[key]}{mark}")
    return "\n".join(lines)


def facts_messages(facts: dict[str, str], pinned: Iterable[str]) -> list[Message]:
    """Pseudo-pair to prepend to the history. Empty facts — nothing.

    Content derived from the dialog must not carry a system prompt's
    authority — a user message the model can disagree with, not a system
    instruction it treats as ground truth (SPEC-w02d10.md §6).
    """
    block = format_facts(facts, pinned)
    if not block:
        return []
    return [
        {"role": "user", "content": f"{FACTS_PREFIX}\n\n{block}"},
        {"role": "assistant", "content": FACTS_ACK},
    ]


def render_note(before: dict[str, str], after: dict[str, str]) -> str | None:
    """One-line diff for the per-turn note, e.g. "facts: +бюджет, ~срок, −платформа".

    None when nothing changed — the caller must not print a note every turn.
    """
    parts: list[str] = []
    for key in sorted(set(before) | set(after), key=_sort_key):
        _, _, name = key.partition(".")
        if key not in before:
            parts.append(f"+{name}")
        elif key not in after:
            parts.append(f"{_REMOVED_MARK}{name}")
        elif before[key] != after[key]:
            parts.append(f"~{name}")
    if not parts:
        return None
    return "facts: " + ", ".join(parts)


__all__ = [
    "FACT_CATEGORIES",
    "FACTS_ACK",
    "FACTS_PREFIX",
    "DeltaResult",
    "apply_delta",
    "facts_messages",
    "format_facts",
    "render_note",
    "validate_key",
]
