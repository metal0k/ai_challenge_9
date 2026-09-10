"""History compaction: what goes into the summary, what stays, who decides.

Pure functions over message lists — no I/O, no network, no state between
calls. The actual call to the summarizer model lives in the agent
(advent_core/agent.py): this module only holds the rules it decides by, and
the text it has to send.
"""

from __future__ import annotations

from collections.abc import Sequence

from advent_core.chat import Message

# Stub reply for the substituted summary. A "message + reply" pair is needed
# because the summary enters the history as a user message, and history must
# keep alternating roles.
SUMMARY_ACK = "Принято."

# Lead-in for the substituted message. The model must understand it's reading
# a summary, not a user turn — otherwise it answers the summary as if it were
# a question.
SUMMARY_PREFIX = "Краткое содержание предыдущей части разговора:"

# Role labels in the text sent to the summarizer. In Russian and as words,
# not "user"/"assistant": the model writes the summary and reads the turns as
# prose, not as markup.
ROLE_LABELS = {"user": "Пользователь", "assistant": "Ассистент"}

# Lead-in before the previous summary. Must sit BEFORE it, not between it and
# the turns: a header sandwiched between two blocks reads as a caption for
# the following one, and the model mistakes the old summary for part of the
# new conversation.
FOLD_HEADER = (
    "Ниже — пересказ более ранней части разговора, составленный прежде. "
    "Его факты обязаны войти в новый пересказ целиком."
)

# Lead-in before the turns. The words "question" and "answer" are avoided on
# purpose: the text is spliced with the turns' own content, and a stray match
# would throw off both reading and checks.
TAIL_HEADER = "Продолжение разговора, которое нужно пересказать:"


def split_history(
    history: Sequence[Message], keep_last: int
) -> tuple[list[Message], list[Message]]:
    """Split history into (older, tail): older goes into the summary, tail stays.

    The boundary is aligned so tail starts on a user turn. Not cosmetic: a
    summary pseudo-pair ending in "Принято." precedes tail, and an assistant
    turn right after it would be an answer without a question. The exact
    tokenizer counts such a history silently (PROBE-w02d09-compact.md §1: 165
    tokens, not a single complaint), so it won't catch the error — it must
    never happen in the first place.

    When the boundary lands on an assistant turn, tail takes ONE MESSAGE
    MORE: an extra turn in context costs tokens, a lost one costs meaning.

    Returns new lists: the input history is not mutated and stays unlinked
    from the result — editing one must not change the other.
    """
    if keep_last <= 0:
        # No tail needed: the whole conversation goes into the summary.
        return list(history), []
    if keep_last >= len(history):
        # Nothing to compact — history is shorter than what's kept anyway.
        return [], list(history)

    cut = len(history) - keep_last
    if history[cut]["role"] != "user":
        cut -= 1

    if cut <= 0:
        # The shift ate everything that could be compacted. A separate branch,
        # not a negative slice: history[:-1] would mean something else entirely.
        return [], list(history)

    return list(history[:cut]), list(history[cut:])


def should_compact(older: Sequence[Message], *, over_budget: bool, compact_every: int) -> bool:
    """Time to compact? Two triggers, and the rule between them lives only here.

    `over_budget` — the request doesn't fit the model's window, i.e. plain
    trimming would have fired. `compact_every` — threshold on the count of
    not-yet-compacted old messages: a scheduled trigger that keeps behavior
    predictable instead of waiting for overflow.

    Empty `older` is always False, even with over_budget: nothing to compact,
    and calling the model for an empty summary would just waste tokens.
    """
    if not older:
        return False
    return len(older) >= compact_every or over_budget


def fold_summary(previous: str | None, messages: Sequence[Message]) -> str:
    """Text of the summarizer's task: the previous summary plus new turns.

    The previous summary is always included — it covers part of the
    conversation that exists nowhere else in the working context. Without it,
    a second compaction round would lose it exactly like plain trimming
    would, defeating the whole point of summarizing on a long conversation.
    """
    parts: list[str] = []
    if previous and previous.strip():
        parts.append(FOLD_HEADER)
        parts.append(previous.strip())
    parts.append(TAIL_HEADER)
    for message in messages:
        label = ROLE_LABELS.get(message["role"], message["role"])
        parts.append(f"{label}: {message['content']}")
    return "\n\n".join(parts)


def summary_messages(summary: str | None) -> list[Message]:
    """Pseudo-pair to prepend to the history. Empty — nothing.

    A pair made of an empty summary and "Принято." would cost tokens for
    nothing, while looking like real context in the history.
    """
    if not summary or not summary.strip():
        return []
    return [
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n\n{summary.strip()}"},
        {"role": "assistant", "content": SUMMARY_ACK},
    ]


__all__ = [
    "FOLD_HEADER",
    "ROLE_LABELS",
    "SUMMARY_ACK",
    "SUMMARY_PREFIX",
    "TAIL_HEADER",
    "fold_summary",
    "should_compact",
    "split_history",
    "summary_messages",
]
