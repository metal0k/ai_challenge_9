"""Pure history-compaction functions: split, trigger, summary fold.

No network here — pure functions over message lists. Tests are tabular: a
set of history shapes, each with its expected split.

The core invariant is checked separately, on several shapes: the tail left
"as is" must start with user. The exact tokenizer does NOT catch a violation
of it — measured (PROBE-w02d09-compact.md §1): a history where a second
assistant follows a fake reply was counted silently, 165 tokens. So this
test is the only guard against a broken pair.
"""

from __future__ import annotations

import pytest

from advent_core.compact import (
    FOLD_HEADER,
    SUMMARY_ACK,
    TAIL_HEADER,
    fold_summary,
    should_compact,
    split_history,
    summary_messages,
)


def u(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def a(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


# Six messages, three complete pairs — the shape of an ordinary conversation.
SIX = [u("в1"), a("о1"), u("в2"), a("о2"), u("в3"), a("о3")]


# --- split_history --------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "keep_last", "expected_older", "expected_tail"),
    [
        # Empty — nothing to split.
        ([], 4, [], []),
        # History shorter than the requested tail: nothing to compact, all stays.
        ([u("в1"), a("о1")], 6, [], [u("в1"), a("о1")]),
        # Even boundary: last two are a whole pair, starts with user.
        (SIX, 2, SIX[:4], [u("в3"), a("о3")]),
        (SIX, 4, SIX[:2], [u("в2"), a("о2"), u("в3"), a("о3")]),
        # Boundary lands on assistant: last 3 would be [о2, в3, о3], a tail
        # starting with a reply and no question. Take one message MORE.
        (SIX, 3, SIX[:2], [u("в2"), a("о2"), u("в3"), a("о3")]),
        (SIX, 5, [], SIX),
        # keep_last bigger than the history — same case as "shorter than the tail".
        (SIX, 99, [], SIX),
        # No tail needed at all: everything goes into the summary.
        (SIX, 0, SIX, []),
    ],
)
def test_split_history_table(history, keep_last, expected_older, expected_tail):
    older, tail = split_history(history, keep_last)
    assert older == expected_older
    assert tail == expected_tail


@pytest.mark.parametrize("keep_last", [0, 1, 2, 3, 4, 5, 6, 7, 20])
@pytest.mark.parametrize(
    "history",
    [
        [],
        [u("в1")],
        [u("в1"), a("о1")],
        SIX,
        # History starting with a reply: happens with a hand-edited session
        # file, and the function must not choke on it.
        [a("о0"), u("в1"), a("о1"), u("в2"), a("о2")],
        # Two questions in a row — a reply was not saved (stream cut off).
        [u("в1"), u("в2"), a("о2"), u("в3"), a("о3")],
    ],
)
def test_split_history_invariants(history, keep_last):
    older, tail = split_history(history, keep_last)

    # 1. Nothing lost, nothing reordered.
    assert older + tail == history
    # 2. Tail starts with user — but only when part of the history goes into
    #    the summary: a fake pair ending in "Принято." precedes the tail, and
    #    an assistant right after it would be a reply with no question. When
    #    older is empty there is nothing to substitute, the history stays as
    #    it was, and it may start with anything — e.g. a reply, if the
    #    session file was hand-edited.
    if older and tail:
        assert tail[0]["role"] == "user"
    # 3. Tail is not shorter than requested (the boundary shift takes MORE, not less).
    if older:
        assert len(tail) >= min(keep_last, len(history))


def test_split_history_does_not_mutate_input():
    history = list(SIX)
    split_history(history, 3)
    assert history == SIX


def test_split_history_returns_new_lists():
    older, tail = split_history(SIX, 2)
    older.append(u("чужое"))
    tail.append(u("чужое"))
    assert len(SIX) == 6


# --- should_compact -------------------------------------------------------


@pytest.mark.parametrize(
    ("older_len", "over_budget", "compact_every", "expected"),
    [
        # Nothing to compact — don't, even if the request doesn't fit.
        (0, True, 10, False),
        (0, False, 10, False),
        # Scheduled trigger: enough old messages accumulated.
        (10, False, 10, True),
        (11, False, 10, True),
        (9, False, 10, False),
        # Budget trigger: threshold not reached, but the request doesn't fit.
        (2, True, 10, True),
        (1, True, 10, True),
        # Both at once — still yes.
        (12, True, 10, True),
    ],
)
def test_should_compact_table(older_len, over_budget, compact_every, expected):
    older = [u(f"в{i}") for i in range(older_len)]
    assert should_compact(older, over_budget=over_budget, compact_every=compact_every) is expected


# --- fold_summary ---------------------------------------------------------


def test_fold_summary_includes_the_messages():
    text = fold_summary(None, [u("как варить плов"), a("рис девзира")])
    assert "как варить плов" in text
    assert "рис девзира" in text


def test_fold_summary_carries_the_previous_summary():
    """A second compaction must inherit the facts of the first.

    Otherwise a codeword from the start of the conversation is lost on the
    second round exactly like with plain trimming — defeating the day's point.
    """
    text = fold_summary("Пользователь просил запомнить КАРАКУМЫ.", [u("новый вопрос")])
    assert "КАРАКУМЫ" in text
    assert "новый вопрос" in text


def test_fold_summary_puts_the_previous_summary_before_the_new_turns():
    """Block order, not just presence — the invariant FOLD_HEADER's comment states.

    A header sandwiched between the two blocks reads as a caption for the one
    below it, and the model takes the old summary for part of the fresh
    conversation. Membership checks (`in text`) pass under a swapped order:
    verified by swapping the blocks — the three tests around this one all
    stayed green.
    """
    text = fold_summary("прежняя сводка", [u("новый вопрос")])

    assert text.index(FOLD_HEADER) < text.index("прежняя сводка") < text.index(TAIL_HEADER)
    assert text.index(TAIL_HEADER) < text.index("новый вопрос")


def test_fold_summary_marks_who_said_what():
    text = fold_summary(None, [u("вопрос"), a("ответ")])
    # Roles must stay distinguishable: a summary where turns blur into one
    # stream gets misattributed by the model.
    assert text.index("вопрос") < text.index("ответ")
    assert text.count("вопрос") == 1


# --- summary_messages -----------------------------------------------------


def test_summary_messages_is_a_pair_starting_with_user():
    messages = summary_messages("сводка разговора")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert "сводка разговора" in messages[0]["content"]
    assert messages[1]["content"] == SUMMARY_ACK


@pytest.mark.parametrize("empty", ["", "   ", "\n", None])
def test_summary_messages_of_nothing_is_nothing(empty):
    """An empty summary is never substituted in.

    A pair of empty text and "Принято." costs tokens and carries nothing —
    worse, it looks like real context.
    """
    assert summary_messages(empty) == []
