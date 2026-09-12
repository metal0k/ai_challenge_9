"""Offline comparison of the four context strategies (SPEC-w02d10.md §13).

Runs the SAME 12-turn ТЗ dialog under window/facts/summary/branch and scores
each on four axes: how many of six planted details survive to the final
"one list" answer, whether a superseded value comes back, how many prompt
tokens the primary calls cost, and how much the strategy's own side calls
(the summarizer or the facts extractor) cost on top of that.

Checked by substring, not an LLM judge — same reasoning as tools/compact_bench.py
and CLAUDE.md's "An LLM judge disagreeing…": there is a known-correct answer
here (the six planted details), and a judge would invent an authority the
measurement doesn't need.

The checklist scores качество/стабильность/расход numerically, but does not
cover all four axes the task asks for on its own: качество ответа and удобство
для пользователя need a human, not a substring match, to look at. So this file
also PRINTS what the checklist can't score — each strategy's own final answer
and each fork branch's own answer (excerpted, C1) — and a one-line, code-
grounded verdict on удобство (usability_line()), rather than inventing a
quality/convenience number the way an LLM judge would (same CLAUDE.md rule:
"отдай оценку креативности человеку, просто выведи данные").

A demo module, like tools/compact_bench.py: lives in tools/, not imported by
the agent. Network access is fine at runtime but not under `--dry-run`, which
builds no Config and touches no network at all.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console, tokens
from advent_core.agent import DEFAULT_KEEP_LAST, Agent, AgentReply
from advent_core.client import capabilities_of, find_model, list_models
from advent_core.config import DEFAULT_SYSTEM_PROMPT, PROJECT_ROOT, Config, ConfigError
from advent_core.errors import AdventError
from advent_core.params import CONTEXT_STRATEGY_CHOICES
from tools import bench_core

DEFAULT_MODEL = "ministral-14b-latest"
# Confirmed against a live run, not just reasoned about: 2026-09-12,
# `--strategies branch`, 12 turns, ministral-14b-latest. Cumulative prompt
# stayed ~12k at BOTH --limit 4000 (11342) and --limit 32000, an 8x wider
# window (11943) — branch's full 12-turn history never came close to
# pressing the 4000 budget, so "4000 is enough for branch's full history"
# holds. What it does NOT explain: branch scored 5/6 at both limits,
# identically — not a window-size question. Closed by a third run on the
# same parameters: 6/6, and the checker was verified directly against the
# model's own wording ("логотип готов"). So the 5/6 was run-to-run variation
# in what the model chose to restate in its final answer — not the window,
# not the checker.
# An earlier note here guessed the budget trim was cutting branch's history
# and causing the 5/6 — that guess was tested directly and refuted by the
# same 4000-vs-32000 comparison above.
DEFAULT_LIMIT = 4000

AGENT_PROMPT_PATH = PROJECT_ROOT / "week_02" / "prompts" / "agent.md"
SUMMARY_PROMPT_PATH = DEFAULT_SYSTEM_PROMPT.parent / "summary.md"
FACTS_PROMPT_PATH = DEFAULT_SYSTEM_PROMPT.parent / "facts.md"

# Six details planted across the first 7 turns (SPEC-w02d10.md §13.1), the
# budget replayed once (480 -> 520). HEAD_TURNS/TAIL_TURNS below are what
# scenario_for() refuses to shorten away.
SCENARIO: tuple[str, ...] = (
    "Нужно ТЗ на мобильное приложение для сети кофеен. Бюджет — 480 тысяч "
    "рублей. Дальше в этом разговоре отвечай коротко — двумя-тремя "
    "предложениями, без списков.",
    "Срок сдачи — к 1 декабря, позже сдвинуть не сможем: открытие точки.",
    "Платформа только Android — на iOS в этом бюджете не остаётся денег.",
    "Оплата только через СБП, банковские карты пока не подключаем.",
    "Вход — по номеру телефона с SMS-кодом, регистрацию по email делать не нужно.",
    "Логотип уже готов, дизайном логотипа заниматься не нужно — просто вставим готовый файл.",
    "Важное уточнение: бюджет пересмотрели, теперь 520 тысяч рублей, а не "
    "480 — используй новую цифру дальше.",
    "Нужны пуш-уведомления о готовности заказа — как их лучше показать пользователю?",
    "Что имеет смысл показать на экране онбординга при первом запуске?",
    "Нужна ли история прошлых заказов в профиле пользователя?",
    "Стоит ли делать программу лояльности с баллами уже в первой версии?",
    "Последний вопрос: выпиши итоговое ТЗ одним списком — со всеми деталями, которые мы обсудили.",
)

# The six planted-detail turns plus the budget replay (indices 0-6) — never
# pruned by scenario_for(). Losing any of these would mean a shortened run
# no longer plants what the detail checklist looks for.
HEAD_TURNS = 7
# The final "one list" ask — the day's only quality check, always last.
TAIL_TURNS = 1
MIN_TURNS = HEAD_TURNS + TAIL_TURNS


def scenario_for(turns: int | None) -> tuple[str, ...]:
    """First `turns` turns of SCENARIO: head (planted details) + trimmed
    filler middle + the final ask, never head or tail alone (SPEC §13.1's
    "shortens the middle and keeps first+last", same shape as
    compact_bench.scenario_for but with a multi-turn head instead of one)."""
    if turns is None or turns >= len(SCENARIO):
        return SCENARIO
    if turns <= MIN_TURNS:
        return (*SCENARIO[:HEAD_TURNS], SCENARIO[-1])
    filler = SCENARIO[HEAD_TURNS : HEAD_TURNS + (turns - MIN_TURNS)]
    return (*SCENARIO[:HEAD_TURNS], *filler, SCENARIO[-1])


# --------------------------------------------------------------------------
# Detail checklist (SPEC §13.2)
# --------------------------------------------------------------------------

# Collapses "520 000" / "520тыс" / "520 тысяч" / "520 тысячи" toward a bare
# "520" so a detail's forms list only needs the number once. Deliberately
# narrow (requires a "000" or "тыс…" marker right after the digits) so it
# never touches an unrelated number like the "1" in "1 декабря".
_THOUSANDS_RE = re.compile(r"(?<!\d)(\d{1,3})[\s ]*(?:000\b|тыс(?:\.|яч[а-я]{0,2})?\b)")


def normalize_numbers(text: str) -> str:
    """520 000 / 520тыс / 520 тысяч -> 520 (SPEC §13.2's own example)."""
    return _THOUSANDS_RE.sub(r"\1", text)


@dataclass(frozen=True, slots=True)
class Detail:
    """One planted fact: a stable id, a human label, and its acceptable forms.

    Any ONE form present (after normalization) counts the detail as kept —
    the six items are alternate phrasings of the SAME fact, not conjunctions
    of several (SPEC §13.2: "у каждой детали список допустимых форм").
    """

    key: str
    label: str
    forms: tuple[str, ...]


# Forms are generous synonyms rather than one exact phrase: a live model's
# wording cannot be predicted, and these are a starting point to tune against
# a real run before the recording — the same caveat as DEFAULT_LIMIT above.
DETAILS: tuple[Detail, ...] = (
    Detail("budget", "бюджет 520 тыс.", ("520",)),
    Detail("deadline", "срок к 1 декабря", ("1 декабря",)),
    Detail("platform", "только Android", ("android",)),
    Detail("payment", "оплата через СБП", ("сбп",)),
    Detail(
        "login",
        "вход по телефону, без email-регистрации",
        ("по номеру телефона", "без регистрации по email", "без email"),
    ),
    Detail(
        "logo",
        "логотип готов, дизайн не нужен",
        ("дизайн логотипа не", "логотип уже готов", "готовый логотип", "логотип готов"),
    ),
)

# The superseded value (SPEC §13.1's "budget 480, replayed as 520") — a
# separate check from DETAILS: a form list says what SHOULD be there, this
# says what should NOT be.
BUDGET_STALE_FORMS: tuple[str, ...] = ("480",)

_DETAIL_LABELS: dict[str, str] = {detail.key: detail.label for detail in DETAILS}


def detail_label(key: str) -> str:
    """Human label for a DETAILS key — `key` itself if it's somehow unknown,
    so a caller can never crash printing a diagnostic line over this."""
    return _DETAIL_LABELS.get(key, key)


def detail_present(text: str, forms: Sequence[str]) -> bool:
    haystack = normalize_numbers(text.lower())
    return any(normalize_numbers(form.lower()) in haystack for form in forms)


@dataclass(frozen=True, slots=True)
class DetailScore:
    """Outcome of checking one final answer against DETAILS/BUDGET_STALE_FORMS."""

    present: tuple[str, ...]
    missing: tuple[str, ...]
    stale_returned: bool


def score_answer(text: str) -> DetailScore:
    present = tuple(detail.key for detail in DETAILS if detail_present(text, detail.forms))
    missing = tuple(detail.key for detail in DETAILS if detail.key not in present)
    return DetailScore(
        present=present,
        missing=missing,
        stale_returned=detail_present(text, BUDGET_STALE_FORMS),
    )


# --------------------------------------------------------------------------
# One strategy's run
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StrategyResult:
    """Result of running SCENARIO end to end under one context_strategy."""

    strategy: str
    turn_prompt_tokens: list[int | None] = field(default_factory=list)
    prompt_total: int | None = None
    missing_usage: int = 0
    side_calls: int = 0
    side_prompt: int = 0
    side_completion: int = 0
    # Among side_calls/side_prompt/side_completion above: calls that were
    # billed and produced NOTHING usable (AgentReply.facts_failed). Already
    # folded into the totals above (bench_core.side_call_totals counts them),
    # tracked separately so a reader sees "paid for, produced nothing" as its
    # own fact rather than a number that quietly vanished into a clean total.
    failed_calls: int = 0
    failed_prompt: int = 0
    failed_completion: int = 0
    # Budget-trim safety net across this run (bench_core.trim_totals) —
    # distinct from side_calls above: those are extra paid calls, this is
    # history the safety net cut from the primary request itself.
    trim: bench_core.TrimTotals = field(default_factory=bench_core.TrimTotals)
    score: DetailScore = field(
        default_factory=lambda: DetailScore(present=(), missing=(), stale_returned=False)
    )
    # The final "one list" answer — printed so a reader can judge quality and
    # convenience themselves (CLAUDE.md: "отдай оценку креативности человеку,
    # просто выведи данные" — no LLM-scored quality number here either).
    answer: str = ""

    @property
    def side_total(self) -> int:
        return self.side_prompt + self.side_completion

    @property
    def grand_total(self) -> int | None:
        """prompt_total + side_total. None — nothing to add it to (no usage at all)."""
        if self.prompt_total is None:
            return None
        return self.prompt_total + self.side_total


def _score_run(strategy: str, replies: Sequence[AgentReply]) -> StrategyResult:
    turn_prompt_tokens = [reply.result.usage.prompt_tokens for reply in replies]
    _rows, prompt_total, missing = bench_core.cumulative_column(turn_prompt_tokens)
    calls, side_prompt, side_completion = bench_core.side_call_totals(replies)
    failed_calls, failed_prompt, failed_completion = bench_core.failed_side_calls(replies)
    last_text = replies[-1].text if replies else ""
    return StrategyResult(
        strategy=strategy,
        turn_prompt_tokens=turn_prompt_tokens,
        prompt_total=prompt_total,
        missing_usage=missing,
        side_calls=calls,
        side_prompt=side_prompt,
        side_completion=side_completion,
        failed_calls=failed_calls,
        failed_prompt=failed_prompt,
        failed_completion=failed_completion,
        trim=bench_core.trim_totals(replies),
        score=score_answer(last_text),
        answer=last_text,
    )


# --------------------------------------------------------------------------
# Answer excerpts (C1: a reader must see the actual text to judge quality and
# удобство, not only the checklist numbers — CLAUDE.md's "отдай оценку
# человеку, просто выведи данные")
# --------------------------------------------------------------------------

# Long enough to show the shape of a "one list" answer, short enough that
# four of them plus the table still fit a demo take on camera.
ANSWER_EXCERPT_CHARS = 360


def excerpt(text: str, limit: int = ANSWER_EXCERPT_CHARS) -> str:
    """`text`, or its first `limit` chars with an explicit truncation note.

    The note names the real length rather than just trailing off with "…" —
    a silent cut reads as "this is the whole answer", which is the same kind
    of silent-vanishing this file's totals fix is about.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}… (обрезано, всего {len(text)} символов)"


# --------------------------------------------------------------------------
# Verdict (SPEC §13.2: "a bare number reads as a verdict on the feature")
# --------------------------------------------------------------------------


def verdict_lines(results: Sequence[StrategyResult]) -> list[str]:
    """Spoken lines next to the table. Facts being expensive is a finding
    (PROBE-w02d10-facts.md: ~850 tokens/turn), not a failure. branch's
    lossless guarantee is about the CONTEXT it sends (nothing dropped from
    history), not about whether the model restates every planted detail in
    its final answer — that's the model's own decision, not a property of
    the strategy. branch's price rank is reported, not judged: measured
    2026-09-12, branch was the CHEAPEST of four strategies, twice, despite
    sending the whole history — accumulated prompt also depends on how long
    the model's own answers are, since those get fed back into every later
    prompt."""
    lines: list[str] = []
    by_name = {result.strategy: result for result in results}

    facts = by_name.get("facts")
    if facts is not None and facts.side_calls:
        lines.append(
            f"facts тратит {facts.side_prompt}/{facts.side_completion} токенов "
            f"({facts.side_calls} вызовов экстрактора) сверх диалога — дороже других "
            "стратегий это ожидаемо (PROBE-w02d10-facts.md: ~850 токенов на ход), "
            "а не поломка"
        )

    failed = [result for result in results if result.failed_calls]
    if failed:
        parts = ", ".join(
            f"{result.strategy}: {result.failed_calls} "
            f"({result.failed_prompt}/{result.failed_completion} токенов)"
            for result in failed
        )
        lines.append(
            f"оплачено и не дало ничего использовать (facts_failed): {parts} — "
            "уже учтено в «цене побочных вызовов» выше, не вычтено из неё"
        )

    branch = by_name.get("branch")
    if branch is not None:
        total = len(DETAILS)
        present = branch.score.present
        # stale_returned still disqualifies "lossless" (a stale value coming
        # back means the context wasn't clean even with nothing missing), but
        # it is reported as its own line (B), never folded into the
        # missing-details wording below — a stale return is a separate fact
        # from a missing detail.
        lossless = len(present) == total and not branch.score.stale_returned

        if lossless:
            detail_line = f"branch — lossless-базлайн: {total}/{total} деталей"
        else:
            missing_labels = ", ".join(detail_label(key) for key in branch.score.missing)
            detail_line = (
                f"branch собрал {len(present)}/{total} деталей (не хватает: {missing_labels}) — "
                "контекст при этом отдан целиком, branch ничего не выбрасывает из истории. "
                "Перепишет ли модель каждую деталь в финальный список — её решение, а не "
                "свойство стратегии: три прогона на одних и тех же параметрах (2026-09-12) "
                "дали 5/6, 5/6 и 6/6."
            )
        # A data-driven addendum, not a replacement for detail_line: the
        # 2026-09-12 hypothesis that the budget trim explains a branch miss
        # was tested directly (4000 vs 32000 --limit) and refuted — so this
        # only fires when THIS run's own numbers show branch was actually cut
        # by the safety net, never as a guess.
        if branch.trim.turns_trimmed:
            tokens_part = (
                "число токенов неизвестно"
                if branch.trim.tokens is None
                else f"{branch.trim.tokens} токенов"
            )
            detail_line += (
                f"; на этом прогоне собственный safety-net всё же резал branch — "
                f"{branch.trim.turns_trimmed} ход(ов), {branch.trim.messages} сообщ., "
                f"{tokens_part} — стоит проверить и это"
            )
        lines.append(detail_line)

        if branch.score.stale_returned:
            lines.append(
                "branch вернул устаревшее значение 480, хотя вся история у него перед "
                "глазами — это тоже про модель, а не про потерю контекста"
            )

        known = [(r.strategy, r.grand_total) for r in results if r.grand_total is not None]
        # A single-strategy run (e.g. `--strategies branch`) has nothing to
        # compare price against — `priciest=False` there would report a
        # comparison that never happened (2026-09-12 finding).
        comparable = len(known) > 1
        priciest = False
        if comparable:
            top = max(value for _, value in known)
            leaders = [name for name, value in known if value == top]
            priciest = leaders == [branch.strategy]

        if not comparable:
            price_line = "цену branch сравнивать не с чем — единственная стратегия в этом прогоне"
        elif priciest:
            price_line = (
                f"branch — самый дорогой прогон ({bench_core.num(branch.grand_total)} токенов), "
                "как и ожидается от стратегии, которая шлёт всю историю"
            )
        else:
            price_line = (
                f"branch не самый дорогой прогон ({bench_core.num(branch.grand_total)} токенов), "
                "хотя шлёт всю историю: накопленный prompt зависит ещё и от того, насколько "
                "длинно отвечает модель"
            )
        lines.append(price_line)

    known_totals = [(r.strategy, r.grand_total) for r in results if r.grand_total is not None]
    if len(known_totals) > 1:
        # max()/min() on a tie silently name a "winner" nobody won
        # (CLAUDE.md) — the tie is listed instead.
        cheapest_total = min(total for _, total in known_totals)
        cheapest = [name for name, total in known_totals if total == cheapest_total]
        if len(cheapest) == 1:
            lines.append(f"дешевле всех — {cheapest[0]} ({cheapest_total} токенов)")
        else:
            lines.append(f"дешевле всех — ничья: {', '.join(cheapest)} ({cheapest_total} токенов)")

    lines.append(usability_line())
    return lines


def usability_line() -> str:
    """The fourth comparison axis (SPEC §1: "удобство для пользователя"),
    otherwise addressed nowhere in this file. Grounded in what the code
    actually requires from the user (advent_core/agent.py, week_02/cli.py's
    command handlers), not an impression:

    - window/summary run unattended every turn — the user issues no command
      for either to work (summary is day 09's own scheduled trigger).
    - facts also runs unattended each turn, but correcting a wrong value
      needs `/fact set|del|unpin`, and switching into it from empty facts
      needs an explicit `/facts backfill` or the agent reads as amnesiac.
    - branch is the only strategy the user drives by hand end to end:
      `/checkpoint`, `/branch`, `/switch`, `/branches` — nothing forks or
      switches on its own.
    """
    return (
        "удобство: window и summary не требуют от пользователя ни одной команды — "
        "обе решают сами на каждом ходу; facts тоже работает без команд на ходу, но "
        "правка неверного значения идёт через /fact set|del|unpin, а включение с "
        "пустых facts требует явного /facts backfill; branch — единственная стратегия, "
        "где /checkpoint, /branch, /switch и /branches держит сам пользователь, иначе "
        "веток просто не появится"
    )


def _trim_cell(trim: bench_core.TrimTotals) -> str:
    """ "ходов/сообщ/ток." — same slash-triple style as the side-call column.
    "—" for tokens only when the safety net cut messages whose freed size
    the counter couldn't say (bench_core.trim_totals: 0 turns cut is a known
    zero, never a gap)."""
    return f"{trim.turns_trimmed}/{trim.messages}/{bench_core.num(trim.tokens)}"


def _missing_details_line(results: Sequence[StrategyResult]) -> str | None:
    """ "нет X/6" alone can't tell a checker gap from a genuine omission — this
    names WHICH detail(s) a strategy's answer is missing, so a reader can look
    at the excerpt (_print_answers) and judge which it is. None when every
    strategy scored 6/6 — nothing to report."""
    parts = [
        f"{result.strategy} — {', '.join(detail_label(key) for key in result.score.missing)}"
        for result in results
        if result.score.missing
    ]
    if not parts:
        return None
    return "не хватает деталей: " + "; ".join(parts)


def _print_table(results: Sequence[StrategyResult], args: argparse.Namespace) -> None:
    """The command's product — stdout, like tools/compact_bench.py's own table."""
    turns = len(results[0].turn_prompt_tokens) if results else 0
    table = Table(
        title=f"стратегии контекста · окно {args.limit} токенов · {turns} ходов · {args.model}"
    )
    table.add_column("стратегия", style="cyan", no_wrap=True)
    table.add_column(f"детали (n/{len(DETAILS)})", justify="right")
    table.add_column("старое значение не вернулось", justify="center")
    table.add_column("prompt накопительно", justify="right")
    table.add_column("обрезка safety-net (ход/сообщ/ток.)", justify="right")
    table.add_column("цена побочных вызовов", justify="right")
    table.add_column("впустую", justify="right")
    table.add_column("итого", justify="right")
    for result in results:
        table.add_row(
            result.strategy,
            f"{len(result.score.present)}/{len(DETAILS)}",
            "да" if not result.score.stale_returned else "нет",
            bench_core.num(result.prompt_total),
            _trim_cell(result.trim),
            f"{result.side_prompt}/{result.side_completion}",
            str(result.failed_calls),
            bench_core.num(result.grand_total),
        )
    console.out.print(table)

    missing = sum(result.missing_usage for result in results)
    if missing:
        console.out.print(
            f"без usage: ходов без прихода токенов — {missing} — "
            "итог там, где есть пропуски, неполон"
        )
    missing_details = _missing_details_line(results)
    if missing_details:
        console.out.print(missing_details)
    for line in verdict_lines(results):
        console.out.print(line)


def _print_answers(results: Sequence[StrategyResult]) -> None:
    """Each strategy's final "one list" answer (C1) — the table scores it,
    this shows it, so a reader judges quality and удобство themselves
    (CLAUDE.md: "отдай оценку креативности человеку, просто выведи данные")."""
    table = Table(title="итоговые ответы («выпиши ТЗ одним списком»)")
    table.add_column("стратегия", style="cyan", no_wrap=True)
    table.add_column("ответ")
    for result in results:
        table.add_row(result.strategy, excerpt(result.answer))
    console.out.print(table)


# --------------------------------------------------------------------------
# Fork isolation (SPEC §13.3)
# --------------------------------------------------------------------------

FORK_CHECKPOINT_TURNS = 6
FORK_BUDGET_A = "520"
FORK_BUDGET_B = "300"
FORK_QUESTION = "Уточнение: бюджет на самом деле {value} тысяч рублей, используй эту цифру."
FORK_ASK = "Напомни, какой у нас сейчас бюджет?"


def run_fork_isolation(agent: Agent, head: Sequence[str]) -> tuple[str, str]:
    """Checkpoints after `head`, forks into two continuations with a
    contradicting budget each, returns each branch's own answer about it.

    The "checkpoint" here is a bench_core.ScenarioState, not a Session file:
    this harness measures whether the STRATEGY keeps facts separate per
    continuation, not the branch-file mechanism advent_core/session.py owns
    and tests on its own (SPEC-w02d10.md §13.3).
    """
    _replies, checkpoint = bench_core.run_scenario(agent, head)

    branch_a = checkpoint.clone()
    bench_core.run_turn(agent, branch_a, FORK_QUESTION.format(value=FORK_BUDGET_A))
    reply_a = bench_core.run_turn(agent, branch_a, FORK_ASK)

    branch_b = checkpoint.clone()
    bench_core.run_turn(agent, branch_b, FORK_QUESTION.format(value=FORK_BUDGET_B))
    reply_b = bench_core.run_turn(agent, branch_b, FORK_ASK)

    return reply_a.text, reply_b.text


def fork_isolation_ok(answer_a: str, answer_b: str) -> tuple[bool, bool]:
    """(a holds its own value, b holds its own value) — each False if that
    branch's answer is missing its own value OR contains the OTHER branch's,
    either of which would mean facts leaked across the fork."""
    norm_a = normalize_numbers(answer_a.lower())
    norm_b = normalize_numbers(answer_b.lower())
    a_ok = FORK_BUDGET_A in norm_a and FORK_BUDGET_B not in norm_a
    b_ok = FORK_BUDGET_B in norm_b and FORK_BUDGET_A not in norm_b
    return a_ok, b_ok


def fork_isolation_report(a_ok: bool, b_ok: bool) -> str:
    if a_ok and b_ok:
        return (
            f"ветки изолированы: своя ветка ответила {FORK_BUDGET_A}, "
            f"другая — {FORK_BUDGET_B}, ни одна не увидела чужой факт"
        )
    leaked = []
    if not a_ok:
        leaked.append(FORK_BUDGET_A)
    if not b_ok:
        leaked.append(FORK_BUDGET_B)
    return (
        f"facts протекли между ветками: ветка(и) с бюджетом {', '.join(leaked)} не удержала "
        "свой ответ — проверь изоляцию facts_upto/facts между ScenarioState-копиями"
    )


def _print_fork_section(answer_a: str, answer_b: str) -> None:
    """Each branch's own answer plus the pass/fail verdict (C1): a reader has
    to be able to see the leak (or its absence) in the actual text, not just
    trust a computed boolean."""
    a_ok, b_ok = fork_isolation_ok(answer_a, answer_b)
    console.out.print(f"ветка бюджета {FORK_BUDGET_A}: {excerpt(answer_a)}")
    console.out.print(f"ветка бюджета {FORK_BUDGET_B}: {excerpt(answer_b)}")
    console.out.print(fork_isolation_report(a_ok, b_ok))


# --------------------------------------------------------------------------
# Prompts, --dry-run, argparse, main
# --------------------------------------------------------------------------


def _persona() -> str | None:
    try:
        return AGENT_PROMPT_PATH.read_text(encoding="utf-8").strip() or None
    except OSError as error:
        console.warn(f"персона агента не прочиталась ({error}) — идём без персоны")
        return None


def _summary_prompt() -> str | None:
    try:
        return SUMMARY_PROMPT_PATH.read_text(encoding="utf-8").strip() or None
    except OSError as error:
        console.warn(
            f"промпт суммаризатора не прочитался ({error}) — strategy=summary "
            "выключится сам, агент скажет об этом вслух"
        )
        return None


def _facts_prompt() -> str | None:
    try:
        return FACTS_PROMPT_PATH.read_text(encoding="utf-8").strip() or None
    except OSError as error:
        console.warn(
            f"промпт экстрактора не прочитался ({error}) — strategy=facts "
            "выродится в window, агент скажет об этом вслух"
        )
        return None


def parse_strategies(raw: str) -> list[str]:
    """Comma-separated, order-preserved, de-duplicated, validated list.

    Validated against CONTEXT_STRATEGY_CHOICES rather than a copy of the four
    names here — advent_core/params.py is the one place that knows them.
    """
    seen: list[str] = []
    for chunk in raw.split(","):
        name = chunk.strip()
        if not name:
            continue
        if name not in CONTEXT_STRATEGY_CHOICES:
            allowed = ", ".join(CONTEXT_STRATEGY_CHOICES)
            raise ConfigError(f"неизвестная стратегия {name!r}, ожидалось одно из: {allowed}")
        if name not in seen:
            seen.append(name)
    if not seen:
        raise ConfigError("--strategies не должен быть пустым")
    return seen


def _print_scenario(
    args: argparse.Namespace, scenario: Sequence[str], strategies: Sequence[str]
) -> None:
    """Scenario and checklist, no network touched at all (`--dry-run`)."""
    table = Table(title=f"Сценарий strategy_bench, {len(scenario)} ходов (--dry-run, без сети)")
    table.add_column("№", justify="right", style="cyan", no_wrap=True)
    table.add_column("реплика")
    for number, question in enumerate(scenario, start=1):
        table.add_row(str(number), question)
    console.out.print(table)

    checklist = Table(title=f"Плантированные детали ({len(DETAILS)})")
    checklist.add_column("№", justify="right", style="cyan", no_wrap=True)
    checklist.add_column("деталь")
    checklist.add_column("допустимые формы")
    for number, detail in enumerate(DETAILS, start=1):
        checklist.add_row(str(number), detail.label, ", ".join(detail.forms))
    console.out.print(checklist)
    console.out.print(f"старое значение (не должно вернуться): {', '.join(BUDGET_STALE_FORMS)}")

    keep_last = args.keep_last if args.keep_last is not None else DEFAULT_KEEP_LAST
    console.out.print(
        f"модель {args.model} · лимит окна {args.limit} · keep_last {keep_last} · "
        f"стратегии {', '.join(strategies)} · fork {'да' if args.fork else 'нет'}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = bench_core.build_parser(
        "Сравнение стратегий контекста: один сценарий на каждой из window/facts/summary/branch.",
        default_model=DEFAULT_MODEL,
        default_limit=DEFAULT_LIMIT,
        limit_help=f"узкое окно context_limit для всех прогонов (по умолчанию {DEFAULT_LIMIT})",
        model_help="модель для всех прогонов",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help=f"сколько ходов сценария прогнать (по умолчанию все {len(SCENARIO)}); "
        "плантированные детали и финальный вопрос всегда остаются",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help=f"хвост window/facts/summary, сообщений (по умолчанию агента: {DEFAULT_KEEP_LAST})",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(CONTEXT_STRATEGY_CHOICES),
        help=f"через запятую, из: {', '.join(CONTEXT_STRATEGY_CHOICES)}",
    )
    parser.add_argument(
        "--fork",
        dest="fork",
        action="store_true",
        default=True,
        help="прогнать секцию изоляции веток (по умолчанию)",
    )
    parser.add_argument(
        "--no-fork",
        dest="fork",
        action="store_false",
        help="пропустить секцию изоляции веток",
    )
    return parser


def _build_agent(
    *,
    strategy: str,
    base_config: Config,
    limit: int,
    keep_last: int | None,
    counter: tokens.TokenCounter | None,
    capabilities: dict | None,
    persona: str | None,
    summary_prompt: str | None,
    facts_prompt: str | None,
) -> Agent:
    params = bench_core.apply_param_overrides(
        base_config.params, context_strategy=strategy, keep_last=keep_last
    )
    config = replace(base_config, params=params)
    return Agent(
        config,
        complete=chat_core.complete,
        stream=chat_core.stream,
        counter=counter,
        capabilities=capabilities,
        context_limit=limit,
        persona=persona,
        dialog_preset=None,
        summary_prompt=summary_prompt,
        facts_prompt=facts_prompt,
        on_warning=console.warn,
    )


def main(argv: list[str] | None = None) -> int:
    console.force_utf8()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit должен быть положительным — нечему резать ни одну стратегию")
    if args.turns is not None and args.turns < MIN_TURNS:
        parser.error(
            f"--turns не меньше {MIN_TURNS}: короче детали негде плантировать и негде терять"
        )
    try:
        strategies = parse_strategies(args.strategies)
    except ConfigError as exc:
        parser.error(str(exc))

    scenario = scenario_for(args.turns)

    if args.dry_run:
        _print_scenario(args, scenario, strategies)
        return 0

    try:
        base_config = Config.resolve(model=args.model)
    except ConfigError as exc:
        console.warn(str(exc))
        return 2

    try:
        models = list_models(base_config)
    except AdventError as exc:
        console.fail(exc)
        return exc.exit_code
    card = find_model(models, base_config.model)
    capabilities = capabilities_of(models, base_config.model)

    # on_notice: a cold-cache tekken download takes minutes — a silent process
    # between launch and the first call reads as hung.
    counter, warning = tokens.counter_for(base_config.model, on_notice=console.note)
    if warning:
        console.warn(warning)

    persona = _persona()
    summary_prompt = _summary_prompt()
    facts_prompt = _facts_prompt()

    window = bench_core.num(tokens.context_limit(card))
    console.note(
        f"модель {base_config.model} · окно карточки {window} · "
        f"лимит прогона {args.limit} · ходов {len(scenario)} · стратегии {', '.join(strategies)}"
    )

    results: list[StrategyResult] = []
    try:
        for index, strategy in enumerate(strategies, start=1):
            console.note(f"прогон {index}/{len(strategies)}: {strategy}")
            agent = _build_agent(
                strategy=strategy,
                base_config=base_config,
                limit=args.limit,
                keep_last=args.keep_last,
                counter=counter,
                capabilities=capabilities,
                persona=persona,
                summary_prompt=summary_prompt,
                facts_prompt=facts_prompt,
            )
            replies, _state = bench_core.run_scenario(agent, scenario)
            results.append(_score_run(strategy, replies))

        _print_table(results, args)
        _print_answers(results)

        if args.fork:
            console.note(f"секция изоляции веток: checkpoint после хода {FORK_CHECKPOINT_TURNS}")
            fork_agent = _build_agent(
                strategy="facts",
                base_config=base_config,
                limit=args.limit,
                keep_last=args.keep_last,
                counter=counter,
                capabilities=capabilities,
                persona=persona,
                summary_prompt=summary_prompt,
                facts_prompt=facts_prompt,
            )
            answer_a, answer_b = run_fork_isolation(fork_agent, SCENARIO[:FORK_CHECKPOINT_TURNS])
            _print_fork_section(answer_a, answer_b)
    except AdventError as exc:
        console.fail(exc)
        return exc.exit_code

    return 0


if __name__ == "__main__":
    sys.exit(main())
