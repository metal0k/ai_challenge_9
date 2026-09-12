"""Offline comparison of history compaction: one scenario, compact off/on.

SPEC-w02d09.md §13. Runs the SAME fixed dialog twice — `compact=False` and
`compact=True` — under an artificially low `context_limit`: at the model's
real window (262144 tokens) trim never fires over a dozen turns, and the
comparison would degenerate into two identical runs.

Quality is checked by substring, not an LLM judge: the first turn plants a
codeword, the last asks for it back, and the check is an exact, free substring
match. A judge would be misplaced here for the same reason as in
week_01/temperature.py: it earns its keep where there is a ground truth it can
disagree with; here the ground truth is checked without any AI at all
(CLAUDE.md, "An LLM judge disagreeing…", SPEC §13).

A demo module, like tools/make_biginput.py: lives in tools/, not imported by
the agent. Network access (model list, tokenizer, chat calls) is fine at
runtime but not under `--dry-run`, which builds no Config and touches no
network at all.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console, tokens
from advent_core.agent import DEFAULT_COMPACT_EVERY, DEFAULT_KEEP_LAST, Agent
from advent_core.client import capabilities_of, find_model, list_models
from advent_core.config import DEFAULT_SYSTEM_PROMPT, PROJECT_ROOT, Config, ConfigError
from advent_core.errors import AdventError
from advent_core.params import GenerationParams
from tools import bench_core

DEFAULT_MODEL = "ministral-14b-latest"
DEFAULT_LIMIT = 2500

# Codeword planted in the first turn. Uppercase, no common meaning — lowers
# the chance the model "guesses" it from general knowledge instead of history.
CODEWORD = "КАРАКУМЫ"

# The scenario is a hardcoded constant, not read from a file: comparing modes
# requires literally the same dialog twice, and an editable scenario file
# between runs would break that (same reasoning as demo_steps() in
# advent_cli/record.py — the scenario is fixed).
#
# The first turn plants the codeword and caps the answer length. That cap is
# load-bearing, and it is the opposite of what it looks like: the scenario
# asked for VERBOSE answers first, and the live run (2026-09-10) came back with
# 2929 completion tokens per turn — a single exchange does not fit a 2500-token
# window, so trim wiped the history on EVERY turn, `older` stayed empty and
# compaction never fired once. Both modes then produced identical numbers and
# forgot the codeword, i.e. the day's comparison showed nothing. With ~150-token
# answers at the same window: compact off starts trimming on turn 8 and loses
# the codeword, compact on compacts on turns 8 and 11 and keeps it. History has
# to grow past the window gradually, not overshoot it in one turn.
#
# The last turn is the day's only quality check: did the agent remember the
# codeword (§13).
SCENARIO: tuple[str, ...] = (
    f"Запомни кодовое слово: {CODEWORD}. Дальше в этом разговоре отвечай на "
    "каждый мой вопрос коротко — двумя-тремя предложениями, без списков.",
    "Как возник Великий шёлковый путь и какие города были на нём ключевыми?",
    "Чем отличаются TCP и UDP по гарантиям доставки?",
    "Каков жизненный цикл звезды от протозвёздного облака до финала?",
    "Как устроено человеческое сердце: камеры, клапаны, круги кровообращения?",
    "Как устроено индексирование в реляционных базах и почему выбирают B-tree?",
    "Что происходило с книгопечатанием от Гутенберга до промышленной революции?",
    "Как работает турбореактивный двигатель — по тактам?",
    "Кто живёт в экосистеме кораллового рифа и почему рифы уязвимы к потеплению?",
    "Как идёт фотосинтез: световая и темновая фазы, роль хлорофилла?",
    "Чем акции отличаются от облигаций по риску и правам держателя?",
    "Последний вопрос: какое кодовое слово я просил тебя запомнить в самом начале разговора?",
)

# Shortest run that still measures anything: plant the codeword, let history
# grow by at least one turn, ask for it back.
MIN_TURNS = 3


def scenario_for(turns: int | None) -> tuple[str, ...]:
    """First `turns` turns of SCENARIO, with the codeword question always last.

    A plain slice would cut the final turn — the day's only quality check
    (§13) — and both arms would then be compared without ever being asked for
    the codeword. A shortened run is therefore "the first turns-1 questions
    plus the last one", never "the first turns questions".
    """
    if turns is None or turns >= len(SCENARIO):
        return SCENARIO
    return (*SCENARIO[: turns - 1], SCENARIO[-1])


# Same persona as `adventagent` (the agent's production path), not the
# project persona (advent_core/prompts/default_system.md): that one says
# "concise assistant, no filler" and actively shortens answers (CLAUDE.md —
# it halves reasoning length on the Day 01 measurement), but this scenario
# needs history to grow via long answers. The agent persona carries no such
# restriction.
AGENT_PROMPT_PATH = PROJECT_ROOT / "week_02" / "prompts" / "agent.md"
# Same path week_02/cli.py uses — the summarizer prompt lives in
# advent_core/prompts, core mechanics rather than a week's persona.
SUMMARY_PROMPT_PATH = DEFAULT_SYSTEM_PROMPT.parent / "summary.md"


@dataclass(slots=True)
class ModeResult:
    """Result of running SCENARIO end to end in one mode (compact off/on).

    `turn_prompt_tokens` — server-reported prompt tokens for EACH turn
    (result.usage.prompt_tokens); None means usage was missing, not zero
    (project rule: unknown never becomes zero).
    """

    turn_prompt_tokens: list[int | None] = field(default_factory=list)
    compactions: int = 0
    compact_prompt: int = 0
    compact_completion: int = 0
    remembered: bool = False


def _num(value: int | None) -> str:
    """The number, or "—" for unknown — never zero."""
    return "—" if value is None else str(value)


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
            f"промпт суммаризатора не прочитался ({error}) — режим compact=True "
            "выключится сам, агент скажет об этом вслух"
        )
        return None


def _mode_params(
    base: GenerationParams, *, compact: bool, keep_last: int | None, compact_every: int | None
) -> GenerationParams:
    """Copy of session params with compact/keep_last/compact_every applied.

    Thin wrapper over bench_core.apply_param_overrides — see there for why
    the bounds check goes through params.set() rather than living here.
    """
    return bench_core.apply_param_overrides(
        base, compact=compact, keep_last=keep_last, compact_every=compact_every
    )


def _run_mode(
    *,
    config: Config,
    limit: int,
    counter: tokens.TokenCounter | None,
    capabilities: dict | None,
    persona: str | None,
    summary_prompt: str | None,
    scenario: Sequence[str],
) -> ModeResult:
    """Runs the scenario end to end in one mode, collects per-turn numbers.

    The agent doesn't hold history itself (advent_core/agent.py, `ask()`
    contract) — this loop tracks `history` and `summary`, same as
    `week_02/cli.py._turn()` does for an interactive session.
    """
    agent = Agent(
        config,
        complete=chat_core.complete,
        stream=chat_core.stream,
        counter=counter,
        capabilities=capabilities,
        context_limit=limit,
        persona=persona,
        dialog_preset=None,
        summary_prompt=summary_prompt,
        on_warning=console.warn,
    )

    replies, _state = bench_core.run_scenario(agent, scenario)
    result = ModeResult()
    for reply in replies:
        if reply.compaction is not None:
            result.compactions += 1
            usage = reply.compaction.result.usage
            # `or 0`: usage may be missing entirely, undercounting THIS
            # compaction's cost — same trick and caveat as
            # week_02/cli.py._report_compaction. Zero here means "server
            # didn't say", not "free".
            result.compact_prompt += usage.prompt_tokens or 0
            result.compact_completion += usage.completion_tokens or 0
        result.turn_prompt_tokens.append(reply.result.usage.prompt_tokens)
    last_text = replies[-1].text if replies else ""
    # Case-insensitive: checks the fact of remembering, not spelling — the
    # model needn't echo the codeword in the same case.
    result.remembered = CODEWORD.lower() in last_text.lower()
    return result


# Same behaviour as `_print_growth_table()` in week_02/cli.py, moved to
# bench_core so tools/strategy_bench.py doesn't reinvent it (SPEC-w02d10 §13).
_cumulative = bench_core.cumulative_column


def _net_savings(off_total: int | None, on_total: int | None, compact_cost: int) -> int | None:
    """(off total) − (on total) − (compaction cost). None means nothing to compute from."""
    if off_total is None or on_total is None:
        return None
    return off_total - on_total - compact_cost


def _verdict(net: int | None, off_remembered: bool, on_remembered: bool) -> str:
    """One line naming the trade the numbers describe.

    A bare negative "чистая экономия" reads as "the feature lost", and on the
    three live runs of 2026-09-10 (ministral-14b, this scenario) it is always
    negative: −3464 at window 2500, −867 at 8000 with one compaction, −4163 at
    8000 with four. One compaction costs ~1450 tokens and saves 150-460 per
    later turn, so twelve turns never repay it. What it buys is the codeword
    trimming destroys — and both halves have to be said, or the table says
    something untrue by omission.
    """
    if net is None:
        return "чистую экономию посчитать не из чего: на части ходов сервер не прислал usage"

    if on_remembered and not off_remembered:
        memory = "; но кодовое слово пережило только сжатие — за это и заплачено"
    elif off_remembered and not on_remembered:
        memory = "; и кодовое слово при этом потеряно — здесь сжатие проигрывает по обоим счётам"
    elif not off_remembered:
        memory = "; кодовое слово забыто в обоих прогонах — на этом окне сравнивать нечего"
    else:
        memory = "; кодовое слово помнят оба — на этой длине истории обрезке нечего терять"

    if net > 0:
        return f"сжатие дешевле обрезки на {net} токенов{memory}"
    if net == 0:
        return f"токенов поровну{memory}"
    return f"сжатие дороже обрезки на {-net} токенов{memory}"


def _print_report(off: ModeResult, on: ModeResult, args: argparse.Namespace) -> None:
    """Before/after table and summary block — the command's product, hence stdout."""
    off_rows, off_total, off_missing = _cumulative(off.turn_prompt_tokens)
    on_rows, on_total, on_missing = _cumulative(on.turn_prompt_tokens)

    turns = len(off.turn_prompt_tokens)
    table = Table(
        title=f"compact off vs on · окно {args.limit} токенов · {turns} ходов · {args.model}"
    )
    table.add_column("ход", justify="right", style="cyan", no_wrap=True)
    table.add_column("prompt off", justify="right")
    table.add_column("накопит. off", justify="right")
    table.add_column("prompt on", justify="right")
    table.add_column("накопит. on", justify="right")
    for number, (off_row, on_row) in enumerate(zip(off_rows, on_rows, strict=True), start=1):
        table.add_row(str(number), *off_row, *on_row)
    console.out.print(table)
    if off_missing or on_missing:
        # Without this line the last visible "cumulative" reads as an exact
        # total, though missing turns never made it in.
        console.out.print(
            f"без usage: off {off_missing} ход(ов), on {on_missing} ход(ов) — "
            "итог там, где есть пропуски, неполон"
        )

    compact_cost = on.compact_prompt + on.compact_completion
    net = _net_savings(off_total, on_total, compact_cost)

    summary = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("prompt-токены всего · compact off", _num(off_total))
    summary.add_row("prompt-токены всего · compact on", _num(on_total))
    summary.add_row("сжатий (compact on)", str(on.compactions))
    summary.add_row(
        "стоимость сжатий (compact on), prompt/completion",
        f"{on.compact_prompt}/{on.compact_completion}",
    )
    summary.add_row("чистая экономия (off − on − цена сжатий)", _num(net))
    summary.add_row("кодовое слово вспомнено · compact off", "да" if off.remembered else "нет")
    summary.add_row("кодовое слово вспомнено · compact on", "да" if on.remembered else "нет")
    console.out.print(summary)
    console.out.print(_verdict(net, off.remembered, on.remembered))


def _print_scenario(args: argparse.Namespace) -> None:
    """Scenario and run settings, no network touched at all (`--dry-run`)."""
    scenario = scenario_for(args.turns)
    table = Table(title=f"Сценарий compact_bench, {len(scenario)} ходов (--dry-run, без сети)")
    table.add_column("№", justify="right", style="cyan", no_wrap=True)
    table.add_column("реплика")
    for number, question in enumerate(scenario, start=1):
        table.add_row(str(number), question)
    console.out.print(table)

    keep_last = args.keep_last if args.keep_last is not None else DEFAULT_KEEP_LAST
    compact_every = args.compact_every if args.compact_every is not None else DEFAULT_COMPACT_EVERY
    console.out.print(
        f"модель {args.model} · лимит окна {args.limit} · keep_last {keep_last} · "
        f"compact_every {compact_every} · кодовое слово {CODEWORD!r}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = bench_core.build_parser(
        "Оффлайн-сравнение сжатия истории: один сценарий, compact off против on.",
        default_model=DEFAULT_MODEL,
        default_limit=DEFAULT_LIMIT,
        limit_help=f"заниженный context_limit для обоих прогонов (по умолчанию {DEFAULT_LIMIT})",
        model_help="модель для обоих прогонов",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help=f"сколько последних сообщений остаётся как есть (по умолчанию агента: "
        f"{DEFAULT_KEEP_LAST})",
    )
    parser.add_argument(
        "--compact-every",
        type=int,
        default=None,
        help=f"порог планового сжатия в сообщениях (по умолчанию агента: {DEFAULT_COMPACT_EVERY})",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help=f"сколько ходов сценария прогнать (по умолчанию все {len(SCENARIO)}); "
        "вопрос про кодовое слово всегда остаётся последним",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    console.force_utf8()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit должен быть положительным — иначе не с чем сравнивать trim/compact")

    if args.turns is not None and args.turns < MIN_TURNS:
        parser.error(
            f"--turns не меньше {MIN_TURNS}: короче кодовому слову негде потеряться, "
            "и сравнение мерит пустоту"
        )

    if args.dry_run:
        _print_scenario(args)
        return 0

    try:
        base_config = Config.resolve(model=args.model)
        off_params = _mode_params(
            base_config.params,
            compact=False,
            keep_last=args.keep_last,
            compact_every=args.compact_every,
        )
        on_params = _mode_params(
            base_config.params,
            compact=True,
            keep_last=args.keep_last,
            compact_every=args.compact_every,
        )
    except ConfigError as exc:
        console.warn(str(exc))
        return 2

    off_config = replace(base_config, params=off_params)
    on_config = replace(base_config, params=on_params)

    try:
        models = list_models(off_config)
    except AdventError as exc:
        console.fail(exc)
        return exc.exit_code
    card = find_model(models, off_config.model)
    capabilities = capabilities_of(models, off_config.model)

    # on_notice: a cold-cache tekken download takes minutes — a silent
    # process between launch and the first call reads as hung.
    counter, warning = tokens.counter_for(off_config.model, on_notice=console.note)
    if warning:
        console.warn(warning)

    persona = _persona()
    summary_prompt = _summary_prompt()

    scenario = scenario_for(args.turns)
    # Triggers named out loud, not just in --dry-run: a shortened run only
    # shows anything when they are tuned to its length, and a table whose
    # thresholds are invisible invites comparison with a run that used others.
    keep_last = args.keep_last if args.keep_last is not None else DEFAULT_KEEP_LAST
    compact_every = args.compact_every if args.compact_every is not None else DEFAULT_COMPACT_EVERY
    console.note(
        f"модель {off_config.model} · окно карточки {_num(tokens.context_limit(card))} · "
        f"лимит прогона {args.limit} · ходов {len(scenario)} · "
        f"keep_last {keep_last} · compact_every {compact_every}"
    )

    try:
        console.note("прогон 1/2: compact off")
        off = _run_mode(
            config=off_config,
            limit=args.limit,
            counter=counter,
            capabilities=capabilities,
            persona=persona,
            summary_prompt=summary_prompt,
            scenario=scenario,
        )
        console.note("прогон 2/2: compact on")
        on = _run_mode(
            config=on_config,
            limit=args.limit,
            counter=counter,
            capabilities=capabilities,
            persona=persona,
            summary_prompt=summary_prompt,
            scenario=scenario,
        )
    except AdventError as exc:
        console.fail(exc)
        return exc.exit_code

    _print_report(off, on, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
