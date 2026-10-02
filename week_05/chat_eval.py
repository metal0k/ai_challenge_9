"""Day 25: `adventrag chat-eval` — long scripted dialogues through the real agent.

The agent is the REPL's (memory strategy + cite + rewrite/rerank), built the way
`adventagent` builds it; working memory lives in the process and is never written to
logs/memory. Everything this command pays for is journaled with week=5, day=25.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.cells import cell_len
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console, tokens
from advent_core import scenario as scenario_core
from advent_core.agent import Agent, AgentReply
from advent_core.client import capabilities_of, find_model, list_models, model_names
from advent_core.config import PROJECT_ROOT, ConfigError
from advent_core.errors import AdventError
from advent_core.journal import log_call
from advent_core.memory import (
    MemoryFailure,
    MemorySnapshot,
    MemoryUpdate,
    StructuredMemory,
    render_delta,
    render_memory,
)
from advent_core.rag import RagContext, RagSettings, check_facts
from advent_core.telemetry import CallResult
from advent_core.tokens import counter_for
from week_02 import cli as agent_cli
from week_05 import rag as rag_module
from week_05 import rag_cli

SCENARIOS_PATH = PROJECT_ROOT / "week_05" / "chat_scenarios.json"
CHAT_DAY = 25
KINDS = ("goal", "question", "clarify", "constraint", "term", "switch")
STATE_FIELDS = ("goal", "clarified", "constraints", "terms")
DASH = "—"
ANSWER_LABEL = "   ответ: "

# --- scenarios ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatMessage:
    text: str
    kind: str
    facts: tuple[tuple[str, ...], ...] = ()
    sources: tuple[str, ...] = ()
    expect_state: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    title: str
    messages: tuple[ChatMessage, ...]
    placeholder: bool = False


def _str_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _parse_message(raw: object, where: str) -> ChatMessage:
    if not isinstance(raw, dict):
        raise AdventError(f"{where}: ожидался объект")
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise AdventError(f"{where}: text должен быть непустой строкой")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise AdventError(f"{where}: kind должен быть одним из {', '.join(KINDS)}")
    facts = raw.get("facts", [])
    if not isinstance(facts, list) or any(not _str_list(alts) for alts in facts):
        raise AdventError(f"{where}: facts должен быть списком непустых списков строк")
    sources = raw.get("sources", [])
    if not isinstance(sources, list) or any(not isinstance(s, str) or not s for s in sources):
        raise AdventError(f"{where}: sources должен быть списком строк")
    expect = raw.get("expect_state", {})
    if not isinstance(expect, dict) or any(
        name not in STATE_FIELDS or not _str_list(needles) for name, needles in expect.items()
    ):
        raise AdventError(
            f"{where}: expect_state должен быть объектом "
            f"{{поле: [подстрока, …]}}, поля из {', '.join(STATE_FIELDS)}"
        )
    return ChatMessage(
        text=text,
        kind=kind,
        facts=tuple(tuple(alts) for alts in facts),
        sources=tuple(sources),
        expect_state={name: tuple(needles) for name, needles in expect.items()},
    )


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    """Load and validate the scenario file; every problem is an AdventError, not a traceback."""
    path = path or SCENARIOS_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdventError(f"Файл сценариев не найден: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise AdventError(f"Файл сценариев повреждён ({path.name}): {exc}") from exc
    items = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise AdventError(f"Файл сценариев ({path.name}): ожидался непустой список scenarios.")
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for n, item in enumerate(items, start=1):
        where = f"Файл сценариев ({path.name}), сценарий {n}"
        if not isinstance(item, dict):
            raise AdventError(f"{where}: ожидался объект")
        sid, title = item.get("id"), item.get("title")
        if not isinstance(sid, str) or not sid.strip() or not isinstance(title, str):
            raise AdventError(f"{where}: id и title должны быть строками")
        if sid in seen:
            raise AdventError(f"Файл сценариев ({path.name}): id повторяются.")
        seen.add(sid)
        raw_messages = item.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise AdventError(f"{where}: messages должен быть непустым списком")
        messages = tuple(
            _parse_message(raw, f"{where}, сообщение {i}")
            for i, raw in enumerate(raw_messages, start=1)
        )
        scenarios.append(
            Scenario(sid, title, messages, placeholder=item.get("placeholder") is True)
        )
    return scenarios


# --- agent -------------------------------------------------------------------------------


def build_chat_agent(
    model: str | None,
    db_path: Path | None = None,
    *,
    strategy: str = "structure",
    k: int = 5,
    k_before: int = 20,
    threshold: float = 5.0,
) -> Agent:
    """The REPL's agent (week_02/cli.py AgentShell.__post_init__) with memory + cite on.

    Without the extractor prompt, the exact counter and the model card (window,
    capabilities) the memory strategy would switch itself off with a warning.
    """
    config = rag_cli.make_config(model, strategy, k, k_before, threshold)
    params = config.params
    params.rag = params.rag_rewrite = params.rag_rerank = params.rag_cite = True
    params.context_strategy = "memory"
    try:
        models = list_models(config)
    except AdventError:
        models = []
    names = model_names(models)
    if names and config.model not in names:
        raise ConfigError(
            console.model_unavailable_text(config.model, sorted(names), base_url=config.base_url)
        )
    card = find_model(models, config.model) if models else None
    capabilities = capabilities_of(models, config.model) if models else None
    counter, warning = counter_for(config.model, on_notice=console.note)
    if warning:
        console.warn(warning)
    return Agent(
        config,
        complete=chat_core.complete,
        stream=chat_core.stream,
        counter=counter,
        capabilities=capabilities,
        context_limit=tokens.context_limit(card),
        persona=agent_cli._persona(config),
        dialog_preset=agent_cli._dialog_preset(),
        summary_prompt=agent_cli._summary_prompt(),
        facts_prompt=agent_cli._facts_prompt(),
        memory_prompt=agent_cli._memory_prompt(),
        on_warning=console.warn,
        retrieve=rag_module.make_retriever(db_path, day=CHAT_DAY, aux_day=CHAT_DAY),
    )


# --- tokens: (prompt, completion) pairs, None = unknown, never zero ------------------------

Pair = tuple[int | None, int | None]
ZERO: Pair = (0, 0)


def _add(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else a + b


def _add_pair(a: Pair, b: Pair) -> Pair:
    return (_add(a[0], b[0]), _add(a[1], b[1]))


def _usage_pair(result: object) -> Pair:
    usage = getattr(result, "usage", None)
    if usage is None:
        return (None, None)
    return (usage.prompt_tokens, usage.completion_tokens)


def _n(value: int | None) -> str:
    return DASH if value is None else str(value)


def _fmt_pair(pair: Pair) -> str:
    return f"{_n(pair[0])}/{_n(pair[1])}"


@dataclass(frozen=True, slots=True)
class TurnTokens:
    answer: Pair = ZERO
    aux: Pair = ZERO  # rewrite + rerank
    embed: int | None = 0
    extractor: Pair = ZERO


# --- one run -----------------------------------------------------------------------------


@dataclass(slots=True)
class TurnRow:
    n: int
    message: ChatMessage
    error: str | None = None
    answered: bool = False
    refusal: str = ""  # short reason tag of a refusal, "" otherwise
    quotes: tuple[int, int] = (0, 0)  # verbatim / total
    facts: tuple[bool, ...] = ()
    goal_blocked: int = 0  # automatic goal rewrites stopped by the pin (!goal in the delta)
    expect_ok: bool | None = None  # None: the message has no expect_state
    goal: tuple[tuple[str, str], ...] = ()  # goal entries after the turn
    goal_pinned: bool = False
    tokens: TurnTokens = field(default_factory=TurnTokens)

    @property
    def done(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class ScenarioResult:
    scenario: Scenario
    rows: list[TurnRow]
    memory: MemorySnapshot


def expect_state_met(expect: dict[str, tuple[str, ...]], working: StructuredMemory) -> bool:
    """Every named field holds an entry VALUE with one of its substrings (keys never count)."""
    for name, needles in expect.items():
        haystacks = [
            value.casefold() for key, value in working.entries.items() if key.startswith(name + ".")
        ]
        if not any(needle.casefold() in text for needle in needles for text in haystacks):
            return False
    return True


def _log_memory(item: MemoryUpdate | MemoryFailure, scenario_id: str, n: int) -> Pair:
    """Journal a paid extractor call (as the REPL does, but day=25) and return its usage."""
    call = item.call_result
    if call is None:
        return ZERO
    if isinstance(item, MemoryFailure):
        extra: dict[str, object] = {"kind": "memory_failed", "reason": item.reason}
    else:
        extra = {
            "kind": "memory",
            "covered": item.memory_upto,
            "applied": len(item.applied),
            "blocked": len(item.blocked),
            "rejected": len(item.rejected),
        }
    extra.update(command="chat_eval", scenario=scenario_id, turn=n)
    log_call(
        call,
        getattr(call, "sent_messages", None) or [],
        week=rag_module.RAG_WEEK,
        day=CHAT_DAY,
        extra=extra,
    )
    return _usage_pair(call)


class _RetrievalMeter:
    """Wraps the agent's retriever: a paid retrieval survives a failed answer call."""

    def __init__(self, inner: Callable[[str, RagSettings], RagContext]) -> None:
        self.inner = inner
        self.calls = 0
        self.aux: Pair = ZERO
        self.embed: int | None = 0

    def __call__(self, question: str, settings: RagSettings) -> RagContext:
        ctx = self.inner(question, settings)
        self.calls += 1
        self.aux = _add_pair(self.aux, (ctx.aux_prompt_tokens, ctx.aux_completion_tokens))
        self.embed = _add(self.embed, ctx.embed_tokens)
        return ctx


def _ask_turn(
    agent: Agent, state: scenario_core.ScenarioState, scenario_id: str, n: int, text: str
) -> tuple[AgentReply | None, str | None, Pair]:
    """One ask with a single retry on a transient error; paid extractor calls are salvaged."""
    spent: Pair = ZERO
    for attempt in (1, 2):
        try:
            return scenario_core.run_turn(agent, state, text), None, spent
        except AdventError as error:
            update, failure = scenario_core.salvage_pending_memory(agent, state)
            for item in (update, failure):
                if item is not None:
                    spent = _add_pair(spent, _log_memory(item, scenario_id, n))
            if attempt == 2 or error.exit_code not in rag_cli.TRANSIENT_EXIT_CODES:
                log_call(
                    CallResult(model_requested=agent.config.model),
                    [{"role": "user", "content": text}],
                    week=rag_module.RAG_WEEK,
                    day=CHAT_DAY,
                    error=error.message,
                    extra={"command": "chat_eval", "scenario": scenario_id, "turn": n},
                )
                return None, error.message, spent
            console.warn(f"ход {n}: {error.message} — повтор")
            rag_cli._pause(rag_cli.RETRY_PAUSE_S)
    raise AssertionError("unreachable")  # pragma: no cover


def run_chat_turn(
    agent: Agent,
    state: scenario_core.ScenarioState,
    scenario_id: str,
    n: int,
    message: ChatMessage,
) -> tuple[TurnRow, AgentReply | None]:
    """Ask, journal, score; the reply is returned only for printing."""
    inner = getattr(agent, "retrieve", None)
    meter = _RetrievalMeter(inner) if inner is not None else None
    if meter is not None:
        agent.retrieve = meter
    try:
        reply, error, salvaged = _ask_turn(agent, state, scenario_id, n, message.text)
    finally:
        if meter is not None:
            agent.retrieve = inner
    working = state.memory.working if state.memory is not None else StructuredMemory()
    row = TurnRow(n=n, message=message, error=error)
    goal_keys = sorted(k for k in working.entries if k.startswith("goal."))
    row.goal = tuple((k, working.entries[k]) for k in goal_keys)
    row.goal_pinned = bool(goal_keys) and all(k in working.pinned for k in goal_keys)
    row.facts = (False,) * len(message.facts)
    if message.expect_state:
        row.expect_ok = error is None and expect_state_met(message.expect_state, working)
    if reply is None:
        row.tokens = TurnTokens(
            aux=meter.aux if meter else ZERO,
            embed=meter.embed if meter else 0,
            extractor=salvaged,
        )
        return row, None

    extractor = salvaged
    for item in (reply.memory_update, reply.memory_failed):
        if item is not None:
            extractor = _add_pair(extractor, _log_memory(item, scenario_id, n))
    answer_tokens: Pair = ZERO
    if reply.model_called:
        answer_tokens = _usage_pair(reply.result)
        log_call(
            reply.result,
            reply.result.sent_messages or [{"role": "user", "content": message.text}],
            week=rag_module.RAG_WEEK,
            day=CHAT_DAY,
            extra={"command": "chat_eval", "scenario": scenario_id, "turn": n, "mode": "cite"},
        )
    ctx = reply.rag
    if meter is not None and meter.calls:
        aux, embed = meter.aux, meter.embed  # every attempt's retrieval, retries included
    else:
        aux = ZERO if ctx is None else (ctx.aux_prompt_tokens, ctx.aux_completion_tokens)
        embed = 0 if ctx is None else ctx.embed_tokens

    cited = reply.cited
    row.answered = bool(cited is not None and cited.quoted)
    if cited is not None:
        row.quotes = (cited.verified_quotes, len(cited.quotes))
        row.refusal = "" if row.answered else (rag_cli._refusal_tag(cited) or "не знаю")
    else:
        row.refusal = "формат"
    if row.answered:
        row.facts = check_facts(cited.answer, message.facts)
    if reply.memory_update is not None:
        row.goal_blocked = sum(1 for _, op, _ in reply.memory_update.blocked if op.field == "goal")
    row.tokens = TurnTokens(answer_tokens, aux, embed, extractor)
    return row, reply


# --- live output (stdout: the demonstration) -----------------------------------------------


def _cell_cut(text: str, limit: int) -> str:
    return rag_cli._cut(" ".join(text.split()), limit)


def _line(text: str) -> None:
    rag_cli._print_line(_cell_cut(text, console.out.width))


def _marks(flags: Sequence[bool]) -> str:
    return "".join("✓" if flag else "✗" for flag in flags)


def print_turn(
    scenario: Scenario,
    row: TurnRow,
    total: int,
    reply: AgentReply | None,
    memory: MemorySnapshot,
    *,
    detail: bool,
) -> None:
    width = console.out.width
    message = row.message
    if not detail:
        head = f"{row.n}/{total} "
        if row.error is not None:
            tail = " · ошибка"
        else:
            src = "источник ✓" if row.answered else f"не знаю·{row.refusal}"
            tail = f" · {src} · факты {sum(row.facts)}/{len(row.facts)}"
        question = _cell_cut(message.text, width - len(head) - cell_len(tail))
        rag_cli._print_line(head + question + tail)
        return
    console.out.print(
        f"── {scenario.id} · ход {row.n}/{total} · {message.kind} ──",
        style="bold",
        markup=False,
        highlight=False,
    )
    console.out.print(f"Вопрос: {message.text}", markup=False, highlight=False)
    if row.error is not None:
        _line(f"   ошибка: {row.error}")
        console.out.print()
        return
    cited = reply.cited if reply is not None else None
    if row.answered and cited is not None:
        room = width - cell_len(ANSWER_LABEL) - 1
        text = rag_cli._snippet(cited.answer, message.facts, room)
        _line(f"{ANSWER_LABEL}«{text}»")
        hits = reply.rag.hits if reply is not None and reply.rag is not None else ()
        for quote in cited.quotes:
            source = hits[quote.n - 1].source if 1 <= quote.n <= len(hits) else "?"
            mark = "✓" if quote.verified else "✗"
            _line(f"   [{quote.n}] {mark} {source} «{quote.text}»")
    else:
        _line(f"   {rag_cli._refusal_line(cited)}")
    if reply is not None and reply.memory_update is not None:
        delta = render_delta(reply.memory_update)
        if delta:
            _line(f"   {delta}")
    if reply is not None and reply.memory_failed is not None:
        _line(f"   memory: не обновлена ({reply.memory_failed.reason})")
    console.out.print(
        agent_cli._task_line(memory, width), highlight=False, emoji=False, no_wrap=True
    )
    parts = []
    if message.facts:
        parts.append(f"факты {_marks(row.facts)}")
    if row.expect_ok is not None:
        parts.append(f"state {'✓' if row.expect_ok else '✗'}")
    if row.goal_blocked:
        parts.append(f"цель: переписать пытались {row.goal_blocked}, закреплена")
    if parts:
        _line("   " + " · ".join(parts))
    console.out.print()


def run_scenario_chat(scenario: Scenario, agent: Agent, *, detail: bool = True) -> ScenarioResult:
    """One scenario on a fresh agent; every turn carries history/transcript/memory like the REPL."""
    state = scenario_core.ScenarioState(memory=MemorySnapshot())
    rows: list[TurnRow] = []
    total = len(scenario.messages)
    for n, message in enumerate(scenario.messages, start=1):
        row, reply = run_chat_turn(agent, state, scenario.id, n, message)
        rows.append(row)
        print_turn(scenario, row, total, reply, state.memory or MemorySnapshot(), detail=detail)
    return ScenarioResult(scenario, rows, state.memory or MemorySnapshot())


# --- metrics and the closing screen --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoalStability:
    pinned_from: int | None
    stable: bool


def goal_stability(rows: Sequence[TurnRow]) -> GoalStability:
    """Pinned from turn N and unchanged (same keys, same text, still pinned) to the end."""
    start = next((r for r in rows if r.goal and r.goal_pinned), None)
    if start is None:
        return GoalStability(None, False)
    stable = all(r.goal_pinned and r.goal == start.goal for r in rows if r.n >= start.n)
    return GoalStability(start.n, stable)


@dataclass(frozen=True, slots=True)
class Metrics:
    planned: int
    done: int
    with_source: int
    refusals: dict[str, int]
    facts_found: int
    facts_total: int
    goal: GoalStability
    goal_blocked: int
    after_switch: tuple[bool, ...]  # per "switch" message that has a next turn
    state_ok: int
    state_total: int
    errors: int


def after_switch(rows: Sequence[TurnRow]) -> tuple[bool, ...]:
    """For each "switch" message with a next turn: is that turn a sourced answer with a fact hit."""
    return tuple(
        nxt.done and nxt.answered and nxt.quotes[0] >= 1 and sum(nxt.facts) >= 1
        for row, nxt in zip(rows, rows[1:], strict=False)
        if row.message.kind == "switch"
    )


def scenario_metrics(result: ScenarioResult) -> Metrics:
    rows = result.rows
    done = [r for r in rows if r.done]
    answered = [r for r in done if r.answered]
    refusals = Counter(r.refusal for r in done if not r.answered)
    return Metrics(
        planned=len(rows),
        done=len(done),
        with_source=sum(1 for r in answered if r.quotes[0] >= 1),
        refusals=dict(refusals),
        facts_found=sum(sum(r.facts) for r in rows),
        facts_total=sum(len(r.message.facts) for r in rows),
        goal=goal_stability(rows),
        goal_blocked=sum(r.goal_blocked for r in rows),
        after_switch=after_switch(rows),
        state_ok=sum(1 for r in rows if r.expect_ok),
        state_total=sum(1 for r in rows if r.expect_ok is not None),
        errors=len(rows) - len(done),
    )


def _goal_cell(goal: GoalStability) -> str:
    if goal.pinned_from is None:
        return DASH
    return f"✓ ход {goal.pinned_from}" if goal.stable else "✗"


def summary_table(results: Sequence[ScenarioResult]) -> Table:
    table = Table(title="Мини-чат: сценарии", title_justify="left")
    for name, justify in (
        ("сценарий", "left"),
        ("ходов", "right"),
        ("источник", "right"),
        ("не знаю", "right"),
        ("факты", "right"),
        ("после побочного", "left"),
        ("цель", "left"),
    ):
        table.add_column(name, justify=justify, no_wrap=True)
    for result in results:
        m = scenario_metrics(result)
        table.add_row(
            result.scenario.id,
            f"{m.done}/{m.planned}",
            f"{m.with_source}/{m.done}",
            str(sum(m.refusals.values())),
            f"{m.facts_found}/{m.facts_total}",
            _marks(m.after_switch) or DASH,
            _goal_cell(m.goal),
        )
    return table


def total_tokens(results: Sequence[ScenarioResult]) -> dict[str, Pair | int | None]:
    """Per-category sums over every turn; one unknown value makes that category unknown."""
    rows = [r for result in results for r in result.rows]
    sums: dict[str, Pair | int | None] = {}
    for name in ("answer", "aux", "extractor"):
        total: Pair = ZERO
        for r in rows:
            total = _add_pair(total, getattr(r.tokens, name))
        sums[name] = total
    embed: int | None = 0
    for r in rows:
        embed = _add(embed, r.tokens.embed)
    sums["embed"] = embed
    return sums


def _out(text: str) -> None:
    console.out.print(text, markup=False, highlight=False)


def print_summary(results: Sequence[ScenarioResult]) -> None:
    """The closing screen: table, per-scenario details, final task state, tokens, caveat."""
    console.out.print(summary_table(results))
    metrics = [scenario_metrics(r) for r in results]
    for result, m in zip(results, metrics, strict=True):
        refused = ", ".join(f"{tag} {count}" for tag, count in sorted(m.refusals.items()))
        goal = m.goal
        goal_text = (
            "не закреплена"
            if goal.pinned_from is None
            else f"закреплена с хода {goal.pinned_from}, "
            + ("не менялась" if goal.stable else "менялась или снята")
        )
        _out(
            f"{result.scenario.id}: цель {goal_text} · state по expect_state "
            f"{m.state_ok}/{m.state_total}"
            f" · попыток переписать цель: {m.goal_blocked}"
            + (f" · не знаю: {refused}" if refused else "")
            + (f" · ходов с ошибкой {m.errors}" if m.errors else "")
        )
    for result in results:
        console.out.print()
        _out(f"Итоговый task state — {result.scenario.id}:")
        for row in render_memory(result.memory, layer="working").splitlines()[1:]:
            _line("  " + row.strip())
        if not result.memory.working.entries:
            _out("  пусто")
    console.out.print()
    t = total_tokens(results)
    _out("Токены prompt/completion по вызовам:")
    _out(f"  ответ {_fmt_pair(t['answer'])} · rewrite+rerank {_fmt_pair(t['aux'])}")
    _out(f"  extractor {_fmt_pair(t['extractor'])} · embed {_n(t['embed'])}")
    if any(r.scenario.placeholder for r in results):
        _out("Сценарии с placeholder: факты ещё не сверены с живым прогоном.")
    _out(
        "Один прогон на сценарий: цифры — наблюдение, не оценка. Метрики цели и факты "
        "считает код, без модели-судьи."
    )


def run_chat_eval(
    scenarios_path: Path | None = None,
    scenario_id: str | None = None,
    model: str | None = None,
    db: str | None = None,
    detail: bool = True,
) -> list[ScenarioResult]:
    """Run the chosen scenarios (all by default) and print the result."""
    available = load_scenarios(scenarios_path)
    if scenario_id is not None:
        chosen = [s for s in available if s.id == scenario_id]
        if not chosen:
            raise AdventError(
                f"Нет сценария {scenario_id!r}.",
                hint="Доступны: " + ", ".join(s.id for s in available) + ".",
            )
    else:
        chosen = available
    db_path = Path(db) if db is not None else None
    rag_module.check_index(db_path, "structure")
    results: list[ScenarioResult] = []
    for scenario in chosen:
        if scenario.placeholder:
            console.note(f"сценарий {scenario.id}: факты не сверены с живым прогоном (placeholder)")
        agent = build_chat_agent(model, db_path)
        console.out.print()
        console.out.print(
            f"Сценарий «{scenario.title}» — {len(scenario.messages)} сообщений",
            style="bold cyan",
            markup=False,
            highlight=False,
        )
        results.append(run_scenario_chat(scenario, agent, detail=detail))
    console.out.print()
    print_summary(results)
    return results


def register(app: typer.Typer) -> None:
    """Register the chat-eval command."""

    @app.command("chat-eval")
    def chat_eval_command(
        scenario: str | None = typer.Option(
            None, "--scenario", help="id сценария; по умолчанию все."
        ),
        detail: bool = typer.Option(
            True,
            "--detail/--no-detail",
            help="Каждый ход подробно: ответ, источники, дельта memory, задача.",
        ),
        model: str | None = typer.Option(None, "--model", help="Модель чата."),
        db: str | None = typer.Option(None, "--db", help="Путь к индексу."),
        scenarios: str | None = typer.Option(None, "--scenarios", help="Файл сценариев."),
    ) -> None:
        """Мини-чат с RAG, цитатами и памятью задачи: длинные сценарии, цель и источники."""
        run_chat_eval(
            Path(scenarios) if scenarios is not None else None, scenario, model, db, detail
        )
