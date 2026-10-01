"""Days 22-24: `adventrag ask` / `eval` / `stages` — one Agent: RAG off, plain, full, cite."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.cells import cell_len
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console
from advent_core.agent import Agent
from advent_core.config import PROJECT_ROOT, Config
from advent_core.errors import AdventError, NetworkError, RateLimitError, ServerError
from advent_core.journal import log_call
from advent_core.rag import (
    RAG_UNKNOWN_PREFIX,
    RAG_UNKNOWN_TEXTS,
    CitedAnswer,
    RagContext,
    RagHit,
    RagSettings,
    check_facts,
    fact_span,
)
from week_05 import index as index_module
from week_05 import rag as rag_module

QUESTIONS_PATH = PROJECT_ROOT / "week_05" / "rag_questions.json"
UNANSWERABLE_PATH = PROJECT_ROOT / "week_05" / "rag_unanswerable.json"
RAG_STRATEGIES = ("fixed", "structure")
MODE_LABELS = {"off": "без RAG", "plain": "RAG", "full": "RAG+rerank", "cite": "RAG+цитаты"}
ASK_MODES = ("both", "rag", "no-rag", "full", "cite")
ALLOWED_PAIRS = (
    ("off", "plain"),
    ("off", "full"),
    ("plain", "full"),
    ("off", "cite"),
    ("full", "cite"),
)
DEFAULT_PAIR = "off,plain"
RETRY_PAUSE_S = 5.0
TRANSIENT_EXIT_CODES = (
    RateLimitError.exit_code,
    ServerError.exit_code,
    NetworkError.exit_code,
)
JOURNAL_DAY_PLAIN = 22
JOURNAL_DAY_FULL = 23
JOURNAL_DAY_CITE = 24
MODE_DAYS = {"off": 22, "plain": 22, "full": 23, "cite": 24}


def _pause(seconds: float) -> None:
    """Module-level seam: tests replace it so a retry never sleeps."""
    time.sleep(seconds)


def parse_pair(pair: str) -> tuple[str, str]:
    """Exactly two different modes, the second never `off`."""
    parts = tuple(part.strip() for part in pair.split(","))
    if parts not in ALLOWED_PAIRS:
        raise AdventError(
            f"Недопустимая пара режимов: {pair!r}.",
            hint="Доступно: off,plain · off,full · plain,full · off,cite · full,cite.",
        )
    return parts


def journal_day(modes: tuple[str, ...]) -> int:
    if "cite" in modes:
        return JOURNAL_DAY_CITE
    return JOURNAL_DAY_FULL if "full" in modes else JOURNAL_DAY_PLAIN


def mode_days(modes: tuple[str, ...]) -> dict[str, int]:
    """Journal day per mode. A pair with `cite` splits days by mode (full 23, cite 24);
    older pairs keep one day for both, as on days 22-23."""
    if "cite" in modes:
        return {m: MODE_DAYS[m] for m in modes}
    day = journal_day(modes)
    return {m: day for m in modes}


@dataclass(frozen=True, slots=True)
class ModeRun:
    text: str
    ctx: RagContext | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int
    mode: str = "off"
    cited: CitedAnswer | None = None
    display_text: str | None = None
    model_called: bool = True


def check_args(strategy: str, k: int, k_before: int, threshold: float) -> None:
    if strategy not in RAG_STRATEGIES:
        raise AdventError(f"Неизвестная стратегия: {strategy!r}. Доступны: fixed, structure.")
    if not 1 <= k <= 20:
        raise AdventError(f"-k должно быть от 1 до 20, получено {k}.")
    if not 1 <= k_before <= 50:
        raise AdventError(f"--k-before должно быть от 1 до 50, получено {k_before}.")
    if not 0 <= threshold <= 10:
        raise AdventError(f"--threshold должен быть от 0 до 10, получено {threshold}.")


def make_config(
    model: str | None,
    strategy: str,
    k: int,
    k_before: int = 20,
    threshold: float = 5.0,
) -> Config:
    check_args(strategy, k, k_before, threshold)
    config = Config.resolve(model=model, stream=False)
    config.params.rag_strategy = strategy
    config.params.rag_k = k
    config.params.rag_k_before = k_before
    config.params.rag_threshold = float(threshold)
    config.params.rag = False
    config.params.rag_rewrite = False
    config.params.rag_rerank = False
    config.params.rag_cite = False
    return config


def _yes_no(flags: tuple[bool, ...]) -> str:
    """Whether at least one of the alternative expected sources is present."""
    return "да" if any(flags) else "нет"


# Every question runs on an empty history with no context_limit wired in, so this
# one fires on every ask and says nothing the user can act on.
_NOISE_WARNING = "max_context_length модели неизвестен"


def _agent_warning(text: str) -> None:
    """Agent warnings to stderr; the unknown-window one is noise here."""
    if text.startswith(_NOISE_WARNING):
        return
    console.warn(text)


def build_agent(config: Config, db_path: Path | None, day: int = JOURNAL_DAY_PLAIN) -> Agent:
    return Agent(
        config,
        on_warning=_agent_warning,
        complete=chat_core.complete,
        stream=chat_core.stream,
        retrieve=rag_module.make_retriever(db_path, day=day, aux_day=day),
    )


def build_agents(config: Config, db_path: Path | None, modes: tuple[str, ...]) -> dict[str, Agent]:
    """One Agent per distinct journal day, shared by the modes that use that day."""
    by_day: dict[int, Agent] = {}
    agents: dict[str, Agent] = {}
    for mode, day in mode_days(modes).items():
        if day not in by_day:
            by_day[day] = build_agent(config, db_path, day)
        agents[mode] = by_day[day]
    return agents


def run_mode(
    agent: Agent,
    question: str,
    *,
    mode: str,
    command: str,
    question_id: int | None = None,
    day: int = JOURNAL_DAY_PLAIN,
) -> ModeRun:
    params = agent.config.params
    params.rag = mode != "off"
    params.rag_rewrite = params.rag_rerank = mode in ("full", "cite")
    params.rag_cite = mode == "cite"
    reply = agent.ask(question, [])
    extra: dict[str, object] = {
        "command": command,
        "rag": mode != "off",
        "question_id": question_id,
    }
    if mode in ("full", "cite"):
        extra["mode"] = mode
    # A refusal made by code has a synthetic CallResult: nothing was sent, nothing to journal.
    if reply.model_called:
        log_call(
            reply.result,
            reply.result.sent_messages or [],
            week=rag_module.RAG_WEEK,
            day=day,
            extra=extra,
        )
    # ctx.warnings already reached stderr through the agent's on_warning.
    return ModeRun(
        text=reply.text,
        ctx=reply.rag,
        prompt_tokens=reply.result.usage.prompt_tokens,
        completion_tokens=reply.result.usage.completion_tokens,
        latency_ms=reply.result.latency_ms,
        mode=mode,
        cited=reply.cited,
        display_text=reply.display_text,
        model_called=reply.model_called,
    )


def _tokens(value: int | None) -> str:
    return "?" if value is None else str(value)


def _num(value: float) -> str:
    return f"{value:g}"


def _print_line(text: str) -> None:
    """Print one line that must never wrap (the caller sizes it to the width)."""
    console.out.print(text, markup=False, highlight=False, no_wrap=True, overflow="ellipsis")


def print_sources(ctx: RagContext, mode: str = "plain") -> None:
    full = mode == "full"
    if full and ctx.rewritten is not None:
        _print_line(_cut(f"rewrite: {' '.join(ctx.rewritten.split())}", console.out.width))
    if not ctx.hits:
        console.out.print("источники: ничего не найдено", markup=False, highlight=False)
    else:
        console.out.print("источники:", markup=False, highlight=False)
        for n, hit in enumerate(ctx.hits, start=1):
            label = f"{hit.source} — {hit.section}" if hit.section.strip() else hit.source
            if full:
                rerank = "?" if hit.rerank is None else _num(hit.rerank)
                was = "—" if hit.rank is None else f"#{hit.rank}"
                head = f"  [{n}] cos {hit.score:.3f} · rerank {rerank} · был {was}  "
            else:
                head = f"  [{n}] {hit.score:.3f}  "
            _print_line(head + label)
    if ctx.dropped > 0:
        console.out.print(f"  не вошло в контекст: {ctx.dropped}", markup=False, highlight=False)
    if full and ctx.passed is not None:
        console.out.print(
            f"кандидатов {ctx.candidates} → прошли порог {ctx.passed} → "
            f"в контексте {len(ctx.hits)}",
            markup=False,
            highlight=False,
        )


def _quotes_note(cited: CitedAnswer) -> str:
    return f"цитаты: {cited.verified_quotes}/{len(cited.quotes)} дословны"


def _aux_tokens_text(ctx: RagContext | None) -> str:
    if ctx is None:
        return ""
    return f"aux {_tokens(ctx.aux_prompt_tokens)}+{_tokens(ctx.aux_completion_tokens)} ток."


def print_cite_answer(run: ModeRun) -> None:
    """Cite screen: the rendered answer already lists sources with chunk_id and quotes ✓/✗."""
    console.out.print(
        (run.display_text or run.text).strip(), markup=False, highlight=False, emoji=False
    )
    ctx = run.ctx
    if ctx is None:
        return
    if ctx.rewritten is not None:
        _print_line(_cut(f"rewrite: {' '.join(ctx.rewritten.split())}", console.out.width))
    if ctx.passed is not None:
        console.out.print(
            f"кандидатов {ctx.candidates} → прошли порог {ctx.passed} → "
            f"в контексте {len(ctx.hits)}",
            markup=False,
            highlight=False,
        )


def _cite_note(run: ModeRun) -> str:
    label = MODE_LABELS["cite"]
    if not run.model_called:
        return (
            f"{label}: не знаю · модель не вызывалась · {_aux_tokens_text(run.ctx)} · "
            f"embed {_tokens(run.ctx.embed_tokens) if run.ctx else '?'}"
        )
    parts = [
        f"{label}: prompt {_tokens(run.prompt_tokens)} · "
        f"completion {_tokens(run.completion_tokens)}"
    ]
    if run.ctx is not None:
        parts.append(f"embed {_tokens(run.ctx.embed_tokens)}")
        parts.append(_aux_tokens_text(run.ctx))
    parts.append(f"{run.latency_ms / 1000:.1f} с")
    return " · ".join(parts)


def print_mode_answer(mode: str, run: ModeRun) -> None:
    label = MODE_LABELS[mode]
    console.out.print(f"── {label} ──", style="bold", markup=False, highlight=False)
    if mode == "cite":
        print_cite_answer(run)
        console.out.print()
        console.note(_cite_note(run))
        # Own line: appended to the token line it wraps the latency off at 80 columns.
        if run.model_called and run.cited is not None and run.cited.quotes:
            console.note(_quotes_note(run.cited))
        return
    console.out.print(run.text.strip(), markup=False, highlight=False)
    if run.ctx is not None:
        print_sources(run.ctx, mode)
    console.out.print()
    embed = f" · embed {_tokens(run.ctx.embed_tokens)}" if run.ctx is not None else ""
    aux = ""
    if mode == "full" and run.ctx is not None:
        aux = (
            f" · aux {_tokens(run.ctx.aux_prompt_tokens)}+"
            f"{_tokens(run.ctx.aux_completion_tokens)} ток."
        )
    console.note(
        f"{label}: prompt {_tokens(run.prompt_tokens)} · "
        f"completion {_tokens(run.completion_tokens)}{embed}{aux} · {run.latency_ms / 1000:.1f} с"
    )


def resolve_ask_modes(mode: str, pair: str | None) -> tuple[str, ...]:
    """Modes to run for `ask`; `--pair` and the single-mode flags exclude each other."""
    if mode not in ASK_MODES:
        raise AdventError(f"Неизвестный режим: {mode!r}. Доступны: {', '.join(ASK_MODES)}.")
    if pair is not None:
        if mode != "both":
            raise AdventError(
                "--pair нельзя сочетать с --rag/--no-rag/--full.",
                hint="Либо пара режимов, либо один режим.",
            )
        return parse_pair(pair)
    return {
        "both": parse_pair(DEFAULT_PAIR),
        "rag": ("plain",),
        "no-rag": ("off",),
        "full": ("full",),
        "cite": ("cite",),
    }[mode]


def ask_question(
    question: str,
    mode: str = "both",
    strategy: str = "structure",
    k: int = 5,
    db: str | None = None,
    model: str | None = None,
    pair: str | None = None,
    k_before: int = 20,
    threshold: float = 5.0,
) -> None:
    modes = resolve_ask_modes(mode, pair)
    if not question.strip():
        raise AdventError("Пустой вопрос.")
    db_path = Path(db) if db is not None else None
    config = make_config(model, strategy, k, k_before, threshold)
    if any(m != "off" for m in modes):
        rag_module.check_index(db_path, strategy)
    days = mode_days(modes)
    agents = build_agents(config, db_path, modes)
    for m in modes:
        run = run_mode(agents[m], question, mode=m, command="ask", day=days[m])
        print_mode_answer(m, run)


def register(app: typer.Typer) -> None:
    @app.command("ask")
    def ask_command(
        question: str = typer.Argument(..., help="Вопрос по базе."),
        rag: bool | None = typer.Option(
            None, "--rag/--no-rag", help="Только с RAG или только без; по умолчанию оба режима."
        ),
        full: bool = typer.Option(False, "--full", help="Только RAG+rerank (rewrite + rerank)."),
        cite: bool = typer.Option(
            False, "--cite", help="Только RAG+цитаты (rewrite + rerank + JSON с цитатами)."
        ),
        pair: str | None = typer.Option(
            None,
            "--pair",
            help="Пара: off,plain (по умолчанию) | off,full | plain,full | off,cite | full,cite.",
        ),
        strategy: str = typer.Option("structure", "--strategy", help="fixed | structure."),
        k: int = typer.Option(5, "-k", "--top-k", min=1, help="Сколько чанков прикладывать."),
        k_before: int = typer.Option(
            20, "--k-before", min=1, help="Кандидатов на запрос до rerank (режим full)."
        ),
        threshold: float = typer.Option(
            5.0, "--threshold", help="Порог оценки reranker'а 0..10 (режим full)."
        ),
        db: str | None = typer.Option(None, "--db", help="Путь к индексу."),
        model: str | None = typer.Option(None, "--model", help="Модель чата."),
    ) -> None:
        """Один вопрос — ответы в двух режимах рядом (без RAG / RAG / RAG+rerank)."""
        if full and rag is not None:
            raise AdventError("--full нельзя сочетать с --rag/--no-rag.")
        if cite and (full or rag is not None):
            raise AdventError("--cite нельзя сочетать с --rag/--no-rag/--full.")
        mode = (
            "cite"
            if cite
            else "full"
            if full
            else "both"
            if rag is None
            else ("rag" if rag else "no-rag")
        )
        ask_question(question, mode, strategy, k, db, model, pair, k_before, threshold)


QUESTION_CELL = 30


@dataclass(frozen=True, slots=True)
class EvalRow:
    question: rag_module.ControlQuestion
    first: ModeRun | None
    second: ModeRun | None
    first_score: rag_module.AnswerScore | None
    second_score: rag_module.AnswerScore | None
    error: str | None = None
    grounding: rag_module.Grounding | None = None
    judge_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _cut(text: str, limit: int) -> str:
    """Cut text to a fixed number of terminal cells."""
    if cell_len(text) <= limit:
        return text
    out = ""
    for ch in text:
        if cell_len(out + ch) > limit - 1:
            break
        out += ch
    return out + "…"


def _sum(values: list[int | None]) -> int | None:
    """Sum values, returning None if any value is None."""
    if any(value is None for value in values):
        return None
    return sum(values)


def _ratio(flags: tuple[bool, ...]) -> str:
    """Format a boolean tuple as found/total."""
    return f"{sum(flags)}/{len(flags)}"


def _labels(pair: tuple[str, str]) -> tuple[str, str]:
    return MODE_LABELS[pair[0]], MODE_LABELS[pair[1]]


def eval_table(rows: list[EvalRow], pair: tuple[str, str] = ("off", "plain")) -> Table:
    """Build a comparison table for evaluated questions."""
    if pair[1] == "cite":
        return cite_eval_table(rows, pair)
    first, second = _labels(pair)
    table = Table(title=f"Контрольные вопросы: {first} и {second}", title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("вопрос")
    table.add_column(first, justify="right")
    table.add_column(second, justify="right")
    table.add_column("в top-k", justify="right")
    table.add_column("в ответе", justify="right")
    for row in rows:
        question_cell = _cut(" ".join(row.question.question.split()), QUESTION_CELL)
        if not row.ok:
            table.add_row(str(row.question.id), question_cell, "ошибка", "ошибка", "—", "—")
        else:
            table.add_row(
                str(row.question.id),
                question_cell,
                _ratio(row.first_score.facts),
                _ratio(row.second_score.facts),
                _yes_no(row.second_score.sources_retrieved),
                _yes_no(row.second_score.sources_cited),
            )
    return table


def _out(text: str) -> None:
    console.out.print(text, markup=False, highlight=False)


def print_summary(rows: list[EvalRow], pair: tuple[str, str] = ("off", "plain")) -> None:
    """Print aggregate evaluation statistics."""
    first, second = _labels(pair)
    done = [row for row in rows if row.ok]
    total = len(rows)
    if not done:
        _out("Ни одна пара ответов не получена — сравнивать нечего.")
        return
    n = len(done)
    facts_total = sum(len(row.question.expect) for row in done)
    first_facts = sum(row.first_score.facts_found for row in done)
    second_facts = sum(row.second_score.facts_found for row in done)
    first_full = sum(row.first_score.complete for row in done)
    second_full = sum(row.second_score.complete for row in done)
    retrieved = sum(any(row.second_score.sources_retrieved) for row in done)
    cited = sum(any(row.second_score.sources_cited) for row in done)
    _out(
        f"Фактов найдено: {first} {first_facts}/{facts_total} · "
        f"{second} {second_facts}/{facts_total}"
    )
    _out(f"Полных ответов: {first} {first_full}/{n} · {second} {second_full}/{n}")
    _out(f"Ожидаемый источник в top-k: {retrieved}/{n} · назван в ответе: {cited}/{n}")
    # Spend counts every completed call, including the first run of a failed pair.
    runs = [run for row in rows for run in (row.first, row.second) if run is not None]
    first_runs = [run for row in rows if (run := row.first) is not None]
    second_runs = [run for row in rows if (run := row.second) is not None]
    first_p = _sum([run.prompt_tokens for run in first_runs])
    first_c = _sum([run.completion_tokens for run in first_runs])
    second_p = _sum([run.prompt_tokens for run in second_runs])
    second_c = _sum([run.completion_tokens for run in second_runs])
    embed = _sum([run.ctx.embed_tokens for run in runs if run.ctx is not None] or [0])
    tokens = (
        f"Токены prompt/completion: {first} {_tokens(first_p)}/{_tokens(first_c)} · "
        f"{second} {_tokens(second_p)}/{_tokens(second_c)} · embed {_tokens(embed)}"
    )
    full_runs = [run for run in runs if run.mode == "full"]
    if full_runs:
        aux_p = _sum([run.ctx.aux_prompt_tokens if run.ctx else None for run in full_runs])
        aux_c = _sum([run.ctx.aux_completion_tokens if run.ctx else None for run in full_runs])
        tokens += f" · aux {_tokens(aux_p)}/{_tokens(aux_c)}"
    _out(tokens)
    dropped = sum(run.ctx.dropped for run in runs if run.ctx is not None)
    if dropped > 0:
        _out(f"Чанков не вошло в контекст: {dropped}")
    if n < total:
        _out(f"Завершено пар: {n}/{total} — сравнение неполное, вердикта нет.")
    elif second_facts > first_facts:
        _out(f"По фактам в этой выборке выше: {second} ({second_facts} против {first_facts}).")
    elif first_facts > second_facts:
        _out(f"По фактам в этой выборке выше: {first} ({first_facts} против {second_facts}).")
    else:
        _out("Ничья по фактам — явного лидера нет.")
    _out(
        f"Вопросов: {total}, один прогон на ячейку — малая выборка, разница в один факт не значима."
    )


def _snippet(answer: str, facts: tuple[tuple[str, ...], ...] | list, width: int) -> str:
    """Window of at most `width` cells around the first found fact."""
    text = " ".join(answer.split())
    if cell_len(text) <= width:
        return text
    span = None
    for fact in facts:
        # The alternative is collapsed too: spans refer to the collapsed text.
        span = fact_span(text, [" ".join(alt.split()) for alt in fact])
        if span is not None:
            break

    def fill(start: int, budget: int) -> int:
        end, used = start, 0
        while end < len(text) and used + cell_len(text[end]) <= budget:
            used += cell_len(text[end])
            end += 1
        return end

    start = 0
    if span is not None:
        s, e = span
        fact_w = cell_len(text[s:e])
        # Fact first (plus both ellipses), then context: left ~1/3 of what remains.
        rem = width - fact_w - (1 if s > 0 else 0) - (1 if e < len(text) else 0)
        if rem < 0:
            return text[s : fill(s, width - 1)] + "…"
        right_all = cell_len(text[e:])
        left = rem // 3
        if right_all <= rem - left:
            left = rem + (1 if e < len(text) else 0) - right_all
        start = s
        used = 0
        while start > 0 and used + cell_len(text[start - 1]) <= left:
            start -= 1
            used += cell_len(text[start])
    prefix = "…" if start > 0 else ""
    end = fill(start, width - len(prefix))
    if end >= len(text):
        return prefix + text[start:]
    return prefix + text[start : fill(start, width - len(prefix) - 1)] + "…"


def _print_pair(n: int, row: EvalRow, detail: int, pair: tuple[str, str]) -> None:
    """Print one question's result right after its pair of answers."""
    if pair[1] == "cite":
        _print_cite_pair(n, row, detail, pair)
        return
    first, second = _labels(pair)
    width = console.out.width
    head = f"#{row.question.id} "
    if not row.ok or row.first_score is None or row.second_score is None:
        tail = " · ошибка"
        _print_line(
            head
            + _cut(" ".join(row.question.question.split()), width - len(head) - len(tail))
            + tail
        )
        return
    question = " ".join(row.question.question.split())
    if n > detail:
        r1, r2 = _ratio(row.first_score.facts), _ratio(row.second_score.facts)
        tail = f" · {first} {r1} · {second} {r2}"
        _print_line(head + _cut(question, width - len(head) - cell_len(tail)) + tail)
        return
    _print_line(head + _cut(question, width - len(head)))
    expected = " · ".join(" ".join(fact[0].split()) for fact in row.question.expect)
    label = "   ожидается: "
    _print_line(label + _cut(expected, width - cell_len(label)))
    marks_w = max(2, len(row.question.expect))
    label_w = max(cell_len(first), cell_len(second))
    for name, run, score in (
        (first, row.first, row.first_score),
        (second, row.second, row.second_score),
    ):
        marks = "".join("✓" if found else "✗" for found in score.facts)
        lead = f"   {name:<{label_w}} {marks:<{marks_w}} «"
        room = width - cell_len(lead) - 1
        _print_line(f"{lead}{_snippet(run.text, row.question.expect, room)}»")
    console.out.print()


def _retry_transient(call, label: str):
    """One retry after a pause, and only for transient API errors (429/5xx/network)."""
    try:
        return call()
    except AdventError as error:
        if error.exit_code not in TRANSIENT_EXIT_CODES:
            raise
        console.warn(f"{label}: {error.message} — повтор")
        _pause(RETRY_PAUSE_S)
        return call()


def _run_mode_retry(
    agent: Agent,
    question: rag_module.ControlQuestion,
    *,
    mode: str,
    day: int = JOURNAL_DAY_PLAIN,
) -> ModeRun:
    return _retry_transient(
        lambda: run_mode(
            agent,
            question.question,
            mode=mode,
            command="eval",
            question_id=question.id,
            day=day,
        ),
        f"вопрос #{question.id}",
    )


def _load_valid_questions(
    questions_path: Path | None, db_path: Path | None, strategy: str
) -> tuple[list[rag_module.ControlQuestion], list]:
    """Questions that the index supports, plus the strategy's chunks."""
    questions = rag_module.load_questions(questions_path or QUESTIONS_PATH)
    if not questions:
        raise AdventError("В наборе контрольных вопросов нет ни одного вопроса.")
    chunks = index_module.load_chunks(db_path, strategy)
    valid, broken = rag_module.validate_questions(questions, chunks)
    if broken:
        problems = "; ".join(f"#{question.id}: {reason}" for question, reason in broken)
        raise AdventError(
            f"Набор контрольных вопросов не согласован с индексом — {problems}.",
            hint="Исправь week_05/rag_questions.json или переиндексируй: `adventrag index`.",
        )
    return valid, chunks


# --- Day 24: cite pair — table, per-question lines, unanswerable set, summary -----------

CITE_QUESTION_CELL = 14
UN_QUESTION_CELL = 30
NOT_JUDGED = "—"


@dataclass(frozen=True, slots=True)
class UnRow:
    question: rag_module.UnanswerableQuestion
    first: ModeRun | None
    second: ModeRun | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _judge(
    question: rag_module.ControlQuestion, run: ModeRun
) -> tuple[rag_module.Grounding | None, str | None]:
    """Judge a quoted cite answer under its own try: its failure never costs the pair."""
    cited = run.cited
    if cited is None or not cited.quoted or run.ctx is None:
        return None, None
    hits = run.ctx.hits
    try:
        grounding = _retry_transient(
            lambda: rag_module.judge_grounding(
                cited, hits, question.question, day=JOURNAL_DAY_CITE
            ),
            f"judge #{question.id}",
        )
    except Exception as error:  # noqa: BLE001 - isolated: the answer and its tokens are kept
        message = getattr(error, "message", None) or str(error)
        console.warn(f"judge #{question.id}: {message}")
        return None, message
    return grounding, None


def _verdict(row: EvalRow) -> str:
    """Judge cell: `н/о` for a quoted answer without a usable verdict, `—` when not judged."""
    cited = row.second.cited if row.second is not None else None
    if cited is None or not cited.quoted:
        return NOT_JUDGED
    if row.grounding is None or row.grounding.verdict is None:
        return NOT_RATED
    return row.grounding.verdict


def _quotes_cell(cited: CitedAnswer | None) -> str:
    if cited is None or not cited.quotes:
        return "—"
    return f"{cited.verified_quotes}/{len(cited.quotes)}"


def cite_eval_table(rows: list[EvalRow], pair: tuple[str, str]) -> Table:
    """Facts of both modes, then the cite columns: sources, quotes, meaning, refusal."""
    first, second = _labels(pair)
    table = Table(title=f"Контрольные вопросы: {first} и {second}", title_justify="left")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("вопрос", no_wrap=True)
    table.add_column(pair[0], justify="right", no_wrap=True)
    table.add_column("cite", justify="right", no_wrap=True)
    for name in ("источн.", "цитаты", "смысл", "не знаю"):
        table.add_column(name, justify="right", no_wrap=True)
    for row in rows:
        question = _cut(" ".join(row.question.question.split()), CITE_QUESTION_CELL)
        if not row.ok:
            table.add_row(str(row.question.id), question, *["ошибка"] * 2, *["—"] * 4)
            continue
        cited = row.second.cited
        answered = cited is not None and cited.quoted
        table.add_row(
            str(row.question.id),
            question,
            _ratio(row.first_score.facts),
            _ratio(row.second_score.facts),
            _yes_no(row.second_score.sources_cited) if answered else "—",
            _quotes_cell(cited),
            _verdict(row),
            _refusal_cell(cited),
        )
    return table


REFUSAL_SHORT = {
    "empty_context": "порог",
    "model_unknown": "модель",
    "unverified": "цитаты",
    "bad_json": "формат",
    "truncated": "формат",
    "no_index": "индекс",
}


def _refusal_tag(cited: CitedAnswer | None) -> str:
    """Short refusal reason, "" when the cite run answered (or has no cite result)."""
    if cited is None or cited.status != "unknown":
        return ""
    return REFUSAL_SHORT.get(cited.reason, cited.reason)


def _refusal_cell(cited: CitedAnswer | None) -> str:
    tag = _refusal_tag(cited)
    return f"да·{tag}" if tag else "нет"


def _answer_text(run: ModeRun) -> str:
    """The text a reader compares: the accepted cite answer, else the whole reply."""
    if run.cited is not None and run.cited.quoted:
        return run.cited.answer
    return run.text


def _refusal_line(cited: CitedAnswer | None) -> str:
    """First line of the refusal as the reader sees it."""
    if cited is None:
        return "не знаю"
    reason = RAG_UNKNOWN_TEXTS.get(cited.reason, cited.reason)
    return f"{RAG_UNKNOWN_PREFIX} {reason}."


def _print_cite_pair(n: int, row: EvalRow, detail: int, pair: tuple[str, str]) -> None:
    width = console.out.width
    head = f"#{row.question.id} "
    question = " ".join(row.question.question.split())
    if not row.ok or row.first_score is None or row.second_score is None:
        tail = " · ошибка"
        _print_line(head + _cut(question, width - len(head) - len(tail)) + tail)
        return
    cited = row.second.cited
    r1, r2 = _ratio(row.first_score.facts), _ratio(row.second_score.facts)
    if cited is not None and cited.quoted:
        extra = f" · цитаты {_quotes_cell(cited)} · смысл {_verdict(row)}"
    else:
        tag = _refusal_tag(cited)
        extra = f" · не знаю·{tag}" if tag else " · не знаю"
    if n > detail:
        tail = f" · {pair[0]} {r1} · cite {r2}{extra}"
        _print_line(head + _cut(question, width - len(head) - cell_len(tail)) + tail)
        return
    _print_line(head + _cut(question, width - len(head)))
    expected = " · ".join(" ".join(fact[0].split()) for fact in row.question.expect)
    label = "   ожидается: "
    _print_line(label + _cut(expected, width - cell_len(label)))
    marks_w = max(2, len(row.question.expect))
    name_w = max(len(pair[0]), 4)
    for name, run, score in (
        (pair[0], row.first, row.first_score),
        ("cite", row.second, row.second_score),
    ):
        marks = "".join("✓" if found else "✗" for found in score.facts)
        lead = f"   {name:<{name_w}} {marks:<{marks_w}} «"
        room = width - cell_len(lead) - 1
        _print_line(f"{lead}{_snippet(_answer_text(run), row.question.expect, room)}»")
    if cited is not None and cited.quoted:
        sources = _yes_no(row.second_score.sources_cited)
        line = (
            f"   цитаты {_quotes_cell(cited)} дословны · источник {sources} · смысл {_verdict(row)}"
        )
        reason = row.grounding.reason if row.grounding is not None else ""
        if reason:
            line += f" — {reason}"
        _print_line(_cut(line, width))
        for quote in cited.quotes:
            mark = "✓" if quote.verified else "✗"
            _print_line(_cut(f"   [{quote.n}] {mark} «{' '.join(quote.text.split())}»", width))
    else:
        _print_line(_cut(f"   {_refusal_line(cited)}", width))
    console.out.print()


OUTCOME_CELLS = {
    "refused": "не знаю ✓",
    "unverified": "неподтв.",
    "format": "ошибка формата",
    "answered": "ответил ✗",
}


def _outcome_cell(run: ModeRun) -> str:
    """`≈` marks the `says_unknown` heuristic used for modes without a structured result."""
    cell = OUTCOME_CELLS[rag_module.unanswerable_outcome(run.text, run.cited)]
    return cell if run.cited is not None else f"≈{cell}"


def _print_un_line(row: UnRow, pair: tuple[str, str]) -> None:
    width = console.out.width
    head = f"#{row.question.id} "
    question = " ".join(row.question.question.split())
    if not row.ok:
        tail = " · ошибка"
    else:
        tail = f" · {pair[0]} {_outcome_cell(row.first)} · cite {_outcome_cell(row.second)}"
    _print_line(head + _cut(question, width - len(head) - cell_len(tail)) + tail)


def unanswerable_table(rows: list[UnRow], pair: tuple[str, str]) -> Table:
    first, second = _labels(pair)
    table = Table(
        title=f"Вопросы без ответа в репо: {first} и {second} (≈ — эвристика по тексту)",
        title_justify="left",
    )
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("вопрос", no_wrap=True)
    table.add_column(pair[0], justify="right", no_wrap=True)
    table.add_column("cite", justify="right", no_wrap=True)
    for row in rows:
        question = _cut(" ".join(row.question.question.split()), UN_QUESTION_CELL)
        if not row.ok:
            table.add_row(str(row.question.id), question, "ошибка", "ошибка")
        else:
            table.add_row(
                str(row.question.id), question, _outcome_cell(row.first), _outcome_cell(row.second)
            )
    return table


def print_cite_summary(rows: list[EvalRow], un_rows: list[UnRow], pair: tuple[str, str]) -> None:
    """Last screen. Every share is over the pairs actually obtained (M), never the plan."""
    first, second = _labels(pair)
    done = [row for row in rows if row.ok]
    planned, got = len(rows), len(done)
    _out(f"Запланировано {planned} · получено {got} · без ответа {planned - got}")
    if done:
        cites = [row.second.cited for row in done]
        sources = sum(1 for c in cites if c is not None and c.quoted_sources)
        with_quotes = sum(1 for c in cites if c is not None and c.quotes)
        verbatim = sum(
            1 for c in cites if c is not None and c.quotes and c.verified_quotes == len(c.quotes)
        )
        unknown = sum(1 for c in cites if c is not None and c.status == "unknown")
        _out(
            f"Источники в ответе {sources}/{got} · цитаты {with_quotes}/{got} · "
            f"все цитаты дословны {verbatim}/{got} · «не знаю» на отвечаемых {unknown}/{got}"
        )
        verdicts = [_verdict(row) for row in done]
        counts = {v: verdicts.count(v) for v in ("да", "частично", "нет", NOT_RATED, NOT_JUDGED)}
        _out(
            f"Смысл по judge, из {got}: да {counts['да']} · частично {counts['частично']} · "
            f"нет {counts['нет']} · н/о {counts[NOT_RATED]} · не оценивался {counts[NOT_JUDGED]}"
        )
        facts_total = sum(len(row.question.expect) for row in done)
        first_facts = sum(row.first_score.facts_found for row in done)
        second_facts = sum(row.second_score.facts_found for row in done)
        _out(
            f"Фактов найдено: {first} {first_facts}/{facts_total} · "
            f"{second} {second_facts}/{facts_total} (из текста ответа, не из цитат)"
        )
    else:
        _out("Ни одна пара ответов не получена — сравнивать нечего.")
    got_un = [row for row in un_rows if row.ok]
    if un_rows:
        _out(
            f"Без ответа в репо: запланировано {len(un_rows)} · получено {len(got_un)} · "
            f"без ответа {len(un_rows) - len(got_un)}"
        )
        outcomes = [rag_module.unanswerable_outcome(r.second.text, r.second.cited) for r in got_un]
        first_refused = sum(
            1
            for r in got_un
            if rag_module.unanswerable_outcome(r.first.text, r.first.cited) == "refused"
        )
        _out(
            f"«Не знаю» на неотвечаемых: cite {outcomes.count('refused')}/{len(got_un)} · "
            f"{first} ≈{first_refused}/{len(got_un)} (эвристика) · "
            f"неподтв. {outcomes.count('unverified')} · ошибка формата {outcomes.count('format')}"
        )
    # Spend counts every completed call, including the first run of a failed pair.
    pair_runs = [run for row in rows for run in (row.first, row.second) if run is not None]
    pair_runs += [run for row in un_rows for run in (row.first, row.second) if run is not None]
    cite_runs = [run for run in pair_runs if run.mode == "cite"]
    _out(f"Без вызова модели: {sum(1 for run in cite_runs if not run.model_called)}")
    first_runs = [run for run in pair_runs if run.mode == pair[0]]
    p1 = _sum([run.prompt_tokens for run in first_runs])
    c1 = _sum([run.completion_tokens for run in first_runs])
    p2 = _sum([run.prompt_tokens for run in cite_runs])
    c2 = _sum([run.completion_tokens for run in cite_runs])
    embed = _sum([run.ctx.embed_tokens for run in pair_runs if run.ctx is not None] or [0])
    _out(
        f"Токены prompt/completion: {first} {_tokens(p1)}/{_tokens(c1)} · "
        f"{second} {_tokens(p2)}/{_tokens(c2)} · embed {_tokens(embed)}"
    )
    staged = [run for run in pair_runs if run.mode in ("full", "cite")]
    judged = [row.grounding.result for row in done if row.grounding and row.grounding.result]
    aux_p = _sum(
        [run.ctx.aux_prompt_tokens if run.ctx else None for run in staged]
        + [result.usage.prompt_tokens for result in judged]
    )
    aux_c = _sum(
        [run.ctx.aux_completion_tokens if run.ctx else None for run in staged]
        + [result.usage.completion_tokens for result in judged]
    )
    _out(f"aux (rewrite+rerank+judge) prompt/completion: {_tokens(aux_p)}/{_tokens(aux_c)}")
    _out(
        "Один прогон на ячейку. «Цитаты дословны» — не «ответ подтверждён»: смысл оценивает "
        "judge (та же ministral-14b), его вердикт — оценка, не истина."
    )


def run_eval(
    questions_path: Path | None = None,
    strategy: str = "structure",
    k: int = 5,
    db: str | None = None,
    model: str | None = None,
    answers: bool = False,
    detail: int = 0,
    pair: str = DEFAULT_PAIR,
    k_before: int = 20,
    threshold: float = 5.0,
    unanswerable: bool = True,
    unanswerable_path: Path | None = None,
) -> list[EvalRow]:
    """Run the control-question evaluation and print the result."""
    modes = parse_pair(pair)
    cite_pair = modes[1] == "cite"
    db_path = Path(db) if db is not None else None
    config = make_config(model, strategy, k, k_before, threshold)
    rag_module.check_index(db_path, strategy)
    valid, _ = _load_valid_questions(questions_path, db_path, strategy)
    un_questions: list[rag_module.UnanswerableQuestion] = []
    if cite_pair and unanswerable:
        un_questions = rag_module.load_unanswerable(unanswerable_path or UNANSWERABLE_PATH)
    days = mode_days(modes)
    agents = build_agents(config, db_path, modes)
    rows: list[EvalRow] = []
    for n, question in enumerate(valid, start=1):
        first = second = None
        try:
            first = _run_mode_retry(agents[modes[0]], question, mode=modes[0], day=days[modes[0]])
            second = _run_mode_retry(agents[modes[1]], question, mode=modes[1], day=days[modes[1]])
        except AdventError as error:
            console.warn(f"вопрос #{question.id} не оценён: {error.message}")
            # A completed first run is already paid for: keep it so the spend
            # totals count it (comparison metrics still use complete pairs only).
            rows.append(EvalRow(question, first, second, None, None, error=error.message))
            _print_pair(n, rows[-1], detail, modes)
            continue
        grounding, judge_error = _judge(question, second) if cite_pair else (None, None)
        rows.append(
            EvalRow(
                question=question,
                first=first,
                second=second,
                first_score=rag_module.score_answer(question, first.text, first.ctx, first.cited),
                second_score=rag_module.score_answer(
                    question, second.text, second.ctx, second.cited
                ),
                grounding=grounding,
                judge_error=judge_error,
            )
        )
        _print_pair(n, rows[-1], detail, modes)
        if answers:
            console.out.print(
                f"#{question.id} {question.question}", style="bold", markup=False, highlight=False
            )
            print_mode_answer(modes[0], first)
            print_mode_answer(modes[1], second)
    un_rows: list[UnRow] = []
    if un_questions:
        console.out.print()
        for uq in un_questions:
            first = second = None
            try:
                first = _run_mode_retry(agents[modes[0]], uq, mode=modes[0], day=days[modes[0]])
                second = _run_mode_retry(agents[modes[1]], uq, mode=modes[1], day=days[modes[1]])
            except AdventError as error:
                console.warn(f"вопрос #{uq.id} не оценён: {error.message}")
                un_rows.append(UnRow(uq, first, second, error=error.message))
            else:
                un_rows.append(UnRow(uq, first, second))
            _print_un_line(un_rows[-1], modes)
    if rows:
        console.out.print()
    console.out.print(eval_table(rows, modes))
    if cite_pair:
        if un_rows:
            console.out.print()
            console.out.print(unanswerable_table(un_rows, modes))
        print_cite_summary(rows, un_rows, modes)
    else:
        print_summary(rows, modes)
    return rows


def register_eval(app: typer.Typer) -> None:
    """Register the eval command."""

    @app.command("eval")
    def eval_command(
        questions: str | None = typer.Option(
            None, "--questions", help="Файл контрольных вопросов."
        ),
        pair: str = typer.Option(
            DEFAULT_PAIR,
            "--pair",
            help="Пара режимов: off,plain | off,full | plain,full | off,cite | full,cite.",
        ),
        strategy: str = typer.Option("structure", "--strategy", help="fixed | structure."),
        k: int = typer.Option(5, "-k", "--top-k", min=1, help="Сколько чанков прикладывать."),
        k_before: int = typer.Option(
            20, "--k-before", min=1, help="Кандидатов на запрос до rerank (режим full)."
        ),
        threshold: float = typer.Option(
            5.0, "--threshold", help="Порог оценки reranker'а 0..10 (режим full)."
        ),
        db: str | None = typer.Option(None, "--db", help="Путь к индексу."),
        model: str | None = typer.Option(None, "--model", help="Модель чата."),
        answers: bool = typer.Option(
            False, "--answers", help="Печатать ответы обоих режимов целиком."
        ),
        detail: int = typer.Option(
            0,
            "--detail",
            min=0,
            help="Сколько первых вопросов показать подробно: ожидаемое и выдержки ответов.",
        ),
        unanswerable: bool = typer.Option(
            True,
            "--unanswerable/--no-unanswerable",
            help="Режим cite: добавить вопросы без ответа в репо (ожидается «не знаю»).",
        ),
    ) -> None:
        """10 контрольных вопросов: ответы в двух режимах, факты и источники."""
        run_eval(
            Path(questions) if questions is not None else None,
            strategy,
            k,
            db,
            model,
            answers,
            detail,
            pair,
            k_before,
            threshold,
            unanswerable,
        )


# --- Day 23: retrieval quality by stage, no answers generated -------------------------

NO_RELEVANT = "нет чанка со всеми фактами"
NOT_RATED = "н/о"


@dataclass(frozen=True, slots=True)
class StageRow:
    question: rag_module.ControlQuestion
    relevant: frozenset[str]
    ctx: RagContext | None
    error: str | None = None

    @property
    def scored(self) -> bool:
        """Counts toward the sums: answered and some chunk holds every fact."""
        return self.error is None and self.ctx is not None and bool(self.relevant)

    def ranks(self) -> tuple[int | None, int | None, int | None, int | None]:
        """Rank of the first relevant chunk: cosine, +rewrite, +rerank, final."""
        ctx = self.ctx
        if ctx is None or ctx.trace is None:
            return (None, None, None, self._final_rank(ctx.hits) if ctx else None)
        trace = ctx.trace
        return (
            _first_rank(trace.original, self.relevant),
            _first_rank(trace.fused, self.relevant),
            _first_rank(trace.reranked, self.relevant),
            self._final_rank(ctx.hits),
        )

    def _final_rank(self, hits: Sequence[RagHit]) -> int | None:
        """Rank in the context, rechecked on the RETAINED text: fit_hits may cut the fact."""
        expect = self.question.expect
        for position, hit in enumerate(hits, start=1):
            if (
                hit.chunk_id in self.relevant
                and hit.source in self.question.sources
                and all(check_facts(hit.text, expect))
            ):
                return position
        return None

    def threshold_losses(self, threshold: float) -> tuple[int, int]:
        """Relevant candidates rated below the threshold, and relevant ones never rated."""
        if not self.scored or self.ctx.trace is None:
            return (0, 0)
        relevant = [h for h in self.ctx.trace.reranked if h.chunk_id in self.relevant]
        below = sum(1 for h in relevant if h.rerank is not None and h.rerank < threshold)
        return below, sum(1 for h in relevant if h.rerank is None)


def _first_rank(hits: Sequence[RagHit], relevant: frozenset[str]) -> int | None:
    for position, hit in enumerate(hits, start=1):
        if hit.chunk_id in relevant:
            return position
    return None


def _cell(rank: int | None) -> str:
    return "—" if rank is None else str(rank)


def _gate(ctx: RagContext | None) -> str:
    return "—" if ctx is None else f"{ctx.passed or 0}/{ctx.candidates}"


def _stage_line(row: StageRow) -> str:
    head = f"#{row.question.id} "
    if row.error is not None:
        return f"{head}ошибка: {row.error}"
    gate = f" · порог {_gate(row.ctx)}"
    if not row.relevant:
        return f"{head}{NO_RELEVANT}{gate}"
    cos, rew, rer, _ = row.ranks()
    return f"{head}cosine {_cell(cos)} → +rewrite {_cell(rew)} → +rerank {_cell(rer)}{gate}"


def stages_table(rows: Sequence[StageRow]) -> Table:
    table = Table(title="Ранг первого релевантного чанка по этапам", title_justify="left")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("вопрос", no_wrap=True)
    for name in ("cosine", "+rewrite", "+rerank", "итог", "порог"):
        table.add_column(name, justify="right", no_wrap=True)
    for row in rows:
        question = _cut(" ".join(row.question.question.split()), 24)
        if row.error is not None:
            table.add_row(str(row.question.id), question, *["ошибка"] * 4, "—")
        elif not row.relevant:
            table.add_row(str(row.question.id), question, *[NOT_RATED] * 4, _gate(row.ctx))
        else:
            table.add_row(
                str(row.question.id), question, *(_cell(r) for r in row.ranks()), _gate(row.ctx)
            )
    return table


def print_stages_summary(rows: Sequence[StageRow], k: int, threshold: float) -> None:
    scored = [row for row in rows if row.scored]
    n = len(scored)
    ranks = [row.ranks() for row in scored]
    _out(f"Релевантный чанк, оцениваемых вопросов: {n}")
    stages = (
        ("cosine", 0, f"в top-{k}", k),
        ("+rewrite", 1, f"в top-{k}", k),
        ("+rerank (до порога)", 2, f"в top-{k}", k),
        ("итог (после порога)", 3, "в контексте", None),
    )
    for name, i, label, limit in stages:
        found = sum(1 for r in ranks if r[i] is not None and (limit is None or r[i] <= limit))
        first = sum(1 for r in ranks if r[i] == 1)
        _out(f"  {name}: {label} {found}/{n} · на 1-м месте {first}/{n}")
    losses = [(row, row.threshold_losses(threshold)) for row in scored]
    cut = [(row.question.id, below) for row, (below, _) in losses if below]
    line = (
        f"Порог {_num(threshold)}: релевантных чанков ниже порога: "
        f"{sum(b for _, b in cut)} (в {len(cut)} вопросах)"
    )
    if cut:
        line += ": " + ", ".join(f"#{i}×{b}" for i, b in cut)
    _out(line)
    not_rated = [(row.question.id, u) for row, (_, u) in losses if u]
    if not_rated:
        _out(
            f"Релевантных чанков без оценки reranker'а: {sum(u for _, u in not_rated)} (в "
            + f"{len(not_rated)} вопросах): "
            + ", ".join(f"#{i}×{u}" for i, u in not_rated)
        )
    unrated = [row.question.id for row in rows if row.error is None and not row.relevant]
    if unrated:
        _out(f"Вне сумм ({NO_RELEVANT}): " + ", ".join(f"#{i}" for i in unrated))
    failed = [row.question.id for row in rows if row.error is not None]
    if failed:
        _out("Не оценено из-за ошибок: " + ", ".join(f"#{i}" for i in failed))
    ctxs = [row.ctx for row in rows if row.ctx is not None]
    embed = _sum([c.embed_tokens for c in ctxs])
    aux_p = _sum([c.aux_prompt_tokens for c in ctxs])
    aux_c = _sum([c.aux_completion_tokens for c in ctxs])
    _out(
        f"Токены: embed {_tokens(embed)} · aux prompt/completion {_tokens(aux_p)}/{_tokens(aux_c)}"
    )
    _out("Один прогон: rewrite и rerank недетерминированы, ранги плавают на 1–2 позиции.")


def run_stages(
    questions_path: Path | None = None,
    strategy: str = "structure",
    k_before: int = 20,
    k: int = 5,
    threshold: float = 5.0,
    db: str | None = None,
) -> list[StageRow]:
    """Per-stage retrieval ranks for every control question; no answers are generated."""
    check_args(strategy, k, k_before, threshold)
    db_path = Path(db) if db is not None else None
    rag_module.check_index(db_path, strategy)
    valid, chunks = _load_valid_questions(questions_path, db_path, strategy)
    retrieve = rag_module.make_retriever(db_path, day=JOURNAL_DAY_FULL)
    settings = RagSettings(strategy, k, k_before, True, True, float(threshold))
    rows: list[StageRow] = []
    for question in valid:
        relevant = rag_module.relevant_chunk_ids(question, chunks)
        try:
            ctx = _retry_transient(
                lambda q=question: retrieve(q.question, settings), f"вопрос #{question.id}"
            )
        except AdventError as error:
            console.warn(f"вопрос #{question.id} не оценён: {error.message}")
            rows.append(StageRow(question, relevant, None, error=error.message))
        else:
            rows.append(StageRow(question, relevant, ctx))
            for warning in ctx.warnings:
                console.warn(warning)
        _print_line(_cut(_stage_line(rows[-1]), console.out.width))
    if rows:
        console.out.print()
    console.out.print(stages_table(rows))
    print_stages_summary(rows, k, float(threshold))
    return rows


def register_stages(app: typer.Typer) -> None:
    """Register the stages command."""

    @app.command("stages")
    def stages_command(
        questions: str | None = typer.Option(
            None, "--questions", help="Файл контрольных вопросов."
        ),
        strategy: str = typer.Option("structure", "--strategy", help="fixed | structure."),
        k_before: int = typer.Option(
            20, "--k-before", min=1, help="Кандидатов на запрос до rerank."
        ),
        k: int = typer.Option(5, "-k", "--top-k", min=1, help="Сколько чанков в контексте."),
        threshold: float = typer.Option(5.0, "--threshold", help="Порог оценки reranker'а 0..10."),
        db: str | None = typer.Option(None, "--db", help="Путь к индексу."),
    ) -> None:
        """Ранг релевантного чанка после cosine, rewrite и rerank — без генерации ответов."""
        run_stages(
            Path(questions) if questions is not None else None,
            strategy,
            k_before,
            k,
            threshold,
            db,
        )
