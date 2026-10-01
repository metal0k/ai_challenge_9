"""Day 22: `adventrag ask` / `adventrag eval` — the same Agent with RAG off and on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.cells import cell_len
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console
from advent_core.agent import Agent
from advent_core.config import PROJECT_ROOT, Config
from advent_core.errors import AdventError
from advent_core.journal import log_call
from advent_core.rag import RagContext, fact_span
from week_05 import index as index_module
from week_05 import rag as rag_module

QUESTIONS_PATH = PROJECT_ROOT / "week_05" / "rag_questions.json"
RAG_STRATEGIES = ("fixed", "structure")
MODE_LABELS = {False: "без RAG", True: "с RAG"}
ASK_MODES = ("both", "rag", "no-rag")


@dataclass(frozen=True, slots=True)
class ModeRun:
    text: str
    ctx: RagContext | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int


def make_config(model: str | None, strategy: str, k: int) -> Config:
    if strategy not in RAG_STRATEGIES:
        raise AdventError(f"Неизвестная стратегия: {strategy!r}. Доступны: fixed, structure.")
    if not 1 <= k <= 20:
        raise AdventError(f"-k должно быть от 1 до 20, получено {k}.")
    config = Config.resolve(model=model, stream=False)
    config.params.rag_strategy = strategy
    config.params.rag_k = k
    config.params.rag = False
    return config


def _yes_no(flags: tuple[bool, ...]) -> str:
    """Whether at least one of the alternative expected sources is present."""
    return "да" if any(flags) else "нет"


def build_agent(config: Config, db_path: Path | None) -> Agent:
    # No on_warning: the only warning this path can raise is the unknown context
    # window, which is noise here — every question runs on an empty history.
    return Agent(
        config,
        complete=chat_core.complete,
        stream=chat_core.stream,
        retrieve=rag_module.make_retriever(db_path),
    )


def run_mode(
    agent: Agent,
    question: str,
    *,
    use_rag: bool,
    command: str,
    question_id: int | None = None,
) -> ModeRun:
    agent.config.params.rag = use_rag
    reply = agent.ask(question, [])
    log_call(
        reply.result,
        reply.result.sent_messages or [],
        week=rag_module.RAG_WEEK,
        day=rag_module.RAG_DAY,
        extra={"command": command, "rag": use_rag, "question_id": question_id},
    )
    return ModeRun(
        text=reply.text,
        ctx=reply.rag,
        prompt_tokens=reply.result.usage.prompt_tokens,
        completion_tokens=reply.result.usage.completion_tokens,
        latency_ms=reply.result.latency_ms,
    )


def _tokens(value: int | None) -> str:
    return "?" if value is None else str(value)


def print_sources(ctx: RagContext) -> None:
    if not ctx.hits:
        console.out.print("источники: ничего не найдено", markup=False, highlight=False)
        return
    console.out.print("источники:", markup=False, highlight=False)
    for n, hit in enumerate(ctx.hits, start=1):
        label = f"{hit.source} — {hit.section}" if hit.section.strip() else hit.source
        console.out.print(
            f"  [{n}] {hit.score:.3f}  {label}",
            markup=False,
            highlight=False,
            no_wrap=True,
            overflow="ellipsis",
        )
    if ctx.dropped > 0:
        console.out.print(f"  не вошло в контекст: {ctx.dropped}", markup=False, highlight=False)


def print_mode_answer(use_rag: bool, run: ModeRun) -> None:
    console.out.print(f"── {MODE_LABELS[use_rag]} ──", style="bold", markup=False, highlight=False)
    console.out.print(run.text.strip(), markup=False, highlight=False)
    if run.ctx is not None:
        print_sources(run.ctx)
    console.out.print()
    embed = f" · embed {_tokens(run.ctx.embed_tokens)}" if run.ctx is not None else ""
    console.note(
        f"{MODE_LABELS[use_rag]}: prompt {_tokens(run.prompt_tokens)} · "
        f"completion {_tokens(run.completion_tokens)}{embed} · {run.latency_ms / 1000:.1f} с"
    )


def ask_question(
    question: str,
    mode: str = "both",
    strategy: str = "structure",
    k: int = 5,
    db: str | None = None,
    model: str | None = None,
) -> None:
    if mode not in ASK_MODES:
        raise AdventError(f"Неизвестный режим: {mode!r}. Доступны: both, rag, no-rag.")
    if not question.strip():
        raise AdventError("Пустой вопрос.")
    db_path = Path(db) if db is not None else None
    config = make_config(model, strategy, k)
    flags = {"both": (False, True), "rag": (True,), "no-rag": (False,)}[mode]
    if True in flags:
        rag_module.check_index(db_path, strategy)
    agent = build_agent(config, db_path)
    for use_rag in flags:
        run = run_mode(agent, question, use_rag=use_rag, command="ask")
        print_mode_answer(use_rag, run)


def register(app: typer.Typer) -> None:
    @app.command("ask")
    def ask_command(
        question: str = typer.Argument(..., help="Вопрос по базе."),
        rag: bool | None = typer.Option(
            None, "--rag/--no-rag", help="Только с RAG или только без; по умолчанию оба режима."
        ),
        strategy: str = typer.Option("structure", "--strategy", help="fixed | structure."),
        k: int = typer.Option(5, "-k", "--top-k", min=1, help="Сколько чанков прикладывать."),
        db: str | None = typer.Option(None, "--db", help="Путь к индексу."),
        model: str | None = typer.Option(None, "--model", help="Модель чата."),
    ) -> None:
        """Один вопрос — ответ модели без RAG и с RAG рядом."""
        mode = "both" if rag is None else ("rag" if rag else "no-rag")
        ask_question(question, mode, strategy, k, db, model)


QUESTION_CELL = 30


@dataclass(frozen=True, slots=True)
class EvalRow:
    question: rag_module.ControlQuestion
    plain: ModeRun | None
    rag: ModeRun | None
    plain_score: rag_module.AnswerScore | None
    rag_score: rag_module.AnswerScore | None
    error: str | None = None

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


def eval_table(rows: list[EvalRow]) -> Table:
    """Build a comparison table for evaluated questions."""
    table = Table(title="Контрольные вопросы: без RAG и с RAG", title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("вопрос")
    table.add_column("без RAG", justify="right")
    table.add_column("с RAG", justify="right")
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
                _ratio(row.plain_score.facts),
                _ratio(row.rag_score.facts),
                _yes_no(row.rag_score.sources_retrieved),
                _yes_no(row.rag_score.sources_cited),
            )
    return table


def print_summary(rows: list[EvalRow]) -> None:
    """Print aggregate evaluation statistics."""
    done = [row for row in rows if row.ok]
    total = len(rows)
    if not done:
        console.out.print(
            "Ни одна пара ответов не получена — сравнивать нечего.", markup=False, highlight=False
        )
        return
    facts_total = sum(len(row.question.expect) for row in done)
    plain_facts = sum(row.plain_score.facts_found for row in done)
    rag_facts = sum(row.rag_score.facts_found for row in done)
    plain_full = sum(row.plain_score.complete for row in done)
    rag_full = sum(row.rag_score.complete for row in done)
    retrieved = sum(any(row.rag_score.sources_retrieved) for row in done)
    cited = sum(any(row.rag_score.sources_cited) for row in done)
    console.out.print(
        f"Фактов найдено: без RAG {plain_facts}/{facts_total} · с RAG {rag_facts}/{facts_total}",
        markup=False,
        highlight=False,
    )
    console.out.print(
        f"Полных ответов: без RAG {plain_full}/{len(done)} · с RAG {rag_full}/{len(done)}",
        markup=False,
        highlight=False,
    )
    console.out.print(
        f"Ожидаемый источник в top-k: {retrieved}/{len(done)} · "
        f"назван в ответе: {cited}/{len(done)}",
        markup=False,
        highlight=False,
    )
    plain_p = _sum([row.plain.prompt_tokens for row in done])
    plain_c = _sum([row.plain.completion_tokens for row in done])
    rag_p = _sum([row.rag.prompt_tokens for row in done])
    rag_c = _sum([row.rag.completion_tokens for row in done])
    embed = _sum([row.rag.ctx.embed_tokens if row.rag.ctx is not None else 0 for row in done])
    console.out.print(
        f"Токены prompt/completion: без RAG {_tokens(plain_p)}/{_tokens(plain_c)} · "
        f"с RAG {_tokens(rag_p)}/{_tokens(rag_c)} · embed {_tokens(embed)}",
        markup=False,
        highlight=False,
    )
    dropped = sum(row.rag.ctx.dropped for row in done if row.rag.ctx is not None)
    if dropped > 0:
        console.out.print(f"Чанков не вошло в контекст: {dropped}", markup=False, highlight=False)
    if len(done) < total:
        console.out.print(
            f"Завершено пар: {len(done)}/{total} — сравнение неполное, вердикта нет.",
            markup=False,
            highlight=False,
        )
    elif rag_facts > plain_facts:
        console.out.print(
            f"По фактам в этой выборке выше: с RAG ({rag_facts} против {plain_facts}).",
            markup=False,
            highlight=False,
        )
    elif plain_facts > rag_facts:
        console.out.print(
            f"По фактам в этой выборке выше: без RAG ({plain_facts} против {rag_facts}).",
            markup=False,
            highlight=False,
        )
    else:
        console.out.print("Ничья по фактам — явного лидера нет.", markup=False, highlight=False)
    console.out.print(
        f"Вопросов: {total}, один прогон на ячейку — малая выборка, "
        "разница в один факт не значима.",
        markup=False,
        highlight=False,
    )


def _print_line(text: str) -> None:
    """Print one line that must never wrap (the caller sizes it to the width)."""
    console.out.print(text, markup=False, highlight=False, no_wrap=True, overflow="ellipsis")


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


def _print_pair(n: int, row: EvalRow, detail: int) -> None:
    """Print one question's result right after its pair of answers."""
    width = console.out.width
    head = f"#{row.question.id} "
    if not row.ok or row.plain_score is None or row.rag_score is None:
        tail = " · ошибка"
        _print_line(
            head
            + _cut(" ".join(row.question.question.split()), width - len(head) - len(tail))
            + tail
        )
        return
    question = " ".join(row.question.question.split())
    if n > detail:
        tail = f" · без RAG {_ratio(row.plain_score.facts)} · с RAG {_ratio(row.rag_score.facts)}"
        _print_line(head + _cut(question, width - len(head) - cell_len(tail)) + tail)
        return
    _print_line(head + _cut(question, width - len(head)))
    expected = " · ".join(" ".join(fact[0].split()) for fact in row.question.expect)
    label = "   ожидается: "
    _print_line(label + _cut(expected, width - cell_len(label)))
    marks_w = max(2, len(row.question.expect))
    for label, run, score in (
        ("без RAG", row.plain, row.plain_score),
        ("с RAG", row.rag, row.rag_score),
    ):
        marks = "".join("✓" if found else "✗" for found in score.facts)
        lead = f"   {label:<7} {marks:<{marks_w}} «"
        room = width - cell_len(lead) - 1
        _print_line(f"{lead}{_snippet(run.text, row.question.expect, room)}»")
    console.out.print()


def _run_mode_retry(
    agent: Agent, question: rag_module.ControlQuestion, *, use_rag: bool
) -> ModeRun:
    """One retry per mode: a transient network error must not cost the whole pair."""
    try:
        return run_mode(
            agent, question.question, use_rag=use_rag, command="eval", question_id=question.id
        )
    except AdventError as error:
        console.warn(f"вопрос #{question.id}: {error.message} — повтор")
        return run_mode(
            agent, question.question, use_rag=use_rag, command="eval", question_id=question.id
        )


def run_eval(
    questions_path: Path | None = None,
    strategy: str = "structure",
    k: int = 5,
    db: str | None = None,
    model: str | None = None,
    answers: bool = False,
    detail: int = 0,
) -> list[EvalRow]:
    """Run the control-question evaluation and print the result."""
    db_path = Path(db) if db is not None else None
    config = make_config(model, strategy, k)
    rag_module.check_index(db_path, strategy)
    questions = rag_module.load_questions(questions_path or QUESTIONS_PATH)
    if not questions:
        raise AdventError("В наборе контрольных вопросов нет ни одного вопроса.")
    valid, broken = rag_module.validate_questions(
        questions, index_module.load_chunks(db_path, strategy)
    )
    if broken:
        problems = "; ".join(f"#{question.id}: {reason}" for question, reason in broken)
        raise AdventError(
            f"Набор контрольных вопросов не согласован с индексом — {problems}.",
            hint="Исправь week_05/rag_questions.json или переиндексируй: `adventrag index`.",
        )
    agent = build_agent(config, db_path)
    rows: list[EvalRow] = []
    for n, question in enumerate(valid, start=1):
        try:
            plain = _run_mode_retry(agent, question, use_rag=False)
            with_rag = _run_mode_retry(agent, question, use_rag=True)
        except AdventError as error:
            console.warn(f"вопрос #{question.id} не оценён: {error.message}")
            rows.append(
                EvalRow(
                    question=question,
                    plain=None,
                    rag=None,
                    plain_score=None,
                    rag_score=None,
                    error=error.message,
                )
            )
            _print_pair(n, rows[-1], detail)
            continue
        rows.append(
            EvalRow(
                question=question,
                plain=plain,
                rag=with_rag,
                plain_score=rag_module.score_answer(question, plain.text, None),
                rag_score=rag_module.score_answer(question, with_rag.text, with_rag.ctx),
            )
        )
        _print_pair(n, rows[-1], detail)
        if answers:
            console.out.print(
                f"#{question.id} {question.question}", style="bold", markup=False, highlight=False
            )
            print_mode_answer(False, plain)
            print_mode_answer(True, with_rag)
    if rows:
        console.out.print()
    console.out.print(eval_table(rows))
    print_summary(rows)
    return rows


def register_eval(app: typer.Typer) -> None:
    """Register the eval command."""

    @app.command("eval")
    def eval_command(
        questions: str | None = typer.Option(
            None, "--questions", help="Файл контрольных вопросов."
        ),
        strategy: str = typer.Option("structure", "--strategy", help="fixed | structure."),
        k: int = typer.Option(5, "-k", "--top-k", min=1, help="Сколько чанков прикладывать."),
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
    ) -> None:
        """10 контрольных вопросов: ответы без RAG и с RAG, факты и источники."""
        run_eval(
            Path(questions) if questions is not None else None,
            strategy,
            k,
            db,
            model,
            answers,
            detail,
        )
