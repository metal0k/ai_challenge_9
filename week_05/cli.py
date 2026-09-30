"""Week 05 entry point `adventrag`: chunk -> embed -> index -> search/compare.

Search and comparison logic lives in week_05/index.py as plain functions, not
inside a typer command body (SPEC-w05d21.md SS0): a later day's MCP server
(week_05/server.py) calls them directly, the way week_04/server.py already
calls week_04 functions in-process. stdout is the product (tables); progress,
notes and errors go to stderr, as everywhere in this repo.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.cells import cell_len
from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import console
from advent_core.client import mistral_client
from advent_core.config import PROJECT_ROOT, Config
from advent_core.embeddings import DEFAULT_EMBED_MODEL, embed_texts
from advent_core.errors import AdventError
from week_05 import corpus as corpus_module
from week_05 import index as index_module
from week_05 import rag_cli
from week_05.chunking import MAX_CHUNK_CHARS, STRATEGIES, Chunk, chunk_corpus

DAY = 21
WEEK = 5
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "week_05" / "eval_questions.json"
# Insertion order of STRATEGIES ("fixed", "structure") -- every table in this
# module lists strategies in this same order.
ALL_STRATEGIES = tuple(STRATEGIES)
SHOW_PREVIEW_LINES = 12
SMALL_CHUNK_LABEL = f"мелких <{index_module.SMALL_CHUNK_CHARS}"
# Human-facing lines only — the DB keeps the full sha (finding 14); at 80
# columns a full 40-char sha wraps the line it sits in.
REV_DISPLAY_CHARS = 12

app = typer.Typer(
    help="Индекс RAG поверх корпуса репозитория: chunking, эмбеддинги, поиск, сравнение стратегий.",
    no_args_is_help=True,
    add_completion=False,
)


def _strategy_list(value: str) -> list[str]:
    if value == "all":
        return list(ALL_STRATEGIES)
    if value not in STRATEGIES:
        known = ", ".join((*ALL_STRATEGIES, "all"))
        raise AdventError(f"Неизвестная стратегия: {value!r}. Доступны: {known}.")
    return [value]


def _db_path(db: str | None) -> Path | None:
    return Path(db) if db is not None else None


def _db_display(db_path: Path | None) -> str:
    return index_module.display_path(db_path)


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Russian plural agreement: 1 строка / 2-4 строки / 5+ и 11-14 строк."""
    if 11 <= n % 100 <= 14:
        return many
    if n % 10 == 1:
        return one
    if 2 <= n % 10 <= 4:
        return few
    return many


def _short_rev(rev: str) -> str:
    return rev[:REV_DISPLAY_CHARS]


# ---------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------


def _new_stats_table(title: str) -> Table:
    table = Table(title=title, title_justify="left")
    table.add_column("стратегия", style="bold")
    table.add_column("чанков", justify="right")
    table.add_column("min/med/p90/max", justify="right")
    table.add_column(SMALL_CHUNK_LABEL, justify="right")
    table.add_column("разрезов", justify="right")
    return table


def _size_cell(stats: index_module.Stats) -> str:
    return f"{stats.min_chars}/{stats.median_chars:.0f}/{stats.p90_chars:.0f}/{stats.max_chars}"


def _add_stats_row(table: Table, strategy: str, stats: index_module.Stats) -> None:
    table.add_row(
        strategy,
        str(stats.n_chunks),
        _size_cell(stats),
        f"{stats.small_share:.0%}",
        f"{stats.cut_share:.0%}",
    )


def _chunks_table(stats: dict[str, index_module.Stats], strategies: list[str]) -> Table:
    """Chunk size distribution — split from cost (finding 13): 9 columns truncated at 80 cols."""
    table = _new_stats_table("Статистика чанков")
    for strat in strategies:
        _add_stats_row(table, strat, stats[strat])
    return table


def _cost_table(runs: dict[str, index_module.RunInfo], strategies: list[str]) -> Table:
    table = Table(title="Стоимость индексации", title_justify="left")
    table.add_column("стратегия", style="bold")
    table.add_column("токенов", justify="right")
    table.add_column("запросов", justify="right")
    table.add_column("время", justify="right")
    table.add_column("$", justify="right")
    for strat in strategies:
        run = runs[strat]
        table.add_row(
            strat,
            str(run.prompt_tokens) if run.prompt_tokens is not None else "?",
            str(run.requests),
            f"{run.latency_ms / 1000:.1f} с",
            f"${run.cost_usd:.4f}" if run.cost_usd is not None else "$?",
        )
    return table


def _quality_table(report: index_module.EvalReport, strategies: list[str]) -> Table:
    table = Table(title="Качество поиска (top-5, eval_questions.json)", title_justify="left")
    table.add_column("стратегия", style="bold")
    table.add_column("вопросов", justify="right")
    table.add_column("hit@1", justify="right")
    table.add_column("hit@3", justify="right")
    table.add_column("hit@5", justify="right")
    table.add_column("MRR@5", justify="right")
    for strat in strategies:
        m = report.per_strategy[strat]
        table.add_row(
            strat,
            str(m.n_questions),
            f"{m.hit_at_1:.0%}",
            f"{m.hit_at_3:.0%}",
            f"{m.hit_at_5:.0%}",
            f"{m.mrr_at_5:.3f}",
        )
    return table


def _cut(text: str, limit: int) -> str:
    """Cut to `limit` terminal CELLS with an ellipsis (emoji are 2 cells wide)."""
    if limit < 1:
        return ""
    if cell_len(text) <= limit:
        return text
    out = ""
    for ch in text:
        if cell_len(out + ch) > limit - 1:
            break
        out += ch
    return out + "…"


def _print_search_hits(strategy: str, hits: list[index_module.Hit]) -> None:
    """Two lines per hit; source:lines is the answer and is never cut, section/snippet are."""
    width = console.out.width
    console.out.print(f"[bold]Поиск — {strategy}[/bold]")
    for rank, hit in enumerate(hits, 1):
        chunk = hit.chunk
        head = f"  {rank}. {hit.score:.3f}  {chunk.source}:{chunk.line_start}-{chunk.line_end}"
        section = " ".join(chunk.section.split())
        room = width - cell_len(head) - len("  · ")
        if section and room >= 4:
            head += f"  · {_cut(section, room)}"
        console.out.print(head, markup=False, highlight=False, no_wrap=True, overflow="ellipsis")
        indent = "     "
        snippet = " ".join(chunk.text.split())
        console.out.print(
            indent + _cut(snippet, width - len(indent) - 1),
            markup=False,
            highlight=False,
            style="dim",
            no_wrap=True,
            overflow="ellipsis",
        )


def _chunk_metadata_table(chunk: Chunk) -> Table:
    table = Table(show_header=False, box=None, title="Метаданные чанка", title_justify="left")
    table.add_column(style="dim")
    table.add_column()
    table.add_row("chunk_id", rich_escape(chunk.chunk_id))
    table.add_row("strategy", chunk.strategy)
    table.add_row("source", rich_escape(chunk.source))
    table.add_row("title", rich_escape(chunk.title))
    table.add_row("section", rich_escape(chunk.section))
    table.add_row("ordinal", str(chunk.ordinal))
    table.add_row("char_start/char_end", f"{chunk.char_start}/{chunk.char_end}")
    table.add_row("line_start/line_end", f"{chunk.line_start}/{chunk.line_end}")
    table.add_row("n_chars", str(chunk.n_chars))
    return table


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def _summary_line(
    strategy: str, chunks: list[Chunk], embed_result, elapsed_s: float, cost: float | None
) -> str:
    tokens = embed_result.prompt_tokens if embed_result.prompt_tokens is not None else "?"
    cost_label = f"${cost:.4f}" if cost is not None else "$?"
    return (
        f"[bold]{strategy}[/bold]: {len(chunks)} чанков · токенов {tokens} · "
        f"запросов {embed_result.requests} · {elapsed_s:.1f} с · {cost_label}"
    )


def _check_chunk_sizes(chunks_by_strategy: dict[str, list[Chunk]]) -> None:
    """Preflight before any embed_texts call (SPEC §2 invariant, finding 11).

    chunk_structure asserts this internally, but chunk_fixed does not, and an
    assert isn't a caller-facing error naming the chunk — this is a common
    gate in front of every strategy's chunks, for every strategy, so a bad
    chunk fails before ANY strategy's API calls, not partway through.
    """
    for chunks in chunks_by_strategy.values():
        for chunk in chunks:
            if chunk.n_chars > MAX_CHUNK_CHARS:
                raise AdventError(
                    f"Чанк {chunk.chunk_id!r} длиннее MAX_CHUNK_CHARS "
                    f"({chunk.n_chars} > {MAX_CHUNK_CHARS} символов) — эмбеддинг не отправлен."
                )


def _progress_reporter(strat: str):
    """Coarse-milestone progress (~every 25% plus completion), stderr only (finding 16).

    Reporting every batch "freezes" the screen with a wall of identical-looking
    lines on a long index run (CLAUDE.md's day-10 lesson) without adding
    information; a handful of milestones still proves the run is alive.
    """
    state = {"last_milestone": -1}

    def on_batch(done: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, int(done * 100 / total))
        milestone = (pct // 25) * 25
        if milestone > state["last_milestone"] or done >= total:
            state["last_milestone"] = max(state["last_milestone"], milestone)
            console.note(f"  {strat}: {done}/{total} чанков ({pct}%)")

    return on_batch


@app.command("index")
def index_command(
    strategy: str = typer.Option("all", "--strategy", help="fixed | structure | all."),
    model: str = typer.Option(DEFAULT_EMBED_MODEL, "--model", help="Модель эмбеддинга."),
    rev: str = typer.Option("HEAD", "--rev", help="Git-ревизия корпуса."),
    db: str | None = typer.Option(
        None, "--db", help="Путь к индексу (по умолчанию data/rag/index.sqlite3)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Только chunking и статистика — без API и без записи."
    ),
) -> None:
    """Собрать корпус, разбить на чанки, посчитать эмбеддинги и записать индекс."""
    strategies = _strategy_list(strategy)
    console.note(f"собираю корпус на ревизии {rev}…")
    corpus_rev, docs = corpus_module.collect_corpus(PROJECT_ROOT, rev=rev)
    corpus_chars = sum(len(d.text) for d in docs)
    console.note(
        f"корпус: {len(docs)} файлов, {corpus_chars} символов, ревизия {_short_rev(corpus_rev)}"
    )

    chunks_by_strategy: dict[str, list[Chunk]] = {}
    for strat in strategies:
        console.note(f"chunking {strat}…")
        chunks_by_strategy[strat] = chunk_corpus(docs, strat)

    if dry_run:
        table = _new_stats_table("Статистика чанков (dry-run, без API и без записи)")
        for strat in strategies:
            stats = index_module.chunk_stats(chunks_by_strategy[strat], docs)
            _add_stats_row(table, strat, stats)
        console.out.print(table)
        console.note("dry-run: без API и без записи индекса")
        return

    _check_chunk_sizes(chunks_by_strategy)

    config = Config.resolve()
    results: dict[str, tuple[list[Chunk], object]] = {}
    with mistral_client(config) as client:
        for strat in strategies:
            chunks = chunks_by_strategy[strat]
            texts = [c.text for c in chunks]
            ids = [c.chunk_id for c in chunks]
            console.note(f"эмбеддинг {strat}: {len(texts)} чанков, модель {model}")

            t0 = time.monotonic()
            embed_result = embed_texts(
                client,
                model,
                texts,
                ids=ids,
                on_batch=_progress_reporter(strat),
                week=WEEK,
                day=DAY,
                journal_extra={"strategy": strat, "command": "index"},
            )
            elapsed = time.monotonic() - t0
            results[strat] = (chunks, embed_result)
            cost = index_module.embed_cost_usd(embed_result.model, embed_result.prompt_tokens)
            console.out.print(_summary_line(strat, chunks, embed_result, elapsed, cost))

    db_path = _db_path(db)
    index_module.write_index(
        db_path,
        results,
        corpus_rev=corpus_rev,
        corpus_files=len(docs),
        corpus_chars=corpus_chars,
    )
    console.note(f"индекс записан: {_db_display(db_path)}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command("compare")
def compare_command(
    db: str | None = typer.Option(None, "--db", help="Путь к индексу."),
) -> None:
    """Сравнить стратегии: статистика чанков + качество поиска на eval-вопросах."""
    db_path = _db_path(db)
    runs = index_module.load_runs(db_path)
    strategies = [s for s in ALL_STRATEGIES if s in runs]
    if len(strategies) < 2:
        raise AdventError(
            "Для сравнения нужны обе стратегии в индексе. Сначала `adventrag index --strategy all`."
        )
    index_module.check_comparable(runs, strategies)
    model = runs[strategies[0]].model
    corpus_rev = runs[strategies[0]].corpus_rev

    console.note(
        f"собираю корпус на ревизии {_short_rev(corpus_rev)} для проверки якорей и py-эвристик…"
    )
    _, docs = corpus_module.collect_corpus(PROJECT_ROOT, rev=corpus_rev)
    docs_by_source = {d.source: d for d in docs}
    # Stats from the STORED chunks, not a fresh chunk_corpus() call (finding
    # 4) — today's chunker could differ from what was actually indexed.
    stats = {
        strat: index_module.chunk_stats(index_module.load_chunks(db_path, strat), docs)
        for strat in strategies
    }
    console.out.print(_chunks_table(stats, strategies))
    console.out.print(_cost_table(runs, strategies))

    questions = index_module.load_eval(EVAL_QUESTIONS_PATH)
    config = Config.resolve()
    with mistral_client(config) as client:
        embed_result = embed_texts(
            client,
            model,
            [q.question for q in questions],
            ids=[str(q.id) for q in questions],
            week=WEEK,
            day=DAY,
            journal_extra={"strategy": "eval", "command": "compare"},
        )

    hits_by_question: dict[str, dict[int, list[index_module.Hit]]] = {s: {} for s in strategies}
    for i, question in enumerate(questions):
        vector = embed_result.vectors[i]
        for strat in strategies:
            hits_by_question[strat][question.id] = index_module.search(db_path, strat, vector, k=5)

    report = index_module.evaluate(hits_by_question, questions, docs_by_source)

    if report.broken:
        console.warn("вопросы исключены из метрики (якорь не найден в снимке корпуса):")
        for broken in report.broken:
            console.warn(f"  #{broken.id} {broken.question} — {broken.reason}")

    # Table first, verdict and caveat under it: the last screen is the answer.
    console.out.print(_quality_table(report, strategies))

    n_scored = report.per_strategy[strategies[0]].n_questions
    if n_scored == 0:
        console.out.print("Ни один вопрос нельзя оценить — вердикта нет.")
        return
    mrr_values = {s: report.per_strategy[s].mrr_at_5 for s in strategies}
    winner = index_module.leader(mrr_values)
    if winner is not None:
        others = ", ".join(f"{v:.3f}" for s, v in mrr_values.items() if s != winner)
        console.out.print(
            f"Выше MRR@5 в этой выборке: [bold]{winner}[/bold] "
            f"({mrr_values[winner]:.3f} против {others})"
        )
    else:
        console.out.print("Ничья по MRR@5 — явного лидера нет.")
    console.out.print(
        f"{n_scored} {_plural(n_scored, 'вопрос', 'вопроса', 'вопросов')} — малая выборка, "
        "различие в одно попадание не значимо.",
        markup=False,
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command("search")
def search_command(
    query: str = typer.Argument(..., help="Текст запроса."),
    strategy: str = typer.Option("all", "--strategy", help="fixed | structure | all."),
    k: int = typer.Option(5, "-k", "--top-k", min=1, help="Сколько результатов на стратегию."),
    db: str | None = typer.Option(None, "--db", help="Путь к индексу."),
) -> None:
    """Найти ближайшие чанки к запросу — по каждой запрошенной стратегии рядом."""
    search_hits(query, strategy, k, db)


def search_hits(query: str, strategy: str = "all", k: int = 5, db: str | None = None) -> None:
    """Plain function behind `search`; validates before the paid embedding call."""
    if k < 1:
        raise AdventError(f"-k должно быть не меньше 1, получено {k}.")
    db_path = _db_path(db)
    strategies = _strategy_list(strategy)
    runs = index_module.load_runs(db_path)
    index_module.check_comparable(runs, strategies)
    model = runs[strategies[0]].model

    config = Config.resolve()
    with mistral_client(config) as client:
        embed_result = embed_texts(
            client,
            model,
            [query],
            week=WEEK,
            day=DAY,
            journal_extra={"strategy": ",".join(strategies), "command": "search"},
        )
    query_vec = embed_result.vectors[0]

    for strat in strategies:
        hits = index_module.search(db_path, strat, query_vec, k=k)
        _print_search_hits(strat, hits)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def _resolve_target(db_path: Path | None, target: str, strategy: str) -> Chunk | None:
    """chunk_id (carries `#ordinal`) or SOURCE:LINE (for a deterministic demo)."""
    if "#" in target:
        return index_module.get_chunk(db_path, target)
    if ":" in target:
        source, _, line_text = target.rpartition(":")
        try:
            line = int(line_text)
        except ValueError as exc:
            raise AdventError(f"Не удалось разобрать номер строки в {target!r}.") from exc
        return index_module.find_chunk_by_line(db_path, strategy, source, line)
    raise AdventError(
        f"Не удалось разобрать {target!r}: ожидается chunk_id (со знаком #) или SOURCE:LINE."
    )


@app.command("show")
def show_command(
    target: str = typer.Argument(..., help="chunk_id (strategy:source#ordinal) или SOURCE:LINE."),
    strategy: str = typer.Option(
        "structure", "--strategy", help="Стратегия для формы SOURCE:LINE."
    ),
    db: str | None = typer.Option(None, "--db", help="Путь к индексу."),
    full: bool = typer.Option(False, "--full", help="Весь текст чанка, без обрезки."),
) -> None:
    """Показать все метаданные и начало текста одного чанка (--full — весь текст)."""
    show_chunk(target, strategy, db, full)


def show_chunk(
    target: str, strategy: str = "structure", db: str | Path | None = None, full: bool = False
) -> None:
    """Plain function behind `show` (typer defaults are OptionInfo objects, truthy)."""
    db_path = Path(db) if db is not None else None
    chunk = _resolve_target(db_path, target, strategy)
    if chunk is None:
        raise AdventError(f"Чанк не найден: {target!r}.")
    console.out.print(_chunk_metadata_table(chunk))
    console.out.print()
    lines = chunk.text.splitlines()
    shown = lines if full else lines[:SHOW_PREVIEW_LINES]
    console.out.print(rich_escape("\n".join(shown)))
    hidden = len(lines) - len(shown)
    if hidden > 0:
        console.out.print(
            f"… ещё {hidden} {_plural(hidden, 'строка', 'строки', 'строк')} (полный текст: --full)",
            style="dim",
            markup=False,
        )


# Day 22 commands (`ask`, `eval`) live in their own module.
rag_cli.register(app)
rag_cli.register_eval(app)


def main() -> None:
    """Entry point `adventrag`: errors as text, not a traceback."""
    console.force_utf8()
    try:
        app()
    except AdventError as error:
        console.fail(error)
        raise SystemExit(error.exit_code) from None
    except KeyboardInterrupt:
        console.note("\nпрервано")
        raise SystemExit(130) from None


# `python -m week_05.cli` is how the demo recorder launches this (Step.module).
if __name__ == "__main__":
    main()
