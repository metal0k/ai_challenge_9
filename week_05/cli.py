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

from advent_core import console, openai_compat
from advent_core.client import mistral_client
from advent_core.config import PROJECT_ROOT, Config, ConfigError
from advent_core.embeddings import (
    DEFAULT_EMBED_MODEL,
    EMBED_MODELS,
    LOCAL_EMBED_MODEL,
    embed_local,
    embed_texts,
)
from advent_core.errors import AdventError
from advent_core.params import DEFAULT_RAG_STRATEGY
from week_05 import chat_eval, rag_cli
from week_05 import corpus as corpus_module
from week_05 import index as index_module
from week_05 import rag as rag_module
from week_05.chunking import (
    MAX_CHUNK_CHARS,
    STRATEGIES,
    Chunk,
    chunk_corpus,
)

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


def _tokens_cell(run: index_module.RunInfo) -> str:
    # LM Studio embeddings report prompt_tokens 0: unknown, not zero tokens.
    if run.endpoint == index_module.ENDPOINT_LOCAL and not run.prompt_tokens:
        return "—"
    return str(run.prompt_tokens) if run.prompt_tokens is not None else "?"


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
            _tokens_cell(run),
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


def _check_chunk_sizes(
    chunks_by_strategy: dict[str, list[Chunk]], cap: int = MAX_CHUNK_CHARS
) -> None:
    """Preflight before any embed_texts call (SPEC §2 invariant, finding 11).

    chunk_structure asserts this internally, but chunk_fixed does not, and an
    assert isn't a caller-facing error naming the chunk — this is a common
    gate in front of every strategy's chunks, for every strategy, so a bad
    chunk fails before ANY strategy's API calls, not partway through.
    """
    for chunks in chunks_by_strategy.values():
        for chunk in chunks:
            if chunk.n_chars > cap:
                raise AdventError(
                    f"Чанк {chunk.chunk_id!r} длиннее MAX_CHUNK_CHARS "
                    f"({chunk.n_chars} > {cap} символов) — эмбеддинг не отправлен."
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


def _local_setup(model: str) -> str:
    """Offline guard on, loopback URL validated, embedding model loaded; returns the URL."""
    config = Config.resolve(offline=True, base_url=openai_compat.default_url())
    url = str(config.base_url)  # canonical: no /v1, no trailing slash
    openai_compat.ensure_ready(model, url, require_state=True)
    return url


def _local_mode_check(db_path: Path | None, strategies: list[str]) -> None:
    """Offline mode accepts a local index only (`check_index` carries the rule)."""
    for strat in strategies:
        rag_module.check_index(db_path, strat)


def _embed_queries(
    run: index_module.RunInfo,
    texts: list[str],
    ids: list[str] | None,
    strategy: str,
    command: str,
):
    """Query vectors in the run's own vector space: a local run never touches the cloud client."""
    extra = {"strategy": strategy, "command": command}
    if run.endpoint == index_module.ENDPOINT_LOCAL:
        return index_module.embed_queries_local(
            run, texts, base_url=None, week=WEEK, day=DAY, journal_extra=extra
        )
    with mistral_client(Config.resolve()) as client:
        return embed_texts(
            client, run.model, texts, ids=ids, week=WEEK, day=DAY, journal_extra=extra
        )


def _index_local(
    strategies: list[str],
    chunks_by_strategy: dict[str, list[Chunk]],
    model: str,
) -> tuple[dict[str, tuple[list[Chunk], object]], dict[str, str]]:
    url = _local_setup(model)
    results: dict[str, tuple[list[Chunk], object]] = {}
    meta: dict[str, str] = {}
    for strat in strategies:
        chunks = chunks_by_strategy[strat]
        texts = [c.text for c in chunks]
        console.note(f"эмбеддинг {strat}: {len(texts)} чанков, модель {model} (локально)")
        t0 = time.monotonic()
        embed_result = embed_local(
            url,
            model,
            texts,
            kind="doc",
            on_batch=_progress_reporter(strat),
            week=WEEK,
            day=DAY,
            journal_extra={"strategy": strat, "command": "index"},
        )
        elapsed = time.monotonic() - t0
        results[strat] = (chunks, embed_result)
        console.out.print(_summary_line(strat, chunks, embed_result, elapsed, 0.0))

        def probe_embed(batch: list[str], _strat: str = strat):
            return embed_local(
                url,
                model,
                batch,
                kind="doc",
                week=WEEK,
                day=DAY,
                journal_extra={"strategy": _strat, "command": "index_probe"},
            ).vectors

        truncated, probed, ids = index_module.truncation_probe(
            chunks, embed_result.vectors, probe_embed
        )
        meta[f"truncated_by_model.{strat}"] = f"{truncated}/{probed}"
        console.out.print(
            f"  обрезано моделью: {truncated} из {probed} проверенных "
            f"(чанки ≥{index_module.PROBE_MIN_CHARS} симв., всего {len(chunks)})"
        )
        for chunk_id in ids[:5]:
            console.warn(f"  обрезан моделью: {chunk_id}")
    return results, meta


@app.command("index")
def index_command(
    strategy: str = typer.Option("all", "--strategy", help="fixed | structure | all."),
    model: str | None = typer.Option(
        None, "--model", help="Модель эмбеддинга (mistral-embed; с --local — nomic-embed)."
    ),
    rev: str = typer.Option("HEAD", "--rev", help="Git-ревизия корпуса."),
    db: str | None = typer.Option(
        None,
        "--db",
        help="Путь к индексу (data/rag/index.sqlite3; с --local — data/rag/index.local.sqlite3).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Только chunking и статистика — без API и без записи."
    ),
    local: bool = typer.Option(
        False, "--local", help="Эмбеддинги локальной моделью (LM Studio), облако отключено."
    ),
) -> None:
    """Собрать корпус, разбить на чанки, посчитать эмбеддинги и записать индекс."""
    local = local is True  # direct calls leave typer's OptionInfo default here
    strategies = _strategy_list(strategy)
    model = (model if isinstance(model, str) else None) or (
        LOCAL_EMBED_MODEL if local else DEFAULT_EMBED_MODEL
    )
    spec = EMBED_MODELS.get(model)
    cap = (spec.chunk_cap if local and spec else 0) or MAX_CHUNK_CHARS
    if local and db is None:
        db = str(index_module.LOCAL_DB)
    console.note(f"собираю корпус на ревизии {rev}…")
    corpus_rev, docs = corpus_module.collect_corpus(PROJECT_ROOT, rev=rev)
    corpus_chars = sum(len(d.text) for d in docs)
    console.note(
        f"корпус: {len(docs)} файлов, {corpus_chars} символов, ревизия {_short_rev(corpus_rev)}"
    )

    chunks_by_strategy: dict[str, list[Chunk]] = {}
    for strat in strategies:
        console.note(f"chunking {strat}…")
        chunks_by_strategy[strat] = (
            chunk_corpus(docs, strat, max_chars=cap) if local else chunk_corpus(docs, strat)
        )

    if dry_run:
        table = _new_stats_table("Статистика чанков (dry-run, без API и без записи)")
        for strat in strategies:
            stats = index_module.chunk_stats(chunks_by_strategy[strat], docs)
            _add_stats_row(table, strat, stats)
        console.out.print(table)
        console.note("dry-run: без API и без записи индекса")
        return

    _check_chunk_sizes(chunks_by_strategy, cap)
    db_path = _db_path(db)
    endpoint = index_module.ENDPOINT_LOCAL if local else index_module.ENDPOINT_CLOUD
    index_module.check_endpoint_free(db_path, endpoint)  # before any embedding is paid for

    meta: dict[str, str] = {}
    if local:
        results, meta = _index_local(strategies, chunks_by_strategy, model)
        meta["chunk_cap"] = str(cap)
    else:
        results = _index_cloud(strategies, chunks_by_strategy, model)

    doc_prefix, query_prefix = index_module.local_prefixes(model) if local else ("", "")
    index_module.write_index(
        db_path,
        results,
        corpus_rev=corpus_rev,
        corpus_files=len(docs),
        corpus_chars=corpus_chars,
        endpoint=endpoint,
        doc_prefix=doc_prefix,
        query_prefix=query_prefix,
        meta=meta or None,
    )
    console.note(f"индекс записан: {_db_display(db_path)}")


def _index_cloud(
    strategies: list[str], chunks_by_strategy: dict[str, list[Chunk]], model: str
) -> dict[str, tuple[list[Chunk], object]]:
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
    return results


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command("compare")
def compare_command(
    db: str | None = typer.Option(None, "--db", help="Путь к индексу."),
    local: bool = typer.Option(
        False, "--local", help="Локальный индекс и локальный embedding запросов, облако отключено."
    ),
) -> None:
    """Сравнить стратегии: статистика чанков + качество поиска на eval-вопросах."""
    local = local is True
    if local and db is None:
        db = str(index_module.LOCAL_DB)
    db_path = _db_path(db)
    if local:
        Config.resolve(offline=True, base_url=openai_compat.default_url())
    runs = index_module.load_runs(db_path)
    strategies = [s for s in ALL_STRATEGIES if s in runs]
    if len(strategies) < 2:
        raise AdventError(
            "Для сравнения нужны обе стратегии в индексе. Сначала `adventrag index --strategy all`."
        )
    index_module.check_comparable(runs, strategies)
    if local:
        _local_mode_check(db_path, strategies)
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
    embed_result = _embed_queries(
        runs[strategies[0]],
        [q.question for q in questions],
        [str(q.id) for q in questions],
        "eval",
        "compare",
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
# check (local RAG prerequisites; the rehearsal runs it before StartRecord)
# ---------------------------------------------------------------------------


@app.command("check")
def check_command(
    db: str | None = typer.Option(None, "--db", help="Путь к индексу (по умолчанию локальный)."),
    strategy: str = typer.Option(DEFAULT_RAG_STRATEGY, "--strategy", help="fixed | structure."),
) -> None:
    """Проверить локальный RAG: индекс с локальным provenance и загруженная embedding-модель."""
    check_local_rag(db, strategy)


def _warm_up_jit_model(run: index_module.RunInfo, base: str) -> None:
    """LM Studio unloads an idle JIT embedding model after its TTL; one request reloads it."""
    found = next((m for m in openai_compat.server_status(base) if m.id == run.model), None)
    if found is None or found.state is None or found.loaded:
        return  # unlisted / stateless server: ensure_ready reports it
    t0 = time.monotonic()
    result = embed_local(base, run.model, ["проверка"], kind="query", prefix=run.query_prefix)
    if result.vectors.shape[1] != run.dim:
        raise AdventError(
            f"Прогрев {run.model}: размерность {result.vectors.shape[1]}, "
            f"а индекс построен с {run.dim}."
        )
    console.err.print(
        f"embedding-модель {run.model} не была загружена — прогрел одним запросом "
        f"({time.monotonic() - t0:.1f} s)"
    )


def check_local_rag(
    db: str | None = None, strategy: str = DEFAULT_RAG_STRATEGY
) -> index_module.RunInfo:
    """Offline mode, a local-endpoint index with the strategy, its embedding model loaded.

    Non-zero exit (AdventError) on any miss: a missing index must stop a take before
    recording instead of surfacing as a warning inside the video.
    """
    config = Config.resolve(offline=True, base_url=openai_compat.default_url())
    db_path = Path(db) if db is not None else index_module.LOCAL_DB
    run = rag_module.check_index(db_path, strategy)  # offline: refuses a cloud-built index
    base = str(config.base_url)
    _warm_up_jit_model(run, base)
    openai_compat.ensure_ready(run.model, base, require_state=True)
    console.out.print(
        f"локальный индекс готов: {_db_display(db_path)}, стратегия {strategy}, "
        f"{run.n_chunks} чанков, модель {run.model} загружена"
    )
    return run


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
    embed_result = _embed_queries(
        runs[strategies[0]], [query], None, ",".join(strategies), "search"
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


# Day 22-23 commands (`ask`, `eval`, `stages`) live in their own module.
rag_cli.register(app)
rag_cli.register_eval(app)
rag_cli.register_stages(app)
chat_eval.register(app)


def main() -> None:
    """Entry point `adventrag`: errors as text, not a traceback."""
    console.force_utf8()
    try:
        app()
    except AdventError as error:
        console.fail(error)
        raise SystemExit(error.exit_code) from None
    except ConfigError as error:  # not an AdventError: offline/loopback refusals land here
        console.fail(AdventError(str(error)))
        raise SystemExit(error.exit_code) from None
    except KeyboardInterrupt:
        console.note("\nпрервано")
        raise SystemExit(130) from None


# `python -m week_05.cli` is how the demo recorder launches this (Step.module).
if __name__ == "__main__":
    main()
