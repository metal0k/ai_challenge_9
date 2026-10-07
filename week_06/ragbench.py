"""Day 28: the same RAG pipeline (rewrite + rerank + cite) on a cloud and a local backend.

No typer here: `week_06/cli.py` registers the command, this module is the bench core.
Order is fixed by the offline guard, which is process-wide and cannot be switched off:
cloud phase first, then `offline.enable()` + `reset_counters()`, then the local phase.
stdout is the product (one line per finished answer, the tables); stage progress and
warnings go to stderr.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from rich.cells import cell_len
from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import config as config_module
from advent_core import console, offline
from advent_core import openai_compat as oc
from advent_core.config import LOCAL_API_KEY, Config, ConfigError
from advent_core.embeddings import embed_cost_usd
from advent_core.errors import AdventError
from advent_core.params import GenerationParams
from advent_core.rag import LedgerEntry, OnCallFn
from week_01.models_bench import PriceTable, call_cost
from week_05 import cli as rag_index_cli
from week_05 import index as index_module
from week_05 import rag as rag_module
from week_05 import rag_cli

WEEK = 6
DAY = 28
BACKENDS = ("cloud", "local")  # execution order, not display order
LOCAL_MODEL = "ornith"
CLOUD_MODEL = "ministral-14b-latest"
STRATEGY = "structure"
K = 5
K_BEFORE = 20
THRESHOLD = 5.0
ANSWER_MAX_TOKENS = 4096
DEFAULT_RUNS = 3
MAX_RUNS = 10
RETRY_PAUSE_S = rag_cli.RETRY_PAUSE_S
PROGRESS_INTERVAL = 3.0
REFUSALS = ("empty_context", "model_unknown")  # a real refusal; no_index is an infrastructure miss
SMALL_SAMPLE = "малая выборка: одно попадание разницы не значимо"
SHOWN = {"local": "local", "cloud": "cloud"}
COST_NOTE = (
    "$: только завершённые вызовы (упавший вызов не виден); local — без электричества и железа"
)
PARTIAL_NOTE = "≥ — у части вызовов сервер не вернул usage: сумма по известным, не полная"
LEGACY_NOTE = (
    "search: нет в файле старого формата (embed взят из ledger; "
    "время поиска не сохранялось и не домысливается)"
)
TPS_NOTE = "tok/s ответа = completion / время ответа; у local внутри reasoning, у cloud — overhead"
PROTOCOLS = {
    "local": "rerank: json_schema-грамматика, до 2 попыток, свой cap; aux без reasoning; "
    "ответ с reasoning, cite-JSON по грамматике",
    "cloud": "rerank: JSON-инструкция в промпте (json mode); aux без reasoning; "
    "ответ ministral-14b",
}


def _pause(seconds: float) -> None:
    """Module-level seam: tests replace it so a retry never sleeps."""
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# configs, backends, pre-flight
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Backend:
    name: str  # "local" | "cloud"
    config: Config
    db_path: Path
    run: Any = None  # index RunInfo (corpus_rev, embed model) for the report

    @property
    def endpoint(self) -> str:
        return "local" if self.config.is_local else "cloud"


def cloud_config(model: str = CLOUD_MODEL) -> Config | None:
    """Cloud config isolated from ADVENT_BASE_URL / ADVENT_OFFLINE / ADVENT_* params.

    `Config.resolve` would inherit them from .env and quietly turn the cloud side local;
    None without a real key. No os.environ mutation.
    """
    config_module.load_env()
    key = (os.environ.get("MISTRAL_API_KEY") or "").strip()
    if not key:
        return None
    return Config(
        api_key=key,
        model=model,
        system_prompt_path=None,
        params=GenerationParams.build(max_tokens=ANSWER_MAX_TOKENS),
        stream=False,
        base_url=None,
        offline=False,
    )


def local_config(url: str | None = None, model: str = LOCAL_MODEL) -> Config:
    """Loopback config, never touching MISTRAL_API_KEY; the guard is enabled by the caller."""
    base = config_module.normalize_base_url(url or oc.default_url())
    if not base:
        raise ConfigError("Не задан адрес локального сервера.")
    config_module.validate_loopback_url(base, "локальный сервер")
    return Config(
        api_key=LOCAL_API_KEY,
        model=model,
        system_prompt_path=None,
        params=GenerationParams.build(max_tokens=ANSWER_MAX_TOKENS),
        stream=False,
        base_url=base,
        offline=True,
    )


def apply_rag_settings(params: GenerationParams) -> None:
    """Every RAG setting explicit; sampling params stay unset (server defaults)."""
    params.rag_strategy = STRATEGY
    params.rag_k = K
    params.rag_k_before = K_BEFORE
    params.rag_threshold = THRESHOLD
    params.rag = True
    params.rag_rewrite = True
    params.rag_rerank = True
    params.rag_cite = True
    params.rag_aux_reasoning = False
    params.max_tokens = ANSWER_MAX_TOKENS
    params.temperature = None
    params.top_p = None


def isolated_config(config: Config) -> Config:
    """A copy with its own params: running one agent never changes another or the input."""
    return dataclasses.replace(config, params=copy.deepcopy(config.params))


def make_cloud_backend(config: Config, db_path: Path | None = None) -> Backend:
    db = db_path or index_module.DEFAULT_DB
    run = rag_module.check_index(db, STRATEGY)
    return Backend("cloud", config, db, run)


def make_local_backend(config: Config, db_path: Path | None = None) -> Backend:
    """Local pre-flight: index built locally, embedding model loaded, ornith loaded.

    Must run only in the local phase: the guard is already on and a cloud-built index
    is refused by `check_index`.
    """
    db = db_path or index_module.LOCAL_DB
    base = str(config.base_url)
    run = rag_module.check_index(db, STRATEGY)
    rag_index_cli._warm_up_jit_model(run, base)
    oc.ensure_ready(run.model, base, require_state=True)
    oc.ensure_ready(config.model, base, require_state=True)
    return Backend("local", config, db, run)


def _index_run(db: Path, backend: str) -> Any:
    """Index run of STRATEGY without the offline guard: plain sqlite, no HTTP."""
    hint = "adventrag index" if backend == "cloud" else "adventrag index --local"
    try:
        runs = index_module.load_runs(db)
    except AdventError:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken sqlite file is a pre-flight failure
        raise AdventError(f"Индекс {db.name} не читается: {exc}", hint=hint) from exc
    run = runs.get(STRATEGY)
    if run is None:
        raise AdventError(f"В индексе {db.name} нет стратегии {STRATEGY!r}.", hint=hint)
    endpoint = getattr(run, "endpoint", index_module.ENDPOINT_CLOUD)
    if (endpoint == index_module.ENDPOINT_LOCAL) != (backend == "local"):
        raise AdventError(
            f"Индекс {db.name} собран через {endpoint}, а нужен для backend'а {backend}.",
            hint=hint,
        )
    return run


@dataclass(frozen=True, slots=True)
class Plan:
    questions: tuple[rag_module.ControlQuestion, ...]
    unanswerable: tuple[rag_module.UnanswerableQuestion, ...]
    corpus_rev: str | None
    notes: tuple[str, ...] = ()


def prepare_plan(
    backends: Sequence[str],
    *,
    ids: Sequence[int] | None = None,
    unanswerable: bool = True,
    cloud_db: Path | None = None,
    local_db: Path | None = None,
    questions_path: Path | None = None,
    unanswerable_path: Path | None = None,
) -> Plan:
    """Index checks (sqlite only) and the question set valid for EVERY requested index."""
    dbs = {"cloud": cloud_db or index_module.DEFAULT_DB, "local": local_db or index_module.LOCAL_DB}
    runs = {name: _index_run(dbs[name], name) for name in BACKENDS if name in backends}
    revs = {name: run.corpus_rev for name, run in runs.items()}
    if len(set(revs.values())) > 1:
        shown = ", ".join(f"{name} {rev[:12]}" for name, rev in revs.items())
        raise AdventError(
            f"Индексы собраны из разных снимков репозитория ({shown}): сравнение нечестное.",
            hint="Пересобери оба: `adventrag index` и `adventrag index --local` (один HEAD).",
        )
    questions = rag_module.load_questions(questions_path or rag_cli.QUESTIONS_PATH)
    if ids is not None:
        known = {q.id for q in questions}
        missing = [i for i in ids if i not in known]
        if missing:
            raise AdventError(
                f"Нет вопросов с id {', '.join(map(str, missing))} в rag_questions.json."
            )
        questions = [q for q in questions if q.id in set(ids)]
    valid_ids: set[int] | None = None
    notes: list[str] = []
    for name in runs:
        chunks = index_module.load_chunks(dbs[name], STRATEGY)
        good, broken = rag_module.validate_questions(questions, chunks)
        good_ids = {q.id for q in good}
        valid_ids = good_ids if valid_ids is None else valid_ids & good_ids
        for q, reason in broken:
            notes.append(f"вопрос #{q.id} не подходит индексу {name}: {reason}")
    common = [q for q in questions if valid_ids is None or q.id in valid_ids]
    if not common:
        raise AdventError("Нет вопросов, валидных для всех выбранных индексов.")
    un: list[rag_module.UnanswerableQuestion] = []
    if unanswerable:
        un = rag_module.load_unanswerable(unanswerable_path or rag_cli.UNANSWERABLE_PATH)
    return Plan(tuple(common), tuple(un), next(iter(revs.values()), None), tuple(notes))


# ---------------------------------------------------------------------------
# one cell: question x run x backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallRecord:
    stage: str
    model: str
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int
    cost: float | None  # None = unknown, never 0


@dataclass(slots=True)
class BenchRun:
    backend: str
    kind: str  # "answerable" | "unanswerable"
    question_id: int
    run: int
    # answered, a refusal reason, unverified, bad_json, truncated, no_index or error
    outcome: str
    correct: bool = False
    error: str | None = None
    facts: tuple[bool, ...] = ()
    facts_total: int = 0
    sources_cited: bool | None = None
    quotes_emitted: int = 0
    quotes_reattributed: int = 0
    quotes_dropped: int = 0
    quotes_verbatim: int = 0
    wall_ms: int = 0
    rewrite_ms: int | None = None
    embed_ms: int | None = None  # summed embed-request ledger latency, all attempts
    search_ms: int | None = None  # summed local cosine + RRF time, all attempts
    embed_search_ms: int | None = None  # legacy: only set when loading a pre-split --save file
    rerank_ms: int | None = None
    answer_ms: int | None = None
    tok_s: float | None = None
    retries: int = 0
    model_called: bool = True
    answer: str = ""
    ledger: tuple[CallRecord, ...] = ()

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def all_facts(self) -> bool:
        return bool(self.facts) and all(self.facts)


Question = rag_module.ControlQuestion | rag_module.UnanswerableQuestion
AskFn = Callable[[Backend, Question, int, OnCallFn], rag_cli.ModeRun]


def default_ask(
    backend: Backend, question: Question, run: int, on_call: OnCallFn
) -> rag_cli.ModeRun:
    """One answer through the permanent pipeline; a fresh agent and params per cell."""
    config = isolated_config(backend.config)
    apply_rag_settings(config.params)
    identity = {
        "command": "ragbench",
        "backend": backend.name,
        "run": run,
        "question_id": question.id,
    }
    agent = rag_cli.build_agent(
        config,
        backend.db_path,
        DAY,
        aux_config=config,
        week=WEEK,
        extra=identity,
        on_call=on_call,
    )
    return rag_cli.run_mode(
        agent,
        question.question,
        mode="cite",
        command="ragbench",
        question_id=question.id,
        day=DAY,
        week=WEEK,
        extra=identity,
        on_call=on_call,
    )


def outcome_of(mode_run: rag_cli.ModeRun) -> str:
    cited = mode_run.cited
    if cited is None:
        return "error"
    return "answered" if cited.status == "answer" else cited.reason or "bad_json"


def ledger_cost(entry: LedgerEntry, prices: PriceTable | None) -> float | None:
    if entry.endpoint == "local" or entry.stage == "search":
        return 0.0
    if entry.result is None:  # cloud embedding
        return embed_cost_usd(entry.model, entry.prompt_tokens)
    price = prices.price_of(entry.model) if prices is not None else None
    if price is None:
        return None
    cached = getattr(getattr(entry.result, "usage", None), "cached_tokens", None)
    return call_cost(entry.prompt_tokens, entry.completion_tokens, cached, price)


def _stage_ms(ledger: Sequence[CallRecord], stage: str) -> int | None:
    values = [c.latency_ms for c in ledger if c.stage == stage]
    return sum(values) if values else None


def _quote_counts(cited: Any) -> tuple[int, int, int, int]:
    if cited is None:
        return 0, 0, 0, 0
    quotes = cited.quotes
    verbatim = sum(1 for q in quotes if q.verified)
    reattributed = sum(1 for q in quotes if q.claimed is not None)
    return len(quotes), reattributed, len(quotes) - verbatim, verbatim


def build_cell(
    backend: str,
    question: Question,
    run: int,
    mode_run: rag_cli.ModeRun,
    ledger: Sequence[CallRecord],
    wall_ms: int,
    retries: int,
) -> BenchRun:
    cited = mode_run.cited
    outcome = outcome_of(mode_run)
    answerable = isinstance(question, rag_module.ControlQuestion)
    emitted, reattributed, dropped, verbatim = _quote_counts(cited)
    cell = BenchRun(
        backend=backend,
        kind="answerable" if answerable else "unanswerable",
        question_id=question.id,
        run=run,
        outcome=outcome,
        quotes_emitted=emitted,
        quotes_reattributed=reattributed,
        quotes_dropped=dropped,
        quotes_verbatim=verbatim,
        wall_ms=wall_ms,
        retries=retries,
        model_called=mode_run.model_called,
        answer=cited.answer if cited is not None and cited.status == "answer" else "",
        ledger=tuple(ledger),
    )
    if outcome == "error":
        cell.error = "нет структурного результата cite"
    if answerable:
        score = rag_module.score_answer(question, mode_run.text, mode_run.ctx, cited)
        cell.facts = score.facts
        cell.facts_total = len(question.expect)
        cell.sources_cited = any(score.sources_cited) if score.sources_cited else None
        cell.correct = outcome == "answered" and score.complete
    else:
        refused = rag_module.unanswerable_outcome(mode_run.text, cited) == "refused"
        cell.correct = refused and outcome in REFUSALS
    cell.rewrite_ms = _stage_ms(ledger, "rewrite")
    cell.rerank_ms = _stage_ms(ledger, "rerank")
    cell.answer_ms = _stage_ms(ledger, "answer")
    cell.embed_ms = _stage_ms(ledger, "embed")
    cell.search_ms = _stage_ms(ledger, "search")
    answer_entry = next((c for c in ledger if c.stage == "answer"), None)
    if (
        answer_entry is not None
        and answer_entry.completion_tokens is not None
        and answer_entry.latency_ms > 0
    ):
        cell.tok_s = answer_entry.completion_tokens / (answer_entry.latency_ms / 1000)
    return cell


def _error_cell(
    backend: str,
    question: Question,
    run: int,
    message: str,
    ledger: Sequence[CallRecord],
    wall_ms: int,
    retries: int,
) -> BenchRun:
    answerable = isinstance(question, rag_module.ControlQuestion)
    return BenchRun(
        backend=backend,
        kind="answerable" if answerable else "unanswerable",
        question_id=question.id,
        run=run,
        outcome="error",
        error=message,
        facts_total=len(question.expect) if answerable else 0,
        wall_ms=wall_ms,
        retries=retries,
        model_called=bool(ledger),
        ledger=tuple(ledger),
    )


NONTTY_INTERVAL = 10.0  # piped stderr: a full line at most this often
_CLEAR_LINE = "\r\x1b[K"
_tick_lock = threading.Lock()
_tick_visible = False  # an in-place tick line is currently on the terminal


def _stderr_is_tty() -> bool:
    return bool(console.err.is_terminal)


def say_err(message: str, **kwargs: Any) -> None:
    """Print a stderr line, first wiping a live in-place tick (no half-overwritten text)."""
    global _tick_visible
    with _tick_lock:
        if _tick_visible:
            console.err.file.write(_CLEAR_LINE)
            console.err.file.flush()
            _tick_visible = False
        console.err.print(message, **kwargs)


class Heartbeat:
    """Stderr pulse so a long local call never looks frozen.

    tty: one line rewritten in place (cleared before any other output);
    non-tty: a full line at most every NONTTY_INTERVAL s.
    """

    def __init__(
        self,
        label: str,
        *,
        is_tty: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.label = label
        self._is_tty = is_tty or _stderr_is_tty  # resolved per tick, never bound at import
        self._clock = clock
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = 0.0
        self._last_line = 0.0

    def _run(self) -> None:
        while not self._stop.wait(PROGRESS_INTERVAL):
            self._pulse()

    def _pulse(self) -> None:
        global _tick_visible
        now = self._clock()
        text = f"  … {self.label} {int(now - self._start)} s"
        with _tick_lock:
            if self._stop.is_set():
                return
            if self._is_tty():
                console.err.file.write(f"{_CLEAR_LINE}{text}")
                console.err.file.flush()
                _tick_visible = True
            elif now - self._last_line >= NONTTY_INTERVAL:
                self._last_line = now
                console.err.print(text, style="dim", markup=False, highlight=False)

    def __enter__(self) -> Heartbeat:
        self._start = self._last_line = self._clock()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        global _tick_visible
        self._stop.set()
        self._thread.join(timeout=2)
        with _tick_lock:
            if _tick_visible:
                console.err.file.write(_CLEAR_LINE)
                console.err.file.flush()
                _tick_visible = False


def _run_cell(
    backend: Backend,
    question: Question,
    run: int,
    *,
    ask: AskFn,
    prices: PriceTable | None,
    clock: Callable[[], float],
    pause: Callable[[float], None],
    heartbeat: Callable[[str], AbstractContextManager] | None,
) -> BenchRun:
    ledger: list[CallRecord] = []
    label = f"{backend.name} #{question.id} прогон {run}"
    say_err(f"→ {label}", style="dim", markup=False)

    def on_call(entry: LedgerEntry) -> None:
        ledger.append(
            CallRecord(
                stage=entry.stage,
                model=entry.model,
                endpoint=entry.endpoint,
                prompt_tokens=entry.prompt_tokens,
                completion_tokens=entry.completion_tokens,
                latency_ms=entry.latency_ms,
                cost=ledger_cost(entry, prices),
            )
        )
        say_err(f"  {entry.stage} {entry.latency_ms / 1000:.1f} s", style="dim", markup=False)

    retries = 0
    started = clock()
    guard = heartbeat(label) if heartbeat is not None else contextlib.nullcontext()
    try:
        with guard:
            try:
                mode_run = ask(backend, question, run, on_call)
            except AdventError as error:
                if error.exit_code not in rag_cli.TRANSIENT_EXIT_CODES:
                    raise
                say_err(f"{label}: {error.message} — повтор", style="yellow", markup=False)
                pause(RETRY_PAUSE_S)
                retries = 1
                mode_run = ask(backend, question, run, on_call)
    except Exception as error:  # noqa: BLE001 - one failed cell must not stop a long bench
        message = getattr(error, "message", None) or str(error) or type(error).__name__
        console.warn(f"{label}: ошибка — {message}")
        return _error_cell(
            backend.name, question, run, message, ledger, int((clock() - started) * 1000), retries
        )
    wall_ms = int((clock() - started) * 1000)  # includes the retry pause
    return build_cell(backend.name, question, run, mode_run, ledger, wall_ms, retries)


# ---------------------------------------------------------------------------
# the bench
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OfflineReport:
    attempted: dict[str, int]
    blocked: dict[str, int]
    completed: dict[str, int]
    ledger_local: int
    ledger_cloud: int

    @property
    def ok(self) -> bool:
        return self.blocked.get("cloud", 0) == 0 and self.ledger_cloud == 0


@dataclass(slots=True)
class BenchResult:
    runs: list[BenchRun] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)  # ran, in execution order
    skipped: list[str] = field(default_factory=list)
    n_runs: int = 0
    plan: Plan | None = None
    offline: OfflineReport | None = None
    failed: str | None = None  # a backend's setup failed after earlier results were kept
    settings: dict[str, Any] = field(default_factory=dict)
    question_count: int | None = None  # set when rebuilt from --save files (no Plan)

    @property
    def n_questions(self) -> int:
        if self.plan is not None:
            return len(self.plan.questions) + len(self.plan.unanswerable)
        return self.question_count or 0


def run_bench(
    factories: Mapping[str, Callable[[], Backend | None]],
    plan: Plan,
    *,
    runs: int,
    ask: AskFn = default_ask,
    prices: PriceTable | None = None,
    offline_mod: Any = offline,
    clock: Callable[[], float] = time.perf_counter,
    pause: Callable[[float], None] | None = None,
    heartbeat: Callable[[str], AbstractContextManager] | None = None,
    on_run: Callable[[BenchRun, int], None] | None = None,
) -> BenchResult:
    """Cloud phase, then guard + counter reset, then local; run-major order in each backend."""
    if "cloud" in factories and offline_mod.is_enabled():
        raise AdventError(
            "Offline-режим уже включён в этом процессе — облачный backend недоступен.",
            hint="Запусти `adventlocal rag --backends local` или сними ADVENT_OFFLINE.",
        )
    sleeper = pause if pause is not None else _pause
    result = BenchResult(n_runs=runs, plan=plan)
    items: list[Question] = [*plan.questions, *plan.unanswerable]
    emit = on_run if on_run is not None else print_run_line
    for name in BACKENDS:
        if name not in factories:
            continue
        if name == "local":
            offline_mod.enable()
            offline_mod.reset_counters()
        try:
            backend = factories[name]()
        except AdventError as error:
            if not result.runs:
                raise
            result.failed = f"{name}: {error.message}" + (f" ({error.hint})" if error.hint else "")
            result.skipped.append(name)
            continue
        if backend is None:
            result.skipped.append(name)
            continue
        result.backends.append(name)
        result.settings[name] = effective_settings(backend)
        mine: list[BenchRun] = []
        for run in range(1, runs + 1):
            for question in items:
                cell = _run_cell(
                    backend,
                    question,
                    run,
                    ask=ask,
                    prices=prices,
                    clock=clock,
                    pause=sleeper,
                    heartbeat=heartbeat,
                )
                mine.append(cell)
                result.runs.append(cell)
                emit(cell, runs)
        if name == "local":
            counts = offline_mod.counters()
            ledger_calls = [c for r in mine for c in r.ledger if c.stage != "search"]
            result.offline = OfflineReport(
                attempted=dict(counts.attempted),
                blocked=dict(counts.blocked),
                completed=dict(counts.completed),
                ledger_local=sum(1 for c in ledger_calls if c.endpoint == "local"),
                ledger_cloud=sum(1 for c in ledger_calls if c.endpoint != "local"),
            )
    return result


def effective_settings(backend: Backend) -> dict[str, Any]:
    run = backend.run
    return {
        "model": backend.config.model,
        "endpoint": backend.endpoint,
        "index": backend.db_path.name,
        "embed_model": getattr(run, "model", None),
        "corpus_rev": getattr(run, "corpus_rev", None),
        "strategy": STRATEGY,
        "k": K,
        "k_before": K_BEFORE,
        "threshold": THRESHOLD,
        "rewrite": True,
        "rerank": True,
        "cite": True,
        "aux_reasoning": False,
        "answer_max_tokens": ANSWER_MAX_TOKENS,
        "temperature": None,
        "top_p": None,
        "protocol": PROTOCOLS[backend.name],
    }


# ---------------------------------------------------------------------------
# per-run product line
# ---------------------------------------------------------------------------

REFUSAL_TAG = {"empty_context": "порог", "model_unknown": "модель"}
OUTCOME_TAG = {
    "unverified": "неподтв.",
    "bad_json": "формат",
    "truncated": "обрыв",
    "no_index": "нет индекса",
}


def cut_cells(text: str, limit: int) -> str:
    """Cut by terminal cells (an emoji or CJK char is two), with an ellipsis."""
    if limit <= 0:
        return ""
    if cell_len(text) <= limit:
        return text
    out = ""
    for ch in text:
        if cell_len(out + ch) > limit - 1:
            break
        out += ch
    return out + "…"


def run_verdict(cell: BenchRun) -> str:
    if cell.error is not None:
        return f"ошибка: {' '.join(cell.error.split())}"
    if cell.kind == "unanswerable":
        if cell.correct:
            return f"не знаю ✓ ({REFUSAL_TAG.get(cell.outcome, cell.outcome)})"
        return "ответил ✗" if cell.outcome == "answered" else f"✗ {_outcome_text(cell.outcome)}"
    if cell.outcome == "answered":
        mark = "✓" if cell.all_facts else "✗"
        return f"{mark} {sum(cell.facts)}/{cell.facts_total}"
    return _outcome_text(cell.outcome)


def _outcome_text(outcome: str) -> str:
    if outcome in REFUSAL_TAG:
        return f"не знаю ({REFUSAL_TAG[outcome]})"
    return OUTCOME_TAG.get(outcome, outcome)


def run_line(cell: BenchRun, n_runs: int, width: int) -> str:
    quotes = f"цитаты {cell.quotes_verbatim}/{cell.quotes_emitted}" if cell.quotes_emitted else ""
    head = (
        f"{cell.backend:<6} #{cell.question_id} прогон {cell.run}/{n_runs}  "
        f"{run_verdict(cell)}  "
        + (f"{quotes}  " if quotes else "")
        + f"{cell.wall_ms / 1000:.1f} s"
    )
    room = width - cell_len(head) - 4  # two spaces and «»
    text = " ".join(cell.answer.split())
    if not text or room < 8:
        return head
    return f"{head}  «{cut_cells(text, room)}»"


def print_run_line(cell: BenchRun, n_runs: int) -> None:
    console.out.print(run_line(cell, n_runs, console.out.width), markup=False, highlight=False)


# ---------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------


def _of(runs: Sequence[BenchRun], backend: str, kind: str | None = None) -> list[BenchRun]:
    return [r for r in runs if r.backend == backend and (kind is None or r.kind == kind)]


def _groups(cells: Sequence[BenchRun]) -> dict[int, list[BenchRun]]:
    out: dict[int, list[BenchRun]] = {}
    for cell in sorted(cells, key=lambda r: (r.question_id, r.run)):
        out.setdefault(cell.question_id, []).append(cell)
    return out


def facts_totals(cells: Sequence[BenchRun]) -> tuple[int, int]:
    """(found, total) over every attempt of answerable questions; an error cell counts 0 found."""
    answerable = [c for c in cells if c.kind == "answerable"]
    return sum(sum(c.facts) for c in answerable), sum(c.facts_total for c in answerable)


def questions_all_runs_complete(cells: Sequence[BenchRun]) -> tuple[int, int]:
    groups = _groups([c for c in cells if c.kind == "answerable"])
    good = sum(1 for g in groups.values() if all(c.ok and c.all_facts for c in g))
    return good, len(groups)


def quote_stats(cells: Sequence[BenchRun]) -> dict[str, int]:
    """Answers with quotes, answers with ALL quotes verbatim, and the raw quote counters."""
    with_quotes = [c for c in cells if c.quotes_emitted > 0]
    return {
        "answers_with_quotes": len(with_quotes),
        "all_verbatim": sum(1 for c in with_quotes if c.quotes_verbatim == c.quotes_emitted),
        "emitted": sum(c.quotes_emitted for c in cells),
        "reattributed": sum(c.quotes_reattributed for c in cells),
        "dropped": sum(c.quotes_dropped for c in cells),
        "verbatim": sum(c.quotes_verbatim for c in cells),
    }


def verdict_char(cell: BenchRun) -> str:
    if cell.error is not None:
        return "!"
    if cell.correct:
        return "✓"
    if cell.kind == "answerable" and cell.outcome in REFUSALS:
        return "б"
    if cell.outcome == "unverified":
        return "н"
    if cell.outcome in ("bad_json", "truncated"):
        return "ф"
    return "✗"


def _verdict_key(cell: BenchRun) -> tuple[str, bool]:
    return (cell.outcome, cell.all_facts if cell.kind == "answerable" else cell.correct)


def stability(cells: Sequence[BenchRun]) -> dict[int, dict[str, Any]]:
    """Per question: verdict pattern, same-verdict flag and all-runs-correct flag.

    A verdict is the outcome plus whether the answer was right (all facts / correct refusal),
    so `✓✗✓` is NOT stable. The two flags stay separate: `✗✗✗` is consistent, never correct.
    """
    out: dict[int, dict[str, Any]] = {}
    for qid, group in _groups(cells).items():
        out[qid] = {
            "pattern": "".join(verdict_char(c) for c in group),
            "same_verdict": len({_verdict_key(c) for c in group}) == 1,
            "all_correct": all(c.correct and c.ok for c in group),
            "times": [c.wall_ms for c in group if c.ok],
        }
    return out


def stability_counts(cells: Sequence[BenchRun]) -> tuple[int, int, int]:
    """(same verdict in all runs, all runs correct, questions)."""
    st = stability(cells)
    return (
        sum(1 for s in st.values() if s["same_verdict"]),
        sum(1 for s in st.values() if s["all_correct"]),
        len(st),
    )


def p90(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(0.9 * len(ordered)) - 1, 0)]


def _med(values: Sequence[float | int | None]) -> float | None:
    known = [v for v in values if v is not None]
    return float(median(known)) if known else None


@dataclass(frozen=True, slots=True)
class Tokens:
    """Sum of known usage; `partial` = some included call reported none (a lower bound)."""

    value: int | None
    partial: bool = False

    def __str__(self) -> str:
        if self.value is None:
            return "?"
        return f"≥{self.value}" if self.partial else str(self.value)


def _sum_known(values: Sequence[int | None]) -> Tokens:
    known = [v for v in values if v is not None]
    return Tokens(sum(known) if known else None, len(known) < len(values))


def token_totals(cells: Sequence[BenchRun]) -> dict[str, Tokens]:
    calls = [c for cell in cells for c in cell.ledger]
    answer = [c for c in calls if c.stage == "answer"]
    aux = [c for c in calls if c.stage in ("rewrite", "rerank")]
    embed = [c for c in calls if c.stage == "embed"]
    return {
        "answer_prompt": _sum_known([c.prompt_tokens for c in answer]),
        "answer_completion": _sum_known([c.completion_tokens for c in answer]),
        "aux_prompt": _sum_known([c.prompt_tokens for c in aux]),
        "aux_completion": _sum_known([c.completion_tokens for c in aux]),
        "embed": _sum_known([c.prompt_tokens for c in embed]),
    }


def cost_total(cells: Sequence[BenchRun]) -> float | None:
    """Sum of completed calls; None when any call has an unknown price."""
    costs = [c.cost for cell in cells for c in cell.ledger]
    if not costs or any(c is None for c in costs):
        return None if costs else 0.0
    return float(sum(c for c in costs if c is not None))


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _secs(ms: float | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.1f} s"


def _frac(num: int, den: int) -> str:
    return f"{num}/{den}"


def _backend_table(title: str, backends: Sequence[str]) -> Table:
    table = Table(title=title, title_justify="left")
    table.add_column("метрика", overflow="fold")
    for name in backends:
        table.add_column(SHOWN[name], justify="right", no_wrap=True, min_width=10)
    return table


def quality_table(runs: Sequence[BenchRun], backends: Sequence[str]) -> Table:
    table = _backend_table("Качество", backends)
    rows: list[tuple[str, list[str]]] = [
        ("факты найдено", []),
        ("вопросов со всеми фактами во всех прогонах", []),
        ("ответов, где все цитаты дословны", []),
        ("цитат: выдано / источник исправлен / отброшено / дословных", []),
        ("источник в цитатах (ответов)", []),
        ("«не знаю» на отвечаемых", []),
        ("неотвечаемые: верный отказ", []),
        ("ошибки", []),
    ]
    for name in backends:
        mine = _of(runs, name)
        ans = _of(runs, name, "answerable")
        un = _of(runs, name, "unanswerable")
        found, total = facts_totals(mine)
        good, n_q = questions_all_runs_complete(mine)
        qs = quote_stats(mine)
        answered = [c for c in ans if c.outcome == "answered" and c.ok]
        sourced = sum(1 for c in answered if c.sources_cited)
        refused = sum(1 for c in ans if c.outcome in REFUSALS)
        cells = [
            _frac(found, total),
            _frac(good, n_q),
            _frac(qs["all_verbatim"], qs["answers_with_quotes"]),
            f"{qs['emitted']}/{qs['reattributed']}/{qs['dropped']}/{qs['verbatim']}",
            _frac(sourced, len(answered)),
            _frac(refused, len(ans)),
            _frac(sum(1 for c in un if c.correct), len(un)) if un else "—",
            _frac(sum(1 for c in mine if not c.ok), len(mine)),
        ]
        for row, cell in zip(rows, cells, strict=True):
            row[1].append(cell)
    for label, cells in rows:
        table.add_row(label, *cells)
    return table


def speed_table(runs: Sequence[BenchRun], backends: Sequence[str]) -> Table:
    table = _backend_table("Скорость", backends)
    rows: list[tuple[str, list[str]]] = [
        (label, [])
        for label in (
            "завершено ответов",
            "стена на вопрос, медиана",
            "стена на вопрос, p90",
            "rewrite, медиана",
            "embed, медиана",
            "search, медиана",
            "rerank, медиана",
            "ответ, медиана",
            "tok/s ответа, медиана",
            "токены ответа prompt/completion",
            "токены aux prompt/completion",
            "токены embed",
            "$ всего",
        )
    ]
    for name in backends:
        mine = _of(runs, name)
        done = [c for c in mine if c.ok]
        tokens = token_totals(mine)
        tps = _med([c.tok_s for c in done])
        cost = cost_total(mine)
        cells = [
            _frac(len(done), len(mine)),
            _secs(_med([c.wall_ms for c in done])),
            _secs(p90([c.wall_ms for c in done])),
            _secs(_med([c.rewrite_ms for c in done])),
            _secs(_med([c.embed_ms for c in done])),
            _secs(_med([c.search_ms for c in done])),
            _secs(_med([c.rerank_ms for c in done])),
            _secs(_med([c.answer_ms for c in done])),
            "—" if tps is None else f"{tps:.1f}",
            f"{tokens['answer_prompt']}/{tokens['answer_completion']}",
            f"{tokens['aux_prompt']}/{tokens['aux_completion']}",
            str(tokens["embed"]),
            "—" if cost is None else f"{cost:.4f}",
        ]
        for row, cell in zip(rows, cells, strict=True):
            row[1].append(cell)
    for label, cells in rows:
        table.add_row(label, *cells)
    notes = [TPS_NOTE, COST_NOTE]
    if any(t.partial for name in backends for t in token_totals(_of(runs, name)).values()):
        notes.append(PARTIAL_NOTE)
    if any(c.embed_search_ms is not None for c in runs if c.backend in backends):
        notes.append(LEGACY_NOTE)
    table.caption = "\n".join(notes)
    table.caption_justify = "left"
    return table


def _spread(times: Sequence[int]) -> str:
    if not times:
        return "—"
    lo, hi = min(times) / 1000, max(times) / 1000
    return f"{lo:.1f} s" if lo == hi else f"{lo:.1f}–{hi:.1f} s"


def stability_table(runs: Sequence[BenchRun], backends: Sequence[str]) -> Table:
    table = Table(title="Стабильность", title_justify="left")
    table.add_column("#", justify="right", no_wrap=True)
    per = {name: stability(_of(runs, name)) for name in backends}
    for name in backends:
        table.add_column(f"{SHOWN[name]} вердикты", no_wrap=True, min_width=len(SHOWN[name]) + 9)
        table.add_column(f"{SHOWN[name]} время", justify="right", no_wrap=True, min_width=9)
    ids = sorted({qid for st in per.values() for qid in st})
    for qid in ids:
        cells: list[str] = []
        for name in backends:
            entry = per[name].get(qid)
            cells += [entry["pattern"], _spread(entry["times"])] if entry else ["—", "—"]
        table.add_row(str(qid), *cells)
    table.caption = (
        "✓ верно · ✗ неверно · н неподтверждено · ф формат · б «не знаю» на отвечаемом · ! ошибка"
    )
    table.caption_justify = "left"
    return table


# ---------------------------------------------------------------------------
# conclusions (every statement is computed; a tie names no leader)
# ---------------------------------------------------------------------------


def _plural(n: int, one: str, few: str, many: str) -> str:
    if 11 <= n % 100 <= 14:
        return many
    return one if n % 10 == 1 else few if 2 <= n % 10 <= 4 else many


def quality_line(runs: Sequence[BenchRun]) -> str:
    lf, lt = facts_totals(_of(runs, "local"))
    cf, ct = facts_totals(_of(runs, "cloud"))
    shown = f"local {lf}/{lt}, cloud {cf}/{ct}"
    if lt == 0 or ct == 0:
        return f"качество: сравнить нечем ({shown})"
    left, right = lf * ct, cf * lt
    if left == right:
        return f"качество: ничья ({shown})"
    leader = "local" if left > right else "cloud"
    note = f" — разница {abs(lf - cf)}, {SMALL_SAMPLE}" if lt == ct and abs(lf - cf) <= 1 else ""
    return f"качество: выше {leader} ({shown}){note}"


def speed_line(runs: Sequence[BenchRun]) -> str:
    local = _med([c.wall_ms for c in _of(runs, "local") if c.ok])
    cloud = _med([c.wall_ms for c in _of(runs, "cloud") if c.ok])
    if not local or not cloud:
        return "скорость: сравнить нечем — нет завершённых ответов у одного из backend'ов"
    shown = f"медиана на вопрос: local {local / 1000:.1f} s, cloud {cloud / 1000:.1f} s"
    if local == cloud:
        return f"скорость: ничья ({shown})"
    if local > cloud:
        return f"скорость: local медленнее в {local / cloud:.1f}× ({shown})"
    return f"скорость: local быстрее в {cloud / local:.1f}× ({shown})"


def stability_lines(runs: Sequence[BenchRun]) -> list[str]:
    l_same, l_ok, l_n = stability_counts(_of(runs, "local"))
    c_same, c_ok, c_n = stability_counts(_of(runs, "cloud"))
    lines = [
        "одинаковый вердикт во всех прогонах (в том числе стабильно неверный; "
        "вердикт = исход + верность): "
        f"local {l_same}/{l_n}, cloud {c_same}/{c_n}",
        f"все прогоны верны: local {l_ok}/{l_n}, cloud {c_ok}/{c_n}",
    ]
    if l_n and c_n:
        left, right = l_ok * c_n, c_ok * l_n
        if left == right:
            lines.append(f"стабильность: ничья по «все прогоны верны» (local {l_ok}, cloud {c_ok})")
        else:
            leader = "local" if left > right else "cloud"
            lines.append(f"стабильность: чаще верен во всех прогонах {leader}")
    return lines


def conclusion_lines(result: BenchResult) -> list[str]:
    """Comparisons only when both backends produced something; one backend prints none."""
    runs = result.runs
    lines: list[str] = []
    if {"local", "cloud"} <= set(result.backends):
        lines += [quality_line(runs), speed_line(runs), *stability_lines(runs)]
    n_q = result.n_questions
    lines.append(
        f"{result.n_runs} {_plural(result.n_runs, 'прогон', 'прогона', 'прогонов')} × "
        f"{n_q} {_plural(n_q, 'вопрос', 'вопроса', 'вопросов')} — малая выборка"
    )
    return lines


def offline_line(report: OfflineReport) -> str:
    http = (
        f"HTTP: попыток локальных {report.attempted.get('local', 0)}, "
        f"облачных {report.attempted.get('cloud', 0)}; "
        f"заблокировано {report.blocked.get('cloud', 0)}; "
        f"завершено локальных {report.completed.get('local', 0)}, "
        f"облачных {report.completed.get('cloud', 0)}"
    )
    return (
        f"локальный прогон: модельных вызовов локальных {report.ledger_local}, "
        f"облачных {report.ledger_cloud} (заблокировано {report.blocked.get('cloud', 0)}) · {http}"
    )


def print_report(result: BenchResult) -> None:
    """Counter line, three tables, conclusions; a failed acceptance repeats in red at the end."""
    out = console.out
    if result.offline is not None:
        out.print("")
        out.print(offline_line(result.offline), markup=False, highlight=False)
    if result.runs:
        out.print("")
        for table in (
            quality_table(result.runs, result.backends),
            speed_table(result.runs, result.backends),
            stability_table(result.runs, result.backends),
        ):
            out.print(table)
        for line in conclusion_lines(result):
            out.print(line, markup=False, highlight=False)
    if result.skipped:
        out.print(f"пропущено: {', '.join(result.skipped)}", markup=False, highlight=False)
    if result.offline is not None and not result.offline.ok:
        out.print(
            "[bold red]offline-приёмка НЕ пройдена: "
            f"облачных попыток {result.offline.attempted.get('cloud', 0)}, "
            f"заблокировано {result.offline.blocked.get('cloud', 0)}, "
            f"облачных вызовов в ledger {result.offline.ledger_cloud}[/bold red]",
            highlight=False,
        )
    if result.failed:
        out.print(f"[bold red]{rich_escape(result.failed)}[/bold red]", highlight=False)


def exit_code(result: BenchResult) -> int:
    if result.offline is not None and not result.offline.ok:
        return 1
    if result.failed:
        return 1
    return 0 if any(c.ok for c in result.runs) else 1


# ---------------------------------------------------------------------------
# --save
# ---------------------------------------------------------------------------


def result_to_json(result: BenchResult) -> dict[str, Any]:
    plan = result.plan
    return {
        "week": WEEK,
        "day": DAY,
        "runs": result.n_runs,
        "backends": result.backends,
        "skipped": result.skipped,
        "failed": result.failed,
        "corpus_rev": plan.corpus_rev if plan else None,
        "questions": [q.id for q in plan.questions] if plan else [],
        "unanswerable": [q.id for q in plan.unanswerable] if plan else [],
        "settings": result.settings,
        "offline": dataclasses.asdict(result.offline) if result.offline else None,
        "results": [dataclasses.asdict(r) for r in result.runs],
    }


def save_json(result: BenchResult, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result_to_json(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        raise AdventError(f"Не удалось записать {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# command entry
# ---------------------------------------------------------------------------


def parse_backends(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    bad = [n for n in names if n not in BACKENDS]
    if bad or not names:
        raise AdventError(
            f"Неизвестный backend: {', '.join(bad) or value!r}.", hint="Доступно: local, cloud."
        )
    return [n for n in BACKENDS if n in names]


def parse_ids(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    try:
        return [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise AdventError(
            f"--questions: нужны целые id через запятую, получено {value!r}."
        ) from exc


def run_rag_command(
    *,
    runs: int = DEFAULT_RUNS,
    questions: str | None = None,
    backends: str = "local,cloud",
    unanswerable: bool = True,
    save: Path | None = None,
    url: str | None = None,
    prices_loader: Callable[[], PriceTable] | None = None,
) -> int:
    """Everything the `rag` command does; returns the process exit code."""
    wanted = parse_backends(backends)
    ids = parse_ids(questions)
    cloud_cfg: Config | None = None
    if "cloud" in wanted:
        # ADVENT_OFFLINE in env/.env turns the guard on at the first HTTP call: refuse before.
        config_module.load_env()
        if offline.is_enabled() or config_module.offline_env():
            raise AdventError(
                "Offline-режим включён (ADVENT_OFFLINE или guard процесса) — "
                "облачный backend недоступен.",
                hint="Запусти `adventlocal rag --backends local` или сними ADVENT_OFFLINE.",
            )
        cloud_cfg = cloud_config()
        if cloud_cfg is None:
            console.warn("MISTRAL_API_KEY не задан — облачная часть пропущена.")
            wanted = [n for n in wanted if n != "cloud"]
            if not wanted:
                raise ConfigError("Не найден MISTRAL_API_KEY, а других backend'ов не выбрано.")
    local_url = None
    if "local" in wanted:
        local_url = config_module.normalize_base_url(url or oc.default_url())
        config_module.validate_loopback_url(str(local_url), "локальный сервер")
    plan = prepare_plan(wanted, ids=ids, unanswerable=unanswerable)
    for note in plan.notes:
        console.warn(note)
    prices: PriceTable | None = None
    if cloud_cfg is not None:
        from week_01.models_bench import load_prices, warn_if_prices_stale

        prices = (prices_loader or load_prices)()
        warn_if_prices_stale(prices)

    factories: dict[str, Callable[[], Backend | None]] = {}
    if cloud_cfg is not None:
        factories["cloud"] = lambda: make_cloud_backend(cloud_cfg)
    if "local" in wanted:
        factories["local"] = lambda: make_local_backend(local_config(local_url))
    result = run_bench(factories, plan, runs=runs, prices=prices, heartbeat=Heartbeat)
    print_report(result)
    if save is not None:
        save_json(result, save)
        console.err.print(f"сохранено: {save}", style="dim", markup=False)
    return exit_code(result)


# ---------------------------------------------------------------------------
# --report: re-render from --save files (no network, no guard)
# ---------------------------------------------------------------------------

_CELL_FIELDS = {f.name for f in dataclasses.fields(BenchRun)}


def _call_from_json(raw: Mapping[str, Any]) -> CallRecord:
    return CallRecord(
        stage=str(raw.get("stage", "")),
        model=str(raw.get("model", "")),
        endpoint=str(raw.get("endpoint", "")),
        prompt_tokens=raw.get("prompt_tokens"),
        completion_tokens=raw.get("completion_tokens"),
        latency_ms=int(raw.get("latency_ms") or 0),
        cost=raw.get("cost"),
    )


def cell_from_json(raw: Mapping[str, Any]) -> BenchRun:
    """A saved cell; every field but the identity is optional (older files lack newer ones)."""
    data = {k: v for k, v in raw.items() if k in _CELL_FIELDS and k != "ledger"}
    data["facts"] = tuple(bool(x) for x in raw.get("facts") or ())
    data["ledger"] = tuple(_call_from_json(c) for c in raw.get("ledger") or ())
    cell = BenchRun(**data)
    if "embed_ms" not in raw:  # pre-split file: embed from the ledger, search stays unknown
        cell.embed_ms = _stage_ms(cell.ledger, "embed")
    return cell


def load_result_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdventError(f"Не удалось прочитать {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise AdventError(f"{path}: это не файл --save дня 28 (нет списка results).")
    try:
        data["_cells"] = [cell_from_json(r) for r in data["results"]]
    except (TypeError, ValueError, AttributeError) as exc:
        raise AdventError(f"{path}: повреждённая запись результата ({exc}).") from exc
    return data


def merge_result_files(paths: Sequence[Path]) -> BenchResult:
    """One BenchResult from several --save files; refuses mixed snapshots or question sets."""
    files = [(path, load_result_file(path)) for path in paths]
    revs = {str(path): data.get("corpus_rev") for path, data in files}
    if len(set(revs.values())) > 1:
        shown = ", ".join(f"{Path(p).name}: {str(r)[:12]}" for p, r in revs.items())
        raise AdventError(
            f"Файлы собраны из разных снимков индекса (corpus_rev: {shown}): сравнение нечестное."
        )
    sets = {
        str(path): (
            frozenset(data.get("questions") or ()),
            frozenset(data.get("unanswerable") or ()),
        )
        for path, data in files
    }
    if len(set(sets.values())) > 1:
        shown = "; ".join(
            f"{Path(p).name}: вопросы {sorted(q)}, без ответа {sorted(u)}"
            for p, (q, u) in sets.items()
        )
        raise AdventError(f"Наборы вопросов в файлах различаются ({shown}).")
    result = BenchResult()
    seen: dict[str, str] = {}
    for path, data in files:
        for name in data.get("backends") or ():
            if name not in BACKENDS:
                raise AdventError(f"{path}: неизвестный backend {name!r}.")
            if name in seen:
                raise AdventError(
                    f"Backend {name} есть и в {Path(seen[name]).name}, и в {path.name}: "
                    "прогоны не склеить без дублей."
                )
            seen[name] = str(path)
        result.runs += data["_cells"]
        result.settings.update(data.get("settings") or {})
        if result.offline is None and isinstance(data.get("offline"), dict):
            raw = data["offline"]
            result.offline = OfflineReport(
                attempted=dict(raw.get("attempted") or {}),
                blocked=dict(raw.get("blocked") or {}),
                completed=dict(raw.get("completed") or {}),
                ledger_local=int(raw.get("ledger_local") or 0),
                ledger_cloud=int(raw.get("ledger_cloud") or 0),
            )
    result.backends = [n for n in BACKENDS if n in seen]
    run_counts = {int(d.get("runs") or 0) for _, d in files}
    if len(run_counts) > 1:
        console.warn(f"Число прогонов в файлах разное ({sorted(run_counts)}) — взято наибольшее.")
    result.n_runs = max(run_counts)
    first = files[0][1]
    result.question_count = len(first.get("questions") or ()) + len(first.get("unanswerable") or ())
    for _, data in files:
        for name in data.get("skipped") or ():
            if name not in result.backends and name not in result.skipped:
                result.skipped.append(name)
        failed = data.get("failed")
        if failed and str(failed).split(":", 1)[0] not in result.backends:
            result.failed = str(failed)
    return result


def run_report_command(paths: Sequence[Path]) -> int:
    """Render the report from saved files; touches no network, guard or LM Studio."""
    result = merge_result_files(paths)
    console.err.print(
        f"отчёт из {len(paths)} файл(ов): {', '.join(p.name for p in paths)} — без сети",
        style="dim",
        markup=False,
    )
    if any(c.embed_search_ms is not None for c in result.runs):
        console.warn("файл старого формата: время поиска (search) в нём не сохранено.")
    print_report(result)
    return exit_code(result)
