"""Week 06 entry point `adventlocal`: a local LLM over plain HTTP, and local-vs-cloud comparison.

stdout is the product (model answer, tables, conclusions); progress, footers,
warnings and the model's reasoning go to stderr, as everywhere in this repo.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import typer
from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import config as config_module
from advent_core import console
from advent_core.config import Config, ConfigError
from advent_core.errors import AdventError, ConfigurationError
from advent_core.journal import log_call, log_service_call
from advent_core.params import GenerationParams
from advent_core.telemetry import CallResult, Usage
from week_01.models_bench import PriceTable, call_cost, load_prices, warn_if_prices_stale
from week_06 import codecheck, profiles, ragbench, tasks, vram
from week_06 import compare as compare_module
from week_06 import local_client as lc
from week_06 import service_client as svc

WEEK = 6
DAY = 26
DEFAULT_CLOUD_MODEL = "ministral-14b-latest"
DEFAULT_TASKS = "fact,alice,palindrome"
PROGRESS_INTERVAL = 3.0
EXCERPT_LINES = 12
CODE_FULL_LINES = 25
LOCAL_COST_NOTE = "* электричество и железо не считаем"
CLOUD_TPS_NOTE = (
    "tok/s облака = completion / всё время ответа: на коротком ответе это overhead, "
    "а не скорость генерации"
)

app = typer.Typer(
    help="Локальная LLM по HTTP (LM Studio): статус, один запрос, сравнение с облаком Mistral.",
    no_args_is_help=True,
    add_completion=False,
)

_URL_OPT = typer.Option(
    None, "--url", help="Адрес сервера (по умолчанию ADVENT_LOCAL_URL или 127.0.0.1:1234)."
)
_MODEL_OPT = typer.Option(lc.DEFAULT_MODEL, "--model", help="Идентификатор локальной модели.")
_MAX_TOKENS_OPT = typer.Option(lc.MAX_TOKENS, "--max-tokens", min=1, help="Лимит токенов ответа.")
_DEADLINE_OPT = typer.Option(
    lc.DEADLINE, "--deadline", min=1.0, help="Общий предел одного локального вызова, секунды."
)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _models_table(models: list[lc.LocalModel], base: str) -> Table:
    table = Table(title=f"Сервер {base}", title_justify="left")
    for name in ("модель", "state", "контекст", "квантизация", "arch"):
        table.add_column(name, overflow="fold")
    for model in models:
        table.add_row(
            model.id,
            model.state or "неизвестно",
            str(model.loaded_context_length) if model.loaded_context_length else "—",
            model.quantization or "—",
            model.arch or "—",
        )
    return table


def _find_lms() -> str | None:
    candidate = Path.home() / ".lmstudio" / "bin" / "lms.exe"
    if candidate.exists():
        return str(candidate)
    return shutil.which("lms")


def _lms_ps() -> str:
    """`lms ps` wakes the headless service and resets its settings: only on explicit --cli."""
    lms = _find_lms()
    if lms is None:
        return "lms не найден — только HTTP"
    try:
        done = subprocess.run(
            [lms, "ps"], capture_output=True, text=True, encoding="utf-8", timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"lms ps не выполнился: {exc}"
    return (done.stdout or done.stderr).strip() or "lms ps: пустой вывод"


@app.command("status")
def status_command(
    model: str = _MODEL_OPT,
    url: str | None = _URL_OPT,
    cli: bool = typer.Option(False, "--cli", help="Дополнительно выполнить `lms ps`."),
    show_all: bool = typer.Option(
        False, "--all", help="Показать и незагруженные модели (по умолчанию только loaded)."
    ),
    vram_flag: bool = typer.Option(False, "--vram", help="Показать занятую VRAM (nvidia-smi)."),
) -> None:
    """Загруженные модели сервера по HTTP; exit 2, если нужная модель не загружена."""
    base = (url or lc.default_url()).rstrip("/")
    models = lc.server_status(base)
    # state None everywhere = /v1/models fallback: nothing to filter on, show all
    state_known = any(m.state for m in models)
    shown = models if show_all or not state_known else [m for m in models if m.state == "loaded"]
    console.out.print(_models_table(shown, base))
    hidden = len(models) - len(shown)
    if hidden > 0:
        console.out.print(f"ещё {hidden} моделей не загружены (--all — показать)", highlight=False)
    if cli:
        console.out.print("через CLI:")
        console.out.print(_lms_ps(), markup=False, highlight=False)
    found = lc.check_ready(models, model, base)
    ctx = f", контекст {found.loaded_context_length}" if found.loaded_context_length else ""
    state = found.state or "неизвестно"
    quant = f", quant {rich_escape(found.quantization)}" if found.quantization else ""
    console.out.print(
        f"модель {rich_escape(model)}: готова (state: {rich_escape(state)}{ctx}{quant})",
        highlight=False,
    )
    if vram_flag:
        sample = vram.query_nvidia_smi()
        report = (
            vram.VramReport(*(sample[0], sample[0], sample[1])) if sample else vram.VramReport()
        )
        console.out.print(rich_escape(vram.describe(report)), highlight=False)


# ---------------------------------------------------------------------------
# shared call plumbing
# ---------------------------------------------------------------------------


class Ticker:
    """Background stderr progress line every PROGRESS_INTERVAL s, so the screen never freezes."""

    def __init__(self, *, thinking: bool) -> None:
        self.thinking = thinking
        self.chars = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = 0.0

    def add_reasoning(self, text: str) -> None:
        self.chars += len(text)

    def _run(self) -> None:
        while not self._stop.wait(PROGRESS_INTERVAL):
            seconds = int(time.monotonic() - self._start)
            suffix = f" (думает: {self.chars} симв.)" if self.thinking else ""
            console.err.print(f"  … {seconds} s{suffix}", style="dim", markup=False)

    def __enter__(self) -> Ticker:
        self._start = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _local_call_result(res: lc.LocalResult, requested: str, actual: str) -> CallResult:
    return CallResult(
        text=res.text,
        model_requested=requested,
        model_actual=actual,
        usage=res.usage or Usage(),
        latency_ms=int(res.latency_ms),
        stream=True,
        truncated=res.truncated,
        finish_reason=res.finish_reason,
        reasoning_text=res.reasoning_text or None,
    )


def _journal(
    result: CallResult,
    messages: list[dict],
    *,
    backend: str,
    task: str,
    run: int,
    ttft_ms: float | None = None,
    content_ms: float | None = None,
    verdict: str | None = None,
    error: str | None = None,
) -> None:
    log_call(
        result,
        messages,
        week=WEEK,
        day=DAY,
        error=error,
        extra={
            "backend": backend,
            "task": task,
            "run": run,
            "ttft_ms": ttft_ms,
            "content_ms": content_ms,
            "reasoning_tokens": result.usage.reasoning_tokens,
            "finish_reason": result.finish_reason,
            "verdict": verdict,
        },
    )


def _secs(ms: float | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.1f} s"


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def _read_question(question: str | None) -> tuple[str, bool]:
    if question is not None and question.strip():
        return question.strip(), False
    if sys.stdin.isatty():
        raise ConfigError("Нет вопроса: передай его аргументом или через stdin.")
    text = sys.stdin.read().strip()
    if not text:
        raise ConfigError("Нет вопроса: stdin пуст.")
    return text, True


def _ask_footer(res: lc.LocalResult, model_id: str) -> None:
    usage = res.usage
    if usage is None or usage.completion_tokens is None:
        tokens = "tokens ?"
    else:
        prompt = "—" if usage.prompt_tokens is None else usage.prompt_tokens
        tokens = f"tokens {prompt}/{usage.completion_tokens}"
        if usage.reasoning_tokens is not None:
            tokens += f" (reasoning {usage.reasoning_tokens})"
    tps = res.tokens_per_second
    parts = [
        f"model {model_id}",
        _secs(res.latency_ms),
        f"TTFT {_secs(res.ttft_ms)}",
        tokens,
        "tok/s —" if tps is None else f"tok/s {tps:.1f}",
    ]
    console.err.print(f"· {'  ·  '.join(parts)}", style="dim", markup=False, highlight=False)
    note = res.cutoff_note
    if note:
        console.warn(note)


@app.command("ask")
def ask_command(
    question: str | None = typer.Argument(None, help="Вопрос; без аргумента читается из stdin."),
    model: str = _MODEL_OPT,
    url: str | None = _URL_OPT,
    show_thinking: bool = typer.Option(
        True, "--show-thinking/--no-show-thinking", help="Показывать reasoning (dim, stderr)."
    ),
    max_tokens: int = _MAX_TOKENS_OPT,
    deadline: float = _DEADLINE_OPT,
) -> None:
    """Один запрос к локальной модели: reasoning на stderr, ответ на stdout."""
    text, from_stdin = _read_question(question)
    if from_stdin:
        console.echo_input(text)
    base = (url or lc.default_url()).rstrip("/")
    found = lc.ensure_ready(model, base)
    messages = [{"role": "user", "content": text}]

    state = {"in_reasoning": False}

    def on_reasoning(chunk: str) -> None:
        if show_thinking:
            state["in_reasoning"] = True
            console.err.print(chunk, style="dim", end="", markup=False, highlight=False)

    def on_content(chunk: str) -> None:
        if state["in_reasoning"]:
            console.err.print("")
            state["in_reasoning"] = False
        console.write_chunk(chunk)

    try:
        res = lc.chat(
            base,
            model,
            messages,
            max_tokens=max_tokens,
            temperature=lc.TEMPERATURE,
            top_p=lc.TOP_P,
            on_reasoning=on_reasoning,
            on_content=on_content,
            deadline=deadline,
        )
    except AdventError as error:
        _journal(
            CallResult(model_requested=model, model_actual=found.id),
            messages,
            backend="local",
            task="ask",
            run=1,
            error=error.message,
        )
        raise
    if state["in_reasoning"]:
        console.err.print("")
    if res.text:
        console.finish_answer()
    _ask_footer(res, found.id)
    _journal(
        _local_call_result(res, model, found.id),
        messages,
        backend="local",
        task="ask",
        run=1,
        ttft_ms=res.ttft_ms,
        content_ms=res.content_ms,
    )


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Probe:
    task: str
    where: str  # "local" | "cloud"
    run: int
    error: str | None = None
    ok: bool | None = None
    label: str = ""
    latency_ms: float | None = None
    completion: int | None = None
    reasoning: int | None = None
    tok_s: float | None = None
    cost: float | None = None  # None = unknown

    @property
    def answered(self) -> bool:
        return self.error is None


def _cloud_config(model: str) -> Config | None:
    """Cloud config isolated from ADVENT_BASE_URL and ADVENT_* params; None without a real key."""
    config_module.load_env()
    key = (os.environ.get("MISTRAL_API_KEY") or "").strip()
    if not key:
        return None
    return Config(
        api_key=key,
        model=model,
        system_prompt_path=None,
        params=GenerationParams.build(max_tokens=lc.MAX_TOKENS),
        stream=False,
        base_url=None,
    )


def _excerpt(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) <= EXCERPT_LINES:
        return "\n".join(lines)
    rest = len(lines) - EXCERPT_LINES
    return "\n".join(lines[:EXCERPT_LINES]) + f"\n[... ещё {rest} строк]"


def _answer_body(task_id: str, text: str) -> str:
    if not text.strip():
        return "(пустой ответ)"
    if task_id == "palindrome":
        code = codecheck.extract_code(text)
        if code is not None and len(code.strip().splitlines()) <= CODE_FULL_LINES:
            return code.strip()
    return _excerpt(text)


def _print_answer(
    where: str, task_id: str, text: str, verdict: tasks.TaskVerdict, note: str | None = None
) -> None:
    detail = f" ({verdict.detail})" if verdict.detail else ""
    console.out.print(
        f"[bold]{where}[/bold]: {rich_escape(verdict.label + detail)}", highlight=False
    )
    if note:
        # Separate from the verdict: a cut answer can still match the key.
        console.out.print(f"  ⚠ {note}", markup=False, highlight=False)
    console.out.print(_answer_body(task_id, text), markup=False, highlight=False)


SHOWN = {"local": "локально", "cloud": "облако"}


def _probe_error(task: str, where: str, run: int, error: AdventError) -> Probe:
    console.warn(f"{SHOWN[where]}: {task} — ошибка: {error.message}")
    return Probe(task, where, run, error=error.message)


def _run_local(
    task: tasks.Task,
    run: int,
    *,
    base: str,
    model: str,
    found: lc.LocalModel,
    max_tokens: int,
    deadline: float,
) -> Probe:
    messages = [{"role": "user", "content": task.prompt}]
    try:
        with Ticker(thinking=True) as ticker:
            res = lc.chat(
                base,
                model,
                messages,
                max_tokens=max_tokens,
                temperature=lc.TEMPERATURE,
                top_p=lc.TOP_P,
                on_reasoning=ticker.add_reasoning,
                deadline=deadline,
            )
    except AdventError as error:
        _journal(
            CallResult(model_requested=model, model_actual=found.id),
            messages,
            backend="local",
            task=task.id,
            run=run,
            error=error.message,
        )
        return _probe_error(task.id, "local", run, error)
    verdict = tasks.check_task(task.id, res.text)
    if res.cutoff_note and not res.text.strip():
        verdict = tasks.TaskVerdict(tasks.V_WRONG, "✗ неверно", False)
    _print_answer("локально", task.id, res.text, verdict, res.cutoff_note)
    _journal(
        _local_call_result(res, model, found.id),
        messages,
        backend="local",
        task=task.id,
        run=run,
        ttft_ms=res.ttft_ms,
        content_ms=res.content_ms,
        verdict=verdict.verdict,
    )
    usage = res.usage
    return Probe(
        task.id,
        "local",
        run,
        ok=verdict.ok,
        label=verdict.label,
        latency_ms=res.latency_ms,
        completion=usage.completion_tokens if usage else None,
        reasoning=usage.reasoning_tokens if usage else None,
        tok_s=res.tokens_per_second,
        cost=0.0,
    )


def _run_cloud(task: tasks.Task, run: int, *, config: Config, prices: PriceTable) -> Probe:
    messages = [{"role": "user", "content": task.prompt}]
    try:
        with Ticker(thinking=False):
            result = chat_core.complete(config, messages)
    except AdventError as error:
        _journal(
            CallResult(model_requested=config.model),
            messages,
            backend="cloud",
            task=task.id,
            run=run,
            error=error.message,
        )
        return _probe_error(task.id, "cloud", run, error)
    verdict = tasks.check_task(task.id, result.text)
    _print_answer("облако", task.id, result.text, verdict)
    _journal(result, messages, backend="cloud", task=task.id, run=run, verdict=verdict.verdict)
    usage = result.usage
    price = prices.price_of(config.model)
    cost = None
    if price is not None:
        cost = call_cost(usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens, price)
    tok_s = None
    if usage.completion_tokens is not None and result.latency_ms > 0:
        tok_s = usage.completion_tokens / (result.latency_ms / 1000)
    return Probe(
        task.id,
        "cloud",
        run,
        ok=verdict.ok,
        label=verdict.label,
        latency_ms=float(result.latency_ms),
        completion=usage.completion_tokens,
        reasoning=usage.reasoning_tokens,
        tok_s=tok_s,
        cost=cost,
    )


def _med(values: list[float | int | None]) -> float | None:
    known = [v for v in values if v is not None]
    return float(median(known)) if known else None


def _tokens_cell(probes: list[Probe]) -> str:
    completion = _med([p.completion for p in probes])
    if completion is None:
        return "—"
    reasoning = _med([p.reasoning for p in probes])
    if reasoning is None:
        return f"{completion:.0f}"
    return f"{completion:.0f} ({reasoning:.0f})"


def _cost_sum(probes: list[Probe]) -> float | None:
    if any(p.cost is None for p in probes):
        return None
    return sum(p.cost or 0.0 for p in probes)


def _frac(correct: int, attempts: int, errors: int) -> str:
    suffix = f" (ошибок {errors})" if errors else ""
    return f"{correct}/{attempts}{suffix}"


def _verdict_cell(cell: list[Probe], runs: int) -> str:
    """Denominator is every executed attempt: errors are shown, never dropped."""
    if runs == 1 and cell[0].answered:
        return cell[0].label
    errors = sum(1 for p in cell if not p.answered)
    return _frac(sum(1 for p in cell if p.ok), len(cell), errors)


def _summary_table(
    task_ids: list[str],
    probes: list[Probe],
    runs: int,
    prices: PriceTable | None,
    cloud_model: str,
) -> Table:
    table = Table(title="Сводка", title_justify="left", pad_edge=False)
    for name in ("задача", "где", "время", "ток. (reasoning)", "tok/s", "верно", "$"):
        # 11 keeps "ток. (reasoning)" from wrapping mid-word at 80 columns.
        table.add_column(name, overflow="fold", min_width=11 if name.startswith("ток") else None)
    for task_id in task_ids:
        for where, shown in (("local", "локально"), ("cloud", "облако")):
            cell = [p for p in probes if p.task == task_id and p.where == where]
            if not cell:
                continue
            good = [p for p in cell if p.answered]
            if not good:
                table.add_row(task_id, shown, "—", "—", "—", "ошибка", "—")
                continue
            latency = _med([p.latency_ms for p in good])
            tps = _med([p.tok_s for p in good])
            if where == "local":
                money = "0*"
            else:
                total = _cost_sum(good)
                money = "—" if total is None else f"{total:.6f}"
            table.add_row(
                task_id,
                shown,
                _secs(latency),
                _tokens_cell(good),
                "—" if tps is None else f"{tps:.1f}",
                _verdict_cell(cell, runs),
                money,
            )
    notes = [LOCAL_COST_NOTE]
    if any(p.where == "cloud" for p in probes):
        notes.append(CLOUD_TPS_NOTE)
        if prices is not None:
            notes.append(f"цены на {prices.checked_on.isoformat()} ({cloud_model})")
    table.caption = "\n".join(notes)
    table.caption_justify = "left"
    return table


def _tally(probes: list[Probe], where: str) -> tuple[int, int, int]:
    """(correct, attempts, errors) over every executed attempt, errors included."""
    mine = [p for p in probes if p.where == where]
    return sum(1 for p in mine if p.ok), len(mine), sum(1 for p in mine if not p.answered)


def _task_time(probes: list[Probe], where: str, task_id: str) -> float | None:
    return _med([p.latency_ms for p in probes if p.where == where and p.task == task_id])


def _task_time_total(probes: list[Probe], where: str, task_ids: list[str]) -> float | None:
    values = [_task_time(probes, where, t) for t in task_ids]
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _common_times(
    probes: list[Probe], task_ids: list[str]
) -> tuple[float, float, list[str]] | None:
    """Median-per-task sums over tasks that BOTH sides completed; None without any."""
    common = [
        t
        for t in task_ids
        if _task_time(probes, "local", t) is not None and _task_time(probes, "cloud", t) is not None
    ]
    if not common:
        return None
    local = sum(_task_time(probes, "local", t) or 0.0 for t in common)
    cloud = sum(_task_time(probes, "cloud", t) or 0.0 for t in common)
    return local, cloud, common


def conclusions(
    probes: list[Probe], task_ids: list[str], runs: int, *, with_cloud: bool
) -> list[str]:
    """Statements computed from the numbers; ties never name a leader."""
    lines: list[str] = []
    if runs == 1:
        lines.append("Один прогон на задачу — это не статистика (--runs N для замеров).")
    l_ok, l_n, l_err = _tally(probes, "local")
    l_frac = _frac(l_ok, l_n, l_err)
    if not with_cloud:
        l_time = _task_time_total(probes, "local", task_ids)
        lines.append(f"Локально: верно {l_frac}, время {_secs(l_time)}, стоит $0 (без облака).")
        return lines
    c_ok, c_n, c_err = _tally(probes, "cloud")
    c_frac = _frac(c_ok, c_n, c_err)
    c_cost = _cost_sum([p for p in probes if p.where == "cloud" and p.answered])
    if l_n and c_n:
        left, right = l_ok * c_n, c_ok * l_n
        if left == right:
            accuracy = f"по точности ничья: {l_frac} против {c_frac}"
        else:
            leader = "локальная" if left > right else "облако"
            accuracy = f"точнее {leader}: локальная {l_frac}, облако {c_frac}"
    else:
        accuracy = f"верно {l_frac} против {c_frac}"
    both = _common_times(probes, task_ids)
    if both is not None and both[0] > 0 and both[1] > 0:
        l_time, c_time, common = both
        names = f" (задачи, где ответили обе стороны: {', '.join(common)})"
        if l_time == c_time:
            speed = "по времени ничья" + names
        elif l_time > c_time:
            speed = f"локальная медленнее в {l_time / c_time:.1f}×{names}"
        else:
            speed = f"локальная быстрее в {c_time / l_time:.1f}×{names}"
    else:
        speed = "время сравнить нечем: нет задач, где ответили обе стороны"
    money = "—" if c_cost is None else f"${c_cost:.6f}"
    lines.append(f"{accuracy}; {speed}; стоит $0 против {money}.")
    return lines


@app.command("compare")
def compare_command(
    task_ids: str = typer.Option(DEFAULT_TASKS, "--tasks", help="Задачи через запятую."),
    cloud_model: str = typer.Option(DEFAULT_CLOUD_MODEL, "--cloud-model", help="Облачная модель."),
    no_cloud: bool = typer.Option(False, "--no-cloud", help="Только локальная модель."),
    runs: int = typer.Option(1, "--runs", min=1, help="Прогонов на задачу и сторону."),
    model: str = _MODEL_OPT,
    url: str | None = _URL_OPT,
    max_tokens: int = _MAX_TOKENS_OPT,
    deadline: float = _DEADLINE_OPT,
) -> None:
    """Задачи разной сложности: локально и в облаке, ответы, verdict и итоговая таблица."""
    loaded = tasks.load_tasks()
    ids = tasks.parse_task_ids(task_ids, loaded)
    base = (url or lc.default_url()).rstrip("/")
    found = lc.ensure_ready(model, base)

    config: Config | None = None
    prices: PriceTable | None = None
    if not no_cloud:
        config = _cloud_config(cloud_model)
        if config is None:
            console.warn("MISTRAL_API_KEY не задан — облачная часть пропущена (как --no-cloud).")
        else:
            prices = load_prices()
            warn_if_prices_stale(prices)
    if "palindrome" in ids:
        console.err.print(codecheck.COMPOSITION, style="dim", markup=False)

    probes: list[Probe] = []
    for task_id in ids:
        task = loaded[task_id]
        console.out.print(f"\n[bold]── {task_id} ({rich_escape(task.level)}) ──[/bold]")
        for run in range(1, runs + 1):
            if runs > 1:
                console.err.print(f"прогон {run}/{runs}", style="dim")
            probes.append(
                _run_local(
                    task,
                    run,
                    base=base,
                    model=model,
                    found=found,
                    max_tokens=max_tokens,
                    deadline=deadline,
                )
            )
            if config is not None and prices is not None:
                probes.append(_run_cloud(task, run, config=config, prices=prices))

    with_cloud = config is not None
    console.out.print("")
    for line in conclusions(probes, ids, runs, with_cloud=with_cloud):
        console.out.print(line, markup=False, highlight=False)
    # Last, so the closing screen of the take is the day's headline (w02d10).
    console.out.print(_summary_table(ids, probes, runs, prices, cloud_model))
    if not any(p.answered for p in probes):
        console.warn("ни один вызов не удался")
        raise typer.Exit(1)


@app.command("rag")
def rag_command(
    runs: int = typer.Option(
        ragbench.DEFAULT_RUNS, "--runs", min=1, max=ragbench.MAX_RUNS, help="Прогонов на вопрос."
    ),
    questions: str | None = typer.Option(
        None, "--questions", help="id контрольных вопросов через запятую (по умолчанию все)."
    ),
    backends: str = typer.Option("local,cloud", "--backends", help="Подмножество: local,cloud."),
    unanswerable: bool = typer.Option(
        True, "--unanswerable/--no-unanswerable", help="Добавить вопросы без ответа в репо."
    ),
    save: Path | None = typer.Option(None, "--save", help="JSON со всеми прогонами."),
    report: list[Path] | None = typer.Option(
        None,
        "--report",
        help="Только отчёт из файла --save (повторяемо: склеивает backend'ы), без сети.",
    ),
    url: str | None = _URL_OPT,
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Профиль настроек локальной модели (adventlocal profiles); не baseline — только "
        "с --backends local.",
    ),
    compare: bool = typer.Option(
        False,
        "--compare",
        help="Профили бок о бок из файлов --save (`--compare F1 F2 …`, первый — опорный), "
        "без сети.",
    ),
    tables: str | None = typer.Option(
        None,
        "--tables",
        help="Только с --compare: какие блоки печатать, через запятую: quality, citations, "
        "speed, resources, tps, build, verdicts (компактные вердикты ворот), gates. "
        "Без опции - всё.",
    ),
    files: list[Path] | None = typer.Argument(None, help="Файлы --save для --compare."),
) -> None:
    """RAG (rewrite + rerank + cite) в облаке и локально: качество, скорость, стабильность."""
    if files and not compare:
        raise ConfigError("Файлы в аргументах нужны только вместе с --compare.")
    if tables is not None and not compare:
        raise ConfigError("--tables нужен только вместе с --compare.")
    if compare:
        if save is not None or report or profile:
            raise ConfigError("--compare не сочетается с --save, --report и --profile.")
        if not files:
            raise ConfigError("--compare: укажи файлы --save (первый — опорный).")
        code = compare_module.run_compare_command(files, tables)
        if code:
            raise typer.Exit(code)
        return
    if report:
        if save is not None:
            raise ConfigError("--report не сочетается с --save: отчёт ничего не запускает.")
        code = ragbench.run_report_command(report)
        if code:
            raise typer.Exit(code)
        return
    code = ragbench.run_rag_command(
        runs=runs,
        questions=questions,
        backends=backends,
        unanswerable=unanswerable,
        save=save,
        url=url,
        profile=profile,
    )
    if code:
        raise typer.Exit(code)


@app.command("profiles")
def profiles_command(
    names: list[str] | None = typer.Argument(
        None, help="Какие профили показать (по умолчанию все); baseline есть всегда."
    ),
) -> None:
    """Профили локального RAG: строки — поля, колонки — профили, отличия от baseline выделены."""
    for table in profiles.profiles_tables(names):
        console.out.print(table)


# ---------------------------------------------------------------------------
# service: the private LLM service (day 30)
# ---------------------------------------------------------------------------

SERVICE_DAY = 30
STATUS_POLL_SECONDS = 2.0
STRESS_PROMPT = (
    "Расскажи подробно, по шагам и не короче двадцати предложений, "
    "как правильно заваривать чай в чайнике."
)
OVERFLOW_TEXT_CHARS = 14_000  # ~3500+ tokens, far below the 64 KB body limit
BODY_LIMIT_BYTES = 65_536

service_app = typer.Typer(
    help="Приватный LLM-сервис по HTTPS: статус, чат, нагрузка, переполнение контекста.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(service_app, name="service")

_SVC_URL_OPT = typer.Option(
    None, "--url", help="Адрес сервиса (по умолчанию ADVENT_SERVICE_URL из .env)."
)
_SVC_KEY_ENV_OPT = typer.Option(
    svc.SERVICE_KEY_ENV,
    "--key-env",
    help="ИМЯ переменной окружения с ключом (не сам ключ: он не должен попасть в argv).",
)
_SVC_MAX_TOKENS_OPT = typer.Option(
    svc.DEFAULT_MAX_TOKENS, "--max-tokens", min=1, max=svc.MAX_TOKENS_CEILING, help="Лимит ответа."
)


def _make_client(target: svc.ServiceTarget) -> svc.ServiceClient:
    """Single seam for the HTTP client (tests substitute a MockTransport one)."""
    return svc.ServiceClient(target)


def _ms(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f} мс" if value < 1000 else f"{value / 1000:.1f} с"


def _num(value: object, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"


def _field(data: object, *path: str) -> object:
    for part in path:
        data = data.get(part) if isinstance(data, dict) else None
    return data


def _counts(data: object) -> str:
    if not isinstance(data, dict) or not data:
        return "—"
    return " · ".join(f"{k} {v}" for k, v in data.items())


def _service_status_table(origin: str, st: svc.ServiceStatus) -> Table:
    table = Table(title=f"Сервис {origin}", title_justify="left", show_header=True)
    table.add_column("параметр", overflow="fold")
    table.add_column("значение", overflow="fold")
    raw = st.raw
    queue = (
        f"ждут {_num(st.queue_waiting)} · активных {_num(st.queue_active)}"
        f" · максимум ожидающих {_num(st.queue_max)}"
    )
    limits = (
        f"rpm {_num(st.rpm)} · burst {_num(st.burst)} · в bucket сейчас {_num(st.bucket_tokens)}"
    )
    tokens = (
        f"{_num(_field(raw, 'limits', 'max_tokens_default'))} по умолчанию · "
        f"потолок {_num(_field(raw, 'limits', 'max_tokens_ceiling'))}"
    )
    rows = [
        ("версия", _num(st.version)),
        ("модель", _num(st.state)),
        ("имя модели", _num(raw.get("model_id"))),
        ("RSS процесса", _num(None if st.rss_mb is None else round(st.rss_mb), " МБ")),
        ("аптайм процесса", _num(st.uptime_s, " с")),
        ("до выгрузки (idle)", _num(st.idle_unload_s, " с")),
        ("аптайм gateway", _num(raw.get("gateway_uptime_s"), " с")),
        ("код выхода модели", _num(raw.get("last_exit_code"))),
        ("очередь", queue),
        ("ключ", _num(_field(raw, "key", "name"))),
        ("лимиты ключа", limits),
        ("контекст", _num(_field(raw, "limits", "n_ctx"), " токенов")),
        ("max_tokens", tokens),
        ("тело запроса", _num(_field(raw, "limits", "body_max_bytes"), " байт")),
        ("таймаут простоя", _num(_field(raw, "limits", "idle_seconds"), " с")),
        ("всего в минуту", _num(_field(raw, "limits", "global_per_minute"))),
        ("ответы по кодам", _counts(raw.get("counters"))),
        ("исходы", _counts(raw.get("outcomes"))),
    ]
    for name, value in rows:
        table.add_row(name, value)
    return table


def _service_journal(
    messages: list[dict],
    *,
    task: str,
    result: svc.ServiceResult | None = None,
    error: AdventError | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    """One `endpoint=private` journal row per service call; the outcome is its own field."""
    res = result or svc.ServiceResult(status=getattr(error, "status", 0))
    call = CallResult(
        text=res.text,
        model_requested="advent-llm",
        model_actual=res.model or "advent-llm",
        usage=res.usage,
        latency_ms=int(res.wall_ms),
        stream=res.stream,
        truncated=res.incomplete,
        finish_reason=res.finish_reason,
    )
    outcome = "error" if error is not None else ("incomplete" if res.incomplete else "complete")
    log_service_call(
        call,
        messages,
        week=WEEK,
        day=SERVICE_DAY,
        outcome=outcome,
        error=error.message if error else res.incomplete_reason,
        extra={
            "backend": "service",
            "task": task,
            "http_status": getattr(error, "status", res.status),
            "error_type": getattr(error, "error_type", None),
            "queue_wait_ms": res.queue_wait_ms,
            "cold_start_ms": res.cold_start_ms,
            "input_tokens": res.input_tokens,
            **(extra or {}),
        },
    )


# --- status -----------------------------------------------------------------


async def _fetch_status(
    target: svc.ServiceTarget, wait_stopped: float | None
) -> tuple[svc.ServiceStatus, bool]:
    """(status, reached): with --wait-stopped poll until the model is `stopped` or time is up."""
    async with _make_client(target) as client:
        st = await client.status()
        if wait_stopped is None:
            return st, True
        deadline = time.monotonic() + wait_stopped
        while st.state != "stopped" and time.monotonic() < deadline:
            console.note(f"жду выгрузки модели: сейчас {st.state or '?'}")
            await asyncio.sleep(min(STATUS_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
            st = await client.status()
        return st, st.state == "stopped"


@service_app.command("status")
def service_status_command(
    url: str | None = _SVC_URL_OPT,
    key_env: str = _SVC_KEY_ENV_OPT,
    wait_stopped: float | None = typer.Option(
        None,
        "--wait-stopped",
        min=0.0,
        help="Ждать до N секунд, пока модель не станет stopped (exit 1, если не дождались).",
    ),
) -> None:
    """Состояние сервиса из /v1/status: модель, RSS, очередь, лимиты ключа, счётчики."""
    target = svc.resolve_target(url, key_env)
    st, reached = svc.run(_fetch_status(target, wait_stopped))
    console.out.print(_service_status_table(target.origin, st))
    if not reached:
        console.warn(f"модель не остановилась за {wait_stopped:g} с (состояние: {st.state or '?'})")
        raise typer.Exit(1)


# --- chat -------------------------------------------------------------------


def _chat_footer(res: svc.ServiceResult) -> None:
    usage = res.usage
    if usage.completion_tokens is None:
        tokens = "tokens ?"
    else:
        prompt = "—" if usage.prompt_tokens is None else usage.prompt_tokens
        tokens = f"tokens {prompt}/{usage.completion_tokens}"
    tps = res.tokens_per_second
    parts = [
        f"cold start {_ms(res.cold_start_ms)}",
        f"очередь {_ms(res.queue_wait_ms)}",
        _ms(res.wall_ms),
        tokens,
        "tok/s —" if tps is None else f"tok/s {tps:.1f}",
        f"finish {res.finish_reason or '—'}",
    ]
    console.err.print(f"· {'  ·  '.join(parts)}", style="dim", markup=False, highlight=False)


async def _chat_once(
    target: svc.ServiceTarget, messages: list[dict], max_tokens: int, on_content
) -> svc.ServiceResult:
    async with _make_client(target) as client:
        return await client.chat(messages, max_tokens=max_tokens, on_content=on_content)


def _chat_turn(
    target: svc.ServiceTarget, history: list[dict], question: str, max_tokens: int
) -> svc.ServiceResult | None:
    """One exchange. The answer goes to stdout, everything else to stderr; an
    incomplete or failed answer is not kept in `history`."""
    messages = [*history, {"role": "user", "content": question}]
    try:
        res = svc.run(_chat_once(target, messages, max_tokens, console.write_chunk))
    except AdventError as error:
        _service_journal(messages, task="chat", error=error)
        raise
    if res.text:
        console.finish_answer()
    _chat_footer(res)
    _service_journal(messages, task="chat", result=res)
    if res.incomplete:
        console.warn(f"обрыв: {res.incomplete_reason}; ответ неполный и не попадёт в историю")
        return None
    history.extend([messages[-1], {"role": "assistant", "content": res.text}])
    return res


def _read_repl_line() -> str | None:
    console.err.print("вы › ", end="", markup=False, highlight=False)
    line = sys.stdin.readline()
    if line == "":
        console.err.print("")
        return None
    text = line.strip()
    if not sys.stdin.isatty():
        console.echo_input(text)
    return text


@service_app.command("chat")
def service_chat_command(
    question: str | None = typer.Argument(
        None, help="Один вопрос и выход; без аргумента — REPL (/exit, /new, /status)."
    ),
    url: str | None = _SVC_URL_OPT,
    key_env: str = _SVC_KEY_ENV_OPT,
    max_tokens: int = _SVC_MAX_TOKENS_OPT,
) -> None:
    """Чат с приватным сервисом: stream в stdout, footer (cold start, очередь, tok/s) в stderr."""
    target = svc.resolve_target(url, key_env)
    history: list[dict] = []
    if question is not None and question.strip():
        _chat_turn(target, history, question.strip(), max_tokens)
        return
    console.note(f"сервис {target.origin} · /exit, /new, /status")
    while True:
        line = _read_repl_line()
        if line is None or line in ("/exit", "/quit"):
            return
        if not line:
            continue
        if line == "/new":
            history.clear()
            console.note("история очищена")
        elif line == "/status":
            try:
                st, _ = svc.run(_fetch_status(target, None))
                console.err.print(_service_status_table(target.origin, st))
            except AdventError as error:
                console.warn(error.message)
        elif line.startswith("/"):
            console.warn("команды: /exit, /new, /status")
        else:
            try:
                _chat_turn(target, history, line, max_tokens)
            except AdventError as error:
                # A refusal (429, 503 ...) must not end the session on camera.
                console.warn(error.message + (f" ({error.hint})" if error.hint else ""))


# --- stress -----------------------------------------------------------------


@dataclass(slots=True)
class StressRow:
    index: int
    http_status: int | None  # None = never got an HTTP answer
    kind: str  # "ok" | "incomplete" | "refused" | "network"
    queue_wait_ms: int | None = None
    cold_start_ms: int | None = None
    wall_ms: float = 0.0
    tokens: str = "—"
    detail: str = "—"  # finish_reason, error type or the cut reason


def _row_from_result(index: int, res: svc.ServiceResult) -> StressRow:
    usage = res.usage
    if usage.completion_tokens is None:
        tokens = "—"
    else:
        prompt = "—" if usage.prompt_tokens is None else usage.prompt_tokens
        tokens = f"{prompt}/{usage.completion_tokens}"
    incomplete = res.incomplete
    return StressRow(
        index=index,
        http_status=res.status,
        kind="incomplete" if incomplete else "ok",
        queue_wait_ms=res.queue_wait_ms,
        cold_start_ms=res.cold_start_ms,
        wall_ms=res.wall_ms,
        tokens=tokens,
        detail=f"обрыв: {res.incomplete_reason}" if incomplete else (res.finish_reason or "—"),
    )


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _stress_table(rows: list[StressRow]) -> Table:
    table = Table(title="Нагрузка: N одновременных запросов", title_justify="left")
    for name in ("#", "статус", "очередь", "cold start", "wall", "tokens", "итог"):
        table.add_column(name, overflow="fold")
    for row in sorted(rows, key=lambda r: r.index):
        table.add_row(
            str(row.index),
            "—" if row.http_status is None else str(row.http_status),
            _ms(row.queue_wait_ms),
            _ms(row.cold_start_ms),
            _ms(row.wall_ms),
            row.tokens,
            row.detail,
        )
    return table


def stress_summary(rows: list[StressRow], preconditions: str) -> list[str]:
    """Plain lines: outcome counts as `<код>: <N>`, then p50/p90 wall of the full answers."""
    counts: dict[str, int] = {}
    for row in rows:
        if row.kind == "network":
            key = "сеть"
        elif row.kind == "incomplete":
            key = "обрыв"
        else:
            key = str(row.http_status)
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (not item[0].isdigit(), item[0]))
    walls = [row.wall_ms for row in rows if row.kind == "ok"]
    p50, p90 = _percentile(walls, 0.5), _percentile(walls, 0.9)
    return [
        "Итого: " + ", ".join(f"{key}: {n}" for key, n in ordered),
        f"wall успешных: p50 {_ms(p50)}, p90 {_ms(p90)}",
        f"предусловия: {preconditions}",
    ]


async def _stress_preconditions(
    client: svc.ServiceClient, settle: float
) -> tuple[bool | None, str]:
    """Wait (up to `settle` s) for an empty queue and a full bucket.

    True = both proven, False = proven NOT met after the wait, None = status cannot say.
    """
    deadline = time.monotonic() + settle
    while True:
        st = await client.status()
        empty, full = st.queue_empty, st.bucket_full
        if empty and full:
            return True, "очередь пуста, bucket полон"
        if empty is None or full is None:
            return None, "статус не сообщает очередь или bucket — не проверены"
        if time.monotonic() >= deadline:
            what = []
            if not empty:
                what.append("очередь не пуста")
            if not full:
                what.append(f"bucket {_num(st.bucket_tokens)} из {_num(st.burst)}")
            return False, "не выполнены (" + ", ".join(what) + ")"
        console.note("жду: очередь пуста и bucket полон")
        await asyncio.sleep(STATUS_POLL_SECONDS)


async def _run_stress(
    target: svc.ServiceTarget, n: int, max_tokens: int, settle: float
) -> tuple[list[StressRow], str]:
    messages = [{"role": "user", "content": STRESS_PROMPT}]
    rows: list[StressRow] = []
    async with _make_client(target) as client:
        proven, precondition = await _stress_preconditions(client, settle)
        if proven is not True:
            console.warn(f"предусловия: {precondition}; точные счётчики не гарантируются")
        opened = await client.warm(n)
        console.note(f"открыто соединений: {opened}/{n}; отправка по барьеру")
        ready = 0
        go = asyncio.Event()

        async def one(index: int) -> None:
            nonlocal ready
            ready += 1
            await go.wait()
            extra = {"run": index}
            try:
                res = await client.chat(messages, max_tokens=max_tokens)
                row = _row_from_result(index, res)
                _service_journal(messages, task="stress", result=res, extra=extra)
            except svc.ServiceError as error:
                row = StressRow(index, error.status, "refused", detail=error.error_type or "—")
                _service_journal(messages, task="stress", error=error, extra=extra)
            except AdventError as error:
                row = StressRow(index, None, "network", detail="сеть")
                _service_journal(messages, task="stress", error=error, extra=extra)
            rows.append(row)
            console.note(f"  #{index}: {row.http_status or 'сеть'} {row.detail}")

        workers = [asyncio.create_task(one(i + 1)) for i in range(n)]
        while ready < n:
            await asyncio.sleep(0)
        go.set()
        await asyncio.gather(*workers)
    return rows, precondition


@service_app.command("stress")
def service_stress_command(
    n: int = typer.Option(6, "-n", min=1, max=64, help="Сколько одновременных запросов."),
    url: str | None = _SVC_URL_OPT,
    key_env: str = _SVC_KEY_ENV_OPT,
    max_tokens: int = typer.Option(128, "--max-tokens", min=1, max=svc.MAX_TOKENS_CEILING),
    settle: float = typer.Option(
        30.0, "--settle", min=0.0, help="Сколько секунд ждать пустую очередь и полный bucket."
    ),
) -> None:
    """N одновременных запросов: таблица по каждому, сколько 200/429/503, p50/p90 wall."""
    target = svc.resolve_target(url, key_env)
    rows, precondition = svc.run(_run_stress(target, n, max_tokens, settle))
    console.out.print(_stress_table(rows))
    for line in stress_summary(rows, precondition):
        console.out.print(line, markup=False, highlight=False)


# --- overflow ---------------------------------------------------------------

# a word list reads better as prose than as 48 quoted items
_OVERFLOW_WORDS = tuple(
    "lantern harbor quartz meadow copper violin thunder orchard saffron glacier "  # noqa: SIM905
    "compass willow ember prism cobalt thistle marble falcon ribbon cinder velvet "
    "granite jasmine anchor beacon cypress dolphin elm fjord garnet hollow ivory "
    "juniper kettle lotus mosaic nectar opal pebble quill raven summit tundra "
    "umber vortex walnut yarrow zephyr".split()
)


def overflow_text(chars: int = OVERFLOW_TEXT_CHARS) -> str:
    """Varied text of about `chars` characters. BPE squeezes monotone input to a few tokens
    (`"а" * 700_000`), so every phrase differs: a number and a word triple that walks the list."""
    words = _OVERFLOW_WORDS
    parts: list[str] = []
    size = 0
    i = 0
    while size < chars:
        phrase = (
            f"{i * 37 + 11}: {words[i % len(words)]} {words[(i * 7 + 3) % len(words)]}"
            f" {words[(i * 13 + 5) % len(words)]} {i * i % 9973}."
        )
        parts.append(phrase)
        size += len(phrase) + 1
        i += 1
    return " ".join(parts)


async def _overflow_once(
    target: svc.ServiceTarget, messages: list[dict], max_tokens: int
) -> svc.ServiceResult:
    async with _make_client(target) as client:
        return await client.chat(messages, max_tokens=max_tokens, stream=False)


@service_app.command("overflow")
def service_overflow_command(
    url: str | None = _SVC_URL_OPT,
    key_env: str = _SVC_KEY_ENV_OPT,
    max_tokens: int = typer.Option(16, "--max-tokens", min=1, max=svc.MAX_TOKENS_CEILING),
) -> None:
    """Запрос длиннее контекста: ожидается 413 context_length_exceeded с числами из ответа."""
    target = svc.resolve_target(url, key_env)
    messages = [{"role": "user", "content": overflow_text()}]
    payload = {"messages": messages, "max_tokens": max_tokens}
    body_bytes = len(json.dumps(payload).encode("utf-8"))
    if body_bytes >= BODY_LIMIT_BYTES:
        raise ConfigError(
            f"Тело {body_bytes} Б не меньше лимита {BODY_LIMIT_BYTES} Б — проверялся бы другой"
        )
    try:
        res = svc.run(_overflow_once(target, messages, max_tokens))
    except svc.ServiceError as error:
        _service_journal(messages, task="overflow", error=error)
        if error.status != 413 or error.error_type != "context_length_exceeded":
            raise AdventError(
                f"ожидался 413 context_length_exceeded, пришло {error.status} "
                f"{error.error_type or '—'}: {error.message}",
                hint="числа не печатаются: это не та ошибка",
            ) from None
        numbers = error.numbers
        missing = [k for k in ("input_tokens", "max_tokens", "n_ctx") if k not in numbers]
        if missing:
            raise AdventError(
                f"в ответе 413 context_length_exceeded нет полей: {', '.join(missing)}"
            ) from None
    else:
        _service_journal(messages, task="overflow", result=res)
        raise AdventError(
            f"переполнения не случилось: сервис ответил {res.status}",
            hint="входной текст должен быть больше n_ctx минус max_tokens",
        )
    total = numbers["input_tokens"] + numbers["max_tokens"]
    table = Table(title="Переполнение контекста", title_justify="left")
    table.add_column("параметр", overflow="fold")
    table.add_column("значение", overflow="fold")
    table.add_row("ответ", "413 context_length_exceeded")
    table.add_row("размер тела", f"{body_bytes} Б (лимит {BODY_LIMIT_BYTES} Б)")
    table.add_row("input_tokens", str(numbers["input_tokens"]))
    table.add_row("max_tokens", str(numbers["max_tokens"]))
    table.add_row("n_ctx", str(numbers["n_ctx"]))
    table.add_row("сумма", f"{total} > {numbers['n_ctx']}")
    console.out.print(table)


# --- keygen -----------------------------------------------------------------


@service_app.command("keygen")
def service_keygen_command() -> None:
    """Новый ключ сервиса и его sha256 (в keys.json идёт только хэш)."""
    key, digest = svc.generate_key()
    console.out.print(f"ключ: {key}", markup=False, highlight=False)
    console.out.print(f"sha256: {digest}", markup=False, highlight=False)
    console.note("ключ показан один раз: положи его в .env, а sha256 — в keys.json на сервере")


def main() -> None:
    """Entry point `adventlocal`: errors as text, not a traceback."""
    console.force_utf8()
    try:
        app()
    except ConfigError as error:
        console.fail(ConfigurationError(str(error)))
        raise SystemExit(2) from None
    except AdventError as error:
        console.fail(error)
        raise SystemExit(error.exit_code) from None
    except KeyboardInterrupt:
        console.note("\nпрервано")
        raise SystemExit(130) from None


# `python -m week_06.cli` is how the demo recorder launches this (Step.module).
if __name__ == "__main__":
    main()
