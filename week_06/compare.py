"""Day 29 `adventlocal rag --compare`: profiles side by side from saved --save files, offline.

The first file is the reference; every other cell carries its delta against it. Nothing here
touches the network, the offline guard or LM Studio.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import console
from advent_core.errors import AdventError
from week_06 import profiles, vram
from week_06 import ragbench as rb

UNKNOWN = "неизвестно"
LEGACY_NAME = "день 28"
MIN_FREE_MIB = 700
SHARE_TOLERANCE = 0.10  # expected-source share may fall by 10 percentage points
REATTRIBUTION_FACTOR = 2
MINUS = "−"
SCREENING_NOTE = "скрининг: 10×1 не различает разницу в один факт"


@dataclass(frozen=True, slots=True)
class Loaded:
    path: Path
    label: str
    profile: str | None  # None = a legacy Day 28 file
    cells: tuple[rb.BenchRun, ...]
    runs: int
    started_at: str | None
    finished_at: str | None
    quant: str | None
    context: int | None
    publisher: str | None
    file_gb: float | None
    vram: vram.VramReport
    identity: Mapping[str, Any]  # the loaded model as the server reported it, nothing filled in
    question_ids: tuple[frozenset[int], frozenset[int]]
    sampling_default: bool  # answer sampling was left to the (unknown) server default


@dataclass(frozen=True, slots=True)
class Metrics:
    found: int
    total: int
    good: int
    n_questions: int
    same: int
    n_all: int
    refused_ok: int
    n_unanswerable: int
    wrong_refusals: int  # answerable questions answered «не знаю»
    sourced: int
    answered: int
    reattributed: int
    truncated: int
    bad_json: int
    errors: int
    median_ms: float | None
    p90_ms: float | None
    rerank_ms: float | None
    answer_ms: float | None
    completion: float | None
    tok_s: float | None
    rerank_tps: float | None  # rerank prompt tokens / rerank time: a lower bound

    @property
    def share(self) -> float | None:
        return self.sourced / self.answered if self.answered else None


# ---------------------------------------------------------------------------
# loading and validation
# ---------------------------------------------------------------------------


def _identity(data: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = data.get("model_identity")
    return raw if isinstance(raw, dict) else {}


def _fields(data: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = data.get("profile_fields")
    return raw if isinstance(raw, dict) else {}


def load_one(path: Path) -> tuple[Loaded, dict[str, Any]]:
    data = rb.load_result_file(path)
    cells = tuple(c for c in data["_cells"] if c.backend == "local")
    if not cells:
        raise AdventError(f"{path.name}: в файле нет локальных прогонов — сравнивать нечего.")
    identity, fields = _identity(data), _fields(data)
    quant = identity.get("quantization")
    context = identity.get("loaded_context_length")
    name = data.get("profile")
    settings = (data.get("settings") or {}).get("local") or {}
    sources = settings.get("sampling_source")
    if isinstance(sources, dict):
        sampling_default = rb.SAMPLING_SERVER_DEFAULT in sources.values()
    else:  # legacy file: unset sampling in the saved settings means the server default
        sampling_default = any(settings.get(k) is None for k in ("temperature", "top_p"))
    loaded = Loaded(
        path=path,
        label=str(name) if name else LEGACY_NAME,
        profile=str(name) if name else None,
        cells=cells,
        runs=int(data.get("runs") or 0),
        started_at=data.get("started_at") or None,
        finished_at=data.get("finished_at") or None,
        quant=str(quant) if quant else None,
        context=context if isinstance(context, int) and not isinstance(context, bool) else None,
        publisher=identity.get("publisher") or None,
        file_gb=fields.get("file_gb") if isinstance(fields.get("file_gb"), (int, float)) else None,
        vram=vram.report_from_dict(data.get("vram")),
        identity=dict(identity),
        question_ids=(
            frozenset(data.get("questions") or ()),
            frozenset(data.get("unanswerable") or ()),
        ),
        sampling_default=sampling_default,
    )
    return loaded, data


def _expected_pairs(data: Mapping[str, Any]) -> set[tuple[int, int]]:
    ids = [*(data.get("questions") or ()), *(data.get("unanswerable") or ())]
    return {(int(q), run) for q in ids for run in range(1, int(data.get("runs") or 0) + 1)}


def load_files(paths: Sequence[Path]) -> list[Loaded]:
    """Load and validate: one corpus_rev, one question set, equal runs, full coverage."""
    if len(paths) < 2:
        raise AdventError("--compare: нужно минимум два файла (первый — опорный).")
    pairs = [load_one(p) for p in paths]
    datas = [d for _, d in pairs]
    revs = {str(d.get("corpus_rev")) for d in datas}
    if len(revs) > 1:
        shown = ", ".join(
            f"{p.name}: {str(d.get('corpus_rev'))[:12]}" for p, d in zip(paths, datas, strict=True)
        )
        raise AdventError(
            f"Файлы собраны из разных снимков индекса (corpus_rev: {shown}): сравнение нечестное."
        )
    sets = {
        (frozenset(d.get("questions") or ()), frozenset(d.get("unanswerable") or ())) for d in datas
    }
    if len(sets) > 1:
        raise AdventError("Наборы вопросов в файлах различаются: сравнение нечестное.")
    runs = {int(d.get("runs") or 0) for d in datas}
    if len(runs) > 1:
        raise AdventError(
            f"Число прогонов в файлах разное ({sorted(runs)}): 10×1 и 13×3 не сравниваются."
        )
    for (loaded, data), path in zip(pairs, paths, strict=True):
        wanted = _expected_pairs(data)
        got = [(c.question_id, c.run) for c in loaded.cells]
        if len(set(got)) != len(got) or set(got) != wanted:
            raise AdventError(
                f"{path.name}: покрытие вопрос×прогон неполное или с дублями "
                f"({len(set(got))} из {len(wanted)}) — файл не сравнить."
            )
    loaded_list = [loaded for loaded, _ in pairs]
    seen: dict[str, int] = {}
    labelled: list[Loaded] = []
    for item in loaded_list:
        seen[item.label] = seen.get(item.label, 0) + 1
        if seen[item.label] > 1:
            item = _relabel(item, f"{item.label}#{seen[item.label]}")
        labelled.append(item)
    return labelled


def _relabel(item: Loaded, label: str) -> Loaded:
    return dataclasses.replace(item, label=label)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _answer_completion(cell: rb.BenchRun) -> int | None:
    entry = next((c for c in cell.ledger if c.stage == "answer"), None)
    return entry.completion_tokens if entry is not None else None


def _rerank_tps(cell: rb.BenchRun) -> float | None:
    calls = [c for c in cell.ledger if c.stage == "rerank"]
    tokens = [c.prompt_tokens for c in calls]
    if not calls or any(t is None for t in tokens) or not cell.rerank_ms:
        return None
    return sum(t for t in tokens if t is not None) / (cell.rerank_ms / 1000)


def compute(cells: Sequence[rb.BenchRun]) -> Metrics:
    ans = [c for c in cells if c.kind == "answerable"]
    un = [c for c in cells if c.kind == "unanswerable"]
    found, total = rb.facts_totals(cells)
    good, n_q = rb.questions_all_runs_complete(cells)
    same, _ok, n_all = rb.stability_counts(cells)
    answered = [c for c in ans if c.outcome == "answered" and c.ok]
    times = [float(c.wall_ms) for c in cells]
    return Metrics(
        found=found,
        total=total,
        good=good,
        n_questions=n_q,
        same=same,
        n_all=n_all,
        refused_ok=sum(1 for c in un if c.correct),
        n_unanswerable=len(un),
        wrong_refusals=sum(1 for c in ans if c.outcome in rb.REFUSALS),
        sourced=sum(1 for c in answered if c.sources_cited),
        answered=len(answered),
        reattributed=sum(c.quotes_reattributed for c in cells),
        truncated=sum(1 for c in cells if c.outcome == "truncated"),
        bad_json=sum(1 for c in cells if c.outcome == "bad_json"),
        errors=sum(1 for c in cells if not c.ok),
        median_ms=rb._med(times),
        p90_ms=rb.p90(times),
        rerank_ms=rb._med([c.rerank_ms for c in cells]),
        answer_ms=rb._med([c.answer_ms for c in cells]),
        completion=rb._med([_answer_completion(c) for c in cells]),
        tok_s=rb._med([c.tok_s for c in cells]),
        rerank_tps=rb._med([_rerank_tps(c) for c in cells]),
    )


def _correct_questions(cells: Sequence[rb.BenchRun]) -> set[int]:
    return {qid for qid, s in rb.stability(cells).items() if s["all_correct"]}


def paired_median(
    ref: Sequence[rb.BenchRun], other: Sequence[rb.BenchRun]
) -> tuple[float | None, float | None, int]:
    """(reference median, other median, n questions): wall time over every attempt of the
    questions that are correct in all runs in BOTH files. Medians are None when there are none."""
    common = _correct_questions(ref) & _correct_questions(other)
    if not common:
        return None, None, 0
    pick = [float(c.wall_ms) for c in ref if c.question_id in common]
    mine = [float(c.wall_ms) for c in other if c.question_id in common]
    return rb._med(pick), rb._med(mine), len(common)


# ---------------------------------------------------------------------------
# cells with deltas
# ---------------------------------------------------------------------------


def _signed(n: float, digits: int = 0) -> str:
    if round(n, digits) == 0:
        return "±0"
    body = f"{abs(n):.{digits}f}"
    return f"+{body}" if n > 0 else f"{MINUS}{body}"


def _pct(value: float, ref: float) -> str:
    if not ref:
        return ""
    d = round((value - ref) / ref * 100)
    return "0%" if d == 0 else (f"+{d}%" if d > 0 else f"{MINUS}{abs(d)}%")


def seconds_cell(ms: float | None, ref_ms: float | None, *, is_ref: bool) -> str:
    if ms is None:
        return "—"
    text = f"{ms / 1000:.1f} s"
    if is_ref or ref_ms is None:
        return text
    delta = _pct(ms, ref_ms)
    return f"{text} ({delta})" if delta else text


def count_cell(num: int, den: int, ref_num: int | None, *, is_ref: bool) -> str:
    text = f"{num}/{den}"
    if is_ref or ref_num is None:
        return text
    return f"{text} ({_signed(num - ref_num)})"


def plain_cell(
    value: float | int | None,
    ref: float | int | None,
    *,
    is_ref: bool,
    unit: str = "",
    digits: int = 0,
) -> str:
    if value is None:
        return "—"
    text = f"{value:.{digits}f}{unit}"
    if is_ref or ref is None:
        return text
    diff = value - ref
    return f"{text} ({_signed(diff, digits)})"


def share_cell(m: Metrics, ref: Metrics, *, is_ref: bool) -> str:
    if m.share is None:
        return "—"
    text = f"{round(m.share * 100)}%"
    if is_ref or ref.share is None:
        return text
    return f"{text} ({_signed(round((m.share - ref.share) * 100))} п.п.)"


def vram_cell(item: Loaded, ref: Loaded, *, is_ref: bool) -> str:
    peak = item.vram.vram_used_peak
    if peak is None:
        return "нет данных"
    text = f"{peak} MiB"
    if is_ref or ref.vram.vram_used_peak is None:
        return text
    return f"{text} ({_signed(peak - ref.vram.vram_used_peak)})"


# ---------------------------------------------------------------------------
# gates (SPEC-w06d29 section 5 and 9a-5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Check:
    ok: bool
    text: str
    known: bool = True  # False = no data: counts as not passed, shown with «?»


def no_data(what: str) -> Check:
    return Check(False, f"{what}: нет данных", known=False)


def reference_for(item: Loaded, files: Sequence[Loaded]) -> Loaded:
    """The first file, except on the quant axis: q3/q5 are measured against q4b (same build)."""
    if item.profile in ("q3", "q5"):
        q4b = next((f for f in files if f.profile == "q4b"), None)
        if q4b is not None:
            return q4b
    return files[0]


def reference_lines(files: Sequence[Loaded]) -> list[str]:
    quant_ref = [f.label for f in files if reference_for(f, files) is not files[0]]
    if not quant_ref:
        return []
    return [
        f"опорный для квантов: q4b (для {', '.join(quant_ref)}); для остальных — {files[0].label}"
    ]


def sampling_caveat(files: Sequence[Loaded]) -> str | None:
    """Different model loads + unknown server sampling default: the default may differ too."""
    identities = {json.dumps(f.identity, sort_keys=True) for f in files}
    if len(identities) > 1 and any(f.sampling_default for f in files):
        return (
            "оговорка: профили измерены в разных загрузках модели, а сэмплинг ответа — дефолт "
            "сервера LM Studio (значения неизвестны); различие дефолтов между загрузками "
            "исключить нельзя"
        )
    return None


def control_set() -> tuple[frozenset[int], frozenset[int]]:
    """Ids of the control set: 10 answerable + 3 unanswerable questions."""
    answerable = rb.rag_module.load_questions(rb.rag_cli.QUESTIONS_PATH)
    unanswerable = rb.rag_module.load_unanswerable(rb.rag_cli.UNANSWERABLE_PATH)
    return frozenset(q.id for q in answerable), frozenset(q.id for q in unanswerable)


FINAL_RUNS = 3


def is_day29_final(files: Sequence[Loaded]) -> bool:
    """3 runs over exactly the control set in every file; anything else is exploratory."""
    try:
        control = control_set()
    except (AdventError, OSError, ValueError):
        return False
    return all(f.runs == FINAL_RUNS and f.question_ids == control for f in files)


def screening_gate(ref: Loaded, item: Loaded, rm: Metrics, im: Metrics) -> list[Check]:
    """10x1 screening: facts >= reference - 1, no new failure kinds, VRAM for ctx/quant levers."""
    if im.total == 0 or rm.total == 0:
        checks = [no_data("факты")]
    else:
        checks = [
            Check(
                im.found >= rm.found - 1,
                f"факты {im.found} {'≥' if im.found >= rm.found - 1 else '<'} {rm.found - 1} "
                f"(опорный {rm.found} − 1)",
            )
        ]
    worse = [
        label
        for label, now, before in (
            ("обрывы", im.truncated, rm.truncated),
            ("формат", im.bad_json, rm.bad_json),
            ("ошибки", im.errors, rm.errors),
        )
        if now > before
    ]
    checks.append(
        Check(not worse, "новых сбоев нет" if not worse else f"новые сбои: {', '.join(worse)}")
    )
    if item.profile in profiles.VRAM_GATED:
        free = item.vram.free_at_peak
        if free is None:
            checks.append(Check(False, f"свободно VRAM ≥ {MIN_FREE_MIB} MiB: нет данных"))
        else:
            checks.append(
                Check(
                    free >= MIN_FREE_MIB,
                    f"свободно VRAM на пике {free} MiB (нужно ≥ {MIN_FREE_MIB})",
                )
            )
    return checks


def final_gate(ref: Loaded, item: Loaded, rm: Metrics, im: Metrics) -> list[Check]:
    """13x3 final: facts, all-runs-correct, refusals, sources, re-attributions, paired speed."""
    checks = []
    if im.total == 0 or rm.total == 0:
        checks.append(no_data("факты"))
    else:
        checks.append(
            Check(
                im.found >= rm.found - 1,
                f"факты {im.found} из {im.total} (опорный {rm.found}, допуск −1)",
            )
        )
    if im.n_questions == 0 or rm.n_questions == 0:
        checks.append(no_data("все прогоны верны"))
    else:
        checks.append(
            Check(
                im.good >= rm.good - 1,
                f"все прогоны верны {im.good}/{im.n_questions} (опорный {rm.good}, допуск −1)",
            )
        )
    if im.n_unanswerable == 0 or rm.n_unanswerable == 0:
        checks.append(no_data("верные отказы на неотвечаемых"))
    else:
        checks.append(
            Check(
                im.refused_ok >= rm.refused_ok,
                f"верные отказы на неотвечаемых {im.refused_ok}/{im.n_unanswerable} "
                f"(опорный {rm.refused_ok}); «не знаю» на отвечаемых {im.wrong_refusals} "
                f"(опорный {rm.wrong_refusals})",
            )
        )
    if im.share is None or rm.share is None:
        checks.append(no_data("источник в цитатах"))
    else:
        ok = im.share >= rm.share - SHARE_TOLERANCE - 1e-9
        checks.append(
            Check(
                ok,
                f"источник в цитатах {round(im.share * 100)}% "
                f"(опорный {round(rm.share * 100)}%, допуск −10 п.п.)",
            )
        )
    checks.append(
        Check(
            im.reattributed <= REATTRIBUTION_FACTOR * rm.reattributed,
            f"исправленных источников {im.reattributed} (опорный {rm.reattributed}, не больше ×2)",
        )
    )
    ref_med, med, n = paired_median(ref.cells, item.cells)
    if ref_med is None or med is None:
        checks.append(no_data("парная медиана (нет вопросов, верных у обоих)"))
    else:
        checks.append(
            Check(
                med < ref_med,
                f"парная медиана {med / 1000:.1f} s против {ref_med / 1000:.1f} s ({n} вопр.)",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _date(value: str | None) -> str:
    if not value:
        return UNKNOWN
    return value[:16].replace("T", " ")


def dates_line(files: Sequence[Loaded]) -> str:
    return "из сохранённых замеров: " + "; ".join(f"{f.label} {_date(f.started_at)}" for f in files)


def _table(title: str, columns: Sequence[str], min_width: int = 13) -> Table:
    table = Table(title=title, title_justify="left")
    table.add_column("профиль", no_wrap=True)
    for name in columns:
        table.add_column(name, justify="right", overflow="fold", min_width=min_width)
    return table


TPS_NOTE = "ток/с ответа = completion / время всего ответа (prefill включён): не скорость генерации"
RERANK_TPS_NOTE = (
    "rerank, ток/с = prompt_tokens / время rerank: нижняя оценка, во времени есть и генерация"
)


def build_tables(files: Sequence[Loaded]) -> list[Table]:
    metrics = {id(f): compute(f.cells) for f in files}
    quality = _table(
        "Качество",
        ["факты", "все прогоны верны", "вердикт стабилен", "отказы на неотвечаемых"],
    )
    cites = _table(
        "Цитаты и сбои",
        ["источник в цитатах", "источник исправлен", "обрыв/формат/ошибка"],
    )
    speed = _table(
        "Скорость, на вопрос",
        ["медиана", "парная медиана", "p90", "rerank, медиана"],
    )
    stages = _table(
        "Ответ и ресурсы",
        ["ответ, медиана", "completion, медиана", "VRAM пик"],
    )
    rates = _table(
        "Токены в секунду",
        ["ток/с ответа (вкл. prefill)", "rerank, ток/с (prompt+gen)"],
    )
    build = _table(
        "Сборка модели", ["ось", "quant", "ctx", "издатель", "файл, ГБ (по HF)"], min_width=8
    )
    paired_notes: list[str] = []
    for item in files:
        ref = reference_for(item, files)
        is_ref = item is ref
        m, rm = metrics[id(item)], metrics[id(ref)]
        label = rich_escape(item.label)
        quality.add_row(
            label,
            count_cell(m.found, m.total, rm.found, is_ref=is_ref),
            count_cell(m.good, m.n_questions, rm.good, is_ref=is_ref),
            count_cell(m.same, m.n_all, rm.same, is_ref=is_ref),
            count_cell(m.refused_ok, m.n_unanswerable, rm.refused_ok, is_ref=is_ref),
        )
        cites.add_row(
            label,
            share_cell(m, rm, is_ref=is_ref),
            plain_cell(m.reattributed, rm.reattributed, is_ref=is_ref),
            f"{m.truncated}/{m.bad_json}/{m.errors}",
        )
        ref_paired, paired, n = paired_median(ref.cells, item.cells)
        if not is_ref:
            paired_notes.append(f"{item.label} {n}")
        speed.add_row(
            label,
            seconds_cell(m.median_ms, rm.median_ms, is_ref=is_ref),
            "—" if is_ref else seconds_cell(paired, ref_paired, is_ref=False),
            seconds_cell(m.p90_ms, rm.p90_ms, is_ref=is_ref),
            seconds_cell(m.rerank_ms, rm.rerank_ms, is_ref=is_ref),
        )
        stages.add_row(
            label,
            seconds_cell(m.answer_ms, rm.answer_ms, is_ref=is_ref),
            plain_cell(m.completion, rm.completion, is_ref=is_ref),
            vram_cell(item, ref, is_ref=is_ref),
        )
        rates.add_row(
            label,
            plain_cell(m.tok_s, rm.tok_s, is_ref=is_ref, digits=1),
            plain_cell(m.rerank_tps, rm.rerank_tps, is_ref=is_ref, digits=0),
        )
        gb = "—" if item.file_gb is None else f"{item.file_gb}"
        build.add_row(
            label,
            profiles.axis_label_for(
                item.profile or "", item.context, reference_for(item, files).context
            ),
            item.quant or "—",
            str(item.context) if item.context else "—",
            rich_escape(item.publisher or UNKNOWN),
            gb,
        )
    speed.caption = (
        "медиана и p90 — по всем попыткам, сбои включены; парная — по вопросам, верным во всех "
        "прогонах у обоих профилей"
        + (f" (вопросов: {', '.join(paired_notes)})" if paired_notes else "")
    )
    speed.caption_justify = "left"
    stages.caption = "ответ с reasoning: completion включает рассуждение; без данных — «—»"
    stages.caption_justify = "left"
    rates.caption = f"{TPS_NOTE}\n{RERANK_TPS_NOTE}"
    rates.caption_justify = "left"
    build.caption = (
        "quant, ctx, издатель — как сообщил сервер («неизвестно» — сервер не сказал); "
        "размер файла записан в профиль по HF; «квант+ctx» — контекст отличается от опорного"
    )
    build.caption_justify = "left"
    return [quality, cites, speed, stages, rates, build]


def gate_lines(files: Sequence[Loaded]) -> list[str]:
    runs = files[0].runs
    screening = runs == 1
    final = not screening and is_day29_final(files)
    if screening:
        title = "Ворота скрининга:"
    elif final:
        title = "Ворота финала:"
    else:
        title = (
            "Пробное сравнение (не финал дня 29: нужны 3 прогона и контрольный набор "
            "10+3) — без вердикта:"
        )
    gate: Callable[[Loaded, Loaded, Metrics, Metrics], list[Check]] = (
        screening_gate if screening else final_gate
    )
    lines = [title]
    for item in files[1:]:
        ref = reference_for(item, files)
        checks = gate(ref, item, compute(ref.cells), compute(item.cells))
        passed = all(c.ok for c in checks)
        suffix = f" (опорный {ref.label})" if ref is not files[0] else ""
        if screening:
            verdict = "пройден" if passed else "не пройден"
        elif final:
            verdict = "успех" if passed else "не выполнено — это результат дня"
        else:
            verdict = "без вердикта"
        lines.append(f"{item.label}{suffix}: {verdict}")
        lines += [f"  {'✓' if c.ok else ('✗' if c.known else '?')} {c.text}" for c in checks]
    if screening:
        lines.append(f"{SCREENING_NOTE}; {rb.SMALL_SAMPLE}")
    return lines


def run_compare_command(paths: Sequence[Path]) -> int:
    files = load_files(paths)
    out = console.out
    out.print(dates_line(files), markup=False, highlight=False)
    if any(f.started_at is None for f in files):
        console.warn("в части файлов нет даты замера (старый формат): показано «неизвестно».")
    for line in reference_lines(files):
        out.print(line, markup=False, highlight=False)
    caveat = sampling_caveat(files)
    if caveat:
        out.print(caveat, markup=False, highlight=False)
    for f in files:
        if not f.vram.known:
            out.print(f"{f.label}: {vram.NO_DATA}", markup=False, highlight=False)
    for table in build_tables(files):
        out.print(table)
    for line in gate_lines(files):
        out.print(line, markup=False, highlight=False)
    return 0
