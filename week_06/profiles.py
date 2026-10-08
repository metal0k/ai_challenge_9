"""Day 29: tuning profiles of the local RAG. A profile is a complete, explicit configuration.

`baseline` is exactly the Day 28 setup; every screening profile differs from it in the fields
it names (one lever, or the build for the quant axis). `tuned` is deliberately absent: it is
added after screening, by the selection rule of SPEC-w06d29 section 5, not guessed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core.config import ConfigError

PROFILE_DAY = 29  # journal/save label of a run made with a profile; Day 28 runs keep 28

OFFICIAL = "ornith-ai"
BARTOWSKI = "bartowski"


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    title: str
    quant: str  # expected /api/v0/models quantization of the loaded model
    context: int  # expected loaded_context_length
    answer_temperature: float | None = None  # None = server default (unknown, same for all)
    answer_top_p: float | None = None
    answer_top_k: int | None = None
    answer_max_tokens: int = 4096
    answer_reasoning: bool = True  # False -> top-level reasoning_effort "none" on the answer
    k_before: int = 20
    rerank_format: str = "objects"  # "positional" -> {"scores": [int x n]}
    cite_prompt: str = "default"  # "local" -> RAG_CITE_INSTRUCTION_LOCAL
    gguf: str = "Ornith-1.5-9B-Q4_K_M.gguf"
    file_gb: float = 5.78  # by the HF file listing, not measured here
    publisher: str = OFFICIAL


BASELINE = Profile(name="baseline", title="до: настройки дня 28", quant="Q4_K_M", context=40960)

_SCREENING = (
    dataclasses.replace(
        BASELINE,
        name="sampling",
        title="сэмплинг по карточке ornith",
        answer_temperature=0.6,
        answer_top_p=0.95,
        answer_top_k=20,
    ),
    dataclasses.replace(BASELINE, name="cap", title="потолок ответа 1536", answer_max_tokens=1536),
    dataclasses.replace(
        BASELINE, name="noreason", title="ответ без reasoning", answer_reasoning=False
    ),
    dataclasses.replace(BASELINE, name="ctx24k", title="контекст 24576", context=24576),
    dataclasses.replace(
        BASELINE, name="positional", title="rerank массивом оценок", rerank_format="positional"
    ),
    dataclasses.replace(BASELINE, name="k12", title="12 кандидатов в rerank", k_before=12),
    dataclasses.replace(
        BASELINE, name="citelocal", title="cite-инструкция под ornith", cite_prompt="local"
    ),
    dataclasses.replace(
        BASELINE,
        name="q4b",
        title="Q4_K_M bartowski (опора для q3/q5)",
        gguf="Ornith-1.5-9B-Q4_K_M.gguf",
        file_gb=5.91,
        publisher=BARTOWSKI,
    ),
    dataclasses.replace(
        BASELINE,
        name="q3",
        title="квант Q3_K_M",
        quant="Q3_K_M",
        gguf="Ornith-1.5-9B-Q3_K_M.gguf",
        file_gb=4.92,
        publisher=BARTOWSKI,
    ),
    # The context may be lowered at measurement time (largest of 40960/32768/24576 that leaves
    # >= 700 MiB free); then the row is labelled "квант+ctx", see axis_label().
    dataclasses.replace(
        BASELINE,
        name="q5",
        title="квант Q5_K_M",
        quant="Q5_K_M",
        gguf="Ornith-1.5-9B-Q5_K_M.gguf",
        file_gb=6.85,
        publisher=BARTOWSKI,
    ),
)

PROFILES: dict[str, Profile] = {p.name: p for p in (BASELINE, *_SCREENING)}

# (field, row label) in display order; the tuple order is also the diff order.
FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("title", "название"),
    ("quant", "quant"),
    ("context", "context"),
    ("answer_temperature", "t ответа"),
    ("answer_top_p", "top_p"),
    ("answer_top_k", "top_k"),
    ("answer_max_tokens", "max_tokens ответа"),
    ("answer_reasoning", "reasoning ответа"),
    ("k_before", "k_before"),
    ("rerank_format", "формат rerank"),
    ("cite_prompt", "cite-инструкция"),
    ("gguf", "GGUF"),
    ("publisher", "издатель"),
    ("file_gb", "файл, ГБ (по HF)"),
)
DIFF_FIELDS = tuple(name for name, _ in FIELD_LABELS if name != "title")
QUANT_AXIS = ("q3", "q4b", "q5")
VRAM_GATED = ("ctx24k", "q3", "q5")  # also need >= 700 MiB free (SPEC section 5)


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(PROFILES)
        raise ConfigError(f"Неизвестный профиль {name!r}. Есть: {known}.") from None


def changed_fields(profile: Profile, base: Profile = BASELINE) -> tuple[str, ...]:
    """Names of the fields (title excluded) in which `profile` differs from `base`."""
    return tuple(f for f in DIFF_FIELDS if getattr(profile, f) != getattr(base, f))


def axis_label_for(name: str, context: int | None, ref_context: int | None) -> str:
    """«квант» or «квант+ctx» for a quant-axis row, judged by the contexts actually saved."""
    if name == "q4b":
        return "опора"
    if name not in ("q3", "q5"):
        return "—"
    moved = context is not None and ref_context is not None and context != ref_context
    return "квант+ctx" if moved else "квант"


def axis_label(profile: Profile) -> str:
    """Registry view of axis_label_for: the profile's declared context against q4b's."""
    if profile.name not in QUANT_AXIS:
        return "—"
    label = axis_label_for(profile.name, profile.context, PROFILES["q4b"].context)
    return "квант" if label == "опора" else label


def settings_dict(profile: Profile) -> dict[str, object]:
    """Every profile field (the --save JSON carries them next to the measured results)."""
    return dataclasses.asdict(profile)


def _shown(field: str, value: object) -> str:
    if value is None:
        return "сервер"
    if isinstance(value, bool):
        return "вкл" if value else "выкл"
    return str(value)


def profiles_tables(names: Sequence[str] | None = None, *, per_table: int = 3) -> list[Table]:
    """Rows = fields, columns = baseline + the chosen profiles; differences are marked.

    More than `per_table` profiles split into several tables so each fits 80 columns
    (every table repeats the baseline column as the yardstick).
    """
    chosen = [get_profile(n) for n in names] if names else list(PROFILES.values())
    others = [p for p in chosen if p.name != BASELINE.name]
    groups = [others[i : i + per_table] for i in range(0, len(others), per_table)] or [[]]
    tables: list[Table] = []
    for group in groups:
        table = Table(title="Профили локального RAG", title_justify="left")
        table.add_column("поле", no_wrap=True)
        columns = [BASELINE, *group]
        for p in columns:
            table.add_column(p.name, overflow="fold", min_width=8)
        for field, label in FIELD_LABELS:
            cells = []
            for p in columns:
                text = rich_escape(_shown(field, getattr(p, field)))
                differs = p is not BASELINE and field != "title"
                differs = differs and getattr(p, field) != getattr(BASELINE, field)
                cells.append(f"[bold yellow]» {text}[/bold yellow]" if differs else text)
            table.add_row(label, *cells)
        table.caption = "» жёлтым — отличие от baseline; «сервер» — дефолт LM Studio, не задаётся"
        table.caption_justify = "left"
        tables.append(table)
    return tables
