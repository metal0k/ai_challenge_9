"""LLM stages: describe, commit message, report draft. The model writes text; code validates it.

The model is reached only through `ctx.complete`. Prompt inputs are limited to the task text,
`git diff --stat` and file names (the report additionally gets public stage facts).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path

from advent_core.config import DEFAULT_MODEL, Config, ConfigError, configured_secrets, redact
from advent_core.journal import log_call
from adventship.ctx import EXIT_CONFIG, EXIT_TRUNCATED, ShipContext, ShipError, StageResult
from adventship.gitutil import changed_paths, git
from adventship.state import STAGES

PROMPTS = Path(__file__).parent / "prompts"
DESCRIBE_MAX = 900
COMMENT_LINE_MAX = 200
COMMIT_LINE_MAX = 72
REPORT_MAX = 3000
RUSSIAN_SHARE = 0.6
DRAFT_MARKER = "<!-- adventship:draft -->"
COMMIT_TYPES = "feat|fix|docs|test|refactor|chore|perf|style"
MODEL_ATTEMPTS = 2

# Built by concatenation: this file is committed, and tools/check_staged.py scans for the
# literal forms.
_NEEDLES = (
    "D" + ":\\",
    "C" + ":\\Users",
    "docs.google" + ".com/spreadsheets",
    "yadi.sk" + "/d/",
    "disk.yandex.ru" + "/d/",
    "github_pat" + "_",
    "gh" + "p_",
)
_DRIVE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:\\")
_HOME_PATH = re.compile(r"/(?:Users|home)/[^\s/]+/")
_API_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")
_FORBIDDEN_TRAILER = re.compile(
    r"^\s*(co-authored-by|claude-session)\s*:", re.IGNORECASE | re.MULTILINE
)
_SHEET_ID = re.compile(r"/d/([A-Za-z0-9_-]{20,})")
_SECRET_ENV_KEYS = (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "MISTRAL_API_KEY",
    "YANDEX_DISK_TOKEN",
    "COURSE_SHEET_URL",
)


class Rejected(ShipError):
    """Model output refused by a validator. `fatal`: a leak, so no retry is worth making."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


# --------------------------------------------------------------------- leak filter


def _secret_values(env: dict[str, str]) -> list[str]:
    values = list(configured_secrets())
    for key in _SECRET_ENV_KEYS:
        value = (env.get(key) or "").strip()
        if len(value) < 8:
            continue
        values.append(value)
        if key == "COURSE_SHEET_URL":
            match = _SHEET_ID.search(value)
            if match:
                values.append(match.group(1))
    return values


def find_leaks(text: str, env: dict[str, str] | None = None) -> list[str]:
    """Reasons `text` must not be published (local paths, tokens, the sheet link).

    Reasons never echo the offending value.
    """
    env = env or {}
    reasons: list[str] = []
    if any(needle in text for needle in _NEEDLES) or _DRIVE_PATH.search(text):
        reasons.append("локальный путь, токен или личная ссылка")
    if _HOME_PATH.search(text):
        reasons.append("домашняя папка пользователя")
    if _API_KEY.search(text):
        reasons.append("похоже на API-ключ")
    if any(value in text for value in _secret_values(env)):
        reasons.append("значение секрета из окружения")
    return reasons


def scrub(text: str, env: dict[str, str] | None = None) -> str:
    """Mask configured secrets before text goes into a prompt, a log or an error."""
    for value in _secret_values(env or {}):
        text = text.replace(value, "***")
    return redact(text)


def _reject_leaks(text: str, env: dict[str, str]) -> None:
    reasons = find_leaks(text, env)
    if reasons:
        raise Rejected("в тексте " + "; ".join(reasons), fatal=True)


def _russian_share(text: str) -> float:
    letters = [c.lower() for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "а" <= c <= "я" or c == "ё") / len(letters)


# --------------------------------------------------------------------- validators


def validate_description(text: str, env: dict[str, str] | None = None) -> str:
    env = env or {}
    text = text.strip().strip('"«»').strip()
    if not text:
        raise Rejected("пустое описание")
    if len(text) > DESCRIBE_MAX:
        raise Rejected(f"описание длиннее {DESCRIBE_MAX} символов ({len(text)})")
    if _russian_share(text) < RUSSIAN_SHARE:
        raise Rejected("описание должно быть на русском языке")
    _reject_leaks(text, env)
    return text


def comment_line(description: str) -> str:
    """First sentence of the description on one line, at most COMMENT_LINE_MAX characters."""
    flat = " ".join(description.split())
    first = re.split(r"(?<=[.!?])\s", flat, maxsplit=1)[0]
    if len(first) <= COMMENT_LINE_MAX:
        return first
    cut = first[: COMMENT_LINE_MAX - 1]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(",;:- ") + "…"


def validate_commit_message(message: str, tag: str, env: dict[str, str] | None = None) -> str:
    """`type(wNNdDD): text`, one line of at most 72 characters, no trailers. Returns the message."""
    env = env or {}
    message = message.strip()
    if _FORBIDDEN_TRAILER.search(message):
        raise Rejected("в сообщении есть Co-Authored-By или Claude-Session", fatal=True)
    if "\n" in message:
        raise Rejected("сообщение должно быть одной строкой")
    if len(message) > COMMIT_LINE_MAX:
        raise Rejected(f"строка длиннее {COMMIT_LINE_MAX} символов ({len(message)})")
    if not re.fullmatch(rf"(?:{COMMIT_TYPES})\({re.escape(tag)}\): \S.*", message):
        raise Rejected(f"формат должен быть type({tag}): описание")
    _reject_leaks(message, env)
    return message


def validate_report(text: str, env: dict[str, str] | None = None) -> str:
    env = env or {}
    text = text.strip()
    if not text:
        raise Rejected("пустой отчёт")
    if len(text) > REPORT_MAX:
        raise Rejected(f"отчёт длиннее {REPORT_MAX} символов ({len(text)})")
    if _russian_share(text) < RUSSIAN_SHARE:
        raise Rejected("отчёт должен быть на русском языке")
    _reject_leaks(text, env)
    return text


# --------------------------------------------------------------------- model call


def _config(max_tokens: int) -> Config:
    base = Config.resolve(model=DEFAULT_MODEL, stream=False, max_tokens=max_tokens, temperature=0.3)
    if not base.is_local:
        return base
    # A loopback ADVENT_BASE_URL (LM Studio) would get the stub key: the pipeline is cloud-only.
    key = os.environ.get("MISTRAL_API_KEY", "")
    if not key:
        raise ConfigError("Не найден MISTRAL_API_KEY.")
    return replace(base, api_key=key, base_url=None, offline=False)


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8").strip()


def _ask(ctx: ShipContext, kind: str, prompt: str, user: str, *, max_tokens: int, validate) -> str:
    """One validated answer: a format rejection gets a single retry, a leak never does."""
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
    for attempt in range(MODEL_ATTEMPTS):
        result = ctx.complete(_config(max_tokens), list(messages))
        log_call(
            result,
            list(messages),
            week=ctx.week,
            day=ctx.day,
            path=ctx.journal_path,
            extra={"kind": kind, "tag": ctx.tag},
        )
        if result.truncated or result.finish_reason in ("length", "model_length"):
            raise ShipError(
                f"{kind}: ответ модели обрезан",
                exit_code=EXIT_TRUNCATED,
                hint="Повтори запуск стадии.",
            )
        try:
            return validate(result.text or "")
        except Rejected as rejected:
            if rejected.fatal or attempt == MODEL_ATTEMPTS - 1:
                raise ShipError(f"{kind}: ответ модели отклонён: {rejected}") from rejected
            messages += [
                {"role": "assistant", "content": result.text},
                {
                    "role": "user",
                    "content": f"Ответ отклонён: {rejected}. Исправь и ответь заново.",
                },
            ]
    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------- prompt inputs


def _task_text(ctx: ShipContext) -> str:
    if ctx.task is None:
        raise ShipError("задание не прочитано (стадия intake)", exit_code=EXIT_CONFIG)
    return ctx.task.text


def gather_inputs(ctx: ShipContext, files: list[str] | None = None) -> str:
    """The whole LLM-visible input: task text, `git diff --stat`, changed file names."""
    stat = git(ctx, "diff", "--stat", "HEAD").stdout.strip()
    names = files if files is not None else changed_paths(ctx)
    block = (
        f"ЗАДАНИЕ:\n{_task_text(ctx).strip()}\n\n"
        f"ИЗМЕНЕНИЯ (git diff --stat):\n{stat or '(нет изменений в отслеживаемых файлах)'}\n\n"
        "ФАЙЛЫ:\n" + ("\n".join(names) if names else "(нет)")
    )
    block = scrub(block, ctx.env)
    reasons = find_leaks(block, ctx.env)
    if reasons:
        raise ShipError(
            "во входе для модели " + "; ".join(reasons),
            hint="Убери это из задания, имён файлов или diff --stat: наружу такое не уходит.",
        )
    return block


# --------------------------------------------------------------------- stages


def run_describe(ctx: ShipContext) -> StageResult:
    text = _ask(
        ctx,
        "ship_describe",
        load_prompt("describe"),
        gather_inputs(ctx),
        max_tokens=700,
        validate=lambda t: validate_description(t, ctx.env),
    )
    line = comment_line(text)
    return StageResult(
        data={"description": text, "comment_line": line},
        summary=f"{len(text)} симв.: {line}",
    )


def generate_commit_message(ctx: ShipContext, files: list[str] | None = None) -> str:
    """Only the first non-empty line of the model's answer is used."""

    def validate(text: str) -> str:
        lines = [ln.strip().strip("`\"'") for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise Rejected("пустое сообщение")
        return validate_commit_message(lines[0], ctx.tag, ctx.env)

    user = f"Метка дня: {ctx.tag}\n\n{gather_inputs(ctx, files)}"
    return _ask(ctx, "ship_commit", load_prompt("commit"), user, max_tokens=120, validate=validate)


def collect_facts(ctx: ShipContext) -> dict[str, dict]:
    """Public stage results for the report: plain values only, volatile `_keys` dropped."""
    facts: dict[str, dict] = {}
    for stage in STAGES:
        entry = ctx.state.entry(stage)
        if stage in ("intake", "report") or entry is None:
            continue
        data = {
            key: (value[:400] if isinstance(value, str) else value)
            for key, value in entry.get("data", {}).items()
            if not key.startswith("_")
            and isinstance(value, str | int | float | bool | type(None) | list)
        }
        facts[stage] = {"status": entry["status"], **data}
    return facts


def run_report(ctx: ShipContext) -> StageResult:
    facts = scrub(json.dumps(collect_facts(ctx), ensure_ascii=False, indent=2), ctx.env)
    if find_leaks(facts, ctx.env):
        raise ShipError(
            "в фактах стадий найдены локальные пути или секреты", hint="Проверь state.json."
        )
    path = ctx.root / "specs" / "reports" / f"{ctx.tag}.md"
    relative = path.relative_to(ctx.root).as_posix()
    if path.exists() and DRAFT_MARKER not in path.read_text(encoding="utf-8"):
        return StageResult(
            data={"report": relative, "kept": True},
            summary="отчёт уже написан вручную, не перезаписан",
        )
    task = scrub(_task_text(ctx).strip(), ctx.env)
    body = _ask(
        ctx,
        "ship_report",
        load_prompt("report"),
        f"Метка дня: {ctx.tag}\n\nЗАДАНИЕ:\n{task}\n\nФАКТЫ:\n{facts}",
        max_tokens=1500,
        validate=lambda t: validate_report(t, ctx.env),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {ctx.tag}: отчёт (черновик)\n\n{DRAFT_MARKER}\n"
        "> Черновик, сгенерирован adventship: проверь и поправь перед использованием.\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return StageResult(data={"report": relative, "chars": len(body)}, summary=relative)
