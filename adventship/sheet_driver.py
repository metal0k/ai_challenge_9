"""The sheet behind an interface: `SheetDriver` Protocol and the Playwright implementation.

The real layout of the course sheet is unknown until `sheet-probe` has run on it, so every
UI-dependent step is best effort, keeps its selectors in `SELECTORS` (the one place to fix
when Google changes the UI) and fails loudly instead of guessing. Playwright is imported
lazily: tests and `adventship status` never need it.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from advent_core.config import redact
from adventship.ctx import EXIT_AUTH, EXIT_CONFIG, ShipError

CELL_RE = re.compile(r"^[A-Z]{1,3}[0-9]{1,7}$")
DEFAULT_PROFILE = Path("data") / "ship" / "chrome-profile"
LOGIN_HINT = "adventship sheet-login"
PAGE_TIMEOUT_MS = 60_000
LOGIN_WAIT_MS = 15 * 60_000

# Every entry: candidate selectors tried in order, first one that matches wins.
SELECTORS: dict[str, list[str]] = {
    "name_box": [
        "#t-name-box input",
        "input[aria-label='Name box']",
        "input[aria-label='Поле имени']",
    ],
    "sheet_title": [
        "input.docs-title-input",
        "[aria-label='Rename']",
        "[aria-label='Переименовать']",
    ],
    "tab_names": [".docs-sheet-tab-name", ".docs-sheet-tab .docs-sheet-tab-caption"],
    "account": ["a[aria-label^='Google Account']", "a[aria-label^='Аккаунт Google']"],
    "comment_input": [
        "[role='dialog'] [contenteditable='true']",
        ".docos-input-contenteditable",
        "div[aria-label='Comment'][contenteditable='true']",
        "div[aria-label='Комментарий'][contenteditable='true']",
    ],
    "comment_submit": [
        "button[aria-label='Comment']",
        "button[aria-label='Комментировать']",
        "button:has-text('Comment')",
        "button:has-text('Комментировать')",
    ],
    "comment_text": [
        ".docos-replyview-body",
        ".docos-anchoredreplyview-body",
        ".docos-docoview-body",
    ],
}


@dataclass
class ProbeResult:
    """What `probe()` saw. `account` is masked: it ends up in state.json."""

    ok: bool = False
    account: str | None = None
    sheet_title: str | None = None
    tabs: list[str] = field(default_factory=list)
    slot: str | None = None
    cell: str | None = None
    comment_controls: bool = False
    comments_before: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProbeResult:
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def summary(self) -> str:
        count = "?" if self.comments_before is None else self.comments_before
        lines = [
            f"таблица: {self.sheet_title or '?'}; листы: {', '.join(self.tabs) or '?'}",
            f"аккаунт: {self.account or '?'}",
            f"слот {self.slot or '?'}: ячейка {self.cell or 'не найдена'}",
            f"комментариев в ячейке: {count}",
            f"контролы комментария: {'есть' if self.comment_controls else 'не найдены'}",
        ]
        if self.notes:
            lines.append(f"заметки: {self.notes}")
        return "\n".join(lines)


class SheetDriver(Protocol):
    def probe(self) -> ProbeResult: ...

    def read_comments(self, cell: str) -> list[str]: ...

    def add_comment(self, cell: str, text: str) -> None: ...


def mask_account(label: str | None) -> str | None:
    match = re.search(r"([\w.+-])[\w.+-]*@([\w.-]+)", label or "")
    return f"{match.group(1)}***@{match.group(2)}" if match else None


class PlaywrightDriver:
    """Persistent system-Chrome profile; one browser session per call."""

    def __init__(self, sheet_url: str | None, slot: str | None, profile_dir: Path) -> None:
        self.sheet_url = sheet_url
        self.slot = slot
        self.profile_dir = profile_dir

    @classmethod
    def from_env(cls, env: dict[str, str], root: Path) -> PlaywrightDriver:
        raw = (env.get("SHEET_PROFILE_DIR") or "").strip()
        profile = Path(raw) if raw else DEFAULT_PROFILE
        if not profile.is_absolute():
            profile = root / profile
        return cls(
            (env.get("COURSE_SHEET_URL") or "").strip() or None,
            (env.get("SHEET_SLOT") or "").strip() or None,
            profile.resolve(),
        )

    # --- plumbing ------------------------------------------------------

    def _require_url(self) -> str:
        if not self.sheet_url:
            raise ShipError(
                "Не задан COURSE_SHEET_URL в .env",
                exit_code=EXIT_CONFIG,
                hint="Положи приватную ссылку на таблицу в .env.",
            )
        return self.sheet_url

    @contextlib.contextmanager
    def _context(self, *, headless: bool = False) -> Iterator[Any]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ShipError(
                "Не установлен playwright",
                exit_code=EXIT_CONFIG,
                hint="uv sync --extra ship (браузер не нужен: берётся системный Chrome).",
            ) from None
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            try:
                context = playwright.chromium.launch_persistent_context(
                    str(self.profile_dir),
                    channel="chrome",
                    headless=headless,
                    args=["--disable-blink-features=AutomationControlled"],
                    ignore_default_args=["--enable-automation"],
                )
            except Exception as error:
                raise ShipError(
                    f"Chrome не запустился: {redact(str(error))[:300]}",
                    hint="Закрой окна Chrome с этим профилем; проверь, что Chrome установлен.",
                ) from None
            try:
                yield context
            finally:
                with contextlib.suppress(Exception):
                    context.close()

    def _open(self, context: Any) -> Any:
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)
        try:
            page.goto(self._require_url(), wait_until="domcontentloaded")
            page.wait_for_load_state("load")
        except ShipError:
            raise
        except Exception as error:
            raise ShipError(f"таблица не открылась: {redact(str(error))[:300]}") from None
        if "accounts.google.com" in page.url:
            raise ShipError(
                "Профиль не авторизован в Google",
                exit_code=EXIT_AUTH,
                hint=f"Запусти: {LOGIN_HINT}",
            )
        return page

    @staticmethod
    def _first(page: Any, key: str) -> Any | None:
        for selector in SELECTORS[key]:
            locator = page.locator(selector)
            if locator.count():
                return locator.first
        return None

    @classmethod
    def _all_texts(cls, page: Any, key: str) -> list[str]:
        for selector in SELECTORS[key]:
            locator = page.locator(selector)
            if locator.count():
                return [t.strip() for t in locator.all_inner_texts() if t.strip()]
        return []

    def _goto_cell(self, page: Any, cell: str) -> None:
        box = self._first(page, "name_box")
        if box is None:
            raise ShipError(
                "не нашёл поле адреса ячейки (Name box)", hint="Обнови SELECTORS['name_box']."
            )
        box.click()
        box.fill(cell)
        page.keyboard.press("Enter")

    def _open_comment(self, page: Any) -> None:
        page.keyboard.press("Control+Alt+m")
        page.wait_for_timeout(800)

    # --- SheetDriver ---------------------------------------------------

    def login(self) -> None:
        """Headed Chrome; the owner signs in by hand and closes the window."""
        with self._context(headless=False) as context:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(self.sheet_url or "https://accounts.google.com/")
            context.wait_for_event("close", timeout=LOGIN_WAIT_MS)

    def probe(self) -> ProbeResult:
        result = ProbeResult(slot=self.slot)
        with self._context() as context:
            page = self._open(context)
            page.wait_for_timeout(3000)
            title = self._first(page, "sheet_title")
            if title is not None:
                with contextlib.suppress(Exception):
                    result.sheet_title = title.input_value() or title.inner_text()
            result.tabs = self._all_texts(page, "tab_names")
            account = self._first(page, "account")
            if account is not None:
                result.account = mask_account(account.get_attribute("aria-label"))
            if not self.slot:
                result.notes = "SHEET_SLOT не задан в .env"
                return result
            result.cell = self._find_slot(page, result)
            if result.cell:
                self._open_comment(page)
                result.comment_controls = self._first(page, "comment_input") is not None
                result.comments_before = len(self._all_texts(page, "comment_text"))
                page.keyboard.press("Escape")
            result.ok = bool(result.cell and result.comment_controls)
            if result.cell and not result.comment_controls:
                result.notes = (result.notes + " контролы комментария не найдены").strip()
        return result

    def _find_slot(self, page: Any, result: ProbeResult) -> str | None:
        """Ctrl+F the slot label, then read the selected cell from the Name box."""
        page.keyboard.press("Control+f")
        page.wait_for_timeout(500)
        page.keyboard.type(self.slot or "")
        page.keyboard.press("Enter")
        page.wait_for_timeout(800)
        page.keyboard.press("Escape")
        box = self._first(page, "name_box")
        value = ""
        if box is not None:
            with contextlib.suppress(Exception):
                value = (box.input_value() or "").strip().upper()
        if not CELL_RE.match(value):
            result.notes = "адрес найденной ячейки не распознан: раскладка неизвестна"
            return None
        result.notes = "первое совпадение поиска; проверь адрес глазами"
        return value

    def read_comments(self, cell: str) -> list[str]:
        self._check_cell(cell)
        with self._context() as context:
            page = self._open(context)
            page.wait_for_timeout(2000)
            self._goto_cell(page, cell)
            self._open_comment(page)
            texts = self._all_texts(page, "comment_text")
            page.keyboard.press("Escape")
            return texts

    def add_comment(self, cell: str, text: str) -> None:
        self._check_cell(cell)
        with self._context() as context:
            page = self._open(context)
            page.wait_for_timeout(2000)
            self._goto_cell(page, cell)
            self._open_comment(page)
            field_ = self._first(page, "comment_input")
            if field_ is None:
                raise ShipError(
                    "не нашёл поле комментария", hint="Обнови SELECTORS['comment_input']."
                )
            field_.click()
            page.keyboard.insert_text(text)
            submit = self._first(page, "comment_submit")
            if submit is None:
                raise ShipError(
                    "не нашёл кнопку отправки", hint="Обнови SELECTORS['comment_submit']."
                )
            submit.click()
            page.wait_for_timeout(1500)

    def screenshot(self, path: Path) -> None:
        """Debug only (`SHIP_DEBUG_SHOTS`): may contain other people's sheet content."""
        with contextlib.suppress(Exception), self._context() as context:
            self._open(context).screenshot(path=str(path))

    @staticmethod
    def _check_cell(cell: str | None) -> None:
        if not cell or not CELL_RE.match(cell):
            raise ShipError(
                "run sheet-probe first", exit_code=EXIT_CONFIG, hint="adventship sheet-probe"
            )
