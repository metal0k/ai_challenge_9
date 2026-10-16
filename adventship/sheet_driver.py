"""The sheet behind an interface: `SheetDriver` Protocol and the Playwright implementation.

The cell address comes from the CSV export (`sheet_layout.resolve_cell`); comments are
invisible there, so reading and writing them goes through the UI. UI selectors live in
`SELECTORS` (the one place to fix when Google changes the UI) and every step fails loudly
instead of guessing. Playwright is imported lazily: tests and `adventship status` never need it.
"""

from __future__ import annotations

import contextlib
import csv
import io
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from advent_core.config import redact
from adventship.ctx import EXIT_AUTH, EXIT_CONFIG, ShipError
from adventship.sheet_layout import resolve_cell

# Host kept apart from the path so the source holds no contiguous sheet-link literal.
GOOGLE_DOCS = "https://docs.google" + ".com"
CELL_RE = re.compile(r"^[A-Z]{1,3}[0-9]{1,7}$")
SHEET_ID_RE = re.compile(r"/spreadsheets/d/([\w-]+)")
GID_RE = re.compile(r"[#?&]gid=(\d+)")
DEFAULT_PROFILE = Path("data") / "ship" / "chrome-profile"
LOGIN_HINT = "adventship sheet-login"
PAGE_TIMEOUT_MS = 60_000
UI_WAIT_MS = 20_000
LOGIN_WAIT_MS = 15 * 60_000

# Every entry: candidate selectors, any visible match wins.
SELECTORS: dict[str, list[str]] = {
    "name_box": [
        "input#t-name-box",
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
        ".docos-input-contenteditable[contenteditable='true']",
        "div[aria-label='Comment draft'][contenteditable='true']",
        "[role='dialog'] [contenteditable='true']",
    ],
    "comment_submit": [
        ".docos-input-post",
        "div[role='button'][aria-label='Post Comment']",
        "button[aria-label='Comment']",
        "button:has-text('Comment')",
        "button:has-text('Комментировать')",
    ],
    "comment_text": [".docos-replyview-body"],
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


def _norm(text: str) -> str:
    return " ".join(text.split())


class PlaywrightDriver:
    """Persistent system-Chrome profile; one browser session per call."""

    def __init__(
        self,
        sheet_url: str | None,
        slot: str | None,
        profile_dir: Path,
        day: int | None = None,
        owner: str | None = None,
    ) -> None:
        self.sheet_url = sheet_url
        self.slot = slot
        self.profile_dir = profile_dir
        self.day = day
        self.owner = owner

    @classmethod
    def from_env(cls, env: dict[str, str], root: Path, day: int | None = None) -> PlaywrightDriver:
        raw = (env.get("SHEET_PROFILE_DIR") or "").strip()
        profile = Path(raw) if raw else DEFAULT_PROFILE
        if not profile.is_absolute():
            profile = root / profile
        owner = (env.get("SHEET_OWNER") or "").strip()
        if not owner and env.get("GITHUB_URL"):
            from advent_cli.links import parse_repo

            with contextlib.suppress(Exception):
                owner = parse_repo(env["GITHUB_URL"])[0]
        return cls(
            (env.get("COURSE_SHEET_URL") or "").strip() or None,
            (env.get("SHEET_SLOT") or "").strip() or None,
            profile.resolve(),
            day,
            owner or None,
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
    def _context(self, *, headless: bool = True) -> Iterator[Any]:
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
                    viewport={"width": 1400, "height": 900},
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

    @staticmethod
    def _auth_error() -> ShipError:
        return ShipError(
            "Профиль не авторизован в Google",
            exit_code=EXIT_AUTH,
            hint=f"Запусти: {LOGIN_HINT}",
        )

    def _open(self, context: Any) -> Any:
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)
        try:
            page.goto(self._require_url(), wait_until="domcontentloaded")
        except ShipError:
            raise
        except Exception as error:
            raise ShipError(f"таблица не открылась: {redact(str(error))[:300]}") from None
        if "accounts.google.com" in page.url:
            raise self._auth_error()
        self._wait(page, "name_box")
        return page

    def _fetch_rows(self, context: Any) -> list[list[str]]:
        """CSV export through the logged-in context; comments are invisible there."""
        url = self._require_url()
        match = SHEET_ID_RE.search(url)
        if not match:
            raise ShipError("не разобрал id таблицы в COURSE_SHEET_URL", exit_code=EXIT_CONFIG)
        export = f"{GOOGLE_DOCS}/spreadsheets/d/{match.group(1)}/export?format=csv"
        gid = GID_RE.search(url)
        if gid:
            export += f"&gid={gid.group(1)}"
        try:
            response = context.request.get(export, timeout=PAGE_TIMEOUT_MS)
        except Exception as error:
            raise ShipError(f"экспорт таблицы не скачался: {redact(str(error))[:300]}") from None
        if "accounts.google.com" in response.url:
            raise self._auth_error()
        if response.status != 200:
            raise ShipError(f"экспорт таблицы вернул HTTP {response.status}")
        text = response.body().decode("utf-8-sig", errors="replace")
        return list(csv.reader(io.StringIO(text)))

    @staticmethod
    def _wait(page: Any, key: str, timeout: int = UI_WAIT_MS) -> Any | None:
        """First visible match of any candidate selector, or None after the timeout."""
        joined = ", ".join(SELECTORS[key])
        try:
            page.wait_for_selector(joined, state="visible", timeout=timeout)
        except Exception:
            return None
        return page.locator(joined).first

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
        box = self._wait(page, "name_box")
        if box is None:
            raise ShipError(
                "не нашёл поле адреса ячейки (Name box)", hint="Обнови SELECTORS['name_box']."
            )
        box.click()
        box.fill(cell)
        page.keyboard.press("Enter")
        page.wait_for_timeout(600)
        shown = (box.input_value() or "").strip().upper()
        if shown != cell:
            raise ShipError(f"переход к {cell} не удался (Name box: {shown or 'пусто'})")

    def _open_comment(self, page: Any) -> bool:
        """Ctrl+Alt+M opens the cell's thread; True once the comment input is visible."""
        for _ in range(2):
            page.keyboard.press("Control+Alt+m")
            if self._wait(page, "comment_input", 5000) is not None:
                return True
            page.keyboard.press("Escape")
        return False

    def _resolve(self, rows: list[list[str]]) -> str:
        if self.day is None or not self.owner:
            raise ShipError(
                "драйверу не заданы день или владелец строки",
                exit_code=EXIT_CONFIG,
                hint="Задай SHEET_OWNER или GITHUB_URL в .env.",
            )
        try:
            slot_row = int(self.slot or "")
        except ValueError:
            raise ShipError(
                "SHEET_SLOT должен быть номером строки", exit_code=EXIT_CONFIG
            ) from None
        return resolve_cell(rows, self.day, slot_row, self.owner)

    # --- SheetDriver ---------------------------------------------------

    def login(self) -> None:
        """Headed Chrome; the owner signs in by hand and closes the window."""
        with self._context(headless=False) as context:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(self.sheet_url or "https://accounts.google.com/")
            context.wait_for_event("close", timeout=LOGIN_WAIT_MS)

    def probe(self) -> ProbeResult:
        result = ProbeResult(slot=self.slot)
        if not self.slot:
            result.notes = "SHEET_SLOT не задан в .env"
            return result
        with self._context() as context:
            result.cell = self._resolve(self._fetch_rows(context))
            page = self._open(context)
            title = self._first(page, "sheet_title")
            if title is not None:
                with contextlib.suppress(Exception):
                    result.sheet_title = title.input_value() or title.inner_text()
            result.tabs = self._all_texts(page, "tab_names")
            account = self._first(page, "account")
            if account is not None:
                result.account = mask_account(account.get_attribute("aria-label"))
            self._goto_cell(page, result.cell)
            result.comment_controls = self._open_comment(page)
            if result.comment_controls:
                result.comments_before = len(self._all_texts(page, "comment_text"))
            page.keyboard.press("Escape")  # discards the empty draft; nothing is submitted
            result.ok = result.comment_controls
            if not result.comment_controls:
                result.notes = "контролы комментария не найдены"
        return result

    def read_comments(self, cell: str) -> list[str]:
        self._check_cell(cell)
        with self._context() as context:
            page = self._open(context)
            self._goto_cell(page, cell)
            if not self._open_comment(page):
                raise ShipError(
                    "не открылся комментарий ячейки", hint="Обнови SELECTORS['comment_input']."
                )
            texts = self._all_texts(page, "comment_text")
            page.keyboard.press("Escape")
            return texts

    def add_comment(self, cell: str, text: str) -> None:
        self._check_cell(cell)
        with self._context() as context:
            page = self._open(context)
            self._goto_cell(page, cell)
            if not self._open_comment(page):
                raise ShipError(
                    "не нашёл поле комментария", hint="Обнови SELECTORS['comment_input']."
                )
            field_ = self._wait(page, "comment_input", 5000)
            if field_ is None:
                raise ShipError("поле комментария пропало", hint="Обнови SELECTORS.")
            field_.click()
            page.keyboard.insert_text(text)
            submit = self._first(page, "comment_submit")
            if submit is None:
                raise ShipError(
                    "не нашёл кнопку отправки", hint="Обнови SELECTORS['comment_submit']."
                )
            submit.click()
            wanted = _norm(text)
            for _ in range(20):
                page.wait_for_timeout(500)
                if any(_norm(t) == wanted for t in self._all_texts(page, "comment_text")):
                    return
            raise ShipError(f"комментарий не появился в {cell} после отправки")

    def screenshot(self, path: Path) -> None:
        """Debug only (`SHIP_DEBUG_SHOTS`): may contain other people's sheet content."""
        with contextlib.suppress(Exception), self._context(headless=False) as context:
            self._open(context).screenshot(path=str(path))

    @staticmethod
    def _check_cell(cell: str | None) -> None:
        if not cell or not CELL_RE.match(cell):
            raise ShipError(
                "run sheet-probe first", exit_code=EXIT_CONFIG, hint="adventship sheet-probe"
            )
