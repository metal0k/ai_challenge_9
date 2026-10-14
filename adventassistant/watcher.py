"""Local PR watcher (W07 D32): poll open same-repo PRs, review each new base/head, comment once.

Orchestration only: diff, RAG and rerank stay in `review`, REST in `github`.
The paid result is written as JSON before the state checkpoint, so a crash between
the two never buys the same review twice. Incoming PR code is fetched as git objects
and never checked out or executed. stdout carries the finished review, stderr the rest.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from advent_core import console
from advent_core.config import ConfigError, redact
from advent_core.errors import (
    AdventError,
    AuthError,
    ConfigurationError,
    NetworkError,
    RateLimitError,
    ServerError,
    StreamTruncated,
    translate,
)
from adventassistant import review, review_index
from adventassistant.github import GithubClient, PullRequest

MIN_INTERVAL = 5
GIT_TIMEOUT = 60.0
CACHE_LOCK_WAIT = 900.0
STATE_VERSION = 1
RESULT_VERSION = 1
EMPTY_TEXT = "изменений нет"
REVIEW_MARKER = "<!-- adventassistant-review -->"

_TRANSIENT = (RateLimitError, ServerError, NetworkError, StreamTruncated)
_SHA = re.compile(r"[0-9a-f]{40}")
_KEY = re.compile(r"(\d+):([0-9a-f]{40}):([0-9a-f]{40})")
_NAME = re.compile(r"[A-Za-z0-9._-]+")
_GIT_CONFIG_VAR = re.compile(r"GIT_CONFIG_(COUNT|KEY_\d+|VALUE_\d+)")
_STATUSES = ("reviewed", "posted", "empty")
_AUTH_MARKERS = (
    "authentication failed",
    "terminal prompts disabled",
    "could not read username",
    "returned error: 401",
    "returned error: 403",
    "permission denied",
    "invalid username or token",
    "repository not found",
)
_MISSING_MARKERS = (
    "not our ref",
    "couldn't find remote ref",
    "unadvertised object",
    "bad object",
    "did not match any",
)


class WatcherFatal(AdventError):
    """Stops the whole watcher: corrupt or unwritable state, a malformed API reply, wrong origin."""


@dataclass(frozen=True, slots=True)
class Key:
    number: int
    base: str
    head: str

    def __post_init__(self) -> None:
        ok = isinstance(self.number, int) and not isinstance(self.number, bool) and self.number > 0
        if not (ok and _SHA.fullmatch(self.base) and _SHA.fullmatch(self.head)):
            raise WatcherFatal(
                f"Некорректный ключ PR: {self.number!r} {self.base!r} {self.head!r}."
            )

    def __str__(self) -> str:
        return f"{self.number}:{self.base}:{self.head}"

    @property
    def stem(self) -> str:
        return f"pr-{self.number}-{self.base}-{self.head}"


@dataclass(frozen=True, slots=True)
class Result:
    empty: bool
    text: str


def parse_github_url(url: str) -> str:
    """`owner/repo` from a plain HTTPS github.com URL. Userinfo, ports and extras are refused."""
    parts = urlsplit(url.strip())
    ok = (
        parts.scheme == "https"
        and (parts.hostname or "").lower() == "github.com"
        and parts.username is None
        and parts.password is None
        and "@" not in parts.netloc
        and ":" not in parts.netloc
        and not parts.query
        and not parts.fragment
    )
    segments = parts.path.strip("/").split("/")
    if len(segments) == 2 and segments[1].endswith(".git"):
        segments[1] = segments[1][:-4]
    if not (ok and len(segments) == 2 and all(_NAME.fullmatch(s) for s in segments)):
        raise ConfigurationError(
            "Нужен HTTPS-адрес вида https://github.com/<owner>/<repo>.git без логина и пароля."
        )
    return f"{segments[0]}/{segments[1]}"


def revision_marker(base: str, head: str) -> str:
    return f"<!-- adventassistant-revision:{base}:{head} -->"


def comment_body(text: str, base: str, head: str) -> str:
    short = review_index.SHORT_SHA
    return (
        f"{REVIEW_MARKER}\n{revision_marker(base, head)}\n{text.rstrip()}\n"
        f"ревью: {base[:short]}..{head[:short]}\n"
    )


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _retry_after(exc: BaseException) -> float:
    value = getattr(exc, "retry_after", None)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, float(value))


# --- atomic files, strict JSON ------------------------------------------------------------


def _read_text(path: Path) -> str | None:
    """None only for a missing file. Other OSError is retried, then fatal, never 'empty'."""
    last: OSError | None = None
    for attempt in range(3):
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            last = exc
            time.sleep(0.05 * (attempt + 1))
    raise WatcherFatal(f"Не удалось прочитать {path.name}: {last}. Файл не тронут.")


def _exact_int(value: object, expected: int) -> bool:
    """True only for a real int equal to `expected`: `True == 1` must not pass a schema check."""
    return type(value) is int and value == expected


def _discard(path: Path, *, quiet: bool) -> None:
    """Remove a leftover file. `quiet`: an earlier error is in flight and must not be masked."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        if not quiet:
            raise WatcherFatal(f"Не удалось удалить {path.name}: {exc}.") from exc


def _atomic_write(path: Path, text: str) -> None:
    tmp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            tmp = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                tmp = None
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except OSError as exc:
        raise WatcherFatal(f"Не удалось записать {path.name}: {exc}. Прежний файл цел.") from exc
    finally:
        if tmp is not None:
            _discard(tmp, quiet=True)


def _load_json(path: Path, what: str) -> dict | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WatcherFatal(
            f"{what} {path.name} повреждён (не JSON). Файл не стёрт: исправь или перенеси его."
        ) from exc
    if not isinstance(data, dict):
        raise WatcherFatal(f"{what} {path.name} повреждён (не объект). Файл не стёрт.")
    return data


def _corrupt(path: Path, what: str, why: str) -> WatcherFatal:
    return WatcherFatal(f"{what} {path.name} повреждён: {why}. Файл не стёрт.")


class State:
    """version, repo and one `status` per PR+base+head key. No token, no diff."""

    def __init__(self, path: Path, repo: str, entries: dict[str, dict]) -> None:
        self.path = path
        self.repo = repo
        self.entries = entries

    @classmethod
    def load(cls, path: Path, repo: str) -> State:
        data = _load_json(path, "state")
        if data is None:
            return cls(path, repo, {})
        if set(data) != {"version", "repo", "entries"} or not _exact_int(
            data["version"], STATE_VERSION
        ):
            raise _corrupt(path, "state", "неизвестные ключи или версия")
        if not isinstance(data["repo"], str) or data["repo"].lower() != repo.lower():
            raise _corrupt(path, "state", f"записан для другого репозитория {data['repo']!r}")
        entries = data["entries"]
        if not isinstance(entries, dict):
            raise _corrupt(path, "state", "entries не объект")
        for raw_key, entry in entries.items():
            match = _KEY.fullmatch(raw_key) if isinstance(raw_key, str) else None
            if match is None or int(match.group(1)) <= 0:
                raise _corrupt(path, "state", f"ключ {raw_key!r}")
            if (
                not isinstance(entry, dict)
                or set(entry) != {"status", "updated"}
                or entry["status"] not in _STATUSES
                or not isinstance(entry["updated"], str)
            ):
                raise _corrupt(path, "state", f"запись {raw_key!r}")
        return cls(path, repo, entries)

    def status(self, key: Key) -> str | None:
        entry = self.entries.get(str(key))
        return None if entry is None else entry["status"]

    def set(self, key: Key, status: str) -> None:
        entries = dict(self.entries)
        entries[str(key)] = {"status": status, "updated": _now()}
        doc = {"version": STATE_VERSION, "repo": self.repo, "entries": entries}
        _atomic_write(self.path, json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
        self.entries = entries


class Store:
    """Result JSON (recovery source) and markdown copy under `reports/`, named by validated key."""

    def __init__(self, directory: Path, repo: str) -> None:
        self.directory = directory
        self.repo = repo

    def _json(self, key: Key) -> Path:
        return self.directory / f"{key.stem}.json"

    def save_result(self, key: Key, result: Result) -> None:
        doc = {
            "version": RESULT_VERSION,
            "repo": self.repo,
            "number": key.number,
            "base": key.base,
            "head": key.head,
            "empty": result.empty,
            "text": result.text,
        }
        _atomic_write(self._json(key), json.dumps(doc, ensure_ascii=False, indent=2) + "\n")

    def save_report(self, key: Key, result: Result) -> None:
        body = EMPTY_TEXT if result.empty else result.text.rstrip()
        header = f"# PR #{key.number}: {key.base}..{key.head}\n\n"
        _atomic_write(self.directory / f"{key.stem}.md", header + body + "\n")

    def load_result(self, key: Key) -> Result | None:
        path = self._json(key)
        data = _load_json(path, "результат")
        if data is None:
            return None
        expected = {"version", "repo", "number", "base", "head", "empty", "text"}
        same = (
            set(data) == expected
            and _exact_int(data["version"], RESULT_VERSION)
            and isinstance(data["repo"], str)
            and data["repo"].lower() == self.repo.lower()
            and _exact_int(data["number"], key.number)
            and data["base"] == key.base
            and data["head"] == key.head
            and isinstance(data["empty"], bool)
            and isinstance(data["text"], str)
            and (data["empty"] or data["text"].strip())
        )
        if not same:
            raise _corrupt(path, "результат", "не соответствует ключу или схеме")
        return Result(empty=data["empty"], text=data["text"])


# --- OS locks -----------------------------------------------------------------------------


class FileLock:
    """Held for the process lifetime; the OS drops it on crash, a stale file means nothing."""

    def __init__(self, path: Path, *, wait: float = 0.0, busy: str = "Файл занят.") -> None:
        self.path = path
        self.wait = wait
        self.busy = busy
        self._handle = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")  # noqa: SIM115 - held until __exit__
        deadline = time.monotonic() + self.wait
        while True:
            try:
                self._lock(handle)
                self._handle = handle
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise AdventError(self.busy) from None
                time.sleep(0.25)

    def __exit__(self, *exc: object) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            self._unlock(handle)
        except OSError:
            pass
        finally:
            handle.close()

    @staticmethod
    def _lock(handle) -> None:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle) -> None:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# --- git ----------------------------------------------------------------------------------


def basic_header(token: str) -> str:
    raw = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
    return f"AUTHORIZATION: basic {raw}"


def git_env(token: str | None) -> dict[str, str]:
    """Env for one git call. The auth header lives only here: not in argv, not in .git/config."""
    env = {k: v for k, v in os.environ.items() if not _GIT_CONFIG_VAR.fullmatch(k.upper())}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    # An unsupported socks ALL_PROXY kills httpx-style clients and git alike; HTTPS_PROXY stays.
    env["ALL_PROXY"] = ""
    env["all_proxy"] = ""
    settings = [("credential.helper", "")]
    if token:
        settings.append(("http.https://github.com/.extraheader", basic_header(token)))
    env["GIT_CONFIG_COUNT"] = str(len(settings))
    for index, (name, value) in enumerate(settings):
        env[f"GIT_CONFIG_KEY_{index}"] = name
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    return env


def run_git(
    args: Sequence[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-c", "credential.helper=", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise NetworkError(f"git {args[0]} не ответил за {timeout:.0f} с.") from None
    except OSError as exc:
        raise AdventError(f"Не удалось запустить git: {exc}") from exc


# --- the watcher --------------------------------------------------------------------------


class Watcher:
    def __init__(
        self,
        *,
        root: Path,
        repo: str,
        token: str | None,
        client,
        state: State,
        store: Store,
        cache_dir: Path,
        complete,
        pr: int | None,
        post: bool,
        build_index,
        review_runner,
        git_runner,
    ) -> None:
        self.root = root
        self.repo = repo
        self.token = token
        self.client = client
        self.state = state
        self.store = store
        self.cache_dir = cache_dir
        self.complete = complete
        self.only = pr
        self.post = post
        self.build_index = build_index
        self.review_runner = review_runner
        self.git_runner = git_runner

    # ---- helpers

    def _gh(self, call):
        """GitHub call: transient and auth errors keep their class, anything else is fatal."""
        try:
            return call()
        except (*_TRANSIENT, AuthError, WatcherFatal):
            raise
        except AdventError as exc:
            raise WatcherFatal(exc.message, hint=exc.hint) from exc
        except Exception as exc:
            raise WatcherFatal(translate(exc).message) from exc

    def _secrets(self) -> list[str]:
        if not self.token:
            return []
        return [basic_header(self.token), basic_header(self.token).split()[-1], self.token]

    def _clean(self, text: str) -> str:
        for secret in self._secrets():
            text = text.replace(secret, "***")
        return redact(text).strip()[:500]

    def _git(self, args: list[str], *, network: bool = False) -> subprocess.CompletedProcess:
        if network and not self.token:
            raise ConfigurationError("Для git fetch нужен GITHUB_TOKEN в .env.")
        env = git_env(self.token if network else None)
        return self.git_runner(args, cwd=self.root, env=env, timeout=GIT_TIMEOUT)

    def _check_origin(self) -> None:
        result = self._git(["remote", "get-url", "origin"])
        if result.returncode != 0:
            raise WatcherFatal("В репозитории нет remote origin.")
        try:
            found = parse_github_url(result.stdout)
        except ConfigurationError as exc:
            raise WatcherFatal(f"origin не подходит для watcher: {exc.message}") from exc
        if found.lower() != self.repo.lower():
            raise WatcherFatal(f"origin ({found}) не совпадает с GITHUB_URL ({self.repo}).")

    def _has(self, sha: str) -> bool:
        return self._git(["cat-file", "-e", f"{sha}^{{commit}}"]).returncode == 0

    def ensure_objects(self, base: str, head: str) -> None:
        if self._has(base) and self._has(head):
            return
        self._check_origin()
        console.note(f"git fetch: {base[:12]} {head[:12]}")
        args = ["fetch", "--quiet", "--no-tags", "--no-recurse-submodules", "origin", base, head]
        result = self._git(args, network=True)
        if result.returncode != 0:
            text = self._clean((result.stderr or "") + (result.stdout or ""))
            low = text.lower()
            if any(marker in low for marker in _AUTH_MARKERS):
                raise AuthError(
                    "GitHub отклонил доступ git, watcher остановлен.",
                    hint="Проверь GITHUB_TOKEN в .env: нужен доступ к репозиторию.",
                )
            if any(marker in low for marker in _MISSING_MARKERS):
                raise AdventError(f"origin не отдал нужный SHA: {text}")
            raise NetworkError(f"git fetch завершился с кодом {result.returncode}: {text}")
        if not (self._has(base) and self._has(head)):
            raise AdventError("После git fetch нужных коммитов нет в репозитории.")

    def _is_empty(self, base: str, head: str) -> bool:
        spec = f"{base}...{head}"
        result = self._git(["diff", "-z", "--name-status", "--find-renames", spec])
        if result.returncode != 0:
            detail = self._clean(result.stderr or "")
            raise AdventError(f"git diff {spec[:12]}… завершился с ошибкой: {detail}")
        return not (result.stdout or "").strip("\0\r\n ")

    def ensure_index(self, base: str) -> Path:
        name = f"review-{base}-v{review_index.REVIEW_INDEX_VERSION}"
        path = self.cache_dir / f"{name}.sqlite3"
        lock = FileLock(
            self.cache_dir / f"{name}.lock",
            wait=CACHE_LOCK_WAIT,
            busy="Индекс этой базы строит другой процесс.",
        )
        with lock:
            if not path.exists():
                part = self.cache_dir / f"{name}.part.sqlite3"
                _discard(part, quiet=False)
                console.note(f"индексирование базы {base[:12]}")
                built = False
                try:
                    self.build_index(self.root, rev=base, db_path=part)
                    os.replace(part, path)
                    built = True
                except (*_TRANSIENT, AuthError, ConfigurationError, WatcherFatal, ConfigError):
                    raise
                except AdventError as exc:
                    # write_index wraps disk errors as AdventError; a retry would repay embed.
                    raise WatcherFatal(exc.message, hint=exc.hint) from exc
                except OSError as exc:
                    raise WatcherFatal(f"Не удалось сохранить индекс {path.name}: {exc}") from exc
                except Exception as exc:
                    translated = translate(exc)
                    if isinstance(translated, (*_TRANSIENT, AuthError, ConfigurationError)):
                        raise translated from exc
                    raise WatcherFatal(translated.message, hint=translated.hint) from exc
                finally:
                    _discard(part, quiet=not built)
            else:
                console.note(f"индекс базы {base[:12]} найден в кэше")
            # A bad local index never heals by retrying: stop instead of looping every pass.
            try:
                return review_index.require_review_index(path, base)
            except AdventError as exc:
                raise WatcherFatal(exc.message, hint=exc.hint) from exc
            except Exception as exc:
                raise WatcherFatal(
                    f"Индекс {path.name} не читается: {redact(str(exc))[:200]}"
                ) from exc

    # ---- one PR

    def produce(self, pr: PullRequest, key: Key) -> Result:
        self.ensure_objects(pr.base_sha, pr.head_sha)
        if self._is_empty(pr.base_sha, pr.head_sha):
            result = Result(empty=True, text="")
        else:
            index = self.ensure_index(pr.base_sha)
            captured = io.StringIO()
            try:
                with contextlib.redirect_stdout(captured):
                    returned = self.review_runner(
                        self.root,
                        base=pr.base_sha,
                        head=pr.head_sha,
                        db_path=index,
                        complete=self.complete,
                        pr=None,
                    )
            except (AdventError, ConfigError):
                raise
            except Exception as exc:
                raise translate(exc) from exc
            text = (returned if isinstance(returned, str) else "").strip()
            text = text or captured.getvalue().strip()
            if not text:
                raise AdventError("Ревью вернуло пустой текст, ничего не сохранено.")
            result = Result(empty=text == EMPTY_TEXT, text="" if text == EMPTY_TEXT else text)
        # Result JSON first: it is what a restart recovers from if the checkpoint fails.
        self.store.save_result(key, result)
        self.store.save_report(key, result)
        return result

    def _emit(self, pr: PullRequest, text: str) -> None:
        short = review_index.SHORT_SHA
        sys.stdout.write(f"PR #{pr.number} ({pr.base_sha[:short]}..{pr.head_sha[:short]})\n")
        sys.stdout.write(text.rstrip() + "\n")
        sys.stdout.flush()

    def process(self, pr: PullRequest) -> None:
        key = Key(pr.number, pr.base_sha, pr.head_sha)
        label = f"PR #{pr.number} {pr.head_sha[:12]}"
        status = self.state.status(key)
        if status in ("posted", "empty"):
            console.note(f"{label}: уже обработан, пропуск")
            return
        if status == "reviewed":
            # Load even with --no-post: a lost or corrupt result must be loud now, not on --post.
            result = self.store.load_result(key)
            if result is None:
                raise WatcherFatal(
                    f"{label}: в state ревью есть, а результата нет. Модель не вызвана.",
                    hint=f"Верни {key.stem}.json в reports/ или удали запись из state.",
                )
            if not self.post:
                console.note(f"{label}: ревью готово, публикация выключена")
                return
        else:
            result = self.store.load_result(key)
            if result is None:
                result = self.produce(pr, key)
            else:
                console.note(f"{label}: найден готовый результат, ревью не повторяется")
                self.store.save_report(key, result)
            if not result.empty:
                self._emit(pr, result.text)
            self.state.set(key, "empty" if result.empty else "reviewed")
            if result.empty:
                console.note(f"{label}: изменений нет, комментарий не нужен")
                return
            if not self.post:
                console.note(f"{label}: ревью сохранено, комментарий не записан (--no-post)")
                return
        self.publish(pr, key, result)

    def publish(self, pr: PullRequest, key: Key, result: Result) -> None:
        label = f"PR #{pr.number}"
        current = self._gh(lambda: self.client.get_pull(pr.number))
        self.validate(current)
        if current.state != "open":
            console.note(f"{label}: закрыт, комментарий не записан")
            return
        if current.head_repo is None or current.head_repo.lower() != current.base_repo.lower():
            console.note(f"{label}: стал форком, комментарий не записан")
            return
        if (current.base_sha, current.head_sha) != (pr.base_sha, pr.head_sha):
            console.note(f"{label}: base/head изменились, ревью устарело, комментарий не записан")
            return
        marker = revision_marker(pr.base_sha, pr.head_sha)
        found = self._gh(lambda: self.client.find_comment(pr.number, marker))
        if found:
            console.note(f"{label}: комментарий уже есть: {found}")
            self.state.set(key, "posted")
            return
        body = comment_body(result.text, pr.base_sha, pr.head_sha)
        url = self._gh(lambda: self.client.post_comment(pr.number, body))
        self.state.set(key, "posted")
        console.note(f"{label}: комментарий записан: {url}")

    def validate(self, pr: PullRequest) -> None:
        number = pr.number
        good = (
            isinstance(number, int)
            and not isinstance(number, bool)
            and number > 0
            and pr.state in ("open", "closed")
            and isinstance(pr.base_sha, str)
            and isinstance(pr.head_sha, str)
            and _SHA.fullmatch(pr.base_sha)
            and _SHA.fullmatch(pr.head_sha)
            and isinstance(pr.base_repo, str)
        )
        if not good:
            raise WatcherFatal("GitHub вернул некорректное описание PR.")
        if pr.base_repo.lower() != self.repo.lower():
            raise WatcherFatal(f"PR #{number} относится к {pr.base_repo}, ожидался {self.repo}.")

    def accept(self, pr: PullRequest) -> bool:
        self.validate(pr)
        if self.only is not None and pr.number != self.only:
            return False
        if pr.state != "open":
            console.note(f"PR #{pr.number}: не открыт, пропуск")
            return False
        if pr.head_repo is None or pr.head_repo.lower() != pr.base_repo.lower():
            console.note(f"PR #{pr.number}: из форка, пропуск")
            return False
        return True

    # ---- one pass

    def run_pass(self) -> tuple[int, float]:
        """(failed PR count, longest retry_after seen). Auth and fatal errors propagate."""
        console.note("проход: запрашиваю открытые PR")
        failures = 0
        wait = 0.0
        try:
            pulls = self._gh(lambda: self.client.list_open_pulls(self.only))
        except _TRANSIENT as exc:
            console.warn(f"список PR не получен: {redact(exc.message)}")
            return 1, _retry_after(exc)
        for pr in sorted(pulls, key=lambda item: item.number):
            try:
                if self.accept(pr):
                    self.process(pr)
            except (AuthError, WatcherFatal, ConfigurationError, ConfigError):
                raise
            except _TRANSIENT as exc:
                failures += 1
                wait = max(wait, _retry_after(exc))
                console.warn(f"PR #{pr.number}: {redact(exc.message)}; повтор в следующем проходе")
            except AdventError as exc:
                failures += 1
                console.warn(f"PR #{pr.number}: {redact(exc.message)}")
                if exc.hint:
                    console.note(redact(exc.hint))
            except Exception as exc:
                failures += 1
                console.warn(f"PR #{pr.number}: {redact(translate(exc).message)}")
        return failures, wait


def run_watch(
    root: Path,
    *,
    complete,
    interval: int = 60,
    once: bool = False,
    pr: int | None = None,
    post: bool = True,
    state_path: Path | None = None,
    cache_dir: Path | None = None,
    client=None,
    build_index=None,
    review_runner=None,
    git_runner=None,
    sleep=None,
) -> int:
    """Poll until Ctrl+C, or one pass with `once`. Exit code 1 if a PR failed in `once`."""
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < MIN_INTERVAL:
        raise ConfigurationError(f"--interval: целое число секунд не меньше {MIN_INTERVAL}.")
    if pr is not None and (isinstance(pr, bool) or not isinstance(pr, int) or pr <= 0):
        raise ConfigurationError("--pr: номер PR должен быть положительным числом.")
    root = Path(root)
    env = dotenv_values(root / ".env")
    url = (env.get("GITHUB_URL") or "").strip()
    if not url:
        raise ConfigurationError("В .env нет GITHUB_URL.")
    repo = parse_github_url(url)
    token = (env.get("GITHUB_TOKEN") or "").strip() or None
    state_path = Path(state_path) if state_path is not None else root / "data/review/watcher.json"
    cache_dir = Path(cache_dir) if cache_dir is not None else root / "data/review/indexes"
    build_index = build_index if build_index is not None else review_index.build_review_index
    review_runner = review_runner if review_runner is not None else review.run_review
    git_runner = git_runner if git_runner is not None else run_git
    sleep = sleep if sleep is not None else time.sleep

    lock = FileLock(
        state_path.with_name(state_path.name + ".lock"),
        busy=f"С этим state уже работает другой watcher: {state_path.name}.",
    )
    with lock:
        state = State.load(state_path, repo)
        owned = client is None
        if owned:
            client = GithubClient.from_project(root)
        watcher = Watcher(
            root=root,
            repo=repo,
            token=token,
            client=client,
            state=state,
            store=Store(state_path.parent / "reports", repo),
            cache_dir=cache_dir,
            complete=complete,
            pr=pr,
            post=post,
            build_index=build_index,
            review_runner=review_runner,
            git_runner=git_runner,
        )
        try:
            while True:
                failures, wait = watcher.run_pass()
                if once:
                    return 1 if failures else 0
                delay = max(float(interval), wait)
                console.note(f"следующий проход через {delay:.0f} с")
                sleep(delay)
        finally:
            if owned:
                client.close()
