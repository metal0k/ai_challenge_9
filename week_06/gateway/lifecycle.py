"""llama-server child process manager (spec 2.2, 10-9..12, 10-14, 10-15).

States: stopped -> starting -> ready -> stopping -> stopped, plus failed. Every transition
is tied to a generation id so a late event of an old process cannot touch a newer one.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import httpx

from .errors import ApiError
from .settings import Settings

TAIL_LINES = 200
DRAIN_JOIN_TIMEOUT = 2.0
KILL_REAP_TIMEOUT = 2.0


def read_mem_available_mb(path: str) -> float | None:
    """None when unreadable (not Linux): the check is then skipped, not failed."""
    try:
        for line in Path(path).read_text(encoding="ascii", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


class ProcessManager:
    def __init__(
        self,
        settings: Settings,
        *,
        busy: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.s = settings
        self.busy = busy
        self.clock = clock
        self.state = "stopped"
        self.gen = 0
        self.proc: asyncio.subprocess.Process | None = None
        self.tail: deque[str] = deque(maxlen=TAIL_LINES)
        self.last_exit: int | None = None
        self.last_error: str | None = None
        self.ready_at: float | None = None
        self.last_activity = clock()
        self.spawns = 0
        self.spawned: deque[asyncio.subprocess.Process] = deque(maxlen=50)  # for leak checks
        self._failed_restarts: deque[float] = deque()
        self._start_task: asyncio.Task | None = None
        self._drains: list[asyncio.Task] = []
        self._watcher: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self._stopped: asyncio.Event | None = None
        self._client: httpx.AsyncClient | None = None

    # -- plumbing -----------------------------------------------------------------------

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{self.s.upstream_port}",
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(self.s.upstream_timeout, connect=3.0),
            )
        return self._client

    @property
    def stopped_event(self) -> asyncio.Event:
        if self._stopped is None:
            self._stopped = asyncio.Event()
            self._stopped.set()
        return self._stopped

    def note_finished(self) -> None:
        """Only a completed generation extends the idle timer (spec 10-15)."""
        self.last_activity = self.clock()

    def idle_remaining(self) -> float | None:
        if self.state != "ready":
            return None
        return max(0.0, self.s.idle_seconds - (self.clock() - self.last_activity))

    # -- start --------------------------------------------------------------------------

    async def ensure_ready(self) -> tuple[int, int]:
        """(generation, cold_start_ms). cold_start_ms is 0 if the model was already up."""
        t0 = self.clock()
        waited = False
        while True:
            if self.state == "ready" and self.proc is not None and self.proc.returncode is not None:
                # Died but the watcher task has not run yet: do not hand out a dead process.
                self.last_exit = self.proc.returncode
                self._fail(f"exit code {self.proc.returncode}")
                self.proc = None
            if self.state == "ready":
                ms = int((self.clock() - t0) * 1000) if waited else 0
                return self.gen, ms
            if self.state == "stopping":
                waited = True
                await self.stopped_event.wait()
                continue
            if self.state in ("stopped", "failed"):
                was_failed = self.state == "failed"
                prev = self.state
                self.state = "starting"  # sync with the check above: one starter only
                task = asyncio.create_task(self._start(was_failed, prev))
                task.add_done_callback(lambda t: t.cancelled() or t.exception())
                self._start_task = task
            waited = True
            task = self._start_task
            assert task is not None
            # Shield: cancelling the first waiter must not abort the shared load.
            await asyncio.shield(task)

    async def _start(self, was_failed: bool, prev_state: str) -> None:
        s = self.s
        now = self.clock()
        if was_failed:
            while self._failed_restarts and now - self._failed_restarts[0] > s.restart_window:
                self._failed_restarts.popleft()
            if len(self._failed_restarts) >= s.restart_max:
                self.state = prev_state
                raise ApiError(
                    503, "restart_throttled",
                    "Модель слишком часто падает, перезапуск временно отключён.",
                    type_="server_error", retry_after=60,
                )  # fmt: skip
            self._failed_restarts.append(now)

        avail = read_mem_available_mb(s.meminfo_path)
        if avail is not None and avail < s.mem_available_min_mb:
            self.state = prev_state
            raise ApiError(
                503, "host_memory_low",
                f"На сервере мало свободной памяти ({avail:.0f} МБ), модель не запускается.",
                type_="server_error", retry_after=30,
            )  # fmt: skip

        self.gen += 1
        gen = self.gen
        self.tail.clear()
        self.last_error = None
        self.stopped_event.clear()
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *s.llama_argv(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=s.child_env(),
            )
        except OSError as e:
            self.proc = None
            self.stopped_event.set()
            self._fail(f"spawn failed: {e}")
            raise ApiError(
                502, "model_start_failed", "Не удалось запустить модель.", type_="server_error"
            ) from None
        self.spawns += 1
        proc = self.proc
        self.spawned.append(proc)
        self._drains = [
            asyncio.create_task(self._drain(proc.stdout)),
            asyncio.create_task(self._drain(proc.stderr)),
        ]
        self._watcher = asyncio.create_task(self._watch(gen, proc))

        try:
            await self._wait_healthy(gen, proc)
            await self._verify_props()
        except ApiError as e:
            await self._teardown(gen, final="failed", reason=e.message)
            raise
        except asyncio.CancelledError:
            await asyncio.shield(self._teardown(gen, final="stopped", reason="start cancelled"))
            raise
        self.state = "ready"
        self.ready_at = self.clock()
        self.last_activity = self.ready_at

    async def _wait_healthy(self, gen: int, proc: asyncio.subprocess.Process) -> None:
        s = self.s
        deadline = self.clock() + s.start_timeout
        while True:
            if proc.returncode is not None:
                self.last_exit = proc.returncode
                raise ApiError(
                    502, "model_start_failed",
                    f"Модель завершилась при старте (код {proc.returncode}).",
                    type_="server_error", extra={"stderr_tail": self.tail_text(5)},
                )  # fmt: skip
            with contextlib.suppress(httpx.HTTPError):
                r = await self.client.get("/health", timeout=1.0)
                if r.status_code == 200:
                    return
            if self.clock() >= deadline:
                raise ApiError(
                    504, "model_start_timeout",
                    f"Модель не запустилась за {s.start_timeout:g} с.", type_="server_error",
                )  # fmt: skip
            await asyncio.sleep(s.health_poll)

    async def _verify_props(self) -> None:
        try:
            r = await self.client.get("/props", timeout=5.0)
            n_ctx = r.json()["default_generation_settings"]["n_ctx"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            n_ctx = None
        if n_ctx != self.s.n_ctx:
            raise ApiError(
                502, "n_ctx_mismatch",
                f"Контекст модели ({n_ctx}) не совпадает с ожидаемым ({self.s.n_ctx}).",
                type_="server_error",
            )  # fmt: skip

    # -- child output -------------------------------------------------------------------

    async def _drain(self, stream: asyncio.StreamReader | None) -> None:
        """Read continuously: an unread pipe fills up and freezes the child."""
        if stream is None:
            return
        partial = ""
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            text = partial + chunk.decode("utf-8", errors="replace")
            *lines, partial = text.split("\n")
            for line in lines:
                self.tail.append(line[-500:])
            partial = partial[-500:]
        if partial:
            self.tail.append(partial)

    def tail_text(self, n: int = 10) -> str:
        return "\n".join(list(self.tail)[-n:])

    async def _watch(self, gen: int, proc: asyncio.subprocess.Process) -> None:
        rc = await proc.wait()
        if self.gen == gen and self.state == "ready":
            self.last_exit = rc
            self._fail(f"exit code {rc}")
            self.proc = None

    def _fail(self, reason: str) -> None:
        self.state = "failed"
        self.last_error = reason

    # -- stop ---------------------------------------------------------------------------

    async def _teardown(self, gen: int, *, final: str, reason: str | None = None) -> None:
        if gen != self.gen or self.state == "stopping":
            if self.state == "stopping":
                await self.stopped_event.wait()
            return
        proc = self.proc
        self.state = "stopping"
        self.stopped_event.clear()
        try:
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), self.s.kill_grace)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
            if proc is not None:
                self.last_exit = proc.returncode
            if self._drains:
                _, pending = await asyncio.wait(self._drains, timeout=DRAIN_JOIN_TIMEOUT)
                for t in pending:
                    t.cancel()
                self._drains = []
        except asyncio.CancelledError:
            # Shutdown cancelled us mid-wait: a SIGTERM-resistant child must still die.
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(proc.wait()), KILL_REAP_TIMEOUT)
            raise
        finally:
            if proc is None or proc.returncode is not None:
                self.proc = None  # only once reaped; otherwise shutdown() still sees it
            self.state = final
            self.last_error = reason if final == "failed" else None
            self.stopped_event.set()

    async def kill_current(self, gen: int) -> None:
        """Hard stop after a deadline; no-op if that generation is already gone."""
        await self._teardown(gen, final="stopped")

    async def slot_busy(self) -> bool:
        try:
            r = await self.client.get("/slots", timeout=2.0)
            return any(sl.get("is_processing") for sl in r.json())
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return False

    async def release_after_timeout(self, gen: int) -> None:
        """Upstream was closed on a deadline: if the slot stays busy, kill the child."""
        end = self.clock() + self.s.slot_check_grace
        while self.clock() < end:
            if not await self.slot_busy():
                return
            await asyncio.sleep(0.1)
        await self.kill_current(gen)

    # -- idle watchdog ------------------------------------------------------------------

    def start_watchdog(self) -> None:
        if self._watchdog is None:
            self._watchdog = asyncio.create_task(self._watchdog_loop())

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.watchdog_interval)
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 - one bad tick must not end the watchdog
                continue

    async def tick(self) -> None:
        idle_for = self.clock() - self.last_activity
        if self.state == "ready" and not self.busy() and idle_for >= self.s.idle_seconds:
            await self._teardown(self.gen, final="stopped")

    async def shutdown(self) -> None:
        if self._watchdog:
            self._watchdog.cancel()
            await asyncio.gather(self._watchdog, return_exceptions=True)
            self._watchdog = None
        if self._start_task and not self._start_task.done():
            self._start_task.cancel()
            await asyncio.gather(self._start_task, return_exceptions=True)
        if self.proc is not None:
            await self._teardown(self.gen, final="stopped")
        if self._watcher:
            self._watcher.cancel()
            await asyncio.gather(self._watcher, return_exceptions=True)
        for t in self._drains:
            t.cancel()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- introspection ------------------------------------------------------------------

    def rss_kb(self) -> int | None:
        if self.proc is None or self.state != "ready":
            return None
        try:
            for line in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        except (OSError, ValueError, IndexError):
            return None
        return None

    def uptime(self) -> float | None:
        if self.state != "ready" or self.ready_at is None:
            return None
        return self.clock() - self.ready_at
