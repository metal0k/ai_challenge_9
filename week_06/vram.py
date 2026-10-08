"""Day 29: GPU memory sampled by a background thread through `nvidia-smi` (no HTTP, no guard).

A missing tool or any failure yields None everywhere; the measurement carries on without it.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

QUERY = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
INTERVAL_S = 1.0
QUERY_TIMEOUT_S = 5
STOP_WAIT_S = QUERY_TIMEOUT_S + 2  # an in-flight nvidia-smi may take its whole timeout
NO_DATA = "VRAM: нет данных"

QueryFn = Callable[[], "tuple[int, int] | None"]


def query_nvidia_smi() -> tuple[int, int] | None:
    """(used MiB, total MiB) of the first GPU; None when the tool is absent or misbehaves."""
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            QUERY,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=QUERY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    first = (done.stdout or "").strip().splitlines()[:1]
    if not first:
        return None
    try:
        used, total = (int(part.strip()) for part in first[0].split(","))
    except ValueError:
        return None
    return used, total


@dataclass(frozen=True, slots=True)
class VramReport:
    vram_used_start: int | None = None
    vram_used_peak: int | None = None
    vram_total: int | None = None

    @property
    def known(self) -> bool:
        return self.vram_used_peak is not None

    @property
    def free_at_peak(self) -> int | None:
        if self.vram_used_peak is None or self.vram_total is None:
            return None
        return self.vram_total - self.vram_used_peak

    def as_dict(self) -> dict[str, int | None]:
        return {
            "vram_used_start": self.vram_used_start,
            "vram_used_peak": self.vram_used_peak,
            "vram_total": self.vram_total,
        }


def report_from_dict(raw: object) -> VramReport:
    """A saved report; anything malformed reads as 'no data'."""
    if not isinstance(raw, dict):
        return VramReport()

    def num(key: str) -> int | None:
        value = raw.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return VramReport(num("vram_used_start"), num("vram_used_peak"), num("vram_total"))


def describe(report: VramReport) -> str:
    if not report.known:
        return NO_DATA
    return (
        f"VRAM: старт {report.vram_used_start} MiB, пик {report.vram_used_peak} MiB "
        f"из {report.vram_total} MiB (свободно на пике {report.free_at_peak} MiB)"
    )


class VramSampler:
    """One synchronous sample on start, then one per `interval` until stop() (plus a last one)."""

    def __init__(
        self,
        query: QueryFn = query_nvidia_smi,
        interval: float = INTERVAL_S,
        stop_wait: float = STOP_WAIT_S,
    ) -> None:
        self._query = query
        self._stop_wait = stop_wait
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._start: int | None = None
        self._peak: int | None = None
        self._total: int | None = None
        self._closed = False

    def _sample(self) -> None:
        try:
            got = self._query()
        except Exception:  # noqa: BLE001 - a sampler must never take the measurement down
            got = None
        if got is None:
            return
        used, total = got
        with self._lock:
            if self._closed:  # the report was already handed out: it never changes again
                return
            if self._start is None:
                self._start = used
            if self._peak is None or used > self._peak:
                self._peak = used
            self._total = total

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def start(self) -> VramSampler:
        self._sample()
        self._thread.start()
        return self

    def stop(self) -> VramReport:
        """Let an in-flight query finish (bounded), take a last sample, freeze the report."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._stop_wait)
        self._sample()
        with self._lock:
            self._closed = True
            return VramReport(self._start, self._peak, self._total)

    def __enter__(self) -> VramSampler:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
