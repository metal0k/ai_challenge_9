"""API keys (hashes only) and rate limiting with an injectable clock."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Clock = Callable[[], float]


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KeyRecord:
    name: str
    sha256: str
    rpm: float
    burst: int


def load_keys(path: str | Path) -> list[KeyRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("keys file must be a JSON list")
    out: list[KeyRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("key entry must be an object")
        digest = str(item["sha256"]).lower()
        if len(digest) != 64:
            raise ValueError(f"key {item.get('name')!r}: sha256 must be 64 hex chars")
        out.append(KeyRecord(str(item["name"]), digest, float(item["rpm"]), int(item["burst"])))
    return out


class KeyStore:
    def __init__(self, records: list[KeyRecord]):
        self.records = records

    def authenticate(self, token: str) -> KeyRecord | None:
        digest = hash_key(token)
        found = None
        # No early exit: the time spent does not depend on which key matched.
        for rec in self.records:
            if hmac.compare_digest(digest, rec.sha256):
                found = rec
        return found


class TokenBucket:
    def __init__(self, rpm: float, burst: int, clock: Clock = time.monotonic):
        self.rate = rpm / 60.0
        self.capacity = float(burst)
        self.clock = clock
        self.tokens = float(burst)
        self.stamp = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.stamp) * self.rate)
        self.stamp = now

    def take(self) -> tuple[bool, int]:
        """(ok, retry_after_seconds); a refused take consumes nothing."""
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0
        if self.rate <= 0:
            return False, 3600
        return False, max(1, math.ceil((1.0 - self.tokens) / self.rate))

    def peek(self) -> float:
        self._refill()
        return self.tokens


class RateLimiter:
    """One bucket per key name."""

    def __init__(self, records: list[KeyRecord], clock: Clock = time.monotonic):
        self.buckets = {r.name: TokenBucket(r.rpm, r.burst, clock) for r in records}

    def take(self, name: str) -> tuple[bool, int]:
        return self.buckets[name].take()

    def left(self, name: str) -> float:
        return self.buckets[name].peek()


class SlidingWindow:
    """Global budget: at most `limit` events within `window` seconds."""

    def __init__(self, limit: int, window: float = 60.0, clock: Clock = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.events: deque[float] = deque()

    def take(self) -> tuple[bool, int]:
        now = self.clock()
        while self.events and now - self.events[0] >= self.window:
            self.events.popleft()
        if len(self.events) >= self.limit:
            return False, max(1, math.ceil(self.window - (now - self.events[0])))
        self.events.append(now)
        return True, 0
