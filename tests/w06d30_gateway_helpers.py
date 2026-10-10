"""Shared harness for the day-30 gateway tests: a live uvicorn thread + a real fake child."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from week_06.gateway.app import create_app
from week_06.gateway.keys import KeyRecord, hash_key
from week_06.gateway.settings import Settings

FAKE = Path(__file__).parent / "fixtures" / "w06d30" / "fake_llama.py"
OWNER_KEY = "adv_test_owner_key_0123456789"
DEMO_KEY = "adv_test_demo_key_9876543210"


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def records(owner_rpm: float = 6000, owner_burst: int = 100) -> list[KeyRecord]:
    return [
        KeyRecord("owner", hash_key(OWNER_KEY), owner_rpm, owner_burst),
        KeyRecord("demo", hash_key(DEMO_KEY), 3, 2),
    ]


def make_settings(tmp_path: Path, fake: dict | None = None, **over) -> Settings:
    port = free_port()
    argv = [getattr(sys, "_base_executable", sys.executable), str(FAKE),
            "--port", str(port), "-c", "2048"]  # fmt: skip
    values = dict(
        upstream_port=port,
        llama_argv_override=argv,
        child_extra_env={"FAKE_LLAMA": json.dumps(fake or {})},
        meminfo_path=str(tmp_path / "no-such-meminfo"),
        idle_seconds=60.0,
        watchdog_interval=0.1,
        kill_grace=2.0,
        health_poll=0.05,
        start_timeout=10.0,
        queue_timeout=20.0,
        gen_deadline=20.0,
        body_read_deadline=5.0,
        slot_check_grace=0.3,
        global_per_minute=10_000,
        version="test-sha",
    )
    values.update(over)
    return Settings(**values)


class Live:
    """Gateway under a real uvicorn in a thread, so disconnects and chunking are real."""

    def __init__(self, settings: Settings, recs=None, clock=None):
        self.settings = settings
        kwargs = {"key_records": recs or records()}
        if clock is not None:
            kwargs["clock"] = clock
        self.app = create_app(settings, **kwargs)
        self.gw = self.app.gateway
        self.port = free_port()
        cfg = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="error",
            limit_concurrency=32, timeout_keep_alive=5, access_log=False,
        )  # fmt: skip
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.url = f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        end = time.time() + 10
        while not self.server.started:
            assert time.time() < end, "uvicorn did not start"
            time.sleep(0.02)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=15)
        assert not self.thread.is_alive(), "gateway thread did not stop"
        # nothing may outlive the gateway
        for proc in self.gw.manager.spawned:
            assert proc.returncode is not None, f"leaked child pid={proc.pid}"

    def client(self, key: str | None = OWNER_KEY, **kw) -> httpx.Client:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return httpx.Client(base_url=self.url, headers=headers, timeout=30, trust_env=False, **kw)

    def backend(self) -> dict:
        r = httpx.get(
            f"http://127.0.0.1:{self.settings.upstream_port}/_stats", timeout=5, trust_env=False
        )
        return r.json()

    def wait_for(self, cond, timeout: float = 8.0, what: str = "condition"):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.03)
        raise AssertionError(f"timeout waiting for {what}")


def chat_body(text: str = "привет", **extra) -> dict:
    return {"messages": [{"role": "user", "content": text}], **extra}


def raw_post(lv: Live, body: dict, key: str = OWNER_KEY) -> socket.socket:
    """Send a chat request over a bare socket; closing it is a real client disconnect."""
    raw = json.dumps(body).encode()
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {key}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    sock = socket.create_connection(("127.0.0.1", lv.port), timeout=10)
    sock.sendall(head + raw)
    return sock


def read_all(sock: socket.socket) -> bytes:
    out = b""
    while chunk := sock.recv(65536):
        out += chunk
    return out


def backend_or_none(lv: Live) -> dict | None:
    try:
        return lv.backend()
    except httpx.HTTPError:
        return None
