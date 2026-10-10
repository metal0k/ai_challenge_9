"""Regression tests for the day-30 code-review findings (gateway side)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from tests.w06d30_gateway_helpers import (
    OWNER_KEY,
    Live,
    backend_or_none,
    chat_body,
    make_settings,
    raw_post,
    read_all,
    records,
)
from week_06.gateway.app import create_app
from week_06.gateway.errors import ApiError
from week_06.gateway.lifecycle import ProcessManager
from week_06.gateway.settings import Settings
from week_06.gateway.validate import parse_chat_request

CHAT = "/v1/chat/completions"


def _events(text: str) -> list[str]:
    return [ln[6:] for ln in text.splitlines() if ln.startswith("data: ")]


# -- #1 pre-header disconnects ----------------------------------------------------------


def test_disconnect_while_queued_releases_ticket_and_never_generates(tmp_path):
    with Live(make_settings(tmp_path)) as lv:
        first = raw_post(lv, chat_body("#chunks=10;delay=0.15;watch=1#a"))
        lv.wait_for(lambda: lv.gw.admission.sem.locked(), what="first holds the slot")
        second = raw_post(lv, chat_body("queued"))
        lv.wait_for(lambda: lv.gw.admission.occupied == 2, what="second queued")
        second.close()
        lv.wait_for(lambda: lv.gw.admission.occupied == 1, timeout=1.0, what="ticket released")
        assert b"200 OK" in read_all(first)
        first.close()
        lv.wait_for(lambda: lv.gw.admission.occupied == 0, what="line empty")
        time.sleep(0.3)
        assert lv.backend()["total"] == 1  # the abandoned request never reached generation


def test_disconnect_during_cold_start_releases_ticket_and_never_generates(tmp_path):
    with Live(make_settings(tmp_path, {"startup_delay": 3.0})) as lv:
        sock = raw_post(lv, chat_body("cold"))
        lv.wait_for(lambda: lv.gw.manager.state == "starting", what="cold start began")
        sock.close()
        lv.wait_for(lambda: lv.gw.admission.occupied == 0, timeout=1.5, what="ticket released")
        lv.wait_for(lambda: lv.gw.manager.state == "ready", timeout=6, what="start finished")
        time.sleep(0.3)
        assert lv.backend()["total"] == 0


@pytest.mark.parametrize(
    ("magic", "stream"),
    [("#chunks=3;hdelay=2.5;watch=1#x", True), ("#chunks=10;delay=0.3;watch=1#x", False)],
    ids=["waiting-for-stream-headers", "non-stream-completion"],
)
def test_disconnect_before_response_headers_aborts_backend_and_frees_admission(
    tmp_path, magic, stream
):
    with Live(make_settings(tmp_path)) as lv:
        sock = raw_post(lv, chat_body(magic, stream=stream))
        lv.wait_for(lambda: (backend_or_none(lv) or {}).get("current") == 1, what="backend busy")
        sock.close()
        lv.wait_for(lambda: lv.gw.admission.occupied == 0, timeout=1.0, what="admission freed")
        lv.wait_for(lambda: lv.backend()["aborted"] >= 1, timeout=1.0, what="backend aborted")
        assert lv.backend()["current"] == 0


# -- #2 streaming must not inherit the short client timeout -----------------------------


@pytest.mark.parametrize("magic", ["#chunks=2;hdelay=1.3#x", "#chunks=2;fdelay=1.3#x"])
def test_stream_survives_silence_longer_than_the_client_read_timeout(tmp_path, magic):
    with Live(make_settings(tmp_path, upstream_timeout=0.5)) as lv, lv.client() as c:
        with c.stream("POST", CHAT, json=chat_body(magic, stream=True)) as r:
            assert r.status_code == 200
            text = "\n".join(r.iter_lines())
        events = _events(text)
        assert events[-1] == "[DONE]"
        assert not [e for e in events if e.startswith('{"error"')]
        lv.wait_for(lambda: lv.gw.stats.outcomes["ok"] == 1, what="outcome ok")


# -- #3 cancelled teardown must not orphan the child ------------------------------------


class _StubbornProc:
    """Ignores terminate(); dies only on kill()."""

    pid = 4242

    def __init__(self):
        self.returncode = None
        self.killed = False
        self._dead = asyncio.Event()

    def terminate(self):
        pass

    def kill(self):
        self.killed = True
        self.returncode = -9
        self._dead.set()

    async def wait(self):
        await self._dead.wait()
        return self.returncode


def test_cancelled_teardown_still_kills_and_reaps_a_sigterm_resistant_child(tmp_path):
    async def main():
        m = ProcessManager(make_settings(tmp_path, kill_grace=30.0))
        proc = _StubbornProc()
        m.proc, m.state, m.gen = proc, "ready", 1  # type: ignore[assignment]
        task = asyncio.create_task(m._teardown(1, final="stopped"))
        await asyncio.sleep(0.1)  # now waiting for the child after SIGTERM
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await m.shutdown()
        assert proc.killed, "child survived the cancelled teardown"
        assert proc.returncode is not None
        assert m.proc is None and m.state == "stopped"

    asyncio.run(main())


# -- #5 admission is kept until the backend slot is really free -------------------------


def test_admission_waits_for_backend_cancel_lag_after_midstream_disconnect(tmp_path):
    settings = make_settings(tmp_path, {"cancel_lag": 0.5}, slot_check_grace=3.0)
    with Live(settings) as lv, lv.client() as c:
        with c.stream("POST", CHAT, json=chat_body("#chunks=200;delay=0.05#x", stream=True)) as r:
            for ln in r.iter_lines():
                if ln.startswith("data: "):
                    break
        # the next request goes in at once, while the backend is still cancelling
        assert c.post(CHAT, json=chat_body("next")).status_code == 200
        assert lv.backend()["max_concurrent"] == 1
        assert lv.gw.manager.spawns == 1  # waited for the slot, did not kill the child


# -- #6 the deadline covers response transmission ---------------------------------------


def test_blocked_downstream_send_is_cut_by_the_generation_deadline(tmp_path):
    settings = make_settings(tmp_path, gen_deadline=0.8)
    body = json.dumps(chat_body("#chunks=200;delay=0.05#x", stream=True)).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "path": CHAT, "raw_path": CHAT.encode(), "query_string": b"", "root_path": "",
        "scheme": "http", "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80),
        "headers": [
            (b"authorization", f"Bearer {OWNER_KEY}".encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    }  # fmt: skip

    async def main():
        app = create_app(settings, key_records=records())
        gw = app.gateway
        sent_first = asyncio.Event()

        async def receive():
            if not sent_first.is_set():
                sent_first.set()
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()  # the client stays connected, forever

        async def send(msg):
            if msg["type"] == "http.response.body" and msg.get("more_body"):
                await asyncio.sleep(3600)  # the client stopped reading

        try:
            t0 = time.monotonic()
            await asyncio.wait_for(app(scope, receive, send), 5.0)
            assert time.monotonic() - t0 < 4.0
            assert gw.admission.occupied == 0
            assert gw.stats.outcomes["timeout"] == 1
        finally:
            await gw.manager.shutdown()

    asyncio.run(main())


# -- #7 [DONE] is an SSE event, not a substring -----------------------------------------


def test_done_inside_content_with_premature_eof_is_not_success(tmp_path):
    fake = {"chunk_text": "see [DONE] marker", "no_done": True}
    with Live(make_settings(tmp_path, fake)) as lv, lv.client() as c:
        with c.stream("POST", CHAT, json=chat_body(stream=True)) as r:
            events = _events("\n".join(r.iter_lines()))
        assert "[DONE]" not in events
        assert json.loads(events[-1])["error"]["code"] == "upstream_closed"
        lv.wait_for(lambda: lv.gw.admission.occupied == 0, what="admission freed")
        assert lv.gw.stats.outcomes["upstream_error"] == 1
        assert lv.gw.stats.outcomes["ok"] == 0


def _feed(parts: list[bytes]) -> bool:
    from week_06.gateway.app import SseDoneTracker

    t = SseDoneTracker()
    for p in parts:
        t.feed(p)
    return t.done


def test_done_tracker_framing():
    assert _feed([b'data: {"a":1}\n\n', b"data: [DONE]\n\n"])
    assert _feed([b"data: [DO", b"NE]\n", b"\n"])  # split marker
    assert _feed([b"data: [DONE]\r\n\r\n"])  # CRLF framing
    assert _feed([b"data: [DONE]\r", b"\n\r", b"\n"])  # CR/LF split across chunks
    assert not _feed([b"data: [DONE]"])  # partial event
    assert not _feed([b"data: [DONE]\n"])  # partial event
    assert not _feed([b'data: {"c":"[DONE]"}\n\n'])  # inside content
    assert not _feed([b'data: {"c":"x"}\n\ndata: [DONE] more\n\n'])  # not the whole data
    assert not _feed([b"data: x\n\n"])  # premature EOF after a normal event


# -- #10 role of the wrong JSON type ----------------------------------------------------


@pytest.mark.parametrize("role", [[], {}, 5, None, True, ["user"]])
def test_non_string_role_is_a_400_not_a_crash(role):
    raw = json.dumps({"messages": [{"role": role, "content": "x"}]}).encode()
    with pytest.raises(ApiError) as ei:
        parse_chat_request(raw, Settings())
    assert ei.value.status == 400 and ei.value.code == "invalid_messages"
