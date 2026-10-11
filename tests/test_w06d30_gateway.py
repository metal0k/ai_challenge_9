"""Gateway HTTP behaviour against a live uvicorn and a REAL fake llama-server child."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.w06d30_gateway_helpers import (
    DEMO_KEY,
    OWNER_KEY,
    FakeClock,
    Live,
    chat_body,
    make_settings,
    records,
)

CHAT = "/v1/chat/completions"


@pytest.fixture
def live(tmp_path):
    with Live(make_settings(tmp_path)) as lv:
        yield lv


# -- auth / static ----------------------------------------------------------------------


def test_missing_and_wrong_key_give_401_and_never_wake_the_model(live):
    with live.client(None) as anon, live.client("adv_wrong") as wrong:
        assert anon.post(CHAT, json=chat_body()).status_code == 401
        r = wrong.post(CHAT, json=chat_body())
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "invalid_api_key"
        assert anon.get("/health").status_code == 401
        assert anon.get("/v1/status").status_code == 401
        assert anon.get("/v1/models").status_code == 401
    assert live.gw.manager.state == "stopped"
    assert live.gw.manager.spawns == 0


def test_anonymous_pages_and_csp(live):
    with live.client(None) as c:
        page = c.get("/")
        js = c.get("/chat.js")
        css = c.get("/chat.css")
    assert page.status_code == js.status_code == css.status_code == 200
    for r in (page, js, css):
        assert r.headers["content-security-policy"].startswith("default-src 'self'")
    assert '<script src="/chat.js"' in page.text


def test_health_does_not_wake_model_and_status_reports_literal_counters(live):
    with live.client() as c, live.client(None) as anon:
        anon.get("/health")  # 401
        r = c.get("/health")
        assert r.json() == {"status": "ok", "model": "stopped"}
        assert c.post(CHAT, json=chat_body("a")).status_code == 200
        st = c.get("/v1/status").json()
    assert st["version"] == "test-sha"
    assert st["model"] == "ready"
    assert st["counters"]["401"] == 1
    assert st["counters"]["200"] == 2  # /health and the chat
    assert st["by_code"]["401"] == 1 and st["by_code"]["200"] == 2
    assert st["key"]["name"] == "owner"
    assert st["limits"]["n_ctx"] == 2048
    assert st["queue"] == {"active": 0, "waiting": 0, "max": 2}


def test_models_lists_one_id(live):
    with live.client() as c:
        data = c.get("/v1/models").json()
    assert [m["owned_by"] for m in data["data"]] == ["advent-llm"]


# -- rate limit -------------------------------------------------------------------------


def test_rate_limit_429_retry_after_with_fake_clock_and_no_wake(tmp_path):
    clock = FakeClock()
    with Live(make_settings(tmp_path), clock=clock) as lv, lv.client(DEMO_KEY) as c:
        assert c.post(CHAT, json=chat_body()).status_code == 200
        assert c.post(CHAT, json=chat_body()).status_code == 200
        r = c.post(CHAT, json=chat_body())
        assert r.status_code == 429
        assert r.headers["retry-after"] == "20"  # 3 rpm -> one token per 20 s
        assert r.json()["error"]["code"] == "rate_limit_exceeded"
        clock.advance(20)
        assert c.post(CHAT, json=chat_body()).status_code == 200
        assert lv.gw.manager.spawns == 1


def test_limited_requests_do_not_start_the_model(tmp_path):
    clock = FakeClock()
    recs = records()
    recs[1] = type(recs[1])("demo", recs[1].sha256, 3, 1)
    with Live(make_settings(tmp_path), recs, clock) as lv, lv.client(DEMO_KEY) as c:
        # burst 1: the first passes validation and wakes the model, the next ones must not
        # touch anything new
        assert c.post(CHAT, json=chat_body()).status_code == 200
        before = lv.gw.manager.last_activity
        for _ in range(3):
            assert c.post(CHAT, json=chat_body()).status_code == 429
        assert lv.gw.manager.last_activity == before
        assert lv.gw.manager.spawns == 1


def test_global_budget_429(tmp_path):
    clock = FakeClock()
    settings = make_settings(tmp_path, global_per_minute=2)
    with Live(settings, clock=clock) as lv, lv.client() as c:
        assert c.post(CHAT, json=chat_body()).status_code == 200
        assert c.post(CHAT, json=chat_body()).status_code == 200
        r = c.post(CHAT, json=chat_body())
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "global_rate_limit"
        clock.advance(61)
        assert c.post(CHAT, json=chat_body()).status_code == 200


# -- admission --------------------------------------------------------------------------


def _slow_post(lv, stream: bool):
    body = chat_body("#chunks=3;delay=0.3#x", stream=stream)
    with lv.client() as c:
        r = c.post(CHAT, json=body)
        return r.status_code, r.headers.get("retry-after")


@pytest.mark.parametrize("stream", [False, True])
def test_queue_full_503_and_backend_concurrency_never_exceeds_one(tmp_path, stream):
    with Live(make_settings(tmp_path)) as lv:
        with ThreadPoolExecutor(8) as pool:
            futs = [pool.submit(_slow_post, lv, stream) for _ in range(6)]
            results = [f.result() for f in futs]
        codes = sorted(c for c, _ in results)
        assert codes.count(200) == 3  # 1 active + queue_max 2
        assert codes.count(503) == 3
        assert all(ra == "10" for c, ra in results if c == 503)
        assert lv.backend()["max_concurrent"] == 1
        st = lv.client().get("/v1/status").json()
        assert st["queue"]["active"] == 0 and st["queue"]["waiting"] == 0
        assert lv.gw.admission.occupied == 0


def test_queue_wait_header_grows_for_waiters(tmp_path):
    with Live(make_settings(tmp_path)) as lv:

        def one(_):
            with lv.client() as c:
                r = c.post(CHAT, json=chat_body("#chunks=2;delay=0.2#x"))
                return int(r.headers["x-advent-queue-wait-ms"])

        with ThreadPoolExecutor(3) as pool:
            waits = sorted(pool.map(one, range(3)))
    assert waits[0] < 300 and waits[-1] >= 300


def test_queue_timeout_503(tmp_path):
    with Live(make_settings(tmp_path, queue_timeout=0.3)) as lv:
        with ThreadPoolExecutor(2) as pool:
            slow = pool.submit(_slow_post, lv, False)
            time.sleep(0.15)
            with lv.client() as c:
                r = c.post(CHAT, json=chat_body("#chunks=10;delay=0.2#x"))
            slow.result()
        # first one finished (0.45 s) or the second timed out waiting; either way it must
        # be a clean answer and the line must be empty afterwards
        assert r.status_code in (200, 503)
        assert lv.gw.admission.occupied == 0


# -- body / validation ------------------------------------------------------------------


def test_body_too_large_with_content_length(live):
    big = json.dumps(chat_body("x" * 70_000))
    with live.client() as c:
        r = c.post(CHAT, content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "body_too_large"
    assert "input_tokens" not in r.json()["error"]
    assert live.gw.manager.spawns == 0


def test_body_too_large_chunked_without_content_length(live):
    def chunks():
        yield b'{"messages":[{"role":"user","content":"'
        for _ in range(80):
            yield b"x" * 1000
        yield b'"}]}'

    with live.client() as c:
        req = c.build_request("POST", CHAT, content=chunks())
        assert "content-length" not in req.headers
        r = c.send(req)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "body_too_large"
    assert live.gw.manager.spawns == 0


def test_content_encoding_rejected_415(live):
    with live.client() as c:
        r = c.post(CHAT, content=b"{}", headers={"Content-Encoding": "gzip"})
    assert r.status_code == 415


def test_exact_context_boundary_ok_and_one_over_rejected(live):
    # the fake counts one token per character, so input_tokens is exact
    with live.client() as c:
        ok = c.post(CHAT, json=chat_body("x" * 1536, max_tokens=512))  # 1536 + 512 == 2048
        over = c.post(CHAT, json=chat_body("x" * 1537, max_tokens=512))  # 2049
    assert ok.status_code == 200
    assert ok.headers["x-advent-input-tokens"] == "1536"
    assert over.status_code == 413
    err = over.json()["error"]
    assert err["code"] == "context_length_exceeded"
    assert (err["input_tokens"], err["max_tokens"], err["n_ctx"]) == (1537, 512, 2048)
    assert live.backend()["total"] == 1  # the rejected one never reached generation


def test_upstream_exceed_context_is_mapped_to_413(live):
    with live.client() as c:
        for stream in (False, True):
            r = c.post(CHAT, json=chat_body("#exceed=1#x", stream=stream))
            assert r.status_code == 413
            err = r.json()["error"]
            assert err["code"] == "context_length_exceeded"
            assert err["input_tokens"] == 3000


@pytest.mark.parametrize("bad", [-1, 0, True, "5", 1.5, 513])
def test_invalid_max_tokens_400(live, bad):
    with live.client() as c:
        r = c.post(CHAT, json=chat_body(max_tokens=bad))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_max_tokens"
    assert live.gw.manager.spawns == 0


def test_max_tokens_default_ceiling_and_null(live):
    with live.client() as c:
        c.post(CHAT, json=chat_body())
        assert live.backend()["last_body"]["max_tokens"] == 256
        c.post(CHAT, json=chat_body(max_tokens=None))
        assert live.backend()["last_body"]["max_tokens"] == 256
        c.post(CHAT, json=chat_body(max_tokens=512))
        assert live.backend()["last_body"]["max_tokens"] == 512


def test_default_temperature_and_field_whitelist_reach_backend(live):
    with live.client() as c:
        r = c.post(CHAT, json=chat_body(model="whatever", seed=7, stop="END"))
        assert r.status_code == 200
        body = live.backend()["last_body"]
        assert body["temperature"] == 0.7
        assert body["stop"] == ["END"] and body["seed"] == 7
        assert "model" not in body


@pytest.mark.parametrize("field", ["n_probs", "slot_id", "grammar", "cache_prompt", "tools"])
def test_unsupported_field_400(live, field):
    with live.client() as c:
        r = c.post(CHAT, json=chat_body(**{field: 1}))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_field"
    assert field in r.json()["error"]["message"]


def test_invalid_json_400(live):
    with live.client() as c:
        r = c.post(CHAT, content=b"{nope", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_json"


# -- proxy hygiene ----------------------------------------------------------------------


def test_client_headers_are_not_forwarded_upstream(live):
    canary = "CANARY-5f3a"
    with live.client() as c:
        r = c.post(
            CHAT,
            json=chat_body(),
            headers={"Cookie": f"s={canary}", "X-Canary": canary, "X-Forwarded-For": canary},
        )
    assert r.status_code == 200
    seen = live.backend()["last_headers"]
    assert set(seen) <= {"host", "content-type", "content-length", "accept", "accept-encoding",
                         "connection", "user-agent"}  # fmt: skip
    assert canary not in json.dumps(seen)
    assert OWNER_KEY not in json.dumps(seen)
    assert seen["content-type"] == "application/json"
    assert canary not in json.dumps(dict(r.headers))


def test_response_headers_are_whitelisted(live):
    with live.client() as c:
        r = c.post(CHAT, json=chat_body())
    names = {k.lower() for k in r.headers}
    assert {"x-advent-queue-wait-ms", "x-advent-cold-start-ms", "x-advent-input-tokens"} <= names
    assert "x-canary" not in names


# -- streaming --------------------------------------------------------------------------


def _events(lines):
    return [ln[6:] for ln in lines if ln.startswith("data: ")]


def test_stream_injects_include_usage_and_arrives_incrementally(live):
    stamps = []
    with live.client() as c, c.stream(
        "POST",
        CHAT,
        json=chat_body("#chunks=5;delay=0.15#x", stream=True,
                       stream_options={"include_usage": False}),
    ) as r:  # fmt: skip
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = []
        for ln in r.iter_lines():
            if ln.startswith("data: "):
                stamps.append(time.monotonic())
            lines.append(ln)
        assert r.headers["x-advent-cold-start-ms"].isdigit()
    events = _events(lines)
    assert events[-1] == "[DONE]"
    usage = [json.loads(e) for e in events[:-1] if '"usage"' in e]
    assert usage and usage[0]["usage"]["completion_tokens"] == 5
    assert live.backend()["last_body"]["stream_options"] == {"include_usage": True}
    assert stamps[-1] - stamps[0] >= 0.4  # chunks were not buffered into one blob


def test_stream_crash_after_first_chunk_yields_sse_error_event_then_restart(live):
    with live.client() as c:
        with c.stream(
            "POST", CHAT, json=chat_body("#chunks=6;delay=0.1;crash=2#x", stream=True)
        ) as r:  # fmt: skip
            assert r.status_code == 200  # headers were already sent
            lines = list(r.iter_lines())
        events = _events(lines)
        contents = [e for e in events if '"delta"' in e]
        errors = [json.loads(e) for e in events if e.startswith('{"error"')]
        assert len(contents) == 2
        assert len(errors) == 1 and errors[0]["error"]["code"] in (
            "upstream_error",
            "upstream_closed",
        )
        assert "[DONE]" not in events
        # the slot is free and the next request restarts the child
        live.wait_for(lambda: live.gw.manager.state == "failed", what="failed state")
        live.wait_for(lambda: live.gw.admission.occupied == 0, what="admission freed")
        ok = c.post(CHAT, json=chat_body("again"))
        assert ok.status_code == 200
        assert live.gw.manager.spawns == 2
        assert live.gw.stats.outcomes["upstream_error"] == 1


def test_client_disconnect_frees_admission_and_generation(live):
    with live.client() as c:
        with c.stream(
            "POST", CHAT, json=chat_body("#chunks=200;delay=0.05#x", stream=True)
        ) as r:  # fmt: skip
            it = r.iter_lines()
            for ln in it:
                if ln.startswith("data: "):
                    break
            assert live.gw.admission.occupied == 1
        # leaving the with-block closes the connection mid-stream
        live.wait_for(lambda: live.gw.admission.occupied == 0, what="admission freed")
        live.wait_for(lambda: live.backend()["aborted"] >= 1, what="backend saw the abort")
        live.wait_for(lambda: live.backend()["current"] == 0, what="slot idle")
        assert live.gw.stats.outcomes["aborted"] == 1
        # and the line is usable at once
        assert c.post(CHAT, json=chat_body("next")).status_code == 200


def test_generation_deadline_504_then_kills_child_with_busy_slot(tmp_path):
    settings = make_settings(tmp_path, gen_deadline=0.4, slot_check_grace=0.3)
    with Live(settings) as lv, lv.client() as c:
        # the fake keeps "processing" after the client left, like a stuck slot
        r = c.post(CHAT, json=chat_body("#chunks=1;delay=2#x"))
        assert r.status_code == 504
        assert r.json()["error"]["code"] == "generation_timeout"
        lv.wait_for(lambda: lv.gw.manager.state == "stopped", what="stuck child killed")
        assert lv.gw.admission.occupied == 0
        assert c.post(CHAT, json=chat_body("after")).status_code == 200


# -- host / lifecycle through HTTP ------------------------------------------------------


def test_host_memory_low_503_and_no_spawn(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 900000 kB\nMemAvailable:  102400 kB\n")
    with Live(make_settings(tmp_path, meminfo_path=str(meminfo))) as lv, lv.client() as c:
        r = c.post(CHAT, json=chat_body())
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "host_memory_low"
        assert lv.gw.manager.spawns == 0
        meminfo.write_text("MemAvailable:  819200 kB\n")
        assert c.post(CHAT, json=chat_body()).status_code == 200


def test_startup_timeout_returns_504_and_reaps_child(tmp_path):
    settings = make_settings(tmp_path, {"never_healthy": True}, start_timeout=0.6)
    with Live(settings) as lv, lv.client() as c:
        r = c.post(CHAT, json=chat_body())
        assert r.status_code == 504
        assert r.json()["error"]["code"] == "model_start_timeout"
        assert lv.gw.manager.state == "failed"
        assert all(p.returncode is not None for p in lv.gw.manager.spawned)
        assert lv.gw.admission.occupied == 0


def test_idle_unload_then_cold_start_again(tmp_path):
    settings = make_settings(tmp_path, {"startup_delay": 0.3}, idle_seconds=0.6)
    with Live(settings) as lv, lv.client() as c:
        first = c.post(CHAT, json=chat_body())
        assert int(first.headers["x-advent-cold-start-ms"]) >= 250
        warm = c.post(CHAT, json=chat_body())
        assert warm.headers["x-advent-cold-start-ms"] == "0"
        lv.wait_for(lambda: lv.gw.manager.state == "stopped", what="idle unload")
        assert lv.gw.manager.spawned[-1].returncode is not None
        assert c.get("/health").json()["model"] == "stopped"
        again = c.post(CHAT, json=chat_body())
        assert int(again.headers["x-advent-cold-start-ms"]) >= 250
        assert lv.gw.manager.spawns == 2


def test_rejected_requests_do_not_extend_idle(live):
    with live.client() as c, live.client(None) as anon:
        assert c.post(CHAT, json=chat_body()).status_code == 200
        before = live.gw.manager.last_activity
        time.sleep(0.05)
        anon.post(CHAT, json=chat_body())  # 401
        c.post(CHAT, json=chat_body(max_tokens=0))  # 400
        c.post(CHAT, json=chat_body(bogus=1))  # 400
        assert live.gw.manager.last_activity == before


def test_n_ctx_mismatch_marks_failed(tmp_path):
    settings = make_settings(tmp_path, {"props_n_ctx": 4096})
    with Live(settings) as lv, lv.client() as c:
        r = c.post(CHAT, json=chat_body())
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "n_ctx_mismatch"
        assert lv.gw.manager.state == "failed"
        assert lv.gw.manager.spawned[-1].returncode is not None


def test_keys_never_reach_the_log(live, caplog):
    caplog.set_level(logging.DEBUG)
    with live.client() as c, live.client("adv_attacker_guess") as bad:
        c.post(CHAT, json=chat_body())
        bad.post(CHAT, json=chat_body())
    text = caplog.text
    assert OWNER_KEY not in text and "adv_attacker_guess" not in text
    assert "key=owner" in text
    assert "attacker" not in text  # 401s are counted, not logged
