"""Gateway units: lifecycle against a real fake child, keys, validation, settings, web page."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from tests.w06d30_gateway_helpers import make_settings
from week_06.gateway.errors import ApiError
from week_06.gateway.keys import (
    KeyRecord,
    KeyStore,
    SlidingWindow,
    TokenBucket,
    hash_key,
    load_keys,
)
from week_06.gateway.lifecycle import ProcessManager, read_mem_available_mb
from week_06.gateway.settings import Settings, load_settings
from week_06.gateway.validate import parse_chat_request

STATIC = Path(__file__).parent.parent / "week_06" / "gateway" / "static"


def run_manager(settings: Settings, scenario):
    """Run `scenario(manager)` and always reap the child, even if the scenario fails."""

    async def main():
        m = ProcessManager(settings)
        try:
            return await scenario(m)
        finally:
            await m.shutdown()
            assert all(p.returncode is not None for p in m.spawned), "leaked child"

    return asyncio.run(main())


# -- lifecycle --------------------------------------------------------------------------


def test_concurrent_waiters_share_one_start(tmp_path):
    async def scenario(m):
        results = await asyncio.gather(*[m.ensure_ready() for _ in range(5)])
        assert m.spawns == 1
        assert {g for g, _ in results} == {1}
        assert all(ms >= 250 for _, ms in results)
        assert (await m.ensure_ready())[1] == 0

    run_manager(make_settings(tmp_path, {"startup_delay": 0.4}), scenario)


def test_cancelled_first_waiter_does_not_abort_the_shared_start(tmp_path):
    async def scenario(m):
        first = asyncio.create_task(m.ensure_ready())
        await asyncio.sleep(0.05)
        second = asyncio.create_task(m.ensure_ready())
        await asyncio.sleep(0.05)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        gen, _ = await second
        assert gen == 1 and m.state == "ready" and m.spawns == 1

    run_manager(make_settings(tmp_path, {"startup_delay": 0.5}), scenario)


def test_startup_timeout_reaps_that_child_and_errors_all_waiters(tmp_path):
    async def scenario(m):
        outcomes = await asyncio.gather(m.ensure_ready(), m.ensure_ready(), return_exceptions=True)
        assert all(isinstance(o, ApiError) and o.code == "model_start_timeout" for o in outcomes)
        assert m.spawns == 1
        assert m.state == "failed"
        assert m.spawned[0].returncode is not None

    run_manager(make_settings(tmp_path, {"never_healthy": True}, start_timeout=0.5), scenario)


def test_child_exit_during_start_is_reported_with_code(tmp_path):
    settings = make_settings(tmp_path)
    settings.llama_argv_override = [settings.llama_argv_override[0], "-c", "raise SystemExit(7)"]

    async def scenario(m):
        with pytest.raises(ApiError) as ei:
            await m.ensure_ready()
        assert ei.value.code == "model_start_failed"
        assert m.state == "failed"

    run_manager(settings, scenario)


def test_spawn_failure_is_a_clean_error(tmp_path):
    settings = make_settings(tmp_path, llama_argv_override=["/definitely/not/here"])

    async def scenario(m):
        with pytest.raises(ApiError) as ei:
            await m.ensure_ready()
        assert ei.value.code == "model_start_failed"
        assert m.state == "failed"

    run_manager(settings, scenario)


def test_stderr_flood_before_health_is_drained_and_tail_is_bounded(tmp_path):
    async def scenario(m):
        await m.ensure_ready()  # would hang forever if the pipe were not read
        assert 0 < len(m.tail) <= 200
        assert "fake-llama listening" in m.tail_text(5)
        assert all(len(line) <= 500 for line in m.tail)

    run_manager(make_settings(tmp_path, {"stderr_bytes": 400_000}), scenario)


def test_request_during_stopping_waits_for_teardown_then_starts_fresh(tmp_path):
    async def scenario(m):
        gen, _ = await m.ensure_ready()
        stopper = asyncio.create_task(m.kill_current(gen))
        await asyncio.sleep(0)
        assert m.state == "stopping"
        gen2, cold = await m.ensure_ready()
        await stopper
        assert gen2 == gen + 1 and m.state == "ready" and m.spawns == 2
        assert m.spawned[0].returncode is not None

    run_manager(make_settings(tmp_path), scenario)


def test_stale_generation_cannot_stop_the_new_process(tmp_path):
    async def scenario(m):
        old, _ = await m.ensure_ready()
        await m.kill_current(old)
        new, _ = await m.ensure_ready()
        await m.kill_current(old)  # late event of the old process
        assert m.state == "ready" and m.gen == new

    run_manager(make_settings(tmp_path), scenario)


def test_crash_marks_failed_and_next_request_restarts(tmp_path):
    async def scenario(m):
        await m.ensure_ready()
        m.proc.kill()
        for _ in range(100):
            if m.state == "failed":
                break
            await asyncio.sleep(0.02)
        assert m.state == "failed" and m.last_exit is not None
        await m.ensure_ready()
        assert m.state == "ready" and m.spawns == 2

    run_manager(make_settings(tmp_path), scenario)


def test_restart_throttle_after_repeated_crashes(tmp_path):
    async def crash(m):
        m.proc.kill()
        for _ in range(100):
            if m.state == "failed":
                return
            await asyncio.sleep(0.02)
        raise AssertionError("no failed state")

    async def scenario(m):
        await m.ensure_ready()
        for _ in range(2):  # two failure-triggered restarts allowed
            await crash(m)
            await m.ensure_ready()
        await crash(m)
        with pytest.raises(ApiError) as ei:
            await m.ensure_ready()
        assert ei.value.code == "restart_throttled" and ei.value.retry_after == 60
        assert m.state == "failed"

    run_manager(make_settings(tmp_path, restart_max=2), scenario)


def test_idle_watchdog_kills_only_when_nothing_is_busy(tmp_path):
    busy = {"v": True}

    async def scenario(m):
        m.busy = lambda: busy["v"]
        await m.ensure_ready()
        m.start_watchdog()
        await asyncio.sleep(0.6)
        assert m.state == "ready"  # idle elapsed, but a request is in the line
        busy["v"] = False
        for _ in range(100):
            if m.state == "stopped":
                break
            await asyncio.sleep(0.03)
        assert m.state == "stopped"
        assert m.spawned[0].returncode is not None

    run_manager(make_settings(tmp_path, idle_seconds=0.3), scenario)


def test_note_finished_extends_idle(tmp_path):
    async def scenario(m):
        await m.ensure_ready()
        m.note_finished()
        assert 0 < m.idle_remaining() <= 60

    run_manager(make_settings(tmp_path), scenario)


def test_mem_available_parser(tmp_path):
    f = tmp_path / "mi"
    f.write_text("MemTotal: 1 kB\nMemAvailable:  2048 kB\n")
    assert read_mem_available_mb(str(f)) == 2.0
    assert read_mem_available_mb(str(tmp_path / "missing")) is None
    f.write_text("MemTotal: 1 kB\n")
    assert read_mem_available_mb(str(f)) is None


def test_child_env_is_built_from_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_THREADS", "99")
    monkeypatch.setenv("SECRET_TOKEN", "leak")
    env = Settings().child_env()
    assert "LLAMA_ARG_THREADS" not in env and "SECRET_TOKEN" not in env
    assert env["LD_LIBRARY_PATH"] == "/opt/advent-llm/lib"
    assert {"PATH", "HOME"} <= set(env)


def test_default_argv_matches_the_launch_contract():
    argv = Settings(upstream_port=8791).llama_argv()
    joined = " ".join(argv)
    for part in ("--host 127.0.0.1", "--port 8791", "-c 2048", "-np 1", "-t 1", "-n 512"):
        assert part in joined
    assert "--no-webui" in argv and "--api-key" not in argv


# -- keys / buckets ---------------------------------------------------------------------


def test_keystore_matches_by_hash_only(tmp_path):
    recs = [KeyRecord("a", hash_key("k-a"), 1, 1), KeyRecord("b", hash_key("k-b"), 1, 1)]
    ks = KeyStore(recs)
    assert ks.authenticate("k-b").name == "b"
    assert ks.authenticate("k-c") is None
    assert ks.authenticate("") is None
    path = tmp_path / "keys.json"
    path.write_text(json.dumps([{"name": "a", "sha256": hash_key("k-a"), "rpm": 30, "burst": 6}]))
    assert load_keys(path) == [KeyRecord("a", hash_key("k-a"), 30.0, 6)]
    assert "k-a" not in path.read_text()


def test_keys_file_rejects_bad_digest(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps([{"name": "a", "sha256": "abc", "rpm": 1, "burst": 1}]))
    with pytest.raises(ValueError, match="64 hex"):
        load_keys(path)


def test_gateway_source_uses_compare_digest():
    src = (STATIC.parent / "keys.py").read_text(encoding="utf-8")
    assert "hmac.compare_digest" in src


def test_token_bucket_with_fake_clock():
    t = [0.0]
    b = TokenBucket(rpm=3, burst=2, clock=lambda: t[0])
    assert b.take() == (True, 0)
    assert b.take() == (True, 0)
    assert b.take() == (False, 20)
    t[0] = 10
    assert b.take() == (False, 10)
    t[0] = 20
    assert b.take() == (True, 0)
    t[0] = 10_000
    assert b.peek() == 2.0  # capped at burst


def test_sliding_window_budget():
    t = [0.0]
    w = SlidingWindow(2, 60.0, clock=lambda: t[0])
    assert w.take() == (True, 0) and w.take() == (True, 0)
    assert w.take() == (False, 60)
    t[0] = 59
    assert w.take() == (False, 1)
    t[0] = 60
    assert w.take() == (True, 0)


# -- validation -------------------------------------------------------------------------

S = Settings()


def _parse(obj):
    return parse_chat_request(json.dumps(obj).encode(), S)


def _msgs(n=1):
    return [{"role": "user", "content": "hi"}] * n


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"messages": []}, "invalid_messages"),
        ({"messages": _msgs(33)}, "invalid_messages"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "invalid_messages"),
        ({"messages": [{"role": "user", "content": 5}]}, "invalid_messages"),
        ({"messages": [{"role": "user", "content": "x", "name": "n"}]}, "invalid_messages"),
        ({"messages": _msgs(), "temperature": 1.6}, "invalid_temperature"),
        ({"messages": _msgs(), "temperature": True}, "invalid_temperature"),
        ({"messages": _msgs(), "top_p": 1.1}, "invalid_top_p"),
        ({"messages": _msgs(), "stop": ["a"] * 5}, "invalid_stop"),
        ({"messages": _msgs(), "stop": "x" * 33}, "invalid_stop"),
        ({"messages": _msgs(), "seed": True}, "invalid_seed"),
        ({"messages": _msgs(), "seed": 1.5}, "invalid_seed"),
        ({"messages": _msgs(), "stream": "yes"}, "invalid_stream"),
        ({"messages": _msgs(), "tools": []}, "unsupported_field"),
    ],
)
def test_validation_rejects(body, code):
    with pytest.raises(ApiError) as ei:
        _parse(body)
    assert ei.value.status == 400 and ei.value.code == code


@pytest.mark.parametrize("raw", [b"{", b"[]", b"5", b"\xff\xfe"])
def test_validation_rejects_non_object_json(raw):
    with pytest.raises(ApiError) as ei:
        parse_chat_request(raw, S)
    assert ei.value.code == "invalid_json"


def test_validation_accepts_edges_and_builds_payload():
    req = _parse({
        "messages": _msgs(32), "max_tokens": 1, "temperature": 0, "top_p": 0,
        "stop": ["a"] * 4, "seed": -3, "stream": True, "model": "x",
        "stream_options": {"include_usage": False},
    })  # fmt: skip
    p = req.upstream_payload()
    assert p["stream_options"] == {"include_usage": True}
    assert p["max_tokens"] == 1 and p["temperature"] == 0.0 and p["seed"] == -3
    assert set(p) == {"messages", "max_tokens", "temperature", "stream", "top_p", "stop",
                      "seed", "stream_options"}  # fmt: skip
    plain = _parse({"messages": _msgs()}).upstream_payload()
    assert "stream_options" not in plain and plain["temperature"] == 0.7


# -- settings ---------------------------------------------------------------------------


def test_settings_precedence_file_then_env(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"idle_seconds": 30, "queue_max": 4}))
    s = load_settings(str(f), env={"ADVENT_GATEWAY_QUEUE_MAX": "1"})
    assert (s.idle_seconds, s.queue_max, s.n_ctx) == (30.0, 1, 2048)


def test_settings_reject_unknown_key(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"idle_second": 30}))
    with pytest.raises(ValueError, match="unknown settings"):
        load_settings(str(f), env={})


def test_spec_default_limits():
    s = Settings()
    assert (s.idle_seconds, s.queue_max, s.n_ctx) == (60.0, 2, 2048)
    assert (s.default_max_tokens, s.max_tokens_ceiling) == (256, 512)
    assert (s.queue_timeout, s.gen_deadline, s.start_timeout) == (150.0, 150.0, 30.0)
    assert (s.mem_available_min_mb, s.global_per_minute, s.body_max_bytes) == (500, 20, 65536)


# -- web page ---------------------------------------------------------------------------


def test_chat_js_never_renders_model_text_as_html():
    js = (STATIC / "chat.js").read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(",
                   "new Function", "srcdoc"):  # fmt: skip
        assert banned not in js, banned
    assert "textContent" in js or "nodeValue" in js
    assert "stream: true" in js  # TextDecoder keeps split multibyte characters
    assert "removeItem" in js  # 'forget key' exists
    assert re.search(r"if \(rememberEl\.checked", js)  # key stored only on opt-in


def test_html_has_no_inline_script_or_handlers():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.findall(r"<script\b[^>]*>", html) == ['<script src="/chat.js" defer>']
    assert not re.search(r"\son\w+\s*=", html)
    assert "<style" not in html
    assert 'name="viewport"' in html and "remember" in html and "forget" in html


def test_stats_counts_exact_codes_alongside_buckets():
    from week_06.gateway.app import Stats

    s = Stats()
    for code in (200, 200, 404, 404, 404, 499, 503, 401):
        s.count_status(code)
    assert dict(s.by_code) == {"200": 2, "404": 3, "499": 1, "503": 1, "401": 1}
    snap = s.snapshot()
    assert snap["200"] == 2 and snap["401"] == 1 and snap["503"] == 1
    assert snap["other"] == 4  # 404 x3 + 499, bucket keys unchanged
