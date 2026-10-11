"""input_tokens call: ordering vs. the slot, logging, deadline, one retry on a dead socket."""

from __future__ import annotations

import logging
import time

from tests.w06d30_gateway_helpers import Live, chat_body, make_settings, raw_post, read_all

CHAT = "/v1/chat/completions"


def test_count_runs_after_admission_so_it_never_waits_behind_a_generation(tmp_path):
    # Child models "input_tokens queued behind the slot"; the gateway must never be caught by it.
    fake = {"count_block_while_processing": True}
    with Live(make_settings(tmp_path, fake, upstream_timeout=1.0)) as lv:
        first = raw_post(lv, chat_body("#chunks=6;delay=0.1#a", stream=True))
        lv.wait_for(lambda: lv.gw.admission.sem.locked(), what="first holds the slot")
        with lv.client() as c:
            r = c.post(CHAT, json=chat_body("b"))
        read_all(first)
        assert r.status_code == 200, r.text


def test_count_failure_is_logged_with_type_and_elapsed_without_content(tmp_path, caplog):
    secret = "СЕКРЕТНЫЙ-ТЕКСТ"
    fake = {"count_drop_first": 2}  # both attempts die: no retry can save it
    with (
        Live(make_settings(tmp_path, fake)) as lv,
        caplog.at_level(logging.WARNING),
        lv.client() as c,
    ):
        r = c.post(CHAT, json=chat_body(secret))
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "token_count_failed"
    rec = [m for m in caplog.messages if "token_count_failed" in m]
    assert rec, caplog.messages
    assert "RemoteProtocolError" in rec[0] and "elapsed_ms=" in rec[0]
    assert secret not in caplog.text


def test_count_is_bounded_by_the_request_deadline_not_the_fixed_client_timeout(tmp_path):
    fake = {"count_delay": 3.0}
    with Live(make_settings(tmp_path, fake, gen_deadline=0.6, upstream_timeout=10.0)) as lv:
        t0 = time.monotonic()
        with lv.client() as c:
            r = c.post(CHAT, json=chat_body("x"))
        took = time.monotonic() - t0
        assert r.status_code == 502, r.text
        assert r.json()["error"]["code"] == "token_count_failed"
        assert took < 2.0, took
        assert lv.backend()["total"] == 0  # a failed count never reached generation


def test_one_retry_on_a_dropped_connection(tmp_path):
    with Live(make_settings(tmp_path, {"count_drop_first": 1})) as lv:
        with lv.client() as c:
            r = c.post(CHAT, json=chat_body("x"))
        assert r.status_code == 200, r.text
        assert lv.backend()["count_calls"] == 2


def test_no_retry_on_timeout(tmp_path):
    fake = {"count_delay": 1.0}
    with Live(make_settings(tmp_path, fake, gen_deadline=0.4)) as lv:
        with lv.client() as c:
            r = c.post(CHAT, json=chat_body("x"))
        assert r.status_code == 502
        assert lv.backend()["count_calls"] == 1
