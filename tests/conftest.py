"""Suite-wide isolation: the offline guard patches httpx, so it must not outlive a test.

Teardown also fails a test that made a cloud attempt under the guard without opting in
(`@pytest.mark.allow_cloud_attempts`): a blocked request the test never looked at is
exactly how a leak hides behind a green run.
"""

from __future__ import annotations

import pytest

from advent_core import offline


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_cloud_attempts: the test deliberately triggers (blocked) cloud attempts "
        "under the offline guard",
    )


@pytest.fixture(autouse=True)
def _offline_off(request, monkeypatch):
    monkeypatch.delenv("ADVENT_OFFLINE", raising=False)
    offline.disable()
    offline.reset_counters()
    yield
    attempted = offline.counters().attempted["cloud"]
    offline.disable()
    offline.reset_counters()
    if attempted and request.node.get_closest_marker("allow_cloud_attempts") is None:
        pytest.fail(
            f"{attempted} unexpected cloud attempt(s) under the offline guard; "
            "assert them in the test and mark it @pytest.mark.allow_cloud_attempts",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _real_journal_untouched(request, monkeypatch, tmp_path_factory):
    """Redirect journal's default dir to a temp one, then fail a test that still reaches
    the real logs/calls.jsonl (the user's live journal) through an explicit path."""
    from advent_core import journal
    from advent_core.config import LOG_DIR

    monkeypatch.setattr(journal, "LOG_DIR", tmp_path_factory.mktemp("journal"))

    real = LOG_DIR / "calls.jsonl"

    def stamp():
        try:
            st = real.stat()
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    before = stamp()
    yield
    after = stamp()
    if after != before:
        pytest.fail(
            f"{request.node.nodeid} wrote to the real journal {real.name}; "
            "point log_path/path at tmp_path or monkeypatch journal.LOG_DIR",
            pytrace=False,
        )
