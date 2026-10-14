"""Day 33 core: fixed support data, isolated index, and scoped RAG pipeline."""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from advent_core.embeddings import DEFAULT_EMBED_MODEL, EmbedResult
from advent_core.errors import AdventError
from advent_core.mcp_client import call_tool_once, connect_and_list
from advent_core.telemetry import CallResult
from adventassistant import cli, support_index, support_pipeline, support_store
from week_05.index import load_meta, load_runs

ROOT = Path(__file__).resolve().parents[1]


def _embed_result(n: int) -> EmbedResult:
    matrix = np.zeros((n, 1024), dtype=np.float32)
    matrix[:, 0] = 1.0
    return EmbedResult(matrix, DEFAULT_EMBED_MODEL, 42, 1, 2)


def _build_index(path: Path) -> Path:
    support_index.build_support_index(
        db_path=path, embed_fn=lambda texts: _embed_result(len(texts))
    )
    return path


def test_index_fingerprint_version_isolation_and_atomic_refusal(tmp_path: Path) -> None:
    path = _build_index(tmp_path / "support.sqlite3")
    meta = load_meta(path)
    assert meta["purpose"] == "w07d33"
    assert meta["support_version"] == support_index.CORPUS_VERSION
    assert meta["support_fingerprint"] == load_runs(path)["support"].corpus_rev
    assert support_index.require_support_index(path) == path
    before = path.read_bytes()

    foreign = tmp_path / "foreign.sqlite3"
    foreign.write_bytes(b"not a sqlite database")
    with pytest.raises(AdventError, match="Нельзя перезаписать"):
        support_index.build_support_index(db_path=foreign, embed_fn=lambda _: pytest.fail("embed"))
    assert foreign.read_bytes() == b"not a sqlite database"

    assert support_index.require_support_index(path) == path
    assert path.read_bytes() == before


def test_failed_rebuild_preserves_last_valid_index(tmp_path: Path) -> None:
    path = _build_index(tmp_path / "support.sqlite3")
    before = path.read_bytes()

    def fail(_texts):
        raise AdventError("fake embed failure")

    with pytest.raises(AdventError, match="fake embed failure"):
        support_index.build_support_index(db_path=path, embed_fn=fail)
    assert path.read_bytes() == before
    assert support_index.require_support_index(path) == path


def test_index_rejects_day31_day32_destinations_before_embedding(
    tmp_path: Path, monkeypatch
) -> None:
    old = tmp_path / "old.sqlite3"
    old.write_bytes(b"untouched")
    monkeypatch.setattr(support_index, "OLD_INDEX_PATHS", (old,))
    with pytest.raises(AdventError, match="другого дня"):
        support_index.build_support_index(db_path=old, embed_fn=lambda _: pytest.fail("embed"))
    assert old.read_bytes() == b"untouched"


def test_index_validates_fingerprint_and_stored_chunks(tmp_path: Path) -> None:
    path = _build_index(tmp_path / "support.sqlite3")
    from week_05.index import load_chunks, write_index

    chunks = load_chunks(path, "support")
    chunks[0] = replace(chunks[0], text="поддельный source")
    result = _embed_result(len(chunks))
    write_index(
        path,
        {"support": (chunks, result)},
        corpus_rev=load_runs(path)["support"].corpus_rev,
        corpus_files=1,
        corpus_chars=1,
        endpoint="cloud",
        meta={
            "purpose": "w07d33",
            "support_version": support_index.CORPUS_VERSION,
            "support_fingerprint": load_meta(path)["support_fingerprint"],
        },
    )
    with pytest.raises(AdventError, match="(?i)содержимое индекса"):
        support_index.require_support_index(path)


def test_index_rejects_stale_metadata_before_model_or_embed(tmp_path: Path) -> None:
    path = _build_index(tmp_path / "support.sqlite3")
    from week_05.index import load_chunks, write_index

    chunks = load_chunks(path, "support")
    write_index(
        path,
        {"support": (chunks, _embed_result(len(chunks)))},
        corpus_rev="stale",
        corpus_files=1,
        corpus_chars=1,
        endpoint="cloud",
        meta={"purpose": "w07d33", "support_version": "0", "support_fingerprint": "stale"},
    )
    with pytest.raises(AdventError, match="устарела"):
        support_index.require_support_index(path)


@pytest.mark.parametrize("blob", [b"short", np.full(1024, np.nan, dtype=np.float32).tobytes()])
def test_index_rejects_corrupt_or_nonfinite_vector_blobs(tmp_path: Path, blob: bytes) -> None:
    path = _build_index(tmp_path / "support.sqlite3")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE chunks SET embedding = ? WHERE strategy = 'support' "
            "AND chunk_id = (SELECT chunk_id FROM chunks WHERE strategy = 'support' LIMIT 1)",
            (blob,),
        )
    with pytest.raises(AdventError, match="(?i)вектор") as caught:
        support_index.require_support_index(path)
    assert "index-support" in caught.value.message


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(version=True),
        lambda data: data["users"][0].update(plan=[]),
        lambda data: data["users"][0].update(account_status={}),
        lambda data: data["users"][0].update(locale=[]),
        lambda data: data["tickets"][0].update(status=[]),
        lambda data: data["tickets"][0].update(category={}),
        lambda data: data["tickets"][0].update(user_id="missing"),
        lambda data: data["tickets"][0].update(extra="secret"),
        lambda data: data["users"].append(data["users"][0].copy()),
        lambda data: data["tickets"].append(data["tickets"][0].copy()),
    ],
)
def test_support_fixture_rejects_invalid_shapes_before_return(tmp_path: Path, mutate) -> None:
    data = json.loads(support_store.DATA_PATH.read_text(encoding="utf-8"))
    mutate(data)
    fixture = tmp_path / "bad.json"
    fixture.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AdventError, match="Файл поддержки повреждён"):
        support_store.load_support_data(fixture)


def test_support_fixture_rejects_oversize_file_with_bounded_read(tmp_path: Path) -> None:
    fixture = tmp_path / "large.json"
    fixture.write_bytes(b" " * (support_store.MAX_FILE_BYTES + 5_000))
    with pytest.raises(AdventError, match="размер больше"):
        support_store.load_support_data(fixture)


def test_support_fixture_rejects_deep_json_without_traceback(tmp_path: Path) -> None:
    fixture = tmp_path / "deep.json"
    depth = min(
        (support_store.MAX_FILE_BYTES - 1) // 2,
        sys.getrecursionlimit() + 1000,
    )
    assert depth > sys.getrecursionlimit()
    fixture.write_text("[" * depth + "0" + "]" * depth, encoding="utf-8")
    with pytest.raises(AdventError, match="Файл поддержки повреждён"):
        support_store.load_support_data(fixture)


def test_support_fixture_rejects_integer_token_over_python_limit(tmp_path: Path) -> None:
    fixture = tmp_path / "huge-int.json"
    fixture.write_text(
        '{"version":' + "1" * 5000 + ',"users":[],"tickets":[]}',
        encoding="utf-8",
    )
    with pytest.raises(AdventError, match="UTF-8 JSON"):
        support_store.load_support_data(fixture)


def test_scoped_context_and_owner_errors() -> None:
    first = support_store.get_support_context("demo-a", "AUTH-101")
    assert first["ticket"]["id"] == "AUTH-101"
    assert first["user"]["id"] == "demo-a"
    assert "demo-b" not in json.dumps(first, ensure_ascii=False)
    assert support_store.get_support_context("demo-a", None)["ticket"] is None
    with pytest.raises(AdventError, match="не принадлежит"):
        support_store.get_support_context("demo-b", "AUTH-101")
    with pytest.raises(AdventError, match="не найден"):
        support_store.get_support_context("demo-a", "AUTH-404")
    with pytest.raises(AdventError, match="не найден"):
        support_store.get_support_context("missing", None)


def test_real_mcp_stdio_lists_and_returns_only_scoped_context() -> None:
    command = [sys.executable, "-m", "adventassistant.support_server"]
    listing = connect_and_list(command, timeout=15, raw=False, sink=lambda _line: None, quiet=True)
    assert [item.name for item in listing.tools] == ["get_support_context"]
    outcome = call_tool_once(
        command,
        "get_support_context",
        {"user_id": "demo-b", "ticket_id": "AUTH-102"},
        timeout=15,
        quiet=True,
    )
    assert not outcome.is_error
    context = json.loads(outcome.text)
    assert context["user"]["id"] == "demo-b"
    assert context["ticket"]["id"] == "AUTH-102"
    assert "demo-a" not in outcome.text
    wrong = call_tool_once(
        command,
        "get_support_context",
        {"user_id": "demo-b", "ticket_id": "AUTH-101"},
        timeout=15,
        quiet=True,
    )
    assert wrong.is_error and "не принадлежит" in wrong.text


def _complete_for_pipeline(
    calls: list[dict],
    *,
    scores: str = '{"scores":[' + ",".join(f'{{"id":{i},"score":9}}' for i in range(1, 9)) + "]}",
):
    def complete(config, messages):
        calls.append({"model": config.model, "messages": messages})
        if len(calls) > 1:
            return CallResult(text="Запросите новую ссылку для входа.")
        return CallResult(text=scores)

    return complete


def test_pipeline_uses_ticket_facts_for_retrieval_and_adds_fixed_sources(
    tmp_path: Path, capsys
) -> None:
    path = _build_index(tmp_path / "support.sqlite3")
    calls: list[dict] = []
    embedded: list[str] = []
    context = {
        "user": {"id": "demo-a", "account_status": "active"},
        "ticket": {"id": "AUTH-101", "facts": {"cause": "expired_magic_link"}},
    }
    result = support_pipeline.run_support(
        "Почему не работает авторизация?",
        user_id="demo-a",
        ticket_id="AUTH-101",
        db_path=path,
        complete=_complete_for_pipeline(calls),
        embed=lambda text: embedded.append(text) or np.eye(1, 1024, dtype=np.float32)[0],
        mcp=lambda user, ticket: context,
    )
    assert "expired_magic_link" in embedded[0]
    assert "Источники:\n- adventassistant/support_faq.md — " in result
    assert "http://" not in result
    assert "Источники:" not in calls[-1]["messages"][1]["content"]
    assert calls[-1]["messages"][1]["content"].find("AUTH-101") >= 0
    assert "источник:" in capsys.readouterr().err


def test_pipeline_refuses_without_answer_call_when_rerank_is_low(tmp_path: Path) -> None:
    path = _build_index(tmp_path / "support.sqlite3")
    calls: list[dict] = []
    result = support_pipeline.run_support(
        "Где купить велосипед?",
        user_id="demo-a",
        ticket_id=None,
        db_path=path,
        complete=_complete_for_pipeline(calls, scores='{"scores":[{"id":1,"score":0}]}'),
        embed=lambda _text: np.eye(1, 1024, dtype=np.float32)[0],
        mcp=lambda _user, _ticket: {"user": {"id": "demo-a"}, "ticket": None},
    )
    assert result == support_pipeline.REFUSAL
    assert len(calls) == 1


def test_cli_registers_support_without_changing_existing_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["adventassistant", "--help"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "index-support" in out and "support" in out
    assert "index-review" in out and "review" in out


def test_support_cli_routes_answer_to_stdout(monkeypatch, capsys) -> None:
    seen: dict = {}

    def fake(question, **kwargs):
        seen.update(question=question, **kwargs)
        return "ответ поддержки"

    monkeypatch.setattr(cli, "run_support", fake)
    monkeypatch.setattr(
        sys,
        "argv",
        ["adventassistant", "support", "вопрос", "--user", "demo-a", "--ticket", "AUTH-101"],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == "ответ поддержки\n"
    assert captured.err == ""
    assert seen["question"] == "вопрос"
    assert seen["user_id"] == "demo-a" and seen["ticket_id"] == "AUTH-101"
    assert seen["complete"] is cli.chat_core.complete
