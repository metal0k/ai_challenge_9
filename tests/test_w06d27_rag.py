"""Day 27: local RAG — embedding table, run provenance, chunk cap, probe, retriever seams."""

from __future__ import annotations

import dataclasses
import json
import re
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from dotenv import load_dotenv
from typer.testing import CliRunner

from advent_core import config as config_module
from advent_core import embeddings as emb
from advent_core import journal as journal_module
from advent_core import offline
from advent_core import openai_compat as oc
from advent_core.config import Config
from advent_core.errors import AdventError
from advent_core.params import GenerationParams
from advent_core.rag import RagSettings
from advent_core.telemetry import CallResult, Usage
from week_05 import cli as rag_cli_module
from week_05 import index as ix
from week_05 import rag
from week_05.chunking import Chunk, Document, chunk_corpus, chunk_structure

NOMIC = emb.NOMIC_EMBED_MODEL
BGE = emb.BGE_M3_EMBED_MODEL
LOCAL_CHUNK_CAP = emb.EMBED_MODELS[NOMIC].chunk_cap
DIM = 64


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(journal_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.delenv("ADVENT_LOCAL_URL", raising=False)


def _vec_of(text: str) -> np.ndarray:
    """Deterministic fake 'embedding': bag of a few letters, L2-normalised."""
    v = np.zeros(DIM, dtype=np.float32)
    for ch in text:
        v[ord(ch) % DIM] += 1.0
    if not v.any():
        v[0] = 1.0
    return v / np.linalg.norm(v)


def _fake_embed(calls: list | None = None, *, window: int | None = None):
    """Stand-in for openai_compat.embed; `window` mimics silent truncation by length."""

    def fake(url, model, texts, *, expected_dim=None, on_batch=None, **kw):
        if calls is not None:
            calls.append({"url": url, "model": model, "texts": list(texts), "dim": expected_dim})
        used = [t[:window] if window else t for t in texts]
        vectors = np.stack([_vec_of(t) for t in used])
        if on_batch:
            on_batch(len(texts), len(texts))
        return emb.EmbedResult(
            vectors=vectors, model=model, prompt_tokens=0, requests=1, latency_ms=1
        )

    return fake


def _chunk(text, *, strategy="structure", source="a.md", n=0):
    return Chunk(
        chunk_id=f"{strategy}:{source}#{n}",
        strategy=strategy,
        source=source,
        title=source,
        section="S",
        ordinal=n,
        char_start=0,
        char_end=len(text),
        line_start=1,
        line_end=1,
        text=text,
    )


def _result(chunks, *, model=NOMIC):
    vectors = np.stack([_vec_of(c.text) for c in chunks])
    return emb.EmbedResult(vectors=vectors, model=model, prompt_tokens=0, requests=1, latency_ms=1)


def _write(db, strategy="structure", *, endpoint="cloud", texts=("alpha beta",), **kw):
    chunks = [_chunk(t, strategy=strategy, n=i) for i, t in enumerate(texts)]
    ix.write_index(
        db,
        {strategy: (chunks, _result(chunks))},
        corpus_rev="rev1",
        corpus_files=1,
        corpus_chars=10,
        endpoint=endpoint,
        **kw,
    )
    return chunks


# --- embedding table ---------------------------------------------------------


def test_embed_table_has_the_nomic_card_values():
    spec = emb.EMBED_MODELS[NOMIC]
    assert (spec.dim, spec.max_tokens, spec.local) == (768, 2048, True)
    assert spec.doc_prefix == "search_document: "
    assert spec.query_prefix == "search_query: "
    cloud = emb.EMBED_MODELS["mistral-embed"]
    assert (cloud.dim, cloud.local, cloud.doc_prefix) == (1024, False, "")


def test_local_embedding_costs_zero_even_without_tokens():
    assert emb.embed_cost_usd(NOMIC, 0) == 0.0
    assert emb.embed_cost_usd(NOMIC, None) == 0.0
    assert emb.embed_cost_usd("mistral-embed", 1_000_000) == pytest.approx(0.1)


def test_embed_local_adds_doc_and_query_prefixes_and_checks_dim(monkeypatch):
    calls: list = []
    monkeypatch.setattr(oc, "embed", _fake_embed(calls))
    emb.embed_local("http://127.0.0.1:1234", NOMIC, ["x", "y"], kind="doc")
    emb.embed_local(None, NOMIC, ["q"], kind="query")
    assert calls[0]["texts"] == ["search_document: x", "search_document: y"]
    assert calls[1]["texts"] == ["search_query: q"]
    assert calls[0]["dim"] == 768
    assert calls[0]["url"] == "http://127.0.0.1:1234"


def test_embed_local_explicit_prefix_wins_and_bad_kind_is_rejected(monkeypatch):
    calls: list = []
    monkeypatch.setattr(oc, "embed", _fake_embed(calls))
    emb.embed_local(None, NOMIC, ["q"], kind="query", prefix="clustering: ")
    assert calls[0]["texts"] == ["clustering: q"]
    with pytest.raises(ValueError):
        emb.embed_local(None, NOMIC, ["q"], kind="other")


def test_embed_local_journals_an_error_row_and_reraises(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AdventError("сервер недоступен")

    monkeypatch.setattr(oc, "embed", boom)
    path = tmp_path / "j.jsonl"
    with pytest.raises(AdventError, match="недоступен"):
        emb.embed_local(None, NOMIC, ["q"], journal_path=path)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "error"
    assert row["endpoint"] == "local"


# --- run provenance and migration --------------------------------------------


def test_write_index_stores_endpoint_prefixes_and_meta(tmp_path):
    db = tmp_path / "i.sqlite3"
    _write(
        db,
        endpoint="local",
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
        meta={"truncated_by_model.structure": "0/3"},
    )
    run = ix.load_runs(db)["structure"]
    assert (run.endpoint, run.doc_prefix, run.query_prefix) == (
        "local",
        "search_document: ",
        "search_query: ",
    )
    assert ix.load_meta(db)["truncated_by_model.structure"] == "0/3"


def _old_schema_db(path):
    """A pre-day-27 index: `runs` without the provenance columns."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, strategy TEXT NOT NULL,
          source TEXT NOT NULL, title TEXT NOT NULL, section TEXT NOT NULL,
          ordinal INTEGER NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
          line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, n_chars INTEGER NOT NULL,
          text TEXT NOT NULL, embedding BLOB NOT NULL);
        CREATE TABLE runs (strategy TEXT PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL,
          n_chunks INTEGER NOT NULL, prompt_tokens INTEGER, requests INTEGER NOT NULL,
          latency_ms INTEGER NOT NULL, cost_usd REAL, corpus_rev TEXT NOT NULL,
          corpus_files INTEGER NOT NULL, corpus_chars INTEGER NOT NULL, created_at TEXT NOT NULL);
        INSERT INTO runs VALUES ('fixed','mistral-embed',1024,1,5,1,10,0.0,'r',1,1,'t');
        """
    )
    conn.commit()
    conn.close()


def test_old_db_without_columns_reads_as_cloud_without_prefixes(tmp_path):
    db = tmp_path / "old.sqlite3"
    _old_schema_db(db)
    run = ix.load_runs(db)["fixed"]
    assert (run.endpoint, run.doc_prefix, run.query_prefix) == ("cloud", "", "")


def test_old_db_is_migrated_on_the_next_cloud_write(tmp_path):
    db = tmp_path / "old.sqlite3"
    _old_schema_db(db)
    _write(db, "structure", endpoint="cloud")
    runs = ix.load_runs(db)
    assert runs["fixed"].endpoint == "cloud"
    assert runs["structure"].endpoint == "cloud"
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(runs)")}
    assert {"endpoint", "doc_prefix", "query_prefix"} <= cols


def test_old_db_refuses_a_local_write_before_embedding(tmp_path):
    db = tmp_path / "old.sqlite3"
    _old_schema_db(db)
    with pytest.raises(AdventError, match="endpoint=cloud"):
        ix.check_endpoint_free(db, "local")


def test_partial_rebuild_cannot_mix_local_into_a_cloud_db(tmp_path):
    db = tmp_path / "i.sqlite3"
    _write(db, "fixed", endpoint="cloud")
    with pytest.raises(AdventError, match="смешивать"):
        ix.check_endpoint_free(db, "local")
    with pytest.raises(AdventError, match="смешивать"):
        _write(db, "structure", endpoint="local")
    assert set(ix.load_runs(db)) == {"fixed"}  # nothing was written


def test_partial_rebuild_cannot_mix_cloud_into_a_local_db(tmp_path):
    db = tmp_path / "i.sqlite3"
    _write(db, "fixed", endpoint="local")
    with pytest.raises(AdventError, match="endpoint=local"):
        ix.check_endpoint_free(db, "cloud")
    _write(db, "structure", endpoint="local")  # same endpoint: fine
    assert set(ix.load_runs(db)) == {"fixed", "structure"}


def test_endpoint_check_passes_on_a_missing_db(tmp_path):
    ix.check_endpoint_free(tmp_path / "nope.sqlite3", "local")


# --- chunk cap ---------------------------------------------------------------


def test_local_caps_are_the_measured_values_per_model():
    assert LOCAL_CHUNK_CAP == 1600
    assert emb.EMBED_MODELS[BGE].chunk_cap == 4000  # no truncation seen up to 4000 chars
    assert emb.LOCAL_EMBED_MODEL == BGE
    spec = emb.EMBED_MODELS[BGE]
    assert (spec.dim, spec.local, spec.doc_prefix, spec.query_prefix) == (1024, True, "", "")


def test_cap_rechunks_a_long_section_and_keeps_offsets_and_lines_exact():
    body = "\n".join(f"строка {i} " + "я" * 60 for i in range(80))
    text = "# Заголовок\n\n" + body + "\n"
    anchor = "строка 45 "
    assert len(text) > 3 * LOCAL_CHUNK_CAP
    chunks = chunk_corpus([Document("d.md", text)], "structure", max_chars=LOCAL_CHUNK_CAP)
    assert len(chunks) >= 3
    assert all(c.n_chars <= LOCAL_CHUNK_CAP for c in chunks)
    hit = [c for c in chunks if anchor in c.text]
    assert hit and hit[0].char_start > LOCAL_CHUNK_CAP  # quote sits past the first window
    for c in hit:
        assert text[c.char_start : c.char_end] == c.text
        assert c.line_start == text.count("\n", 0, c.char_start) + 1
        assert c.chunk_id == f"structure:d.md#{c.ordinal}"
        assert "[part" in c.section


def test_default_cap_is_unchanged_and_fixed_stays_inside_the_local_cap():
    doc = Document("d.md", "# T\n\n" + "слово " * 1500)
    wide = chunk_corpus([doc], "structure")
    assert max(c.n_chars for c in wide) > LOCAL_CHUNK_CAP  # 4000 cap by default
    assert wide == chunk_structure(doc)
    fixed = chunk_corpus([doc], "fixed", max_chars=LOCAL_CHUNK_CAP)
    assert max(c.n_chars for c in fixed) == 1000


# --- vector probe --------------------------------------------------------------


def test_probe_flags_chunks_the_model_truncates():
    window = 1300
    long_chunk = _chunk("ab" * 700 + "cdefgh" * 100, n=0)  # 2000 chars, tail past the window
    short_chunk = _chunk("a" * 400 + "bcdefgh" * 125, n=1)  # 1275 chars, fits
    chunks = [long_chunk, short_chunk, _chunk("tiny", n=2)]
    embed = _fake_embed(window=window)
    vectors = embed(None, NOMIC, [c.text for c in chunks]).vectors

    def embed_fn(batch):
        return embed(None, NOMIC, batch).vectors

    truncated, probed, ids = ix.truncation_probe(chunks, vectors, embed_fn)
    assert (truncated, probed) == (1, 2)  # "tiny" is below the probe threshold
    assert ids == [long_chunk.chunk_id]


def test_probe_with_nothing_long_enough_makes_no_call():
    def never(batch):
        raise AssertionError("must not embed")

    chunks = [_chunk("short")]
    assert ix.truncation_probe(chunks, np.zeros((1, DIM), np.float32), never) == (0, 0, [])


# --- check_index and mode ------------------------------------------------------


def test_check_index_refuses_a_cloud_index_in_offline_mode(tmp_path):
    db = tmp_path / "i.sqlite3"
    _write(db, endpoint="cloud")
    offline.enable()
    with pytest.raises(AdventError, match="облако") as info:
        rag.check_index(db, "structure")
    assert "adventrag index --local" in (info.value.hint or "")


def test_check_index_accepts_local_offline_and_either_when_online(tmp_path):
    local, cloud = tmp_path / "l.sqlite3", tmp_path / "c.sqlite3"
    _write(local, endpoint="local")
    _write(cloud, endpoint="cloud")
    assert rag.check_index(cloud, "structure").endpoint == "cloud"
    assert rag.check_index(local, "structure").endpoint == "local"
    offline.enable()
    assert rag.check_index(local, "structure").endpoint == "local"


def test_check_index_still_reports_a_missing_strategy(tmp_path):
    db = tmp_path / "i.sqlite3"
    _write(db, "fixed", endpoint="local")
    with pytest.raises(AdventError, match="нет стратегии 'structure'"):
        rag.check_index(db, "structure")


# --- retriever -----------------------------------------------------------------


def _local_db(tmp_path):
    db = tmp_path / "local.sqlite3"
    texts = ("aaa bbb ccc", "xyz xyz xyz", "mno pqr stu")
    _write(
        db,
        endpoint="local",
        texts=texts,
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
    )
    return db


def _no_cloud(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("cloud client must not be created")

    monkeypatch.setattr(rag, "mistral_client", forbidden)
    monkeypatch.setattr(rag, "embed_texts", forbidden)


def _call_result(text):
    return CallResult(
        text=text,
        model_requested="m",
        model_actual="m",
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        latency_ms=1,
        stream=False,
    )


def test_retriever_embeds_the_query_locally_with_the_run_prefix(monkeypatch, tmp_path):
    db = _local_db(tmp_path)
    calls: list = []
    monkeypatch.setattr(oc, "embed", _fake_embed(calls))
    _no_cloud(monkeypatch)
    aux = Config.resolve(offline=True, base_url="http://127.0.0.1:1234", model="ornith")
    retrieve = rag.make_retriever(db, aux_config=aux)
    ctx = retrieve("xyz", RagSettings("structure", 2, 2, False, False, 0.0))
    assert calls[0]["texts"] == ["search_query: xyz"]
    assert calls[0]["url"] == "http://127.0.0.1:1234"
    assert ctx.embed_model == NOMIC
    assert ctx.hits[0].text == "xyz xyz xyz"


def test_aux_calls_use_the_agent_config_not_the_cloud_aux_model(monkeypatch, tmp_path):
    db = _local_db(tmp_path)
    monkeypatch.setattr(oc, "embed", _fake_embed())
    _no_cloud(monkeypatch)
    seen: list = []

    def fake_complete(config, messages):
        seen.append(config)
        if len(seen) == 2:  # rewrite first, then rerank
            ids = [{"id": i, "score": 9} for i in (1, 2, 3)]
            return _call_result(json.dumps({"scores": ids}))
        return _call_result("xyz xyz")

    monkeypatch.setattr(rag.chat_core, "complete", fake_complete)
    aux = Config.resolve(offline=True, base_url="http://127.0.0.1:1234", model="ornith")
    aux = dataclasses.replace(aux, stream=True)
    retrieve = rag.make_retriever(db, aux_config=aux)
    ctx = retrieve("xyz", RagSettings("structure", 2, 3, True, True, 5.0))
    assert [c.model for c in seen] == ["ornith", "ornith"]
    assert all(c.base_url == "http://127.0.0.1:1234" for c in seen)
    assert all(c.api_key == "lm-studio-local" for c in seen)
    assert all(c.stream is False for c in seen)  # a stream=True agent config is forced off
    assert [c.params.temperature for c in seen] == [0, 0]
    assert rag.RAG_AUX_MODEL not in {c.model for c in seen}
    assert [c.params.format for c in seen] == [None, None]  # LM Studio refuses json_object
    # reasoning eats max_tokens: 200 / 2000 cloud budgets are raised to the local floor
    # default aux_reasoning is off, so the rerank is capped to what its JSON needs
    assert [c.params.max_tokens for c in seen] == [4096, rag.local_rerank_cap(3)]
    assert ctx.rewritten == "xyz xyz"
    # only the local rerank call carries the grammar-enforced schema, rewrite does not
    assert [c.params.response_format for c in seen] == [None, rag.LOCAL_RERANK_RESPONSE_FORMAT]


def test_local_rerank_request_body_has_the_literal_json_schema(monkeypatch):
    from advent_core import chat as chat_core_mod

    cfg = Config.resolve(offline=True, base_url="http://127.0.0.1:1234", model="ornith")
    cfg = dataclasses.replace(
        cfg,
        params=GenerationParams(
            temperature=0,
            max_tokens=544,
            format=None,
            response_format=rag.LOCAL_RERANK_RESPONSE_FORMAT,
        ),
    )
    payload, _, _, _ = chat_core_mod._payload(cfg, [{"role": "user", "content": "q"}])
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "rerank",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                            },
                            "required": ["id", "score"],
                        },
                    }
                },
                "required": ["scores"],
            },
        },
    }
    # an OpenAI-style json_schema survives the wire rewrite untouched
    assert oc._wire_payload(payload)["response_format"] == payload["response_format"]


def test_cloud_rerank_call_keeps_json_object_and_no_schema(monkeypatch):
    seen: list = []

    def fake_complete(config, messages):
        seen.append(config)
        return _call_result("{}")

    monkeypatch.setattr(rag.chat_core, "complete", fake_complete)
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    rag._aux_call(
        "p", command="rag_rerank", max_tokens=100, json_mode=True, day=1, rerank_schema=True
    )
    assert seen[0].params.format == "json"
    assert seen[0].params.response_format is None


def test_without_aux_config_the_cloud_aux_model_is_still_used(monkeypatch, tmp_path):
    db = tmp_path / "c.sqlite3"
    _write(db, endpoint="cloud", texts=("aaa", "xyz"))
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    seen: list = []

    def fake_complete(config, messages):
        seen.append(config)
        return _call_result("xyz")

    class Client:
        def __enter__(self):
            return object()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(rag.chat_core, "complete", fake_complete)
    monkeypatch.setattr(rag, "mistral_client", lambda config: Client())
    monkeypatch.setattr(
        rag,
        "embed_texts",
        lambda client, model, texts, **kw: emb.EmbedResult(
            vectors=np.stack([_vec_of(t) for t in texts]),
            model=model,
            prompt_tokens=3,
            requests=1,
            latency_ms=1,
        ),
    )
    retrieve = rag.make_retriever(db)
    retrieve("xyz", RagSettings("structure", 1, 1, True, False, 0.0))
    assert seen[0].model == rag.RAG_AUX_MODEL


NARRATED_RERANK = 'Вот оценки: {"scores": [{"id": 1, "score": 9}]} — готово.'


def test_local_retriever_tolerates_a_narrated_rerank_reply(monkeypatch, tmp_path):
    db = _local_db(tmp_path)
    monkeypatch.setattr(oc, "embed", _fake_embed())
    _no_cloud(monkeypatch)
    monkeypatch.setattr(
        rag.chat_core, "complete", lambda config, messages: _call_result(NARRATED_RERANK)
    )
    aux = Config.resolve(offline=True, base_url="http://127.0.0.1:1234", model="ornith")
    ctx = rag.make_retriever(db, aux_config=aux)(
        "xyz", RagSettings("structure", 2, 3, False, True, 5.0)
    )
    assert ctx.passed == 1


def test_cloud_retriever_keeps_the_strict_rerank_contract(monkeypatch, tmp_path):
    db = tmp_path / "c.sqlite3"
    _write(db, endpoint="cloud", texts=("aaa", "xyz"))
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setattr(
        rag.chat_core, "complete", lambda config, messages: _call_result(NARRATED_RERANK)
    )

    class Client:
        def __enter__(self):
            return object()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(rag, "mistral_client", lambda config: Client())
    monkeypatch.setattr(
        rag,
        "embed_texts",
        lambda client, model, texts, **kw: emb.EmbedResult(
            vectors=np.stack([_vec_of(t) for t in texts]),
            model=model,
            prompt_tokens=3,
            requests=1,
            latency_ms=1,
        ),
    )
    with pytest.raises(AdventError):
        rag.make_retriever(db)("xyz", RagSettings("structure", 1, 1, False, True, 5.0))


def test_offline_retriever_refuses_a_cloud_index_before_any_call(monkeypatch, tmp_path):
    db = tmp_path / "c.sqlite3"
    _write(db, endpoint="cloud")
    offline.enable()
    _no_cloud(monkeypatch)
    retrieve = rag.make_retriever(db)
    with pytest.raises(AdventError, match="облако"):
        retrieve("q", RagSettings("structure", 1, 1, False, False, 0.0))


# --- adventrag CLI -------------------------------------------------------------


@pytest.fixture
def small_corpus(monkeypatch):
    long_text = "# Заголовок\n\n" + "\n".join(f"Строка {i} " + "а" * 90 for i in range(40)) + "\n"
    docs = [Document("a.md", long_text), Document("b.md", "# B\n\nкороткий текст\n")]
    monkeypatch.setattr(
        rag_cli_module.corpus_module, "collect_corpus", lambda root, rev="HEAD": ("f" * 40, docs)
    )
    monkeypatch.setattr(
        oc, "ensure_ready", lambda model, url=None, require_state=False: SimpleNamespace(id=model)
    )
    return docs


def _no_cloud_cli(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("cloud client must not be created")

    monkeypatch.setattr(rag_cli_module, "mistral_client", forbidden)


def test_index_local_writes_a_local_run_with_prefixes_cap_and_probe(
    monkeypatch, tmp_path, small_corpus
):
    db = tmp_path / "local.sqlite3"
    calls: list = []
    monkeypatch.setattr(oc, "embed", _fake_embed(calls, window=1300))
    _no_cloud_cli(monkeypatch)
    rag_cli_module.index_command(
        strategy="structure", model=NOMIC, rev="HEAD", db=str(db), dry_run=False, local=True
    )
    run = ix.load_runs(db)["structure"]
    assert (run.endpoint, run.model, run.dim) == ("local", NOMIC, DIM)
    assert (run.doc_prefix, run.query_prefix) == ("search_document: ", "search_query: ")
    assert run.cost_usd == 0.0
    assert offline.is_enabled()
    chunks = ix.load_chunks(db, "structure")
    assert max(c.n_chars for c in chunks) <= LOCAL_CHUNK_CAP
    assert all(t.startswith("search_document: ") for t in calls[0]["texts"])
    meta = ix.load_meta(db)
    assert meta["chunk_cap"] == str(LOCAL_CHUNK_CAP)
    assert re.fullmatch(r"\d+/[1-9]\d*", meta["truncated_by_model.structure"])


def test_index_local_default_db_is_the_local_file(monkeypatch, tmp_path, small_corpus):
    target = tmp_path / "index.local.sqlite3"
    monkeypatch.setattr(ix, "LOCAL_DB", target)
    monkeypatch.setattr(oc, "embed", _fake_embed())
    rag_cli_module.index_command(
        strategy="fixed", model=None, rev="HEAD", db=None, dry_run=False, local=True
    )
    assert target.exists()


def test_index_local_into_a_cloud_db_is_refused_before_any_embedding(
    monkeypatch, tmp_path, small_corpus
):
    db = tmp_path / "cloud.sqlite3"
    _write(db, "fixed", endpoint="cloud")
    calls: list = []
    monkeypatch.setattr(oc, "embed", _fake_embed(calls))
    with pytest.raises(AdventError, match="смешивать"):
        rag_cli_module.index_command(
            strategy="structure", model=None, rev="HEAD", db=str(db), dry_run=False, local=True
        )
    assert calls == []


def test_index_local_refuses_a_non_loopback_url(monkeypatch, tmp_path, small_corpus):
    monkeypatch.setenv("ADVENT_LOCAL_URL", "http://example.com:1234")
    monkeypatch.setattr(oc, "embed", _fake_embed())
    with pytest.raises(config_module.ConfigError, match="loopback"):
        rag_cli_module.index_command(
            strategy="fixed",
            model=None,
            rev="HEAD",
            db=str(tmp_path / "x.sqlite3"),
            dry_run=False,
            local=True,
        )


def test_compare_local_embeds_questions_locally_and_prints_the_tables(
    monkeypatch, tmp_path, small_corpus
):
    db = tmp_path / "local.sqlite3"
    monkeypatch.setattr(oc, "embed", _fake_embed(window=1300))
    for strat in ("fixed", "structure"):
        rag_cli_module.index_command(
            strategy=strat, model=NOMIC, rev="HEAD", db=str(db), dry_run=False, local=True
        )
    offline.disable()
    calls: list = []
    monkeypatch.setattr(oc, "embed", _fake_embed(calls))
    _no_cloud_cli(monkeypatch)
    qpath = tmp_path / "q.json"
    qpath.write_text(
        json.dumps(
            [{"id": 1, "question": "Строка 3", "expected": [{"source": "b.md", "anchor": "текст"}]}]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_cli_module, "EVAL_QUESTIONS_PATH", qpath)
    result = CliRunner().invoke(rag_cli_module.app, ["compare", "--local", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert offline.is_enabled()
    assert calls[0]["texts"] == ["search_query: Строка 3"]
    assert "Качество поиска" in result.output
    assert result.output.count("MRR@5") == 1


def test_compare_local_refuses_a_cloud_index(monkeypatch, tmp_path, small_corpus):
    db = tmp_path / "c.sqlite3"
    _write(db, "fixed", endpoint="cloud")
    _write(db, "structure", endpoint="cloud")
    _no_cloud_cli(monkeypatch)
    with pytest.raises(AdventError, match="облако"):
        rag_cli_module.compare_command(db=str(db), local=True)


def test_main_turns_a_config_error_into_text_and_exit_2(monkeypatch, capsys):
    def boom():
        raise config_module.ConfigError("не loopback")

    monkeypatch.setattr(rag_cli_module, "app", boom)
    with pytest.raises(SystemExit) as info:
        rag_cli_module.main()
    assert info.value.code == 2
    assert "не loopback" in capsys.readouterr().err


@pytest.mark.parametrize("aux_reasoning, effort", [(False, "none"), (True, None)])
def test_aux_reasoning_off_sends_reasoning_effort_none_on_local_aux_calls(
    monkeypatch, tmp_path, aux_reasoning, effort
):
    db = _local_db(tmp_path)
    monkeypatch.setattr(oc, "embed", _fake_embed())
    _no_cloud(monkeypatch)
    seen: list = []

    def fake_complete(config, messages):
        seen.append(config)
        if len(seen) == 2:
            return _call_result(json.dumps({"scores": [{"id": i, "score": 9} for i in (1, 2, 3)]}))
        return _call_result("xyz xyz")

    monkeypatch.setattr(rag.chat_core, "complete", fake_complete)
    aux = Config.resolve(offline=True, base_url="http://127.0.0.1:1234", model="ornith")
    retrieve = rag.make_retriever(db, aux_config=aux)
    retrieve("xyz", RagSettings("structure", 2, 3, True, True, 5.0, aux_reasoning=aux_reasoning))
    assert len(seen) == 2  # rewrite and rerank both
    assert [c.params.reasoning_effort for c in seen] == [effort, effort]
    payload, _ = seen[0].params.as_payload(None)
    assert payload.get("reasoning_effort") == effort


def test_aux_reasoning_never_reaches_a_cloud_aux_call(monkeypatch):
    seen: list = []

    def fake_complete(config, messages):
        seen.append(config)
        return _call_result("xyz")

    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setattr(rag.chat_core, "complete", fake_complete)
    monkeypatch.setattr(rag, "log_call", lambda *a, **k: None)
    rag._aux_call(
        "p", command="rag_rewrite", max_tokens=10, json_mode=False, day=23, reasoning=False
    )
    assert seen[0].params.reasoning_effort is None


def test_rag_aux_reasoning_param_defaults_to_the_measured_default_and_is_a_bool():
    from advent_core.params import DEFAULT_RAG_AUX_REASONING, GenerationParams

    p = GenerationParams()
    p.apply_defaults("agent")
    assert p.rag_aux_reasoning is DEFAULT_RAG_AUX_REASONING
    p.set("rag_aux_reasoning", "off")
    assert p.rag_aux_reasoning is False
    p.set("rag_aux_reasoning", "on")
    assert p.rag_aux_reasoning is True


def test_parse_rerank_finds_the_json_inside_a_narrated_reply():
    raw = (
        "Рассмотрю фрагменты. Ключевой факт: {потолок} 1.5.\n\n```json\n"
        '{"scores": [{"id": 1, "score": 9}, {"id": 2, "score": 0}]}\n```\n\n'
        '**Комментарий**: оценка {"id": 3, "score": 7} здесь не в счёт.'
    )
    assert rag.parse_rerank(raw, 2, tolerant=True) == {1: 9.0, 2: 0.0}
    with pytest.raises(AdventError):
        rag.parse_rerank("просто текст без JSON {не json}", 2, tolerant=True)


def test_parse_rerank_cloud_stays_strict_about_narrated_replies():
    raw = 'Вот ответ: {"scores": [{"id": 1, "score": 9}]} — готово.'
    with pytest.raises(AdventError):
        rag.parse_rerank(raw, 1)  # default = cloud contract
    assert rag.parse_rerank(raw, 1, tolerant=True) == {1: 9.0}


def test_parse_rerank_tolerant_rejects_several_different_score_objects():
    raw = (
        'Пример формата: {"scores": [{"id": 1, "score": 9}]} '
        'Итог: {"scores": [{"id": 1, "score": 0}]}'
    )
    with pytest.raises(AdventError) as info:
        rag.parse_rerank(raw, 1, tolerant=True)
    assert "несколько" in info.value.message


def test_parse_rerank_tolerant_accepts_the_same_object_repeated():
    obj = '{"scores": [{"id": 1, "score": 4}]}'
    assert rag.parse_rerank(f"{obj} повторю: {obj}", 1, tolerant=True) == {1: 4.0}


# --- review finding 8: URL resolved after .env, canonical everywhere -----------------


def _env_file(monkeypatch, tmp_path, text):
    (tmp_path / ".env").write_text(text, encoding="utf-8")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    # the module fixture mutes load_env: put a real one back for these tests
    monkeypatch.setattr(
        config_module, "load_env", lambda: load_dotenv(config_module.PROJECT_ROOT / ".env")
    )
    # monkeypatch restores the variable that load_dotenv sets behind its back
    monkeypatch.setenv("ADVENT_LOCAL_URL", "placeholder")
    monkeypatch.delenv("ADVENT_LOCAL_URL")


def test_default_url_reads_advent_local_url_from_dot_env_on_the_first_call(monkeypatch, tmp_path):
    _env_file(monkeypatch, tmp_path, "ADVENT_LOCAL_URL=http://127.0.0.1:4321/v1\n")
    assert oc.default_url() == "http://127.0.0.1:4321"  # loaded AND normalized


def test_local_setup_uses_the_canonical_url_for_readiness(monkeypatch, tmp_path):
    _env_file(monkeypatch, tmp_path, "ADVENT_LOCAL_URL=http://127.0.0.1:4321/v1/\n")
    seen: list = []
    monkeypatch.setattr(
        oc,
        "ensure_ready",
        lambda model, url=None, require_state=False: seen.append((model, url)) or None,
    )

    url = rag_cli_module._local_setup("nomic")

    assert url == "http://127.0.0.1:4321" and seen == [("nomic", "http://127.0.0.1:4321")]


# --- review finding 14: `adventrag check`, the rehearsal's local-RAG gate ---------------


def _serve(monkeypatch, state="loaded"):
    monkeypatch.setattr(
        oc,
        "server_status",
        lambda url=None, **kw: [oc.LocalModel(NOMIC, "embeddings", state, 2048)],
    )


def test_check_passes_for_a_local_index_with_its_model_loaded(monkeypatch, tmp_path, capsys):
    db = _local_db(tmp_path)
    _serve(monkeypatch)

    run = rag_cli_module.check_local_rag(str(db))

    assert run.endpoint == "local" and run.model == NOMIC
    assert "локальный индекс готов" in capsys.readouterr().out
    assert offline.is_enabled()  # the check itself runs under the guard


def test_check_refuses_a_missing_index(monkeypatch, tmp_path):
    _serve(monkeypatch)
    with pytest.raises(AdventError):
        rag_cli_module.check_local_rag(str(tmp_path / "nope.sqlite3"))


def test_check_refuses_an_index_built_through_the_cloud(monkeypatch, tmp_path):
    db = tmp_path / "c.sqlite3"
    _write(db, endpoint="cloud")
    _serve(monkeypatch)
    with pytest.raises(AdventError, match="облако"):
        rag_cli_module.check_local_rag(str(db))


def test_check_refuses_a_missing_strategy(monkeypatch, tmp_path):
    db = _local_db(tmp_path)  # holds "structure" only
    _serve(monkeypatch)
    with pytest.raises(AdventError, match="fixed"):
        rag_cli_module.check_local_rag(str(db), "fixed")


@pytest.mark.parametrize("state", ["not-loaded", None])
def test_check_refuses_an_embedding_model_that_is_not_loaded(monkeypatch, tmp_path, state):
    db = _local_db(tmp_path)
    _serve(monkeypatch, state)
    with pytest.raises(AdventError) as info:
        rag_cli_module.check_local_rag(str(db))
    assert info.value.exit_code == 2


def test_check_command_exits_non_zero_on_a_miss(monkeypatch, tmp_path):
    _serve(monkeypatch)
    result = CliRunner().invoke(
        rag_cli_module.app, ["check", "--db", str(tmp_path / "nope.sqlite3")]
    )
    assert result.exit_code != 0


def _rerank_run(monkeypatch, tmp_path, replies, *, aux_reasoning, local=True):
    """Drive one rewrite+rerank retrieve with scripted replies; returns (configs, prompts)."""
    db = _local_db(tmp_path)
    monkeypatch.setattr(oc, "embed", _fake_embed())
    _no_cloud(monkeypatch)
    monkeypatch.setattr(rag, "log_call", lambda *a, **k: None)
    seen: list = []
    prompts: list = []
    queue = list(replies)

    def fake_complete(config, messages):
        seen.append(config)
        prompts.append(messages[0]["content"])
        return _call_result(queue.pop(0))

    monkeypatch.setattr(rag.chat_core, "complete", fake_complete)
    if local:
        aux = Config.resolve(offline=True, base_url="http://127.0.0.1:1234", model="ornith")
        retrieve = rag.make_retriever(db, aux_config=aux)
    else:
        monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
        retrieve = rag.make_retriever(db)
    settings = RagSettings("structure", 2, 3, True, True, 5.0, aux_reasoning=aux_reasoning)
    return seen, prompts, lambda: retrieve("xyz", settings)


GOOD = json.dumps({"scores": [{"id": i, "score": 9} for i in (1, 2, 3)]})


def test_local_rerank_prompt_repeats_the_json_shape_after_the_trailing_question():
    from advent_core.rag import RAG_RERANK_LOCAL_TAIL, RagHit, build_rerank_prompt

    hits = [RagHit("c", "a.md", "S", 0.5, "текст", rank=1)]
    local = build_rerank_prompt("q?", hits, local=True)
    cloud = build_rerank_prompt("q?", hits)
    assert local == cloud + "\n\n" + RAG_RERANK_LOCAL_TAIL.replace("{n}", "1")
    assert local.endswith("вокруг.")
    assert '{"scores": [{"id": 1, "score": 0}, ...]}' in local.rsplit("Вопрос", 1)[1]


def test_cloud_rerank_prompt_is_byte_identical_to_the_pre_day27_text():
    from advent_core.rag import RagHit, build_rerank_prompt

    hits = [RagHit("c", "a.md", "Раздел", 0.5, "текст", rank=1)]
    expected = (
        "Оцени, насколько каждый фрагмент помогает ответить на вопрос. "
        "Шкала 0–10: 10 — фрагмент содержит прямой ответ, 0 — не относится. "
        'Верни JSON {"scores": [{"id": <номер>, "score": <0-10>}, ...]} '
        "для всех 1 фрагментов без пропусков."
        "\n\nВопрос: q?\n\n[1] a.md — Раздел\nтекст\n\nВопрос: q?"
    )
    assert build_rerank_prompt("q?", hits) == expected


def test_local_rerank_reasoning_off_caps_max_tokens_reasoning_on_keeps_4096(monkeypatch, tmp_path):
    for reasoning, expected in ((False, rag.local_rerank_cap(3)), (True, 4096)):
        seen, _, go = _rerank_run(monkeypatch, tmp_path, ["xyz", GOOD], aux_reasoning=reasoning)
        go()
        assert seen[0].params.max_tokens == 4096  # rewrite untouched
        assert seen[1].params.max_tokens == expected
    assert rag.local_rerank_cap(3) == 512
    assert rag.local_rerank_cap(40) == 64 + 24 * 40


def test_local_rerank_retries_once_on_non_json_and_succeeds(monkeypatch, tmp_path):
    seen, prompts, go = _rerank_run(
        monkeypatch, tmp_path, ["xyz", "<tool_call> мусор", GOOD], aux_reasoning=False
    )
    ctx = go()
    assert len(seen) == 3
    assert prompts[1] == prompts[2]
    assert seen[1].params == seen[2].params
    assert ctx.hits


def test_local_rerank_fails_after_the_second_non_json_answer(monkeypatch, tmp_path):
    seen, _, go = _rerank_run(
        monkeypatch, tmp_path, ["xyz", "мусор", "опять мусор"], aux_reasoning=False
    )
    with pytest.raises(AdventError, match="не JSON"):
        go()
    assert len(seen) == 3


def test_local_rerank_retries_on_ambiguous_answer(monkeypatch, tmp_path):
    ambiguous = '{"scores": [{"id": 1, "score": 1}]} и ещё {"scores": [{"id": 1, "score": 9}]}'
    seen, _, go = _rerank_run(monkeypatch, tmp_path, ["xyz", ambiguous, GOOD], aux_reasoning=False)
    go()
    assert len(seen) == 3


def test_cloud_rerank_has_no_retry_no_tail_and_unchanged_params(monkeypatch, tmp_path):
    seen, prompts, go = _rerank_run(
        monkeypatch, tmp_path, ["xyz", "мусор"], aux_reasoning=False, local=False
    )
    with pytest.raises(AdventError, match="не JSON"):
        go()
    assert len(seen) == 2
    assert "ТОЛЬКО JSON" not in prompts[1]
    assert seen[1].params.max_tokens == rag.RERANK_MAX_TOKENS
    assert seen[1].params.reasoning_effort is None
