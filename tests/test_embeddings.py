from __future__ import annotations

import json

import numpy as np
import pytest

import advent_core.embeddings as embeddings_module
from advent_core.embeddings import (
    BATCH_CHAR_BUDGET,
    BATCH_MAX_INPUTS,
    DEFAULT_EMBED_MODEL,
    _batches,
    embed_cost_usd,
    embed_texts,
)
from advent_core.errors import AdventError


class FakeAPIError(Exception):
    """Duck-typed like the SDK's own error: a status_code attribute + message text."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeUsage:
    def __init__(self, prompt_tokens: int | None) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = 0
        self.total_tokens = prompt_tokens


class FakeData:
    def __init__(self, embedding: list[float], index: int) -> None:
        self.embedding = embedding
        self.index = index


class FakeResponse:
    def __init__(self, data: list[FakeData], model: str, prompt_tokens: int | None) -> None:
        self.data = data
        self.model = model
        self.usage = FakeUsage(prompt_tokens)


class FakeEmbeddings:
    """Records every call's inputs; `handler(inputs)` decides the outcome."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[list[str]] = []

    def create(self, *, model: str, inputs: list[str]) -> FakeResponse:
        self.calls.append(list(inputs))
        return self.handler(inputs)


class FakeClient:
    def __init__(self, handler) -> None:
        self.embeddings = FakeEmbeddings(handler)


def _identity_response(inputs: list[str], *, dim: int = 3, tokens_per_input: int = 10):
    data = [FakeData([float(pos + 1)] * dim, index=pos) for pos in range(len(inputs))]
    tokens = tokens_per_input * len(inputs)
    return FakeResponse(data, model=DEFAULT_EMBED_MODEL, prompt_tokens=tokens)


def _reordered_one_hot_response(inputs: list[str]):
    """Returns data in REVERSE index order, as one-hot vectors.

    Proves two things at once: embed_texts must place each vector by
    data[i].index (not by call position), and one-hot vectors survive
    L2-normalization unchanged, so the assertion can be an exact match.
    """
    n = len(inputs)

    def one_hot(pos: int) -> list[float]:
        vec = [0.0] * n
        vec[pos] = 1.0
        return vec

    data = [FakeData(one_hot(pos), index=n - 1 - pos) for pos in range(n)]
    return FakeResponse(data, model=DEFAULT_EMBED_MODEL, prompt_tokens=10 * n)


# --- embed_cost_usd -----------------------------------------------------


def test_embed_cost_usd_known_model():
    assert embed_cost_usd("mistral-embed", 1_000_000) == pytest.approx(0.10)


def test_embed_cost_usd_unknown_model_or_missing_tokens_is_none():
    assert embed_cost_usd("some-other-model", 1000) is None
    assert embed_cost_usd("mistral-embed", None) is None


# --- batching -------------------------------------------------------------


def test_batches_split_by_char_budget():
    texts = ["a" * 20_000, "b" * 20_000, "c" * 20_000]
    batches = _batches(texts)
    # first two fill the budget exactly (40_000), the third starts a new batch
    assert batches == [(0, 2), (2, 3)]


def test_batches_split_by_max_input_count():
    texts = ["x"] * (BATCH_MAX_INPUTS + 36)
    batches = _batches(texts)
    assert batches == [(0, BATCH_MAX_INPUTS), (BATCH_MAX_INPUTS, BATCH_MAX_INPUTS + 36)]


def test_a_single_oversized_text_still_gets_its_own_batch():
    texts = ["a" * (BATCH_CHAR_BUDGET + 5000)]
    assert _batches(texts) == [(0, 1)]


# --- ordering / normalization ---------------------------------------------


def test_vectors_ordered_by_response_index_not_call_position():
    client = FakeClient(_reordered_one_hot_response)
    texts = ["one", "two", "three"]

    result = embed_texts(client, DEFAULT_EMBED_MODEL, texts, journal_path=None)

    expected = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    assert np.array_equal(result.vectors, expected)


def test_vectors_are_l2_normalized_float32():
    client = FakeClient(lambda inputs: _identity_response(inputs, dim=4))
    result = embed_texts(client, DEFAULT_EMBED_MODEL, ["a", "b"], journal_path=None)

    assert result.vectors.dtype == np.float32
    norms = np.linalg.norm(result.vectors, axis=1)
    assert norms == pytest.approx([1.0, 1.0])
    assert result.dim == 4


def test_prompt_tokens_summed_across_batches(monkeypatch):
    monkeypatch.setattr(embeddings_module, "BATCH_MAX_INPUTS", 1)
    client = FakeClient(lambda inputs: _identity_response(inputs, tokens_per_input=7))

    result = embed_texts(client, DEFAULT_EMBED_MODEL, ["a", "b", "c"], journal_path=None)

    assert len(client.embeddings.calls) == 3  # one text per batch, forced above
    assert result.prompt_tokens == 21
    assert result.requests == 3


def test_prompt_tokens_none_if_any_batch_lacked_usage(monkeypatch):
    monkeypatch.setattr(embeddings_module, "BATCH_MAX_INPUTS", 1)
    calls = {"n": 0}

    def handler(inputs):
        calls["n"] += 1
        tokens = None if calls["n"] == 2 else 5
        return FakeResponse(
            [FakeData([1.0, 0.0], index=0)], model=DEFAULT_EMBED_MODEL, prompt_tokens=tokens
        )

    client = FakeClient(handler)
    result = embed_texts(client, DEFAULT_EMBED_MODEL, ["a", "b", "c"], journal_path=None)

    assert result.prompt_tokens is None


# --- adaptive splitting on "Too many tokens overall" -----------------------


def test_too_many_tokens_overall_splits_batch_and_retries():
    def handler(inputs):
        if len(inputs) > 1:
            raise FakeAPIError("Too many tokens overall, split into more batches.", status_code=400)
        return _identity_response(inputs)

    client = FakeClient(handler)
    texts = ["t0", "t1", "t2", "t3"]

    result = embed_texts(client, DEFAULT_EMBED_MODEL, texts, journal_path=None)

    # 1 failed call at size 4, 2 failed at size 2, 4 successful at size 1
    assert len(client.embeddings.calls) == 7
    assert [len(c) for c in client.embeddings.calls if len(c) == 1] == [1, 1, 1, 1]
    assert result.requests == 7
    assert result.vectors.shape == (4, 3)


def test_single_input_batch_still_failing_raises_advent_error():
    def handler(inputs):
        raise FakeAPIError("Too many tokens overall, split into more batches.", status_code=400)

    client = FakeClient(handler)

    with pytest.raises(AdventError):
        embed_texts(client, DEFAULT_EMBED_MODEL, ["only one"], journal_path=None)


# --- the 8192-token-per-input 400 --------------------------------------


def test_input_exceeding_8192_tokens_names_the_chunk_and_does_not_rechunk():
    def handler(inputs):
        raise FakeAPIError(
            "Input id 1 has 9000 tokens, exceeding max 8192 tokens.", status_code=400
        )

    client = FakeClient(handler)
    ids = ["fixed:CLAUDE.md#0", "fixed:CLAUDE.md#1", "fixed:CLAUDE.md#2"]

    with pytest.raises(AdventError) as excinfo:
        embed_texts(client, DEFAULT_EMBED_MODEL, ["c0", "c1", "c2"], ids=ids, journal_path=None)

    assert "fixed:CLAUDE.md#1" in str(excinfo.value)
    assert "9000" in str(excinfo.value)
    # no re-chunk-and-retry: exactly the one failed call, nothing further
    assert len(client.embeddings.calls) == 1


def test_input_exceeding_8192_tokens_without_ids_names_the_index():
    def handler(inputs):
        raise FakeAPIError(
            "Input id 0 has 9000 tokens, exceeding max 8192 tokens.", status_code=400
        )

    client = FakeClient(handler)

    with pytest.raises(AdventError) as excinfo:
        embed_texts(client, DEFAULT_EMBED_MODEL, ["c0"], journal_path=None)

    assert "'0'" in str(excinfo.value)


# --- batch response validation (missing index becomes a zero vector) ------


def test_batch_response_missing_item_raises():
    def handler(inputs):
        return FakeResponse(
            [FakeData([1.0, 0.0], index=0)], model=DEFAULT_EMBED_MODEL, prompt_tokens=5
        )  # 1 item back for 2 inputs sent

    client = FakeClient(handler)
    with pytest.raises(AdventError):
        embed_texts(client, DEFAULT_EMBED_MODEL, ["a", "b"], journal_path=None)


def test_batch_response_duplicate_index_raises():
    def handler(inputs):
        data = [FakeData([1.0, 0.0], index=0), FakeData([0.0, 1.0], index=0)]
        return FakeResponse(data, model=DEFAULT_EMBED_MODEL, prompt_tokens=10)

    client = FakeClient(handler)
    with pytest.raises(AdventError):
        embed_texts(client, DEFAULT_EMBED_MODEL, ["a", "b"], journal_path=None)


def test_batch_response_out_of_range_index_raises():
    def handler(inputs):
        data = [FakeData([1.0, 0.0], index=0), FakeData([0.0, 1.0], index=5)]
        return FakeResponse(data, model=DEFAULT_EMBED_MODEL, prompt_tokens=10)

    client = FakeClient(handler)
    with pytest.raises(AdventError):
        embed_texts(client, DEFAULT_EMBED_MODEL, ["a", "b"], journal_path=None)


def test_batch_response_dimension_mismatch_across_batches_raises(monkeypatch):
    monkeypatch.setattr(embeddings_module, "BATCH_MAX_INPUTS", 1)
    calls = {"n": 0}

    def handler(inputs):
        calls["n"] += 1
        dim = 3 if calls["n"] == 1 else 4
        return FakeResponse(
            [FakeData([1.0] * dim, index=0)], model=DEFAULT_EMBED_MODEL, prompt_tokens=5
        )

    client = FakeClient(handler)
    with pytest.raises(AdventError):
        embed_texts(client, DEFAULT_EMBED_MODEL, ["a", "b"], journal_path=None)


def test_batch_response_nan_value_raises():
    def handler(inputs):
        data = [FakeData([1.0, float("nan")], index=0), FakeData([0.0, 1.0], index=1)]
        return FakeResponse(data, model=DEFAULT_EMBED_MODEL, prompt_tokens=10)

    client = FakeClient(handler)
    with pytest.raises(AdventError):
        embed_texts(client, DEFAULT_EMBED_MODEL, ["a", "b"], journal_path=None)


# --- other API errors delegate to errors.translate() -----------------------


def test_other_400_delegates_to_translate():
    def handler(inputs):
        raise FakeAPIError("Something else entirely.", status_code=400)

    client = FakeClient(handler)

    with pytest.raises(AdventError):
        embed_texts(client, DEFAULT_EMBED_MODEL, ["a"], journal_path=None)


# --- journal ----------------------------------------------------------------


def test_journal_logs_every_batch_without_leaking_text(tmp_path):
    journal_path = tmp_path / "calls.jsonl"
    secret_marker = "SECRET_CHUNK_TEXT_DO_NOT_LEAK"

    def handler(inputs):
        if len(inputs) > 1:
            raise FakeAPIError("Too many tokens overall, split into more batches.", status_code=400)
        return _identity_response(inputs)

    client = FakeClient(handler)
    texts = [f"{secret_marker}-{i}" for i in range(4)]

    embed_texts(
        client,
        DEFAULT_EMBED_MODEL,
        texts,
        week=5,
        day=21,
        journal_path=journal_path,
        journal_extra={"strategy": "fixed"},
    )

    raw = journal_path.read_text(encoding="utf-8")
    assert secret_marker not in raw

    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 7  # matches the 7 API calls made above
    assert {row["status"] for row in rows} == {"ok", "error"}
    assert all(row["kind"] == "embed" for row in rows)
    assert all(row["week"] == 5 and row["day"] == 21 for row in rows)
    assert all(row["strategy"] == "fixed" for row in rows)
    assert all("text" not in row and "messages" not in row for row in rows)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    assert sum(row["usage"]["prompt_tokens"] for row in ok_rows) > 0


class _NoIndexData:
    def __init__(self, embedding: list[float], index: int | None) -> None:
        self.embedding = embedding
        self.index = index


def _resp(data) -> FakeResponse:
    return FakeResponse(data, model=DEFAULT_EMBED_MODEL, prompt_tokens=10)


def test_batch_response_mixed_none_and_int_index_raises():
    def handler(inputs):
        return _resp([_NoIndexData([1.0, 0.0], None), _NoIndexData([0.0, 1.0], 0)])

    with pytest.raises(AdventError, match="смешаны"):
        embed_texts(FakeClient(handler), DEFAULT_EMBED_MODEL, ["a", "b"], journal_path=None)


def test_batch_response_all_none_index_is_positional():
    def handler(inputs):
        return _resp([_NoIndexData([1.0, 0.0], None), _NoIndexData([0.0, 1.0], None)])

    result = embed_texts(FakeClient(handler), DEFAULT_EMBED_MODEL, ["a", "b"], journal_path=None)
    assert result.vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_value_finite_in_float64_but_inf_in_float32_raises():
    def handler(inputs):
        return _resp([FakeData([1e40, 1.0], index=0)])

    with pytest.raises(AdventError, match="NaN/Inf"):
        embed_texts(FakeClient(handler), DEFAULT_EMBED_MODEL, ["a"], journal_path=None)


def test_all_zero_vector_raises():
    def handler(inputs):
        return _resp([FakeData([0.0, 0.0], index=0), FakeData([0.0, 1.0], index=1)])

    with pytest.raises(AdventError, match="нулевой вектор"):
        embed_texts(FakeClient(handler), DEFAULT_EMBED_MODEL, ["a", "b"], journal_path=None)


# --- on_request: every successful request is reported, even when a later one fails ---------


def test_on_request_survives_a_failed_second_batch(monkeypatch):
    monkeypatch.setattr(embeddings_module, "BATCH_MAX_INPUTS", 1)
    state = {"n": 0}

    def handler(inputs):
        state["n"] += 1
        if state["n"] == 2:
            raise FakeAPIError("boom", 500)
        return _identity_response(inputs, tokens_per_input=7)

    seen: list[tuple[int | None, int]] = []
    client = FakeClient(handler)
    with pytest.raises(AdventError):
        embed_texts(
            client,
            DEFAULT_EMBED_MODEL,
            ["a", "b"],
            journal_path=None,
            on_request=lambda tokens, ms: seen.append((tokens, ms)),
        )
    assert len(seen) == 1 and seen[0][0] == 7


def test_on_request_reports_the_good_half_when_the_split_retry_fails():
    state = {"n": 0}

    def handler(inputs):
        state["n"] += 1
        if state["n"] == 1:
            raise FakeAPIError("Too many tokens overall, split into more batches.", 400)
        if state["n"] == 3:
            raise FakeAPIError("boom", 500)
        return _identity_response(inputs, tokens_per_input=5)

    seen: list[int | None] = []
    with pytest.raises(AdventError):
        embed_texts(
            FakeClient(handler),
            DEFAULT_EMBED_MODEL,
            ["a", "b"],
            journal_path=None,
            on_request=lambda tokens, ms: seen.append(tokens),
        )
    assert seen == [5]
