from __future__ import annotations

import io
import json
import re

import pytest
from rich.console import Console

from advent_core import config as config_module
from advent_core import console
from advent_core.errors import AdventError
from advent_core.rag import RagContext, RagHit
from advent_core.telemetry import CallResult, Usage
from week_05 import rag_cli
from week_05.chunking import Chunk

CHUNK_TEXT = "ТЕКСТ-ЧАНКА: окно 262144 токена, тег w01d01"


_ANSI = re.compile(r"\[[0-9;]*m")


def _plain(text: str) -> str:
    # FORCE_COLOR in the environment colours output even under no_color.
    return _ANSI.sub("", text)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", _plain(text)).strip()


def _chunk(text: str = CHUNK_TEXT, source: str = "CLAUDE.md") -> Chunk:
    return Chunk(
        chunk_id=f"structure:{source}#1",
        strategy="structure",
        source=source,
        title=source,
        section="Раздел",
        ordinal=1,
        char_start=0,
        char_end=len(text),
        line_start=1,
        line_end=1,
        text=text,
    )


class Fakes:
    def __init__(self) -> None:
        self.complete_calls: list[list[dict]] = []
        self.retrieve_calls: list[tuple[str, str, int]] = []
        self.check_calls: list[str] = []
        self.journal: list[dict] = []
        self.answer_plain = "не знаю"
        self.answer_rag = "окно 262 144, тег w01d01, см. CLAUDE.md"
        self.fail_on_call: int | None = None
        self.chunks = [_chunk()]

    def complete(self, config, messages, capabilities=None, **kwargs):
        self.complete_calls.append(list(messages))
        if self.fail_on_call == len(self.complete_calls):
            raise AdventError("сервер недоступен")
        has_chunk = any(CHUNK_TEXT in str(m.get("content", "")) for m in messages)
        return CallResult(
            text=self.answer_rag if has_chunk else self.answer_plain,
            usage=Usage(prompt_tokens=100 if has_chunk else 10, completion_tokens=5),
            latency_ms=1500,
            stream=False,
        )

    def retrieve(self, question, strategy, k):
        self.retrieve_calls.append((question, strategy, k))
        hit = RagHit(
            chunk_id="structure:CLAUDE.md#1",
            source="CLAUDE.md",
            section="Раздел",
            score=0.9,
            text=CHUNK_TEXT,
        )
        return RagContext(
            hits=(hit,),
            strategy=strategy,
            k=k,
            embed_model="mistral-embed",
            embed_tokens=7,
            corpus_rev="abc",
        )


@pytest.fixture
def fakes(monkeypatch):
    f = Fakes()
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setattr(rag_cli.chat_core, "complete", f.complete)
    monkeypatch.setattr(rag_cli.rag_module, "make_retriever", lambda db_path=None: f.retrieve)
    monkeypatch.setattr(
        rag_cli.rag_module, "check_index", lambda db_path, strategy: f.check_calls.append(strategy)
    )
    monkeypatch.setattr(rag_cli.index_module, "load_chunks", lambda db_path, strategy: f.chunks)
    monkeypatch.setattr(
        rag_cli,
        "log_call",
        lambda result, messages, *, week, day, extra=None: f.journal.append(
            {"week": week, "day": day, "extra": extra}
        ),
    )
    return f


@pytest.fixture
def out(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=80, no_color=True))
    return buffer


@pytest.fixture
def err(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(console, "err", Console(file=buffer, width=80, no_color=True))
    return buffer


def _questions(tmp_path, items):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


Q1 = {"id": 1, "question": "Какое окно?", "expect": [["262144"]], "sources": ["CLAUDE.md"]}
Q2 = {"id": 2, "question": "Какой тег?", "expect": [["w01d01"]], "sources": ["CLAUDE.md"]}


def test_ask_both_prints_plain_then_rag(fakes, out, err):
    rag_cli.ask_question("Какое окно?")
    text = _plain(out.getvalue())
    assert text.count("── без RAG ──") == 1
    assert text.count("── с RAG ──") == 1
    assert text.index("не знаю") < text.index("── с RAG ──") < text.index("окно 262 144")
    assert "  [1] 0.900  CLAUDE.md — Раздел" in text
    assert fakes.retrieve_calls == [("Какое окно?", "structure", 5)]
    assert len(fakes.complete_calls) == 2


def test_ask_rag_request_contains_chunk_and_plain_does_not(fakes, out, err):
    rag_cli.ask_question("Какое окно?")
    plain_last = fakes.complete_calls[0][-1]
    rag_last = fakes.complete_calls[1][-1]
    assert plain_last["content"] == "Какое окно?"
    assert CHUNK_TEXT in rag_last["content"]
    assert rag_last["content"].endswith("Вопрос: Какое окно?")


def test_ask_no_rag_never_touches_the_index(fakes, out, err):
    rag_cli.ask_question("Какое окно?", mode="no-rag")
    assert fakes.retrieve_calls == []
    assert fakes.check_calls == []
    assert "── с RAG ──" not in _plain(out.getvalue())
    assert "источники" not in _plain(out.getvalue())


def test_ask_rag_only(fakes, out, err):
    rag_cli.ask_question("Какое окно?", mode="rag", strategy="fixed", k=3)
    assert fakes.check_calls == ["fixed"]
    assert fakes.retrieve_calls == [("Какое окно?", "fixed", 3)]
    assert "── без RAG ──" not in _plain(out.getvalue())
    assert len(fakes.complete_calls) == 1


def test_ask_token_line_goes_to_stderr(fakes, out, err):
    rag_cli.ask_question("Какое окно?")
    assert "prompt" not in _plain(out.getvalue())
    flat_err = _flat(_plain(err.getvalue()))
    assert "без RAG: prompt 10 · completion 5 · 1.5 с" in flat_err
    assert "с RAG: prompt 100 · completion 5 · embed 7 · 1.5 с" in flat_err


def test_ask_journals_both_modes_as_day_22(fakes, out, err):
    rag_cli.ask_question("Какое окно?")
    assert fakes.journal == [
        {"week": 5, "day": 22, "extra": {"command": "ask", "rag": False, "question_id": None}},
        {"week": 5, "day": 22, "extra": {"command": "ask", "rag": True, "question_id": None}},
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"question": "  "},
        {"question": "q", "mode": "maybe"},
        {"question": "q", "strategy": "semantic"},
        {"question": "q", "k": 0},
        {"question": "q", "k": 21},
    ],
)
def test_ask_rejects_bad_arguments_before_any_call(fakes, out, err, kwargs):
    with pytest.raises(AdventError):
        rag_cli.ask_question(**kwargs)
    assert fakes.complete_calls == []
    assert fakes.retrieve_calls == []


def test_eval_table_and_verdict_at_80_columns(fakes, out, err, tmp_path):
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    text = _plain(out.getvalue())
    flat = _flat(text)
    assert text.count("Контрольные вопросы: без RAG и с RAG") == 1
    assert "│ 1 │ Какое окно? │ 0/1 │ 1/1 │ да │ да │" in flat
    assert "│ 2 │ Какой тег? │ 0/1 │ 1/1 │ да │ да │" in flat
    assert "…" not in text
    assert "Фактов найдено: без RAG 0/2 · с RAG 2/2" in flat
    assert "Полных ответов: без RAG 0/2 · с RAG 2/2" in flat
    assert "Ожидаемый источник в top-k: 2/2 · назван в ответе: 2/2" in flat
    assert "Токены prompt/completion: без RAG 20/10 · с RAG 200/10 · embed 14" in flat
    assert "По фактам в этой выборке выше: с RAG (2 против 0)." in flat
    assert flat.endswith("разница в один факт не значима.")
    assert len(rows) == 2
    assert max(len(line) for line in text.splitlines()) <= 80


def test_eval_tie_names_no_winner(fakes, out, err, tmp_path):
    fakes.answer_plain = "окно 262144"
    rag_cli.run_eval(_questions(tmp_path, [Q1]))
    flat = _flat(_plain(out.getvalue()))
    assert "Ничья по фактам — явного лидера нет." in flat
    assert "выше:" not in flat


def test_eval_plain_winner(fakes, out, err, tmp_path):
    fakes.answer_plain = "262144"
    fakes.answer_rag = "не знаю"
    rag_cli.run_eval(_questions(tmp_path, [Q1]))
    flat = _flat(_plain(out.getvalue()))
    assert "По фактам в этой выборке выше: без RAG (1 против 0)." in flat


def test_eval_error_row_means_no_verdict(fakes, out, err, tmp_path):
    fakes.fail_on_call = 3
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    flat = _flat(_plain(out.getvalue()))
    assert "│ 2 │ Какой тег? │ ошибка │ ошибка │ — │ — │" in flat
    assert "Завершено пар: 1/2 — сравнение неполное, вердикта нет." in flat
    assert "выше:" not in flat
    assert "Ничья" not in flat
    assert "Фактов найдено: без RAG 0/1 · с RAG 1/1" in flat
    assert rows[1].ok is False
    assert "вопрос #2 не оценён: сервер недоступен" in _flat(_plain(err.getvalue()))


def test_eval_all_errors(fakes, out, err, tmp_path):
    fakes.fail_on_call = 1
    rag_cli.run_eval(_questions(tmp_path, [Q1]))
    flat = _flat(_plain(out.getvalue()))
    assert "Ни одна пара ответов не получена — сравнивать нечего." in flat


def test_eval_refuses_a_question_set_the_index_does_not_support(fakes, out, err, tmp_path):
    fakes.chunks = [_chunk("совсем другой текст")]
    with pytest.raises(AdventError) as exc:
        rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    assert "не согласован" in exc.value.message
    assert "#1" in exc.value.message
    assert "#2" in exc.value.message
    assert fakes.complete_calls == []


def test_eval_refuses_an_empty_set(fakes, out, err, tmp_path):
    with pytest.raises(AdventError):
        rag_cli.run_eval(_questions(tmp_path, []))
    assert fakes.complete_calls == []


def test_eval_answers_flag_prints_both_answers(fakes, out, err, tmp_path):
    rag_cli.run_eval(_questions(tmp_path, [Q1]), answers=True)
    text = _plain(out.getvalue())
    assert "#1 Какое окно?" in text
    assert text.count("── без RAG ──") == 1
    assert text.count("── с RAG ──") == 1
    assert text.index("── с RAG ──") < text.index("Контрольные вопросы")


def test_eval_without_answers_flag_prints_no_answers(fakes, out, err, tmp_path):
    rag_cli.run_eval(_questions(tmp_path, [Q1]))
    assert "── без RAG ──" not in _plain(out.getvalue())


def test_eval_journals_question_ids(fakes, out, err, tmp_path):
    rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    extras = [row["extra"] for row in fakes.journal]
    assert extras == [
        {"command": "eval", "rag": False, "question_id": 1},
        {"command": "eval", "rag": True, "question_id": 1},
        {"command": "eval", "rag": False, "question_id": 2},
        {"command": "eval", "rag": True, "question_id": 2},
    ]


def test_eval_progress_goes_to_stderr(fakes, out, err, tmp_path):
    rag_cli.run_eval(_questions(tmp_path, [Q1]))
    assert "вопрос 1/1 · без RAG" in _flat(_plain(err.getvalue()))
    assert "вопрос 1/1" not in _plain(out.getvalue())


def test_long_question_is_cut_in_cells(fakes, out, err, tmp_path):
    q = dict(Q1, question="Очень длинный вопрос про окно модели и ещё много слов подряд")
    rag_cli.run_eval(_questions(tmp_path, [q]))
    text = _plain(out.getvalue())
    assert "…" in text
    assert max(len(line) for line in text.splitlines()) <= 80


def test_commands_are_registered_on_the_adventrag_app():
    import typer.main

    from week_05 import cli

    names = [c.name for c in typer.main.get_command(cli.app).commands.values()]
    assert "ask" in names
    assert "eval" in names
    assert "index" in names


def test_eval_error_in_the_rag_half_of_a_pair_drops_the_whole_pair(fakes, out, err, tmp_path):
    fakes.fail_on_call = 4
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    flat = _flat(out.getvalue())
    assert "│ 2 │ Какой тег? │ ошибка │ ошибка │ — │ — │" in flat
    assert "Завершено пар: 1/2 — сравнение неполное, вердикта нет." in flat
    assert "Фактов найдено: без RAG 0/1 · с RAG 1/1" in flat
    assert "Токены prompt/completion: без RAG 10/5 · с RAG 100/5 · embed 7" in flat
    assert "выше:" not in flat and "Ничья" not in flat
    assert rows[1].ok is False and rows[1].plain is None
    assert len(fakes.complete_calls) == 4
