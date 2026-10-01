from __future__ import annotations

import io
import json
import re

import pytest
from rich.cells import cell_len
from rich.console import Console

from advent_core import chat as chat_core
from advent_core import config as config_module
from advent_core import console
from advent_core.errors import AdventError, NetworkError
from advent_core.rag import RagContext, RagHit, RetrievalTrace
from advent_core.telemetry import CallResult, Usage
from week_05 import rag_cli
from week_05.chunking import Chunk

_REAL_COMPLETE = chat_core.complete  # captured before any fixture patches it

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
        self.fail_count = 1
        self.fail_exc = NetworkError
        self.retriever_days: list[int] = []
        self.pauses: list[float] = []
        self.on_call = None
        self.chunks = [_chunk()]
        self.on_retrieve = None
        self.settings_seen: list = []
        self.full_by_question: dict[str, RagContext] = {}
        self.full_default: RagContext | None = None

    def complete(self, config, messages, capabilities=None, **kwargs):
        self.complete_calls.append(list(messages))
        n = len(self.complete_calls)
        if self.on_call is not None:
            self.on_call(n)
        if (
            self.fail_on_call is not None
            and self.fail_on_call <= n < self.fail_on_call + self.fail_count
        ):
            raise self.fail_exc("сервер недоступен")
        has_chunk = any(CHUNK_TEXT in str(m.get("content", "")) for m in messages)
        return CallResult(
            text=self.answer_rag if has_chunk else self.answer_plain,
            usage=Usage(prompt_tokens=100 if has_chunk else 10, completion_tokens=5),
            latency_ms=1500,
            stream=False,
        )

    def retrieve(self, question, settings):
        strategy, k = settings.strategy, settings.k
        if self.on_retrieve is not None:
            self.on_retrieve(question)
        self.retrieve_calls.append((question, strategy, k))
        self.settings_seen.append(settings)
        if settings.rerank:
            if question in self.full_by_question:
                return self.full_by_question[question]
            if self.full_default is not None:
                return self.full_default
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

    def make_retriever(db_path=None, *, day=22, **_kw):
        f.retriever_days.append(day)
        return f.retrieve

    monkeypatch.setattr(rag_cli.rag_module, "make_retriever", make_retriever)
    monkeypatch.setattr(rag_cli, "_pause", f.pauses.append)
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
    assert text.count("── RAG ──") == 1
    assert text.index("не знаю") < text.index("── RAG ──") < text.index("окно 262 144")
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
    assert "── RAG ──" not in _plain(out.getvalue())
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
    assert "RAG: prompt 100 · completion 5 · embed 7 · 1.5 с" in flat_err


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
    assert text.count("Контрольные вопросы: без RAG и RAG") == 1
    assert "│ 1 │ Какое окно? │ 0/1 │ 1/1 │ да │ да │" in flat
    assert "│ 2 │ Какой тег? │ 0/1 │ 1/1 │ да │ да │" in flat
    assert "…" not in text
    assert "Фактов найдено: без RAG 0/2 · RAG 2/2" in flat
    assert "Полных ответов: без RAG 0/2 · RAG 2/2" in flat
    assert "Ожидаемый источник в top-k: 2/2 · назван в ответе: 2/2" in flat
    assert "Токены prompt/completion: без RAG 20/10 · RAG 200/10 · embed 14" in flat
    assert "По фактам в этой выборке выше: RAG (2 против 0)." in flat
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
    fakes.fail_count = 2
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    flat = _flat(_plain(out.getvalue()))
    assert "│ 2 │ Какой тег? │ ошибка │ ошибка │ — │ — │" in flat
    assert "Завершено пар: 1/2 — сравнение неполное, вердикта нет." in flat
    assert "выше:" not in flat
    assert "Ничья" not in flat
    assert "Фактов найдено: без RAG 0/1 · RAG 1/1" in flat
    assert rows[1].ok is False
    assert "вопрос #2 не оценён: сервер недоступен" in _flat(_plain(err.getvalue()))


def test_eval_all_errors(fakes, out, err, tmp_path):
    fakes.fail_on_call = 1
    fakes.fail_count = 2
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
    assert text.count("── RAG ──") == 1
    assert text.index("── RAG ──") < text.index("Контрольные вопросы")


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


def test_eval_has_no_stderr_progress_lines_any_more(fakes, out, err, tmp_path):
    rag_cli.run_eval(_questions(tmp_path, [Q1]))
    assert "вопрос 1/1" not in _plain(err.getvalue())
    assert "вопрос 1/1" not in _plain(out.getvalue())


Q3 = {
    "id": 3,
    "question": "Какой тег и окно?",
    "expect": [["262144"], ["w01d01"]],
    "sources": ["CLAUDE.md"],
}


def _lines(out) -> list[str]:
    return _plain(out.getvalue()).splitlines()


def test_eval_detail_prints_blocks_then_short_lines_before_the_table(fakes, out, err, tmp_path):
    rag_cli.run_eval(_questions(tmp_path, [Q1, Q2, Q3]), detail=2)
    text = _plain(out.getvalue())
    lines = text.splitlines()
    assert text.count("#1 ") == 1
    assert lines[0] == "#1 Какое окно?"
    assert lines[1] == "   ожидается: 262144"
    assert lines[2] == "   без RAG ✗  «не знаю»"
    assert lines[3] == "   RAG     ✓  «окно 262 144, тег w01d01, см. CLAUDE.md»"
    assert lines[4] == ""
    assert lines[5] == "#2 Какой тег?"
    assert lines[6] == "   ожидается: w01d01"
    assert lines[9] == ""
    assert "#3 Какой тег и окно? · без RAG 0/2 · RAG 2/2" in lines
    assert "   ожидается: 262144 · w01d01" not in text
    assert lines[10] == "#3 Какой тег и окно? · без RAG 0/2 · RAG 2/2"
    assert lines[11] == ""
    assert lines[12].strip() == "Контрольные вопросы: без RAG и RAG"
    assert text.index("#3 Какой тег") < text.index("Контрольные вопросы")


def test_eval_detail_zero_prints_short_lines_only(fakes, out, err, tmp_path):
    rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    lines = _lines(out)
    assert "ожидается" not in "\n".join(lines)
    assert lines[0] == "#1 Какое окно? · без RAG 0/1 · RAG 1/1"
    assert lines[1] == "#2 Какой тег? · без RAG 0/1 · RAG 1/1"
    assert "«" not in "\n".join(lines)


def test_eval_two_fact_block_has_two_marks_and_aligned_answers(fakes, out, err, tmp_path):
    fakes.answer_plain = "тег w01d01 и всё"
    rag_cli.run_eval(_questions(tmp_path, [Q3, Q1]), detail=2)
    lines = _lines(out)
    assert lines[1] == "   ожидается: 262144 · w01d01"
    assert lines[2] == "   без RAG ✗✓ «тег w01d01 и всё»"
    assert lines[3].startswith("   RAG     ✓✓ «")
    q1_plain = lines[7]
    assert q1_plain.startswith("   без RAG ✗  «")
    assert {line.index("«") for line in (lines[2], lines[3], q1_plain, lines[8])} == {14}


def test_eval_error_pair_prints_an_error_line_and_still_the_table(fakes, out, err, tmp_path):
    fakes.fail_on_call = 3
    fakes.fail_count = 2
    rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]), detail=5)
    text = _plain(out.getvalue())
    assert "#2 Какой тег? · ошибка" in text.splitlines()
    assert text.index("#2 Какой тег? · ошибка") < text.index("Контрольные вопросы")
    assert "Завершено пар: 1/2" in _flat(text)


def test_eval_detail_lines_fit_80_cells_and_snippet_centres_on_the_fact(fakes, out, err, tmp_path):
    q = dict(Q1, question="Очень длинный вопрос про окно модели 🚀 и ещё много слов подряд " * 3)
    fakes.answer_rag = "слово " * 60 + "окно 262 144 токена" + " слово" * 60
    fakes.answer_plain = "начало " * 60
    rag_cli.run_eval(_questions(tmp_path, [q]), detail=1)
    lines = _lines(out)
    assert all(cell_len(line) <= 80 for line in lines)
    rag_line = next(line for line in lines if line.startswith("   RAG     ✓  «"))
    assert rag_line.startswith("   RAG     ✓  «…")
    assert rag_line.endswith("…»")
    assert "262 144" in rag_line
    assert lines[0].endswith("…")


def test_eval_snippet_without_the_fact_is_the_start_of_the_answer(fakes, out, err, tmp_path):
    fakes.answer_plain = "Начало ответа без факта " + "слово " * 60
    rag_cli.run_eval(_questions(tmp_path, [Q1]), detail=1)
    plain_line = _lines(out)[2]
    assert plain_line.startswith("   без RAG ✗  «Начало ответа без факта слово")
    assert plain_line.endswith("…»")
    assert "«…" not in plain_line


def test_eval_answers_come_after_the_row(fakes, out, err, tmp_path):
    rag_cli.run_eval(_questions(tmp_path, [Q1]), answers=True)
    text = _plain(out.getvalue())
    assert text.index("#1 Какое окно? · без RAG 0/1 · RAG 1/1") < text.index("── без RAG ──")


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


def test_eval_error_in_the_rag_half_drops_the_pair_from_metrics_but_not_from_spend(
    fakes, out, err, tmp_path
):
    fakes.fail_on_call = 4
    fakes.fail_count = 2
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    flat = _flat(out.getvalue())
    assert "│ 2 │ Какой тег? │ ошибка │ ошибка │ — │ — │" in flat
    assert "Завершено пар: 1/2 — сравнение неполное, вердикта нет." in flat
    assert "Фактов найдено: без RAG 0/1 · RAG 1/1" in flat
    # The failed pair's completed first run (10/5) is paid for and stays in the totals.
    assert "Токены prompt/completion: без RAG 20/10 · RAG 100/5 · embed 7" in flat
    assert "выше:" not in flat and "Ничья" not in flat
    assert rows[1].ok is False and rows[1].first is not None and rows[1].second is None
    assert len(fakes.complete_calls) == 5


def test_eval_block_is_printed_before_the_next_question_starts(fakes, out, err, tmp_path):
    seen = []

    def check(n):
        if n == 3:
            seen.append("   ожидается: 262144" in _plain(out.getvalue()))

    fakes.on_call = check
    rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]), detail=1)
    assert seen == [True]


def test_eval_short_line_is_printed_before_the_next_question_starts(fakes, out, err, tmp_path):
    seen = []

    def check(n):
        if n == 3:
            seen.append("#1 Какое окно? · без RAG 0/1 · RAG 1/1" in _plain(out.getvalue()))

    fakes.on_call = check
    rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    assert seen == [True]


def test_eval_error_line_is_printed_before_the_next_pair_starts(fakes, out, err, tmp_path):
    seen = []

    def check(n):
        if n == 5:
            seen.append("#2 Какой тег? · ошибка" in _plain(out.getvalue()))

    fakes.on_call = check
    fakes.fail_on_call = 3
    fakes.fail_count = 2
    rag_cli.run_eval(_questions(tmp_path, [Q1, Q2, Q3]))
    assert seen == [True]


def test_eval_retries_a_failed_call_once(fakes, out, err, tmp_path):
    fakes.fail_on_call = 3
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    assert rows[1].ok is True
    assert len(fakes.complete_calls) == 5
    assert _plain(err.getvalue()).count("вопрос #2: сервер недоступен — повтор") == 1
    assert "не оценён" not in _flat(err.getvalue())


def test_eval_two_failures_in_a_row_make_an_error_row(fakes, out, err, tmp_path):
    fakes.fail_on_call = 3
    fakes.fail_count = 2
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]))
    assert rows[1].ok is False
    assert len(fakes.complete_calls) == 4
    assert _plain(err.getvalue()).count("— повтор") == 1
    assert "вопрос #2 не оценён: сервер недоступен" in _flat(err.getvalue())


def test_eval_blank_line_separates_live_rows_from_the_table(fakes, out, err, tmp_path):
    rag_cli.run_eval(_questions(tmp_path, [Q1]))
    lines = _lines(out)
    assert lines[0] == "#1 Какое окно? · без RAG 0/1 · RAG 1/1"
    assert lines[1] == ""
    assert lines[2].strip() == "Контрольные вопросы: без RAG и RAG"


def test_snippet_keeps_a_fact_wider_than_the_left_third_whole():
    text = "x " * 60 + "A" * 50
    got = rag_cli._snippet(text, [("A" * 50,)], 65)
    assert got.startswith("…")
    assert got.endswith("A" * 50)
    assert cell_len(got) <= 65


def test_snippet_fact_wider_than_the_budget_shows_its_start():
    got = rag_cli._snippet("x " * 10 + "A" * 50, [("A" * 50,)], 20)
    assert got == "A" * 19 + "…"


def test_snippet_measures_a_wide_fact_in_cells():
    fact = "界" * 25
    got = rag_cli._snippet("слово " * 20 + fact + " слово" * 20, [(fact,)], 60)
    assert fact in got
    assert cell_len(got) <= 60


def test_snippet_finds_a_fact_with_double_space_inside():
    text = "слово " * 30 + "foo  bar" + " слово" * 30
    got = rag_cli._snippet(text, [("foo  bar",)], 40)
    assert "foo bar" in got
    assert got.startswith("…")


def test_expected_line_collapses_whitespace_inside_a_fact(fakes, out, err, tmp_path):
    q = dict(Q1, expect=[["x" + chr(10) + "  y", "262144"]])
    rag_cli.run_eval(_questions(tmp_path, [q]), detail=1)
    assert "   ожидается: x y" in _lines(out)


def test_snippet_falls_back_to_the_start_when_casefold_and_ignorecase_disagree():
    from advent_core.rag import check_facts

    answer = "Straße " + "слово " * 30
    assert check_facts(answer, [["strasse"]]) == (True,)
    got = rag_cli._snippet(answer, [("strasse",)], 40)
    assert got.startswith("Straße слово")
    assert got.endswith("…")


# --- Day 23: pairs, full mode, retries, stages ------------------------------------------

REL = "structure:CLAUDE.md#1"


def _h(cid, *, score=0.8, rerank=None, rank=None, text="т", source="CLAUDE.md"):
    return RagHit(
        chunk_id=cid, source=source, section="Раздел", score=score, text=text,
        rerank=rerank, rank=rank,
    )  # fmt: skip


def _full_ctx(*, hits, original=(), fused=(), reranked=(), candidates=31, passed=None,
              warnings=(), rewritten="окно токенов context window"):  # fmt: skip
    return RagContext(
        hits=tuple(hits),
        strategy="structure",
        k=5,
        embed_model="mistral-embed",
        embed_tokens=7,
        corpus_rev="abc",
        rewritten=rewritten,
        candidates=candidates,
        passed=len(hits) if passed is None else passed,
        threshold=5.0,
        aux_prompt_tokens=100,
        aux_completion_tokens=10,
        warnings=tuple(warnings),
        trace=RetrievalTrace(tuple(original), tuple(fused), tuple(reranked)),
    )


def _relevant_hit(**kw):
    return _h(REL, text=CHUNK_TEXT, **kw)


@pytest.mark.parametrize(
    "pair", ["off", "plain,plain", "full,plain", "plain,off", "off,full,plain", "a,b", ""]
)
def test_pair_is_validated_before_any_call(fakes, out, err, tmp_path, pair):
    with pytest.raises(AdventError):
        rag_cli.ask_question("q", pair=pair)
    with pytest.raises(AdventError):
        rag_cli.run_eval(_questions(tmp_path, [Q1]), pair=pair)
    assert fakes.complete_calls == []
    assert fakes.retrieve_calls == []
    assert fakes.check_calls == []


@pytest.mark.parametrize("mode", ["rag", "no-rag", "full"])
def test_pair_conflicts_with_single_mode_flags(fakes, out, err, mode):
    with pytest.raises(AdventError) as exc:
        rag_cli.ask_question("q", mode=mode, pair="plain,full")
    assert "--pair" in exc.value.message
    assert fakes.complete_calls == []


def test_ask_full_flag_conflicts_with_rag_flag_in_the_command(fakes, out, err):
    from week_05 import cli

    with pytest.raises(AdventError):
        cli.app(["ask", "q", "--full", "--rag"], standalone_mode=False)
    assert fakes.complete_calls == []


def test_ask_pair_plain_full_prints_candidates_line_and_was_rank(fakes, out, err):
    fakes.full_default = _full_ctx(
        hits=[_relevant_hit(score=0.812, rerank=9.0, rank=14)],
        candidates=31,
        passed=4,
    )
    rag_cli.ask_question("Какое окно?", pair="plain,full")
    text = _plain(out.getvalue())
    assert text.count("── RAG ──") == 1
    assert text.count("── RAG+rerank ──") == 1
    assert "rewrite: окно токенов context window" in text
    assert "  [1] cos 0.812 · rerank 9 · был #14  CLAUDE.md — Раздел" in text
    assert "кандидатов 31 → прошли порог 4 → в контексте 1" in text
    assert "  [1] 0.900  CLAUDE.md — Раздел" in text
    assert "…" not in text
    assert text.index("── RAG ──") < text.index("── RAG+rerank ──")
    flat_err = _flat(err.getvalue())
    assert "RAG+rerank: prompt 100 · completion 5 · embed 7 · aux 100+10 ток." in flat_err


def test_ask_full_only_runs_one_mode_and_journals_day_23(fakes, out, err):
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    rag_cli.ask_question("Какое окно?", mode="full", k_before=30, threshold=6)
    assert len(fakes.complete_calls) == 1
    assert fakes.retriever_days == [23]
    assert fakes.journal == [
        {
            "week": 5,
            "day": 23,
            "extra": {"command": "ask", "rag": True, "question_id": None, "mode": "full"},
        }
    ]
    s = fakes.settings_seen[0]
    assert (s.rewrite, s.rerank, s.k_before, s.threshold) == (True, True, 30, 6.0)


def test_ask_plain_settings_have_no_rewrite_or_rerank(fakes, out, err):
    rag_cli.ask_question("Какое окно?", mode="rag")
    s = fakes.settings_seen[0]
    assert (s.rewrite, s.rerank) == (False, False)
    assert fakes.retriever_days == [22]


def test_ask_off_plain_pair_is_still_day_22(fakes, out, err):
    rag_cli.ask_question("Какое окно?", pair="off,plain")
    assert [row["day"] for row in fakes.journal] == [22, 22]


def test_ask_full_with_nothing_passed_says_so(fakes, out, err):
    fakes.full_default = _full_ctx(hits=[], candidates=28, passed=0)
    rag_cli.ask_question("Как приготовить борщ?", mode="full")
    text = _plain(out.getvalue())
    assert "источники: ничего не найдено" in text
    assert "кандидатов 28 → прошли порог 0 → в контексте 0" in text


def test_ctx_warnings_are_printed_by_the_cli(fakes, out, err):
    fakes.full_default = _full_ctx(
        hits=[_relevant_hit(rerank=9.0, rank=1)], warnings=("reranker не оценил 2 из 31 чанков",)
    )
    rag_cli.ask_question("Какое окно?", mode="full")
    assert "reranker не оценил 2 из 31 чанков" in _flat(err.getvalue())
    assert "reranker не оценил" not in _plain(out.getvalue())


def test_ask_rejects_bad_k_before_and_threshold(fakes, out, err):
    for kwargs in ({"k_before": 0}, {"k_before": 51}, {"threshold": -1}, {"threshold": 10.5}):
        with pytest.raises(AdventError):
            rag_cli.ask_question("q", mode="full", **kwargs)
    assert fakes.complete_calls == []


def test_eval_pair_plain_full_labels_and_aux_tokens(fakes, out, err, tmp_path):
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=2)], passed=1)
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]), pair="plain,full")
    text = _plain(out.getvalue())
    flat = _flat(text)
    assert text.count("Контрольные вопросы: RAG и RAG+rerank") == 1
    assert "│ 1 │ Какое окно? │ 1/1 │ 1/1 │ да │ да │" in flat
    assert "#1 Какое окно? · RAG 1/1 · RAG+rerank 1/1" in text
    assert "Фактов найдено: RAG 2/2 · RAG+rerank 2/2" in flat
    assert (
        "Токены prompt/completion: RAG 200/10 · RAG+rerank 200/10 · embed 28 · aux 200/20" in flat
    )
    assert "Ничья по фактам — явного лидера нет." in flat
    assert "…" not in text
    assert [r["day"] for r in fakes.journal] == [23, 23, 23, 23]
    assert fakes.retriever_days == [23]
    assert len(rows) == 2


def test_eval_pair_off_full_verdict_names_full(fakes, out, err, tmp_path):
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    rag_cli.run_eval(_questions(tmp_path, [Q1]), pair="off,full")
    flat = _flat(out.getvalue())
    assert "По фактам в этой выборке выше: RAG+rerank (1 против 0)." in flat
    assert "Контрольные вопросы: без RAG и RAG+rerank" in flat


def test_eval_detail_labels_align_for_the_longest_label(fakes, out, err, tmp_path):
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    rag_cli.run_eval(_questions(tmp_path, [Q1]), pair="plain,full", detail=1)
    lines = _lines(out)
    assert lines[2].startswith("   RAG        ✓  «")
    assert lines[3].startswith("   RAG+rerank ✓  «")


def test_eval_retry_waits_five_seconds_for_transient_errors(fakes, out, err, tmp_path):
    fakes.fail_on_call = 1
    rag_cli.run_eval(_questions(tmp_path, [Q1]))
    assert fakes.pauses == [5.0]
    assert _plain(err.getvalue()).count("— повтор") == 1


@pytest.mark.parametrize("code", [4, 5, 6])
def test_transient_exit_codes_are_retried(fakes, out, err, tmp_path, code):
    from advent_core import errors

    fakes.fail_exc = {
        4: errors.RateLimitError,
        5: errors.ServerError,
        6: errors.NetworkError,
    }[code]
    fakes.fail_on_call = 1
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1]))
    assert rows[0].ok
    assert fakes.pauses == [5.0]


def test_non_transient_errors_are_not_retried(fakes, out, err, tmp_path):
    fakes.fail_exc = AdventError
    fakes.fail_on_call = 1
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1]))
    assert rows[0].ok is False
    assert fakes.pauses == []
    assert len(fakes.complete_calls) == 1
    assert "— повтор" not in _plain(err.getvalue())
    assert "вопрос #1 не оценён: сервер недоступен" in _flat(err.getvalue())


def _stage_fakes(fakes):
    def chunk(n, text):
        return Chunk(
            chunk_id=f"structure:CLAUDE.md#{n}", strategy="structure", source="CLAUDE.md",
            title="t", section="s", ordinal=n, char_start=0, char_end=len(text),
            line_start=1, line_end=1, text=text,
        )  # fmt: skip

    fakes.chunks = [_chunk(), chunk(2, "альфа"), chunk(3, "бета")]
    others = [_h(f"structure:x#{i}") for i in range(1, 8)]
    fakes.full_by_question = {
        # relevant: cosine 4 -> rewrite 2 -> rerank 1 -> final 1, 3 of 27 passed
        "Какое окно?": _full_ctx(
            hits=[_relevant_hit(rerank=9.0, rank=2)],
            original=[*others[:3], _relevant_hit()],
            fused=[others[0], _relevant_hit(), *others[1:3]],
            reranked=[_relevant_hit(rerank=9.0), *others[:3]],
            candidates=27,
            passed=3,
        ),
        # relevant sits at cosine 6 but the reranker gave it 2: cut by the threshold 5
        "Какой тег?": _full_ctx(
            hits=[],
            original=[*others[:5], _relevant_hit()],
            fused=[*others[:2], _relevant_hit(), others[2]],
            reranked=[others[0], others[1], _relevant_hit(rerank=2.0, rank=3)],
            candidates=20,
            passed=0,
        ),
        # no single chunk holds both facts
        "Альфа и бета?": _full_ctx(hits=[], candidates=12, passed=0),
    }
    q3 = {
        "id": 3,
        "question": "Альфа и бета?",
        "expect": [["альфа"], ["бета"]],
        "sources": ["CLAUDE.md"],
    }
    return [Q1, Q2, q3]


def test_stages_table_cells_summary_and_exclusions_at_80_columns(fakes, out, err, tmp_path):
    items = _stage_fakes(fakes)
    rows = rag_cli.run_stages(_questions(tmp_path, items))
    text = _plain(out.getvalue())
    flat = _flat(text)
    assert "│ 1 │ Какое окно? │ 4 │ 2 │ 1 │ 1 │ 3/27 │" in flat
    assert "│ 2 │ Какой тег? │ 6 │ 3 │ 3 │ — │ 0/20 │" in flat
    assert "│ 3 │ Альфа и бета? │ н/о │ н/о │ н/о │ н/о │ 0/12 │" in flat
    assert text.count("Ранг первого релевантного чанка по этапам") == 1
    assert flat.count("│ # │ вопрос │ cosine │ +rewrite │ +rerank │ итог │ порог │") == 1
    assert "…" not in text
    assert max(len(line) for line in text.splitlines()) <= 80
    assert "Релевантный чанк, оцениваемых вопросов: 2" in flat
    assert "cosine: в top-5 1/2 · на 1-м месте 0/2" in flat
    assert "+rewrite: в top-5 2/2 · на 1-м месте 0/2" in flat
    assert "+rerank (до порога): в top-5 2/2 · на 1-м месте 1/2" in flat
    assert "итог (после порога): в контексте 1/2 · на 1-м месте 1/2" in flat
    assert "Порог 5: релевантных чанков ниже порога: 1 (в 1 вопросах): #2×1" in flat
    assert "Вне сумм (нет чанка со всеми фактами): #3" in flat
    assert "Токены: embed 21 · aux prompt/completion 300/30" in flat
    assert flat.endswith("ранги плавают на 1–2 позиции.")
    assert [r.ctx is not None for r in rows] == [True, True, True]
    assert all(c.rewrite and c.rerank for c in fakes.settings_seen)
    assert fakes.retriever_days == [23]
    assert fakes.complete_calls == []


def test_stages_final_rank_is_rechecked_on_the_retained_truncated_text(fakes, out, err, tmp_path):
    items = _stage_fakes(fakes)
    cut_hit = _h(REL, text="ТЕКСТ-ЧАНКА: окно", rerank=9.0, rank=1)  # fit_hits cut the fact
    fakes.full_by_question["Какое окно?"] = _full_ctx(
        hits=[cut_hit],
        original=[_relevant_hit()],
        fused=[_relevant_hit()],
        reranked=[_relevant_hit(rerank=9.0)],
        passed=1,
    )
    rows = rag_cli.run_stages(_questions(tmp_path, items[:1]))
    assert rows[0].ranks() == (1, 1, 1, None)
    assert "│ 1 │ Какое окно? │ 1 │ 1 │ 1 │ — │" in _flat(out.getvalue())


def test_stages_threshold_loss_counts_every_rated_relevant_candidate(fakes, out, err, tmp_path):
    import dataclasses

    items = _stage_fakes(fakes)
    second_id = "structure:CLAUDE.md#9"
    fakes.chunks = [*fakes.chunks, dataclasses.replace(_chunk(), chunk_id=second_id)]
    second = _h(second_id, text=CHUNK_TEXT, rerank=2.0)
    fakes.full_by_question["Какое окно?"] = _full_ctx(
        hits=[_relevant_hit(rerank=9.0, rank=1)],
        original=[_relevant_hit(), second],
        fused=[_relevant_hit(), second],
        reranked=[_relevant_hit(rerank=9.0), second],
        passed=1,
    )
    rows = rag_cli.run_stages(_questions(tmp_path, items[:1]))
    assert rows[0].threshold_losses(5.0) == (1, 0)
    assert rows[0].threshold_losses(10.0) == (2, 0)
    flat = _flat(out.getvalue())
    assert "релевантных чанков ниже порога: 1 (в 1 вопросах): #1×1" in flat
    assert "без оценки" not in flat


def test_stages_reports_unrated_relevant_candidates_separately(fakes, out, err, tmp_path):
    items = _stage_fakes(fakes)
    fakes.full_by_question["Какое окно?"] = _full_ctx(
        hits=[],
        original=[_relevant_hit()],
        fused=[_relevant_hit()],
        reranked=[_relevant_hit()],
        passed=0,
    )
    rows = rag_cli.run_stages(_questions(tmp_path, items[:1]))
    assert rows[0].threshold_losses(5.0) == (0, 1)
    flat = _flat(out.getvalue())
    assert "ниже порога: 0 (в 0 вопросах)" in flat
    assert "без оценки reranker'а: 1 (в 1 вопросах): #1×1" in flat


def test_stages_prints_a_live_line_before_the_next_question_starts(fakes, out, err, tmp_path):
    items = _stage_fakes(fakes)
    seen = []

    def check(question):
        if question == "Какой тег?":
            seen.append(
                "#1 cosine 4 → +rewrite 2 → +rerank 1 · порог 3/27" in _plain(out.getvalue())
            )

    fakes.on_retrieve = check
    rag_cli.run_stages(_questions(tmp_path, items))
    assert seen == [True]
    lines = _lines(out)
    assert "#2 cosine 6 → +rewrite 3 → +rerank 3 · порог 0/20" in lines
    assert "#3 нет чанка со всеми фактами · порог 0/12" in lines


def test_stages_passes_options_to_the_retriever(fakes, out, err, tmp_path):
    items = _stage_fakes(fakes)
    rag_cli.run_stages(_questions(tmp_path, items[:1]), k_before=30, k=3, threshold=7)
    s = fakes.settings_seen[0]
    assert (s.k_before, s.k, s.threshold, s.strategy) == (30, 3, 7.0, "structure")


def test_stages_error_row_is_excluded_and_reported(fakes, out, err, tmp_path):
    items = _stage_fakes(fakes)

    def boom(question):
        if question == "Какой тег?":
            raise AdventError("reranker сломался")

    fakes.on_retrieve = boom
    rows = rag_cli.run_stages(_questions(tmp_path, items))
    flat = _flat(out.getvalue())
    assert rows[1].error == "reranker сломался"
    assert "#2 ошибка: reranker сломался" in _lines(out)
    assert "Не оценено из-за ошибок: #2" in flat
    assert "Релевантный чанк, оцениваемых вопросов: 1" in flat
    assert fakes.pauses == []


def test_stages_retries_a_transient_error_once(fakes, out, err, tmp_path):
    items = _stage_fakes(fakes)
    calls = []

    def flaky(question):
        calls.append(question)
        if len(calls) == 1:
            raise NetworkError("нет связи")

    fakes.on_retrieve = flaky
    rows = rag_cli.run_stages(_questions(tmp_path, items[:1]))
    assert rows[0].error is None
    assert fakes.pauses == [5.0]


def test_stages_rejects_bad_arguments_before_any_call(fakes, out, err, tmp_path):
    for kwargs in ({"k": 0}, {"k_before": 51}, {"threshold": 11}, {"strategy": "x"}):
        with pytest.raises(AdventError):
            rag_cli.run_stages(_questions(tmp_path, [Q1]), **kwargs)
    assert fakes.retrieve_calls == []


def test_stages_command_is_registered():
    import typer.main

    from week_05 import cli

    names = [c.name for c in typer.main.get_command(cli.app).commands.values()]
    assert "stages" in names


# --- Day 24: cite mode ------------------------------------------------------------------

QUOTE = "окно 262144 токена, тег w01d01"
CITE_JSON = json.dumps(
    {
        "status": "answer",
        "answer": "окно 262144 токена, тег w01d01",
        "sources": [1],
        "quotes": [{"id": 1, "text": QUOTE}],
    },
    ensure_ascii=False,
)


def _refusal_ctx():
    return _full_ctx(hits=[], reranked=[_h("structure:CLAUDE.md#9", rerank=2.0)], passed=0)


def _grounded(verdict="да", reason="цитата прямо называет окно"):
    from week_05.rag import Grounding

    return Grounding(verdict, reason, CallResult(usage=Usage(30, 5), stream=False))


@pytest.fixture
def judge(monkeypatch):
    calls: list[tuple] = []
    state = {"value": _grounded(), "raise": None}

    def fake(cited, hits, question, *, day):
        calls.append((question, day))
        if state["raise"] is not None:
            raise state["raise"]
        return state["value"]

    monkeypatch.setattr(rag_cli.rag_module, "judge_grounding", fake)
    fake.calls = calls
    fake.state = state
    return fake


def _unanswerable(tmp_path, items):
    path = tmp_path / "unanswerable.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


U1 = {"id": 101, "question": "Как приготовить борщ?", "note": "н"}
U2 = {"id": 102, "question": "Какая PostgreSQL?", "note": "н"}


def test_ask_cite_prints_rendered_answer_with_chunk_id_and_quote_marks(fakes, out, err):
    fakes.answer_rag = CITE_JSON
    rag_cli.ask_question("Какое окно?", mode="cite")
    text = _plain(out.getvalue())
    assert text.count("── RAG+цитаты ──") == 1
    assert "Источники:" in text
    assert "[1] CLAUDE.md — Раздел · structure:CLAUDE.md#1 · цитата ✓" in text
    assert "[1] ✓ «окно 262144 токена, тег w01d01»" in text
    assert "цитаты: 1/1 дословны" in _flat(err.getvalue())
    assert fakes.retriever_days == [24]
    assert fakes.journal == [
        {
            "week": 5,
            "day": 24,
            "extra": {"command": "ask", "rag": True, "question_id": None, "mode": "cite"},
        }
    ]
    s = fakes.settings_seen[0]
    assert (s.rewrite, s.rerank, s.cite) == (True, True, True)


def test_ask_cite_empty_context_makes_no_model_call_and_no_journal_row(fakes, out, err):
    fakes.full_default = _refusal_ctx()
    rag_cli.ask_question("Как приготовить борщ?", mode="cite")
    text = _plain(out.getvalue())
    assert fakes.complete_calls == []
    assert fakes.journal == []
    assert "Не знаю: в документации проекта не нашлось фрагментов, относящихся к вопросу." in text
    assert "Уточните вопрос — возможно, вы про:" in text
    assert "· CLAUDE.md — Раздел" in text
    assert "кандидатов 31 → прошли порог 0 → в контексте 0" in text
    flat_err = _flat(err.getvalue())
    assert "RAG+цитаты: не знаю · модель не вызывалась · aux 100+10 ток." in flat_err
    assert "prompt" not in flat_err


def test_ask_cite_garbage_json_is_journaled_and_shown_as_refusal(fakes, out, err):
    fakes.answer_rag = "это не JSON"
    rag_cli.ask_question("Какое окно?", mode="cite")
    assert [r["day"] for r in fakes.journal] == [24]
    assert "Не знаю: ответ модели не удалось разобрать." in _plain(out.getvalue())


def test_ask_cite_unverified_shows_the_draft_on_screen(fakes, out, err):
    fakes.answer_rag = json.dumps(
        {
            "status": "answer",
            "answer": "выдумка",
            "sources": [1],
            "quotes": [{"id": 1, "text": "такого в чанке нет совсем"}],
        },
        ensure_ascii=False,
    )
    rag_cli.ask_question("Какое окно?", mode="cite")
    text = _plain(out.getvalue())
    assert "Не знаю: ни одна цитата модели не нашлась дословно во фрагментах." in text
    assert "неподтверждённый ответ модели: «выдумка»" in text


def test_ask_pair_full_cite_splits_the_journal_days_by_mode(fakes, out, err):
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    fakes.answer_rag = CITE_JSON
    rag_cli.ask_question("Какое окно?", pair="full,cite")
    text = _plain(out.getvalue())
    assert text.index("── RAG+rerank ──") < text.index("── RAG+цитаты ──")
    assert fakes.retriever_days == [23, 24]
    assert [(r["day"], r["extra"]["mode"]) for r in fakes.journal] == [(23, "full"), (24, "cite")]


def test_pairs_with_cite_are_allowed_and_old_pairs_keep_one_day():
    assert rag_cli.parse_pair("off,cite") == ("off", "cite")
    assert rag_cli.parse_pair("full,cite") == ("full", "cite")
    assert rag_cli.mode_days(("plain", "full")) == {"plain": 23, "full": 23}
    assert rag_cli.mode_days(("off", "plain")) == {"off": 22, "plain": 22}
    assert rag_cli.mode_days(("full", "cite")) == {"full": 23, "cite": 24}
    with pytest.raises(AdventError):
        rag_cli.parse_pair("cite,full")


def test_ask_cite_flag_conflicts_with_other_mode_flags(fakes, out, err):
    from week_05 import cli

    for extra in (["--rag"], ["--full"]):
        with pytest.raises(AdventError):
            cli.app(["ask", "q", "--cite", *extra], standalone_mode=False)
    with pytest.raises(AdventError):
        rag_cli.ask_question("q", mode="cite", pair="full,cite")
    assert fakes.complete_calls == []


def test_eval_cite_pair_table_columns_summary_and_no_ellipsis_in_numbers(
    fakes, out, err, tmp_path, judge
):
    fakes.answer_rag = CITE_JSON
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    fakes.full_by_question["Как приготовить борщ?"] = _refusal_ctx()
    rows = rag_cli.run_eval(
        _questions(tmp_path, [Q1, Q2]),
        pair="full,cite",
        unanswerable_path=_unanswerable(tmp_path, [U1, U2]),
    )
    text = _plain(out.getvalue())
    flat = _flat(text)
    assert len(rows) == 2
    assert text.count("Контрольные вопросы: RAG+rerank и RAG+цитаты") == 1
    assert text.count("источн.") == 1
    assert "│ 1 │ Какое окно? │ 1/1 │ 1/1 │ да │ 1/1 │ да │ нет │" in flat
    assert "#1 Какое окно? · full 1/1 · cite 1/1 · цитаты 1/1 · смысл да" in text
    assert text.count("Вопросы без ответа в репо") == 1
    assert "│ 101 │ Как приготовить борщ? │ ≈не знаю ✓ │ не знаю ✓ │" in flat
    assert "│ 102 │ Какая PostgreSQL? │ ≈ответил ✗ │ ответил ✗ │" in flat
    assert text.index("Вопросы без ответа в репо") < text.index("Запланировано 2")
    assert "Запланировано 2 · получено 2 · без ответа 0" in flat
    assert (
        "Источники в ответе 2/2 · цитаты 2/2 · все цитаты дословны 2/2 · "
        "«не знаю» на отвечаемых 0/2" in flat
    )
    assert "Смысл по judge, из 2: да 2 · частично 0 · нет 0 · н/о 0 · не оценивался 0" in flat
    assert "Без ответа в репо: запланировано 2 · получено 2 · без ответа 0" in flat
    assert "«Не знаю» на неотвечаемых: cite 1/2 · RAG+rerank ≈1/2 (эвристика)" in flat
    assert "Без вызова модели: 1" in flat
    assert flat.rstrip().endswith("его вердикт — оценка, не истина.")
    for line in text.splitlines():
        if line.startswith("│ 1 "):
            assert "…" not in line
    assert [q for q, _ in judge.calls] == ["Какое окно?", "Какой тег?"]
    assert {d for _, d in judge.calls} == {24}


def test_eval_cite_aux_tokens_include_the_judge(fakes, out, err, tmp_path, judge):
    fakes.answer_rag = CITE_JSON
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    rag_cli.run_eval(_questions(tmp_path, [Q1]), pair="full,cite", unanswerable=False)
    flat = _flat(out.getvalue())
    assert "aux (rewrite+rerank+judge) prompt/completion: 230/25" in flat
    assert "Вопросы без ответа" not in flat


def test_eval_cite_counts_the_cite_answer_only_never_the_draft(fakes, out, err, tmp_path, judge):
    fakes.answer_rag = json.dumps(
        {
            "status": "answer",
            "answer": "окно 262144 токена",
            "sources": [1],
            "quotes": [{"id": 1, "text": "такого в чанке нет совсем"}],
        },
        ensure_ascii=False,
    )
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1]), pair="full,cite", unanswerable=False)
    assert rows[0].second_score.facts == (False,)
    assert rows[0].second_score.sources_cited == (False,)
    assert judge.calls == []
    flat = _flat(out.getvalue())
    assert "│ 1 │ Какое окно? │ 1/1 │ 0/1 │ — │ 0/1 │ — │ да·цитаты │" in flat
    assert "не оценивался 1" in flat
    text = _plain(out.getvalue())
    assert "#1 Какое окно? · full 1/1 · cite 0/1 · не знаю·цитаты" in text
    for line in text.splitlines():
        assert cell_len(line) <= 80, line
        if line.startswith("│ 1 "):
            assert "…" not in line
    assert text.count("источн.") == 1


def test_eval_judge_garbage_and_exception_show_no_rating_and_keep_the_pair(
    fakes, out, err, tmp_path, judge
):
    fakes.answer_rag = CITE_JSON
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    judge.state["value"] = _grounded(None, "")
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1]), pair="full,cite", unanswerable=False)
    assert "│ 1 │ Какое окно? │ 1/1 │ 1/1 │ да │ 1/1 │ н/о │ нет │" in _flat(out.getvalue())
    assert rows[0].ok

    out.truncate(0)
    out.seek(0)
    judge.state["raise"] = ValueError("boom")
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1]), pair="full,cite", unanswerable=False)
    flat = _flat(out.getvalue())
    assert rows[0].ok and rows[0].judge_error == "boom"
    assert "│ 1 │ Какое окно? │ 1/1 │ 1/1 │ да │ 1/1 │ н/о │ нет │" in flat
    assert "н/о 1" in flat


def test_eval_detail_cite_shows_expected_answer_quotes_and_judge_reason(
    fakes, out, err, tmp_path, judge
):
    fakes.answer_rag = CITE_JSON
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    rag_cli.run_eval(_questions(tmp_path, [Q1]), pair="full,cite", detail=1, unanswerable=False)
    text = _plain(out.getvalue())
    assert "   ожидается: 262144" in text
    assert "   full ✓" in text
    assert "   cite ✓" in text
    assert "   цитаты 1/1 дословны · источник да · смысл да — цитата прямо называет окно" in text
    assert "   [1] ✓ «окно 262144 токена, тег w01d01»" in text


def test_eval_cite_refusal_on_an_answerable_question_is_counted_apart(
    fakes, out, err, tmp_path, judge
):
    fakes.full_default = _refusal_ctx()
    rag_cli.run_eval(_questions(tmp_path, [Q1]), pair="off,cite", unanswerable=False)
    flat = _flat(out.getvalue())
    assert "│ 1 │ Какое окно? │ 0/1 │ 0/1 │ — │ — │ — │ да·порог │" in flat
    text = _plain(out.getvalue())
    assert "#1 Какое окно? · off 0/1 · cite 0/1 · не знаю·порог" in text
    assert text.count("Контрольные вопросы: без RAG и RAG+цитаты") == 1
    for line in text.splitlines():
        assert cell_len(line) <= 80, line
    assert "«не знаю» на отвечаемых 1/1" in flat
    assert "Без вызова модели: 1" in flat
    assert judge.calls == []


def test_eval_cite_failed_pair_keeps_the_denominators_honest(fakes, out, err, tmp_path, judge):
    fakes.answer_rag = CITE_JSON
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    fakes.fail_on_call, fakes.fail_count, fakes.fail_exc = 1, 1, AdventError
    rows = rag_cli.run_eval(_questions(tmp_path, [Q1, Q2]), pair="full,cite", unanswerable=False)
    flat = _flat(out.getvalue())
    assert [r.ok for r in rows] == [False, True]
    assert "Запланировано 2 · получено 1 · без ответа 1" in flat
    assert "цитаты 1/1" in flat


def test_unanswerable_outcome_covers_every_cite_reason_and_the_heuristic():
    from advent_core.rag import unknown_answer
    from week_05.rag import unanswerable_outcome

    assert unanswerable_outcome("", unknown_answer("empty_context")) == "refused"
    assert unanswerable_outcome("", unknown_answer("model_unknown")) == "refused"
    assert unanswerable_outcome("", unknown_answer("no_index")) == "refused"
    assert unanswerable_outcome("", unknown_answer("unverified")) == "unverified"
    assert unanswerable_outcome("", unknown_answer("bad_json")) == "format"
    assert unanswerable_outcome("", unknown_answer("truncated")) == "format"
    assert unanswerable_outcome("Не знаю, в документации нет.", None) == "refused"
    assert unanswerable_outcome("Борщ варят так: ...", None) == "answered"


# --- Day 24 review fixes (SPEC-w05d24 §9c) ------------------------------------------------


def test_eval_cite_pair_failure_on_the_cite_run_keeps_the_full_run_spend(
    fakes, out, err, judge, tmp_path
):
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    fakes.answer_rag = CITE_JSON
    fakes.fail_on_call = 2
    fakes.fail_count = 2
    rag_cli.run_eval(_questions(tmp_path, [Q1]), pair="full,cite", unanswerable=False)
    flat = _flat(out.getvalue())
    assert "Запланировано 1 · получено 0 · без ответа 1" in flat
    assert "Токены prompt/completion: RAG+rerank 100/5 · RAG+цитаты 0/0" in flat


def test_ask_cite_quotes_note_is_its_own_line_and_token_line_fits_80(fakes, out, err):
    fakes.answer_rag = CITE_JSON
    fakes.full_default = _full_ctx(hits=[_relevant_hit(rerank=9.0, rank=1)], passed=1)
    rag_cli.ask_question("Какое окно?", mode="cite")
    lines = [_plain(line) for line in err.getvalue().splitlines()]
    token_lines = [line for line in lines if line.startswith("RAG+цитаты:")]
    assert len(token_lines) == 1
    assert "цитаты: 1/1" not in token_lines[0] and token_lines[0].rstrip().endswith("1.5 с")
    assert "цитаты: 1/1 дословны" in lines
    assert all(cell_len(line) <= 80 for line in lines)


def test_ask_forwards_agent_warnings_to_stderr_once(fakes, out, err, monkeypatch):
    real = rag_cli.make_config

    def with_stop(*args, **kwargs):
        config = real(*args, **kwargs)
        config.params.stop = ["###"]
        return config

    monkeypatch.setattr(rag_cli, "make_config", with_stop)
    fakes.answer_rag = CITE_JSON
    rag_cli.ask_question("Какое окно?", mode="cite")
    text = _flat(err.getvalue())
    assert text.count("в режиме rag_cite stop не отправляется") == 1
    assert "max_context_length" not in text


def test_ask_without_cite_prints_no_unknown_window_noise(fakes, out, err):
    rag_cli.ask_question("Какое окно?", pair="plain,full")
    assert "max_context_length" not in _flat(err.getvalue())


def test_cite_answer_of_nested_brackets_survives_the_real_complete_and_is_journaled(
    fakes, out, err, monkeypatch
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    raw = "[" * 10000  # json.loads raises RecursionError on this, not JSONDecodeError
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason="stop")],
        model="m",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    @contextmanager
    def fake_client(config):
        yield SimpleNamespace(chat=SimpleNamespace(complete=lambda **kw: response))

    monkeypatch.setattr(rag_cli.chat_core, "complete", _REAL_COMPLETE)
    monkeypatch.setattr(chat_core, "mistral_client", fake_client)
    journaled: list[str] = []
    monkeypatch.setattr(
        rag_cli,
        "log_call",
        lambda result, messages, *, week, day, extra=None: journaled.append(result.text),
    )

    rag_cli.ask_question("Какое окно?", mode="cite")

    assert journaled == [raw]
    assert "Не знаю: ответ модели не удалось разобрать." in _plain(out.getvalue())
