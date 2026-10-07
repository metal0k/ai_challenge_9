"""week_06/cli.py (`adventlocal`): status, ask, compare, record scenario. No network, no credits.

Both clients are doubles: `lc.chat`/`lc.server_status` for the local server and
`cli.chat_core.complete` for the cloud. The journal is replaced by a recorder, so
nothing is written to logs/calls.jsonl.
"""

from __future__ import annotations

import io
import json
import sys
import time

import pytest
from rich.console import Console

import week_06.cli as cli
from advent_cli import record as record_mod
from advent_core import config as config_module
from advent_core import console
from advent_core.errors import NetworkError, ServerError
from advent_core.telemetry import CallResult, Usage
from week_06 import local_client as lc

GOOD_CODE = (
    "```python\n"
    "def is_palindrome(t):\n"
    "    s = [c.lower() for c in t if c.isalnum()]\n"
    "    return s == s[::-1]\n"
    "```"
)
ASCII_CODE = (
    "```python\n"
    "import re\n"
    "def is_palindrome(t):\n"
    "    s = re.sub(r'[^a-zA-Z0-9]', '', t).lower()\n"
    "    return s == s[::-1]\n"
    "```"
)

LOCAL_ANSWERS = {
    "Столица Австралии": "Канберра",
    "У Алисы": "Считаю.\nОТВЕТ: 3",
    "is_palindrome": GOOD_CODE,
}
CLOUD_ANSWERS = {
    "Столица Австралии": "Сидней",
    "У Алисы": "ОТВЕТ: 1",
    "is_palindrome": ASCII_CODE,
}


def _pick(answers: dict[str, str], prompt: str) -> str:
    return next(text for key, text in answers.items() if key in prompt)


def local_result(
    text: str, *, latency_ms=10000.0, ttft_ms=1000.0, usage="default"
) -> lc.LocalResult:
    if usage == "default":
        usage = Usage(
            prompt_tokens=20, completion_tokens=900, total_tokens=920, reasoning_tokens=800
        )
    return lc.LocalResult(
        text=text,
        reasoning_text="думаю",
        usage=usage,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        content_ms=ttft_ms + 100,
        finish_reason="stop",
        model="ornith",
    )


class Doubles:
    def __init__(self) -> None:
        self.local_calls: list[dict] = []
        self.cloud_calls: list[dict] = []
        self.journal: list[dict] = []
        self.models = [lc.LocalModel("ornith", "llm", "loaded", 57344, "Q4_K_M", "qwen35")]
        self.local_factory = lambda prompt: local_result(_pick(LOCAL_ANSWERS, prompt))
        self.cloud_factory = self._default_cloud
        self.cloud_error: Exception | None = None
        self.local_error: Exception | None = None

    @staticmethod
    def _default_cloud(prompt: str, config) -> CallResult:
        return CallResult(
            text=_pick(CLOUD_ANSWERS, prompt),
            model_requested=config.model,
            model_actual=config.model,
            usage=Usage(
                prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000
            ),
            latency_ms=2000,
            stream=False,
            finish_reason="stop",
        )

    def chat(self, url, model, messages, **kw):
        self.local_calls.append({"url": url, "model": model, "messages": messages, **kw})
        if self.local_error is not None:
            raise self.local_error
        res = self.local_factory(messages[-1]["content"])
        if kw.get("on_reasoning"):
            kw["on_reasoning"](res.reasoning_text)
        if kw.get("on_content"):
            kw["on_content"](res.text)
        return res

    def complete(self, config, messages, *args, **kw):
        self.cloud_calls.append({"config": config, "messages": messages})
        if self.cloud_error is not None:
            raise self.cloud_error
        return self.cloud_factory(messages[-1]["content"], config)

    def log_call(self, result, messages, **kw):
        self.journal.append({"result": result, "messages": messages, **kw})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    doubles = Doubles()
    monkeypatch.setattr(lc, "chat", doubles.chat)
    monkeypatch.setattr(lc, "server_status", lambda url=None, **kw: doubles.models)
    monkeypatch.setattr(cli.chat_core, "complete", doubles.complete)
    monkeypatch.setattr(cli, "log_call", doubles.log_call)
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.delenv("ADVENT_BASE_URL", raising=False)
    monkeypatch.delenv("ADVENT_LOCAL_URL", raising=False)
    monkeypatch.setattr(cli, "PROGRESS_INTERVAL", 3600.0)
    return doubles


@pytest.fixture
def out(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(
        console, "out", Console(file=buffer, width=80, force_terminal=False, color_system=None)
    )
    return buffer


@pytest.fixture
def err(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(
        console, "err", Console(file=buffer, width=80, force_terminal=False, color_system=None)
    )
    return buffer


def run_main(monkeypatch, *args: str, stdin: str | None = None) -> int:
    monkeypatch.setattr(sys, "argv", ["adventlocal", *args])
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    try:
        cli.main()
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_ready(monkeypatch, out, no_network):
    assert run_main(monkeypatch, "status") == 0
    text = out.getvalue()
    assert "ornith" in text and "loaded" in text and "57344" in text
    assert "готова" in text
    assert text.count("квантизация") == 1


def test_status_model_not_loaded_exits_2(monkeypatch, out, err, no_network):
    no_network.models = [lc.LocalModel("ornith", "llm", "not-loaded")]
    assert run_main(monkeypatch, "status") == 2
    assert "-Context 40960" in err.getvalue()


def test_status_model_missing_exits_2(monkeypatch, out, err, no_network):
    no_network.models = [lc.LocalModel("other", "llm", "loaded")]
    assert run_main(monkeypatch, "status") == 2


def test_status_server_down_exits_6(monkeypatch, out, err):
    def boom(url=None, **kw):
        raise NetworkError("не отвечает", hint=lc.START_HINT)

    monkeypatch.setattr(lc, "server_status", boom)
    assert run_main(monkeypatch, "status") == 6


def test_status_fallback_state_unknown(monkeypatch, out, no_network):
    no_network.models = [lc.LocalModel("ornith")]
    assert run_main(monkeypatch, "status") == 0
    assert "неизвестно" in out.getvalue()


def _three_models(no_network):
    no_network.models = [
        lc.LocalModel("ornith", "llm", "loaded"),
        lc.LocalModel("idle-one", "llm", "not-loaded"),
        lc.LocalModel("idle-two", "llm", "not-loaded"),
    ]


def test_status_hides_not_loaded_by_default(monkeypatch, out, no_network):
    _three_models(no_network)
    assert run_main(monkeypatch, "status") == 0
    text = out.getvalue()
    assert "ornith" in text
    assert "idle-one" not in text and "idle-two" not in text
    assert text.count("ещё 2 моделей не загружены (--all — показать)") == 1


def test_status_all_shows_not_loaded(monkeypatch, out, no_network):
    _three_models(no_network)
    assert run_main(monkeypatch, "status", "--all") == 0
    text = out.getvalue()
    assert "idle-one" in text and "idle-two" in text
    assert "не загружены" not in text


def test_status_fallback_shows_all_without_counter(monkeypatch, out, no_network):
    no_network.models = [lc.LocalModel("ornith"), lc.LocalModel("other")]
    assert run_main(monkeypatch, "status") == 0
    text = out.getvalue()
    assert "ornith" in text and "other" in text
    assert "не загружены" not in text


def test_status_lms_only_with_cli_flag(monkeypatch, out, no_network):
    calls = []
    monkeypatch.setattr(cli, "_lms_ps", lambda: calls.append(1) or "ornith  LOADED")
    run_main(monkeypatch, "status")
    assert calls == []
    run_main(monkeypatch, "status", "--cli")
    assert calls == [1]
    assert "через CLI:" in out.getvalue() and "ornith  LOADED" in out.getvalue()


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_question_from_stdin(monkeypatch, out, err, capsys, no_network):
    code = run_main(monkeypatch, "ask", stdin="Столица Австралии?\n")
    assert code == 0
    call = no_network.local_calls[0]
    assert call["messages"] == [{"role": "user", "content": "Столица Австралии?"}]
    assert call["temperature"] == 0.6 and call["top_p"] == 0.95 and call["max_tokens"] == 8192
    assert capsys.readouterr().out == "Канберра\n"
    stderr = err.getvalue()
    assert "Столица Австралии?" in stderr  # input echo
    assert "думаю" in stderr  # reasoning on stderr
    assert "TTFT 1.0 s" in stderr and "tokens 20/900 (reasoning 800)" in stderr
    assert "tok/s 100.0" in stderr  # 900 / (10 - 1) s


def test_ask_argument_wins_and_no_thinking_hides_reasoning(
    monkeypatch, out, err, capsys, no_network
):
    run_main(monkeypatch, "ask", "Столица Австралии?", "--no-show-thinking")
    assert "думаю" not in err.getvalue()
    assert capsys.readouterr().out == "Канберра\n"


def test_ask_journals_the_call(monkeypatch, out, err, no_network):
    run_main(monkeypatch, "ask", "Столица Австралии?")
    row = no_network.journal[0]
    assert row["week"] == 6 and row["day"] == 26
    assert row["extra"]["backend"] == "local" and row["extra"]["task"] == "ask"
    assert row["extra"]["reasoning_tokens"] == 800 and row["extra"]["ttft_ms"] == 1000.0
    assert row["result"].model_actual == "ornith"


def test_ask_empty_stdin_exits_2(monkeypatch, out, err, no_network):
    assert run_main(monkeypatch, "ask", stdin="  \n") == 2
    assert no_network.local_calls == []


def test_ask_server_error_is_journaled_and_exits(monkeypatch, out, err, no_network):
    no_network.local_error = ServerError("500")
    assert run_main(monkeypatch, "ask", "привет") == 5
    assert no_network.journal[0]["error"] == "500"


def test_ask_not_loaded_exits_2_without_calling(monkeypatch, out, err, no_network):
    no_network.models = [lc.LocalModel("ornith", "llm", "not-loaded")]
    assert run_main(monkeypatch, "ask", "привет") == 2
    assert no_network.local_calls == []


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_table_is_last_and_clean_at_width_80(monkeypatch, out, err, no_network):
    assert run_main(monkeypatch, "compare") == 0
    text = out.getvalue()
    assert "…" not in text
    assert text.count("задача") == 1  # header once
    table_start = text.index("Сводка")
    # nothing but the table (and its caption) after the title
    assert "── " not in text[table_start:]
    assert text.rstrip().splitlines()[-1] != ""
    assert text.index("── palindrome") < table_start
    # rows
    rows = [
        ln
        for ln in text[table_start:].splitlines()
        if ln.startswith("│") and ("локально" in ln or "облако" in ln)
    ]
    assert len(rows) == 6
    local_fact = next(ln for ln in rows if ln.lstrip("│ ").startswith("fact") and "локально" in ln)
    assert "0*" in local_fact and "✓" in local_fact
    assert "900 (800)" in next(ln for ln in rows if "alice" in ln and "локально" in ln)
    assert "электричество" in text[table_start:]
    box = [ln for ln in text[table_start:].splitlines() if ln[:1] in "┌│├└"]
    assert box and all(ln.rstrip()[-1] in "┐│┤┘" for ln in box)  # right border not clipped
    assert "(reasoning)" in text[table_start:]  # header not split mid-word


def _cloud_row_cells(text: str, task: str) -> list[str]:
    row = next(
        ln for ln in text[text.index("Сводка") :].splitlines() if "облако" in ln and task in ln
    )
    return [c.strip() for c in row.strip("│ ").split("│")]


def test_compare_cloud_cost_cell_with_cached_tokens_and_runs(monkeypatch, out, err, no_network):
    no_network.cloud_factory = lambda prompt, config: CallResult(
        text="Сидней",
        model_requested=config.model,
        usage=Usage(
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
            total_tokens=1_500_000,
            cached_tokens=400_000,
        ),
        latency_ms=2000,
        stream=False,
    )
    run_main(monkeypatch, "compare", "--tasks", "fact", "--runs", "2")
    text = out.getvalue()
    # per call, ministral-14b prices (0.2 in / 0.02 cached / 0.2 out per 1M):
    # 600k*0.2 + 400k*0.02 + 500k*0.2 = 228000 / 1e6 = 0.228; two runs = 0.456
    cells = _cloud_row_cells(text, "fact")
    assert cells[-1] == "0.456000"
    assert "цены на" in text


def test_compare_costs_and_prices_line(monkeypatch, out, err, no_network):
    run_main(monkeypatch, "compare", "--tasks", "fact")
    text = out.getvalue()
    # 1M prompt * 0.2 + 1M completion * 0.2 per 1M = $0.4
    assert _cloud_row_cells(text, "fact")[-1] == "0.400000"
    assert "цены на" in text


def test_compare_unknown_price_is_dash(monkeypatch, out, err, no_network):
    run_main(monkeypatch, "compare", "--tasks", "fact", "--cloud-model", "no-such-model")
    text = out.getvalue()
    cloud_row = next(ln for ln in text.splitlines() if "облако" in ln and "fact" in ln)
    assert cloud_row.rstrip("│ ").endswith("—")
    assert "$0 против —" in text


def test_compare_conclusions_from_numbers(monkeypatch, out, err, no_network):
    run_main(monkeypatch, "compare")
    flat = " ".join(out.getvalue().split())
    # local 10 s per task x3 against cloud 2 s per task x3 -> 5.0x slower
    assert "локальная медленнее в 5.0×" in flat
    assert "точнее локальная: локальная 3/3, облако 0/3" in flat
    assert "$0 против $1.200000" in flat
    assert "Один прогон" in flat


def test_compare_tie_names_no_leader(monkeypatch, out, err, no_network):
    no_network.cloud_factory = lambda prompt, config: CallResult(
        text=_pick(LOCAL_ANSWERS, prompt),
        model_requested=config.model,
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        latency_ms=10000,
        stream=False,
    )
    run_main(monkeypatch, "compare")
    flat = " ".join(out.getvalue().split())
    assert "по точности ничья: 3/3 против 3/3" in flat
    assert "по времени ничья" in flat
    assert "точнее" not in flat and "медленнее" not in flat and "быстрее" not in flat


def test_compare_no_cloud_is_local_only_summary(monkeypatch, out, err, no_network):
    assert run_main(monkeypatch, "compare", "--no-cloud") == 0
    assert no_network.cloud_calls == []
    text = out.getvalue()
    assert "облако" not in text
    flat = " ".join(text.split())
    assert "Локально: верно 3/3" in flat
    assert "медленнее" not in flat and "точнее" not in flat
    assert text.count("задача") == 1


def test_compare_cloud_error_does_not_abort(monkeypatch, out, err, no_network):
    no_network.cloud_error = ServerError("облако легло")
    assert run_main(monkeypatch, "compare") == 0
    text = out.getvalue()
    assert text.count("ошибка") == 3  # one cell per task
    assert "облако легло" in err.getvalue()
    assert len(no_network.local_calls) == 3
    errors = [r for r in no_network.journal if r["error"]]
    assert len(errors) == 3 and all(r["extra"]["backend"] == "cloud" for r in errors)


def test_compare_local_error_keeps_cloud(monkeypatch, out, err, no_network):
    no_network.local_error = ServerError("local 500")
    assert run_main(monkeypatch, "compare", "--tasks", "fact") == 0
    assert len(no_network.cloud_calls) == 1


def test_compare_all_calls_failed_exits_1(monkeypatch, out, err, no_network):
    no_network.local_error = ServerError("local 500")
    no_network.cloud_error = ServerError("cloud 500")
    assert run_main(monkeypatch, "compare", "--tasks", "fact") == 1
    assert "ни один вызов не удался" in err.getvalue()
    assert "Сводка" in out.getvalue()


def test_compare_without_key_skips_cloud_with_warning(monkeypatch, out, err, no_network):
    monkeypatch.delenv("MISTRAL_API_KEY")
    monkeypatch.setenv("ADVENT_BASE_URL", "http://127.0.0.1:1234")
    assert run_main(monkeypatch, "compare", "--tasks", "fact") == 0
    assert no_network.cloud_calls == []
    assert "MISTRAL_API_KEY не задан" in err.getvalue()


def test_compare_cloud_ignores_advent_base_url(monkeypatch, out, err, no_network):
    monkeypatch.setenv("ADVENT_BASE_URL", "http://127.0.0.1:1234")
    run_main(monkeypatch, "compare", "--tasks", "fact")
    assert no_network.cloud_calls[0]["config"].base_url is None


def test_compare_sent_params(monkeypatch, out, err, no_network):
    monkeypatch.setenv("MISTRAL_TEMPERATURE", "1.2")
    monkeypatch.setenv("MISTRAL_MAX_TOKENS", "50")
    run_main(monkeypatch, "compare", "--tasks", "fact", "--cloud-model", "ministral-14b-latest")
    cloud = no_network.cloud_calls[0]
    config = cloud["config"]
    assert [m["role"] for m in cloud["messages"]] == ["user"]  # no persona
    assert config.system_prompt_path is None and config.system_prompt() is None
    assert config.params.max_tokens == 8192 and config.params.temperature is None
    assert config.model == "ministral-14b-latest" and config.stream is False
    local = no_network.local_calls[0]
    assert [m["role"] for m in local["messages"]] == ["user"]
    assert local["messages"] == cloud["messages"]
    assert local["max_tokens"] == 8192 and local["temperature"] == 0.6
    assert local["top_p"] == 0.95


def test_compare_journal_rows(monkeypatch, out, err, no_network):
    run_main(monkeypatch, "compare", "--tasks", "palindrome")
    assert len(no_network.journal) == 2
    local_row, cloud_row = no_network.journal
    assert local_row["week"] == 6 and local_row["day"] == 26
    assert local_row["extra"]["backend"] == "local" and local_row["extra"]["task"] == "palindrome"
    assert local_row["extra"]["verdict"] == "ok" and local_row["extra"]["run"] == 1
    assert cloud_row["extra"]["backend"] == "cloud"
    assert cloud_row["extra"]["verdict"] == "partial"


def test_compare_shows_code_whole_and_verdict(monkeypatch, out, err, no_network):
    run_main(monkeypatch, "compare", "--tasks", "palindrome")
    text = out.getvalue()
    assert "def is_palindrome(t):" in text and "✓ 7/7" in text
    assert "✗ 5/7" in text


def test_compare_runs_aggregate_k_of_n(monkeypatch, out, err, no_network):
    answers = iter(["Канберра", "Сидней", "Канберра"])
    no_network.local_factory = lambda prompt: local_result(next(answers))
    run_main(monkeypatch, "compare", "--tasks", "fact", "--no-cloud", "--runs", "3")
    text = out.getvalue()
    assert "2/3" in text[text.index("Сводка") :]
    assert "прогон 3/3" in err.getvalue()
    assert "прогон" not in text
    assert len(no_network.local_calls) == 3


def test_compare_runs_with_errors_keep_them_in_the_denominator(monkeypatch, out, err, no_network):
    """One correct and two errored local calls is 1/3, not 1/1 (review #4)."""
    calls = iter([None, "ERR", "ERR"])

    def factory(prompt):
        item = next(calls)
        if item == "ERR":
            raise ServerError("упал")
        return local_result("Канберра")

    def chat(url, model, messages, **kw):
        no_network.local_calls.append(kw)
        return factory(messages[-1]["content"])

    monkeypatch.setattr(lc, "chat", chat)
    run_main(monkeypatch, "compare", "--tasks", "fact", "--no-cloud", "--runs", "3")
    text = out.getvalue()
    table = text[text.index("Сводка") :]
    assert "1/3 (ошибок 2)" in table
    assert "верно 1/3 (ошибок 2)" in " ".join(text.split())
    assert "1/1" not in text


def test_compare_speed_uses_only_tasks_both_sides_completed(monkeypatch, out, err, no_network):
    """Review #5: local fact 1 s + palindrome 100 s, cloud fact 2 s and a failed palindrome."""
    times = {"Столица Австралии": 1000.0, "is_palindrome": 100000.0}
    no_network.local_factory = lambda prompt: local_result(
        _pick(LOCAL_ANSWERS, prompt),
        latency_ms=next(v for k, v in times.items() if k in prompt)
        if any(k in prompt for k in times)
        else 5000.0,
    )
    base = no_network.cloud_factory

    def cloud(prompt, config):
        if "is_palindrome" in prompt:
            raise ServerError("облако легло")
        return base(prompt, config)

    no_network.cloud_factory = cloud
    run_main(monkeypatch, "compare", "--tasks", "fact,palindrome")
    flat = " ".join(out.getvalue().split())
    assert "50.5" not in flat
    assert "локальная быстрее в 2.0×" in flat
    assert "задачи, где ответили обе стороны: fact" in flat


def test_compare_no_common_tasks_names_no_speed_direction(monkeypatch, out, err, no_network):
    no_network.cloud_error = ServerError("облако легло")
    run_main(monkeypatch, "compare", "--tasks", "fact")
    flat = " ".join(out.getvalue().split())
    assert "медленнее" not in flat and "быстрее" not in flat
    assert "время сравнить нечем" in flat


def test_compare_truncated_nonempty_answer_still_shows_the_cutoff(
    monkeypatch, out, err, no_network
):
    def cut(prompt):
        res = local_result("Канберра")
        res.finish_reason = None
        res.truncated = True
        return res

    no_network.local_factory = cut
    run_main(monkeypatch, "compare", "--tasks", "fact", "--no-cloud")
    text = out.getvalue()
    assert "✓ верно" in text  # content check is independent of the cutoff
    assert "обрыв: поток закончился раньше времени" in text


def test_compare_palindrome_shows_own_asserts_independently(monkeypatch, out, err, no_network):
    """Review #7: correct function, failing own assert -> 7/7 AND «свои тесты: упали»."""
    code = GOOD_CODE.replace("\n```", "\nassert not is_palindrome('aba')\n```")
    no_network.local_factory = lambda prompt: local_result(code)
    run_main(monkeypatch, "compare", "--tasks", "palindrome", "--no-cloud")
    text = out.getvalue()
    assert "✓ 7/7" in text
    assert "свои тесты: упали" in text


def test_compare_palindrome_own_asserts_ok_and_none(monkeypatch, out, err, no_network):
    run_main(monkeypatch, "compare", "--tasks", "palindrome", "--no-cloud")
    assert "свои тесты: нет" in out.getvalue()
    code = GOOD_CODE.replace("\n```", "\nassert is_palindrome('aba')\n```")
    no_network.local_factory = lambda prompt: local_result(code)
    out.truncate(0)
    out.seek(0)
    run_main(monkeypatch, "compare", "--tasks", "palindrome", "--no-cloud")
    assert "свои тесты: ок" in out.getvalue()


def test_compare_unknown_usage_gives_dash_for_tps(monkeypatch, out, err, no_network):
    no_network.local_factory = lambda prompt: local_result("Канберра", usage=None)
    run_main(monkeypatch, "compare", "--tasks", "fact", "--no-cloud")
    row = next(ln for ln in out.getvalue().splitlines() if "локально" in ln and "fact" in ln)
    assert row.count("—") >= 2  # tokens and tok/s both unknown


def test_compare_unknown_task_exits_2(monkeypatch, out, err, no_network):
    assert run_main(monkeypatch, "compare", "--tasks", "nope") == 2
    assert no_network.local_calls == []


def test_compare_progress_ticker_prints_to_stderr(monkeypatch, out, err, no_network):
    monkeypatch.setattr(cli, "PROGRESS_INTERVAL", 0.05)

    def slow(prompt):
        time.sleep(0.3)
        return local_result("Канберра")

    no_network.local_factory = slow
    run_main(monkeypatch, "compare", "--tasks", "fact", "--no-cloud")
    assert "… " in err.getvalue() and "думает" in err.getvalue()
    assert "думает" not in out.getvalue()


def test_compare_cutoff_with_empty_content_is_wrong(monkeypatch, out, err, no_network):
    def cut(prompt):
        res = local_result("")
        res.finish_reason = "length"
        return res

    no_network.local_factory = cut
    run_main(monkeypatch, "compare", "--tasks", "fact", "--no-cloud")
    assert "reasoning съел max_tokens" in out.getvalue()


def test_compare_does_not_write_the_real_journal(monkeypatch, out, err, no_network, tmp_path):
    # the autouse double replaced log_call: no file appears under logs/ from this run
    json.dumps([r["extra"] for r in no_network.journal])
    run_main(monkeypatch, "compare", "--tasks", "fact", "--no-cloud")
    assert len(no_network.journal) == 1


# ---------------------------------------------------------------------------
# record scenario
# ---------------------------------------------------------------------------


def test_module_command_caption():
    assert record_mod.MODULE_COMMANDS["week_06.cli"] == "adventlocal"


def test_demo_steps_w06d26():
    steps = record_mod.demo_steps(6, 26)
    assert all(s.module == "week_06.cli" for s in steps)
    assert [s.args[0] for s in steps] == ["status", "ask", "compare"]
    assert steps[1].args[1] == "Столица Австралии? Ответь одним словом."
    compare = steps[-1]
    assert compare.args[0] == "compare" and "--deadline" in compare.args
    deadline = float(compare.args[compare.args.index("--deadline") + 1])
    assert compare.timeout > 3 * deadline + 60  # three local calls + cloud + codecheck
    ask = steps[1]
    assert ask.timeout > float(ask.args[ask.args.index("--deadline") + 1])
    assert all("--cli" not in s.args for s in steps)  # lms ps never in the demo


def test_rehearsal_steps_w06d26_status_first_then_cyrillic_ask_via_stdin():
    steps = record_mod.rehearsal_steps(6, 26)
    assert [s.args[0] for s in steps] == ["status", "ask"]
    assert steps[1].args[0] == "ask" and "--deadline" in steps[1].args
    assert steps[1].timeout > float(steps[1].args[steps[1].args.index("--deadline") + 1])
    assert any(ord(ch) > 127 for ch in steps[1].stdin_lines[0])
    assert all("--cli" not in s.args for s in steps)
    assert all(not s.expect_failure for s in steps)
