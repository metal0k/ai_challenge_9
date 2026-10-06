"""Day 27: adventagent entirely on a local LLM (REPL, agent plumbing, MCP env, demo).

No network: LM Studio is a double (`openai_compat.server_status`), the model calls are
fakes, the MCP child is a real local subprocess that only echoes its environment.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import week_02.cli as cli
from advent_core import console, mcp_client, mcp_router, offline, openai_compat
from advent_core import journal as journal_module
from advent_core.agent import Agent
from advent_core.config import LOCAL_API_KEY, Config
from advent_core.errors import AdventError, CloudBlockedError, ConfigurationError, MCPError
from advent_core.openai_compat import LocalModel
from advent_core.params import AGENT_COMMAND, GenerationParams
from advent_core.rag import RagContext
from advent_core.telemetry import CallResult, Usage
from advent_core.tokens import EstimateCounter

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
URL = "http://127.0.0.1:1234"
NOMIC = "text-embedding-nomic-embed-text-v1.5"
READY = [
    LocalModel("ornith", "llm", "loaded", 57344),
    LocalModel(NOMIC, "embeddings", "loaded", 2048),
    LocalModel("other", "llm", "not-loaded", None),
]
CARDS = [{"id": "ornith"}, {"id": "other"}, {"id": NOMIC}]


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", _ANSI.sub("", text))


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    monkeypatch.setattr(cli, "log_call", lambda result, messages, **kwargs: None)
    monkeypatch.setattr(cli, "log_internal_call", lambda result, **kwargs: None)
    monkeypatch.setattr(cli, "list_models", lambda config: CARDS)
    monkeypatch.setattr(cli, "counter_for", lambda model, **kwargs: (EstimateCounter(), None))
    monkeypatch.setattr(cli, "MCP_REGISTRY_PATH", Path("no-such-registry.json"))
    monkeypatch.setattr(openai_compat, "server_status", lambda url=None, **kw: list(READY))
    monkeypatch.delenv("ADVENT_MCP_TIMEOUT", raising=False)


def _config(*, local=True, model="ornith", stream=False, **params) -> Config:
    config = Config(
        api_key=LOCAL_API_KEY if local else "k" * 32,
        model=model,
        system_prompt_path=None,
        params=GenerationParams.build(**params),
        base_url=URL if local else None,
        offline=local,
    )
    config.params.apply_defaults(AGENT_COMMAND)
    config.stream = stream
    return config


def _reply(text="ответ", **kwargs):
    return CallResult(
        text=text,
        model_requested="ornith",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        finish_reason="stop",
        **kwargs,
    )


def _complete(config, messages, capabilities=None, **kwargs):
    return _reply(sent_messages=messages)


def _shell(monkeypatch, tmp_path, *, local=True, stream_fake=None, **kw):
    monkeypatch.setattr(cli.chat_core, "complete", _complete)
    if stream_fake is not None:
        monkeypatch.setattr(cli.chat_core, "stream", stream_fake)
    config = _config(local=local, stream=stream_fake is not None, **kw.pop("params", {}))
    return cli.AgentShell(config, directory=tmp_path, **kw)


# --- readiness and the window -----------------------------------------------


def test_local_shell_takes_the_window_from_the_server_not_the_card(monkeypatch, tmp_path):
    shell = _shell(monkeypatch, tmp_path)

    assert shell.agent.context_limit == 57344
    assert shell.local_model.loaded_context_length == 57344


def test_context_limit_override_beats_the_server_window(monkeypatch, tmp_path):
    shell = _shell(monkeypatch, tmp_path, params={"context_limit": 2500})

    assert shell.agent.context_limit == 2500


@pytest.mark.parametrize(
    "listed",
    [
        [LocalModel("ornith", "llm", "not-loaded", None)],
        [LocalModel("ornith", "llm", None, None)],
        [LocalModel("другая", "llm", "loaded", 4096)],
    ],
    ids=["not-loaded", "no-state-fallback", "model-missing"],
)
def test_local_mode_refuses_before_the_first_turn(monkeypatch, tmp_path, listed):
    monkeypatch.setattr(openai_compat, "server_status", lambda url=None, **kw: listed)

    with pytest.raises(ConfigurationError) as info:
        _shell(monkeypatch, tmp_path)

    assert info.value.exit_code == 2
    assert "start-local-llm.ps1" in info.value.hint


def test_bare_base_url_is_not_strict_but_still_gets_the_window(monkeypatch, tmp_path):
    config = _config(local=False)
    config.base_url = URL
    monkeypatch.setattr(cli.chat_core, "complete", _complete)

    shell = cli.AgentShell(config, directory=tmp_path)

    assert shell.is_local is False
    assert shell.agent.context_limit == 57344

    def down(url=None, **kw):
        raise AdventError("сервер не отвечает")

    monkeypatch.setattr(openai_compat, "server_status", down)
    shell.retarget()  # a dead server is no reason to die outside --local
    assert shell.local_model is None


def test_cloud_blocked_error_is_not_swallowed_by_refresh(monkeypatch, tmp_path):
    shell = _shell(monkeypatch, tmp_path)

    def blocked(config):
        raise CloudBlockedError("облако отключено (--local): list_models")

    monkeypatch.setattr(cli, "list_models", blocked)

    with pytest.raises(CloudBlockedError):
        shell.refresh()


def test_model_retarget_rechecks_readiness_and_rolls_back(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path)

    cli._dispatch("/model other", shell)

    err = _flat(capsys.readouterr().err)
    assert "не загружена" in err
    assert shell.config.model == "ornith"
    assert shell.agent.context_limit == 57344


# --- /local, start line, flags ----------------------------------------------


def test_local_command_prints_the_proof_screen_to_stdout(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path)
    capsys.readouterr()

    cli._dispatch("/local", shell)

    captured = capsys.readouterr()
    assert captured.err == ""
    out = captured.out
    assert "сервер: 127.0.0.1:1234" in out
    assert "чат-модель: ornith, loaded" in out
    assert f"embed-модели: {NOMIC} (loaded)" in out
    assert "окно: 57344 токенов" in out
    assert "вызовов: локальных 0, облачных 0 (заблокировано 0)" in out
    assert "токены сессии:" in out
    assert re.search(r"время сессии: \d+ с", out)


@pytest.mark.allow_cloud_attempts
def test_local_command_counts_a_blocked_cloud_attempt(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path)
    offline.enable()
    with pytest.raises(CloudBlockedError):
        offline.assert_cloud_allowed("RAG rewrite")
    capsys.readouterr()

    cli._dispatch("/local", shell)

    assert "облачных 1 (заблокировано 1)" in capsys.readouterr().out


def test_local_command_outside_local_mode_says_so_on_stdout(monkeypatch, tmp_path, capsys):
    shell = _shell(monkeypatch, tmp_path, local=False)
    capsys.readouterr()

    cli._dispatch("/local", shell)

    assert "локальный режим выключен" in capsys.readouterr().out


def _run_main(monkeypatch, argv, *, shell_cls=None, env=None):
    created: list = []

    class FakeShell:
        is_local = True

        def __init__(self, config, **kwargs):
            self.config = config
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(cli, "AgentShell", shell_cls or FakeShell)
    monkeypatch.setattr(cli, "_loop", lambda shell: None)
    monkeypatch.setattr(sys, "argv", ["adventagent", *argv])
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    try:
        cli.main()
    except SystemExit as exit_:
        if exit_.code not in (0, None):  # typer exits 0 after a normal run
            raise
    return created[0]


def test_local_flag_resolves_url_model_offline_and_prints_the_start_line(monkeypatch, capsys):
    monkeypatch.setenv("ADVENT_LOCAL_URL", "http://127.0.0.1:4321")

    shell = _run_main(monkeypatch, ["--local"])

    config = shell.config
    assert config.base_url == "http://127.0.0.1:4321"
    assert config.model == "ornith"
    assert config.offline is True
    assert config.api_key == "lm-studio-local"
    assert "локальный режим: ornith @ 127.0.0.1:4321, облако отключено" in _flat(
        capsys.readouterr().err
    )


def test_base_url_flag_beats_the_env_url_and_model_flag_beats_the_default(monkeypatch):
    monkeypatch.setenv("ADVENT_LOCAL_URL", "http://127.0.0.1:4321")

    shell = _run_main(monkeypatch, ["--local", "--base-url", URL, "--model", "other"])

    assert shell.config.base_url == URL
    assert shell.config.model == "other"


@pytest.mark.parametrize("via", ["flag", "env"])
def test_non_loopback_local_url_exits_2_before_any_readiness_probe(monkeypatch, via):
    def boom(url=None, **kw):
        raise AssertionError("readiness must not be probed")

    monkeypatch.setattr(openai_compat, "server_status", boom)
    argv = ["--local"]
    if via == "flag":
        argv += ["--base-url", "http://example.com:1234"]
    else:
        monkeypatch.setenv("ADVENT_LOCAL_URL", "http://example.com:1234")

    with pytest.raises(SystemExit) as info:
        _run_main(monkeypatch, argv)

    assert info.value.code == 2


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:bad", "http://127.0.0.1@evil.com", "ftp://127.0.0.1:1234"]
)
def test_malformed_local_url_exits_2_without_a_traceback(monkeypatch, capsys, url):
    def boom(url=None, **kw):
        raise AssertionError("readiness must not be probed")

    monkeypatch.setattr(openai_compat, "server_status", boom)

    with pytest.raises(SystemExit) as info:
        _run_main(monkeypatch, ["--local", "--base-url", url])

    assert info.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_local_flag_reads_the_url_from_dot_env_and_strips_v1(monkeypatch, tmp_path):
    from advent_core import config as config_module

    (tmp_path / ".env").write_text("ADVENT_LOCAL_URL=http://127.0.0.1:4321/v1\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ADVENT_LOCAL_URL", "placeholder")
    monkeypatch.delenv("ADVENT_LOCAL_URL")  # not in the process env: only .env has it

    shell = _run_main(monkeypatch, ["--local"])

    assert shell.config.base_url == "http://127.0.0.1:4321"


def test_no_thinking_and_rag_db_flags_reach_the_shell(monkeypatch, tmp_path):
    db = tmp_path / "x.sqlite3"

    shell = _run_main(monkeypatch, ["--local", "--no-thinking", "--rag-db", str(db)])

    assert shell.config.params.show_thinking is False
    assert shell.kwargs["rag_db"] == db
    assert "show_thinking" in shell.kwargs["explicit"]


def test_show_thinking_defaults_on_and_is_a_set_param(monkeypatch, tmp_path):
    shell = _shell(monkeypatch, tmp_path)
    assert shell.config.params.show_thinking is True

    cli._dispatch("/set show_thinking off", shell)

    assert shell.config.params.show_thinking is False


# --- show_thinking survives a restart and a session switch (review finding 10) ----


def _saved(tmp_path, name="default", **state):
    from advent_core.session import Session

    session = Session.new(name, directory=tmp_path)
    session.state.update(state)
    session.save()
    return session


def test_show_thinking_off_is_written_to_the_session_and_restored(monkeypatch, tmp_path):
    from advent_core.session import Session

    shell = _shell(monkeypatch, tmp_path)
    cli._dispatch("/set show_thinking off", shell)

    assert Session.load("default", directory=tmp_path).state["show_thinking"] is False
    again = _shell(monkeypatch, tmp_path)
    assert again.config.params.show_thinking is False


def test_no_thinking_flag_beats_the_saved_value(monkeypatch, tmp_path):
    _saved(tmp_path, show_thinking=True)
    explicit = cli._explicit_local_flags(mode=None, done=None, context_limit=None, no_thinking=True)
    assert explicit == {"show_thinking"}
    assert cli._explicit_local_flags(mode=None, done=None, context_limit=None) == frozenset()

    shell = _shell(monkeypatch, tmp_path, params={"show_thinking": False}, explicit=explicit)

    assert shell.config.params.show_thinking is False


def test_show_thinking_three_cases_on_a_session_switch(monkeypatch, tmp_path):
    _saved(tmp_path, "off", show_thinking=False)
    _saved(tmp_path, "nul", show_thinking=None)
    _saved(tmp_path, "absent")
    shell = _shell(monkeypatch, tmp_path)
    assert shell.config.params.show_thinking is True

    cli._dispatch("/set session off", shell)
    assert shell.config.params.show_thinking is False  # a bool applies
    cli._dispatch("/set session absent", shell)
    assert shell.config.params.show_thinking is False  # absent leaves what is loaded
    cli._dispatch("/set session nul", shell)
    assert shell.config.params.show_thinking is True  # null resets to the command default


@pytest.mark.parametrize("bad", ["off", 0, [False]])
def test_show_thinking_garbage_in_a_session_file_warns(monkeypatch, tmp_path, capsys, bad):
    _saved(tmp_path, show_thinking=bad)

    shell = _shell(monkeypatch, tmp_path)

    assert shell.config.params.show_thinking is True
    assert f"негодное show_thinking={bad!r}" in _flat(capsys.readouterr().err)


# --- reasoning: stderr only, never history -----------------------------------


def _reasoning_stream(seen):
    def stream(config, messages, on_chunk, capabilities=None, *, on_reasoning=None):
        seen.append(on_reasoning)
        if on_reasoning is not None:
            on_reasoning("взвешиваю ")
            on_reasoning("варианты")
        on_chunk("Ответ.")
        return _reply("Ответ.", stream=True, sent_messages=messages)

    return stream


def test_reasoning_goes_to_stderr_between_label_and_answer_and_not_into_history(
    monkeypatch, tmp_path, capsys
):
    seen: list = []
    shell = _shell(monkeypatch, tmp_path, stream_fake=_reasoning_stream(seen))
    capsys.readouterr()

    cli._turn(shell, "привет")

    captured = capsys.readouterr()
    assert captured.out.strip() == "Ответ."
    err = _ANSI.sub("", captured.err)
    assert "взвешиваю варианты" in err
    assert err.index("агент ›") < err.index("взвешиваю") < err.index("model ")
    assert "взвешиваю" not in captured.out
    assert all("взвешиваю" not in m["content"] for m in shell.history)
    assert all("взвешиваю" not in turn.content for turn in shell.session.turns)


def test_show_thinking_off_passes_no_callback_and_prints_no_reasoning(
    monkeypatch, tmp_path, capsys
):
    seen: list = []
    shell = _shell(monkeypatch, tmp_path, stream_fake=_reasoning_stream(seen))
    cli._dispatch("/set show_thinking off", shell)
    capsys.readouterr()

    cli._turn(shell, "привет")

    assert seen == [None]
    assert "взвешиваю" not in capsys.readouterr().err


def test_cloud_turn_calls_stream_without_the_reasoning_kwarg(monkeypatch, tmp_path):
    kwargs_seen: list = []

    def old_style_stream(config, messages, on_chunk, capabilities=None, **kwargs):
        kwargs_seen.append(kwargs)
        on_chunk("ок")
        return _reply("ок", stream=True, sent_messages=messages)

    shell = _shell(monkeypatch, tmp_path, local=False, stream_fake=old_style_stream)

    cli._turn(shell, "привет")

    assert kwargs_seen == [{}]


def test_agent_ask_forwards_on_reasoning_only_when_set():
    calls: list = []

    def stream(config, messages, on_chunk, capabilities=None, **kwargs):
        calls.append(kwargs)
        on_chunk("x")
        return _reply("x", stream=True)

    agent = Agent(_config(stream=True), complete=_complete, stream=stream)

    agent.ask("привет", [], on_chunk=lambda text: None)
    cb = lambda text: None  # noqa: E731
    agent.ask("привет", [], on_chunk=lambda text: None, on_reasoning=cb)

    assert calls == [{}, {"on_reasoning": cb}]


def test_dim_reasoning_is_closed_before_an_empty_answer_footer(capsys):
    chunks = console.LabelledChunks()

    chunks.reasoning("думаю")
    chunks.close_reasoning()
    chunks.close_reasoning()  # idempotent

    err = _ANSI.sub("", capsys.readouterr().err)
    assert err.count("агент ›") == 1
    assert err.endswith("думаю\n")


# --- progress ticker for aux calls -------------------------------------------


def test_ticker_prints_a_pulse_on_stderr_and_stops_with_the_block(capsys):
    with console.ticker("rerank", interval=0.02):
        time.sleep(0.15)
    after = _ANSI.sub("", capsys.readouterr().err)
    time.sleep(0.1)

    assert re.search(r"… \d+ s \(rerank\)", after)
    assert capsys.readouterr().err == ""


def test_local_agent_gets_the_ticker_and_cloud_agent_does_not(monkeypatch, tmp_path):
    assert _shell(monkeypatch, tmp_path).agent.progress is console.ticker
    assert _shell(monkeypatch, tmp_path, local=False).agent.progress is None


def test_agent_labels_the_aux_calls_it_wraps():
    labels: list[str] = []

    @contextmanager
    def progress(label):
        labels.append(f"+{label}")
        yield
        labels.append(f"-{label}")

    def retrieve(question, settings):
        labels.append("retrieve")
        return RagContext(
            hits=(),
            strategy="structure",
            k=5,
            embed_model="m",
            embed_tokens=1,
            corpus_rev="0" * 16,
            dropped=0,
        )

    config = _config(rag=True)
    agent = Agent(config, complete=_complete, stream=None, retrieve=retrieve)
    agent.progress = progress

    agent.ask("вопрос", [])

    assert labels == ["+rag", "retrieve", "-rag", "+answer", "-answer"]


# --- RAG target: local index and the agent's own model for aux ---------------


def test_local_shell_points_the_retriever_at_the_local_index_and_its_own_config(
    monkeypatch, tmp_path
):
    captured: dict = {}
    monkeypatch.setattr(
        cli, "_rag_retriever", lambda **kw: captured.update(kw) or (lambda q, s: None)
    )

    shell = _shell(monkeypatch, tmp_path)

    assert captured["db_path"] == cli.LOCAL_RAG_DB
    assert captured["db_path"].as_posix().endswith("data/rag/index.local.sqlite3")
    assert captured["aux_config"] is shell.config


def test_rag_db_flag_overrides_the_local_default_and_cloud_gets_only_the_override(
    monkeypatch, tmp_path
):
    seen: list[dict] = []
    monkeypatch.setattr(cli, "_rag_retriever", lambda **kw: seen.append(kw) or (lambda q, s: None))
    db = tmp_path / "mine.sqlite3"

    local = _shell(monkeypatch, tmp_path, rag_db=db)
    _shell(monkeypatch, tmp_path, local=False, rag_db=db)
    plain = _shell(monkeypatch, tmp_path, local=False)

    assert seen[0] == {"db_path": db, "aux_config": local.config}
    assert seen[1] == {"db_path": db}
    assert seen[2] == {}
    assert plain.rag_target() == {}


def test_index_check_in_local_mode_gets_the_local_db(monkeypatch, tmp_path):
    shell = _shell(monkeypatch, tmp_path)
    checks: list = []

    def check(strategy, **kwargs):
        checks.append((strategy, kwargs))
        return SimpleNamespace(corpus_rev="0" * 16, n_chunks=1, model="m")

    monkeypatch.setattr(cli, "_rag_check_index", check)

    cli._dispatch("/rag plain", shell)

    assert checks[0] == ("structure", {"db_path": cli.LOCAL_RAG_DB})


def test_make_retriever_receives_db_and_aux_config_only_when_set(monkeypatch):
    import week_05.rag as rag

    calls: list[dict] = []

    def fake_make(db_path=None, *, week=5, day=22, aux_day=22, **kwargs):
        calls.append({"db_path": db_path, "day": day, **kwargs})
        return lambda question, settings: "ctx"

    monkeypatch.setattr(rag, "make_retriever", fake_make)
    config = _config()
    settings = SimpleNamespace(cite=False, rewrite=False, rerank=False)

    cli._rag_retriever()("q", settings)
    cli._rag_retriever(db_path=Path("x.sqlite3"), aux_config=config)("q", settings)

    assert calls[0] == {"db_path": None, "day": 22}
    assert calls[1] == {"db_path": Path("x.sqlite3"), "day": 22, "aux_config": config}


# --- MCP: explicit child env, offline npx, timeout ----------------------------


def test_mcp_env_in_local_mode_is_explicit_and_blanks_the_cloud_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTRAL_API_KEY", "cloud-key-from-dotenv")
    shell = _shell(monkeypatch, tmp_path)

    assert cli._mcp_env(shell) == {
        "ADVENT_OFFLINE": "1",
        "ADVENT_BASE_URL": URL,
        "ADVENT_SUMMARIZE_MODEL": "ornith",
        "MISTRAL_API_KEY": "",
    }
    assert cli._mcp_env(_shell(monkeypatch, tmp_path, local=False)) is None


def test_the_real_child_sees_the_offline_env_and_not_the_parents_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTRAL_API_KEY", "cloud-key-from-dotenv")
    script = tmp_path / "envprobe.py"
    script.write_text(
        "import json, os\n"
        "from mcp.server import MCPServer\n"
        "mcp = MCPServer('envprobe', version='0.1.0', log_level='WARNING')\n"
        "@mcp.tool()\n"
        "def show_env() -> str:\n"
        "    keys = ['ADVENT_OFFLINE', 'ADVENT_BASE_URL', 'ADVENT_SUMMARIZE_MODEL',\n"
        "            'MISTRAL_API_KEY']\n"
        "    return json.dumps({k: os.environ.get(k) for k in keys})\n"
        "mcp.run()\n",
        encoding="utf-8",
    )
    shell = _shell(monkeypatch, tmp_path)
    env = cli._mcp_env(shell)

    outcome = mcp_client.call_tool_once(
        [sys.executable, str(script)], "show_env", {}, timeout=60.0, env=env
    )

    import json

    assert json.loads(outcome.text) == {
        "ADVENT_OFFLINE": "1",
        "ADVENT_BASE_URL": URL,
        "ADVENT_SUMMARIZE_MODEL": "ornith",
        "MISTRAL_API_KEY": "",
    }


def test_week04_server_switches_its_own_guard_on_from_the_env():
    code = "import week_04.server; from advent_core import offline; print(offline.is_enabled())"
    root = Path(cli.__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**__import__("os").environ, "ADVENT_OFFLINE": "1"},
        timeout=120,
    )

    assert out.stdout.strip().splitlines()[-1] == "True"


def test_summarize_text_uses_the_env_model_url_and_a_reasoning_sized_budget(monkeypatch):
    import week_04.server as server

    monkeypatch.setenv("ADVENT_SUMMARIZE_MODEL", "ornith")
    monkeypatch.setenv("ADVENT_BASE_URL", URL)
    monkeypatch.setenv("MISTRAL_API_KEY", "cloud-key-from-dotenv")
    seen: list[Config] = []

    def fake_complete(config, messages, *a, **kw):
        seen.append(config)
        return _reply("сводка")

    monkeypatch.setattr(server.chat_core, "complete", fake_complete)
    monkeypatch.setattr(server.journal, "log_call", lambda *a, **kw: None)

    assert server.summarize_text("длинный текст") == "сводка"

    config = seen[0]
    assert config.model == "ornith"
    assert config.base_url == URL
    assert config.api_key == "lm-studio-local"
    assert config.params.max_tokens == 2048


def test_summarize_text_without_the_env_keeps_the_cloud_defaults(monkeypatch):
    import week_04.server as server

    monkeypatch.delenv("ADVENT_SUMMARIZE_MODEL", raising=False)
    monkeypatch.delenv("ADVENT_BASE_URL", raising=False)
    monkeypatch.delenv("ADVENT_OFFLINE", raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", "cloud-key-from-dotenv")
    seen: list[Config] = []
    monkeypatch.setattr(
        server.chat_core,
        "complete",
        lambda config, messages, *a, **kw: seen.append(config) or _reply(),
    )
    monkeypatch.setattr(server.journal, "log_call", lambda *a, **kw: None)

    server.summarize_text("текст")

    assert seen[0].model == "ministral-14b-latest"
    assert seen[0].params.max_tokens == 300


def _spec(name, command):
    return mcp_router.ServerSpec(name=name, command=command, timeout=5.0)


def _router_fakes(monkeypatch, *, fail_npx=False):
    listed: list[tuple[list[str], dict]] = []
    called: list[tuple[list[str], dict]] = []

    def fake_list(command, **kwargs):
        listed.append((command, kwargs))
        if fail_npx and "npx" in command[0]:
            raise MCPError("handshake: сервер не ответил")
        return SimpleNamespace(tools=[])

    def fake_call(command, name, arguments, **kwargs):
        called.append((command, kwargs))
        return mcp_client.ToolCallOutcome(text="ok", is_error=False)

    monkeypatch.setattr(mcp_router.mcp_client, "connect_and_list", fake_list)
    monkeypatch.setattr(mcp_router.mcp_client, "call_tool_once", fake_call)
    return listed, called


def test_offline_router_adds_npx_offline_and_passes_the_env(monkeypatch):
    listed, _ = _router_fakes(monkeypatch)
    env = {"ADVENT_OFFLINE": "1"}
    router = mcp_router.Router(
        [_spec("fs", ["npx", "-y", "@scope/pkg"]), _spec("repo", ["python", "-m", "x"])],
        env=env,
        offline=True,
    )

    router.connect()

    by_name = {cmd[0]: (cmd, kw) for cmd, kw in listed}
    assert by_name["npx"][0] == ["npx", "--offline", "-y", "@scope/pkg"]
    assert by_name["python"][0] == ["python", "-m", "x"]
    assert by_name["npx"][1]["env"] == env
    assert by_name["python"][1]["env"] == env


def test_online_router_leaves_the_command_alone_and_sends_no_env(monkeypatch):
    listed, _ = _router_fakes(monkeypatch)

    mcp_router.Router([_spec("fs", ["npx", "-y", "@scope/pkg"])]).connect()

    command, kwargs = listed[0]
    assert command == ["npx", "-y", "@scope/pkg"]
    assert "env" not in kwargs


def test_offline_npx_failure_names_the_cold_cache(monkeypatch):
    _router_fakes(monkeypatch, fail_npx=True)
    router = mcp_router.Router([_spec("fs", ["npx", "-y", "@scope/pkg"])], offline=True)

    report = router.connect()

    assert report.failed == ["fs"]
    assert "npx-пакет не в кэше: прогрей онлайн `npx -y" in report.warnings[0]


def test_offline_router_call_goes_through_the_same_command_and_env(monkeypatch):
    listing = SimpleNamespace(tools=[SimpleNamespace(name="t", description="", input_schema=None)])
    _, called = _router_fakes(monkeypatch)
    monkeypatch.setattr(mcp_router.mcp_client, "connect_and_list", lambda command, **kw: listing)
    env = {"ADVENT_OFFLINE": "1"}
    router = mcp_router.Router([_spec("fs", ["npx", "-y", "@scope/pkg"])], env=env, offline=True)
    router.connect()

    router.call("fs__t", {})

    command, kwargs = called[0]
    assert command == ["npx", "--offline", "-y", "@scope/pkg"]
    assert kwargs["env"] == env


def test_bridge_passes_env_and_timeout_only_when_given(monkeypatch):
    seen: list[dict] = []

    def fake_call(command, name, arguments, **kwargs):
        seen.append(kwargs)
        return mcp_client.ToolCallOutcome(text="ok", is_error=False)

    monkeypatch.setattr(mcp_client, "call_tool_once", fake_call)

    cli._mcp_bridge(["srv"])("t", {})
    cli._mcp_bridge(["srv"], {"A": "1"}, 99.0)("t", {})

    assert seen == [{"timeout": cli.MCP_TIMEOUT}, {"timeout": 99.0, "env": {"A": "1"}}]


def test_mcp_timeout_env_applies_in_local_mode_only(monkeypatch, tmp_path):
    local = _shell(monkeypatch, tmp_path)
    cloud = _shell(monkeypatch, tmp_path, local=False)
    monkeypatch.setenv("ADVENT_MCP_TIMEOUT", "240")

    assert cli._mcp_timeout(local, 15.0) == 240.0
    assert cli._mcp_timeout(cloud, 15.0) == 15.0
    monkeypatch.setenv("ADVENT_MCP_TIMEOUT", "мусор")
    assert cli._mcp_timeout(local, 15.0) == 15.0
    assert cli._mcp_env(local)["ADVENT_MCP_TIMEOUT"] == "мусор"


# --- journal ------------------------------------------------------------------


def test_journal_rows_get_endpoint_local_only_in_offline_mode(tmp_path):
    path = tmp_path / "calls.jsonl"
    messages = [{"role": "user", "content": "привет"}]

    journal_module.log_call(_reply(), messages, week=6, day=27, path=path)
    offline.enable()
    journal_module.log_call(_reply(), messages, week=6, day=27, path=path)

    import json

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert "endpoint" not in rows[0]
    assert rows[1]["endpoint"] == "local"


# --- demo and rehearsal -------------------------------------------------------


def test_demo_steps_w06d27_run_the_local_agent_with_a_blank_cloud_key():
    from advent_cli import record

    steps = record.demo_steps(6, 27)

    assert len(steps) == 1
    step = steps[0]
    assert step.module == "week_02.cli"
    assert step.env == {"MISTRAL_API_KEY": ""}
    assert step.args[0] == "--local"
    assert step.stdin_lines[0] == "/new w06d27-take"
    for line in ("/strategy facts", "/rag cite", "/local", "/exit"):
        assert line in step.stdin_lines
    assert step.stdin_lines.index("/rag cite") < step.stdin_lines.index("/local")
    assert step.timeout >= 600


def test_rehearsal_w06d27_checks_readiness_then_a_short_local_turn():
    from advent_cli import record

    steps = record.rehearsal_steps(6, 27)

    assert [s.module for s in steps] == ["week_06.cli", "week_05.cli", "week_02.cli"]
    assert steps[0].args == ["status"]
    assert steps[1].args == ["check"]  # the local RAG index gate runs before the agent turn
    local_step = steps[2]
    assert local_step.env == {"MISTRAL_API_KEY": ""}
    assert "--local" in local_step.args
    assert local_step.stdin_lines[0] == "/new w06d27-rehearsal"
    assert any(re.search("[а-я]", line) for line in local_step.stdin_lines)
    assert "/local" in local_step.stdin_lines


def test_w06d27_take_and_rehearsal_start_from_a_pristine_session(monkeypatch, tmp_path):
    from advent_cli import record
    from advent_core import session as session_module

    assert record.demo_steps(6, 27)[0].fresh_sessions == ["w06d27-take"]
    assert record.rehearsal_steps(6, 27)[2].fresh_sessions == ["w06d27-rehearsal"]

    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    for name in ("w06d27-take.json", "w06d27-take.json.2026-10-06T10-00-00.bak", "other.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(record.time, "sleep", lambda _: None)
    seen: list[list[str]] = []

    def fake_run(command, **kwargs):
        # The session file must already be gone when the child starts.
        seen.append(sorted(p.name for p in tmp_path.iterdir()))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(record.subprocess, "run", fake_run)
    record._run_step(record.Step(title="t", args=["x"], fresh_sessions=["w06d27-take"]))
    assert seen == [["other.json"]]


def test_reset_sessions_tolerates_missing_files(monkeypatch, tmp_path):
    from advent_cli import record
    from advent_core import session as session_module

    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    record._reset_sessions(["never-existed"])
