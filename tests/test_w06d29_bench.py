"""Day 29: pre-flight against the loaded model, VRAM sampler, command wiring, saved JSON."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from advent_core import config as config_module
from advent_core import console
from advent_core import openai_compat as oc
from advent_core.config import ConfigError
from advent_core.errors import AdventError
from advent_core.rag import CitedAnswer, LedgerEntry
from week_05 import rag as rag_module
from week_05 import rag_cli
from week_06 import profiles, vram
from week_06 import ragbench as rb

LOCAL_URL = "http://127.0.0.1:1234"
Q1 = rag_module.ControlQuestion(1, "что такое alpha?", (("alpha",),), ("a.md",))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(config_module, "load_env", lambda: None)
    monkeypatch.delenv("ADVENT_LOCAL_URL", raising=False)
    monkeypatch.setattr(rb, "_pause", lambda seconds: None)


# --- pre-flight -----------------------------------------------------------------------------


def _server(monkeypatch, *, quant="Q4_K_M", ctx=40960, state="loaded", publisher=None):
    models = [
        oc.LocalModel("text-embedding-bge-m3", "embeddings", "loaded", 8192, "Q8_0", "bert"),
        oc.LocalModel("ornith", "llm", state, ctx, quant, "qwen35", publisher),
    ]
    monkeypatch.setattr(oc, "server_status", lambda url=None, timeout=10.0: models)
    monkeypatch.setattr(
        rb.rag_module,
        "check_index",
        lambda db, strategy: SimpleNamespace(model="text-embedding-bge-m3"),
    )
    monkeypatch.setattr(rb.rag_index_cli, "_warm_up_jit_model", lambda run, base: None)


def _local():
    return rb.local_config(LOCAL_URL)


def test_matching_model_passes_and_its_identity_is_recorded(monkeypatch):
    _server(monkeypatch, publisher="ornith-ai")
    backend = rb.make_local_backend(_local(), Path("x.sqlite3"), profiles.PROFILES["cap"])
    assert backend.profile is profiles.PROFILES["cap"]
    assert backend.identity == {
        "id": "ornith",
        "quantization": "Q4_K_M",
        "loaded_context_length": 40960,
        "arch": "qwen35",
        "publisher": "ornith-ai",
    }


def test_publisher_is_none_when_the_api_does_not_offer_it(monkeypatch):
    _server(monkeypatch)
    backend = rb.make_local_backend(_local(), Path("x.sqlite3"), profiles.PROFILES["cap"])
    assert backend.identity["publisher"] is None


def test_quant_mismatch_refuses_with_the_load_command_in_the_hint(monkeypatch):
    _server(monkeypatch, quant="Q4_K_M")
    with pytest.raises(AdventError) as info:
        rb.make_local_backend(_local(), Path("x.sqlite3"), profiles.PROFILES["q3"])
    assert "quant Q4_K_M вместо Q3_K_M" in info.value.message
    assert "Ornith-1.5-9B-Q3_K_M.gguf" in info.value.hint
    assert "start-local-llm.ps1 -Model ornith-1.5-9b@q3_k_m -Context 40960" in info.value.hint


def test_context_mismatch_refuses_with_the_context_in_the_hint(monkeypatch):
    _server(monkeypatch, ctx=40960)
    with pytest.raises(AdventError) as info:
        rb.make_local_backend(_local(), Path("x.sqlite3"), profiles.PROFILES["ctx24k"])
    assert "контекст 40960 вместо 24576" in info.value.message
    assert "-Model ornith-ai/ornith-1.5-9b -Context 24576" in info.value.hint


def test_unknown_quant_or_context_is_a_refusal_not_a_pass(monkeypatch):
    _server(monkeypatch, quant=None, ctx=None)
    with pytest.raises(AdventError) as info:
        rb.make_local_backend(_local(), Path("x.sqlite3"), profiles.BASELINE)
    assert "quant неизвестен" in info.value.message and "контекст неизвестен" in info.value.message


def test_without_a_profile_nothing_is_compared(monkeypatch):
    _server(monkeypatch, quant="Q3_K_M", ctx=1)
    backend = rb.make_local_backend(_local(), Path("x.sqlite3"))
    assert backend.profile is None and backend.identity["quantization"] == "Q3_K_M"


def test_model_not_loaded_is_refused_before_any_comparison(monkeypatch):
    _server(monkeypatch, state="not-loaded")
    with pytest.raises(Exception, match="не загружена"):
        rb.make_local_backend(_local(), Path("x.sqlite3"), profiles.BASELINE)


# --- VRAM sampler ---------------------------------------------------------------------------


def test_sampler_tracks_start_peak_and_total():
    readings = iter([(6000, 8192), (6474, 8192), (6200, 8192), (6100, 8192)])
    sampler = vram.VramSampler(query=lambda: next(readings, (6100, 8192)), interval=0.001)
    sampler.start()
    ticked = threading.Event()
    threading.Timer(0.05, ticked.set).start()
    ticked.wait(1)
    report = sampler.stop()
    assert report.vram_used_start == 6000
    assert report.vram_used_peak == 6474
    assert report.vram_total == 8192
    assert report.free_at_peak == 8192 - 6474


def test_sampler_without_nvidia_smi_reports_none_and_does_not_raise():
    sampler = vram.VramSampler(query=lambda: None, interval=0.001)
    report = sampler.start().stop()
    assert (report.vram_used_start, report.vram_used_peak, report.vram_total) == (None,) * 3
    assert report.known is False and vram.describe(report) == "VRAM: нет данных"


def test_sampler_survives_a_raising_query():
    def boom():
        raise RuntimeError("driver gone")

    report = vram.VramSampler(query=boom, interval=0.001).start().stop()
    assert report.known is False


def test_missing_nvidia_smi_binary_yields_none(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(vram.subprocess, "run", missing)
    assert vram.query_nvidia_smi() is None


def test_nvidia_smi_output_is_parsed_from_the_first_gpu(monkeypatch):
    done = SimpleNamespace(returncode=0, stdout="6474, 8192\n1000, 8192\n")
    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: done)
    assert vram.query_nvidia_smi() == (6474, 8192)


@pytest.mark.parametrize("stdout", ["", "garbage", "1, x"])
def test_nvidia_smi_garbage_yields_none(monkeypatch, stdout):
    done = SimpleNamespace(returncode=0, stdout=stdout)
    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: done)
    assert vram.query_nvidia_smi() is None


def test_nvidia_smi_is_asked_with_the_documented_query(monkeypatch):
    seen: list = []

    def run(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout="1, 2")

    monkeypatch.setattr(vram.subprocess, "run", run)
    vram.query_nvidia_smi()
    assert seen == [
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ]


def test_saved_vram_report_round_trips_and_garbage_reads_as_no_data():
    saved = vram.VramReport(1, 3, 8).as_dict()
    assert vram.report_from_dict(saved) == vram.VramReport(1, 3, 8)
    assert vram.report_from_dict("junk") == vram.VramReport()
    assert vram.report_from_dict({"vram_used_peak": True}).known is False


# --- run_bench + save -----------------------------------------------------------------------


class _Offline:
    def is_enabled(self):
        return False

    def enable(self):
        pass

    def reset_counters(self):
        pass

    def counters(self):
        from advent_core import offline

        return offline.Counters(
            attempted={"local": 1, "cloud": 0},
            blocked={"local": 0, "cloud": 0},
            completed={"local": 1, "cloud": 0},
        )


class _Sampler:
    def __init__(self):
        self.log = []

    def start(self):
        self.log.append("start")

    def stop(self):
        self.log.append("stop")
        return vram.VramReport(6000, 6474, 8192)


def _ask(backend, question, run, on_call):
    on_call(LedgerEntry("answer", "ornith", "local", 10, 5, 1000))
    cited = CitedAnswer("answer", "alpha", (), (), (), "")
    return rag_cli.ModeRun(
        text="alpha", ctx=None, prompt_tokens=10, completion_tokens=5, latency_ms=100,
        mode="cite", cited=cited, model_called=True,
    )  # fmt: skip


def _bench(profile, sampler=None):
    backend = rb.Backend(
        "local",
        rb.local_config(LOCAL_URL),
        Path("index.local.sqlite3"),
        profile=profile,
        identity={"id": "ornith", "quantization": "Q4_K_M", "loaded_context_length": 40960},
    )
    return rb.run_bench(
        {"local": lambda: backend},
        rb.Plan((Q1,), (), "rev1"),
        runs=1,
        ask=_ask,
        offline_mod=_Offline(),
        on_run=lambda cell, n: None,
        vram_sampler=(lambda: sampler) if sampler is not None else None,
    )


def test_sampler_runs_around_the_local_phase_and_lands_in_the_result():
    sampler = _Sampler()
    result = _bench(profiles.PROFILES["cap"], sampler)
    assert sampler.log == ["start", "stop"]
    assert result.vram == {"vram_used_start": 6000, "vram_used_peak": 6474, "vram_total": 8192}
    assert result.profile is profiles.PROFILES["cap"]


def test_no_sampler_means_no_vram_key_content_and_day_28_behaviour():
    result = _bench(None)
    assert result.vram is None and result.profile is None
    data = rb.result_to_json(result)
    assert data["day"] == 28 and data["profile"] is None and data["vram"] is None


def test_saved_json_carries_profile_fields_identity_vram_and_iso_dates(tmp_path):
    result = _bench(profiles.PROFILES["sampling"], _Sampler())
    path = tmp_path / "out.json"
    rb.save_json(result, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["day"] == 29 and data["profile"] == "sampling"
    assert data["profile_fields"]["answer_top_k"] == 20
    assert data["model_identity"]["quantization"] == "Q4_K_M"
    assert data["vram"]["vram_used_peak"] == 6474
    assert data["started_at"] and data["finished_at"]
    assert data["started_at"] <= data["finished_at"]
    assert data["settings"]["local"]["profile"] == "sampling"
    assert data["settings"]["local"]["temperature"] == 0.6


def test_report_prints_profile_identity_and_vram_once_each(monkeypatch):
    result = _bench(profiles.PROFILES["cap"], _Sampler())
    buf = io.StringIO()
    monkeypatch.setattr(
        console, "out", Console(file=buf, width=80, no_color=True, force_terminal=False)
    )
    rb.print_report(result)
    text = buf.getvalue()
    assert text.count("профиль: cap") == 1
    assert text.count("модель: id ornith, quantization Q4_K_M") == 1
    assert text.count("VRAM: старт 6000 MiB, пик 6474 MiB из 8192 MiB") == 1


# --- command wiring -------------------------------------------------------------------------


def _wire(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(rb, "prepare_plan", lambda wanted, **kw: rb.Plan((Q1,), (), "rev1"))

    def fake_bench(factories, plan, **kw):
        captured["factories"] = set(factories)
        captured["kw"] = kw
        captured["factory"] = factories.get("local")
        return rb.BenchResult(runs=[], backends=["local"], n_runs=1, plan=plan)

    monkeypatch.setattr(rb, "run_bench", fake_bench)
    monkeypatch.setattr(rb, "print_report", lambda result: None)
    monkeypatch.setattr(rb.oc, "default_url", lambda: LOCAL_URL)
    return captured


@pytest.mark.parametrize("backends", ["local,cloud", "cloud"])
def test_a_non_baseline_profile_with_any_other_backend_is_a_config_error(monkeypatch, backends):
    monkeypatch.setenv("MISTRAL_API_KEY", "realkey123456")
    captured = _wire(monkeypatch)
    with pytest.raises(ConfigError) as info:
        rb.run_rag_command(backends=backends, profile="cap")
    assert "--backends local" in str(info.value)
    assert "factories" not in captured


def test_unknown_profile_is_a_config_error(monkeypatch):
    _wire(monkeypatch)
    with pytest.raises(ConfigError):
        rb.run_rag_command(backends="local", profile="nope")


def test_baseline_profile_is_allowed_with_both_backends(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    captured = _wire(monkeypatch)
    assert rb.run_rag_command(backends="local,cloud", profile="baseline") in (0, 1)
    assert captured["factories"] == {"local"}


def test_profile_run_builds_the_local_backend_with_the_profile_and_a_vram_sampler(monkeypatch):
    built: list = []
    monkeypatch.setattr(
        rb, "make_local_backend", lambda cfg, db=None, profile=None: built.append(profile)
    )
    captured = _wire(monkeypatch)
    rb.run_rag_command(backends="local", profile="noreason")
    captured["factory"]()
    assert built == [profiles.PROFILES["noreason"]]
    assert captured["kw"]["vram_sampler"] is vram.VramSampler


def test_without_a_profile_no_sampler_and_the_day_28_factory_call(monkeypatch):
    calls: list = []
    monkeypatch.setattr(rb, "make_local_backend", lambda cfg, db=None: calls.append(cfg))
    captured = _wire(monkeypatch)
    rb.run_rag_command(backends="local")
    captured["factory"]()
    assert len(calls) == 1 and captured["kw"]["vram_sampler"] is None


def test_cli_rejects_compare_together_with_profile_or_save():
    from typer.testing import CliRunner

    from week_06 import cli

    runner = CliRunner()
    result = runner.invoke(cli.app, ["rag", "--compare", "a.json", "--profile", "cap"])
    assert result.exit_code != 0
    result = runner.invoke(cli.app, ["rag", "a.json"])
    assert result.exit_code != 0


# --- review fixes ---------------------------------------------------------------------------


def test_a_known_publisher_that_differs_from_the_profile_is_refused(monkeypatch):
    _server(monkeypatch, publisher="bartowski")
    with pytest.raises(AdventError) as info:
        rb.make_local_backend(_local(), Path("x.sqlite3"), profiles.PROFILES["cap"])
    assert "издатель bartowski вместо ornith-ai" in info.value.message
    assert "Ornith-1.5-9B-Q4_K_M.gguf" in info.value.hint


def test_the_same_quant_from_the_other_publisher_passes_the_matching_profile(monkeypatch):
    _server(monkeypatch, publisher="bartowski")
    backend = rb.make_local_backend(_local(), Path("x.sqlite3"), profiles.PROFILES["q4b"])
    assert backend.identity["publisher"] == "bartowski"


def test_an_absent_publisher_is_allowed_and_never_filled_from_the_profile(monkeypatch):
    _server(monkeypatch, publisher=None)
    backend = rb.make_local_backend(_local(), Path("x.sqlite3"), profiles.PROFILES["q4b"])
    assert backend.identity["publisher"] is None


def test_unset_sampling_is_saved_as_the_unknown_server_default(tmp_path):
    backend = rb.Backend("local", rb.local_config(LOCAL_URL), Path("x"))
    settings = rb.effective_settings(backend, tmp_path)
    assert settings["sampling_source"] == {
        "temperature": "server_default_unknown",
        "top_p": "server_default_unknown",
        "top_k": "server_default_unknown",
    }
    assert settings["sampling_note"] == (
        "дефолт сервера LM Studio (значения неизвестны; per-model override не найден)"
    )
    assert settings["model_override_found"] is False and settings["model_override_file"] is None


def test_profile_sampling_is_marked_as_coming_from_the_profile(tmp_path):
    backend = rb.Backend(
        "local", rb.local_config(LOCAL_URL), Path("x"), profile=profiles.PROFILES["sampling"]
    )
    settings = rb.effective_settings(backend, tmp_path)
    assert set(settings["sampling_source"].values()) == {"profile"}
    assert "sampling_note" not in settings


def test_a_per_model_override_file_is_found_read_only_by_the_model_key(tmp_path):
    folder = tmp_path / "user-concrete-model-default-config"
    (folder / "ornith-ai").mkdir(parents=True)
    (folder / "ornith-ai" / "ornith-1.5-9b.json").write_text("{}", encoding="utf-8")
    (folder / "other.json").write_text("{}", encoding="utf-8")
    assert rb.find_model_override("ornith", folder) == "ornith-1.5-9b.json"
    assert rb.find_model_override("nothing-like-it", folder) is None
    assert rb.find_model_override("ornith", tmp_path / "missing") is None
    backend = rb.Backend("local", rb.local_config(LOCAL_URL), Path("x"))
    settings = rb.effective_settings(backend, folder)
    assert settings["model_override_found"] is True
    assert "найден per-model override: ornith-1.5-9b.json" in settings["sampling_note"]
    assert (folder / "other.json").read_text(encoding="utf-8") == "{}"  # nothing was touched


def test_the_cloud_backend_has_no_sampling_provenance(tmp_path):
    settings = rb.effective_settings(
        rb.Backend("cloud", rb.local_config(LOCAL_URL), Path("x")), tmp_path
    )
    assert "sampling_source" not in settings


# --- VRAM sampler: stop() is a hard boundary --------------------------------------------------


def test_stop_waits_for_an_in_flight_query_and_the_report_never_changes_afterwards():
    release = threading.Event()
    entered = threading.Event()
    calls = []

    def query():
        calls.append(1)
        if len(calls) == 2:  # the worker's first poll blocks like a hung nvidia-smi
            entered.set()
            release.wait(5)
            return (9999, 8192)  # a peak that arrives late, after stop() began
        return (6000, 8192)

    sampler = vram.VramSampler(query=query, interval=0.001, stop_wait=5)
    sampler.start()
    assert entered.wait(2)
    threading.Timer(0.05, release.set).start()
    report = sampler.stop()
    assert not sampler._thread.is_alive()
    assert report.vram_used_peak == 9999  # the in-flight sample was awaited, not abandoned
    frozen = (report.vram_used_start, report.vram_used_peak, report.vram_total)
    sampler._query = lambda: (12345, 8192)
    sampler._sample()
    after = sampler.stop()
    assert (after.vram_used_start, after.vram_used_peak, after.vram_total) == frozen


def test_a_worker_that_outlives_stop_cannot_alter_the_returned_report():
    release = threading.Event()
    entered = threading.Event()
    calls = []

    def query():
        calls.append(1)
        if len(calls) == 2:
            entered.set()
            release.wait(5)
            return (9999, 8192)
        return (6000, 8192)

    sampler = vram.VramSampler(query=query, interval=0.001, stop_wait=0.05)
    sampler.start()
    assert entered.wait(2)
    report = sampler.stop()  # gives up waiting after 50 ms; the worker is still blocked
    assert sampler._thread.is_alive()
    release.set()
    sampler._thread.join(2)
    assert not sampler._thread.is_alive()
    assert report.vram_used_peak == 6000
    assert sampler.stop().vram_used_peak == 6000  # the late 9999 never landed
