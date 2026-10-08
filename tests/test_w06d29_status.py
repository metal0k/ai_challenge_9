"""Day 29: `adventlocal status --vram` shows the loaded quant and the GPU memory, no real tools."""

from __future__ import annotations

from tests.test_week06_cli import no_network, out, run_main  # noqa: F401 (fixtures)
from week_06 import vram


def test_status_vram_prints_used_memory_from_the_sampler_query(monkeypatch, out, no_network):  # noqa: F811
    monkeypatch.setattr(vram, "query_nvidia_smi", lambda: (6474, 8192))
    assert run_main(monkeypatch, "status", "--vram") == 0
    text = out.getvalue()
    assert text.count("VRAM: старт 6474 MiB, пик 6474 MiB из 8192 MiB") == 1
    assert "готова" in text


def test_status_vram_without_nvidia_smi_says_no_data_and_still_succeeds(
    monkeypatch,
    out,  # noqa: F811
    no_network,  # noqa: F811
):
    monkeypatch.setattr(vram, "query_nvidia_smi", lambda: None)
    assert run_main(monkeypatch, "status", "--vram") == 0
    assert out.getvalue().count("VRAM: нет данных") == 1


def test_status_without_the_flag_never_asks_for_the_gpu(monkeypatch, out, no_network):  # noqa: F811
    def boom():
        raise AssertionError("nvidia-smi must not run without --vram")

    monkeypatch.setattr(vram, "query_nvidia_smi", boom)
    assert run_main(monkeypatch, "status") == 0
    assert "VRAM" not in out.getvalue()
