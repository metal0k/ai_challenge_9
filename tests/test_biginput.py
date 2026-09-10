"""Генератор гигантского ввода: чистая сборка текста и проводка main().

Сети здесь нет и быть не может: счётчик в тестах — подставная функция от длины
строки, window_for и counter_for в проводке main() перекрыты monkeypatch.
Точный tekken-прогон — рантайм-дело скрипта (PROBE-w02d08-overflow.md).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "make_biginput", ROOT / "tools" / "make_biginput.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_biginput = _load_module()


def _char_counter(text: str) -> int:
    """Подставной счётчик: детерминирован, ни во что не ходит."""
    return len(text) // 4


# --- чистая сборка текста ---------------------------------------------------


def test_build_text_is_one_line_without_any_newline():
    text = make_biginput.build_text(_char_counter, 1000)
    assert "\n" not in text
    assert "\r" not in text


def test_build_text_is_deterministic_byte_for_byte():
    # Один и тот же target — один и тот же текст: демо-шаг записи сравнивает
    # напечатанное число токенов с тем, что показал прогон генератора.
    first = make_biginput.build_text(_char_counter, 2000)
    second = make_biginput.build_text(_char_counter, 2000)
    assert first == second


def test_build_text_reaches_target_by_the_given_counter():
    target = 1500
    text = make_biginput.build_text(_char_counter, target)
    assert _char_counter(text) >= target


def test_build_text_numbers_phrases_sequentially():
    text = make_biginput.build_text(_char_counter, 800)
    phrases = text.split(" ")
    # Каждая фраза начинается со своего номера: «1) …», «2) …» и так далее.
    numbers = [int(p.rstrip(")")) for p in phrases if p.endswith(")")]
    assert numbers == list(range(1, len(numbers) + 1))


def test_build_text_does_not_wrap_past_target_much():
    # Запас 0.95 сжимает перелёт: текст обязан превысить окно, но не вдвое.
    target = 4000
    text = make_biginput.build_text(_char_counter, target)
    assert _char_counter(text) < target * 1.6


def test_build_text_rejects_non_positive_target():
    with pytest.raises(ValueError):
        make_biginput.build_text(_char_counter, 0)


def test_build_text_surfaces_uncountable_text():
    def _broken(text: str) -> int | None:
        return None

    with pytest.raises(RuntimeError, match="не смог посчитать"):
        make_biginput.build_text(_broken, 100)


def test_phrase_pool_is_large_and_compositions_differ():
    # Монотонный повтор BPE сожмёт в горсть токенов (PROBE, факт 4): пул фраз
    # обязан быть большим, а соседние фразы — не повторять друг друга.
    assert len(make_biginput.FRAGMENTS) >= 30
    assert len({make_biginput._phrase(n) for n in range(1, 50)}) == 49


# --- проводка main(): stderr/stdout, файл, код возврата ---------------------


class _FakeCounter:
    """Двойник счётчика токенов: токен = 4 символа, точность настраивается."""

    def __init__(self, *, exact: bool = True) -> None:
        self.exact = exact
        self.name = "fake"

    def count(self, messages) -> int | None:
        return sum(len(m.get("content") or "") for m in messages) // 4

    def calibrate(self, messages, prompt_tokens) -> None:
        return None


def _run_main(monkeypatch, tmp_path: Path, *, exact: bool = True, warning: str | None = None):
    """main() без сети: окно подставное, счётчик подставной, вывод в tmp."""
    monkeypatch.setattr(make_biginput, "window_for", lambda model: 1000)
    monkeypatch.setattr(
        make_biginput.tokens,
        "counter_for",
        lambda model, on_notice=None: (_FakeCounter(exact=exact), warning),
    )
    out = tmp_path / "demo" / "biginput.txt"
    code = make_biginput.main(["--model", "fake-model", "--out", str(out)])
    return code, out


def test_main_writes_single_line_file_and_prints_three_numbers(monkeypatch, tmp_path, capsys):
    code, out = _run_main(monkeypatch, tmp_path)
    assert code == 0

    text = out.read_text(encoding="utf-8")
    assert "\n" not in text
    assert len(text) // 4 >= 1000 + 3000  # окно + дефолтный margin

    captured = capsys.readouterr()
    # Три числа — в stdout, при точном счётчике без тильды.
    fields = captured.out.split()
    assert fields[1] == str(len(text))  # символов
    assert fields[3] == str(len(text) // 4)  # токенов
    assert fields[6] == "1000"  # лимит окна
    assert "~" not in captured.out
    assert captured.out.count("\n") == 1  # одна строка, без хвоста


def test_main_marks_estimate_with_tilde_but_still_writes(monkeypatch, tmp_path, capsys):
    code, out = _run_main(monkeypatch, tmp_path, exact=False, warning="нет точного токенизатора")
    assert code == 0
    assert out.exists()
    captured = capsys.readouterr()
    assert "~" in captured.out
    # Предупреждение об оценке — в stderr, продукт (числа) не замусорен.
    assert "нет точного токенизатора" in captured.err


def test_main_fails_without_known_window(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(make_biginput, "window_for", lambda model: None)
    code = make_biginput.main(["--out", str(tmp_path / "x.txt")])
    assert code != 0
    assert not (tmp_path / "x.txt").exists()
    assert "окна" in capsys.readouterr().err


def test_main_rejects_non_positive_margin(monkeypatch, tmp_path):
    monkeypatch.setattr(make_biginput, "window_for", lambda model: 1000)
    with pytest.raises(SystemExit):
        make_biginput.main(["--margin", "0"])
