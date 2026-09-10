"""Генератор гигантского ввода: чистая сборка текста и проводка main().

Сети здесь нет и быть не может: счётчик в тестах — подставная функция от длины
строки, window_for и counter_for в проводке main() перекрыты monkeypatch.
Точный tekken-прогон — рантайм-дело скрипта (PROBE-w02d08-overflow.md).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Снимает ANSI-подсветку Rich.

    Rich красит числа своим highlighter'ом, и при FORCE_COLOR в окружении
    (его выставляют CI и часть терминалов) цвет доезжает даже до capsys:
    сравнение полей рвётся на покрашенном числе вместо голого. Проверяется
    содержимое строки, а не то, покрашена ли она.
    """
    return _ANSI.sub("", text)


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


def _fragments_in(phrase: str) -> tuple[str, ...]:
    """Фрагменты пула, реально обнаруженные в теле фразы (после «n) »).

    Не переигрывает формулу индексов _phrase() — просто ищет, какие строки
    пула входят в текст. Так тест ловит именно то, ради чего заведён пул:
    лексическое разнообразие тела фразы, а не различие числового префикса,
    которое делает строки разными само по себе безо всякого разнообразия
    (прежняя версия теста сравнивала строки целиком и не заметила бы, стань
    все три фрагмента одинаковыми).
    """
    remainder = phrase.split(") ", 1)[1]
    return tuple(f for f in make_biginput.FRAGMENTS if f in remainder)


def test_phrase_pool_is_large_and_compositions_differ():
    # Монотонный повтор BPE сожмёт в горсть токенов (PROBE, факт 4): пул фраз
    # обязан быть большим, а композиция из трёх фрагментов — реально меняться
    # от фразы к фразе, а не только номер в начале строки.
    assert len(make_biginput.FRAGMENTS) >= 30

    compositions = [_fragments_in(make_biginput._phrase(n)) for n in range(1, 50)]
    distinct = {frozenset(c) for c in compositions}
    used = {fragment for c in compositions for fragment in c}

    # Хвост от 49 — модульная арифметика _phrase() изредка выбирает один и тот
    # же фрагмент дважды в одной фразе, и это ожидаемо; порог существенно
    # ниже 49, но существенно выше «все композиции совпали».
    assert len(distinct) >= 30, "композиции фрагментов почти не меняются от фразы к фразе"
    # Пул должен реально быть в деле, а не наполовину простаивать.
    assert len(used) >= len(make_biginput.FRAGMENTS) - 2, "пул фрагментов задействован не весь"


def test_biginput_file_path_agrees_with_the_generators_default_out():
    """record._BIGINPUT_FILE (шаг 4б) и make_biginput.DEFAULT_OUT (шаг 4а,
    записывает файл) обязаны указывать на один и тот же путь — иначе 4б подаст
    в агента не тот файл, который написал 4а. Ничего это раньше не сверяло:
    оба места жили каждое своей константой."""
    from advent_cli import record as record_mod

    expected = (record_mod.PROJECT_ROOT / record_mod._BIGINPUT_FILE).resolve()
    assert make_biginput.DEFAULT_OUT.resolve() == expected


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

    printed = _plain(capsys.readouterr().out)
    # Три числа — в stdout, при точном счётчике без тильды.
    fields = printed.split()
    assert fields[1] == str(len(text))  # символов
    assert fields[3] == str(len(text) // 4)  # токенов
    assert fields[6] == "1000"  # лимит окна
    assert "~" not in printed
    assert printed.count("\n") == 1  # одна строка, без хвоста


def test_main_marks_estimate_with_tilde_but_still_writes(monkeypatch, tmp_path, capsys):
    code, out = _run_main(monkeypatch, tmp_path, exact=False, warning="нет точного токенизатора")
    assert code == 0
    assert out.exists()
    captured = capsys.readouterr()
    assert "~" in _plain(captured.out)
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


# main() отказывает четырьмя путями; выше проверены успех, отсутствие окна и
# argparse-ошибка margin. Ниже — оставшиеся три: счётчик, который не считает,
# и оба исключения из window_for(). Сети в них нет: window_for и counter_for
# подставные, как и во всех тестах main() выше.


def test_main_reports_uncountable_text_as_exit_code_1(monkeypatch, tmp_path, capsys):
    """RuntimeError из build_text (счётчик молчит вместо числа) должна дойти
    до кода 1 — эта ветка main() раньше не проверялась ни одним тестом."""
    monkeypatch.setattr(make_biginput, "window_for", lambda model: 1000)

    class _BrokenCounter:
        exact = True
        name = "broken"

        def count(self, messages):
            return None

    monkeypatch.setattr(
        make_biginput.tokens,
        "counter_for",
        lambda model, on_notice=None: (_BrokenCounter(), None),
    )
    out = tmp_path / "x.txt"

    code = make_biginput.main(["--model", "fake-model", "--out", str(out)])

    assert code == 1
    assert not out.exists()
    assert "не смог посчитать" in capsys.readouterr().err


def test_main_reports_config_error_as_exit_code_2(monkeypatch, tmp_path, capsys):
    """window_for() читает Config.resolve() — конфигурационная ошибка (ключ,
    .env) обязана давать код 2 и человеческий текст, не traceback."""

    def _broken_window_for(model: str) -> int | None:
        raise make_biginput.ConfigError("нет ключа API — читается из .env")

    monkeypatch.setattr(make_biginput, "window_for", _broken_window_for)
    out = tmp_path / "x.txt"

    code = make_biginput.main(["--out", str(out)])

    assert code == 2
    assert not out.exists()
    assert "нет ключа API" in capsys.readouterr().err


def test_main_propagates_the_exit_code_of_an_advent_error(monkeypatch, tmp_path, capsys):
    """window_for() ходит в живой API (list_models) — сетевая/серверная
    ошибка приходит как AdventError с СВОИМ exit_code (errors.py: 3 auth, 4
    rate limit, 5 server, 6 network), и main() обязан отдать именно его, а не
    захардкоженную единицу."""

    class _KnownExitError(make_biginput.AdventError):
        exit_code = 6

    def _broken_window_for(model: str) -> int | None:
        raise _KnownExitError("сеть недоступна")

    monkeypatch.setattr(make_biginput, "window_for", _broken_window_for)
    out = tmp_path / "x.txt"

    code = make_biginput.main(["--out", str(out)])

    assert code == 6
    assert not out.exists()
    assert "сеть недоступна" in capsys.readouterr().err
