"""Счёт токенов: точный, оценочный, сверка с фактом сервера.

Сети не касается ни разу. Точный счётчик берёт tekken.json, который лежит
внутри самого пакета mistral_common, — скачивание при этом не отключается
заглушкой «на всякий случай», а прямо запрещается (download=False), чтобы
случайный поход в сеть падал тестом, а не проходил незамеченным.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from advent_core import tokens as tokens_mod
from advent_core.tokens import (
    DEFAULT_CHARS_PER_TOKEN,
    EstimateCounter,
    MistralCounter,
    TokenizerUnavailable,
    context_limit,
    counter_for,
    reconcile,
    repo_for_model,
)

MESSAGES = [
    {"role": "system", "content": "Ты помощник."},
    {"role": "user", "content": "Сколько будет два плюс два?"},
]


def _bundled_tekken() -> Path:
    """tekken.json из самого пакета mistral_common — локальный файл, не сеть."""
    import mistral_common

    candidates = sorted((Path(mistral_common.__file__).parent / "data").glob("tekken*.json"))
    if not candidates:
        pytest.skip("в mistral_common нет встроенного tekken.json")
    return candidates[-1]


class _FakeCounter:
    """Двойник счётчика: отдаёт заранее заданное число, ни во что не ходит."""

    def __init__(self, value: int | None, *, exact: bool = True) -> None:
        self.value = value
        self.exact = exact
        self.name = "fake"
        self.calibrated_with: list[int | None] = []

    def count(self, messages) -> int | None:
        return self.value

    def calibrate(self, messages, prompt_tokens) -> None:
        self.calibrated_with.append(prompt_tokens)


# --- оценка ---------------------------------------------------------------


def test_estimate_is_marked_as_estimate():
    counter = EstimateCounter()
    assert counter.exact is False


def test_estimate_never_returns_zero_for_non_empty_messages():
    counter = EstimateCounter()
    tokens = counter.count([{"role": "user", "content": "а"}])
    assert tokens is not None and tokens >= 1


def test_estimate_of_empty_messages_is_unknown_not_zero():
    # Пустой запрос всё равно несёт накладные расходы шаблона чата, а сколько
    # именно — мы не знаем. Ноль здесь читался бы как «бесплатно».
    assert EstimateCounter().count([]) is None


def test_estimate_calibrates_towards_server_usage():
    counter = EstimateCounter()
    before = counter.chars_per_token
    assert before == DEFAULT_CHARS_PER_TOKEN
    long_message = [{"role": "user", "content": "я" * 1000}]
    counter.calibrate(long_message, prompt_tokens=500)
    assert counter.calibrated is True
    assert counter.chars_per_token == pytest.approx(2.0)
    assert counter.count(long_message) == 500


def test_estimate_ignores_missing_usage_when_calibrating():
    counter = EstimateCounter()
    counter.calibrate(MESSAGES, prompt_tokens=None)
    assert counter.calibrated is False
    assert counter.chars_per_token == DEFAULT_CHARS_PER_TOKEN


# --- точный счёт ----------------------------------------------------------


def test_exact_counter_counts_messages_without_network():
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    counter = MistralCounter(MistralTokenizer.from_file(str(_bundled_tekken())))
    tokens = counter.count(MESSAGES)
    assert counter.exact is True
    assert tokens is not None and tokens > 0


def test_exact_counter_returns_none_when_it_cannot_count():
    # Токенизатор придирчив к форме диалога; отказ посчитать — «неизвестно»,
    # а не ноль и не исключение посреди разговора.
    class _Broken:
        def encode_chat_completion(self, request):
            raise RuntimeError("нет")

    assert MistralCounter(_Broken()).count(MESSAGES) is None


# --- выбор счётчика -------------------------------------------------------


def test_unknown_model_falls_back_to_estimate_with_warning():
    counter, warning = counter_for("совершенно-своя-модель", download=False)
    assert counter.exact is False
    assert warning and "оценочный" in warning


def test_known_model_without_network_degrades_to_estimate(tmp_path):
    # Модель в таблице есть, файла в кэше нет, скачивание запрещено — агент
    # обязан продолжить работать на оценке, а не упасть.
    assert repo_for_model("ministral-14b-latest")
    counter, warning = counter_for("ministral-14b-latest", cache_dir=tmp_path, download=False)
    assert counter.exact is False
    assert warning is not None


def test_download_failure_does_not_raise_out_of_counter_for(tmp_path, monkeypatch):
    def _boom(repo, target):
        raise TokenizerUnavailable("сеть недоступна")

    monkeypatch.setattr(tokens_mod, "download_tokenizer", _boom)
    counter, warning = counter_for("ministral-14b-latest", cache_dir=tmp_path)
    assert counter.exact is False
    assert warning and "сеть недоступна" in warning


def test_cached_tokenizer_is_used_without_download(tmp_path, monkeypatch):
    repo = repo_for_model("ministral-14b-latest")
    assert repo is not None
    cached = tokens_mod._cache_path(repo, tmp_path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(_bundled_tekken().read_bytes())

    def _never(repo_id, target):
        raise AssertionError("скачивание не должно вызываться при наличии кэша")

    monkeypatch.setattr(tokens_mod, "download_tokenizer", _never)
    counter, warning = counter_for("ministral-14b-latest", cache_dir=tmp_path)
    assert warning is None
    assert counter.exact is True
    assert counter.count(MESSAGES)


# --- скачивание токенизатора ----------------------------------------------


def test_download_is_announced_before_it_starts(tmp_path, monkeypatch):
    """16.7 МБ идут минуты, и молчащий процесс читается как зависший.

    Не косметика: `advent record` снимает шаг по таймауту и называет причину
    «шаг завис» — по неозвученной скачке отличить одно от другого нельзя.
    """
    said: list[str] = []

    def _no_network(repo, **kwargs):
        raise TokenizerUnavailable("сеть недоступна")

    monkeypatch.setattr(tokens_mod, "load_tokenizer", _no_network)

    counter_for("ministral-14b-latest", cache_dir=tmp_path, on_notice=said.append)

    assert said and "качаю токенизатор" in said[0]


def test_nothing_is_announced_when_the_tokenizer_is_already_cached(tmp_path, monkeypatch):
    repo = repo_for_model("ministral-14b-latest")
    assert repo is not None
    cached = tokens_mod._cache_path(repo, tmp_path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(_bundled_tekken().read_bytes())
    said: list[str] = []

    counter_for("ministral-14b-latest", cache_dir=tmp_path, on_notice=said.append)

    assert said == []


def test_download_timeout_fits_inside_the_record_step_timeout():
    """Скачка, пережившая таймаут шага записи, убила бы дубль с неверно
    названной причиной: `advent record` сказал бы «шаг завис»."""
    from advent_cli import record

    assert tokens_mod.DOWNLOAD_TIMEOUT < record.STEP_TIMEOUT


def test_download_gives_up_on_the_total_deadline(tmp_path, monkeypatch):
    """Потолок считается по всему времени скачки, а не по одной операции сокета:
    httpx применяет свой timeout поштучно, и медленный, но живой канал не
    упирается в него никогда."""

    class _Response:
        def raise_for_status(self):
            return None

        def iter_bytes(self):
            # Поток конечный намеренно: без проверки общего срока скачка
            # дойдёт до конца и завершится успехом — то есть тест покраснеет
            # на отсутствии исключения, а не повиснет.
            for _ in range(10):
                yield b"x" * 1024

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Clock:
        """Часы, которые прыгают вперёд быстрее, чем идёт цикл чтения."""

        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            self.now += 30.0
            return self.now

    monkeypatch.setattr(tokens_mod, "time", _Clock())
    monkeypatch.setattr(tokens_mod.httpx, "stream", lambda *a, **k: _Response())

    target = tmp_path / "tekken.json"
    with pytest.raises(TokenizerUnavailable):
        tokens_mod.download_tokenizer("mistralai/что-нибудь", target)

    # Недокачанный файл не остаётся ни целью, ни .part: иначе следующий запуск
    # получит точный счётчик, который молча не открывается.
    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


# --- сверка ---------------------------------------------------------------


def test_reconcile_matches_recorded_prompt_tokens():
    # На записанном ответе и без сети: сверка закрепляется как механизм, а не
    # как конкретное число — числа зависят от версии токенизатора.
    check = reconcile(_FakeCounter(37), MESSAGES, prompt_tokens=37)
    assert check.matches is True
    assert check.delta == 0
    assert check.warning() is None


def test_reconcile_reports_mismatch_of_exact_counter():
    check = reconcile(_FakeCounter(37), MESSAGES, prompt_tokens=42)
    assert check.matches is False
    assert check.delta == -5
    warning = check.warning()
    assert warning and "37" in warning and "42" in warning


def test_reconcile_stays_quiet_for_estimate():
    # У оценки расхождение — норма и повод откалиброваться, а не сообщать о
    # поломке таблицы моделей.
    check = reconcile(_FakeCounter(30, exact=False), MESSAGES, prompt_tokens=42)
    assert check.matches is False
    assert check.warning() is None


def test_reconcile_without_usage_is_unknown_not_mismatch():
    check = reconcile(_FakeCounter(37), MESSAGES, prompt_tokens=None)
    assert check.delta is None
    assert check.matches is None
    assert check.warning() is None
    assert check.server is None


def test_reconcile_without_counter_keeps_none():
    check = reconcile(None, MESSAGES, prompt_tokens=42)
    assert check.local is None
    assert check.matches is None


# --- окно модели ----------------------------------------------------------


def test_context_limit_reads_model_card():
    assert context_limit({"max_context_length": 131072}) == 131072
    assert context_limit({"context_length": "8192"}) == 8192


def test_unknown_context_limit_is_none_not_a_number():
    assert context_limit(None) is None
    assert context_limit({}) is None
    assert context_limit({"max_context_length": 0}) is None
    assert context_limit({"max_context_length": True}) is None
