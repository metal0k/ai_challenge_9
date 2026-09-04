"""Результат одного вызова LLM: текст, usage, latency, и сумма по нескольким."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(slots=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # Из usage.prompt_tokens_details.cached_tokens (SPEC-w01d05.md §10):
    # промпт-кэш биллится по input в 10 раз дешевле. Опциональная деталь
    # биллинга, а не признак того, что usage не пришёл, — поэтому в
    # is_empty() не участвует: отсутствие prompt_tokens_details означает
    # «кэша не было», а не «usage не пришёл».
    cached_tokens: int | None = None

    @classmethod
    def from_raw(cls, raw: object) -> Usage:
        """Собирает usage и из dict, и из pydantic-модели SDK.

        prompt_tokens_details — вложенный объект той же двойной природы, что
        и сам usage: и dict, и pydantic-модель SDK, — поэтому разбирается тем
        же приёмом (get/getattr), а не напрямую raw["prompt_tokens_details"].
        """
        if raw is None:
            return cls()
        get = raw.get if isinstance(raw, dict) else lambda k: getattr(raw, k, None)
        details = get("prompt_tokens_details")
        if details is None:
            cached_tokens = None
        else:
            get_detail = (
                details.get if isinstance(details, dict) else lambda k: getattr(details, k, None)
            )
            cached_tokens = get_detail("cached_tokens")
        return cls(
            prompt_tokens=get("prompt_tokens"),
            completion_tokens=get("completion_tokens"),
            total_tokens=get("total_tokens"),
            cached_tokens=cached_tokens,
        )

    def is_empty(self) -> bool:
        """Нечем считать: usage не пришёл вовсе или пришёл частично.

        Частичный usage (пришёл completion, но не пришёл prompt) считается
        отсутствующим намеренно. Иначе вызов не попадает в missing_usage,
        неизвестное слагаемое суммируется как 0, и Totals.tokens_label()
        печатает «0/7» без оговорки — то есть показывает неизвестное значение
        точным нулём. Ровно эту ошибку missing_usage и заведён предотвращать.

        cached_tokens сюда не входит: это опциональная деталь биллинга, а не
        признак прихода usage — модель без промпт-кэша законно не пришлёт
        prompt_tokens_details вовсе, и это не то же самое, что «usage не
        пришёл».
        """
        if self.total_tokens is not None:
            return False
        return self.prompt_tokens is None or self.completion_tokens is None


@dataclass(slots=True)
class CallResult:
    text: str = ""
    model_requested: str = ""
    model_actual: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    stream: bool = True
    truncated: bool = False
    # Параметры, не отправленные из-за отсутствующей capability модели.
    skipped_params: list[str] = field(default_factory=list)
    # stop | length | model_length | error | tool_calls (complete()); в
    # stream() набор без model_length — SDK его в чанках не присылает
    # (SPEC-w01d02.md §2). None — SDK не вернул значение вовсе.
    finish_reason: str | None = None
    # Вердикт по формату из formats.verify(): True/False — проверено,
    # None — для text/yaml/md вердикт принципиально не выносится (нет ни
    # API-гарантии, ни дешёвой верификации).
    format_ok: bool | None = None
    # Человекочитаемая расшифровка вердикта для footer, например
    # "JSON ✓ · схема ✓ · items: 3" или "—" для форматов без вердикта.
    format_detail: str | None = None
    # Сообщения, фактически ушедшие в Mistral — то есть messages ДО этой
    # функции плюс инструкция пресета формата, дописанная chat._payload().
    # Журнал (week_01/cli.py → log_call) исторически писал в logs/calls.jsonl
    # тот messages, что был построен ДО дописывания инструкции формата —
    # source of truth для недель 2/5 расходился с тем, что реально видела
    # модель. None — только когда вызов не дошёл до _payload() вовсе
    # (например, CallResult(model_requested=...) для error-веток в cli.py,
    # где messages для лога и так есть отдельно).
    sent_messages: list[dict[str, str]] | None = None
    # Лимит запросов в минуту ДЛЯ МОДЕЛИ ЭТОГО ВЫЗОВА, из заголовка
    # x-ratelimit-limit-req-minute (SPEC-w01d05.md §13). None — «неизвестно»:
    # либо шов регистрации хука в SDK отвалился, либо ответ пришёл без
    # заголовка. Ноль здесь — законное значение, а не «нет данных»: именно
    # ноль отдаёт Mistral для модели, недоступной на тарифе, и путать его с
    # None нельзя.
    rate_limit_rpm: int | None = None


@dataclass(slots=True, frozen=True)
class Totals:
    """Сумма по нескольким CallResult: вызовы, токены, время.

    Живёт в core, а не в week_01: цена ответа складывается из нескольких
    вызовов начиная с Day 03 (панель экспертов — четыре вызова на один
    ответ), а неделя 2 с её учётом токенов и компактизацией контекста будет
    складывать ровно то же самое. Отдельный тип, а не кортеж, потому что
    складывать приходится и суммы между собой (прогоны одной стратегии).

    frozen: сумма считается один раз по готовому списку вызовов и дальше
    только читается — случайная правка поля означала бы, что телеметрия
    разошлась с журналом.

    missing_usage — число вызовов, у которых usage не пришёл вовсе или пришёл
    частично (см. Usage.is_empty). Без него "0 токенов" читается как
    «бесплатно», хотя на деле это «неизвестно»: стрим отдаёт usage только
    последним чанком, и оборванный ответ приходит без него совсем.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    missing_usage: int = 0

    @classmethod
    def of(cls, results: Iterable[CallResult]) -> Totals:
        """Складывает результаты вызовов в одну сумму."""
        calls = prompt = completion = total = latency = missing = 0
        for result in results:
            calls += 1
            latency += result.latency_ms
            usage = result.usage
            if usage.is_empty():
                missing += 1
            prompt += usage.prompt_tokens or 0
            completion += usage.completion_tokens or 0
            # total_tokens сервер присылает сам; когда его нет, а слагаемые
            # есть — считаем сами, иначе колонка "Токены" покажет 0 при
            # ненулевых prompt/completion.
            total += usage.total_tokens or (
                (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
            )
        return cls(calls, prompt, completion, total, latency, missing)

    def __add__(self, other: Totals) -> Totals:
        if not isinstance(other, Totals):
            return NotImplemented
        return Totals(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
            missing_usage=self.missing_usage + other.missing_usage,
        )

    def has_usage(self) -> bool:
        return self.calls > self.missing_usage

    def tokens_label(self) -> str:
        """prompt/completion для таблицы сравнения (§9 SPEC-w01d03)."""
        if not self.has_usage():
            return "—"
        label = f"{self.prompt_tokens}/{self.completion_tokens}"
        if self.missing_usage:
            # Часть вызовов не отдала usage — сумма занижена, и молчать об
            # этом нельзя: колонка сравнивает стоимость стратегий.
            label += f" (без usage: {self.missing_usage})"
        return label

    def time_label(self) -> str:
        """Суммарное время: миллисекунды до секунды, дальше секунды."""
        if self.latency_ms < 1000:
            return f"{self.latency_ms} ms"
        return f"{self.latency_ms / 1000:.1f} s"
