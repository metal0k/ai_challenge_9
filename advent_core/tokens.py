"""Счёт токенов ДО отправки: точный токенизатор Mistral и калиброванная оценка.

Зачем вообще считать локально: заполненность окна показывается ДО запроса, а
не после (SPEC-w02d06.md §7.4), и порог обрезки контекста берётся в токенах, а
не в символах (§8). `prompt_tokens` сервера приходит уже постфактум и на этот
вопрос не отвечает.

Главное правило модуля, оно же самый частый класс бага во всей разведке
(`specs/RESEARCH-cli-agents.md`, рекомендация 1): **неизвестно — это `None`,
никогда не `0`.** Ноль здесь означает ровно «нечего считать», а не «посчитать
не удалось».
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from advent_core.chat import Message
from advent_core.config import LOG_DIR

# Таблица «имя модели в API → repo id на HuggingFace». ВТОРОЙ ИСТОЧНИК ИСТИНЫ
# и известный риск (SPEC-w02d06.md §21.1): имена в API и на HF живут своей
# жизнью и однажды разъедутся. Защита — не аккуратность, а сверка после
# каждого ответа (reconcile() ниже): расхождение точного счёта с
# `prompt_tokens` сервера означает, что таблица врёт, и агент говорит об этом
# вслух.
#
# Сюда попадает только то, что ИЗМЕРЕНО живьём. ministral-14b-latest →
# Ministral-3-14B-Instruct-2512 проверен 2026-09-07 на пяти формах запроса,
# дельта 0 (specs/PROBE-w02d06-tokenizer.md). Остальные модели линейки НЕ
# добавлены намеренно: догадка про repo id даст неверные числа, выглядящие
# точными, а модель, которой в таблице нет, честно уходит на оценку.
MODEL_TOKENIZERS: dict[str, str] = {
    "ministral-14b-latest": "mistralai/Ministral-3-14B-Instruct-2512",
}

# Файл токенизатора в репозитории HF. Не gated, токен не нужен (проверено
# через api/models), 16.7 МБ — поэтому кэш на диск обязателен.
HF_FILE_URL = "https://huggingface.co/{repo}/resolve/main/tekken.json"
TOKENIZER_FILE = "tekken.json"
TOKENIZER_CACHE_DIR = LOG_DIR / "cache" / "tokenizers"
TOKENIZER_SIZE_LABEL = "~16.7 МБ"
# Потолок на ВСЮ скачку, а не на одну операцию сокета: httpx применяет свой
# timeout поштучно, и медленный, но живой канал не упирается в него никогда.
# Считается по общему прошедшему времени в download_tokenizer().
#
# 120 с, а не 300: шаг демо снимается по таймауту через 180 с
# (advent_cli.record.STEP_TIMEOUT), и скачка, пережившая этот порог, убивала
# бы дубль записи с сообщением «шаг завис» — то есть с неверно названной
# причиной. Отказ скачки при этом безобиден: counter_for деградирует до
# оценки с предупреждением.
DOWNLOAD_TIMEOUT = 120.0

# Стартовый коэффициент символы/токен для оценочного счётчика. Замер
# 2026-09-07 на tekken (включая накладные расходы шаблона чата): русский
# текст — 2.2 симв/токен, английский — 3.9. 3.0 — середина, и она всё равно
# только стартовая: с первого же ответа коэффициент калибруется по фактическим
# `prompt_tokens` этой же сессии.
DEFAULT_CHARS_PER_TOKEN = 3.0

# Ключи карточки модели, под которыми встречается размер окна. Первый — у
# Mistral, остальные — у OpenAI-совместимых серверов, которые называют то же
# самое по-своему. Ни одного не нашлось — «неизвестно», а не выдуманное число
# (SPEC-w02d06.md §7.4, §21.2).
_CONTEXT_KEYS = ("max_context_length", "context_length", "context_window", "max_model_len")


class TokenizerUnavailable(Exception):
    """Точный токенизатор недоступен: нет в таблице, не скачался, не открылся.

    Не ошибка уровня пользователя и не повод падать: вызывающий код
    (counter_for) обязан деградировать до оценки с предупреждением наверх.
    """


class TokenCounter(Protocol):
    """Счётчик токенов для списка сообщений.

    `exact` — не украшение, а часть контракта: оценка обязана быть помечена
    как оценка везде, где она показывается (SPEC-w02d06.md §7.2). `count()`
    возвращает None, когда посчитать не удалось — не 0.
    """

    #: True — точный токенизатор модели, False — оценка по коэффициенту.
    exact: bool
    #: Человекочитаемое имя источника счёта, для панели токенов.
    name: str

    def count(self, messages: Sequence[Message]) -> int | None: ...

    def calibrate(self, messages: Sequence[Message], prompt_tokens: int | None) -> None: ...


def _chars(messages: Sequence[Message]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


@dataclass(slots=True)
class EstimateCounter:
    """Оценка по коэффициенту символы/токен, калибруемая живыми usage.

    Для произвольной локальной модели токенизатора у нас нет, а токенизатор
    Mistral дал бы для неё неверные числа, выглядящие точными — это хуже
    честной оценки. Коэффициент подтягивается по фактическим `prompt_tokens`
    той же сессии (calibrate), поэтому к третьему-четвёртому ходу он
    описывает именно ту модель, с которой идёт разговор.
    """

    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    exact: bool = field(default=False, init=False)
    name: str = "оценка"
    # Накопленное для калибровки: сумма символов и сумма фактических
    # prompt_tokens. Копим, а не берём последнее значение, — один короткий ход
    # иначе перекосил бы коэффициент целиком.
    _chars_seen: int = field(default=0, init=False)
    _tokens_seen: int = field(default=0, init=False)

    def count(self, messages: Sequence[Message]) -> int | None:
        if not messages:
            # Пустой список — не «ноль токенов»: даже пустой запрос несёт
            # накладные расходы шаблона чата, а сколько именно — мы не знаем.
            return None
        chars = _chars(messages)
        if self.chars_per_token <= 0:
            return None
        # max(1, ...): непустой запрос не может стоить ноль токенов, а
        # округление вниз на очень коротком тексте дало бы именно ноль —
        # то есть «бесплатно» вместо «мало».
        return max(1, round(chars / self.chars_per_token))

    def calibrate(self, messages: Sequence[Message], prompt_tokens: int | None) -> None:
        """Подтягивает коэффициент по фактическому usage сервера.

        `prompt_tokens is None` — usage не пришёл (частая история у локальных
        OpenAI-совместимых серверов). Тогда калибровать нечем и коэффициент
        остаётся прежним; выдумывать ноль здесь нельзя.
        """
        if prompt_tokens is None or prompt_tokens <= 0:
            return
        chars = _chars(messages)
        if chars <= 0:
            return
        self._chars_seen += chars
        self._tokens_seen += prompt_tokens
        self.chars_per_token = self._chars_seen / self._tokens_seen

    @property
    def calibrated(self) -> bool:
        """Был ли коэффициент хоть раз подтянут живыми данными."""
        return self._tokens_seen > 0


@dataclass(slots=True)
class MistralCounter:
    """Точный счёт через mistral-common: tekken.json + encode_chat_completion.

    `MistralTokenizer.from_model()` сюда не годится: метод deprecated (удаление
    в 1.13.0), требует точного версионного имени, алиасов `-latest` не
    понимает, и нашей модели в его таблице нет вовсе
    (specs/PROBE-w02d06-tokenizer.md). Рабочий путь — from_file() по
    скачанному tekken.json.
    """

    tokenizer: Any
    name: str = "tekken"
    exact: bool = field(default=True, init=False)

    def count(self, messages: Sequence[Message]) -> int | None:
        if not messages:
            return None
        try:
            from mistral_common.protocol.instruct.request import ChatCompletionRequest

            request = ChatCompletionRequest.from_openai(list(messages))
            return len(self.tokenizer.encode_chat_completion(request).tokens)
        except Exception:  # noqa: BLE001
            # Токенизатор придирчив к форме диалога (например, требует, чтобы
            # последним было сообщение пользователя). Отказ посчитать — это
            # «неизвестно», а не ноль и не падение агента посреди разговора.
            return None

    def calibrate(self, messages: Sequence[Message], prompt_tokens: int | None) -> None:
        """Точному счёту калибровка не нужна — метод есть ради общего протокола.

        Расхождение с сервером у точного счётчика означает не «коэффициент
        подстроить», а «таблица моделей разъехалась» — это ловит reconcile().
        """
        return


def repo_for_model(model: str) -> str | None:
    """Repo id на HF для имени модели в API. None — модели нет в таблице."""
    return MODEL_TOKENIZERS.get(model.strip().lower())


def _cache_path(repo: str, cache_dir: Path | None = None) -> Path:
    directory = cache_dir or TOKENIZER_CACHE_DIR
    return directory / f"{repo.replace('/', '__')}--{TOKENIZER_FILE}"


def download_tokenizer(repo: str, target: Path) -> Path:
    """Качает tekken.json репозитория в target. Запись атомарная.

    Атомарность здесь не педантизм: 16.7 МБ по сети успевают оборваться на
    середине, а недокачанный файл в кэше — это точный счётчик, который молча
    не открывается при каждом следующем запуске.
    """
    url = HF_FILE_URL.format(repo=repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.part")
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT
    try:
        with httpx.stream("GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
                    # Общий срок, а не только таймаут сокета: медленный, но
                    # живой канал иначе тянет файл дольше, чем длится весь шаг
                    # записи демо, и срывает дубль под чужим предлогом.
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"не уложились в {DOWNLOAD_TIMEOUT:.0f} с")
        os.replace(tmp, target)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise TokenizerUnavailable(f"не удалось скачать токенизатор {repo}: {exc}") from exc
    return target


def tokenizer_file(repo: str, *, cache_dir: Path | None = None, download: bool = True) -> Path:
    """Путь к tekken.json репозитория: из кэша, иначе — скачать."""
    path = _cache_path(repo, cache_dir)
    if path.is_file():
        return path
    if not download:
        raise TokenizerUnavailable(f"токенизатор {repo} не скачан, скачивание выключено")
    return download_tokenizer(repo, path)


def load_tokenizer(repo: str, *, cache_dir: Path | None = None, download: bool = True) -> Any:
    """MistralTokenizer по repo id. TokenizerUnavailable на любом отказе."""
    path = tokenizer_file(repo, cache_dir=cache_dir, download=download)
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        return MistralTokenizer.from_file(str(path))
    except TokenizerUnavailable:
        raise
    except Exception as exc:
        raise TokenizerUnavailable(f"токенизатор {repo} не открылся: {exc}") from exc


def counter_for(
    model: str,
    *,
    cache_dir: Path | None = None,
    download: bool = True,
    on_notice: Callable[[str], None] | None = None,
) -> tuple[TokenCounter, str | None]:
    """Счётчик для модели и предупреждение наверх (None — всё в порядке).

    Никогда не бросает: сети нет, пакета нет, модели нет в таблице — во всех
    случаях возвращается оценочный счётчик и текст, который CLI напечатает в
    stderr. Агент из-за токенизатора падать не имеет права: счёт токенов —
    удобство, а разговор — продукт.

    Предупреждение возвращается, а не печатается: печатает только CLI
    (SPEC-w02d06.md §5, §14). `on_notice` — по той же причине колбэк, а не
    print: он зовётся ДО скачки, потому что скачка идёт минуты, а молчащий
    процесс между «сессия default» и приглашением выглядит как зависший.
    """
    repo = repo_for_model(model)
    if repo is None:
        return (
            EstimateCounter(),
            f"точного токенизатора для {model} нет в таблице — счёт токенов оценочный (~)",
        )
    if on_notice is not None and download and not _cache_path(repo, cache_dir).is_file():
        on_notice(
            f"качаю токенизатор {repo} ({TOKENIZER_SIZE_LABEL}, разово, дальше из кэша) — "
            "первый запуск дольше обычного"
        )
    try:
        tokenizer = load_tokenizer(repo, cache_dir=cache_dir, download=download)
    except TokenizerUnavailable as exc:
        return EstimateCounter(), f"{exc}; счёт токенов оценочный (~)"
    return MistralCounter(tokenizer, name=repo), None


@dataclass(slots=True, frozen=True)
class TokenCheck:
    """Сверка локального счёта с фактом сервера.

    Живёт отдельным типом, а не парой чисел, потому что у сверки три исхода, а
    не два: совпало, разошлось и «сверять не с чем» (usage не пришёл). Третий
    исход обязан отличаться от второго, иначе отсутствующий usage читается как
    поломка таблицы моделей.
    """

    local: int | None
    server: int | None
    exact: bool

    @property
    def delta(self) -> int | None:
        """Локально минус сервер. None — сверять не с чем."""
        if self.local is None or self.server is None:
            return None
        return self.local - self.server

    @property
    def matches(self) -> bool | None:
        """True/False — сошлось или нет; None — сверять не с чем."""
        delta = self.delta
        return None if delta is None else delta == 0

    def warning(self) -> str | None:
        """Текст предупреждения наверх, либо None.

        Предупреждаем ТОЛЬКО про расхождение точного счёта: у оценки
        расхождение — это норма и повод откалиброваться, а не сообщать о
        поломке. Расхождение точного счётчика означает ровно одно: таблица
        «имя модели в API → repo id на HF» разъехалась (SPEC-w02d06.md §7.1).
        """
        if not self.exact or self.matches is not False:
            return None
        return (
            f"счёт токенов разошёлся с сервером: локально {self.local}, "
            f"сервер {self.server} (дельта {self.delta:+d}) — "
            "похоже, таблица токенизаторов разъехалась с моделью"
        )


def reconcile(
    counter: TokenCounter | None,
    messages: Sequence[Message],
    prompt_tokens: int | None,
) -> TokenCheck:
    """Сверяет наш локальный счёт с `prompt_tokens` ответа.

    Функция чистая: калибровку оценки она НЕ делает, хотя данные для этого у
    неё есть. Сверка и подстройка — разные события (одну хочется звать в
    тестах и в `/tokens`, другая меняет состояние счётчика), и склеивание их в
    одном вызове означало бы, что посмотреть на расхождение нельзя, не
    изменив счётчик.

    ВАЖНО, что сюда передаётся: сверять надо с теми сообщениями, которые
    РЕАЛЬНО ушли в API (`CallResult.sent_messages`), а не с теми, что собрал
    вызывающий код. Слой формата (`format=json/schema/…`) дописывает
    инструкцию к system уже внутри chat._payload(), и сверка «до слоя» дала бы
    стабильную дельту на ровном месте — то есть ложное «таблица разъехалась».
    """
    if counter is None:
        return TokenCheck(local=None, server=prompt_tokens, exact=False)
    return TokenCheck(local=counter.count(messages), server=prompt_tokens, exact=counter.exact)


def context_limit(card: dict | None) -> int | None:
    """Размер окна из карточки модели `/v1/models`. None — неизвестно.

    Неизвестно — это именно None: выдуманный лимит хуже отсутствующего, потому
    что от него считается порог обрезки. Разведка отдельно предупреждает не
    доверять заявленному лимиту слепо (goose #6185: заявлено 128K, реально
    262K), поэтому значение — ориентир для показа заполненности, а не
    гарантия (SPEC-w02d06.md §21.2).
    """
    if not card:
        return None
    for key in _CONTEXT_KEYS:
        value = card.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                continue
            if parsed > 0:
                return parsed
    return None
