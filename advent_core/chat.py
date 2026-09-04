"""Сборка сообщений и вызов Mistral: complete и stream."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from advent_core import formats
from advent_core.client import mistral_client, requests_per_minute
from advent_core.config import Config, ConfigError
from advent_core.errors import AdventError, ConfigurationError, translate
from advent_core.telemetry import CallResult, Usage

Message = dict[str, str]

# Грубый лимит истории REPL. Считаем по символам, а не по токенам: точный
# счётчик потребовал бы токенизатора, а задача здесь — не дать диалогу
# дорасти до ошибки context length.
HISTORY_CHAR_BUDGET = 24_000


def build_messages(
    user_input: str,
    *,
    system: str | None = None,
    history: Iterable[Message] = (),
) -> list[Message]:
    """system → история → новый вопрос."""
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return messages


def trim_history(
    history: list[Message], budget: int = HISTORY_CHAR_BUDGET
) -> tuple[list[Message], int]:
    """Выкидывает старые пары сообщений, пока история не влезет в бюджет.

    Возвращает обрезанную историю и число выброшенных сообщений — вызывающий
    код обязан сказать об этом пользователю, иначе контекст теряется молча.
    """
    dropped = 0
    while history and sum(len(m["content"]) for m in history) > budget:
        # Пара user+assistant выкидывается целиком, чтобы диалог не остался
        # с вопросом без ответа.
        count = min(2, len(history))
        del history[:count]
        dropped += count
    return history, dropped


def should_stream(config: Config) -> bool:
    """Единая точка решения «стримить ответ или нет».

    format=json/schema выключает стрим всегда: вердикт по формату и поиск
    маркера завершения диалога возможны только по целому ответу, а не по
    чанкам (SPEC-w01d02.md §6.6). Вызывающий код (CLI) обязан звать эту
    функцию вместо прямого чтения config.stream — иначе правило продублируется
    в двух местах и однажды разъедется.
    """
    if config.params.format in ("json", "schema"):
        return False
    return config.stream


def _extract_delta(event: object) -> tuple[str, object | None, str | None]:
    """Достаёт кусок текста, usage и имя модели из чанка стрима.

    SDK оборачивает SSE-событие в объект с полем `data`; внутри — привычная
    структура choices[].delta.content. Идём через getattr, чтобы не падать
    на изменении обёртки между версиями SDK.
    """
    data = getattr(event, "data", event)
    model = getattr(data, "model", None)
    usage = getattr(data, "usage", None)

    choices = getattr(data, "choices", None) or []
    if not choices:
        return "", usage, model

    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None) if delta is not None else None

    if content is None:
        return "", usage, model
    if isinstance(content, str):
        return content, usage, model
    # Мультимодальный ответ приходит списком блоков — берём только текстовые.
    parts = [getattr(chunk, "text", "") or "" for chunk in content]
    return "".join(parts), usage, model


def _extract_finish_reason(event: object) -> str | None:
    """Достаёт finish_reason из чанка стрима.

    Отдельная функция, а не ещё один элемент кортежа _extract_delta: тесты
    (tests/test_chat.py) распаковывают её результат как (text, usage, model),
    и менять арность там, где вызывающий код этого не ждёт, — тихий способ
    всё сломать. Значение непустое только в последнем чанке, ровно там же,
    где уже забирается usage.

    В стриме, в отличие от complete(), значение model_length не приходит —
    проверено по SDK (SPEC-w01d02.md §2): набор ограничивается stop | length |
    error | tool_calls.
    """
    data = getattr(event, "data", event)
    choices = getattr(data, "choices", None) or []
    if not choices:
        return None
    return getattr(choices[0], "finish_reason", None)


def _with_format_instruction(
    messages: list[Message], format_name: str, schema: dict | None
) -> list[Message]:
    """Дописывает инструкцию пресета формата к system-сообщению.

    И _ask_once, и REPL зовут build_messages() сами по себе, каждый со своим
    system. Дописывать инструкцию формата здесь, в единственном месте перед
    отправкой, а не в обоих вызывающих кодах — единственный способ не
    получить два разных поведения (SPEC-w01d02.md §4). messages не
    мутируется: возвращается новый список, старый остаётся как был (важно
    для REPL — он хранит messages для истории отдельно).
    """
    if format_name == "text":
        return messages
    has_system = messages and messages[0].get("role") == "system"
    base_system = messages[0]["content"] if has_system else None
    system = formats.build_system(format_name, base_system, schema)
    if system is None:
        return messages
    rest = messages[1:] if base_system is not None else messages
    return [{"role": "system", "content": system}, *rest]


def complete(
    config: Config, messages: list[Message], capabilities: dict | None = None
) -> CallResult:
    """Один ответ целиком, без стрима."""
    payload, skipped, format_name, schema = _payload(config, messages, capabilities)
    started = time.perf_counter()

    with mistral_client(config) as mistral:
        try:
            response = mistral.chat.complete(**payload)
        except Exception as exc:
            raise translate(exc) from exc
        # Внутри with: клиент (а с ним и проба) живёт только здесь.
        rate_limit_rpm = requests_per_minute(mistral)

    latency_ms = int((time.perf_counter() - started) * 1000)
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise AdventError("Mistral вернул ответ без вариантов — текст сгенерировать не удалось.")
    choice = choices[0]
    text = getattr(choice.message, "content", "") or ""
    if not isinstance(text, str):
        text = "".join(getattr(part, "text", "") or "" for part in text)

    # stop | length | model_length | error | tool_calls (SPEC-w01d02.md §2).
    # getattr, как и у usage выше: обёртка SDK меняется между версиями.
    finish_reason = getattr(choice, "finish_reason", None)
    verdict = formats.verify(format_name, text, schema)

    return CallResult(
        text=text,
        model_requested=config.model,
        model_actual=getattr(response, "model", None),
        usage=Usage.from_raw(getattr(response, "usage", None)),
        latency_ms=latency_ms,
        stream=False,
        skipped_params=skipped,
        finish_reason=finish_reason,
        format_ok=verdict.ok,
        format_detail=verdict.detail,
        sent_messages=payload["messages"],
        rate_limit_rpm=rate_limit_rpm,
    )


def stream(
    config: Config,
    messages: list[Message],
    on_chunk: Callable[[str], None],
    capabilities: dict | None = None,
) -> CallResult:
    """Стрим ответа. Каждый кусок отдаётся в on_chunk по мере прихода.

    usage приходит в последнем чанке, поэтому его нельзя брать раньше конца
    цикла. Ctrl+C и обрыв соединения не теряют уже напечатанный текст —
    он возвращается с пометкой truncated.
    """
    payload, skipped, format_name, schema = _payload(config, messages, capabilities)
    started = time.perf_counter()

    result = CallResult(
        text="",
        model_requested=config.model,
        stream=True,
        skipped_params=skipped,
        sent_messages=payload["messages"],
    )
    parts: list[str] = []

    with mistral_client(config) as mistral:
        try:
            response = mistral.chat.stream(**payload, stream=True)
            with response as events:
                for event in events:
                    chunk, usage, model = _extract_delta(event)
                    if model and not result.model_actual:
                        result.model_actual = model
                    if usage is not None:
                        result.usage = Usage.from_raw(usage)
                    # Приходит только в последнем чанке — там же, где usage.
                    finish_reason = _extract_finish_reason(event)
                    if finish_reason:
                        result.finish_reason = finish_reason
                    if chunk:
                        parts.append(chunk)
                        on_chunk(chunk)
        except KeyboardInterrupt:
            result.truncated = True
        except Exception as exc:
            if parts:
                # Часть ответа уже на экране — отдаём её, а не теряем.
                result.truncated = True
            else:
                raise translate(exc) from exc
        # Внутри with и ПОСЛЕ except: оборванный стрим всё равно успел
        # получить заголовки ответа, и лимит из них знать полезнее всего
        # именно тогда, когда что-то пошло не так.
        result.rate_limit_rpm = requests_per_minute(mistral)

    result.text = "".join(parts)
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    # should_stream() исключает json/schema из стрима, но text/yaml/md сюда
    # доходят, а у них тоже есть вердикт — пусть и всегда ok=None, detail="—"
    # (formats.verify). footer должен видеть его так же, как для complete().
    verdict = formats.verify(format_name, result.text, schema)
    result.format_ok = verdict.ok
    result.format_detail = verdict.detail
    return result


def _payload(
    config: Config, messages: list[Message], capabilities: dict | None = None
) -> tuple[dict, list[str], str, dict | None]:
    """Payload запроса, отсеянные по capabilities параметры, и (format, schema).

    response_format и system-инструкция формата собираются здесь, а не в
    CLI: _ask_once и REPL строят messages независимо друг от друга, и это
    единственное место, гарантирующее им одинаковое поведение
    (SPEC-w01d02.md §4). format_name и schema возвращаются вызывающему коду
    (complete/stream), потому что formats.verify() по готовому ответу нужен
    именно им, а вычислять format/schema дважды — плодить второй источник
    истины.
    """
    extra, skipped = config.params.as_payload(capabilities)

    format_name = config.params.format or "text"
    try:
        schema = (
            formats.load_schema(config.params.schema_file)
            if format_name == "schema" and config.params.schema_file
            else None
        )
        payload_messages = _with_format_instruction(messages, format_name, schema)
        response_format = formats.response_format_for(format_name, schema)
    except ConfigError as exc:
        # formats.py поднимает ConfigError (файла схемы нет / format=schema
        # без schema_file) — ошибка конфигурации, а не сбоя API.
        # complete()/stream() до сих пор обещали вызывающему коду только
        # AdventError — REPL (week_01/cli.py) ловит именно его, а не
        # ConfigError, чтобы не убить сессию. Раньше здесь заворачивали в
        # голый AdventError (exit_code=1) — это держало обещание про тип, но
        # теряло верный exit_code=2 для одношотового вызова (advent_cli/app.py
        # транслирует именно error.exit_code, а не тип). ConfigurationError —
        # AdventError, значит REPL по-прежнему не падает, и exit_code=2, как у
        # исходной ConfigError.
        raise ConfigurationError(str(exc)) from exc

    if response_format is not None:
        # Уходит всегда, без гейтинга по capabilities: такого флага не
        # отдаёт список моделей ни у одной из 48 моделей аккаунта
        # (SPEC-w01d02.md §2, §6.4). Не «чинить» через Spec.requires в
        # params.py — отказ модели переводится в понятную ошибку в
        # errors.translate() по факту 400, а не гейтится заранее по
        # недокументированной догадке.
        extra["response_format"] = response_format

    payload = {"model": config.model, "messages": payload_messages, **extra}
    return payload, skipped, format_name, schema
