"""Сборка сообщений и вызов Mistral: complete и stream."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from advent_core.client import mistral_client
from advent_core.config import Config
from advent_core.errors import AdventError, translate
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


def complete(
    config: Config, messages: list[Message], capabilities: dict | None = None
) -> CallResult:
    """Один ответ целиком, без стрима."""
    payload, skipped = _payload(config, messages, capabilities)
    started = time.perf_counter()

    with mistral_client(config) as mistral:
        try:
            response = mistral.chat.complete(**payload)
        except Exception as exc:
            raise translate(exc) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise AdventError("Mistral вернул ответ без вариантов — текст сгенерировать не удалось.")
    choice = choices[0]
    text = getattr(choice.message, "content", "") or ""
    if not isinstance(text, str):
        text = "".join(getattr(part, "text", "") or "" for part in text)

    return CallResult(
        text=text,
        model_requested=config.model,
        model_actual=getattr(response, "model", None),
        usage=Usage.from_raw(getattr(response, "usage", None)),
        latency_ms=latency_ms,
        stream=False,
        skipped_params=skipped,
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
    payload, skipped = _payload(config, messages, capabilities)
    started = time.perf_counter()

    result = CallResult(
        text="", model_requested=config.model, stream=True, skipped_params=skipped
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

    result.text = "".join(parts)
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


def _payload(
    config: Config, messages: list[Message], capabilities: dict | None = None
) -> tuple[dict, list[str]]:
    """Payload запроса и список параметров, отсеянных по capabilities модели."""
    extra, skipped = config.params.as_payload(capabilities)
    return {"model": config.model, "messages": messages, **extra}, skipped
