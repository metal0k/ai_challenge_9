"""Starlette app: auth -> rate limit -> body/validation -> admission -> model -> proxy."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import httpx
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .errors import ApiError
from .keys import KeyRecord, KeyStore, RateLimiter, SlidingWindow, load_keys
from .lifecycle import ProcessManager
from .settings import Settings
from .validate import ChatRequest, parse_chat_request

log = logging.getLogger("advent.gateway")
STATIC = Path(__file__).parent / "static"
CSP = "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
SEND_GRACE = 2.0  # lets the terminal error event out after the generation deadline
STATUS_BUCKETS = ("200", "401", "413", "429", "503", "5xx")


# -- admission --------------------------------------------------------------------------


class Ticket:
    """A place in the line. release() is idempotent and frees exactly what was taken."""

    def __init__(self, admission: Admission):
        self.adm = admission
        self.acquired = False
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.adm.occupied -= 1
        if self.acquired:
            self.adm.sem.release()


class Admission:
    """One generation at a time plus `queue_max` waiting; held until the stream is cleaned up."""

    def __init__(self, queue_max: int, retry_after: int):
        self.cap = 1 + queue_max
        self.retry_after = retry_after
        self.occupied = 0
        self.sem = asyncio.Semaphore(1)

    @property
    def waiting(self) -> int:
        return max(0, self.occupied - (1 if self.sem.locked() else 0))

    def reserve(self) -> Ticket:
        if self.occupied >= self.cap:
            raise ApiError(
                503, "queue_full", "Очередь заполнена, попробуйте позже.",
                type_="server_error", retry_after=self.retry_after,
            )  # fmt: skip
        self.occupied += 1  # no await between the check and this line
        return Ticket(self)

    async def wait_turn(self, ticket: Ticket, timeout: float) -> None:
        try:
            async with asyncio.timeout(timeout):
                await self.sem.acquire()
        except TimeoutError:
            raise ApiError(
                503,
                "queue_timeout",
                "Слишком долго ждали очереди.",
                type_="server_error",
                retry_after=self.retry_after,
            ) from None
        ticket.acquired = True


# -- plumbing ---------------------------------------------------------------------------


class Stats:
    def __init__(self):
        self.started = time.time()
        self.status: Counter[str] = Counter()
        self.outcomes: Counter[str] = Counter()

    def count_status(self, code: int) -> None:
        key = str(code) if str(code) in STATUS_BUCKETS else ("5xx" if code >= 500 else "other")
        self.status[key] += 1

    def snapshot(self) -> dict:
        snap = {k: self.status.get(k, 0) for k in STATUS_BUCKETS}
        snap["other"] = self.status.get("other", 0)
        return snap


class CountingMiddleware:
    def __init__(self, app, stats: Stats):
        self.app, self.stats = app, stats

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def counting_send(message):
            if message["type"] == "http.response.start":
                self.stats.count_status(message["status"])
            await send(message)

        await self.app(scope, receive, counting_send)


class SseDoneTracker:
    """Incremental SSE framing: `done` only when a complete event's data is exactly [DONE]."""

    KEEP = 65536  # a [DONE] event is a few bytes; never hold more than this

    def __init__(self):
        self.buf = b""
        self.done = False

    @property
    def pending(self) -> bool:
        return bool(self.buf)

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk
        if b"\r" in self.buf:
            self.buf = self.buf.replace(b"\r\n", b"\n")
        while (i := self.buf.find(b"\n\n")) != -1:
            event, self.buf = self.buf[:i], self.buf[i + 2 :]
            data = [
                ln[5:].removeprefix(b" ") for ln in event.split(b"\n") if ln.startswith(b"data:")
            ]
            if data and b"\n".join(data) == b"[DONE]":
                self.done = True
        if len(self.buf) > self.KEEP:
            self.buf = self.buf[-self.KEEP :]


class GuardedStream:
    """SSE response whose cleanup always runs: on finish, error, disconnect or cancel.

    Starlette's StreamingResponse leaves a suspended generator to the GC when the client
    drops, and the admission slot would be held until then.
    """

    def __init__(
        self,
        gen: AsyncIterator[bytes],
        cleanup: Callable[[], Awaitable[None]],
        headers: dict[str, str],
        deadline: float | None = None,
        on_send_timeout: Callable[[], None] = lambda: None,
    ):
        self.gen = gen
        self.cleanup = cleanup
        self.deadline = deadline  # loop time; bounds every send() so a stalled reader is cut
        self.on_send_timeout = on_send_timeout
        # No Content-Length: the body is open-ended.
        self.raw_headers = [(b"content-type", b"text/event-stream; charset=utf-8")] + [
            (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()
        ]

    async def __call__(self, scope, receive, send):
        async def watch(scope_):
            while True:
                if (await receive())["type"] == "http.disconnect":
                    scope_.cancel()
                    return

        async def bounded_send(msg):
            if self.deadline is None:
                await send(msg)
                return
            async with asyncio.timeout_at(self.deadline):
                await send(msg)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(watch, tg.cancel_scope)
                try:
                    await bounded_send(
                        {
                            "type": "http.response.start",
                            "status": 200,
                            "headers": self.raw_headers,
                        }
                    )
                    async for chunk in self.gen:
                        await bounded_send(
                            {"type": "http.response.body", "body": chunk, "more_body": True}
                        )
                    await bounded_send(
                        {"type": "http.response.body", "body": b"", "more_body": False}
                    )
                except TimeoutError:
                    self.on_send_timeout()
                tg.cancel_scope.cancel()
        finally:
            with anyio.CancelScope(shield=True):
                with contextlib.suppress(Exception):
                    await self.gen.aclose()  # type: ignore[attr-defined]
                await self.cleanup()


async def _wait_disconnect(request: Request) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass


def sse_error(err: ApiError) -> bytes:
    return b"data: " + json.dumps(err.body(), ensure_ascii=False).encode() + b"\n\n"


def json_error(err: ApiError) -> JSONResponse:
    return JSONResponse(
        err.body(),
        status_code=err.status,
        headers={**err.headers(), "Cache-Control": "no-store"},
    )


def api(fn):
    @functools.wraps(fn)
    async def wrapper(self, request: Request):
        try:
            return await fn(self, request)
        except ApiError as e:
            return json_error(e)
        except Exception:
            log.exception("unhandled error in %s", fn.__name__)
            return json_error(
                ApiError(500, "internal_error", "Внутренняя ошибка.", type_="server_error")
            )

    return wrapper


# -- app --------------------------------------------------------------------------------


class Gateway:
    def __init__(
        self,
        settings: Settings,
        manager: ProcessManager | None,
        records: list[KeyRecord],
        clock: Callable[[], float],
    ):
        self.s = settings
        self.stats = Stats()
        self.keys = KeyStore(records)
        self.limiter = RateLimiter(records, clock)
        self.global_budget = SlidingWindow(settings.global_per_minute, 60.0, clock)
        self.admission = Admission(settings.queue_max, settings.queue_retry_after)
        self.manager = manager or ProcessManager(settings)
        self.manager.busy = lambda: self.admission.occupied > 0

    # -- auth ---------------------------------------------------------------------------

    def authenticate(self, request: Request) -> KeyRecord:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        rec = self.keys.authenticate(token.strip()) if scheme.lower() == "bearer" else None
        if rec is None:
            raise ApiError(
                401, "invalid_api_key", "Неверный или отсутствующий ключ.",
                type_="authentication_error",
            )  # fmt: skip
        return rec

    # -- simple endpoints ---------------------------------------------------------------

    @api
    async def health(self, request: Request):
        self.authenticate(request)
        return JSONResponse({"status": "ok", "model": self.manager.state})

    @api
    async def models(self, request: Request):
        self.authenticate(request)
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": self.s.model_id, "object": "model", "owned_by": "advent-llm"}],
            }
        )

    @api
    async def status(self, request: Request):
        rec = self.authenticate(request)
        m, s = self.manager, self.s
        rss = m.rss_kb()
        body = {
            "version": s.version,
            "model": m.state,
            "model_id": s.model_id,
            "rss_mb": round(rss / 1024, 1) if rss is not None else None,
            "model_uptime_s": _round(m.uptime()),
            "idle_unload_in_s": _round(m.idle_remaining()),
            "gateway_uptime_s": round(time.time() - self.stats.started, 1),
            "last_exit_code": m.last_exit,
            "queue": {
                "active": 1 if self.admission.sem.locked() else 0,
                "waiting": self.admission.waiting,
                "max": s.queue_max,
            },
            "limits": {
                "n_ctx": s.n_ctx,
                "max_tokens_default": s.default_max_tokens,
                "max_tokens_ceiling": s.max_tokens_ceiling,
                "body_max_bytes": s.body_max_bytes,
                "idle_seconds": s.idle_seconds,
                "global_per_minute": s.global_per_minute,
            },
            "key": {
                "name": rec.name,
                "rpm": rec.rpm,
                "burst": rec.burst,
                "tokens_left": round(self.limiter.left(rec.name), 2),
            },
            "counters": self.stats.snapshot(),
            "outcomes": dict(self.stats.outcomes),
        }
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    # -- chat ---------------------------------------------------------------------------

    async def read_body(self, request: Request) -> bytes:
        s = self.s
        enc = request.headers.get("content-encoding", "identity").strip().lower()
        if enc not in ("", "identity"):
            raise ApiError(415, "unsupported_encoding", "Content-Encoding не поддерживается.")
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                too_big = int(declared) > s.body_max_bytes
            except ValueError:
                raise ApiError(
                    400, "invalid_content_length", "Некорректный Content-Length."
                ) from None
            if too_big:
                raise self._too_big()
        buf = bytearray()
        try:
            async with asyncio.timeout(s.body_read_deadline):
                async for chunk in request.stream():
                    buf += chunk
                    if len(buf) > s.body_max_bytes:
                        raise self._too_big()
        except TimeoutError:
            raise ApiError(
                408, "body_read_timeout", "Тело запроса пришло слишком медленно."
            ) from None
        except ClientDisconnect:
            raise ApiError(400, "client_disconnected", "Клиент закрыл соединение.") from None
        return bytes(buf)

    def _too_big(self) -> ApiError:
        return ApiError(
            413, "body_too_large",
            f"Тело запроса больше {self.s.body_max_bytes} байт.",
            extra={"limit_bytes": self.s.body_max_bytes},
        )  # fmt: skip

    @api
    async def chat(self, request: Request):
        s = self.s
        rec = self.authenticate(request)

        ok, retry = self.limiter.take(rec.name)
        if not ok:
            raise ApiError(
                429, "rate_limit_exceeded", "Слишком много запросов для этого ключа.",
                type_="rate_limit_error", retry_after=retry,
            )  # fmt: skip

        req = parse_chat_request(await self.read_body(request), s)

        ok, retry = self.global_budget.take()
        if not ok:
            raise ApiError(
                429, "global_rate_limit", "Исчерпан общий лимит генераций сервиса.",
                type_="rate_limit_error", retry_after=retry,
            )  # fmt: skip

        work = asyncio.ensure_future(self._admit_and_serve(req, rec))
        watcher = asyncio.ensure_future(_wait_disconnect(request))
        try:
            await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            work.cancel()
            watcher.cancel()
            await asyncio.shield(asyncio.gather(work, watcher, return_exceptions=True))
            raise
        if work.done():
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            return work.result()
        # Client left before the response started: stop the work; its cleanup frees the line.
        work.cancel()
        await asyncio.shield(asyncio.gather(work, return_exceptions=True))
        return Response(status_code=499)

    async def _admit_and_serve(self, req: ChatRequest, rec: KeyRecord):
        s, m = self.s, self.manager
        ticket = self.admission.reserve()
        handed_off = False
        t_wait = time.monotonic()
        gen = 0
        try:
            await self.admission.wait_turn(ticket, s.queue_timeout)
            queue_ms = int((time.monotonic() - t_wait) * 1000)
            gen, cold_ms = await m.ensure_ready()
            deadline = asyncio.get_running_loop().time() + s.gen_deadline  # cold start excluded
            input_tokens = await self._count_tokens(req, deadline)
            if input_tokens + req.max_tokens > s.n_ctx:
                raise self._context_error(input_tokens, req.max_tokens)
            headers = {
                "Cache-Control": "no-store",
                "X-Advent-Queue-Wait-Ms": str(queue_ms),
                "X-Advent-Cold-Start-Ms": str(cold_ms),
                "X-Advent-Input-Tokens": str(input_tokens),
            }
            if req.stream:
                resp = await self._stream(req, ticket, gen, headers, rec)
                handed_off = True
                return resp
            return await self._complete(req, gen, headers, rec)
        finally:
            if not handed_off:
                ticket.release()

    def _context_error(self, input_tokens: int, max_tokens: int) -> ApiError:
        s = self.s
        return ApiError(
            413, "context_length_exceeded",
            f"Запрос не помещается в контекст: {input_tokens} токенов во входе + "
            f"{max_tokens} на ответ > {s.n_ctx}.",
            extra={"input_tokens": input_tokens, "max_tokens": max_tokens, "n_ctx": s.n_ctx},
        )  # fmt: skip

    async def _count_tokens(self, req: ChatRequest, deadline: float) -> int:
        """Runs only while this request holds the admission; bounded by the request deadline."""
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        failure: BaseException | None = None
        for attempt in (1, 2):
            remaining = deadline - loop.time()
            try:
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    r = await self.manager.client.post(
                        "/v1/chat/completions/input_tokens",
                        json=req.counting_payload(),
                        timeout=httpx.Timeout(remaining, connect=min(3.0, remaining)),
                    )
                r.raise_for_status()
                n = r.json()["input_tokens"]
                if not isinstance(n, int) or isinstance(n, bool):
                    raise ValueError("input_tokens is not an int")
                return n
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
                failure = e  # transient (e.g. a stale keep-alive socket): one retry
                if attempt == 1:
                    continue
                break
            except (TimeoutError, httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                failure = e  # a timeout is never retried
                break
        log.warning(
            "token_count_failed error=%s elapsed_ms=%d",
            type(failure).__name__,
            int((time.monotonic() - t0) * 1000),
        )  # type and timing only: never message content
        raise ApiError(
            502, "token_count_failed", "Не удалось посчитать токены запроса.",
            type_="server_error",
        ) from None  # fmt: skip

    def _upstream_error(self, resp: httpx.Response, req: ChatRequest) -> ApiError:
        """Map an upstream non-200 (body already read) to our error."""
        try:
            err = resp.json().get("error", {})
        except (ValueError, AttributeError):
            err = {}
        if isinstance(err, dict) and err.get("type") == "exceed_context_size_error":
            n = err.get("n_prompt_tokens")
            if isinstance(n, int):
                return self._context_error(n, req.max_tokens)
        return ApiError(
            502, "upstream_error", f"Модель вернула ошибку {resp.status_code}.",
            type_="server_error",
        )  # fmt: skip

    async def _complete(self, req: ChatRequest, gen: int, headers: dict, rec: KeyRecord):
        m = self.manager
        outcome = "aborted"
        verify = True  # the backend may still be generating unless it answered or refused
        try:
            try:
                async with asyncio.timeout(self.s.gen_deadline):
                    r = await m.client.post(
                        "/v1/chat/completions",
                        json=req.upstream_payload(),
                        timeout=httpx.Timeout(None, connect=3.0),
                    )
            except TimeoutError:
                outcome = "timeout"
                raise ApiError(
                    504, "generation_timeout", "Генерация не уложилась в срок.",
                    type_="server_error",
                ) from None  # fmt: skip
            except httpx.HTTPError:
                outcome = "upstream_error"
                raise ApiError(
                    502, "upstream_error", "Модель оборвала соединение.", type_="server_error"
                ) from None
            if r.status_code != 200:
                outcome = "upstream_error"
                verify = False
                raise self._upstream_error(r, req)
            outcome = "ok"
            verify = False
            m.note_finished()
            return Response(r.content, media_type="application/json", headers=headers)
        finally:
            self.stats.outcomes[outcome] += 1
            if verify:
                await asyncio.shield(m.release_after_timeout(gen))
            log.info("chat key=%s stream=false outcome=%s", rec.name, outcome)

    async def _stream(
        self, req: ChatRequest, ticket: Ticket, gen: int, headers: dict, rec: KeyRecord
    ):
        m, s = self.manager, self.s
        deadline = asyncio.get_running_loop().time() + s.gen_deadline
        upstream = m.client.build_request(
            "POST",
            "/v1/chat/completions",
            json=req.upstream_payload(),
            timeout=httpx.Timeout(None, connect=3.0),  # only `deadline` bounds a generation
        )
        try:
            async with asyncio.timeout_at(deadline):
                resp = await m.client.send(upstream, stream=True)
        except TimeoutError:
            await asyncio.shield(m.release_after_timeout(gen))
            raise ApiError(
                504, "generation_timeout", "Генерация не уложилась в срок.", type_="server_error"
            ) from None
        except (httpx.HTTPError, asyncio.CancelledError) as e:
            await asyncio.shield(m.release_after_timeout(gen))
            if isinstance(e, asyncio.CancelledError):
                raise
            raise ApiError(
                502, "upstream_error", "Модель оборвала соединение.", type_="server_error"
            ) from None
        if resp.status_code != 200:
            try:
                await resp.aread()
                raise self._upstream_error(resp, req)
            finally:
                await resp.aclose()

        state = {"outcome": "aborted", "done": False}

        async def gen_events() -> AsyncIterator[bytes]:
            it = resp.aiter_raw()
            sse = SseDoneTracker()
            while True:
                try:
                    async with asyncio.timeout_at(deadline):
                        chunk = await anext(it)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    state["outcome"] = "timeout"
                    yield self._tail_fix(sse) + sse_error(
                        ApiError(504, "generation_timeout", "Генерация не уложилась в срок.",
                                 type_="server_error")
                    )  # fmt: skip
                    return
                except httpx.HTTPError:
                    state["outcome"] = "upstream_error"
                    yield self._tail_fix(sse) + sse_error(
                        ApiError(502, "upstream_error", "Модель оборвала соединение.",
                                 type_="server_error")
                    )  # fmt: skip
                    return
                sse.feed(chunk)
                yield chunk
            if sse.done:
                state["outcome"] = "ok"
                return
            state["outcome"] = "upstream_error"
            yield self._tail_fix(sse) + sse_error(
                ApiError(502, "upstream_closed", "Модель закончила поток без [DONE].",
                         type_="server_error")
            )  # fmt: skip

        cleaned = False

        async def cleanup() -> None:
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            try:
                with contextlib.suppress(Exception):
                    await resp.aclose()
                if state["outcome"] == "ok":
                    m.note_finished()
                else:
                    await m.release_after_timeout(gen)
            finally:
                self.stats.outcomes[state["outcome"]] += 1
                log.info("chat key=%s stream=true outcome=%s", rec.name, state["outcome"])
                ticket.release()  # capacity returns only after the slot check above

        def send_timed_out() -> None:
            state["outcome"] = "timeout"

        return GuardedStream(gen_events(), cleanup, headers, deadline + SEND_GRACE, send_timed_out)

    @staticmethod
    def _tail_fix(sse: SseDoneTracker) -> bytes:
        """Close a half-written event so the error event starts on a fresh line."""
        return b"\n\n" if sse.pending else b""

    # -- static -------------------------------------------------------------------------

    def static(self, name: str, media: str):
        body = (STATIC / name).read_bytes()

        async def handler(request: Request):
            return Response(
                body,
                media_type=media,
                headers={
                    "Content-Security-Policy": CSP,
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-cache",
                },
            )

        return handler


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


def create_app(
    settings: Settings,
    *,
    manager: ProcessManager | None = None,
    key_records: list[KeyRecord] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Starlette:
    records = key_records if key_records is not None else load_keys(settings.keys_file)
    gw = Gateway(settings, manager, records, clock)

    @asynccontextmanager
    async def lifespan(app):
        gw.manager.start_watchdog()
        try:
            yield
        finally:
            await gw.manager.shutdown()

    routes = [
        Route("/", gw.static("index.html", "text/html; charset=utf-8"), methods=["GET"]),
        Route("/chat.js", gw.static("chat.js", "text/javascript; charset=utf-8"), methods=["GET"]),
        Route("/chat.css", gw.static("chat.css", "text/css; charset=utf-8"), methods=["GET"]),
        Route("/health", gw.health, methods=["GET"]),
        Route("/v1/status", gw.status, methods=["GET"]),
        Route("/v1/models", gw.models, methods=["GET"]),
        Route("/v1/chat/completions", gw.chat, methods=["POST"]),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    wrapped = CountingMiddleware(app, gw.stats)
    wrapped.gateway = gw  # type: ignore[attr-defined]
    return wrapped  # type: ignore[return-value]
