"""Stand-in for llama-server (stdlib only): a REAL child process for lifecycle tests.

Argv is parsed like the real one (--port, -c). Behaviour comes from the FAKE_LLAMA env var
(JSON): startup_delay, never_healthy, stderr_bytes, props_n_ctx, chunks, chunk_delay,
gen_delay, crash_after_chunks, cancel_lag, chunk_text, no_done. A request can override chunks/delay/
crash with a magic prefix in its last message: `#chunks=5;delay=0.05;crash=2;exceed=1#`.
More magic keys: hdelay (before response headers), fdelay (after headers, before the first
chunk), watch=1 (notice a client disconnect while waiting, like llama-server's task cancel),
lag (seconds the slot stays busy after an abort; models the ~0.5 s cancel lag).
GET /_stats exposes what the backend saw (max concurrency, last headers/body, aborts).
"""

from __future__ import annotations

import json
import os
import re
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CFG = json.loads(os.environ.get("FAKE_LLAMA", "{}"))
PORT = 8791
N_CTX = 2048
argv = sys.argv[1:]
for i, a in enumerate(argv):
    if a == "--port":
        PORT = int(argv[i + 1])
    if a == "-c":
        N_CTX = int(argv[i + 1])

LOCK = threading.Lock()
STATS = {"current": 0, "max_concurrent": 0, "total": 0, "aborted": 0, "completed": 0,
         "last_headers": {}, "last_body": None, "pid": os.getpid()}  # fmt: skip
PROCESSING = {"n": 0}
MAGIC = re.compile(r"^#([\w=;.]+)#")


def count_tokens(messages) -> int:
    return sum(len(m.get("content", "")) for m in messages)


def magic(messages) -> dict:
    out: dict = {}
    m = MAGIC.match(messages[-1].get("content", "")) if messages else None
    if m:
        for part in m.group(1).split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k] = float(v)
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def wait(self, secs: float, watch: bool) -> None:
        """Sleep; with `watch`, raise ConnectionError as soon as the client has gone."""
        end = time.time() + secs
        while True:
            left = end - time.time()
            if left <= 0:
                return
            if not watch:
                time.sleep(left)
                return
            r, _, _ = select.select([self.connection], [], [], min(left, 0.02))
            if r and self.connection.recv(1, socket.MSG_PEEK) == b"":
                raise ConnectionError("client gone")

    def _json(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            if CFG.get("never_healthy"):
                self._json(503, {"error": {"message": "Loading model"}})
            else:
                self._json(200, {"status": "ok"})
        elif self.path == "/props":
            n = CFG.get("props_n_ctx", N_CTX)
            self._json(200, {"default_generation_settings": {"n_ctx": n}, "total_slots": 1})
        elif self.path == "/slots":
            self._json(200, [{"id": 0, "is_processing": PROCESSING["n"] > 0}])
        elif self.path == "/_stats":
            with LOCK:
                self._json(200, dict(STATS))
        else:
            self._json(404, {"error": "nope"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        with LOCK:
            STATS["last_headers"] = {k.lower(): v for k, v in self.headers.items()}
            STATS["last_body"] = body
        msgs = body.get("messages", [])
        if self.path == "/v1/chat/completions/input_tokens":
            self._json(200, {"input_tokens": count_tokens(msgs)})
            return
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "nope"})
            return
        mg = magic(msgs)
        prompt = count_tokens(msgs)
        if mg.get("exceed") or prompt >= N_CTX:
            self._json(400, {"error": {"code": 400, "type": "exceed_context_size_error",
                                       "message": "too long", "n_prompt_tokens": max(prompt, 3000),
                                       "n_ctx": N_CTX}})  # fmt: skip
            return
        with LOCK:
            STATS["current"] += 1
            STATS["total"] += 1
            STATS["max_concurrent"] = max(STATS["max_concurrent"], STATS["current"])
            PROCESSING["n"] += 1
        try:
            try:
                self._generate(body, int(mg.get("chunks", CFG.get("chunks", 3))),
                               mg.get("delay", CFG.get("chunk_delay", 0.0)),
                               int(mg.get("crash", CFG.get("crash_after_chunks", 0))),
                               prompt, mg)  # fmt: skip
            except OSError:
                with LOCK:
                    STATS["aborted"] += 1
                time.sleep(mg.get("lag", CFG.get("cancel_lag", 0.0)))  # slot frees late
        finally:
            with LOCK:
                STATS["current"] -= 1
                PROCESSING["n"] -= 1

    def _generate(self, body, chunks, delay, crash, prompt, mg):
        watch = bool(mg.get("watch"))
        if not body.get("stream"):
            self.wait(CFG.get("gen_delay", 0.0) + delay * chunks, watch)
            text = "ok " * chunks
            self._json(200, {"choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": text}}],
                             "usage": {"prompt_tokens": prompt, "completion_tokens": chunks,
                                       "total_tokens": prompt + chunks}})  # fmt: skip
            with LOCK:
                STATS["completed"] += 1
            return
        self.wait(mg.get("hdelay", 0.0), watch)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(text: str):
            data = text.encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            self.wait(mg.get("fdelay", 0.0), watch)
            for i in range(chunks):
                if crash and i == crash:
                    os._exit(3)
                text = CFG.get("chunk_text", f"т{i} ")
                obj = {"choices": [{"index": 0, "delta": {"content": text}}]}
                emit(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n")
                self.wait(delay, watch)
            if crash and crash >= chunks:
                os._exit(3)
            fin = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            emit(f"data: {json.dumps(fin)}\n\n")
            if (body.get("stream_options") or {}).get("include_usage"):
                u = {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": chunks,
                                              "total_tokens": prompt + chunks},
                     "timings": {"predicted_per_second": 9.5}}  # fmt: skip
                emit(f"data: {json.dumps(u)}\n\n")
            if not CFG.get("no_done"):
                emit("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            with LOCK:
                STATS["completed"] += 1
        except (BrokenPipeError, ConnectionError, OSError):
            with LOCK:
                STATS["aborted"] += 1
            time.sleep(mg.get("lag", CFG.get("cancel_lag", 0.0)))  # slot frees late


def main():
    n = int(CFG.get("stderr_bytes", 0))
    if n:
        sys.stderr.write("x" * n + "\nfake-llama noisy start\n")
        sys.stderr.flush()
    time.sleep(CFG.get("startup_delay", 0.0))
    sys.stderr.write("fake-llama listening\n")
    sys.stderr.flush()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
