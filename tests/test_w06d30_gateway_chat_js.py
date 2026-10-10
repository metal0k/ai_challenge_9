"""chat.js executed for real (node + stubbed DOM): incomplete streams must not pass as answers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CHAT_JS = Path(__file__).parent.parent / "week_06" / "gateway" / "static" / "chat.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const [, , jsPath, scenarioPath] = process.argv;
const scenario = JSON.parse(fs.readFileSync(scenarioPath, "utf8"));

function mkEl(tag) {
  const e = {
    tag, children: [], className: "", value: "", checked: false, disabled: false, _text: "",
    handlers: {}, style: {},
    classList: { add(c) { if (!e.className.split(" ").includes(c)) e.className = (e.className + " " + c).trim(); } },
    addEventListener(t, f) { e.handlers[t] = f; },
    scrollIntoView() {},
    appendChild(c) { e.children.push(c); return c; },
    insertBefore(c, ref) {
      const i = e.children.indexOf(ref);
      if (i < 0) e.children.push(c); else e.children.splice(i, 0, c);
      return c;
    },
    removeChild(c) { e.children.splice(e.children.indexOf(c), 1); },
    get firstChild() { return e.children[0] || null; },
  };
  Object.defineProperty(e, "textContent", {
    get() { return e._text; },
    set(v) { e._text = v; e.children = []; },
  });
  return e;
}

const ids = {};
["key", "remember", "forget", "log", "form", "input", "send", "reset"].forEach((i) => { ids[i] = mkEl(i); });
const enc = new TextEncoder();
const bodies = [];
let turn = 0;

const sandbox = {
  document: {
    getElementById: (i) => ids[i],
    createElement: (t) => mkEl(t),
    createTextNode: (t) => ({ nodeType: 3, nodeValue: t }),
  },
  window: { localStorage: { getItem: () => null, setItem() {}, removeItem() {} } },
  TextDecoder, performance,
  fetch: async (url, opts) => {
    bodies.push(JSON.parse(opts.body).messages);
    const t = scenario.turns[turn++];
    let i = 0;
    return {
      ok: t.status === 200, status: t.status,
      headers: { get: (n) => (t.headers && n in t.headers ? String(t.headers[n]) : null) },
      json: async () => t.json,
      body: { getReader: () => ({ read: async () => (i < t.chunks.length ? { done: false, value: enc.encode(t.chunks[i++]) } : { done: true }) }) },
    };
  },
};
vm.runInNewContext(fs.readFileSync(jsPath, "utf8"), sandbox);

function dump(n) {
  if (n.nodeType === 3) return n.nodeValue;
  return { cls: n.className, text: n.textContent, kids: n.children.map(dump) };
}

(async () => {
  ids.key.value = "adv_key";
  for (const text of scenario.inputs) {
    ids.input.value = text;
    ids.form.handlers.submit({ preventDefault() {} });
    while (ids.send.disabled) await new Promise((r) => setImmediate(r));
  }
  process.stdout.write(JSON.stringify({ bodies, log: ids.log.children.map(dump) }));
})();
"""  # noqa: E501 - embedded JS


def ev(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def content(text: str) -> str:
    return ev({"choices": [{"index": 0, "delta": {"content": text}}]})


FINISH = ev({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
USAGE = ev({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
DONE = "data: [DONE]\n\n"
NEXT_OK = [content("второй"), FINISH, USAGE, DONE]


def run(tmp_path, first_chunks, first_status=200):
    scenario = {
        "inputs": ["первый", "второй вопрос"],
        "turns": [
            {"status": first_status, "chunks": first_chunks, "headers": {}},
            {"status": 200, "chunks": NEXT_OK, "headers": {}},
        ],
    }
    (tmp_path / "harness.js").write_text(HARNESS, encoding="utf-8")
    (tmp_path / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")
    p = subprocess.run(
        ["node", str(tmp_path / "harness.js"), str(CHAT_JS), str(tmp_path / "scenario.json")],
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )  # fmt: skip
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def assistant(out, n):
    msgs = [m for m in out["log"] if "assistant" in m["cls"]]
    return msgs[n]


def foot_text(msg) -> str:
    return msg["kids"][-1]["text"]


def test_complete_stream_is_kept_in_history(tmp_path):
    out = run(tmp_path, [content("при"), content("вет"), FINISH, USAGE, DONE])
    first = assistant(out, 0)
    assert "ответ оборван" not in foot_text(first)
    assert "error" not in first["cls"]
    assert out["bodies"][1] == [
        {"role": "user", "content": "первый"},
        {"role": "assistant", "content": "привет"},
        {"role": "user", "content": "второй вопрос"},
    ]


@pytest.mark.parametrize(
    "chunks",
    [
        [content("часть"), content(" ответа")],  # clean close, no terminal event
        [content("текст [DONE] внутри"), FINISH],  # the marker is content, not an event
        [content("часть"), "data: [DONE]"],  # terminal event never completed
        [content("часть"), DONE],  # [DONE] without a finish_reason
        [content("часть"), "data: {broken json\n\n", FINISH, DONE],  # a lost event
    ],
    ids=["eof", "done-in-content", "partial-done", "no-finish-reason", "malformed-event"],
)
def test_incomplete_stream_is_marked_and_excluded_from_history(tmp_path, chunks):
    out = run(tmp_path, chunks)
    first = assistant(out, 0)
    assert "ответ оборван" in foot_text(first)
    assert "error" in first["cls"]
    assert out["bodies"][1] == [{"role": "user", "content": "второй вопрос"}]


def test_server_error_event_still_excluded(tmp_path):
    err = ev({"error": {"code": "upstream_error", "message": "обрыв модели"}})
    out = run(tmp_path, [content("часть"), err])
    assert "обрыв: обрыв модели" in foot_text(assistant(out, 0))
    assert out["bodies"][1] == [{"role": "user", "content": "второй вопрос"}]
