"use strict";
// All untrusted text (model output, server errors) goes in through textContent only.
(function () {
  var KEY_STORE = "advent-llm-key";
  var keyEl = document.getElementById("key");
  var rememberEl = document.getElementById("remember");
  var forgetEl = document.getElementById("forget");
  var logEl = document.getElementById("log");
  var form = document.getElementById("form");
  var inputEl = document.getElementById("input");
  var sendEl = document.getElementById("send");
  var resetEl = document.getElementById("reset");
  var history = [];

  function loadKey() {
    try {
      var saved = window.localStorage.getItem(KEY_STORE);
      if (saved) { keyEl.value = saved; rememberEl.checked = true; }
    } catch (e) { /* storage may be blocked */ }
  }
  function saveKey() {
    try {
      if (rememberEl.checked && keyEl.value) window.localStorage.setItem(KEY_STORE, keyEl.value);
      else window.localStorage.removeItem(KEY_STORE);
    } catch (e) { /* ignore */ }
  }
  forgetEl.addEventListener("click", function () {
    keyEl.value = "";
    rememberEl.checked = false;
    try { window.localStorage.removeItem(KEY_STORE); } catch (e) { /* ignore */ }
  });
  rememberEl.addEventListener("change", saveKey);
  keyEl.addEventListener("change", saveKey);

  function addMsg(kind, text) {
    var el = document.createElement("div");
    el.className = "msg " + kind;
    el.textContent = text;
    logEl.appendChild(el);
    el.scrollIntoView({ block: "end" });
    return el;
  }
  function addFoot(parent) {
    var f = document.createElement("div");
    f.className = "foot";
    parent.appendChild(f);
    return f;
  }

  // Incremental SSE parser: a UTF-8 decoder with a buffer; an event ends at a blank line.
  function makeSseParser(onEvent) {
    var decoder = new TextDecoder("utf-8");
    var buf = "";
    return {
      push: function (bytes) {
        buf += decoder.decode(bytes, { stream: true });
        buf = buf.replace(/\r\n/g, "\n");
        var idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          var raw = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          var data = [];
          raw.split("\n").forEach(function (line) {
            if (line.indexOf("data:") === 0) data.push(line.slice(5).replace(/^ /, ""));
          });
          if (data.length) onEvent(data.join("\n"));
        }
      },
      finish: function () {
        buf += decoder.decode();
      },
      pending: function () { return buf.trim() !== ""; }
    };
  }

  function fmtMs(v) { return v === null ? "—" : (Number(v) / 1000).toFixed(1) + " с"; }

  async function send(text) {
    var key = keyEl.value.trim();
    if (!key) { addMsg("error", "Введите ключ доступа."); return; }
    saveKey();
    history.push({ role: "user", content: text });
    addMsg("user", text);
    var out = addMsg("assistant", "");
    var foot = addFoot(out);
    var answer = "";
    var usage = null;
    var timings = null;
    var failed = null;
    var sawDone = false;
    var finishReason = null;
    var malformed = false;
    var firstAt = null;
    sendEl.disabled = true;

    try {
      var resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": "Bearer " + key },
        body: JSON.stringify({ messages: history, stream: true })
      });
      var hdr = function (n) { var v = resp.headers.get(n); return v === null ? null : Number(v); };
      var cold = hdr("X-Advent-Cold-Start-Ms");
      var queued = hdr("X-Advent-Queue-Wait-Ms");
      var inTok = hdr("X-Advent-Input-Tokens");

      if (!resp.ok) {
        var msg = "Ошибка " + resp.status;
        try {
          var j = await resp.json();
          if (j && j.error && j.error.message) msg += ": " + j.error.message;
        } catch (e) { /* not JSON */ }
        var ra = resp.headers.get("Retry-After");
        if (ra) msg += " (повторить через " + ra + " с)";
        history.pop();
        out.className = "msg error";
        out.insertBefore(document.createTextNode(msg), foot);
        return;
      }

      var textNode = document.createTextNode("");
      out.insertBefore(textNode, foot);
      var parser = makeSseParser(function (data) {
        if (data === "[DONE]") { sawDone = true; return; }
        var obj;
        try { obj = JSON.parse(data); } catch (e) { malformed = true; return; }
        if (obj.error) { failed = obj.error.message || obj.error.code || "ошибка"; return; }
        var choice = obj.choices && obj.choices[0];
        if (choice && typeof choice.finish_reason === "string") finishReason = choice.finish_reason;
        if (choice && choice.delta && typeof choice.delta.content === "string") {
          if (firstAt === null) firstAt = performance.now();
          answer += choice.delta.content;
          textNode.nodeValue = answer;
        }
        if (obj.usage) usage = obj.usage;
        if (obj.timings) timings = obj.timings;
      });
      var reader = resp.body.getReader();
      for (;;) {
        var step = await reader.read();
        if (step.done) break;
        parser.push(step.value);
      }
      parser.finish();

      var parts = [];
      parts.push("холодный старт: " + (cold ? fmtMs(cold) : "нет"));
      parts.push("очередь: " + fmtMs(queued));
      var outTok = usage ? usage.completion_tokens : null;
      parts.push("токены: " + (inTok === null ? "?" : inTok) + " → " + (outTok === null ? "?" : outTok));
      var tps = null;
      if (timings && timings.predicted_per_second) tps = timings.predicted_per_second;
      else if (outTok && firstAt !== null && performance.now() > firstAt) {
        tps = outTok / ((performance.now() - firstAt) / 1000);
      }
      parts.push("скорость: " + (tps === null ? "?" : tps.toFixed(1) + " ток/с"));
      // Complete = terminal [DONE] event after a finish_reason, nothing lost or left half-parsed.
      var complete = sawDone && finishReason !== null && !malformed && !parser.pending();
      if (failed) { parts.push("обрыв: " + failed); out.classList.add("error"); }
      else if (!complete) {
        parts.push("ответ оборван — в историю не попал");
        out.classList.add("error");
      }
      foot.textContent = parts.join(" · ");
      if (answer && !failed && complete) history.push({ role: "assistant", content: answer });
      else history.pop();
    } catch (e) {
      out.classList.add("error");
      foot.textContent = "Сеть: " + (e && e.message ? e.message : "ошибка");
      history.pop();
    } finally {
      sendEl.disabled = false;
    }
  }

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    send(text);
  });
  resetEl.addEventListener("click", function () {
    history = [];
    while (logEl.firstChild) logEl.removeChild(logEl.firstChild);
  });
  loadKey();
})();
