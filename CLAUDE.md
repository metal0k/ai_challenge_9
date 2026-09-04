# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coursework for [AI Advent](https://mobiledeveloper.tech/ai_advent_9), 9th cohort
(hence the `_9` in the URL and the repo name) — **7 weeks**, five tasks a week,
built on the Mistral API. Each finished day is submitted as a pair of links
(code + video) commented into a shared Google Sheet.

### Schedule and deadlines

A task drops every weekday at **14:00 MSK**; the week's five are due by **14:00
MSK the following Monday**. Miss the set and you are dropped from the chat and
get no further tasks. A theory video lands on Mondays, ahead of that week's
tasks. Every submission is working code **plus a demo video** plus a short
explanation — public demonstration is mandatory, not optional. The course budgets
1–2 hours a day, which is the real constraint: `advent record` and
`advent submit` exist so that shipping a day costs minutes, not an evening.

### Programme

Theme names are the course's own, kept verbatim.

| Week | Theme | Covers |
|------|-------|--------|
| 1 | Основы LLM | entry into the topic |
| 2 | База про агентов | agent implementation, keeping and **compacting** context, token accounting |
| 3 | Оптимизация агента | state management |
| 4 | MCP | Model-Context-Protocol |
| 5 | RAG | retrieval augmented generation applied to your own projects |
| 6 | Локальный ИИ | a local LLM run as a private service |
| 7 | Пайплайн | integrating an LLM into your own tasks and automating them |

An Advanced track (+3 weeks: Code Assistance, FineTune, Security) is a separate
product, not part of this one.

**What the programme implies for this repo.** Weeks 2–3 grow an agent on top of
the existing client, so context compaction and token accounting belong in
`advent_core`, not inside a week folder. Weeks 4 and 6 do not: an MCP server and
a local model server are separate processes, so "one app that grows day by day"
holds *within* a week, not across the course — `week_04` and `week_06` will need
their own entry points rather than another `advent wNN` typer group. The JSONL
call journal and the token telemetry written on Day 01 are week-2 and week-5
material, not decoration.

## Commands

```bash
uv sync                                   # install/refresh the environment
uv run advent w01 chat "вопрос"           # one-shot answer
uv run advent w01 chat                    # REPL with history
uv run advent w01 models                  # models from the live API (--all for non-chat)
uv run advent w01 solve                   # one problem, four reasoning strategies, judged
uv run advent w01 solve --strategy panel --runs 3 --no-judge
uv run advent w01 temp                    # one prompt at t=0 / 0.7 / 1.2, no judge
uv run advent w01 bench                   # one prompt across ministral 3b/8b/14b
uv run advent w01 bench --runs 10 --problem children
uv run advent record --day 1              # record the demo through OBS
uv run advent record --day 1 --dry-run    # run the demo without recording
uv run advent submit --day 1              # verify tag + video, print the sheet comment

uv run ruff check . && uv run ruff format .
uv run pytest
uv run pytest tests/test_params.py::test_reasoning_effort_is_skipped_for_incapable_model
```

Tests never hit the network and never spend API credits — the client is mocked.

## Architecture

Three layers, deliberately separated:

- **`advent_core/`** — shared across all weeks: config, generation params, Mistral
  client, chat/streaming, telemetry, JSONL journal, console output.
- **`week_NN/`** — all code for that week in one folder, with the task statements
  in one `tasks.md` beside it (`## Day NN` sections, not a file per day). The
  week is **one application that grows day by day**, not one file per day.
- **`advent_cli/`** — the `advent` entry point, plus `record` (OBS) and `submit`
  (GitHub + Yandex Disk). A week registers its own typer group, so `advent w01 …`
  keeps working after later weeks move ahead.

Folder numbering goes **at the end** (`week_01`, not `01_Week`) so the name is a
valid Python identifier and the folder imports as a package.

Since the app grows by day, the state of a submitted day is pinned by a git tag
`wNNdDD` (`w01d01`). Sheet links point at `tree/<tag>`, never at `main` — the
reviewer checks on Mondays and must see what the video showed.

## Things that will bite you

**The `mistralai` SDK is v2.** Imports moved from `mistralai` to
`mistralai.client` (`mistralai.client.utils` for `RetryConfig`/`BackoffStrategy`).
Any snippet using `from mistralai import Mistral` is v1 and will not work. The
version is pinned in `pyproject.toml` for this reason.

**Unset generation params are not sent at all.** The API knows a recommended
`default_model_temperature` per model — 0.3 for `mistral-small-2603`, 1.0 for
`magistral-medium`. A hardcoded client-side constant would be worse for half the
models. `None` means "let the server decide"; the recommendation is only
*displayed* (the `t°` column).

**Params are filtered against the model's `capabilities`.** `reasoning_effort`
goes to models with `reasoning: true` and is dropped with a stderr warning for
the rest. Add a new gated param by setting `requires=` on its `Spec` in
`advent_core/params.py` — that is the single place that knows the rules.

**The API echoes back the model name you sent**, so `-latest` stays `-latest` in
the response. The concrete version shown in the footer and the log is resolved
from the models list (`resolve_alias`), not from the response.

**JSON mode does not decide the shape of the JSON — only that it's valid.**
`response_format={"type": "json_object"}` guarantees parseable JSON, nothing
about its keys. The Mistral docs say it in one sentence: "When using JSON mode
you MUST also instruct the model to produce JSON yourself with a system or a
user message." Skipping the prompt instruction because the parameter "already
guarantees it" is the most common way to get valid-but-wrong JSON.

**A `stop` sequence never reaches the output.** Per the Mistral docs: "The
output will not contain the stop sequence." That rules out `stop` as the
completion marker for a multi-turn dialog loop — if the model is told to end
the conversation with a marker string and that same string is also passed as
`stop`, the API strips it before the caller ever sees it, and there is nothing
left to detect. The marker and the `stop` value must never be the same string.

**No model in this account's 48-model list carries a capability for
structured output.** `capabilities` has `function_calling`, `vision`,
`reasoning`, and a few more — nothing named for `response_format`/JSON
schema. So `response_format` cannot be gated the way `reasoning_effort` is;
it is sent unconditionally, and a model's refusal comes back as a 400 that
`errors.py` translates into a hinted message instead of a traceback.

Gating it on `function_calling` as a proxy was considered and rejected — and a
live sweep later proved the point: **all 29 chat models on this account accept
`json_schema`, and all 29 carry `function_calling: true`**. The proxy would have
filtered out nothing at all, while looking like a safeguard. A demo step built on
"this model refuses structured output" had to be rewritten for the same reason:
no model on this account refuses.

**Prompting strategies only diverge on a narrow class of problem.** A live sweep
of 18 candidates, 4 runs each on `mistral-small-latest`, is what the comparison
day rests on, and almost nothing separated the four strategies:

- Classic trick riddles ("the fifth daughter", "five machines in five minutes",
  counting letters in a word) are solved 4/4. They sit in the training data —
  the model recalls them rather than reasoning, so every strategy wins.
- Long arithmetic chains are solved 4/4 too. The model counts better than the
  premise of the day assumed.
- The one class that broke consistently: **counting relations where the child is
  not counted among their own siblings**. `children` (answer 7) missed 7 times
  out of 8 on a direct answer and was right 3 out of 4 step by step.
- **Two of the hand-written reference answers were themselves wrong** (socks: 6,
  not 7; trains: 265.7 km, not 240 — and there the model had been right all four
  times). A key in the problem bank is as much an object of verification as the
  model's output; `"placeholder": true` marks one that has not been checked
  against a live run.

So the demo problem is pinned by an explicit `"default": true` flag in the
bank rather than by "first file alphabetically" — that implicit link breaks the
moment a problem with an earlier id is added.

**A candidate that misses is not yet a candidate that separates the strategies,
and only the finished pipeline can tell you which it is.** `children` (answer 7)
survived the 18-candidate sweep on those numbers and was then measured again on
the assembled `solve` command, 5 runs per strategy: step-by-step was right 2 out
of 5 — exactly as often as the direct answer, so the day had nothing to compare.
The problem that shipped is `alice` (answer 3, "how many sisters does Alice's
brother have"): **1 out of 10 direct against 10 out of 10 step by step**.
Re-measure the gap on the same path that will run on camera, not on the sweep
harness — and re-measure it with enough runs. An early 5-run sample put `steps`
at 4/5; ten runs put the same prompt at 5/10. Five runs cannot tell 50% from 80%.

**The wording of the step-by-step instruction moved the result more than
anything else in the day.** Three prompts, 10 runs each on `alice`:

| instruction | correct |
|---|---|
| direct answer, no instruction | 0/10 |
| numbered rules ending in "substitute the result back and check" | 5/10 |
| "break the statement apart, introduce notation, write the equations" | 9/10 |

The verification rule was the problem: asked to check its own answer, the model
re-derived the sibling count from the wrong point of view and talked itself out
of the right answer. Naming the entities up front beat checking the result
afterwards, so `reason_steps.md` is the short algebraic form.

A fourth variant scored 9/10 too by adding "note whose point of view the
question is asked from" — a hint aimed straight at this problem's trap. It was
rejected: a strategy prompt tuned to the benchmark problem measures the tuning,
not the strategy.

**The project's default system prompt is deliberately NOT mixed into a strategy
call, because that persona cancels the effect the day measures.**
`advent_core/prompts/default_system.md` says "be a concise assistant, answer to
the point, no filler", and the model obeys it over "reason step by step".
Measured on `children`, `mistral-small`, 5 runs each: with the persona, `steps`
is right **2 out of 5** at an average answer length of **679 characters**;
without it, **4 out of 5** at **1232 characters**. The persona halves the
reasoning and doubles the error rate. So `build_strategy_system()`
(`week_01/strategies.py`) drops `config.system_prompt()` whenever the path is
still `DEFAULT_SYSTEM_PROMPT`, and keeps it only when the user set `--system` /
`ADVENT_SYSTEM_PROMPT` explicitly — an explicit choice is not ours to ignore.
The rule generalises: **any day that measures reasoning quality has to disable
the default persona first, or it is measuring the persona.** Anything layered on
top of `chat_core` later (the week-2 agent) inherits this trap.

**`strategy=direct` must stay the plain chat path, not a strategy that happens to
add nothing.** Every strategy appends the same `ОТВЕТ:` marker instruction and
runs without streaming. Routing `direct` through that machinery would make an
ordinary `advent w01 chat "привет"` demand a marker line and lose the stream, so
`chat` sends a question into `week_01/strategies.py` only when the strategy is
*not* `direct`.

**A default value differs per command, and the params registry is the only place
that knows it.** `strategy` defaults to `direct` in `chat` and `all` in `solve`;
`judge` is on in `solve`, off in `chat`. That lives in `Spec.defaults` +
`apply_defaults(command)`, not in the two typer signatures — otherwise one rule
is written down twice and drifts. Consequence for the REPL: `/set <param>
default` restores the *command's* default (in the REPL, always `chat`'s), which
is not the same thing as `None`.

**A default argument binds at import time, which defeats the way these tests
remove the network.** `strategies.solve()/run_strategy()/judge()` take
`complete: CompleteFn = chat_core.complete`; that default is captured when
`week_01.strategies` is imported, so `monkeypatch.setattr(cli.chat_core,
"complete", fake)` — how the existing tests mock the client — never reaches it.
The CLI therefore passes `complete=chat_core.complete` explicitly at every call
site. Any new seam with a function default needs the same treatment.

**An LLM judge disagreeing with the reference answer is a result, not a bug.**
The judge is given the four answers under labels A–D with the strategy names
stripped (otherwise "panel of experts" wins on its name alone) and without the
key, so it grades the reasoning. When its first place is wrong by the key, the
output says so in words instead of hiding it. Its known limitation — a fixed
A–D order, hence positional bias — is stated rather than compensated for.

**stdout/stderr contract:** the command's *product* goes to stdout, everything
else (footer, warnings, REPL prompt, input echo, the problem statement) to
stderr. This keeps `advent w01 chat "…" > answer.txt` honest. Do not print
status to stdout.

For `chat` the product is the model's answer. For the analysis commands it is
the analysis: `models`, `solve` (`print_comparison`), `temp`
(`print_cell_table`, `print_conclusions`) and `bench` all print their tables,
verdicts and links to stdout, and have since Day 01, Day 03 and Day 04
respectively. A Day 05 review flagged `bench` for this; the finding was
rejected and the rule reworded instead, because the alternative was either one
day that behaves unlike the rest or a rewrite of an already-submitted day's
output. Per-run model answers still go to stdout in those commands too — they
are part of the product a reader compares.

**Windows encoding, twice over:**
- `console.force_utf8()` runs first thing in `main()`; without it the console
  falls back to cp1251 and Cyrillic in the answer becomes garbage on camera.
- `subprocess.Popen(text=True)` takes its pipe encoding from the locale (cp1252
  here), so `encoding="utf-8"` is passed explicitly wherever Cyrillic is fed to a
  child process. This one only fails inside Windows Terminal, not from a shell
  that already happens to be UTF-8 — it will pass your tests and fail on camera.

**Errors never reach the user as a traceback.** `advent_core/errors.py` maps
status codes to messages and exit codes (2 config, 3 auth, 4 rate limit, 5 server,
6 network, 7 truncated stream). `translate()` sniffs the status via duck typing
rather than importing SDK exception classes, which move between versions.
`ConfigError` is separate from `AdventError` — the REPL catches both so a typo in
`/model` or `/set` warns instead of killing the session.

**Mistral's `temperature` ceiling is 1.5, not 2.0.** Verified 2026-09-03 by raw
REST against `mistral-small-latest` and `magistral-medium-latest`: `1.5` returns
200, `1.51` and `2.0` return 422 with the server's own validator text —
`"Input should be less than or equal to 1.5"`, `ctx: {"le": 1.5}`. The 0–2 range
is OpenAI's; Mistral does not have it. `params.py` shipped `maximum=2.0` from
Day 01, so `--temperature 2` passed local validation and died on the API. The
bound now lives in exactly one place, and the three prose copies of "0..2" that
had drifted out of sync with it (`cli.py` flag help, `.env.example`,
`specs/SPEC.md`) are a standing reminder: a range written as text next to a
range written as code is a second source of truth.

**`temperature=0` is not deterministic, and `random_seed` does not fix it.**
Same prompt, 5 runs: 2 distinct answers at `t=0`, and still 2 with
`random_seed=42`. On the assembled `temp` pipeline at 10 runs: 2 distinct out of
10 on `alice`, **5 out of 10** on `digits5`, 2 out of 10 on `coffee` — not one
cell reached "1 of 10". The seed is not broken: at `t=0` decoding is greedy,
there is no sampling step for a seed to control, and what is left comes from
floating-point ordering in parallel server-side computation — a property of
every LLM API. `digits5` diverges most because its reasoning chain is the
longest and therefore has the most places to diverge. Mistral's docs promise
"deterministic results" from `random_seed`; that promise does not survive
measurement. The narrow, defensible claim is "`t=0` does not guarantee the same
answer", not "seed does not work".

**Temperature does not raise or lower accuracy — it widens the spread around
whatever the model already considers most likely.** Measured on the assembled
pipeline, 10 runs per cell, `mistral-small-latest`:

| task | t=0 | t=0.7 | t=1.2 |
|---|---|---|---|
| `digits5` (model's modal answer is **right**) | **10/10** | 8/10 | 5/10 |
| `alice` (model's modal answer is **wrong**) | **0/10** | 2/10 | 5/10 |

When the modal answer is right, spread can only hurt. When it is wrong, spread
is the only source of a correct answer — `alice` goes from 0/10 to 5/10 as
temperature rises. Both converge on 5/10 at 1.2, where the model leans less on
its own preference and more on chance. So **"low temperature means accuracy" is
a myth of the same kind as "t=0 means deterministic"**: low temperature buys
*repeatability*, and if the modal answer is wrong it guarantees being wrong
every single time. The practical consequence is the opposite of the intuitive
one — for a task the model reliably fails, several high-temperature runs plus a
vote beat one run at zero.

This is also why `TEMP_PROBLEMS` holds **both** `digits5` and `alice`. Either
one alone produces a confident and wrong generalisation, and the question "which
temperature is more accurate" has no answer that is not a lie until you name the
task. Never drop one of the pair to shorten a demo.

**Format compliance is a property of the prompt, not of the temperature.** An
early REST probe showed the model emitting three variants with commentary at
`t≥1.2` when asked to "answer in one line", which looked like temperature
breaking the format. With `«без пояснений и без вариантов»` added to the
statement, compliance is **10/10 at every temperature on every task**. The
day's own problem-bank `note` had predicted the breakdown and was rewritten
against the measurement. Two rules follow: a claim that a parameter degrades
output has to be re-tested against a *strong* instruction before it is believed,
and a `note` in the bank must never be printed before the run that would
confirm it — `print_problem_header()` deliberately does not print it, because
pre-announcing the finding turns a measurement into a formality.

**Diversity and creativity are different things, and only diversity is
monotonic in temperature.** Distinct answers rise on every task (5→10→10,
2→10→10, 2→8→9). Creativity does not follow: a judge run during development
ranked `t=0.7` above both `t=0` and `t=1.2` on the creative task, faulting the
1.2 answer for "almost duplicating answer A, losing originality to letter
case". Measuring "creativity" as "distinct count" would have inverted the
result — the two must never be collapsed into one number.

**Creativity is not scored by a model in this project; the day prints the data
and the person judges.** Day 04 shipped an LLM judge first and it was removed on
the user's instruction — "отдай оценку креативности человеку, просто выведи
данные". What replaced it is every run's answer, grouped by temperature, so a
reader compares them at a glance. The point is not that the judge was
inaccurate: it is that on an open task with no reference there is nothing to be
accurate *against*, so a number invents an authority the measurement does not
have. Note the asymmetry with Day 03, where the judge stays: there it grades
*reasoning* against a problem that has a right answer, and its disagreement with
the key is itself a reportable result. A judge is defensible where a ground
truth exists to disagree with, and decorative where none does. When removing
such a thing, remove it — prompt, schema, parser, column, flags and tests — do
not leave it behind a default-off flag; a switch nobody turns on is the same
dead weight plus a maintenance claim.

**A "winner" printed on a tie is a lie the day cannot afford.** `max()` returns
the first maximum, so a flat accuracy column (0/3, 0/3, 0/3 — a likely outcome
on `alice` at three runs) would have printed "точнее всего — t=0.0 (0/3)",
naming a winner where nobody won. Every summary line built from `max()`/`min()`
over measured cells needs an explicit tie check before it claims a leader.

**A per-module `DAY` constant stops working the moment one module holds two
days' commands.** `week_01/cli.py` writes `day=DAY` into the JSONL journal;
`temp` was added on Day 04 while `chat`/`solve` were last touched on Day 03, so
one constant would mislabel one of them. There are now `DAY` and `TEMP_DAY`.
The test that was supposed to catch this compared the journal against
`cli.DAY` — the same constant the journal is filled from — so it passed at any
value. **A test whose expected value comes from the same source as the code
under test cannot go red**; assert the literal.

**Price per token and price per response rank models in opposite order — "which
model is cheaper" is not a well-formed question without a length.** Measured
2026-09-04 on the `children` task, 5 runs, t=0: `ministral-8b-latest` is
cheaper than `ministral-14b-latest` by a third on the price list ($0.15 vs
$0.20 per 1M output tokens), and **twice as expensive per actual response**
($0.068 vs $0.035 per 1000 answers) — because it writes a median of 289
output tokens where 14b writes 9. A price-list column alone answers a
question nobody asked; the day prints both `$/1M tokens` and `$/1000
responses` for exactly this reason (SPEC-w01d05.md §5, §10).

**Latency is not a property of a model — it is a property of how much the
model writes.** Same measurement: `ministral-8b-latest` is the **slowest**
of the three by total response time (5.3 s median vs 2.6 s/2.9 s) and the
**fastest by 16x** on ms-per-output-token (18 ms vs 289 ms/319 ms) — it is
slow because it is verbose, not because it is slow. Caveat that must travel
with the ms/token column: at a 9-token median response (3b, 14b), "ms per
token" is almost entirely fixed overhead (network, queue, prefill), not
generation speed — the ratio is honest, reading it as "generation speed" is
only valid where the response is actually long.

**Accuracy is measurable only together with a format instruction — without
one, a 0/N score means "not measured", not "wrong".** `check_answer()`
returns a `no_marker` verdict, distinct from a wrong-answer verdict, when the
`ОТВЕТ:` marker it's told to extract from never appears. Dropping the marker
instruction to see a "raw" accuracy number does not measure worse — it
measures nothing, and a naive reader sees a zero and reads it as a quality
failure of the model rather than an absence of the thing being checked for.

**Format-instruction compliance does not scale monotonically with model
size.** With the `ОТВЕТ:` marker instruction, `ministral-3b-latest` and
`ministral-14b-latest` both comply (median 9 output tokens); the *middle*
model, `ministral-8b-latest`, does not (483 tokens) — measured
2026-09-04, 3 runs (SPEC-w01d05.md §6). "Bigger/smaller model" is not a
predictor of "follows this instruction or not"; it has to be measured per
model, not assumed to move with parameter count.

**A model that answered yesterday can return a stable 429 today, and it is
not the 403 this project already has a message for.** `mistral-small-latest`
and `mistral-medium*` went from working on Day 02 to `x-ratelimit-limit-req-minute:
0` on 2026-09-04 — a per-model block, distinct from the `tier_not_allowed` 403
that `mistral-large-latest` returns. `0` is not "rate exhausted, retry later";
four probes 30 s apart all returned `0` for the same model while
`ministral-14b` returned 200 in the same seconds. `errors.py`'s 429 hint now
says to try another model via `--model` — retrying the same model on a
timer will not recover from this shape of 429; check the header, don't infer
the cause from the status code alone.

**The cold-first-call hypothesis was tested and refuted, on this project's own
data.** "The first call to a model is cold and should be discarded" sounds
plausible enough to design around — it was measured instead: first-run median
vs. rest-of-cell median across 5 cells × 5 runs per model (2026-09-04) is
2685 ms vs 2918 ms (3b), 4845 ms vs 5506 ms (8b), 2741 ms vs 2885 ms (14b).
The first call is not slower — if anything it is marginally faster. `bench`
ships with no warm-up call for this reason: a warm-up would have been a cost
paid against an effect that does not exist.

## OBS recording

`advent record` drives OBS through obs-websocket. Non-obvious constraints, all
learned the hard way:

- **Configure and probe the scene while it is the active program scene.** Window
  capture only refreshes its texture while the source is showing; a screenshot
  taken before `program_scene` returns a black frame on a perfectly good setup.
- `ensure_scene()` **re-applies the source settings every run.** A source created
  while no terminal window existed stays empty forever — OBS does not re-bind.
- `verify_capture()` checks the frame is not black **before** recording starts.
  A black video is the worst failure mode: you only find out afterwards.
- `fit_to_canvas()` scales the capture to the canvas. Without it a small terminal
  sits in a corner of a 1920x1080 frame surrounded by black.
- The record directory and program scene are switched temporarily and restored
  with verification and retries — the user's active profile may write into an
  unrelated project's folder.
- OBS may record `.mkv`. The remux to `.mp4` writes to `0101.part.mp4`, **keeping
  the `.mp4` extension** — ffmpeg picks the container from the extension and
  cannot make sense of `0101.mp4.part`. The old target is replaced only after
  ffmpeg succeeds.
- `record` rehearses the subprocess/pipe machinery (with Cyrillic input) before
  starting the recording, so encoding bugs surface in seconds instead of in a
  finished file.
- **obsws-python prints a traceback for every failed request, including ones the
  caller catches and retries successfully.** `reqs.py` does
  `logger.exception(...)` immediately before `raise`, so a traceback on screen
  means "one attempt failed", not "the program crashed". On the Day 03 recording
  `SetRecordDirectory` returned 500 on the first restore attempt — OBS was still
  finalising the container after `StopRecord` — and succeeded on the second: the
  directory came back, the exit code was zero, and a traceback sat on the console
  anyway. `connect()` therefore raises the `obsws_python` logger to CRITICAL. Our
  own errors are unaffected: they surface as `AdventError` text or as the
  explicit warning from `_restore()`.
- **The demo prints into the current console, and OBS captures a window by
  title**, so recording only works from a visible Windows Terminal whose title
  matches — `tools/record_demo.ps1` sets it and is the entry point. A headless or
  background shell produces a black frame, which `verify_capture()` catches
  before recording rather than after.
- **OBS accepts a record directory that does not exist** — `SetRecordDirectory`
  to `Q:/nope` returns success and changes the setting. It is not a validation
  point, so never probe request failures with it; a bad scene name fails cleanly
  and changes nothing.

The demo scenario lives in `demo_steps()` in `advent_cli/record.py` and is
deliberately fixed: the model answers differently every time, the script must not.

## Submitting a day

`advent submit --day N` refuses to print links until the tag exists **on the
remote** — a local-only tag yields a 404 for the reviewer. It then publishes the
video via the Yandex Disk API (`PUT /resources/publish`, then read `public_url`)
and prints the comment. `--video <url>` skips publication when the token is absent.

## Publishing to the public repo

This repository is public, so two habits are mandatory.

**Run `uv run python tools/check_staged.py` before every commit that will be
pushed.** It reads the staged content and exits non-zero on local paths, tokens,
the course spreadsheet link and personal Disk links. Searching for API-key
patterns alone is not enough — personal data hides in defaults and docs. A local
path leaked through `.env.example`'s `VIDEO_DIR=` default and through a table row
in the spec, and a key-pattern scan walks straight past both.

Three things that script gets right and an inline `grep` pipeline did not, all
found by feeding it a deliberate canary:

- It reads content from the **index** (`git show :file`), not the working tree —
  what gets committed and what sits on disk diverge more often than you expect.
- It matches **fixed strings**. Backslash patterns expand differently in the
  shell and in ERE; `grep -E 'D:\\Denis'` silently matched nothing while
  `grep -F` found it.
- It skips files that document the patterns themselves, so `CLAUDE.md` does not
  trip the check by quoting it.

Personal notes stay out of the repo entirely: `spec.md` and `specs/` are
gitignored. Anything naming a person, a local path, or a shared course document
(which carries *other people's* names) does not belong here.

**Pasted chat transcripts are the leak this repo actually came closest to
shipping.** Clarifications from the course chat were pasted verbatim into
`week_01/tasks.md` — three real participant names, staged for a public push, and
`check_staged.py` passed it: its needles look for paths and tokens, and a
person's name is neither. Names cannot be added to the guard either, since the
guard itself is public and the list would leak with it. What the guard now
matches is the **shape** of a chat export — a `[dd.mm.yyyy hh:mm]` timestamp at
the head of a line, plus `in reply to` — which catches a pasted transcript
without knowing anything about who is in it. When a task statement needs a
clarification from the chat, paraphrase it; keep verbatim quotes in `specs/`,
which is gitignored.

**Never put a token in a URL you hand to git.** `git push -u <url-with-token>`
writes that URL into `.git/config` as the branch's upstream, where a later
`git config --get` will happily print it. Push to `origin` with credentials
supplied out of band instead.

**Make the guard abort, not merely print.** Two shapes that look like a check and
are not, both of which shipped a commit here that should have been blocked:

- `check && echo STOP || echo clean && git commit` — `&&` and `||` are
  left-associative, so `git commit` runs after the `echo` succeeds, whichever
  branch was taken.
- A failing command on its own line followed by more lines. Nothing propagates:
  a heredoc that raised an exception was followed by `git commit && git push`,
  and both ran.

Run the check as its own step, honour its exit code, and only then commit — or
put `set -e` at the top of the script. The same applies to every pre-flight here:
`verify_capture()` raises instead of warning, `_check()` in `record.py` raises on
an unexpected exit code. A check whose failure path continues is worse than no
check, because it reads as verified.

**A force-push does not erase anything on GitHub.** Orphaned commits stay
reachable by SHA until a garbage collection that may never run on a public repo.
If secrets or personal data reach the remote, the only reliable fix without
GitHub Support is deleting and recreating the repository — which is why the
pre-push scan matters more than the cleanup.

## Secrets

`.env` is gitignored from the very first commit, `.env.example` documents every
key. `redact()` strips `.env` values from anything printed or logged, because SDK
error text and HTTP bodies can echo the key back. `spec.md` (personal notes with
local paths and a Yandex Disk folder link) is gitignored deliberately — the
sanitised specification lives in `specs/SPEC.md`.

## Working conventions

- `specs/SPEC.md` is the agreed specification; `specs/TODO.md` is the live queue.
  Keep both current — the queue must survive a session restart.
- **Every finished day gets a short report in `specs/reports/wNNdDD.md`**, written
  after `advent submit` succeeds. The spec says what was planned and the queue says
  what is left; neither answers "what came out of it and what did we trip over".
  Template and rationale: `specs/reports/README.md`. Whatever in it is permanent
  project knowledge also goes into "Things that will bite you" above — the report
  is the source, not the substitute.
- README files and chat are in Russian; this file and code comments are English.
- Comments explain *why*, especially where the code looks odd on purpose (the
  traps above). Do not add comments that restate the code.

**Rate-limit headers are reachable through the SDK's hook chain, not through
the response object.** `mistral.chat.complete()` returns only the unmarshalled
body, so `x-ratelimit-limit-req-minute` is genuinely absent from what the call
hands back — and a first pass concluded from that it could not be had at all,
and paced the Day 05 sweep off a hardcoded snapshot instead. The SDK
(Speakeasy-generated) runs a hook chain, and `after_success` receives the raw
`httpx.Response` with its headers. `SDKHooks.after_success()` iterates the
registered hooks and calls `hook.after_success(ctx, response)` with no type
check, so a plain object with that method is enough — no need to subclass the
private `mistralai.client._hooks.types.AfterSuccessHook`. The hook **must**
return the response: the chain assigns the return value back.

Registration is still private (`client.sdk_configuration._hooks`, planted via
`__dict__`), so it is wrapped in a broad `except` — telemetry must never take
down the call — and when the seam disappears `CallResult.rate_limit_rpm` stays
`None`, which reads as *unknown*, never as a default. `0` is a legitimate
value there, not missing data: zero is exactly what Mistral returns for a
model the tier does not allow, and no amount of pausing fixes that.

**The rate limit is per model, not per key.** On 2026-09-04 the same key
reported 750 req/min for `ministral-3b-latest`, 188 for `8b`, 30 for `14b`,
and 0 (plus a 429) for the whole `mistral-small` family — measured within the
same few seconds. So a limit cached per account would be wrong for every model
but one, and a model going 429 while its neighbour returns 200 is not a
transient blip to retry through.

**A model that answered yesterday can return 429 today, and it is not a 403.**
`mistral-small-latest` — the project's `DEFAULT_MODEL` from Day 01 through
Day 04 — began returning 429 with `x-ratelimit-limit-req-minute: 0`, stable
across four probes 30 seconds apart, while `ministral-14b` answered 200 in the
same seconds. `mistral-large-latest` refuses differently, with 403
`tier_not_allowed`, and `mistral-medium*` additionally reports **empty
`capabilities`** in `/v1/models`, so `chat_models()` filters it out and
`advent w01 models` never shows it. Three different shapes of "you cannot use
this model", none of which is an outage — check the headers before diagnosing
one.
