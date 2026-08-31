# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coursework for [AI Advent 9](https://mobiledeveloper.tech/ai_advent_9) — 9 weeks,
one task per day, built on the Mistral API. Each finished day is submitted as a
pair of links (code + video) commented into a shared Google Sheet.

## Commands

```bash
uv sync                                   # install/refresh the environment
uv run advent w01 chat "вопрос"           # one-shot answer
uv run advent w01 chat                    # REPL with history
uv run advent w01 models                  # models from the live API (--all for non-chat)
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
  (`task_NN.md`) beside it. The week is **one application that grows day by day**,
  not one file per day.
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

**stdout/stderr contract:** the model's answer goes to stdout, everything else
(footer, warnings, REPL prompt, input echo) to stderr. This keeps
`advent w01 chat "…" > answer.txt` honest. Do not print status to stdout.

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
- README files and chat are in Russian; this file and code comments are English.
- Comments explain *why*, especially where the code looks odd on purpose (the
  traps above). Do not add comments that restate the code.
