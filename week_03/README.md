# Week 03, Day 11 — explicit memory layers

Day 11 continues the `adventagent` from Week 02. The default remains
`context_strategy=summary`; memory is opt-in:

```powershell
uv run adventagent --session demo11
```

Then select the strategy in the REPL:

```text
/strategy memory
```

The agent keeps three distinct layers:

- `short-term` — the current session transcript (`logs/sessions/<name>.json`);
- `working` — task-local goal, constraints, decisions and open items
  (`logs/memory/working/<name>.json`);
- `long-term` — explicit profile, preferences and knowledge shared by all
  sessions (`logs/memory/long_term.json`).

Automatic routing requires exact evidence from a user message. Manual edits
are available through `/memory set|del|pin|unpin|move`; manual `set` pins the
value. `/memory retry` retries dirty file writes, while `/facts` and `/fact`
remain compatibility aliases for the session-scoped working layer.

Useful commands:

```text
/memory
/memory working
/memory set working goal.primary release the migration
/memory set long preferences.answer_language Russian prose with English technical terms
/memory pin working goal.primary
/memory move working long source.key target.key
/memory backfill
/memory retry
/tokens
/new other-session
```

`summary` is still a short-term optimization, not a fourth layer. `/new`
clears short-term and working memory but leaves long-term memory intact;
`/switch`, checkpoints and branches load/copy working memory without copying
long-term memory.

The offline demo uses the real routing, storage and request assembly code and
never calls the Mistral API:

```powershell
uv run python tools/memory_demo.py --dry-run
```

The deterministic recording walks through one complete lifecycle: session A
routes a goal and constraint to `working`, a language preference to
`long-term`, and a code word to `short-term`; session B keeps only the global
preference; returning to A restores its session-local layers. The same run
also shows a pinned manual value blocking an automatic conflict, a
credential-like value being rejected, and `/new` clearing only session-local
memory. The final lines name all three physical storage locations.

The demo has two complementary takes. `logs/videos/0311.mp4`
is the deterministic offline demo: it uses the real memory operations with
fixed fake model replies, so every layer and lifecycle transition is visible
without API calls. `logs/videos/0311-live.mp4` is an optional live Mistral
take, recorded with:

```powershell
uv run advent record --week 3 --day 11 --live
```

The live take uses the real Mistral model and demonstrates manual `/memory`
updates, session A/B switching, recall after returning to A, and `/new`.
During that take the extractor proposed invalid deltas; the strict validation
guard rejected them safely. This is shown as a safe failure, not presented as
successful automatic routing.

# Week 03, Day 12 — named profiles

Named global profiles live in `logs/profiles/<name>.json` as flexible
`key=value` preferences. The active profile is persisted additively in
`Session.state.active_profile`, while profile context is injected into each
request and counted separately. Current request settings have priority over
the active profile; no active profile preserves the previous request behavior.

```text
/profile create developer style=technical format=code-first
/profile create manager style=brief format=bullets
/profile use developer
/profile show
/profile set style=concise
/profile del style
/profile delete manager
/tokens
```

Credential-like keys and values are rejected (and unsafe legacy fields are
omitted on load); profile context is not written to the conversation journal.
The deterministic offline utility remains available for development checks:

```powershell
uv run python -m tools.profile_demo
```

Implementation: `advent_core/profiles.py`, `advent_core/agent.py`,
`week_02/cli.py`, `advent_cli/record.py`, with focused tests in
`tests/test_profiles.py` and `tests/test_agent.py`. The submission recording is
the real `adventagent` REPL with the project-default Mistral model, streaming,
two profile-conditioned answers and `--max-tokens 220`:

```powershell
uv run advent record --week 3 --day 12
```
