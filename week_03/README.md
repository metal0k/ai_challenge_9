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
