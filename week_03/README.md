# Неделя 03, день 11 — явные memory layers

День 11 продолжает `adventagent` из недели 02. `default` по-прежнему —
`context_strategy=summary`; memory работает в режиме opt-in:

```powershell
uv run adventagent --session demo11
```

Затем выберите strategy в REPL:

```text
/strategy memory
```

Agent хранит три отдельных layers:

- `short-term` — transcript текущей session (`logs/sessions/<name>.json`);
- `working` — task-local goal, constraints, decisions и open items
  (`logs/memory/working/<name>.json`);
- `long-term` — explicit profile, preferences и knowledge, общие для всех sessions
  (`logs/memory/long_term.json`).

Для automatic routing нужны точные данные из user message. Manual edits
доступны через `/memory set|del|pin|unpin|move`; manual `set` закрепляет value.
`/memory retry` повторяет dirty file writes, а `/facts` и `/fact` остаются
compatibility aliases для session-scoped working layer.

Полезные команды:

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

`summary` по-прежнему остаётся short-term optimization, а не четвёртым layer.
`/new` очищает short-term и working memory, но сохраняет long-term memory;
`/switch`, checkpoints и branches загружают или копируют working memory, не
копируя long-term memory.

Offline demo использует реальный код routing, storage и request assembly и
никогда не вызывает Mistral API:

```powershell
uv run python tools/memory_demo.py --dry-run
```

Deterministic recording показывает полный lifecycle: session A направляет goal
и constraint в `working`, language preference — в `long-term`, а code word — в
`short-term`; session B сохраняет только global preference; возврат в A
восстанавливает его session-local layers. Тот же run также показывает, как
закреплённое manual value блокирует automatic conflict, credential-like value
отклоняется, а `/new` очищает только session-local memory. Последние строки
называют все три физических storage locations.

У demo есть два взаимодополняющих takes. `logs/videos/0311.mp4` — это
deterministic offline demo: оно использует реальные memory operations с
фиксированными fake model replies, поэтому каждый layer и lifecycle transition
видны без API calls. `logs/videos/0311-live.mp4` — optional live Mistral take,
записанный командой:

```powershell
uv run advent record --week 3 --day 11 --live
```

Live take использует реальную Mistral model и показывает manual `/memory`
updates, переключение sessions A/B, recall после возврата в A и `/new`. Во
время этого take extractor предложил invalid deltas; strict validation guard
безопасно их отклонил. Это показано как safe failure, а не как успешный
automatic routing.

# Неделя 03, день 12 — named profiles

Named global profiles хранятся в `logs/profiles/<name>.json` как гибкие
`key=value` preferences. Active profile с additive semantics сохраняется в
`Session.state.active_profile`, а profile context внедряется в каждый request и
учитывается отдельно. Current request settings имеют приоритет над active
profile; отсутствие active profile сохраняет прежнее request behavior.

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

Credential-like keys и values отклоняются, а unsafe legacy fields пропускаются
при load; profile context не записывается в conversation journal.
Deterministic offline utility остаётся доступной для development checks:

```powershell
uv run python -m tools.profile_demo
```

Implementation: `advent_core/profiles.py`, `advent_core/agent.py`,
`week_02/cli.py`, `advent_cli/record.py`; focused tests находятся в
`tests/test_profiles.py` и `tests/test_agent.py`. Submission recording — это
реальный `adventagent` REPL с project-default Mistral model, streaming, двумя
profile-conditioned answers и `--max-tokens 220`:

```powershell
uv run advent record --week 3 --day 12
```

# Неделя 03, день 13 — Task State Machine

Одна session хранит одну formal task с lifecycle
`planning → execution → validation → done`. Из `validation` разрешён retry в
`execution`. Только explicit `/task` commands меняют state; model answer может
предложить следующий transition, но не выполняет его автоматически.

```text
/task start Выпустить API :: Составить release plan :: Подтвердить риски
/task show
/task update Уточнить rollback plan :: Получить approval
/task advance execution Выполнить canary deploy :: Проверить metrics
/task pause Ожидаем metrics
/task resume
/task advance validation Проверить metrics :: Решить, нужен ли rollback
/task complete Production stable
/task clear
```

Active unpaused task добавляется в каждый model request как protected counted
context: goal, phase, current step и expected action. Current user request имеет
priority. Paused task является hard gate — ordinary prompts, `/again`, model и
context mutations блокируются до `/task resume` или `/task clear`; read-only
status/navigation commands остаются доступны.

Task переживает restart, `/reset`, checkpoint и branch. `/new` очищает task
вместе с dialog content. Branch получает snapshot и дальше изменяется
независимо. Done-state остаётся видимым через `/task show`, но больше не
injected в requests.

Live demo использует project-default Mistral model, streaming и два процесса,
чтобы показать restart continuity:

```powershell
uv run advent record --week 3 --day 13
```

# Неделя 03, день 14 — Invariants and State Constraints

`/invariant` добавляет session-scoped policy, которая имеет priority выше
Task State и текущего user request. Правила не являются частью dialogue и
сохраняются через restart, `/reset`, checkpoint и branch; `/new` их не удаляет.

```text
/invariant add no-public-network :: Do not expose the service to the public network.
/invariant add approval-before-deploy :: Obtain explicit approval before deploy.
/invariant list
/invariant remove approval-before-deploy
/invariant clear
```

При активных rules `adventagent` сначала выполняет отдельный structured JSON
assessment. `compliant` запускает обычный answer call; в stderr после token
panel видно `Invariant check: compliant`. Это не добавляется в stdout, поэтому
`format=json` и `format=schema` сохраняют valid model-product output.

`conflict`, `policy_conflict`, malformed JSON или assessment/API failure
fail closed: answer call не запускается, Session dialogue не меняется, а REPL
показывает named rule(s), explanation и safe alternative либо explicit
`/invariant remove <id>`/`/invariant clear`. Raw assessment prompt и JSON не
попадают в conversation journal; `/tokens` показывает rules block и отдельную
стоимость assessment calls.

`/invariant add|list|remove|clear` разрешены при paused Task State, хотя все
model calls до `/task resume` по-прежнему blocked.
