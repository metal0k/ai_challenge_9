# Неделя 04 — MCP

День 16: минимальный MCP-клиент. Подключается к серверу по stdio, делает
handshake и печатает список инструментов.

## Запуск

```bash
uv run adventmcp tools                    # собственный сервер репозитория (advent-repo)
uv run adventmcp tools --raw              # + JSON-RPC кадры в stderr (→ клиент, ← сервер)
uv run adventmcp tools --server "python -m my_server" --expect a,b
```

Флаги `tools`:

- `--server "<команда>"` — команда запуска чужого MCP-сервера по stdio
  (пути Windows с пробелами — в кавычках). По умолчанию `python -m week_04.server`.
- Ограничение разбора: `--flag="a b"` (кавычки внутри токена) на Windows не поддерживается — берите значение целиком в кавычки: `"--flag=a b"`.
- `--timeout 15` — секунд на handshake и на каждую страницу `tools/list`.
- `--raw/--no-raw` — зеркалить каждый кадр JSON-RPC в stderr; stdout не меняется.
- `--expect имя,имя` — сверить список с ожидаемым. Для собственного сервера
  по умолчанию ожидаются все его инструменты (`list_days`, `get_task`,
  `count_tokens`, `git_log`, `schedule_job`, `repo_activity_summary`), для чужого проверки нет. Расхождение — exit 1.

stdout — продукт (таблица инструментов и вердикт сверки); шапка сервера,
raw-кадры и ошибки — в stderr.

## Сервер

`week_04/server.py` — шесть инструментов с разными схемами:
`list_days` (без аргументов), `get_task(week, day)` (обязательные int),
`count_tokens(text, model)` (обязательный + опциональный enum),
`git_log` (день 17, read-only), `schedule_job(interval_seconds)` и
`repo_activity_summary(minutes)` (день 18). В stdout сервер пишет только протокол.

## Ошибки: exit code 8

Любой сбой соединения — `MCPError`, одна строка причины и совет, без traceback:
исполняемый файл не найден; процесс сервера завершился до конца handshake;
сервер пишет в stdout не JSON-RPC; сервер не ответил за `--timeout`; протокольная ошибка. Дочерний процесс
при сбое гасится.

## Демо

```bash
uv run advent record --week 4 --day 16 --dry-run   # прогон без записи
uv run advent record --week 4 --day 16             # запись через OBS
```

Сценарий: успех и сверка → сервер не стартует (exit 8) → сырой лог кадров.

## День 17: git_log

Инструмент `git_log` отдаёт последние коммиты репозитория; агент подключает
сервер через `/mcp on` и вызывает его как обычную function-calling функцию.

## День 18: планировщик и фоновые задачи

Отдельный демон-процесс выполняет job по расписанию, а MCP-тулы только
пишут и читают общий файл состояния `logs/scheduler/repo_activity.json`
(в `.gitignore`). Сам демон MCP-тулы не запускают и не останавливают.

```bash
uv run adventmcp scheduler run                 # демон: цикл, Ctrl+C — остановка
uv run adventmcp scheduler run --interval 15   # + создать/перенастроить job на старте
uv run adventmcp scheduler run --once          # одна проверка и выход, без sleep
```

Демон раз в секунду перечитывает файл, поэтому интервал, изменённый через
`schedule_job`, подхватывается без перезапуска. Весь вывод демона — в stderr,
stdout пуст. Ctrl+C — «прервано» и exit 130.

Инструменты сервера:

- `schedule_job(interval_seconds)` — создать job или сменить интервал
  (5..3600 с; вне границ — ошибка без traceback).
- `repo_activity_summary(minutes=None)` — сводка: число тиков, новые коммиты
  (`N+`, если между тиками их было больше окна сканирования), ошибки тиков,
  последние новые коммиты. Без job'а — ошибка «сначала вызови schedule_job».

Подготовка к демо:

1. Заранее, не в кадре: `uv run adventmcp scheduler run --interval 15`
   в отдельном терминале — к началу записи накопится несколько тиков.
2. В кадре: агент (`/mcp on`) дважды вызывает `repo_activity_summary` с
   паузой между вызовами — число тиков растёт.
3. После записи остановить демон через Ctrl+C, иначе он проработает до
   перезагрузки машины.
