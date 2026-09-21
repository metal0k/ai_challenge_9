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
  по умолчанию ожидаются его три инструмента (`list_days`, `get_task`,
  `count_tokens`), для чужого проверки нет. Расхождение — exit 1.

stdout — продукт (таблица инструментов и вердикт сверки); шапка сервера,
raw-кадры и ошибки — в stderr.

## Сервер

`week_04/server.py` — read-only, три инструмента с разными схемами:
`list_days` (без аргументов), `get_task(week, day)` (обязательные int),
`count_tokens(text, model)` (обязательный + опциональный enum).
В stdout сервер пишет только протокол.

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
