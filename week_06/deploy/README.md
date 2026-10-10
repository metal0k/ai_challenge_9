# Деплой приватного LLM-сервиса (день 30)

Gateway (`week_06/gateway/`) и llama-server живут в одном systemd unit
`advent-llm.service` на VPS. Деплой идёт через SSH-слой `VPS_Manager`.

## Подготовка

- `VPS_MANAGER_DIR` — каталог `VPS_Manager` (в нём `tools/vps.py`); в коде путей нет.
- `ADVENT_SERVICE_KEY` и `ADVENT_SERVICE_DEMO_KEY` — в `.env` или окружении. На сервер
  уходят только их sha256 (`/etc/advent-llm/keys.json`).
- Ключи делает `uv run adventlocal service keygen`.

## Команды

```
python week_06/deploy/deploy.py plan                 # план установки, без SSH
python week_06/deploy/deploy.py install --yes        # preflight, user, uv, релиз, unit, smoke
python week_06/deploy/deploy.py release --yes        # новый релиз + атомарный symlink + restart
python week_06/deploy/deploy.py rollback --yes       # current -> предыдущий релиз, restart, smoke
python week_06/deploy/deploy.py uninstall --yes      # только созданное deploy'ем
python week_06/deploy/deploy.py status               # только чтение
```

Без `--yes` любая изменяющая команда печатает план и ничего не делает. Полезные флаги:
`--version` (имя релиза, по умолчанию git sha; при незакоммиченном gateway к нему
добавляется хэш содержимого), `--model-sha256` (эталон для модели), `--radio-url`
(замер p95 статики радио во время длинной генерации, ворота: все 200 и p95 <= 0.5 с; по умолчанию включён, `--no-radio` выключает).

Релиз собирается в `releases/.staging-*`, проверяется под пользователем сервиса и только потом переименовывается в `releases/<версия>`; существующая версия принимается лишь с тем же манифестом (`MANIFEST`), каталог под `current` не меняется. После переключения `current` — явный `systemctl restart` и опрос `/health` (до 30 с). Если задан `VPS_MANAGER_DIR`, в конце запускается `tools/smoke.py` соседних сервисов.

`install` заодно удаляет остатки probe: `dl/` и две невыбранные модели.
`uninstall` не трогает `bin/`, `lib/`, выбранную модель и `probe/`.

## Caddy

Скрипт Caddy **не** меняет. Блок `Caddyfile.advent-llm` добавь в зеркало конфигов
`VPS_Manager` и залей его `push_config.py` (backup, validate, reload, rollback).
Без `flush_interval`: для `text/event-stream` Caddy и так сбрасывает ответ сразу.

## Цена сервиса

`CPUWeight=20` и `Nice=10` делают сервис вежливым, но не бесплатным: статика радио во время
генерации отвечает примерно вдвое медленнее (0.07 -> 0.15 с), все ответы 200.
`CPUQuota` не ставим: замер показал, что радио он не помогает, а генерацию замедляет.

## Склейка видео

Терминальный дубль (`advent record --day 30`) и web-клип снимаются отдельно. Клеить нужно
через `trim`/`concat`, а не `select`: у OBS два потока (видео и звук), и `concat` берёт оба
из одного списка интервалов, поэтому они не расходятся.

```
ffmpeg -i terminal.mp4 -i web.mp4 -filter_complex \
  "[0:v]fps=30,scale=1920:1080,setsar=1[v0];[1:v]fps=30,scale=1920:1080,setsar=1[v1];\
   [0:a]aresample=48000[a0];[1:a]aresample=48000[a1];\
   [v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -c:v libx264 -crf 20 -c:a aac out.mp4
```

После web-клипа повторно проверь `adventlocal service status`: модель должна быть `stopped`.
Готовое видео проверь по кадрам: `ffmpeg -ss <t> -i out.mp4 -frames:v 1 frame.png`.
