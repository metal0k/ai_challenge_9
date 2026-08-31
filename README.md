# AI Advent 9

Задания курса [AI Advent](https://mobiledeveloper.tech/ai_advent_9) — 9 недель, задания по дням.
Python + [Mistral API](https://docs.mistral.ai/).

## Установка

```bash
git clone https://github.com/metal0k/ai_challenge_9.git
cd ai_challenge_9
uv sync
cp .env.example .env      # и вписать MISTRAL_API_KEY
```

Ключ берётся в [console.mistral.ai](https://console.mistral.ai/api-keys).
Нужен Python 3.12+ и [uv](https://docs.astral.sh/uv/).

## Быстрый старт

```bash
uv run advent w01 chat "Что такое LLM?"   # один вопрос — один ответ
uv run advent w01 chat                     # интерактивный диалог
uv run advent w01 models                   # какие модели доступны аккаунту
```

## Прогресс

| Неделя | День | Задание | Код | Видео |
|--------|------|---------|-----|-------|
| 01 | 01 | Первый запрос к LLM через API | [`w01d01`](https://github.com/metal0k/ai_challenge_9/tree/w01d01) | — |

## Структура

```
advent_core/    общий слой: конфиг, клиент Mistral, стрим, логирование, вывод
advent_cli/     команды advent: корневой app, record, submit
week_01/        код первой недели + условия задач task_NN.md
tests/          тесты без обращения к сети
specs/          спецификация проекта
```

Неделя — явная подкоманда (`advent w01 …`), поэтому команды прошлых недель
продолжают работать после того, как приложение уехало вперёд. Состояние
каждого сданного дня зафиксировано тегом `wNNdDD`.

## Команды

| Команда | Что делает |
|---------|-----------|
| `advent w01 chat [вопрос]` | Вопрос модели; без аргумента — REPL с историей |
| `advent w01 models` | Список моделей из живого API |
| `advent record --day 1` | Записать демо через OBS и положить как `WWDD.mp4` |
| `advent submit --day 1` | Проверить тег и видео, напечатать комментарий для таблицы |

Полезные флаги `chat`: `--model`, `--system`, `--temperature`, `--top-p`,
`--max-tokens`, `--seed`, `--stop`, `--reasoning-effort`, `--no-stream`, `--verbose`.

Команды внутри REPL:

| Команда | Что делает |
|---------|-----------|
| `/help` | список команд |
| `/model` | текущая модель |
| `/model <имя>` | переключить модель |
| `/model list` | таблица chat-моделей (`list all` — вообще все) |
| `/model info` | карточка модели: контекст, возможности, рекомендованная t° |
| `/params` | текущие параметры генерации |
| `/set <параметр> <значение>` | изменить параметр (`default` — сбросить) |
| `/reset` | очистить историю |
| `/exit` | выход |

Незаданный параметр **не отправляется вовсе** — Mistral применит рекомендованное
для модели значение. Оно видно в колонке `t°`. Параметры, которых модель не
поддерживает, отсеиваются с предупреждением: `reasoning_effort` уйдёт в
`mistral-small`, но не в `codestral`, у которого `reasoning=false`.

Ответ идёт в stdout, телеметрия — в stderr, поэтому редирект работает как надо:

```bash
uv run advent w01 chat "напиши хокку" > ответ.txt
```

## Разработка

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

Тесты не ходят в сеть и не тратят API — клиент замокан.

## Что не в репозитории

`.env` с ключами и `logs/` с историей вызовов в `.gitignore`. Видео живёт на
Яндекс.Диске, не в git. Шаблон переменных — в `.env.example`.
