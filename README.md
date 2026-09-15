# AI Advent 9

Задания курса [AI Advent](https://mobiledeveloper.tech/ai_advent_9), 9-й поток —
7 недель, по заданию каждый будний день.
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
uv run advent w01 solve                    # одна задача четырьмя способами рассуждения
uv run advent w01 temp                     # один запрос при temperature 0 / 0.7 / 1.2
uv run advent w01 bench                    # один запрос на лестнице ministral 3b/8b/14b
```

## Программа

| Неделя | Тема |
|--------|------|
| 01 | Основы LLM |
| 02 | База про агентов: контекст, его сжатие, токены |
| 03 | Оптимизация агента: state management |
| 04 | MCP — Model-Context-Protocol |
| 05 | RAG |
| 06 | Локальный ИИ: своя LLM как приватный сервис |
| 07 | Пайплайн: интеграция LLM в свои задачи |

Пять заданий в неделю — новое каждый будний день в 14:00 МСК; сдать всю пятёрку
надо до 14:00 МСК следующего понедельника. Формат сдачи — видео + код.

## Прогресс

| Неделя | День | Задание | Код | Видео |
|--------|------|---------|-----|-------|
| 01 | 01 | Первый запрос к LLM через API | [`w01d01`](https://github.com/metal0k/ai_challenge_9/tree/w01d01) | — |
| 01 | 02 | Формат ответа | [`w01d02`](https://github.com/metal0k/ai_challenge_9/tree/w01d02) | — |
| 01 | 03 | Разные способы рассуждения | [`w01d03`](https://github.com/metal0k/ai_challenge_9/tree/w01d03) | — |
| 01 | 04 | Температура | [`w01d04`](https://github.com/metal0k/ai_challenge_9/tree/w01d04) | — |
| 01 | 05 | Версии моделей | [`w01d05`](https://github.com/metal0k/ai_challenge_9/tree/w01d05) | — |
| 02 | 06 | Первый агент | [`w02d06`](https://github.com/metal0k/ai_challenge_9/tree/w02d06) | — |
| 02 | 07 | Сохранение контекста | [`w02d07`](https://github.com/metal0k/ai_challenge_9/tree/w02d07) | — |
| 02 | 08 | Работа с токенами | [`w02d08`](https://github.com/metal0k/ai_challenge_9/tree/w02d08) | — |
| 02 | 09 | Управление контекстом: сжатие истории | [`w02d09`](https://github.com/metal0k/ai_challenge_9/tree/w02d09) | — |
| 02 | 10 | Управление контекстом: разные стратегии | [`w02d10`](https://github.com/metal0k/ai_challenge_9/tree/w02d10) | — |
| 03 | 11 | Модель памяти агента | [`w03d11`](https://github.com/metal0k/ai_challenge_9/tree/w03d11) | [Yandex Disk](https://yadi.sk/i/Udf4MNU_ocQaAw) |

## Структура

```
advent_core/    общий слой: конфиг, клиент Mistral, стрим, логирование, вывод
advent_cli/     команды advent: корневой app, record, submit
week_01/        код первой недели + условия задач в tasks.md (разделы ## Day NN)
tests/          тесты без обращения к сети
tools/          проверка индекса перед публикацией
```

Неделя — явная подкоманда (`advent w01 …`), поэтому команды прошлых недель
продолжают работать после того, как приложение уехало вперёд. Состояние
каждого сданного дня зафиксировано тегом `wNNdDD`.

## Команды

| Команда | Что делает |
|---------|-----------|
| `advent w01 chat [вопрос]` | Вопрос модели; без аргумента — REPL с историей |
| `advent w01 models` | Список моделей из живого API |
| `advent w01 solve` | Решить задачу четырьмя способами рассуждения и сравнить |
| `advent w01 temp` | Один запрос при разных `temperature`: точность, формат, разнообразие |
| `advent w01 bench` | Один запрос на лестнице моделей: точность, latency, токены, цена |
| `advent record --day 1` | Записать демо через OBS и положить как `WWDD.mp4` |
| `advent submit --day 1` | Проверить тег и видео, напечатать комментарий для таблицы |

Полезные флаги `chat`: `--model`, `--system`, `--temperature`, `--top-p`,
`--max-tokens`, `--seed`, `--stop`, `--reasoning-effort`, `--no-stream`,
`--verbose`, `--format`, `--schema-file`, `--done`, `--mode`, `--max-turns`,
`--strategy`.

Флаги `solve`: `--problem`, `--strategy`, `--runs`, `--judge/--no-judge`,
`--judge-model`, `--model`, `--system`, `--verbose`. Способы рассуждения —
`direct` (прямой ответ), `steps` (пошагово), `meta` (модель сама пишет промпт
для решения), `panel` (аналитик, инженер, критик и синтез); `all` прогоняет все
четыре и печатает таблицу сравнения с вердиктом по эталону и оценкой
LLM-судьи. Подробности — в [`week_01/README.md`](week_01/README.md#day-03--разные-способы-рассуждения).

Команды внутри REPL:

| Команда | Что делает |
|---------|-----------|
| `/help` | список команд |
| `/model` | текущая модель |
| `/model <имя>` | переключить модель |
| `/model list` | таблица chat-моделей (`list all` — вообще все) |
| `/model info` | карточка модели: контекст, возможности, рекомендованная t° |
| `/params` | текущие параметры генерации и итоговый system prompt |
| `/set <параметр> <значение>` | изменить параметр, включая `format`/`schema_file`/`done`/`mode`/`max_turns`/`strategy` (`default` — вернуть умолчание команды) |
| `/again` | повторить последний вопрос с текущими настройками, без истории |
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
