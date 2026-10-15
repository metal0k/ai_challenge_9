# AdventAssistant — команды и возможности

Ассистент разработчика для работы с проектом, документацией и автоматизации задач.

## Список команд

### Вопросы о проекте
- **`adventassistant help "вопрос"`**
  Отвечает на вопросы о проекте, используя документацию и текущую ветку Git.
  Пример: `adventassistant help "На какой ветке сейчас репозиторий?"`.

- **`adventassistant index`**
  Собирает индекс документации для поиска (требуется перед использованием `help`).
  Пример: `adventassistant index --rev HEAD`.

### Ревью кода
- **`adventassistant index-review`**
  Собирает индекс для анализа изменений в коде (требуется перед `review`).
  Пример: `adventassistant index-review --rev main`.

- **`adventassistant review --base <ref> --head <ref>`**
  Анализирует изменения в диапазоне `base...head`, выявляя баги и архитектурные проблемы.
  Пример: `adventassistant review --base main --head feature-branch`.

- **`adventassistant watch-pr`**
  Автоматически следит за открытыми PR и публикует ревью в комментарии.
  Примеры:
  - `adventassistant watch-pr --once` — один проход.
  - `adventassistant watch-pr --pr 123` — проверка только PR №123.
  - `adventassistant watch-pr --interval 30` — изменение интервала проверки.

### Поддержка пользователей
- **`adventassistant index-support`**
  Собирает FAQ-индекс для поддержки.

- **`adventassistant support "вопрос" --user <id> --ticket <id>`**
  Отвечает на вопросы пользователей с учётом контекста тикета.
  Пример: `adventassistant support "Почему не работает авторизация?" --user demo-a --ticket AUTH-101`.

### Работа с файлами
- **`adventassistant files "цель"`**
  Ассистент выполняет задачи с файлами (поиск, анализ, изменение) через MCP.
  Пример: `adventassistant files "Обновить README.md на основе изменений в коде"`.

- **`adventassistant files-apply <run_id>`**
  Применяет сохранённые изменения из предыдущего прогона.
  Пример: `adventassistant files-apply abc123`.

## Требования
- Для работы с GitHub PR (`watch-pr`) необходимы переменные окружения:
  `GITHUB_URL`, `GITHUB_TOKEN`, `MISTRAL_API_KEY`.

- Все команды, кроме `help`, требуют предварительной индексации (`index`, `index-review`, `index-support`).