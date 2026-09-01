"""Блокирует коммит, если в индекс попали личные данные.

Запускается перед каждым push в публичный репо:

    uv run python tools/check_staged.py

Ненулевой код возврата означает «не коммитить». Проверка читает содержимое
из индекса (`git show :file`), а не из рабочего дерева: коммитится именно
то, что застейджено, и эти две вещи расходятся чаще, чем кажется.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Вывод здесь кириллический, а stdout при запуске из хука или через пайп берёт
# кодировку из локали — на этой машине cp1252. Без принудительного UTF-8
# проверка падает с UnicodeEncodeError на собственном сообщении об успехе:
# guard, который умирает на выводе, читается как сломанный и будет обойдён.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Фиксированные строки, а не регулярки: паттерны с обратными слэшами
# по-разному раскрываются в shell и в ERE и молча перестают срабатывать.
NEEDLES = (
    ("D:\\", "локальный путь Windows"),
    ("C:\\Users", "домашняя папка пользователя"),
    ("docs.google.com/spreadsheets", "ссылка на таблицу курса — там чужие имена"),
    ("yadi.sk/i/", "персональная ссылка на файл Яндекс.Диска"),
    ("github_pat_", "GitHub token"),
    ("ghp_", "GitHub token"),
    ("sk-", "похоже на API-ключ"),
    ("in reply to", "похоже на дамп переписки — там чужие имена"),
)

# Регулярки — только там, где ловится ФОРМА, а не конкретная строка.
# Имена людей фиксированной строкой не поймать, а вписать их сюда нельзя:
# этот файл сам уходит в публичный репозиторий, и список имён утёк бы вместе
# с ним. Поэтому ловим характерную разметку экспорта чата — «[дд.мм.гггг чч:мм]»
# в начале строки. Так guard срабатывает на вставленную переписку, ничего не
# зная о её участниках.
PATTERNS = ((re.compile(r"\[\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}\]"), "дамп чата с датами реплик"),)

# Файлы, которые документируют сами паттерны и потому находят себя.
SELF_REFERENTIAL = {"CLAUDE.md", "tools/check_staged.py"}


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [f for f in out.split("\0") if f]


def staged_content(path: str) -> str | None:
    result = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None  # бинарный файл


def main() -> int:
    findings: list[str] = []

    for path in staged_files():
        if path in SELF_REFERENTIAL:
            continue
        content = staged_content(path)
        if content is None:
            continue
        for line_no, line in enumerate(content.splitlines(), 1):
            for needle, why in NEEDLES:
                if needle in line:
                    findings.append(f"  {path}:{line_no}  {why}\n    {line.strip()[:100]}")
            for pattern, why in PATTERNS:
                if pattern.search(line):
                    findings.append(f"  {path}:{line_no}  {why}\n    {line.strip()[:100]}")

    if findings:
        print("В индексе личные данные — коммит остановлен:\n", file=sys.stderr)
        print("\n".join(findings), file=sys.stderr)
        print(
            "\nУбери их или добавь файл в .gitignore. Помни: force-push уже не спасёт —"
            "\nстарые объекты остаются доступны по SHA.",
            file=sys.stderr,
        )
        return 1

    print(f"проверено файлов: {len(staged_files())} — личных данных нет")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
