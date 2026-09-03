"""Day 03: четыре способа рассуждения над одной задачей, сверка и судья.

Модуль знает всё про день: банк задач, промпты стратегий, извлечение ответа,
сверку с эталоном, судью и таблицу сравнения. CLI (week_01/cli.py) только
разбирает флаги и зовёт `solve()` / `judge()` — вся механика здесь, чтобы
`advent w01 solve` и `advent w01 chat --strategy ...` не разъехались в
поведении (SPEC-w01d03.md §3).

Исполнение строго последовательное: ни потоков, ни asyncio. Девять вызовов
подряд — это медленно и так задумано (SPEC-w01d03.md §2): параллельный запуск
смешал бы порядок вывода на видео и упёрся бы в rate limit ровно в момент
записи.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rich.markup import escape as rich_escape
from rich.table import Table

from advent_core import chat as chat_core
from advent_core import console
from advent_core.chat import Message
from advent_core.config import DEFAULT_SYSTEM_PROMPT, Config, ConfigError
from advent_core.errors import AdventError
from advent_core.journal import log_call
from advent_core.params import STRATEGIES
from advent_core.telemetry import CallResult, Totals

WEEK_DIR = Path(__file__).resolve().parent
PROBLEMS_DIR = WEEK_DIR / "problems"
PROMPTS_DIR = WEEK_DIR / "prompts"
JUDGE_SCHEMA_PATH = WEEK_DIR / "schemas" / "judge.json"

# КРИТИЧНО: этот маркер НЕЛЬЗЯ передавать в параметр `stop`. Установлено на
# Day 02 и записано в CLAUDE.md: "The output will not contain the stop
# sequence" — API вырезает стоп-последовательность из ответа, и детектировать
# станет нечего. Маркер ищется в уже полученном тексте, см. extract_answer().
ANSWER_MARKER = "ОТВЕТ:"

# STRATEGIES импортирован из advent_core.params: список стратегий нужен и
# реестру параметров (для валидации `--strategy`/`/set strategy`), и здесь —
# а два кортежа с одними и теми же строками разъезжаются при первой правке.
STRATEGY_CHOICES = (*STRATEGIES, "all")

# Метки судьи. Названий стратегий судья не видит вообще: "панель экспертов"
# получила бы фору просто из-за названия (SPEC-w01d03.md §8).
JUDGE_LABELS = ("A", "B", "C", "D")

# Три состояния, а не два. "Маркера нет" — не то же самое, что "ответ
# неверный": в первом случае модель, возможно, решила задачу правильно, но не
# выполнила формат, и на видео это не должно выглядеть одинаково
# (SPEC-w01d03.md §6).
VERDICT_OK = "ok"
VERDICT_WRONG = "wrong"
VERDICT_NO_MARKER = "no_marker"

VERDICT_LABELS = {
    VERDICT_OK: "✓",
    VERDICT_WRONG: "✗",
    VERDICT_NO_MARKER: "нет маркера",
}

# Инъекция вызова: тесты подменяют её заглушкой, сеть при этом не трогается
# вовсе (SPEC-w01d03.md §11). Сигнатура повторяет chat.complete().
CompleteFn = Callable[[Config, list[Message], dict[str, Any] | None], CallResult]
Notify = Callable[[str], None]

# Колбэк «вызов состоялся». Зовётся сразу после успешного complete(), до того
# как стратегия сделает следующий вызов, — именно на нём держится обещание
# журнала: седьмой упавший вызов не уносит с собой шесть уже оплаченных.
StepHook = Callable[["Step"], None]


# --------------------------------------------------------------------------
# Банк задач
# --------------------------------------------------------------------------

# Значения Problem.format_check. Константы, а не голые строки в load/сравнении
# — "marker" и "single_line" иначе разбросаны по _read_problem(),
# week_01/temperature.py и тестам, и опечатка в одном месте молча не совпадёт
# с другим (SPEC-w01d04.md §6).
FORMAT_CHECK_MARKER = "marker"
FORMAT_CHECK_SINGLE_LINE = "single_line"
FORMAT_CHECKS = (FORMAT_CHECK_MARKER, FORMAT_CHECK_SINGLE_LINE)


@dataclass(slots=True, frozen=True)
class Problem:
    """Одна задача из банка week_01/problems/<id>.json (SPEC-w01d03.md §5)."""

    id: str
    title: str
    statement: str
    answer: str
    accept: tuple[str, ...] = ()
    note: str = ""
    # Задача-заглушка: структура полная, но эталон не выверен живым прогоном.
    # Ставить `"placeholder": true` в файле имеет смысл ровно до того, как
    # задача проверена на модели — вывод дня держится на этом эталоне.
    placeholder: bool = False
    # Задача дня по умолчанию. Флаг, а не первая позиция по алфавиту: неявная
    # связь с именем файла ломается молча в тот день, когда в банк добавят
    # задачу с более ранним id, и демо поедет не на той задаче
    # (SPEC-w01d03.md §5).
    default: bool = False
    # open=True значит не «ответ пустой», а «правильного ответа не существует
    # в принципе» (SPEC-w01d04.md §6) — у задачи вроде "придумай название
    # кофейни" нет ключа, с которым можно сверяться. check_answer() на такой
    # задаче звать нельзя: она молча сверит пустой answer с чем угодно и
    # выдаст ложный вердикт. Вызывающий код (week_01/temperature.py) обязан
    # проверять этот флаг сам и вместо сверки с эталоном звать судью.
    open: bool = False
    # Как проверяется соблюдение формата: "marker" — присутствие ОТВЕТ: (как у
    # всех задач Day 03), "single_line" — ровно одна непустая строка после
    # strip() (SPEC-w01d04.md §7). По умолчанию "marker" — это сохраняет
    # поведение всего существующего банка без правки его файлов.
    format_check: str = FORMAT_CHECK_MARKER


def load_problems(directory: Path | None = None) -> dict[str, Problem]:
    """Читает весь банк задач. Ключи — id, порядок — алфавитный по id."""
    folder = directory or PROBLEMS_DIR
    if not folder.is_dir():
        raise ConfigError(f"Банк задач не найден: {folder}")

    problems: dict[str, Problem] = {}
    for path in sorted(folder.glob("*.json")):
        problem = _read_problem(path)
        if problem.id in problems:
            raise ConfigError(f"Две задачи с одинаковым id {problem.id!r} — вторая в {path}")
        problems[problem.id] = problem

    if not problems:
        raise ConfigError(f"В банке задач нет ни одного файла: {folder}/*.json")
    return dict(sorted(problems.items()))


def load_problem(problem_id: str | None = None, directory: Path | None = None) -> Problem:
    """Задача по id; без id — помеченная `"default": true` в файле."""
    problems = load_problems(directory)
    if problem_id is None:
        return _default_problem(problems)

    problem = problems.get(problem_id)
    if problem is None:
        available = ", ".join(problems)
        raise ConfigError(f"Задача {problem_id!r} не найдена. Есть: {available}")
    return problem


def _default_problem(problems: dict[str, Problem]) -> Problem:
    """Задача, на которой идёт демо: та, у которой в файле `"default": true`.

    Без флага падать нельзя — банк остаётся рабочим и с одними безымянными
    задачами, — поэтому запасной вариант прежний, первая по алфавиту. А вот
    два флага сразу это уже неоднозначность, которая иначе разрешалась бы
    порядком файлов, то есть молча и не в ту сторону.
    """
    marked = [problem for problem in problems.values() if problem.default]
    if len(marked) > 1:
        names = ", ".join(problem.id for problem in marked)
        raise ConfigError(f'Флаг "default": true стоит сразу у нескольких задач: {names}')
    if marked:
        return marked[0]
    return next(iter(problems.values()))


def problem_ids(directory: Path | None = None) -> list[str]:
    return list(load_problems(directory))


# id задачи-обёртки для вопроса из chat: в журнале по нему видно, что задача
# пришла не из банка, а из командной строки.
CHAT_PROBLEM_ID = "chat"


def question_as_problem(question: str) -> Problem:
    """Заворачивает произвольный вопрос `chat` в Problem — без эталона.

    Вторая поверхность дня (`chat --strategy ...`) прогоняет ту же механику по
    вопросу пользователя, а эталона у него нет и быть не может. Вердикт в
    получившемся StrategyRun.check поэтому бессмысленный (пустой эталон ни с
    чем не совпадёт) и печатать его нельзя: вторая поверхность печатает шаги
    (print_step) и не зовёт print_verdict().
    """
    return Problem(id=CHAT_PROBLEM_ID, title="вопрос из чата", statement=question, answer="")


def _read_problem(path: Path) -> Problem:
    """Разбирает один файл задачи.

    ConfigError, а не голый traceback, на любой поломке файла: банк правится
    руками между прогонами, и опечатка в JSON должна читаться как «почини
    вот этот файл», а не как падение программы.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Не читается файл задачи {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Файл задачи {path} — не валидный JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Задача в {path} должна быть JSON-объектом")

    # answer обязателен для всех задач, КРОМЕ open=true: у open-задачи
    # ("coffee") эталона нет по определению (SPEC-w01d04.md §6), и требовать
    # непустой answer означало бы заставить банк врать про наличие ключа,
    # которого нет.
    required = ("id", "title", "statement")
    if not data.get("open"):
        required += ("answer",)
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ConfigError(f"В задаче {path} не заполнены поля: {', '.join(missing)}")

    accept = data.get("accept") or []
    if not isinstance(accept, list):
        raise ConfigError(f'Поле "accept" в {path} должно быть списком строк')

    # Незаданный format_check — это FORMAT_CHECK_MARKER, поведение всего
    # банка Day 03 без правки его файлов. А вот ЗАДАННОЕ, но опечатанное
    # значение (например "makrer") обязано падать явной ошибкой, а не тихо
    # повести себя как marker, — иначе опечатку заметят только по неверной
    # колонке в таблице, через несколько дней после того, как файл правили
    # (SPEC-w01d04.md §6).
    format_check = str(data.get("format_check") or FORMAT_CHECK_MARKER)
    if format_check not in FORMAT_CHECKS:
        allowed = ", ".join(FORMAT_CHECKS)
        raise ConfigError(
            f'Поле "format_check" в {path} = {format_check!r}, ожидалось одно из: {allowed}'
        )

    # Неизвестные ключи игнорируются молча: банк — данные, а не схема, и
    # заметка автора задачи не должна ронять загрузчик.
    return Problem(
        id=str(data["id"]),
        title=str(data["title"]),
        statement=str(data["statement"]).strip(),
        answer=str(data["answer"]).strip(),
        accept=tuple(str(item).strip() for item in accept if str(item).strip()),
        note=str(data.get("note") or "").strip(),
        placeholder=bool(data.get("placeholder")),
        default=bool(data.get("default")),
        open=bool(data.get("open")),
        format_check=format_check,
    )


# --------------------------------------------------------------------------
# Промпты
# --------------------------------------------------------------------------


def load_prompt(name: str) -> str:
    """Читает week_01/prompts/reason_<name>.md.

    Сознательно без кэша: промпты подбираются итерациями прямо между
    прогонами, и кэш означал бы «правка не применилась, перезапусти» ровно в
    тот момент, когда идёт подбор формулировки.
    """
    path = PROMPTS_DIR / f"reason_{name}.md"
    if not path.is_file():
        raise ConfigError(f"Не найден промпт стратегии: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigError(f"Промпт стратегии пуст: {path}")
    return text


def marker_instruction() -> str:
    """Общая добавка про маркер ответа — одна и та же для всех стратегий.

    Одинаковость здесь принципиальна: если бы добавка отличалась, сравнение
    стратегий сравнивало бы заодно и формулировки инструкции (SPEC-w01d03.md
    §6).
    """
    return load_prompt("marker")


def build_strategy_system(config: Config, *parts: str | None, marker: bool = True) -> str | None:
    """Собирает system: пользовательский → промпт стратегии → добавка маркера.

    marker=False нужен ровно одному вызову — первому шагу `meta`, где модель
    пишет промпт и ничего не решает: требовать от неё строку `ОТВЕТ:` там
    означало бы прямо подтолкнуть её решить задачу, чего этот шаг как раз
    запрещает.

    Штатный system-промпт проекта («лаконичный ассистент, отвечай по делу, без
    воды») в стратегии НЕ подмешивается: он прямо противоречит инструкции
    рассуждать пошагово, и модель слушается его. Замер на children,
    mistral-small, по 5 прогонов: с персоной steps даёт верный ответ 2 раза из
    5 при средней длине ответа 679 символов, без неё — 4 из 5 при 1232
    символах. То есть персона вдвое укорачивает рассуждение и вдвое повышает
    долю ошибок, обнуляя ровно тот эффект, который этот день измеряет.

    Явно заданный пользователем system (--system или ADVENT_SYSTEM_PROMPT)
    при этом сохраняется: он выбран осознанно, и молча его игнорировать
    означало бы врать про то, что ушло в API.
    """
    explicit_system = (
        None if config.system_prompt_path == DEFAULT_SYSTEM_PROMPT else config.system_prompt()
    )
    pieces = [explicit_system, *parts]
    if marker:
        pieces.append(marker_instruction())
    kept = [piece.strip() for piece in pieces if piece and piece.strip()]
    return "\n\n".join(kept) or None


# --------------------------------------------------------------------------
# Извлечение ответа и сверка с эталоном
# --------------------------------------------------------------------------

# Слово маркера отдельно от двоеточия: между ними модель охотно вставляет
# markdown ("**ОТВЕТ:**"). Обе регулярки собираются из ANSWER_MARKER, чтобы
# правка константы не разошлась с разбором.
_MARKER_WORD = ANSWER_MARKER.rstrip(":")

# Строгий маркер: слово ЗАГЛАВНЫМИ в начале строки — ровно то, что требует
# reason_marker.md ("Слово ОТВЕТ пиши заглавными буквами", "она должна быть
# последней"). Регистр здесь значим намеренно: заглавные буквы и есть признак
# того, что модель выполняла формат, а не произнесла слово «ответ» в прозе.
# Перед маркером допускается markdown-обвес: "### **ОТВЕТ:**", "- ОТВЕТ:".
_STRICT_MARKER_RE = re.compile(rf"^[ \t>\-*_#]*{_MARKER_WORD}[ \t*_`]*:", re.MULTILINE)

# Мягкий запасной вариант: маркер в любом регистре и в любом месте строки.
# \b по краям слова не даёт совпасть с "Ответственность:".
_MARKER_RE = re.compile(rf"\b{_MARKER_WORD}\b[ \t*_`]*:", re.IGNORECASE)

# Мусор вокруг самого значения: жирный markdown, кавычки, концевая точка.
_TRIM = " \t*_`\"'«»"


def extract_answer(text: str) -> str | None:
    """Достаёт значение из строки с маркером. None — маркера нет вовсе.

    Сначала ищется СТРОГИЙ маркер (заглавными, в начале строки), и только при
    его отсутствии — мягкий. Порядок здесь несущий, а не косметический: модель
    регулярно дописывает прозу ПОСЛЕ строки маркера («ОТВЕТ: 1\\n\\nЕсли считать,
    что козу нельзя оставлять одну, то ответ: 3»), и мягкая регулярка забирала
    из неё последнее «ответ:», отдавая 3 вместо 1. Верный ответ получал вердикт
    ✗ — то есть ломался главный вывод дня, а не оформление.

    Внутри выбранного вида берётся ПОСЛЕДНЕЕ вхождение: модель охотно упоминает
    требуемый формат по ходу рассуждения («в конце напишу ОТВЕТ: …»), и первое
    вхождение почти всегда не то (SPEC-w01d03.md §6).

    None — отдельное состояние, не «пустой ответ»: вызывающий код обязан
    отличать «не выполнен формат» от «ответ неверный».
    """
    matches = list(_STRICT_MARKER_RE.finditer(text)) or list(_MARKER_RE.finditer(text))
    if not matches:
        return None

    tail = text[matches[-1].end() :]
    for line in tail.splitlines():
        # Значение обычно на той же строке, что и маркер; если модель
        # перенесла его на следующую — берём первую непустую.
        value = line.strip().strip(_TRIM).strip()
        if value:
            return value
    return ""


def normalize(text: str) -> str:
    """Приводит ответ к сравнимому виду: регистр, пробелы, точка, ё → е."""
    lowered = text.strip().lower().replace("ё", "е")
    collapsed = re.sub(r"\s+", " ", lowered)
    return collapsed.strip(_TRIM + ".").strip()


@dataclass(slots=True, frozen=True)
class Check:
    """Вердикт по эталону: что извлекли, совпало ли, и был ли маркер."""

    verdict: str
    answer: str | None
    expected: str

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_OK

    @property
    def has_marker(self) -> bool:
        return self.verdict != VERDICT_NO_MARKER

    @property
    def label(self) -> str:
        return VERDICT_LABELS[self.verdict]


def check_answer(text: str, problem: Problem) -> Check:
    """Сверяет ответ модели с эталоном задачи и её списком accept."""
    answer = extract_answer(text)
    if answer is None:
        return Check(VERDICT_NO_MARKER, None, problem.answer)

    candidates = _reference_forms(problem)
    verdict = VERDICT_OK if normalize(answer) in candidates else VERDICT_WRONG
    return Check(verdict, answer, problem.answer)


def _reference_forms(problem: Problem) -> set[str]:
    """Все нормализованные формы верного ответа: эталон плюс accept."""
    forms = {normalize(problem.answer), *(normalize(item) for item in problem.accept)}
    forms.discard("")
    return forms


def mentions_expected_answer(text: str, problem: Problem) -> bool:
    """Встречается ли эталон в произвольном тексте (не в ответе на задачу).

    Нужно ровно одному месту — первому шагу `meta`, который обязан НАПИСАТЬ
    промпт и не решать. Если он вопреки запрету решил, эталон уезжает в system
    второго вызова, и ✓ в таблице у meta не значит ничего: способ не сработал,
    а получил ответ в подарок (SPEC-w01d03.md §7, §13). Вывод дня о точности
    способов при этом ложный, поэтому такой прогон помечается вслух.

    Поиск по границам слова, а не подстрокой: эталон «7» иначе нашёлся бы в
    любом «шаг 17». Ложные срабатывания всё же возможны (эталон «7» и фраза
    «проверь 7 раз»), и это осознанный размен: лишняя оговорка честнее, чем
    молчаливая ✓, которую нечем проверить.
    """
    needles = _reference_forms(problem)
    if not needles:
        return False
    haystack = normalize(text)
    return any(re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) for needle in needles)


# --------------------------------------------------------------------------
# Один вызов внутри стратегии
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Step:
    """Один вызов LLM внутри стратегии: как называется, что ушло, что пришло.

    Шаг знает свою стратегию и номер прогона, хотя их знает и StrategyRun.
    Так он самодостаточен: и журнал (`log_step`), и печать (`print_step`)
    работают с шагом сразу после вызова — то есть тогда, когда объекта
    StrategyRun ещё не существует и может не появиться вовсе.
    """

    name: str
    title: str
    messages: list[Message]
    result: CallResult
    strategy: str
    run: int

    @property
    def text(self) -> str:
        return self.result.text


@dataclass(slots=True)
class _Caller:
    """Замыкание вокруг chat.complete: копит шаги и рапортует о прогрессе."""

    config: Config
    capabilities: dict[str, Any] | None
    complete: CompleteFn
    notify: Notify
    strategy: str
    run: int = 1
    # Зовётся после КАЖДОГО успешного вызова. Всё, что должно пережить падение
    # стратегии на середине (журнал, печать ответа), висит именно здесь.
    on_step: StepHook | None = None
    steps: list[Step] = field(default_factory=list)
    # Короткие пометки о том, что делает вердикт по эталону недостоверным
    # (пока такая ровно одна — эталон, утёкший в промпт meta). Список, а не
    # флаг: они попадают в таблицу одной строкой и должны складываться.
    caveats: list[str] = field(default_factory=list)

    def ask(self, name: str, title: str, *, system: str | None, user: str) -> Step:
        self.notify(f"  → {title}")
        messages = chat_core.build_messages(user, system=system)
        result = self.complete(self.config, messages, self.capabilities)
        step = Step(
            name=name,
            title=title,
            # sent_messages — то, что реально ушло в API после дописывания
            # инструкции формата в chat._payload(); журнал должен видеть
            # именно его, иначе source of truth расходится (как в cli.py).
            messages=result.sent_messages or messages,
            result=result,
            strategy=self.strategy,
            run=self.run,
        )
        self.steps.append(step)
        if self.on_step is not None:
            self.on_step(step)
        return step


# --------------------------------------------------------------------------
# Стратегии
# --------------------------------------------------------------------------


def _run_direct(problem: Problem, caller: _Caller) -> None:
    """Один вызов: задача как есть, поверх только общая добавка про маркер."""
    caller.ask(
        "answer",
        "прямой ответ",
        system=build_strategy_system(caller.config),
        user=problem.statement,
    )


def _run_steps(problem: Problem, caller: _Caller) -> None:
    """Один вызов: та же задача плюс инструкция рассуждать пошагово."""
    caller.ask(
        "answer",
        "пошаговое решение",
        system=build_strategy_system(caller.config, load_prompt("steps")),
        user=problem.statement,
    )


def _run_meta(problem: Problem, caller: _Caller) -> None:
    """Два вызова: модель пишет промпт, затем решает по нему.

    Первый вызов идёт без добавки про маркер (см. build_system) и с явным
    запретом решать — известный риск дня в том, что модель охотно начинает
    решать задачу вместо написания промпта (SPEC-w01d03.md §7, §13).
    """
    written = caller.ask(
        "prompt",
        "модель пишет промпт",
        system=build_strategy_system(caller.config, load_prompt("meta"), marker=False),
        user=problem.statement,
    )

    generated = written.text.strip()
    if not generated:
        # Пустой промпт — не повод терять стратегию: второй вызов вырождается
        # в прямой ответ, и об этом надо сказать вслух, иначе в таблице meta
        # окажется неотличима от direct без всякого объяснения.
        console.warn("meta: модель вернула пустой промпт — второй вызов пойдёт без него")
    elif mentions_expected_answer(generated, problem):
        # Тот самый риск способа, который день и должен показывать: модель
        # вопреки запрету решила задачу прямо в промпте. Дальше эталон уедет в
        # system второго вызова, тот его перепишет, и в таблице появится ✓,
        # неотличимая от честной работы способа. Говорим вслух — так же явно,
        # как про пустой промпт, — и метим прогон, чтобы оговорка дожила до
        # таблицы, а не осталась строкой в stderr над девятью вызовами.
        caller.caveats.append("эталон в промпте")
        console.warn(
            f"meta: в сгенерированном промпте уже есть эталонный ответ ({problem.answer}) — "
            "второй вызов получает его в подарок, и ✓ у meta ничего не доказывает"
        )

    caller.ask(
        "solve",
        "решение по сгенерированному промпту",
        system=build_strategy_system(caller.config, generated or None),
        user=problem.statement,
    )


def _run_panel(problem: Problem, caller: _Caller) -> None:
    """Четыре вызова: три независимых эксперта и синтез.

    Независимость буквальная: каждый эксперт получает свежие messages из
    system + условия, без истории и без чужих решений. Расплата за это —
    критику нечего критиковать, он решает задачу со своей позиции; размен
    осознанный, ради возможности увидеть, сошлись ли эксперты сами по себе
    (SPEC-w01d03.md §7).
    """
    roles = (
        ("analyst", "аналитик"),
        ("engineer", "инженер"),
        ("critic", "критик"),
    )
    solutions: list[str] = []
    for name, title in roles:
        step = caller.ask(
            name,
            title,
            system=build_strategy_system(caller.config, load_prompt(f"panel_{name}")),
            user=problem.statement,
        )
        solutions.append(step.text)

    # Сошлись ли эксперты, считаем сами и печатаем в stderr. Раньше об этом
    # просили синтез, и его ответ начинался с «аналитик и инженер сошлись…» —
    # то есть текст, который уходит судье как «Ответ D», сам сообщал, что D это
    # работа группы экспертов. Метки A–D формально анонимны, а фора из-за
    # названия возвращалась через содержание (SPEC-w01d03.md §8).
    console.note(f"панель: {_panel_agreement(solutions)}")

    # Роли не названы и в user-сообщении синтеза: «Решение, которое дал
    # критик:» модель охотно цитирует обратно. Нумерация ничего не отнимает —
    # сами ответы экспертов печатаются целиком и подписаны ролями в stderr.
    parts = [f"Задача:\n{problem.statement}"]
    parts += [f"Черновое решение {index}:\n{text}" for index, text in enumerate(solutions, start=1)]
    caller.ask(
        "synthesis",
        "синтез",
        system=build_strategy_system(caller.config, load_prompt("panel_synthesis")),
        user="\n\n".join(parts),
    )


def _panel_agreement(solutions: Sequence[str]) -> str:
    """Сошлись ли эксперты — по извлечённым из их ответов значениям.

    Признак дешёвый и детерминированный, в отличие от «спросить у синтеза».
    Ради этого наблюдения независимость экспертов и куплена ценой того, что
    критику нечего критиковать (SPEC-w01d03.md §7), — потерять его вместе с
    переписанным промптом синтеза было бы обидно.
    """
    # None и "" — разные состояния (формат не выполнен против пустого
    # значения), и схлопывать их в одну строку тут значило бы повторить ровно
    # то смешивание, которое запрещает §6.
    extracted = [extract_answer(text) for text in solutions]
    answers = ["нет маркера" if answer is None else answer for answer in extracted]
    distinct = list(dict.fromkeys(answers))
    if len(distinct) == 1:
        return f"эксперты сошлись на «{distinct[0]}»"
    return "эксперты разошлись: " + ", ".join(f"«{answer}»" for answer in answers)


_RUNNERS: dict[str, Callable[[Problem, _Caller], None]] = {
    "direct": _run_direct,
    "steps": _run_steps,
    "meta": _run_meta,
    "panel": _run_panel,
}


@dataclass(slots=True)
class StrategyRun:
    """Один прогон одной стратегии: все её вызовы и вердикт по эталону."""

    strategy: str
    problem_id: str
    steps: list[Step]
    check: Check
    # Оговорки к вердикту: то, из-за чего ✓/✗ этого прогона нельзя читать
    # буквально. Доезжают до таблицы через StrategyOutcome.verdict_label.
    caveats: list[str] = field(default_factory=list)

    @property
    def final(self) -> Step:
        """Последний вызов: его ответ и есть ответ стратегии."""
        return self.steps[-1]

    @property
    def text(self) -> str:
        return self.final.text

    @property
    def answer(self) -> str | None:
        return self.check.answer

    @property
    def verdict(self) -> str:
        return self.check.verdict

    @property
    def results(self) -> list[CallResult]:
        return [step.result for step in self.steps]

    @property
    def totals(self) -> Totals:
        return Totals.of(self.results)


def run_strategy(
    strategy: str,
    problem: Problem,
    config: Config,
    *,
    capabilities: dict[str, Any] | None = None,
    complete: CompleteFn = chat_core.complete,
    notify: Notify = console.note,
    warn_unsafe_params: bool = True,
    index: int = 1,
    on_step: StepHook | None = None,
) -> StrategyRun:
    """Один прогон одной стратегии. Вызовы идут строго по очереди.

    `warn_unsafe_params=False` ставит solve(): предупреждения про `stop` и
    `format` он печатает один раз на весь прогон. Сама правка конфига при этом
    происходит всегда — run_strategy() остаётся самодостаточным и безопасным
    при прямом вызове (вторая поверхность дня зовёт именно его).

    `on_step` зовётся после каждого успешного вызова, а не после прогона. Это
    несущее свойство, а не удобство: `panel` — четыре последовательных вызова,
    и 429 на четвёртом раньше уносил три уже оплаченных ответа и с экрана, и
    из journal'а, хотя журнал — единственное, что этот день сохраняет.
    `index` — номер прогона при `--runs N`, он попадает в каждый шаг и оттуда
    в строку журнала.
    """
    runner = _RUNNERS.get(strategy)
    if runner is None:
        allowed = ", ".join(STRATEGIES)
        raise ConfigError(f"Неизвестная стратегия {strategy!r}. Доступны: {allowed}")

    caller = _Caller(
        config=_solving_config(config, warn=warn_unsafe_params),
        capabilities=capabilities,
        complete=complete,
        notify=notify,
        strategy=strategy,
        run=index,
        on_step=on_step,
    )
    runner(problem, caller)
    check = check_answer(caller.steps[-1].result.text, problem)
    return StrategyRun(
        strategy=strategy,
        problem_id=problem.id,
        steps=caller.steps,
        check=check,
        caveats=caller.caveats,
    )


def _solving_config(config: Config, *, warn: bool = True) -> Config:
    """Убирает из `stop` всё, что съело бы маркер ответа.

    Прямое следствие факта из CLAUDE.md: API вырезает стоп-последовательность
    из вывода. Если пользователь выставил `--stop ОТВЕТ` (или любую строку,
    пересекающуюся с маркером), извлекать станет нечего — и это выглядело бы
    как «модель не выполнила формат», хотя виноват параметр. Исходный config
    не мутируется: REPL держит те же params дальше.

    format пользователя при этом НЕ подменяется, только предупреждается:
    `--format json` вместе со стратегией — осознанный выбор пользователя, и
    молча переписывать его флаг хуже, чем сказать вслух, что маркера в JSON
    может не оказаться.

    `warn=False` глушит оба предупреждения, не трогая саму правку: конфиг
    считается один раз на прогон в solve(), а без этого одно и то же
    предупреждение уходило в stderr на каждый вызов run_strategy() — двенадцать
    одинаковых строк на `--runs 3` со всеми четырьмя стратегиями, ровно между
    шагами демо.
    """
    if warn and config.params.format in ("json", "schema"):
        console.warn(
            f"format={config.params.format}: ответ придёт JSON'ом, и строки "
            f"{ANSWER_MARKER} в нём может не быть — сверка с эталоном тогда "
            "скажет «нет маркера»"
        )

    stop = config.params.stop
    if not stop:
        return config

    marker = ANSWER_MARKER.lower()
    unsafe = [item for item in stop if item.lower() in marker or marker in item.lower()]
    if not unsafe:
        return config

    if warn:
        console.warn(
            f"stop={', '.join(unsafe)} пересекается с маркером {ANSWER_MARKER} — "
            "на время решения этот stop не отправляется, иначе маркер вырежется из ответа"
        )
    safe = [item for item in stop if item not in unsafe]
    return replace(config, params=replace(config.params, stop=safe or None))


# --------------------------------------------------------------------------
# Несколько прогонов на стратегию
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StrategyOutcome:
    """Итог по стратегии: N прогонов, суммарная цена, вердикт k/N."""

    strategy: str
    runs: list[StrategyRun]

    @property
    def first(self) -> StrategyRun:
        return self.runs[0]

    @property
    def text(self) -> str:
        """Ответ, который идёт судье: финальный текст ПЕРВОГО прогона.

        При --runs N судья всё равно видит по одному ответу на стратегию —
        иначе один вызов на всех (SPEC-w01d03.md §8) превратился бы в N.
        """
        return self.first.text

    @property
    def correct(self) -> int:
        return sum(1 for run in self.runs if run.check.ok)

    @property
    def no_marker(self) -> int:
        return sum(1 for run in self.runs if not run.check.has_marker)

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def totals(self) -> Totals:
        summed = Totals()
        for run in self.runs:
            summed = summed + run.totals
        return summed

    @property
    def answer_label(self) -> str:
        """Ответ для таблицы. При N > 1 показывает разброс, а не первый попавшийся."""
        distinct: list[str] = []
        for run in self.runs:
            shown = run.answer if run.answer is not None else "нет маркера"
            if shown not in distinct:
                distinct.append(shown)
        if len(distinct) <= 2:
            return " / ".join(distinct)
        return f"{distinct[0]} и ещё {len(distinct) - 1}"

    @property
    def caveats(self) -> list[str]:
        """Оговорки всех прогонов без повторов, в порядке появления."""
        collected: list[str] = []
        for run in self.runs:
            for note in run.caveats:
                if note not in collected:
                    collected.append(note)
        return collected

    @property
    def verdict_label(self) -> str:
        """Колонка «Эталон»: символ при одном прогоне, k/N при нескольких.

        Оговорка (например, «эталон в промпте» у meta) приписывается прямо
        сюда: без неё ✓ в таблице читается как «способ сработал», хотя способ
        мог просто получить ответ в подарок, — и вывод дня о точности способов
        становится ложным.
        """
        if self.total == 1:
            label = self.first.check.label
        else:
            label = f"{self.correct}/{self.total}"
            if self.no_marker:
                label += f" · без маркера: {self.no_marker}"
        for note in self.caveats:
            label += f" · {note}"
        return label


def solve(
    problem: Problem,
    config: Config,
    *,
    strategies: Sequence[str] = STRATEGIES,
    runs: int = 1,
    capabilities: dict[str, Any] | None = None,
    complete: CompleteFn = chat_core.complete,
    notify: Notify = console.note,
    on_step: StepHook | None = None,
    on_run: Callable[[StrategyRun], None] | None = None,
) -> list[StrategyOutcome]:
    """Прогоняет стратегии по очереди, каждую `runs` раз.

    Порядок результатов повторяет порядок `strategies` — от него зависят и
    таблица, и метки судьи, поэтому он фиксирован и не сортируется.

    Два колбэка, потому что у них разный момент истинности. `on_step` зовётся
    по факту каждого успешного вызова — на нём висит журнал и печать ответа,
    и он единственный переживает падение стратегии на середине. `on_run` —
    после законченного прогона, когда есть чем сверяться с эталоном: вердикт
    раньше последнего шага не существует. Без них весь вывод копился до
    возврата, то есть до девятого вызова: экран молчал минуту, а упавший
    прогон не оставлял вообще ничего.
    """
    if runs < 1:
        raise ConfigError(f"runs должен быть не меньше 1, получено {runs}")

    # Небезопасный stop снимается и предупреждения печатаются ОДИН раз на весь
    # прогон, а не внутри каждого run_strategy(): предупреждение зависит от
    # конфига, а не от стратегии, и повторённое по разу на каждый из двенадцати
    # прогонов оно забивает stderr ровно там, где на записи идёт ожидание API.
    config = _solving_config(config)

    outcomes: list[StrategyOutcome] = []
    for strategy in strategies:
        collected: list[StrategyRun] = []
        for index in range(1, runs + 1):
            suffix = f" (прогон {index}/{runs})" if runs > 1 else ""
            notify(f"{strategy}{suffix}")
            run = run_strategy(
                strategy,
                problem,
                config,
                capabilities=capabilities,
                complete=complete,
                notify=notify,
                warn_unsafe_params=False,
                index=index,
                on_step=on_step,
            )
            collected.append(run)
            if on_run is not None:
                on_run(run)
        outcomes.append(StrategyOutcome(strategy=strategy, runs=collected))
    return outcomes


def log_step(step: Step, *, week: int, day: int, problem: str) -> None:
    """Пишет ОДИН состоявшийся вызов в JSONL-журнал Day 01.

    Единица записи — вызов, а не прогон и не стратегия. Это и есть то, что
    обещают README и SPEC §2: отдельного `--out` у дня нет, журнал и есть
    сохранение результата, поэтому вызов, за который уже заплачено, обязан
    попасть в файл независимо от того, доживёт ли стратегия до конца.
    Зовётся из `on_step`, то есть сразу после успешного complete().

    Строка помечена стратегией и ролью шага (`panel` / `critic`): один ответ
    этого дня складывается из нескольких вызовов, и без меток строки журнала
    неразличимы — а именно по ним неделя 2 будет считать токены.
    """
    log_call(
        step.result,
        step.messages,
        week=week,
        day=day,
        extra={
            "strategy": step.strategy,
            "step": step.name,
            "problem": problem,
            "run": step.run,
        },
    )


def log_run(run: StrategyRun, *, week: int, day: int) -> None:
    """Пишет все вызовы одного законченного прогона.

    Путь «записать всё разом» — для вызывающего, который не подключил
    `on_step` (тесты, будущие поверхности). CLI им не пользуется: там журнал
    ведётся пошагово, иначе упавшая стратегия не записала бы ничего.
    """
    for step in run.steps:
        log_step(step, week=week, day=day, problem=run.problem_id)


def log_calls(outcomes: Iterable[StrategyOutcome], *, week: int, day: int) -> None:
    """Пишет каждый вызов каждого прогона — то же, что log_run, но целиком."""
    for outcome in outcomes:
        for run in outcome.runs:
            log_run(run, week=week, day=day)


# --------------------------------------------------------------------------
# Судья
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class JudgeScore:
    label: str
    score: int
    comment: str


@dataclass(slots=True)
class JudgeVerdict:
    """Оценка судьи. error != None означает «судья не отработал», но прогон жив."""

    labels: dict[str, str] = field(default_factory=dict)
    ranking: list[str] = field(default_factory=list)
    scores: dict[str, JudgeScore] = field(default_factory=dict)
    result: CallResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def totals(self) -> Totals:
        return Totals.of([self.result] if self.result is not None else [])

    def label_of(self, strategy: str) -> str | None:
        for label, name in self.labels.items():
            if name == strategy:
                return label
        return None

    def rank_of(self, strategy: str) -> int | None:
        """Место стратегии в ранжировании, начиная с 1. None — судья её не назвал."""
        label = self.label_of(strategy)
        if label is None or label not in self.ranking:
            return None
        return self.ranking.index(label) + 1

    def score_of(self, strategy: str) -> JudgeScore | None:
        label = self.label_of(strategy)
        return self.scores.get(label) if label else None

    def cell(self, strategy: str) -> str:
        """Колонка «Судья» одной строкой: место и балл."""
        if not self.ok:
            return "—"
        rank = self.rank_of(strategy)
        score = self.score_of(strategy)
        parts = []
        if rank is not None:
            parts.append(f"#{rank}")
        if score is not None:
            parts.append(f"{score.score}/10")
        return " · ".join(parts) or "—"


def judge(
    problem: Problem,
    outcomes: Sequence[StrategyOutcome],
    config: Config,
    *,
    capabilities: dict[str, Any] | None = None,
    complete: CompleteFn = chat_core.complete,
    notify: Notify = console.note,
    model: str | None = None,
) -> JudgeVerdict:
    """Один вызов на все ответы: ранжирование плюс балл и комментарий.

    Эталона судья не видит — оценивает качество рассуждения, а не совпадение
    с ключом. Расхождение судьи с эталоном это результат дня, а не сбой
    (SPEC-w01d03.md §8).

    Судья не имеет права уронить прогон: любая ошибка — от отказа API до
    невалидного JSON — превращается в JudgeVerdict с заполненным error, а уже
    полученные ответы стратегий остаются на экране.

    `capabilities` — карточка ИМЕННО судейской модели (`model`), а не той, что
    решала. Отсев параметров по чужим capabilities отправил бы `magistral`-ий
    `reasoning_effort` в `mistral-small` и получил 400 вместо оценки. Считать
    её здесь нечем — списка моделей аккаунта модуль не знает, — поэтому это
    обязанность вызывающего (week_01/cli.py).
    """
    # strict=False намеренно: стратегий может быть меньше четырёх
    # (--strategy direct), и лишние метки просто не используются.
    labels = {
        label: outcome.strategy for label, outcome in zip(JUDGE_LABELS, outcomes, strict=False)
    }
    if not labels:
        return JudgeVerdict(error="нечего оценивать: ни одной стратегии не отработало")

    notify("судья")
    parts = [f"Задача:\n{problem.statement}"]
    parts += [
        # Порядок фиксирован; позиционная предвзятость судьи — известное
        # ограничение, названное в отчёте, а не замаскированное перетасовкой.
        #
        # Рядом с ней живёт второе ограничение той же природы: анонимность
        # держится не только на метках. Текст ответа сам может себя выдать —
        # синтез панели раньше начинался с «аналитик и инженер сошлись…», и
        # судья видел, что D это работа группы экспертов. Промпт синтеза это
        # теперь запрещает, но гарантии нет: остаточный риск назван в
        # SPEC-w01d03.md §8, а не замолчан.
        f"Ответ {label}:\n{outcome.text.strip() or '(пустой ответ)'}"
        for label, outcome in zip(JUDGE_LABELS, outcomes, strict=False)
    ]

    judge_config = _judge_config(config, model)

    try:
        # load_prompt() и сборка system стоят ВНУТРИ try намеренно. Промпты
        # правятся руками между прогонами (см. docstring load_prompt), и
        # опечатка в файле поднимает ConfigError — а он не AdventError и
        # снаружи try убил бы весь прогон после восьми уже оплаченных вызовов,
        # не напечатав таблицу. Судья не имеет права уронить прогон ни по
        # какой причине, а не только из-за отказа API.
        messages = chat_core.build_messages(
            "\n\n".join(parts),
            system=build_strategy_system(judge_config, load_prompt("judge"), marker=False),
        )
        result = complete(judge_config, messages, capabilities)
    except ConfigError as error:
        return JudgeVerdict(labels=labels, error=str(error))
    except AdventError as error:
        return JudgeVerdict(labels=labels, error=error.message)

    try:
        ranking, scores = _parse_judge(result.text, labels)
    except ValueError as error:
        return JudgeVerdict(labels=labels, result=result, error=str(error))

    return JudgeVerdict(labels=labels, ranking=ranking, scores=scores, result=result)


def log_judge(verdict: JudgeVerdict, *, week: int, day: int, problem: str | None = None) -> None:
    """Пишет вызов судьи в тот же журнал, что и вызовы стратегий.

    Судья не проходит через log_calls(): это отдельный вызов, не принадлежащий
    ни одной стратегии, — в журнале он помечен strategy=judge. Ошибка разбора
    ответа пишется в поле error и не отменяет запись: сам вызов состоялся и
    стоил токенов. Если же судья упал на самом обращении к API (result is
    None), писать нечего — что именно ушло в API, знает только judge(), и
    придумывать за него список сообщений хуже, чем не записать строку.
    """
    if verdict.result is None:
        return
    log_call(
        verdict.result,
        verdict.result.sent_messages or [],
        week=week,
        day=day,
        error=verdict.error,
        # problem передаётся снаружи: JudgeVerdict про задачу ничего не знает,
        # а без неё строку судьи не связать с прогоном, который он оценивал.
        extra={"strategy": "judge", "step": "judge", "problem": problem},
    )


def _judge_config(config: Config, model: str | None) -> Config:
    """Конфиг судьи: та же модель (или judge_model) и format=schema.

    Схема подаётся через штатный механизм Day 02 (advent_core/formats.py):
    response_format=json_schema плюс дословная инструкция в system. Один
    параметр без инструкции гарантировал бы валидный, но произвольный JSON —
    это записано в CLAUDE.md и здесь ровно тот случай.

    Копия, а не правка на месте: пользовательские format/schema_file
    относятся к его собственным вопросам и не должны исчезнуть после solve.
    """
    params = replace(config.params, format="schema", schema_file=str(JUDGE_SCHEMA_PATH))
    updated = replace(config, params=params)
    return replace(updated, model=model) if model else updated


def _parse_judge(text: str, labels: dict[str, str]) -> tuple[list[str], dict[str, JudgeScore]]:
    """Разбирает ответ судьи. ValueError — ответ непригоден.

    Схема уже проверена chat.complete() через formats.verify(), но её
    результат сюда не доезжает, а полагаться на «схема была строгой» нельзя:
    strict=True — обещание API, а не гарантия смысла. Метки вне поданного
    набора игнорируются, дубликаты в ranking схлопываются.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ответ судьи не разбирается как JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("ответ судьи — не JSON-объект")

    scores: dict[str, JudgeScore] = {}
    for item in data.get("scores") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().upper()
        if label not in labels:
            continue
        try:
            score = int(item.get("score"))
        except (TypeError, ValueError):
            continue
        scores[label] = JudgeScore(label, score, str(item.get("comment") or "").strip())

    ranking: list[str] = []
    for raw in data.get("ranking") or []:
        label = str(raw).strip().upper()
        if label in labels and label not in ranking:
            ranking.append(label)

    if not ranking and not scores:
        raise ValueError("судья не назвал ни одной известной метки")
    return ranking, scores


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------


def print_problem(problem: Problem) -> None:
    """Условие и эталон — в stderr: это обстановка, а не ответ модели."""
    # Квадратные скобки вокруг id тут были бы съедены Rich-markup как
    # незакрытый тег (та же грабля, что в cli.py вокруг done/схемы).
    console.note(f"задача {problem.id} — {problem.title}")
    console.err.print(rich_escape(problem.statement))
    console.note(f"эталон: {problem.answer}")
    if problem.note:
        console.note(problem.note)
    if problem.placeholder:
        console.warn(f"задача {problem.id} помечена placeholder — эталон не выверен живым прогоном")


def print_step(step: Step) -> None:
    """Один вызов: заголовок в stderr, текст ответа в stdout.

    Печатать умеет отдельный шаг, а не только законченный прогон: CLI вешает
    это на `on_step` и показывает ответ сразу, как он пришёл. Иначе экран
    молчит все четыре вызова панели, а упавший пятый уносит с собой четыре
    уже полученных ответа.
    """
    console.note(f"── {step.strategy}: {step.title} ──")
    console.write_chunk(step.text)
    console.finish_answer()
    console.footer(step.result)


def print_verdict(run: StrategyRun) -> None:
    """Только сверка с эталоном — без ответов модели.

    Отдельно от print_step(), потому что вердикта не существует до последнего
    вызова прогона, а сами ответы печатаются по мере поступления. Вторая
    поверхность дня (`chat --strategy ...`) не зовёт эту функцию вовсе: у
    произвольного вопроса эталона нет, и «✗» там был бы прямой ложью.
    """
    if run.check.has_marker:
        # rich_escape: извлечённое значение пришло от модели, и «ОТВЕТ: [b]7[/b]»
        # console.note() съел бы как markup — на экране осталось бы «извлечено: 7»
        # без скобок, то есть пользователь видит ✗ и не видит, из-за чего.
        console.note(
            f"извлечено: {rich_escape(run.answer or '')}  ·  "
            f"эталон: {rich_escape(run.check.expected)}  {run.check.label}"
        )
    else:
        console.warn(
            f"маркер {ANSWER_MARKER} в ответе не найден — "
            "ответ не извлечён (это не то же самое, что неверный ответ)"
        )


def print_comparison(
    problem: Problem,
    outcomes: Sequence[StrategyOutcome],
    verdict: JudgeVerdict | None = None,
) -> None:
    """Таблица сравнения (SPEC-w01d03.md §9) — в stdout, это итог дня."""
    runs = outcomes[0].total if outcomes else 1
    title = f"{problem.title} · эталон: {problem.answer}"
    if runs > 1:
        title += f" · прогонов на стратегию: {runs}"

    table = Table(title=title)
    table.add_column("стратегия", style="cyan", no_wrap=True)
    table.add_column("ответ")
    table.add_column("эталон", justify="center")
    table.add_column("судья", justify="center")
    table.add_column("вызовов", justify="right")
    table.add_column("токены", justify="right")
    table.add_column("время", justify="right")

    for outcome in outcomes:
        totals = outcome.totals
        style = {VERDICT_OK: "green", VERDICT_WRONG: "red"}.get(outcome.first.verdict)
        if outcome.total > 1:
            # При нескольких прогонах строка целиком не «зелёная» и не
            # «красная» — k/N говорит сам за себя, а покраска по первому
            # прогону врала бы.
            style = None
        table.add_row(
            outcome.strategy,
            rich_escape(outcome.answer_label),
            outcome.verdict_label,
            verdict.cell(outcome.strategy) if verdict else "—",
            str(totals.calls),
            totals.tokens_label(),
            totals.time_label(),
            style=style,
        )

    console.out.print(table)

    if verdict is not None:
        _print_judge(outcomes, verdict)


def _print_judge(outcomes: Sequence[StrategyOutcome], verdict: JudgeVerdict) -> None:
    """Комментарии судьи и честная пометка о расхождении с эталоном."""
    if not verdict.ok:
        # Не ошибка прогона: стратегии отработали, судья — нет. Предупреждение
        # в stderr, чтобы таблица в stdout осталась чистой.
        console.warn(f"судья не отработал: {verdict.error}")
        return

    order = (
        " → ".join(f"{label} ({verdict.labels.get(label, '?')})" for label in verdict.ranking)
        or "—"
    )
    console.out.print(f"\n[bold]ранжирование судьи:[/bold] {order}")

    for outcome in outcomes:
        score = verdict.score_of(outcome.strategy)
        if score is None:
            continue
        label = verdict.label_of(outcome.strategy)
        console.out.print(
            f"  [cyan]{label}[/cyan] {outcome.strategy}: {score.score}/10 — "
            f"{rich_escape(score.comment)}"
        )

    best = verdict.ranking[0] if verdict.ranking else None
    winner = verdict.labels.get(best) if best else None
    top = next((item for item in outcomes if item.strategy == winner), None)
    if top is None or top.correct:
        return

    # Судья эталона не видит — расхождение это результат дня, а не сбой, и
    # формулировка не должна выглядеть как ошибка программы.
    #
    # Две ветки, а не одна: «неверен по эталону» и «ответ не извлекался» —
    # разные состояния (§6), и вторую нельзя называть первой. Судья оценивал
    # текст ПЕРВОГО прогона (StrategyOutcome.text), поэтому и смотрим на него:
    # обвинить стратегию в неверном ответе, когда ответа не извлекли вовсе,
    # значит соврать прямо в кадре.
    if not top.first.check.has_marker:
        console.note(
            f"судья поставил на первое место {winner}, но в этом ответе не было строки "
            f"{ANSWER_MARKER} — сверять с эталоном нечего; судья оценивал рассуждение"
        )
    else:
        console.note(
            f"судья поставил на первое место {winner}, хотя по эталону этот ответ неверен — "
            "судья оценивает рассуждение, а не совпадение с ключом"
        )
