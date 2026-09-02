"""Day 03: стратегии рассуждения, сверка с эталоном, судья и таблица.

Клиент Mistral не мокается здесь вовсе — он даже не создаётся: `run_strategy()`
и `judge()` принимают функцию вызова параметром (`complete=`), и тесты подают
свою (`_Complete` ниже). Сеть не трогается ни разу, кредиты не тратятся.

`complete` передаётся явно, а не через monkeypatch модуля: значение по
умолчанию у этих функций связывается в момент импорта strategies, и подмена
`chat_core.complete` до него уже не дотягивается — та же грабля описана в
week_01/cli.py.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import jsonschema
import pytest
from rich.console import Console

from advent_core import console, formats
from advent_core.config import Config, ConfigError
from advent_core.errors import AdventError
from advent_core.params import GenerationParams
from advent_core.telemetry import CallResult, Usage
from week_01 import strategies

# --------------------------------------------------------------------------
# Инструменты: конфиг без сети, задача и запись вызовов
# --------------------------------------------------------------------------


def _config(**param_kwargs) -> Config:
    """Конфиг без system prompt: тогда весь system — это добавки стратегий."""
    return Config(
        api_key="k" * 32,
        model="mistral-small-latest",
        system_prompt_path=None,
        params=GenerationParams.build(**param_kwargs),
    )


PROBLEM = strategies.Problem(
    id="demo",
    title="Демонстрационная задача",
    statement="Сколько будет дважды два умножить на десять с половиной?",
    answer="42",
    accept=("сорок два", "42 штуки"),
)


@dataclass(slots=True)
class _Call:
    """Один перехваченный вызов: с каким конфигом и какими messages пришли."""

    config: Config
    messages: list[dict[str, str]]
    capabilities: dict | None

    @property
    def system(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "system"), "")

    @property
    def user(self) -> str:
        return self.messages[-1]["content"]

    @property
    def whole(self) -> str:
        """Весь запрос одной строкой — для проверок «этого там нет вообще»."""
        return "\n".join(m["content"] for m in self.messages)


_DEFAULT_ANSWER = "ОТВЕТ: 42"


class _Complete:
    """Замена chat.complete: отдаёт заготовленные тексты и копит вызовы.

    Кончились заготовки — отдаёт верный ответ по умолчанию, а не падает:
    число вызовов каждый тест проверяет явным assert'ом, и падение здесь
    маскировало бы его сообщением не о том.
    """

    def __init__(self, *texts: str, latency_ms: int = 100, usage: tuple[int, int] = (10, 5)):
        self.texts = list(texts)
        self.calls: list[_Call] = []
        self.latency_ms = latency_ms
        self.usage = usage

    def __call__(self, config, messages, capabilities=None) -> CallResult:
        self.calls.append(_Call(config, messages, capabilities))
        text = self.texts.pop(0) if self.texts else _DEFAULT_ANSWER
        prompt, completion = self.usage
        return CallResult(
            text=text,
            model_requested=config.model,
            model_actual="mistral-small-2603",
            usage=Usage(prompt, completion, prompt + completion),
            latency_ms=self.latency_ms,
            stream=False,
            finish_reason="stop",
            sent_messages=messages,
        )


def _silent(_message: str) -> None:
    """notify по умолчанию печатает в stderr; тестам это только мешает."""


def _run(strategy: str, complete: _Complete, *, problem=PROBLEM, config=None):
    return strategies.run_strategy(
        strategy,
        problem,
        config or _config(),
        complete=complete,
        notify=_silent,
    )


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    """Схлопывает перенос строки Rich в обычный пробел.

    console.note()/warn() печатают через Rich, а он переносит длинную строку
    по ширине консоли. Без схлопывания подстрочная проверка рвётся посреди
    честного и неизменного текста — то есть тест краснеет от ширины
    терминала, а не от поведения кода.
    """
    return re.sub(r"\s+", " ", _ANSI.sub("", text)).strip()


@pytest.fixture
def stdout_capture(monkeypatch):
    """Подменяет console.out широкой консолью в буфер.

    Rich по умолчанию режет таблицу под 80 колонок, и проверка «в таблице есть
    все столбцы» падала бы на переносах, а не на настоящем расхождении.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(console, "out", Console(file=buffer, width=200))
    return buffer


# --------------------------------------------------------------------------
# Извлечение маркера (SPEC-w01d03.md §6)
# --------------------------------------------------------------------------


def test_marker_takes_the_last_occurrence():
    """Модель охотно проговаривает формат по ходу — первое вхождение не то."""
    text = (
        "Сейчас посчитаю и в конце напишу ОТВЕТ: <краткий ответ>.\n"
        "Дважды два — четыре, умножаем на десять с половиной.\n"
        "ОТВЕТ: 42"
    )
    assert strategies.extract_answer(text) == "42"


def test_missing_marker_is_none_not_empty_string():
    """None — отдельное состояние: «формат не выполнен», а не «ответ пуст»."""
    assert strategies.extract_answer("Ответственность за подсчёт лежит на мне. Будет 42.") is None


def test_garbage_after_marker_is_cut_off():
    """Модель дописывает вежливость после маркера — в ответ она попасть не должна."""
    text = "Рассуждение.\n\nОТВЕТ: 42\n\nНадеюсь, я помог! Если что — спрашивай."
    assert strategies.extract_answer(text) == "42"


def test_markdown_around_marker_and_value_is_stripped():
    assert strategies.extract_answer("### **ОТВЕТ:** **42**") == "42"


def test_value_on_the_next_line_is_still_found():
    assert strategies.extract_answer("Итог.\nОТВЕТ:\n\n42") == "42"


def test_marker_without_value_is_empty_string_not_none():
    """Маркер был — значит формат выполнен; пустое значение это уже неверный ответ."""
    assert strategies.extract_answer("рассуждение\nОТВЕТ:") == ""


# --- НАХОДКА 1: проза со словом «ответ» после строки маркера ---


def test_prose_after_the_marker_line_does_not_hijack_the_answer():
    """Главный класс отказа дня: верный ответ получал ✗ из-за собственной прозы.

    Мягкая регулярка берёт ПОСЛЕДНЕЕ «ответ:» в любом регистре, а модель
    регулярно дописывает оговорку после строки маркера. Извлекалось «3» вместо
    «1» — вердикт по эталону становился ложным, то есть ломался вывод дня, а
    не оформление.
    """
    text = "Рассуждение.\nОТВЕТ: 1\n\nЕсли считать, что козу нельзя оставлять одну, то ответ: 3."
    assert strategies.extract_answer(text) == "1"


def test_prose_after_the_marker_keeps_the_verdict_correct():
    """Та же грабля на уровне вердикта: ✓ не должна превращаться в ✗."""
    text = "ОТВЕТ: 42\n\nПримечание: если считать иначе, ответ: 41."
    assert strategies.check_answer(text, PROBLEM).verdict == strategies.VERDICT_OK


def test_soft_marker_is_still_the_fallback_when_the_format_was_not_followed():
    """Строгий поиск не отменяет мягкий, а идёт первым.

    Модель, не выполнившая формат вовсе, всё равно должна быть разобрана: без
    запасного варианта такой ответ получал бы «нет маркера», хотя ответ в нём
    назван.
    """
    assert strategies.extract_answer("длинное рассуждение, и тогда ответ: 8") == "8"


def test_strict_marker_wins_over_a_later_soft_one_even_in_markdown():
    """Markdown-обвес вокруг маркера — норма, и он не должен ронять строгую ветку."""
    text = "### **ОТВЕТ:** 42\n\nи ещё раз повторю: ответ: 41"
    assert strategies.extract_answer(text) == "42"


def test_last_strict_marker_wins_over_earlier_strict_ones():
    """Внутри строгой ветки правило прежнее: последнее вхождение."""
    text = "ОТВЕТ: 41\nпересчитал\nОТВЕТ: 42"
    assert strategies.extract_answer(text) == "42"


# --- НАХОДКА 16: промпт маркера, константа и регулярка не должны разъехаться ---


def _marker_example_line() -> str:
    """Строка-образец из reason_marker.md — ровно то, что модель копирует.

    Берётся из файла промпта, а не из литерала в тесте: смысл проверки в том,
    чтобы правка формулировки в промпте не разошлась молча с ANSWER_MARKER и
    регуляркой извлечения.
    """
    for line in strategies.marker_instruction().splitlines():
        stripped = line.strip()
        if "<" in stripped and ">" in stripped and ":" in stripped:
            return stripped
    raise AssertionError("в reason_marker.md больше нет строки-образца вида «МАРКЕР: <ответ>»")


def test_marker_prompt_asks_for_exactly_the_constant():
    """Переименование маркера в промпте обязано ронять тест, а не демо.

    Промпты правятся руками между прогонами (SPEC §7). Заменить в
    reason_marker.md `ОТВЕТ:` на `ИТОГ:` — и модель послушно пишет `ИТОГ: 7`,
    extract_answer() возвращает None, а колонка «эталон» у всех четырёх
    стратегий показывает «нет маркера». Раньше этого не замечал ни один тест:
    один сравнивал промпт сам с собой, другой кормил парсер строкой, которую
    написал сам.
    """
    assert _marker_example_line().startswith(strategies.ANSWER_MARKER)


def test_extraction_parses_the_line_the_prompt_dictates():
    """Стык «что просим» → «что разбираем», собранный из файла промпта.

    Значение подставляется в образец из файла, а не пишется в тесте, поэтому
    сломать разбор правкой промпта незаметно нельзя.
    """
    filled = re.sub(r"<[^>]*>", "42", _marker_example_line())

    assert strategies._STRICT_MARKER_RE.search(filled), (
        "образец из промпта обязан попадать в СТРОГУЮ ветку: заглавные буквы в "
        "начале строки — то самое требование reason_marker.md, на котором она держится"
    )
    assert strategies.extract_answer(f"рассуждение\n{filled}") == "42"


def test_marker_is_never_used_as_a_stop_sequence():
    """CLAUDE.md: API вырезает stop из вывода — маркер в stop убил бы разбор.

    Проверка на константу, а не на поведение: строку легко «улучшить» до
    совпадения со стоп-последовательностью из демо, и тогда день молча
    перестанет извлекать ответы.
    """
    config = _config(stop="ОТВЕТ:")
    solving = strategies._solving_config(config)
    assert solving.params.stop is None
    assert config.params.stop == ["ОТВЕТ:"], "исходный config не должен мутироваться"


def test_unrelated_stop_sequence_survives():
    config = _config(stop="###")
    assert strategies._solving_config(config).params.stop == ["###"]


# --- НАХОДКА 14: снятие stop проверяется на прогоне, а не только на хелпере ---


def test_run_strategy_really_applies_the_safe_config_to_the_outgoing_call():
    """Хелпер мало проверить — важно, что прогон им пользуется.

    Прежде `_solving_config()` тестировался в одиночку: убрать его вызов из
    `run_strategy()` можно было, не покрасив ни одного теста. В живом запуске
    `advent w01 solve --stop 'ОТВЕТ:'` API вырезал бы маркер из ответа, и все
    четыре стратегии дали бы «нет маркера» — при полностью зелёных тестах.
    """
    complete = _Complete("ОТВЕТ: 42")
    _run("direct", complete, config=_config(stop="ОТВЕТ:"))

    assert complete.calls[0].config.params.stop is None


def test_every_call_of_a_multi_step_strategy_gets_the_safe_config():
    """panel — четыре вызова, и небезопасный stop не должен уцелеть ни в одном."""
    complete = _Complete(*PANEL_TEXTS)
    _run("panel", complete, config=_config(stop="ОТВЕТ:,###"))

    assert len(complete.calls) == 4
    for call in complete.calls:
        assert call.config.params.stop == ["###"], "чужой stop остаётся, маркерный снимается"


def test_solve_strips_the_unsafe_stop_and_warns_exactly_once(capsys):
    """Правка — на каждый вызов, предупреждение — одно на прогон.

    Двенадцать одинаковых строк в stderr (4 стратегии × 3 прогона) забивали
    экран ровно там, где на записи идёт ожидание ответа API.
    """
    complete = _Complete()
    strategies.solve(
        PROBLEM,
        _config(stop="ОТВЕТ:"),
        strategies=("direct", "steps"),
        runs=3,
        complete=complete,
        notify=_silent,
    )

    assert len(complete.calls) == 6
    assert all(call.config.params.stop is None for call in complete.calls)
    assert _flat(capsys.readouterr().err).count("пересекается с маркером") == 1


def test_json_format_is_warned_about_but_never_silently_overridden(capsys):
    """Вторая ветка `_solving_config()`: `--format json` вместе со стратегией.

    Ответ придёт JSON'ом, и строки маркера в нём может не быть — сказать об
    этом надо, а молча переписывать выбранный пользователем флаг нельзя.
    """
    complete = _Complete("ОТВЕТ: 42")
    _run("direct", complete, config=_config(format="json"))

    assert complete.calls[0].config.params.format == "json", "формат пользователя не подменяется"
    assert "format=json" in _flat(capsys.readouterr().err)


def test_warn_unsafe_params_false_silences_the_warning_but_not_the_fix(capsys):
    """Флаг гасит печать, а не саму правку — иначе вторая поверхность сломалась бы.

    `chat --strategy ...` зовёт `run_strategy()` напрямую; если бы снятие
    небезопасного stop зависело от того же флага, что и печать, маркер уехал
    бы в API и день молча перестал бы извлекать ответы.
    """
    complete = _Complete("ОТВЕТ: 42")
    strategies.run_strategy(
        "direct",
        PROBLEM,
        _config(stop="ОТВЕТ:", format="json"),
        complete=complete,
        notify=_silent,
        warn_unsafe_params=False,
    )

    assert complete.calls[0].config.params.stop is None
    assert _flat(capsys.readouterr().err) == ""


# --------------------------------------------------------------------------
# Нормализация и accept-синонимы (SPEC-w01d03.md §5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "42",
        " 42 ",
        "42.",
        "**42**",
        "«42»",
    ],
)
def test_answer_matches_reference_after_normalisation(raw):
    check = strategies.check_answer(f"рассуждение\nОТВЕТ: {raw}", PROBLEM)
    assert check.verdict == strategies.VERDICT_OK
    assert check.ok


@pytest.mark.parametrize("raw", ["сорок два", "Сорок Два", "сорок  два", "42 штуки"])
def test_accept_synonyms_count_as_correct(raw):
    assert strategies.check_answer(f"ОТВЕТ: {raw}", PROBLEM).ok


def test_yo_and_ye_are_the_same_letter():
    problem = replace(PROBLEM, answer="четыре сестры", accept=())
    assert strategies.check_answer("ОТВЕТ: Четыре сёстры.", problem).ok


def test_normalize_collapses_case_spaces_and_trailing_dot():
    assert strategies.normalize("  Сорок   Два.  ") == "сорок два"


def test_wrong_answer_is_wrong_not_missing_marker():
    check = strategies.check_answer("ОТВЕТ: 41", PROBLEM)
    assert check.verdict == strategies.VERDICT_WRONG
    assert check.has_marker is True
    assert check.answer == "41"


def test_missing_marker_is_its_own_verdict():
    """Три состояния, а не два: «нет маркера» не должно читаться как «неверно»."""
    check = strategies.check_answer("Ответ где-то здесь, но формат я не выполнил.", PROBLEM)
    assert check.verdict == strategies.VERDICT_NO_MARKER
    assert check.has_marker is False
    assert check.ok is False
    assert check.answer is None
    assert check.label != strategies.VERDICT_LABELS[strategies.VERDICT_WRONG]


# --------------------------------------------------------------------------
# Банк задач
# --------------------------------------------------------------------------


def _write_problem(directory: Path, **fields) -> None:
    payload = {"title": "t", "statement": "s", "answer": "a", **fields}
    (directory / f"{payload['id']}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_default_problem_is_the_flagged_one_not_the_alphabetically_first(tmp_path):
    """`"default": true` решает, какая задача поедет в демо, а не имя файла."""
    _write_problem(tmp_path, id="aaa")
    _write_problem(tmp_path, id="zzz", default=True)

    assert strategies.load_problem(directory=tmp_path).id == "zzz"


def test_default_falls_back_to_first_when_nothing_is_flagged(tmp_path):
    _write_problem(tmp_path, id="aaa")
    _write_problem(tmp_path, id="zzz")

    assert strategies.load_problem(directory=tmp_path).id == "aaa"


def test_two_default_flags_are_an_error_not_a_coin_toss(tmp_path):
    _write_problem(tmp_path, id="aaa", default=True)
    _write_problem(tmp_path, id="zzz", default=True)

    with pytest.raises(ConfigError):
        strategies.load_problem(directory=tmp_path)


def test_shipped_problem_bank_has_exactly_one_default():
    """Регрессия на банк: демо дня едет на задаче, помеченной явным флагом."""
    problems = strategies.load_problems()
    flagged = [problem.id for problem in problems.values() if problem.default]
    assert len(flagged) == 1, f"ровно один default ожидается, найдено: {flagged}"
    assert strategies.load_problem().id == flagged[0]


def test_default_problem_is_not_a_placeholder():
    """Выверен обязан быть эталон ЗАДАЧИ ДНЯ, а не всех задач банка.

    Проверка сознательно сужена. Прежняя версия требовала «ни одного
    placeholder во всём банке» и тем запрещала ровно тот сценарий, ради
    которого флаг заведён (docstring `Problem.placeholder`): свежий кандидат
    кладётся в банк ДО того, как эталон проверен живым прогоном. То есть тест
    краснел на неизменённом коде и толкал не ставить флаг вовсе — то есть к
    молчаливо неверному эталону. Значение имеет одно: вердикт ✓/✗ в кадре
    держится на эталоне той задачи, на которой едет демо.
    """
    assert not strategies.load_problem().placeholder


def test_a_fresh_candidate_may_stay_a_placeholder(tmp_path):
    """Флаг обязан оставаться рабочим состоянием, а не запрещённым."""
    _write_problem(tmp_path, id="socks", placeholder=True)
    _write_problem(tmp_path, id="alice", default=True)

    bank = strategies.load_problems(tmp_path)

    assert bank["socks"].placeholder is True
    assert strategies.load_problem(directory=tmp_path).id == "alice"


def test_placeholder_problem_says_so_out_loud(capsys):
    """Непроверенный эталон не должен выглядеть на экране как проверенный."""
    strategies.print_problem(replace(PROBLEM, placeholder=True))

    assert "placeholder" in _flat(capsys.readouterr().err)


def test_demo_scenario_problem_exists_in_the_shipped_bank():
    """Сценарий записи привязан к банку здесь, а не за минуту до записи.

    `advent record --day 3` подставляет `_PROBLEM_ID` в пять шагов и берёт
    оттуда же текст условия. Переименование или удаление файла банка — штатная
    операция (спека §5 прямо предполагает подбор задачи), и без этого теста она
    ломает сценарий ConfigError'ом в момент сборки шагов, то есть перед камерой.
    """
    from advent_cli import record

    problem = strategies.load_problem(record._PROBLEM_ID)
    assert problem.id == record._PROBLEM_ID
    # placeholder = «эталон не выверен живым прогоном»: снимать демо, где ключ
    # под вопросом, нельзя — вердикт ✗/✓ в кадре не будет ничего значить.
    assert not problem.placeholder


def test_unknown_problem_id_names_the_available_ones(tmp_path):
    _write_problem(tmp_path, id="aaa")
    with pytest.raises(ConfigError) as excinfo:
        strategies.load_problem("нет-такой", directory=tmp_path)
    assert "aaa" in str(excinfo.value)


def test_broken_problem_file_is_a_config_error_not_a_traceback(tmp_path):
    (tmp_path / "broken.json").write_text("{не json", encoding="utf-8")
    with pytest.raises(ConfigError):
        strategies.load_problems(tmp_path)


# --------------------------------------------------------------------------
# panel: четыре вызова, независимость экспертов, синтез видит всех
# --------------------------------------------------------------------------


PANEL_TEXTS = (
    "разбор аналитика ЛАМБДА\nОТВЕТ: 42",
    "выкладка инженера КАППА\nОТВЕТ: 41",
    "возражение критика ОМЕГА\nОТВЕТ: 42",
    "сведение трёх мнений\nОТВЕТ: 42",
)


def test_panel_makes_exactly_four_calls():
    complete = _Complete(*PANEL_TEXTS)
    run = _run("panel", complete)

    assert len(complete.calls) == 4
    assert [step.name for step in run.steps] == ["analyst", "engineer", "critic", "synthesis"]


def test_panel_experts_do_not_see_each_others_solutions():
    """Независимость буквальная: ни истории, ни чужих решений (SPEC §7)."""
    complete = _Complete(*PANEL_TEXTS)
    _run("panel", complete)

    analyst, engineer, critic = complete.calls[:3]
    for call in (analyst, engineer, critic):
        # Ровно system + условие: пары user/assistant из чужих ответов нет.
        assert [m["role"] for m in call.messages] == ["system", "user"]
        assert call.user == PROBLEM.statement

    # Проверяются только те пары «вызов ↔ маркер», где утечка вообще возможна:
    # ответ должен уже существовать к моменту вызова. Раньше в цикл входил и
    # аналитик — первый вызов, когда ни одного ответа ещё нет, — и три из
    # девяти проверок были истинны при ЛЮБОЙ реализации, включая заведомо
    # сломанную. Тест, который не может упасть, читается как проверенное.
    leakable = (
        (engineer, "ЛАМБДА"),  # аналитик уже ответил
        (critic, "ЛАМБДА"),
        (critic, "КАППА"),  # и инженер тоже
    )
    for call, marker in leakable:
        assert marker not in call.whole


def test_the_first_expert_is_asked_exactly_like_a_lone_call():
    """Аналитику утечь нечему — у него проверяется форма запроса, а не маркеры.

    Утверждаемое свойство именно это: первый эксперт получает такой же запрос,
    какой получил бы вызов, сделанный в одиночку, — system своей роли плюс
    условие, и ничего сверх того.
    """
    panel = _Complete(*PANEL_TEXTS)
    _run("panel", panel)
    alone = _Complete("ОТВЕТ: 42")
    _run("direct", alone)

    analyst = panel.calls[0]
    assert [m["role"] for m in analyst.messages] == [m["role"] for m in alone.calls[0].messages]
    assert analyst.user == alone.calls[0].user
    assert not any(m["role"] == "assistant" for m in analyst.messages)


def test_panel_experts_get_their_own_role_prompts():
    complete = _Complete(*PANEL_TEXTS)
    _run("panel", complete)

    for call, role in zip(complete.calls[:3], ("analyst", "engineer", "critic"), strict=True):
        assert strategies.load_prompt(f"panel_{role}") in call.system


def test_panel_synthesis_sees_all_three_solutions():
    complete = _Complete(*PANEL_TEXTS)
    _run("panel", complete)

    synthesis = complete.calls[3]
    assert strategies.load_prompt("panel_synthesis") in synthesis.system
    for marker in ("ЛАМБДА", "КАППА", "ОМЕГА"):
        assert marker in synthesis.user
    assert PROBLEM.statement in synthesis.user


# --- НАХОДКА 4: анонимность держится не только на метках A–D ---

# Слова, по которым судья опознал бы в «Ответе D» работу группы экспертов и
# получил бы ровно ту фору из-за названия, которую метки должны были убрать.
ROLE_WORDS = ("аналитик", "инженер", "критик", "эксперт", "панель")

# Ответы экспертов БЕЗ названий ролей внутри: PANEL_TEXTS их содержат («разбор
# аналитика»), и проверка «ролей во входе синтеза нет» ловила бы на них саму
# фикстуру вместо кода. Здесь проверяется ровно то, что дописывает код —
# подписи вокруг чужих решений.
NEUTRAL_PANEL_TEXTS = (
    "первое рассуждение ЛАМБДА\nОТВЕТ: 42",
    "второе рассуждение КАППА\nОТВЕТ: 41",
    "третье рассуждение ОМЕГА\nОТВЕТ: 42",
    "сведение\nОТВЕТ: 42",
)


def test_synthesis_input_never_names_the_roles():
    """Промпт бесполезен, если модель может процитировать роли прямо из входа.

    Было «Решение, которое дал критик:» — и синтез охотно начинал ответ с
    «аналитик и инженер сошлись…», а этот текст уходит судье как «Ответ D».
    Проверяется именно user-сообщение: в system роли есть и должны быть, там
    они стоят под запретом (см. соседний тест).
    """
    complete = _Complete(*NEUTRAL_PANEL_TEXTS)
    _run("panel", complete)

    user = complete.calls[3].user
    for marker in ("ЛАМБДА", "КАППА", "ОМЕГА"):
        assert marker in user, "решения экспертов синтез видеть обязан — иначе тест пустой"
    for word in ROLE_WORDS:
        assert word not in user.lower(), f"роль {word!r} утекла в user-сообщение синтеза"


def test_synthesis_prompt_forbids_naming_the_roles():
    """Вторая половина той же защиты — запрет живёт в промпте, а не в коде."""
    prompt = strategies.load_prompt("panel_synthesis").lower()

    for word in ("аналитик", "инженер", "критик"):
        assert word in prompt, "промпт обязан называть запрещённые слова явно"
    assert "от первого лица" in prompt


def test_panel_agreement_is_computed_in_code_not_asked_of_the_model(capsys):
    """Наблюдение «сошлись или разошлись» сохранено, но судье не показывается.

    Ради него куплена независимость экспертов (SPEC §7), поэтому терять его
    вместе с переписанным промптом синтеза было нельзя. Считается детерминированно
    по извлечённым ответам, печатается в stderr и не стоит ни одного вызова API.
    """
    complete = _Complete(*PANEL_TEXTS)
    _run("panel", complete)

    assert len(complete.calls) == 4, "на сравнение ответов лишний вызов не тратится"
    stderr = _flat(capsys.readouterr().err)
    assert "эксперты разошлись" in stderr, "42 / 41 / 42 — это расхождение"
    assert "«41»" in stderr


def test_panel_agreement_names_the_common_answer_when_experts_agree(capsys):
    complete = _Complete(
        "аналитик\nОТВЕТ: 42", "инженер\nОТВЕТ: 42", "критик\nОТВЕТ: 42", "синтез\nОТВЕТ: 42"
    )
    _run("panel", complete)

    assert "эксперты сошлись на «42»" in _flat(capsys.readouterr().err)


def test_panel_agreement_never_reaches_the_judge():
    """Сообщение о согласии экспертов идёт в stderr — в вызов судьи оно попасть не должно."""
    outcomes = _solve_all(_Complete(*ALL_TEXTS))
    complete = _Complete(JUDGE_JSON)
    strategies.judge(PROBLEM, outcomes, _config(), complete=complete, notify=_silent)

    whole = complete.calls[0].whole.lower()
    for word in ("эксперт", "сошлись", "разошлись", "панель"):
        assert word not in whole


def test_panel_answer_is_the_synthesis_not_the_experts():
    """Ответом стратегии считается последний вызов — иначе «панель» мерила бы аналитика."""
    complete = _Complete(*PANEL_TEXTS)
    run = _run("panel", complete)

    assert run.final.name == "synthesis"
    assert run.text == PANEL_TEXTS[3]
    assert run.check.ok


def test_every_solving_call_carries_the_same_marker_instruction():
    """Добавка одинаковая для всех — иначе сравнивались бы формулировки, а не стратегии."""
    complete = _Complete(*PANEL_TEXTS)
    _run("panel", complete)

    instruction = strategies.marker_instruction()
    assert all(instruction in call.system for call in complete.calls)


# --------------------------------------------------------------------------
# meta: два вызова, второй работает по тексту первого
# --------------------------------------------------------------------------


META_PROMPT = "Разбери условие по частям, выпиши все величины, потом проверь себя."
META_TEXTS = (META_PROMPT, "решаю по сгенерированному промпту\nОТВЕТ: 42")


def test_meta_makes_exactly_two_calls():
    complete = _Complete(*META_TEXTS)
    run = _run("meta", complete)

    assert len(complete.calls) == 2
    assert [step.name for step in run.steps] == ["prompt", "solve"]


def test_meta_second_call_uses_the_generated_prompt():
    complete = _Complete(*META_TEXTS)
    _run("meta", complete)

    writing, solving = complete.calls
    assert META_PROMPT not in writing.system, "первый вызов промпт ещё не видел"
    assert META_PROMPT in solving.system, "второй вызов обязан идти по сгенерированному промпту"
    assert solving.user == PROBLEM.statement


def test_meta_first_call_gets_no_marker_instruction():
    """На первом шаге модель пишет промпт, а не решает: требовать ОТВЕТ: там —
    прямо подтолкнуть её решить задачу, чего этот шаг как раз запрещает."""
    complete = _Complete(*META_TEXTS)
    _run("meta", complete)

    instruction = strategies.marker_instruction()
    assert instruction not in complete.calls[0].system
    assert strategies.load_prompt("meta") in complete.calls[0].system
    assert instruction in complete.calls[1].system


def test_meta_survives_an_empty_generated_prompt(capsys):
    """Пустой промпт не должен терять стратегию — второй вызов всё равно идёт."""
    complete = _Complete("   ", "ОТВЕТ: 42")
    run = _run("meta", complete)

    assert len(complete.calls) == 2
    assert run.check.ok
    assert "пустой промпт" in capsys.readouterr().err


# --- НАХОДКА 3: эталон, утёкший в сгенерированный промпт ---


META_LEAKY_PROMPT = (
    "Разбери условие по частям и выпиши все величины. Для самопроверки: если "
    "всё сделано верно, получится 42."
)


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("шаг 142 из инструкции", False),
        ("проверь себя 4 или 2 раза", False),
        ("если всё верно, получится 42", True),
        ("итог должен выйти сорок два", True),  # accept-форма считается так же
        ("", False),
    ],
)
def test_reference_is_searched_by_word_boundaries_not_by_substring(text, found):
    """Эталон «42» иначе находился бы в любом «шаг 142» — и оговорка стояла бы всегда."""
    assert strategies.mentions_expected_answer(text, PROBLEM) is found


def test_a_question_without_a_reference_can_never_match():
    """У вопроса из чата эталона нет — проверять нечего, и оговорки быть не должно."""
    question = strategies.question_as_problem("?")
    assert not strategies.mentions_expected_answer("что угодно 42", question)


def test_meta_flags_a_generated_prompt_that_already_holds_the_answer(capsys):
    """✓ у meta не должна быть неотличима от честной работы способа.

    Главный риск способа (SPEC §7, §13): первый вызов вопреки запрету решает
    задачу, эталон уезжает в system второго, тот его переписывает — и в таблице
    появляется ✓, по которой невозможно понять, сработал промпт или ответ был
    подарен. Вывод дня о точности способов при этом ложный.
    """
    complete = _Complete(META_LEAKY_PROMPT, "решаю по промпту\nОТВЕТ: 42")
    outcome = strategies.solve(
        PROBLEM, _config(), strategies=("meta",), complete=complete, notify=_silent
    )[0]

    assert outcome.first.check.ok, "второй вызов «угадал» — и именно это ничего не доказывает"
    assert outcome.caveats == ["эталон в промпте"]
    label = outcome.verdict_label
    assert label.endswith("· эталон в промпте"), "оговорка обязана дожить до таблицы"
    assert "эталонный ответ" in _flat(capsys.readouterr().err)


def test_meta_leaves_an_honest_run_unmarked(capsys):
    """Оговорка — не украшение: на честном прогоне её быть не должно."""
    outcome = strategies.solve(
        PROBLEM, _config(), strategies=("meta",), complete=_Complete(*META_TEXTS), notify=_silent
    )[0]

    assert outcome.caveats == []
    assert outcome.verdict_label == strategies.VERDICT_LABELS[strategies.VERDICT_OK]
    assert "эталонный ответ" not in _flat(capsys.readouterr().err)


def test_a_question_from_chat_never_triggers_the_reference_caveat():
    """У произвольного вопроса эталона нет — проверка утечки на нём выключена."""
    problem = strategies.question_as_problem("Сколько будет дважды два?")
    complete = _Complete("промпт, в котором зачем-то есть число 42", "ОТВЕТ: 4")

    run = _run("meta", complete, problem=problem)

    assert run.caveats == []


def test_direct_gets_nothing_but_the_marker_instruction():
    """direct — базовая линия сравнения: никаких добавок, кроме общей."""
    complete = _Complete("ОТВЕТ: 42")
    _run("direct", complete)

    call = complete.calls[0]
    assert call.system == strategies.marker_instruction()
    assert call.user == PROBLEM.statement


def test_steps_adds_the_step_by_step_prompt():
    complete = _Complete("шаг 1 … шаг 2 …\nОТВЕТ: 42")
    _run("steps", complete)

    assert strategies.load_prompt("steps") in complete.calls[0].system


def test_unknown_strategy_is_a_config_error():
    with pytest.raises(ConfigError):
        _run("телепатия", _Complete())


# --------------------------------------------------------------------------
# Судья (SPEC-w01d03.md §8)
# --------------------------------------------------------------------------


JUDGE_JSON = json.dumps(
    {
        "scores": [
            {"label": "A", "score": 3, "comment": "ответ без разбора условия"},
            {"label": "B", "score": 9, "comment": "полный разбор и проверка"},
            {"label": "C", "score": 7, "comment": "хороший промпт, решение короткое"},
            {"label": "D", "score": 8, "comment": "эксперты сошлись"},
        ],
        "ranking": ["B", "D", "C", "A"],
    },
    ensure_ascii=False,
)


def _outcomes(*names: str) -> list[strategies.StrategyOutcome]:
    """Готовые outcome'ы без вызовов API — судье нужен только текст ответа.

    Текст ответа намеренно не содержит имени стратегии: иначе проверка
    «судья названий не видит» проходила бы на подсказке, которую подложил
    сам тест.
    """
    built = []
    for index, name in enumerate(names, start=1):
        complete = _Complete(f"рассуждение номер {index}\nОТВЕТ: 42")
        built.append(strategies.StrategyOutcome(strategy=name, runs=[_run("direct", complete)]))
    return built


def test_judge_parses_ranking_and_scores():
    complete = _Complete(JUDGE_JSON)
    verdict = strategies.judge(
        PROBLEM, _outcomes(*strategies.STRATEGIES), _config(), complete=complete, notify=_silent
    )

    assert verdict.ok
    assert verdict.error is None
    assert verdict.labels == dict(zip(strategies.JUDGE_LABELS, strategies.STRATEGIES, strict=True))
    assert verdict.ranking == ["B", "D", "C", "A"]
    assert verdict.rank_of("steps") == 1
    assert verdict.rank_of("direct") == 4
    assert verdict.score_of("steps").score == 9
    assert verdict.cell("steps") == "#1 · 9/10"


def test_judge_is_a_single_call_for_all_strategies():
    complete = _Complete(JUDGE_JSON)
    strategies.judge(
        PROBLEM, _outcomes(*strategies.STRATEGIES), _config(), complete=complete, notify=_silent
    )

    assert len(complete.calls) == 1


def test_judge_sees_labels_but_never_strategy_names_or_the_key():
    """Названия дали бы «панели экспертов» фору, эталон превратил бы судью в сверку.

    Эталон здесь — строка, которой заведомо нет ни в условии, ни в ответах:
    искать «42» бессмысленно, потому что ответы его и содержат, а вопрос в
    том, подаёт ли judge() поле problem.answer отдельным блоком.
    """
    secret = replace(PROBLEM, answer="СЕКРЕТНЫЙ-ЭТАЛОН", accept=(), note="подсказка автора задачи")
    complete = _Complete(JUDGE_JSON)
    strategies.judge(
        secret, _outcomes(*strategies.STRATEGIES), _config(), complete=complete, notify=_silent
    )

    whole = complete.calls[0].whole
    for name in strategies.STRATEGIES:
        assert name not in whole
    assert secret.answer not in whole
    assert secret.note not in whole
    assert secret.statement in whole, "условие задачи судья видеть обязан"
    for label in strategies.JUDGE_LABELS:
        assert f"Ответ {label}:" in whole


def test_judge_asks_for_the_schema_without_touching_user_format():
    config = _config()
    complete = _Complete(JUDGE_JSON)
    strategies.judge(PROBLEM, _outcomes("direct"), config, complete=complete, notify=_silent)

    judge_params = complete.calls[0].config.params
    assert judge_params.format == "schema"
    assert judge_params.schema_file == str(strategies.JUDGE_SCHEMA_PATH)
    assert config.params.format is None, "формат пользователя менять нельзя"


def test_judge_model_overrides_the_solving_model():
    complete = _Complete(JUDGE_JSON)
    strategies.judge(
        PROBLEM,
        _outcomes("direct"),
        _config(),
        complete=complete,
        notify=_silent,
        model="ministral-8b-latest",
    )

    assert complete.calls[0].config.model == "ministral-8b-latest"


def test_judge_schema_file_is_loadable():
    """Схема едет в API штатным механизмом Day 02 — битый файл сорвал бы демо."""
    schema = formats.load_schema(str(strategies.JUDGE_SCHEMA_PATH))
    assert set(schema["properties"]) == {"scores", "ranking"}


def test_judge_fixture_validates_against_the_shipped_schema():
    """Фикстура тестов и файл схемы не должны разъехаться (SPEC §11).

    Все judge-тесты кормят парсер JSON'ом, написанным руками здесь же. Если
    во внутреннем объекте схемы переименовать `label`/`score`/`comment` или
    сделать `score` строкой, проверка «на верхнем уровне два свойства»
    останется истинной, тесты — зелёными, а живая модель вернёт
    схемо-валидный ответ с другими именами полей: `_parse_judge()` отбросит
    все метки и колонка «Судья» станет «—» прямо на записи. Прогон фикстуры
    через сам файл схемы связывает их намертво.
    """
    schema = formats.load_schema(str(strategies.JUDGE_SCHEMA_PATH))

    jsonschema.validate(json.loads(JUDGE_JSON), schema)


def test_the_parser_understands_exactly_what_the_schema_promises():
    """Вторая половина стыка: имена полей из схемы обязан читать `_parse_judge()`.

    Ответ собирается по описанию схемы (обязательные поля вложенного объекта),
    а не по памяти автора теста, — иначе схема и парсер снова расходятся молча.
    """
    schema = formats.load_schema(str(strategies.JUDGE_SCHEMA_PATH))
    item = schema["properties"]["scores"]["items"]
    assert item["required"] == ["label", "score", "comment"], "схема поменялась — проверь парсер"

    built = json.dumps(
        {
            "scores": [{"label": "A", "score": 6, "comment": "ровно по схеме"}],
            "ranking": ["A"],
        }
    )
    jsonschema.validate(json.loads(built), schema)

    verdict = strategies.judge(
        PROBLEM, _outcomes("direct"), _config(), complete=_Complete(built), notify=_silent
    )

    assert verdict.ok
    assert verdict.cell("direct") == "#1 · 6/10"


@pytest.mark.parametrize(
    "text",
    [
        "не json вовсе",
        "[1, 2, 3]",
        '{"ranking": ["X", "Y"], "scores": [{"label": "Z", "score": 5, "comment": ""}]}',
        '{"ranking": [], "scores": []}',
    ],
)
def test_invalid_judge_answer_does_not_break_the_run(text):
    """Судья не имеет права уронить прогон: ошибка живёт в verdict.error."""
    verdict = strategies.judge(
        PROBLEM, _outcomes("direct"), _config(), complete=_Complete(text), notify=_silent
    )

    assert verdict.ok is False
    assert verdict.error
    assert verdict.cell("direct") == "—"


def test_judge_api_failure_becomes_a_verdict_not_an_exception():
    def failing(config, messages, capabilities=None):
        raise AdventError("сервис недоступен")

    verdict = strategies.judge(
        PROBLEM, _outcomes("direct"), _config(), complete=failing, notify=_silent
    )

    assert verdict.ok is False
    assert "недоступен" in verdict.error
    assert verdict.result is None


def test_a_broken_judge_prompt_does_not_kill_the_run(monkeypatch):
    """Судья не имеет права уронить прогон ни по какой причине, а не только из-за API.

    Промпты правятся руками между прогонами (docstring `load_prompt`), и
    опечатка в имени файла поднимает ConfigError — а он не AdventError. Снаружи
    try он убивал бы `solve` после восьми уже оплаченных вызовов, не напечатав
    таблицу, то есть теряя итог дня целиком.
    """
    real = strategies.load_prompt

    def broken(name: str) -> str:
        if name == "judge":
            raise ConfigError("Промпт стратегии пуст: reason_judge.md")
        return real(name)

    monkeypatch.setattr(strategies, "load_prompt", broken)
    complete = _Complete(JUDGE_JSON)

    verdict = strategies.judge(
        PROBLEM, _outcomes("direct"), _config(), complete=complete, notify=_silent
    )

    assert verdict.ok is False
    assert "reason_judge.md" in verdict.error
    assert complete.calls == [], "до API дело не дошло — промпт собрать не удалось"
    assert verdict.cell("direct") == "—"


def test_judge_ignores_unknown_labels_but_keeps_known_ones():
    text = json.dumps(
        {
            "scores": [
                {"label": "A", "score": 6, "comment": "нормально"},
                {"label": "Z", "score": 10, "comment": "метки Z не существует"},
                {"label": "B", "score": "не число", "comment": "балл не разобрать"},
            ],
            "ranking": ["Z", "A", "A"],
        },
        ensure_ascii=False,
    )
    verdict = strategies.judge(
        PROBLEM,
        _outcomes("direct", "steps"),
        _config(),
        complete=_Complete(text),
        notify=_silent,
    )

    assert verdict.ok
    assert verdict.ranking == ["A"], "дубликаты схлопываются, чужие метки отбрасываются"
    assert verdict.score_of("steps") is None
    assert verdict.cell("direct") == "#1 · 6/10"


def test_judge_without_outcomes_reports_instead_of_calling_api():
    complete = _Complete(JUDGE_JSON)
    verdict = strategies.judge(PROBLEM, [], _config(), complete=complete, notify=_silent)

    assert verdict.ok is False
    assert complete.calls == []


# --------------------------------------------------------------------------
# --runs N: агрегация k/N, токены и время (SPEC-w01d03.md §9)
# --------------------------------------------------------------------------


def test_runs_repeats_the_strategy_and_aggregates_k_of_n():
    complete = _Complete("ОТВЕТ: 42", "ОТВЕТ: 41", "ОТВЕТ: 42")
    outcome = strategies.solve(
        PROBLEM,
        _config(),
        strategies=("direct",),
        runs=3,
        complete=complete,
        notify=_silent,
    )[0]

    assert len(complete.calls) == 3
    assert outcome.total == 3
    assert outcome.correct == 2
    assert outcome.no_marker == 0
    assert outcome.verdict_label == "2/3"


def test_runs_counts_missing_marker_separately_from_wrong():
    complete = _Complete("ОТВЕТ: 42", "формат не выполнен", "ОТВЕТ: 41")
    outcome = strategies.solve(
        PROBLEM, _config(), strategies=("direct",), runs=3, complete=complete, notify=_silent
    )[0]

    assert outcome.correct == 1
    assert outcome.no_marker == 1
    assert outcome.verdict_label == "1/3 · без маркера: 1"


def test_runs_sums_tokens_and_time_across_runs_and_calls():
    complete = _Complete(latency_ms=250, usage=(10, 5))
    outcome = strategies.solve(
        PROBLEM, _config(), strategies=("panel",), runs=2, complete=complete, notify=_silent
    )[0]

    totals = outcome.totals
    assert len(complete.calls) == 8, "четыре вызова панели × два прогона"
    assert totals.calls == 8
    assert totals.prompt_tokens == 80
    assert totals.completion_tokens == 40
    assert totals.total_tokens == 120
    assert totals.latency_ms == 2000
    assert totals.tokens_label() == "80/40"
    assert totals.time_label() == "2.0 s"


def test_missing_usage_is_reported_not_counted_as_free():
    """usage не пришёл — это «неизвестно», а не «ноль токенов»."""
    results = [
        CallResult(usage=Usage(10, 5, 15), latency_ms=100),
        CallResult(usage=Usage(), latency_ms=100),
    ]
    totals = strategies.Totals.of(results)

    assert totals.missing_usage == 1
    assert "без usage: 1" in totals.tokens_label()


@pytest.mark.parametrize(
    "partial",
    [
        Usage(None, 7, None),  # пришёл completion, не пришёл prompt
        Usage(7, None, None),  # и наоборот
    ],
)
def test_partial_usage_counts_as_missing_not_as_zero(partial):
    """Неизвестное слагаемое не должно показываться точным нулём.

    Раньше `is_empty()` смотрел только на prompt и total: вызов с
    `completion_tokens=7` и остальными None не попадал в `missing_usage`,
    неизвестный prompt суммировался как 0, и таблица печатала сумму без
    оговорки — ровно та ошибка, ради предотвращения которой missing_usage и
    заведён.
    """
    results = [
        CallResult(usage=Usage(10, 5, 15), latency_ms=100),
        CallResult(usage=partial, latency_ms=100),
    ]
    totals = strategies.Totals.of(results)

    assert partial.is_empty() is True
    assert totals.missing_usage == 1
    assert "без usage: 1" in totals.tokens_label()


def test_complete_usage_is_not_mistaken_for_a_partial_one():
    """Обратная сторона: полный usage обязан считаться, а не отбрасываться."""
    totals = strategies.Totals.of([CallResult(usage=Usage(10, 5, 15), latency_ms=100)])

    assert totals.missing_usage == 0
    assert totals.tokens_label() == "10/5"


def test_single_run_shows_a_symbol_not_a_fraction():
    outcome = strategies.solve(
        PROBLEM, _config(), strategies=("direct",), complete=_Complete("ОТВЕТ: 42"), notify=_silent
    )[0]

    assert outcome.verdict_label == strategies.VERDICT_LABELS[strategies.VERDICT_OK]


def test_answer_label_shows_the_spread_across_runs():
    complete = _Complete("ОТВЕТ: 42", "ОТВЕТ: 41")
    outcome = strategies.solve(
        PROBLEM, _config(), strategies=("direct",), runs=2, complete=complete, notify=_silent
    )[0]

    assert outcome.answer_label == "42 / 41"


def test_runs_below_one_is_rejected_before_any_call():
    complete = _Complete()
    with pytest.raises(ConfigError):
        strategies.solve(PROBLEM, _config(), runs=0, complete=complete, notify=_silent)
    assert complete.calls == []


# --------------------------------------------------------------------------
# --strategy all: порядок и полнота таблицы
# --------------------------------------------------------------------------


ALL_TEXTS = (
    "прямо и мимо\nОТВЕТ: 41",  # direct
    "по шагам\nОТВЕТ: 42",  # steps
    "текст сгенерированного промпта",  # meta, вызов 1
    "решение по нему\nОТВЕТ: 42",  # meta, вызов 2
    "аналитик\nОТВЕТ: 42",  # panel ×4
    "инженер\nОТВЕТ: 42",
    "критик\nОТВЕТ: 41",
    "синтез\nОТВЕТ: 42",
)


def _solve_all(complete: _Complete) -> list[strategies.StrategyOutcome]:
    return strategies.solve(
        PROBLEM,
        _config(),
        strategies=strategies.STRATEGIES,
        complete=complete,
        notify=_silent,
    )


def test_all_runs_every_strategy_in_a_fixed_order():
    """Порядок задаёт и таблицу, и метки судьи A–D — сортировать его нельзя."""
    complete = _Complete(*ALL_TEXTS)
    outcomes = _solve_all(complete)

    assert [outcome.strategy for outcome in outcomes] == ["direct", "steps", "meta", "panel"]
    assert list(strategies.STRATEGIES) == ["direct", "steps", "meta", "panel"]


def test_all_costs_exactly_eight_calls():
    """Бюджет дня: 1 + 1 + 2 + 4 (SPEC-w01d03.md §2)."""
    complete = _Complete(*ALL_TEXTS)
    outcomes = _solve_all(complete)

    assert len(complete.calls) == 8
    assert [outcome.totals.calls for outcome in outcomes] == [1, 1, 2, 4]


def test_all_verdicts_are_per_strategy():
    outcomes = _solve_all(_Complete(*ALL_TEXTS))
    verdicts = {outcome.strategy: outcome.first.verdict for outcome in outcomes}

    assert verdicts["direct"] == strategies.VERDICT_WRONG
    assert verdicts["steps"] == strategies.VERDICT_OK
    assert verdicts["meta"] == strategies.VERDICT_OK
    assert verdicts["panel"] == strategies.VERDICT_OK


def test_comparison_table_has_every_column_and_every_strategy(stdout_capture):
    outcomes = _solve_all(_Complete(*ALL_TEXTS))
    verdict = strategies.judge(
        PROBLEM, outcomes, _config(), complete=_Complete(JUDGE_JSON), notify=_silent
    )

    strategies.print_comparison(PROBLEM, outcomes, verdict)
    printed = stdout_capture.getvalue()

    for column in ("стратегия", "ответ", "эталон", "судья", "вызовов", "токены", "время"):
        assert column in printed
    for name in strategies.STRATEGIES:
        assert name in printed
    assert "#1 · 9/10" in printed, "колонка судьи заполнена местом и баллом"
    assert "ранжирование судьи" in printed


def test_comparison_table_without_judge_leaves_a_dash(stdout_capture):
    outcomes = _solve_all(_Complete(*ALL_TEXTS))

    strategies.print_comparison(PROBLEM, outcomes, None)
    printed = stdout_capture.getvalue()

    for name in strategies.STRATEGIES:
        assert name in printed
    assert "ранжирование судьи" not in printed


def test_failed_judge_keeps_the_table_and_warns_in_stderr(stdout_capture, capsys):
    """Судья не отработал — таблица в stdout остаётся чистой, ошибка уходит в stderr."""
    outcomes = _solve_all(_Complete(*ALL_TEXTS))
    verdict = strategies.judge(
        PROBLEM, outcomes, _config(), complete=_Complete("не json"), notify=_silent
    )

    strategies.print_comparison(PROBLEM, outcomes, verdict)

    printed = stdout_capture.getvalue()
    assert "direct" in printed
    assert "судья не отработал" in capsys.readouterr().err


# --- НАХОДКА 17: расхождение судьи с эталоном — результат дня, а не сбой ---


def _judge_json(*ranking: str) -> str:
    """Ответ судьи с заданным порядком меток — по одному баллу на метку."""
    return json.dumps(
        {
            "scores": [
                {"label": label, "score": 10 - index, "comment": f"комментарий про {label}"}
                for index, label in enumerate(ranking)
            ],
            "ranking": list(ranking),
        },
        ensure_ascii=False,
    )


def test_judge_disagreeing_with_the_reference_is_explained_not_hidden(stdout_capture, capsys):
    """Заявленный результат дня §8, который до сих пор не выполнялся ни разу.

    В прежних тестах судья всегда ставил первой стратегию с верным ответом,
    поэтому ветка пояснения не исполнялась вовсе. Здесь первой стоит метка A —
    это `direct`, который по ALL_TEXTS отвечает «41» при эталоне «42».
    """
    outcomes = _solve_all(_Complete(*ALL_TEXTS))
    verdict = strategies.judge(
        PROBLEM,
        outcomes,
        _config(),
        complete=_Complete(_judge_json("A", "B", "C", "D")),
        notify=_silent,
    )
    assert verdict.labels["A"] == "direct"

    strategies.print_comparison(PROBLEM, outcomes, verdict)

    stderr = _flat(capsys.readouterr().err)
    assert "судья поставил на первое место direct" in stderr
    assert "по эталону этот ответ неверен" in stderr
    # Пояснение — в stderr, таблица в stdout остаётся итогом дня без пометок
    # об ошибке: расхождение это результат, а не сбой программы.
    printed = stdout_capture.getvalue()
    assert "по эталону этот ответ неверен" not in printed
    assert "ранжирование судьи" in printed


def test_agreeing_judge_says_nothing_extra(stdout_capture, capsys):
    """Ветка обязана уметь молчать — иначе предыдущий тест ловил бы что угодно."""
    outcomes = _solve_all(_Complete(*ALL_TEXTS))
    verdict = strategies.judge(
        PROBLEM,
        outcomes,
        _config(),
        complete=_Complete(_judge_json("B", "A", "C", "D")),  # B — steps, ответ верный
        notify=_silent,
    )

    strategies.print_comparison(PROBLEM, outcomes, verdict)

    assert "на первое место" not in _flat(capsys.readouterr().err)


def test_a_top_answer_without_a_marker_is_not_called_wrong(stdout_capture, capsys):
    """«Маркера не было» и «ответ неверен» — разные состояния (§6).

    Судья видел текст первого прогона; если в нём не было строки ОТВЕТ:,
    сверять с эталоном нечего. Назвать это «неверным по эталону» — ложное
    обвинение стратегии прямо в кадре.
    """
    texts = ("формат я не выполнил", *ALL_TEXTS[1:])
    outcomes = _solve_all(_Complete(*texts))
    assert not outcomes[0].first.check.has_marker

    verdict = strategies.judge(
        PROBLEM,
        outcomes,
        _config(),
        complete=_Complete(_judge_json("A", "B", "C", "D")),
        notify=_silent,
    )
    strategies.print_comparison(PROBLEM, outcomes, verdict)

    stderr = _flat(capsys.readouterr().err)
    assert f"не было строки {strategies.ANSWER_MARKER}" in stderr
    assert "по эталону этот ответ неверен" not in stderr


def test_all_table_reports_the_number_of_runs(stdout_capture):
    outcomes = strategies.solve(
        PROBLEM,
        _config(),
        strategies=("direct", "steps"),
        runs=2,
        complete=_Complete(),
        notify=_silent,
    )

    strategies.print_comparison(PROBLEM, outcomes, None)

    assert "прогонов на стратегию: 2" in stdout_capture.getvalue()


# --------------------------------------------------------------------------
# Журнал: метки вызовов
# --------------------------------------------------------------------------


def test_log_calls_labels_every_call_with_strategy_step_and_run(monkeypatch):
    """Без меток строки журнала неразличимы — а по ним неделя 2 будет считать токены."""
    records: list[dict] = []
    monkeypatch.setattr(
        strategies,
        "log_call",
        lambda result, messages, **kwargs: records.append(kwargs),
    )

    outcomes = strategies.solve(
        PROBLEM,
        _config(),
        strategies=("meta",),
        runs=2,
        complete=_Complete(),
        notify=_silent,
    )
    strategies.log_calls(outcomes, week=1, day=3)

    assert len(records) == 4  # два вызова meta × два прогона
    assert [r["extra"]["step"] for r in records] == ["prompt", "solve", "prompt", "solve"]
    assert [r["extra"]["run"] for r in records] == [1, 1, 2, 2]
    assert {r["extra"]["strategy"] for r in records} == {"meta"}
    assert {r["extra"]["problem"] for r in records} == {PROBLEM.id}
    assert {r["day"] for r in records} == {3}


def test_on_step_fires_per_call_so_a_mid_strategy_failure_keeps_paid_calls(monkeypatch):
    """Четвёртый вызов панели упал — три оплаченных обязаны остаться в журнале.

    Именно ради этого запись пошаговая, а не пострановая: журнал — единственное,
    что этот день сохраняет, а `panel` — четыре последовательных вызова, где
    429 на любом из них раньше уносил все предыдущие.
    """
    records: list[dict] = []
    monkeypatch.setattr(
        strategies,
        "log_call",
        lambda result, messages, **kwargs: records.append(kwargs),
    )

    class _FailsOnSynthesis(_Complete):
        def __call__(self, config, messages, capabilities=None) -> CallResult:
            if len(self.calls) == 3:
                raise AdventError("rate limit")
            return super().__call__(config, messages, capabilities)

    complete = _FailsOnSynthesis()
    with pytest.raises(AdventError):
        strategies.run_strategy(
            "panel",
            PROBLEM,
            _config(),
            complete=complete,
            notify=_silent,
            on_step=lambda step: strategies.log_step(step, week=1, day=3, problem=PROBLEM.id),
        )

    assert [r["extra"]["step"] for r in records] == ["analyst", "engineer", "critic"]
    assert {r["extra"]["strategy"] for r in records} == {"panel"}


def test_solve_hooks_carry_the_run_number_into_every_step(monkeypatch):
    """`--runs N`: номер прогона доезжает до строки журнала через сам шаг."""
    seen: list[tuple[str, str, int]] = []

    strategies.solve(
        PROBLEM,
        _config(),
        strategies=("meta",),
        runs=2,
        complete=_Complete(),
        notify=_silent,
        on_step=lambda step: seen.append((step.strategy, step.name, step.run)),
    )

    assert seen == [
        ("meta", "prompt", 1),
        ("meta", "solve", 1),
        ("meta", "prompt", 2),
        ("meta", "solve", 2),
    ]


def test_log_judge_marks_the_call_as_judge(monkeypatch):
    records: list[dict] = []
    monkeypatch.setattr(
        strategies,
        "log_call",
        lambda result, messages, **kwargs: records.append(kwargs),
    )

    verdict = strategies.judge(
        PROBLEM, _outcomes("direct"), _config(), complete=_Complete(JUDGE_JSON), notify=_silent
    )
    strategies.log_judge(verdict, week=1, day=3, problem=PROBLEM.id)

    assert len(records) == 1
    assert records[0]["extra"]["strategy"] == "judge"
    assert records[0]["error"] is None


def test_log_judge_writes_nothing_when_the_call_never_happened(monkeypatch):
    """Вызов не состоялся — придумывать за judge() список сообщений хуже, чем промолчать."""
    records: list[dict] = []
    monkeypatch.setattr(
        strategies, "log_call", lambda result, messages, **kwargs: records.append(kwargs)
    )

    strategies.log_judge(strategies.JudgeVerdict(error="упал"), week=1, day=3)

    assert records == []


# --------------------------------------------------------------------------
# Вторая поверхность: chat --strategy
# --------------------------------------------------------------------------


def test_question_from_chat_has_no_reference_answer():
    """У произвольного вопроса эталона нет — вердикт печатать нельзя."""
    problem = strategies.question_as_problem("Как дела?")

    assert problem.answer == ""
    assert problem.id == strategies.CHAT_PROBLEM_ID


def test_print_step_prints_answers_to_stdout_and_headers_to_stderr(capsys):
    run = _run("meta", _Complete(*META_TEXTS))

    for step in run.steps:
        strategies.print_step(step)
    captured = capsys.readouterr()

    assert META_PROMPT in captured.out, "ответ модели — в stdout"
    assert "модель пишет промпт" in captured.err, "заголовки и телеметрия — в stderr"


def test_print_verdict_names_the_missing_marker_explicitly(capsys):
    run = _run("direct", _Complete("формат не выполнен"))

    strategies.print_verdict(run)

    assert strategies.ANSWER_MARKER in capsys.readouterr().err


def test_extracted_value_is_escaped_before_it_reaches_rich(capsys):
    """Ответ пришёл от модели, и Rich съел бы в нём квадратные скобки как тег.

    Пользователь видел бы вердикт ✗ и при этом не видел, что именно было
    извлечено, — то есть самое нужное на экране место молча пустело.
    """
    run = _run("direct", _Complete("ОТВЕТ: [b]7[/b]"))

    strategies.print_verdict(run)

    assert "[b]7[/b]" in _flat(capsys.readouterr().err)


# --------------------------------------------------------------------------
# Демо-сценарий
# --------------------------------------------------------------------------


def test_demo_caption_keeps_square_brackets_from_the_problem_statement(monkeypatch, stdout_capture):
    """С дня 03 в подпись шага подставляется условие из банка — его пишет человек.

    Квадратные скобки в условии («последовательности [a, b, c]») Rich съел бы
    как незакрытый тег, и кусок подписи пропал бы прямо в кадре, без единой
    ошибки и без единого красного теста.
    """
    from advent_cli import record

    monkeypatch.setattr(record, "_run_step", lambda step: None)
    monkeypatch.setattr(record.time, "sleep", lambda seconds: None)
    step = record.Step(title="шаг", args=["w01", "chat", "Сколько всего [a, b, c] вариантов?"])

    record._play([step])

    assert "[a, b, c]" in _flat(stdout_capture.getvalue())
