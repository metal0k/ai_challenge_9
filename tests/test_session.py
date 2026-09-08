"""Именованные сессии: имя, атомарная запись, битый файл, usage без нулей.

Сети здесь нет по построению — модуль работает с файлами и словарями.
"""

from __future__ import annotations

import json
import os

import pytest

from advent_core.config import ConfigError
from advent_core.session import (
    SESSION_VERSION,
    Session,
    list_sessions,
    validate_name,
)
from advent_core.telemetry import Usage


def _session(tmp_path, name="default"):
    return Session.new(name, directory=tmp_path)


# --- имя сессии -----------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["../secrets", "..", "a/b", "a\\b", "C:/tmp/x", "", "   ", "."],
)
def test_traversal_and_empty_names_are_rejected(name):
    # Имя сессии — часть пути к файлу, поэтому "../" в нём недопустим.
    with pytest.raises(ConfigError):
        validate_name(name)


def test_normal_names_are_accepted():
    assert validate_name(" рецепты ") == "рецепты"
    assert validate_name("day-06_демо") == "day-06_демо"


def test_path_for_refuses_traversal(tmp_path):
    with pytest.raises(ConfigError):
        Session.path_for("../evil", tmp_path)


# --- запись и чтение ------------------------------------------------------


def test_roundtrip_keeps_turns_model_and_usage(tmp_path):
    session = _session(tmp_path)
    session.record(
        "привет",
        "здравствуй",
        model="ministral-14b-latest",
        usage=Usage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )
    session.save()

    loaded = Session.load("default", directory=tmp_path)
    assert loaded.warnings == []
    assert [t.role for t in loaded.turns] == ["user", "assistant"]
    assert loaded.turns[1].usage.total_tokens == 13
    # model пишется на КАЖДЫЙ ход, а не на сессию целиком (SPEC §11).
    assert all(turn.model == "ministral-14b-latest" for turn in loaded.turns)
    assert loaded.last_model() == "ministral-14b-latest"


def test_file_shape_matches_spec(tmp_path):
    session = _session(tmp_path, "демо")
    session.record("вопрос", "ответ", model="m", usage=Usage(1, 2, 3))
    path = session.save()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == SESSION_VERSION
    assert raw["name"] == "демо"
    assert raw["created"] == session.created
    assert set(raw["turns"][0]) == {"role", "content", "ts", "model"}
    assert set(raw["turns"][1]) == {"role", "content", "ts", "model", "usage"}
    assert set(raw["turns"][1]["usage"]) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }


def test_missing_usage_never_becomes_zero(tmp_path):
    session = _session(tmp_path)
    session.record("вопрос", "ответ", model="m", usage=None)
    path = session.save()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "usage" not in raw["turns"][1]

    loaded = Session.load("default", directory=tmp_path)
    assert loaded.turns[1].usage is None
    # Сумма по сессии без единого usage — «неизвестно», а не ноль.
    assert loaded.token_total() is None
    assert loaded.missing_usage() == 1


def test_partial_usage_is_kept_as_none_not_filled_in(tmp_path):
    session = _session(tmp_path)
    session.record("в", "о", model="m", usage=Usage(prompt_tokens=5))
    session.save()

    loaded = Session.load("default", directory=tmp_path)
    usage = loaded.turns[1].usage
    assert usage.prompt_tokens == 5
    assert usage.completion_tokens is None
    assert usage.total_tokens is None
    # Частичный usage считается отсутствующим (Usage.is_empty) — иначе сумма
    # молча занизится и покажется точной.
    assert loaded.token_total() is None


def test_secrets_are_redacted_on_write(tmp_path, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "supersecretkey1234")
    session = _session(tmp_path)
    session.record("ключ supersecretkey1234", "ответ supersecretkey1234", model="m")
    path = session.save()

    text = path.read_text(encoding="utf-8")
    assert "supersecretkey1234" not in text
    assert "***" in text


# --- атомарность ----------------------------------------------------------


def test_save_replaces_atomically_from_the_same_directory(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.record("в", "о", model="m")
    seen = {}
    real_replace = os.replace

    def _spy(src, dst):
        seen["src"] = str(src)
        seen["dst"] = str(dst)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _spy)
    session.save()

    # Временный файл лежит РЯДОМ: os.replace атомарен только внутри одной
    # файловой системы, через границу он деградирует до копирования.
    assert os.path.dirname(seen["src"]) == os.path.dirname(seen["dst"])
    assert seen["dst"] == str(session.path)


def test_failed_save_leaves_old_file_intact_and_no_leftovers(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.record("первый", "ответ", model="m")
    session.save()
    before = session.path.read_text(encoding="utf-8")

    session.record("второй", "ответ", model="m")

    def _boom(src, dst):
        raise OSError("диск кончился")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        session.save()

    assert session.path.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("*.tmp")) == []


# --- битый файл -----------------------------------------------------------


def test_broken_file_gives_warning_and_empty_session(tmp_path):
    path = tmp_path / "default.json"
    path.write_text("{это не json", encoding="utf-8")

    session = Session.load("default", directory=tmp_path)
    assert session.turns == []
    assert session.warnings and "default" in session.warnings[0]
    # Испорченный файл не затирается молча — его можно починить руками.
    assert list(tmp_path.glob("*.bak"))


def test_unknown_version_is_not_guessed(tmp_path):
    path = tmp_path / "default.json"
    path.write_text(
        json.dumps({"version": 99, "name": "default", "created": "x", "turns": []}),
        encoding="utf-8",
    )
    session = Session.load("default", directory=tmp_path)
    assert session.turns == []
    assert any("99" in w for w in session.warnings)


def test_broken_turn_entries_are_skipped_with_warning(tmp_path):
    path = tmp_path / "default.json"
    path.write_text(
        json.dumps(
            {
                "version": SESSION_VERSION,
                "name": "default",
                "created": "2026-09-07T14:00:00+03:00",
                "turns": [
                    {"role": "user", "content": "цел", "ts": "t"},
                    {"role": "нечто", "content": "битый"},
                    "вообще не объект",
                ],
            }
        ),
        encoding="utf-8",
    )
    session = Session.load("default", directory=tmp_path)
    # Структура файла цела — уцелевший разговор не выбрасывается из-за двух
    # испорченных записей, но об их числе говорится вслух.
    assert len(session.turns) == 1
    assert any("2" in w for w in session.warnings)


def test_missing_file_is_just_an_empty_session(tmp_path):
    session = Session.load("нету", directory=tmp_path)
    assert session.turns == []
    assert session.warnings == []


# --- история и список -----------------------------------------------------


def test_history_returns_plain_messages_for_the_agent(tmp_path):
    session = _session(tmp_path)
    session.record("вопрос", "ответ", model="m", usage=Usage(1, 2, 3))
    history = session.history()
    assert history == [
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"},
    ]
    # Копия, а не ссылка: правка сообщения не должна менять файл сессии.
    history[0]["content"] = "подмена"
    assert session.turns[0].content == "вопрос"


def test_clear_forgets_turns_but_keeps_the_name(tmp_path):
    session = _session(tmp_path, "тема")
    session.record("в", "о", model="m")
    session.clear()
    assert session.turns == []
    assert session.name == "тема"


def test_list_sessions_reports_name_date_turns_and_tokens(tmp_path):
    first = _session(tmp_path, "первая")
    first.record("в", "о", model="m", usage=Usage(10, 5, 15))
    first.save()
    second = _session(tmp_path, "вторая")
    second.record("в", "о", model="m", usage=None)
    second.save()

    infos, warnings = list_sessions(tmp_path)
    assert warnings == []
    by_name = {info.name: info for info in infos}
    assert by_name["первая"].turns == 2
    assert by_name["первая"].tokens == 15
    assert by_name["первая"].created
    # Сессия без единого usage показывает «неизвестно», а не ноль токенов.
    assert by_name["вторая"].tokens is None
    assert by_name["вторая"].missing_usage == 1


def test_list_sessions_does_not_rename_anything_on_disk(tmp_path):
    """`/sessions` — команда «покажи, что у меня есть», а не «переложи файлы».

    Пока карантин делался и при листинге, первый вызов уносил битую сессию в
    .bak, а второй не показывал ни строки, ни предупреждения — запись
    пропадала бесследно, и пользователь решал, что разговора не было.
    """
    (tmp_path / "битая.json").write_text(
        json.dumps({"version": 0, "name": "битая", "turns": []}), encoding="utf-8"
    )
    _session(tmp_path, "живая").save()

    first_infos, first_warnings = list_sessions(tmp_path)
    second_infos, second_warnings = list_sessions(tmp_path)

    assert (tmp_path / "битая.json").is_file(), "листинг переименовал файл сессии"
    assert list(tmp_path.glob("*.bak")) == []
    # Множествами, а не списками: порядок тут ни при чём, а дата начала у
    # нечитаемой сессии берётся из момента чтения — два вызова подряд легко
    # оказываются по разные стороны секунды и переставляют строки местами.
    assert {info.name for info in first_infos} == {info.name for info in second_infos}
    assert "битая" in {info.name for info in second_infos}
    assert first_warnings and second_warnings, "во второй раз о проблеме промолчали"
    # Нечитаемая сессия не притворяется пустой: «ходов 0» — это утверждение о
    # разговоре, которого мы не видели.
    assert {info.name: info.broken for info in second_infos} == {"битая": True, "живая": False}


def test_opening_a_session_for_writing_still_quarantines_the_broken_file(tmp_path):
    """Карантин никуда не делся — он переехал туда, где файл всё равно будет
    перезаписан следующим save()."""
    (tmp_path / "default.json").write_text("{это не json", encoding="utf-8")

    session = Session.load("default", directory=tmp_path)

    assert session.broken is True
    assert list(tmp_path.glob("*.bak"))


def test_list_sessions_on_empty_directory(tmp_path):
    infos, warnings = list_sessions(tmp_path / "нет-такой")
    assert infos == []
    assert warnings == []
