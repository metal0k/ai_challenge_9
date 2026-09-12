"""Именованные сессии агента: JSON-файл на сессию, атомарная запись, ходы.

Почему JSON, а не SQLite: файл читается глазами, показывается на камере и
чинится руками. SQLite надёжнее при нескольких процессах, но пользователь у
нас один, а содержимое сессии — предмет демонстрации (SPEC-w02d06.md §6).

Почему запись атомарная: прямой прецедент — goose мигрировал с JSONL на
SQLite именно из-за порчи сессий (issue #3200, зависание при чтении битого
файла). Мы берём из этой истории не миграцию, а вывод: писать во временный
файл рядом и заменять целиком, а на битом файле — предупреждать и начинать
пустым, а не падать (тот же фикс-паттерн у shell_gpt #786 и gptme #3498).

Почему НЕ в temp: shell_gpt по умолчанию держит сессии в
`gettempdir()/chat_cache`, где их тихо стирает уборка временных файлов.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from advent_core.chat import Message
from advent_core.config import LOG_DIR, ConfigError, redact
from advent_core.telemetry import Usage

SESSION_VERSION = 1
SESSIONS_DIR = LOG_DIR / "sessions"
DEFAULT_SESSION = "default"

# Имя сессии — часть пути к файлу, поэтому проверяется, а не подставляется как
# есть. \w покрывает и кириллицу (флага re.ASCII нет намеренно: `/session
# рецепты` на видео читается), и при этом не пропускает ни "/", ни "\", ни
# "..", ни двоеточие диска — то есть ни одной формы выхода из каталога сессий.
# "--" passes it too (two ASCII hyphens are just two of the allowed chars) —
# SPEC-w02d10.md §7.2 leans on exactly that instead of bumping _NAME_RE.
_NAME_RE = re.compile(r"^[\w-]{1,64}$")

# Branch/checkpoint naming (SPEC-w02d10.md §7). A branch file is
# "<parent>--<name>", a checkpoint is "<parent>--cp-<name>" — the prefix is
# what a name-only parser (build_tree below) tells them apart by, with no
# file read needed.
BRANCH_SEP = "--"
CHECKPOINT_PREFIX = "cp-"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# state keys that are conversation CONTENT, not a run setting. clear() erases
# them along with the turns — unlike mode/done/context_limit, which survive
# `/new` on purpose (SPEC-w02d09.md §8). The list lives here, next to clear(),
# because "setting or content" is a fact about storage, not about the
# interface — a second list in the CLI would drift on the first new key.
CONTENT_STATE_KEYS = ("summary", "summary_upto", "facts", "facts_pinned", "facts_upto")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def validate_name(name: str) -> str:
    """Проверяет имя сессии и возвращает его. ConfigError на негодном.

    Отдельная функция, а не проверка внутри load(): CLI обязана отбить
    `--session ../../.env` до того, как что-то будет прочитано или записано.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise ConfigError("Имя сессии не может быть пустым")
    if not _NAME_RE.match(cleaned):
        raise ConfigError(
            f"Недопустимое имя сессии {name!r}: разрешены буквы, цифры, дефис и "
            "подчёркивание, до 64 символов (имя становится именем файла)"
        )
    return cleaned


def _validate_segment(name: str, *, kind: str) -> str:
    """A branch/checkpoint's OWN segment, before it is joined onto a parent.

    Separate from validate_name(): the combined "<parent>--<segment>" is
    checked there (length, charset), but "must not start with cp-" and "must
    not itself contain --" are rules about the segment alone, and only make
    sense before joining — after joining, both fold indistinguishably into
    the combined string.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise ConfigError(f"имя {kind} не может быть пустым")
    if BRANCH_SEP in cleaned:
        raise ConfigError(
            f"имя {kind} {name!r} не может содержать {BRANCH_SEP!r} — это "
            "разделитель дерева сессий, а не обычный дефис"
        )
    if kind == "ветки" and cleaned.startswith(CHECKPOINT_PREFIX):
        raise ConfigError(
            f"имя ветки не может начинаться с {CHECKPOINT_PREFIX!r} — так "
            "помечаются checkpoint'ы, и след пропал бы неразличимо"
        )
    return cleaned


def branch_file_name(parent: str, name: str) -> str:
    """File-name (without .json) for a new branch of `parent`.

    Length/charset of the COMBINED name are checked by validate_name() itself
    — an over-long "<parent>--<name>" raises ConfigError with the same
    readable message as any other bad session name, never a silent truncation.
    """
    parent = validate_name(parent)
    segment = _validate_segment(name, kind="ветки")
    return validate_name(f"{parent}{BRANCH_SEP}{segment}")


def checkpoint_file_name(parent: str, name: str) -> str:
    """File-name (without .json) for a checkpoint snapshot of `parent`."""
    parent = validate_name(parent)
    segment = _validate_segment(name, kind="checkpoint'а")
    return validate_name(f"{parent}{BRANCH_SEP}{CHECKPOINT_PREFIX}{segment}")


def parse_name(name: str) -> tuple[str | None, str, bool]:
    """Split a session name into (parent, own segment, is_checkpoint).

    Pure name algebra, no file read: `parent` is None for a root (no `--` in
    the name at all). Checkpoint-ness is read off the LAST segment only, via
    rpartition — so a root literally named "cp-foo" (no `--`) is never
    mistaken for a checkpoint, and "root--mvp--lite" reports its immediate
    parent as "root--mvp", not "root".
    """
    if BRANCH_SEP not in name:
        return None, name, False
    parent, _sep, last = name.rpartition(BRANCH_SEP)
    return parent, last, last.startswith(CHECKPOINT_PREFIX)


def root_of(name: str) -> str:
    """Topmost ancestor by name shape alone — string algebra, no file reads.

    Used to group `/branches`/`/sessions` by tree even when an intermediate
    file is missing (orphaned branch): the root string is still derivable.
    """
    parent, _segment, _is_checkpoint = parse_name(name)
    return root_of(parent) if parent else name


@dataclass(slots=True, frozen=True)
class TreeEntry:
    """One row of the session tree for `/branches` / an indented `/sessions`.

    `parent` is None both for an actual root AND for an orphan (parent name
    not present in the set the caller passed in) — `orphaned` is what tells
    those two apart, so the CLI never re-derives the parent string itself.
    """

    name: str
    parent: str | None
    root: str
    is_checkpoint: bool
    depth: int
    orphaned: bool


def build_tree(names: Iterable[str]) -> list[TreeEntry]:
    """Group session names into roots/branches/checkpoints, purely by name.

    A branch whose parent file is gone — deleted, or from before this
    feature existed — is reported with `orphaned=True`, not raised on: SPEC
    §7 gives branches no repair story, so a dangling one must stay listable.
    Segments never contain "--" themselves (validated at creation time), so
    `name.count(BRANCH_SEP)` is an exact depth, not an estimate.
    """
    known = set(names)
    entries: list[TreeEntry] = []
    for name in sorted(known):
        parent, _segment, is_checkpoint = parse_name(name)
        orphaned = parent is not None and parent not in known
        entries.append(
            TreeEntry(
                name=name,
                parent=None if orphaned else parent,
                root=root_of(name),
                is_checkpoint=is_checkpoint,
                depth=name.count(BRANCH_SEP),
                orphaned=orphaned,
            )
        )
    return entries


@dataclass(slots=True)
class Turn:
    """Один ход сессии.

    `model` хранится НА КАЖДЫЙ ход, а не на сессию целиком (SPEC-w02d06.md
    §11): модель меняется командой `/model` посреди разговора, и без пометки
    на ходу нельзя ни предупредить о смене, ни понять, каким окном и каким
    токенизатором мерить историю. Прямой пробел, найденный разведкой —
    simonw/llm #1140 подменяет модель молча уже годами.

    `usage` есть только у ответа модели и только тогда, когда сервер его
    прислал. None — «неизвестно», и нулём оно не становится нигде.
    """

    role: str
    content: str
    ts: str
    model: str | None = None
    usage: Usage | None = None

    def to_json(self) -> dict:
        """Запись хода в файл. redact() — как в journal.py: текст ответа или
        промпта может содержать значение ключа, эхом вернувшееся от SDK."""
        data: dict[str, object] = {
            "role": self.role,
            "content": redact(self.content),
            "ts": self.ts,
            "model": self.model,
        }
        if self.usage is not None:
            data["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        return data

    @classmethod
    def from_json(cls, raw: object) -> Turn | None:
        """Ход из файла. None — запись не похожа на ход (её пропустят с warn)."""
        if not isinstance(raw, dict):
            return None
        role = raw.get("role")
        content = raw.get("content")
        if role not in (ROLE_USER, ROLE_ASSISTANT) or not isinstance(content, str):
            return None
        usage_raw = raw.get("usage")
        # Usage.from_raw() умеет и dict, и объект SDK, и None. Отсутствующие
        # поля остаются None — «сервер не прислал», а не ноль.
        usage = Usage.from_raw(usage_raw) if isinstance(usage_raw, dict) else None
        model = raw.get("model")
        ts = raw.get("ts")
        return cls(
            role=role,
            content=content,
            ts=ts if isinstance(ts, str) else "",
            model=model if isinstance(model, str) else None,
            usage=usage,
        )


@dataclass(slots=True, frozen=True)
class SessionInfo:
    """Строка списка `/sessions`: имя, дата, число ходов, сумма токенов.

    `tokens=None` означает «ни один ход не принёс usage» — это не ноль
    токенов. `missing_usage` считает ходы ассистента без usage: без него
    неполная сумма выглядела бы полной.

    `broken` — файл не прочитался (битый JSON, чужая версия). Такая сессия
    остаётся в списке отдельной строкой: показать её как пустую значило бы
    сказать «ходов 0» про разговор, который просто не удалось прочитать.
    """

    name: str
    created: str
    turns: int
    tokens: int | None
    missing_usage: int
    broken: bool = False


@dataclass(slots=True)
class Session:
    """Именованная память агента на диске.

    Сессия — только хранилище: она не ходит в сеть, ничего не печатает и не
    знает про Agent. Проблемы чтения складываются в `warnings`, откуда их
    забирает CLI, — печатать здесь означало бы размазать контракт
    stdout/stderr по двум слоям (SPEC-w02d06.md §5, §14).
    """

    name: str
    path: Path
    created: str = field(default_factory=_now)
    turns: list[Turn] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Файл был на месте, но прочитать его не удалось. Отдельно от warnings:
    # предупреждение бывает и у целого файла (пропущенные записи), а этот флаг
    # означает «ходов не видно вовсе», и списку сессий надо их различать.
    broken: bool = False
    # Служебное состояние запуска (режим, маркер done, счётчик диалога) —
    # opaque blob: Session остаётся хранилищем и не знает, что означают ключи,
    # интерпретирует их только CLI. Пишется в файл БЕЗ bump SESSION_VERSION:
    # ключ необязательный и аддитивный — код по тегу w02d06 такой файл читает
    # (ключ игнорируется), новый код читает старые файлы (ключа нет — дефолты).
    # Bump объявил бы новые файлы «чужой версией» для старого кода и уводил бы
    # их в .bak-карантин — потеря сессии при переключении между тегами.
    state: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def path_for(name: str, directory: Path | None = None) -> Path:
        return (directory or SESSIONS_DIR) / f"{validate_name(name)}.json"

    @classmethod
    def new(cls, name: str = DEFAULT_SESSION, *, directory: Path | None = None) -> Session:
        """Пустая сессия под этим именем — то, что делает `/new`.

        Файл при этом не трогается: он будет перезаписан целиком при первом
        save(), и до него у пользователя ещё есть шанс передумать.
        """
        return cls(name=validate_name(name), path=cls.path_for(name, directory))

    @classmethod
    def load(
        cls,
        name: str = DEFAULT_SESSION,
        *,
        directory: Path | None = None,
        quarantine: bool = True,
    ) -> Session:
        """Читает сессию с диска. Битый файл — предупреждение и пустая сессия.

        Не исключение и не падение: сессия — удобство, а не единственный
        носитель смысла, и потерять разговор из-за одного сломанного байта
        хуже, чем начать заново с предупреждением (shell_gpt #786, gptme
        #3498). Испорченный файл при этом не затирается молча — он
        откладывается рядом с суффиксом .bak, чтобы его можно было починить
        руками.

        `quarantine=False` — чтение без единого изменения на диске. Нужно
        листингу: `/sessions` обязана оставаться командой «покажи, что у меня
        есть». Пока карантин делался и при листинге, второй вызов той же
        команды не показывал битую сессию вовсе и не говорил о ней ни слова —
        пользователь решал, что разговора не было. Откладывать файл имеет
        право только открытие сессии на запись.
        """
        session = cls.new(name, directory=directory)
        if not session.path.is_file():
            return session

        try:
            raw = json.loads(session.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            session.warnings.append(
                session._broken(f"файл сессии не читается: {exc}", quarantine=quarantine)
            )
            return session

        if not isinstance(raw, dict) or not isinstance(raw.get("turns"), list):
            session.warnings.append(
                session._broken(
                    "файл сессии не похож на сессию (нет объекта с turns)", quarantine=quarantine
                )
            )
            return session

        version = raw.get("version")
        if version != SESSION_VERSION:
            # Незнакомая версия — не повод гадать по полям: разобрать её
            # «как получится» означало бы тихо потерять или переврать часть
            # ходов. Лучше сказать вслух и начать пустым.
            session.warnings.append(
                session._broken(
                    f"версия файла сессии {version!r}, ожидалась {SESSION_VERSION}",
                    quarantine=quarantine,
                )
            )
            return session

        created = raw.get("created")
        if isinstance(created, str) and created:
            session.created = created

        # Файлу не доверяем вслепую: state — dict или его нет. Иное (строка,
        # число) — это чужая или битая запись, и молча подставлять её как
        # «пусто» значило бы стереть настоящие значения дефолтами.
        raw_state = raw.get("state")
        if isinstance(raw_state, dict):
            session.state = raw_state
        elif raw_state is not None:
            session.warnings.append(
                f"в сессии {session.name} ключ state не похож на объект — игнорирован"
            )

        skipped = 0
        for item in raw["turns"]:
            turn = Turn.from_json(item)
            if turn is None:
                skipped += 1
                continue
            session.turns.append(turn)
        if skipped:
            # Отдельный случай от «файл битый»: структура файла цела, испорчены
            # отдельные записи. Выбрасывать из-за них уцелевший разговор — хуже,
            # чем сказать, сколько записей пропущено.
            session.warnings.append(f"в сессии {session.name} пропущено битых записей: {skipped}")
        return session

    def _broken(self, reason: str, *, quarantine: bool) -> str:
        """Помечает сессию нечитаемой и возвращает текст предупреждения.

        При `quarantine=True` испорченный файл дополнительно откладывается
        рядом с суффиксом .bak — чтобы его можно было починить руками, а не
        обнаружить затёртым следующим save().
        """
        self.broken = True
        if not quarantine:
            return f"сессия {self.name}: {reason} — файл оставлен как есть"

        backup = self.path.with_name(f"{self.path.name}.{_now().replace(':', '-')}.bak")
        try:
            os.replace(self.path, backup)
            where = f", старый файл сохранён как {backup.name}"
        except OSError:
            # Не смогли отложить — это не повод отменять загрузку пустой
            # сессии: предупреждение всё равно уйдёт наверх.
            where = ""
        return f"сессия {self.name}: {reason} — начинаем пустую{where}"

    def save(self) -> Path:
        """Пишет сессию целиком: временный файл рядом + os.replace().

        Рядом, а не в системный temp: os.replace атомарен только внутри одной
        файловой системы, а через границу он деградирует до копирования — то
        есть ровно до того полузаписанного файла, ради защиты от которого всё
        и затевалось.

        Ошибку записи наружу НЕ глотаем (в отличие от journal.log_call, где
        лог — побочный эффект): потерянная сессия — это потерянный разговор,
        и молчать об этом нельзя.
        """
        payload = {
            "version": SESSION_VERSION,
            "name": self.name,
            "created": self.created,
            "turns": [turn.to_json() for turn in self.turns],
            # Аддитивный ключ без bump версии — совместимость с тегом w02d06
            # в обе стороны, подробности у поля state выше.
            "state": self.state,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                # fsync до replace: без него содержимое может остаться в
                # буфере ОС, и после сбоя питания на месте сессии окажется
                # файл нулевой длины — формально «атомарно заменённый».
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
        return self.path

    def add_turn(
        self,
        role: str,
        content: str,
        *,
        model: str | None = None,
        usage: Usage | None = None,
    ) -> Turn:
        turn = Turn(role=role, content=content, ts=_now(), model=model, usage=usage)
        self.turns.append(turn)
        return turn

    def record(
        self,
        user_input: str,
        assistant_text: str,
        *,
        model: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        """Записывает обмен «вопрос — ответ» одним движением.

        `assistant_text` берётся из `reply.history[-1]["content"]`, а НЕ из
        `reply.text`: при прерывании стрима агент дописывает в историю пометку
        об обрыве (advent_core/agent.py), и в файле сессии она должна быть —
        иначе следующий запуск подсунет модели её же оборванный ответ как
        законченный (SPEC-w02d06.md §12).
        """
        self.add_turn(ROLE_USER, user_input, model=model)
        if assistant_text:
            self.add_turn(ROLE_ASSISTANT, assistant_text, model=model, usage=usage)

    def history(self) -> list[Message]:
        """История как список сообщений для передачи в агента.

        Новый список из новых словарей: агент и CLI работают со своей копией,
        а правка сообщения на месте не должна незаметно менять файл сессии.
        """
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]

    def last_model(self) -> str | None:
        """Модель последнего хода, у которого она записана. None — неизвестно.

        Отсюда берётся предупреждение о смене модели внутри сессии
        (SPEC-w02d06.md §11): контекст остаётся валидным, но окно и счёт
        токенов теперь другие. Запрета нет — есть предупреждение.
        """
        for turn in reversed(self.turns):
            if turn.model:
                return turn.model
        return None

    def clear(self) -> None:
        """`/new`: забыть ходы, оставить имя. Дата начала — новая.

        Run settings in state are NOT erased: mode/done/context_limit are not
        conversation content; the CLI resets the dialog turn counter, so it
        hits the file already at zero.

        CONTENT_STATE_KEYS, however, are erased along with the turns — the
        boundary sits exactly here (SPEC-w02d09.md §8). A summary is a
        compressed form of the conversation, not a setting: a summary that
        survived `/new` would mean an agent remembering things no longer in
        the history — on a demo that looks like a ghost of the previous
        session.
        """
        self.turns.clear()
        self.created = _now()
        for key in CONTENT_STATE_KEYS:
            self.state.pop(key, None)

    def token_total(self) -> int | None:
        """Сумма total_tokens по ходам. None — ни один ход не принёс usage."""
        total = 0
        seen = False
        for turn in self.turns:
            usage = turn.usage
            if usage is None or usage.is_empty():
                continue
            seen = True
            total += usage.total_tokens or (
                (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
            )
        return total if seen else None

    def missing_usage(self) -> int:
        """Сколько ответов модели пришло без usage — сумма ниже настоящей."""
        return sum(
            1
            for turn in self.turns
            if turn.role == ROLE_ASSISTANT and (turn.usage is None or turn.usage.is_empty())
        )

    def info(self) -> SessionInfo:
        return SessionInfo(
            name=self.name,
            created=self.created,
            turns=len(self.turns),
            tokens=self.token_total(),
            missing_usage=self.missing_usage(),
            broken=self.broken,
        )


def make_checkpoint(session: Session, name: str, *, directory: Path | None = None) -> Session:
    """Full snapshot of `session` right now — turns AND state, own file.

    Not an index into `session.turns`: SPEC-w02d10.md §7.1 draws this
    straight from the summary_upto lesson (SPEC-w02d09.md §8) — an index
    into a living list goes stale silently the moment /new or /reset touches
    that list. A snapshot costs disk space and nothing else.

    deepcopy, not a fresh list of the same Turn/dict objects: the test that
    matters is "mutating the parent afterwards does not change the
    checkpoint", and a shallow copy of `turns` would still share the dict
    payload of `state`.
    """
    directory = directory or session.path.parent
    cp_name = checkpoint_file_name(session.name, name)
    checkpoint = Session(
        name=cp_name,
        path=Session.path_for(cp_name, directory),
        turns=copy.deepcopy(session.turns),
        state=copy.deepcopy(session.state),
    )
    checkpoint.state["kind"] = "checkpoint"
    checkpoint.save()
    return checkpoint


def make_branch(
    checkpoint_or_parent: Session, name: str, *, directory: Path | None = None
) -> Session:
    """New branch, copying turns+state from `checkpoint_or_parent`, saved.

    The new branch's PARENT is the checkpoint's own root, not the checkpoint
    file's name: `/checkpoint mvp` then `/branch cheap` yields
    "demo10--cheap", never "demo10--cp-mvp--cheap" (SPEC-w02d10.md §7.2's own
    demo trace) — a checkpoint is a snapshot to branch FROM, not an address
    to attach under. Passing a plain (non-checkpoint) session branches
    directly off it, `fork_at` stays None.
    """
    parent_name, _segment, is_checkpoint = parse_name(checkpoint_or_parent.name)
    parent = parent_name if is_checkpoint else checkpoint_or_parent.name
    fork_at = checkpoint_or_parent.name if is_checkpoint else None
    directory = directory or checkpoint_or_parent.path.parent
    branch_name = branch_file_name(parent, name)
    branch = Session(
        name=branch_name,
        path=Session.path_for(branch_name, directory),
        turns=copy.deepcopy(checkpoint_or_parent.turns),
        state=copy.deepcopy(checkpoint_or_parent.state),
    )
    branch.state["parent"] = parent
    branch.state["fork_at"] = fork_at
    branch.state.pop("kind", None)  # a branch is not itself a checkpoint
    branch.save()
    return branch


def delete_branch(name: str, *, directory: Path | None = None) -> None:
    """Delete a branch's file. `/branch --delete` (SPEC-w02d10.md §7.2/§7.3).

    Refuses on a checkpoint-kind file: this command creates and removes what
    `/branch` creates, and a checkpoint is addressed and reasoned about
    differently (§7.1 — you can only fork from one, never `/switch` into
    it). Refuses on a root for the same reason: a root isn't a branch either.
    Refusing to delete the CURRENTLY OPEN branch is the CLI's job — only it
    knows which session is active; this function only knows files.
    """
    path = Session.path_for(name, directory)
    if not path.is_file():
        raise ConfigError(f"ветка {name!r} не найдена")

    parent_name, _segment, _is_checkpoint = parse_name(name)
    if parent_name is None:
        raise ConfigError(f"{name!r} — корневая сессия, а не ветка; эта команда её не трогает")

    session = Session.load(name, directory=directory, quarantine=False)
    if session.state.get("kind") == "checkpoint":
        raise ConfigError(
            f"{name!r} — это checkpoint, а не ветка; удаление checkpoint'ов эта команда не делает"
        )
    path.unlink()


def list_sessions(directory: Path | None = None) -> tuple[list[SessionInfo], list[str]]:
    """Список сессий и предупреждения по нечитаемым файлам.

    Возвращает и то, и другое: сессия, которую не удалось прочитать, обязана
    попасть в вывод как проблема, а не исчезнуть из списка молча — иначе
    пользователь решит, что разговор не сохранился вовсе.

    Читает БЕЗ карантина (`quarantine=False`): команда «покажи, что у меня
    есть» не имеет права переименовывать файлы. Пока карантин делался и
    отсюда, первый `/sessions` уносил битую сессию в .bak, а второй не
    показывал ни строки, ни предупреждения — запись пропадала бесследно.
    """
    folder = directory or SESSIONS_DIR
    infos: list[SessionInfo] = []
    warnings: list[str] = []
    if not folder.is_dir():
        return infos, warnings

    for path in sorted(folder.glob("*.json")):
        try:
            name = validate_name(path.stem)
        except ConfigError:
            # Файл с именем, которое мы сами бы не создали, — чужой; молча
            # пропускаем, но говорим об этом.
            warnings.append(f"пропущен файл с недопустимым именем сессии: {path.name}")
            continue
        session = Session.load(name, directory=folder, quarantine=False)
        warnings.extend(session.warnings)
        infos.append(session.info())

    # По дате начала, свежие сверху: список нужен, чтобы вернуться к
    # последнему разговору, а не чтобы искать по алфавиту.
    infos.sort(key=lambda info: (info.created, info.name), reverse=True)
    return infos, warnings
