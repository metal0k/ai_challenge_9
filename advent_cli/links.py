"""Ссылки для сдачи: GitHub-тег и публичная ссылка на видео с Яндекс.Диска."""

from __future__ import annotations

import os
import re
from pathlib import Path

import httpx

from advent_core.errors import AdventError

GITHUB_API = "https://api.github.com"
YANDEX_API = "https://cloud-api.yandex.net/v1/disk"


# ---------------------------------------------------------------- GitHub


def parse_repo(github_url: str) -> tuple[str, str]:
    """`https://github.com/owner/repo.git` → `(owner, repo)`."""
    match = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", github_url.strip())
    if not match:
        raise AdventError(f"Не разобрал GITHUB_URL: {github_url!r}")
    return match.group(1), match.group(2)


def tag_url(github_url: str, tag: str) -> str:
    owner, repo = parse_repo(github_url)
    return f"https://github.com/{owner}/{repo}/tree/{tag}"


def _github(method: str, path: str, token: str, **kwargs) -> httpx.Response:
    return httpx.request(
        method,
        GITHUB_API + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30.0,
        **kwargs,
    )


def remote_tag_exists(github_url: str, token: str, tag: str) -> bool:
    """Есть ли тег на remote.

    Локального тега мало: ссылка в таблице ведёт на GitHub, и если тег не
    запушен, проверяющий увидит 404.
    """
    owner, repo = parse_repo(github_url)
    response = _github("GET", f"/repos/{owner}/{repo}/git/ref/tags/{tag}", token)
    if response.status_code == 200:
        return True
    if response.status_code == 404:
        return False
    raise AdventError(f"GitHub ответил {response.status_code} на проверку тега {tag}")


def create_remote_tag(github_url: str, token: str, tag: str, sha: str) -> str:
    """Создаёт lightweight-тег на указанный коммит."""
    owner, repo = parse_repo(github_url)
    response = _github(
        "POST",
        f"/repos/{owner}/{repo}/git/refs",
        token,
        json={"ref": f"refs/tags/{tag}", "sha": sha},
    )
    if response.status_code in (200, 201):
        return tag_url(github_url, tag)
    if response.status_code == 422:
        # 422 здесь чаще всего означает не «тег занят», а «коммита нет на
        # remote»: sha берётся из локального HEAD, который мог быть не запушен.
        raise AdventError(
            f"GitHub отклонил тег {tag} (422): {_reason(response)}",
            hint=f"Скорее всего коммит {sha[:8]} ещё не запушен. Сделай git push и повтори.",
        )
    raise AdventError(
        f"GitHub отклонил создание тега {tag} ({response.status_code}): {_reason(response)}"
    )


def _reason(response: httpx.Response) -> str:
    """Достаёт человекочитаемую причину из тела ответа GitHub."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    parts = [body.get("message", "")]
    parts += [e.get("message", "") for e in body.get("errors", []) if isinstance(e, dict)]
    return "; ".join(p for p in parts if p) or response.text[:200]


# ---------------------------------------------------------- Яндекс.Диск


def find_sync_root(start: Path) -> Path | None:
    """Ищет корень синхронизации Яндекс.Диска вверх от папки с видео.

    Признак корня — служебная папка `.sync`. Автоопределение избавляет от
    ещё одной переменной в .env и не ломается при переносе папки.
    """
    for candidate in [start, *start.parents]:
        if (candidate / ".sync").exists():
            return candidate
    return None


def disk_path(local_file: Path) -> str:
    """Локальный путь → путь внутри Яндекс.Диска (`/Media/Study/...`)."""
    local_file = local_file.resolve()
    root_env = os.getenv("YANDEX_DISK_LOCAL_ROOT")
    root = Path(root_env).resolve() if root_env else find_sync_root(local_file.parent)

    if root is None:
        raise AdventError(
            f"Не нашёл корень Яндекс.Диска выше {local_file.parent}.",
            hint="Задай YANDEX_DISK_LOCAL_ROOT в .env — путь к синхронизируемой папке.",
        )
    try:
        relative = local_file.relative_to(root)
    except ValueError as exc:
        raise AdventError(f"{local_file} лежит вне папки Яндекс.Диска {root}") from exc

    return "/" + relative.as_posix()


def _yandex(method: str, path: str, token: str, params: dict) -> httpx.Response:
    return httpx.request(
        method,
        YANDEX_API + path,
        headers={"Authorization": f"OAuth {token}"},
        params=params,
        timeout=30.0,
    )


def publish_and_get_url(token: str, remote_path: str) -> str:
    """Публикует файл и возвращает персональную публичную ссылку.

    Два шага по документации Яндекса: PUT /resources/publish делает ресурс
    публичным, после чего public_url читается из метаданных ресурса.
    Публиковать может только владелец; токену нужны scopes
    cloud_api:disk.read и cloud_api:disk.write.
    """
    response = _yandex("PUT", "/resources/publish", token, {"path": remote_path})
    if response.status_code == 401:
        raise AdventError(
            "Яндекс.Диск отклонил токен (401).",
            hint=(
                "Проверь YANDEX_DISK_TOKEN и его scopes: cloud_api:disk.read, cloud_api:disk.write."
            ),
        )
    if response.status_code == 404:
        raise AdventError(
            f"Файл не найден на Яндекс.Диске: {remote_path}",
            hint="Дождись, пока клиент Яндекс.Диска зальёт файл, и повтори.",
        )
    if response.status_code not in (200, 409):
        raise AdventError(f"Яндекс.Диск ответил {response.status_code} на публикацию.")

    meta = _yandex("GET", "/resources", token, {"path": remote_path, "fields": "public_url"})
    if meta.status_code != 200:
        raise AdventError(f"Не удалось прочитать метаданные файла ({meta.status_code}).")

    public_url = meta.json().get("public_url")
    if not public_url:
        raise AdventError(f"Яндекс.Диск не вернул public_url для {remote_path}.")
    return public_url
