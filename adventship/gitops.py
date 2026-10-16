"""Stages `commit` and `tag`: explicit add list, index scan, human gates, ephemeral-auth push.

Nothing here forces anything: no `--force`, no `--no-verify`, no `commit -a`. The GitHub token
lives only in the argv of one `git push` call and never reaches a URL or `.git/config`.
"""

from __future__ import annotations

import base64
import fnmatch
import re
import sys

from advent_cli.links import _github, parse_repo, tag_url
from adventship.ctx import (
    EXIT_AUTH,
    EXIT_CHECKS_FAIL,
    EXIT_CONFIG,
    EXIT_STAGE_FAILED,
    ShipContext,
    ShipError,
    StageResult,
)
from adventship.gates import confirm
from adventship.gitutil import changed_paths, dirty_tracked_paths, git
from adventship.llm import generate_commit_message, scrub, validate_commit_message

BRANCH = "main"
PUSH_TIMEOUT = 300.0
COMMIT_TIMEOUT = 300.0
# Never part of a day's commit: they must be removed or git-ignored before shipping.
JUNK_PATTERNS = ("*.log", ".env", ".env.*", "*.bak", "*.tmp")


# --------------------------------------------------------------------- helpers


def _head(ctx: ShipContext) -> str:
    return git(ctx, "rev-parse", "HEAD").stdout.strip()


def _dirty(ctx: ShipContext) -> list[str]:
    """Tracked paths still dirty (empty list = clean); untracked files never block."""
    return dirty_tracked_paths(ctx)


def _untracked_count(ctx: ShipContext) -> int:
    out = git(ctx, "ls-files", "--others", "--exclude-standard").stdout
    return len([line for line in out.splitlines() if line.strip()])


def _is_junk(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, pattern) for pattern in JUNK_PATTERNS)


def _env(ctx: ShipContext, key: str) -> str:
    value = (ctx.env.get(key) or "").strip()
    if not value:
        raise ShipError(f"Не найден {key} в .env", exit_code=EXIT_CONFIG)
    return value


def fetch_remote_ref(github_url: str, token: str, ref: str) -> tuple[str, str] | None:
    """`(object type, sha)` of `tags/<tag>` or `heads/<branch>` on GitHub; None when absent."""
    owner, repo = parse_repo(github_url)
    response = _github("GET", f"/repos/{owner}/{repo}/git/ref/{ref}", token)
    if response.status_code == 404:
        return None
    if response.status_code in (401, 403):
        raise ShipError(
            f"GitHub отклонил токен ({response.status_code})",
            exit_code=EXIT_AUTH,
            hint="Проверь GITHUB_TOKEN и его права на репозиторий.",
        )
    if response.status_code != 200:
        raise ShipError(f"GitHub ответил {response.status_code} на проверку {ref}")
    obj = response.json().get("object", {})
    return str(obj.get("type", "")), str(obj.get("sha", ""))


def _remote_ref(ctx: ShipContext, ref: str) -> tuple[str, str] | None:
    # Looked up on the module at call time so tests can replace the HTTP seam.
    return fetch_remote_ref(_env(ctx, "GITHUB_URL"), _env(ctx, "GITHUB_TOKEN"), ref)


def push_argv(token: str, tag: str) -> list[str]:
    """The exact auth contract: no credential helper, one ephemeral Basic header, no URL token."""
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
    return [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        f"http.extraheader=AUTHORIZATION: basic {basic}",
        "push",
        "origin",
        BRANCH,
        f"refs/tags/{tag}",
    ]


def _mask_push_secrets(text: str, token: str) -> str:
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
    return scrub(text.replace(basic, "***").replace(token, "***"), {"GITHUB_TOKEN": token})


# --------------------------------------------------------------------- commit


def run_commit(ctx: ShipContext) -> StageResult:
    paths = changed_paths(ctx)
    # Work already committed: untracked files are never the day's change set, so HEAD is reused.
    if not paths or (not dirty_tracked_paths(ctx) and _head_is_days_commit(ctx)):
        return _reuse_head_commit(ctx)
    junk = [p for p in paths if _is_junk(p)]
    if junk:
        raise ShipError(
            "в рабочем дереве файлы, которые нельзя коммитить: " + ", ".join(junk),
            hint="Удали их или добавь в .gitignore: после commit дерево должно быть чистым.",
        )
    message = generate_commit_message(ctx, paths)
    git(ctx, "add", "--", *paths)
    scan = ctx.run([sys.executable, str(ctx.root / "tools" / "check_staged.py")], timeout=120)
    if scan.returncode != 0:
        raise ShipError(
            "check_staged.py остановил commit",
            exit_code=EXIT_CHECKS_FAIL,
            hint=scrub(
                "\n".join((scan.stdout or scan.stderr or "").strip().splitlines()[-15:]), ctx.env
            ),
        )
    stat = git(ctx, "diff", "--cached", "--stat").stdout.strip()
    confirm(
        ctx,
        "Сделать commit?",
        f"Сообщение: {message}\nФайлов: {len(paths)}\n{stat}",
    )
    # Re-validated right before use: the message is the only model text that reaches git history.
    message = validate_commit_message(message, ctx.tag, ctx.env)
    git(ctx, "commit", "-m", message, timeout=COMMIT_TIMEOUT)

    sha = _head(ctx)
    dirty = _dirty(ctx)
    if dirty:
        raise ShipError(
            "после commit рабочее дерево не чистое: " + ", ".join(dirty[:10]),
            exit_code=EXIT_STAGE_FAILED,
        )
    return StageResult(
        data={"sha": sha, "message": message, "files": paths},
        summary=f"{sha[:8]} {message}",
    )


_SUBJECT = re.compile(r"^(?:feat|fix|docs|test|refactor|chore|perf|style)\((w\d{2}d\d{2})\): ")


def _head_is_days_commit(ctx: ShipContext) -> bool:
    match = _SUBJECT.match(git(ctx, "log", "-1", "--format=%s").stdout.strip())
    return bool(match) and match.group(1) == ctx.tag


def _reuse_head_commit(ctx: ShipContext) -> StageResult:
    """Clean tree: a previous run may have committed and died before saving state."""
    subject = git(ctx, "log", "-1", "--format=%s").stdout.strip()
    match = _SUBJECT.match(subject)
    if not match or match.group(1) != ctx.tag:
        raise ShipError(
            "нечего коммитить: рабочее дерево чистое, а HEAD не коммит этого дня",
            hint=f"Ожидался коммит вида type({ctx.tag}): ...",
        )
    sha = _head(ctx)
    return StageResult(
        data={"sha": sha, "message": subject, "files": [], "reused": True},
        summary=f"{sha[:8]} (уже закоммичено) {subject}",
    )


def recheck_commit(ctx: ShipContext) -> bool:
    recorded = ctx.state.data("commit").get("sha")
    try:
        return bool(recorded) and _head(ctx) == recorded and not _dirty(ctx)
    except ShipError:
        return False


# --------------------------------------------------------------------- tag


def _local_tag_sha(ctx: ShipContext, tag: str) -> str | None:
    result = git(ctx, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}", check=False)
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _verify_remote(ctx: ShipContext, sha: str) -> None:
    tag_ref = _remote_ref(ctx, f"tags/{ctx.tag}")
    if tag_ref is None or tag_ref[1] != sha:
        raise ShipError(
            f"на GitHub тег {ctx.tag} не указывает на коммит {sha[:8]}",
            hint=f"Ответ GitHub: {tag_ref[1][:8] if tag_ref else 'тега нет'}.",
        )
    branch_ref = _remote_ref(ctx, f"heads/{BRANCH}")
    if branch_ref is None or branch_ref[1] != sha:
        raise ShipError(f"на GitHub {BRANCH} не указывает на коммит {sha[:8]}")


def run_tag(ctx: ShipContext) -> StageResult:
    tag = ctx.tag
    sha = ctx.state.data("commit").get("sha")
    if not sha:
        raise ShipError("нет sha коммита: стадия commit не выполнена")
    if _head(ctx) != sha:
        raise ShipError(
            "HEAD ушёл от коммита дня",
            hint="Код менялся после commit: повтори с --from commit.",
        )
    dirty = _dirty(ctx)
    if dirty:
        raise ShipError("рабочее дерево не чистое: " + ", ".join(dirty[:10]))
    untracked = _untracked_count(ctx)
    branch = git(ctx, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != BRANCH:
        raise ShipError(f"тег ставится из ветки {BRANCH}, сейчас {branch}")

    github_url = _env(ctx, "GITHUB_URL")
    token = _env(ctx, "GITHUB_TOKEN")
    remote_tag = _remote_ref(ctx, f"tags/{tag}")
    if remote_tag is not None and remote_tag[1] != sha:
        raise ShipError(
            f"тег {tag} уже запушен на другой коммит ({remote_tag[1][:8]})",
            hint=f"Перетегировать запрещено (без --force): реши вручную, например {tag}-2.",
        )

    local = _local_tag_sha(ctx, tag)
    if local is not None and local != sha:
        git(ctx, "tag", "-d", tag)  # not pushed (checked above), so moving it is safe
        local = None
    if local is None:
        git(ctx, "tag", tag, sha)

    already_pushed = remote_tag is not None
    branch_ref = _remote_ref(ctx, f"heads/{BRANCH}")
    if not (already_pushed and branch_ref is not None and branch_ref[1] == sha):
        confirm(
            ctx,
            "Запушить main и тег?",
            f"Тег: {tag}\nКоммит: {sha[:8]}\nПуш: origin {BRANCH} + refs/tags/{tag}",
        )
        _push(ctx, token, tag)
    _verify_remote(ctx, sha)
    data: dict = {
        "tag": tag,
        "sha": sha,
        "code_url": tag_url(github_url, tag),
        "remote_verified": True,
    }
    if untracked:
        data["warning"] = f"untracked: {untracked}"  # count only, never names
    return StageResult(
        data=data,
        summary=f"{tag} -> {sha[:8]} проверен через GitHub API",
    )


def _push(ctx: ShipContext, token: str, tag: str) -> None:
    env = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"}
    leak: str | None = None
    try:
        result = ctx.run(push_argv(token, tag), timeout=PUSH_TIMEOUT, env=env)
    finally:
        # Runs on every exit path: the header must not survive in any config scope.
        check = git(ctx, "config", "--get-all", "http.extraheader", check=False)
        if check.stdout.strip():
            leak = "в git config остался http.extraheader"
    if leak:
        raise ShipError(leak, hint="Удали его: git config --unset-all http.extraheader.")
    if result.returncode != 0:
        detail = _mask_push_secrets(
            "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-6:]), token
        )
        auth = re.search(r"\b(401|403)\b|authentication|denied|could not read", detail, re.I)
        raise ShipError(
            f"git push завершился с кодом {result.returncode}",
            exit_code=EXIT_AUTH if auth else EXIT_STAGE_FAILED,
            hint=detail or None,
        )


def recheck_tag(ctx: ShipContext) -> bool:
    data = ctx.state.data("tag")
    try:
        remote = _remote_ref(ctx, f"tags/{ctx.tag}")
    except ShipError:
        return False
    if remote is None or remote[1] != data.get("sha"):
        return False
    # A moved main means the tag no longer describes the published code.
    try:
        head = _remote_ref(ctx, f"heads/{BRANCH}")
    except ShipError:
        return False
    return head is not None and head[1] == data.get("sha")


__all__ = [
    "fetch_remote_ref",
    "push_argv",
    "recheck_commit",
    "recheck_tag",
    "run_commit",
    "run_tag",
]
