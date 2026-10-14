"""Minimal GitHub REST client for the PR watcher: fixed host, no redirects, typed errors.

Pagination uses our own page/per_page on the fixed host; `Link` URLs are never followed,
so the token cannot be sent anywhere GitHub did not name in the docs.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
from dotenv import dotenv_values

from advent_core.errors import (
    AdventError,
    AuthError,
    ConfigurationError,
    NetworkError,
    RateLimitError,
    ServerError,
)

API_URL = "https://api.github.com"
TIMEOUT_S = 30.0
PER_PAGE = 100
MAX_PAGES = 200
MAX_COMMENT_CHARS = 65536
MAX_RETRY_AFTER_S = 3600
BASIC_USER = "x-access-token"

_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_URL_PATH_RE = re.compile(r"/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?")
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_COMMENT_URL_PREFIX = "https://github.com/"


@dataclass(frozen=True, slots=True)
class PullRequest:
    number: int
    state: str
    base_sha: str
    head_sha: str
    base_repo: str
    head_repo: str  # "" when the fork's repository is gone

    @property
    def is_same_repo(self) -> bool:
        return bool(self.head_repo) and self.head_repo.lower() == self.base_repo.lower()


def is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _valid_repo(repo: object) -> bool:
    if not isinstance(repo, str) or _REPO_RE.fullmatch(repo) is None:
        return False
    return all(part not in (".", "..") for part in repo.split("/"))


def _valid_number(number: object) -> bool:
    return isinstance(number, int) and not isinstance(number, bool) and number > 0


def parse_repo_url(url: str) -> str:
    """`https://github.com/owner/repo[.git][/]` -> `owner/repo`; anything else is refused."""
    bad = ConfigurationError(
        "GITHUB_URL должен быть вида https://github.com/<owner>/<repo> "
        "без логина, пароля, порта, запроса и фрагмента."
    )
    if not isinstance(url, str) or url != url.strip():
        raise bad
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise bad from None
    if (
        parts.scheme != "https"
        or parts.netloc != "github.com"
        or port is not None
        or parts.query
        or parts.fragment
    ):
        raise bad
    match = _URL_PATH_RE.fullmatch(parts.path)
    if match is None:
        raise bad
    repo = f"{match.group(1)}/{match.group(2)}"
    if not _valid_repo(repo):
        raise bad
    return repo


def _retry_after(response: httpx.Response) -> int | None:
    raw = response.headers.get("retry-after", "").strip()
    if raw.isdigit():
        return min(int(raw), MAX_RETRY_AFTER_S)
    reset = response.headers.get("x-ratelimit-reset", "").strip()
    if reset.isdigit():
        return max(0, min(int(reset) - int(time.time()), MAX_RETRY_AFTER_S))
    return None


def _with_retry(exc: AdventError, seconds: int | None) -> AdventError:
    exc.retry_after = seconds  # type: ignore[attr-defined]
    return exc


def _is_rate_limited(response: httpx.Response) -> bool:
    if response.headers.get("x-ratelimit-remaining", "").strip() == "0":
        return True
    if "retry-after" in response.headers:
        return True
    try:
        return "rate limit" in response.text.lower()
    except Exception:  # noqa: BLE001 - classification only, the body is never printed
        return False


class GithubClient:
    def __init__(self, repo: str, token: str, client: httpx.Client | None = None) -> None:
        if not _valid_repo(repo):
            raise ConfigurationError("Репозиторий должен быть вида owner/repo.")
        if not isinstance(token, str) or not token or re.search(r"\s", token):
            raise ConfigurationError("GITHUB_TOKEN пуст или содержит пробельные символы.")
        self.repo = repo
        self._auth = httpx.BasicAuth(BASIC_USER, token)
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=TIMEOUT_S, follow_redirects=False)
        self._user_id: int | None = None

    @classmethod
    def from_project(cls, root: str | Path, client: httpx.Client | None = None) -> GithubClient:
        values = dotenv_values(Path(root) / ".env")
        url = (values.get("GITHUB_URL") or "").strip()
        token = (values.get("GITHUB_TOKEN") or "").strip()
        if not url or not token:
            raise ConfigurationError(
                "В .env нужны GITHUB_URL и GITHUB_TOKEN.",
                hint="GITHUB_URL=https://github.com/<owner>/<repo>",
            )
        return cls(parse_repo_url(url), token, client)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        body: dict[str, object] | None = None,
    ) -> object:
        try:
            response = self._client.request(
                method,
                API_URL + path,
                params=params,
                json=body,
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                auth=self._auth,
                timeout=TIMEOUT_S,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            # type name only: exception text may carry request details
            raise _with_retry(
                NetworkError(f"GitHub недоступен ({type(exc).__name__})."), None
            ) from None
        self._raise_for_status(response)
        try:
            return response.json()
        except ValueError:
            raise AdventError("GitHub вернул не JSON.") from None

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        if status == 401:
            raise AuthError("GitHub отклонил токен (401).", hint="Проверьте GITHUB_TOKEN в .env.")
        if status == 429 or (status == 403 and _is_rate_limited(response)):
            raise _with_retry(
                RateLimitError(f"Лимит запросов GitHub ({status})."), _retry_after(response)
            )
        if status == 403:
            raise AuthError(
                "GitHub отказал в доступе (403).",
                hint="Токену нужен доступ к репозиторию и комментариям PR.",
            )
        if status >= 500:
            raise _with_retry(
                ServerError(f"Ошибка сервера GitHub ({status})."), _retry_after(response)
            )
        if 300 <= status < 400:
            raise AdventError(f"GitHub ответил редиректом ({status}); он не выполняется.")
        if status == 404:
            raise AdventError(
                "GitHub: не найдено (404).", hint="Проверьте GITHUB_URL и права токена."
            )
        raise AdventError(f"GitHub отклонил запрос ({status}).")

    def _pages(self, path: str, params: dict[str, object]) -> list[object]:
        items: list[object] = []
        for page in range(1, MAX_PAGES + 1):
            data = self._request("GET", path, params={**params, "per_page": PER_PAGE, "page": page})
            if not isinstance(data, list):
                raise AdventError("GitHub вернул неожиданную форму списка.")
            items.extend(data)
            if len(data) < PER_PAGE:
                return items
        raise AdventError("Слишком много страниц в ответе GitHub.")

    def _repo_path(self) -> str:
        owner, name = self.repo.split("/")
        return f"/repos/{quote(owner, safe='')}/{quote(name, safe='')}"

    def _parse_pull(self, item: object) -> PullRequest:
        bad = AdventError("GitHub вернул PR в неожиданном формате.")
        if not isinstance(item, dict):
            raise bad
        number, state = item.get("number"), item.get("state")
        base, head = item.get("base"), item.get("head")
        if not _valid_number(number) or state not in ("open", "closed"):
            raise bad
        if not isinstance(base, dict) or not isinstance(head, dict):
            raise bad
        base_info, head_info = base.get("repo"), head.get("repo")
        base_repo = base_info.get("full_name") if isinstance(base_info, dict) else None
        head_repo = head_info.get("full_name") if isinstance(head_info, dict) else ""
        base_sha, head_sha = base.get("sha"), head.get("sha")
        if not is_sha(base_sha) or not is_sha(head_sha) or not _valid_repo(base_repo):
            raise bad
        if head_repo != "" and not _valid_repo(head_repo):
            raise bad
        if base_repo.lower() != self.repo.lower():
            raise AdventError("GitHub вернул PR чужого репозитория.")
        return PullRequest(number, state, base_sha, head_sha, base_repo, head_repo)

    def current_user_id(self) -> int:
        if self._user_id is None:
            data = self._request("GET", "/user")
            uid = data.get("id") if isinstance(data, dict) else None
            if not _valid_number(uid):
                raise AdventError("GitHub не вернул id пользователя токена.")
            self._user_id = uid
        return self._user_id

    def list_open_pulls(self, pr: int | None = None) -> list[PullRequest]:
        if pr is not None:
            pull = self.get_pull(pr)
            return [pull] if pull.state == "open" else []
        pulls = [
            self._parse_pull(item)
            for item in self._pages(f"{self._repo_path()}/pulls", {"state": "open"})
        ]
        if any(p.state != "open" for p in pulls):
            raise AdventError("GitHub вернул закрытый PR в списке открытых.")
        return pulls

    def get_pull(self, number: int) -> PullRequest:
        if not _valid_number(number):
            raise AdventError("Номер PR должен быть положительным целым.")
        pull = self._parse_pull(self._request("GET", f"{self._repo_path()}/pulls/{number}"))
        if pull.number != number:
            raise AdventError("GitHub вернул другой PR, чем запрошен.")
        return pull

    def find_comment(self, number: int, marker: str) -> str | None:
        """URL of the first comment written by the token's own user that contains `marker`."""
        if not _valid_number(number):
            raise AdventError("Номер PR должен быть положительным целым.")
        if not isinstance(marker, str) or not marker:
            raise AdventError("Маркер комментария пуст.")
        uid = self.current_user_id()
        for item in self._pages(f"{self._repo_path()}/issues/{number}/comments", {}):
            if not isinstance(item, dict):
                raise AdventError("GitHub вернул комментарий в неожиданном формате.")
            user, body = item.get("user"), item.get("body")
            if not isinstance(user, dict) or user.get("id") != uid or not isinstance(body, str):
                continue
            if marker in body:
                return _comment_url(item)
        return None

    def post_comment(self, number: int, body: str) -> str:
        if not _valid_number(number):
            raise AdventError("Номер PR должен быть положительным целым.")
        if not isinstance(body, str) or not body.strip():
            raise AdventError("Текст комментария пуст.")
        if len(body) > MAX_COMMENT_CHARS:
            raise AdventError(f"Комментарий длиннее лимита GitHub ({MAX_COMMENT_CHARS} символов).")
        data = self._request(
            "POST", f"{self._repo_path()}/issues/{number}/comments", body={"body": body}
        )
        if not isinstance(data, dict):
            raise AdventError("GitHub вернул комментарий в неожиданном формате.")
        return _comment_url(data)


def _comment_url(item: dict) -> str:
    url = item.get("html_url")
    if not isinstance(url, str) or not url.startswith(_COMMENT_URL_PREFIX):
        raise AdventError("GitHub вернул комментарий без корректной ссылки.")
    return url
