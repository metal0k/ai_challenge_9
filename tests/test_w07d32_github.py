"""GitHub client for the day-32 watcher: MockTransport only, synthetic credentials."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from advent_core.errors import (
    AdventError,
    AuthError,
    ConfigurationError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from adventassistant.github import (
    API_URL,
    GithubClient,
    is_sha,
    parse_repo_url,
)

TOKEN = "synthetic-token-123"
REPO = "acme/widgets"
BASE = "a" * 40
HEAD = "b" * 40


def pull_json(number=1, state="open", base=BASE, head=HEAD, head_repo=REPO, base_repo=REPO):
    return {
        "number": number,
        "state": state,
        "base": {"sha": base, "repo": {"full_name": base_repo}},
        "head": {"sha": head, "repo": {"full_name": head_repo} if head_repo else None},
    }


def make(handler):
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    http = httpx.Client(transport=httpx.MockTransport(wrapped))
    return GithubClient(REPO, TOKEN, http), seen, http


def test_repo_url_accepts_plain_git_and_slash():
    for url in (
        "https://github.com/acme/widgets",
        "https://github.com/acme/widgets.git",
        "https://github.com/acme/widgets/",
    ):
        assert parse_repo_url(url) == REPO


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/acme/widgets",
        "https://user:pw@github.com/acme/widgets",
        "https://github.com:444/acme/widgets",
        "https://github.com.evil.com/acme/widgets",
        "https://evil.com/github.com/acme/widgets",
        "https://github.com/acme/widgets?x=1",
        "https://github.com/acme/widgets#frag",
        "https://github.com/acme",
        "https://github.com/acme/widgets/extra",
        "https://github.com/../widgets",
        " https://github.com/acme/widgets",
        "",
    ],
)
def test_repo_url_rejects(url):
    with pytest.raises(ConfigurationError):
        parse_repo_url(url)


def test_sha_validation():
    assert is_sha("0" * 40)
    for bad in ("A" * 40, "a" * 39, "a" * 41, "g" * 40, None, 5):
        assert not is_sha(bad)


def test_from_project_reads_only_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "from-environment")
    monkeypatch.setenv("GITHUB_URL", "https://github.com/evil/env")
    (tmp_path / ".env").write_text(
        f"GITHUB_URL=https://github.com/{REPO}.git\nGITHUB_TOKEN={TOKEN}\n", encoding="utf-8"
    )
    client = GithubClient.from_project(tmp_path)
    try:
        assert client.repo == REPO
    finally:
        client.close()


def test_from_project_missing_values_is_config_error_without_leak(tmp_path):
    (tmp_path / ".env").write_text(f"GITHUB_TOKEN={TOKEN}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError) as exc:
        GithubClient.from_project(tmp_path)
    assert TOKEN not in str(exc.value) + (exc.value.hint or "")
    with pytest.raises(ConfigurationError):
        GithubClient.from_project(tmp_path / "nowhere")


def test_from_project_bad_url_is_refused(tmp_path):
    (tmp_path / ".env").write_text(
        f"GITHUB_URL=https://x:y@github.com/{REPO}\nGITHUB_TOKEN={TOKEN}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigurationError):
        GithubClient.from_project(tmp_path)


def test_request_goes_to_fixed_host_with_basic_auth_and_no_redirects():
    def handler(request):
        return httpx.Response(200, json={"id": 7})

    client, seen, _ = make(handler)
    assert client.current_user_id() == 7
    request = seen[0]
    assert str(request.url) == f"{API_URL}/user"
    user, _, password = (
        base64.b64decode(request.headers["authorization"].split()[1]).decode().partition(":")
    )
    assert password == TOKEN and user
    assert client.current_user_id() == 7 and len(seen) == 1  # cached


def test_redirect_is_not_followed():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://evil.example/x"})

    client, seen, _ = make(handler)
    with pytest.raises(AdventError) as exc:
        client.get_pull(1)
    assert "evil" not in exc.value.message
    assert len(seen) == 1


def test_list_open_pulls_paginates_fully_and_ignores_link_header():
    def handler(request):
        assert request.url.host == "api.github.com"
        page = int(request.url.params["page"])
        assert request.url.params["per_page"] == "100"
        assert request.url.params["state"] == "open"
        if page == 1:
            items = [pull_json(n) for n in range(1, 101)]
            return httpx.Response(
                200, json=items, headers={"link": '<https://evil.example/p2>; rel="next"'}
            )
        return httpx.Response(200, json=[pull_json(101, head_repo="fork/widgets"), pull_json(102)])

    client, seen, _ = make(handler)
    pulls = client.list_open_pulls()
    assert [p.number for p in pulls][-2:] == [101, 102] and len(pulls) == 102
    assert len(seen) == 2
    assert pulls[100].is_same_repo is False and pulls[0].is_same_repo is True


def test_exact_full_page_requests_one_more_page():
    def handler(request):
        page = int(request.url.params["page"])
        return httpx.Response(200, json=[pull_json(n) for n in range(1, 101)] if page == 1 else [])

    client, seen, _ = make(handler)
    assert len(client.list_open_pulls()) == 100
    assert len(seen) == 2


def test_draft_and_deleted_fork_repo_parse():
    item = pull_json(3, head_repo="")
    item["draft"] = True
    client, _, _ = make(lambda r: httpx.Response(200, json=[item]))
    (pull,) = client.list_open_pulls()
    assert pull.head_repo == "" and not pull.is_same_repo


def test_get_pull_returns_closed_state_and_checks_number():
    client, _, _ = make(lambda r: httpx.Response(200, json=pull_json(5, state="closed")))
    assert client.get_pull(5).state == "closed"
    with pytest.raises(AdventError):
        client.get_pull(6)  # server answered with PR 5


def test_list_with_pr_filter_skips_closed():
    client, seen, _ = make(lambda r: httpx.Response(200, json=pull_json(5, state="closed")))
    assert client.list_open_pulls(pr=5) == []
    assert seen[0].url.path.endswith("/pulls/5")


@pytest.mark.parametrize("number", [0, -1, True, "1", None])
def test_invalid_pr_number_rejected_before_request(number):
    client, seen, _ = make(lambda r: httpx.Response(200, json={}))
    with pytest.raises(AdventError):
        client.get_pull(number)
    with pytest.raises(AdventError):
        client.post_comment(number, "x")
    assert not seen


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(number=0),
        lambda d: d.update(state="merged"),
        lambda d: d["base"].update(sha="abc"),
        lambda d: d["head"].update(sha="B" * 40),
        lambda d: d["base"]["repo"].update(full_name="../x"),
        lambda d: d["base"].update(repo=None),
        lambda d: d.update(base="nope"),
    ],
)
def test_malformed_pull_is_fatal_adventerror(mutate):
    item = pull_json()
    mutate(item)
    client, _, _ = make(lambda r: httpx.Response(200, json=[item]))
    with pytest.raises(AdventError) as exc:
        client.list_open_pulls()
    assert type(exc.value) is AdventError


def test_foreign_repo_pull_is_refused():
    client, _, _ = make(lambda r: httpx.Response(200, json=[pull_json(base_repo="other/repo")]))
    with pytest.raises(AdventError):
        client.list_open_pulls()


def test_non_list_and_non_json_responses():
    client, _, _ = make(lambda r: httpx.Response(200, json={"oops": 1}))
    with pytest.raises(AdventError):
        client.list_open_pulls()
    client, _, _ = make(lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(AdventError):
        client.get_pull(1)


def comment(cid, uid, body, url=None):
    return {
        "id": cid,
        "user": {"id": uid} if uid is not None else None,
        "body": body,
        "html_url": url or f"https://github.com/{REPO}/pull/1#issuecomment-{cid}",
    }


def test_find_comment_matches_only_own_author_id_across_pages():
    marker = "<!-- adventassistant-revision:x:y -->"

    def handler(request):
        if request.url.path == "/user":
            return httpx.Response(200, json={"id": 42, "login": "someone"})
        page = int(request.url.params["page"])
        if page == 1:
            return httpx.Response(
                200,
                json=[comment(i, 99, marker) for i in range(100)],  # foreign author, same text
            )
        return httpx.Response(
            200,
            json=[
                comment(500, None, marker),
                comment(501, 42, "no marker"),
                comment(502, 42, marker),
            ],
        )

    client, _, _ = make(handler)
    assert client.find_comment(1, marker).endswith("issuecomment-502")


def test_find_comment_none_when_only_foreign_marker():
    def handler(request):
        if request.url.path == "/user":
            return httpx.Response(200, json={"id": 42})
        return httpx.Response(200, json=[comment(1, 7, "m")])

    client, _, _ = make(handler)
    assert client.find_comment(1, "m") is None


def test_find_comment_error_propagates_not_none():
    def handler(request):
        if request.url.path == "/user":
            return httpx.Response(200, json={"id": 42})
        return httpx.Response(502)

    client, _, _ = make(handler)
    with pytest.raises(ServerError):
        client.find_comment(1, "m")


def test_bad_user_id_is_fatal():
    client, _, _ = make(lambda r: httpx.Response(200, json={"id": "x"}))
    with pytest.raises(AdventError):
        client.find_comment(1, "m")


def test_post_comment_direct_post_returns_url():
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == f"/repos/{REPO}/issues/3/comments"
        assert json.loads(request.read()) == {"body": "привет"}
        return httpx.Response(201, json=comment(9, 42, "x"))

    client, seen, _ = make(handler)
    assert client.post_comment(3, "привет").endswith("issuecomment-9")
    assert len(seen) == 1  # no GET before POST


def test_post_comment_rejects_empty_oversized_and_bad_url_reply():
    client, seen, _ = make(
        lambda r: httpx.Response(201, json={"html_url": "https://evil.example/x"})
    )
    for body in ("", "  ", "x" * 65537):
        with pytest.raises(AdventError):
            client.post_comment(1, body)
    assert not seen
    with pytest.raises(AdventError):
        client.post_comment(1, "ok")


@pytest.mark.parametrize(
    ("status", "headers", "text", "exc_type"),
    [
        (401, {}, "", AuthError),
        (403, {}, "Resource not accessible", AuthError),
        (403, {"x-ratelimit-remaining": "0"}, "", RateLimitError),
        (403, {"retry-after": "7"}, "", RateLimitError),
        (403, {}, "You have exceeded a secondary rate limit", RateLimitError),
        (429, {}, "", RateLimitError),
        (500, {}, "", ServerError),
        (503, {"retry-after": "3"}, "", ServerError),
        (404, {}, "", AdventError),
        (422, {}, "", AdventError),
    ],
)
def test_status_mapping(status, headers, text, exc_type):
    client, _, _ = make(lambda r: httpx.Response(status, headers=headers, text=text))
    with pytest.raises(AdventError) as exc:
        client.get_pull(1)
    assert type(exc.value) is exc_type


def test_retry_after_is_exposed_and_capped():
    client, _, _ = make(lambda r: httpx.Response(429, headers={"retry-after": "99999"}))
    with pytest.raises(RateLimitError) as exc:
        client.get_pull(1)
    assert exc.value.retry_after == 3600
    client, _, _ = make(lambda r: httpx.Response(403, headers={"retry-after": "12"}))
    with pytest.raises(RateLimitError) as exc:
        client.get_pull(1)
    assert exc.value.retry_after == 12


def test_network_and_timeout_errors_are_typed_without_details():
    def boom(request):
        raise httpx.ConnectTimeout(f"timeout for {TOKEN}")

    client, _, _ = make(boom)
    with pytest.raises(NetworkError) as exc:
        client.get_pull(1)
    assert TOKEN not in exc.value.message
    assert getattr(exc.value, "retry_after", "missing") is None


def test_error_messages_never_contain_body_or_token():
    client, _, _ = make(lambda r: httpx.Response(500, text=f"secret body {TOKEN}"))
    with pytest.raises(ServerError) as exc:
        client.get_pull(1)
    assert TOKEN not in exc.value.message and "secret body" not in exc.value.message


def test_injected_client_is_not_closed_but_owned_one_is():
    client, _, http = make(lambda r: httpx.Response(200, json={}))
    client.close()
    assert not http.is_closed
    owned = GithubClient(REPO, TOKEN)
    inner = owned._client
    owned.close()
    assert inner.is_closed
    assert inner.follow_redirects is False


@pytest.mark.parametrize(
    ("repo", "token"),
    [("bad", TOKEN), ("a/b/c", TOKEN), ("../x", TOKEN), (REPO, ""), (REPO, "a b")],
)
def test_constructor_validation(repo, token):
    with pytest.raises(ConfigurationError):
        GithubClient(repo, token)
