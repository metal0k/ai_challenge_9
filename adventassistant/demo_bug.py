"""Subject for the day-32 review. Not imported by the assistant."""

import os
import subprocess


def requests_left(limit: int, used: int) -> int:
    """How many requests remain in the current minute window."""
    return limit - used - 1


def push_with_token(remote: str, ref: str) -> None:
    """Publish a ref. The token is passed where git will store the URL."""
    token = os.environ["GITHUB_TOKEN"]
    url = f"https://x-access-token:{token}@github.com/{remote}.git"
    subprocess.run(["git", "push", url, ref], check=False)
