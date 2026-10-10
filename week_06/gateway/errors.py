"""One error type: carries the HTTP status and renders the OpenAI-style body."""

from __future__ import annotations


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        type_: str = "invalid_request_error",
        retry_after: int | None = None,
        extra: dict | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.type = type_
        self.retry_after = retry_after
        self.extra = extra or {}

    def body(self) -> dict:
        return {
            "error": {"message": self.message, "type": self.type, "code": self.code, **self.extra}
        }

    def headers(self) -> dict[str, str]:
        return {"Retry-After": str(self.retry_after)} if self.retry_after else {}
