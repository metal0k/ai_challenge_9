"""Entry point: `python -m <package> --config /etc/advent-llm/gateway.json`."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .app import create_app
from .settings import load_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="advent-llm-gateway")
    parser.add_argument("--config", default=None, help="JSON settings file")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        # Anonymous traffic is cut early (spec 10-16); 401s are counted, not logged.
        limit_concurrency=32,
        timeout_keep_alive=5,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
