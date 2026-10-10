"""Gateway configuration: dataclass defaults < JSON file < ADVENT_GATEWAY_<FIELD> env."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ENV_PREFIX = "ADVENT_GATEWAY_"


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8790
    upstream_port: int = 8791
    keys_file: str = "/etc/advent-llm/keys.json"
    version: str = "dev"
    model_id: str = "qwen2.5-0.5b-instruct-q4_k_m"

    # child process
    llama_bin: str = "/opt/advent-llm/bin/llama-b11541/llama-server"
    model_path: str = "/opt/advent-llm/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    lib_dir: str = "/opt/advent-llm/lib"
    child_home: str = "/var/lib/advent-llm"
    # Tests replace the whole command; production builds it from the fields above.
    llama_argv_override: list[str] | None = None
    # Extra env for the child (tests only); the base env is built from scratch.
    child_extra_env: dict[str, str] = field(default_factory=dict)

    # limits
    n_ctx: int = 2048
    default_max_tokens: int = 256
    max_tokens_ceiling: int = 512
    default_temperature: float = 0.7
    body_max_bytes: int = 65536
    queue_max: int = 2
    queue_retry_after: int = 10
    global_per_minute: int = 20
    restart_max: int = 5
    restart_window: float = 600.0
    mem_available_min_mb: int = 500
    meminfo_path: str = "/proc/meminfo"

    # lifecycle / deadlines (seconds)
    idle_seconds: float = 60.0
    watchdog_interval: float = 5.0
    kill_grace: float = 10.0
    health_poll: float = 0.2
    start_timeout: float = 30.0
    queue_timeout: float = 150.0
    gen_deadline: float = 150.0
    body_read_deadline: float = 10.0
    slot_check_grace: float = 5.0
    upstream_timeout: float = 10.0  # short upstream calls only; generation uses gen_deadline

    def validate(self) -> None:
        if not 1 <= self.default_max_tokens <= self.max_tokens_ceiling:
            raise ValueError("default_max_tokens must be within 1..max_tokens_ceiling")
        if self.queue_max < 0 or self.idle_seconds <= 0 or self.n_ctx <= 0:
            raise ValueError("queue_max/idle_seconds/n_ctx out of range")

    def llama_argv(self) -> list[str]:
        if self.llama_argv_override is not None:
            return list(self.llama_argv_override)
        return [
            self.llama_bin, "-m", self.model_path,
            "--host", "127.0.0.1", "--port", str(self.upstream_port),
            "-c", str(self.n_ctx), "-np", "1", "-t", "1",
            "-n", str(self.max_tokens_ceiling), "--no-webui",
        ]  # fmt: skip

    def child_env(self) -> dict[str, str]:
        # From scratch: inherited LLAMA_ARG_* must not reach the server.
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LD_LIBRARY_PATH": self.lib_dir,
            "HOME": self.child_home,
        }
        if os.name == "nt":  # tests on Windows need these for the interpreter to start
            for name in ("SYSTEMROOT", "PATH"):
                if name in os.environ:
                    env[name] = os.environ[name]
        env.update(self.child_extra_env)
        return env


def _cast(raw: str, current):
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def load_settings(path: str | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    path = path or env.get(ENV_PREFIX + "CONFIG")
    data: dict = {}
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("gateway config must be a JSON object")
    known = {f.name for f in fields(Settings)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    settings = Settings(**data)
    for f in fields(Settings):
        raw = env.get(ENV_PREFIX + f.name.upper())
        if raw is not None and f.name not in ("llama_argv_override", "child_extra_env"):
            setattr(settings, f.name, _cast(raw, getattr(settings, f.name)))
    settings.validate()
    return settings
