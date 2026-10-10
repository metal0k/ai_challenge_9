#!/usr/bin/env python3
"""Deploy the day-30 gateway to the VPS over the VPS_Manager SSH layer.

Every subcommand first builds a list of steps (plan); nothing touches the server until a
mutating command gets an explicit --yes. The SSH layer is imported from
$VPS_MANAGER_DIR/tools/vps.py, so no local path lives in this file.

Keys never leave the machine except as sha256 hashes in keys.json; the smoke test is the one
place a raw key is sent, in a mode-600 curl config that is removed in an always-run step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GATEWAY_DIR = REPO / "week_06" / "gateway"

BASE = "/opt/advent-llm"
RELEASES = f"{BASE}/releases"
CURRENT = f"{BASE}/current"
UV_DIR = f"{BASE}/uv"
UV_BIN = f"{UV_DIR}/uv"
PY_DIR = f"{BASE}/python"
LLAMA_BIN = f"{BASE}/bin/llama-b11541/llama-server"
LIBGOMP = f"{BASE}/lib/libgomp.so.1"
MODELS = f"{BASE}/models"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
# verified against Hugging Face and the server's copy
MODEL_SHA256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
MODEL_BYTES = 491400032
KEYS_PATH = "/etc/advent-llm/keys.json"
UNIT_PATH = "/etc/systemd/system/advent-llm.service"
SERVICE = "advent-llm"
SYS_USER = "advent-llm"
LOOPBACK = "http://127.0.0.1:8790"
PUBLIC_ORIGIN = "https://advent-llm.duckdns.org"
SMOKE_CURLRC = "/run/advent-llm-smoke.curlrc"
MIN_AVAILABLE_MB = 500
MIN_DISK_MB = 1500
READY_DEADLINE_S = 30
RADIO_SAMPLES = 20
# static radio page of the neighbouring VPS_Manager deployment (its own smoke.py checks it)
DEFAULT_RADIO_URL = "https://lala-radio.duckdns.org/"

# name -> (env var holding the raw key, rpm, burst); spec section 2.4
KEY_SPECS = (
    ("owner", "ADVENT_SERVICE_KEY", 30, 6),
    ("demo", "ADVENT_SERVICE_DEMO_KEY", 3, 2),
)


HttpGet = Callable[[str, str | None], int]


class DeployError(RuntimeError):
    """A step failed; the message is shown to the owner as is."""


# --- remote layer ---------------------------------------------------------------------------


class Remote(Protocol):
    def run(self, cmd: str) -> tuple[int, str, str]: ...
    def write(self, path: str, text: str, mode: str) -> None: ...
    def put(self, local: Path, remote: str) -> None: ...
    def close(self) -> None: ...


class VpsRemote:
    """Thin adapter over VPS_Manager's tools/vps.py (paramiko)."""

    def __init__(self, manager_dir: str) -> None:
        path = Path(manager_dir) / "tools" / "vps.py"
        if not path.is_file():
            raise DeployError(f"в VPS_MANAGER_DIR нет tools/vps.py: {path}")
        spec = importlib.util.spec_from_file_location("vps_manager_vps", path)
        assert spec and spec.loader
        self._vps = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._vps)
        self._ssh = self._vps.connect()

    def run(self, cmd: str) -> tuple[int, str, str]:
        return self._vps.run(self._ssh, cmd, check=False)

    def write(self, path: str, text: str, mode: str) -> None:
        self._vps.write_remote(self._ssh, path, text, mode)

    def put(self, local: Path, remote: str) -> None:
        self._vps.put(self._ssh, local, remote)

    def close(self) -> None:
        self._ssh.close()


# --- ops and steps ----------------------------------------------------------------------------


@dataclass
class Cmd:
    cmd: str
    # returns an error text, or None when the output is acceptable
    verify: Callable[[str], str | None] | None = None
    # runs even when an earlier op of the step failed (cleanup)
    always: bool = False
    tolerate: bool = False  # a non-zero rc is fine

    def describe(self) -> str:
        return f"$ {self.cmd}"


@dataclass
class Write:
    path: str
    # a callable keeps the content out of the plan: it is resolved only when executing
    content: str | Callable[[], str]
    mode: str = "644"
    chgrp: str | None = None
    always: bool = False

    def describe(self) -> str:
        return f"запись {self.path} (mode {self.mode})"


@dataclass
class Put:
    label: str
    make_local: Callable[[], Path]
    remote: str
    always: bool = False

    def describe(self) -> str:
        return f"загрузка {self.label} -> {self.remote}"


@dataclass
class LocalCheck:
    """A check run on this machine (external HTTP), shaped like an op for the executor."""

    label: str
    fn: Callable[[], None]
    always: bool = False

    def describe(self) -> str:
        return f"локально: {self.label}"


Op = Cmd | Write | Put | LocalCheck


@dataclass
class Step:
    title: str
    ops: list[Op] = field(default_factory=list)
    mutating: bool = True


def render_plan(steps: list[Step]) -> str:
    lines: list[str] = []
    for i, st in enumerate(steps, 1):
        tag = "запись" if st.mutating else "чтение"
        lines.append(f"[{i}] {st.title} ({tag})")
        lines.extend(f"      {op.describe()}" for op in st.ops)
    return "\n".join(lines)


def execute(steps: list[Step], remote: Remote, out: Callable[[str], None] = print) -> None:
    for i, st in enumerate(steps, 1):
        out(f"\n=== [{i}/{len(steps)}] {st.title}")
        error: str | None = None
        for op in st.ops:
            if error is not None and not op.always:
                continue
            try:
                _run_op(op, remote, out)
            except DeployError as exc:
                error = error or str(exc)
        if error is not None:
            raise DeployError(f"шаг «{st.title}» не прошёл: {error}")


def _run_op(op: Op, remote: Remote, out: Callable[[str], None]) -> None:
    out(f"  {op.describe()}")
    if isinstance(op, Cmd):
        rc, stdout, stderr = remote.run(op.cmd)
        if stdout.strip():
            out("    " + stdout.strip().replace("\n", "\n    "))
        if rc != 0 and not op.tolerate:
            raise DeployError(f"rc={rc}: {(stderr or stdout).strip()[-400:]}")
        if op.verify is not None:
            problem = op.verify(stdout)
            if problem:
                raise DeployError(problem)
    elif isinstance(op, Write):
        text = op.content() if callable(op.content) else op.content
        remote.write(op.path, text, op.mode)
        if op.chgrp:
            rc, _, err = remote.run(f"chgrp {op.chgrp} {op.path}")
            if rc != 0:
                raise DeployError(f"chgrp {op.path}: {err.strip()}")
    elif isinstance(op, LocalCheck):
        op.fn()
    else:
        remote.put(op.make_local(), op.remote)


# --- local inputs -----------------------------------------------------------------------------


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def local_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Process env over the repo's .env, which is where the owner's keys live."""
    merged = load_dotenv(REPO / ".env")
    merged.update(environ if environ is not None else os.environ)
    return merged


def require_key(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise DeployError(f"{name} не задан (ни в окружении, ни в .env): ключи берутся локально")
    return value


def build_keys_json(env: Mapping[str, str]) -> str:
    """Hashes only. Raw keys never enter this text."""
    records = [
        {"name": name, "sha256": hash_key(require_key(env, var)), "rpm": rpm, "burst": burst}
        for name, var, rpm, burst in KEY_SPECS
    ]
    return json.dumps(records, indent=2) + "\n"


def build_gateway_json(version: str) -> str:
    return json.dumps({"version": version, "keys_file": KEYS_PATH}, indent=2) + "\n"


def gateway_files() -> list[Path]:
    return sorted(
        p
        for p in GATEWAY_DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".log"}
    )


def content_digest() -> str:
    h = hashlib.sha256()
    for p in gateway_files():
        h.update(p.relative_to(GATEWAY_DIR).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


def default_version() -> str:
    """Git short sha; a dirty gateway dir gets a content hash so two builds never collide."""

    def git(*args: str) -> str:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise DeployError(f"git {' '.join(args)}: {r.stderr.strip()}")
        return r.stdout.strip()

    sha = git("rev-parse", "--short=12", "HEAD")
    if git("status", "--porcelain", "--", "week_06/gateway"):
        return f"{sha}-{content_digest()}"
    return sha


def build_release_tar() -> Path:
    """app/week_06/{__init__.py,gateway/**} + requirements.txt, in a temp file."""
    tmp = Path(tempfile.mkdtemp(prefix="advent-release-")) / "release.tgz"
    with tarfile.open(tmp, "w:gz") as tar:
        init = b'"""Week 06 (server copy: gateway only)."""\n'
        info = tarfile.TarInfo("app/week_06/__init__.py")
        info.size = len(init)
        tar.addfile(info, io.BytesIO(init))
        for p in gateway_files():
            rel = p.relative_to(GATEWAY_DIR).as_posix()
            tar.add(p, arcname=f"app/week_06/gateway/{rel}")
        tar.add(GATEWAY_DIR / "requirements.txt", arcname="requirements.txt")
    return tmp


# --- step builders ----------------------------------------------------------------------------


def _int_at_least(minimum: int, what: str) -> Callable[[str], str | None]:
    def check(out: str) -> str | None:
        try:
            value = int(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return f"{what}: не удалось прочитать число из «{out.strip()}»"
        return None if value >= minimum else f"{what}: {value} МБ < {minimum} МБ"

    return check


def preflight_step(model_sha256: str | None = None) -> Step:
    model = f"{MODELS}/{MODEL_FILE}"
    expected = (model_sha256 or MODEL_SHA256).lower()

    def sha_check(out: str) -> str | None:
        got = out.split()[0].lower() if out.split() else ""
        return None if got == expected else f"sha256 модели {got or '?'} не совпал с {expected}"

    def size_check(out: str) -> str | None:
        got = out.strip().splitlines()[-1] if out.strip() else ""
        return None if got == str(MODEL_BYTES) else f"размер модели {got or '?'} != {MODEL_BYTES}"

    return Step(
        "Проверки перед установкой",
        [
            Cmd(
                "awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo",
                verify=_int_at_least(MIN_AVAILABLE_MB, "свободная RAM"),
            ),
            Cmd(
                "df --output=avail -BM /opt | tail -1 | tr -dc 0-9",
                verify=_int_at_least(MIN_DISK_MB, "свободный диск"),
            ),
            Cmd(f"test -x {LLAMA_BIN}"),
            Cmd(f"test -f {LIBGOMP}"),
            Cmd(f"test -f {model}"),
            Cmd(f"stat -c %s {model}", verify=size_check),
            Cmd(f"sha256sum {model}", verify=sha_check),
        ],
        mutating=False,
    )


def system_step() -> Step:
    return Step(
        "Системный пользователь и каталоги",
        [
            Cmd(
                f"id {SYS_USER} >/dev/null 2>&1 || useradd --system --user-group "
                f"--no-create-home --home-dir /var/lib/advent-llm "
                f"--shell /usr/sbin/nologin {SYS_USER}"
            ),
            Cmd(f"install -d -o {SYS_USER} -g {SYS_USER} -m 750 /var/lib/advent-llm"),
            Cmd(f"install -d -o root -g {SYS_USER} -m 750 /etc/advent-llm"),
            Cmd(f"install -d -m 755 {BASE} {RELEASES}"),
            Cmd(f"chmod -R a+rX {BASE}/bin {BASE}/lib {MODELS}"),
        ],
    )


def cleanup_step() -> Step:
    """Probe leftovers (spec section 8-7): dl/ and the two unselected models."""
    return Step(
        "Уборка остатков probe: dl/ и две невыбранные модели",
        [
            Cmd(f"rm -rf {BASE}/dl"),
            Cmd(
                f"find {MODELS} -maxdepth 1 -type f "
                r"\( -iname 'qwen3-0.6b*' -o -iname 'gemma-3-270m*' \) -print -delete"
            ),
        ],
    )


def runtime_step(uv_version: str | None) -> Step:
    url = (
        f"https://astral.sh/uv/{uv_version}/install.sh"
        if uv_version
        else "https://astral.sh/uv/install.sh"
    )
    env = f"UV_PYTHON_INSTALL_DIR={PY_DIR}"
    return Step(
        "uv и Python 3.12 в /opt/advent-llm",
        [
            Cmd(f"test -x {UV_BIN} || (curl -LsSf {url} | env UV_UNMANAGED_INSTALL={UV_DIR} sh)"),
            Cmd(f"{env} {UV_BIN} python install 3.12"),
            Cmd(f"chmod -R a+rX {PY_DIR} {UV_DIR}"),
        ],
    )


def keys_step(env: Mapping[str, str]) -> Step:
    return Step(
        "keys.json из sha256 локальных ключей",
        [Write(KEYS_PATH, lambda: build_keys_json(env), "640", chgrp=SYS_USER)],
    )


def release_digest(version: str) -> str:
    """Manifest of a release: version + every shipped file, so an equal hash means equal content."""
    h = hashlib.sha256(version.encode())
    for p in gateway_files():
        h.update(p.relative_to(GATEWAY_DIR).as_posix().encode())
        h.update(p.read_bytes())
    h.update((GATEWAY_DIR / "requirements.txt").read_bytes())
    return h.hexdigest()


def _import_check(root: str) -> Cmd:
    py = f"{root}/venv/bin/python"
    # the service user must be able to import it, not only root
    return Cmd(
        f"cd {root}/app && runuser -u {SYS_USER} -- env PYTHONPATH={root}/app "
        f"{py} -c 'import week_06.gateway.app'"
    )


def release_build_steps(version: str, stage_id: str | None = None) -> list[Step]:
    """Build in a unique staging dir, validate, rename into an immutable releases/<version>.

    `current` may point at releases/<version>: that directory is never written to.
    """
    rel = f"{RELEASES}/{version}"
    stage = f"{RELEASES}/.staging-{version}-{stage_id or uuid.uuid4().hex[:8]}"
    tgz = f"/tmp/advent-llm-{stage.rsplit('-', 1)[1]}-{version}.tgz"
    py = f"{stage}/venv/bin/python"
    env = f"UV_PYTHON_INSTALL_DIR={PY_DIR}"
    finalize = (
        f"if [ -e {rel} ]; then cmp -s {stage}/MANIFEST {rel}/MANIFEST || "
        f'{{ echo "релиз {version} уже есть с другим содержимым" >&2; exit 1; }}; '
        f'echo "релиз {version} уже есть, содержимое совпало"; '
        f'else mv -T {stage} {rel} && echo "релиз {version} готов"; fi'
    )
    return [
        Step(
            f"Релиз {version}: staging, код, venv, зависимости",
            [
                Cmd(f"mkdir -p {RELEASES} && mkdir {stage}"),
                Put("release.tgz", build_release_tar, tgz),
                Cmd(f"tar -xzf {tgz} -C {stage} && rm -f {tgz}"),
                Write(f"{stage}/gateway.json", build_gateway_json(version)),
                Write(f"{stage}/MANIFEST", release_digest(version) + "\n"),
                Cmd(f"{env} {UV_BIN} venv --python 3.12 {stage}/venv"),
                Cmd(f"{UV_BIN} pip install --python {py} -r {stage}/requirements.txt"),
                Cmd(f"chown -R root:root {stage} && chmod -R a+rX {stage}"),
                _import_check(stage),
                Cmd(finalize),
                _import_check(rel),
                Cmd(f"rm -rf {stage}", always=True, tolerate=True),
            ],
        ),
        Step(
            f"Переключение current -> {version} (атомарно)",
            [Cmd(f"ln -sfn {rel} {BASE}/current.new && mv -Tf {BASE}/current.new {CURRENT}")],
        ),
    ]


def unit_step() -> Step:
    return Step(
        "systemd unit: установить, enable, restart",
        [
            Write(UNIT_PATH, (HERE / "advent-llm.service").read_text(encoding="utf-8")),
            Cmd("systemctl daemon-reload"),
            Cmd(f"systemctl enable {SERVICE}"),
            Cmd(f"systemctl restart {SERVICE}"),
        ],
    )


def restart_step() -> Step:
    return Step(
        "Явный restart сервиса",
        [Cmd(f"systemctl restart {SERVICE}"), Cmd(f"systemctl is-active {SERVICE}")],
    )


def _split_http(out: str) -> tuple[str, str]:
    """(body, code) of curl output ending in `-w '\\n%{http_code}'`."""
    body, _, code = out.rstrip("\n").rpartition("\n")
    return body, code.strip()


def _completion_ok(out: str) -> str | None:
    """HTTP 200 and a parsed chat completion: content, usage numbers, finish_reason."""
    body, code = _split_http(out)
    if code != "200":
        return f"генерация: HTTP {code or '?'}, ожидался 200"
    try:
        data = json.loads(body)
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data["usage"]
    except (ValueError, KeyError, IndexError, TypeError):
        return "ответ генерации не разобрался как chat completion"
    if not isinstance(content, str) or not content.strip():
        return "пустой content в ответе"
    if not choice.get("finish_reason"):
        return "в ответе нет finish_reason"
    for field_name in ("prompt_tokens", "completion_tokens"):
        n = usage.get(field_name) if isinstance(usage, dict) else None
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            return f"usage.{field_name} не число: {n!r}"
    return None


def _version_ok(out: str) -> str | None:
    """First line is the release dir name, the rest is /v1/status; compare for real."""
    release, _, body = out.strip().partition("\n")
    try:
        got = json.loads(body).get("version")
    except (ValueError, AttributeError):
        return "/v1/status не разобрался как JSON"
    return None if release and got == release else f"версия {got!r} != релиз {release!r}"


def _http_code(expected: str) -> Callable[[str], str | None]:
    def check(out: str) -> str | None:
        got = out.strip().splitlines()[-1] if out.strip() else ""
        return None if got == expected else f"HTTP {got or '?'}, ожидался {expected}"

    return check


def _radio_check(out: str) -> str | None:
    if "gen-not-started" in out:
        return "генерация не началась: замеры радио не во время генерации"
    times: list[float] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            if parts[0] != "200":
                return f"радио ответило {parts[0]} во время генерации"
            times.append(float(parts[1]))
    if len(times) < RADIO_SAMPLES:
        return f"снято {len(times)} замеров радио из {RADIO_SAMPLES}"
    if "gen-alive" not in out:
        return "генерация закончилась раньше замеров: p95 не относится к работе модели"
    p95 = sorted(times)[int(len(times) * 0.95) - 1]
    return None if p95 <= 0.5 else f"p95 статики радио {p95:.2f} с > 0.5 с"


def _run_local_smoke(manager_dir: str) -> None:
    """VPS_Manager's tools/smoke.py (read-only): the neighbouring services must stay green."""
    path = Path(manager_dir) / "tools" / "smoke.py"
    if not path.is_file():
        raise DeployError(f"в VPS_MANAGER_DIR нет tools/smoke.py: {path}")
    r = subprocess.run(
        [sys.executable, str(path)],
        cwd=path.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    tail = (r.stdout + r.stderr).strip()[-600:]
    print("    " + tail.replace("\n", "\n    "))
    if r.returncode != 0:
        raise DeployError(f"smoke.py соседних сервисов: rc={r.returncode}")


def smoke_steps(
    env: Mapping[str, str],
    origin: str = PUBLIC_ORIGIN,
    http_get: HttpGet | None = None,
    radio_url: str | None = None,
    manager_dir: str | None = None,
) -> list[Step]:
    """Spec 10-30. The raw owner key exists only in the temp curl config (removed after)."""
    get = http_get or _urllib_get
    curl = f"curl -sS -m 160 -K {SMOKE_CURLRC}"
    short = {"messages": [{"role": "user", "content": "Скажи одно слово: привет"}]}
    gen_body = json.dumps({**short, "max_tokens": 16}, ensure_ascii=False)
    # long enough to outlast the radio samples; max_tokens_ceiling is 512
    long_msg = {"role": "user", "content": "Считай вслух от 1 до 300, каждое число словами."}
    long_body = json.dumps({"messages": [long_msg], "max_tokens": 400}, ensure_ascii=False)

    def external(expected: int, with_key: bool) -> Callable[[], None]:
        def run() -> None:
            key = require_key(env, "ADVENT_SERVICE_KEY") if with_key else None
            code = get(f"{origin}/health", key)
            if code != expected:
                raise DeployError(f"{origin}/health: HTTP {code}, ожидался {expected}")

        return run

    # authenticated /health with a bounded deadline: Type=simple does not wait for the bind
    ready = (
        f"end=$(( $(date +%s) + {READY_DEADLINE_S} )); "
        f"until [ \"$(curl -sS -m 3 -K {SMOKE_CURLRC} -o /dev/null -w '%{{http_code}}' "
        f'{LOOPBACK}/health 2>/dev/null)" = 200 ]; do '
        f"[ $(date +%s) -ge $end ] && "
        f"{{ echo 'сервис не ответил на /health за {READY_DEADLINE_S} с' >&2; exit 1; }}; "
        f"sleep 1; done; echo ready"
    )
    version_check = f'basename "$(readlink {CURRENT})"; {curl} {LOOPBACK}/v1/status'
    steps = [
        Step(
            "Smoke 1-2: loopback /health, версия, одна cold-генерация",
            [
                Write(
                    SMOKE_CURLRC,
                    lambda: (
                        f'header = "Authorization: Bearer {require_key(env, KEY_SPECS[0][1])}"\n'
                    ),
                    "600",
                ),
                Cmd(ready),
                Cmd(
                    f"{curl} -o /dev/null -w '%{{http_code}}' {LOOPBACK}/health",
                    verify=_http_code("200"),
                ),
                Cmd(version_check, verify=_version_ok),
                Cmd(
                    f"{curl} -H 'Content-Type: application/json' -w '\\n%{{http_code}}' "
                    f"-d '{gen_body}' {LOOPBACK}/v1/chat/completions",
                    verify=_completion_ok,
                ),
            ],
        ),
    ]
    if radio_url:
        started = f"curl -s -m 3 -K {SMOKE_CURLRC} {LOOPBACK}/v1/status | grep -Eq '\"active\": *1'"
        loop = (
            f"({curl} -o /dev/null -H 'Content-Type: application/json' -d '{long_body}' "
            f"{LOOPBACK}/v1/chat/completions) & gen=$!; "
            f"ok=0; for i in $(seq 30); do {started} && ok=1 && break; sleep 0.5; done; "
            f"[ $ok = 1 ] || echo gen-not-started; "
            f"for i in $(seq {RADIO_SAMPLES}); do "
            f"curl -s -o /dev/null -m 10 -w '%{{http_code}} %{{time_total}}\\n' {radio_url}; "
            f"done; kill -0 $gen 2>/dev/null && echo gen-alive; wait $gen"
        )
        steps[0].ops.append(Cmd(loop, verify=_radio_check))
    steps[0].ops.append(Cmd(f"rm -f {SMOKE_CURLRC}", always=True, tolerate=True))
    steps.append(
        Step(
            "Smoke 3: снаружи без ключа 401, с ключом 200",
            [
                LocalCheck("без ключа -> 401", external(401, False)),
                LocalCheck("с ключом owner -> 200", external(200, True)),
            ],
        )
    )
    if manager_dir:
        steps.append(
            Step(
                "Smoke 4: соседние сервисы (VPS_Manager tools/smoke.py)",
                [LocalCheck("smoke.py", lambda: _run_local_smoke(manager_dir))],
                mutating=False,
            )
        )
    return steps


def _urllib_get(url: str, key: str | None) -> int:
    # httpx, not urllib: the project's offline guard only covers httpx.
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.get(url, headers=headers, timeout=20, follow_redirects=False).status_code


def caddy_note() -> str:
    return (
        "Caddy этим скриптом НЕ меняется. Добавь week_06/deploy/Caddyfile.advent-llm в зеркало "
        "конфигов VPS_Manager и залей его push_config.py (backup, validate, reload, rollback)."
    )


def _smoke_kwargs(args: argparse.Namespace, env: Mapping[str, str]) -> dict:
    return {
        "radio_url": None if args.no_radio else args.radio_url,
        "manager_dir": env.get("VPS_MANAGER_DIR") or None,
    }


def install_plan(
    args: argparse.Namespace, env: Mapping[str, str], version: str, http_get: HttpGet | None = None
) -> list[Step]:
    return [
        preflight_step(args.model_sha256),
        system_step(),
        cleanup_step(),
        runtime_step(args.uv_version),
        keys_step(env),
        *release_build_steps(version),
        unit_step(),
        *smoke_steps(env, args.origin, http_get, **_smoke_kwargs(args, env)),
    ]


def release_plan(
    args: argparse.Namespace, env: Mapping[str, str], version: str, http_get: HttpGet | None = None
) -> list[Step]:
    return [
        preflight_step(args.model_sha256),
        *release_build_steps(version),
        restart_step(),
        *smoke_steps(env, args.origin, http_get, **_smoke_kwargs(args, env)),
    ]


def rollback_plan(
    args: argparse.Namespace, env: Mapping[str, str], http_get: HttpGet | None = None
) -> list[Step]:
    pick = (
        f"cur=$(readlink {CURRENT}); prev=$(ls -1dt {RELEASES}/*/ | sed 's#/$##' | "
        f'grep -vxF "$cur" | head -1); '
        f'[ -n "$prev" ] || {{ echo "нет предыдущего релиза" >&2; exit 1; }}; '
        f'ln -sfn "$prev" {BASE}/current.new && mv -Tf {BASE}/current.new {CURRENT} '
        f'&& echo "-> $prev"'
    )
    return [
        Step("Откат: current -> предыдущий релиз", [Cmd(pick)]),
        restart_step(),
        *smoke_steps(env, args.origin, http_get, **_smoke_kwargs(args, env)),
    ]


def uninstall_plan() -> list[Step]:
    """Only what deploy created. bin/, lib/, the chosen model and probe/ stay."""
    return [
        Step(
            "Удаление созданного deploy'ем",
            [
                Cmd(f"systemctl disable --now {SERVICE}", tolerate=True),
                Cmd(f"rm -f {UNIT_PATH}"),
                Cmd("systemctl daemon-reload"),
                Cmd(f"rm -rf {RELEASES} {CURRENT} {BASE}/current.new {PY_DIR} {UV_DIR}"),
                Cmd("rm -rf /etc/advent-llm /var/lib/advent-llm"),
                Cmd(f"id {SYS_USER} >/dev/null 2>&1 && userdel {SYS_USER} || true"),
            ],
        )
    ]


def status_plan() -> list[Step]:
    return [
        Step(
            "Состояние сервиса (только чтение)",
            [
                Cmd(f"systemctl is-active {SERVICE}", tolerate=True),
                Cmd(f"readlink {CURRENT}", tolerate=True),
                Cmd(f"ls -1t {RELEASES}", tolerate=True),
                Cmd(f"systemctl show {SERVICE} -p MemoryCurrent -p NRestarts", tolerate=True),
            ],
            mutating=False,
        )
    ]


# --- CLI --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Деплой приватного LLM-сервиса (день 30)")
    ap.add_argument(
        "command",
        nargs="?",
        default="plan",
        choices=["plan", "install", "release", "rollback", "uninstall", "status"],
    )
    ap.add_argument("--yes", action="store_true", help="выполнить изменения на сервере")
    ap.add_argument("--dry-run", action="store_true", help="только план (по умолчанию)")
    ap.add_argument("--version", help="имя релиза (по умолчанию git sha)")
    ap.add_argument(
        "--model-sha256",
        default=os.environ.get("ADVENT_MODEL_SHA256") or MODEL_SHA256,
        help="эталон sha256 модели; несовпадение прерывает деплой",
    )
    ap.add_argument("--uv-version", help="закрепить версию uv установщика")
    ap.add_argument("--origin", default=PUBLIC_ORIGIN)
    ap.add_argument(
        "--radio-url",
        default=DEFAULT_RADIO_URL,
        help="URL статики радио для замера p95 во время генерации",
    )
    ap.add_argument("--no-radio", action="store_true", help="не мерить радио (только для отладки)")
    return ap


def main(
    argv: list[str] | None = None,
    *,
    remote_factory: Callable[[], Remote] | None = None,
    environ: Mapping[str, str] | None = None,
    out: Callable[[str], None] = print,
    http_get: HttpGet | None = None,
) -> int:
    # Windows consoles and pipes default to cp1252; the plan is Russian.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    env = local_env(environ)
    try:
        if args.command == "status":
            steps = status_plan()
        elif args.command == "uninstall":
            steps = uninstall_plan()
        elif args.command == "rollback":
            steps = rollback_plan(args, env, http_get)
        else:
            version = args.version or default_version()
            planner = release_plan if args.command == "release" else install_plan
            steps = planner(args, env, version, http_get)
        out(render_plan(steps))
        mutating = any(s.mutating for s in steps)
        if args.command == "plan" or args.dry_run or (mutating and not args.yes):
            if args.command != "plan" and not args.dry_run:
                out("\nПлан не выполнен: нужен явный --yes.")
            if args.command in ("plan", "install", "release"):
                out("\n" + caddy_note())
            return 0
        if remote_factory is None:
            manager = os.environ.get("VPS_MANAGER_DIR")
            if not manager:
                raise DeployError("задай VPS_MANAGER_DIR: каталог VPS_Manager с tools/vps.py")
            remote: Remote = VpsRemote(manager)
        else:
            remote = remote_factory()
        try:
            execute(steps, remote, out)
        finally:
            remote.close()
        if args.command in ("install", "release"):
            out("\n" + caddy_note())
        return 0
    except DeployError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
