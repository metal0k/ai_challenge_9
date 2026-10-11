"""Day 30 deploy tooling and recording scenario: no SSH, no network, no real server."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import typer.main

from advent_cli import record
from week_06 import cli
from week_06.deploy import deploy

# built at runtime so the public-repo guard does not flag the canary
OWNER_CANARY = "adv" + "_" + "CANARYownerKEY0123456789abcdefghijklmnopqrs"
DEMO_CANARY = "adv" + "_" + "CANARYdemoKEY00123456789abcdefghijklmnopqrst"
ENV = {"ADVENT_SERVICE_KEY": OWNER_CANARY, "ADVENT_SERVICE_DEMO_KEY": DEMO_CANARY}
DEPLOY_DIR = Path(deploy.__file__).parent


VERSION = "v1"
GOOD_COMPLETION = {
    "choices": [{"message": {"role": "assistant", "content": "Привет"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
}
RADIO_OK = "\n".join(["200 0.080"] * 20 + ["gen-alive"])
R = deploy.RELEASES

# strict allowlist: every command shape the plans may send; anything else fails the test
ALLOWED = [
    r"awk '/MemAvailable",
    r"df --output=avail",
    r"test -[xf] /opt/advent-llm/",
    r"stat -c %s ",
    r"sha256sum ",
    r"id advent-llm",
    r"install -d ",
    r"chmod -R a\+rX ",
    r"rm -rf /opt/advent-llm/dl$",
    r"find /opt/advent-llm/models ",
    r"test -x /opt/advent-llm/uv/uv \|\| ",
    r"UV_PYTHON_INSTALL_DIR=\S+ \S+ python install 3\.12$",
    r"chgrp advent-llm ",
    rf"mkdir -p {R} && mkdir {R}/\.staging-",
    r"tar -xzf /tmp/advent-llm-\S+\.tgz -C \S+ && rm -f ",
    r"UV_PYTHON_INSTALL_DIR=\S+ \S+ venv --python 3\.12 ",
    r"/opt/advent-llm/uv/uv pip install --python ",
    r"chown -R root:root ",
    r"cd \S+/app && runuser -u advent-llm -- env PYTHONPATH=",
    r"if \[ -e ",
    rf"rm -rf {R}/\.staging-",
    r"ln -sfn \S+ /opt/advent-llm/current\.new && mv -Tf ",
    r"systemctl (daemon-reload|enable advent-llm|restart advent-llm|is-active advent-llm)$",
    r"end=\$\(\( \$\(date \+%s\) \+ 30 \)\); until ",
    r"curl -sS -m 160 -K \S+ -o /dev/null -w '%\{http_code\}' \S+/health$",
    r'basename "\$\(readlink ',
    r"curl -sS -m 160 -K \S+ -H 'Content-Type: application/json' -w '\\n%\{http_code\}' ",
    r"\(curl -sS -m 160 -K \S+ -o /dev/null -H .* gen=\$!; ",
    r"rm -f /run/advent-llm-smoke\.curlrc$",
    r"cur=\$\(readlink ",
]


class FakeRemote:
    """Answers only the command shapes in ALLOWED, with output a real server would give."""

    def __init__(self, answers=None, *, version=VERSION, status_version=VERSION):
        self.cmds: list[str] = []
        self.writes: list[tuple[str, str, str]] = []
        self.puts: list[tuple[str, str]] = []
        self.closed = False
        self.answers = answers or {}
        self.version = version
        self.status_version = status_version

    def run(self, cmd):
        self.cmds.append(cmd)
        for needle, ans in self.answers.items():
            if needle in cmd:
                return ans
        if not any(re.match(pattern, cmd, re.DOTALL) for pattern in ALLOWED):
            raise AssertionError(f"unexpected command: {cmd[:200]}")
        if "MemAvailable" in cmd:
            return 0, "600\n", ""
        if cmd.startswith("df --output"):
            return 0, "20000\n", ""
        if cmd.startswith("stat -c %s"):
            return 0, f"{deploy.MODEL_BYTES}\n", ""
        if cmd.startswith("sha256sum"):
            return 0, f"{deploy.MODEL_SHA256}  model\n", ""
        if cmd.startswith("basename"):
            status = json.dumps({"version": self.status_version, "model": "ready"})
            return 0, f"{self.version}\n{status}", ""
        if "-w '\\n%{http_code}'" in cmd:
            return 0, json.dumps(GOOD_COMPLETION) + "\n200", ""
        if "gen=$!" in cmd:
            return 0, RADIO_OK, ""
        if "-w '%{http_code}'" in cmd:
            return 0, "200", ""
        return 0, "", ""

    def write(self, path, text, mode):
        self.writes.append((path, text, mode))

    def put(self, local, remote):
        assert Path(local).is_file()
        self.puts.append((str(local), remote))

    def close(self):
        self.closed = True


def run_main(argv, remote=None):
    lines: list[str] = []
    rc = deploy.main(
        argv,
        remote_factory=(lambda: remote) if remote is not None else None,
        environ=ENV,
        out=lines.append,
        http_get=lambda url, key: 200 if key else 401,
    )
    return rc, "\n".join(lines)


def _args(*argv):
    return deploy.build_parser().parse_args(list(argv))


def test_plan_needs_no_ssh_and_lists_every_stage():
    # no remote_factory and no VPS_MANAGER_DIR: touching SSH would fail the run
    rc, text = run_main(["plan", "--version", "v1"])
    assert rc == 0
    for needle in (
        "Проверки перед установкой",
        "Системный пользователь",
        "Уборка остатков probe",
        "uv и Python 3.12",
        "keys.json",
        "Релиз v1",
        "systemd unit",
        "Smoke 1-2",
        "Smoke 3",
        "Caddy этим скриптом НЕ меняется",
    ):
        assert needle in text
    assert "UV_PYTHON_INSTALL_DIR=/opt/advent-llm/python" in text
    assert "rm -rf /opt/advent-llm/dl" in text
    assert "qwen3-0.6b*" in text and "gemma-3-270m*" in text


def test_mutating_command_without_yes_is_only_a_plan():
    remote = FakeRemote()
    rc, text = run_main(["install", "--version", "v1"], remote)
    assert rc == 0
    assert "нужен явный --yes" in text
    assert remote.cmds == [] and remote.writes == []


def test_install_with_yes_runs_in_order_and_closes_the_connection():
    remote = FakeRemote()
    rc, _ = run_main(["install", "--yes", "--version", "v1"], remote)
    assert rc == 0 and remote.closed
    joined = "\n".join(remote.cmds)
    order = [
        "MemAvailable",
        "useradd --system",
        "rm -rf /opt/advent-llm/dl",
        "python install 3.12",
        "venv --python 3.12 /opt/advent-llm/releases/.staging-v1-",
        "mv -T /opt/advent-llm/releases/.staging-v1-",
        "mv -Tf /opt/advent-llm/current.new /opt/advent-llm/current",
        "systemctl enable advent-llm",
        "systemctl restart advent-llm",
        "date +%s",
        "/v1/chat/completions",
    ]
    positions = [joined.index(x) for x in order]
    assert positions == sorted(positions)
    assert "enable --now" not in joined


def test_keys_json_has_hashes_only():
    text = deploy.build_keys_json(ENV)
    records = json.loads(text)
    assert records == [
        {"name": "owner", "sha256": deploy.hash_key(OWNER_CANARY), "rpm": 30, "burst": 6},
        {"name": "demo", "sha256": deploy.hash_key(DEMO_CANARY), "rpm": 3, "burst": 2},
    ]
    assert len(records[0]["sha256"]) == 64
    assert OWNER_CANARY not in text and DEMO_CANARY not in text


def test_missing_key_is_a_clear_error():
    with pytest.raises(deploy.DeployError, match="ADVENT_SERVICE_DEMO_KEY не задан"):
        deploy.build_keys_json({"ADVENT_SERVICE_KEY": OWNER_CANARY})


def test_raw_keys_never_appear_in_plan_or_commands_and_only_in_the_temp_curlrc():
    remote = FakeRemote()
    seen_http: list[tuple[str, str | None]] = []

    def fake_get(url, key):
        seen_http.append((url, key))
        return 200 if key else 401

    _, plan_text = run_main(["plan", "--version", "v1"])
    for canary in (OWNER_CANARY, DEMO_CANARY):
        assert canary not in plan_text
    steps = deploy.install_plan(_args("install"), ENV, "v1")
    # swap the smoke steps for ones with a fake external HTTP client
    steps = steps[:-2] + deploy.smoke_steps(ENV, http_get=fake_get)
    deploy.execute(steps, remote, out=lambda _: None)
    for cmd in remote.cmds:
        assert OWNER_CANARY not in cmd and DEMO_CANARY not in cmd
    holders = [w for w in remote.writes if OWNER_CANARY in w[1] or DEMO_CANARY in w[1]]
    assert [w[0] for w in holders] == [deploy.SMOKE_CURLRC]
    assert holders[0][2] == "600" and DEMO_CANARY not in holders[0][1]
    assert any(c.startswith(f"rm -f {deploy.SMOKE_CURLRC}") for c in remote.cmds)
    keys_write = next(w for w in remote.writes if w[0] == deploy.KEYS_PATH)
    assert OWNER_CANARY not in keys_write[1] and keys_write[2] == "640"
    assert seen_http == [
        ("https://advent-llm.duckdns.org/health", None),
        ("https://advent-llm.duckdns.org/health", OWNER_CANARY),
    ]


def test_smoke_cleanup_runs_even_when_generation_fails():
    remote = FakeRemote({"/v1/chat/completions": (0, "{}\n200", "")})
    steps = deploy.smoke_steps(ENV, http_get=lambda u, k: 200)
    with pytest.raises(deploy.DeployError, match="chat completion"):
        deploy.execute(steps, remote, out=lambda _: None)
    assert any(c.startswith("rm -f /run/advent-llm-smoke.curlrc") for c in remote.cmds)


def test_failed_preflight_stops_before_any_write():
    remote = FakeRemote({"MemAvailable": (0, "300\n", "")})
    rc, _ = run_main(["install", "--yes", "--version", "v1"], remote)
    assert rc == 1
    assert remote.writes == [] and not any("useradd" in c for c in remote.cmds)


def test_external_smoke_fails_when_unauthenticated_request_is_accepted():
    steps = deploy.smoke_steps(ENV, http_get=lambda u, k: 200)
    with pytest.raises(deploy.DeployError, match="HTTP 200, ожидался 401"):
        deploy.execute(steps[-1:], FakeRemote(), out=lambda _: None)


def test_unit_file_directives():
    text = (DEPLOY_DIR / "advent-llm.service").read_text(encoding="utf-8")
    lines = {ln.strip() for ln in text.splitlines()}
    for directive in (
        "MemoryMax=780M",
        "MemorySwapMax=0",
        "CPUWeight=20",
        "Nice=10",
        "OOMPolicy=continue",
        "Restart=on-failure",
        "User=advent-llm",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "ReadWritePaths=/var/lib/advent-llm",
        "PrivateTmp=yes",
    ):
        assert directive in lines
    for banned in ("MemoryHigh", "CPUQuota"):
        assert banned not in text
    assert "-m week_06.gateway --config /opt/advent-llm/current/gateway.json" in text


def test_caddy_block():
    text = (DEPLOY_DIR / "Caddyfile.advent-llm").read_text(encoding="utf-8")
    assert "advent-llm.duckdns.org {" in text
    assert "reverse_proxy 127.0.0.1:8790" in text
    assert "max_size 64KB" in text
    assert "flush_interval" not in text


def test_uninstall_leaves_probe_artifacts_alone():
    text = "\n".join(op.describe() for st in deploy.uninstall_plan() for op in st.ops)
    for kept in ("/opt/advent-llm/bin", "/opt/advent-llm/lib", "/opt/advent-llm/models", "probe"):
        assert kept not in text
    assert "rm -rf /etc/advent-llm" in text


def test_rollback_picks_the_previous_release():
    steps = deploy.rollback_plan(_args("rollback"), ENV)
    text = "\n".join(op.describe() for st in steps for op in st.ops)
    assert "grep -vxF" in text and "ls -1dt /opt/advent-llm/releases/" in text
    assert "systemctl restart advent-llm" in text


def test_release_tar_contains_gateway_and_requirements_only():
    with tarfile.open(deploy.build_release_tar()) as tar:
        names = tar.getnames()
    assert "requirements.txt" in names
    assert "app/week_06/__init__.py" in names
    assert "app/week_06/gateway/app.py" in names
    assert not any("__pycache__" in n or n.endswith((".pyc", ".log")) for n in names)
    assert not any("advent_core" in n for n in names)


def test_plan_prints_russian_utf8_into_a_cp1252_pipe():
    # no PYTHONIOENCODING, PYTHONUTF8=0: the child's stdout falls back to the locale codepage
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "VPS_MANAGER_DIR")}
    env["PYTHONUTF8"] = "0"
    r = subprocess.run(
        [sys.executable, "week_06/deploy/deploy.py", "plan"],
        cwd=deploy.REPO,
        env=env,
        capture_output=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "Проверки" in r.stdout.decode("utf-8")


def test_no_local_absolute_paths_in_deploy_files():
    # built at runtime so the public-repo guard does not flag the needles
    drive = "D:" + "\\"
    users = "C:" + "\\" + "Users"
    for path in DEPLOY_DIR.iterdir():
        if path.suffix in {".py", ".md", ".service"} or path.name.startswith("Caddyfile"):
            body = path.read_text(encoding="utf-8")
            assert drive not in body and users not in body, path.name


# --- recording scenario -----------------------------------------------------------------------


def _parse(args: list[str]):
    """Resolve a step's args through the real typer tree; an unknown command or flag fails."""
    cmd = typer.main.get_command(cli.app)
    rest = list(args)
    while rest and hasattr(cmd, "get_command") and not rest[0].startswith("-"):
        name = rest.pop(0)
        sub = cmd.get_command(None, name)
        assert sub is not None, f"нет команды {name}"
        cmd = sub
    return cmd.make_context(cmd.name or "x", rest)


def test_demo_steps_w06d30_shape_and_flags_parse():
    steps = record.demo_steps(6, 30)
    assert [s.title.split(".")[0] for s in steps] == ["0", "1", "2", "3", "4", "5"]
    assert steps[0].args == ["service", "status", "--wait-stopped", "120"]
    assert steps[-1].args == ["service", "status", "--wait-stopped", "120"]
    assert steps[2].args[:4] == ["service", "stress", "-n", "6"]
    assert steps[3].args[-2:] == ["--key-env", "ADVENT_SERVICE_DEMO_KEY"]
    assert steps[4].args == ["service", "overflow"]
    assert len(steps[1].stdin_lines) == 3 and steps[1].stdin_lines[-1] == "/exit"
    for step in steps:
        assert step.module == "week_06.cli"
        _parse(step.args)
    assert _parse(steps[0].args).params["wait_stopped"] == 120.0
    assert _parse(steps[2].args).params["n"] == 6
    assert _parse(steps[3].args).params["n"] == 4
    assert _parse(steps[3].args).params["key_env"] == "ADVENT_SERVICE_DEMO_KEY"


def test_unknown_flag_is_caught_by_the_parser_helper():
    with pytest.raises(Exception, match="No such option"):
        _parse(["service", "status", "--no-such-flag"])


def test_wait_stopped_timeouts_clear_idle_unload():
    steps = record.demo_steps(6, 30)
    # idle_seconds is 60; each step must outlive its own 120 s wait
    assert steps[0].timeout > 120 and steps[-1].timeout > 120
    assert steps[2].timeout >= 300


def test_rehearsal_steps_w06d30_parse():
    steps = record.rehearsal_steps(6, 30)
    assert len(steps) == 3
    for step in steps:
        assert step.module == "week_06.cli"
        _parse(step.args)
    assert steps[2].stdin_lines[-1] == "/exit"
    assert any(ord(ch) > 0x400 for ch in steps[2].stdin_lines[0])


# --- review fixes: #4 immutable releases, #11 smoke, #12 readiness, preflight ------------------


def test_fake_remote_rejects_unexpected_commands():
    with pytest.raises(AssertionError, match="unexpected command"):
        FakeRemote().run("echo hello")


def _build_ops(version="v1", stage_id="abc"):
    return [op for st in deploy.release_build_steps(version, stage_id) for op in st.ops]


def test_release_is_built_in_staging_and_never_written_in_place():
    ops = _build_ops()
    rel, stage = f"{deploy.RELEASES}/v1", f"{deploy.RELEASES}/.staging-v1-abc"
    fin = next(i for i, o in enumerate(ops) if isinstance(o, deploy.Cmd) and "mv -T " in o.cmd)
    for op in ops[:fin]:
        text = op.cmd if isinstance(op, deploy.Cmd) else getattr(op, "path", "")
        if "runuser" in text:
            continue
        # before the rename nothing may name the final directory
        assert rel not in text.replace(stage, ""), text
    writes = [o.path for o in ops if isinstance(o, deploy.Write)]
    assert writes and all(w.startswith(stage + "/") for w in writes)
    # the staging dir is removed even when a step failed
    last = deploy.release_build_steps("v1", "abc")[0].ops[-1]
    assert last.cmd == f"rm -rf {stage}" and last.always and last.tolerate
    # validated twice as the service user: in staging and at the final path
    checks = [o.cmd for o in ops if isinstance(o, deploy.Cmd) and "runuser" in o.cmd]
    assert len(checks) == 2 and stage in checks[0] and rel in checks[1]


def test_two_builds_of_one_version_get_distinct_staging_dirs():
    a = deploy.release_build_steps("v1")[0].ops[0].cmd
    b = deploy.release_build_steps("v1")[0].ops[0].cmd
    assert a != b


def _bash():
    import shutil
    import subprocess

    exe = shutil.which("bash")
    if not exe:
        pytest.skip("no bash")
    probe = subprocess.run(
        [exe, "-c", "echo ok; command -v cmp mv"], capture_output=True, text=True
    )
    if probe.returncode != 0 or "ok" not in probe.stdout or probe.stdout.count("\n") < 3:
        pytest.skip("bash without coreutils")
    return exe


def _run_finalize(tmp_path, *, existing: str | None, staged: str = "H1"):
    import subprocess

    exe = _bash()
    base = tmp_path.as_posix()
    stage = tmp_path / "releases" / ".staging-v1-abc"
    stage.mkdir(parents=True)
    (stage / "MANIFEST").write_text(staged)
    (stage / "marker").write_text("new")
    if existing is not None:
        (tmp_path / "releases" / "v1").mkdir()
        (tmp_path / "releases" / "v1" / "MANIFEST").write_text(existing)
        (tmp_path / "releases" / "v1" / "marker").write_text("old")
    cmd = next(
        o.cmd for o in _build_ops(stage_id="abc") if isinstance(o, deploy.Cmd) and "mv -T " in o.cmd
    )
    cmd = cmd.replace(deploy.BASE, base)
    r = subprocess.run([exe, "-c", cmd], capture_output=True, text=True, encoding="utf-8")
    return r, tmp_path / "releases" / "v1"


def test_finalize_renames_a_new_version(tmp_path):
    r, rel = _run_finalize(tmp_path, existing=None)
    assert r.returncode == 0 and (rel / "marker").read_text() == "new"
    assert not (tmp_path / "releases" / ".staging-v1-abc").exists()


def test_finalize_refuses_an_existing_version_with_other_content(tmp_path):
    r, rel = _run_finalize(tmp_path, existing="H0")
    assert r.returncode == 1 and "уже есть с другим содержимым" in r.stderr
    assert (rel / "marker").read_text() == "old"  # the live release is untouched


def test_finalize_accepts_an_existing_version_with_the_same_manifest(tmp_path):
    r, rel = _run_finalize(tmp_path, existing="H1")
    assert r.returncode == 0 and "содержимое совпало" in r.stdout
    assert (rel / "marker").read_text() == "old"


def test_release_manifest_changes_with_version_and_is_written():
    assert deploy.release_digest("v1") != deploy.release_digest("v2")
    ops = _build_ops()
    manifest = [o for o in ops if isinstance(o, deploy.Write) and o.path.endswith("MANIFEST")]
    assert manifest[0].content == deploy.release_digest("v1") + "\n"


def test_service_is_restarted_then_polled_before_any_check():
    steps = deploy.install_plan(_args("install"), ENV, "v1")
    cmds = [op.cmd for st in steps for op in st.ops if isinstance(op, deploy.Cmd)]
    restart = cmds.index("systemctl restart advent-llm")
    poll = next(i for i, c in enumerate(cmds) if "date +%s" in c)
    first_check = next(i for i, c in enumerate(cmds) if "/v1/status" in c)
    gen = next(i for i, c in enumerate(cmds) if "/v1/chat/completions" in c)
    assert restart < poll < first_check < gen
    assert "-K /run/advent-llm-smoke.curlrc" in cmds[poll]  # authenticated
    assert "+ 30 ))" in cmds[poll] and "exit 1" in cmds[poll]  # bounded


def test_release_and_rollback_restart_before_the_poll():
    for steps in (
        deploy.release_plan(_args("release"), ENV, "v1"),
        deploy.rollback_plan(_args("rollback"), ENV),
    ):
        cmds = [op.cmd for st in steps for op in st.ops if isinstance(op, deploy.Cmd)]
        assert cmds.index("systemctl restart advent-llm") < next(
            i for i, c in enumerate(cmds) if "date +%s" in c
        )


def test_readiness_poll_gives_up_with_a_clear_error():
    remote = FakeRemote({"date +%s": (1, "", "сервис не ответил на /health за 30 с")})
    steps = deploy.smoke_steps(ENV, http_get=lambda u, k: 200)
    with pytest.raises(deploy.DeployError, match="не ответил на /health"):
        deploy.execute(steps, remote, out=lambda _: None)
    assert any(c.startswith("rm -f /run/advent-llm-smoke.curlrc") for c in remote.cmds)


def _completion(**patch):
    data = json.loads(json.dumps(GOOD_COMPLETION))
    for key, value in patch.items():
        if key == "content":
            data["choices"][0]["message"]["content"] = value
        elif key == "finish_reason":
            data["choices"][0]["finish_reason"] = value
        elif key == "usage":
            data["usage"] = value
    return json.dumps(data) + "\n200"


@pytest.mark.parametrize(
    ("out", "problem"),
    [
        (json.dumps(GOOD_COMPLETION) + "\n200", None),
        (_completion(content=""), "пустой content"),
        (_completion(content="   "), "пустой content"),
        (_completion(finish_reason=None), "finish_reason"),
        (_completion(usage={}), "usage.prompt_tokens"),
        (_completion(usage={"prompt_tokens": "7", "completion_tokens": 1}), "usage.prompt_tokens"),
        (_completion(usage={"prompt_tokens": 1, "completion_tokens": True}), "completion_tokens"),
        ('{"choices": []}\n200', "chat completion"),
        ("<html>\n200", "chat completion"),
        (json.dumps(GOOD_COMPLETION) + "\n503", "HTTP 503"),
        ("", "HTTP ?"),
    ],
)
def test_completion_verdicts(out, problem):
    got = deploy._completion_ok(out)
    assert (got is None) if problem is None else (problem in (got or ""))


def test_version_is_really_compared():
    status = json.dumps({"version": "v1"})
    assert deploy._version_ok(f"v1\n{status}") is None
    assert "v1" in deploy._version_ok(f"v2\n{status}")
    assert deploy._version_ok("v1\nnot json") is not None
    assert deploy._version_ok("") is not None


def test_smoke_fails_on_version_mismatch_and_empty_generation():
    steps = deploy.smoke_steps(ENV, http_get=lambda u, k: 200 if k else 401)
    with pytest.raises(deploy.DeployError, match="версия"):
        deploy.execute(steps, FakeRemote(status_version="old"), out=lambda _: None)
    bad = {"/v1/chat/completions": (0, _completion(content=""), "")}
    with pytest.raises(deploy.DeployError, match="пустой content"):
        deploy.execute(steps, FakeRemote(bad), out=lambda _: None)


def test_radio_gate_needs_20_ok_samples_and_a_generation_still_running():
    assert deploy._radio_check(RADIO_OK) is None
    assert "закончилась раньше" in deploy._radio_check(RADIO_OK.replace("gen-alive", ""))
    assert "не началась" in deploy._radio_check("gen-not-started\n" + RADIO_OK)
    assert "19 замеров" in deploy._radio_check("\n".join(["200 0.1"] * 19 + ["gen-alive"]))
    assert "502" in deploy._radio_check("\n".join(["502 0.1"] + ["200 0.1"] * 19 + ["gen-alive"]))
    assert "p95" in deploy._radio_check("\n".join(["200 0.9"] * 20 + ["gen-alive"]))


def test_radio_sampling_runs_against_a_long_generation_and_is_on_by_default():
    args = _args("install")
    assert args.radio_url == deploy.DEFAULT_RADIO_URL and not args.no_radio
    steps = deploy.smoke_steps(ENV, radio_url=args.radio_url)
    loop = next(op.cmd for op in steps[0].ops if isinstance(op, deploy.Cmd) and "gen=$!" in op.cmd)
    assert '"max_tokens": 400' in loop  # not the 16-token request
    assert loop.index("gen=$!") < loop.index("seq 20") < loop.index("kill -0")
    assert "wait $gen" in loop and "gen-alive" in loop


def test_smoke_py_of_neighbouring_services_runs_when_manager_dir_is_set(tmp_path, capsys):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "smoke.py").write_text("print('сервисы зелёные')\n", encoding="utf-8")
    env = {**ENV, "VPS_MANAGER_DIR": str(tmp_path)}
    steps = deploy.install_plan(_args("install"), env, "v1")
    assert steps[-1].title.startswith("Smoke 4") and not steps[-1].mutating
    deploy.execute(steps[-1:], FakeRemote(), out=lambda _: None)
    assert "сервисы зелёные" in capsys.readouterr().out
    (tools / "smoke.py").write_text("import sys; sys.exit(1)\n", encoding="utf-8")
    with pytest.raises(deploy.DeployError, match="smoke.py соседних сервисов: rc=1"):
        deploy.execute(steps[-1:], FakeRemote(), out=lambda _: None)


def test_smoke_py_step_is_absent_without_manager_dir():
    steps = deploy.install_plan(_args("install"), ENV, "v1")
    assert not any(st.title.startswith("Smoke 4") for st in steps)


def test_model_sha256_default_is_pinned_and_a_mismatch_aborts():
    assert deploy.MODEL_SHA256 == "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
    assert deploy.MODEL_BYTES == 491400032
    assert _args("install").model_sha256 == deploy.MODEL_SHA256
    remote = FakeRemote({"sha256sum": (0, "deadbeef  model\n", "")})
    rc, _ = run_main(["install", "--yes", "--version", "v1"], remote)
    assert rc == 1 and remote.writes == [] and not any("useradd" in c for c in remote.cmds)
    short = FakeRemote({"stat -c": (0, "123\n", "")})
    assert run_main(["install", "--yes", "--version", "v1"], short)[0] == 1


def test_default_external_get_ignores_proxy_env(monkeypatch):
    seen: dict = {}

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None):
            return type("R", (), {"status_code": 401})()

    monkeypatch.setattr(deploy.httpx, "Client", FakeClient)
    assert deploy._urllib_get("https://x/health", None) == 401
    assert seen["trust_env"] is False and seen["follow_redirects"] is False
