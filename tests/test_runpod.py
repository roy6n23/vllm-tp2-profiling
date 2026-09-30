"""scripts/runpod.py against a local mock of the RunPod REST API; syntax and argv of the box scripts."""
from __future__ import annotations

import http.server
import importlib.util
import json
import os
import subprocess
import sys
import threading

import pytest

from tests.conftest import ROOT

SCRIPTS = ROOT / "scripts"
PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTests roy@mac"

# The pod-create body from the task brief, verbatim (JSON text, so the escapes are the brief's own).
EXPECTED_CREATE_JSON = r"""
{"name": "tp2prof", "imageName": "vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90",
 "gpuTypeIds": ["NVIDIA H100 80GB HBM3"], "gpuCount": 2, "cloudType": "SECURE", "allowedCudaVersions": ["13.0"],
 "containerDiskInGb": 80, "volumeInGb": 200, "volumeMountPath": "/workspace", "ports": ["22/tcp"],
 "dockerEntrypoint": ["bash", "-c"],
 "dockerStartCmd": ["set -e; export -p > /etc/profile.d/00-image-env.sh; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends openssh-server rsync git tmux curl ca-certificates zstd >/dev/null; mkdir -p /root/.ssh /run/sshd; printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity"],
 "env": {"PUBLIC_KEY": "<contents of pubkey file>"}}
"""


class MockRunPod:
    """Records every request; answers from per-method scripted responses (the last one repeats)."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.responses: dict[str, list[tuple[int, object]]] = {
            "POST": [(200, {"id": "pod123", "desiredStatus": "RUNNING"})],
            "GET": [(200, {"id": "pod123", "publicIp": "", "portMappings": {}})],
            "DELETE": [(200, None)],
        }

    def respond(self, method: str) -> tuple[int, object]:
        queue = self.responses[method]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def of(self, method: str) -> list[dict]:
        return [r for r in self.requests if r["method"] == method]


class _Handler(http.server.BaseHTTPRequestHandler):
    def _handle(self) -> None:
        mock: MockRunPod = self.server.mock  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        mock.requests.append({"method": self.command, "path": self.path,
                              "headers": {k.lower(): v for k, v in self.headers.items()},
                              "body": json.loads(raw) if raw else None})
        status, payload = mock.respond(self.command)
        data = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_DELETE = _handle

    def log_message(self, format: str, *args: object) -> None:  # keep test output clean
        pass


@pytest.fixture
def mock_api(monkeypatch):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.mock = MockRunPod()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    monkeypatch.setenv("RUNPOD_API_BASE", f"http://127.0.0.1:{server.server_address[1]}/v1")
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_test_key")
    yield server.mock  # type: ignore[attr-defined]
    server.shutdown()
    server.server_close()


@pytest.fixture
def runpod(monkeypatch):
    spec = importlib.util.spec_from_file_location("runpod_script", SCRIPTS / "runpod.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "POLL_INTERVAL_S", 0.0)
    return module


@pytest.fixture
def pubkey(tmp_path):
    path = tmp_path / "id_ed25519.pub"
    path.write_text(PUBKEY + "\n")
    return path


def test_create_posts_exactly_the_brief_body(runpod, mock_api, pubkey, capsys):
    rc = runpod.main(["create", "--pubkey", str(pubkey)])
    assert rc == 0
    (req,) = mock_api.of("POST")
    assert req["path"] == "/v1/pods"
    assert req["headers"]["authorization"] == "Bearer rpa_test_key"
    assert req["headers"]["content-type"] == "application/json"
    expected = json.loads(EXPECTED_CREATE_JSON)
    expected["env"]["PUBLIC_KEY"] = PUBKEY
    assert list(req["body"]) == list(expected)
    assert req["body"] == expected
    assert len(req["body"]["dockerStartCmd"]) == 1
    assert r"printf '%s\n' " in req["body"]["dockerStartCmd"][0]
    assert "pod123" in capsys.readouterr().out
    assert mock_api.of("GET") == [] and mock_api.of("DELETE") == []


def test_create_options_change_only_their_fields(runpod, mock_api, pubkey):
    rc = runpod.main(["create", "--pubkey", str(pubkey), "--gpus", "1", "--name", "p1",
                      "--volume-gb", "50", "--disk-gb", "40"])
    assert rc == 0
    body = mock_api.of("POST")[0]["body"]
    expected = json.loads(EXPECTED_CREATE_JSON)
    expected.update(name="p1", gpuCount=1, volumeInGb=50, containerDiskInGb=40)
    expected["env"]["PUBLIC_KEY"] = PUBKEY
    assert body == expected


def test_create_with_missing_pubkey_exits_2_without_calling_api(runpod, mock_api, tmp_path, capsys):
    rc = runpod.main(["create", "--pubkey", str(tmp_path / "nope.pub")])
    assert rc == 2
    assert "nope.pub" in capsys.readouterr().err
    assert mock_api.requests == []


def test_create_without_id_in_response_fails(runpod, mock_api, pubkey, capsys):
    mock_api.responses["POST"] = [(200, {"error": "no capacity"})]
    rc = runpod.main(["create", "--pubkey", str(pubkey)])
    assert rc == 1
    assert "no capacity" in capsys.readouterr().err


def test_wait_polls_until_ip_and_port_22_then_prints_ssh_line(runpod, mock_api, capsys):
    mock_api.responses["GET"] = [
        (200, {"id": "pod123", "publicIp": "", "portMappings": {}}),
        (200, {"id": "pod123", "publicIp": "203.0.113.7", "portMappings": None}),
        (200, {"id": "pod123", "publicIp": "203.0.113.7", "portMappings": {"22": 10341}}),
    ]
    rc = runpod.main(["wait", "pod123"])
    assert rc == 0
    gets = mock_api.of("GET")
    assert [g["path"] for g in gets] == ["/v1/pods/pod123"] * 3
    assert gets[0]["headers"]["authorization"] == "Bearer rpa_test_key"
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ssh -p 10341 root@203.0.113.7"


def test_wait_retries_through_a_server_error(runpod, mock_api, capsys):
    mock_api.responses["GET"] = [
        (503, {"error": "upstream"}),
        (200, {"id": "pod123", "publicIp": "203.0.113.7", "portMappings": {"22": 10341}}),
    ]
    assert runpod.main(["wait", "pod123"]) == 0
    assert len(mock_api.of("GET")) == 2
    assert "ssh -p 10341 root@203.0.113.7" in capsys.readouterr().out


def test_wait_timeout_prints_terminate_hint_and_exits_1(runpod, mock_api, capsys):
    rc = runpod.main(["wait", "pod123", "--timeout", "0"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "runpod.py terminate pod123" in captured.out + captured.err
    assert "ssh -p" not in captured.out


def test_wait_aborts_on_auth_error(runpod, mock_api, capsys):
    mock_api.responses["GET"] = [(401, {"error": "invalid api key"})]
    assert runpod.main(["wait", "pod123", "--timeout", "900"]) == 1
    assert len(mock_api.of("GET")) == 1
    assert "401" in capsys.readouterr().err


def test_status_prints_pod_json_and_ssh_line(runpod, mock_api, capsys):
    pod = {"id": "pod123", "desiredStatus": "RUNNING", "publicIp": "203.0.113.7", "portMappings": {"22": 10341}}
    mock_api.responses["GET"] = [(200, pod)]
    assert runpod.main(["status", "pod123"]) == 0
    out = capsys.readouterr().out
    assert '"desiredStatus": "RUNNING"' in out
    assert "ssh -p 10341 root@203.0.113.7" in out
    assert mock_api.of("GET")[0]["path"] == "/v1/pods/pod123"


def test_terminate_sends_delete(runpod, mock_api, capsys):
    assert runpod.main(["terminate", "pod123"]) == 0
    (req,) = mock_api.requests
    assert req["method"] == "DELETE"
    assert req["path"] == "/v1/pods/pod123"
    assert req["headers"]["authorization"] == "Bearer rpa_test_key"
    assert "pod123" in capsys.readouterr().out


def test_terminate_http_error_exits_1_with_status(runpod, mock_api, capsys):
    mock_api.responses["DELETE"] = [(404, {"error": "pod not found"})]
    assert runpod.main(["terminate", "pod123"]) == 1
    err = capsys.readouterr().err
    assert "404" in err and "pod not found" in err


@pytest.mark.parametrize("argv", [["create"], ["wait", "pod123"], ["status", "pod123"], ["terminate", "pod123"]])
def test_missing_api_key_exits_2_with_message(runpod, mock_api, monkeypatch, capsys, argv):
    monkeypatch.delenv("RUNPOD_API_KEY")
    assert runpod.main(argv) == 2
    assert "RUNPOD_API_KEY" in capsys.readouterr().err
    assert mock_api.requests == []


def test_script_runs_standalone_and_exits_2_without_key(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("RUNPOD_API_KEY", "PYTHONPATH")}
    proc = subprocess.run([sys.executable, str(SCRIPTS / "runpod.py"), "status", "pod123"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2
    assert "RUNPOD_API_KEY" in proc.stderr


@pytest.mark.parametrize("name", ["bootstrap_box.sh", "sync_to_box.sh"])
def test_box_scripts_pass_bash_syntax_check(name):
    proc = subprocess.run(["bash", "-n", str(SCRIPTS / name)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def _fake_rsync_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "rsync.jsonl"
    rsync = bin_dir / "rsync"
    rsync.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                     "with open(os.environ['FAKE_RSYNC_LOG'], 'a') as f:\n"
                     "    f.write(json.dumps({'cwd': os.getcwd(), 'argv': sys.argv[1:]}) + '\\n')\n")
    rsync.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", HOME=str(home),
               FAKE_RSYNC_LOG=str(log))
    return env, home, log


def _rsync_calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_sync_to_box_rsyncs_repo_with_brief_argv(tmp_path):
    env, _home, log = _fake_rsync_env(tmp_path)
    proc = subprocess.run(["bash", str(SCRIPTS / "sync_to_box.sh"), "203.0.113.7", "10341"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    (call,) = _rsync_calls(log)
    assert os.path.realpath(call["cwd"]) == os.path.realpath(ROOT)
    argv = call["argv"]
    assert argv[:7] == ["-az", "--delete", "--exclude", ".venv", "--exclude", "results/dryrun", "-e"]
    assert argv[7] == "ssh -p 10341"
    assert argv[-2:] == ["./", "root@203.0.113.7:/workspace/vllm-tp2-profiling/"]
    # A re-sync must never delete run records the box has produced under results/.
    assert "protect /results/" in argv
    assert "Liger" in proc.stdout


def test_sync_to_box_also_copies_triton_fa2_forward_when_present(tmp_path):
    env, home, log = _fake_rsync_env(tmp_path)
    fa2 = home / "Documents" / "Personal" / "triton-fa2-forward"
    fa2.mkdir(parents=True)
    proc = subprocess.run(["bash", str(SCRIPTS / "sync_to_box.sh"), "203.0.113.7", "10341"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    calls = _rsync_calls(log)
    assert len(calls) == 2
    argv = calls[1]["argv"]
    assert "ssh -p 10341" in argv
    assert argv[-2:] == [f"{fa2}/", "root@203.0.113.7:/workspace/triton-fa2-forward/"]


def test_sync_to_box_usage_error_without_args(tmp_path):
    env, _home, log = _fake_rsync_env(tmp_path)
    proc = subprocess.run(["bash", str(SCRIPTS / "sync_to_box.sh")], env=env,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()
    assert _rsync_calls(log) == []
