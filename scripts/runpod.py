#!/usr/bin/env python3
"""RunPod REST client for the rental box (spec 9.2, AM28). Standard library only.

    python3 scripts/runpod.py create [--gpus 2] [--name tp2prof] [--volume-gb 200] [--disk-gb 80]
                                     [--pubkey ~/.ssh/id_ed25519.pub]
    python3 scripts/runpod.py wait POD_ID [--timeout 900]
    python3 scripts/runpod.py status POD_ID
    python3 scripts/runpod.py terminate POD_ID

Needs RUNPOD_API_KEY in the environment. RUNPOD_API_BASE overrides the API base URL (tests).
Exit codes: 0 ok, 1 API error or wait timeout, 2 usage error (missing key or public key file).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpprof.constants import H100_SXM, VLLM_IMAGE  # noqa: E402

DEFAULT_API_BASE = "https://rest.runpod.io/v1"
POLL_INTERVAL_S = 10.0
HTTP_TIMEOUT_S = 30.0

# The image ENTRYPOINT is `vllm serve` (D9-3), so the pod runs `bash -c <START_CMD>` instead. SSH sessions
# do not inherit the Docker ENV, hence the export -p snapshot that bootstrap_box.sh sources (AM28).
START_CMD = (
    "set -e; export -p > /etc/profile.d/00-image-env.sh; apt-get update -qq; "
    "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
    "openssh-server rsync git tmux curl ca-certificates zstd >/dev/null; "
    "mkdir -p /root/.ssh /run/sshd; printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; "
    "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity"
)


class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        """Network failures and 5xx are worth another poll; 4xx (bad key, unknown pod) are not."""
        return self.status is None or self.status >= 500


def api_base() -> str:
    return os.environ.get("RUNPOD_API_BASE", DEFAULT_API_BASE).rstrip("/")


def request(method: str, path: str, api_key: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(api_base() + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").strip()[:2000]
        raise ApiError(f"{method} {path} -> HTTP {exc.code}: {detail}", status=exc.code) from None
    except (urllib.error.URLError, OSError) as exc:
        raise ApiError(f"{method} {path} failed: {getattr(exc, 'reason', exc)}") from None
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ApiError(f"{method} {path} returned non-JSON: {raw[:200]!r}") from None


def create_body(name: str, gpus: int, volume_gb: int, disk_gb: int, public_key: str) -> dict:
    return {
        "name": name,
        "imageName": VLLM_IMAGE,
        "gpuTypeIds": [H100_SXM.name],
        "gpuCount": gpus,
        "cloudType": "SECURE",
        "allowedCudaVersions": ["13.0"],   # guarantees a host driver >= 580 (D9-2)
        "containerDiskInGb": disk_gb,
        "volumeInGb": volume_gb,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [START_CMD],
        "env": {"PUBLIC_KEY": public_key},
    }


def ssh_target(pod: dict) -> tuple[str, int] | None:
    """(public IP, mapped port for 22) once both are known, else None."""
    ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    if ip and port:
        return str(ip), int(port)
    return None


def ssh_line(target: tuple[str, int]) -> str:
    ip, port = target
    return f"ssh -p {port} root@{ip}"


def terminate_hint(pod_id: str) -> str:
    return f"The pod is still billed until you run: python3 scripts/runpod.py terminate {pod_id}"


def cmd_create(args: argparse.Namespace, api_key: str) -> int:
    pubkey_path = pathlib.Path(args.pubkey).expanduser()
    if not pubkey_path.is_file():
        print(f"error: SSH public key not found: {pubkey_path} (pass --pubkey PATH)", file=sys.stderr)
        return 2
    public_key = pubkey_path.read_text().strip()
    body = create_body(args.name, args.gpus, args.volume_gb, args.disk_gb, public_key)
    pod = request("POST", "/pods", api_key, body)
    pod_id = pod.get("id")
    if not pod_id:
        raise ApiError(f"POST /pods returned no pod id: {json.dumps(pod)[:2000]}")
    print(f"created pod {pod_id}")
    print(f"next: python3 scripts/runpod.py wait {pod_id}")
    return 0


def cmd_wait(args: argparse.Namespace, api_key: str) -> int:
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            pod = request("GET", f"/pods/{args.pod_id}", api_key)
        except ApiError as exc:
            if not exc.retryable:
                raise
            print(f"poll failed, retrying: {exc}", file=sys.stderr)
            pod = {}
        target = ssh_target(pod)
        if target:
            print(ssh_line(target))
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"timed out after {args.timeout:g} s: pod {args.pod_id} has no public IP and port 22 "
                  f"mapping yet (publicIp={pod.get('publicIp')!r}, portMappings={pod.get('portMappings')!r})",
                  file=sys.stderr)
            print(terminate_hint(args.pod_id), file=sys.stderr)
            return 1
        if pod:
            print(f"waiting: desiredStatus={pod.get('desiredStatus')!r} publicIp={pod.get('publicIp')!r} "
                  f"portMappings={pod.get('portMappings')!r}", file=sys.stderr)
        time.sleep(min(POLL_INTERVAL_S, remaining))


def cmd_status(args: argparse.Namespace, api_key: str) -> int:
    pod = request("GET", f"/pods/{args.pod_id}", api_key)
    print(json.dumps(pod, indent=2, sort_keys=True))
    target = ssh_target(pod)
    if target:
        print(ssh_line(target))
    return 0


def cmd_terminate(args: argparse.Namespace, api_key: str) -> int:
    request("DELETE", f"/pods/{args.pod_id}", api_key)
    print(f"terminated pod {args.pod_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runpod.py", description="RunPod pod lifecycle for the tp2 box")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create the 2x H100 SXM pod")
    create.add_argument("--gpus", type=int, default=2)
    create.add_argument("--name", default="tp2prof")
    create.add_argument("--volume-gb", type=int, default=200)
    create.add_argument("--disk-gb", type=int, default=80)
    create.add_argument("--pubkey", default="~/.ssh/id_ed25519.pub")
    create.set_defaults(func=cmd_create)
    wait = sub.add_parser("wait", help="poll until SSH is reachable, then print the ssh command")
    wait.add_argument("pod_id")
    wait.add_argument("--timeout", type=float, default=900.0)
    wait.set_defaults(func=cmd_wait)
    status = sub.add_parser("status", help="print the pod JSON")
    status.add_argument("pod_id")
    status.set_defaults(func=cmd_status)
    terminate = sub.add_parser("terminate", help="delete the pod (and its /workspace volume)")
    terminate.add_argument("pod_id")
    terminate.set_defaults(func=cmd_terminate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("error: RUNPOD_API_KEY is not set; export RUNPOD_API_KEY=<your RunPod API key>", file=sys.stderr)
        return 2
    try:
        return args.func(args, api_key)
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
