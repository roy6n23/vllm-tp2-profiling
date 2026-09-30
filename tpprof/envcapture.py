"""env.json: what the box was, for the record (spec 7.5, 8 row "Version drift").

Every tool is optional: a missing or failing tool leaves its key None and records why under
"errors", so the capture never fails. Secrets in the captured environment are redacted.
"""
from __future__ import annotations

import os
import platform
import re
import sys
import time

import tpprof
from tpprof import preflight
from tpprof.preflight import PreflightContext

PIP_TIMEOUT_S = 120
ENV_PREFIXES = ("VLLM_", "NCCL_", "HF_", "CUDA", "TPPROF_")
ENV_NAMES = frozenset({"PATH", "LD_LIBRARY_PATH"})
_SECRET = re.compile(r"TOKEN|SECRET|PASSWORD|API_KEY|_KEY$")
REDACTED = "<redacted>"


def filtered_environ(env: dict[str, str]) -> dict[str, str]:
    """The VLLM_/NCCL_/HF_/CUDA*/TPPROF_ variables plus PATH and LD_LIBRARY_PATH, secrets redacted."""
    out = {}
    for key in sorted(env):
        if key in ENV_NAMES or key.startswith(ENV_PREFIXES):
            out[key] = REDACTED if _SECRET.search(key) else env[key]
    return out


def capture_env(ctx: PreflightContext) -> dict:
    """The env.json content for the box described by `ctx`."""
    env = preflight.environ(ctx)
    errors: dict[str, str] = {}

    def tool(key: str, argv: list[str], timeout_s: float = preflight.TOOL_TIMEOUT_S) -> str | None:
        out, err = preflight.tool_output(argv, env, timeout_s)
        if out is None:
            errors[key] = err
        return out

    def read(key: str, path: str) -> str | None:
        try:
            with open(path) as fh:
                return fh.read()
        except OSError as exc:
            errors[key] = f"cannot read {path}: {exc}"
            return None

    smi = ctx.nvidia_smi
    gpus, gpu_err = preflight.query_gpus(ctx)
    if gpu_err:
        errors["gpus"] = gpu_err
    drivers = sorted({g["driver_version"] for g in gpus})
    return {
        "t_wall": time.time(),
        "t_mono": time.monotonic(),
        "tpprof_version": tpprof.__version__,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "driver": ", ".join(drivers) if drivers else None,
        "gpu_names": [g["name"] for g in gpus],
        "nvidia_smi_q": tool("nvidia_smi_q", [smi, "-q"], preflight.NVIDIA_SMI_TIMEOUT_S),
        "topo_m": tool("topo_m", [smi, "topo", "-m"], preflight.NVIDIA_SMI_TIMEOUT_S),
        "nvlink_s": tool("nvlink_s", [smi, "nvlink", "-s"], preflight.NVIDIA_SMI_TIMEOUT_S),
        "monitor_field": preflight.monitor_field(ctx),
        "nsys_path": preflight.nsys_path(ctx),
        "nsys_version": tool("nsys_version", [preflight.nsys_path(ctx), "--version"]),
        "pip_freeze": tool("pip_freeze", [sys.executable, "-m", "pip", "freeze"], PIP_TIMEOUT_S),
        "uname": tool("uname", ["uname", "-a"]),
        "os_release": read("os_release", "/etc/os-release"),
        "lscpu": tool("lscpu", ["lscpu"]),
        "free_g": tool("free_g", ["free", "-g"]),
        "vllm_build_commit": env.get("VLLM_BUILD_COMMIT"),
        "environ": filtered_environ(env),
        "errors": errors,
    }
