"""Helpers shared by the fake executables in tests/fake_bin (stdlib only).

Environment variables are the C5 contract: FAKE_TIME_SCALE scales every fake sleep, and
FAKE_GPU_STATE_DIR holds gpu<i>.json = {"used_mib": int, "pid": int} for GPUs a live fake holds.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def time_scale() -> float:
    return float(os.environ.get("FAKE_TIME_SCALE") or 1.0)


def fake_sleep(seconds: float) -> None:
    """Sleep `seconds` of fake time (scaled by FAKE_TIME_SCALE)."""
    if seconds > 0:
        time.sleep(seconds * time_scale())


def env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def env_bool01(name: str, default: str) -> bool:
    """vLLM's `bool(int(os.getenv(name, default)))` env gates."""
    return bool(int(os.environ.get(name, default)))


def visible_gpus(default_count: int) -> list[int]:
    """Physical GPU indices from CUDA_VISIBLE_DEVICES, else 0..default_count-1."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return list(range(default_count))
    return [int(x) for x in cvd.split(",") if x.strip()]


def gpu_state_dir() -> pathlib.Path | None:
    value = os.environ.get("FAKE_GPU_STATE_DIR")
    return pathlib.Path(value) if value else None


def write_gpu_state(gpus: list[int], used_mib: int, pid: int) -> None:
    root = gpu_state_dir()
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    for i in gpus:
        tmp = root / f".gpu{i}.json.{pid}"
        tmp.write_text(json.dumps({"used_mib": used_mib, "pid": pid}))
        tmp.replace(root / f"gpu{i}.json")


def remove_gpu_state(gpus: list[int], pid: int) -> None:
    """Delete this process's state files; files another pid has written are left alone."""
    root = gpu_state_dir()
    if root is None:
        return
    for i in gpus:
        path = root / f"gpu{i}.json"
        state = read_gpu_state_file(path)
        if state is not None and state.get("pid") == pid:
            path.unlink(missing_ok=True)


def read_gpu_state_file(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def gpu_states() -> dict[int, dict]:
    """{gpu index: state} for every gpu<i>.json in FAKE_GPU_STATE_DIR."""
    root = gpu_state_dir()
    out: dict[int, dict] = {}
    if root is None or not root.is_dir():
        return out
    for path in sorted(root.glob("gpu*.json")):
        idx = path.stem[3:]
        state = read_gpu_state_file(path)
        if idx.isdigit() and state is not None:
            out[int(idx)] = state
    return out


def raise_nofile(target: int = 65535) -> int:
    """Raise the soft RLIMIT_NOFILE to min(hard, target) (spec AM12); returns the soft limit in effect.

    macOS rejects soft limits above kern.maxfilesperproc even when the hard limit is unlimited,
    so on ValueError fall back to OPEN_MAX (10240).
    """
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = target if hard == resource.RLIM_INFINITY else min(hard, target)
    if soft >= want:
        return soft
    for candidate in (want, min(want, 10240)):
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (candidate, hard))
            return candidate
        except (ValueError, OSError):
            continue
    return soft


def percentile(values: list[float], p: float) -> float:
    """numpy.percentile with the default linear interpolation."""
    xs = sorted(values)
    if not xs:
        raise ValueError("percentile of an empty list")
    rank = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def pstdev(values: list[float]) -> float:
    """numpy.std (population)."""
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def median(values: list[float]) -> float:
    return percentile(values, 50)
