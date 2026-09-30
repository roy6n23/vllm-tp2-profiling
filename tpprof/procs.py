"""Child processes: own process group, log file, timeout, stop escalation, GPU-memory waits.

Every child runs in a new session (so it leads its own process group), reads nothing
from stdin, appends stdout and stderr to a log file, and carries the env tag
TPPROF_RUN_ID=<run_id> (spec AM32). Stopping sends SIGINT, then SIGTERM, then SIGKILL
to the whole group (spec 7.4), then kills any process that escaped the group but still
carries the tag.
"""
from __future__ import annotations

import math
import os
import resource
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import psutil

from tpprof.constants import GPU_FREE_MIB

RUN_ID_ENV = "TPPROF_RUN_ID"
NOFILE_TARGET = 65535          # spec AM12: raise RLIMIT_NOFILE to min(hard, 65535)
KILL_GRACE_S = 10.0            # how long to wait for the group to vanish after SIGKILL
SWEEP_GRACE_S = 5.0
POLL_S = 0.05
QUERY_TIMEOUT_S = 30.0


@dataclass
class Proc:
    popen: subprocess.Popen
    argv: list[str]
    log_path: str
    run_id: str
    t_wall_start: float
    t_mono_start: float


@dataclass
class Outcome:
    exit_code: int | None      # None when the run timed out
    timed_out: bool
    duration_s: float
    t_wall_end: float
    t_mono_end: float


def _nofile_raiser() -> Callable[[], None] | None:
    """A preexec_fn that raises the soft RLIMIT_NOFILE, or None if it is already high enough.

    The target is computed here, in the parent, so the child only makes one system call
    between fork and exec.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft_v = math.inf if soft == resource.RLIM_INFINITY else soft
    hard_v = math.inf if hard == resource.RLIM_INFINITY else hard
    target = int(min(hard_v, NOFILE_TARGET))
    if soft_v >= target:
        return None                # never lower an already higher limit

    def _raise() -> None:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    return _raise


def spawn(argv: Sequence[str], env: Mapping[str, str], log_path: str, run_id: str, cwd: str | None = None,
          raise_nofile: bool = False) -> Proc:
    """Start argv in a new session with its output appended to log_path."""
    argv = [str(a) for a in argv]
    child_env = dict(env)
    child_env[RUN_ID_ENV] = run_id
    preexec = _nofile_raiser() if raise_nofile else None
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    t_wall, t_mono = time.time(), time.monotonic()
    with open(log_path, "a", buffering=1) as log:
        popen = subprocess.Popen(argv, env=child_env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=preexec)
    return Proc(popen=popen, argv=argv, log_path=log_path, run_id=run_id,
                t_wall_start=t_wall, t_mono_start=t_mono)


def wait(p: Proc, timeout_s: float | None) -> int | None:
    """The exit code, or None if the process is still running after timeout_s."""
    try:
        return p.popen.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _wait_group_gone(p: Proc, timeout_s: float) -> bool:
    """True once the leader has exited (and is reaped) and the process group is empty."""
    pgid = p.popen.pid
    deadline = time.monotonic() + timeout_s
    while True:
        if p.popen.poll() is not None and not _group_alive(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_S)


def stop(p: Proc, grace_int_s: float = 20.0, grace_term_s: float = 10.0, *, sweep: bool = True) -> int:
    """Stop the whole process group and return the leader's exit code.

    SIGINT, wait grace_int_s; SIGTERM, wait grace_term_s; SIGKILL. A signal is only sent
    while the group still has members, so calling stop again is harmless. With sweep
    (the default), processes that left the group but carry TPPROF_RUN_ID=p.run_id are
    killed afterwards.
    """
    pgid = p.popen.pid
    for sig, grace in ((signal.SIGINT, grace_int_s), (signal.SIGTERM, grace_term_s),
                       (signal.SIGKILL, KILL_GRACE_S)):
        if _wait_group_gone(p, 0):
            break
        _signal_group(pgid, sig)
        if _wait_group_gone(p, grace):
            break
    if sweep:
        sweep_tagged(p.run_id)
    rc = p.popen.poll()
    if rc is None:
        raise RuntimeError(f"process {pgid} ({p.argv[0]}) is still running {KILL_GRACE_S:.0f} s after SIGKILL")
    return rc


def _alive(proc: psutil.Process) -> bool:
    try:
        return proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def sweep_tagged(run_id: str) -> list[int]:
    """SIGKILL every process whose environment has TPPROF_RUN_ID == run_id; return their pids.

    The calling process and its ancestors are never touched. Processes whose environment
    cannot be read (other users, kernel tasks) are skipped. Killed processes are not
    reaped here, so a killed direct child still reports its real exit code to its Popen.
    """
    me = psutil.Process()
    spare = {me.pid, *(a.pid for a in me.parents())}
    victims: list[psutil.Process] = []
    for proc in psutil.process_iter():
        if proc.pid in spare:
            continue
        try:
            if proc.environ().get(RUN_ID_ENV) != run_id:
                continue
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
            continue
        victims.append(proc)
    deadline = time.monotonic() + SWEEP_GRACE_S
    pending = victims
    while pending and time.monotonic() < deadline:
        time.sleep(POLL_S)
        pending = [v for v in pending if _alive(v)]
    return [v.pid for v in victims]


def run(argv: Sequence[str], env: Mapping[str, str], log_path: str, run_id: str, timeout_s: float,
        cwd: str | None = None) -> Outcome:
    """spawn, wait up to timeout_s, and stop the process group on timeout or on an exception."""
    p = spawn(argv, env, log_path, run_id, cwd=cwd)
    try:
        code = wait(p, timeout_s)
    except BaseException:          # e.g. KeyboardInterrupt: the child is in its own session
        stop(p)
        raise
    timed_out = code is None
    if timed_out:
        stop(p)
    t_wall_end, t_mono_end = time.time(), time.monotonic()
    return Outcome(exit_code=code, timed_out=timed_out, duration_s=t_mono_end - p.t_mono_start,
                   t_wall_end=t_wall_end, t_mono_end=t_mono_end)


def _query(nvidia_smi: str, args: Sequence[str], env: Mapping[str, str] | None) -> list[list[str]]:
    """Run one nvidia-smi csv query and return the comma-split, stripped fields of each line."""
    argv = [nvidia_smi, *args]
    r = subprocess.run(argv, env=None if env is None else dict(env), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=QUERY_TIMEOUT_S, start_new_session=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {r.returncode}: {(r.stderr or r.stdout).strip()}")
    return [[f.strip() for f in line.split(",")] for line in r.stdout.splitlines() if line.strip()]


def _int_or_none(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


def gpu_memory_used(nvidia_smi: str = "nvidia-smi", env: Mapping[str, str] | None = None) -> dict[int, int]:
    """{gpu index: memory.used in MiB} (C8 query)."""
    out: dict[int, int] = {}
    for fields in _query(nvidia_smi, ["--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], env):
        if len(fields) == 2 and _int_or_none(fields[0]) is not None and _int_or_none(fields[1]) is not None:
            out[int(fields[0])] = int(fields[1])
    return out


def gpu_processes(nvidia_smi: str = "nvidia-smi", env: Mapping[str, str] | None = None) -> list[tuple[int, int]]:
    """[(pid, used_memory MiB)] of GPU compute processes (C8 query); -1 when the memory is not reported."""
    out: list[tuple[int, int]] = []
    for fields in _query(nvidia_smi, ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], env):
        if len(fields) == 2 and _int_or_none(fields[0]) is not None:
            used = _int_or_none(fields[1])
            out.append((int(fields[0]), -1 if used is None else used))
    return out


def wait_gpu_memory_free(gpus: Sequence[int], threshold_mib: int = GPU_FREE_MIB, timeout_s: float = 90,
                         nvidia_smi: str = "nvidia-smi", env: Mapping[str, str] | None = None) -> bool:
    """Poll until every listed GPU uses < threshold_mib; False if that does not happen within timeout_s.

    A GPU missing from nvidia-smi's output never counts as free.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        used = gpu_memory_used(nvidia_smi, env)
        if all(g in used and used[g] < threshold_mib for g in gpus):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(1.0, remaining))


def tail(path: str, n: int = 40) -> str:
    """The last n lines of a text file ("" if it does not exist)."""
    if n <= 0:
        return ""
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return ""
    with f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        data = b""
        while pos > 0 and data.count(b"\n") <= n:
            step = min(64 * 1024, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    return "\n".join(data.decode("utf-8", "replace").splitlines()[-n:])
