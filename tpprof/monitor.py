"""Monitors that run alongside a run: nvidia-smi every 200 ms, and psutil per-process CPU.

gpu.csv is nvidia-smi's own csv output (C8 monitor query) under one header line.
cpu.csv has one row per sampled process per interval.
"""
from __future__ import annotations

import csv
import os
import threading
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import psutil

from tpprof import procs

GPU_FIELDS = ("timestamp", "index", "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu",
              "utilization.gpu", "memory.used")
EVENT_REASONS_FIELD = "clocks_event_reasons.active"
LEGACY_EVENT_REASONS_FIELD = "clocks_throttle_reasons.active"
MONITOR_PERIOD_MS = 200
MONITOR_STOP_GRACE_S = 2.0

# SwPowerCap 0x4, HwSlowdown 0x8, SwThermalSlowdown 0x20, HwThermalSlowdown 0x40, HwPowerBrakeSlowdown 0x80.
# GpuIdle 0x1, ApplicationsClocksSetting 0x2, SyncBoost 0x10 and DisplayClockSetting 0x100 are not throttling.
THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80

CPU_FIELDS = ("t_wall", "pid", "ppid", "name", "cmd", "cpu_percent", "rss_mib")


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


class GpuMonitor:
    """Context manager: `nvidia-smi --query-gpu=... -lms 200` appending to csv_path.

    The nvidia-smi process is tagged with run_id like every other child, so a leaked
    monitor is caught by the run's sweep. Leaving the context stops only the monitor's
    own process group; it does not sweep the run's other processes.
    """

    def __init__(self, csv_path: str, run_id: str, field: str = EVENT_REASONS_FIELD,
                 nvidia_smi: str = "nvidia-smi", env: Mapping[str, str] | None = None):
        self.csv_path = csv_path
        self.run_id = run_id
        self.field = field
        self.nvidia_smi = nvidia_smi
        self.env = env
        self.proc: procs.Proc | None = None

    def argv(self) -> list[str]:
        return [self.nvidia_smi, "--query-gpu=" + ",".join((*GPU_FIELDS, self.field)),
                "--format=csv,noheader,nounits", "-lms", str(MONITOR_PERIOD_MS)]

    def __enter__(self) -> GpuMonitor:
        _ensure_dir(self.csv_path)
        with open(self.csv_path, "a") as fh:
            fh.write(",".join((*GPU_FIELDS, self.field)) + "\n")
        env = os.environ if self.env is None else self.env
        self.proc = procs.spawn(self.argv(), env, self.csv_path, self.run_id)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None:
            procs.stop(self.proc, MONITOR_STOP_GRACE_S, MONITOR_STOP_GRACE_S, sweep=False)


class CpuSampler:
    """Context manager: a thread that samples the root pids and all their descendants.

    Every interval_s it writes one row per live process. A process's first observation
    only primes psutil's cpu_percent (which would read 0.0) and writes no row.
    """

    def __init__(self, csv_path: str, roots: Callable[[], Sequence[int]], interval_s: float = 1.0):
        self.csv_path = csv_path
        self.roots = roots
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._known: dict[int, psutil.Process] = {}
        self._error: BaseException | None = None

    def __enter__(self) -> CpuSampler:
        _ensure_dir(self.csv_path)
        self._fh = open(self.csv_path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if self._fh.tell() == 0:
            self._writer.writerow(CPU_FIELDS)
        self._thread = threading.Thread(target=self._loop, name="tpprof-cpu-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._fh.close()
        if self._error is not None and exc[0] is None:
            raise RuntimeError(f"CpuSampler for {self.csv_path} failed") from self._error

    def _loop(self) -> None:
        try:
            while True:
                self._sample()
                if self._stop.wait(self.interval_s):
                    return
        except BaseException as e:  # surfaced by __exit__
            self._error = e

    def _processes(self) -> list[psutil.Process]:
        found: dict[int, psutil.Process] = {}
        for pid in self.roots():
            try:
                root = psutil.Process(pid)
                found[root.pid] = root
                for child in root.children(recursive=True):
                    found[child.pid] = child
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return list(found.values())

    def _sample(self) -> None:
        t_wall = time.time()
        known: dict[int, psutil.Process] = {}
        for proc in self._processes():
            prev = self._known.get(proc.pid)
            try:
                if prev is None or prev != proc:        # new process (or a reused pid): prime it
                    proc.cpu_percent(None)
                    known[proc.pid] = proc
                    continue
                with prev.oneshot():
                    cpu = prev.cpu_percent(None)
                    row = (f"{t_wall:.3f}", prev.pid, prev.ppid(), prev.name(), _cmd(prev),
                           f"{cpu:.1f}", f"{prev.memory_info().rss / 2**20:.1f}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            known[proc.pid] = prev
            self._writer.writerow(row)
        self._known = known
        self._fh.flush()


def _cmd(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline())
    except (psutil.AccessDenied, psutil.ZombieProcess):
        return ""


def _parse_int(s: str, base: int = 10) -> int | None:
    try:
        return int(s, base)
    except ValueError:
        return None


def throttle_flags(csv_path: str) -> dict:
    """Throttle reasons and SM clock range seen in a GpuMonitor csv.

    Header lines, error text and a truncated last line are skipped; "rows" counts the
    sample lines that were read, so an empty or failed monitor is visible (rows == 0).
    """
    n_fields = len(GPU_FIELDS) + 1
    rows = 0
    masks: set[int] = set()
    clocks: list[int] = []
    with open(csv_path, errors="replace") as fh:
        for line in fh:
            fields = [f.strip() for f in line.split(",")]
            if len(fields) != n_fields or _parse_int(fields[1]) is None:
                continue
            rows += 1
            sm = _parse_int(fields[2])
            if sm is not None:
                clocks.append(sm)
            mask = _parse_int(fields[-1], 16)
            if mask is not None and mask & THROTTLE_MASK:
                masks.add(mask)
    return {"throttled": bool(masks), "masks": [f"0x{m:016x}" for m in sorted(masks)],
            "sm_clock_min": min(clocks) if clocks else None, "sm_clock_max": max(clocks) if clocks else None,
            "rows": rows}


def cpu_summary(csv_path: str) -> dict:
    """{process name: {"p50", "p90", "max"}} of cpu_percent over a CpuSampler csv."""
    by_name: dict[str, list[float]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            by_name.setdefault(row["name"], []).append(float(row["cpu_percent"]))
    return {name: {"p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)), "max": float(max(v))}
            for name, v in by_name.items()}
