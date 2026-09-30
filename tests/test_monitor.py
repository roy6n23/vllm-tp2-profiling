from __future__ import annotations

import csv
import os
import pathlib
import subprocess
import sys
import textwrap
import time

import psutil
import pytest

from tpprof import monitor

PY = sys.executable
C8_MONITOR_QUERY = ("--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                    "utilization.gpu,memory.used,clocks_event_reasons.active")


def _fake_monitor_smi(tmp_path: pathlib.Path, argv_out: pathlib.Path) -> str:
    """nvidia-smi that records its argv, then prints two GPU rows per 50 ms tick until killed."""
    script = tmp_path / "nvidia-smi"
    script.write_text(textwrap.dedent(f"""\
        #!{PY}
        import os, sys, time
        open({str(argv_out)!r}, "w").write("\\n".join(sys.argv[1:]) + "\\n" + os.environ.get("TPPROF_RUN_ID", ""))
        while True:
            for i in (0, 1):
                print("2026/09/30 12:00:00.000, %d, 1980, 2619, 120.50, 35, 0, 1, 0x0000000000000000" % i, flush=True)
            time.sleep(0.05)
        """))
    script.chmod(0o755)
    return str(script)


def _rows(path: pathlib.Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.startswith("2026/")]


def test_gpu_monitor_writes_rows_and_stops(tmp_path):
    argv_out = tmp_path / "argv"
    smi = _fake_monitor_smi(tmp_path, argv_out)
    csv_path = tmp_path / "gpu.csv"
    with monitor.GpuMonitor(str(csv_path), "run-mon", nvidia_smi=smi) as mon:
        time.sleep(0.5)
        pid = mon.proc.popen.pid
    assert len(_rows(csv_path)) >= 2
    assert argv_out.read_text().splitlines() == [
        C8_MONITOR_QUERY, "--format=csv,noheader,nounits", "-lms", "200", "run-mon"]
    assert csv_path.read_text().splitlines()[0].startswith("timestamp,index,clocks.sm")
    assert mon.proc.popen.poll() is not None
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_gpu_monitor_legacy_field(tmp_path):
    argv_out = tmp_path / "argv"
    smi = _fake_monitor_smi(tmp_path, argv_out)
    with monitor.GpuMonitor(str(tmp_path / "gpu.csv"), "r", field="clocks_throttle_reasons.active", nvidia_smi=smi):
        deadline = time.monotonic() + 5
        while not argv_out.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
    assert argv_out.read_text().splitlines()[0].endswith(",memory.used,clocks_throttle_reasons.active")


def test_gpu_monitor_does_not_kill_other_processes_of_the_run(tmp_path):
    smi = _fake_monitor_smi(tmp_path, tmp_path / "argv")
    env = dict(os.environ, TPPROF_RUN_ID="run-shared")
    other = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"], env=env, start_new_session=True)
    try:
        with monitor.GpuMonitor(str(tmp_path / "gpu.csv"), "run-shared", nvidia_smi=smi):
            time.sleep(0.2)
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()


HEADER = ("timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,"
          "clocks_event_reasons.active\n")


def _row(i: int, sm: int, mask: str) -> str:
    return f"2026/09/30 12:00:00.000, {i}, {sm}, 2619, 650.00, 70, 100, 70000, {mask}\n"


def test_throttle_flags_detects_power_cap(tmp_path):
    f = tmp_path / "gpu.csv"
    f.write_text(HEADER + _row(0, 1980, "0x0000000000000000") + _row(1, 1755, "0x0000000000000004")
                 + _row(0, 1980, "0x0000000000000001"))
    flags = monitor.throttle_flags(str(f))
    assert flags["throttled"] is True
    assert flags["masks"] == ["0x0000000000000004"]
    assert flags["sm_clock_min"] == 1755 and flags["sm_clock_max"] == 1980
    assert flags["rows"] == 3


def test_throttle_flags_ignores_benign_reasons_and_junk(tmp_path):
    f = tmp_path / "gpu.csv"
    # 0x1 GpuIdle, 0x2 ApplicationsClocksSetting and 0x100 SyncBoost are not throttling.
    f.write_text(HEADER + _row(0, 1980, "0x0000000000000001") + _row(1, 1980, "0x0000000000000102")
                 + _row(0, 1980, "[N/A]") + "Unable to determine the device handle\n"
                 + "2026/09/30 12:00:00.000, 1, 19")   # truncated last line from a killed monitor
    flags = monitor.throttle_flags(str(f))
    assert flags == {"throttled": False, "masks": [], "sm_clock_min": 1980, "sm_clock_max": 1980, "rows": 3}


@pytest.mark.parametrize("bit", [0x4, 0x8, 0x20, 0x40, 0x80])
def test_throttle_mask_bits(tmp_path, bit):
    f = tmp_path / "gpu.csv"
    f.write_text(HEADER + _row(0, 1500, f"0x{bit:016x}"))
    assert monitor.throttle_flags(str(f))["throttled"] is True


def test_throttle_flags_empty_csv(tmp_path):
    f = tmp_path / "gpu.csv"
    f.write_text(HEADER)
    assert monitor.throttle_flags(str(f)) == {"throttled": False, "masks": [], "sm_clock_min": None,
                                              "sm_clock_max": None, "rows": 0}


def test_cpu_sampler_on_current_process(tmp_path):
    csv_path = tmp_path / "cpu.csv"
    with monitor.CpuSampler(str(csv_path), lambda: [os.getpid()], interval_s=0.1):
        t_end = time.monotonic() + 0.6
        while time.monotonic() < t_end:
            sum(range(10000))
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "no rows written"
    assert set(rows[0]) == {"t_wall", "pid", "ppid", "name", "cmd", "cpu_percent", "rss_mib"}
    mine = [r for r in rows if int(r["pid"]) == os.getpid()]
    assert len(mine) >= 2
    assert all(float(r["rss_mib"]) > 1 for r in mine)
    assert max(float(r["cpu_percent"]) for r in mine) > 0


def test_cpu_sampler_includes_descendants_and_survives_exits(tmp_path):
    child = subprocess.Popen([PY, "-c", "import time; time.sleep(1.0)"])
    try:
        csv_path = tmp_path / "cpu.csv"
        with monitor.CpuSampler(str(csv_path), lambda: [os.getpid(), 999999999], interval_s=0.1):
            time.sleep(0.5)
            child.wait(5)       # the child exits while it is being sampled
            time.sleep(0.3)
        with open(csv_path, newline="") as fh:
            pids = {int(r["pid"]) for r in csv.DictReader(fh)}
        assert child.pid in pids and os.getpid() in pids
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_cpu_summary(tmp_path):
    f = tmp_path / "cpu.csv"
    lines = ["t_wall,pid,ppid,name,cmd,cpu_percent,rss_mib"]
    for i, v in enumerate([10, 20, 30, 40, 100]):
        lines.append(f"{i}.0,1,0,EngineCore,vllm serve,{v},100.0")
    lines.append("0.0,2,1,APIServer,vllm serve,5,50.0")
    f.write_text("\n".join(lines) + "\n")
    s = monitor.cpu_summary(str(f))
    assert set(s) == {"EngineCore", "APIServer"}
    assert s["EngineCore"]["p50"] == pytest.approx(30.0)
    assert s["EngineCore"]["p90"] == pytest.approx(76.0)
    assert s["EngineCore"]["max"] == pytest.approx(100.0)
    assert s["APIServer"] == {"p50": 5.0, "p90": 5.0, "max": 5.0}
