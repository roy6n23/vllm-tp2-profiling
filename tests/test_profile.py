from __future__ import annotations

import json
import os
import sys

import pytest

from tests.conftest import FAKE_BIN, fake_env
from tpprof import constants, engine, profile, traces

NSYS = str(FAKE_BIN / "nsys")
STEPS = 20

# Stands in for `python -m tpprof.offline --profile ...` under the fake engine: appends one C4 line
# per rank to $FAKE_NSYS_EVENTS (contracts C4) and records the environment it was started with.
STAND_IN = (
    "import json, os, sys, time\n"
    "if os.environ.get('HANG_UNLESS_SYMM_MEM_OFF') == '1' and os.environ.get('VLLM_ALLREDUCE_USE_SYMM_MEM') != '0':\n"
    "    time.sleep(3600)\n"
    "with open(os.environ['ENV_DUMP'], 'w') as f:\n"
    "    json.dump(dict(os.environ), f)\n"
    "with open(os.environ['FAKE_NSYS_EVENTS'], 'a') as f:\n"
    "    for rank in (0, 1):\n"
    "        f.write(json.dumps({'pid': os.getpid() * 10 + rank, 'device': rank, 'rank': rank, 'tp': 2,\n"
    f"                            'ar_backend': 'trtllm', 'batch': 1, 'steps': [[0, 0, 1, 1]] * {STEPS}}}) + '\\n')\n"
    "sys.exit(int(os.environ.get('STAND_IN_EXIT', '0')))\n"
)
TARGET = [sys.executable, "-c", STAND_IN]


def _env(tmp_path, **overrides):
    env = fake_env(tmp_path, ENV_DUMP=str(tmp_path / "env.json"), **overrides)
    env["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "1"          # the base engine env (contracts C1)
    return env


def _run(tmp_path, env, target=TARGET, timeout_s=60.0, **kw):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return profile.run_trace(str(run_dir), "T14-test", target, env, tp=2, min_steps=STEPS, timeout_s=timeout_s, **kw)


# --- argv ---------------------------------------------------------------------------------------

def test_profile_argv_is_exact():
    assert profile.nsys_profile_argv("/r/trace", ["python", "-m", "tpprof.offline"], nsys="/x/nsys") == [
        "/x/nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--trace-fork-before-exec=true",
        "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--force-overwrite=true", "--output=/r/trace", "python", "-m", "tpprof.offline"]


def test_fallback_argv_is_exact_and_has_no_capture_range_end():
    argv = profile.nsys_profile_argv("/r/trace", ["python", "t.py"], capture="none", nsys="/x/nsys")
    assert argv == [
        "/x/nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--trace-fork-before-exec=true",
        "--sample=none", "--cpuctxsw=none", "--capture-range=none",
        "--force-overwrite=true", "--output=/r/trace", "python", "t.py"]
    assert not any(a.startswith("--capture-range-end") for a in argv)      # AM17


def test_profile_argv_rejects_unknown_capture():
    with pytest.raises(ValueError, match="capture"):
        profile.nsys_profile_argv("/r/trace", ["python"], capture="nvtx", nsys="/x/nsys")


def test_export_argv_is_exact():
    assert profile.nsys_export_argv("/r/trace.nsys-rep", "/r/trace.sqlite", nsys="/x/nsys") == [
        "/x/nsys", "export", "--type=sqlite", "--force-overwrite=true", "--output=/r/trace.sqlite",
        "/r/trace.nsys-rep"]


def test_default_nsys_is_the_pinned_path(monkeypatch):
    monkeypatch.delenv("TPPROF_NSYS", raising=False)
    assert profile.nsys_profile_argv("o", ["t"])[0] == constants.NSYS_DEFAULT_PATH
    assert profile.nsys_export_argv("o.nsys-rep", "o.sqlite")[0] == constants.NSYS_DEFAULT_PATH
    monkeypatch.setenv("TPPROF_NSYS", NSYS)
    assert profile.nsys_profile_argv("o", ["t"])[0] == NSYS


def test_nsys_env_is_exact():
    assert profile.NSYS_ENV == {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "60"}


# --- run_trace ----------------------------------------------------------------------------------

def test_trace_happy_path(tmp_path):
    out = _run(tmp_path, _env(tmp_path))
    run_dir = str(tmp_path / "run")
    assert out.ok, out.reason
    assert out.summary["gate"] == {"ok": True, "reasons": []}
    assert out.sqlite_path == os.path.join(run_dir, "trace.sqlite") and os.path.exists(out.sqlite_path)
    assert [r["steps"] for r in out.summary["ranks"]] == [STEPS, STEPS]
    assert len(out.attempts) == 1
    a = out.attempts[0]
    assert a["argv"] == profile.nsys_profile_argv(os.path.join(run_dir, "trace"), TARGET, nsys=NSYS)
    assert a["capture"] == "cudaProfilerApi" and a["exit_code"] == 0 and not a["timed_out"]
    assert a["gate_ok"] is True and a["gate_reasons"] == []
    assert a["export"]["argv"] == profile.nsys_export_argv(
        os.path.join(run_dir, "trace.nsys-rep"), out.sqlite_path, nsys=NSYS)
    assert a["export"]["exit_code"] == 0
    for rec in (a, a["export"]):
        assert rec["t_wall_end"] >= rec["t_wall_start"] and rec["t_mono_end"] >= rec["t_mono_start"]
        assert rec["duration_s"] >= 0 and os.path.exists(rec["log"])
    # the environment the profiled target actually saw
    seen = json.loads((tmp_path / "env.json").read_text())
    assert seen["NSYS_TMPDIR"] == os.path.join(run_dir, "nsys_tmp") and os.path.isdir(seen["NSYS_TMPDIR"])
    assert seen["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert seen["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] == "60"
    assert seen["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "1"
    assert seen["TPPROF_RUN_ID"] == "T14-test"
    assert a["env_overrides"] == {**profile.NSYS_ENV, "NSYS_TMPDIR": seen["NSYS_TMPDIR"]}
    json.dumps(out.attempts)                  # the caller writes them to cmd.json


def test_trace_passes_untraced_step_time_to_the_summary(tmp_path):
    out = _run(tmp_path, _env(tmp_path), untraced_step_s=0.5)
    assert out.ok and out.summary["idle_est"] is not None


def test_trace_falls_back_to_capture_range_none(tmp_path):
    out = _run(tmp_path, _env(tmp_path, FAKE_NSYS_DROP_RANK_TAIL="1"))
    assert out.ok, out.reason
    assert len(out.attempts) == 2
    first, second = out.attempts
    assert first["capture"] == "cudaProfilerApi" and first["gate_ok"] is False
    assert any("rank 1 has 17 steps" in r for r in first["gate_reasons"])
    assert second["capture"] == "none" and second["gate_ok"] is True
    assert "--capture-range=none" in second["argv"]
    assert not any(a.startswith("--capture-range-end") for a in second["argv"])    # AM17
    assert out.summary["gate"]["ok"] and [r["steps"] for r in out.summary["ranks"]] == [STEPS, STEPS]
    assert "capture-range=none" in out.reason


def test_trace_fallback_without_measure_range_is_rejected(tmp_path, monkeypatch):
    real = traces.load_trace

    def without_measure(path):
        td = real(path)
        td.ranges = [r for r in td.ranges if r.text != traces.MEASURE_RANGE]
        return td

    monkeypatch.setattr(traces, "load_trace", without_measure)
    out = _run(tmp_path, _env(tmp_path, FAKE_NSYS_DROP_RANK_TAIL="1"))
    assert not out.ok
    assert len(out.attempts) == 2
    assert traces.MEASURE_RANGE in out.reason
    assert out.attempts[1]["gate_ok"] is False
    assert any(traces.MEASURE_RANGE in r for r in out.attempts[1]["gate_reasons"])


def test_trace_gate_fails_on_both_attempts(tmp_path):
    out = profile.run_trace(str(tmp_path / "run"), "T14-test", TARGET, _env(tmp_path), tp=2,
                            min_steps=STEPS + 1, timeout_s=60)
    assert not out.ok and len(out.attempts) == 2
    assert [a["capture"] for a in out.attempts] == ["cudaProfilerApi", "none"]
    assert out.summary is not None and not out.summary["gate"]["ok"]
    assert "fewer than min_steps" in out.reason


def test_trace_timeout_retries_without_symm_mem(tmp_path):
    out = _run(tmp_path, _env(tmp_path, FAKE_NSYS_HANG="1"), timeout_s=1.0)
    assert not out.ok
    assert len(out.attempts) == 2
    first, second = out.attempts
    assert first["timed_out"] and first["exit_code"] is None and first["export"] is None
    assert "VLLM_ALLREDUCE_USE_SYMM_MEM" not in first["env_overrides"]
    assert second["timed_out"]
    assert second["env_overrides"]["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0"
    assert second["argv"] == first["argv"]
    assert out.summary is None and out.sqlite_path is None
    assert "timed out" in out.reason


def test_trace_recovers_when_symm_mem_off_stops_the_hang(tmp_path):
    # the target hangs like vLLM #48486 unless VLLM_ALLREDUCE_USE_SYMM_MEM=0
    out = _run(tmp_path, _env(tmp_path, HANG_UNLESS_SYMM_MEM_OFF="1"), timeout_s=3.0)
    assert out.ok, out.reason
    assert [a["timed_out"] for a in out.attempts] == [True, False]
    seen = json.loads((tmp_path / "env.json").read_text())
    assert seen["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0"
    assert "VLLM_ALLREDUCE_USE_SYMM_MEM=0" in out.reason


def test_trace_nonzero_exit_fails_without_retry(tmp_path):
    out = _run(tmp_path, _env(tmp_path, STAND_IN_EXIT="3"))
    assert not out.ok and len(out.attempts) == 1
    assert out.attempts[0]["exit_code"] == 3 and out.attempts[0]["export"] is None
    assert "exited 3" in out.reason


def test_trace_uses_the_nsys_named_in_env(tmp_path, monkeypatch):
    monkeypatch.delenv("TPPROF_NSYS", raising=False)
    out = _run(tmp_path, _env(tmp_path))
    assert out.ok and out.attempts[0]["argv"][0] == NSYS


def test_trace_happy_path_with_offline_driver(tmp_path):
    pytest.importorskip("tpprof.offline")
    model_dir = str(tmp_path / "model")
    cfg = engine.base_config("TP2")
    target = [sys.executable, "-m", "tpprof.offline", "--out", str(tmp_path / "points"),
              "--meta", json.dumps({"config": "TP2", "arm": "base"}),
              "--profile", "decode:b1", "--profile-warmup", "1", "--profile-iters", "1",
              "--", *cfg.offline_args(model_dir)]
    env = cfg.environment(fake_env(tmp_path, TPPROF_FAKE="1"))
    out = profile.run_trace(str(tmp_path / "run"), "T14-offline", target, env, tp=2, min_steps=200, timeout_s=120)
    assert out.ok, (out.reason, out.attempts)
    assert out.summary["gate"]["ok"] and len(out.attempts) == 1


def test_trace_export_failure_is_reported_and_no_stale_trace_is_returned(tmp_path):
    wrapper = tmp_path / "nsys"
    wrapper.write_text(f'#!/bin/sh\nif [ "$1" = export ]; then echo "export broke" >&2; exit 2; fi\n'
                       f'exec {sys.executable} {NSYS} "$@"\n')
    wrapper.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "trace.sqlite").write_text("stale")
    out = _run(tmp_path, _env(tmp_path, TPPROF_NSYS=str(wrapper)))
    assert not out.ok and len(out.attempts) == 1
    assert out.attempts[0]["export"]["exit_code"] == 2
    assert "nsys export exited 2" in out.reason
    assert out.sqlite_path is None and out.summary is None
    assert not (run_dir / "trace.sqlite").exists()
