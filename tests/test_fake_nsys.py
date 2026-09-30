from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.conftest import FAKE_BIN, fake_env
from tpprof import traces

NSYS = str(FAKE_BIN / "nsys")

# Stands in for the profiled engine: appends one C4 line per rank to $FAKE_NSYS_EVENTS.
WRITE_EVENTS = (
    "import json, os\n"
    "with open(os.environ['FAKE_NSYS_EVENTS'], 'a') as f:\n"
    "    for rank in (0, 1):\n"
    "        f.write(json.dumps({'pid': 4242 * 10 + rank, 'device': rank, 'rank': rank, 'tp': 2,\n"
    "                            'ar_backend': 'trtllm', 'batch': 1,\n"
    "                            'steps': [[0, 0, 1, 1]] * 20}) + '\\n')\n"
)

CAPTURE_API = ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
CAPTURE_NONE = ["--capture-range=none"]


def _profile_and_export(tmp_path, capture, **env_overrides):
    env = fake_env(tmp_path, **env_overrides)
    out = str(tmp_path / "t")
    prof = subprocess.run([NSYS, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", *capture,
                           "--force-overwrite=true", f"--output={out}", sys.executable, "-c", WRITE_EVENTS],
                          env=env, capture_output=True, text=True, timeout=60)
    assert prof.returncode == 0, prof.stderr
    exp = subprocess.run([NSYS, "export", "--type=sqlite", "--force-overwrite=true",
                          f"--output={out}.sqlite", f"{out}.nsys-rep"],
                         env=env, capture_output=True, text=True, timeout=60)
    assert exp.returncode == 0, exp.stderr
    return out


def test_fake_nsys_is_executable():
    assert os.access(NSYS, os.X_OK)


def test_version(tmp_path):
    r = subprocess.run([NSYS, "--version"], env=fake_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0
    assert r.stdout.strip() == "NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0"


def test_status_environment(tmp_path):
    r = subprocess.run([NSYS, "status", "--environment"], env=fake_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0
    assert sum(line.endswith("OK") for line in r.stdout.splitlines()) >= 3


def test_profile_writes_rep_and_export_passes_gate(tmp_path):
    out = _profile_and_export(tmp_path, CAPTURE_API)
    rep = json.loads(open(f"{out}.nsys-rep").read())
    assert rep["events"] == f"{out}.events.jsonl"
    assert rep["capture_range"] == "cudaProfilerApi"
    assert "--capture-range-end=stop" in rep["argv"]
    summary = traces.summarize_trace(f"{out}.sqlite", tp=2, min_steps=5)
    assert summary["gate"]["ok"] is True
    assert [r["steps"] for r in summary["ranks"]] == [20, 20]


def test_drop_rank_tail_fails_the_gate(tmp_path):
    out = _profile_and_export(tmp_path, CAPTURE_API, FAKE_NSYS_DROP_RANK_TAIL="1")
    summary = traces.summarize_trace(f"{out}.sqlite", tp=2, min_steps=5)
    assert summary["gate"]["ok"] is False
    assert "rank 1 has 17 steps, rank 0 has 20" in summary["gate"]["reasons"]


def test_capture_range_none_is_not_affected_by_the_drop(tmp_path):
    out = _profile_and_export(tmp_path, CAPTURE_NONE, FAKE_NSYS_DROP_RANK_TAIL="1")
    assert json.loads(open(f"{out}.nsys-rep").read())["capture_range"] == "none"
    summary = traces.summarize_trace(f"{out}.sqlite", tp=2, min_steps=5)
    assert summary["gate"]["ok"] is True


def test_profile_rerun_replaces_old_events(tmp_path):
    out = _profile_and_export(tmp_path, CAPTURE_API)
    out = _profile_and_export(tmp_path, CAPTURE_API)
    assert len(open(f"{out}.events.jsonl").read().splitlines()) == 2


def test_profile_propagates_target_exit_code(tmp_path):
    out = str(tmp_path / "t")
    r = subprocess.run([NSYS, "profile", *CAPTURE_API, f"--output={out}", sys.executable, "-c",
                        "import sys; sys.exit(3)"], env=fake_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 3
    assert os.path.exists(f"{out}.nsys-rep")


def test_profile_hang(tmp_path):
    out = str(tmp_path / "t")
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run([NSYS, "profile", *CAPTURE_API, f"--output={out}", sys.executable, "-c", "pass"],
                       env=fake_env(tmp_path, FAKE_NSYS_HANG="1"), capture_output=True, timeout=1)


def test_export_refuses_to_overwrite_without_force(tmp_path):
    out = _profile_and_export(tmp_path, CAPTURE_API)
    r = subprocess.run([NSYS, "export", "--type=sqlite", f"--output={out}.sqlite", f"{out}.nsys-rep"],
                       env=fake_env(tmp_path), capture_output=True, text=True)
    assert r.returncode != 0
    assert "exists" in r.stderr


def test_unknown_subcommand_fails(tmp_path):
    r = subprocess.run([NSYS, "analyze"], env=fake_env(tmp_path), capture_output=True, text=True)
    assert r.returncode != 0
