"""End-to-end dry run of the full matrix, the way `make dry-run` runs it (plan Task 18).

`python -m tpprof run --tier all --dry-run`, then `analyze`, then `report`, each as a subprocess whose working
directory, HOME and temp directories are inside tmp_path, so any write that escapes the results directory shows.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from tests.conftest import ROOT
from tests.test_runner import free_port_base, leftover_fakes
from tpprof import report

PY = sys.executable
RUN_TIMEOUT_S = 1200            # the full matrix takes about 3 min at the dry-run time scale
STEP_TIMEOUT_S = 300
ALLOWED_SKIPS = {"dry_run:no_nccl_tests"}      # comm_m4 needs the nccl-tests build, which exists only on the box
FIB_ARM = "-FIBtrtllm-"                        # skippable only when the fake TP2 smoke did not pick mnnvl
HYPOTHESES = [f"H{i}" for i in range(1, 9)]
TMP_ENTRIES = {"results", "home", "tmp"}


def repo_snapshot() -> dict[str, tuple[int, int]]:
    """Every file in the checkout except .git and bytecode caches: path -> (size, mtime_ns)."""
    snap = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for name in filenames:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            snap[os.path.relpath(path, ROOT)] = (st.st_size, st.st_mtime_ns)
    return snap


def read(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.slow
def test_full_matrix_dry_run_then_analyze_and_report(tmp_path):
    results, home, tmpdir = tmp_path / "results", tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    tmpdir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("TPPROF_RESULTS_DIR", "TPPROF_MODEL_DIR")}
    env.update({"HOME": str(home), "TMPDIR": str(tmpdir), "XDG_CACHE_HOME": str(home / ".cache"),
                "MPLCONFIGDIR": str(home / ".matplotlib"), "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH")) if p)})
    before = repo_snapshot()

    def tpprof(*args: str, timeout: float = STEP_TIMEOUT_S) -> subprocess.CompletedProcess:
        return subprocess.run([PY, "-m", "tpprof", *args, "--results-dir", str(results)], cwd=str(tmp_path),
                              env=env, capture_output=True, text=True, timeout=timeout)

    try:
        run = tpprof("run", "--tier", "all", "--dry-run", "--port-base", str(free_port_base()),
                     timeout=RUN_TIMEOUT_S)
        assert run.returncode == 0, (run.stdout[-4000:], run.stderr[-4000:])

        # Every spec of the matrix the operator is shown ran: done, or skipped for an allowed reason.
        summary = read(results / "raw" / "_last_run.json")
        assert summary["failed"] == [] and not summary["interrupted"], summary["failed"]
        smoke = [d for d in os.listdir(results / "raw") if d.startswith("P0-smoke-TP2-base-")]
        assert len(smoke) == 1, smoke
        fi_backend = read(results / "raw" / smoke[0] / "effective_config.json").get("fi_backend")
        for entry in summary["skipped"]:
            allowed = entry["reason"] in ALLOWED_SKIPS or (FIB_ARM in entry["run_id"] and fi_backend != "mnnvl")
            assert allowed, entry
        listed = tpprof("matrix", "--list")
        assert listed.returncode == 0, listed.stderr[-2000:]
        planned = {line.split()[0] for line in listed.stdout.splitlines()[1:] if line.strip()}
        ran = set(summary["done"]) | {e["run_id"] for e in summary["skipped"]}
        assert planned == ran, {"planned_not_run": sorted(planned - ran), "ran_not_planned": sorted(ran - planned)}
        for run_id in summary["done"]:
            assert os.path.exists(results / "raw" / run_id / "done.json"), run_id
            assert not os.path.exists(results / "raw" / run_id / "failed.json"), run_id

        analyzed = tpprof("analyze")
        assert analyzed.returncode == 0, (analyzed.stdout[-4000:], analyzed.stderr[-4000:])
        reported = tpprof("report")
        assert reported.returncode == 0, (reported.stdout[-4000:], reported.stderr[-4000:])

        lines = (results / "SUMMARY.md").read_text().splitlines()
        assert lines[0] == report.FAKE_WATERMARK
        rows = [m.group(1) for line in lines if (m := re.match(r"^\| (H\d) \|", line))]
        assert rows == HYPOTHESES, rows
    finally:
        left = leftover_fakes(results)
    assert left == []

    # Nothing was written outside tmp_path: the checkout is unchanged and tmp_path holds only what the test made.
    assert repo_snapshot() == before
    assert {p.name for p in tmp_path.iterdir()} <= TMP_ENTRIES, sorted(p.name for p in tmp_path.iterdir())
