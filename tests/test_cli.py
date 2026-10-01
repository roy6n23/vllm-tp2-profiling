"""`python -m tpprof <subcommand>`."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests import synth_trace
from tests.conftest import FAKE_BIN, ROOT
from tests.test_runner import free_port_base, leftover_fakes
from tpprof import cli, matrix, model

PY = sys.executable
HOST_GATES = ("--skip-gate", "shm", "disk", "--skip-gate", "nofile")


@pytest.fixture
def fake_path(monkeypatch, tmp_path):
    """The fake nvidia-smi / vllm / nsys first on PATH, as on a box where they are installed."""
    monkeypatch.setenv("PATH", os.pathsep.join([str(FAKE_BIN), os.path.dirname(PY), os.environ["PATH"]]))
    monkeypatch.setenv("FAKE_GPU_STATE_DIR", str(tmp_path / "gpustate"))
    monkeypatch.setenv("TPPROF_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("TPPROF_MODEL_DIR", raising=False)


def test_cli_matrix_estimate_prints_hours(capsys, tmp_path):
    assert cli.main(["matrix", "--estimate", "--results-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for tier in ("P0", "P1", "P2"):
        assert any(line.startswith(f"{tier}  ") and " h  $" in line for line in out.splitlines()), out
    assert "Total" in out and " h " in out
    # without a measured grid, the sweeps are estimated on the model's saturation rates
    specs = matrix.build_matrix(matrix.TIERS, matrix.rate_grid(matrix.model_mu_rps()), "mnnvl")
    total_h = sum(e.minutes for e in matrix.estimate(specs)) / 60
    assert f"Total  {len(specs)} runs  {total_h:.2f} h" in out
    assert "rate grid: a-priori model" in out


def test_cli_matrix_uses_the_measured_grid(capsys, tmp_path):
    mu = {"TP1": 20.0, "TP2": 40.0, "DP2": 50.0}
    os.makedirs(tmp_path / "raw")
    (tmp_path / "raw" / "rate_grid.json").write_text(json.dumps(
        {"mu_rps": mu, "grid": matrix.rate_grid(mu), "sources": []}))
    assert cli.main(["matrix", "--tier", "P1", "--list", "--results-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    sweeps = matrix.p1_sweep_specs(matrix.rate_grid(mu))
    for s in sweeps:
        assert s.run_id in out
    assert "rate grid: measured" in out


def test_cli_matrix_rejects_unknown_tier(capsys):
    with pytest.raises(SystemExit):
        cli.main(["matrix", "--tier", "P9"])
    assert "P9" in capsys.readouterr().err


def test_cli_preflight_quick(fake_path, capsys, monkeypatch):
    assert cli.main(["preflight", "--quick", *HOST_GATES]) == 0
    out = capsys.readouterr().out
    assert "PASS  gpu_count" in out and "preflight OK" in out
    monkeypatch.setenv("FAKE_NVSMI_SCENARIO", "nv12")
    assert cli.main(["preflight", "--quick", *HOST_GATES]) == 1
    assert "FAIL  topology" in capsys.readouterr().out
    assert cli.main(["preflight", "--quick", "--accept-topology", *HOST_GATES]) == 0


def test_cli_preflight_quick_needs_only_the_standard_library(fake_path, tmp_path):
    """AM27: the quick preflight runs before any download, so neither numpy nor psutil may be imported."""
    code = ("import sys\n"
            "for m in ('numpy', 'psutil'):\n"
            "    sys.modules[m] = None\n"
            "from tpprof import cli\n"
            f"rc = cli.main(['preflight', '--quick', {', '.join(repr(a) for a in HOST_GATES)}])\n"
            "assert 'numpy' not in [m for m, v in sys.modules.items() if v is not None]\n"
            "sys.exit(rc)\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "preflight OK" in r.stdout


def test_cli_full_preflight_reports_model_dir(fake_path, capsys, tmp_path):
    assert cli.main(["preflight", *HOST_GATES, "--model-dir", str(tmp_path / "nomodel")]) == 1
    assert "FAIL  model_files" in capsys.readouterr().out


def test_cli_env_prints_env_json(fake_path, capsys):
    assert cli.main(["env"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["gpu_names"] == ["NVIDIA H100 80GB HBM3"] * 2
    assert "topo_m" in doc and "environ" in doc


def test_cli_predict_writes_predictions(tmp_path, capsys):
    from tpprof import predict

    out = tmp_path / "predictions.md"
    assert cli.main(["predict", "--out", str(out)]) == 0
    assert out.read_text() == predict.render_predictions_md(model.predictions())


def test_cli_run_requires_a_model_dir_without_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("TPPROF_MODEL_DIR", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["run", "--tier", "P0", "--results-dir", str(tmp_path)])
    assert "--model-dir" in capsys.readouterr().err


def test_cli_run_dry_run_subset(tmp_path, capsys):
    res = tmp_path / "results"
    rc = cli.main(["run", "--tier", "P0", "--dry-run", "--only", "preflight,envcapture", "--results-dir", str(res),
                   "--port-base", str(free_port_base())])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 done, 0 failed, 0 skipped" in out
    last = json.loads((res / "raw" / "_last_run.json").read_text())
    assert last["dry_run"] is True and len(last["done"]) == 2
    assert last["only"] == ["preflight", "envcapture"]
    # space-separated kinds work too; both are done already
    assert cli.main(["run", "--tier", "all", "--dry-run", "--only", "preflight", "envcapture",
                     "--results-dir", str(res)]) == 0
    assert "0 done, 0 failed, 2 skipped" in capsys.readouterr().out
    assert leftover_fakes(res) == []
    with pytest.raises(SystemExit):
        cli.main(["run", "--tier", "P0", "--dry-run", "--only", "bogus", "--results-dir", str(res)])


def _trace_record(results_dir, spec, other_frac: float, ar_ops: int, gate_ok: bool = True) -> None:
    d = results_dir / "raw" / spec.run_id
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps(spec.to_dict()))
    tp = 2 if spec.config == "TP2" else 1
    ag = model.allgathers_per_step(tp)
    ranks = [{"rank": r, "pid": 100 + r, "device": r, "steps": 256, "decode_steps": 255,
              "ar_ops_per_step": {"min": ar_ops, "max": ar_ops, "mode": ar_ops},
              "ag_ops_per_step": {"min": ag, "max": ag, "mode": ag}, "mean_step_ms": 4.0, "busy_frac": 0.9,
              "category_frac": {"gemm": 1 - other_frac, "other": other_frac},
              "unclassified_top": [["mystery", other_frac]]} for r in range(tp)]
    (d / "trace_summary.json").write_text(json.dumps(
        {"gate": {"ok": gate_ok, "reasons": [] if gate_ok else ["rank 1 lost its tail"]}, "ranks": ranks,
         "ar_wire_s_per_step": None, "ar_sync_wait_s_per_step": None, "idle_est": None,
         "window_ns": [0, 1], "launch_ts_missing": 0}))
    (d / "done.json").write_text(json.dumps({"run_id": spec.run_id, "status": "done", "duration_s": 1,
                                             "artifacts": []}))


def test_cli_traces_check(tmp_path, capsys):
    traces = [s for s in matrix.p0_specs() if s.kind == "trace"]
    tp1 = next(s for s in traces if s.config == "TP1")
    tp2 = next(s for s in traces if s.config == "TP2")
    _trace_record(tmp_path, tp1, 0.001, 0)
    _trace_record(tmp_path, tp2, 0.002, model.allreduces_per_step(2))
    assert cli.main(["traces", "--check", "--results-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert tp1.run_id in out and tp2.run_id in out and "gate ok" in out and "H6 ok" in out

    bad = next(s for s in traces if s.config == "TP2" and s.p("points") == "decode:b32")
    _trace_record(tmp_path, bad, 0.02, model.allreduces_per_step(2) + 1)
    assert cli.main(["traces", "--check", "--results-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "unclassified 2.00%" in out and "H6 off" in out


def test_cli_traces_check_without_traces(tmp_path, capsys):
    assert cli.main(["traces", "--check", "--results-dir", str(tmp_path)]) == 1
    assert "no trace" in capsys.readouterr().out


def test_cli_traces_requires_check(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["traces", "--results-dir", str(tmp_path)])


def test_python_dash_m_tpprof_help(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([PY, "-m", "tpprof", "--help"], env=env, capture_output=True, text=True, timeout=60,
                       cwd=str(tmp_path))
    assert r.returncode == 0
    for sub in ("preflight", "env", "matrix", "run", "traces", "analyze", "report", "predict"):
        assert sub in r.stdout


def test_cli_analyze_and_report_on_a_dry_run(tmp_path, capsys):
    pytest.importorskip("tpprof.analyze")
    pytest.importorskip("tpprof.report")
    res = tmp_path / "results"
    assert cli.main(["run", "--tier", "P0", "--dry-run", "--only", "preflight,smoke", "--results-dir", str(res),
                     "--port-base", str(free_port_base())]) == 0
    assert cli.main(["analyze", "--results-dir", str(res)]) == 0
    assert (res / "tidy" / "hypotheses.json").exists()
    assert cli.main(["report", "--results-dir", str(res)]) == 0
    summary = (res / "SUMMARY.md").read_text()
    assert summary.splitlines()[0].startswith("> FAKE DATA")
    assert leftover_fakes(res) == []


def test_run_progress_reaches_a_pipe_line_by_line(tmp_path):
    """Review I4: RUN_ON_GPU.md pipes `run` through tee; progress must not sit in an 8 KiB block buffer."""
    res = tmp_path / "results"
    smokes = [s for s in matrix.p0_specs() if s.kind == "smoke"]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH")) if p)
    proc = subprocess.Popen([PY, "-m", "tpprof", "run", "--tier", "P0", "--dry-run", "--only", "smoke",
                             "--results-dir", str(res), "--port-base", str(free_port_base())],
                            cwd=str(tmp_path), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        first = proc.stdout.readline()
        last_done = os.path.exists(os.path.join(str(res), "raw", smokes[-1].run_id, "done.json"))
    finally:
        rest = proc.communicate(timeout=300)[0]
    assert proc.returncode == 0, rest[-3000:]
    assert first.strip() and not last_done, (first, rest[-2000:])
    assert leftover_fakes(res) == []


def _odd_step_trace(results_dir, spec, short_steps: list[int], steps: int = 200) -> None:
    """A TP2 trace whose trace.sqlite has `short_steps` one all-reduce short, the summary matching it."""
    ar = model.allreduces_per_step(2)
    _trace_record(results_dir, spec, 0.001, ar)
    d = results_dir / "raw" / spec.run_id
    summ = json.loads((d / "trace_summary.json").read_text())
    for rk in summ["ranks"]:
        rk["steps"] = steps
        rk["ar_ops_per_step"] = {"min": ar - 1 if short_steps else ar, "max": ar, "mode": ar}
    (d / "trace_summary.json").write_text(json.dumps(summ))
    ranks = [{"pid": 100 + r, "device": r, "tp": 2, "ar_backend": "trtllm", "batch": 1,
              "steps": [[0, 0, 1, 1]] * steps, "drop_last": 0, "short_ar_steps": short_steps} for r in range(2)]
    synth_trace.build_trace_db(str(d / "trace.sqlite"), ranks, measure=synth_trace.trace_span(ranks))


def test_cli_traces_check_applies_the_h6_rule(tmp_path, capsys):
    """Review M5: `traces --check` gates P1 on H6 as the spec states it (>= 99% of steps exact, counted from
    trace.sqlite), not on every step: one odd step in 200 passes, three do not."""
    tp2 = next(s for s in matrix.p0_specs() if s.kind == "trace" and s.config == "TP2")
    _odd_step_trace(tmp_path, tp2, [7])
    assert cli.main(["traces", "--check", "--results-dir", str(tmp_path)]) == 0, capsys.readouterr().out
    assert "199/200" in capsys.readouterr().out

    other = tmp_path / "other"
    _odd_step_trace(other, tp2, [7, 70, 170])
    assert cli.main(["traces", "--check", "--results-dir", str(other)]) == 1
    assert "197/200" in capsys.readouterr().out
