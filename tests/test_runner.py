"""Runner: resumable run records, per-kind handlers, the frozen rate grid, dependency skips, interrupts.

Everything runs against the fakes in dry-run mode (tests/fake_bin, TPPROF_FAKE=1).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time

import psutil
import pytest

from tests.conftest import FAKE_BIN, ROOT
from tpprof import constants, matrix, preflight, runner

PY = sys.executable
GRID_MU = {"TP1": 0.5, "TP2": 0.8, "DP2": 1.0}      # req/s of fake time; gives 200-prompt runs (capped)


def free_port_base() -> int:
    """A port p with p and p + 1 free (DP2-rand uses base and base + 1)."""
    for _ in range(200):
        base = random.randrange(20000, 60000)
        socks = []
        try:
            for p in (base, base + 1):
                s = socket.socket()
                socks.append(s)
                s.bind(("127.0.0.1", p))
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free port pair")


def make_ctx(results_dir, **env_overrides) -> runner.RunContext:
    ctx = runner.dry_run_context(str(results_dir))
    ctx.port_base = free_port_base()
    ctx.rounds = 1
    ctx.base_env.update(env_overrides)
    ctx.log = lambda msg: None
    return ctx


def leftover_fakes(results_dir) -> list[str]:
    """Live processes started for this results dir (the fake GPU state dir is in their environment)."""
    state_dir = os.path.join(os.path.abspath(str(results_dir)), "_gpustate")
    left = []
    for proc in psutil.process_iter():
        try:
            if proc.status() == psutil.STATUS_ZOMBIE:
                continue
            if proc.environ().get("FAKE_GPU_STATE_DIR") != state_dir:
                continue
            left.append(" ".join(proc.cmdline()))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
            continue
    return left


def run_dirs(results_dir) -> dict[str, str]:
    raw = os.path.join(str(results_dir), "raw")
    return {name: os.path.join(raw, name) for name in os.listdir(raw) if not name.startswith("_")}


def status_of(results_dir, run_id: str) -> str | None:
    d = os.path.join(str(results_dir), "raw", run_id)
    for name, status in (("done.json", "done"), ("failed.json", "failed")):
        if os.path.exists(os.path.join(d, name)):
            return status
    return None


def read(path):
    with open(path) as f:
        return json.load(f)


def skipped_reasons(summary: dict) -> dict[str, str]:
    return {e["run_id"]: e["reason"] for e in summary["skipped"]}


def write_grid(results_dir, mu=GRID_MU) -> dict:
    raw = os.path.join(str(results_dir), "raw")
    os.makedirs(raw, exist_ok=True)
    doc = {"mu_rps": mu, "grid": matrix.rate_grid(mu), "sources": ["test"]}
    with open(os.path.join(raw, "rate_grid.json"), "w") as f:
        json.dump(doc, f)
    return doc


# ---------------------------------------------------------------- dry-run context


def test_dry_run_context(tmp_path):
    ctx = runner.dry_run_context(str(tmp_path / "res"))
    env = ctx.base_env
    path = env["PATH"].split(os.pathsep)
    assert path[0] == str(FAKE_BIN)
    assert path[1] == os.path.dirname(sys.executable)
    assert "" not in path
    assert env["TPPROF_FAKE"] == "1"
    assert env["FAKE_TIME_SCALE"] == "0.002"
    assert env["FAKE_GPU_STATE_DIR"] == str(tmp_path / "res" / "_gpustate")
    assert env["TPPROF_NSYS"] == str(FAKE_BIN / "nsys")
    assert str(ROOT) in env["PYTHONPATH"].split(os.pathsep)
    assert ctx.dry_run and ctx.results_dir == str(tmp_path / "res")
    assert ctx.model_dir == str(tmp_path / "res" / "_model")
    assert preflight.check_model_files(ctx.model_dir).ok
    for name, size in constants.MODEL_SHARD_SIZES.items():
        assert os.path.getsize(os.path.join(ctx.model_dir, name)) == size
    # idempotent: a second context over the same dir reuses the model dir
    assert runner.dry_run_context(str(tmp_path / "res")).model_dir == ctx.model_dir


# ---------------------------------------------------------------- P0


@pytest.fixture(scope="module")
def p0(tmp_path_factory):
    res = tmp_path_factory.mktemp("p0") / "results"
    ctx = make_ctx(res)
    t0 = time.monotonic()
    summary = runner.Runner(ctx).run_tiers(["P0"])
    return res, ctx, summary, time.monotonic() - t0


@pytest.mark.slow
def test_dry_run_p0_completes(p0):
    res, _, summary, _ = p0
    specs = matrix.p0_specs()
    assert summary["failed"] == [] and summary["skipped"] == []
    assert summary["done"] == [s.run_id for s in specs]
    for s in specs:
        assert status_of(res, s.run_id) == "done", s.run_id
        d = os.path.join(str(res), "raw", s.run_id)
        assert read(os.path.join(d, "spec.json")) == json.loads(json.dumps(s.to_dict()))
        cmds = read(os.path.join(d, "cmd.json"))
        assert (cmds == []) == (s.kind in ("preflight", "envcapture")), s.run_id
        for c in cmds:
            for key in ("argv", "env_overrides", "cwd", "t_wall_start", "t_mono_start", "t_wall_end",
                        "t_mono_end", "exit_code", "timed_out", "log"):
                assert key in c, (s.run_id, key)
        done = read(os.path.join(d, "done.json"))
        assert done["run_id"] == s.run_id and done["status"] == "done" and done["artifacts"]
    last = read(os.path.join(str(res), "raw", "_last_run.json"))
    assert last["done"] == summary["done"]

    for s in specs:
        d = os.path.join(str(res), "raw", s.run_id)
        if s.kind == "offline":
            from tpprof.offline import parse_points
            for p in parse_points(str(s.p("points"))):
                point = read(os.path.join(d, "points", p.filename()))
                assert point["tpprof"]["config"] == s.config and point["tpprof"]["arm"] == s.arm
                assert point["tpprof"]["engine"] == "fake"
        elif s.kind == "trace":
            summ = read(os.path.join(d, "trace_summary.json"))
            assert summ["gate"]["ok"], (s.run_id, summ["gate"])
            assert summ["idle_est"] is not None      # the matching offline run gave the untraced step time
            assert os.path.exists(os.path.join(d, "trace.sqlite"))
        elif s.kind == "smoke":
            eff = read(os.path.join(d, "effective_config.json"))
            assert eff["violations"] == [], (s.run_id, eff["violations"])
            assert eff["startup_complete"] and eff["kv_cache_tokens"]
            assert read(os.path.join(d, "sub-0", "validation.json"))["valid"]
            if s.config == "TP2":
                assert eff["fi_backend"] == "mnnvl"
            if s.p("gpu1_check"):
                point = read(os.path.join(d, "points", "point-decode-b1-i1024-o64.json"))
                assert point["tpprof"]["config"] == "TP1"
                cmd = [c for c in read(os.path.join(d, "cmd.json")) if "tpprof.offline" in c["argv"]]
                assert cmd and cmd[0]["env_overrides"]["CUDA_VISIBLE_DEVICES"] == "1"
        elif s.kind in ("comm_m2", "comm_m3"):
            with open(os.path.join(d, "comm_rows.jsonl")) as f:
                assert len([line for line in f if line.strip()]) > 10
        elif s.kind == "bench_latency_xcheck":
            assert read(os.path.join(d, "bench_latency.json"))["latencies"]
        elif s.kind == "preflight":
            report = read(os.path.join(d, "preflight.json"))
            assert report["ok"]
        elif s.kind == "envcapture":
            assert "topo_m" in read(os.path.join(d, "env.json"))
        for name in ("effective_config.json",):
            path = os.path.join(d, name)
            if os.path.exists(path):
                assert read(path)["violations"] == [], (s.run_id, read(path)["violations"])
    assert leftover_fakes(res) == []


@pytest.mark.slow
def test_resume_skips_done(p0):
    res, ctx, first, _ = p0
    t0 = time.monotonic()
    summary = runner.Runner(ctx).run_tiers(["P0"])
    elapsed = time.monotonic() - t0
    assert summary["done"] == [] and summary["failed"] == []
    assert list(skipped_reasons(summary).items()) == [(r, "already_done") for r in first["done"]]
    assert elapsed < 2.0, elapsed
    assert leftover_fakes(res) == []


def test_p0_subset_through_only(tmp_path):
    res = tmp_path / "results"
    kinds = ("preflight", "smoke", "bench_latency_xcheck")
    summary = runner.Runner(make_ctx(res)).run_tiers(["P0"], only_kinds=kinds)
    specs = [s for s in matrix.p0_specs() if s.kind in kinds]
    assert summary["failed"] == [] and summary["skipped"] == []
    assert summary["done"] == [s.run_id for s in specs]
    assert set(run_dirs(res)) == {s.run_id for s in specs}
    for s in specs:
        d = os.path.join(str(res), "raw", s.run_id)
        assert os.path.exists(os.path.join(d, "done.json"))
        if s.kind == "smoke":
            assert read(os.path.join(d, "effective_config.json"))["violations"] == []
            assert read(os.path.join(d, "sub-0", "validation.json"))["valid"]
            for name in ("gpu.csv", "cpu.csv"):
                assert os.path.getsize(os.path.join(d, name)) > 0
    assert leftover_fakes(res) == []


def test_preflight_failure_stops_the_tier(tmp_path):
    res = tmp_path / "results"
    summary = runner.Runner(make_ctx(res, FAKE_NVSMI_SCENARIO="one_gpu")).run_tiers(["P0"])
    pre = matrix.p0_specs()[0]
    assert pre.kind == "preflight"
    assert summary["failed"] == [pre.run_id] and summary["done"] == []
    failed = read(os.path.join(str(res), "raw", pre.run_id, "failed.json"))
    assert failed["reason"] == "preflight_failed" and "gpu_count" in failed["detail"]
    reasons = skipped_reasons(summary)
    assert list(reasons) == [s.run_id for s in matrix.p0_specs()[1:]]
    assert set(reasons.values()) == {f"preflight_failed:{pre.run_id}"}
    assert set(run_dirs(res)) == {pre.run_id}


def test_gpus_busy_fails_the_precheck(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "GPU_PRECHECK_TIMEOUT_S", 0.5)
    res = tmp_path / "results"
    summary = runner.Runner(make_ctx(res, FAKE_NVSMI_SCENARIO="busy")).run_tiers(["P0"], only_kinds=("comm_m2",))
    [run_id] = summary["failed"]
    failed = read(os.path.join(str(res), "raw", run_id, "failed.json"))
    assert failed["status"] == "failed" and failed["reason"] == "gpus_busy"
    assert "30000" in failed["detail"]
    assert read(os.path.join(str(res), "raw", run_id, "cmd.json")) == []


def test_comm_runs_and_m4_without_nccl_tests(tmp_path):
    res = tmp_path / "results"
    kinds = ("comm_m1", "comm_m3", "comm_m4")
    summary = runner.Runner(make_ctx(res)).run_tiers(["P2"], only_kinds=kinds)
    specs = [s for s in matrix.p2_specs(None, None) if s.kind in kinds]
    m4 = [s.run_id for s in specs if s.kind == "comm_m4"]
    assert summary["failed"] == []
    assert summary["done"] == [s.run_id for s in specs if s.kind != "comm_m4"]
    assert skipped_reasons(summary) == {m4[0]: "dry_run:no_nccl_tests"}
    for run_id in summary["done"]:
        with open(os.path.join(str(res), "raw", run_id, "comm_rows.jsonl")) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert rows
    m3 = [s for s in specs if s.kind == "comm_m3"][0]
    cmd = read(os.path.join(str(res), "raw", m3.run_id, "cmd.json"))[0]
    algo, proto = m3.p("variant").split(":")
    assert cmd["env_overrides"]["NCCL_ALGO"] == f"allreduce:{algo}"
    assert cmd["env_overrides"]["NCCL_PROTO"] == f"allreduce:{proto}"


def test_p2_sessions_without_grid_are_skipped(tmp_path):
    res = tmp_path / "results"
    ctx = make_ctx(res, TPPROF_TEST_FAIL_CONFIG="TP2")
    summary = runner.Runner(ctx).run_tiers(["P2"], only_kinds=("serve_session",))
    reasons = skipped_reasons(summary)
    assert sorted(reasons.values()) == ["no_rate_grid"] * 3          # A-PC base, A-PC on, A-RAND
    api2 = [s for s in matrix.p2_specs(None, None) if s.kind == "serve_session"]
    tp2 = [s.run_id for s in api2 if s.config == "TP2"]
    dp2 = [s.run_id for s in api2 if s.config == "DP2"]
    assert summary["failed"] == tp2 and summary["done"] == dp2
    assert read(os.path.join(str(res), "raw", tp2[0], "failed.json"))["reason"] == "server_start_failed"
    assert leftover_fakes(res) == []


def test_server_crash_mid_session_is_recorded(tmp_path):
    """Review focus 2: the crashed client run is recorded invalid with a reason, the session failed."""
    res = tmp_path / "results"
    r = runner.Runner(make_ctx(res, FAKE_VLLM_CRASH_AFTER="30"))
    sat = [s for s in matrix.p1_sat_specs() if s.kind == "serve_session"][0]
    assert r.run_spec(sat) == "failed" and r.last_reason == "server_died"
    d = os.path.join(str(res), "raw", sat.run_id)
    failed = read(os.path.join(d, "failed.json"))
    assert failed["reason"] == "server_died" and "exited" in failed["detail"]
    v = read(os.path.join(d, "sub-0", "validation.json"))
    assert not v["valid"] and any("failed=" in x or "errors" in x for x in v["violations"]), v
    assert read(os.path.join(d, "sub-0", "meta.json"))["k"] == 0
    assert not os.path.exists(os.path.join(d, "sub-1"))
    assert leftover_fakes(res) == []


# ---------------------------------------------------------------- P1


@pytest.mark.slow
def test_failed_session_skips_dependents_and_resumes(tmp_path):
    res = tmp_path / "results"
    grid = write_grid(res)
    ctx = make_ctx(res, TPPROF_TEST_FAIL_CONFIG="TP2")
    summary = runner.Runner(ctx).run_tiers(["P1"])
    sat = {s.config: s for s in matrix.p1_sat_specs() if s.kind == "serve_session"}
    sweeps = {s.config: s for s in matrix.p1_sweep_specs(grid["grid"], 1)}
    tok = [s for s in matrix.p1_sat_specs() if s.kind == "tokbench"][0]

    assert summary["failed"] == [sat["TP2"].run_id]
    failed = read(os.path.join(str(res), "raw", sat["TP2"].run_id, "failed.json"))
    assert failed["reason"] == "server_start_failed" and "EngineCore failed to start" in failed["detail"]
    assert failed["log_tails"] and all(isinstance(v, list) for v in failed["log_tails"].values())
    assert skipped_reasons(summary) == {sweeps["TP2"].run_id: f"dependency_failed:{sat['TP2'].run_id}"}
    assert sorted(summary["done"]) == sorted([sat["TP1"].run_id, sat["DP2"].run_id, tok.run_id,
                                               sweeps["TP1"].run_id, sweeps["DP2"].run_id])
    assert leftover_fakes(res) == []

    # without --retry-failed the failure stays recorded and is not retried
    again = runner.Runner(make_ctx(res)).run_tiers(["P1"])
    assert again["done"] == [] and again["failed"] == []
    assert skipped_reasons(again)[sat["TP2"].run_id] == "previously_failed:server_start_failed"
    assert skipped_reasons(again)[sweeps["TP2"].run_id] == f"dependency_failed:{sat['TP2'].run_id}"

    # the hook removed and --retry-failed: the TP2 sessions run and the old failure is kept aside
    retry = runner.Runner(make_ctx(res)).run_tiers(["P1"], retry_failed=True)
    assert retry["failed"] == []
    assert retry["done"] == [sat["TP2"].run_id, sweeps["TP2"].run_id]
    d = os.path.join(str(res), "raw", sat["TP2"].run_id)
    assert os.path.exists(os.path.join(d, "done.json")) and not os.path.exists(os.path.join(d, "failed.json"))
    assert read(os.path.join(d, "failed.prev.json"))["reason"] == "server_start_failed"
    assert leftover_fakes(res) == []


@pytest.mark.slow
def test_rate_grid_written_once(tmp_path):
    res = tmp_path / "results"
    ctx = make_ctx(res)
    summary = runner.Runner(ctx).run_tiers(["P1"])
    assert summary["failed"] == [] and summary["skipped"] == []
    path = os.path.join(str(res), "raw", "rate_grid.json")
    doc = read(path)
    sat = [s for s in matrix.p1_sat_specs() if s.kind == "serve_session"]
    assert set(doc) == {"mu_rps", "grid", "sources"}
    assert doc["sources"] == [s.run_id for s in sat]
    assert doc["grid"] == matrix.rate_grid(doc["mu_rps"])
    sweeps = matrix.p1_sweep_specs(doc["grid"], 1)
    assert [r for r in summary["done"] if "-serve_session-" in r][3:] == [s.run_id for s in sweeps]

    # R4 sub-run layout of a sweep session
    d = os.path.join(str(res), "raw", sweeps[0].run_id)
    rates = sorted(sweeps[0].p("rates"))
    for k, rate in enumerate(rates):
        sub = os.path.join(d, f"sub-{k}")
        for name in ("result.json", "client.log", "metrics_before.prom", "metrics_after.prom",
                     "validation.json", "meta.json"):
            assert os.path.exists(os.path.join(sub, name)), (k, name)
        meta = read(os.path.join(sub, "meta.json"))
        assert meta == {"phase": "sweep", "rate": rate, "seed": sweeps[0].p("seed_base") + k,
                        "config": sweeps[0].config, "arm": "base", "round": 1, "k": k}
        v = read(os.path.join(sub, "validation.json"))
        assert v["valid"], v["violations"]
        result = read(os.path.join(sub, "result.json"))
        assert result["tpprof_config"] == sweeps[0].config and result["request_rate"] == rate
    sat_sub = os.path.join(str(res), "raw", sat[0].run_id, "sub-0")
    assert read(os.path.join(sat_sub, "meta.json"))["rate"] == "inf"
    assert os.path.exists(os.path.join(sat_sub, "gauges.csv"))

    before = (os.path.getmtime(path), open(path).read())
    again = runner.Runner(make_ctx(res)).run_tiers(["P1"])
    assert again["done"] == [] and again["failed"] == []
    assert (os.path.getmtime(path), open(path).read()) == before
    assert leftover_fakes(res) == []


@pytest.mark.slow
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_interrupt_marks_failed_and_exits_130(tmp_path, sig):
    """Ctrl-C, and (review I1) SIGTERM from a kill or SIGHUP from a killed tmux window, stop the same way."""
    res = tmp_path / "results"
    port = free_port_base()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p))
    log = open(tmp_path / "runner.log", "w")
    proc = subprocess.Popen([PY, "-m", "tpprof", "run", "--tier", "P1", "--dry-run", "--rounds", "1",
                             "--results-dir", str(res), "--port-base", str(port)],
                            cwd=str(tmp_path), env=env, stdout=log, stderr=subprocess.STDOUT)
    sat = [s for s in matrix.p1_sat_specs() if s.kind == "serve_session"][1]     # TP2, mid-P1
    sub = os.path.join(str(res), "raw", sat.run_id, "sub-0")
    deadline = time.monotonic() + 120
    while not os.path.exists(sub) and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is None, open(tmp_path / "runner.log").read()
    proc.send_signal(sig)
    assert proc.wait(timeout=120) == 130, open(tmp_path / "runner.log").read()
    log.close()
    failed = read(os.path.join(str(res), "raw", sat.run_id, "failed.json"))
    assert failed["reason"] == "interrupted" and signal.Signals(sig).name in failed["detail"]
    last = read(os.path.join(str(res), "raw", "_last_run.json"))
    assert last["interrupted"] is True and last["failed"] == [sat.run_id]
    assert leftover_fakes(res) == []
    assert _own_fake_vllm(port) == []

    # resume: an interrupted run is retried without --retry-failed
    ctx = make_ctx(res)
    ctx.port_base = port
    r = runner.Runner(ctx)
    assert r.run_spec(sat) == "done"
    assert read(os.path.join(str(res), "raw", sat.run_id, "failed.prev.json"))["reason"] == "interrupted"
    assert leftover_fakes(res) == []


def _own_fake_vllm(port: int) -> list[psutil.Process]:
    out = []
    for p in psutil.process_iter():
        try:
            cmd = p.cmdline()
            if any(str(FAKE_BIN / "vllm") == c for c in cmd) and str(port) in cmd:
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
            continue
    return out


# ---------------------------------------------------------------- offline points and the engine check (review C1)

LOGS = ROOT / "tests" / "fixtures" / "logs"
OFFLINE_TP1 = matrix.RunSpec("offline", "TP1", "base", "P0", {"points": "decode:b1,b8"})


def box_ctx(results_dir, engine_log) -> runner.RunContext:
    """The dry-run context with the effective-config check on, as on the box: the fake engine prints a real
    engine's startup lines (FAKE_ENGINE_LOG)."""
    ctx = make_ctx(results_dir, FAKE_ENGINE_LOG=str(engine_log))
    ctx.dry_run = False
    return ctx


def point_files(run_dir) -> dict[str, float]:
    """Point file name -> the t_wall_start the driver recorded in it."""
    d = os.path.join(run_dir, "points")
    if not os.path.isdir(d):
        return {}
    return {n: read(os.path.join(d, n))["tpprof"]["t_wall_start"] for n in sorted(os.listdir(d))}


def test_offline_config_violation_quarantines_points_and_retry_remeasures(tmp_path):
    res = tmp_path / "results"
    r = runner.Runner(box_ctx(res, LOGS / "tp1_sampling_override.log"))
    assert r.run_spec(OFFLINE_TP1) == "failed" and r.last_reason == "effective_config"
    d = os.path.join(str(res), "raw", OFFLINE_TP1.run_id)
    assert point_files(d) == {}                  # a misconfigured engine's points are not data
    rejected = [n for n in os.listdir(d) if n.startswith("rejected-points")]
    assert len(rejected) == 1 and len(os.listdir(os.path.join(d, rejected[0]))) == 4

    t_retry = time.time()
    r = runner.Runner(box_ctx(res, LOGS / "tp1_base.log"))
    r.retry_failed = True
    assert r.run_spec(OFFLINE_TP1) == "done", r.last_reason
    points = point_files(d)
    assert len(points) == 4 and min(points.values()) >= t_retry      # measured again, by the vetted engine
    assert read(os.path.join(d, "effective_config.json"))["violations"] == []


def test_offline_retry_with_every_point_vetted_skips_the_driver(tmp_path):
    res = tmp_path / "results"
    ctx = box_ctx(res, LOGS / "tp1_base.log")
    assert runner.Runner(ctx).run_spec(OFFLINE_TP1) == "done"
    d = os.path.join(str(res), "raw", OFFLINE_TP1.run_id)
    before = point_files(d)
    # the driver wrote every point with a vetted engine, then failed (say, a timeout in engine teardown)
    os.remove(os.path.join(d, "done.json"))
    with open(os.path.join(d, "failed.json"), "w") as f:
        json.dump({"run_id": OFFLINE_TP1.run_id, "status": "failed", "reason": "timeout", "detail": "",
                   "log_tails": {}}, f)
    r = runner.Runner(ctx)
    r.retry_failed = True
    assert r.run_spec(OFFLINE_TP1) == "done", r.last_reason
    assert point_files(d) == before
    assert read(os.path.join(d, "effective_config.json"))["violations"] == []
    assert not os.path.exists(os.path.join(d, "offline.log"))        # the driver was not started again


def test_rerun_after_a_failed_preflight_runs_it_again_with_the_new_gate_options(tmp_path):
    """Review I3: RUN_ON_GPU.md's recovery is the same command plus --accept-topology or --skip-gate."""
    res = tmp_path / "results"
    kinds = ("preflight", "envcapture")
    pf = next(s for s in matrix.p0_specs() if s.kind == "preflight")
    ctx = make_ctx(res, FAKE_NVSMI_SCENARIO="nv12")
    summary = runner.Runner(ctx).run_tiers(["P0"], only_kinds=kinds)
    assert summary["failed"] == [pf.run_id]
    ctx.accept_topology = True
    summary = runner.Runner(ctx).run_tiers(["P0"], only_kinds=kinds)
    assert summary["failed"] == [] and summary["skipped"] == [], summary
    assert pf.run_id in summary["done"]


def test_shutdown_lines_the_stop_provokes_do_not_fail_a_finished_run(tmp_path):
    """Review I5: the final engine check reads the server logs only up to the stop the runner itself sent."""
    res = tmp_path / "results"
    ctx = make_ctx(res, FAKE_VLLM_SHUTDOWN_LINE="Worker proc VllmWorker-0 died unexpectedly, shutting down executor.")
    smoke = next(s for s in matrix.p0_specs() if s.kind == "smoke" and s.config == "TP2")
    r = runner.Runner(ctx)
    assert r.run_spec(smoke) == "done", r.last_reason
    d = os.path.join(str(res), "raw", smoke.run_id)
    [log] = [n for n in os.listdir(d) if n.startswith("server-") and n.endswith(".log")]
    assert "died unexpectedly" in open(os.path.join(d, log)).read()       # logged, but after the stop
    assert read(os.path.join(d, "effective_config.json"))["violations"] == []


def test_resume_after_a_hard_kill_sweeps_orphans_and_sets_the_attempt_aside(tmp_path):
    """Review I1: a runner killed without cleanup leaves a run dir with neither done.json nor failed.json,
    orphans tagged with its run_id, and logs the next attempt would append to."""
    res = tmp_path / "results"
    ctx = make_ctx(res)
    smoke = next(s for s in matrix.p0_specs() if s.kind == "smoke" and s.config == "TP1")
    d = os.path.join(str(res), "raw", smoke.run_id)
    server_log = f"server-{ctx.port_base}.log"
    os.makedirs(os.path.join(d, "sub-0"))
    with open(os.path.join(d, "spec.json"), "w") as f:
        json.dump(smoke.to_dict(), f)
    with open(os.path.join(d, "cmd.json"), "w") as f:
        json.dump([{"argv": ["the-killed-attempt"], "log": server_log}], f)
    shutil.copy(LOGS / "tp1_base.log", os.path.join(d, server_log))
    orphan = subprocess.Popen([PY, "-c", "import time; time.sleep(120)"], start_new_session=True,
                              env=dict(os.environ, TPPROF_RUN_ID=smoke.run_id))
    try:
        summary = runner.Runner(ctx).run_tiers(["P0"], only_kinds=("smoke",))
        assert orphan.wait(timeout=10) is not None                # killed by the runner, not by this test
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait()
    assert smoke.run_id in summary["done"], summary
    assert read(os.path.join(d, "effective_config.json"))["violations"] == []
    assert sorted(os.listdir(os.path.join(d, "prev-attempt"))) == sorted(["cmd.json", server_log, "sub-0"])
    assert all(c["argv"] != ["the-killed-attempt"] for c in read(os.path.join(d, "cmd.json")))


def test_after_gpus_busy_the_invocation_skips_its_other_gpu_runs(tmp_path, monkeypatch):
    """Review I1: one busy pre-check stops the invocation's GPU runs instead of waiting before each of them."""
    monkeypatch.setattr(runner, "GPU_PRECHECK_TIMEOUT_S", 0.5)
    res = tmp_path / "results"
    summary = runner.Runner(make_ctx(res, FAKE_NVSMI_SCENARIO="busy")).run_tiers(["P0"], only_kinds=("smoke",))
    smokes = [s.run_id for s in matrix.p0_specs() if s.kind == "smoke"]
    assert summary["failed"] == smokes[:1]
    reasons = skipped_reasons(summary)
    assert [reasons.get(r) for r in smokes[1:]] == [f"gpus_busy:{smokes[0]}"] * 2


def test_a_client_timeout_stops_the_session(tmp_path, monkeypatch):
    """Review I2: a client still running at its own timeout ends the session (the engine may be wedged); the
    next clients do not each wait that long again behind an invalid sub-run."""
    monkeypatch.setattr(runner, "MIN_TIMEOUT_S", 1.0)
    monkeypatch.setattr(runner, "TIMEOUT_FACTOR", 0.0)
    res = tmp_path / "results"
    ctx = make_ctx(res, FAKE_VLLM_TTFT_S="2000")            # 4 s per request at the dry-run time scale
    sat = next(s for s in matrix.p1_sat_specs() if s.kind == "serve_session")
    r = runner.Runner(ctx)
    t0 = time.monotonic()
    assert r.run_spec(sat) == "failed" and r.last_reason == "client_timeout"
    d = os.path.join(str(res), "raw", sat.run_id)
    assert sorted(n for n in os.listdir(d) if n.startswith("sub-")) == ["sub-0"]
    assert time.monotonic() - t0 < 60
    assert leftover_fakes(res) == []


def test_a_saturation_session_without_a_valid_run_fails(tmp_path):
    """Review I2: a done sat session with no valid sub-run would block P1 for good: the rate grid can never be
    computed, and --retry-failed does not redo done runs."""
    res = tmp_path / "results"
    ctx = make_ctx(res, FAKE_VLLM_PROMPT_TOKENS="999")      # every request reports the wrong input length
    sat = next(s for s in matrix.p1_sat_specs() if s.kind == "serve_session")
    r = runner.Runner(ctx)
    assert r.run_spec(sat) == "failed" and r.last_reason == "no_valid_saturation_run"
    assert leftover_fakes(res) == []


# ---------------------------------------------------------------- M3 NCCL verification (review I6, spec 4.6)

def m3_spec(variant: str) -> matrix.RunSpec:
    return next(s for s in matrix.p2_specs(None, "mnnvl", True) if s.kind == "comm_m3" and s.p("variant") == variant)


def test_m3_variant_is_verified_from_nccl_tuning_lines(tmp_path):
    """A short diagnostic run of the variant under NCCL_DEBUG=INFO must show the requested algorithm and protocol
    in NCCL's TUNING lines; the loaded NCCL version and the NVLS multicast line are recorded."""
    res = tmp_path / "results"
    spec = m3_spec("ring:LL")
    r = runner.Runner(make_ctx(res))
    assert r.run_spec(spec) == "done", r.last_reason
    d = os.path.join(str(res), "raw", spec.run_id)
    v = read(os.path.join(d, "nccl_verify.json"))
    assert v["verified"] is True and v["observed"] == {"algos": ["RING"], "protos": ["LL"]}
    assert v["nccl_version"] == constants.NCCL_EXPECTED and v["nvls_support"] is True
    assert os.path.exists(os.path.join(d, "comm_rows.jsonl"))


def test_m3_variant_nccl_did_not_run_fails_without_rows(tmp_path):
    res = tmp_path / "results"
    spec = m3_spec("ring:LL")
    r = runner.Runner(make_ctx(res, FAKE_NCCL_TUNING_ALGO="TREE"))
    assert r.run_spec(spec) == "failed" and r.last_reason == "nccl_variant_mismatch"
    assert not os.path.exists(os.path.join(str(res), "raw", spec.run_id, "comm_rows.jsonl"))


def test_m3_variant_nccl_rejects_is_recorded_as_unsupported(tmp_path):
    res = tmp_path / "results"
    spec = m3_spec("tree:LL128")
    r = runner.Runner(make_ctx(res, FAKE_NCCL_UNSUPPORTED="tree:LL128"))
    assert r.run_spec(spec) == "failed" and r.last_reason == "nccl_unsupported"
    assert read(os.path.join(str(res), "raw", spec.run_id, "nccl_verify.json"))["unsupported"] is True


def test_m3_nvls_bind_failure_is_retried_without_nvls(tmp_path):
    res = tmp_path / "results"
    ctx = make_ctx(res, FAKE_NCCL_NVLS_BIND_FAIL="1")
    spec = m3_spec("ring:LL")
    r = runner.Runner(ctx)
    assert r.run_spec(spec) == "done", r.last_reason
    d = os.path.join(str(res), "raw", spec.run_id)
    assert read(os.path.join(d, "nccl_verify.json"))["nvls_disabled_retry"] is True
    first, retry = read(os.path.join(d, "cmd.json"))[:2]
    assert (first["exit_code"], retry["exit_code"]) == (1, 0)
    assert retry["env_overrides"].get("NCCL_NVLS_ENABLE") == "0"
    # NVLS itself cannot run without NVLS: unsupported, not retried
    nvls = m3_spec("nvls:Simple")
    r = runner.Runner(ctx)
    assert r.run_spec(nvls) == "failed" and r.last_reason == "nccl_unsupported"
