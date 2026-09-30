from __future__ import annotations

import csv
import http.server
import json
import math
import os
import socket
import sys
import threading
import time

import psutil
import pytest

from tests.conftest import FAKE_BIN, FIXTURES, fake_env
from tpprof import engine, helpflags, procs, promparse, results
from tpprof.client import ClientSpec, client_argv, run_clients
from tpprof.constants import GPU_FREE_MIB
from tpprof.server import (MetricsPoller, ServerHandle, ServerStartError, scrape_metrics, start_server,
                           stop_server, stop_servers)

VLLM = str(FAKE_BIN / "vllm")
NVSMI = str(FAKE_BIN / "nvidia-smi")
MODEL_DIR = "/models/llama"


def free_ports(n: int) -> list[int]:
    socks = []
    try:
        for _ in range(n):
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def sample_spec(**overrides) -> ClientSpec:
    kw = dict(base_url="http://127.0.0.1:8000", tokenizer="/models/llama", input_len=1024, output_len=256,
              prefix_len=0, num_prompts=3000, request_rate=math.inf, seed=1, result_dir="/runs/x",
              result_filename="sat.json", request_id_prefix="P1-serve_session-TP1-base-r0-0123abcd-",
              metadata=(("tpprof_config", "TP1"), ("tpprof_rate", "inf"), ("tpprof_round", "0")))
    kw.update(overrides)
    return ClientSpec(**kw)


# ---------------------------------------------------------------- client argv


def test_client_argv_flags_known_to_real_bench_serve():
    help_text = (FIXTURES / "bench_serve_help.txt").read_text()
    assert helpflags.missing_flags(client_argv(sample_spec()), help_text) == []
    assert helpflags.missing_flags(client_argv(sample_spec(request_rate=12.3, prefix_len=64)), help_text) == []


def test_client_argv_exact_order():
    assert client_argv(sample_spec(), vllm_bin="/usr/bin/vllm") == [
        "/usr/bin/vllm", "bench", "serve", "--backend", "vllm", "--base-url", "http://127.0.0.1:8000",
        "--endpoint", "/v1/completions", "--model", "llama-3.1-8b-instruct",
        "--tokenizer", "/models/llama", "--dataset-name", "random", "--random-input-len", "1024",
        "--random-output-len", "256",
        "--random-range-ratio", "0.0", "--random-prefix-len", "0", "--num-prompts", "3000", "--request-rate", "inf",
        "--burstiness", "1.0", "--seed", "1", "--ignore-eos", "--temperature", "1.0", "--top-p", "1.0",
        "--num-warmups", "16",
        "--ready-check-timeout-sec", "60", "--save-result", "--save-detailed", "--result-dir", "/runs/x",
        "--result-filename", "sat.json",
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "25,50,75,90,99", "--disable-tqdm",
        "--request-id-prefix", "P1-serve_session-TP1-base-r0-0123abcd-",
        "--metadata", "tpprof_config=TP1", "tpprof_rate=inf", "tpprof_round=0",
    ]
    argv = client_argv(sample_spec(request_rate=12.3))
    assert argv[0] == "vllm"
    assert argv[argv.index("--request-rate") + 1] == "12.3"


@pytest.mark.parametrize("metadata", [
    (("config", "TP1"),),
    (("tpprof_config", "TP1"), ("rate", "1.0")),
    (("tpprof_", "x"),),
    (("tpprof_a=b", "x"),),
    (("tpprof_a b", "x"),),
])
def test_metadata_keys_must_be_prefixed(metadata):
    with pytest.raises(ValueError, match="tpprof_"):
        client_argv(sample_spec(metadata=metadata))


@pytest.mark.parametrize("rate", [0.0, -1.0, math.nan])
def test_request_rate_must_be_positive(rate):
    with pytest.raises(ValueError, match="request_rate"):
        sample_spec(request_rate=rate)


# ---------------------------------------------------------------- server lifecycle


def test_start_scrape_stop_with_fake(tmp_path):
    env = fake_env(tmp_path)
    (port,) = free_ports(1)
    cfg = engine.base_config("TP1")
    h = start_server(cfg, MODEL_DIR, port, str(tmp_path / "run"), env, "run-t1", startup_timeout_s=30,
                     vllm_bin=VLLM)
    try:
        assert h.base_url == f"http://127.0.0.1:{port}" and h.port == port and h.cfg is cfg
        assert h.log_path == str(tmp_path / "run" / f"server-{port}.log")
        assert 0 < h.ready_s < 30
        assert h.proc.argv == cfg.serve_argv(MODEL_DIR, port, VLLM)
        summary = promparse.scrape_summary(scrape_metrics(h.base_url))
        assert set(summary) == {*promparse.TRACKED, *promparse.GAUGES}
        assert procs.gpu_memory_used(NVSMI, env)[0] >= GPU_FREE_MIB
    finally:
        freed = stop_server(h, nvidia_smi=NVSMI, env=env)
    assert freed is True
    assert h.proc.popen.poll() is not None and gone(h.proc.popen.pid)
    assert procs.gpu_memory_used(NVSMI, env)[0] < GPU_FREE_MIB
    log = (tmp_path / "run" / f"server-{port}.log").read_text()
    assert "Application startup complete." in log


def test_start_failure_raises_with_log_tail(tmp_path):
    env = fake_env(tmp_path, FAKE_VLLM_FAIL_START="1")
    (port,) = free_ports(1)
    t0 = time.monotonic()
    with pytest.raises(ServerStartError) as ei:
        start_server(engine.base_config("TP1"), MODEL_DIR, port, str(tmp_path), env, "run-fail",
                     startup_timeout_s=30, vllm_bin=VLLM)
    assert time.monotonic() - t0 < 15
    msg = str(ei.value)
    assert "exited with code 1" in msg
    assert "EngineCore failed to start." in msg
    assert f"server-{port}.log" in msg


def test_start_hang_times_out(tmp_path):
    env = fake_env(tmp_path, FAKE_VLLM_HANG_START="1")
    (port,) = free_ports(1)
    run_dir = tmp_path / "run"
    t0 = time.monotonic()
    with pytest.raises(ServerStartError, match="not healthy after 1 s") as ei:
        start_server(engine.base_config("TP1"), MODEL_DIR, port, str(run_dir), env, "run-hang",
                     startup_timeout_s=1, vllm_bin=VLLM)
    assert time.monotonic() - t0 < 15
    assert "Initializing a V1 LLM engine" in str(ei.value)
    # The hung server was stopped: nothing carries the run tag, and its GPU state is gone.
    assert procs.sweep_tagged("run-hang") == []
    assert list((tmp_path / "gpustate").glob("gpu*.json")) == []


def test_start_refuses_a_port_that_already_serves(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        with pytest.raises(ServerStartError, match="already accepts connections"):
            start_server(engine.base_config("TP1"), MODEL_DIR, port, str(tmp_path), fake_env(tmp_path),
                         "run-busy", startup_timeout_s=5, vllm_bin=VLLM)
    assert not (tmp_path / f"server-{port}.log").exists()


def test_stop_servers_stops_every_server_even_if_one_survives_sigkill(tmp_path, monkeypatch):
    calls = []

    def fake_stop(p, *, sweep=True):
        calls.append(("stop", p.run_id, sweep))
        if p is handles[0].proc:
            raise RuntimeError("still running after SIGKILL")
        return 0

    monkeypatch.setattr(procs, "stop", fake_stop)
    monkeypatch.setattr(procs, "sweep_tagged", lambda run_id: calls.append(("sweep", run_id)) or [])
    monkeypatch.setattr(procs, "wait_gpu_memory_free",
                        lambda gpus, **kw: calls.append(("wait", tuple(gpus))) or True)
    handles = [ServerHandle(cfg=engine.base_config(name), port=port, proc=procs.Proc(
                   popen=None, argv=["vllm"], log_path="x.log", run_id="run-x", t_wall_start=0.0, t_mono_start=0.0),
                   log_path="x.log", base_url=f"http://127.0.0.1:{port}", ready_s=1.0)
               for name, port in (("DP2rand0", 8000), ("DP2rand1", 8001))]
    with pytest.raises(RuntimeError, match="SIGKILL"):
        stop_servers(handles)
    assert calls == [("stop", "run-x", False), ("stop", "run-x", False), ("sweep", "run-x"), ("wait", (0, 1))]


# ---------------------------------------------------------------- clients


def test_two_clients_concurrently_against_two_servers(tmp_path):
    env = fake_env(tmp_path)
    ports = free_ports(2)
    run_dir = tmp_path / "run"
    handles = []
    try:
        for name, port in zip(("DP2rand0", "DP2rand1"), ports):
            handles.append(start_server(engine.base_config(name), MODEL_DIR, port, str(run_dir), env, "run-dp2r",
                                        startup_timeout_s=30, vllm_bin=VLLM))
        assert sorted(p.name for p in (tmp_path / "gpustate").glob("gpu*.json")) == ["gpu0.json", "gpu1.json"]
        specs = [ClientSpec(base_url=h.base_url, tokenizer="/tok", input_len=1024, output_len=16, prefix_len=0,
                            num_prompts=12, request_rate=40.0, seed=1000 + k, result_dir=str(run_dir / f"sub-{k}"),
                            result_filename="result.json", request_id_prefix=f"run-dp2r-{k}-",
                            metadata=(("tpprof_config", h.cfg.name), ("tpprof_rate", "40.0")),
                            num_warmups=2, ready_check_timeout_s=5)
                 for k, h in enumerate(handles)]
        poll_csv = tmp_path / "gauges.csv"
        with MetricsPoller([h.base_url for h in handles], str(poll_csv), interval_s=0.05) as poller:
            outcomes = run_clients(specs, env, str(run_dir), "run-dp2r", timeout_s=60, vllm_bin=VLLM)
            time.sleep(0.2)
    finally:
        freed = stop_servers(handles, nvidia_smi=NVSMI, env=env)

    # Stopped together: the tag sweep after the first server does not SIGKILL its sibling.
    assert freed is True
    assert list((tmp_path / "gpustate").glob("gpu*.json")) == []
    assert all(gone(h.proc.popen.pid) and h.proc.popen.returncode == 0 for h in handles)

    assert [o.spec for o in outcomes] == specs
    starts = []
    for k, o in enumerate(outcomes):
        assert (o.exit_code, o.timed_out) == (0, False), open(o.log_path).read()
        assert o.result_path == str(run_dir / f"sub-{k}" / "result.json")
        assert o.log_path == str(run_dir / f"client-{k}.log") and os.path.getsize(o.log_path) > 0
        assert 0 < o.duration_s < 60
        assert o.argv == client_argv(specs[k], VLLM)
        r = results.load_serve_result(o.result_path)
        assert results.validate_serve(r, expect_in=1024, expect_out=16) == []
        assert r.metadata == {"tpprof_config": specs[k].metadata[0][1], "tpprof_rate": "40.0"}
        starts.append(o.t_mono_start)
    assert abs(starts[0] - starts[1]) < 1.0          # launched together, not one after the other
    # Both clients ran at once: each started before the other finished.
    assert outcomes[0].t_mono_start < outcomes[1].t_mono_end and outcomes[1].t_mono_start < outcomes[0].t_mono_end

    with open(poll_csv, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["t_wall", "name", "value"]
    assert poller.rows == len(rows) - 1 and poller.rows >= len(promparse.GAUGES)
    assert {r[1] for r in rows[1:]} == set(promparse.GAUGES)
    assert all(float(r[0]) > 0 and float(r[2]) >= 0 for r in rows[1:])


def test_run_clients_stops_a_client_past_its_timeout(tmp_path):
    # No server listens, so the client's 60 s ready check keeps it busy past the 1 s timeout.
    (port,) = free_ports(1)
    spec = sample_spec(base_url=f"http://127.0.0.1:{port}", num_prompts=4, result_dir=str(tmp_path / "res"),
                       metadata=())
    t0 = time.monotonic()
    (o,) = run_clients([spec], fake_env(tmp_path), str(tmp_path), "run-slow", timeout_s=1, vllm_bin=VLLM)
    assert time.monotonic() - t0 < 30
    assert (o.exit_code, o.timed_out) == (None, True)
    assert procs.sweep_tagged("run-slow") == [] and not os.path.exists(o.result_path)


def test_run_clients_rejects_colliding_outputs(tmp_path):
    spec = sample_spec(result_dir=str(tmp_path))
    with pytest.raises(ValueError, match="same"):
        run_clients([spec, spec], fake_env(tmp_path), str(tmp_path), "r", timeout_s=5, vllm_bin=VLLM)


def test_run_clients_pins_the_client_environment(tmp_path):
    dump = tmp_path / "env.json"
    fake = tmp_path / "vllm"
    fake.write_text(f"#!{sys.executable}\nimport json, os\njson.dump(dict(os.environ), open({str(dump)!r}, 'w'))\n")
    fake.chmod(0o755)
    env = fake_env(tmp_path, OPENAI_API_KEY="sk-x", SAVE_TO_PYTORCH_BENCHMARK_FORMAT="1", VLLM_USE_RUST_BENCH="1",
                   VLLM_USE_V2_MODEL_RUNNER="1")
    (o,) = run_clients([sample_spec(result_dir=str(tmp_path / "res"))], env, str(tmp_path), "run-env",
                       timeout_s=30, vllm_bin=str(fake))
    assert (o.exit_code, o.timed_out) == (0, False)
    seen = json.loads(dump.read_text())
    assert not {"OPENAI_API_KEY", "SAVE_TO_PYTORCH_BENCHMARK_FORMAT", "VLLM_USE_V2_MODEL_RUNNER"} & set(seen)
    assert seen["VLLM_USE_RUST_BENCH"] == "0" and seen["HF_HUB_OFFLINE"] == "1"
    assert seen["TPPROF_RUN_ID"] == "run-env" and seen["FAKE_TIME_SCALE"] == "0.01"


# ---------------------------------------------------------------- metrics poller


class _StubMetrics(http.server.BaseHTTPRequestHandler):
    body = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("content-length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


def test_metrics_poller_raises_on_a_missing_gauge_but_tolerates_a_dead_server(tmp_path):
    handler = type("H", (_StubMetrics,), {"body": b"vllm:num_requests_running{engine=\"0\"} 3\n"})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(RuntimeError, match="vllm:num_requests_waiting"):
            with MetricsPoller([f"http://127.0.0.1:{srv.server_address[1]}"], str(tmp_path / "a.csv"),
                               interval_s=0.05):
                time.sleep(0.3)
    finally:
        srv.shutdown()
        srv.server_close()

    (port,) = free_ports(1)
    with MetricsPoller([f"http://127.0.0.1:{port}"], str(tmp_path / "b.csv"), interval_s=0.05) as poller:
        time.sleep(0.3)
    assert poller.rows == 0 and poller.scrape_errors >= 1
    assert (tmp_path / "b.csv").read_text().splitlines() == ["t_wall,name,value"]
