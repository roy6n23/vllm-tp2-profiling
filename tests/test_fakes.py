"""Behavior of the fake vllm / nvidia-smi / torchrun executables in tests/fake_bin (Task 8)."""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import pathlib
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest

from tests.conftest import FAKE_BIN, FIXTURES, fake_env
from tests.fakes import common

VLLM = str(FAKE_BIN / "vllm")
NVSMI = str(FAKE_BIN / "nvidia-smi")
TORCHRUN = str(FAKE_BIN / "torchrun")
SERVED = "llama-3.1-8b-instruct"

# C6 line shapes (D5-6): vLLM logger lines and uvicorn lines, both with the per-process prefix.
VLLM_LINE = re.compile(r"^\((?P<proc>\w+) pid=(?P<pid>\d+)\) (?P<lvl>DEBUG|INFO|WARNING|ERROR|CRITICAL) "
                       r"(?P<ts>\d\d-\d\d \d\d:\d\d:\d\d) \[(?P<src>[^\]]+):(?P<ln>\d+)\] (?P<msg>.*)$")
UVICORN_LINE = re.compile(r"^\((?P<proc>\w+) pid=(?P<pid>\d+)\) INFO:     (?P<msg>.*)$")

# C1: common engine flags, TP2 parallelism flags and serve-only flags, token for token.
COMMON_FLAGS = [
    "--dtype", "bfloat16", "--max-model-len", "9216", "--gpu-memory-utilization", "0.90",
    "--max-num-seqs", "1024", "--max-num-batched-tokens", "8192", "--block-size", "16",
    "--kv-cache-dtype", "auto", "--seed", "0", "--no-enable-prefix-caching", "--enable-chunked-prefill",
    "--async-scheduling", "--stream-interval", "1", "--no-enable-dbo", "--no-enable-batch-sharded-sampling",
    "--performance-mode", "balanced", "--optimization-level", "2", "--attention-backend", "FLASH_ATTN",
    "--attention-config", '{"flash_attn_version":3}', "--generation-config", "vllm", "--fail-on-environ-validation",
]
CC_TP2 = '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_allreduce_rms":true}}'
CC_TP2_NOFUSE = '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_allreduce_rms":false}}'
CC_TP1 = '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
BASE_ENV = {
    "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "VLLM_ALLREDUCE_USE_FLASHINFER": "1",
    "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "0", "VLLM_ALLREDUCE_USE_SYMM_MEM": "1",
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND": "auto", "VLLM_LOGGING_LEVEL": "INFO", "VLLM_USE_NCCL_SYMM_MEM": "0",
    "VLLM_USE_RUST_BENCH": "0", "VLLM_USE_RUST_FRONTEND": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "60",
}
AR_OFF_ENV = {"VLLM_ALLREDUCE_USE_FLASHINFER": "0", "VLLM_ALLREDUCE_USE_SYMM_MEM": "0"}
SERVE_ONLY = ["--served-model-name", SERVED, "--host", "127.0.0.1"]
SERVE_TAIL = ["--api-server-count", "1", "--disable-uvicorn-access-log", "--no-enable-log-requests"]


def engine_flags(config: str, arm: str = "base") -> list[str]:
    if config == "TP2":
        par = ["--tensor-parallel-size", "2", "--distributed-executor-backend", "mp"]
        cc = CC_TP2_NOFUSE if arm in ("AR1", "AR2", "AR3") else CC_TP2
    elif config == "DP2":
        par, cc = ["--tensor-parallel-size", "1", "--data-parallel-size", "2"], CC_TP1
    else:
        par, cc = ["--tensor-parallel-size", "1", "--distributed-executor-backend", "mp"], CC_TP1
    if arm == "AR3":
        par = par + ["--disable-custom-all-reduce"]
    if arm == "G2":
        common = [t for t in COMMON_FLAGS if t not in ("--optimization-level", "2")]
        return common + par + ["--enforce-eager"]
    return COMMON_FLAGS + par + ["--compilation-config", cc]


def arm_env(tmp_path: pathlib.Path, config: str, arm: str = "base", **extra: str) -> dict[str, str]:
    env = dict(BASE_ENV)
    if arm in ("AR2", "AR3"):
        env.update(AR_OFF_ENV)
    env["CUDA_VISIBLE_DEVICES"] = "0" if config == "TP1" else "0,1"
    env.update(extra)
    return fake_env(tmp_path, **env)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_status(port: int, method: str, path: str, body: dict | None = None) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        conn.request(method, path, body=None if body is None else json.dumps(body),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        return resp.status
    finally:
        conn.close()


def http_body(port: int, path: str) -> str:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        assert resp.status == 200
        return resp.read().decode()
    finally:
        conn.close()


def wait_healthy(port: int, proc: subprocess.Popen, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        assert proc.poll() is None, f"fake vllm serve exited with {proc.returncode}"
        with contextlib.suppress(OSError):
            if http_status(port, "GET", "/health") == 200:
                return
        time.sleep(0.02)
    raise AssertionError("fake vllm serve never became healthy")


def stop(proc: subprocess.Popen) -> int:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return proc.returncode


def low_nofile() -> None:
    """preexec_fn: the default macOS soft RLIMIT_NOFILE (256), which the fakes must raise themselves."""
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))


@contextlib.contextmanager
def fake_server(tmp_path: pathlib.Path, flags: list[str], env: dict[str, str], model: str = "/m",
                wait: bool = True, preexec_fn=None):
    port = free_port()
    log = tmp_path / f"serve-{port}.log"
    with open(log, "wb") as f:
        proc = subprocess.Popen([VLLM, "serve", model, *flags, "--port", str(port)], env=env, stdout=f,
                                stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=preexec_fn)
    try:
        if wait:
            wait_healthy(port, proc)
        yield proc, port, log
    finally:
        stop(proc)


def serve_log(tmp_path: pathlib.Path, flags: list[str], env: dict[str, str]) -> str:
    with fake_server(tmp_path, flags, env) as (proc, _port, log):
        assert stop(proc) == 0
    return log.read_text()


def messages(text: str) -> list[str]:
    """vLLM-logger and uvicorn message parts (prefix stripped), in order."""
    out = []
    for line in text.splitlines():
        m = VLLM_LINE.match(line) or UVICORN_LINE.match(line)
        if m:
            out.append(m.group("msg"))
    return out


def lines_with(text: str, needle: str) -> list[re.Match]:
    return [m for m in map(VLLM_LINE.match, text.splitlines()) if m and needle in m.group("msg")]


def sse_messages(port: int, body: dict, headers: dict[str, str] | None = None) -> list:
    """POST /v1/completions and split the stream like the real client: on blank lines, 'data: ' stripped."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", "/v1/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type", "").startswith("text/event-stream")
        raw = resp.read().decode()
    finally:
        conn.close()
    assert raw.endswith("\n\n")
    out = []
    for msg in raw.split("\n\n")[:-1]:
        assert msg.startswith("data: ")
        payload = msg[len("data: "):]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def completion_body(max_tokens: int, model: str = "/m") -> dict:
    return {"model": model, "prompt": "a " * 1023, "repetition_penalty": 1.0, "max_tokens": max_tokens,
            "logprobs": None, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}


def metric_value(text: str, name: str, **labels: str) -> float:
    total, found = 0.0, False
    for line in text.splitlines():
        m = re.match(r"^(?P<n>[^{\s]+)\{(?P<l>[^}]*)\} (?P<v>\S+)$", line)
        if not m or m.group("n") != name:
            continue
        got = dict(re.findall(r'(\w+)="([^"]*)"', m.group("l")))
        if all(got.get(k) == v for k, v in labels.items()):
            total, found = total + float(m.group("v")), True
    assert found, f"{name} {labels} not in /metrics"
    return total


def state_files(tmp_path: pathlib.Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "gpustate").glob("gpu*.json"))


# ---------------------------------------------------------------- (a) version and help


def test_version_prints_pinned_version(tmp_path):
    out = subprocess.run([VLLM, "--version"], env=fake_env(tmp_path), capture_output=True, text=True, check=True)
    assert out.stdout == "0.30.0\n"


@pytest.mark.parametrize("argv,fixture", [
    (["serve"], "serve_help_all.txt"),
    (["bench", "serve"], "bench_serve_help.txt"),
    (["bench", "latency"], "bench_latency_help.txt"),
])
def test_help_all_is_the_real_fixture_and_plain_help_is_its_head(tmp_path, argv, fixture):
    text = (FIXTURES / fixture).read_text()
    full = subprocess.run([VLLM, *argv, "--help=all"], env=fake_env(tmp_path), capture_output=True, text=True)
    assert full.returncode == 0
    assert full.stdout == text
    short = subprocess.run([VLLM, *argv, "--help"], env=fake_env(tmp_path), capture_output=True, text=True)
    assert short.returncode == 0
    assert short.stdout.splitlines() == text.splitlines()[:60]


# ---------------------------------------------------------------- (b) serve lifecycle


def test_serve_health_stream_metrics_and_clean_sigint(tmp_path):
    env = fake_env(tmp_path, CUDA_VISIBLE_DEVICES="0,1")
    with fake_server(tmp_path, ["--tensor-parallel-size", "2"], env) as (proc, port, log):
        for i in (0, 1):
            state = json.loads((tmp_path / "gpustate" / f"gpu{i}.json").read_text())
            assert state["pid"] == proc.pid and state["used_mib"] > 1024

        models = json.loads(http_body(port, "/v1/models"))
        assert models["object"] == "list"
        card = models["data"][0]
        assert (card["id"], card["root"], card["object"], card["owned_by"]) == ("/m", "/m", "model", "vllm")
        assert card["max_model_len"] > 0

        assert http_status(port, "POST", "/tokenize", {"model": "/m", "prompt": "a", "add_special_tokens": False}) == 404

        msgs = sse_messages(port, completion_body(5), headers={"x-request-id": "tpprof-t-0"})
        tokens = msgs[:-2]
        assert len(tokens) == 5
        assert all(m["object"] == "text_completion" and m["model"] == "/m" for m in tokens)
        assert [m["choices"][0]["finish_reason"] for m in tokens] == [None] * 4 + ["length"]
        assert all(set(m["choices"][0]) == {"index", "text", "logprobs", "finish_reason", "stop_reason",
                                            "prompt_token_ids", "token_ids"} for m in tokens)
        usage = msgs[-2]
        assert usage["choices"] == []
        assert usage["usage"] == {"prompt_tokens": 1024, "total_tokens": 1029, "completion_tokens": 5}
        assert msgs[-1] == "[DONE]"

        metrics = http_body(port, "/metrics")
        assert metric_value(metrics, "vllm:request_success_total", engine="0", finished_reason="length",
                            model_name="/m") >= 1
        assert metric_value(metrics, "vllm:generation_tokens_total", engine="0") >= 5
        assert metric_value(metrics, "vllm:prompt_tokens_total", engine="0") >= 1024
        for gauge in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"):
            assert metric_value(metrics, gauge, engine="0", model_name="/m") == 0
        for counter in ("vllm:num_preemptions_total", "vllm:prefix_cache_queries_total",
                        "vllm:prefix_cache_hits_total"):
            metric_value(metrics, counter, engine="0", model_name="/m")
        assert "# TYPE vllm:request_success_total counter" in metrics

        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=2) == 0
    assert state_files(tmp_path) == []
    assert "Application startup complete." in messages(log.read_text())


@pytest.mark.parametrize("pc_flag,queries,hits", [
    # D5: no lookup runs with --no-enable-prefix-caching, so both counters stay 0 (hit rate 0/0 on the box).
    ("--no-enable-prefix-caching", 0, 0),
    # Lookup runs: every prompt token is queried; the repeated prompt hits its full blocks.
    ("--enable-prefix-caching", 2 * 1024, 1024),
])
def test_serve_prefix_cache_counters_follow_the_flag(tmp_path, pc_flag, queries, hits):
    with fake_server(tmp_path, [pc_flag], fake_env(tmp_path)) as (_proc, port, _log):
        for _ in range(2):
            sse_messages(port, completion_body(2))
        metrics = http_body(port, "/metrics")
    labels = {"engine": "0", "model_name": "/m"}
    assert metric_value(metrics, "vllm:prefix_cache_queries_total", **labels) == queries
    assert metric_value(metrics, "vllm:prefix_cache_hits_total", **labels) == hits
    assert metric_value(metrics, "vllm:prompt_tokens_total", **labels) == 2 * 1024


def test_serve_rejects_unknown_model_like_the_real_server(tmp_path):
    with fake_server(tmp_path, ["--served-model-name", SERVED], fake_env(tmp_path)) as (_proc, port, _log):
        assert http_status(port, "POST", "/v1/completions", completion_body(2, model="other")) == 404
        assert sse_messages(port, completion_body(2, model=SERVED))[-1] == "[DONE]"


def test_serve_refuses_connections_until_ready_when_hanging(tmp_path):
    env = fake_env(tmp_path, FAKE_VLLM_HANG_START="1", CUDA_VISIBLE_DEVICES="0")
    with fake_server(tmp_path, [], env, wait=False) as (proc, port, _log):
        deadline = time.monotonic() + 5
        while not state_files(tmp_path) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert state_files(tmp_path) == ["gpu0.json"]
        time.sleep(0.3)
        assert proc.poll() is None
        with pytest.raises(OSError):
            http_status(port, "GET", "/health")
        assert stop(proc) == 0
    assert state_files(tmp_path) == []


def test_serve_fail_start_exits_1_with_error_line(tmp_path):
    env = fake_env(tmp_path, FAKE_VLLM_FAIL_START="1")
    out = subprocess.run([VLLM, "serve", "/m", "--port", str(free_port())], env=env, capture_output=True,
                         text=True, timeout=10)
    assert out.returncode == 1
    assert "EngineCore failed to start." in messages(out.stdout + out.stderr)
    assert state_files(tmp_path) == []


def test_serve_fails_fast_when_port_is_taken(tmp_path):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        out = subprocess.run([VLLM, "serve", "/m", "--port", str(port)], env=fake_env(tmp_path),
                             capture_output=True, text=True, timeout=10)
    assert out.returncode == 1
    assert state_files(tmp_path) == []


def test_serve_ignore_sigint_needs_sigterm(tmp_path):
    env = fake_env(tmp_path, FAKE_VLLM_IGNORE_SIGINT="1", CUDA_VISIBLE_DEVICES="0")
    with fake_server(tmp_path, [], env) as (proc, _port, _log):
        proc.send_signal(signal.SIGINT)
        time.sleep(0.5)
        assert proc.poll() is None
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=2) == 0
    assert state_files(tmp_path) == []


def test_serve_leak_mem_keeps_state_files(tmp_path):
    env = fake_env(tmp_path, FAKE_VLLM_LEAK_MEM="1", CUDA_VISIBLE_DEVICES="1")
    with fake_server(tmp_path, [], env) as (proc, _port, _log):
        assert stop(proc) == 0
    assert state_files(tmp_path) == ["gpu1.json"]


# ---------------------------------------------------------------- C6 startup lines


@pytest.mark.parametrize("arm,ar_list", [
    ("base", "['FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL']"),
    ("AR2", "['CUSTOM', 'PYNCCL']"),
    ("AR3", "['PYNCCL']"),
])
def test_tp2_startup_lines_follow_env_gates_and_flags(tmp_path, arm, ar_list):
    text = serve_log(tmp_path, engine_flags("TP2", arm) + SERVE_ONLY + SERVE_TAIL, arm_env(tmp_path, "TP2", arm))
    msgs = messages(text)
    potential = ("['FLASHINFER_PCIE_IPC', 'FLASHINFER', 'NCCL_SYMM_MEM', 'QUICK_REDUCE', 'AITER_CUSTOM', "
                 "'CUSTOM', 'SYMM_MEM', 'PYNCCL']")
    assert (f"Using {ar_list} all-reduce backends (in dispatch order) for group 'tp:0' out of potential "
            f"backends: {potential}.") in msgs
    assert "vLLM is using nccl==2.30.7" in msgs
    assert ("Enabled custom fusions: allreduce_rms" in msgs) == (arm == "base")
    workspace = "Initialized FlashInfer Allreduce norm fusion workspace with backend=mnnvl"
    assert (workspace in msgs) == (arm == "base")
    kv = lines_with(text, "KV cache size:")
    assert [m.group("msg") for m in kv] == [
        "GPU KV cache size: 955,000 tokens, Maximum concurrency for 9,216 tokens per request: 103.62x"]
    assert kv[0].group("proc") == "EngineCore"
    assert {m.group("proc") for m in lines_with(text, "Model loading took")} == {"Worker_TP0", "Worker_TP1"}
    assert lines_with(text, "all-reduce backends")[0].group("proc") == "Worker"
    banner = lines_with(text, "Initializing a V1 LLM engine (v0.30.0) with config: ")
    assert len(banner) == 1
    assert "tensor_parallel_size=2, " in banner[0].group("msg")
    assert f"disable_custom_all_reduce={arm == 'AR3'}, " in banner[0].group("msg")
    for needed in ("Chunked prefill is enabled with max_num_batched_tokens=8192.", "Using V2 Model Runner",
                   "Using AttentionBackendEnum.FLASH_ATTN backend.", "Using FlashAttention version 3"):
        assert needed in msgs
    assert any(m.startswith("Graph capturing finished in ") for m in msgs)
    ready = msgs.index("Application startup complete.")
    assert ready > max(i for i, m in enumerate(msgs) if "KV cache size" in m or m.startswith("Graph capturing"))
    for line in text.splitlines():
        assert VLLM_LINE.match(line) or UVICORN_LINE.match(line), line


def test_tp2_without_multicast_logs_the_fallback_and_trtllm(tmp_path):
    env = arm_env(tmp_path, "TP2", FAKE_VLLM_NO_MULTICAST="1")
    msgs = messages(serve_log(tmp_path, engine_flags("TP2") + SERVE_ONLY + SERVE_TAIL, env))
    assert "FlashInfer MNNVL multicast is unavailable on the current topology." in msgs
    assert any(m.startswith("Failed to initialize FlashInfer All Reduce workspace: ")
               and "This is expected on GPUs without NVSwitch" in m for m in msgs)
    assert ("FlashInfer mnnvl allreduce workspace unavailable (likely no NVSwitch multicast support); "
            "falling back to trtllm backend for single node.") in msgs
    assert "Initialized FlashInfer Allreduce norm fusion workspace with backend=trtllm" in msgs
    assert not any("backend=mnnvl" in m for m in msgs)


def test_tp2_fib_trtllm_env_selects_trtllm_directly(tmp_path):
    env = arm_env(tmp_path, "TP2", VLLM_FLASHINFER_ALLREDUCE_BACKEND="trtllm")
    msgs = messages(serve_log(tmp_path, engine_flags("TP2") + SERVE_ONLY + SERVE_TAIL, env))
    assert "Initialized FlashInfer Allreduce norm fusion workspace with backend=trtllm" in msgs
    assert not any("falling back" in m for m in msgs)


def test_tp1_lines_come_from_enginecore_and_have_no_allreduce(tmp_path):
    text = serve_log(tmp_path, engine_flags("TP1") + SERVE_ONLY + SERVE_TAIL, arm_env(tmp_path, "TP1"))
    kv = lines_with(text, "KV cache size:")
    assert [(m.group("proc"), m.group("msg")) for m in kv] == [
        ("EngineCore", "GPU KV cache size: 420,959 tokens, Maximum concurrency for 9,216 tokens per request: 45.68x")]
    assert {m.group("proc") for m in lines_with(text, "Model loading took")} == {"EngineCore"}
    msgs = messages(text)
    assert not any("all-reduce backends" in m or "nccl==" in m or "custom fusions" in m for m in msgs)
    assert "Worker" not in text


def test_g2_enforce_eager_logs_eager_and_no_graphs_or_fusion(tmp_path):
    msgs = messages(serve_log(tmp_path, engine_flags("TP2", "G2") + SERVE_ONLY + SERVE_TAIL,
                              arm_env(tmp_path, "TP2", "G2")))
    assert ("Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting "
            "-cc.mode=none -cc.cudagraph_mode=none") in msgs
    assert "Enabled custom fusions: allreduce_rms" not in msgs
    assert not any(m.startswith("Graph capturing finished") for m in msgs)


def test_dp2_two_engines_two_kv_lines_and_periodic_stats(tmp_path):
    env = arm_env(tmp_path, "DP2", VLLM_LOG_STATS_INTERVAL="0.2")
    with fake_server(tmp_path, engine_flags("DP2") + SERVE_ONLY + SERVE_TAIL, env) as (proc, _port, log):
        assert state_files(tmp_path) == ["gpu0.json", "gpu1.json"]
        deadline = time.monotonic() + 5
        while "Engine 001: Avg prompt throughput" not in log.read_text() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stop(proc) == 0
    text = log.read_text()
    kv = lines_with(text, "KV cache size:")
    assert [m.group("proc") for m in kv] == ["EngineCore_DP0", "EngineCore_DP1"]
    assert all("420,959 tokens" in m.group("msg") for m in kv)
    stats = lines_with(text, "Engine 000: Avg prompt throughput: ")
    assert stats and stats[0].group("proc") == "APIServer"
    assert re.fullmatch(r"Engine 000: Avg prompt throughput: \d+\.\d tokens/s, Avg generation throughput: "
                        r"\d+\.\d tokens/s, Running: \d+ reqs, Waiting: \d+ reqs, GPU KV cache usage: "
                        r"\d+\.\d%, Prefix cache hit rate: \d+\.\d%", stats[0].group("msg"))


@pytest.mark.parametrize("value,expected", [
    (None, 10.0), ("0.2", 0.2), ("30", 30.0), ("0", 10.0), ("-1", 10.0),
])
def test_stats_interval_is_vllm_log_stats_interval(monkeypatch, value, expected):
    """C6 / D5-20: every VLLM_LOG_STATS_INTERVAL s (default 10.0; <= 0 falls back to 10), not scaled."""
    monkeypatch.setenv("FAKE_TIME_SCALE", "0.01")
    if value is None:
        monkeypatch.delenv("VLLM_LOG_STATS_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("VLLM_LOG_STATS_INTERVAL", value)
    assert common.log_stats_interval() == expected


def test_serve_default_stats_cadence_is_not_one_second(tmp_path):
    with fake_server(tmp_path, [], fake_env(tmp_path)) as (proc, _port, log):
        time.sleep(1.5)
        assert stop(proc) == 0
    assert not lines_with(log.read_text(), "Avg prompt throughput")


# ---------------------------------------------------------------- (d) logparse on fake TP2 logs


@pytest.mark.parametrize("arm", ["base", "AR2", "AR3"])
def test_fake_tp2_logs_pass_logparse_checks(tmp_path, arm):
    logparse = pytest.importorskip("tpprof.logparse")
    text = serve_log(tmp_path, engine_flags("TP2", arm) + SERVE_ONLY + SERVE_TAIL, arm_env(tmp_path, "TP2", arm))
    eff = logparse.parse_engine_log(text)
    assert eff.kv_cache_tokens == [955000]
    assert logparse.check(eff, logparse.expectation_for("TP2", arm, True)) == []


# ---------------------------------------------------------------- (c), (f), (g) bench serve client


def bench_argv(port: int, out_dir: pathlib.Path, filename: str, *, num_prompts: int, rate: str,
               output_len: int = 16, warmups: int = 0, ready_s: int = 0, percentiles: str = "25,50,75,99.9",
               metadata: tuple[str, ...] = ("tp=2", "config=TP2")) -> list[str]:
    """The Task 10 client_argv shape (spec 4.4), with test-sized values."""
    return [VLLM, "bench", "serve", "--backend", "vllm", "--base-url", f"http://127.0.0.1:{port}",
            "--endpoint", "/v1/completions", "--model", SERVED, "--tokenizer", "/tok", "--dataset-name", "random",
            "--random-input-len", "1024", "--random-output-len", str(output_len), "--random-range-ratio", "0.0",
            "--random-prefix-len", "0", "--num-prompts", str(num_prompts), "--request-rate", rate,
            "--burstiness", "1.0", "--seed", "1", "--ignore-eos", "--temperature", "1.0", "--top-p", "1.0",
            "--num-warmups", str(warmups), "--ready-check-timeout-sec", str(ready_s), "--save-result",
            "--save-detailed", "--result-dir", str(out_dir), "--result-filename", filename,
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", percentiles, "--disable-tqdm",
            "--request-id-prefix", "tpprof-t-", "--metadata", *metadata]


def run_bench(argv: list[str], env: dict[str, str], timeout: float = 60, preexec_fn=None
              ) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout, preexec_fn=preexec_fn)


def test_bench_serve_writes_the_real_detailed_schema(tmp_path):
    env = fake_env(tmp_path)
    with fake_server(tmp_path, ["--served-model-name", SERVED], env) as (_proc, port, _log):
        out = run_bench(bench_argv(port, tmp_path / "res", "r.json", num_prompts=20, rate="50", warmups=2,
                                   ready_s=5), env)
        assert out.returncode == 0, out.stderr
        metrics = http_body(port, "/metrics")
    raw = (tmp_path / "res" / "r.json").read_text()
    assert "\n" not in raw.strip()
    res = json.loads(raw)
    fixture = json.loads((FIXTURES / "bench_serve_result_detailed.json").read_text())
    assert list(res) == list(fixture)
    assert (res["tp"], res["config"]) == ("2", "TP2")
    assert (res["endpoint_type"], res["backend"], res["label"], res["model_id"], res["tokenizer_id"]) == (
        "vllm", "vllm", None, SERVED, "/tok")
    assert (res["num_prompts"], res["request_rate"], res["burstiness"], res["max_concurrency"]) == (20, 50.0, 1.0, None)
    assert (res["completed"], res["failed"]) == (20, 0)
    assert res["input_lens"] == [1024] * 20 and res["output_lens"] == [16] * 20
    assert res["total_input_tokens"] == 20 * 1024 and res["total_output_tokens"] == 20 * 16
    assert res["errors"] == [""] * 20
    assert all(len(itl) == 15 for itl in res["itls"])
    assert all(t > 0 for t in res["ttfts"]) and all(lat >= t for lat, t in zip(res["latencies"], res["ttfts"]))
    assert res["start_times"] == sorted(res["start_times"])
    assert res["request_goodput"] is None
    assert res["request_throughput"] == pytest.approx(20 / res["duration"])
    assert re.fullmatch(r"\d{8}-\d{6}", res["date"])
    tpots = sorted((lat - t) / 15 for lat, t in zip(res["latencies"], res["ttfts"]))
    assert res["median_tpot_ms"] == pytest.approx(1000 * (tpots[9] + tpots[10]) / 2)
    # ready check (1) + warmups (2) + main run (20) all reached the server
    assert metric_value(metrics, "vllm:request_success_total", finished_reason="length") == 23


def test_bench_serve_counts_failures_when_server_crashes(tmp_path):
    env = fake_env(tmp_path, FAKE_VLLM_CRASH_AFTER="3", FAKE_VLLM_ITL_S="0.5", CUDA_VISIBLE_DEVICES="0")
    with fake_server(tmp_path, ["--served-model-name", SERVED], env) as (proc, port, log):
        out = run_bench(bench_argv(port, tmp_path, "crash.json", num_prompts=20, rate="50"), env)
        assert out.returncode == 0, out.stderr
        assert proc.wait(timeout=5) == 1
    res = json.loads((tmp_path / "crash.json").read_text())
    assert (res["completed"], res["failed"]) == (3, 17)
    for err, olen, lat in zip(res["errors"], res["output_lens"], res["latencies"]):
        assert (err == "") == (olen == 16)
        if err:
            assert olen == 0 and lat == 0.0
    assert state_files(tmp_path) == []
    assert any("died unexpectedly" in m for m in messages(log.read_text()))


def test_bench_serve_300_concurrent_at_inf_rate(tmp_path):
    # Both fakes start with a 256 soft fd limit and must raise it (AM12) to hold 300 open streams.
    env = fake_env(tmp_path)
    with fake_server(tmp_path, ["--served-model-name", SERVED], env, preexec_fn=low_nofile) as (_proc, port, _log):
        out = run_bench(bench_argv(port, tmp_path, "inf.json", num_prompts=300, rate="inf", output_len=8,
                                   percentiles="25,50,75,90,99", metadata=("tpprof_config=TP1",)), env, timeout=120,
                        preexec_fn=low_nofile)
        assert out.returncode == 0, out.stderr
    res = json.loads((tmp_path / "inf.json").read_text())
    assert (res["completed"], res["failed"], res["request_rate"]) == (300, 0, "inf")
    assert res["tpprof_config"] == "TP1"
    assert list(res)[7] == "tpprof_config" and list(res)[8] == "request_rate"
    assert [k for k in res if k.startswith("p") and k.endswith("_ttft_ms")] == [
        "p25_ttft_ms", "p50_ttft_ms", "p75_ttft_ms", "p90_ttft_ms", "p99_ttft_ms"]


def test_bench_serve_without_server_writes_json_with_all_failed(tmp_path):
    env = fake_env(tmp_path)
    out = run_bench(bench_argv(free_port(), tmp_path, "down.json", num_prompts=3, rate="inf"), env)
    assert out.returncode == 0, out.stderr
    res = json.loads((tmp_path / "down.json").read_text())
    assert (res["completed"], res["failed"]) == (0, 3)
    assert all(res["errors"]) and res["output_lens"] == [0, 0, 0]
    assert res["input_lens"] == [1023] * 3


# ---------------------------------------------------------------- bench latency


def test_bench_latency_writes_c3_schema(tmp_path):
    out_json = tmp_path / "lat.json"
    out = subprocess.run([VLLM, "bench", "latency", "--model", "/m", "--tensor-parallel-size", "2",
                          "--dtype", "bfloat16", "--batch-size", "8", "--input-len", "1024", "--output-len", "64",
                          "--num-iters-warmup", "2", "--num-iters", "5", "--output-json", str(out_json)],
                         env=fake_env(tmp_path), capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    res = json.loads(out_json.read_text())
    assert list(res) == ["avg_latency", "latencies", "percentiles"]
    assert len(res["latencies"]) == 5
    assert list(res["percentiles"]) == ["10", "25", "50", "75", "90", "99"]
    expected = (0.005 + 1e-5 * 8) * 64 / 2 * 0.01
    assert min(res["latencies"]) >= expected * 0.9
    assert res["avg_latency"] == pytest.approx(sum(res["latencies"]) / 5)
    assert "Avg latency: " in out.stdout


# ---------------------------------------------------------------- (e) nvidia-smi


FULL_QUERY = ["--query-gpu=index,name,driver_version,memory.total,memory.used,power.limit,pci.bus_id",
              "--format=csv,noheader,nounits"]


def nvsmi(tmp_path: pathlib.Path, *args: str, scenario: str = "ok") -> str:
    out = subprocess.run([NVSMI, *args], env=fake_env(tmp_path, FAKE_NVSMI_SCENARIO=scenario),
                         capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_nvsmi_ok_full_query(tmp_path):
    assert nvsmi(tmp_path, *FULL_QUERY).splitlines() == [
        "0, NVIDIA H100 80GB HBM3, 580.95.05, 81559, 0, 700.00, 00000000:18:00.0",
        "1, NVIDIA H100 80GB HBM3, 580.95.05, 81559, 0, 700.00, 00000000:19:00.0",
    ]


@pytest.mark.parametrize("scenario,expected", [
    ("one_gpu", ["0, NVIDIA H100 80GB HBM3, 580.95.05, 81559, 0, 700.00, 00000000:18:00.0"]),
    ("pcie", ["0, NVIDIA H100 PCIe, 580.95.05, 81559, 0, 350.00, 00000000:18:00.0",
              "1, NVIDIA H100 PCIe, 580.95.05, 81559, 0, 350.00, 00000000:19:00.0"]),
    ("old_driver", ["0, NVIDIA H100 80GB HBM3, 575.57.08, 81559, 0, 700.00, 00000000:18:00.0",
                    "1, NVIDIA H100 80GB HBM3, 575.57.08, 81559, 0, 700.00, 00000000:19:00.0"]),
    ("busy", ["0, NVIDIA H100 80GB HBM3, 580.95.05, 81559, 30000, 700.00, 00000000:18:00.0",
              "1, NVIDIA H100 80GB HBM3, 580.95.05, 81559, 30000, 700.00, 00000000:19:00.0"]),
    ("power_capped", ["0, NVIDIA H100 80GB HBM3, 580.95.05, 81559, 0, 500.00, 00000000:18:00.0",
                      "1, NVIDIA H100 80GB HBM3, 580.95.05, 81559, 0, 500.00, 00000000:19:00.0"]),
])
def test_nvsmi_scenarios_change_one_aspect(tmp_path, scenario, expected):
    assert nvsmi(tmp_path, *FULL_QUERY, scenario=scenario).splitlines() == expected


def topo_cell(text: str, row: str, col: str) -> str:
    lines = text.splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        cells = line.split("\t")
        if cells[0] == row:
            return cells[header.index(col)].strip()
    raise AssertionError(f"no row {row}")


@pytest.mark.parametrize("scenario,link,nlinks", [("ok", "NV18", 18), ("nv12", "NV12", 12), ("pcie", "PIX", 0)])
def test_nvsmi_topology_and_nvlink(tmp_path, scenario, link, nlinks):
    topo = nvsmi(tmp_path, "topo", "-m", scenario=scenario)
    assert topo.splitlines()[0] == "\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID"
    assert topo_cell(topo, "GPU0", "GPU0") == "X"
    assert topo_cell(topo, "GPU0", "GPU1") == link
    assert topo_cell(topo, "GPU1", "GPU0") == link
    links = nvsmi(tmp_path, "nvlink", "-s", scenario=scenario)
    assert len(re.findall(r"^\s+Link \d+: [\d.]+ GB/s$", links, re.M)) == 2 * nlinks


def test_nvsmi_one_gpu_topology(tmp_path):
    topo = nvsmi(tmp_path, "topo", "-m", scenario="one_gpu")
    assert topo.splitlines()[0] == "\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID"


@pytest.mark.parametrize("scenario,state", [("ok", "Completed"), ("no_fabric", "In Progress")])
def test_nvsmi_q_fabric_block(tmp_path, scenario, state):
    text = nvsmi(tmp_path, "-q", scenario=scenario)
    blocks = re.findall(r"^\s+Fabric\n\s+State\s+: (.+)\n\s+Status\s+: (.+)$", text, re.M)
    assert len(blocks) == 2
    assert all(b[0] == state for b in blocks)
    assert "Driver Version" in text and "NVIDIA H100 80GB HBM3" in text


def test_nvsmi_memory_and_apps_follow_state_files(tmp_path):
    (tmp_path / "gpustate").mkdir(exist_ok=True)
    (tmp_path / "gpustate" / "gpu1.json").write_text(json.dumps({"used_mib": 73403, "pid": 4321}))
    assert nvsmi(tmp_path, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits").splitlines() == [
        "0, 0", "1, 73403"]
    assert nvsmi(tmp_path, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
                 ).splitlines() == ["4321, 73403"]
    busy = nvsmi(tmp_path, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
                 scenario="busy").splitlines()
    assert len(busy) == 2 and "4321, 73403" in busy


def test_nvsmi_idle_has_no_compute_apps(tmp_path):
    assert nvsmi(tmp_path, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits") == ""


def test_nvsmi_monitor_loop_repeats_until_killed(tmp_path):
    query = ("--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,"
             "memory.used,clocks_event_reasons.active")
    proc = subprocess.Popen([NVSMI, query, "--format=csv,noheader,nounits", "-lms", "50"], env=fake_env(tmp_path),
                            stdout=subprocess.PIPE, text=True, start_new_session=True)
    rows = []
    try:
        deadline = time.monotonic() + 5
        while len(rows) < 6 and time.monotonic() < deadline:
            rows.append(proc.stdout.readline().rstrip("\n"))
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        proc.stdout.close()
    assert len(rows) == 6
    for row in rows:
        cells = row.split(", ")
        assert len(cells) == 9
        assert re.fullmatch(r"\d{4}/\d\d/\d\d \d\d:\d\d:\d\d\.\d{3}", cells[0])
        assert cells[1] in ("0", "1")
        assert re.fullmatch(r"0x[0-9A-F]{16}", cells[8])


def test_nvsmi_unknown_field_fails(tmp_path):
    out = subprocess.run([NVSMI, "--query-gpu=bogus.field", "--format=csv"], env=fake_env(tmp_path),
                         capture_output=True, text=True)
    assert out.returncode != 0


# ---------------------------------------------------------------- torchrun


def stub_tpprof(tmp_path: pathlib.Path) -> pathlib.Path:
    """A stand-in tpprof package whose modules record their argv and rank env, so the test sees what ran."""
    root = tmp_path / "stub"
    pkg = root / "tpprof"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    rec = ("import json, os, sys\n"
           "import importlib.util\n"
           "with open(os.environ['STUB_EXE'], 'a') as f:\n"
           "    f.write(json.dumps({'exe': sys.executable,\n"
           "                        'numpy': importlib.util.find_spec('numpy') is not None}) + '\\n')\n"
           "with open(os.environ['STUB_OUT'], 'a') as f:\n"
           "    f.write(json.dumps({'mod': __name__, 'argv': sys.argv[1:], 'rank': os.environ.get('RANK'),\n"
           "                        'world': os.environ.get('WORLD_SIZE')}) + '\\n')\n")
    (pkg / "comm_bench.py").write_text(rec)
    (pkg / "vendored.py").write_text(rec)
    return root


def run_torchrun(tmp_path: pathlib.Path, *args: str, **extra_env: str) -> tuple[subprocess.CompletedProcess,
                                                                              list[dict]]:
    stub = stub_tpprof(tmp_path)
    out_file = tmp_path / "calls.jsonl"
    env = fake_env(tmp_path, STUB_OUT=str(out_file), STUB_EXE=str(tmp_path / "exe.jsonl"), **extra_env)
    env["PYTHONPATH"] = os.pathsep.join([str(stub), env["PYTHONPATH"]])
    out = subprocess.run([TORCHRUN, *args], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    calls = [json.loads(line) for line in out_file.read_text().splitlines()] if out_file.exists() else []
    return out, calls


def test_torchrun_comm_bench_runs_synthetic_once(tmp_path):
    out, calls = run_torchrun(tmp_path, "--nproc-per-node", "2", "--master-port", "29511", "-m",
                              "tpprof.comm_bench", "--out", "m3.jsonl", "--mode", "eager,graph")
    assert out.returncode == 0, out.stderr
    assert calls == [{"mod": "__main__", "argv": ["--synthetic", "--out", "m3.jsonl", "--mode", "eager,graph"],
                      "rank": "0", "world": "2"}]


@pytest.mark.parametrize("which", ["m1", "m2"])
def test_torchrun_vendored_runs_synthetic_once(tmp_path, which):
    out, calls = run_torchrun(tmp_path, "--nproc-per-node", "2", "-m", "tpprof.vendored", which, "--out", "x")
    assert out.returncode == 0, out.stderr
    assert [c["argv"] for c in calls] == [[which, "--out", "x", "--synthetic"]]


def test_torchrun_child_interpreter_defaults_to_its_own_and_honors_override(tmp_path):
    args = ("--nproc-per-node", "2", "-m", "tpprof.comm_bench", "--out", "m3.jsonl")
    own = shutil.which("python3", path=fake_env(tmp_path)["PATH"])
    out, _ = run_torchrun(tmp_path, *args)
    assert out.returncode == 0, out.stderr
    default = json.loads((tmp_path / "exe.jsonl").read_text())
    assert os.path.realpath(default["exe"]) == os.path.realpath(own)

    # tpprof's own interpreter (the venv, with numpy) even when PATH's python3 lacks numpy (Homebrew on the Mac).
    (tmp_path / "exe.jsonl").unlink()
    shutil.rmtree(tmp_path / "stub")
    out, _ = run_torchrun(tmp_path, *args, FAKE_TORCHRUN_PYTHON=sys.executable)
    assert out.returncode == 0, out.stderr
    assert json.loads((tmp_path / "exe.jsonl").read_text()) == {"exe": sys.executable, "numpy": True}


@pytest.mark.parametrize("args", [
    ["--nproc-per-node", "2", "-m", "other.module"],
    ["--nproc-per-node", "2", "script.py", "--x"],
    ["--nproc-per-node", "2", "-m", "tpprof.vendored", "m3"],
])
def test_torchrun_rejects_anything_else(tmp_path, args):
    out, calls = run_torchrun(tmp_path, *args)
    assert out.returncode == 2
    assert out.stderr.strip()
    assert calls == []
