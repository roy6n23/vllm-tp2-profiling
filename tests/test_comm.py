"""Communication microbenchmarks M1-M4 (Task 12): argv, NCCL log parsing, synthetic sweeps, vendored scripts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

from tests.conftest import FAKE_BIN, FIXTURES, ROOT, fake_env
from tpprof import comm_bench, nccltests, stats, vendored

TORCHRUN = str(FAKE_BIN / "torchrun")
VENDORED = ROOT / "third_party" / "vllm_benchmarks"
M3_KEYS = {"impl", "variant", "mode", "bytes", "n", "world_size", "median_us", "p25_us", "p75_us",
           "algbw_GBps", "busbw_GBps"}


def run(argv: list[str], tmp_path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=fake_env(tmp_path, **env), cwd=tmp_path, capture_output=True, text=True,
                          timeout=60)


# ---------------------------------------------------------------- conftest (ruling R14)


def test_fake_env_puts_this_interpreter_right_after_fake_bin(tmp_path):
    parts = fake_env(tmp_path)["PATH"].split(os.pathsep)
    assert parts[:2] == [str(FAKE_BIN), os.path.dirname(sys.executable)]
    assert "" not in parts


# ---------------------------------------------------------------- M3: variants and argv


def test_sizes_and_variants():
    assert comm_bench.SIZES[0] == 2 * 1024 and comm_bench.SIZES[-1] == 256 * 2**20 and len(comm_bench.SIZES) == 18
    assert comm_bench.VARIANTS == ((None, None), ("ring", "LL"), ("ring", "LL128"), ("ring", "Simple"),
                                   ("tree", "LL"), ("tree", "LL128"), ("tree", "Simple"), ("nvls", "Simple"))


def test_variant_env_uses_the_per_function_form():
    assert comm_bench.variant_env(None, None) == {}
    assert comm_bench.variant_env("ring", "LL") == {"NCCL_ALGO": "allreduce:ring", "NCCL_PROTO": "allreduce:LL"}
    assert comm_bench.variant_env("nvls", "Simple") == {"NCCL_ALGO": "allreduce:nvls",
                                                        "NCCL_PROTO": "allreduce:Simple"}
    assert comm_bench.variant_env("tree", None) == {"NCCL_ALGO": "allreduce:tree"}


def test_variant_env_diagnostic_run_adds_nccl_debug():
    env = comm_bench.variant_env("tree", "LL128", debug_dir="/r/x")
    assert env == {"NCCL_ALGO": "allreduce:tree", "NCCL_PROTO": "allreduce:LL128", "NCCL_DEBUG": "INFO",
                   "NCCL_DEBUG_SUBSYS": "INIT,ENV,TUNING", "NCCL_DEBUG_FILE": "/r/x/nccl.%h.%p.log"}


def test_torchrun_argv_exact():
    assert comm_bench.torchrun_argv("/r/m3.jsonl", "eager,graph", ("ring", "LL")) == [
        "torchrun", "--nproc-per-node", "2", "-m", "tpprof.comm_bench", "--out", "/r/m3.jsonl",
        "--mode", "eager,graph", "--iters", "50", "--warmup", "10", "--graph-ops", "20", "--variant", "ring:LL"]
    argv = comm_bench.torchrun_argv("o.jsonl", "graph", (None, None), torchrun="/x/torchrun")
    assert argv[0] == "/x/torchrun"
    assert argv[-2:] == ["--variant", "none:none"]


# ---------------------------------------------------------------- NCCL debug log parsing (D7-4)


def test_parse_tuning_lines_on_fixture():
    text = (FIXTURES / "nccl_debug_sample.log").read_text()
    assert comm_bench.parse_tuning_lines(text) == [
        {"func": "AllReduce", "bytes": 8192, "algo": "RING", "proto": "LL"},
        {"func": "AllReduce", "bytes": 268435456, "algo": "RING", "proto": "SIMPLE"},
        {"func": "AllGather", "bytes": 1026048, "algo": "RING", "proto": "SIMPLE"},
    ]
    assert comm_bench.parse_tuning_lines("") == []


def test_parse_nvls_support():
    assert comm_bench.parse_nvls_support((FIXTURES / "nccl_debug_sample.log").read_text()) is False
    assert comm_bench.parse_nvls_support(
        "h:1:2 [0] NCCL INFO NVLS multicast support is available on dev 0 (NVLS_NCHANNELS 16)\n") is True
    assert comm_bench.parse_nvls_support("h:1:2 [0] NCCL INFO Channel 00/02 : 0 1\n") is None


def test_parse_nccl_version_prefixed_and_bare():
    assert comm_bench.parse_nccl_version((FIXTURES / "nccl_debug_sample.log").read_text()) == "2.30.7"
    assert comm_bench.parse_nccl_version("NCCL version 2.29.7+cuda13.0\n") == "2.29.7"
    assert comm_bench.parse_nccl_version("vLLM is using nccl==2.30.7\n") is None


# ---------------------------------------------------------------- M3: synthetic sweep through the fake torchrun


def test_synthetic_m3_through_fake_torchrun_recovers_alpha_beta(tmp_path):
    out = tmp_path / "m3.jsonl"
    p = run(comm_bench.torchrun_argv(str(out), "eager,graph", (None, None), torchrun=TORCHRUN), tmp_path)
    assert p.returncode == 0, p.stderr
    rows = comm_bench.load_rows(str(out))
    assert len(rows) == 2 * len(comm_bench.SIZES)
    for r in rows:
        assert set(r) == M3_KEYS
        assert r["impl"] == "torch_nccl" and r["variant"] == "none:none" and r["world_size"] == 2
        assert r["n"] == 50
        assert r["p25_us"] <= r["median_us"] <= r["p75_us"]
        assert r["busbw_GBps"] == pytest.approx(r["algbw_GBps"])
        assert r["algbw_GBps"] == pytest.approx(r["bytes"] / (r["median_us"] * 1e-6) / 1e9)
    for mode in ("eager", "graph"):
        sel = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["bytes"])
        assert [r["bytes"] for r in sel] == list(comm_bench.SIZES)
        alpha, beta = stats.fit_alpha_beta([r["bytes"] for r in sel], [r["median_us"] * 1e-6 for r in sel])
        assert alpha == pytest.approx(6e-6, rel=0.05)
        assert beta == pytest.approx(2.6e11, rel=0.05)


def test_synthetic_m3_is_deterministic_and_labels_the_variant(tmp_path):
    outs = []
    for i in range(2):
        out = tmp_path / f"m3-{i}.jsonl"
        env = comm_bench.variant_env("ring", "LL")
        p = run(comm_bench.torchrun_argv(str(out), "graph", ("ring", "LL"), torchrun=TORCHRUN), tmp_path, **env)
        assert p.returncode == 0, p.stderr
        outs.append(out.read_text())
    assert outs[0] == outs[1]
    rows = comm_bench.load_rows(str(tmp_path / "m3-0.jsonl"))
    assert {r["variant"] for r in rows} == {"ring:LL"} and {r["mode"] for r in rows} == {"graph"}


def test_m3_refuses_a_variant_label_that_does_not_match_the_nccl_env(tmp_path):
    out = tmp_path / "m3.jsonl"
    p = run(comm_bench.torchrun_argv(str(out), "eager", ("ring", "LL"), torchrun=TORCHRUN), tmp_path)
    assert p.returncode == 2
    assert "NCCL_ALGO" in p.stderr and "allreduce:ring" in p.stderr
    assert not out.exists()


def test_synthetic_diagnostic_run_writes_a_parseable_nccl_debug_file(tmp_path):
    out = tmp_path / "m3.jsonl"
    env = comm_bench.variant_env("tree", "LL128", debug_dir=str(tmp_path))
    p = run(comm_bench.torchrun_argv(str(out), "eager", ("tree", "LL128"), torchrun=TORCHRUN), tmp_path, **env)
    assert p.returncode == 0, p.stderr
    logs = list(tmp_path.glob("nccl.*.log"))
    assert len(logs) == 1 and "%" not in logs[0].name
    text = logs[0].read_text()
    assert "NCCL_ALGO set by environment to allreduce:tree" in text
    tuning = comm_bench.parse_tuning_lines(text)
    assert [t["bytes"] for t in tuning] == list(comm_bench.SIZES)
    assert {(t["func"], t["algo"], t["proto"]) for t in tuning} == {("AllReduce", "TREE", "LL128")}
    assert comm_bench.parse_nccl_version(text) == "2.30.7"
    assert comm_bench.parse_nvls_support(text) in (True, False)


@pytest.mark.parametrize("content, needle", [("", "no rows"), ('{"impl": "torch_nccl"}\n', "median_us"),
                                             ("{not json\n", "line 1")])
def test_load_rows_rejects_empty_or_malformed_files(tmp_path, content, needle):
    path = tmp_path / "m3.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match=needle) as ei:
        comm_bench.load_rows(str(path))
    assert str(path) in str(ei.value)


# ---------------------------------------------------------------- M1 / M2: vendored vLLM benchmarks


def test_m1_m2_argv_exact():
    assert vendored.M2_TOKENS == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
    assert vendored.m2_argv("/r/m2.jsonl") == ["torchrun", "--nproc-per-node", "2", "-m", "tpprof.vendored", "m2",
                                               "--out", "/r/m2.jsonl"]
    assert vendored.m1_argv("/r/m1.json", torchrun="/t") == ["/t", "--nproc-per-node", "2", "-m", "tpprof.vendored",
                                                             "m1", "--out", "/r/m1.json"]


def test_vendored_script_argv_matches_am31():
    tokens = [str(t) for t in vendored.M2_TOKENS]
    assert vendored.m2_script_argv("o.jsonl") == [
        "x", "--hidden-dim", "4096", "--num-tokens", *tokens, "--dtypes", "bfloat16", "--quant-modes", "none",
        "--warmup", "5", "--trials", "50", "--output-file", "o.jsonl"]
    assert vendored.m1_script_argv("o.json") == [
        "x", "--sequence-lengths", *tokens, "--num-warmup", "5", "--num-trials", "200", "--output-json", "o.json"]


@pytest.mark.parametrize("name", ["benchmark_device_communicators.py", "benchmark_fused_collective.py"])
def test_vendored_files_keep_the_apache_header(name):
    head = (VENDORED / name).read_text().splitlines()[:4]
    assert "# SPDX-License-Identifier: Apache-2.0" in head
    assert "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project" in head


def test_source_md_names_commit_license_and_checksums():
    src = (VENDORED / "SOURCE.md").read_text()
    assert "ced6857afa0e" in src and "v0.30.0" in src and "Apache-2.0" in src
    assert "benchmarks/kernels/benchmark_fused_collective.py" in src
    assert "-import pandas as pd" in src   # the recorded diff
    sums = {name: sha for sha, name in re.findall(r"`([0-9a-f]{64})`\s+`([\w./]+)`", src)}
    for name in ("benchmark_device_communicators.py", "benchmark_fused_collective.py"):
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == sums[f"vendored/{name}"]
    # benchmark_device_communicators.py is unmodified: vendored checksum == upstream checksum
    assert sums["vendored/benchmark_device_communicators.py"] == sums["upstream/benchmark_device_communicators.py"]
    assert sums["vendored/benchmark_fused_collective.py"] != sums["upstream/benchmark_fused_collective.py"]


def test_fused_collective_has_no_pandas_and_no_markdown():
    tree = ast.parse((VENDORED / "benchmark_fused_collective.py").read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "pandas" not in imported
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "format_results_markdown" not in funcs and "save_results_to_file" in funcs


def _vendored_save_results_to_file():
    """The modified save_results_to_file from the vendored script, run without importing torch or vllm."""
    src = (VENDORED / "benchmark_fused_collective.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "save_results_to_file")
    ns = {"json": json, "argparse": argparse, "logger": None}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "vendored", "exec"), ns)
    return ns["save_results_to_file"]


def test_vendored_writer_output_parses_with_parse_m2(tmp_path):
    out = tmp_path / "m2.jsonl"
    entries = [{"num_tokens": n, "hidden_dim": 4096, "dtype": "bfloat16", "use_residual": True,
                "quant_modes": ["none"], "input_size_mb": n * 4096 * 2 / 2**20,
                "results": {"standard_allreduce__native_rms_norm": 0.02,
                            "flashinfer_trtllm_fused_allreduce_rmsnorm_oneshot": 0.008,
                            "flashinfer_mnnvl_fused_allreduce_rmsnorm_twoshot": float("inf")}}
               for n in (1, 8192)]
    save = _vendored_save_results_to_file()
    save(entries, 2, argparse.Namespace(output_file=str(out)), 1)   # rank 1 writes nothing
    assert not out.exists()
    save(entries, 2, argparse.Namespace(output_file=str(out)), 0)
    save(entries, 2, argparse.Namespace(output_file=str(out)), 0)   # a rerun overwrites, never duplicates
    assert all(json.loads(line) for line in out.read_text().splitlines())
    rows = vendored.parse_m2(str(out))
    assert len(rows) == 6
    assert rows[0] == {"op": "standard_allreduce__native_rms_norm", "num_tokens": 1, "bytes": 8192,
                       "backend": "standard", "oneshot": None, "ms": 0.02}
    assert rows[1] == {"op": "flashinfer_trtllm_fused_allreduce_rmsnorm_oneshot", "num_tokens": 1, "bytes": 8192,
                       "backend": "trtllm", "oneshot": True, "ms": 0.008}
    assert rows[2]["backend"] == "mnnvl" and rows[2]["oneshot"] is False and rows[2]["ms"] is None  # FAILED
    assert rows[5]["bytes"] == 64 * 2**20


def test_parse_m2_on_a_synthetic_file(tmp_path):
    out = tmp_path / "m2.jsonl"
    assert vendored.main(["m2", "--out", str(out), "--synthetic"]) == 0
    rows = vendored.parse_m2(str(out))
    assert sorted({r["num_tokens"] for r in rows}) == list(vendored.M2_TOKENS)
    assert {r["backend"] for r in rows} == {"standard", "trtllm", "mnnvl"}
    assert {r["oneshot"] for r in rows if r["backend"] != "standard"} == {True, False}
    for r in rows:
        assert set(r) == {"op", "num_tokens", "bytes", "backend", "oneshot", "ms"}
        assert r["bytes"] == r["num_tokens"] * 4096 * 2
        assert isinstance(r["ms"], float) and r["ms"] > 0


def test_parse_m2_names_a_missing_key(tmp_path):
    out = tmp_path / "m2.jsonl"
    out.write_text('{"op": "standard_allreduce__native_rms_norm", "num_tokens": 1, "hidden_dim": 4096}\n')
    with pytest.raises(ValueError, match="dtype"):
        vendored.parse_m2(str(out))
    out.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        vendored.parse_m2(str(out))


def test_parse_m1_on_a_synthetic_file(tmp_path):
    out = tmp_path / "m1.json"
    assert vendored.main(["m1", "--out", str(out), "--synthetic"]) == 0
    doc = json.loads(out.read_text())
    assert doc["hidden_size"] == 4096 and doc["sequence_lengths"] == list(vendored.M2_TOKENS)
    rows = vendored.parse_m1(str(out))
    impls = {r["impl"] for r in rows}
    assert {"ca_1stage", "pynccl", "flashinfer_trtllm", "flashinfer_mnnvl"} <= impls
    assert len(rows) == len(impls) * len(vendored.M2_TOKENS)
    for r in rows:
        assert set(r) == {"impl", "mode", "bytes", "mean_us"} and r["mode"] == "graph"
    one = next(r for r in rows if r["impl"] == "pynccl" and r["bytes"] == 8192)
    assert one["mean_us"] == pytest.approx(1000 * doc["results"]["1"]["timings"]["pynccl"])


def test_parse_m1_rejects_empty_results(tmp_path):
    out = tmp_path / "m1.json"
    out.write_text(json.dumps({"hidden_size": 4096, "dtype": "torch.bfloat16", "results": {}}))
    with pytest.raises(ValueError, match="no rows"):
        vendored.parse_m1(str(out))


@pytest.mark.parametrize("which, parse", [("m1", vendored.parse_m1), ("m2", vendored.parse_m2)])
def test_vendored_through_fake_torchrun(tmp_path, which, parse):
    out = tmp_path / f"{which}.out"
    argv = getattr(vendored, f"{which}_argv")(str(out), torchrun=TORCHRUN)
    p = run(argv, tmp_path)
    assert p.returncode == 0, p.stderr
    assert parse(str(out))


def test_vendored_modules_load_by_path():
    # Loading does not import torch here: the spec is resolved without executing the module.
    spec = vendored.script_spec("benchmark_fused_collective")
    assert spec.origin == str(VENDORED / "benchmark_fused_collective.py")


# ---------------------------------------------------------------- M4: nccl-tests


def _has_seq(argv, *seq):
    return any(list(argv[i:i + len(seq)]) == list(seq) for i in range(len(argv)))


def test_nccltests_run_argv_graph_and_eager():
    g = nccltests.run_argv("/w/build", "/r/g.json", graph=True)
    e = nccltests.run_argv("/w/build", "/r/e.json", graph=False)
    for argv, out in ((g, "/r/g.json"), (e, "/r/e.json")):
        assert argv[0] == "/w/build/all_reduce_perf"
        assert _has_seq(argv, "-t", "2", "-g", "1") and _has_seq(argv, "-d", "bfloat16")
        assert _has_seq(argv, "-b", "8", "-e", "256M", "-f", "2") and _has_seq(argv, "-J", out)
    assert "-G" in g and "-I" not in g
    assert _has_seq(e, "-I", "1", "-U", "1") and "-G" not in e


def test_nccltests_build_script(tmp_path):
    script = nccltests.build_script()
    assert "MPI=0" in script and "-gencode=arch=compute_90,code=sm_90" in script
    assert "ln -sf libnccl.so.2" in script and "lib/libnccl.so" in script
    assert "https://github.com/NVIDIA/nccl-tests/archive/refs/tags/v2.20.0.tar.gz" in script
    assert "git clone" not in script
    assert "cuda-cudart-dev-13-0" in script and "command -v make" in script
    assert "find_spec('nvidia.nccl')" in script
    assert "echo /opt/nccl" in nccltests.build_script("echo /opt/nccl")
    path = tmp_path / "b.sh"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_nccltests_run_env_points_the_loader_at_the_image_nccl():
    assert nccltests.run_env("/w/nccl-home") == {"LD_LIBRARY_PATH": "/w/nccl-home/lib"}


def _nccl_tests_doc(graph: int) -> dict:
    """The -J layout written by nccl-tests v2.20.0 src/util.cu (writeBenchmarkLine*, writePerIterReport)."""
    def place(t):
        return {"time": t, "alg_bw": 8192 / t / 1e3, "bus_bw": 8192 / t / 1e3, "nwrong": 0.0}
    res = {"size": 8192, "count": 4096, "type": "bfloat16", "redop": "sum", "root": "    -1",
           "out_of_place": place(7.5), "in_place": place(7.25), "actual_iterations": 200,
           "experiment_name": "AllReduce"}
    if not graph:
        res["out_of_place_per_iter"] = {"p50_us": 7.4, "p99_us": 9.0}
        res["tuning"] = {"implementation": "Coll", "algo": "RING", "proto": "LL", "#channels": 2}
    return {"version": 4, "nccl_version": 23007, "config": {"nthreads": 2, "ngpus": 1, "graph": graph},
            "results": [res]}


def test_nccltests_parse_json(tmp_path):
    path = tmp_path / "e.json"
    path.write_text(json.dumps(_nccl_tests_doc(graph=0)))
    rows = nccltests.parse_json(str(path))
    assert [r["place"] for r in rows] == ["out_of_place", "in_place"]
    assert rows[0] == {"impl": "nccl_tests", "mode": "eager", "place": "out_of_place", "bytes": 8192,
                       "time_us": 7.5, "p50_us": 7.4, "algbw_GBps": pytest.approx(8192 / 7.5 / 1e3),
                       "busbw_GBps": pytest.approx(8192 / 7.5 / 1e3), "nwrong": 0.0, "algo": "RING", "proto": "LL",
                       "nccl_version": 23007}
    path.write_text(json.dumps(_nccl_tests_doc(graph=20)))
    rows = nccltests.parse_json(str(path))
    assert {r["mode"] for r in rows} == {"graph"} and rows[0]["algo"] is None and rows[0]["p50_us"] is None


def test_nccltests_parse_json_names_missing_results(tmp_path):
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"version": 4, "nccl_version": 23007, "config": {"graph": 0}}))
    with pytest.raises(ValueError, match="results"):
        nccltests.parse_json(str(path))
