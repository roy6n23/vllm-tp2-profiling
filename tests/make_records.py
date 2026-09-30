"""Synthetic results directories in the C2 run-record layout (ruling R4 for serve sessions).

Analysis tests use this instead of the runner. Every run directory is named by a real
`matrix.RunSpec.run_id` and holds `spec.json` (= `RunSpec.to_dict()`), `cmd.json`, and
`done.json` or `failed.json`, plus the kind's outputs:

- offline: C3 point files under `<run_dir>/points/`
- serve_session: `sub-<k>/` with `result.json` (C7), `client.log`, `metrics_before.prom`,
  `metrics_after.prom`, `validation.json`, `meta.json` {phase, rate, seed, config, arm, round, k}
- trace: `trace_summary.json` shaped like `traces.summarize_trace`
- comm_*: `comm_rows.jsonl` (M3 / M2 / M1 / M4 row shapes)
- engine runs: `effective_config.json` = `dataclasses.asdict(logparse.EffectiveConfig)` + "violations"
- `raw/rate_grid.json` = {mu_rps, grid, sources}; `raw/_last_run.json` lists done/failed/skipped

The numbers come from the central prediction model, so a complete synthetic set is expected to
hit H1, H2, H3, H5, H6, H7 and H8.
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
from collections.abc import Callable, Iterable

import numpy as np

from tpprof import logparse, matrix, model, results
from tpprof.constants import (
    METRIC_PERCENTILES,
    ONLINE_INPUT_LEN,
    ONLINE_OUTPUT_LEN,
    SERVED_MODEL_NAME,
    XCHECK,
)
from tpprof.offline import parse_points

FIXTURE_RESULT = os.path.join(os.path.dirname(__file__), "fixtures", "bench_serve_result_detailed.json")
T0 = 1_790_000_000.0
C = model.CONSTANTS["central"]

MU_RPS = {"TP1": 20.0, "TP2": 40.0, "DP2": 46.0}             # synthetic saturation, req/s
TPOT_SPREAD = (0.5, 1.5)                                     # per-request TPOT factor, uniform
# Unloaded TPOT at the 90th percentile of the spread: 6 / 4 / 5.4 ms, so the goodput curves cross near 14.7 ms.
TPOT0_S = {c: t / 1.4 for c, t in {"TP1": 0.0060, "TP2": 0.0040, "DP2": 0.0054, "DP2rand": 0.0056}.items()}
KV_TOKENS = {"TP1": 420_000, "TP2": 950_000}
COLD_KV_FACTOR = 0.95                                          # D5-22: a cold boot has ~5% less KV
ARM_STEP_FACTOR = {"base": 1.0, "AR1": 1.02, "AR2": 1.04, "AR3": 1.0, "G1": 1.15, "G2": 1.8,
                   "EXECuni": 0.99, "FIBtrtllm": 1.01}
IDLE_B1 = {("TP1", "base"): 0.03, ("TP1", "G1"): 0.08, ("TP1", "G2"): 0.20,
           ("TP2", "base"): 0.05, ("TP2", "G1"): 0.15, ("TP2", "G2"): 0.40}
GPU1_STEP_FACTOR = 1.005                                       # AM14: GPU1 0.5% slower than GPU0
SAT_TPOT_S = 0.001
ITL_CHUNKS = 15                                                # SSE chunks per request (D4-14)


def default_fail(spec: matrix.RunSpec) -> bool:
    return spec.kind == "offline" and spec.arm == "EXECuni"


def default_skip(spec: matrix.RunSpec) -> bool:
    return spec.kind == "tokbench"


def build(results_dir: str, tiers: Iterable[str] = ("P0", "P1", "P2"), rounds: int = 3,
          fi_backend: str = "trtllm", engine: str = "fake", sat_prompts: int = 100, sweep_prompts: int = 40,
          fail: Callable[[matrix.RunSpec], bool] = default_fail,
          skip: Callable[[matrix.RunSpec], bool] = default_skip) -> list[matrix.RunSpec]:
    """Write a complete synthetic results dir; returns the specs of the matrix (done, failed and skipped)."""
    raw = os.path.join(results_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    grid = matrix.rate_grid(MU_RPS)
    specs = matrix.build_matrix(tiers, grid, fi_backend, rounds)
    sat_ids = [s.run_id for s in specs if s.kind == "serve_session" and s.p("phase") == "sat" and s.arm == "base"]
    _write_json(os.path.join(raw, "rate_grid.json"), {"mu_rps": MU_RPS, "grid": grid, "sources": sat_ids})
    summary: dict[str, list] = {"done": [], "failed": [], "skipped": []}
    booted: set[str] = set()
    for i, spec in enumerate(specs):
        if skip(spec):
            summary["skipped"].append({"run_id": spec.run_id, "reason": "dependency_failed:synthetic"})
            continue
        t = T0 + 60.0 * i
        run_dir = os.path.join(raw, spec.run_id)
        os.makedirs(run_dir, exist_ok=True)
        _write_json(os.path.join(run_dir, "spec.json"), spec.to_dict())
        _write_json(os.path.join(run_dir, "cmd.json"), [_cmd(spec, t)])
        if fail(spec):
            _write_json(os.path.join(run_dir, "failed.json"),
                        {"run_id": spec.run_id, "status": "failed", "reason": "engine_start_failed",
                         "detail": "EngineCore failed to start.", "log_tails": {"engine.log": ["EngineCore failed"]}})
            summary["failed"].append(spec.run_id)
            continue
        _write_outputs(spec, run_dir, engine, fi_backend, booted, sat_prompts, sweep_prompts)
        _write_json(os.path.join(run_dir, "done.json"),
                    {"run_id": spec.run_id, "status": "done", "duration_s": 30.0, "artifacts": []})
        summary["done"].append(spec.run_id)
    _write_json(os.path.join(raw, "_last_run.json"), summary)
    return specs


# ------------------------------------------------------------------------------------------ helpers

def _write_json(path: str, doc: object) -> None:
    with open(path, "w") as f:
        json.dump(doc, f)


def _cmd(spec: matrix.RunSpec, t: float) -> dict:
    return {"argv": ["synthetic", spec.kind], "env_overrides": {}, "cwd": None, "t_wall_start": t,
            "t_mono_start": t - T0, "t_wall_end": t + 30.0, "t_mono_end": t - T0 + 30.0, "exit_code": 0,
            "timed_out": False, "log": "run.log"}


def _tp(config: str) -> int:
    return 2 if config == "TP2" else 1


def _engine_names(config: str) -> list[str]:
    if config == "DP2rand":
        return list(matrix.DP2RAND_ENGINES)
    return [config]


def _write_outputs(spec: matrix.RunSpec, run_dir: str, engine: str, fi_backend: str, booted: set[str],
                   sat_prompts: int, sweep_prompts: int) -> None:
    kind = spec.kind
    if kind in ("smoke", "offline", "bench_latency_xcheck", "trace", "serve_session"):
        write_effective_config(run_dir, spec, fi_backend, cold=spec.config not in booted)
        booted.add(spec.config)
    if kind == "preflight":
        _write_json(os.path.join(run_dir, "preflight.json"), {"ok": True, "hard": [], "soft": []})
    elif kind == "envcapture":
        _write_json(os.path.join(run_dir, "env.json"), {"vllm": "0.30.0", "nccl": "2.30.7"})
        for name in ("topo.txt", "nvlink.txt"):
            with open(os.path.join(run_dir, name), "w") as f:
                f.write("GPU0\tX\tNV18\n")
    elif kind == "smoke" and spec.p("gpu1_check"):
        write_points(os.path.join(run_dir, "points"), spec.config, spec.arm, "decode:b1", engine,
                     step_factor=GPU1_STEP_FACTOR)
    elif kind == "offline":
        write_points(os.path.join(run_dir, "points"), spec.config, spec.arm, str(spec.p("points")), engine)
    elif kind == "bench_latency_xcheck":
        x = XCHECK
        lat = _decode_latency(spec.config, spec.arm, x["batch"], x["input_len"], x["output_len"]) * 1.01
        lats = [lat * (1 + 0.001 * (j - x["iters"] / 2)) for j in range(x["iters"])]
        pcts = np.percentile(lats, results.LATENCY_PERCENTILES)
        _write_json(os.path.join(run_dir, "bench_latency.json"),
                    {"avg_latency": float(np.mean(lats)), "latencies": lats,
                     "percentiles": {str(p): float(v) for p, v in zip(results.LATENCY_PERCENTILES, pcts)}})
    elif kind == "trace":
        _write_json(os.path.join(run_dir, "trace_summary.json"),
                    trace_summary(spec.config, spec.arm, str(spec.p("points"))))
    elif kind == "serve_session":
        write_session(run_dir, spec, sat_prompts, sweep_prompts)
    elif kind.startswith("comm_"):
        with open(os.path.join(run_dir, "comm_rows.jsonl"), "w") as f:
            for row in comm_rows(kind, str(spec.p("variant", "none:none"))):
                f.write(json.dumps(row) + "\n")


def write_effective_config(run_dir: str, spec: matrix.RunSpec, fi_backend: str, cold: bool) -> None:
    per_engine = []
    for name in _engine_names(spec.config):
        per_engine += [KV_TOKENS.get(name, KV_TOKENS["TP1"])] * (2 if name == "DP2" else 1)
    factor = (COLD_KV_FACTOR if cold else 1.0) * (1.02 if spec.arm in ("G1", "G2") else 1.0)
    tp2 = spec.config == "TP2"
    eff = logparse.EffectiveConfig(
        vllm_version="0.30.0", kv_cache_tokens=[int(k * factor) for k in per_engine],
        max_concurrency=[round(k * factor / 9216, 2) for k in per_engine],
        available_kv_gib=[round(k * factor * (65536 if tp2 else 131072) / 2**30, 2) for k in per_engine],
        chunked_prefill_tokens=8192, v2_model_runner=True, attention_backend="FLASH_ATTN",
        attention_explicit=True, flash_attn_version=3,
        ar_backends=["FLASHINFER", "CUSTOM", "SYMM_MEM", "PYNCCL"] if tp2 else None,
        fi_backend=fi_backend if tp2 else None, fi_backend_fallback=tp2 and fi_backend == "trtllm",
        nccl_version="2.30.7" if tp2 else None, executor=None if spec.config == "DP2" else "mp",
        enforce_eager=spec.arm == "G2", startup_complete=spec.kind in ("serve_session", "smoke"),
        graph_capture=[] if spec.arm == "G2" else ["Graph capturing finished in 18 secs, took 0.52 GiB"])
    doc = dataclasses.asdict(eff)
    doc["violations"] = []
    _write_json(os.path.join(run_dir, "effective_config.json"), doc)


# ------------------------------------------------------------------------------------------ offline

def _decode_step(config: str, arm: str, batch: int) -> float:
    ar_path = "nccl_unfused" if arm == "AR3" else "fused"
    return model.decode_step_time(_tp(config), batch, model.DECODE_MEAN_CTX, C, ar_path) * ARM_STEP_FACTOR[arm]


def _decode_latency(config: str, arm: str, batch: int, input_len: int, output_len: int,
                    step_factor: float = 1.0) -> float:
    prefill = 0.5 * batch * model.prefill_time(_tp(config), input_len, C)
    return prefill + output_len * _decode_step(config, arm, batch) * step_factor


def write_points(points_dir: str, config: str, arm: str, points: str, engine: str,
                 step_factor: float = 1.0) -> None:
    os.makedirs(points_dir, exist_ok=True)
    for p in parse_points(points):
        if p.kind == "decode":
            lat = _decode_latency(config, arm, p.batch, p.input_len, p.output_len, step_factor)
        else:
            lat = model.prefill_time(_tp(config), p.input_len, C) * ARM_STEP_FACTOR[arm]
        lats = [lat * (1 + 0.001 * (j - (p.iters - 1) / 2)) for j in range(p.iters)]
        meta = {"kind": p.kind, "batch": p.batch, "input_len": p.input_len, "output_len": p.output_len,
                "warmup": p.warmup, "iters": p.iters, "config": config, "arm": arm,
                "t_wall_start": T0, "t_mono_start": 0.0, "engine": engine}
        results.write_latency_result(os.path.join(points_dir, p.filename()), lats, meta)


# ------------------------------------------------------------------------------------------ traces

def trace_summary(config: str, arm: str, points: str) -> dict:
    tp = _tp(config)
    decode = points.startswith("decode")
    steps = 256 if decode else 5
    idle = IDLE_B1.get((config, arm), 0.06) if points == "decode:b1" else 0.04
    batch = int(points.split(":b")[1].split(",")[0]) if decode else 1
    step_ms = (_decode_step(config, arm, batch) if decode
               else model.prefill_time(tp, int(points.split(":")[1]), C)) * 1e3 * 1.1
    ar, ag = model.allreduces_per_step(tp), model.allgathers_per_step(tp)
    frac = ({"all_reduce": 0.12, "all_gather": 0.01, "gemm": 0.62, "attention": 0.1, "norm_act_rope": 0.08,
             "sampling": 0.04, "memcpy": 0.02, "other": 0.01} if tp == 2 else
            {"all_reduce": 0.0, "all_gather": 0.0, "gemm": 0.7, "attention": 0.1, "norm_act_rope": 0.13,
             "sampling": 0.04, "memcpy": 0.02, "other": 0.01})
    ranks = [{"rank": r, "pid": 42420 + r, "device": r, "steps": steps, "decode_steps": steps if decode else 0,
              "ar_ops_per_step": {"min": ar, "max": ar, "mode": ar},
              "ag_ops_per_step": {"min": ag, "max": ag, "mode": ag},
              "mean_step_ms": step_ms, "busy_frac": 1 - idle, "category_frac": dict(frac),
              "unclassified_top": [["mystery_kernel", 0.01]]} for r in range(tp)]
    return {"gate": {"ok": True, "reasons": []}, "ranks": ranks,
            "ar_wire_s_per_step": 3e-4 if tp == 2 else None, "ar_sync_wait_s_per_step": 2e-5 if tp == 2 else None,
            "idle_est": idle, "window_ns": [0, int(steps * step_ms * 1e6)], "launch_ts_missing": 0}


# ------------------------------------------------------------------------------------------ comm

def comm_rows(kind: str, variant: str) -> list[dict]:
    rows = []
    if kind == "comm_m3":
        for mode, alpha in (("eager", 12e-6), ("graph", 6e-6)):
            for k in range(11, 29):
                s = 2 ** k
                t = alpha + s / 2.6e11
                rows.append({"impl": "torch_nccl", "variant": variant, "mode": mode, "bytes": s, "n": 50,
                             "median_us": t * 1e6, "p25_us": t * 0.99e6, "p75_us": t * 1.01e6,
                             "algbw_GBps": s / t / 1e9, "busbw_GBps": s / t / 1e9})
    elif kind == "comm_m2":
        for tokens in matrix.M2_TOKENS:
            nbytes = model.ar_message_bytes(tokens)
            for op, backend, oneshot, alpha, beta in (
                    ("flashinfer_trtllm_fused_allreduce_rmsnorm_oneshot", "trtllm", True, 5e-6, 3.0e11),
                    ("standard_allreduce_rmsnorm", "standard", None, 8e-6, 2.4e11)):
                rows.append({"op": op, "num_tokens": tokens, "bytes": nbytes, "backend": backend,
                             "oneshot": oneshot, "ms": (alpha + nbytes / beta) * 1e3})
    elif kind == "comm_m1":
        for tokens in matrix.M2_TOKENS:
            nbytes = model.ar_message_bytes(tokens)
            for impl, alpha in (("ca_1stage", 5e-6), ("pynccl", 7e-6)):
                rows.append({"impl": impl, "mode": "graph", "bytes": nbytes, "mean_us": (alpha + nbytes / 2.5e11) * 1e6})
    elif kind == "comm_m4":
        for k in range(3, 29):
            s = 2 ** k
            t = 7e-6 + s / 2.5e11
            rows.append({"impl": "nccl_tests", "mode": "eager", "place": "out_of_place", "bytes": s,
                         "time_us": t * 1e6, "p50_us": t * 1e6, "algbw_GBps": s / t / 1e9, "busbw_GBps": s / t / 1e9,
                         "nwrong": 0, "algo": "RING", "proto": "LL", "nccl_version": 23007})
    return rows


# ------------------------------------------------------------------------------------------ serve

def _sub_runs(spec: matrix.RunSpec) -> list[tuple[str, float, int]]:
    """(phase, rate, seed) per sub-run, in the runner's order."""
    phase = spec.p("phase")
    if phase == "sat":
        return [("sat", float("inf"), int(s)) for s in spec.p("seeds")]
    if phase == "sweep":
        return [("sweep", float(r), int(spec.p("seed_base")) + i) for i, r in enumerate(spec.p("rates"))]
    if phase == "pc":
        subs = [("pc", float(spec.p("rate")), 1000 * k) for k in range(1, int(spec.p("repeats")) + 1)]
        return subs + [("pc", float("inf"), 1) for _ in range(int(spec.p("sat_extra")))]
    raise ValueError(f"unknown phase {phase!r}")


def write_session(run_dir: str, spec: matrix.RunSpec, sat_prompts: int, sweep_prompts: int) -> None:
    _write_cpu_csv(os.path.join(run_dir, "cpu.csv"), api_cpu=90.0 if (spec.config, spec.p("phase")) == ("DP2", "sat")
                   and spec.arm == "base" else 35.0)
    with open(os.path.join(run_dir, "gpu.csv"), "w") as f:
        f.write("timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,"
                "clocks_event_reasons.active\n")
        for i in range(3):
            f.write(f"2026/10/01 10:00:0{i}.000, 0, 1980, 2619, 400.00, 55, 90, 70000, 0x0000000000000000\n")
    for k, (phase, rate, seed) in enumerate(_sub_runs(spec)):
        sub = os.path.join(run_dir, f"sub-{k}")
        os.makedirs(sub, exist_ok=True)
        n = sat_prompts if rate == float("inf") else sweep_prompts
        doc = serve_result(spec.config, spec.arm, spec.round, rate, seed, n)
        with open(os.path.join(sub, "result.json"), "w") as f:
            json.dump(doc, f)
        with open(os.path.join(sub, "client.log"), "w") as f:
            f.write("============ Serving Benchmark Result ============\n")
        preempt = 12 if rate == float("inf") and spec.config in ("TP1", "DP2") else 0
        hits = 400 if spec.arm == "PCon" else 0
        before = _prom(100 * k, 0, 0)
        after = _prom(100 * k + preempt, hits, n)
        for name, text in (("metrics_before.prom", before), ("metrics_after.prom", after)):
            with open(os.path.join(sub, name), "w") as f:
                f.write(text)
        _write_json(os.path.join(sub, "validation.json"),
                    {"valid": True, "violations": [], "flags": ["preempted"] if preempt else []})
        _write_json(os.path.join(sub, "meta.json"),
                    {"phase": phase, "rate": "inf" if rate == float("inf") else rate, "seed": seed,
                     "config": spec.config, "arm": spec.arm, "round": spec.round, "k": k})


def _write_cpu_csv(path: str, api_cpu: float) -> None:
    with open(path, "w") as f:
        f.write("t_wall,pid,ppid,name,cmd,cpu_percent,rss_mib\n")
        for i in range(10):
            f.write(f"{T0 + i:.3f},100,1,python3,/usr/bin/python3 /usr/local/bin/vllm serve /models --port 8000,"
                    f"{api_cpu - 5 + i:.1f},900.0\n")
            f.write(f"{T0 + i:.3f},101,100,VLLM::EngineCore,VLLM::EngineCore,99.0,2000.0\n")
            f.write(f"{T0 + i:.3f},200,1,vllm,/usr/bin/python3 /usr/local/bin/vllm bench serve --backend vllm,"
                    f"{20 + i:.1f},300.0\n")


def _prom(preemptions: float, hits: float, requests: float) -> str:
    labels = f'engine="0",model_name="{SERVED_MODEL_NAME}"'
    values = {"vllm:num_preemptions_total": preemptions, "vllm:prefix_cache_queries_total": 1000 + hits,
              "vllm:prefix_cache_hits_total": hits, "vllm:prompt_tokens_total": requests * ONLINE_INPUT_LEN,
              "vllm:generation_tokens_total": requests * ONLINE_OUTPUT_LEN,
              "vllm:request_success_total": requests, "vllm:num_requests_running": 0,
              "vllm:num_requests_waiting": 0, "vllm:kv_cache_usage_perc": 0.0}
    return "".join(f"{name}{{{labels}}} {value}\n" for name, value in values.items())


def _tpot(config: str, rate: float) -> float:
    cap = MU_RPS["DP2" if config == "DP2rand" else config]
    return TPOT0_S[config] / max(1 - rate / cap, 0.02)


def serve_result(config: str, arm: str, rnd: int, rate: float, seed: int, n: int) -> dict:
    """A C7 result JSON (key order of the real fixture, --metadata keys after num_prompts)."""
    rng = random.Random(f"{config}-{arm}-{rnd}-{rate}-{seed}")
    spread = list(np.linspace(*TPOT_SPREAD, n))
    rng.shuffle(spread)
    sat = rate == float("inf")
    if sat:
        mu = MU_RPS["DP2" if config == "DP2rand" else config] * (1 + 0.01 * (seed - 2))
        starts = [100.0 + i / mu for i in range(n)]
        tpots = [SAT_TPOT_S] * n
        ttfts = [0.05] * n
    else:
        starts = [100.0 + i / rate for i in range(n)]
        base = _tpot(config, rate) * (0.97 if arm == "PCon" else 1.0) * (1 + 0.02 * (rnd - 2))
        tpots = [base * f for f in spread]
        ttfts = [0.05 + 0.1 * f for f in spread]      # always inside the 0.5 s TTFT SLO
    out = ONLINE_OUTPUT_LEN
    itls = [[(out - 1) * tp / ITL_CHUNKS] * ITL_CHUNKS for tp in tpots]
    lats = [tt + (out - 1) * tp for tt, tp in zip(ttfts, tpots)]
    duration = max(s + lat for s, lat in zip(starts, lats)) - min(starts)
    with open(FIXTURE_RESULT) as f:
        fixture = json.load(f)
    doc = {key: fixture[key] for key in list(fixture)[:list(fixture).index("num_prompts")]}
    doc["label"] = config
    doc["num_prompts"] = n
    doc["tpprof_config"], doc["tpprof_rate"], doc["tpprof_round"] = config, "inf" if sat else str(rate), str(rnd)
    doc.update({"request_rate": "inf" if sat else rate, "burstiness": 1.0, "max_concurrency": None,
                "duration": duration, "completed": n, "failed": 0, "total_input_tokens": n * ONLINE_INPUT_LEN,
                "total_output_tokens": n * out, "request_throughput": n / duration, "request_goodput": None,
                "output_throughput": n * out / duration, "total_token_throughput": n * (out + ONLINE_INPUT_LEN) / duration,
                "input_lens": [ONLINE_INPUT_LEN] * n, "output_lens": [out] * n, "ttfts": ttfts, "itls": itls,
                "latencies": lats, "start_times": starts, "queue_times": [0.0] * n, "generated_texts": [""] * n,
                "errors": [""] * n, "max_output_tokens_per_s": 0.0, "max_concurrent_requests": n, "rtfx": 0.0})
    per_metric = {"ttft": [t * 1e3 for t in ttfts], "tpot": [t * 1e3 for t in tpots],
                  "itl": [x * 1e3 for row in itls for x in row], "e2el": [x * 1e3 for x in lats]}
    for m, xs in per_metric.items():
        doc[f"mean_{m}_ms"], doc[f"median_{m}_ms"], doc[f"std_{m}_ms"] = (
            float(np.mean(xs)), float(np.median(xs)), float(np.std(xs)))
        for p in METRIC_PERCENTILES:
            doc[f"p{int(p)}_{m}_ms"] = float(np.percentile(xs, p))
    return doc

