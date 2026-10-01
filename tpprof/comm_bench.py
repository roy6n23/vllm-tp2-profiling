"""M3: our own torch.distributed NCCL all-reduce sweep, eager and CUDA graph (spec 4.6, D7-2, D7-4, D7-10).

Run on the box as ``torchrun --nproc-per-node 2 -m tpprof.comm_bench ...`` (see ``torchrun_argv``). The NCCL
variant is chosen by the environment of the torchrun process (``variant_env``), because NCCL reads NCCL_ALGO and
NCCL_PROTO once at communicator init; ``--variant`` only labels the rows, so the caller passes the label and the
``variant_env`` of the same variant together. ``--synthetic`` writes rows of the same shape without importing
torch (tests, dry runs).

Rank 0 writes one JSONL row per (mode, size):
``{"impl", "variant", "mode", "bytes", "n", "median_us", "p25_us", "p75_us", "algbw_GBps", "busbw_GBps"}``,
where ``n`` is the number of timed samples and busbw = algbw * 2 (w - 1) / w, which is algbw for the 2-rank run.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import sys
import threading

from tpprof import stats
from tpprof.constants import NCCL_EXPECTED, TORCH_CUDA

SIZES = tuple(2 ** k for k in range(11, 29))           # 2 KiB .. 256 MiB
VARIANTS = ((None, None), ("ring", "LL"), ("ring", "LL128"), ("ring", "Simple"),
            ("tree", "LL"), ("tree", "LL128"), ("tree", "Simple"), ("nvls", "Simple"))
MODES = ("eager", "graph")
IMPL = "torch_nccl"
ROW_KEYS = ("impl", "variant", "mode", "bytes", "n", "median_us", "p25_us", "p75_us", "algbw_GBps",
            "busbw_GBps")
ITERS, WARMUP, GRAPH_OPS = 50, 10, 20

# Synthetic timing model: alpha 6 us, beta 260 GB/s, +-2 % deterministic noise.
SYN_ALPHA_S, SYN_BETA_BPS, SYN_NOISE = 6e-6, 2.6e11, 0.02

# NCCL_DEBUG=INFO line formats (D7-4, NCCL 2.29.7 src/enqueue.cc:793, src/transport/nvls.cc:202, src/init.cc:649).
_TUNING = re.compile(r"NCCL INFO (?P<func>\w+): (?P<bytes>\d+) Bytes -> Algo (?P<algo>\w+) proto (?P<proto>\w+) "
                     r"channel\{Lo\.\.Hi\}=\{\d+\.\.\d+\}")
_NVLS = re.compile(r"NVLS multicast support is (?P<not>not )?available on dev \d+")
# A forced algorithm/protocol NCCL cannot run is a hard error since 2.24 (D7-2); a failed NVLS multicast
# bind fails communicator init in 2.29+ (D7-3, src/transport/nvls.cc:287). Spec 4.6: the first is recorded as
# unsupported, the second retried with NCCL_NVLS_ENABLE=0.
NCCL_UNSUPPORTED = "no algorithm/protocol available"
NVLS_BIND_FAILED = "Failed to bind NVLink SHARP (NVLS) Multicast memory"
_VERSION = re.compile(r"(?:^|NCCL INFO )NCCL version (?P<v>\d+\.\d+\.\d+)(?:\+cuda[\d.]+)?", re.MULTILINE)


def _label(x: str | None) -> str:
    return "none" if x is None else x


def variant_label(variant: tuple) -> str:
    """``("ring", "LL")`` -> ``"ring:LL"``; None renders as ``none`` (ruling R5)."""
    algo, proto = variant
    return f"{_label(algo)}:{_label(proto)}"


def parse_variant(label: str) -> tuple[str | None, str | None]:
    algo, sep, proto = label.partition(":")
    if not sep or not algo or not proto:
        raise ValueError(f"variant must be ALGO:PROTO (none for the default), got {label!r}")
    return (None if algo == "none" else algo), (None if proto == "none" else proto)


def variant_env(algo: str | None, proto: str | None, *, debug_dir: str | None = None) -> dict[str, str]:
    """Environment for one NCCL variant, in the per-function form so other collectives keep their defaults
    (D7-2). A diagnostic run also logs INIT, ENV and TUNING (D7-4) to ``<debug_dir>/nccl.%h.%p.log``; the
    keyword-only ``debug_dir`` supplies that ``<dir>`` and marks the run as diagnostic."""
    env = {}
    if algo is not None:
        env["NCCL_ALGO"] = f"allreduce:{algo}"
    if proto is not None:
        env["NCCL_PROTO"] = f"allreduce:{proto}"
    if debug_dir is not None:
        env.update({"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,ENV,TUNING",
                    "NCCL_DEBUG_FILE": os.path.join(debug_dir, "nccl.%h.%p.log")})
    return env


def torchrun_argv(out_jsonl: str, modes: str, variant: tuple, torchrun: str = "torchrun", *, iters: int = ITERS,
                  warmup: int = WARMUP) -> list[str]:
    return [torchrun, "--nproc-per-node", "2", "-m", "tpprof.comm_bench", "--out", out_jsonl, "--mode", modes,
            "--iters", str(iters), "--warmup", str(warmup), "--graph-ops", str(GRAPH_OPS),
            "--variant", variant_label(variant)]


def parse_tuning_lines(text: str) -> list[dict]:
    """Per-collective TUNING lines (rank 0 only, one per enqueue; a graph-captured op logs once, at capture).
    Algo and proto are as NCCL prints them (upper case: RING, TREE, NVLS; LL, LL128, SIMPLE)."""
    return [{"func": m["func"], "bytes": int(m["bytes"]), "algo": m["algo"], "proto": m["proto"]}
            for m in _TUNING.finditer(text)]


def parse_nvls_support(text: str) -> bool | None:
    """False if any device reports no NVLS multicast, True if all that report have it, None if none report."""
    found = [m["not"] is None for m in _NVLS.finditer(text)]
    return all(found) if found else None


def parse_nccl_version(text: str) -> str | None:
    """The loaded NCCL version, e.g. ``2.30.7``, from the ``NCCL version 2.30.7+cuda13.0`` line (bare under
    NCCL_DEBUG=VERSION, INFO-prefixed under NCCL_DEBUG=INFO)."""
    m = _VERSION.search(text)
    return m["v"] if m else None


def load_rows(path: str) -> list[dict]:
    """Rows of an M3 JSONL file. Raises ValueError naming the file and the problem (never returns [])."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {i} is not JSON: {e}") from e
            missing = [k for k in ROW_KEYS if k not in row]
            if missing:
                raise ValueError(f"{path}: line {i} lacks {', '.join(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _row(variant: str, mode: str, size: int, world_size: int, times_s: list[float]) -> dict:
    s = stats.summarize(times_s)
    algbw = size / s["median"] / 1e9
    return {"impl": IMPL, "variant": variant, "mode": mode, "bytes": size, "n": s["n"],
            "median_us": s["median"] * 1e6, "p25_us": s["p25"] * 1e6, "p75_us": s["p75"] * 1e6,
            "algbw_GBps": algbw, "busbw_GBps": algbw * 2 * (world_size - 1) / world_size}


def _write_rows(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _synthetic_nccl_debug(variant: tuple, modes: list[str]) -> None:
    """Under NCCL_DEBUG=INFO with NCCL_DEBUG_FILE, write what rank 0's NCCL would log (D7-4 formats)."""
    if os.environ.get("NCCL_DEBUG") != "INFO" or not os.environ.get("NCCL_DEBUG_FILE"):
        return
    host, pid = socket.gethostname(), os.getpid()
    path = os.environ["NCCL_DEBUG_FILE"].replace("%h", host).replace("%p", str(pid))
    pre = f"{host}:{pid}:{threading.get_native_id()} [0] NCCL INFO "
    nvls = os.environ.get("FAKE_VLLM_NO_MULTICAST") != "1"
    lines = [f"NCCL version {NCCL_EXPECTED}+cuda{TORCH_CUDA}"]
    lines += [f"{k} set by environment to {os.environ[k]}" for k in ("NCCL_ALGO", "NCCL_PROTO") if k in os.environ]
    lines.append(f"NVLS multicast support is {'' if nvls else 'not '}available on dev 0 "
                 f"(NVLS_NCHANNELS {16 if nvls else 0})")
    algo, proto = variant
    shown_algo = os.environ.get("FAKE_NCCL_TUNING_ALGO") or (algo or "ring").upper()   # tests: NCCL ignored it
    for _ in modes:
        for size in SIZES:
            p = proto or ("LL" if size <= 64 * 1024 else "Simple")
            lines.append(f"AllReduce: {size} Bytes -> Algo {shown_algo} proto {p.upper()} "
                         f"channel{{Lo..Hi}}={{0..1}}")
    with open(path, "a") as f:
        f.write("".join(pre + ln + "\n" for ln in lines))


def _synthetic(args, variant: tuple, modes: list[str], rank: int, world_size: int) -> int:
    if rank != 0:
        return 0
    # tests: the two NCCL init/launch failures spec 4.6 handles (FAKE_NCCL_NVLS_BIND_FAIL, FAKE_NCCL_UNSUPPORTED)
    if os.environ.get("FAKE_NCCL_NVLS_BIND_FAIL") == "1" and os.environ.get("NCCL_NVLS_ENABLE") != "0":
        print(f"NCCL WARN Cuda failure 1 'invalid argument'\nNCCL WARN {NVLS_BIND_FAILED}", file=sys.stderr)
        return 1
    if args.variant in os.environ.get("FAKE_NCCL_UNSUPPORTED", "").split(","):
        print(f"torch.distributed.DistBackendError: NCCL error: invalid usage (run with NCCL_DEBUG=WARN for "
              f"details), NCCL version {NCCL_EXPECTED}\nLast error:\n{NCCL_UNSUPPORTED}", file=sys.stderr)
        return 1
    _synthetic_nccl_debug(variant, modes)
    rows = []
    for mode in modes:
        for size in SIZES:
            rng = random.Random(f"{mode}:{size}")
            t = SYN_ALPHA_S + size / SYN_BETA_BPS
            rows.append(_row(args.variant, mode, size, world_size,
                             [t * (1 + rng.uniform(-SYN_NOISE, SYN_NOISE)) for _ in range(args.iters)]))
    _write_rows(args.out, rows)
    return 0


def _real(args, modes: list[str], rank: int, local_rank: int, world_size: int) -> int:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rows = []
    try:
        for size in SIZES:
            x = torch.zeros(size // 2, dtype=torch.bfloat16, device="cuda")   # zeros: sums never overflow
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(max(3, args.warmup)):
                    dist.all_reduce(x)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            for mode in modes:
                dist.barrier()
                times = _time_eager(torch, dist, x, args) if mode == "eager" else _time_graph(torch, dist, x, args)
                if rank == 0:
                    rows.append(_row(args.variant, mode, size, world_size, times))
            del x
        if rank == 0:
            _write_rows(args.out, rows)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


def _time_eager(torch, dist, x, args) -> list[float]:
    """Per-op times from CUDA events around each eager all_reduce (includes ProcessGroupNCCL CPU cost)."""
    for _ in range(args.warmup):
        dist.all_reduce(x)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)]
    for start, end in ev:
        start.record()
        dist.all_reduce(x)
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) / 1e3 for start, end in ev]


def _time_graph(torch, dist, x, args) -> list[float]:
    """``--graph-ops`` all_reduce calls captured into one CUDA graph; per-op time = replay time / ops."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.graph_ops):
            dist.all_reduce(x)
    for _ in range(args.warmup):
        graph.replay()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)]
    for start, end in ev:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    times = [start.elapsed_time(end) / 1e3 / args.graph_ops for start, end in ev]
    del graph
    return times


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tpprof.comm_bench", allow_abbrev=False)
    ap.add_argument("--out", required=True, help="JSONL rows (written by rank 0)")
    ap.add_argument("--mode", default="eager,graph", help="comma list of eager, graph")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--graph-ops", type=int, default=GRAPH_OPS)
    ap.add_argument("--variant", default="none:none", help="ALGO:PROTO row label (none for the default); the env sets NCCL_ALGO/NCCL_PROTO")
    ap.add_argument("--synthetic", action="store_true", help="no torch: rows from t = 6 us + s / 260 GB/s")
    args = ap.parse_args(argv)

    modes = [m for m in args.mode.split(",") if m]
    if not modes or any(m not in MODES for m in modes):
        ap.error(f"--mode must be a comma list of {', '.join(MODES)}, got {args.mode!r}")
    try:
        variant = parse_variant(args.variant)
    except ValueError as e:
        ap.error(str(e))
    args.variant = variant_label(variant)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))   # torchrun sets it; the harness is 2-rank
    if args.synthetic:
        return _synthetic(args, variant, modes, rank, world_size)
    return _real(args, modes, rank, local_rank, world_size)


if __name__ == "__main__":
    sys.exit(main())
