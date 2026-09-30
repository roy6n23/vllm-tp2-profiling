"""M1 / M2: thin wrappers around vLLM v0.30.0's own communicator benchmarks (spec 4.6, AM25, AM31, D7-9).

The scripts are vendored in ``third_party/vllm_benchmarks`` (see its SOURCE.md) and loaded by file path, so an
editable install of tpprof finds them without packaging ``third_party``.

- **M2** (P0): ``benchmark_fused_collective.py``, FlashInfer fused AR+RMSNorm (trtllm / mnnvl, one- and two-shot)
  vs vLLM all-reduce + RMSNorm. Modified to write one JSON line per (num_tokens, op); see SOURCE.md.
- **M1** (P2, cross-check): ``benchmark_device_communicators.py``, unmodified. The wrapper sets
  ``HIDDEN_SIZE = 4096`` and initializes vLLM's distributed environment first; without that call v0.30.0 drops
  FlashInfer from the results (D7-9).

Run on the box as ``torchrun --nproc-per-node 2 -m tpprof.vendored m1|m2 --out PATH``. ``--synthetic`` writes
files in the scripts' output formats without importing torch or vllm.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

from tpprof.constants import LLAMA31_8B

M2_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
HIDDEN = LLAMA31_8B.hidden
THIRD_PARTY = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "vllm_benchmarks"
_ITEMSIZE = {"bfloat16": 2, "float16": 2, "float32": 4}
_M2_KEYS = ("num_tokens", "hidden_dim", "dtype", "op", "ms")

# Result names the scripts produce (quant mode "none"), used by the synthetic outputs.
M1_IMPLS = ("ca_1stage", "ca_2stage", "pynccl", "pynccl-symm", "symm_mem_multimem", "symm_mem_two_shot",
            "flashinfer_trtllm", "flashinfer_mnnvl", "pynccl_ag", "pynccl_rs")
M2_OPS = ("standard_allreduce__native_rms_norm", "standard_allreduce__custom_rms_norm",
          "standard_allreduce_rmsnorm_native_compiled",
          "flashinfer_trtllm_fused_allreduce_rmsnorm_oneshot", "flashinfer_trtllm_fused_allreduce_rmsnorm_twoshot",
          "flashinfer_mnnvl_fused_allreduce_rmsnorm_oneshot", "flashinfer_mnnvl_fused_allreduce_rmsnorm_twoshot")


def m1_argv(out_json: str, torchrun: str = "torchrun") -> list[str]:
    return [torchrun, "--nproc-per-node", "2", "-m", "tpprof.vendored", "m1", "--out", out_json]


def m2_argv(out_jsonl: str, torchrun: str = "torchrun") -> list[str]:
    return [torchrun, "--nproc-per-node", "2", "-m", "tpprof.vendored", "m2", "--out", out_jsonl]


def m1_script_argv(out_json: str) -> list[str]:
    """``sys.argv`` for benchmark_device_communicators.main() (AM31: same token list as M2)."""
    return ["x", "--sequence-lengths", *map(str, M2_TOKENS), "--num-warmup", "5", "--num-trials", "200",
            "--output-json", out_json]


def m2_script_argv(out_jsonl: str) -> list[str]:
    """``sys.argv`` for benchmark_fused_collective.main(), exactly as AM31 gives it."""
    return ["x", "--hidden-dim", str(HIDDEN), "--num-tokens", *map(str, M2_TOKENS), "--dtypes", "bfloat16",
            "--quant-modes", "none", "--warmup", "5", "--trials", "50", "--output-file", out_jsonl]


def parse_m1(path: str) -> list[dict]:
    """benchmark_device_communicators.py ``--output-json`` -> rows ``{"impl", "mode": "graph", "bytes",
    "mean_us"}``; bytes is the (seq_len, hidden_size) tensor size, mean_us the graph-replay mean per op."""
    with open(path) as f:
        doc = json.load(f)
    try:
        hidden, dtype, results = doc["hidden_size"], doc["dtype"], doc["results"]
    except KeyError as e:
        raise ValueError(f"{path}: M1 JSON lacks {e.args[0]!r}") from e
    itemsize = _ITEMSIZE[dtype.replace("torch.", "")]
    rows = [{"impl": impl, "mode": "graph", "bytes": int(seq_len) * hidden * itemsize, "mean_us": ms * 1e3}
            for seq_len, entry in results.items() for impl, ms in entry["timings"].items()]
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _op_backend(op: str) -> tuple[str, bool | None]:
    backend = op.split("_")[1] if op.startswith("flashinfer_") else "standard"
    oneshot = True if op.endswith("_oneshot") else False if op.endswith("_twoshot") else None
    return backend, oneshot


def parse_m2(path: str) -> list[dict]:
    """The vendored M2 JSONL -> rows ``{"op", "num_tokens", "bytes", "backend", "oneshot", "ms"}``.
    backend is ``trtllm`` / ``mnnvl`` for FlashInfer ops, ``standard`` for vLLM's all-reduce dispatch + RMSNorm;
    oneshot is None for standard ops. ms is a float (AM31); a failed op is ``inf``, the script's own failure value,
    so a failure is never read as a time."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            missing = [k for k in _M2_KEYS if k not in d]
            if missing:
                raise ValueError(f"{path}: line {i} lacks {', '.join(missing)}")
            if isinstance(d["ms"], bool) or not isinstance(d["ms"], (int, float)):
                raise ValueError(f"{path}: line {i} ms is not a number: {d['ms']!r}")
            backend, oneshot = _op_backend(d["op"])
            rows.append({"op": d["op"], "num_tokens": d["num_tokens"],
                         "bytes": d["num_tokens"] * d["hidden_dim"] * _ITEMSIZE[d["dtype"]],
                         "backend": backend, "oneshot": oneshot,
                         "ms": float(d["ms"])})
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def script_spec(name: str):
    return importlib.util.spec_from_file_location(f"third_party.vllm_benchmarks.{name}", THIRD_PARTY / f"{name}.py")


def _load(name: str):
    spec = script_spec(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_ms(i: int, nbytes: int) -> float:
    return (5e-6 + 1e-6 * i + nbytes / 2.6e11) * 1e3


def _synthetic_m1(out: str) -> None:
    results = {str(n): {"timings": {impl: _synthetic_ms(i, n * HIDDEN * 2) for i, impl in enumerate(M1_IMPLS)},
                        "speedup_info": {}} for n in M2_TOKENS}
    doc = {"world_size": int(os.environ.get("WORLD_SIZE", "2")), "dtype": "torch.bfloat16", "hidden_size": HIDDEN,
           "sequence_lengths": list(M2_TOKENS), "num_warmup": 5, "num_trials": 200,
           "cuda_graph_capture_cycles": 10, "results": results}
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)


def _synthetic_m2(out: str) -> None:
    with open(out, "w") as f:
        for n in M2_TOKENS:
            for i, op in enumerate(M2_OPS):
                f.write(json.dumps({"num_tokens": n, "hidden_dim": HIDDEN, "dtype": "bfloat16", "use_residual": True,
                                    "op": op, "ms": _synthetic_ms(i, n * HIDDEN * 2)}) + "\n")


def _run_m1(out: str) -> None:
    from vllm.distributed.parallel_state import init_distributed_environment

    module = _load("benchmark_device_communicators")
    module.HIDDEN_SIZE = HIDDEN
    rank = int(os.environ["RANK"])
    init_distributed_environment(int(os.environ["WORLD_SIZE"]), rank, "env://",
                                 int(os.environ.get("LOCAL_RANK", rank)), backend="gloo")
    sys.argv = m1_script_argv(out)
    module.main()


def _run_m2(out: str) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config

    module = _load("benchmark_fused_collective")
    sys.argv = m2_script_argv(out)
    with set_current_vllm_config(VllmConfig()):   # as the script's own __main__ block does
        module.main()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tpprof.vendored", allow_abbrev=False)
    ap.add_argument("which", choices=("m1", "m2"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="write the output format without torch or vllm")
    args = ap.parse_args(argv)
    if args.synthetic:
        if int(os.environ.get("RANK", "0")) == 0:
            (_synthetic_m1 if args.which == "m1" else _synthetic_m2)(args.out)
        return 0
    saved = sys.argv
    try:
        (_run_m1 if args.which == "m1" else _run_m2)(args.out)
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    sys.exit(main())
