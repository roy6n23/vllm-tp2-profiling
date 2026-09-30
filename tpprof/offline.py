"""Offline driver: the vLLM `LLM` class with argv parsed by vLLM's own EngineArgs (spec 4.3, 4.5).

It mirrors `vllm bench latency` (D4-2, D4-3): random token-id prompts without BOS, the bench's
SamplingParams, and `time.perf_counter()` around each `llm.generate`. One engine start serves many
points, and each point is written to its own C3 file, so a crashed session resumes per point (AM13).

    python -m tpprof.offline --out DIR --meta JSON [--points SPEC]
        [--profile POINTSPEC --profile-warmup W --profile-iters K] -- <engine args>

With TPPROF_FAKE=1 a fake engine stands in for vLLM, so the driver runs on a Mac. vllm and torch
are imported only inside VllmEngine, which runs only on the box.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import multiprocessing
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterator, Mapping, Protocol, Sequence

import numpy as np

from tpprof.constants import (DECODE_BATCHES, DECODE_INPUT_LEN, DECODE_ITERS, DECODE_L1, DECODE_L2,
                              DECODE_WARMUP, PREFILL_ITERS, PREFILL_LENS, PREFILL_WARMUP)
from tpprof.results import ResultFormatError, load_latency_result, write_latency_result

log = logging.getLogger("tpprof.offline")

PROFILER_CONFIG = '{"profiler":"cuda"}'           # spec 4.5, D6-2: workers call cudaProfilerStart/Stop
MEASURE_RANGE = "tpprof:measure"                  # driver-side NVTX range around the measured iterations
PROFILE_SLEEP_BEFORE_STOP_S = 2.0                 # spec 4.5: sleep 2 s, then stop_profile
# A decode trace is one iteration of this many output tokens: the 256 steps the L1/L2 pair measures
# (spec 4.5 "about 256 decode steps"; plan T16 "1 iteration of output_len=256").
PROFILE_DECODE_OUTPUT_LEN = DECODE_L2 - DECODE_L1
PROMPT_VOCAB = 10000                              # bench latency: np.random.randint(10000, ...) (D4-2)
KINDS = ("prefill", "decode")


# ---------------------------------------------------------------------------------- points

@dataclass(frozen=True)
class Point:
    kind: str            # "prefill" | "decode"
    batch: int
    input_len: int
    output_len: int
    warmup: int
    iters: int

    def filename(self) -> str:
        """The C3 point file name."""
        return f"point-{self.kind}-b{self.batch}-i{self.input_len}-o{self.output_len}.json"


def _prefill(input_len: int) -> Point:
    return Point("prefill", 1, input_len, 1, PREFILL_WARMUP, PREFILL_ITERS)


def _decode_pair(batch: int) -> list[Point]:
    return [Point("decode", batch, DECODE_INPUT_LEN, n, DECODE_WARMUP, DECODE_ITERS) for n in (DECODE_L1, DECODE_L2)]


def default_points() -> list[Point]:
    """Spec 4.3: prefill bs 1 x {512, 2048, 8192}; decode B x {L1, L2} at input 1024."""
    return [_prefill(n) for n in PREFILL_LENS] + [p for b in DECODE_BATCHES for p in _decode_pair(b)]


def parse_points(spec: str) -> list[Point]:
    """`all`, or `;`-separated sections: `prefill`, `decode`, `decode:b1,b32`, `prefill:2048,8192`.

    A decode batch always yields its L1 and L2 points. Duplicates are dropped, first occurrence wins.
    """
    spec = spec.strip()
    if spec == "all":
        return default_points()
    points: list[Point] = []
    for section in spec.split(";"):
        section = section.strip()
        kind, sep, items = section.partition(":")
        if kind not in KINDS:
            raise ValueError(f"point spec {spec!r}: section {section!r} must start with one of {KINDS}")
        if not sep:
            points += [p for p in default_points() if p.kind == kind]
            continue
        for item in items.split(","):
            item = item.strip()
            m = re.fullmatch(r"b([0-9]+)" if kind == "decode" else r"([0-9]+)", item)
            if not m or int(m[1]) < 1:
                form = "b<batch>, e.g. b32" if kind == "decode" else "<input_len>, e.g. 2048"
                raise ValueError(f"point spec {spec!r}: {kind} item {item!r} is not {form}")
            points += _decode_pair(int(m[1])) if kind == "decode" else [_prefill(int(m[1]))]
    return list(dict.fromkeys(points))


def profile_point(spec: str, warmup: int, iters: int) -> Point:
    """The single point a --profile run traces. Decode uses PROFILE_DECODE_OUTPUT_LEN output tokens."""
    shapes = list(dict.fromkeys((p.kind, p.batch, p.input_len) for p in parse_points(spec)))
    if len(shapes) != 1:
        raise ValueError(f"--profile {spec!r} must name exactly one point, got {len(shapes)}: {shapes}")
    if warmup < 0 or iters < 1:
        raise ValueError(f"--profile needs warmup >= 0 and iters >= 1, got {warmup} and {iters}")
    kind, batch, input_len = shapes[0]
    return Point(kind, batch, input_len, 1 if kind == "prefill" else PROFILE_DECODE_OUTPUT_LEN, warmup, iters)


# ---------------------------------------------------------------------------------- engines

class Engine(Protocol):
    name: str            # "vllm" | "fake", recorded in the C3 meta

    def generate(self, prompts: list[list[int]], output_len: int) -> None: ...
    def start_profile(self) -> None: ...
    def stop_profile(self) -> None: ...
    def nvtx_range(self, name: str) -> contextlib.AbstractContextManager: ...


class VllmEngine:
    """The real engine (box only). Builds LLM exactly like `vllm bench latency` does (D2-10)."""

    name = "vllm"

    def __init__(self, engine_args: Sequence[str], profile: bool) -> None:
        if "--model" not in engine_args:
            # Without --model, ModelConfig silently defaults to Qwen/Qwen3-0.6B (D2-10).
            raise ValueError(f"engine args must contain --model, got {list(engine_args)}")
        try:
            from vllm import LLM, SamplingParams
            from vllm.engine.arg_utils import EngineArgs
            from vllm.inputs import TokensPrompt
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except ImportError as exc:
            raise ImportError(f"vllm is not importable ({exc}); the real engine runs only on the box. "
                              "Set TPPROF_FAKE=1 for the fake engine.") from exc
        parser = FlexibleArgumentParser()
        EngineArgs.add_cli_args(parser)
        args = list(engine_args)
        if profile:
            args += ["--profiler-config", PROFILER_CONFIG]
        engine_config = EngineArgs.from_cli_args(parser.parse_args(args))
        self._sampling_params = SamplingParams
        self._tokens_prompt = TokensPrompt
        self.llm = LLM.from_engine_args(engine_config)

    def generate(self, prompts: list[list[int]], output_len: int) -> None:
        sp = self._sampling_params(n=1, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=output_len,
                                   detokenize=True)
        self.llm.generate([self._tokens_prompt(prompt_token_ids=p) for p in prompts], sampling_params=sp,
                          use_tqdm=False)

    def start_profile(self) -> None:
        self.llm.start_profile()

    def stop_profile(self) -> None:
        self.llm.stop_profile()

    @contextlib.contextmanager
    def nvtx_range(self, name: str) -> Iterator[None]:
        import torch

        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


def _spawn_child_main() -> None:
    """Target of the fake engine's spawn child. It only has to be importable and return."""


def _check_spawn(timeout_s: float = 60.0) -> None:
    """Start one spawn child, as vLLM does for its workers (VLLM_WORKER_MULTIPROC_METHOD=spawn).

    The child re-imports the entry module. If that module calls main() without an
    `if __name__ == "__main__":` guard, the child fails to bootstrap, and so does this check (AM32).
    """
    proc = multiprocessing.get_context("spawn").Process(target=_spawn_child_main, name="tpprof-fake-worker")
    proc.start()
    proc.join(timeout_s)
    if proc.exitcode is None:
        proc.kill()
        proc.join()
        raise RuntimeError(f"fake engine: the spawn child did not exit within {timeout_s} s")
    if proc.exitcode != 0:
        raise RuntimeError(f"fake engine: the spawn child exited with code {proc.exitcode}. Under spawn the "
                           "child re-imports the entry module, so an entry point without an "
                           "`if __name__ == \"__main__\":` guard fails like this, as vLLM's workers would.")
    log.info("fake engine: spawn child ok")


def _flag_value(args: Sequence[str], flag: str) -> str | None:
    for i, arg in enumerate(args):
        if arg == flag:
            if i + 1 >= len(args):
                raise ValueError(f"engine args: {flag} has no value")
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


class FakeEngine:
    """Stands in for LLM with TPPROF_FAKE=1: sleeps a plausible time and, in profile mode, writes C4 lines."""

    name = "fake"

    def __init__(self, engine_args: Sequence[str], profile: bool, meta: Mapping[str, object]) -> None:
        args = list(engine_args)
        self.tp = int(_flag_value(args, "--tensor-parallel-size") or 1)
        dp = int(_flag_value(args, "--data-parallel-size") or 1)
        if dp > 1:
            raise ValueError(f"the LLM class rejects data_parallel_size={dp} > 1 (vLLM raises the same); "
                             "offline DP2 is derived from TP1 at half the batch")
        self.enforce_eager = "--enforce-eager" in args
        cc_text = _flag_value(args, "--compilation-config")
        try:
            cc = json.loads(cc_text) if cc_text is not None else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"--compilation-config is not valid JSON: {cc_text!r} ({exc})") from exc
        self.cudagraph_mode = "NONE" if self.enforce_eager else str(cc.get("cudagraph_mode", "FULL_AND_PIECEWISE"))
        self.arm = str(meta.get("arm", "base"))
        self.ar_backend = self._ar_backend("--disable-custom-all-reduce" in args)
        self.profile = profile
        self.time_scale = float(os.environ.get("FAKE_TIME_SCALE", "1.0"))
        log_list = meta.get("call_log")
        self._call_log = log_list if isinstance(log_list, list) else None
        self._profiling = False
        self._steps: list[list[int]] = []
        self._window_batch = 0
        _check_spawn()
        log.info("fake engine: tp=%d arm=%s ar_backend=%s cudagraph_mode=%s profile=%s", self.tp, self.arm,
                 self.ar_backend, self.cudagraph_mode, profile)

    def _ar_backend(self, disable_custom_ar: bool) -> str:
        if self.tp == 1:
            return "none"
        if os.environ.get("VLLM_ALLREDUCE_USE_FLASHINFER") == "0":
            return "nccl" if disable_custom_ar else "custom"
        if (os.environ.get("VLLM_FLASHINFER_ALLREDUCE_BACKEND") == "trtllm"
                or os.environ.get("FAKE_VLLM_NO_MULTICAST") == "1"):
            return "trtllm"
        return "mnnvl"

    def _record(self, *entry: object) -> None:
        if self._call_log is not None:
            self._call_log.append(entry)

    def generate(self, prompts: list[list[int]], output_len: int) -> None:
        batch, input_len = len(prompts), len(prompts[0])
        self._record("generate", batch, input_len, output_len, prompts)
        decode_s = (0.006 / self.tp + 2e-6 * batch) * output_len
        prefill_s = 3e-5 * batch * input_len / self.tp
        time.sleep((decode_s + prefill_s) * self.time_scale)
        if self._profiling:
            self._window_batch = batch
            self._steps.append([batch, batch * input_len, 0, 0])
            self._steps += [[0, 0, batch, batch] for _ in range(output_len - 1)]

    def _require_profiler(self) -> None:
        if not self.profile:
            raise RuntimeError("Profiling is not enabled. Please set --profiler-config to enable profiling.")

    def start_profile(self) -> None:
        self._require_profiler()
        self._record("start_profile")
        self._profiling, self._steps, self._window_batch = True, [], 0

    def stop_profile(self) -> None:
        """Append one C4 line per rank for the steps generated since start_profile."""
        self._require_profiler()
        self._record("stop_profile")
        path = os.environ.get("FAKE_NSYS_EVENTS")
        if path and self._profiling:
            with open(path, "a") as f:
                for rank in range(self.tp):
                    f.write(json.dumps({"pid": os.getpid() * 10 + rank, "device": rank, "rank": rank, "tp": self.tp,
                                        "ar_backend": self.ar_backend, "batch": self._window_batch,
                                        "steps": self._steps}) + "\n")
        self._profiling, self._steps = False, []

    @contextlib.contextmanager
    def nvtx_range(self, name: str) -> Iterator[None]:
        self._record("range_push", name)
        try:
            yield
        finally:
            self._record("range_pop")


def make_engine(engine_args: list[str], profile: bool, meta: dict) -> Engine:
    """TPPROF_FAKE=1 gives the fake engine; otherwise the real LLM class (box only)."""
    if os.environ.get("TPPROF_FAKE") == "1":
        return FakeEngine(engine_args, profile, meta)
    return VllmEngine(engine_args, profile)


# ---------------------------------------------------------------------------------- running points

def _prompts(batch: int, input_len: int, seed: int) -> list[list[int]]:
    return np.random.default_rng(seed).integers(0, PROMPT_VOCAB, size=(batch, input_len)).tolist()


def _is_done(path: str, warn: bool) -> bool:
    if not os.path.exists(path):
        return False
    try:
        load_latency_result(path)
    except (ResultFormatError, OSError, ValueError) as exc:
        if warn:
            log.warning("redoing %s: existing file does not parse (%s)", path, exc)
        return False
    return True


def pending_points(points: Sequence[Point], out_dir: str, warn: bool = True) -> list[Point]:
    """The points whose file is missing or does not parse (AM13 per-point resume)."""
    return [p for p in dict.fromkeys(points) if not _is_done(os.path.join(out_dir, p.filename()), warn)]


def _groups(points: Sequence[Point]) -> list[list[Point]]:
    """Decode points at the same batch and input run together, interleaved; prefill points run alone."""
    groups: dict[tuple, list[Point]] = {}
    for i, p in enumerate(points):
        key = ("decode", p.batch, p.input_len) if p.kind == "decode" else ("prefill", i)
        groups.setdefault(key, []).append(p)
    return list(groups.values())


def _run_group(engine: Engine, group: list[Point], out_dir: str, meta: Mapping[str, object],
               seed: int) -> list[str]:
    """Warmup each point in turn (L1 x3, L2 x3), then measured iterations alternating (L1, L2) x 10."""
    prompts = _prompts(group[0].batch, group[0].input_len, seed)
    t_wall_start, t_mono_start = time.time(), time.monotonic()
    for p in group:
        for _ in range(p.warmup):
            engine.generate(prompts, p.output_len)
    latencies: dict[Point, list[float]] = {p: [] for p in group}
    for i in range(max(p.iters for p in group)):
        for p in group:
            if i < p.iters:
                t0 = time.perf_counter()
                engine.generate(prompts, p.output_len)
                latencies[p].append(time.perf_counter() - t0)
    written = []
    for p in group:
        path = os.path.join(out_dir, p.filename())
        c3 = {"kind": p.kind, "batch": p.batch, "input_len": p.input_len, "output_len": p.output_len,
              "warmup": p.warmup, "iters": p.iters, "config": meta["config"], "arm": meta["arm"],
              "t_wall_start": t_wall_start, "t_mono_start": t_mono_start, "engine": engine.name}
        write_latency_result(path, latencies[p], c3)
        log.info("%s: median %.6f s over %d iterations", p.filename(), float(np.median(latencies[p])), p.iters)
        written.append(path)
    return written


def run_points(engine: Engine, points: Sequence[Point], out_dir: str, meta: dict, seed: int = 0) -> list[str]:
    """Run the points not already on disk and return the paths written by this call.

    Prompts are `default_rng(seed).integers(0, 10000, (B, input_len))`, the same for warmup and
    measured iterations and for both decode lengths. Each point's C3 meta takes config and arm from
    `meta`; t_wall_start / t_mono_start are when its group (the decode pair) started.
    """
    missing = [k for k in ("config", "arm") if k not in meta]
    if missing:
        raise ValueError(f"meta is missing keys {missing}")
    os.makedirs(out_dir, exist_ok=True)
    todo = pending_points(points, out_dir)
    for p in points:
        if p not in todo:
            log.info("%s: already done, skipping", p.filename())
    written = []
    for group in _groups(todo):
        written += _run_group(engine, group, out_dir, meta, seed)
    return written


def run_profile(engine: Engine, point: Point, warmup: int, iters: int,
                sleep_before_stop_s: float = PROFILE_SLEEP_BEFORE_STOP_S) -> None:
    """Spec 4.5: warmup, start_profile, NVTX `tpprof:measure` around K iterations, sleep, stop_profile."""
    prompts = _prompts(point.batch, point.input_len, 0)
    for _ in range(warmup):
        engine.generate(prompts, point.output_len)
    engine.start_profile()
    with engine.nvtx_range(MEASURE_RANGE):
        for _ in range(iters):
            engine.generate(prompts, point.output_len)
    time.sleep(sleep_before_stop_s)
    engine.stop_profile()
    log.info("profiled %s: %d warmup + %d measured iterations", point.filename(), warmup, iters)


# ---------------------------------------------------------------------------------- CLI

def _setup_logging() -> None:
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s [tpprof.offline] %(message)s",
                                               "%m-%d %H:%M:%S"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tpprof.offline",
                                 description="Offline driver; engine args follow `--`.")
    ap.add_argument("--out", required=True, help="directory for the point files")
    ap.add_argument("--meta", required=True, help='JSON object with at least "config" and "arm"')
    ap.add_argument("--points", default=None,
                    help='"all" (default), "prefill", "decode", or e.g. "decode:b1,b32;prefill:2048"')
    ap.add_argument("--profile", metavar="POINTSPEC", default=None, help="trace one point instead of running points")
    ap.add_argument("--profile-warmup", type=int, default=None)
    ap.add_argument("--profile-iters", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, engine_args = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = _parser()
    args = ap.parse_args(ours)
    if args.profile is not None:
        if args.points is not None:
            ap.error("--points and --profile are mutually exclusive")
        if args.profile_warmup is None or args.profile_iters is None:
            ap.error("--profile needs --profile-warmup and --profile-iters")
    try:
        meta = json.loads(args.meta)
        if not isinstance(meta, dict):
            raise ValueError(f"--meta must be a JSON object, got {args.meta!r}")
        missing = [k for k in ("config", "arm") if k not in meta]
        if missing:
            raise ValueError(f"--meta is missing keys {missing}")
        if args.profile is not None:
            point = profile_point(args.profile, args.profile_warmup, args.profile_iters)
            engine = make_engine(engine_args, True, meta)
            sleep_s = PROFILE_SLEEP_BEFORE_STOP_S * getattr(engine, "time_scale", 1.0)
            run_profile(engine, point, args.profile_warmup, args.profile_iters, sleep_before_stop_s=sleep_s)
            return 0
        points = parse_points(args.points or "all")
        if not pending_points(points, args.out, warn=False):
            log.info("all %d points already done in %s; not starting the engine", len(points), args.out)
            return 0
        engine = make_engine(engine_args, False, meta)
        written = run_points(engine, points, args.out, meta)
        log.info("wrote %d of %d points to %s", len(written), len(points), args.out)
        return 0
    except Exception:
        log.exception("offline driver failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
