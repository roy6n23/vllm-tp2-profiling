"""The runner: executes RunSpecs by kind into self-describing, resumable run records (spec 7.3, 7.4, contract C2).

Each spec gets `<results>/raw/<run_id>/` with spec.json, cmd.json (one entry per command), its logs and
outputs, and exactly one of done.json / failed.json on exit.

- Resume: a run whose done.json carries its run_id is skipped. A failed run is skipped unless
  `retry_failed` (its failed.json then becomes failed.prev.json); an interrupted run is always retried.
- Before every run that uses a GPU, both GPUs must be free (< GPU_FREE_MIB); otherwise it fails `gpus_busy`.
- Failures never stop a tier, except a failed preflight: spec 7.5 "any failure stops the run", so every
  later spec of this invocation is skipped with `preflight_failed:<run_id>`. A serve session that fails
  at start makes the later sessions of the same config in that tier skip with `dependency_failed:<run_id>`.
- P1: once the three saturation sessions are done, mu per config (median over seeds of the AM6
  token-timeline rate) gives `raw/rate_grid.json`, written once and never rewritten (AM7). Specs that
  need the grid are skipped with `no_rate_grid` while it is missing.
- The first Ctrl-C stops the current run's processes and marks it failed(`interrupted`); the runner
  then returns and the CLI exits 130. Further Ctrl-Cs during that cleanup are ignored.

Dry run (`dry_run_context`): the fakes in tests/fake_bin, the fake offline engine, time scaled by
FAKE_TIME_SCALE, and a sparse model dir that passes the preflight model gate. Client runs are capped at
DRY_RUN_MAX_PROMPTS prompts (the fake server cannot take thousands of concurrent connections on macOS),
the saturation rate is converted from wall time to fake time (x FAKE_TIME_SCALE) so the rate grid is in
the same units the fake client paces arrivals in, and the offline-driver kinds skip the effective-config
check because the fake engine prints no vLLM log lines.
"""
from __future__ import annotations

import contextlib
import dataclasses
import glob
import http.client
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from tpprof import (client, comm_bench, engine, envcapture, logparse, matrix, monitor, nccltests, preflight, procs,
                    profile, promparse, results, server, stats, vendored)
from tpprof.constants import (DECODE_WARMUP, GPU_FREE_MIB, MODEL_REQUIRED_FILES, MODEL_SHARD_SIZES,
                              MODEL_TENSOR_BYTES_TOTAL, NCCL_EXPECTED, NCCL_TESTS_BIN_DIR, ONLINE_INPUT_LEN,
                              ONLINE_OUTPUT_LEN, PORT_BASE, PREFILL_WARMUP, ROUNDS, SAT_NUM_PROMPTS, SAT_SEEDS,
                              XCHECK)
from tpprof.matrix import RunSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_BIN = os.path.join(REPO_ROOT, "tests", "fake_bin")

DRY_RUN_TIME_SCALE = "0.002"
DRY_RUN_MAX_PROMPTS = 100
# Host gates a Mac or a CI runner cannot pass whatever the fakes say (no nvcc, no FlashInfer, little disk,
# no /dev/shm, a low open-files limit). They are overridden in the dry run and recorded as skipped.
DRY_RUN_SKIP_GATES = ("nvcc", "flashinfer_jit_cache", "disk", "shm", "nofile")
TEST_FAIL_CONFIG_ENV = "TPPROF_TEST_FAIL_CONFIG"      # C5: dry-run hook, FAKE_VLLM_FAIL_START for one config

ALL_GPUS = (0, 1)
MONITOR_TAG_SUFFIX = "-monitor"      # the GPU monitor's TPPROF_RUN_ID, so a server stop's tag sweep spares it
GPU_PRECHECK_TIMEOUT_S = 60
FIRST_START_TIMEOUT_S, START_TIMEOUT_S = 900.0, 300.0
TIMEOUT_FACTOR = 3                  # spec 7.4: a run's timeout is 3 x its estimate (+ the startup allowance)
MIN_TIMEOUT_S = 120.0
LOG_TAIL_LINES = 40
TRACEBACK_CHARS = 4000
STARTUP_LINE_WAIT_S = 10.0          # the "Application startup complete." line may trail the first /health 200

SMOKE_RATE = 4.0                     # plan T16: 8 prompts at rate 4
PC_INPUT_LEN, PC_PREFIX_LEN = 512, 512   # spec 4.7 A-PC: 512 random + 512 shared prefix = 1024 with BOS
DP2RAND_SEED_OFFSET = 500           # the second DP2-rand client's seed: seed + 500 (spec 4.1: different seeds)
START_SKEW_FLAG_S = 1.0             # spec 4.4: DP2-rand start skew above 1 s is flagged
# Kinds that start nothing on a GPU skip the GPU-memory pre-check (spec 7.4: "before the next start");
# the preflight has its own gpu_idle gate with the fix text.
NO_GPU_KINDS = frozenset({"preflight", "envcapture", "tokbench"})
START_FAILURES = ("server_start_failed", "effective_config")
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
# Decode traces: one iteration of PROFILE_DECODE_OUTPUT_LEN tokens; prefill traces: 5 iterations (spec 4.5).
TRACE_WINDOW = {"decode": (DECODE_WARMUP, 1, 200), "prefill": (PREFILL_WARMUP, 5, 5)}   # warmup, iters, min_steps


def _nccl_code(version: str) -> int:
    """"2.30.7" -> 23007, the form nccl-tests reports (NCCL_VERSION_CODE)."""
    major, minor, patch = (int(x) for x in version.split("."))
    return major * 10000 + minor * 100 + patch


NCCL_EXPECTED_CODE = _nccl_code(NCCL_EXPECTED)


@dataclass
class RunContext:
    results_dir: str
    model_dir: str
    dry_run: bool = False
    vllm_bin: str = "vllm"
    nvidia_smi: str = "nvidia-smi"
    torchrun: str = "torchrun"
    python: str = sys.executable
    base_env: dict[str, str] = field(default_factory=dict)      # empty -> os.environ
    port_base: int = PORT_BASE
    rounds: int = ROUNDS
    log: Callable[[str], None] = print
    startup_timeout_first_s: float = FIRST_START_TIMEOUT_S
    startup_timeout_s: float = START_TIMEOUT_S
    skip_gates: tuple[str, ...] = ()
    accept_topology: bool = False


class RunFailure(Exception):
    """A run failed for a known reason; `reason` is a short token, `detail` the evidence."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


# ================================================================ dry-run context


def _write_fake_model(model_dir: str) -> None:
    """Sparse shards of the exact AM26 sizes, minimal JSON files, and an index with the right total_size."""
    os.makedirs(model_dir, exist_ok=True)
    for name, size in MODEL_SHARD_SIZES.items():
        path = os.path.join(model_dir, name)
        if not os.path.exists(path) or os.path.getsize(path) != size:
            with open(path, "wb") as f:
                f.truncate(size)
    for name in MODEL_REQUIRED_FILES:
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("{}\n")
    index = {"metadata": {"total_size": MODEL_TENSOR_BYTES_TOTAL},
             "weight_map": {"lm_head.weight": "model-00004-of-00004.safetensors"}}
    _write_json(os.path.join(model_dir, "model.safetensors.index.json"), index)


def dry_run_context(results_dir: str) -> RunContext:
    """A RunContext that runs the whole matrix against the fakes (spec 7.4 "Dry run", ruling R14)."""
    results_dir = os.path.abspath(results_dir)
    env = dict(os.environ)
    rest = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    env["PATH"] = os.pathsep.join([FAKE_BIN, os.path.dirname(sys.executable), *rest])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (REPO_ROOT, env.get("PYTHONPATH")) if p)
    gpu_state = os.path.join(results_dir, "_gpustate")
    os.makedirs(gpu_state, exist_ok=True)
    env.update({"TPPROF_FAKE": "1", "FAKE_TIME_SCALE": DRY_RUN_TIME_SCALE, "FAKE_GPU_STATE_DIR": gpu_state,
                "TPPROF_NSYS": os.path.join(FAKE_BIN, "nsys")})
    model_dir = os.path.join(results_dir, "_model")
    _write_fake_model(model_dir)
    return RunContext(results_dir=results_dir, model_dir=model_dir, dry_run=True,
                      vllm_bin=os.path.join(FAKE_BIN, "vllm"), nvidia_smi=os.path.join(FAKE_BIN, "nvidia-smi"),
                      torchrun=os.path.join(FAKE_BIN, "torchrun"), python=sys.executable, base_env=env,
                      skip_gates=DRY_RUN_SKIP_GATES)


# ================================================================ small helpers


def _write_json(path: str, doc: object) -> None:
    """Atomic JSON write: a crash or Ctrl-C never leaves half a record."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: str) -> object:
    with open(path) as f:
        return json.load(f)


def _rate_label(rate: float) -> str:
    return "inf" if math.isinf(rate) else repr(float(rate))


def _read_text(path: str) -> str:
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _truncate(text: str, limit: int = TRACEBACK_CHARS) -> str:
    """Head and tail of a long detail: the head says what failed, the tail holds the traceback or log end."""
    if len(text) <= limit:
        return text
    head = limit // 4
    return f"{text[:head]}\n[... {len(text) - limit} characters cut ...]\n{text[-(limit - head):]}"


def _log_tails(run_dir: str) -> dict[str, list[str]]:
    tails = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "*.log"), recursive=True)):
        text = procs.tail(path, LOG_TAIL_LINES)
        if text:
            tails[os.path.relpath(path, run_dir)] = text.splitlines()
    return tails


def _artifacts(run_dir: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(run_dir):
        for name in files:
            rel = os.path.relpath(os.path.join(dirpath, name), run_dir)
            if rel not in ("done.json", "failed.json"):
                out.append(rel)
    return sorted(out)


def _env_overrides(base: Mapping[str, str], env: Mapping[str, str]) -> dict[str, str | None]:
    """What env changes relative to base: new or changed values, and None for removed names."""
    out: dict[str, str | None] = {k: v for k, v in sorted(env.items()) if base.get(k) != v}
    out.update({k: None for k in sorted(base) if k not in env})
    return out


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _slice_csv(src: str, offset: int, dst: str) -> None:
    """dst = the header of src plus the lines src gained after byte `offset` (per-client-run monitor rows)."""
    if not os.path.exists(src):
        return
    with open(src, errors="replace") as f:
        header = f.readline()
        f.seek(max(offset, f.tell()))
        body = f.read()
    with open(dst, "w") as f:
        f.write(header + body)


def _wait_for_line(path: str, line: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with open(path, errors="replace") as f:
            if line in f.read():
                return
        time.sleep(0.1)


def _kept_across_attempts(name: str) -> bool:
    """What a retry resumes from: the spec, the previous failure, the offline points (AM13) with the engine
    checks that vouch for them (effective_config*.json, review C1), and points already rejected."""
    return (name in ("spec.json", "failed.prev.json", "points", "prev-attempt") or name.startswith("rejected-points-")
            or (name.startswith("effective_config") and name.endswith(".json")))


def _set_aside_attempt(run_dir: str) -> None:
    """Before a retry, or a rerun of a run a killed invocation left incomplete: everything else of the previous
    attempt goes to prev-attempt/, so logparse reads only the new attempt's engine logs (logs are appended to),
    sub-runs and monitors start afresh, and cmd.json lists only the new attempt's commands (review I1, M2)."""
    prev = os.path.join(run_dir, "prev-attempt")
    shutil.rmtree(prev, ignore_errors=True)
    stale = [n for n in os.listdir(run_dir) if not _kept_across_attempts(n)]
    if stale:
        os.makedirs(prev)
        for name in stale:
            os.replace(os.path.join(run_dir, name), os.path.join(prev, name))


def _incomplete(run_dir: str) -> bool:
    """A run a killed invocation left behind: started (spec.json), but neither done.json nor failed.json."""
    started, done, failed = (os.path.exists(os.path.join(run_dir, name))
                             for name in ("spec.json", "done.json", "failed.json"))
    return started and not done and not failed


@dataclass
class _SubRun:
    phase: str
    rate: float
    seed: int
    num_prompts: int
    input_len: int = ONLINE_INPUT_LEN
    prefix_len: int = 0


# ================================================================ the runner


class Runner:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.results_dir = os.path.abspath(ctx.results_dir)
        self.model_dir = os.path.abspath(ctx.model_dir) if ctx.model_dir else ""
        self.raw = os.path.join(self.results_dir, "raw")
        self.env = dict(ctx.base_env) if ctx.base_env else dict(os.environ)
        self.retry_failed = False
        self.interrupted = False
        self.last_reason: str | None = None
        self._stop_requested = False
        self._started_configs: set[str] = set()
        self._dead: dict[tuple[str, str], str] = {}       # (tier, config) -> run_id of the session that failed to start
        self._preflight_stop: str | None = None                  # set by a failed preflight: skip everything after it
        self._busy_stop: str | None = None                       # set by a busy GPU pre-check: skip GPU runs after it
        self._stop_signal = "SIGINT"

    # ------------------------------------------------------------ tiers

    def run_tiers(self, tiers: Sequence[str], only_kinds: Sequence[str] = (), retry_failed: bool = False) -> dict:
        """Run the tiers in P0, P1, P2 order.

        Returns {"done": [run_id], "failed": [run_id], "skipped": [{"run_id", "reason"}]} and writes it, with the
        invocation's options, to raw/_last_run.json."""
        wanted = [t for t in matrix.TIERS if t in set(tiers)]
        unknown = sorted(set(tiers) - set(matrix.TIERS))
        if unknown:
            raise ValueError(f"unknown tiers {unknown}; expected a subset of {matrix.TIERS}")
        bad = sorted(set(only_kinds) - set(matrix.KINDS))
        if bad:
            raise ValueError(f"unknown kinds {bad}; expected a subset of {matrix.KINDS}")
        self.retry_failed = retry_failed
        summary: dict[str, list] = {"done": [], "failed": [], "skipped": []}
        os.makedirs(self.raw, exist_ok=True)
        if self.ctx.dry_run:
            _write_json(os.path.join(self.raw, "_dry_run.json"),
                        {"dry_run": True, "t_wall": time.time(), "t_mono": time.monotonic()})
        previous = self._install_sigint()
        t_wall, t_mono = time.time(), time.monotonic()
        self._preflight_stop = None
        self._busy_stop = None
        self._sweep_orphans()
        try:
            for tier in wanted:
                for phase in self._phases(tier):
                    specs, missing = phase()
                    for s in specs:
                        if only_kinds and s.kind not in only_kinds:
                            continue
                        self._record(summary, s.run_id, self._run_or_skip(s))
                        if self.interrupted:
                            return summary
                    for placeholder, kind, reason in missing:
                        if not only_kinds or kind in only_kinds:
                            self._record(summary, placeholder, ("skipped", reason))
        except KeyboardInterrupt:
            self.interrupted = True
            self.ctx.log("interrupted between runs")
        finally:
            self._restore_sigint(previous)
            _write_json(os.path.join(self.raw, "_last_run.json"), {
                **summary, "tiers": wanted, "only": list(only_kinds), "retry_failed": retry_failed,
                "dry_run": self.ctx.dry_run, "interrupted": self.interrupted,
                "t_wall_start": t_wall, "t_mono_start": t_mono,
                "t_wall_end": time.time(), "t_mono_end": time.monotonic()})
        return summary

    def _record(self, summary: dict, run_id: str, outcome: tuple[str, str | None]) -> None:
        status, reason = outcome
        if status == "skipped":
            summary["skipped"].append({"run_id": run_id, "reason": reason})
            self.ctx.log(f"skip   {run_id}: {reason}")
        else:
            summary[status].append(run_id)

    def _run_or_skip(self, spec: RunSpec) -> tuple[str, str | None]:
        if self._preflight_stop is not None:
            return "skipped", self._preflight_stop
        if self._busy_stop is not None and spec.kind not in NO_GPU_KINDS:
            return "skipped", self._busy_stop
        status = self.run_spec(spec)
        if spec.kind == "preflight" and os.path.exists(os.path.join(self.raw, spec.run_id, "failed.json")):
            self._preflight_stop = f"preflight_failed:{spec.run_id}"
        if status == "failed" and self.last_reason == "gpus_busy":
            # the GPUs are held by something this invocation cannot free: stop, rather than wait before each run
            self._busy_stop = f"gpus_busy:{spec.run_id}"
        return status, self.last_reason

    def _sweep_orphans(self) -> None:
        """Kill what a killed invocation left running: processes tagged with a run it left incomplete (review I1)."""
        if not os.path.isdir(self.raw):
            return
        for name in sorted(os.listdir(self.raw)):
            d = os.path.join(self.raw, name)
            if not name.startswith("_") and os.path.isdir(d) and _incomplete(d):
                for tag in (name, f"{name}{MONITOR_TAG_SUFFIX}"):
                    killed = procs.sweep_tagged(tag)
                    if killed:
                        self.ctx.log(f"killed leftover processes {killed} of an earlier invocation, tagged {tag}")

    def _phases(self, tier: str) -> list[Callable[[], tuple[list[RunSpec], list[tuple[str, str, str]]]]]:
        """Each phase returns (specs, [(placeholder id, kind, skip reason)]) when it is reached, so the P1
        sweeps are built only after the saturation sessions ran."""
        if tier == "P0":
            return [lambda: (matrix.p0_specs(self.ctx.rounds), [])]
        if tier == "P1":
            return [lambda: (matrix.p1_sat_specs(), []), self._p1_sweeps]
        return [self._p2]

    def _p1_sweeps(self) -> tuple[list[RunSpec], list[tuple[str, str, str]]]:
        grid = self._ensure_rate_grid()
        if grid is not None:
            return matrix.p1_sweep_specs(grid, self.ctx.rounds), []
        missing = [(f"P1-serve_session-{c}-base-r{r}-sweep", "serve_session", "no_rate_grid")
                   for r in range(1, self.ctx.rounds + 1)
                   for c in matrix.LATIN_SQUARE[(r - 1) % len(matrix.LATIN_SQUARE)]]
        return [], missing

    def _p2(self) -> tuple[list[RunSpec], list[tuple[str, str, str]]]:
        grid = self._ensure_rate_grid()
        specs = matrix.p2_specs(grid, self._fi_backend(), self._multicast())
        missing = []
        if grid is None:
            missing = [(f"P2-serve_session-TP2-{a}-r0-pc", "serve_session", "no_rate_grid") for a in ("base", "PCon")]
            missing.append((f"P2-serve_session-{matrix.DP2RAND}-base-r1-sweep", "serve_session", "no_rate_grid"))
        return specs, missing

    # ------------------------------------------------------------ SIGINT

    def _install_sigint(self) -> dict | None:
        """Ctrl-C, SIGTERM (a kill) and SIGHUP (a closed tmux window) all stop the current run cleanly: its
        processes are stopped, it is marked failed(interrupted), and a rerun resumes (review I1)."""
        if threading.current_thread() is not threading.main_thread():
            return None

        def handler(signum: int, frame: object) -> None:
            if self._stop_requested:
                self.ctx.log("already stopping; waiting for the current run's processes to exit")
                return
            self._stop_requested = True
            self._stop_signal = signal.Signals(signum).name
            raise KeyboardInterrupt

        return {sig: signal.signal(sig, handler) for sig in STOP_SIGNALS}

    def _restore_sigint(self, previous: dict | None) -> None:
        for sig, handler in (previous or {}).items():
            signal.signal(sig, handler)

    # ------------------------------------------------------------ one spec

    def run_spec(self, spec: RunSpec) -> str:
        """Run one spec into raw/<run_id>; returns "done", "failed" or "skipped" (reason in self.last_reason)."""
        self.last_reason = None
        run_dir = os.path.join(self.raw, spec.run_id)
        done_path, failed_path = os.path.join(run_dir, "done.json"), os.path.join(run_dir, "failed.json")
        if os.path.exists(done_path):
            try:
                if _read_json(done_path).get("run_id") == spec.run_id:
                    self.last_reason = "already_done"
                    return "skipped"
            except (OSError, ValueError, AttributeError):
                pass
            os.remove(done_path)                     # unreadable or foreign: redo the run
        if os.path.exists(failed_path):
            try:
                prev = _read_json(failed_path)
                prev_reason = str(prev.get("reason"))
            except (OSError, ValueError, AttributeError):
                prev_reason = "unreadable failed.json"
            # A failed preflight always reruns: it takes a minute, and the rerun may carry the gate options
            # (--accept-topology, --skip-gate) that RUN_ON_GPU.md prescribes for it (review I3).
            if not (self.retry_failed or prev_reason == "interrupted" or spec.kind == "preflight"):
                if spec.kind == "serve_session" and prev_reason in START_FAILURES:
                    self._dead[(spec.tier, spec.config)] = spec.run_id     # its dependents stay skipped too
                self.last_reason = f"previously_failed:{prev_reason}"
                return "skipped"
            os.replace(failed_path, os.path.join(run_dir, "failed.prev.json"))
            _set_aside_attempt(run_dir)
        elif os.path.isdir(run_dir) and _incomplete(run_dir):
            self._sweep(spec)
            self.ctx.log(f"       {spec.run_id} was left incomplete by an earlier invocation; rerunning it")
            _set_aside_attempt(run_dir)
        if spec.kind == "serve_session" and (spec.tier, spec.config) in self._dead:
            self.last_reason = f"dependency_failed:{self._dead[(spec.tier, spec.config)]}"
            return "skipped"
        skip = self._precondition_skip(spec)
        if skip is not None:
            self.last_reason = skip
            return "skipped"

        os.makedirs(run_dir, exist_ok=True)
        _write_json(os.path.join(run_dir, "spec.json"), spec.to_dict())
        if not os.path.exists(os.path.join(run_dir, "cmd.json")):
            _write_json(os.path.join(run_dir, "cmd.json"), [])         # entries are appended per command
        self.ctx.log(f"run    {spec.run_id}")
        t0 = time.monotonic()
        try:
            if spec.kind not in NO_GPU_KINDS and not procs.wait_gpu_memory_free(
                    ALL_GPUS, timeout_s=GPU_PRECHECK_TIMEOUT_S, nvidia_smi=self.ctx.nvidia_smi, env=self.env):
                raise RunFailure("gpus_busy", self._gpu_state())
            self._handler(spec.kind)(spec, run_dir)
        except KeyboardInterrupt:
            self.interrupted = True
            self._sweep(spec)
            self._fail(spec, run_dir, "interrupted", f"stopped by {self._stop_signal}")
            return "failed"
        except RunFailure as e:
            if spec.kind == "serve_session" and e.reason in START_FAILURES:
                self._dead[(spec.tier, spec.config)] = spec.run_id
            self._fail(spec, run_dir, e.reason, e.detail)
            return "failed"
        except Exception as e:
            self._sweep(spec)
            self._fail(spec, run_dir, type(e).__name__, f"{e}\n{traceback.format_exc()}")
            return "failed"
        duration = time.monotonic() - t0
        _write_json(done_path, {"run_id": spec.run_id, "status": "done", "duration_s": duration,
                                "artifacts": _artifacts(run_dir)})
        self.ctx.log(f"done   {spec.run_id} ({duration:.1f} s)")
        return "done"

    def _sweep(self, spec: RunSpec) -> None:
        """Safety net after an interrupt or an unexpected error: kill anything still carrying the run's tags
        (AM32), including a monitor whose stop the interrupt cut short."""
        for tag in (spec.run_id, f"{spec.run_id}{MONITOR_TAG_SUFFIX}"):
            killed = procs.sweep_tagged(tag)
            if killed:
                self.ctx.log(f"       killed leftover processes {killed} tagged {tag}")

    def _fail(self, spec: RunSpec, run_dir: str, reason: str, detail: str) -> None:
        self.last_reason = reason
        _write_json(os.path.join(run_dir, "failed.json"), {
            "run_id": spec.run_id, "status": "failed", "reason": reason, "detail": _truncate(detail),
            "log_tails": _log_tails(run_dir)})
        first = detail.strip().splitlines()[0] if detail.strip() else ""
        self.ctx.log(f"failed {spec.run_id}: {reason}{': ' + first[:200] if first else ''}")

    def _gpu_state(self) -> str:
        try:
            used = procs.gpu_memory_used(self.ctx.nvidia_smi, self.env)
            apps = procs.gpu_processes(self.ctx.nvidia_smi, self.env)
        except (RuntimeError, OSError) as e:
            return f"nvidia-smi failed: {e}"
        return (f"GPU memory used after {GPU_PRECHECK_TIMEOUT_S} s: {used} MiB (need < "
                f"{GPU_FREE_MIB} MiB on {list(ALL_GPUS)}); compute apps: {apps}")

    def _precondition_skip(self, spec: RunSpec) -> str | None:
        if spec.kind == "comm_m4" and self.ctx.dry_run and not os.path.exists(self._nccl_tests_bin()):
            return "dry_run:no_nccl_tests"          # no fake all_reduce_perf; on the box a missing build fails
        return None

    def _handler(self, kind: str) -> Callable[[RunSpec, str], None]:
        return {"preflight": self._preflight, "envcapture": self._envcapture, "smoke": self._smoke,
                "comm_m1": self._comm, "comm_m2": self._comm, "comm_m3": self._comm, "comm_m4": self._comm_m4,
                "offline": self._offline, "bench_latency_xcheck": self._xcheck, "trace": self._trace,
                "serve_session": self._serve_session, "tokbench": self._tokbench}[kind]

    # ------------------------------------------------------------ commands, monitors, timeouts

    def _append_cmd(self, run_dir: str, entry: dict) -> None:
        path = os.path.join(run_dir, "cmd.json")
        cmds = _read_json(path) if os.path.exists(path) else []
        cmds.append(entry)
        _write_json(path, cmds)

    def _run_cmd(self, spec: RunSpec, run_dir: str, argv: Sequence[str], env: Mapping[str, str], log_name: str,
                 timeout_s: float) -> procs.Outcome:
        """procs.run (own process group, log file, timeout) plus a cmd.json entry; fails on timeout or exit != 0."""
        argv = [str(a) for a in argv]
        log = os.path.join(run_dir, log_name)
        t_wall, t_mono = time.time(), time.monotonic()
        out = None
        try:
            out = procs.run(argv, env, log, spec.run_id, timeout_s)
        finally:
            self._append_cmd(run_dir, {
                "argv": argv, "env_overrides": _env_overrides(self.env, env), "cwd": None,
                "t_wall_start": t_wall, "t_mono_start": t_mono,
                "t_wall_end": out.t_wall_end if out else time.time(),
                "t_mono_end": out.t_mono_end if out else time.monotonic(),
                "exit_code": out.exit_code if out else None, "timed_out": out.timed_out if out else False,
                "log": log_name})
        if out.timed_out:
            raise RunFailure("timeout", f"{argv[0]} ... still running after {timeout_s:.0f} s; see {log_name}\n"
                                        f"{procs.tail(log, LOG_TAIL_LINES)}")
        if out.exit_code != 0:
            raise RunFailure("exit_code", f"{os.path.basename(argv[0])} exited {out.exit_code}; see {log_name}\n"
                                          f"{procs.tail(log, LOG_TAIL_LINES)}")
        return out

    @contextlib.contextmanager
    def _monitors(self, spec: RunSpec, run_dir: str) -> Iterator[None]:
        """nvidia-smi every 200 ms into gpu.csv and per-process CPU into cpu.csv, for this process tree.

        The monitor carries its own run tag, so the run's tag sweep after a server stop leaves it alone."""
        gpu = monitor.GpuMonitor(os.path.join(run_dir, "gpu.csv"), f"{spec.run_id}{MONITOR_TAG_SUFFIX}",
                                 nvidia_smi=self.ctx.nvidia_smi, env=self.env)
        cpu = monitor.CpuSampler(os.path.join(run_dir, "cpu.csv"), lambda: [os.getpid()])
        with gpu, cpu:
            yield

    def _mu_rps(self) -> dict[str, float] | None:
        path = os.path.join(self.raw, "rate_grid.json")
        try:
            return dict(_read_json(path)["mu_rps"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _estimate_s(self, spec: RunSpec) -> float:
        return matrix.estimate([spec], self._mu_rps())[0].minutes * 60

    def _startup_timeout(self, config: str) -> float:
        first = config not in self._started_configs
        self._started_configs.add(config)
        return self.ctx.startup_timeout_first_s if first else self.ctx.startup_timeout_s

    def _timeout(self, spec: RunSpec, engine_start: bool) -> float:
        t = max(MIN_TIMEOUT_S, TIMEOUT_FACTOR * self._estimate_s(spec))
        return t + (self._startup_timeout(spec.config) if engine_start else 0.0)

    def _engine_configs(self, spec: RunSpec) -> list[engine.EngineConfig]:
        names = matrix.DP2RAND_ENGINES if spec.config == matrix.DP2RAND else (spec.config,)
        return [engine.arm_config(name, spec.arm) for name in names]

    def _preflight_ctx(self) -> preflight.PreflightContext:
        return preflight.PreflightContext(
            nvidia_smi=self.ctx.nvidia_smi, vllm_bin=self.ctx.vllm_bin, model_dir=self.model_dir,
            results_dir=self.results_dir, shm_path=self.results_dir if self.ctx.dry_run else "/dev/shm",
            on_box=not self.ctx.dry_run, accept_topology=self.ctx.accept_topology,
            skip_gates=tuple(self.ctx.skip_gates), env=self.env)

    # ------------------------------------------------------------ effective config

    def _check_engine_logs(self, spec: RunSpec, run_dir: str, logs: Sequence[tuple[str, str]], serve: bool,
                           out_name: str = "effective_config.json", upto: Mapping[str, int] | None = None
                           ) -> list[str]:
        """logparse every engine log against its expectation; writes effective_config.json and returns violations.

        upto maps a log path to the byte size to read up to: a server's log as it was before the runner stopped
        it, so lines the stop itself provokes are not charged to the run (review I5)."""
        texts, violations = [], []
        for name, path in logs:
            with open(path, "rb") as f:
                data = f.read()
            if upto is not None and path in upto:
                data = data[:upto[path]]
            text = data.decode(errors="replace")
            texts.append(text)
            v = logparse.check(logparse.parse_engine_log(text), logparse.expectation_for(name, spec.arm, serve))
            violations += [f"{name}: {x}" for x in v] if len(logs) > 1 else v
        doc = dataclasses.asdict(logparse.parse_engine_log("\n".join(texts)))
        doc["violations"] = violations
        _write_json(os.path.join(run_dir, out_name), doc)
        return violations

    def _check_offline_log(self, spec: RunSpec, run_dir: str, log_path: str, out_name: str = "effective_config.json"
                           ) -> None:
        """The offline driver's engine log (serve=False). The dry run's fake engine prints no vLLM lines."""
        if self.ctx.dry_run:
            return
        v = self._check_engine_logs(spec, run_dir, [(spec.config, log_path)], serve=False, out_name=out_name)
        if v:
            raise RunFailure("effective_config", "; ".join(v))

    # ------------------------------------------------------------ kinds without an engine

    def _preflight(self, spec: RunSpec, run_dir: str) -> None:
        checks = preflight.full_checks(self._preflight_ctx())
        preflight.write_report(checks, os.path.join(run_dir, "preflight.json"))
        ok, text = preflight.verdict(checks)
        with open(os.path.join(run_dir, "preflight.log"), "w") as f:
            f.write(text + "\n")
        self.ctx.log(text)
        if not ok:
            raise RunFailure("preflight_failed", text.splitlines()[-1] + "\n" + text)

    def _envcapture(self, spec: RunSpec, run_dir: str) -> None:
        doc = envcapture.capture_env(self._preflight_ctx())
        _write_json(os.path.join(run_dir, "env.json"), doc)
        for key, name in (("topo_m", "topo.txt"), ("nvlink_s", "nvlink.txt")):
            if doc.get(key):
                with open(os.path.join(run_dir, name), "w") as f:
                    f.write(doc[key])

    def _tokbench(self, spec: RunSpec, run_dir: str) -> None:
        argv = [self.ctx.python, "-m", "tpprof.tokbench", "--model-dir", self.model_dir,
                "--input-len", str(ONLINE_INPUT_LEN), "--output-len", str(ONLINE_OUTPUT_LEN),
                "--out", os.path.join(run_dir, "tokbench.json"), *(["--synthetic"] if self.ctx.dry_run else [])]
        env = {**self.env, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        with self._monitors(spec, run_dir):
            self._run_cmd(spec, run_dir, argv, env, "tokbench.log", self._timeout(spec, False))

    def _comm_env(self) -> dict[str, str]:
        return {**self.env, "CUDA_VISIBLE_DEVICES": ",".join(map(str, ALL_GPUS))}

    def _comm(self, spec: RunSpec, run_dir: str) -> None:
        if spec.kind == "comm_m3":
            self._comm_m3(spec, run_dir)
            return
        env = self._comm_env()
        if spec.kind == "comm_m1":
            out = os.path.join(run_dir, "m1.json")
            argv, parse = vendored.m1_argv(out, self.ctx.torchrun), vendored.parse_m1
        else:
            out = os.path.join(run_dir, "m2.jsonl")
            argv, parse = vendored.m2_argv(out, self.ctx.torchrun), vendored.parse_m2
        with self._monitors(spec, run_dir):
            self._run_cmd(spec, run_dir, argv, env, f"{spec.kind}.log", self._timeout(spec, False))
        self._write_rows(run_dir, parse(out))

    def _comm_m3(self, spec: RunSpec, run_dir: str) -> None:
        """M3 and its NCCL verification (spec 4.6, review I6): the timed sweep, then a short diagnostic run of
        the same variant under NCCL_DEBUG=INFO, whose TUNING lines must show the forced algorithm and protocol;
        the rows are written only then. A variant NCCL rejects is recorded as unsupported, and an NVLS bind
        failure at init is retried once with NCCL_NVLS_ENABLE=0 (nvls itself cannot be: it is unsupported)."""
        variant = comm_bench.parse_variant(str(spec.p("variant")))
        algo, proto = variant
        label = comm_bench.variant_label(variant)
        env = {**self._comm_env(), **comm_bench.variant_env(*variant)}   # the label and the env of one variant
        out = os.path.join(run_dir, "m3.jsonl")
        argv = comm_bench.torchrun_argv(out, str(spec.p("modes")), variant, self.ctx.torchrun)
        verify_path = os.path.join(run_dir, "nccl_verify.json")
        record: dict = {"variant": label, "requested": {"algo": algo, "proto": proto}, "unsupported": False,
                        "nvls_disabled_retry": False}
        timeout = self._timeout(spec, False)
        debug_dir = os.path.join(run_dir, "nccl-debug")
        with self._monitors(spec, run_dir):
            try:
                self._run_cmd(spec, run_dir, argv, env, "comm_m3.log", timeout)
            except RunFailure:
                text = _read_text(os.path.join(run_dir, "comm_m3.log"))
                nvls_failed = comm_bench.NVLS_BIND_FAILED in text
                if comm_bench.NCCL_UNSUPPORTED in text or (nvls_failed and algo == "nvls"):
                    record["unsupported"] = True
                    _write_json(verify_path, record)
                    why = comm_bench.NVLS_BIND_FAILED if nvls_failed else comm_bench.NCCL_UNSUPPORTED
                    raise RunFailure("nccl_unsupported", f'{label}: NCCL reports "{why}"; recorded as unsupported '
                                                         "(spec 4.6)") from None
                if not nvls_failed:
                    raise
                self.ctx.log(f"       {label}: NVLS multicast memory could not be bound; rerun with NCCL_NVLS_ENABLE=0")
                env["NCCL_NVLS_ENABLE"] = "0"
                record["nvls_disabled_retry"] = True
                self._run_cmd(spec, run_dir, argv, env, "comm_m3-nvls-off.log", timeout)
            os.makedirs(debug_dir, exist_ok=True)
            diag_argv = comm_bench.torchrun_argv(os.path.join(run_dir, "m3-diagnostic.jsonl"), "eager", variant,
                                                 self.ctx.torchrun, iters=1, warmup=1)
            self._run_cmd(spec, run_dir, diag_argv, {**env, **comm_bench.variant_env(*variant, debug_dir=debug_dir)},
                          "comm_m3-diagnostic.log", timeout)
        text = "".join(_read_text(p) for p in sorted(glob.glob(os.path.join(debug_dir, "*"))))
        tuning = [t for t in comm_bench.parse_tuning_lines(text) if t["func"] == "AllReduce"]
        algos, protos = sorted({t["algo"].upper() for t in tuning}), sorted({t["proto"].upper() for t in tuning})
        mismatch = []
        if algo is not None and algos != [algo.upper()]:
            mismatch.append(f"algorithms {algos} where {algo.upper()} was forced")
        if proto is not None and protos != [proto.upper()]:
            mismatch.append(f"protocols {protos} where {proto.upper()} was forced")
        forced = algo is not None or proto is not None
        record.update(observed={"algos": algos, "protos": protos}, tuning_lines=len(tuning),
                      nccl_version=comm_bench.parse_nccl_version(text),
                      nvls_support=comm_bench.parse_nvls_support(text),
                      verified=(not mismatch) if tuning or not forced else None)
        _write_json(verify_path, record)
        if mismatch:
            raise RunFailure("nccl_variant_mismatch", f"{label}: NCCL's TUNING lines show {'; '.join(mismatch)}, so "
                                                      "the rows would carry the wrong label; none written")
        if record["verified"] is None:
            self.ctx.log(f"       warning: {label}: NCCL logged no AllReduce TUNING lines; the variant is unverified")
        self._write_rows(run_dir, comm_bench.load_rows(out))

    def _nccl_tests_bin(self) -> str:
        return os.path.join(NCCL_TESTS_BIN_DIR, "all_reduce_perf")

    def _comm_m4(self, spec: RunSpec, run_dir: str) -> None:
        if not os.path.exists(self._nccl_tests_bin()):
            raise RunFailure("nccl_tests_missing", f"{self._nccl_tests_bin()} does not exist; run "
                                                   "scripts/bootstrap_box.sh (nccl-tests build)")
        env = self._comm_env()
        for key, value in nccltests.run_env().items():          # prepend, keeping the image's library path
            env[key] = os.pathsep.join(p for p in (value, env.get(key)) if p)
        rows = []
        with self._monitors(spec, run_dir):
            for graph in (False, True):
                mode = "graph" if graph else "eager"
                out = os.path.join(run_dir, f"m4-{mode}.json")
                self._run_cmd(spec, run_dir, nccltests.run_argv(NCCL_TESTS_BIN_DIR, out, graph), env,
                              f"comm_m4-{mode}.log", self._timeout(spec, False) / 2)
                rows += nccltests.parse_json(out)
        wrong = sorted({r["nccl_version"] for r in rows} - {NCCL_EXPECTED_CODE})
        self._write_rows(run_dir, rows)
        if wrong:
            raise RunFailure("nccl_version", f"all_reduce_perf loaded NCCL {wrong}, expected {NCCL_EXPECTED_CODE} "
                                             f"({NCCL_EXPECTED}); a system libnccl was measured (D7-11)")

    def _write_rows(self, run_dir: str, rows: Sequence[dict]) -> None:
        with open(os.path.join(run_dir, "comm_rows.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------ offline-driver kinds

    def _offline_argv(self, cfg: engine.EngineConfig, run_dir: str, extra: Sequence[str],
                      meta: Mapping[str, object] | None = None) -> list[str]:
        meta = {"config": cfg.name, "arm": cfg.arm, **(meta or {})}
        return [self.ctx.python, "-m", "tpprof.offline", "--out", os.path.join(run_dir, "points"),
                "--meta", json.dumps(meta, sort_keys=True), *extra, "--", *cfg.offline_args(self.model_dir)]

    def _offline(self, spec: RunSpec, run_dir: str) -> None:
        [cfg] = self._engine_configs(spec)
        with self._monitors(spec, run_dir):
            self._offline_points(spec, run_dir, cfg, str(spec.p("points")), cfg.environment(self.env), "offline.log",
                                 "effective_config.json")

    def _offline_points(self, spec: RunSpec, run_dir: str, cfg: engine.EngineConfig, points_spec: str,
                        env: Mapping[str, str], log_name: str, eff_name: str,
                        meta: Mapping[str, object] | None = None) -> None:
        """The offline driver for points_spec into run_dir/points, keeping only points whose engine passed the
        effective-config check (review C1).

        The driver resumes per point (AM13) and starts no engine when every point exists, so the check is tied
        to the attempt that wrote the points: each attempt's engine log is checked, also when the driver fails,
        and a violation moves points/ aside. A retry whose points all exist, with a clean check on record, does
        not start the driver again."""
        from tpprof.offline import parse_points, pending_points

        points_dir = os.path.join(run_dir, "points")
        points = parse_points(points_spec)
        if not self.ctx.dry_run and os.path.isdir(points_dir) and os.listdir(points_dir):
            eff = self._effective_config(run_dir, eff_name)
            if eff is None or eff.get("violations"):
                self._quarantine_points(run_dir, f"no clean engine check on record ({eff_name})")
            elif not pending_points(points, points_dir, warn=False):
                self.ctx.log(f"       every point exists and its engine passed the check ({eff_name}); not rerun")
                return
        argv = self._offline_argv(cfg, run_dir, ["--points", points_spec], meta)
        log_path = os.path.join(run_dir, log_name)
        try:
            self._run_cmd(spec, run_dir, argv, env, log_name, self._timeout(spec, True))
        except RunFailure as e:
            violations = self._vet_engine_log(spec, run_dir, log_path, eff_name)
            if violations:
                raise RunFailure("effective_config", "; ".join(violations) + f"\n(the driver also failed: "
                                                                             f"{e.reason}: {e.detail})") from None
            raise
        except BaseException:
            self._vet_engine_log(spec, run_dir, log_path, eff_name)
            raise
        violations = self._vet_engine_log(spec, run_dir, log_path, eff_name)
        if violations:
            raise RunFailure("effective_config", "; ".join(violations))
        if violations is None:                         # exited 0 without starting an engine
            eff = self._effective_config(run_dir, eff_name)
            if eff is None or eff.get("violations"):
                self._quarantine_points(run_dir, f"{log_name}: the driver started no engine")
                raise RunFailure("effective_config", f"{log_name}: the driver started no engine and no clean "
                                                     f"engine check is on record ({eff_name}); points set aside")
        missing = [p.filename() for p in points if not os.path.exists(os.path.join(points_dir, p.filename()))]
        if missing:
            raise RunFailure("missing_points", f"the driver exited 0 but wrote no {', '.join(missing)}")

    def _vet_engine_log(self, spec: RunSpec, run_dir: str, log_path: str, eff_name: str) -> list[str] | None:
        """Check an offline driver attempt's engine log (serve=False) into eff_name: its violations, or None
        when no engine started in it. Violations move points/ aside (review C1). The dry run's fake engine
        prints no vLLM lines, so the dry run checks nothing."""
        if self.ctx.dry_run:
            return []
        try:
            with open(log_path, errors="replace") as f:
                if logparse.parse_engine_log(f.read()).vllm_version is None:
                    return None
        except OSError:
            return None
        violations = self._check_engine_logs(spec, run_dir, [(spec.config, log_path)], serve=False,
                                             out_name=eff_name)
        if violations:
            self._quarantine_points(run_dir, f"{eff_name}: {violations[0]}")
        return violations

    def _effective_config(self, run_dir: str, eff_name: str) -> dict | None:
        try:
            doc = _read_json(os.path.join(run_dir, eff_name))
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def _quarantine_points(self, run_dir: str, why: str) -> None:
        """Move points/ to rejected-points-<n>/: kept for inspection, never read as data."""
        src = os.path.join(run_dir, "points")
        if not os.path.isdir(src) or not os.listdir(src):
            return
        n = 1
        while os.path.exists(os.path.join(run_dir, f"rejected-points-{n}")):
            n += 1
        os.replace(src, os.path.join(run_dir, f"rejected-points-{n}"))
        self.ctx.log(f"       points set aside as rejected-points-{n}: {why}")

    def _xcheck(self, spec: RunSpec, run_dir: str) -> None:
        [cfg] = self._engine_configs(spec)
        x = XCHECK
        out = os.path.join(run_dir, "bench_latency.json")
        argv = cfg.bench_latency_argv(self.model_dir, x["batch"], x["input_len"], x["output_len"], x["warmup"],
                                      x["iters"], out, self.ctx.vllm_bin)
        with self._monitors(spec, run_dir):
            self._run_cmd(spec, run_dir, argv, cfg.environment(self.env), "bench_latency.log",
                          self._timeout(spec, True))
        self._check_offline_log(spec, run_dir, os.path.join(run_dir, "bench_latency.log"))
        results.load_latency_result(out)                      # raises ResultFormatError naming what is missing

    def _untraced_step_s(self, spec: RunSpec) -> float | None:
        """The matching offline run's median step (decode: the L1/L2 step; prefill: the point's median), if done."""
        from tpprof.offline import parse_points

        point = parse_points(str(spec.p("points")))[0]
        for d in sorted(glob.glob(os.path.join(self.raw, f"*-offline-{spec.config}-{spec.arm}-*"))):
            if not os.path.exists(os.path.join(d, "done.json")):
                continue
            try:
                rows = results.offline_rows(d)
            except (results.ResultFormatError, OSError, ValueError):
                continue
            for r in rows:
                if (r["kind"], r["batch"], r["input_len"]) == (point.kind, point.batch, point.input_len):
                    value = r["step_s"] if point.kind == "decode" else r["median_s"]
                    if value is not None and math.isfinite(value):
                        return float(value)
        return None

    def _trace(self, spec: RunSpec, run_dir: str) -> None:
        [cfg] = self._engine_configs(spec)
        from tpprof.offline import parse_points

        kind = parse_points(str(spec.p("points")))[0].kind
        warmup, iters, min_steps = TRACE_WINDOW[kind]
        target = self._offline_argv(cfg, run_dir, ["--profile", str(spec.p("points")),
                                                   "--profile-warmup", str(warmup), "--profile-iters", str(iters)])
        timeout = self._timeout(spec, True)
        with self._monitors(spec, run_dir):
            outcome = profile.run_trace(run_dir, spec.run_id, target, cfg.environment(self.env), cfg.tp, min_steps,
                                        timeout, self._untraced_step_s(spec))
        for attempt in outcome.attempts:
            self._append_cmd(run_dir, {**attempt, "log": os.path.relpath(attempt["log"], run_dir)})
        if outcome.summary is not None:
            _write_json(os.path.join(run_dir, "trace_summary.json"), outcome.summary)
        if not outcome.ok:
            raise RunFailure("trace", outcome.reason)
        good = [a for a in outcome.attempts if a.get("gate_ok")]
        if good:
            self._check_offline_log(spec, run_dir, good[-1]["log"])
        self.ctx.log(f"       {outcome.reason}")

    # ------------------------------------------------------------ servers

    def _server_env(self, spec: RunSpec, cfg: engine.EngineConfig) -> dict[str, str]:
        env = dict(self.env)
        if self.ctx.dry_run and env.get(TEST_FAIL_CONFIG_ENV) in (cfg.name, spec.config):
            env["FAKE_VLLM_FAIL_START"] = "1"
        return env

    def _start_servers(self, spec: RunSpec, run_dir: str) -> list[server.ServerHandle]:
        """Start the spec's servers (DP2-rand: two, on port_base and port_base + 1); the caller stops them."""
        handles: list[server.ServerHandle] = []
        timeout = self._startup_timeout(spec.config)
        for i, cfg in enumerate(self._engine_configs(spec)):
            port = self.ctx.port_base + i
            env = self._server_env(spec, cfg)
            t_wall, t_mono = time.time(), time.monotonic()
            entry = {"argv": cfg.serve_argv(self.model_dir, port, self.ctx.vllm_bin),
                     "env_overrides": _env_overrides(self.env, cfg.environment(env)), "cwd": None,
                     "t_wall_start": t_wall, "t_mono_start": t_mono, "log": f"server-{port}.log"}
            try:
                handles.append(server.start_server(cfg, self.model_dir, port, run_dir, env, spec.run_id, timeout,
                                                   self.ctx.vllm_bin))
            except server.ServerStartError as e:
                self._append_cmd(run_dir, {**entry, "t_wall_end": time.time(), "t_mono_end": time.monotonic(),
                                           "exit_code": None, "timed_out": False, "ready_s": None})
                raise RunFailure("server_start_failed", str(e)) from None
            self._append_cmd(run_dir, {**entry, "t_wall_end": None, "t_mono_end": None, "exit_code": None,
                                       "timed_out": False, "ready_s": handles[-1].ready_s})
        return handles

    def _stop_servers(self, run_dir: str, handles: Sequence[server.ServerHandle]) -> None:
        if not handles:
            return
        freed = server.stop_servers(handles, nvidia_smi=self.ctx.nvidia_smi, env=self.env)
        path = os.path.join(run_dir, "cmd.json")
        cmds = _read_json(path) if os.path.exists(path) else []
        by_log = {f"server-{h.port}.log": h for h in handles}
        for c in cmds:                                   # close the servers' cmd.json entries
            h = by_log.get(c.get("log"))
            if h is not None and c.get("t_mono_end") is None:
                c.update(t_wall_end=time.time(), t_mono_end=time.monotonic(), exit_code=h.proc.popen.returncode)
        _write_json(path, cmds)
        if not freed:
            self.ctx.log(f"warning: GPU memory not released within the wait after stopping {len(handles)} server(s)")

    def _server_logs(self, handles: Sequence[server.ServerHandle]) -> list[tuple[str, str]]:
        return [(h.cfg.name, h.log_path) for h in handles]

    def _check_server_logs(self, spec: RunSpec, run_dir: str, handles: Sequence[server.ServerHandle],
                           upto: Mapping[str, int] | None = None) -> None:
        v = self._check_engine_logs(spec, run_dir, self._server_logs(handles), serve=True, upto=upto)
        if v:
            raise RunFailure("effective_config", "; ".join(v))

    def _dead_server(self, handles: Sequence[server.ServerHandle]) -> None:
        for h in handles:
            code = h.proc.popen.poll()
            if code is not None:
                raise RunFailure("server_died", f"{h.cfg.name} on port {h.port} exited {code}\n"
                                                f"{procs.tail(h.log_path, LOG_TAIL_LINES)}")

    def _scrape(self, handles: Sequence[server.ServerHandle]) -> str:
        """/metrics of every server; concatenated, so promparse's sums run over both DP2-rand servers."""
        return "".join(f"# tpprof server {h.base_url}\n" + server.scrape_metrics(h.base_url) for h in handles)

    # ------------------------------------------------------------ smoke

    def _smoke(self, spec: RunSpec, run_dir: str) -> None:
        [cfg] = self._engine_configs(spec)
        handles: list[server.ServerHandle] = []
        with self._monitors(spec, run_dir):
            try:
                handles = self._start_servers(spec, run_dir)
                _wait_for_line(handles[0].log_path, logparse.STARTUP_COMPLETE, STARTUP_LINE_WAIT_S)
                self._check_server_logs(spec, run_dir, handles)
                sub = _SubRun("smoke", SMOKE_RATE, 0, int(spec.p("prompts", 8)))
                sub_dir = os.path.join(run_dir, "sub-0")
                os.makedirs(sub_dir, exist_ok=True)
                violations, _ = self._client_run(spec, run_dir, sub_dir, handles, sub, 0, ["result.json"],
                                                 self._timeout(spec, False))
                self._dead_server(handles)
            finally:
                before_stop = {h.log_path: _file_size(h.log_path) for h in handles}
                self._stop_servers(run_dir, handles)
            self._check_server_logs(spec, run_dir, handles, before_stop)
            if violations:
                raise RunFailure("invalid_smoke", "; ".join(violations))
            if spec.p("gpu1_check"):
                # AM14: the same TP1 decode bs-1 point on GPU1, to compare with GPU0 (DP2's second rank)
                env = {**cfg.environment(self.env), "CUDA_VISIBLE_DEVICES": "1"}
                self._offline_points(spec, run_dir, cfg, "decode:b1", env, "offline-gpu1.log",
                                     "effective_config_gpu1.json", {"gpu": 1})

    # ------------------------------------------------------------ serve sessions

    def _sub_runs(self, spec: RunSpec) -> list[_SubRun]:
        phase = spec.p("phase")
        if phase == "sat":
            n = int(spec.p("num_prompts", SAT_NUM_PROMPTS))
            subs = [_SubRun("sat", math.inf, int(seed), n) for seed in spec.p("seeds")]
        elif phase == "sweep":
            base = int(spec.p("seed_base"))
            subs = [_SubRun("sweep", float(r), base + i, matrix.num_prompts_for(float(r)))
                    for i, r in enumerate(sorted(spec.p("rates")))]
        elif phase == "pc":
            rate = float(spec.p("rate"))
            subs = [_SubRun("pc", rate, 1000 * (i + 1), matrix.num_prompts_for(rate), PC_INPUT_LEN, PC_PREFIX_LEN)
                    for i in range(int(spec.p("repeats")))]
            subs += [_SubRun("pc", math.inf, SAT_SEEDS[i % len(SAT_SEEDS)], SAT_NUM_PROMPTS, PC_INPUT_LEN,
                             PC_PREFIX_LEN) for i in range(int(spec.p("sat_extra")))]
        else:
            raise ValueError(f"{spec.run_id}: unknown serve_session phase {phase!r}")
        if self.ctx.dry_run:
            for s in subs:
                s.num_prompts = min(s.num_prompts, DRY_RUN_MAX_PROMPTS)
        return subs

    def _serve_session(self, spec: RunSpec, run_dir: str) -> None:
        handles: list[server.ServerHandle] = []
        valid = 0
        with self._monitors(spec, run_dir):
            try:
                handles = self._start_servers(spec, run_dir)
                for h in handles:
                    _wait_for_line(h.log_path, logparse.STARTUP_COMPLETE, STARTUP_LINE_WAIT_S)
                self._check_server_logs(spec, run_dir, handles)
                deadline = time.monotonic() + self._timeout(spec, False)       # spec 7.4: 3x the estimate
                for k, sub in enumerate(self._sub_runs(spec)):
                    sub_dir = os.path.join(run_dir, f"sub-{k}")
                    os.makedirs(sub_dir, exist_ok=True)
                    names = ["result.json"] if len(handles) == 1 else [f"result-{i}.json" for i in range(len(handles))]
                    violations, _ = self._client_run(spec, run_dir, sub_dir, handles, sub, k, names,
                                                     self._client_timeout(spec, sub, k, deadline))
                    valid += not violations
                    self._dead_server(handles)
            finally:
                before_stop = {h.log_path: _file_size(h.log_path) for h in handles}
                self._stop_servers(run_dir, handles)
        self._check_server_logs(spec, run_dir, handles, before_stop)
        if spec.p("phase") == "sat" and not valid:
            raise RunFailure("no_valid_saturation_run", "no saturation client run was valid, so the rate grid "
                                                        "(AM6) cannot use this session; see sub-*/validation.json")

    def _client_timeout(self, spec: RunSpec, sub: _SubRun, k: int, deadline: float) -> float:
        """3x the client run's own estimate (spec 7.4), within what is left of the session's 3x (review I2)."""
        left = deadline - time.monotonic()
        if left <= 0:
            raise RunFailure("timeout", f"the session passed 3x its estimate before sub-{k}")
        own = TIMEOUT_FACTOR * matrix.client_run_s(spec.config, sub.rate, sub.num_prompts, self._mu_rps())
        return min(max(MIN_TIMEOUT_S, own), left)

    def _client_specs(self, spec: RunSpec, handles: Sequence[server.ServerHandle], sub: _SubRun, k: int,
                      result_dir: str, names: Sequence[str]) -> list[client.ClientSpec]:
        n = len(handles)
        specs = []
        for i, h in enumerate(handles):
            seed = sub.seed + DP2RAND_SEED_OFFSET * i
            prompts = sub.num_prompts // n + (1 if i < sub.num_prompts % n else 0)
            metadata = (("tpprof_arm", spec.arm), ("tpprof_config", spec.config), ("tpprof_phase", sub.phase),
                        ("tpprof_rate", _rate_label(sub.rate)), ("tpprof_round", str(spec.round)),
                        ("tpprof_seed", str(seed)), ("tpprof_sub", str(k)))
            specs.append(client.ClientSpec(
                base_url=h.base_url, tokenizer=self.model_dir, input_len=sub.input_len, output_len=ONLINE_OUTPUT_LEN,
                prefix_len=sub.prefix_len, num_prompts=prompts, request_rate=sub.rate / n, seed=seed,
                result_dir=result_dir, result_filename=names[i], request_id_prefix=f"{spec.run_id}-s{k}-{i}-",
                metadata=metadata))
        return specs

    def _client_run(self, spec: RunSpec, run_dir: str, sub_dir: str, handles: Sequence[server.ServerHandle],
                    sub: _SubRun, k: int, names: Sequence[str], timeout_s: float) -> tuple[list[str], list[str]]:
        """One client run (two concurrent clients for DP2-rand) with /metrics before and after, validation and
        meta; returns (violations, flags). The R4 files live in sub_dir."""
        specs = self._client_specs(spec, handles, sub, k, sub_dir, names)
        gpu_csv, cpu_csv = os.path.join(run_dir, "gpu.csv"), os.path.join(run_dir, "cpu.csv")
        gpu_off, cpu_off = _file_size(gpu_csv), _file_size(cpu_csv)
        before_text = self._scrape_or_none(handles)
        if before_text is None:
            self._dead_server(handles)
            raise RunFailure("metrics_unreachable", f"GET /metrics failed before sub-{k} on a live server")
        with open(os.path.join(sub_dir, "metrics_before.prom"), "w") as f:
            f.write(before_text)
        # spec 4.4: the gauges are sampled once per second during saturation (rate inf) runs, when the
        # server has them while idle (not with two API servers, see EngineConfig.exposes_gauges)
        gauges = all(cfg.exposes_gauges for cfg in self._engine_configs(spec))
        poller = (server.MetricsPoller([h.base_url for h in handles], os.path.join(sub_dir, "gauges.csv"))
                  if math.isinf(sub.rate) and gauges else contextlib.nullcontext())
        with poller:
            outcomes = client.run_clients(specs, self.env, sub_dir, spec.run_id, timeout_s, self.ctx.vllm_bin)
        if len(outcomes) == 1 and os.path.exists(outcomes[0].log_path):
            client_log = os.path.join(sub_dir, "client.log")          # R4 name for the single client
            os.replace(outcomes[0].log_path, client_log)
            outcomes[0].log_path = client_log
        child_env = client.client_environment(self.env)
        for o in outcomes:
            self._append_cmd(run_dir, {
                "argv": o.argv, "env_overrides": _env_overrides(self.env, child_env), "cwd": None,
                "t_wall_start": o.t_wall_start, "t_mono_start": o.t_mono_start, "t_wall_end": o.t_wall_end,
                "t_mono_end": o.t_mono_end, "exit_code": o.exit_code, "timed_out": o.timed_out,
                "log": os.path.relpath(o.log_path, run_dir)})
        late = [os.path.relpath(o.log_path, run_dir) for o in outcomes if o.timed_out]
        if late:
            self._dead_server(handles)                  # a dead server is the better reason
            raise RunFailure("client_timeout", f"sub-{k}: the client was still running after {timeout_s:.0f} s "
                                               f"(3x its estimate) with the server up; see {', '.join(late)}")
        after_text = self._scrape_or_none(handles)
        if after_text is not None:
            with open(os.path.join(sub_dir, "metrics_after.prom"), "w") as f:
                f.write(after_text)

        violations: list[str] = []
        flags: list[str] = [] if gauges else ["no_gauges"]
        for o in outcomes:
            if o.exit_code != 0:
                violations.append(f"client exited {o.exit_code}; see {os.path.basename(o.log_path)}")
        merged = None
        try:
            parts = [results.load_serve_result(o.result_path) for o in outcomes]
            merged = parts[0] if len(parts) == 1 else results.merge_serve_results(parts)
            violations += results.validate_serve(merged, ONLINE_INPUT_LEN, ONLINE_OUTPUT_LEN)
        except (results.ResultFormatError, OSError, ValueError) as e:
            violations.append(f"result: {e}")
        preemptions = None
        try:
            before = promparse.scrape_summary(before_text, gauges=gauges)
            if after_text is None:
                raise ValueError("no /metrics scrape after the run (server gone?)")
            delta = promparse.deltas(before, promparse.scrape_summary(after_text, gauges=gauges))
            preemptions = delta["vllm:num_preemptions_total"]
            if preemptions > 0:
                flags.append("preempted")
        except ValueError as e:
            violations.append(f"metrics: {e}")
        _slice_csv(gpu_csv, gpu_off, os.path.join(sub_dir, "gpu.csv"))
        _slice_csv(cpu_csv, cpu_off, os.path.join(sub_dir, "cpu.csv"))
        throttle = monitor.throttle_flags(os.path.join(sub_dir, "gpu.csv")) if os.path.exists(
            os.path.join(sub_dir, "gpu.csv")) else None
        if throttle and throttle["throttled"]:
            flags.append("throttled")
        if merged is not None and len(outcomes) > 1:
            skew = float(merged.metadata.get("tpprof_start_skew_s", "0"))
            if skew > START_SKEW_FLAG_S:
                flags.append("start_skew")
        _write_json(os.path.join(sub_dir, "validation.json"), {
            "valid": not violations, "violations": violations, "flags": flags, "preemptions": preemptions,
            "throttle": throttle})
        _write_json(os.path.join(sub_dir, "meta.json"), {
            "phase": sub.phase, "rate": _rate_label(sub.rate) if math.isinf(sub.rate) else sub.rate,
            "seed": sub.seed, "config": spec.config, "arm": spec.arm, "round": spec.round, "k": k})
        state = "valid" if not violations else f"INVALID: {violations[0][:160]}"
        self.ctx.log(f"       sub-{k} {sub.phase} rate {_rate_label(sub.rate)} seed {sub.seed}: {state}"
                     f"{' [' + ', '.join(flags) + ']' if flags else ''}")
        return violations, flags

    def _scrape_or_none(self, handles: Sequence[server.ServerHandle]) -> str | None:
        try:
            return self._scrape(handles)
        except (OSError, http.client.HTTPException, RuntimeError, ValueError):
            return None

    # ------------------------------------------------------------ rate grid, P2 inputs

    def _time_scale(self) -> float:
        """Fake seconds per wall second in the dry run (FAKE_TIME_SCALE); 1 on the box."""
        return float(self.env.get("FAKE_TIME_SCALE", "1")) if self.ctx.dry_run else 1.0

    def _ensure_rate_grid(self) -> dict[str, list[float]] | None:
        """raw/rate_grid.json's grid; written once, when the three P1 saturation sessions are done (AM6, AM7)."""
        path = os.path.join(self.raw, "rate_grid.json")
        if os.path.exists(path):
            return dict(_read_json(path)["grid"])
        sats = [s for s in matrix.p1_sat_specs() if s.kind == "serve_session"]
        if not all(os.path.exists(os.path.join(self.raw, s.run_id, "done.json")) for s in sats):
            return None
        mu_rps: dict[str, float] = {}
        for s in sats:
            tps = []
            for sub in sorted(glob.glob(os.path.join(self.raw, s.run_id, "sub-*"))):
                try:
                    if not _read_json(os.path.join(sub, "validation.json"))["valid"]:
                        continue
                    r = results.load_serve_result(os.path.join(sub, "result.json"))
                except (OSError, ValueError, KeyError):
                    continue
                mu = stats.saturation_tps(r.start_times, r.ttfts, r.itls, r.output_lens, r.ok_mask)
                if math.isfinite(mu) and mu > 0:
                    tps.append(mu * self._time_scale())
            if not tps:
                self.ctx.log(f"no valid saturation run in {s.run_id}; the rate grid cannot be computed")
                return None
            mu_rps[s.config] = stats.percentile(tps, 50) / ONLINE_OUTPUT_LEN
        grid = matrix.rate_grid(mu_rps)
        _write_json(path, {"mu_rps": mu_rps, "grid": grid, "sources": [s.run_id for s in sats]})
        self.ctx.log(f"rate grid written to {path}: mu_rps {mu_rps}")
        return grid

    def _p0_record(self, kind: str, config: str, name: str) -> dict | None:
        for s in matrix.p0_specs():
            if s.kind == kind and s.config == config:
                d = os.path.join(self.raw, s.run_id)
                if os.path.exists(os.path.join(d, "done.json")) and os.path.exists(os.path.join(d, name)):
                    return _read_json(os.path.join(d, name))
        return None

    def _fi_backend(self) -> str | None:
        """The FlashInfer all-reduce backend the TP2 smoke engine chose (gates A-FIB)."""
        eff = self._p0_record("smoke", "TP2", "effective_config.json")
        return eff.get("fi_backend") if eff else None

    def _multicast(self) -> bool | None:
        """The preflight's multicast attribute (spec 7.5); None when not probed (off the box)."""
        report = self._p0_record("preflight", "none", "preflight.json")
        for c in (report or {}).get("checks", []):
            if c.get("name") == "multicast":
                return bool(c.get("ok"))
        return None
