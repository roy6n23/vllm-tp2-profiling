"""`vllm bench serve` client: the exact argv (spec 4.4) and a runner for concurrent clients.

DP2-rand runs two clients at once, one per single-GPU server (spec 4.1, D4-9), so
run_clients starts every client before waiting for any of them.
"""
from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tpprof import procs
from tpprof.constants import METRIC_PERCENTILES, NUM_WARMUPS, READY_CHECK_TIMEOUT_S, SERVED_MODEL_NAME
from tpprof.engine import BASE_ENV, UNSET_ALWAYS

METADATA_PREFIX = "tpprof_"
# bench serve splits each --metadata item at its first "=" and flattens the key into the
# result JSON's top level (D4-11), so keys are plain identifiers with a unique prefix.
_METADATA_KEY = re.compile(r"tpprof_[A-Za-z0-9_]+")
PERCENTILE_METRICS = "ttft,tpot,itl,e2el"
POLL_S = 0.05


@dataclass(frozen=True)
class ClientSpec:
    base_url: str
    tokenizer: str
    input_len: int
    output_len: int
    prefix_len: int
    num_prompts: int
    request_rate: float            # math.inf -> "inf"
    seed: int
    result_dir: str
    result_filename: str
    request_id_prefix: str
    metadata: tuple[tuple[str, str], ...]    # keys must start with "tpprof_"
    num_warmups: int = NUM_WARMUPS
    ready_check_timeout_s: int = READY_CHECK_TIMEOUT_S
    model_name: str = SERVED_MODEL_NAME

    def __post_init__(self) -> None:
        bad = [k for k, _ in self.metadata if not _METADATA_KEY.fullmatch(k)]
        if bad:
            raise ValueError(f"metadata keys must match {METADATA_PREFIX}<letters, digits, _>: {bad}")
        if not self.request_rate > 0:
            raise ValueError(f"request_rate must be > 0 (math.inf for saturation), got {self.request_rate}")

    @property
    def result_path(self) -> str:
        return os.path.join(self.result_dir, self.result_filename)


def _rate(rate: float) -> str:
    return "inf" if math.isinf(rate) else repr(float(rate))


def client_argv(spec: ClientSpec, vllm_bin: str = "vllm") -> list[str]:
    """The bench-serve argv of spec 4.4, token for token (D4-7..D4-10)."""
    metadata = [f"{k}={v}" for k, v in spec.metadata]
    return [
        vllm_bin, "bench", "serve", "--backend", "vllm", "--base-url", spec.base_url,
        "--endpoint", "/v1/completions", "--model", spec.model_name,
        "--tokenizer", spec.tokenizer, "--dataset-name", "random",
        "--random-input-len", str(spec.input_len), "--random-output-len", str(spec.output_len),
        "--random-range-ratio", "0.0", "--random-prefix-len", str(spec.prefix_len),
        "--num-prompts", str(spec.num_prompts), "--request-rate", _rate(spec.request_rate),
        "--burstiness", "1.0", "--seed", str(spec.seed), "--ignore-eos", "--temperature", "1.0", "--top-p", "1.0",
        "--num-warmups", str(spec.num_warmups), "--ready-check-timeout-sec", str(spec.ready_check_timeout_s),
        "--save-result", "--save-detailed", "--result-dir", spec.result_dir,
        "--result-filename", spec.result_filename,
        "--percentile-metrics", PERCENTILE_METRICS,
        "--metric-percentiles", ",".join(str(p) for p in METRIC_PERCENTILES),
        "--disable-tqdm", "--request-id-prefix", spec.request_id_prefix,
        *(["--metadata", *metadata] if metadata else []),
    ]


def client_environment(env: Mapping[str, str]) -> dict[str, str]:
    """env without UNSET_ALWAYS, plus BASE_ENV, as for the engines (spec 4.2).

    For the client this matters: VLLM_USE_RUST_BENCH=1 would exec a Rust client with its
    own output, SAVE_TO_PYTORCH_BENCHMARK_FORMAT adds a result file, and OPENAI_API_KEY
    adds an Authorization header (D4 research, endpoint_request_func.py:155).
    """
    out = {k: v for k, v in env.items() if k not in UNSET_ALWAYS}
    out.update(BASE_ENV)
    return out


@dataclass
class ClientOutcome:
    spec: ClientSpec
    exit_code: int | None          # None when the client timed out and was stopped
    timed_out: bool
    result_path: str
    log_path: str
    duration_s: float
    argv: list[str] = field(default_factory=list)
    t_wall_start: float = 0.0
    t_mono_start: float = 0.0
    t_wall_end: float = 0.0
    t_mono_end: float = 0.0


def run_clients(specs: Sequence[ClientSpec], env: Mapping[str, str], log_dir: str, run_id: str,
                timeout_s: float, vllm_bin: str = "vllm") -> list[ClientOutcome]:
    """Start one bench-serve client per spec at once, wait for all, return outcomes in spec order.

    Client i logs to log_dir/client-<i>.log. Each client runs in its own session with a
    raised RLIMIT_NOFILE (AM12) and the environment client_environment(env). Clients still
    running timeout_s after the start are stopped. The stop does not sweep the run tag,
    which the run's servers share.
    """
    log_paths = [os.path.join(log_dir, f"client-{i}.log") for i in range(len(specs))]
    result_paths = [s.result_path for s in specs]
    if len(set(result_paths)) != len(result_paths):
        raise ValueError(f"two clients would write the same result file: {result_paths}")
    child_env = client_environment(env)
    started: list[procs.Proc] = []
    try:
        for spec, log_path in zip(specs, log_paths):
            os.makedirs(spec.result_dir, exist_ok=True)
            started.append(procs.spawn(client_argv(spec, vllm_bin), child_env, log_path, run_id, raise_nofile=True))
        ends: list[tuple[float, float] | None] = [None] * len(started)
        deadline = time.monotonic() + timeout_s
        while any(e is None for e in ends) and time.monotonic() < deadline:
            for i, p in enumerate(started):
                if ends[i] is None and p.popen.poll() is not None:
                    ends[i] = (time.time(), time.monotonic())
            time.sleep(POLL_S)
    except BaseException:          # e.g. KeyboardInterrupt: the clients are in their own sessions
        for p in started:
            procs.stop(p, sweep=False)
        raise
    outcomes = []
    for spec, p, log_path, end in zip(specs, started, log_paths, ends):
        if end is None and p.popen.poll() is not None:       # exited between the last poll and the deadline
            end = (time.time(), time.monotonic())
        timed_out = end is None
        if timed_out:
            procs.stop(p, sweep=False)
            end = (time.time(), time.monotonic())
        outcomes.append(ClientOutcome(
            spec=spec, exit_code=None if timed_out else p.popen.returncode, timed_out=timed_out,
            result_path=spec.result_path, log_path=log_path, duration_s=end[1] - p.t_mono_start, argv=p.argv,
            t_wall_start=p.t_wall_start, t_mono_start=p.t_mono_start, t_wall_end=end[0], t_mono_end=end[1]))
    return outcomes
