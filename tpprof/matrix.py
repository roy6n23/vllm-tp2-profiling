"""The run matrix: RunSpec and its run_id, the P0/P1/P2 tiers, the rate grid and the estimator.

Contract C2. Spec sections 4.3-4.7 and 9.3, amendments AM6-AM11, AM14, AM20 and AM21.

- P0 decides every hypothesis: smoke, M2/M3, the offline grids with the AR3 and G2 cells,
  the bench-latency cross-check and the P0 traces (AM20).
- P1 measures saturation first (AM6), then sweeps the per-config rate grid (AM7) over
  Latin-square-ordered rounds (AM9).
- P2 holds the ablations (spec 4.7, AM10, AM11).

A RunSpec's run_id hashes the spec and the engine's `to_dict()`. Ports, paths and the run_id
itself are run-local and never enter the hash (build-4). Config "DP2rand" denotes the engine
pair DP2rand0 + DP2rand1 (ruling R3).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from tpprof import engine, model
from tpprof.constants import (
    DECODE_BATCHES,
    DECODE_INPUT_LEN,
    DECODE_ITERS,
    DECODE_L1,
    DECODE_L2,
    DECODE_WARMUP,
    LATIN_SQUARE,
    MIN_PROMPTS,
    NUM_WARMUPS,
    ONLINE_INPUT_LEN,
    ONLINE_OUTPUT_LEN,
    PC_LOAD_FRACTION,
    PREFILL_ITERS,
    PREFILL_LENS,
    PREFILL_WARMUP,
    RATE_FRACTIONS,
    ROUNDS,
    SAT_NUM_PROMPTS,
    SAT_SEEDS,
    SWEEP_WINDOW_S,
    XCHECK,
)

KINDS = ("preflight", "envcapture", "smoke", "comm_m1", "comm_m2", "comm_m3", "comm_m4",
         "offline", "bench_latency_xcheck", "trace", "serve_session", "tokbench")
NO_ENGINE_KINDS = frozenset({"preflight", "envcapture", "tokbench", "comm_m1", "comm_m2", "comm_m3", "comm_m4"})
TIERS = ("P0", "P1", "P2")
DP2RAND = "DP2rand"
DP2RAND_ENGINES = ("DP2rand0", "DP2rand1")
ONLINE_CONFIGS = ("TP1", "TP2", "DP2")
M3_MODES = "eager,graph"

# Estimator minutes and seconds, exactly as plan Task 15 renders AM21.
START_MIN, FIRST_START_MIN = 1.5, 3.0
FIXED_MIN = {"preflight": 1.0, "envcapture": 1.0, "comm_m1": 4.0, "comm_m2": 4.0, "comm_m4": 10.0,
             "tokbench": 2.0, "trace": 2.5}
COMM_M3_MIN = 3.0
SMOKE_MIN = 2.0
RUN_OVERHEAD_S = 20.0            # per client run: health, /metrics scrapes, result write
DEFAULT_PRICE_PER_HOUR = 6.98


# ---------------------------------------------------------------- names owned by other tasks
# M2_TOKENS (Task 12, tpprof.vendored), VARIANTS (Task 12, tpprof.comm_bench) and parse_points
# (Task 11, tpprof.offline) are imported from their owners. The fallbacks below apply only while an
# owner module is absent from the tree; any other import error propagates. tests/test_matrix.py
# fails, rather than skips, once an owner module exists and the import does not resolve to it.

try:
    from tpprof.vendored import M2_TOKENS
except ModuleNotFoundError as exc:
    if exc.name != "tpprof.vendored":
        raise
    M2_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

try:
    from tpprof.comm_bench import VARIANTS as M3_VARIANTS
except ModuleNotFoundError as exc:
    if exc.name != "tpprof.comm_bench":
        raise
    M3_VARIANTS = ((None, None), ("ring", "LL"), ("ring", "LL128"), ("ring", "Simple"),
                   ("tree", "LL"), ("tree", "LL128"), ("tree", "Simple"), ("nvls", "Simple"))


class _Point(NamedTuple):
    """The fields of Task 11's offline.Point that the estimator reads."""
    kind: str
    batch: int
    input_len: int
    output_len: int
    warmup: int
    iters: int


def _fallback_parse_points(spec: str) -> list[_Point]:
    """Task 11's offline.parse_points grammar, used only while tpprof/offline.py is absent:
    `all`, or `;`-separated sections `prefill`, `decode`, `decode:b1,b32`, `prefill:2048`.
    Duplicates are dropped, first occurrence wins."""
    def prefill(n: int) -> _Point:
        return _Point("prefill", 1, n, 1, PREFILL_WARMUP, PREFILL_ITERS)

    def decode(b: int) -> list[_Point]:
        return [_Point("decode", b, DECODE_INPUT_LEN, n, DECODE_WARMUP, DECODE_ITERS) for n in (DECODE_L1, DECODE_L2)]

    defaults = [prefill(n) for n in PREFILL_LENS] + [p for b in DECODE_BATCHES for p in decode(b)]
    spec = spec.strip()
    if spec == "all":
        return defaults
    points: list[_Point] = []
    for section in spec.split(";"):
        section = section.strip()
        kind, sep, items = section.partition(":")
        if kind not in ("prefill", "decode"):
            raise ValueError(f"point spec {spec!r}: section {section!r} must start with one of ('prefill', 'decode')")
        if not sep:
            points += [p for p in defaults if p.kind == kind]
            continue
        for item in items.split(","):
            item = item.strip()
            mt = re.fullmatch(r"b([0-9]+)" if kind == "decode" else r"([0-9]+)", item)
            if not mt or int(mt[1]) < 1:
                form = "b<batch>, e.g. b32" if kind == "decode" else "<input_len>, e.g. 2048"
                raise ValueError(f"point spec {spec!r}: {kind} item {item!r} is not {form}")
            points += decode(int(mt[1])) if kind == "decode" else [prefill(int(mt[1]))]
    return list(dict.fromkeys(points))


try:
    from tpprof.offline import parse_points
except ModuleNotFoundError as exc:
    if exc.name != "tpprof.offline":
        raise
    parse_points = _fallback_parse_points


def _freeze(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _render_variant(algo: str | None, proto: str | None) -> str:
    """comm_bench `--variant` string; None renders as "none" (ruling R5)."""
    return f"{algo or 'none'}:{proto or 'none'}"


@dataclass(frozen=True)
class RunSpec:
    kind: str
    config: str          # EngineConfig name, "DP2rand" (the pair) or "none"
    arm: str             # "base" or an arm name; "none" when config is "none"
    tier: str            # "P0" | "P1" | "P2"
    params: tuple[tuple[str, object], ...] = ()   # sorted by key; lists become tuples; a Mapping is accepted
    round: int = 0

    def __post_init__(self) -> None:
        items = self.params.items() if isinstance(self.params, Mapping) else self.params
        object.__setattr__(self, "params", tuple(sorted((str(k), _freeze(v)) for k, v in items)))
        if self.kind not in KINDS:
            raise ValueError(f"unknown run kind {self.kind!r}; expected one of {KINDS}")
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}; expected one of {TIERS}")
        if self.kind in NO_ENGINE_KINDS:
            if (self.config, self.arm) != ("none", "none"):
                raise ValueError(f"{self.kind} runs no engine: config and arm must be 'none', "
                                 f"got {self.config!r}/{self.arm!r}")
        elif self.config == "none":
            raise ValueError(f"{self.kind} needs an engine config, got config 'none'")
        else:
            self._engine_configs()        # raises ValueError for an unknown config or arm

    def p(self, key: str, default: object = None) -> object:
        return dict(self.params).get(key, default)

    def _engine_configs(self) -> list[engine.EngineConfig]:
        names = DP2RAND_ENGINES if self.config == DP2RAND else (self.config,)
        return [engine.arm_config(name, self.arm) for name in names]

    def to_dict(self) -> dict:
        """The hashed payload (C2); spec.json is this dict. Never holds ports, paths or the run_id."""
        if self.config == "none":
            eng: object = None
        else:
            dicts = [cfg.to_dict() for cfg in self._engine_configs()]
            eng = dicts if self.config == DP2RAND else dicts[0]
        return {"kind": self.kind, "config": self.config, "arm": self.arm, "tier": self.tier,
                "params": self.params, "round": self.round, "engine": eng}

    @property
    def run_id(self) -> str:
        sha8 = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.tier}-{self.kind}-{self.config}-{self.arm}-r{self.round}-{sha8}"


def _none(kind: str, tier: str, **params: object) -> RunSpec:
    return RunSpec(kind, "none", "none", tier, params)


def _sat_params() -> dict[str, object]:
    return {"phase": "sat", "seeds": SAT_SEEDS, "num_prompts": SAT_NUM_PROMPTS}


# ---------------------------------------------------------------- tiers

def p0_specs(rounds: int = ROUNDS) -> list[RunSpec]:
    """P0 in run order (spec 9.3 block 1, AM14, AM20). P0 has no rounds; `rounds` is accepted so
    every tier builder takes the same keyword."""
    del rounds
    t = "P0"
    specs = [_none("preflight", t), _none("envcapture", t)]
    specs.append(RunSpec("smoke", "TP1", "base", t, {"prompts": 8, "gpu1_check": True}))
    specs += [RunSpec("smoke", c, "base", t, {"prompts": 8}) for c in ("TP2", "DP2")]
    specs.append(_none("comm_m2", t, tokens=M2_TOKENS))
    specs.append(_none("comm_m3", t, variant=_render_variant(*M3_VARIANTS[0]), modes=M3_MODES))
    offline = [("TP1", "base", "all"), ("TP2", "base", "all"), ("TP2", "AR3", "decode:b1,b32"),
               ("TP1", "G2", "decode:b1"), ("TP2", "G2", "decode:b1")]
    specs += [RunSpec("offline", c, a, t, {"points": pts}) for c, a, pts in offline]
    specs += [RunSpec("bench_latency_xcheck", c, "base", t) for c in ("TP1", "TP2")]
    specs += [RunSpec("trace", c, "base", t, {"points": pts})
              for c in ("TP1", "TP2") for pts in ("decode:b1", "decode:b32", "prefill:2048")]
    specs.append(RunSpec("trace", "TP2", "G2", t, {"points": "decode:b1"}))
    return specs


def p1_sat_specs() -> list[RunSpec]:
    """The P1 part that does not need the rate grid: saturation per config (AM6), then tokbench."""
    specs = [RunSpec("serve_session", c, "base", "P1", _sat_params()) for c in ONLINE_CONFIGS]
    specs.append(_none("tokbench", "P1"))
    return specs


def p1_sweep_specs(grid: Mapping[str, Sequence[float]], rounds: int = ROUNDS) -> list[RunSpec]:
    """One sweep session per (round, config); round r uses LATIN_SQUARE[r-1] (AM9), cycling
    if rounds > len(LATIN_SQUARE)."""
    missing = [c for c in ONLINE_CONFIGS if c not in grid]
    if missing:
        raise ValueError(f"rate grid has no rates for {missing}")
    specs = []
    for r in range(1, rounds + 1):
        for config in LATIN_SQUARE[(r - 1) % len(LATIN_SQUARE)]:
            params = {"phase": "sweep", "rates": tuple(grid[config]), "seed_base": 1000 * r}
            specs.append(RunSpec("serve_session", config, "base", "P1", params, round=r))
    return specs


def p2_specs(grid: Mapping[str, Sequence[float]] | None, fi_backend: str | None,
             multicast: bool | None = None) -> list[RunSpec]:
    """P2 ablations (spec 4.7). The A-PC and A-RAND sessions need the rate grid and are left out
    when `grid` is None. A-FIB runs only if `fi_backend == "mnnvl"` (R11).

    The NVLS M3 variant runs only if multicast is present (spec 4.6), which is the preflight's
    multicast attribute (spec 7.5). If `multicast` is None (not probed), `fi_backend == "mnnvl"`
    stands in for it: vLLM keeps mnnvl exactly when the NVSwitch multicast workspace can be created
    (spec 3, AM4)."""
    t = "P2"
    mnnvl = fi_backend == "mnnvl"
    nvls = mnnvl if multicast is None else bool(multicast)
    offline = [("TP2", "AR1", "decode:b1,b32,b128;prefill:2048"),
               ("TP2", "AR2", "decode:b1,b32,b128;prefill:2048"),
               ("TP2", "AR3", "decode:b128;prefill:2048"),
               ("TP1", "G1", "decode:b1,b32"), ("TP2", "G1", "decode:b1,b32"),
               ("TP1", "G2", "decode:b32"), ("TP2", "G2", "decode:b32"),
               ("TP1", "EXECuni", "decode:b1,b32")]
    if mnnvl:
        offline.append(("TP2", "FIBtrtllm", "decode:b1,b32,b128"))
    specs = [RunSpec("offline", c, a, t, {"points": pts}) for c, a, pts in offline]
    specs += [RunSpec("trace", c, a, t, {"points": "decode:b1"})
              for c, a in (("TP2", "AR1"), ("TP2", "AR2"), ("TP2", "AR3"), ("TP1", "G1"), ("TP2", "G1"))]
    if grid is not None:
        pc_rate = grid["TP2"][RATE_FRACTIONS.index(PC_LOAD_FRACTION)]      # 0.60 x mu_TP2 (AM10)
        specs += [RunSpec("serve_session", "TP2", a, t, {"phase": "pc", "rate": pc_rate, "repeats": 3,
                                                          "sat_extra": 1})
                  for a in ("base", "PCon")]
        specs.append(RunSpec("serve_session", DP2RAND, "base", t,
                             {"phase": "sweep", "rates": tuple(grid["DP2"]), "seed_base": 1000}, round=1))
    specs += [RunSpec("serve_session", c, "API2", t, _sat_params()) for c in ("TP2", "DP2")]
    specs.append(_none("comm_m1", t))
    specs += [_none("comm_m3", t, variant=_render_variant(algo, proto), modes=M3_MODES)
              for algo, proto in M3_VARIANTS[1:] if algo != "nvls" or nvls]
    specs.append(_none("comm_m4", t))
    return specs


def build_matrix(tiers: Iterable[str], grid: Mapping[str, Sequence[float]] | None = None,
                 fi_backend: str | None = None, rounds: int = ROUNDS,
                 multicast: bool | None = None) -> list[RunSpec]:
    """The specs of `tiers`, always in P0, P1, P2 order. Without a grid, P1 holds only its
    saturation part and P2 drops its grid-dependent sessions."""
    wanted = set(tiers)
    unknown = sorted(wanted - set(TIERS))
    if unknown:
        raise ValueError(f"unknown tiers {unknown}; expected a subset of {TIERS}")
    specs: list[RunSpec] = []
    if "P0" in wanted:
        specs += p0_specs(rounds)
    if "P1" in wanted:
        specs += p1_sat_specs()
        if grid is not None:
            specs += p1_sweep_specs(grid, rounds)
    if "P2" in wanted:
        specs += p2_specs(grid, fi_backend, multicast)
    return specs


# ---------------------------------------------------------------- rate grid

def rate_grid(mu_rps: Mapping[str, float]) -> dict[str, list[float]]:
    """Per config, RATE_FRACTIONS x mu_c in req/s, rounded to 0.01 (AM7)."""
    grid = {}
    for config, mu in mu_rps.items():
        mu = float(mu)
        if not (math.isfinite(mu) and mu > 0):
            raise ValueError(f"saturation rate for {config} must be a positive number of req/s, got {mu!r}")
        grid[config] = [round(f * mu, 2) for f in RATE_FRACTIONS]
    return grid


def num_prompts_for(rate: float) -> int:
    """About SWEEP_WINDOW_S seconds of arrivals, at least MIN_PROMPTS (spec 4.4)."""
    return max(MIN_PROMPTS, round(rate * SWEEP_WINDOW_S))


def model_mu_rps() -> dict[str, float]:
    """A-priori saturation rate in req/s per online config: central model tok/s / output_len."""
    c = model.CONSTANTS["central"]
    return {cfg: model.saturation_output_tps(cfg, c) / ONLINE_OUTPUT_LEN for cfg in ONLINE_CONFIGS}


# ---------------------------------------------------------------- estimator

@dataclass
class Estimate:
    run_id: str
    kind: str
    tier: str
    minutes: float


def _ar_path(arm: str) -> str:
    return "nccl_unfused" if arm == "AR3" else "fused"


def _generate_s(tp: int, arm: str, batch: int, input_len: int, output_len: int) -> float:
    """Model latency of one generate(): batch prompts prefilled, then output_len - 1 decode steps."""
    c = model.CONSTANTS["central"]
    t = batch * model.prefill_time(tp, input_len, c)
    if output_len > 1:
        ctx = input_len + output_len // 2
        t += (output_len - 1) * model.decode_step_time(tp, batch, ctx, c, _ar_path(arm))
    return t


def _points_s(tp: int, arm: str, spec: str) -> float:
    """Offline: sum over points of (warmup + iters) x model latency."""
    return sum((p.warmup + p.iters) * _generate_s(tp, arm, p.batch, p.input_len, p.output_len)
               for p in parse_points(spec))


def _engine_shape(config: str) -> tuple[int, int]:
    """(tp, engines) of a config; DP2rand is two single-GPU engines."""
    if config == DP2RAND:
        return 1, len(DP2RAND_ENGINES)
    cfg = engine.base_config(config)
    return cfg.tp, cfg.dp


def _warmups_s(config: str) -> float:
    """The NUM_WARMUPS warmup requests of a sweep run. `vllm bench serve` sends them concurrently
    (vllm/benchmarks/serve.py:888-907, no semaphore without --max-concurrency), so they take one
    generate() of NUM_WARMUPS / engines requests."""
    tp, engines = _engine_shape(config)
    return _generate_s(tp, "base", max(1, NUM_WARMUPS // engines), ONLINE_INPUT_LEN, ONLINE_OUTPUT_LEN)


def _mu(config: str, mu_rps: Mapping[str, float] | None) -> float:
    key = "DP2" if config == DP2RAND else config
    if mu_rps and key in mu_rps:
        return float(mu_rps[key])
    return model_mu_rps()[key]


def _sat_run_s(mu: float, num_prompts: int = SAT_NUM_PROMPTS) -> float:
    """One saturation run: num_prompts / mu + 20 s."""
    return num_prompts / mu + RUN_OVERHEAD_S


def _rate_run_s(config: str, rate: float, mu: float) -> float:
    """One fixed-rate run: max(90 s, N / mu) + 20 s + the warmups."""
    return max(SWEEP_WINDOW_S, num_prompts_for(rate) / mu) + RUN_OVERHEAD_S + _warmups_s(config)


def _work_min(s: RunSpec, mu_rps: Mapping[str, float] | None) -> float:
    """Minutes of a spec excluding engine starts."""
    if s.kind in FIXED_MIN:
        return FIXED_MIN[s.kind]
    if s.kind == "comm_m3":
        return COMM_M3_MIN
    if s.kind == "smoke":
        return SMOKE_MIN + (_points_s(1, "base", "decode:b1") / 60 if s.p("gpu1_check") else 0.0)
    tp = _engine_shape(s.config)[0]
    if s.kind == "offline":
        return _points_s(tp, s.arm, str(s.p("points"))) / 60
    if s.kind == "bench_latency_xcheck":
        x = XCHECK
        return (x["warmup"] + x["iters"]) * _generate_s(tp, s.arm, x["batch"], x["input_len"], x["output_len"]) / 60
    if s.kind == "serve_session":
        mu = _mu(s.config, mu_rps)
        phase = s.p("phase")
        if phase == "sat":
            return len(s.p("seeds")) * _sat_run_s(mu, int(s.p("num_prompts", SAT_NUM_PROMPTS))) / 60
        if phase == "sweep":
            return sum(_rate_run_s(s.config, r, mu) for r in s.p("rates")) / 60
        if phase == "pc":
            return (s.p("repeats") * _rate_run_s(s.config, s.p("rate"), mu) + s.p("sat_extra") * _sat_run_s(mu)) / 60
        raise ValueError(f"{s.run_id}: unknown serve_session phase {phase!r}")
    raise ValueError(f"no estimate for kind {s.kind!r}")


def _starts(s: RunSpec) -> int:
    """Engine starts of a spec; the TP1 smoke adds the GPU1 offline check (AM14). The two DP2rand
    servers start concurrently on separate GPUs, so they count as one start of wall time."""
    if s.kind in NO_ENGINE_KINDS:
        return 0
    return 2 if s.kind == "smoke" and s.p("gpu1_check") else 1


def estimate(specs: Sequence[RunSpec], mu_rps: Mapping[str, float] | None = None) -> list[Estimate]:
    """Minutes per spec, in order (AM21, as plan Task 15 gives it). Each engine start costs
    START_MIN, or FIRST_START_MIN for the first start of a config (the RunSpec config, whatever the
    arm) in `specs`. mu_rps (req/s) defaults to the central model."""
    seen: set[str] = set()
    out = []
    for s in specs:
        minutes = _work_min(s, mu_rps)
        for _ in range(_starts(s)):
            minutes += START_MIN if s.config in seen else FIRST_START_MIN
            seen.add(s.config)
        out.append(Estimate(s.run_id, s.kind, s.tier, minutes))
    return out


def format_estimate(ests: Sequence[Estimate], price_per_hour: float = DEFAULT_PRICE_PER_HOUR) -> str:
    """Per tier: runs, hours and cost, then minutes per kind; then the total."""
    lines = []
    for tier in TIERS:
        rows = [e for e in ests if e.tier == tier]
        if not rows:
            continue
        hours = sum(e.minutes for e in rows) / 60
        lines.append(f"{tier}  {len(rows)} runs  {hours:.2f} h  ${hours * price_per_hour:.2f}")
        lines.append(f"    {'kind':<22}{'runs':>6}{'minutes':>10}")
        for kind in KINDS:
            of_kind = [e.minutes for e in rows if e.kind == kind]
            if of_kind:
                lines.append(f"    {kind:<22}{len(of_kind):>6}{sum(of_kind):>10.1f}")
    total_h = sum(e.minutes for e in ests) / 60
    lines.append(f"Total  {len(ests)} runs  {total_h:.2f} h  ${total_h * price_per_hour:.2f}"
                 f" at ${price_per_hour:.2f}/h")
    return "\n".join(lines) + "\n"
