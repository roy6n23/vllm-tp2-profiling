"""A-priori performance model for Llama-3.1-8B on H100 SXM: TP1 vs TP2 vs DP2 (spec section 5).

Three parts:
- derived model quantities (parameter, byte, FLOP and collective counts), exact integers;
- a roofline + alpha-beta step-time model for decode and prefill, a KV capacity model and a
  steady-state saturation model, all following D8's reference scripts
  (tests/fixtures/d8/model.py and sat.py);
- a Little's-law operating-point model for TPOT at a request rate, goodput curves and the
  crossover SLO s* (AM24).

The ``d8`` constant set reproduces D8's printed oracle (tests/fixtures/d8_model_oracle.txt);
the ``central`` set differs from it only in beta = 260 GB/s (AM22). Times are in seconds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from tpprof.constants import (
    DECODE_BATCHES,
    DECODE_INPUT_LEN,
    DECODE_L1,
    DECODE_L2,
    LLAMA31_8B,
    ONLINE_INPUT_LEN,
    ONLINE_OUTPUT_LEN,
    PREFILL_LENS,
    TPOT_SLOS_MS,
    TTFT_SLO_S,
    ModelShape,
)

GIB = 2**30
CONFIG_TP = {"TP1": 1, "TP2": 2, "DP2": 1}          # TP degree of one engine
CONFIG_ENGINES = {"TP1": 1, "TP2": 1, "DP2": 2}     # engines per config
AR_PATHS = ("fused", "nccl_unfused")
REQUEST_TOKENS = ONLINE_INPUT_LEN + ONLINE_OUTPUT_LEN               # 1280 KV tokens per request
ONLINE_MEAN_CTX = ONLINE_INPUT_LEN + ONLINE_OUTPUT_LEN // 2         # 1152, mean online decode context
DECODE_MEAN_CTX = DECODE_INPUT_LEN + (DECODE_L1 + DECODE_L2) // 2   # 1216, mean offline decode context
GOODPUT_GRID_POINTS = 200
_FIXED_POINT_MAX_ITERS = 200
_FIXED_POINT_TOL = 1e-9
_BAND_SETS = ("optimistic", "central", "pessimistic")


def _check_tp(tp: int) -> None:
    if tp not in (1, 2):
        raise ValueError(f"tp must be 1 or 2, got {tp!r}")


def _check_config(config: str) -> None:
    if config not in CONFIG_TP:
        raise ValueError(f"config must be one of {sorted(CONFIG_TP)}, got {config!r}")


# ---------------------------------------------------------------- derived model quantities

def _layer_matmul_params(s: ModelShape, tp: int = 1) -> int:
    """qkv + o + gate/up/down parameters of one decoder layer, per rank."""
    q, kv = s.heads * s.head_dim, s.kv_heads * s.head_dim
    attn = (q // tp + 2 * (kv // tp)) * s.hidden + (q // tp) * s.hidden
    mlp = 3 * (s.intermediate // tp) * s.hidden
    return attn + mlp


def _norm_params(s: ModelShape) -> int:
    """All RMSNorm weights: 2 per layer plus the final norm. Replicated under TP."""
    return s.layers * 2 * s.hidden + s.hidden


def _lm_head_params(s: ModelShape) -> int:
    return s.vocab * s.hidden


def param_count(s: ModelShape = LLAMA31_8B) -> int:
    """Total parameters: embedding, layers, norms and the untied lm_head."""
    return s.layers * _layer_matmul_params(s) + 2 * _lm_head_params(s) + _norm_params(s)


def weight_bytes(tp: int = 1, s: ModelShape = LLAMA31_8B) -> int:
    """Weight bytes held by one rank: column/row-parallel layers, vocab-parallel embedding and
    lm_head (the vocab divides by 64 * tp, so no padding), norms replicated."""
    _check_tp(tp)
    params = s.layers * _layer_matmul_params(s, tp) + 2 * (s.vocab // tp) * s.hidden + _norm_params(s)
    return params * s.bytes_per_param


def streamed_weight_bytes(tp: int = 1, s: ModelShape = LLAMA31_8B) -> float:
    """Weight bytes read per decode step per GPU: every matmul weight including lm_head, plus the
    replicated norms. Embedding rows read per step are negligible (D8 W1/W2)."""
    _check_tp(tp)
    matmul = s.bytes_per_param * (s.layers * _layer_matmul_params(s) + _lm_head_params(s))
    return matmul / tp + s.bytes_per_param * _norm_params(s)


def kv_bytes_per_token(tp: int = 1, s: ModelShape = LLAMA31_8B) -> int:
    """K and V bytes per token per GPU (KV heads split across TP ranks)."""
    _check_tp(tp)
    return 2 * s.layers * s.kv_heads * s.head_dim * s.bytes_per_param // tp


def linear_flops_per_token(include_lm_head: bool = False, s: ModelShape = LLAMA31_8B) -> int:
    """Matmul FLOPs per token at 2 FLOP/MAC, whole model."""
    params = s.layers * _layer_matmul_params(s) + (_lm_head_params(s) if include_lm_head else 0)
    return 2 * params


def prefill_flops(n: int, s: ModelShape = LLAMA31_8B) -> tuple[float, float]:
    """(linear FLOPs with lm_head on the last token only, causal attention FLOPs) for n tokens.

    Attention counts QK^T and PV over the n(n+1)/2 causal pairs at 2 FLOP/MAC."""
    linear = linear_flops_per_token(False, s) * n + 2 * _lm_head_params(s)
    attention = 2 * s.layers * s.heads * s.head_dim * n * (n + 1)
    return float(linear), float(attention)


def allreduces_per_step(tp: int, s: ModelShape = LLAMA31_8B) -> int:
    """TP all-reduces per forward step: the embedding plus o_proj and down_proj per layer."""
    _check_tp(tp)
    return 0 if tp == 1 else 2 * s.layers + 1


def allgathers_per_step(tp: int) -> int:
    """Logits all-gathers per forward step."""
    _check_tp(tp)
    return 0 if tp == 1 else 1


def ar_message_bytes(tokens: int, s: ModelShape = LLAMA31_8B) -> int:
    """All-reduce message size: one hidden-state row per token."""
    return tokens * s.hidden * s.bytes_per_param


# ---------------------------------------------------------------- constant sets

@dataclass(frozen=True)
class Constants:
    name: str
    bw_eff: float            # B/s streaming bandwidth
    gemm_flops: float        # FLOP/s for linear layers
    attn_flops: float        # FLOP/s for prefill attention
    t_fixed: float           # s per step, not shrunk by TP (D8 o1)
    t_extra_tp2: float       # s per step, TP2 only (D8 o2_extra)
    alpha: float             # s per all-reduce op on the default (fused) path
    beta: float              # B/s effective all-reduce bandwidth
    alpha_nccl_graph: float = 6e-6         # AM23
    unfused_norm_s: float = 2.5e-6         # s per standalone residual+RMSNorm kernel at small batch (AR1-AR3, TP1)
    ag_alpha: float = 10e-6                # logits all-gather fixed cost
    ew_bytes_tp1: float = 177e3            # non-GEMM bytes/token/layer (prefill, saturation)
    ew_bytes_tp2: float = 120.5e3          # = 64e3 + (177e3 - 64e3) / 2
    total_mem_gib: float = 79.11
    gpu_mem_util: float = 0.90
    weights_gib_tp1: float = 14.99
    weights_gib_tp2: float = 7.51
    nonkv_gib_tp1: float = 4.82
    nonkv_gib_tp2: float = 5.40


CONSTANTS: dict[str, Constants] = {
    "d8": Constants("d8", 3.0e12, 700e12, 400e12, 0.8e-3, 0.1e-3, 5e-6, 350e9),
    "central": Constants("central", 3.0e12, 700e12, 400e12, 0.8e-3, 0.1e-3, 5e-6, 260e9),
    "optimistic": Constants("optimistic", 3.1e12, 790e12, 500e12, 0.4e-3, 0.05e-3, 4e-6, 370e9,
                            nonkv_gib_tp1=2.5, nonkv_gib_tp2=2.9),
    "pessimistic": Constants("pessimistic", 2.7e12, 600e12, 350e12, 1.2e-3, 0.2e-3, 5e-6, 150e9,
                             nonkv_gib_tp1=6.0, nonkv_gib_tp2=7.0),
}


# ---------------------------------------------------------------- step-time models

def _ar_alpha(c: Constants, ar_path: str) -> float:
    if ar_path == "fused":
        return c.alpha
    if ar_path == "nccl_unfused":
        return c.alpha_nccl_graph
    raise ValueError(f"ar_path must be one of {AR_PATHS}, got {ar_path!r}")


def _allreduce_time(tokens: float, alpha: float, c: Constants, s: ModelShape = LLAMA31_8B) -> float:
    """All 65 all-reduces of one step, each alpha + message / beta."""
    return allreduces_per_step(2, s) * (alpha + tokens * s.hidden * s.bytes_per_param / c.beta)


def _allgather_time(sampled_tokens: float, c: Constants, s: ModelShape = LLAMA31_8B) -> float:
    """Logits all-gather: fixed cost plus each rank's vocab half per sampled token."""
    return c.ag_alpha + sampled_tokens * (s.vocab // 2) * s.bytes_per_param / c.beta


def decode_comm_time(tp: int, batch: int, c: Constants, ar_path: str = "fused") -> float:
    """TP collective time of one decode step (all-reduces plus the logits all-gather); 0 at TP1."""
    _check_tp(tp)
    alpha = _ar_alpha(c, ar_path)
    if tp == 1:
        return 0.0
    return _allreduce_time(batch, alpha, c) + _allgather_time(batch, c)


def decode_step_time(tp: int, batch: int, ctx: int, c: Constants, ar_path: str = "fused") -> float:
    """Decode step time in seconds for `batch` sequences at mean context `ctx` (D8 decode()).

    ar_path "nccl_unfused" (AR3) uses the graph-mode NCCL alpha and adds one standalone
    residual+RMSNorm kernel per all-reduce (AM23)."""
    _check_tp(tp)
    comm = decode_comm_time(tp, batch, c, ar_path)       # validates ar_path
    s = LLAMA31_8B
    t_lin = max(streamed_weight_bytes(tp) / c.bw_eff,
                batch * linear_flops_per_token(True) / tp / c.gemm_flops)
    t_kv = batch * ctx * kv_bytes_per_token(tp) / c.bw_eff
    t = t_lin + t_kv + c.t_fixed
    if tp == 2:
        t += c.t_extra_tp2 + comm
        if ar_path == "nccl_unfused":
            t += allreduces_per_step(tp, s) * c.unfused_norm_s
    return t


def prefill_time(tp: int, n: int, c: Constants) -> float:
    """Prefill time in seconds for one n-token prompt (D8 prefill())."""
    _check_tp(tp)
    s = LLAMA31_8B
    linear, attention = prefill_flops(n, s)
    ew_bytes = c.ew_bytes_tp1 if tp == 1 else c.ew_bytes_tp2
    t = linear / tp / c.gemm_flops + attention / tp / c.attn_flops + ew_bytes * s.layers * n / c.bw_eff + c.t_fixed
    if tp == 2:
        t += c.t_extra_tp2 + _allreduce_time(n, c.alpha, c) + _allgather_time(1, c)
    return t


# ---------------------------------------------------------------- capacity and saturation

def kv_capacity_tokens(tp: int, c: Constants) -> int:
    """KV cache tokens per engine: requested memory minus weights and non-KV overhead."""
    _check_tp(tp)
    requested = c.total_mem_gib * GIB * c.gpu_mem_util
    weights, nonkv = (c.weights_gib_tp1, c.nonkv_gib_tp1) if tp == 1 else (c.weights_gib_tp2, c.nonkv_gib_tp2)
    return int((requested - weights * GIB - nonkv * GIB) / kv_bytes_per_token(tp))


def running_limit(config: str, c: Constants, max_num_seqs: int = 1024) -> int:
    """Running requests per engine: min(max_num_seqs, KV tokens / (input + output tokens))."""
    _check_config(config)
    return min(max_num_seqs, kv_capacity_tokens(CONFIG_TP[config], c) // REQUEST_TOKENS)


def _saturation_step(batch: int, tp: int, c: Constants) -> tuple[float, float]:
    """(step time, output tokens per step) of one engine in steady state (D8 sat.py step()).

    Each step decodes `batch` requests; batch / output_len of them finish and are replaced by
    prefilling as many new prompts (chunked prefill), so concurrency stays constant."""
    s = LLAMA31_8B
    finished = batch / ONLINE_OUTPUT_LEN
    prefill_tokens = finished * ONLINE_INPUT_LEN
    tokens = batch + prefill_tokens
    sampled = batch + finished
    lin_flops = (tokens * linear_flops_per_token(False) + sampled * 2 * _lm_head_params(s)) / tp
    t_lin = max(lin_flops / c.gemm_flops, streamed_weight_bytes(tp) / c.bw_eff)
    t_att = (batch * ONLINE_MEAN_CTX * kv_bytes_per_token(tp) / c.bw_eff
             + 2 * s.layers * s.heads * s.head_dim * prefill_tokens * ONLINE_INPUT_LEN / tp / c.attn_flops)
    ew_bytes = c.ew_bytes_tp1 if tp == 1 else c.ew_bytes_tp2
    t_ew = ew_bytes * s.layers * tokens / c.bw_eff
    t = t_lin + t_att + t_ew + c.t_fixed
    if tp == 2:
        t += _allreduce_time(tokens, c.alpha, c) + _allgather_time(batch, c) + c.t_extra_tp2
    return t, sampled


def saturation_output_tps(config: str, c: Constants, max_num_seqs: int = 1024) -> float:
    """Saturation output throughput (tok/s) of the whole config; DP2 counts both engines.

    This is a no-preemption bound (AM6): every engine runs exactly running_limit() requests,
    with none preempted or waiting on KV."""
    _check_config(config)
    t, sampled = _saturation_step(running_limit(config, c, max_num_seqs), CONFIG_TP[config], c)
    return CONFIG_ENGINES[config] * sampled / t


# ---------------------------------------------------------------- Little's law, goodput, s*

def _operating_point(config: str, rate_rps: float, c: Constants) -> tuple[float, float] | None:
    """(TPOT s, TTFT s) at a Poisson request rate, or None past saturation or the running limit.

    Per engine: rho = r_e * output_len / mu_e; TTFT = prefill(input_len) / (1 - rho);
    TPOT = decode_step(B) * (1 + r_e * prefill(input_len)); B = r_e * E2E iterated to a
    fixed point from B = 1 (AM24)."""
    _check_config(config)
    tp, engines = CONFIG_TP[config], CONFIG_ENGINES[config]
    r_e = rate_rps / engines
    mu_e = saturation_output_tps(config, c) / engines
    rho = r_e * ONLINE_OUTPUT_LEN / mu_e
    if rho >= 1:
        return None
    t_prefill = prefill_time(tp, ONLINE_INPUT_LEN, c)
    prefill_share = r_e * t_prefill
    ttft = t_prefill / (1 - rho)
    limit = running_limit(config, c)
    batch = 1.0
    tpot = 0.0
    for _ in range(_FIXED_POINT_MAX_ITERS):
        tpot = decode_step_time(tp, max(1, round(batch)), ONLINE_MEAN_CTX, c) * (1 + prefill_share)
        new_batch = r_e * (ttft + (ONLINE_OUTPUT_LEN - 1) * tpot)
        if new_batch > limit:
            return None
        converged = abs(new_batch - batch) < _FIXED_POINT_TOL
        batch = new_batch
        if converged:
            break
    return tpot, ttft


def tpot_at_rate(config: str, rate_rps: float, c: Constants) -> float | None:
    """Predicted TPOT in seconds at `rate_rps` requests/s for the whole config; None if the
    rate is at or past saturation or needs more than running_limit() requests in flight."""
    point = _operating_point(config, rate_rps, c)
    return None if point is None else point[0]


def goodput_curve(config: str, c: Constants, tpot_slos_ms=TPOT_SLOS_MS, ttft_slo_s=TTFT_SLO_S) -> list[float]:
    """For each TPOT SLO (ms), the largest rate (req/s) on a 200-point grid in (0, mu / output_len)
    whose predicted TPOT and TTFT both meet the SLOs; 0.0 if none does."""
    r_max = saturation_output_tps(config, c) / ONLINE_OUTPUT_LEN
    rates = [r_max * k / (GOODPUT_GRID_POINTS + 1) for k in range(1, GOODPUT_GRID_POINTS + 1)]
    points = [(r, _operating_point(config, r, c)) for r in rates]
    curve = []
    for slo_ms in tpot_slos_ms:
        ok = [r for r, p in points if p is not None and p[0] * 1e3 <= slo_ms and p[1] <= ttft_slo_s]
        curve.append(max(ok) if ok else 0.0)
    return curve


def s_star(c: Constants) -> float | None:
    """Crossover TPOT SLO in ms: the smallest SLO where DP2 goodput >= TP2 goodput, linearly
    interpolated between SLO grid points. Only SLOs where either config has goodput count, and
    DP2 must lose at the first counted SLO (a sign change, AM8); otherwise None."""
    tp2, dp2 = goodput_curve("TP2", c), goodput_curve("DP2", c)
    pts = [(s, d - t) for s, d, t in zip(TPOT_SLOS_MS, dp2, tp2) if max(d, t) > 0]
    for (s0, d0), (s1, d1) in zip(pts, pts[1:]):
        if d0 < 0 <= d1:
            return s0 + (0 - d0) * (s1 - s0) / (d1 - d0)
    return None


# ---------------------------------------------------------------- predictions

def _sig4(x: float) -> float:
    return float(f"{x:.4g}")


def _band(values: list[float | None]) -> list[float] | None:
    kept = [v for v in values if v is not None]
    return [_sig4(min(kept)), _sig4(max(kept))] if kept else None


def _hypothesis_values(c: Constants) -> dict[str, float | None]:
    def speedup(b: int) -> float:
        return decode_step_time(1, b, DECODE_MEAN_CTX, c) / decode_step_time(2, b, DECODE_MEAN_CTX, c)

    b_max = max(DECODE_BATCHES)
    return {
        "H1": speedup(1),
        "H2": speedup(b_max) / 2 - speedup(1) / 2,
        "H3": saturation_output_tps("DP2", c) / saturation_output_tps("TP2", c),
        "H4": s_star(c),
        "H5": kv_capacity_tokens(2, c) / kv_capacity_tokens(1, c),
        "H8": (decode_step_time(2, 1, DECODE_MEAN_CTX, c, ar_path="nccl_unfused")
               / decode_step_time(2, 1, DECODE_MEAN_CTX, c)),
    }


def predictions() -> dict:
    """Every a-priori number predictions.md reports, keyed by constant set.

    decode/prefill are ms and saturation_tps is tok/s, at full precision. Bands are the min/max over
    the optimistic, central and pessimistic sets (not d8); bands, hypotheses_central, the KV ratio
    and s* are rounded to 4 significant digits."""
    names = list(CONSTANTS)
    s_stars = {n: s_star(CONSTANTS[n]) for n in names}
    hyp = {n: _hypothesis_values(CONSTANTS[n]) for n in _BAND_SETS}
    return {
        "constants": {n: asdict(c) for n, c in CONSTANTS.items()},
        "decode": {n: {str(tp): {str(b): decode_step_time(tp, b, DECODE_MEAN_CTX, CONSTANTS[n]) * 1e3
                                 for b in DECODE_BATCHES} for tp in (1, 2)} for n in names},
        "prefill": {n: {str(tp): {str(x): prefill_time(tp, x, CONSTANTS[n]) * 1e3
                                  for x in PREFILL_LENS} for tp in (1, 2)} for n in names},
        "kv": {n: {"TP1": kv_capacity_tokens(1, CONSTANTS[n]), "TP2": kv_capacity_tokens(2, CONSTANTS[n]),
                   "ratio": _sig4(kv_capacity_tokens(2, CONSTANTS[n]) / kv_capacity_tokens(1, CONSTANTS[n]))}
               for n in names},
        "saturation_tps": {n: {cfg: saturation_output_tps(cfg, CONSTANTS[n]) for cfg in ("TP1", "TP2", "DP2")}
                           for n in names},
        "s_star_ms": {n: None if v is None else _sig4(v) for n, v in s_stars.items()},
        "bands": {h: _band([hyp[n][h] for n in _BAND_SETS]) for h in ("H1", "H2", "H3", "H4", "H5", "H8")},
        "hypotheses_central": {h: None if v is None else _sig4(v) for h, v in hyp["central"].items()},
    }
