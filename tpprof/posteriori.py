"""A-posteriori stage of the decode step-time model (spec 5.1, AM25).

The a-priori constants in model.CONSTANTS came from the literature and were committed before the run. This
stage replaces the ones that can be measured without a TP2 engine:

- bw_eff and t_fixed: the slope and intercept of the TP1 decode step time against the bytes one step reads
  (weights + batch x context x KV bytes), least squares over the measured batches;
- alpha and beta of the default all-reduce path: M2, the fused one-shot kernel of the FlashInfer backend the
  TP2 engines ran; alpha and beta of pure NCCL (AR3): M3, graph mode, default variant;
- t_extra_tp2 is set to 0, because no TP1 measurement can fit it.

Every other constant keeps its central a-priori value. TP2 is then an out-of-sample prediction: no TP2 step
time feeds a constant, and measured - predicted is what a TP1-calibrated model does not explain.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from tpprof import model

APRIORI = "central"
M2_IMPL = "flashinfer_{backend}_fused_allreduce_rmsnorm_oneshot"    # decode messages use the one-shot kernel
M3_IMPL, M3_MODE, M3_VARIANT = "torch_nccl", "graph", "none:none"
# (config, arm) -> (TP degree, model AR path, "in" if the points fed the fit else "out")
PREDICTED = {("TP1", "base"): (1, "fused", "in"), ("TP2", "base"): (2, "fused", "out"),
             ("TP2", "AR3"): (2, "nccl_unfused", "out")}
UNITS = {"bw_eff": "B/s", "t_fixed": "s", "t_extra_tp2": "s", "alpha": "s", "beta": "B/s",
         "alpha_nccl_graph": "s", "beta_nccl": "B/s"}


def _num(x: object) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _decode_steps(offline: Sequence[Mapping], config: str, arm: str) -> dict[int, tuple[float, float]]:
    """batch -> (median measured step s, mean context) of one config and arm; derived rows are left out."""
    groups: dict[int, list[tuple[float, float]]] = {}
    for r in offline:
        step = _num(r.get("step_s"))
        if ((r.get("config"), r.get("arm"), r.get("kind")) == (config, arm, "decode") and not r.get("derived")
                and step is not None and r.get("batch") is not None):
            groups.setdefault(int(r["batch"]), []).append((step, _num(r.get("ctx_mean")) or model.DECODE_MEAN_CTX))
    return {b: (float(np.median([s for s, _ in v])), float(np.median([c for _, c in v])))
            for b, v in sorted(groups.items())}


def step_bytes(tp: int, batch: int, ctx: float) -> float:
    """Bytes one decode step reads per GPU: the streamed weights plus every running sequence's KV."""
    return model.streamed_weight_bytes(tp) + batch * ctx * model.kv_bytes_per_token(tp)


def tp1_fit(offline: Sequence[Mapping]) -> dict | None:
    """bw_eff (B/s) and t_fixed (s) from the measured TP1 baseline decode points: step = bytes / bw_eff + t_fixed.

    None with fewer than two batches or a line that does not rise. max_resid_frac is the largest
    |fit - measured| / measured over the fitted points."""
    steps = _decode_steps(offline, "TP1", "base")
    if len(steps) < 2:
        return None
    x = np.array([step_bytes(1, b, ctx) for b, (_, ctx) in steps.items()])
    y = np.array([s for s, _ in steps.values()])
    scale = float(x.max())
    slope_scaled, intercept = np.polyfit(x / scale, y, 1)
    slope = float(slope_scaled) / scale
    if not slope > 0:
        return None
    fitted = slope * x + intercept
    return {"bw_eff": 1.0 / slope, "t_fixed": float(intercept), "n": len(steps),
            "max_resid_frac": float(np.max(np.abs(fitted - y) / y))}


def _comm_fit(comm: Sequence[Mapping], source: str, impl: str, mode: str | None = None,
              variant: str | None = None) -> tuple[float | None, float | None]:
    for r in comm:
        if (r.get("row") == "fit" and r.get("source") == source and r.get("impl") == impl
                and (mode is None or r.get("mode") == mode) and (variant is None or r.get("variant") == variant)):
            alpha, beta = _num(r.get("alpha_us")), _num(r.get("beta_GBps"))
            return (None if alpha is None else alpha * 1e-6), (None if beta is None else beta * 1e9)
    return None, None


def comm_constants(comm: Sequence[Mapping], fi_backend: str | None) -> dict:
    """alpha (s) and beta (B/s) of the two all-reduce paths the model has, from the comm table's fit rows; a
    value is None when its fit is missing. impl names the M2 row of the engines' FlashInfer backend."""
    impl = M2_IMPL.format(backend=fi_backend) if fi_backend else None
    alpha, beta = _comm_fit(comm, "comm_m2", impl) if impl else (None, None)
    alpha_nccl, beta_nccl = _comm_fit(comm, "comm_m3", M3_IMPL, M3_MODE, M3_VARIANT)
    return {"alpha": alpha, "beta": beta, "impl": impl, "alpha_nccl_graph": alpha_nccl, "beta_nccl": beta_nccl}


def _constant_rows(fit: Mapping, comm: Mapping) -> list[dict]:
    c0 = model.CONSTANTS[APRIORI]
    tp1 = f"TP1 baseline decode, least squares over {fit['n']} batches (largest misfit {fit['max_resid_frac']:.2%})"
    m2 = f"M2 {comm['impl']}" if comm["impl"] else "M2"
    m3 = f"M3 {M3_IMPL}, {M3_MODE}, default variant"
    chosen = (
        ("bw_eff", fit["bw_eff"], c0.bw_eff, tp1),
        ("t_fixed", fit["t_fixed"], c0.t_fixed, tp1),
        ("t_extra_tp2", 0.0, c0.t_extra_tp2, "set to 0: no TP1 measurement can fit it, so it stays in the residual"),
        ("alpha", comm["alpha"], c0.alpha, m2),
        ("beta", comm["beta"], c0.beta, m2),
        ("alpha_nccl_graph", comm["alpha_nccl_graph"], c0.alpha_nccl_graph, m3),
        ("beta_nccl", comm["beta_nccl"], c0.beta, m3),
    )
    return [{"row": "constant", "name": name, "value": apriori if value is None else value, "unit": UNITS[name],
             "apriori": apriori, "source": f"a priori (no fit: {source})" if value is None else source}
            for name, value, apriori, source in chosen]


def constant_sets(rows: Sequence[Mapping]) -> dict[str, model.Constants] | None:
    """The model inputs per AR path (model.AR_PATHS) from the table's constant rows; None without them.

    The model has one beta, so the pure NCCL path gets its own set with M3's beta."""
    v = {r["name"]: r["value"] for r in rows if r.get("row") == "constant"}
    if not v:
        return None
    fused = replace(model.CONSTANTS[APRIORI], name="posteriori", bw_eff=v["bw_eff"], t_fixed=v["t_fixed"],
                    t_extra_tp2=v["t_extra_tp2"], alpha=v["alpha"], beta=v["beta"],
                    alpha_nccl_graph=v["alpha_nccl_graph"])
    return {"fused": fused, "nccl_unfused": replace(fused, beta=v["beta_nccl"])}


def rows(tables: Mapping[str, Sequence[Mapping]], fi_backend: str | None) -> list[dict]:
    """The tidy `posteriori` table: one "constant" row per refitted constant (value, a-priori value, source),
    then one "point" row per measured decode batch of TP1 (in sample), TP2 and TP2/AR3 (out of sample) with
    the measured, a-priori and a-posteriori step (ms) and residual = measured - a posteriori. [] without a
    TP1 fit."""
    offline = tables.get("offline_points", [])
    fit = tp1_fit(offline)
    if fit is None:
        return []
    out = _constant_rows(fit, comm_constants(tables.get("comm", []), fi_backend))
    sets = constant_sets(out)
    c0 = model.CONSTANTS[APRIORI]
    for (config, arm), (tp, ar_path, sample) in PREDICTED.items():
        for batch, (step, ctx) in _decode_steps(offline, config, arm).items():
            measured = step * 1e3
            predicted = model.decode_step_time(tp, batch, ctx, sets[ar_path], ar_path) * 1e3
            out.append({"row": "point", "config": config, "arm": arm, "batch": batch, "sample": sample,
                        "measured_ms": measured,
                        "apriori_ms": model.decode_step_time(tp, batch, ctx, c0, ar_path) * 1e3,
                        "posteriori_ms": predicted, "residual_ms": measured - predicted,
                        "residual_frac": (measured - predicted) / measured})
    return out
