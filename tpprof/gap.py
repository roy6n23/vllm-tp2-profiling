"""Where TP2's missing speedup goes (spec 1.1, deliverable 2): the gap between the TP2 decode step and half
the TP1 step, split by kernel category.

For each batch that has gate-passing baseline traces of both configs, a component's time per step is the mean
over the pure decode steps of its kernels (TP2: also the mean over the two ranks). The step totals are the
untraced medians, as in AM16's idle_est; what the kernels leave of the step is "not on the GPU". A component's
excess is its TP2 time minus half its TP1 time, so the excesses add up to the gap exactly.

The kernel times come from the traced runs and the step totals from the untraced ones. The two can differ by a
few percent run to run; the step row reports both ratios, and "not on the GPU" absorbs the difference.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

COMPONENTS = ("gemm", "attention", "norm_act_rope", "comm", "other_gpu", "not_gpu")
LABELS = {
    "step": "decode step",
    "gemm": "GEMM (weights)",
    "attention": "attention",
    "norm_act_rope": "norm, residual, RoPE, activation",
    "comm": "all-reduce and all-gather",
    "other_gpu": "sampling, copies, other kernels",
    "not_gpu": "not on the GPU (CPU, launch gaps)",
}
_OTHER_GPU = ("sampling", "memcpy", "other")
_DECODE_POINT = re.compile(r"^decode:b(\d+)$")


def _mean(steps: Sequence[Mapping], key: str) -> float:
    return float(np.mean([s[key] for s in steps]))


def _traces(trace_steps: Sequence[Mapping]) -> dict[tuple[int, str], list[list[dict]]]:
    """(batch, config) -> per rank, the steps the statistics use: pure decode steps when there are any (as
    analyze._stat_mean does), else every step. Baseline arm, gate-passing traces only."""
    by_rank: dict[tuple[int, str, object], list[dict]] = {}
    for s in trace_steps:
        m = _DECODE_POINT.match(str(s.get("points", "")))
        if m and s.get("arm") == "base" and s.get("gate_ok") is True:
            by_rank.setdefault((int(m.group(1)), s["config"], s["rank"]), []).append(s)
    out: dict[tuple[int, str], list[list[dict]]] = {}
    for (batch, config, _), steps in sorted(by_rank.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        out.setdefault((batch, config), []).append([s for s in steps if s["pure_decode"]] or steps)
    return out


def _untraced_ms(offline: Sequence[Mapping], config: str, batch: int) -> float | None:
    vals = [r["t_ms"] for r in offline if (r.get("config"), r.get("arm"), r.get("kind")) == (config, "base", "decode")
            and r.get("batch") == batch and not r.get("derived") and r.get("t_ms") is not None]
    return float(np.median(vals)) if vals else None


def _traced_step_ms(ranks: Sequence[Sequence[Mapping]]) -> float:
    """Median traced step; the last step of a trace has no next start, so its duration is left out."""
    return float(np.median([s["step_ms"] for steps in ranks for s in steps if not s["last"]]
                           or [s["step_ms"] for steps in ranks for s in steps]))


def _components(ranks: Sequence[Sequence[Mapping]], step_ms: float, norm_in_ar_ms: float) -> dict[str, float]:
    """Per-step ms by component, mean over ranks. norm_in_ar_ms is the residual add + RMSNorm time that the
    fused all-reduce kernel carries (AM16): it counts as norm work, not as communication."""
    def cat(name: str) -> float:
        return float(np.mean([_mean(steps, f"cat_{name}_ms") for steps in ranks]))

    out = {"gemm": cat("gemm"), "attention": cat("attention"),
           "norm_act_rope": cat("norm_act_rope") + norm_in_ar_ms,
           "comm": cat("all_reduce") + cat("all_gather") - norm_in_ar_ms,
           "other_gpu": sum(cat(c) for c in _OTHER_GPU)}
    out["not_gpu"] = step_ms - sum(out.values())
    return out


def rows(tables: Mapping[str, Sequence[Mapping]]) -> list[dict]:
    """The tidy `tp2_gap` table: per traced batch, a "step" row (untraced TP1 and TP2 step, the gap to half of
    TP1, traced / untraced step of both runs) and one row per component with its TP1 and TP2 ms per step, half
    of TP1, the excess and its share of the gap. A batch without both traces and both untraced points has no
    rows."""
    traces = _traces(tables.get("trace_steps", []))
    offline = tables.get("offline_points", [])
    out: list[dict] = []
    for batch in sorted({b for b, _ in traces}):
        tp1, tp2 = traces.get((batch, "TP1")), traces.get((batch, "TP2"))
        t1, t2 = _untraced_ms(offline, "TP1", batch), _untraced_ms(offline, "TP2", batch)
        if not tp1 or not tp2 or t1 is None or t2 is None:
            continue
        tp1_norm = float(np.mean([_mean(steps, "fused_add_rms_norm_ms") for steps in tp1]))
        tp2_norm = float(np.mean([_mean(steps, "fused_add_rms_norm_ms") for steps in tp2]))
        # TP2 without standalone norm kernels runs that work inside the fused all-reduce kernel.
        c1 = _components(tp1, t1, 0.0)
        c2 = _components(tp2, t2, tp1_norm if tp2_norm == 0 else 0.0)
        gap = t2 - t1 / 2
        out.append({"batch": batch, "component": "step", "label": LABELS["step"], "tp1_ms": t1, "tp2_ms": t2,
                    "half_tp1_ms": t1 / 2, "excess_ms": gap, "share": 1.0,
                    "tp1_traced_over_untraced": _traced_step_ms(tp1) / t1,
                    "tp2_traced_over_untraced": _traced_step_ms(tp2) / t2})
        for c in COMPONENTS:
            excess = c2[c] - c1[c] / 2
            out.append({"batch": batch, "component": c, "label": LABELS[c], "tp1_ms": c1[c], "tp2_ms": c2[c],
                        "half_tp1_ms": c1[c] / 2, "excess_ms": excess, "share": excess / gap if gap else None})
    return out
