"""Figures from the tidy tables (spec 7.6). matplotlib is optional (AM30): it is imported lazily with the
Agg backend, and make_figures returns [] when it is not installed. A figure whose inputs are missing is
skipped, so a partial results dir still gets every figure it has data for.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from tpprof import kernels, model
from tpprof.constants import PREFILL_LENS, TTFT_SLO_S

DPI = 150
FIGURES = ("decode_step_vs_batch.png", "efficiency_vs_tokens.png", "ttft_tpot_vs_rate.png", "throughput_vs_rate.png",
           "goodput_vs_slo.png", "allreduce_latency_vs_size.png", "trace_breakdown.png", "ar_ladder.png",
           "graphs_ablation.png", "tp2_gap_waterfall.png")
ONLINE_CONFIGS = ("TP1", "TP2", "DP2", "DP2rand")
DECODE_MARK_TOKENS = (1, 32, 128)          # AR message sizes marked on the all-reduce figure: 8 KiB x tokens
AR_LADDER = ("base", "AR1", "AR2", "AR3")
GRAPH_ARMS = ("base", "G1", "G2")

Tables = Mapping[str, Sequence[dict]]


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _median_by(rows: Sequence[dict], key: str, value: str) -> tuple[list[float], list[float]]:
    groups: dict[float, list[float]] = {}
    for r in rows:
        if r.get(key) is not None and r.get(value) is not None:
            groups.setdefault(r[key], []).append(r[value])
    xs = sorted(groups)
    return xs, [float(np.median(groups[x])) for x in xs]


def _decode_rows(t: Tables, config: str, arm: str, derived: bool = False) -> list[dict]:
    return [r for r in t.get("offline_points", []) if r.get("kind") == "decode" and r.get("config") == config
            and r.get("arm") == arm and bool(r.get("derived")) == derived and r.get("t_ms") is not None]


def _decode_step_vs_batch(plt, t: Tables, ax) -> bool:
    drawn = False
    for config, derived, label in (("TP1", False, "TP1"), ("TP2", False, "TP2"), ("DP2", True, "DP2 (derived)")):
        rows = _decode_rows(t, config, "base", derived)
        xs, ys = _median_by(rows, "batch", "t_ms")
        if not xs:
            continue
        line, = ax.plot(xs, ys, "o-", label=f"{label} measured")
        mx, my = _median_by(rows, "batch", "model_ms")
        if mx:
            _, lo = _median_by(rows, "batch", "model_lo_ms")
            _, hi = _median_by(rows, "batch", "model_hi_ms")
            ax.plot(mx, my, "--", color=line.get_color(), label=f"{label} model (central)")
            if len(lo) == len(mx) == len(hi):
                ax.fill_between(mx, lo, hi, color=line.get_color(), alpha=0.15)
        post = [r for r in t.get("posteriori", []) if r.get("row") == "point" and r.get("arm") == "base"
                and r.get("config") == config]
        px, py = _median_by(post, "batch", "posteriori_ms")
        if px:
            how = "fitted" if post[0].get("sample") == "in" else "out of sample"
            ax.plot(px, py, ":", color=line.get_color(), label=f"{label} a posteriori ({how})")
        drawn = True
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch (sequences)")
    ax.set_ylabel("decode step (ms)")
    ax.set_title("Decode step time vs batch (band: optimistic..pessimistic model)")
    return drawn


def _efficiency_vs_tokens(plt, t: Tables, ax) -> bool:
    drawn = False
    for config, derived, label in (("TP2", False, "TP2"), ("DP2", True, "DP2 (derived)")):
        for kind, marker in (("decode", "o-"), ("prefill", "s-")):
            rows = [r for r in t.get("offline_points", []) if r.get("config") == config and r.get("arm") == "base"
                    and r.get("kind") == kind and bool(r.get("derived")) == derived and r.get("efficiency") is not None]
            xs, ys = _median_by(rows, "tokens_per_step", "efficiency")
            if xs:
                ax.plot(xs, ys, marker, label=f"{label} {kind}")
                drawn = True
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tokens per step")
    ax.set_ylabel("scaling efficiency (speedup vs TP1 / GPUs)")
    ax.set_title("Two-GPU scaling efficiency vs tokens per step")
    return drawn


def _sweep_rows(t: Tables, config: str) -> list[dict]:
    return [r for r in t.get("online_runs", []) if r.get("phase") == "sweep" and r.get("config") == config
            and r.get("arm") == "base" and r.get("valid")]


def _scaled(ys: list[float], k: float) -> list[float]:
    return [y * k for y in ys]


def _ttft_tpot_vs_rate(plt, t: Tables, axes) -> bool:
    drawn = False
    for config in ONLINE_CONFIGS:
        rows = _sweep_rows(t, config)
        for ax, col in ((axes[0], "ttft_p50_s"), (axes[0], "ttft_p90_s"), (axes[1], "tpot_p50_s"),
                        (axes[1], "tpot_p90_s")):
            xs, ys = _median_by(rows, "rate_target", col)
            if xs:
                ax.plot(xs, _scaled(ys, 1e3), "o-" if "p50" in col else "x--", label=f"{config} {col[5:8]}")
                drawn = True
    axes[0].set_ylabel("TTFT (ms)")
    axes[1].set_ylabel("TPOT (ms)")
    for ax in axes:
        ax.set_xlabel("offered rate (req/s)")
    axes[0].set_title("TTFT vs rate (median over rounds)")
    axes[1].set_title("TPOT vs rate (median over rounds)")
    return drawn


def _throughput_vs_rate(plt, t: Tables, ax) -> bool:
    drawn = False
    for config in ONLINE_CONFIGS:
        xs, ys = _median_by(_sweep_rows(t, config), "rate_target", "tput_uniform_tps")
        if xs:
            ax.plot(xs, ys, "o-", label=config)
            drawn = True
    ax.set_xlabel("offered rate (req/s)")
    ax.set_ylabel("output throughput (tok/s)")
    ax.set_title("Output throughput vs offered rate (median over rounds)")
    return drawn


def _goodput_vs_slo(plt, t: Tables, ax) -> bool:
    drawn = False
    for config in ONLINE_CONFIGS:
        rows = [r for r in t.get("goodput", []) if r.get("config") == config and r.get("arm") == "base"
                and r.get("ttft_slo_s") == TTFT_SLO_S]
        xs, ys = _median_by(rows, "tpot_slo_ms", "goodput_rps")
        if xs:
            ax.plot(xs, ys, "o-", label=config)
            drawn = True
    star = next((r for r in t.get("s_star", []) if r.get("row") == "all"), None)
    if drawn and star and star.get("s_star_ms") is not None:
        ax.axvline(star["s_star_ms"], color="black", ls="--", lw=1, label=f"s* {star['s_star_ms']:.3g} ms")
        if star.get("s_star_min") is not None:
            ax.axvspan(star["s_star_min"], star["s_star_max"], color="grey", alpha=0.2)
    ax.set_xlabel("TPOT SLO (ms)")
    ax.set_ylabel(f"goodput (req/s, TTFT <= {TTFT_SLO_S:g} s)")
    ax.set_title("Goodput vs TPOT SLO (median over rounds)")
    return drawn


def _allreduce_latency_vs_size(plt, t: Tables, ax) -> bool:
    comm = t.get("comm", [])
    points = [r for r in comm if r.get("row") == "point" and r.get("lat_us") is not None]
    if not points:
        return False
    groups: dict[tuple, list[dict]] = {}
    for r in points:
        if r.get("variant") in (None, "none:none", "out_of_place"):   # defaults only; variants are in comm.csv
            groups.setdefault((r["source"], r["impl"], r.get("mode")), []).append(r)
    fits = {(r["source"], r["impl"], r.get("mode")): r for r in comm if r.get("row") == "fit"
            and r.get("variant") in (None, "none:none", "out_of_place")}
    for key, rows in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rows.sort(key=lambda r: r["bytes"])
        xs = [r["bytes"] for r in rows]
        line, = ax.plot(xs, [r["lat_us"] for r in rows], "o", ms=3, label=f"{key[0][5:]} {key[1]} {key[2] or ''}")
        fit = fits.get(key)
        if fit and fit.get("alpha_us") is not None and fit.get("beta_GBps"):
            s = np.geomspace(min(xs), max(xs), 50)
            ax.plot(s, fit["alpha_us"] + s / (fit["beta_GBps"] * 1e9) * 1e6, "-", color=line.get_color(), lw=0.8)
    for tokens, color in [(n, "tab:green") for n in DECODE_MARK_TOKENS] + [(n, "tab:red") for n in PREFILL_LENS]:
        ax.axvline(model.ar_message_bytes(tokens), color=color, ls=":", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("message size (bytes)\ndotted: 8 KiB x tokens, decode 1/32/128 (green), prefill 512/2048/8192 (red)")
    ax.set_ylabel("latency per op (us)")
    ax.set_title("All-reduce latency vs size, with alpha-beta fits")
    return True


def _trace_breakdown(plt, t: Tables, axes) -> bool:
    drawn = False
    for ax, kind in zip(axes, ("decode", "prefill")):
        rows = [r for r in t.get("trace_summary", []) if r.get("arm") == "base" and r.get("rank") == 0
                and r.get("gate_ok") is True and str(r.get("points", "")).startswith(kind)
                and r.get("mean_step_ms") is not None]
        if not rows:
            continue
        rows.sort(key=lambda r: (str(r.get("points")), r["config"]))
        labels = [f"{r['config']}\n{r.get('points')}" for r in rows]
        bottom = np.zeros(len(rows))
        for c in (*kernels.CATEGORIES, "idle"):
            vals = np.array([r.get("idle_ms" if c == "idle" else f"cat_{c}_ms") or 0.0 for r in rows])
            ax.bar(labels, vals, bottom=bottom, label=c)
            bottom += vals
        ax.set_ylabel("traced step time (ms), rank 0")
        ax.set_title(f"Per-step trace breakdown, {kind}, TP1 vs TP2 (baseline)")
        drawn = True
    return drawn


def _arm_bars(plt, t: Tables, ax, configs: Sequence[str], arms: Sequence[str], title: str) -> bool:
    cells = []
    for config in configs:
        for arm in arms:
            xs, ys = _median_by(_decode_rows(t, config, arm), "batch", "t_ms")
            cells += [(config, arm, b, y) for b, y in zip(xs, ys)]
    groups = sorted({(c, b) for c, _, b, _ in cells}, key=lambda g: (g[0], g[1]))
    if not groups:
        return False
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        vals = {(c, b): y for c, a, b, y in cells if a == arm}
        ax.bar([j + i * width for j in range(len(groups))], [vals.get(g, 0.0) for g in groups], width, label=arm)
    ax.set_xticks([j + width * (len(arms) - 1) / 2 for j in range(len(groups))])
    ax.set_xticklabels([f"{c} b{b}" for c, b in groups])
    ax.set_ylabel("decode step (ms)")
    ax.set_title(title)
    return True


def _ar_ladder(plt, t: Tables, ax) -> bool:
    return _arm_bars(plt, t, ax, ("TP2",), AR_LADDER, "All-reduce ladder: TP2 decode step by arm")


def _graphs_ablation(plt, t: Tables, ax) -> bool:
    return _arm_bars(plt, t, ax, ("TP1", "TP2"), GRAPH_ARMS, "CUDA-graph arms: decode step (G2 = eager)")


def _tp2_gap_waterfall(plt, t: Tables, ax) -> bool:
    """Horizontal waterfall of the smallest traced batch: half of TP1's step, one bar per component's excess
    (largest first), then TP2's measured step."""
    rows = t.get("tp2_gap", [])
    if not rows:
        return False
    batch = min(r["batch"] for r in rows)
    step = next(r for r in rows if r["batch"] == batch and r["component"] == "step")
    parts = sorted((r for r in rows if r["batch"] == batch and r["component"] != "step"),
                   key=lambda r: -r["excess_ms"])
    ax.figure.set_size_inches(9, 4.6)
    half, tp2 = step["half_tp1_ms"], step["tp2_ms"]
    labels = [f"half of TP1's step ({step['tp1_ms']:.2f} ms / 2)"]
    ax.barh(0, half, color="0.6")
    ax.text(half / 2, 0, f"{half:.2f} ms: what 2x would be", va="center", ha="center", fontsize=8, color="white")
    left = half
    for i, r in enumerate(parts, start=1):
        x = r["excess_ms"]
        ax.barh(i, x, left=left, color="tab:orange" if x >= 0 else "tab:green")
        ax.plot([left, left], [i - 1, i], color="0.4", lw=0.6, ls=":")
        share = "" if r["share"] is None else f" ({r['share']:.0%})"
        ax.text(max(left, left + x) + 0.03, i, f"{x:+.3f} ms{share}", va="center", fontsize=8)
        labels.append(f"+ {r['label']}")
        left += x
    n = len(parts) + 1
    ax.plot([left, left], [n - 1, n], color="0.4", lw=0.6, ls=":")
    ax.barh(n, tp2, color="tab:blue")
    ax.text(tp2 / 2, n, f"{tp2:.2f} ms measured", va="center", ha="center", fontsize=8, color="white")
    labels.append("TP2's step")
    ax.set_yticks(range(n + 1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(tp2, left) * 1.2)
    ax.set_xlabel(f"decode step time (ms); in brackets: share of the {step['excess_ms']:.2f} ms gap")
    ax.set_title(f"Why TP=2 is {step['tp1_ms'] / tp2:.2f}x and not 2x: decode step, batch {batch}", fontsize=10)
    return True


_DRAW: dict[str, tuple[Callable, int]] = {
    "decode_step_vs_batch.png": (_decode_step_vs_batch, 1),
    "efficiency_vs_tokens.png": (_efficiency_vs_tokens, 1),
    "ttft_tpot_vs_rate.png": (_ttft_tpot_vs_rate, 2),
    "throughput_vs_rate.png": (_throughput_vs_rate, 1),
    "goodput_vs_slo.png": (_goodput_vs_slo, 1),
    "allreduce_latency_vs_size.png": (_allreduce_latency_vs_size, 1),
    "trace_breakdown.png": (_trace_breakdown, 2),
    "ar_ladder.png": (_ar_ladder, 1),
    "graphs_ablation.png": (_graphs_ablation, 1),
    "tp2_gap_waterfall.png": (_tp2_gap_waterfall, 1),
}


def make_figures(tables: Tables, out_dir: str) -> list[str]:
    """Writes each figure that has data to out_dir as a 150-dpi PNG; returns the paths written."""
    plt = _pyplot()
    if plt is None:
        return []
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name in FIGURES:
        draw, ncols = _DRAW[name]
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5))
        try:
            if draw(plt, tables, axes):
                for ax in np.atleast_1d(axes):
                    if ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                fig.tight_layout()
                path = os.path.join(out_dir, name)
                fig.savefig(path, dpi=DPI)
                paths.append(path)
        finally:
            plt.close(fig)
    return paths
