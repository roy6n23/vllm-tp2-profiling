"""SLO attainment, goodput interpolation, the TP2/DP2 crossover s* and its request-level bootstrap.

Definitions follow spec 4.4 and AM8. TTFT/TPOT SLOs are inclusive, as in vLLM (D4-13).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from tpprof.constants import ATTAINMENT


@dataclass(frozen=True)
class RatePoint:
    rate: float                # offered req/s
    ttft_s: tuple[float, ...]  # successful requests only
    tpot_s: tuple[float, ...]


def _attainments(ttft_s, tpot_s, ttft_slo_s: float, tpot_slos_s: Sequence[float]) -> np.ndarray:
    """Attainment at each TPOT SLO; NaN for every SLO when there are no requests."""
    ttft, tpot = np.asarray(ttft_s, dtype=float), np.asarray(tpot_s, dtype=float)
    if ttft.shape != tpot.shape:
        raise ValueError(f"attainment: {ttft.size} TTFTs but {tpot.size} TPOTs")
    slos = np.asarray(tpot_slos_s, dtype=float)
    if not ttft.size:
        return np.full(slos.shape, np.nan)
    good = (ttft <= ttft_slo_s)[:, None] & (tpot[:, None] <= slos[None, :])
    return good.mean(axis=0)


def attainment(ttft_s, tpot_s, ttft_slo_s: float, tpot_slo_s: float) -> float:
    """Fraction of requests with TTFT <= ttft_slo_s and TPOT <= tpot_slo_s (inclusive); NaN if empty."""
    return float(_attainments(ttft_s, tpot_s, ttft_slo_s, [tpot_slo_s])[0])


def _goodput_from(rates: Sequence[float], atts: Sequence[float], target: float) -> tuple[float, str]:
    """rates ascending. A NaN attainment (no successful requests) counts as a failure."""
    fails = [not (a >= target) for a in atts]
    if not rates or fails[0]:
        return 0.0, "zero"
    if not any(fails):
        return float(rates[-1]), "lower_bound"
    j = fails.index(True)
    r0, r1, a0, a1 = rates[j - 1], rates[j], atts[j - 1], atts[j]
    a1 = 0.0 if np.isnan(a1) else a1
    return float(r0 + (a0 - target) / (a0 - a1) * (r1 - r0)), "ok"


def goodput(points: Sequence[RatePoint], ttft_slo_s: float, tpot_slo_s: float,
            target: float = ATTAINMENT) -> tuple[float, str]:
    """G(c, s), AM8: the interpolated first down-crossing of attainment == target over ascending rates.

    Returns (rate, flag): flag "lower_bound" when every rate passes (G >= top rate), "zero" when the lowest
    rate fails (or there are no points), else "ok".
    """
    pts = sorted(points, key=lambda p: p.rate)
    atts = [attainment(p.ttft_s, p.tpot_s, ttft_slo_s, tpot_slo_s) for p in pts]
    return _goodput_from([p.rate for p in pts], atts, target)


def crossover(slos_ms: Sequence[float], g_tp2: Sequence[float], g_dp2: Sequence[float]) -> float | None:
    """s*, AM8: the first sign change of G(DP2) - G(TP2) from < 0 to >= 0 along ascending SLOs, linearly
    interpolated on the SLO axis. SLOs where both goodputs are 0 are ignored. None if there is no change."""
    kept = sorted((float(s), float(b) - float(a)) for s, a, b in zip(slos_ms, g_tp2, g_dp2) if max(a, b) > 0)
    for (s0, d0), (s1, d1) in zip(kept, kept[1:]):
        if d0 < 0 <= d1:
            return s0 + (0.0 - d0) / (d1 - d0) * (s1 - s0)
    return None


def _curve(triples, ttft_slo_s: float, tpot_slos_ms: Sequence[float]) -> list[tuple[float, str]]:
    """goodput() at every TPOT SLO from (rate, ttfts, tpots) triples, one attainment pass per rate."""
    triples = sorted(triples, key=lambda t: t[0])
    slos_s = [s / 1000.0 for s in tpot_slos_ms]
    per_rate = [_attainments(ttft, tpot, ttft_slo_s, slos_s) for _, ttft, tpot in triples]
    rates = [t[0] for t in triples]
    return [_goodput_from(rates, [a[k] for a in per_rate], ATTAINMENT) for k in range(len(slos_s))]


def goodput_rows(points_by_config: Mapping[str, Sequence[RatePoint]], gpus: Mapping[str, int],
                 tpot_slos_ms: Sequence[float], ttft_slo_s: float) -> list[dict]:
    """One row per (config, TPOT SLO): goodput in req/s, per-GPU goodput and the goodput flag."""
    rows = []
    for config, points in points_by_config.items():
        curve = _curve([(p.rate, p.ttft_s, p.tpot_s) for p in points], ttft_slo_s, tpot_slos_ms)
        for slo_ms, (g, flag) in zip(tpot_slos_ms, curve):
            rows.append({"config": config, "tpot_slo_ms": slo_ms, "ttft_slo_s": ttft_slo_s,
                         "goodput_rps": g, "per_gpu_rps": g / gpus[config], "flag": flag})
    return rows


def _resampled_s_star(arrays: Mapping[str, list], tpot_slos_ms: Sequence[float], ttft_slo_s: float,
                      rng: np.random.Generator) -> float | None:
    curves = {}
    for config in ("TP2", "DP2"):
        triples = []
        for rate, ttft, tpot in arrays[config]:
            idx = rng.integers(0, ttft.size, size=ttft.size) if ttft.size else slice(None)
            triples.append((rate, ttft[idx], tpot[idx]))
        curves[config] = [g for g, _ in _curve(triples, ttft_slo_s, tpot_slos_ms)]
    return crossover(tpot_slos_ms, curves["TP2"], curves["DP2"])


def bootstrap_s_star(points_by_config: Mapping[str, Sequence[RatePoint]], tpot_slos_ms: Sequence[float],
                     ttft_slo_s: float, n_boot: int = 500, seed: int = 0) -> tuple[float | None, float | None]:
    """Request-level bootstrap of s* (AM8): every RatePoint of TP2 and DP2 has its requests resampled with
    replacement (TTFT/TPOT pairs kept together), s* is recomputed, and the 2.5/97.5 percentiles of the
    replicates that have a crossover are returned. (None, None) if TP2 or DP2 is missing or no replicate
    has a crossover."""
    if "TP2" not in points_by_config or "DP2" not in points_by_config:
        return None, None
    arrays = {c: [(p.rate, np.asarray(p.ttft_s, dtype=float), np.asarray(p.tpot_s, dtype=float))
                  for p in points_by_config[c]] for c in ("TP2", "DP2")}
    rng = np.random.default_rng(seed)
    stars = [s for s in (_resampled_s_star(arrays, tpot_slos_ms, ttft_slo_s, rng) for _ in range(n_boot))
             if s is not None]
    if not stars:
        return None, None
    lo, hi = np.percentile(stars, [2.5, 97.5])
    return float(lo), float(hi)
