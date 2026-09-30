"""Summary statistics, seeded bootstrap CIs, throughput definitions and the alpha-beta fit.

Every function accepts lists or numpy arrays. Empty input gives ``float("nan")`` (spec 4.3, 4.4, AM6).
Percentiles are numpy's default linear interpolation throughout.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

NAN = float("nan")


def _arr(xs) -> np.ndarray:
    return np.asarray(xs, dtype=float).ravel()


def percentile(xs: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, identical to ``np.percentile(xs, p)``; p is 0-100."""
    a = _arr(xs)
    return float(np.percentile(a, p)) if a.size else NAN


def summarize(xs: Sequence[float]) -> dict:
    """{"n", "median", "p25", "p75", "min", "max", "mean"}; n == 0 gives NaN for every statistic."""
    a = _arr(xs)
    if not a.size:
        return {"n": 0, "median": NAN, "p25": NAN, "p75": NAN, "min": NAN, "max": NAN, "mean": NAN}
    p25, median, p75 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "median": float(median), "p25": float(p25), "p75": float(p75),
            "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean())}


def _interval(boots: np.ndarray, ci: float) -> tuple[float, float]:
    lo, hi = np.percentile(boots, [50 * (1 - ci), 50 * (1 + ci)])
    return float(lo), float(hi)


def bootstrap_ci(stat: Callable[[np.ndarray], float], xs, n_boot: int = 2000, seed: int = 0,
                 ci: float = 0.95) -> tuple[float, float]:
    """Percentile-interval bootstrap CI of ``stat(xs)``, resampling xs with replacement."""
    a = _arr(xs)
    if not a.size:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    return _interval(np.array([stat(a[i]) for i in idx], dtype=float), ci)


def diff_ci(xs_a, xs_b, stat: Callable[[np.ndarray], float] = np.median, n_boot: int = 2000,
            seed: int = 0, ci: float = 0.95) -> tuple[float, float, float]:
    """(stat(b) - stat(a), lo, hi): each array is resampled independently from one default_rng(seed)."""
    a, b = _arr(xs_a), _arr(xs_b)
    if not a.size or not b.size:
        return NAN, NAN, NAN
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    boots = np.array([stat(b[j]) - stat(a[i]) for i, j in zip(ia, ib)], dtype=float)
    lo, hi = _interval(boots, ci)
    return float(stat(b) - stat(a)), lo, hi


def decode_step_from_lengths(lat_l1, lat_l2, l1: int, l2: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """Two-length decode step (spec 4.3): (median(lat_l2) - median(lat_l1)) / (l2 - l1), with a bootstrap
    95% CI of the difference of medians (iterations resampled independently, fixed seed)."""
    if l2 <= l1:
        raise ValueError(f"decode_step_from_lengths needs l2 > l1, got l1={l1} l2={l2}")
    point, lo, hi = diff_ci(lat_l1, lat_l2, np.median, n_boot=n_boot, seed=seed)
    steps = l2 - l1
    return {"step_s": point / steps, "ci_lo_s": lo / steps, "ci_hi_s": hi / steps,
            "n1": int(_arr(lat_l1).size), "n2": int(_arr(lat_l2).size)}


def _ok_indices(ok_mask) -> np.ndarray:
    return np.flatnonzero(np.asarray(ok_mask, dtype=bool))


def _request_times(start: float, ttft: float, itl, output_len: int | None = None) -> np.ndarray:
    """Emission times of one request: start + ttft, then + cumsum(itl). With output_len, tokens that shared
    an SSE chunk (len(itl) + 1 < output_len, D4-14) are spread uniformly over the request's [ttft, latency]."""
    first = float(start) + float(ttft)
    times = first + np.concatenate(([0.0], np.cumsum(_arr(itl))))
    missing = 0 if output_len is None else int(output_len) - times.size
    if missing > 0:
        fill = first + (times[-1] - first) * np.arange(1, missing + 1) / (missing + 1)
        times = np.concatenate((times, fill))
    return times


def token_emission_times(start_times, ttfts, itls, ok_mask) -> np.ndarray:
    """AM6 token timeline of the successful requests, sorted: per request start + ttft, then + cumsum(itls)."""
    parts = [_request_times(start_times[i], ttfts[i], itls[i]) for i in _ok_indices(ok_mask)]
    return np.sort(np.concatenate(parts)) if parts else np.empty(0)


def saturation_tps(start_times, ttfts, itls, output_lens, ok_mask, lo: float = 0.10, hi: float = 0.90) -> float:
    """Saturation output throughput from the token-emission timeline (AM6).

    Counts tokens emitted in [q_lo, q_hi], the lo/hi quantiles of all emission times, divided by q_hi - q_lo.
    A request whose SSE chunks carried several tokens (len(itls) + 1 < output_len, D4-14) gets its missing
    tokens spread uniformly over its own [ttft, latency] interval; vLLM's client sets latency = ttft + sum(itls).
    """
    parts = [_request_times(start_times[i], ttfts[i], itls[i], output_lens[i]) for i in _ok_indices(ok_mask)]
    if not parts:
        return NAN
    times = np.concatenate(parts)
    q_lo, q_hi = np.percentile(times, [100 * lo, 100 * hi])
    if q_hi <= q_lo:
        return NAN
    return float(np.count_nonzero((times >= q_lo) & (times <= q_hi)) / (q_hi - q_lo))


def uniform_throughput(start_times, latencies, output_lens, ok_mask) -> float:
    """Output tok/s = sum(output_lens) / (max(start + latency) - min(start)) over successful requests (D4-28)."""
    ok = _ok_indices(ok_mask)
    if not ok.size:
        return NAN
    start = _arr(start_times)[ok]
    span = float(np.max(start + _arr(latencies)[ok]) - np.min(start))
    return float(np.sum(_arr(output_lens)[ok]) / span) if span > 0 else NAN


def fit_alpha_beta(sizes_bytes, times_s, small_max: int = 64 * 1024, large_min: int = 8 * 2**20) -> tuple[float, float]:
    """Latency-bandwidth fit of a collective sweep: alpha = median time over sizes <= small_max (s);
    beta = 1 / slope of the least-squares line t = a + s / beta over sizes >= large_min (bytes/s)."""
    s, t = _arr(sizes_bytes), _arr(times_s)
    if s.size != t.size:
        raise ValueError(f"fit_alpha_beta: {s.size} sizes but {t.size} times")
    small = s <= small_max
    alpha = float(np.median(t[small])) if small.any() else NAN
    large = s >= large_min
    if np.unique(s[large]).size < 2:
        return alpha, NAN
    scale = float(s[large].max())
    slope_scaled, _ = np.polyfit(s[large] / scale, t[large], 1)
    slope = slope_scaled / scale
    return alpha, (1.0 / slope if slope > 0 else NAN)
