from __future__ import annotations

import numpy as np
import pytest

from tpprof import goodput as g


def _point(rate: float, attain: float, n: int = 20, tpot: float = 0.010) -> g.RatePoint:
    """n requests, the first round(attain * n) meet TTFT 1 s / TPOT `tpot`, the rest miss TPOT."""
    good = round(attain * n)
    return g.RatePoint(rate, tuple([0.1] * n), tuple([tpot] * good + [tpot * 10] * (n - good)))


def test_attainment_boundary_is_inclusive():
    assert g.attainment([1.0, 0.5], [0.010, 0.010], 1.0, 0.010) == 1.0
    assert g.attainment([1.0, 1.01], [0.010, 0.001], 1.0, 0.010) == 0.5
    assert g.attainment([0.1, 0.1], [0.0101, 0.001], 1.0, 0.010) == 0.5


def test_goodput_interpolates_first_down_crossing():
    pts = [_point(3, 0.5), _point(1, 1.0), _point(2, 0.95)]   # unsorted on purpose
    rate, flag = g.goodput(pts, 1.0, 0.010)
    assert flag == "ok"
    assert rate == pytest.approx(2 + (0.95 - 0.9) / (0.95 - 0.5))
    assert rate == pytest.approx(2.111, abs=1e-3)


def test_goodput_lower_bound_when_all_pass():
    assert g.goodput([_point(1, 1.0), _point(2, 0.9), _point(3, 0.95)], 1.0, 0.010) == (3, "lower_bound")


def test_goodput_zero_when_first_rate_fails():
    assert g.goodput([_point(1, 0.85), _point(2, 1.0)], 1.0, 0.010) == (0.0, "zero")


def test_goodput_empty_and_request_less_points():
    assert g.goodput([], 1.0, 0.010) == (0.0, "zero")
    # a rate point with no successful requests has attainment 0: 1 + (1.0 - 0.9) / (1.0 - 0) = 1.1
    assert g.goodput([_point(1, 1.0), g.RatePoint(2, (), ())], 1.0, 0.010) == (pytest.approx(1.1), "ok")


def test_crossover_equal_point_counts():
    assert g.crossover([5, 10, 15, 20], [0, 5, 5, 5], [0, 3, 5, 8]) == 15.0


def test_crossover_interpolates_and_ignores_both_zero():
    # both zero at 5 is ignored; d = -2 at 10, +2 at 15 -> 12.5
    assert g.crossover([5, 10, 15], [0, 4, 4], [0, 2, 6]) == 12.5


def test_crossover_none_when_dp2_never_catches_up():
    assert g.crossover([5, 10, 15, 20], [0, 5, 6, 7], [0, 3, 4, 5]) is None
    assert g.crossover([5, 10], [0, 0], [0, 0]) is None
    assert g.crossover([], [], []) is None


def test_goodput_rows_per_gpu_and_flags():
    pts = {"TP1": [_point(1, 1.0), _point(2, 0.95), _point(3, 0.5)],
           "DP2": [_point(1, 1.0), _point(2, 1.0)]}
    rows = g.goodput_rows(pts, {"TP1": 1, "DP2": 2}, [5, 10], 1.0)
    assert [(r["config"], r["tpot_slo_ms"]) for r in rows] == [("TP1", 5), ("TP1", 10), ("DP2", 5), ("DP2", 10)]
    tp1_10 = rows[1]
    assert tp1_10["ttft_slo_s"] == 1.0 and tp1_10["flag"] == "ok"
    assert tp1_10["goodput_rps"] == pytest.approx(2.111, abs=1e-3)
    assert tp1_10["per_gpu_rps"] == tp1_10["goodput_rps"]
    assert rows[0]["flag"] == "zero" and rows[0]["goodput_rps"] == 0.0
    assert rows[3] == {"config": "DP2", "tpot_slo_ms": 10, "ttft_slo_s": 1.0,
                       "goodput_rps": 2, "per_gpu_rps": 1.0, "flag": "lower_bound"}


def _synthetic_crossover() -> dict[str, list[g.RatePoint]]:
    """TP2 has ~8 ms TPOT but its TTFT blows up above 2 req/s; DP2 has ~13 ms TPOT and fine TTFT."""
    rng = np.random.default_rng(11)
    n = 200
    tp2, dp2 = [], []
    for rate in (1.0, 2.0, 3.0, 4.0):
        ttft_bad = 0.0 if rate <= 2 else 0.5
        ttft_tp2 = np.where(rng.random(n) < ttft_bad, 2.0, 0.2)
        tp2.append(g.RatePoint(rate, tuple(ttft_tp2), tuple(rng.normal(0.008, 0.0005, n))))
        dp2.append(g.RatePoint(rate, tuple([0.2] * n), tuple(rng.normal(0.013, 0.0005, n))))
    return {"TP2": tp2, "DP2": dp2, "TP1": tp2[:1]}


def test_bootstrap_s_star_brackets_synthetic_crossover():
    pts = _synthetic_crossover()
    slos = [5, 10, 15, 20]
    point = g.crossover(slos, [g.goodput(pts["TP2"], 1.0, s / 1000)[0] for s in slos],
                        [g.goodput(pts["DP2"], 1.0, s / 1000)[0] for s in slos])
    assert point is not None and 10 < point < 15
    lo, hi = g.bootstrap_s_star(pts, slos, 1.0, n_boot=100, seed=0)
    assert lo is not None and hi is not None
    assert 10 <= lo <= hi <= 15
    assert (lo, hi) == g.bootstrap_s_star(pts, slos, 1.0, n_boot=100, seed=0)
    # the point estimate of the statistic the bootstrap resamples
    assert g.s_star(pts, slos, 1.0) == point
    assert g.s_star({"TP2": pts["TP2"]}, slos, 1.0) is None


def test_bootstrap_s_star_none_without_crossover_or_configs():
    pts = _synthetic_crossover()
    assert g.bootstrap_s_star(pts, [20, 30], 1.0, n_boot=20) == (None, None)
    assert g.bootstrap_s_star({"TP2": pts["TP2"]}, [5, 10, 15], 1.0, n_boot=20) == (None, None)


def test_bootstrap_s_star_tolerates_request_less_points():
    pts = _synthetic_crossover()
    pts["DP2"] = pts["DP2"] + [g.RatePoint(5.0, (), ())]
    lo, hi = g.bootstrap_s_star(pts, [5, 10, 15, 20], 1.0, n_boot=20)
    assert lo is not None and lo <= hi


def test_attainment_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="2 TTFTs but 1 TPOTs"):
        g.attainment([0.1, 0.2], [0.01], 1.0, 0.01)
