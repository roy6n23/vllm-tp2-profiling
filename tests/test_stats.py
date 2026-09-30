from __future__ import annotations

import math

import numpy as np
import pytest

from tpprof import stats


def test_percentile_is_numpy_linear():
    assert stats.percentile([1, 2, 3, 4], 25) == 1.75
    assert stats.percentile(np.array([4.0, 1.0, 3.0, 2.0]), 50) == 2.5


def test_summarize_known_data():
    s = stats.summarize([1, 2, 3, 4])
    assert s == {"n": 4, "median": 2.5, "p25": 1.75, "p75": 3.25, "min": 1.0, "max": 4.0, "mean": 2.5}


def test_bootstrap_ci_is_seeded_and_brackets_the_statistic():
    xs = np.arange(1.0, 21.0)
    lo, hi = stats.bootstrap_ci(np.median, xs, n_boot=500, seed=3)
    assert (lo, hi) == stats.bootstrap_ci(np.median, list(xs), n_boot=500, seed=3)
    assert lo < np.median(xs) < hi
    assert stats.bootstrap_ci(np.median, [2.0] * 5) == (2.0, 2.0)


def test_decode_step_from_lengths_constant_data():
    r = stats.decode_step_from_lengths([1.0] * 10, [1.256] * 10, 64, 320)
    assert r["step_s"] == pytest.approx(0.001)
    assert r["ci_lo_s"] == pytest.approx(0.001) and r["ci_hi_s"] == pytest.approx(0.001)
    assert r["ci_hi_s"] - r["ci_lo_s"] == pytest.approx(0.0, abs=1e-15)
    assert (r["n1"], r["n2"]) == (10, 10)


def test_decode_step_ci_brackets_point_on_noisy_data():
    rng = np.random.default_rng(7)
    l1 = 1.0 + rng.normal(0, 0.01, 10)
    l2 = 1.256 + rng.normal(0, 0.01, 10)
    r = stats.decode_step_from_lengths(l1, l2, 64, 320, n_boot=1000, seed=0)
    assert r["ci_lo_s"] < r["step_s"] < r["ci_hi_s"]
    assert r["step_s"] == pytest.approx((np.median(l2) - np.median(l1)) / 256)


def test_diff_ci_point_is_b_minus_a():
    point, lo, hi = stats.diff_ci([1.0, 1.0, 1.0], [3.0, 3.0, 3.0])
    assert (point, lo, hi) == (2.0, 2.0, 2.0)
    point, lo, hi = stats.diff_ci([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 9.0], stat=np.mean, n_boot=300)
    assert point == pytest.approx(4.25)
    assert lo <= point <= hi


def test_token_emission_times():
    t = stats.token_emission_times([0.0, 10.0, 5.0], [1.0, 0.5, 9.0], [[1.0, 2.0], [0.25], [1.0]],
                                   [True, True, False])
    assert list(t) == [1.0, 2.0, 4.0, 10.5, 10.75]


def test_saturation_tps_token_timeline():
    # tokens at 1, 2, 3, 4 for both requests; lo/hi 0/1 covers [1, 4] -> 8 tokens over 3 s
    tps = stats.saturation_tps([0.0, 0.0], [1.0, 1.0], [[1, 1, 1], [1, 1, 1]], [4, 4], [True, True], lo=0.0, hi=1.0)
    assert tps == pytest.approx(8 / 3)


def test_saturation_tps_bundled_chunks_count_every_token():
    # output_len 4 but only 2 chunks (ttft at 1, one ITL of 3 s): the 2 missing tokens are spread over [1, 4]
    tps = stats.saturation_tps([0.0], [1.0], [[3.0]], [4], [True], lo=0.0, hi=1.0)
    assert tps == pytest.approx(4 / 3)


def test_saturation_tps_excludes_failed_and_trims_window():
    # 11 tokens of one request at t = 1..11; failed request would add tokens far away
    tps = stats.saturation_tps([0.0, 0.0], [1.0, 50.0], [[1.0] * 10, [1.0]], [11, 2], [True, False])
    # q10 = 2, q90 = 10 -> 9 tokens over 8 s
    assert tps == pytest.approx(9 / 8)


def test_uniform_throughput_two_requests():
    # (10 + 20) tokens / (max(0+2, 1+3) - min(0, 1)) = 30 / 4
    assert stats.uniform_throughput([0.0, 1.0], [2.0, 3.0], [10, 20], [True, True]) == 7.5
    # a failed request is ignored entirely
    assert stats.uniform_throughput([0.0, 1.0, 0.0], [2.0, 3.0, 99.0], [10, 20, 500], [True, True, False]) == 7.5


def test_fit_alpha_beta_recovers_synthetic_constants():
    sizes = [2**k for k in range(3, 31)]           # 8 B .. 1 GiB
    times = [5e-6 + s / 2.6e11 for s in sizes]
    alpha, beta = stats.fit_alpha_beta(sizes, times)
    assert alpha == pytest.approx(5e-6, rel=0.05)
    assert beta == pytest.approx(2.6e11, rel=0.01)


@pytest.mark.parametrize("call", [
    lambda: stats.percentile([], 50),
    lambda: stats.bootstrap_ci(np.median, [])[0],
    lambda: stats.decode_step_from_lengths([], [1.0], 64, 320)["step_s"],
    lambda: stats.diff_ci([], [1.0])[0],
    lambda: stats.saturation_tps([], [], [], [], []),
    lambda: stats.uniform_throughput([], [], [], []),
    lambda: stats.fit_alpha_beta([], [])[0],
    lambda: stats.fit_alpha_beta([], [])[1],
    lambda: stats.summarize([])["median"],
])
def test_empty_input_is_nan(call):
    assert math.isnan(call())


def test_empty_emission_timeline_and_summary():
    assert stats.token_emission_times([], [], [], []).size == 0
    assert stats.summarize([])["n"] == 0


def test_decode_step_rejects_non_increasing_lengths():
    with pytest.raises(ValueError, match="l2 > l1"):
        stats.decode_step_from_lengths([1.0], [1.0], 320, 64)
