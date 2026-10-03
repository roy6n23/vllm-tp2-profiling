from __future__ import annotations

from dataclasses import replace

import pytest

from tpprof import model, posteriori
from tpprof.constants import DECODE_BATCHES

CTX = model.DECODE_MEAN_CTX
# A box that differs from every a-priori set in the four constants the stage refits.
TRUE = replace(model.CONSTANTS["central"], name="true", bw_eff=2.5e12, t_fixed=0.6e-3, t_extra_tp2=0.0,
               alpha=4.5e-6, beta=1.8e11)
NCCL = replace(TRUE, alpha_nccl_graph=12e-6, beta=3.1e11)
M2 = "flashinfer_{}_fused_allreduce_rmsnorm_oneshot"


def _decode(config: str, arm: str, batch: int, step_s: float, derived: bool = False) -> dict:
    return {"config": config, "arm": arm, "kind": "decode", "batch": batch, "step_s": step_s, "t_ms": step_s * 1e3,
            "ctx_mean": float(CTX), "derived": derived}


def _tp1(c: model.Constants = TRUE, batches=DECODE_BATCHES) -> list[dict]:
    return [_decode("TP1", "base", b, model.decode_step_time(1, b, CTX, c)) for b in batches]


def _tp2(extra_s: float = 0.0) -> list[dict]:
    return [_decode("TP2", "base", b, model.decode_step_time(2, b, CTX, TRUE) + extra_s) for b in DECODE_BATCHES]


def _fit(source: str, impl: str, alpha_us: float, beta_gbps: float, mode=None, variant=None) -> dict:
    return {"row": "fit", "source": source, "impl": impl, "mode": mode, "variant": variant, "n": 14,
            "alpha_us": alpha_us, "beta_GBps": beta_gbps}


def _comm() -> list[dict]:
    return [_fit("comm_m2", M2.format("mnnvl"), 4.5, 180.0),
            _fit("comm_m2", M2.format("trtllm"), 5.3, 176.0),
            _fit("comm_m2", "flashinfer_mnnvl_fused_allreduce_rmsnorm_twoshot", 5.0, 184.0),
            _fit("comm_m3", "torch_nccl", 27.0, 319.0, mode="eager", variant="none:none"),
            _fit("comm_m3", "torch_nccl", 12.0, 310.0, mode="graph", variant="none:none"),
            _fit("comm_m3", "torch_nccl", 12.5, 112.0, mode="graph", variant="ring:LL"),
            {"row": "point", "source": "comm_m2", "impl": M2.format("mnnvl"), "bytes": 8192, "lat_us": 4.5}]


def _tables(offline: list[dict], comm: list[dict] | None = None) -> dict:
    return {"offline_points": offline, "comm": _comm() if comm is None else comm}


def _constants(rows: list[dict]) -> dict[str, dict]:
    return {r["name"]: r for r in rows if r["row"] == "constant"}


def _points(rows: list[dict], config: str, arm: str = "base") -> dict[int, dict]:
    return {r["batch"]: r for r in rows if r["row"] == "point" and (r["config"], r["arm"]) == (config, arm)}


# ------------------------------------------------------------------------------------------ the TP1 fit

def test_tp1_fit_recovers_bandwidth_and_fixed_time():
    fit = posteriori.tp1_fit(_tp1())
    assert fit["bw_eff"] == pytest.approx(2.5e12, rel=1e-9)
    assert fit["t_fixed"] == pytest.approx(0.6e-3, rel=1e-9)
    assert fit["n"] == len(DECODE_BATCHES)
    assert fit["max_resid_frac"] == pytest.approx(0.0, abs=1e-9)


def test_tp1_fit_uses_only_measured_tp1_baseline_decode_points():
    noise = (_tp2(extra_s=1e-3) + [_decode("TP1", "G2", b, 0.02) for b in (1, 32)]
             + [_decode("DP2", "base", 2 * b, 0.5) for b in DECODE_BATCHES]
             + [_decode("TP1", "base", 4, 0.5, derived=True)]
             + [{"config": "TP1", "arm": "base", "kind": "prefill", "batch": 1, "step_s": None, "t_ms": 17.8,
                 "ctx_mean": None, "derived": False}])
    fit = posteriori.tp1_fit(_tp1() + noise)
    assert (fit["bw_eff"], fit["t_fixed"], fit["n"]) == (pytest.approx(2.5e12), pytest.approx(0.6e-3), 8)


def test_tp1_fit_reports_its_own_misfit():
    rows = _tp1()
    rows[3]["step_s"] *= 1.02                                 # one batch 2% off the line
    fit = posteriori.tp1_fit(rows)
    assert 0.005 < fit["max_resid_frac"] < 0.02


def test_tp1_fit_needs_two_batches_and_a_rising_line():
    assert posteriori.tp1_fit(_tp1(batches=(1,))) is None
    assert posteriori.tp1_fit([]) is None
    falling = [_decode("TP1", "base", b, 0.01 - 1e-5 * b) for b in DECODE_BATCHES]
    assert posteriori.tp1_fit(falling) is None


# ------------------------------------------------------------------------------------------ comm constants

def test_comm_constants_follow_the_engines_backend_and_the_nccl_graph_default():
    got = posteriori.comm_constants(_comm(), "mnnvl")
    assert (got["alpha"], got["beta"]) == (pytest.approx(4.5e-6), pytest.approx(180e9))
    assert got["impl"] == M2.format("mnnvl")
    assert (got["alpha_nccl_graph"], got["beta_nccl"]) == (pytest.approx(12e-6), pytest.approx(310e9))
    assert posteriori.comm_constants(_comm(), "trtllm")["alpha"] == pytest.approx(5.3e-6)


def test_comm_constants_are_none_without_a_fit():
    got = posteriori.comm_constants([], "mnnvl")
    assert got["alpha"] is None and got["beta"] is None and got["alpha_nccl_graph"] is None
    assert posteriori.comm_constants(_comm(), None)["alpha"] is None       # backend unknown: no M2 row is chosen
    nan = [_fit("comm_m2", M2.format("mnnvl"), 4.5, None)]                    # beta fit failed (too few large sizes)
    got = posteriori.comm_constants(nan, "mnnvl")
    assert got["alpha"] == pytest.approx(4.5e-6) and got["beta"] is None


# ------------------------------------------------------------------------------------------ the table

def test_constants_come_from_tp1_and_the_microbenchmarks():
    rows = posteriori.rows(_tables(_tp1() + _tp2()), "mnnvl")
    c = _constants(rows)
    assert c["bw_eff"]["value"] == pytest.approx(2.5e12) and c["bw_eff"]["apriori"] == 3.0e12
    assert c["t_fixed"]["value"] == pytest.approx(0.6e-3) and c["t_fixed"]["apriori"] == 0.8e-3
    assert c["alpha"]["value"] == pytest.approx(4.5e-6) and M2.format("mnnvl") in c["alpha"]["source"]
    assert c["beta"]["value"] == pytest.approx(180e9)
    assert c["alpha_nccl_graph"]["value"] == pytest.approx(12e-6) and c["beta_nccl"]["value"] == pytest.approx(310e9)
    assert c["t_extra_tp2"]["value"] == 0.0 and c["t_extra_tp2"]["apriori"] == 0.1e-3
    assert all(r["unit"] and r["source"] for r in c.values())


def test_tp2_is_an_out_of_sample_prediction():
    unexplained = 0.2e-3
    rows = posteriori.rows(_tables(_tp1() + _tp2(extra_s=unexplained)), "mnnvl")
    used = replace(TRUE, beta=180e9)                                    # the M2 beta of _comm()
    tp1, tp2 = _points(rows, "TP1"), _points(rows, "TP2")
    assert set(tp1) == set(tp2) == set(DECODE_BATCHES)
    for b in DECODE_BATCHES:
        assert tp1[b]["sample"] == "in" and tp2[b]["sample"] == "out"
        assert tp1[b]["residual_ms"] == pytest.approx(0.0, abs=1e-9)
        want = model.decode_step_time(2, b, CTX, used) * 1e3
        assert tp2[b]["posteriori_ms"] == pytest.approx(want)
        assert tp2[b]["measured_ms"] == pytest.approx(tp2[b]["posteriori_ms"] + tp2[b]["residual_ms"])
        assert tp2[b]["residual_frac"] == pytest.approx(tp2[b]["residual_ms"] / tp2[b]["measured_ms"])
        apriori = model.decode_step_time(2, b, CTX, model.CONSTANTS["central"]) * 1e3
        assert tp2[b]["apriori_ms"] == pytest.approx(apriori)
    assert tp2[1]["residual_ms"] == pytest.approx(0.2, abs=0.01)        # only the beta mismatch moves it
    # No TP2 measurement feeds a constant: other TP2 step times leave every constant and prediction as it was.
    other = posteriori.rows(_tables(_tp1() + _tp2(extra_s=5e-3)), "mnnvl")
    assert _constants(other) == _constants(rows)
    assert [r["posteriori_ms"] for r in other if r["row"] == "point"] == \
        [r["posteriori_ms"] for r in rows if r["row"] == "point"]


def test_ar3_is_predicted_with_the_nccl_constants():
    ar3 = [_decode("TP2", "AR3", b, model.decode_step_time(2, b, CTX, NCCL, "nccl_unfused")) for b in (1, 32, 128)]
    rows = posteriori.rows(_tables(_tp1() + _tp2() + ar3), "mnnvl")
    used = replace(TRUE, alpha_nccl_graph=12e-6, beta=310e9)
    got = _points(rows, "TP2", "AR3")
    assert set(got) == {1, 32, 128}
    for b, r in got.items():
        assert r["sample"] == "out"
        assert r["posteriori_ms"] == pytest.approx(model.decode_step_time(2, b, CTX, used, "nccl_unfused") * 1e3)
    assert got[1]["residual_ms"] == pytest.approx(0.0, abs=1e-3)
    assert abs(got[128]["residual_ms"]) < 0.01                           # 310 vs the true 311 GB/s


def test_arms_the_model_has_no_path_for_get_no_rows():
    extra = [_decode("TP2", "G2", 1, 0.0138), _decode("TP2", "AR1", 1, 0.0041), _decode("TP1", "G1", 1, 0.0086)]
    rows = posteriori.rows(_tables(_tp1() + _tp2() + extra), "mnnvl")
    assert {(r["config"], r["arm"]) for r in rows if r["row"] == "point"} == {("TP1", "base"), ("TP2", "base")}


def test_a_missing_microbenchmark_keeps_the_a_priori_constant_and_says_so():
    rows = posteriori.rows(_tables(_tp1() + _tp2(), comm=[]), "mnnvl")
    c = _constants(rows)
    central = model.CONSTANTS["central"]
    assert c["alpha"]["value"] == central.alpha and "a priori" in c["alpha"]["source"]
    assert c["beta"]["value"] == central.beta and "a priori" in c["beta"]["source"]
    assert c["bw_eff"]["value"] == pytest.approx(2.5e12)                 # the TP1 fit does not need them
    assert _points(rows, "TP2")[1]["posteriori_ms"] == pytest.approx(
        model.decode_step_time(2, 1, CTX, replace(TRUE, alpha=central.alpha, beta=central.beta)) * 1e3)


def test_no_tp1_fit_no_table():
    assert posteriori.rows(_tables(_tp2()), "mnnvl") == []
    assert posteriori.rows({}, None) == []


def test_constant_sets_rebuild_the_model_inputs_from_the_table():
    rows = posteriori.rows(_tables(_tp1() + _tp2()), "mnnvl")
    sets = posteriori.constant_sets(rows)
    assert set(sets) == set(model.AR_PATHS)
    assert sets["fused"].bw_eff == pytest.approx(2.5e12) and sets["fused"].beta == pytest.approx(180e9)
    assert sets["nccl_unfused"].beta == pytest.approx(310e9)
    assert sets["nccl_unfused"].alpha_nccl_graph == pytest.approx(12e-6)
    assert posteriori.constant_sets([]) is None
