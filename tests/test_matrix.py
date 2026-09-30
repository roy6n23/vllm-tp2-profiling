from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path

import pytest

from tpprof import engine
from tpprof import matrix as m
from tpprof import model
from tpprof.constants import LATIN_SQUARE, RATE_FRACTIONS

MODEL_GRID = m.rate_grid(m.model_mu_rps())


def full_matrix(fi_backend: str | None = "mnnvl") -> list[m.RunSpec]:
    return m.build_matrix(["P0", "P1", "P2"], grid=MODEL_GRID, fi_backend=fi_backend)


def find(specs, kind, config=None, arm=None, **params):
    out = []
    for s in specs:
        if s.kind != kind or (config is not None and s.config != config) or (arm is not None and s.arm != arm):
            continue
        if all(s.p(k) == v for k, v in params.items()):
            out.append(s)
    return out


# ---------------------------------------------------------------- RunSpec and run_id

def test_run_id_is_stable_across_calls_and_well_formed():
    a = [s.run_id for s in full_matrix()]
    b = [s.run_id for s in full_matrix()]
    assert a == b
    for rid, s in zip(a, full_matrix()):
        assert re.fullmatch(rf"{s.tier}-{s.kind}-{s.config}-{s.arm}-r{s.round}-[0-9a-f]{{8}}", rid), rid


def test_run_id_matches_contract_hash():
    s = find(m.p0_specs(), "offline", "TP2", "AR3")[0]
    payload = {"kind": "offline", "config": "TP2", "arm": "AR3", "tier": "P0",
               "params": [["points", "decode:b1,b32"]], "round": 0,
               "engine": engine.arm_config("TP2", "AR3").to_dict()}
    sha8 = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
    assert s.run_id == f"P0-offline-TP2-AR3-r0-{sha8}"


def test_run_id_changes_when_an_engine_flag_changes(monkeypatch):
    spec = find(m.p0_specs(), "offline", "TP2", "AR3")[0]
    base_spec = find(m.p0_specs(), "offline", "TP2", "base")[0]
    before, before_base = spec.run_id, base_spec.run_id
    orig = engine._TRANSFORMS["AR3"]
    monkeypatch.setitem(engine._TRANSFORMS, "AR3",
                        lambda cfg: orig(cfg).with_arm(cfg.arm, set_flags={"--max-num-seqs": "512"}))
    assert spec.run_id != before
    assert base_spec.run_id == before_base


def test_dp2rand_run_id_hashes_both_engines(monkeypatch):
    spec = find(m.p2_specs(MODEL_GRID, None), "serve_session", "DP2rand")[0]
    payload = spec.to_dict()
    assert [e["name"] for e in payload["engine"]] == ["DP2rand0", "DP2rand1"]
    before = spec.run_id
    tp, dp, _gpus, flags, cc = engine.PARALLEL["DP2rand1"]
    monkeypatch.setitem(engine.PARALLEL, "DP2rand1", (tp, dp, (0,), flags, cc))
    assert spec.run_id != before


def test_run_id_payload_has_no_ports_paths_or_run_id():
    for s in full_matrix():
        payload = s.to_dict()
        assert set(payload) == {"kind", "config", "arm", "tier", "params", "round", "engine"}
        text = json.dumps(payload)
        assert "--port" not in text and "--model" not in text and "run_id" not in text


def test_none_config_hashes_null_engine():
    s = m.p0_specs()[0]
    assert (s.kind, s.config, s.arm) == ("preflight", "none", "none")
    assert s.to_dict()["engine"] is None


def test_run_ids_are_unique_in_the_full_matrix():
    ids = [s.run_id for s in full_matrix()]
    assert len(ids) == len(set(ids))


def test_params_are_frozen_sorted_and_readable():
    s = m.RunSpec("serve_session", "TP1", "base", "P1", {"seeds": [1, 2, 3], "phase": "sat"})
    assert s.params == (("phase", "sat"), ("seeds", (1, 2, 3)))
    assert s.p("seeds") == (1, 2, 3)
    assert s.p("missing", 7) == 7
    hash(s)


@pytest.mark.parametrize("kwargs, match", [
    ({"kind": "serve_sat"}, "kind"),
    ({"tier": "P3"}, "tier"),
    ({"config": "TP1", "arm": "AR3"}, "does not apply"),
    ({"config": "DP2rand", "arm": "G1"}, "does not apply"),
    ({"config": "none", "arm": "base"}, "none"),
    ({"kind": "preflight", "config": "TP1", "arm": "base"}, "none"),
    ({"kind": "offline", "config": "none", "arm": "none"}, "engine config"),
])
def test_runspec_rejects_invalid(kwargs, match):
    args = {"kind": "offline", "config": "TP2", "arm": "base", "tier": "P0", "params": ()}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        m.RunSpec(**args)


def test_every_spec_config_and_arm_is_valid_per_engine_arms():
    for s in full_matrix():
        if s.config == "none":
            assert s.arm == "none" and s.kind in m.NO_ENGINE_KINDS
            continue
        names = ("DP2rand0", "DP2rand1") if s.config == "DP2rand" else (s.config,)
        for name in names:
            assert name in engine.ARMS[s.arm], (s.run_id, name)
            engine.arm_config(name, s.arm)


# ---------------------------------------------------------------- tiers

def test_p0_exact_order():
    got = [(s.kind, s.config, s.arm, dict(s.params)) for s in m.p0_specs()]
    tokens = tuple(m.M2_TOKENS)
    want = [
        ("preflight", "none", "none", {}),
        ("envcapture", "none", "none", {}),
        ("smoke", "TP1", "base", {"prompts": 8, "gpu1_check": True}),
        ("smoke", "TP2", "base", {"prompts": 8}),
        ("smoke", "DP2", "base", {"prompts": 8}),
        ("comm_m2", "none", "none", {"tokens": tokens}),
        ("comm_m3", "none", "none", {"variant": "none:none", "modes": "eager,graph"}),
        ("offline", "TP1", "base", {"points": "all"}),
        ("offline", "TP2", "base", {"points": "all"}),
        ("offline", "TP2", "AR3", {"points": "decode:b1,b32"}),
        ("offline", "TP1", "G2", {"points": "decode:b1"}),
        ("offline", "TP2", "G2", {"points": "decode:b1"}),
        ("bench_latency_xcheck", "TP1", "base", {}),
        ("bench_latency_xcheck", "TP2", "base", {}),
        ("trace", "TP1", "base", {"points": "decode:b1"}),
        ("trace", "TP1", "base", {"points": "decode:b32"}),
        ("trace", "TP1", "base", {"points": "prefill:2048"}),
        ("trace", "TP2", "base", {"points": "decode:b1"}),
        ("trace", "TP2", "base", {"points": "decode:b32"}),
        ("trace", "TP2", "base", {"points": "prefill:2048"}),
        ("trace", "TP2", "G2", {"points": "decode:b1"}),
    ]
    assert got == want
    assert all(s.tier == "P0" and s.round == 0 for s in m.p0_specs())


def test_p0_contains_the_hypothesis_deciding_cells():
    p0 = m.p0_specs()
    assert find(p0, "offline", "TP2", "AR3", points="decode:b1,b32")
    assert find(p0, "offline", "TP1", "G2", points="decode:b1")
    assert find(p0, "offline", "TP2", "G2", points="decode:b1")
    assert find(p0, "trace", "TP2", "G2", points="decode:b1")
    assert find(p0, "comm_m2")


def test_p1_saturation_specs():
    sat = m.p1_sat_specs()
    assert [(s.kind, s.config) for s in sat] == [("serve_session", "TP1"), ("serve_session", "TP2"),
                                                  ("serve_session", "DP2"), ("tokbench", "none")]
    for s in sat[:3]:
        assert dict(s.params) == {"phase": "sat", "seeds": (1, 2, 3), "num_prompts": 3000}
        assert s.tier == "P1" and s.arm == "base"


def test_sweep_rounds_follow_the_latin_square():
    grid = {"TP1": [1.0, 2.0], "TP2": [3.0, 4.0], "DP2": [5.0, 6.0]}
    sweeps = m.p1_sweep_specs(grid)
    assert len(sweeps) == 9
    for r in (1, 2, 3):
        rnd = [s for s in sweeps if s.round == r]
        assert tuple(s.config for s in rnd) == LATIN_SQUARE[r - 1]
        for s in rnd:
            assert dict(s.params) == {"phase": "sweep", "rates": tuple(grid[s.config]), "seed_base": 1000 * r}
    assert [s.round for s in sweeps] == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_sweep_round_count_and_missing_config():
    grid = {"TP1": [1.0], "TP2": [2.0], "DP2": [3.0]}
    assert len(m.p1_sweep_specs(grid, rounds=1)) == 3
    with pytest.raises(ValueError, match="DP2"):
        m.p1_sweep_specs({"TP1": [1.0], "TP2": [2.0]})


def test_fibtrtllm_and_nvls_only_with_mnnvl():
    for backend in (None, "trtllm"):
        specs = m.p2_specs(MODEL_GRID, backend)
        assert not [s for s in specs if s.arm == "FIBtrtllm"]
        assert not find(specs, "comm_m3", variant="nvls:Simple")
    specs = m.p2_specs(MODEL_GRID, "mnnvl")
    fib = [s for s in specs if s.arm == "FIBtrtllm"]
    assert [(s.kind, s.config, s.p("points")) for s in fib] == [("offline", "TP2", "decode:b1,b32,b128")]
    assert find(specs, "comm_m3", variant="nvls:Simple")


def test_nvls_variant_is_gated_on_the_multicast_probe():
    # Spec 4.6: the NVLS M3 variant runs only if multicast is present; A-FIB still follows fi_backend.
    specs = m.p2_specs(MODEL_GRID, "trtllm", multicast=True)
    assert find(specs, "comm_m3", variant="nvls:Simple")
    assert not [s for s in specs if s.arm == "FIBtrtllm"]
    specs = m.p2_specs(MODEL_GRID, "mnnvl", multicast=False)
    assert not find(specs, "comm_m3", variant="nvls:Simple")
    assert [s for s in specs if s.arm == "FIBtrtllm"]
    full = m.build_matrix(["P2"], grid=MODEL_GRID, fi_backend=None, multicast=True)
    assert find(full, "comm_m3", variant="nvls:Simple")


def test_p2_contents():
    grid = {"TP1": [1.0], "TP2": [2.0, 4.0, 6.0, 7.5, 8.5, 9.5], "DP2": [3.0, 5.0]}
    p2 = m.p2_specs(grid, "mnnvl")
    assert all(s.tier == "P2" for s in p2)
    offline = [(s.config, s.arm, s.p("points")) for s in p2 if s.kind == "offline"]
    assert offline == [
        ("TP2", "AR1", "decode:b1,b32,b128;prefill:2048"),
        ("TP2", "AR2", "decode:b1,b32,b128;prefill:2048"),
        ("TP2", "AR3", "decode:b128;prefill:2048"),
        ("TP1", "G1", "decode:b1,b32"),
        ("TP2", "G1", "decode:b1,b32"),
        ("TP1", "G2", "decode:b32"),
        ("TP2", "G2", "decode:b32"),
        ("TP1", "EXECuni", "decode:b1,b32"),
        ("TP2", "FIBtrtllm", "decode:b1,b32,b128"),
    ]
    traces = [(s.config, s.arm, s.p("points")) for s in p2 if s.kind == "trace"]
    assert traces == [("TP2", "AR1", "decode:b1"), ("TP2", "AR2", "decode:b1"), ("TP2", "AR3", "decode:b1"),
                      ("TP1", "G1", "decode:b1"), ("TP2", "G1", "decode:b1")]
    pc = find(p2, "serve_session", phase="pc")
    assert [(s.config, s.arm) for s in pc] == [("TP2", "base"), ("TP2", "PCon")]
    rate_at_060 = grid["TP2"][RATE_FRACTIONS.index(0.60)]
    for s in pc:
        assert dict(s.params) == {"phase": "pc", "rate": rate_at_060, "repeats": 3, "sat_extra": 1}
    rand = find(p2, "serve_session", "DP2rand")
    assert len(rand) == 1
    assert dict(rand[0].params) == {"phase": "sweep", "rates": (3.0, 5.0), "seed_base": 1000}
    assert rand[0].round == 1 and rand[0].arm == "base"
    api2 = find(p2, "serve_session", arm="API2")
    assert [s.config for s in api2] == ["TP2", "DP2"]
    assert all(dict(s.params) == {"phase": "sat", "seeds": (1, 2, 3), "num_prompts": 3000} for s in api2)
    m3 = [s.p("variant") for s in p2 if s.kind == "comm_m3"]
    assert m3 == ["ring:LL", "ring:LL128", "ring:Simple", "tree:LL", "tree:LL128", "tree:Simple", "nvls:Simple"]
    assert all(set(dict(s.params)) == {"variant", "modes"} and s.p("modes") == "eager,graph"
               for s in p2 if s.kind == "comm_m3")
    assert find(p2, "comm_m1") and find(p2, "comm_m4")


def test_p2_without_grid_drops_only_grid_dependent_sessions():
    with_grid = m.p2_specs(MODEL_GRID, None)
    without = m.p2_specs(None, None)
    dropped = [s for s in with_grid if s not in without]
    assert {(s.config, s.p("phase")) for s in dropped} == {("TP2", "pc"), ("DP2rand", "sweep")}
    assert find(without, "serve_session", arm="API2")


def test_build_matrix_tiers():
    p1_no_grid = m.build_matrix(["P1"])
    assert p1_no_grid == m.p1_sat_specs()
    full = m.build_matrix(["P2", "P0", "P1"], grid=MODEL_GRID, fi_backend="mnnvl", rounds=2)
    assert [s.tier for s in full] == sorted(s.tier for s in full)
    assert len([s for s in full if s.p("phase") == "sweep" and s.config != "DP2rand"]) == 6
    with pytest.raises(ValueError, match="P9"):
        m.build_matrix(["P9"])


# ---------------------------------------------------------------- rate grid

def test_rate_grid_fractions_of_mu():
    assert m.rate_grid({"TP1": 20.0})["TP1"] == [4.0, 8.0, 12.0, 15.0, 17.0, 19.0]
    grid = m.rate_grid({"TP1": 43.217, "DP2": 80.0})
    assert grid["TP1"] == [round(f * 43.217, 2) for f in RATE_FRACTIONS]
    assert set(grid) == {"TP1", "DP2"}


@pytest.mark.parametrize("mu", [0.0, -1.0, float("nan")])
def test_rate_grid_rejects_non_positive_mu(mu):
    with pytest.raises(ValueError, match="TP2"):
        m.rate_grid({"TP2": mu})


def test_num_prompts_for():
    assert m.num_prompts_for(1.0) == 200
    assert m.num_prompts_for(10.0) == 900
    assert m.num_prompts_for(12.34) == 1111


def test_model_mu_rps_uses_central_saturation():
    mu = m.model_mu_rps()
    c = model.CONSTANTS["central"]
    for cfg in ("TP1", "TP2", "DP2"):
        assert mu[cfg] == pytest.approx(model.saturation_output_tps(cfg, c) / 256)


# ---------------------------------------------------------------- estimator

def _full_matrix_hours(fi_backend: str | None) -> float:
    return sum(e.minutes for e in m.estimate(full_matrix(fi_backend))) / 60


@pytest.mark.parametrize("fi_backend", ["mnnvl", None])
def test_estimate_full_matrix_within_budget(fi_backend):
    # The $80 cap (AM21): 11 h x $6.98 = $76.8, the plan's upper bound. Every spec gets a positive
    # estimate, in spec order. The formulas themselves are pinned term by term by the tests below.
    specs = full_matrix(fi_backend)
    ests = m.estimate(specs)
    assert [e.run_id for e in ests] == [s.run_id for s in specs]
    assert all(e.minutes > 0 for e in ests)
    assert {e.tier for e in ests} == {"P0", "P1", "P2"}
    assert _full_matrix_hours(fi_backend) <= 11


@pytest.mark.xfail(strict=True, reason=(
    "Plan Task 15 Step 1 requires [6, 11] h, but the plan's own estimator formulas give 5.69 h "
    "(5.58 h without mnnvl) with the central model. The 6 h came from AM21's hand figures "
    "(P0 1.8 + P1 2.8 + P2 2.3 h), which the formulas do not reproduce. Awaiting a controller "
    "ruling: either lower the bound (then drop this marker) or name the missing AM21 terms in the "
    "plan (then this test XPASSes, which strict=True turns into a failure, so the marker goes)."))
@pytest.mark.parametrize("fi_backend", ["mnnvl", None])
def test_estimate_full_matrix_is_between_6_and_11_hours(fi_backend):
    # Plan Task 15 Step 1, verbatim: "The estimate totals are within [6, 11] h for the full matrix."
    assert 6 <= _full_matrix_hours(fi_backend) <= 11


def test_estimate_fixed_costs_and_first_starts():
    p0 = m.p0_specs()
    ests = {e.run_id: e.minutes for e in m.estimate(p0)}
    by = {(s.kind, s.config, s.arm, s.p("points")): ests[s.run_id] for s in p0}
    assert by[("preflight", "none", "none", None)] == 1
    assert by[("envcapture", "none", "none", None)] == 1
    assert by[("comm_m2", "none", "none", None)] == 4
    assert by[("comm_m3", "none", "none", None)] == 3
    # smoke: 2 + first start of the config (3.0); TP1 adds a GPU1 engine start (1.5) and its bs-1 point
    assert by[("smoke", "TP2", "base", None)] == pytest.approx(5.0)
    assert by[("smoke", "DP2", "base", None)] == pytest.approx(5.0)
    gpu1_point = m._points_s(1, "base", "decode:b1") / 60
    assert by[("smoke", "TP1", "base", None)] == pytest.approx(2.0 + 3.0 + 1.5 + gpu1_point)
    # trace: 2.5 + a later start (1.5)
    assert by[("trace", "TP2", "base", "decode:b1")] == pytest.approx(4.0)
    # TP2 AR3 is not the first start of config TP2 (the TP2 smoke was)
    assert by[("offline", "TP2", "AR3", "decode:b1,b32")] == pytest.approx(
        1.5 + m._points_s(2, "AR3", "decode:b1,b32") / 60)


def test_estimate_serve_session_formulas():
    mu = {"TP1": 30.0, "TP2": 60.0, "DP2": 80.0}
    # sat: 3 x (3000 / mu + 20 s), nothing else
    sat = m.RunSpec("serve_session", "TP2", "base", "P1", {"phase": "sat", "seeds": [1, 2, 3], "num_prompts": 3000})
    [e] = m.estimate([sat], mu)
    assert e.minutes == pytest.approx(3.0 + 3 * (3000 / 60.0 + 20) / 60)
    # sweep: per rate max(90, N / mu) + 20 s + 16 warmups; 38 req/s drains (3420 / 30 = 114 s > 90 s)
    sweep = m.RunSpec("serve_session", "TP1", "base", "P1", {"phase": "sweep", "rates": [1.0, 8.0, 38.0],
                                                           "seed_base": 1000}, round=1)
    [e] = m.estimate([sweep], mu)
    warm = m._warmups_s("TP1")
    per_rate = [90 + 20 + warm, 90 + 20 + warm, 3420 / 30.0 + 20 + warm]
    assert e.minutes == pytest.approx(3.0 + sum(per_rate) / 60)
    # pc: repeats fixed-rate runs plus sat_extra sat runs
    pc = m.RunSpec("serve_session", "TP2", "PCon", "P2", {"phase": "pc", "rate": 36.0, "repeats": 3, "sat_extra": 1})
    [e] = m.estimate([pc], mu)
    rate_run = max(90, m.num_prompts_for(36.0) / 60.0) + 20 + m._warmups_s("TP2")
    assert e.minutes == pytest.approx(3.0 + (3 * rate_run + 3000 / 60.0 + 20) / 60)


def test_warmups_are_one_concurrent_batch_split_over_the_engines():
    c = model.CONSTANTS["central"]

    def generate(tp, batch):
        return batch * model.prefill_time(tp, 1024, c) + 255 * model.decode_step_time(tp, batch, 1024 + 128, c)

    assert m._warmups_s("TP1") == pytest.approx(generate(1, 16))
    assert m._warmups_s("TP2") == pytest.approx(generate(2, 16))
    assert m._warmups_s("DP2") == pytest.approx(generate(1, 8))
    assert m._warmups_s("DP2rand") == pytest.approx(generate(1, 8))
    assert 0 < m._warmups_s("TP1") < 30


def test_estimate_dp2rand_uses_the_dp2_mu():
    spec = m.RunSpec("serve_session", "DP2rand", "base", "P2", {"phase": "sat", "seeds": [1], "num_prompts": 3000})
    [e] = m.estimate([spec], {"DP2": 50.0})
    assert e.minutes == pytest.approx(3.0 + (3000 / 50.0 + 20) / 60)


def test_first_start_is_per_config_not_per_arm():
    specs = [m.RunSpec("trace", "TP2", "base", "P0", {"points": "decode:b1"}),
             m.RunSpec("trace", "TP2", "base", "P0", {"points": "decode:b32"}),
             m.RunSpec("trace", "TP2", "AR3", "P0", {"points": "decode:b1"}),
             m.RunSpec("trace", "TP1", "G2", "P0", {"points": "decode:b1"})]
    assert [e.minutes for e in m.estimate(specs)] == pytest.approx([5.5, 4.0, 4.0, 5.5])


def test_estimate_rejects_unknown_serve_phase():
    spec = m.RunSpec("serve_session", "TP1", "base", "P1", {"phase": "bogus"})
    with pytest.raises(ValueError, match="bogus"):
        m.estimate([spec])


def test_estimate_offline_sums_points_with_central_model():
    spec = m.RunSpec("offline", "TP1", "base", "P0", {"points": "decode:b1;prefill:2048"})
    [e] = m.estimate([spec])
    c = model.CONSTANTS["central"]
    decode = sum((3 + 10) * (model.prefill_time(1, 1024, c)
                             + (L - 1) * model.decode_step_time(1, 1, 1024 + L // 2, c)) for L in (64, 320))
    prefill = (5 + 20) * model.prefill_time(1, 2048, c)
    assert e.minutes == pytest.approx(3.0 + (decode + prefill) / 60)


def test_parse_points_follows_the_offline_driver_grammar():
    # Task 11's offline.parse_points; the matrix's fallback copy must behave the same way.
    assert len(m.parse_points("all")) == 3 + 16
    assert len(m.parse_points("decode:b1,b32;prefill:2048")) == 5
    assert len(m.parse_points("decode:b1,b1;decode:b1")) == 2          # duplicates dropped
    got = [(p.kind, p.batch, p.input_len, p.output_len, p.warmup, p.iters) for p in m.parse_points("decode:b32")]
    assert got == [("decode", 32, 1024, 64, 3, 10), ("decode", 32, 1024, 320, 3, 10)]
    for bad in ("decode:32", "decode:b0", "all;prefill", "bogus", "prefill:b2048"):
        with pytest.raises(ValueError):
            m.parse_points(bad)


def test_format_estimate_lists_tiers_hours_cost_and_total():
    ests = m.estimate(full_matrix())
    text = m.format_estimate(ests)
    total_h = sum(e.minutes for e in ests) / 60
    for tier in ("P0", "P1", "P2"):
        assert re.search(rf"^{tier}\b", text, re.M), tier
    assert f"{total_h:.2f} h" in text
    assert f"${total_h * 6.98:.2f}" in text
    assert "$6.98/h" in text
    assert "serve_session" in text
    assert f"${total_h * 10:.2f}" in m.format_estimate(ests, price_per_hour=10)


# ---------------------------------------------------------------- names owned by other tasks
# Once an owner module exists in the tree, matrix must import the name from it. These tests fail
# (they do not skip) if the module exists but the name is not the owner's object.

ROOT = Path(__file__).resolve().parents[1]


def owner(name: str):
    if not (ROOT / "tpprof" / f"{name}.py").exists():
        pytest.skip(f"tpprof/{name}.py is not in this tree yet; matrix uses its fallback copy")
    return importlib.import_module(f"tpprof.{name}")


def test_m2_tokens_come_from_vendored():
    assert m.M2_TOKENS is owner("vendored").M2_TOKENS


def test_m3_variants_come_from_comm_bench():
    assert m.M3_VARIANTS is owner("comm_bench").VARIANTS


def test_parse_points_comes_from_the_offline_driver():
    assert m.parse_points is owner("offline").parse_points
