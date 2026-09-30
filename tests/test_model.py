from __future__ import annotations

import pytest
from tpprof import model as m

D8 = m.CONSTANTS["d8"]

def test_param_and_byte_counts():
    assert m.param_count() == 8_030_261_248
    assert m.weight_bytes(1) == 16_060_522_496
    assert m.weight_bytes(2) == 8_030_527_488
    assert m.streamed_weight_bytes(1) == 15_009_849_344
    assert m.streamed_weight_bytes(2) == 7_505_190_912
    assert m.kv_bytes_per_token(1) == 131_072 and m.kv_bytes_per_token(2) == 65_536
    assert m.linear_flops_per_token() == 13_958_643_712
    assert m.linear_flops_per_token(include_lm_head=True) == 15_009_316_864
    assert m.allreduces_per_step(2) == 65 and m.allreduces_per_step(1) == 0
    assert m.allgathers_per_step(2) == 1 and m.ar_message_bytes(1) == 8192

@pytest.mark.parametrize("batch,tp1_ms,tp2_ms", [(1, 5.85, 3.76), (8, 6.16, 3.93), (32, 7.23, 4.51), (128, 11.53, 6.84)])
def test_decode_reproduces_d8_oracle(batch, tp1_ms, tp2_ms):
    assert m.decode_step_time(1, batch, 1024, D8) * 1e3 == pytest.approx(tp1_ms, abs=0.01)
    assert m.decode_step_time(2, batch, 1024, D8) * 1e3 == pytest.approx(tp2_ms, abs=0.01)

@pytest.mark.parametrize("n,tp1_ms,tp2_ms", [(512, 12.15, 7.86), (2048, 48.26, 28.78), (8192, 223.61, 127.90)])
def test_prefill_reproduces_d8_oracle(n, tp1_ms, tp2_ms):
    assert m.prefill_time(1, n, D8) * 1e3 == pytest.approx(tp1_ms, abs=0.01)
    assert m.prefill_time(2, n, D8) * 1e3 == pytest.approx(tp2_ms, abs=0.01)

def test_kv_capacity_reproduces_d8_oracle():
    t1, t2 = m.kv_capacity_tokens(1, D8), m.kv_capacity_tokens(2, D8)
    assert t1 == pytest.approx(421_000, rel=0.005) and t2 == pytest.approx(955_000, rel=0.005)
    assert t2 / t1 == pytest.approx(2.269, abs=0.003)

def test_saturation_reproduces_d8_oracle_at_d8_limits():
    # D8 used 330 running requests for TP1 and 512 for TP2
    assert m.saturation_output_tps("TP1", D8, max_num_seqs=330) == pytest.approx(6040, rel=0.01)
    assert m.saturation_output_tps("DP2", D8, max_num_seqs=330) == pytest.approx(12090, rel=0.01)
    assert m.saturation_output_tps("TP2", D8, max_num_seqs=512) == pytest.approx(10690, rel=0.01)

def test_running_limit_kv_vs_max_num_seqs():
    c = m.CONSTANTS["central"]
    assert m.running_limit("TP1", c) == m.kv_capacity_tokens(1, c) // 1280       # KV-limited, < 1024
    assert m.running_limit("TP2", c) == min(1024, m.kv_capacity_tokens(2, c) // 1280)

def test_nccl_unfused_path_is_slower():
    c = m.CONSTANTS["central"]
    fused, nccl = m.decode_step_time(2, 1, 1152, c), m.decode_step_time(2, 1, 1152, c, ar_path="nccl_unfused")
    assert 1.02 < nccl / fused < 1.25

def test_tpot_at_rate_monotone_and_none_past_saturation():
    c = m.CONSTANTS["central"]
    mu = m.saturation_output_tps("TP2", c) / 256
    a, b = m.tpot_at_rate("TP2", 0.2 * mu, c), m.tpot_at_rate("TP2", 0.8 * mu, c)
    assert a is not None and b is not None and a < b
    assert m.tpot_at_rate("TP2", 1.05 * mu, c) is None

def test_s_star_exists_for_central():
    s = m.s_star(m.CONSTANTS["central"])
    assert s is not None and 5 < s < 60

def test_predictions_has_bands_for_every_hypothesis():
    p = m.predictions()
    for h in ("H1", "H2", "H3", "H4", "H5", "H8"):
        lo, hi = p["bands"][h]
        assert lo <= hi


def _run_d8_script(name: str) -> dict:
    import contextlib
    import io
    from tests.conftest import FIXTURES
    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec((FIXTURES / "d8" / name).read_text(), ns)
    return ns

def test_matches_d8_reference_functions_to_float_precision():
    # Beyond the 2-decimal oracle: D8's own decode(), prefill() and sat.py step() agree to float precision.
    d8m, d8s = _run_d8_script("model.py"), _run_d8_script("sat.py")
    for batch in (1, 8, 32, 128):
        for ctx in (1024, 1152):
            t1, t2, _ = d8m["decode"](batch, ctx, 3.0e12, 0.8e-3, 0.1e-3, 5e-6, 350e9)
            assert m.decode_step_time(1, batch, ctx, D8) == pytest.approx(t1, rel=1e-12)
            assert m.decode_step_time(2, batch, ctx, D8) == pytest.approx(t2, rel=1e-12)
    for n in (512, 2048, 8192):
        t1, t2, _ = d8m["prefill"](n, 700e12, 400e12, 3.0e12, 0.8e-3, 5e-6, 350e9)
        assert m.prefill_time(1, n, D8) == pytest.approx(t1, rel=1e-12)
        assert m.prefill_time(2, n, D8) == pytest.approx(t2, rel=1e-12)
    _, tps1 = d8s["step"](m.running_limit("TP1", D8), 1)
    _, tps2 = d8s["step"](512, 2)
    assert m.saturation_output_tps("TP1", D8) == pytest.approx(tps1, rel=1e-12)
    assert m.saturation_output_tps("TP2", D8, max_num_seqs=512) == pytest.approx(tps2, rel=1e-12)

def test_rejects_unknown_tp_ar_path_and_config():
    with pytest.raises(ValueError, match="tp must be 1 or 2"):
        m.decode_step_time(4, 1, 1024, D8)
    with pytest.raises(ValueError, match="ar_path"):
        m.decode_step_time(2, 1, 1024, D8, ar_path="nccl")
    with pytest.raises(ValueError, match="config"):
        m.running_limit("DP4", D8)

@pytest.mark.parametrize("config,tp,batch", [("TP1", 1, 1), ("TP1", 1, 4), ("TP1", 1, 64), ("TP1", 1, 330),
                                             ("TP2", 2, 1), ("TP2", 2, 4), ("TP2", 2, 64), ("TP2", 2, 512)])
def test_saturation_matches_sat_py_including_memory_bound_steps(config, tp, batch):
    # Small batches make sat.py's step memory-bound, so its 15.01e9 / tp weight literal is exercised.
    d8s = _run_d8_script("sat.py")
    _, tps = d8s["step"](m.running_limit(config, D8, max_num_seqs=batch), tp)   # TP1 is KV-bound at 328
    assert m.saturation_output_tps(config, D8, max_num_seqs=batch) == pytest.approx(tps, rel=1e-12)


def _floats(obj):
    if isinstance(obj, float):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _floats(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _floats(v)


def test_predictions_floats_have_at_most_4_significant_digits():
    p = m.predictions()
    outputs = [x for k, v in p.items() if k != "constants" for x in _floats(v)]
    assert outputs
    for x in outputs:
        assert float(f"{x:.4g}") == x, x
    # Constants are inputs and are reported verbatim; every one already has at most 4 significant digits.
    for x in _floats(p["constants"]):
        assert float(f"{x:.4g}") == x, x


def test_predictions_h2_has_a_prefill_leg():
    p = m.predictions()
    c = m.CONSTANTS["central"]

    def e(n: int) -> float:
        return m.prefill_time(1, n, c) / m.prefill_time(2, n, c) / 2

    lo, hi = p["bands"]["H2_prefill"]
    assert lo <= hi
    central = m.hypothesis_values(c)["H2_prefill"]
    assert central == pytest.approx(e(8192) - e(512), rel=1e-3)
    assert lo <= central <= hi


BRIEF_KEYS = {"constants", "decode", "prefill", "kv", "saturation_tps", "s_star_ms", "bands"}


def test_predictions_keys_are_the_brief_schema_plus_ratios_only():
    # The brief's schema, plus the documented "ratios" extension; no "hypotheses_central".
    assert set(m.predictions()) == BRIEF_KEYS | {"ratios"}


def test_hypothesis_values_are_rounded_and_lie_in_their_bands():
    p = m.predictions()
    for name in ("optimistic", "central", "pessimistic"):
        values = m.hypothesis_values(m.CONSTANTS[name])
        assert set(values) == set(p["bands"])
        for h, (lo, hi) in p["bands"].items():
            assert float(f"{values[h]:.4g}") == values[h], (name, h)
            assert lo <= values[h] <= hi, (name, h)
    # The bands are exactly the min/max of the per-set values.
    per_set = [m.hypothesis_values(m.CONSTANTS[n]) for n in ("optimistic", "central", "pessimistic")]
    for h, band in p["bands"].items():
        assert band == [min(v[h] for v in per_set), max(v[h] for v in per_set)], h


def test_hypothesis_values_h8_is_the_nccl_unfused_over_fused_ratio():
    c = m.CONSTANTS["central"]
    full = m.decode_step_time(2, 1, 1216, c, ar_path="nccl_unfused") / m.decode_step_time(2, 1, 1216, c)
    assert m.hypothesis_values(c)["H8"] == float(f"{full:.4g}")


def test_predictions_ratios_are_computed_before_rounding():
    p, c = m.predictions(), m.CONSTANTS["central"]
    full = m.decode_step_time(1, 1, 1216, c) / m.decode_step_time(2, 1, 1216, c)
    assert p["ratios"]["decode_speedup"]["central"]["1"] == float(f"{full:.4g}")
    central = m.hypothesis_values(c)
    assert p["ratios"]["decode_speedup"]["central"]["1"] == central["H1"]
    assert p["ratios"]["dp2_over_tp2_saturation"]["central"] == central["H3"]
    assert set(p["ratios"]["prefill_speedup"]["central"]) == set(p["prefill"]["central"]["1"])
