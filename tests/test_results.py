from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tests.conftest import FIXTURES
from tpprof import results as R

DETAILED = str(FIXTURES / "bench_serve_result_detailed.json")
NODETAIL = str(FIXTURES / "bench_serve_result_nodetail.json")
DETAILED_8 = str(FIXTURES / "bench_serve_result.json")   # also --save-detailed, 8 requests


def _serve(n=4, *, start0=100.0, rate=2.0, in_len=1024, out_len=256, failed=0, completed=None,
           num_prompts=None, errors=None, output_lens=None, metadata=None, duration=10.0):
    """A synthetic ServeResult with n requests, 1 s apart, ttft 0.1 s, latency 2.65 s."""
    errors = errors if errors is not None else [""] * n
    output_lens = output_lens if output_lens is not None else [out_len] * n
    return R.ServeResult(
        path=f"synthetic-{start0}", raw={}, completed=n - failed if completed is None else completed,
        failed=failed, duration=duration, num_prompts=n if num_prompts is None else num_prompts,
        request_rate=rate, input_lens=[in_len] * n, output_lens=list(output_lens),
        ttfts=[0.1] * n, itls=[[0.01] * (out_len - 1) for _ in range(n)], latencies=[2.65] * n,
        start_times=[start0 + i for i in range(n)], errors=list(errors),
        metadata=dict(metadata or {}))


# ---------------------------------------------------------------- bench serve: loading

@pytest.mark.parametrize("path", [DETAILED, DETAILED_8])
def test_load_detailed_fixture(path):
    raw = json.loads(open(path).read())
    r = R.load_serve_result(path)
    assert r.path == path and r.raw == raw
    n = raw["num_prompts"]
    assert r.num_prompts == n and r.completed == raw["completed"] == n and r.failed == 0
    for arr in (r.input_lens, r.output_lens, r.ttfts, r.itls, r.latencies, r.start_times, r.errors):
        assert len(arr) == n
    assert len(r.ttfts) == r.completed
    assert r.ok_mask == [True] * n
    assert r.request_rate == raw["request_rate"] and math.isfinite(r.request_rate)
    assert r.duration == raw["duration"]
    assert r.input_lens == raw["input_lens"] and r.output_lens == raw["output_lens"]
    assert r.start_times == raw["start_times"] and r.itls == raw["itls"]
    # --metadata keys are flattened to the top level between num_prompts and request_rate (C7)
    assert r.metadata == {"tp": raw["tp"], "config": raw["config"]}


def test_detailed_fixture_counts():
    r = R.load_serve_result(DETAILED)
    assert r.num_prompts == 6 and r.metadata == {"tp": "2", "config": "TP2"}
    assert R.load_serve_result(DETAILED_8).num_prompts == 8


def test_missing_detailed_keys_raises_with_names():
    with pytest.raises(R.ResultFormatError) as ei:
        R.load_serve_result(NODETAIL)
    msg = str(ei.value)
    for key in ("ttfts", "start_times", "input_lens", "output_lens", "itls", "errors"):
        assert key in msg
    assert "--save-detailed" in msg


def test_request_rate_inf_string(tmp_path):
    raw = json.loads(open(DETAILED).read())
    raw["request_rate"] = "inf"
    p = tmp_path / "inf.json"
    p.write_text(json.dumps(raw))
    assert R.load_serve_result(str(p)).request_rate == math.inf


def test_truncated_json_raises(tmp_path):
    p = tmp_path / "trunc.json"
    p.write_text(open(DETAILED).read()[:500])
    with pytest.raises(R.ResultFormatError, match="trunc.json"):
        R.load_serve_result(str(p))


def test_array_length_mismatch_raises(tmp_path):
    raw = json.loads(open(DETAILED).read())
    raw["ttfts"] = raw["ttfts"][:-1]
    p = tmp_path / "short.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(R.ResultFormatError, match="ttfts"):
        R.load_serve_result(str(p))


def test_failed_request_is_not_ok(tmp_path):
    raw = json.loads(open(DETAILED).read())
    raw["errors"][2] = "ClientConnectorError"
    raw["output_lens"][2] = 0
    raw["completed"], raw["failed"] = 5, 1
    p = tmp_path / "f.json"
    p.write_text(json.dumps(raw))
    r = R.load_serve_result(str(p))
    assert r.ok_mask == [True, True, False, True, True, True]
    assert len(R.request_metrics(r)) == 5


# ---------------------------------------------------------------- per-request metrics

def test_request_metrics_tpot_matches_vllm_definition():
    raw = json.loads(open(DETAILED).read())
    rows = R.request_metrics(R.load_serve_result(DETAILED))
    assert len(rows) == raw["completed"]
    for i, row in enumerate(rows):
        assert set(row) == {"ttft_s", "tpot_s", "e2e_s", "start_s", "in_len", "out_len"}
        lat, ttft, out = raw["latencies"][i], raw["ttfts"][i], raw["output_lens"][i]
        assert row["tpot_s"] == pytest.approx((lat - ttft) / (out - 1))
        assert row["ttft_s"] == ttft and row["e2e_s"] == lat
        assert row["start_s"] == raw["start_times"][i]
        assert row["in_len"] == raw["input_lens"][i] and row["out_len"] == out
    # vLLM's own mean TPOT over the same requests (D4-13)
    assert np.mean([x["tpot_s"] for x in rows]) * 1e3 == pytest.approx(raw["mean_tpot_ms"])


def test_request_metrics_tpot_zero_for_single_token():
    r = _serve(2, out_len=1)
    assert [x["tpot_s"] for x in R.request_metrics(r)] == [0.0, 0.0]


# ---------------------------------------------------------------- validity

def test_validate_accepts_clean_run():
    assert R.validate_serve(_serve(), 1024, 256) == []


def test_validate_rejects_failed_and_wrong_lengths():
    failed = _serve(4, failed=1, errors=["", "", "timeout", ""], output_lens=[256, 256, 0, 256])
    v = R.validate_serve(failed, 1024, 256)
    assert any("failed=1" in s for s in v)
    assert any("timeout" in s for s in v)

    short = _serve(4, output_lens=[256, 255, 256, 256])
    v = R.validate_serve(short, 1024, 256)
    assert len(v) == 1 and "output_lens" in v[0] and "255" in v[0]

    wrong_in = _serve(4, in_len=1023)
    v = R.validate_serve(wrong_in, 1024, 256)
    assert len(v) == 1 and "input_lens" in v[0] and "1023" in v[0]

    missing = _serve(4, num_prompts=5)
    v = R.validate_serve(missing, 1024, 256)
    assert any("completed=4" in s and "num_prompts=5" in s for s in v)


def test_validate_fixture_against_its_own_lengths():
    r = R.load_serve_result(DETAILED)
    assert R.validate_serve(r, 1536, 32) == []
    assert R.validate_serve(r, 1024, 256) != []


def test_validate_rejects_empty_run():
    assert R.validate_serve(_serve(0), 1024, 256) != []


# ---------------------------------------------------------------- DP2-rand merge

def test_merge_two_halves():
    a = _serve(3, start0=100.0, rate=2.5, metadata={"tpprof_config": "DP2rand", "tpprof_round": "1"})
    b = _serve(4, start0=100.4, rate=2.5, failed=1, errors=["", "x", "", ""],
               metadata={"tpprof_config": "DP2rand", "tpprof_round": "1"})
    m = R.merge_serve_results([a, b])
    assert m.num_prompts == 7 and m.completed == 3 + 3 and m.failed == 1
    assert m.request_rate == pytest.approx(5.0)
    assert m.start_times == a.start_times + b.start_times
    assert m.errors == a.errors + b.errors and m.itls == a.itls + b.itls
    assert len(m.ttfts) == len(m.latencies) == len(m.input_lens) == len(m.output_lens) == 7
    assert float(m.metadata["tpprof_start_skew_s"]) == pytest.approx(0.4)
    assert m.metadata["tpprof_config"] == "DP2rand" and m.metadata["tpprof_round"] == "1"
    assert m.ok_mask == a.ok_mask + b.ok_mask
    assert m.duration == max(a.duration, b.duration)
    assert R.validate_serve(m, 1024, 256) != []            # the failed request is still visible


def test_merge_skew_uses_first_start_of_each_half_and_inf_rate():
    a = _serve(2, start0=50.0, rate=math.inf)
    b = _serve(2, start0=48.5, rate=math.inf)
    m = R.merge_serve_results([a, b])
    assert float(m.metadata["tpprof_start_skew_s"]) == pytest.approx(1.5)
    assert m.request_rate == math.inf


def test_merge_requires_parts():
    with pytest.raises(ValueError):
        R.merge_serve_results([])


# ---------------------------------------------------------------- online rows

ONLINE_KEYS = {"config", "arm", "round", "rate_target", "completed", "failed", "valid", "violations",
               "duration_s", "tput_uniform_tps", "tput_client_tps", "ttft_p50_s", "ttft_p90_s", "ttft_p99_s",
               "tpot_p50_s", "tpot_p90_s", "tpot_p99_s", "e2e_p50_s", "e2e_p99_s"}


def test_online_row_on_fixture():
    pytest.importorskip("tpprof.stats")
    r = R.load_serve_result(DETAILED)
    row = R.online_row(r, {"config": "TP2", "arm": "base", "round": 1, "rate_target": 20.0, "sub": "sub-0"})
    assert set(row) == ONLINE_KEYS | {"sub"}
    assert row["config"] == "TP2" and row["arm"] == "base" and row["round"] == 1
    assert row["completed"] == 6 and row["failed"] == 0
    # the fixture is 1536 in / 32 out, so the 1024/256 online rules reject it
    assert row["valid"] is False and "output_lens" in row["violations"]
    assert row["duration_s"] == r.duration
    raw = r.raw
    ends = [s + l for s, l in zip(raw["start_times"], raw["latencies"])]
    assert row["tput_uniform_tps"] == pytest.approx(sum(raw["output_lens"]) / (max(ends) - min(raw["start_times"])))
    assert row["tput_client_tps"] == pytest.approx(raw["output_throughput"])
    assert row["ttft_p50_s"] * 1e3 == pytest.approx(raw["p50_ttft_ms"])
    assert row["tpot_p50_s"] * 1e3 == pytest.approx(raw["p50_tpot_ms"])
    assert row["e2e_p50_s"] * 1e3 == pytest.approx(raw["p50_e2el_ms"])
    assert row["ttft_p50_s"] <= row["ttft_p90_s"] <= row["ttft_p99_s"]


def test_online_row_defaults_from_metadata_and_merged_client_tput():
    pytest.importorskip("tpprof.stats")
    md = {"tpprof_config": "DP2rand", "tpprof_rate": "5.0", "tpprof_round": "2"}
    a, b = _serve(3, start0=100.0, metadata=md), _serve(3, start0=100.2, metadata=md)
    a.raw["output_throughput"], b.raw["output_throughput"] = 70.0, 80.0
    row = R.online_row(R.merge_serve_results([a, b]), {})
    assert row["config"] == "DP2rand" and row["round"] == 2 and row["rate_target"] == 5.0
    assert row["arm"] is None                                  # never invented: no tpprof_arm metadata
    assert row["valid"] is True and row["violations"] == ""
    assert row["tput_client_tps"] == pytest.approx(150.0)     # D4-28 (a): sum of the halves' own throughput
    assert row["tput_uniform_tps"] == pytest.approx(6 * 256 / (102.2 + 2.65 - 100.0))


def test_online_row_no_successful_requests_gives_none_not_zero():
    pytest.importorskip("tpprof.stats")
    r = _serve(2, failed=2, errors=["e", "e"], output_lens=[0, 0])
    row = R.online_row(r, {})
    assert row["valid"] is False
    assert row["ttft_p50_s"] is None and row["tput_uniform_tps"] is None


# ---------------------------------------------------------------- bench latency / C3

META = {"kind": "decode", "batch": 8, "input_len": 1024, "output_len": 64, "warmup": 3, "iters": 10,
        "config": "TP2", "arm": "base", "t_wall_start": 1.0, "t_mono_start": 2.0, "engine": "fake"}


def test_latency_round_trip(tmp_path):
    p = str(tmp_path / "point-decode-b8-i1024-o64.json")
    lats = [0.10, 0.12, 0.11, 0.13, 0.30]
    R.write_latency_result(p, lats, META)
    raw = json.loads(open(p).read())
    assert list(raw) == ["avg_latency", "latencies", "percentiles", "tpprof"]
    assert list(raw["percentiles"]) == ["10", "25", "50", "75", "90", "99"]
    assert raw["tpprof"] == META
    res = R.load_latency_result(p)
    assert res.path == p and res.latencies == lats and res.meta == META
    assert res.avg_latency == pytest.approx(np.mean(lats))
    assert set(res.percentiles) == {"10", "25", "50", "75", "90", "99"}
    for k, v in res.percentiles.items():
        assert v == pytest.approx(np.percentile(lats, float(k)))
    assert not list(tmp_path.glob("*.tmp"))


def test_load_latency_result_accepts_vllm_bench_latency_output(tmp_path):
    p = tmp_path / "xcheck.json"
    p.write_text(json.dumps({"avg_latency": 0.2, "latencies": [0.2, 0.2],
                             "percentiles": {"10": 0.2, "25": 0.2, "50": 0.2, "75": 0.2, "90": 0.2, "99": 0.2}},
                            indent=4))
    res = R.load_latency_result(str(p))
    assert res.meta == {} and res.percentiles["50"] == 0.2


def test_load_latency_result_names_missing_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"latencies": [0.1]}))
    with pytest.raises(R.ResultFormatError) as ei:
        R.load_latency_result(str(p))
    assert "avg_latency" in str(ei.value) and "percentiles" in str(ei.value)


# ---------------------------------------------------------------- offline rows

def _point(d, kind, batch, input_len, output_len, lats, config="TP2", arm="base"):
    meta = dict(META, kind=kind, batch=batch, input_len=input_len, output_len=output_len, config=config, arm=arm)
    R.write_latency_result(str(d / f"point-{kind}-b{batch}-i{input_len}-o{output_len}.json"), lats, meta)


def test_offline_rows(tmp_path):
    pytest.importorskip("tpprof.stats")
    pts = tmp_path / "points"
    pts.mkdir()
    pre = [0.010, 0.011, 0.012, 0.013, 0.020]
    _point(pts, "prefill", 1, 2048, 1, pre)
    # decode b8: L2 - L1 = 0.256 s over 256 steps -> 1 ms/step
    _point(pts, "decode", 8, 1024, 64, [0.50] * 10)
    _point(pts, "decode", 8, 1024, 320, [0.756] * 10)
    # decode b32: 2 ms/step
    _point(pts, "decode", 32, 1024, 64, [1.0, 1.1, 0.9, 1.0, 1.0, 1.05, 0.95, 1.0, 1.0, 1.0])
    _point(pts, "decode", 32, 1024, 320, [1.512, 1.612, 1.412, 1.512, 1.512, 1.562, 1.462, 1.512, 1.512, 1.512])
    # decode b64 has only its L1 file (a partial session)
    _point(pts, "decode", 64, 1024, 64, [2.0] * 10)
    rows = R.offline_rows(str(tmp_path))
    assert len({tuple(sorted(r)) for r in rows}) == 1              # one uniform schema (tidy CSV)

    prefill = [r for r in rows if r["kind"] == "prefill"]
    assert len(prefill) == 1
    p = prefill[0]
    assert (p["config"], p["arm"], p["batch"], p["input_len"], p["output_len"]) == ("TP2", "base", 1, 2048, 1)
    assert p["n"] == 5 and p["median_s"] == pytest.approx(0.012)
    assert p["p25_s"] == pytest.approx(np.percentile(pre, 25)) and p["p75_s"] == pytest.approx(np.percentile(pre, 75))
    assert p["step_s"] is None

    dec = {r["batch"]: r for r in rows if r["kind"] == "decode"}
    assert set(dec) == {8, 32, 64}
    d8 = dec[8]
    assert d8["step_s"] == pytest.approx(0.001) and d8["ci_lo_s"] == pytest.approx(0.001)
    assert d8["ci_hi_s"] == pytest.approx(0.001)
    assert d8["ctx_mean"] == 1024 + (64 + 320) / 2 == 1216
    assert (d8["l1"], d8["l2"], d8["n1"], d8["n2"]) == (64, 320, 10, 10)
    assert dec[32]["step_s"] == pytest.approx(0.002)
    assert dec[32]["ci_lo_s"] <= dec[32]["step_s"] <= dec[32]["ci_hi_s"]
    assert dec[64]["step_s"] is None and "320" in dec[64]["note"]


def test_offline_rows_pairs_only_within_config_and_arm(tmp_path):
    pytest.importorskip("tpprof.stats")
    _point(tmp_path, "decode", 1, 1024, 64, [0.5] * 10, arm="base")
    _point(tmp_path, "decode", 1, 1024, 320, [0.756] * 10, arm="AR3")
    rows = R.offline_rows(str(tmp_path))
    assert len(rows) == 2 and all(r["step_s"] is None for r in rows)


def test_offline_rows_names_missing_meta(tmp_path):
    p = tmp_path / "point-decode-b1-i1024-o64.json"
    R.write_latency_result(str(p), [0.1], {"kind": "decode"})
    with pytest.raises(R.ResultFormatError) as ei:
        R.offline_rows(str(tmp_path))
    assert "batch" in str(ei.value) and "config" in str(ei.value)
