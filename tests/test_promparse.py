from __future__ import annotations

import pytest

from tpprof.promparse import GAUGES, TRACKED, deltas, metric_sum, parse_prometheus, scrape_summary

# prometheus_client text exposition in the vLLM 0.30.0 shape (D5-1, D5-5): labels sorted, counters
# exposed as _total plus a _created gauge, histogram triplets, and non-vllm series from the instrumentator.
SCRAPE = """\
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 1234.0
http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 42.0
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="llama-3.1-8b-instruct"} 12.0
vllm:num_requests_running{engine="1",model_name="llama-3.1-8b-instruct"} 9.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="llama-3.1-8b-instruct"} 3.0
vllm:num_requests_waiting{engine="1",model_name="llama-3.1-8b-instruct"} 0.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="llama-3.1-8b-instruct"} 0.25
vllm:kv_cache_usage_perc{engine="1",model_name="llama-3.1-8b-instruct"} 0.5
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="llama-3.1-8b-instruct"} 2.0
vllm:num_preemptions_total{engine="1",model_name="llama-3.1-8b-instruct"} 1.0
# TYPE vllm:num_preemptions_created gauge
vllm:num_preemptions_created{engine="0",model_name="llama-3.1-8b-instruct"} 1.7592e+09
vllm:num_preemptions_created{engine="1",model_name="llama-3.1-8b-instruct"} 1.7592e+09
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="llama-3.1-8b-instruct"} 0.0
vllm:prefix_cache_queries_total{engine="1",model_name="llama-3.1-8b-instruct"} 0.0
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="llama-3.1-8b-instruct"} 0.0
vllm:prefix_cache_hits_total{engine="1",model_name="llama-3.1-8b-instruct"} 0.0
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="llama-3.1-8b-instruct"} 102400.0
vllm:prompt_tokens_total{engine="1",model_name="llama-3.1-8b-instruct"} 51200.0
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="llama-3.1-8b-instruct"} 25600.0
vllm:generation_tokens_total{engine="1",model_name="llama-3.1-8b-instruct"} 12800.0
# TYPE vllm:request_success_total counter
vllm:request_success_total{engine="0",finished_reason="stop",model_name="llama-3.1-8b-instruct"} 1.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="llama-3.1-8b-instruct"} 99.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="llama-3.1-8b-instruct"} 0.0
vllm:request_success_total{engine="1",finished_reason="stop",model_name="llama-3.1-8b-instruct"} 2.0
vllm:request_success_total{engine="1",finished_reason="length",model_name="llama-3.1-8b-instruct"} 48.0
vllm:request_success_total{engine="1",finished_reason="abort",model_name="llama-3.1-8b-instruct"} 1.0
# TYPE vllm:request_success_created gauge
vllm:request_success_created{engine="0",finished_reason="stop",model_name="llama-3.1-8b-instruct"} 1.7592e+09
vllm:request_success_created{engine="1",finished_reason="stop",model_name="llama-3.1-8b-instruct"} 1.7592e+09
# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_bucket{engine="0",le="512.0",model_name="llama-3.1-8b-instruct"} 10.0
vllm:iteration_tokens_total_bucket{engine="0",le="+Inf",model_name="llama-3.1-8b-instruct"} 20.0
vllm:iteration_tokens_total_count{engine="0",model_name="llama-3.1-8b-instruct"} 20.0
vllm:iteration_tokens_total_sum{engine="0",model_name="llama-3.1-8b-instruct"} 9000.0
"""


def test_parse_skips_comments_and_reads_labels():
    samples = parse_prometheus(SCRAPE)
    assert ("vllm:kv_cache_usage_perc", {"engine": "1", "model_name": "llama-3.1-8b-instruct"}, 0.5) in samples
    assert ("vllm:iteration_tokens_total_bucket",
            {"engine": "0", "le": "+Inf", "model_name": "llama-3.1-8b-instruct"}, 20.0) in samples
    assert all(not name.startswith("#") for name, _, _ in samples)


def test_created_series_are_ignored():
    names = {name for name, _, _ in parse_prometheus(SCRAPE)}
    assert not any(n.endswith("_created") for n in names)
    assert metric_sum(parse_prometheus(SCRAPE), "vllm:num_preemptions_total") == 3.0


def test_request_success_sums_over_engines_and_finished_reasons():
    assert metric_sum(parse_prometheus(SCRAPE), "vllm:request_success_total") == 151.0


def test_metric_sum_raises_naming_a_missing_metric():
    with pytest.raises(ValueError, match="vllm:request_success_total"):
        metric_sum(parse_prometheus("# only a comment\n"), "vllm:request_success_total")


def test_escaped_label_values_timestamps_and_special_floats():
    text = ('m{a="x\\"y,z",b="p\\\\q\\n"} 1.5 1700000000000\n'
            "n +Inf\n"
            "o{} -2e-3\n")
    assert parse_prometheus(text) == [("m", {"a": 'x"y,z', "b": "p\\q\n"}, 1.5),
                                      ("n", {}, float("inf")),
                                      ("o", {}, -0.002)]


def test_malformed_sample_line_raises():
    with pytest.raises(ValueError, match="garbage here"):
        parse_prometheus("vllm:num_requests_running 1.0\ngarbage here\n")


def test_scrape_summary_has_every_tracked_counter_and_gauge():
    summary = scrape_summary(SCRAPE)
    assert set(summary) == set(TRACKED) | set(GAUGES)
    assert summary["vllm:request_success_total"] == 151.0
    assert summary["vllm:prompt_tokens_total"] == 153600.0
    assert summary["vllm:num_requests_running"] == 21.0
    assert summary["vllm:kv_cache_usage_perc"] == 0.75


def test_scrape_summary_raises_listing_every_missing_metric():
    text = "\n".join(ln for ln in SCRAPE.splitlines()
                     if "prefix_cache" not in ln and "num_requests_waiting" not in ln)
    with pytest.raises(ValueError) as err:
        scrape_summary(text)
    for name in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:num_requests_waiting"):
        assert name in str(err.value)


def test_scrape_summary_without_gauges_for_multi_api_server_runs():
    # vLLM 0.30.0 drops the gauges when --api-server-count > 1 (2026-10-02 box); the counters stay
    text = "\n".join(ln for ln in SCRAPE.splitlines() if not any(g in ln for g in GAUGES))
    summary = scrape_summary(text, gauges=False)
    assert set(summary) == set(TRACKED)
    with pytest.raises(ValueError):
        scrape_summary(text)
    with pytest.raises(ValueError):            # the counters are still required
        scrape_summary(text.replace("vllm:request_success_total", "vllm:x"), gauges=False)


def test_deltas():
    before = scrape_summary(SCRAPE)
    after_text = (SCRAPE.replace('finished_reason="length",model_name="llama-3.1-8b-instruct"} 99.0',
                                 'finished_reason="length",model_name="llama-3.1-8b-instruct"} 199.0')
                        .replace('vllm:num_requests_running{engine="0",model_name="llama-3.1-8b-instruct"} 12.0',
                                 'vllm:num_requests_running{engine="0",model_name="llama-3.1-8b-instruct"} 2.0'))
    d = deltas(before, scrape_summary(after_text))
    assert set(d) == set(before)
    assert d["vllm:request_success_total"] == 100.0
    assert d["vllm:num_preemptions_total"] == 0.0
    assert d["vllm:num_requests_running"] == -10.0  # gauges may go down


def test_deltas_rejects_counter_reset_and_mismatched_keys():
    before = {"vllm:generation_tokens_total": 100.0}
    with pytest.raises(ValueError, match="vllm:generation_tokens_total"):
        deltas(before, {"vllm:generation_tokens_total": 5.0})
    with pytest.raises(ValueError, match="vllm:prompt_tokens_total"):
        deltas(before, {"vllm:generation_tokens_total": 150.0, "vllm:prompt_tokens_total": 1.0})
