"""Prometheus text exposition from vLLM 0.30.0 /metrics -> samples, summaries and deltas (research D5-1).

Counters are summed over every label set (engine, finished_reason, ...); model_name is constant for a
server, so this is the spec AM32 rule. prometheus_client's *_created series are dropped (D5-1).
A tracked metric missing from a scrape raises with its name rather than reading as zero.
"""
from __future__ import annotations

import re

TRACKED = ("vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
           "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:request_success_total")
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")

_SAMPLE = re.compile(r"(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)(?:\s+-?\d+)?\s*$")
_LABEL = re.compile(r'\s*(?P<k>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<v>(?:[^"\\]|\\.)*)"\s*(?:,|$)')
_UNESCAPE = {"\\\\": "\\", '\\"': '"', "\\n": "\n"}


def _labels(raw: str, line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    pos = 0
    while pos < len(raw):
        m = _LABEL.match(raw, pos)
        if not m:
            raise ValueError(f"malformed labels in Prometheus line: {line!r}")
        out[m.group("k")] = re.sub(r"\\[\\\"n]", lambda e: _UNESCAPE[e.group(0)], m.group("v"))
        pos = m.end()
    return out


def parse_prometheus(text: str) -> list[tuple[str, dict[str, str], float]]:
    """(name, labels, value) for every sample line; skips blank and '#' lines and *_created series."""
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            raise ValueError(f"malformed Prometheus line: {line!r}")
        try:
            value = float(m.group("value"))
        except ValueError:
            raise ValueError(f"malformed Prometheus value in line: {line!r}") from None
        if m.group("name").endswith("_created"):
            continue
        samples.append((m.group("name"), _labels(m.group("labels") or "", line), value))
    return samples


def metric_sum(samples, name: str) -> float:
    """Sum of `name` over all label sets; raises if the scrape has no sample of that name."""
    values = [value for n, _, value in samples if n == name]
    if not values:
        raise ValueError(f"metric {name} not found in the scrape")
    return sum(values)


def scrape_summary(text: str) -> dict[str, float]:
    """TRACKED counters and GAUGES, each summed over label sets -> {name: value}."""
    samples = parse_prometheus(text)
    present = {n for n, _, _ in samples}
    missing = [n for n in (*TRACKED, *GAUGES) if n not in present]
    if missing:
        raise ValueError(f"metrics missing from the scrape: {', '.join(missing)}")
    return {n: metric_sum(samples, n) for n in (*TRACKED, *GAUGES)}


def deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """after - before per metric. Both scrapes must carry the same names; a tracked counter may not go down."""
    if set(before) != set(after):
        odd = sorted(set(before) ^ set(after))
        raise ValueError(f"scrapes differ in metric names: {', '.join(odd)}")
    out = {n: after[n] - before[n] for n in before}
    reset = [n for n in TRACKED if out.get(n, 0.0) < 0]
    if reset:
        raise ValueError(f"counter went down between scrapes (server restarted?): {', '.join(reset)}")
    return out
