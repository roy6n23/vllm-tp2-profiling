"""Result files: `vllm bench serve` JSON (C7), bench-latency / offline point files (C3), validity, DP2-rand merge.

Parsers never guess. A missing key or a truncated file raises ResultFormatError naming what is
missing, so a schema drift on the box is loud instead of a column of zeros.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from tpprof.constants import DECODE_L1, DECODE_L2, ONLINE_INPUT_LEN, ONLINE_OUTPUT_LEN

# vLLM 0.30.0 serve.py:1286-1308 and 2272-2305 (D4-11). The arrays exist only with --save-detailed,
# except `latencies`, which is always kept.
SERVE_SUMMARY_KEYS = ("num_prompts", "request_rate", "duration", "completed", "failed")
SERVE_ARRAY_KEYS = ("input_lens", "output_lens", "ttfts", "itls", "latencies", "start_times", "errors")
LATENCY_PERCENTILES = (10, 25, 50, 75, 90, 99)          # vllm/benchmarks/latency.py
LATENCY_KEYS = ("avg_latency", "latencies", "percentiles")
POINT_META_KEYS = ("kind", "batch", "input_len", "output_len", "config", "arm")
OFFLINE_ROW_KEYS = ("config", "arm", "kind", "engine", "batch", "input_len", "output_len", "l1", "l2",
                    "n", "n1", "n2", "median_s", "p25_s", "p75_s", "step_s", "ci_lo_s", "ci_hi_s",
                    "ctx_mean", "note")


class ResultFormatError(ValueError):
    """A result file is missing keys, truncated, or internally inconsistent."""


def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ResultFormatError(f"{path}: not valid JSON (truncated?): {e}") from e
    if not isinstance(raw, dict):
        raise ResultFormatError(f"{path}: expected a JSON object, got {type(raw).__name__}")
    return raw


# ---------------------------------------------------------------------------------- bench serve

@dataclass
class ServeResult:
    path: str
    raw: dict
    completed: int
    failed: int
    duration: float
    num_prompts: int
    request_rate: float            # math.inf for "inf"
    input_lens: list[int]
    output_lens: list[int]
    ttfts: list[float]
    itls: list[list[float]]
    latencies: list[float]
    start_times: list[float]       # time.perf_counter() of the client: CLOCK_MONOTONIC, shared per host (D4-12)
    errors: list[str]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def ok_mask(self) -> list[bool]:
        """A request succeeded iff its error string is empty (D4-12)."""
        return [e == "" for e in self.errors]


def _parse_rate(path: str, value: object) -> float:
    if value == "inf":
        return math.inf
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ResultFormatError(f"{path}: request_rate must be a number or 'inf', got {value!r}")


def _metadata(raw: dict) -> dict[str, str]:
    """--metadata keys: flattened to the top level between num_prompts and request_rate (C7, D4-11)."""
    keys = list(raw)
    lo, hi = keys.index("num_prompts"), keys.index("request_rate")
    return {k: str(raw[k]) for k in keys[lo + 1:hi]}


def load_serve_result(path: str) -> ServeResult:
    raw = _read_json(path)
    missing = [k for k in SERVE_SUMMARY_KEYS + SERVE_ARRAY_KEYS if k not in raw]
    if missing:
        hint = ("; the per-request arrays exist only with --save-detailed"
                if set(missing) & set(SERVE_ARRAY_KEYS) else "")
        raise ResultFormatError(f"{path}: bench serve result is missing keys {missing}{hint}")
    n = raw["num_prompts"]
    bad = [k for k in SERVE_ARRAY_KEYS if not isinstance(raw[k], list) or len(raw[k]) != n]
    if bad:
        lens = {k: len(raw[k]) if isinstance(raw[k], list) else type(raw[k]).__name__ for k in bad}
        raise ResultFormatError(f"{path}: per-request arrays {bad} do not have num_prompts={n} entries: {lens}")
    return ServeResult(
        path=path, raw=raw, completed=int(raw["completed"]), failed=int(raw["failed"]),
        duration=float(raw["duration"]), num_prompts=int(n), request_rate=_parse_rate(path, raw["request_rate"]),
        input_lens=list(raw["input_lens"]), output_lens=list(raw["output_lens"]), ttfts=list(raw["ttfts"]),
        itls=[list(x) for x in raw["itls"]], latencies=list(raw["latencies"]),
        start_times=list(raw["start_times"]), errors=list(raw["errors"]), metadata=_metadata(raw))


def request_metrics(r: ServeResult) -> list[dict]:
    """Per successful request, vLLM's own definitions (D4-13): TPOT = (e2e - ttft) / (out - 1), 0 if out <= 1."""
    rows = []
    for i, ok in enumerate(r.ok_mask):
        if not ok:
            continue
        out, lat, ttft = r.output_lens[i], r.latencies[i], r.ttfts[i]
        rows.append({"ttft_s": ttft, "tpot_s": (lat - ttft) / (out - 1) if out > 1 else 0.0, "e2e_s": lat,
                     "start_s": r.start_times[i], "in_len": r.input_lens[i], "out_len": out})
    return rows


def _length_violation(name: str, lens: Sequence[int], ok: Sequence[bool], expect: int) -> str | None:
    wrong = [x for x, good in zip(lens, ok) if good and x != expect]
    if not wrong:
        return None
    return f"{len(wrong)} successful requests have {name} != {expect} (values: {sorted(set(wrong))[:5]})"


def validate_serve(r: ServeResult, expect_in: int, expect_out: int) -> list[str]:
    """Spec 4.4 validity rules. An empty list means valid; every entry says what is wrong."""
    ok = r.ok_mask
    v = []
    if r.failed != 0:
        v.append(f"failed={r.failed} (must be 0)")
    if r.completed != r.num_prompts:
        v.append(f"completed={r.completed} != num_prompts={r.num_prompts}")
    errs = [(i, e) for i, e in enumerate(r.errors) if e]
    if errs:
        i, e = errs[0]
        v.append(f"{len(errs)} requests have errors; first [{i}]: {e[:200]}")
    if sum(ok) != r.completed:
        v.append(f"completed={r.completed} but {sum(ok)} requests have an empty error")
    if not any(ok):
        v.append("no successful requests")
    for name, lens, expect in (("output_lens", r.output_lens, expect_out), ("input_lens", r.input_lens, expect_in)):
        msg = _length_violation(name, lens, ok, expect)
        if msg:
            v.append(msg)
    return v


def _client_tps(r: ServeResult) -> float | None:
    """The client's own output throughput (vLLM: output tokens / its duration)."""
    tps = r.raw.get("output_throughput")
    if tps is not None:
        return float(tps)
    if r.duration > 0:
        return sum(o for o, good in zip(r.output_lens, r.ok_mask) if good) / r.duration
    return None


def merge_serve_results(parts: Sequence[ServeResult]) -> ServeResult:
    """DP2-rand: concatenate the halves' per-request arrays (start_times share CLOCK_MONOTONIC, D4-12).

    Counts and offered rates add. `duration` is the longest half's client duration, and the merged
    client throughput (raw["output_throughput"]) is the sum of the halves' own values (D4-28 (a)).
    metadata["tpprof_start_skew_s"] is the spread of the halves' first start times; > 1 s is flagged.
    """
    if not parts:
        raise ValueError("merge_serve_results needs at least one part")
    empty = [p.path for p in parts if not p.start_times]
    if empty:
        raise ValueError(f"cannot merge parts without requests: {empty}")
    firsts = [min(p.start_times) for p in parts]
    keys = list(dict.fromkeys(k for p in parts for k in p.metadata))
    metadata = {}
    for k in keys:
        vals = [p.metadata.get(k, "") for p in parts]
        metadata[k] = vals[0] if len(set(vals)) == 1 else "+".join(vals)
    metadata["tpprof_start_skew_s"] = repr(max(firsts) - min(firsts))
    tps = [_client_tps(p) for p in parts]
    raw = {"tpprof_merged_from": [p.path for p in parts],
           "output_throughput": None if None in tps else sum(tps)}

    def cat(name: str) -> list:
        return [x for p in parts for x in getattr(p, name)]

    return ServeResult(
        path="+".join(p.path for p in parts), raw=raw,
        completed=sum(p.completed for p in parts), failed=sum(p.failed for p in parts),
        duration=max(p.duration for p in parts), num_prompts=sum(p.num_prompts for p in parts),
        request_rate=sum(p.request_rate for p in parts),
        input_lens=cat("input_lens"), output_lens=cat("output_lens"), ttfts=cat("ttfts"),
        itls=[list(x) for x in cat("itls")], latencies=cat("latencies"), start_times=cat("start_times"),
        errors=cat("errors"), metadata=metadata)


def online_row(r: ServeResult, extra: Mapping[str, object]) -> dict:
    """One tidy online_runs row. config/round/rate default to the tpprof_* metadata; `extra` overrides."""
    from tpprof import stats

    md = r.metadata
    viol = validate_serve(r, ONLINE_INPUT_LEN, ONLINE_OUTPUT_LEN)
    ms = request_metrics(r)
    ttft = [m["ttft_s"] for m in ms]
    tpot = [m["tpot_s"] for m in ms if m["out_len"] > 1]      # vLLM's TPOT percentiles skip out_len <= 1
    e2e = [m["e2e_s"] for m in ms]

    def pct(xs: list[float], p: float) -> float | None:
        return float(stats.percentile(xs, p)) if xs else None

    row = {
        "config": md.get("tpprof_config"),
        "arm": md.get("tpprof_arm"),
        "round": int(md["tpprof_round"]) if "tpprof_round" in md else None,
        "rate_target": float(md["tpprof_rate"]) if "tpprof_rate" in md else None,
        "completed": r.completed,
        "failed": r.failed,
        "valid": not viol,
        "violations": "; ".join(viol),
        "duration_s": r.duration,
        "tput_uniform_tps": (float(stats.uniform_throughput(r.start_times, r.latencies, r.output_lens, r.ok_mask))
                             if ms else None),
        "tput_client_tps": _client_tps(r),
        "ttft_p50_s": pct(ttft, 50), "ttft_p90_s": pct(ttft, 90), "ttft_p99_s": pct(ttft, 99),
        "tpot_p50_s": pct(tpot, 50), "tpot_p90_s": pct(tpot, 90), "tpot_p99_s": pct(tpot, 99),
        "e2e_p50_s": pct(e2e, 50), "e2e_p99_s": pct(e2e, 99),
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------------- bench latency / C3

@dataclass
class LatencyResult:
    path: str
    avg_latency: float
    latencies: list[float]
    percentiles: dict[str, float]
    meta: dict


def write_latency_result(path: str, latencies: Sequence[float], meta: Mapping[str, object]) -> None:
    """C3: the bench-latency schema plus a `tpprof` key. Written atomically, so a crash leaves no partial file."""
    lats = np.asarray(latencies, dtype=float)
    if lats.size == 0:
        raise ValueError(f"{path}: no latencies to write")
    pcts = np.percentile(lats, LATENCY_PERCENTILES)
    doc = {"avg_latency": float(lats.mean()), "latencies": lats.tolist(),
           "percentiles": {str(p): float(v) for p, v in zip(LATENCY_PERCENTILES, pcts)},
           "tpprof": dict(meta)}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=4)
    os.replace(tmp, path)


def load_latency_result(path: str) -> LatencyResult:
    """Reads C3 point files and plain `vllm bench latency --output-json` files (meta is then {})."""
    raw = _read_json(path)
    missing = [k for k in LATENCY_KEYS if k not in raw]
    if missing:
        raise ResultFormatError(f"{path}: latency result is missing keys {missing}")
    return LatencyResult(path=path, avg_latency=float(raw["avg_latency"]),
                         latencies=[float(x) for x in raw["latencies"]],
                         percentiles={str(k): float(v) for k, v in raw["percentiles"].items()},
                         meta=dict(raw.get("tpprof", {})))


def _point_files(session_dir: str) -> list[str]:
    """Point files directly in the session dir or in its points/ subdir (the offline driver's --out)."""
    return sorted(glob.glob(os.path.join(session_dir, "point-*.json"))
                  + glob.glob(os.path.join(session_dir, "points", "point-*.json")))


def _row(**kw: object) -> dict:
    row = dict.fromkeys(OFFLINE_ROW_KEYS)
    row.update(kw)
    return row


def offline_rows(session_dir: str) -> list[dict]:
    """Tidy offline_points rows: one per prefill point, one per decode (L1, L2) pair (spec 4.3).

    A decode point without its partner still gets a row, with step_s None and a note, so a partial
    session shows up as a gap instead of disappearing.
    """
    points: dict[tuple, LatencyResult] = {}
    for path in _point_files(session_dir):
        res = load_latency_result(path)
        missing = [k for k in POINT_META_KEYS if k not in res.meta]
        if missing:
            raise ResultFormatError(f"{path}: tpprof meta is missing keys {missing}")
        m = res.meta
        if m["kind"] not in ("prefill", "decode"):
            raise ResultFormatError(f"{path}: unknown point kind {m['kind']!r}")
        key = (m["kind"], m["config"], m["arm"], int(m["batch"]), int(m["input_len"]), int(m["output_len"]))
        if key in points:
            raise ResultFormatError(f"{path}: duplicate point, already read from {points[key].path}")
        points[key] = res

    from tpprof import stats

    def point_row(key: tuple, res: LatencyResult, note: str | None = None) -> dict:
        kind, config, arm, batch, input_len, output_len = key
        s = stats.summarize(res.latencies)
        return _row(config=config, arm=arm, kind=kind, engine=res.meta.get("engine"), batch=batch,
                    input_len=input_len, output_len=output_len, n=s["n"], median_s=s["median"],
                    p25_s=s["p25"], p75_s=s["p75"], note=note)

    rows = []
    for key, res in points.items():
        kind, config, arm, batch, input_len, output_len = key
        if kind == "prefill":
            rows.append(point_row(key, res))
            continue
        partner_len = {DECODE_L1: DECODE_L2, DECODE_L2: DECODE_L1}.get(output_len)
        partner = points.get((kind, config, arm, batch, input_len, partner_len))
        if partner is None:
            want = partner_len if partner_len is not None else f"{DECODE_L1}/{DECODE_L2}"
            rows.append(point_row(key, res, note=f"unpaired decode point: no output_len={want} partner"))
        elif output_len == DECODE_L1:
            d = stats.decode_step_from_lengths(res.latencies, partner.latencies, DECODE_L1, DECODE_L2)
            rows.append(_row(config=config, arm=arm, kind=kind, engine=res.meta.get("engine"), batch=batch,
                             input_len=input_len, l1=DECODE_L1, l2=DECODE_L2, n1=d["n1"], n2=d["n2"],
                             step_s=d["step_s"], ci_lo_s=d["ci_lo_s"], ci_hi_s=d["ci_hi_s"],
                             ctx_mean=input_len + (DECODE_L1 + DECODE_L2) / 2))
    order = {"prefill": 0, "decode": 1}
    rows.sort(key=lambda r: (order[r["kind"]], r["config"], r["arm"], r["batch"], r["input_len"],
                             r["output_len"] or 0))
    return rows
