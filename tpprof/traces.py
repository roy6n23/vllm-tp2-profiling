"""nsys SQLite export -> per-rank steps, completeness gate, and a trace summary (spec 4.5).

Joins follow research D6-7/D6-8: kernels are matched to their launching runtime-API
record on (pid, correlationId), because CUPTI correlationIds are per process and collide
across TP workers. Steps are assigned by that CPU launch time, never by the GPU start
(AM15). Category time uses interval unions with overlap given to the earlier-starting
kernel (AM16), because PDL lets adjacent kernels overlap (research D3-14).
"""
from __future__ import annotations

import bisect
import functools
import re
import sqlite3
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from tpprof import kernels as kn

STEP_RE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")
MEASURE_RANGE = "tpprof:measure"

KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
RUNTIME_TABLE = "CUPTI_ACTIVITY_KIND_RUNTIME"
MEMCPY_TABLE = "CUPTI_ACTIVITY_KIND_MEMCPY"
MEMSET_TABLE = "CUPTI_ACTIVITY_KIND_MEMSET"
NVTX_TABLE = "NVTX_EVENTS"
STRINGS_TABLE = "StringIds"
NVTX_PUSH_POP = 59
NVTX_START_END = 60
MEMCPY_NAME = "[CUDA memcpy]"
MEMSET_NAME = "[CUDA memset]"
COPY_NAMES = (MEMCPY_NAME, MEMSET_NAME)
# Without these the analysis has no kernels, no steps, or no kernel names (Review Focus 5).
REQUIRED_TABLES = (KERNEL_TABLE, NVTX_TABLE, STRINGS_TABLE)
TOP_UNCLASSIFIED = 20

_ID_SPAN = 0x1000000   # globalTid = <HW:8><VM:8><PID:24><TID:24> (research D6-8)


@dataclass(frozen=True)
class Kernel:
    pid: int
    device: int
    start: int
    end: int
    name: str
    launch_ts: int
    graph_id: int | None


@dataclass(frozen=True)
class Range:
    pid: int
    tid: int
    start: int
    end: int
    text: str


@dataclass
class TraceData:
    kernels: list[Kernel]
    copies: list[Kernel]
    ranges: list[Range]
    tables: set[str]
    launch_ts_missing: int = 0   # kernels/copies without a runtime record; their launch_ts is the GPU start


@dataclass
class Gate:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _pid(global_id: int) -> int:
    return global_id // _ID_SPAN % _ID_SPAN


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}


@functools.lru_cache(maxsize=None)
def _category(name: str) -> str:
    return kn.categorize(name)


def load_trace(db_path: str) -> TraceData:
    con = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if KERNEL_TABLE in tables:
            con.execute(f"CREATE INDEX IF NOT EXISTS ix_k ON {KERNEL_TABLE}(globalPid, correlationId)")
        if RUNTIME_TABLE in tables:
            con.execute(f"CREATE INDEX IF NOT EXISTS ix_r ON {RUNTIME_TABLE}(globalTid, correlationId)")
        con.commit()
        strings: dict[int, str] = {}
        if STRINGS_TABLE in tables:
            strings = dict(con.execute(f"SELECT id, value FROM {STRINGS_TABLE}"))

        # (pid, correlationId) -> CPU start of the launching call; for graph-node kernels
        # this is the cudaGraphLaunch record.
        launches: dict[tuple[int, int], int] = {}
        if RUNTIME_TABLE in tables:
            for gtid, cid, start in con.execute(f"SELECT globalTid, correlationId, start FROM {RUNTIME_TABLE}"):
                if gtid is not None and cid is not None:
                    launches.setdefault((_pid(gtid), cid), start)
        missing = 0

        def launch_ts(pid: int, cid: int | None, start: int) -> int:
            nonlocal missing
            ts = launches.get((pid, cid))
            if ts is None:
                missing += 1
                return start
            return ts

        found: list[Kernel] = []
        if KERNEL_TABLE in tables:
            graph = "graphId" if "graphId" in _columns(con, KERNEL_TABLE) else "NULL"
            for start, end, dev, gpid, cid, dname, sname, gid in con.execute(
                    f"SELECT start, end, deviceId, globalPid, correlationId, demangledName, shortName, {graph} "
                    f"FROM {KERNEL_TABLE}"):
                if gpid is None:
                    continue
                pid = _pid(gpid)
                name = strings.get(dname) or strings.get(sname) or ""
                found.append(Kernel(pid, dev, start, end, name, launch_ts(pid, cid, start), gid))

        copies: list[Kernel] = []
        for table, label in ((MEMCPY_TABLE, MEMCPY_NAME), (MEMSET_TABLE, MEMSET_NAME)):
            if table not in tables:
                continue
            graph = "graphId" if "graphId" in _columns(con, table) else "NULL"
            for start, end, dev, gpid, cid, gid in con.execute(
                    f"SELECT start, end, deviceId, globalPid, correlationId, {graph} FROM {table}"):
                if gpid is None:
                    continue
                pid = _pid(gpid)
                copies.append(Kernel(pid, dev, start, end, label, launch_ts(pid, cid, start), gid))

        ranges: list[Range] = []
        if NVTX_TABLE in tables:
            text_id = "textId" if "textId" in _columns(con, NVTX_TABLE) else "NULL"
            for start, end, etype, text, tid, gtid in con.execute(
                    f"SELECT start, end, eventType, text, {text_id}, globalTid FROM {NVTX_TABLE} "
                    f"WHERE eventType IN ({NVTX_PUSH_POP}, {NVTX_START_END})"):
                if text is None:
                    text = strings.get(tid)
                if gtid is None or text is None:
                    continue
                if etype == NVTX_PUSH_POP or text == MEASURE_RANGE:
                    ranges.append(Range(_pid(gtid), gtid % _ID_SPAN, start, start if end is None else end, text))

        found.sort(key=lambda k: k.start)
        copies.sort(key=lambda k: k.start)
        ranges.sort(key=lambda r: r.start)
        return TraceData(found, copies, ranges, tables, missing)
    finally:
        con.close()


def worker_ranks(td: TraceData) -> list[tuple[int, int]]:
    """(pid, device) pairs that ran kernels, sorted by device; rank = index."""
    return sorted({(k.pid, k.device) for k in td.kernels}, key=lambda p: (p[1], p[0]))


def _measure(td: TraceData) -> tuple[int, int] | None:
    for r in td.ranges:
        if r.text == MEASURE_RANGE:
            return r.start, r.end
    return None


def _is_forward(text: str) -> bool:
    """False for execute_context_0(0)_generation_0(0): vLLM opens it when nothing is scheduled and it
    launches no kernels, so it is not a forward step (2026-10-01 box: one after the last request)."""
    m = STEP_RE.match(text)
    return m is not None and (m.group(1) != "0" or m.group(3) != "0")


def step_ranges(td: TraceData, pid: int) -> list[Range]:
    """vLLM's execute_context_* forward ranges of that pid, by start; inside tpprof:measure if present."""
    window = _measure(td)
    return [r for r in td.ranges
            if r.pid == pid and _is_forward(r.text)
            and (window is None or window[0] <= r.start <= window[1])]


def union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    total, cur_start, cur_end = 0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        elif end > cur_end:
            cur_end = end
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def assign_steps(td: TraceData, pid: int, device: int) -> list[dict]:
    """Step k = [start_k, start_{k+1}); the last step ends at the max launch_ts + 1.

    Every kernel and copy of (pid, device) goes to the step containing its CPU launch time (AM15).
    """
    ranges = step_ranges(td, pid)
    if not ranges:
        return []
    window = _measure(td)
    events = [e for e in td.kernels + td.copies
              if e.pid == pid and e.device == device
              and (window is None or window[0] <= e.launch_ts <= window[1])]
    starts = [r.start for r in ranges]
    last_end = max(max((e.launch_ts for e in events), default=starts[-1]) + 1, starts[-1] + 1)
    buckets: list[list[Kernel]] = [[] for _ in ranges]
    for e in events:
        i = bisect.bisect_right(starts, e.launch_ts) - 1
        if i >= 0:
            buckets[i].append(e)
    steps = []
    for i, (r, evs) in enumerate(zip(ranges, buckets)):
        evs.sort(key=lambda e: (e.launch_ts, e.start))
        nc, nct, ng, ngt = (int(g) for g in STEP_RE.match(r.text).groups())
        steps.append({
            "index": i, "nvtx": r.text,
            "start": r.start, "end": starts[i + 1] if i + 1 < len(starts) else last_end,
            "n_ctx_reqs": nc, "n_ctx_tokens": nct, "n_gen_reqs": ng, "n_gen_tokens": ngt,
            "ar_ops": sum(kn.is_ar_launch(e.name) for e in evs),
            "ag_ops": sum(kn.is_all_gather(e.name) for e in evs),
            "kernels": evs,
            "gpu_busy_ns": union_ns((e.start, e.end) for e in evs),
        })
    return steps


def _attributed(ks: Sequence[Kernel], window: tuple[int, int]) -> list[tuple[Kernel, int]]:
    """Per kernel, the ns of the window it alone is credited with: overlap goes to the earlier start."""
    lo, hi = window
    clipped = [(max(k.start, lo), min(k.end, hi), k) for k in ks if min(k.end, hi) > max(k.start, lo)]
    clipped.sort(key=lambda c: (c[0], c[1]))
    out, covered = [], lo
    for start, end, k in clipped:
        out.append((k, max(0, end - max(start, covered))))
        covered = max(covered, end)
    return out


def attribute(kernels: Sequence[Kernel], window: tuple[int, int]) -> dict[str, int]:
    """ns per category inside `window`; the values sum to the union of the clipped intervals (AM16)."""
    out = dict.fromkeys(kn.CATEGORIES, 0)
    for k, ns in _attributed(kernels, window):
        out[_category(k.name)] += ns
    return out


def _gate(pairs: list[tuple[int, int]], steps_by_rank: list[list[dict]], td: TraceData,
          tp: int, min_steps: int) -> Gate:
    reasons: list[str] = []
    missing = [t for t in REQUIRED_TABLES if t not in td.tables]
    if missing:
        reasons.append(f"trace is missing required tables: {', '.join(missing)}")
    # (a) exactly tp (pid, device) pairs, on distinct devices
    if len(pairs) != tp:
        counts = Counter((k.pid, k.device) for k in td.kernels)
        found = ", ".join(f"(pid {p}, device {d}, {counts[(p, d)]} kernels)" for p, d in pairs)
        reasons.append(f"expected {tp} (pid, device) pairs, found {len(pairs)}" + (f": {found}" if found else ""))
    if len({d for _, d in pairs}) != len(pairs):
        reasons.append(f"(pid, device) pairs are not one per GPU: {pairs}")
    # (b) equal execute-range counts on every rank, and at least min_steps
    counts = [len(s) for s in steps_by_rank]
    if counts:
        ref = counts.index(max(counts))
        for i, c in enumerate(counts):
            if c != counts[ref]:
                reasons.append(f"rank {i} has {c} steps, rank {ref} has {counts[ref]}")
        for i, c in enumerate(counts):
            if c < min_steps:
                reasons.append(f"rank {i} has {c} steps, fewer than min_steps={min_steps}")
    # (c) equal per-step AR op counts across ranks
    if steps_by_rank:
        ar0 = [s["ar_ops"] for s in steps_by_rank[0]]
        for i, steps in enumerate(steps_by_rank[1:], start=1):
            ari = [s["ar_ops"] for s in steps]
            diffs = [j for j, (a, b) in enumerate(zip(ar0, ari)) if a != b]
            if diffs:
                j = diffs[0]
                reasons.append(f"rank {i} per-step AR op counts differ from rank 0 in {len(diffs)} steps; "
                               f"first at step {j}: {ari[j]} vs {ar0[j]}")
    # (d) at least one kernel (not a memcpy/memset) in the last step on every rank
    for i, steps in enumerate(steps_by_rank):
        if steps and not any(e.name not in COPY_NAMES for e in steps[-1]["kernels"]):
            reasons.append(f"rank {i} has no kernels in its last step (step {len(steps) - 1})")
    return Gate(not reasons, reasons)


def completeness_gate(td: TraceData, tp: int, min_steps: int) -> Gate:
    pairs = worker_ranks(td)
    return _gate(pairs, [assign_steps(td, p, d) for p, d in pairs], td, tp, min_steps)


def _is_pure_decode(step: dict) -> bool:
    return (step["n_ctx_reqs"] == 0 and step["n_ctx_tokens"] == 0
            and step["n_gen_reqs"] == step["n_gen_tokens"] > 0)


def _stat_steps(steps: list[dict]) -> list[dict]:
    """Per-step statistics use pure decode steps when there are any (AM15), else every step."""
    return [s for s in steps if _is_pure_decode(s)] or steps


def _min_max_mode(values: list[int]) -> dict:
    if not values:
        return {"min": None, "max": None, "mode": None}
    counts = Counter(values)
    top = max(counts.values())
    return {"min": min(values), "max": max(values), "mode": min(v for v, n in counts.items() if n == top)}


def _ar_op_durations(step: dict) -> list[int]:
    """GPU time of each AR op in launch order; an rmsNormLamport tail joins the preceding op."""
    ops: list[list[Kernel]] = []
    for k in step["kernels"]:
        if kn.is_ar_launch(k.name):
            ops.append([k])
        elif kn.is_ar_tail(k.name) and ops:
            ops[-1].append(k)
    return [union_ns((k.start, k.end) for k in op) for op in ops]


def _ar_wire_and_sync(steps_by_rank: list[list[dict]]) -> tuple[float | None, float | None]:
    """AM16: per AR op, min duration across ranks = wire time, max - min = sync wait; summed per step."""
    if not steps_by_rank:
        return None, None
    n = min(len(s) for s in steps_by_rank)
    indices = [s["index"] for s in _stat_steps(steps_by_rank[0][:n])]
    wire, sync = [], []
    for j in indices:
        per_rank = [_ar_op_durations(steps[j]) for steps in steps_by_rank]
        if not per_rank[0] or len({len(d) for d in per_rank}) != 1:
            continue
        cols = list(zip(*per_rank))
        wire.append(sum(min(c) for c in cols))
        sync.append(sum(max(c) - min(c) for c in cols))
    if not wire:
        return None, None
    return statistics.fmean(wire) / 1e9, statistics.fmean(sync) / 1e9


def summarize_trace(db_path: str, tp: int, min_steps: int, untraced_step_s: float | None = None) -> dict:
    td = load_trace(db_path)
    pairs = worker_ranks(td)
    steps_by_rank = [assign_steps(td, p, d) for p, d in pairs]
    gate = _gate(pairs, steps_by_rank, td, tp, min_steps)

    events = td.kernels + td.copies
    window = _measure(td)
    if window is None:
        window = (min((e.start for e in events), default=0), max((e.end for e in events), default=0))
    span = window[1] - window[0]

    ranks, busy_per_step = [], []
    for i, ((pid, device), steps) in enumerate(zip(pairs, steps_by_rank)):
        per_kernel = _attributed([e for e in events if e.pid == pid and e.device == device], window)
        busy = sum(ns for _, ns in per_kernel)
        by_cat = dict.fromkeys(kn.CATEGORIES, 0)
        unclassified: Counter[str] = Counter()
        for k, ns in per_kernel:
            cat = _category(k.name)
            by_cat[cat] += ns
            if cat == "other":
                unclassified[k.name] += ns
        stat = _stat_steps(steps)
        timed = [s for s in stat if s["index"] != len(steps) - 1] or stat   # the last step has no next start
        if stat:
            busy_per_step.append(statistics.fmean(s["gpu_busy_ns"] for s in stat))
        ranks.append({
            "rank": i, "pid": pid, "device": device,
            "steps": len(steps),
            "decode_steps": sum(_is_pure_decode(s) for s in steps),
            "ar_ops_per_step": _min_max_mode([s["ar_ops"] for s in steps]),
            "ag_ops_per_step": _min_max_mode([s["ag_ops"] for s in steps]),
            "mean_step_ms": statistics.fmean(s["end"] - s["start"] for s in timed) / 1e6 if timed else None,
            "busy_frac": busy / span if span > 0 else 0.0,
            "category_frac": {c: (ns / busy if busy else 0.0) for c, ns in by_cat.items()},
            "unclassified_top": [[name, ns / busy] for name, ns in unclassified.most_common(TOP_UNCLASSIFIED)],
        })

    wire, sync = _ar_wire_and_sync(steps_by_rank)
    idle_est = None
    if untraced_step_s is not None and busy_per_step:
        idle_est = 1 - statistics.fmean(busy_per_step) / 1e9 / untraced_step_s
    return {
        "gate": {"ok": gate.ok, "reasons": gate.reasons},
        "ranks": ranks,
        "ar_wire_s_per_step": wire,
        "ar_sync_wait_s_per_step": sync,
        "idle_est": idle_est,
        "window_ns": [window[0], window[1]],
        "launch_ts_missing": td.launch_ts_missing,
    }
