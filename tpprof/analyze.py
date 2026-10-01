"""Run records -> tidy CSVs and hypothesis verdicts (spec 4.8, 7.6; AM6-AM11, AM14, AM16, AM22-AM24).

Reads the C2 run-record layout under <results_dir>/raw, with serve-session sub-runs in the ruling-R4
layout (sub-<k>/ with result.json, metrics_before/after.prom, validation.json, meta.json). Writes
<results_dir>/tidy/<table>.csv for every table and tidy/hypotheses.json.

A record that cannot be read never stops the analysis (Review Focus 2). The problem becomes a row of the
`gaps` table, next to failed, skipped and incomplete runs and to the matrix runs that have no record at
all, and SUMMARY.md lists every one of them.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tpprof import goodput, kernels, logparse, matrix, model, monitor, promparse, results, stats, traces
from tpprof.constants import (
    DECODE_BATCHES,
    DECODE_INPUT_LEN,
    DECODE_L1,
    DECODE_L2,
    ONLINE_OUTPUT_LEN,
    PREFILL_LENS,
    ROUNDS,
    SAT_SEEDS,
    TPOT_SLOS_MS,
    TTFT_SLO_S,
    TTFT_SLO_SENSITIVITY_S,
    XCHECK,
)

TABLES = ("offline_points", "online_runs", "saturation", "goodput", "s_star", "comm", "kv_capacity",
          "trace_summary", "trace_steps")
# Also written to tidy/: the H2 bootstrap, the spec section 6 evidence table and the gap list.
EXTRA_TABLES = ("efficiency_delta", "confounders", "gaps")
ALL_TABLES = TABLES + EXTRA_TABLES
VERDICTS = ("hit", "miss", "insufficient_data")

CONFIG_GPUS = {"TP1": 1, "TP2": 2, "DP2": 2, matrix.DP2RAND: 2}
CPU_FLAG_PCT = 80.0          # AM11: a process above 80% of one core flags the run
GPU_ASYM_FLAG = 0.01         # AM14: GPU0/GPU1 decode difference that flags the DP2 derivation
XCHECK_TOL = 0.03            # spec 4.3: bench latency vs the offline driver
H6_MIN_EXACT = 0.99          # spec 4.8: exact 65 AR / 1 AG in >= 99% of steps
H7_BASE_MAX, H7_G2_MIN = 0.10, 0.30
# AM16: vLLM's standalone residual-add + RMSNorm kernel (vllm::fused_add_rms_norm_kernel, kernel_names.tsv).
FUSED_ADD_RMS_NORM = re.compile(r"fused_add_rms_norm")
UNFUSED_AR_ARMS = ("AR1", "AR2", "AR3")      # spec 4.7: the RMSNorm is not fused into the all-reduce
N_BOOT = 2000
S_STAR_BOOT = 500
MAX_EVIDENCE_PATHS = 3
RECORD_FILES = {"spec.json", "cmd.json", "done.json", "failed.json", "failed.prev.json", "skipped.json",
                "effective_config.json", "validation.json"}
_PARSE_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError)


# ------------------------------------------------------------------------------------------ records

@dataclass
class Run:
    run_id: str
    path: str
    spec: dict
    status: str                  # "done" | "failed" | "skipped" | "incomplete"
    reason: str | None
    t_order: float               # first cmd.json t_wall_start, else the spec.json mtime

    @property
    def kind(self) -> str:
        return self.spec.get("kind", "")

    @property
    def config(self) -> str:
        return self.spec.get("config", "")

    @property
    def arm(self) -> str:
        return self.spec.get("arm", "")

    @property
    def tier(self) -> str:
        return self.spec.get("tier", "")

    @property
    def round(self) -> int:
        return int(self.spec.get("round", 0))

    @property
    def tp(self) -> int | None:
        eng = self.spec.get("engine")
        return eng.get("tp") if isinstance(eng, dict) else None

    def p(self, key: str, default: object = None) -> object:
        return {k: v for k, v in self.spec.get("params", [])}.get(key, default)


class _Gaps:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, type_: str, detail: str, run: Run | None = None, run_id: str | None = None,
            config: str | None = None, kind: str | None = None) -> None:
        self.rows.append({"type": type_, "run_id": run.run_id if run else run_id,
                          "config": run.config if run else config, "kind": run.kind if run else kind,
                          "detail": detail})


def _read_json(path: str) -> object:
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{os.path.basename(path)}: not valid JSON (truncated?): {e}") from e


def _t_order(run_dir: str) -> float:
    try:
        cmds = _read_json(os.path.join(run_dir, "cmd.json"))
        starts = [c["t_wall_start"] for c in cmds if isinstance(c, dict) and c.get("t_wall_start") is not None]
        if starts:
            return float(min(starts))
    except _PARSE_ERRORS:
        pass
    return os.path.getmtime(os.path.join(run_dir, "spec.json"))


def _failure_reason(doc: object) -> str:
    if not isinstance(doc, dict):
        return "unknown"
    detail = str(doc.get("detail") or "").strip().splitlines()
    return str(doc.get("reason") or "unknown") + (f": {detail[0][:200]}" if detail else "")


def load_runs(raw_dir: str, gaps: _Gaps) -> list[Run]:
    """Every run directory under raw/, oldest first. Names starting with '_' are the runner's own files."""
    runs = []
    if not os.path.isdir(raw_dir):
        gaps.add("missing", f"no run records: {raw_dir} does not exist")
        return runs
    for name in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, name)
        if name.startswith("_") or not os.path.isdir(path):
            continue
        try:
            spec = _read_json(os.path.join(path, "spec.json"))
            if not isinstance(spec, dict):
                raise ValueError("spec.json is not an object")
        except _PARSE_ERRORS as e:
            gaps.add("unreadable", f"spec.json: {e}", run_id=name)
            continue
        status, reason = "incomplete", "no done.json or failed.json (interrupted?)"
        try:
            if os.path.exists(os.path.join(path, "done.json")):
                status, reason = "done", None
            elif os.path.exists(os.path.join(path, "failed.json")):
                status, reason = "failed", _failure_reason(_read_json(os.path.join(path, "failed.json")))
            elif os.path.exists(os.path.join(path, "skipped.json")):
                status, reason = "skipped", _failure_reason(_read_json(os.path.join(path, "skipped.json")))
        except _PARSE_ERRORS as e:
            reason = f"unreadable status file: {e}"
        runs.append(Run(name, path, spec, status, reason, _t_order(path)))
    runs.sort(key=lambda r: (r.t_order, r.run_id))
    return runs


def _skipped_from_last_run(raw_dir: str) -> dict[str, str]:
    """run_id -> reason from the runner's raw/_last_run.json (entries: ids, {run_id, reason} or [id, reason])."""
    path = os.path.join(raw_dir, "_last_run.json")
    if not os.path.exists(path):
        return {}
    try:
        doc = _read_json(path)
    except _PARSE_ERRORS:
        return {}
    out = {}
    for entry in doc.get("skipped", []) if isinstance(doc, dict) else []:
        if isinstance(entry, str):
            out[entry] = "skipped"
        elif isinstance(entry, dict) and "run_id" in entry:
            out[str(entry["run_id"])] = str(entry.get("reason") or "skipped")
        elif isinstance(entry, (list, tuple)) and entry:
            out[str(entry[0])] = str(entry[1]) if len(entry) > 1 else "skipped"
    return out


def _of_kind(runs: Iterable[Run], *kinds: str) -> list[Run]:
    return [r for r in runs if r.kind in kinds]


# ------------------------------------------------------------------------------------------ numbers

def finite(x: object) -> float | None:
    """A finite float, or None (NaN and None both mean 'no value')."""
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def median_or_none(xs: Iterable[float | None]) -> float | None:
    vals = [v for v in (finite(x) for x in xs) if v is not None]
    return float(np.median(vals)) if vals else None


def _rate(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)                 # "inf" -> inf
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------ offline

def _point_latencies(points_dir: str) -> dict[tuple, list[float]]:
    out = {}
    for path in sorted(glob.glob(os.path.join(points_dir, "point-*.json"))):
        res = results.load_latency_result(path)
        m = res.meta
        out[(m["config"], m["arm"], m["kind"], int(m["batch"]), int(m["input_len"]), int(m["output_len"]))] = \
            res.latencies
    return out


def _metric(row: Mapping) -> float | None:
    return finite(row.get("step_s") if row.get("kind") == "decode" else row.get("median_s"))


def _model_ms(pred: Mapping, row: Mapping) -> tuple[float | None, float | None, float | None]:
    """(central, lo, hi) model ms of a base-arm row; DP2-derived rows use TP1 at half the batch."""
    if row["arm"] != "base" or row["kind"] not in ("decode", "prefill"):
        return None, None, None
    tp = "2" if row["config"] == "TP2" else "1"
    if row["kind"] == "decode":
        section, key = "decode", str(row["batch"] // 2 if row["derived"] else row["batch"])
    else:
        section, key = "prefill", str(row["input_len"])
    vals = [pred[section][s][tp].get(key) for s in ("central", "optimistic", "pessimistic")]
    if vals[0] is None:
        return None, None, None
    return vals[0], min(v for v in vals if v is not None), max(v for v in vals if v is not None)


def _engine_checked(run: Run, eff_name: str, rows: Sequence[Mapping], gaps: _Gaps, what: str) -> bool:
    """Review C1: a real engine's numbers count only when its effective-config check is on record and clean.
    The fake engine of a dry run prints no vLLM lines and is never checked."""
    if rows and all(r.get("engine") == "fake" for r in rows):
        return True
    try:
        doc = _read_json(os.path.join(run.path, eff_name))
    except (OSError, ValueError):
        doc = None
    violations = doc.get("violations") if isinstance(doc, dict) else None
    if violations is None:
        gaps.add("effective_config", f"{what}: no engine check on record ({eff_name}); not used", run)
        return False
    if violations:
        gaps.add("effective_config", f"{what}: the engine failed its check ({eff_name}: {violations[0]}); not used",
                 run)
        return False
    return True


def _smoke_gpu1_step(runs: Sequence[Run], gaps: _Gaps) -> float | None:
    """AM14: the TP1 decode bs-1 step the smoke measured on GPU1 (points/, with effective_config_gpu1.json)."""
    steps = []
    for run in _of_kind(runs, "smoke"):
        d = os.path.join(run.path, "points")
        if run.config != "TP1" or not os.path.isdir(d):
            continue
        try:
            rows = results.offline_rows(d)
        except _PARSE_ERRORS as e:
            gaps.add("unreadable", f"smoke GPU1 points: {e}", run)
            continue
        if not _engine_checked(run, "effective_config_gpu1.json", rows, gaps, "smoke GPU1 points"):
            continue
        steps += [r["step_s"] for r in rows if r["kind"] == "decode" and r["batch"] == 1 and r["step_s"]]
    return median_or_none(steps)


def _xcheck_rows(runs: Sequence[Run], lat: Mapping[tuple, list[float]], gaps: _Gaps) -> list[dict]:
    """Spec 4.3: `vllm bench latency` vs the offline driver at the same point."""
    rows = []
    x = XCHECK
    for run in _of_kind(runs, "bench_latency_xcheck"):
        files = [p for p in sorted(glob.glob(os.path.join(run.path, "*.json")))
                 if os.path.basename(p) not in RECORD_FILES]
        if not files:
            if run.status == "done":
                gaps.add("unreadable", "no bench latency JSON in the run dir", run)
            continue
        if os.path.exists(os.path.join(run.path, "effective_config.json")) and not _engine_checked(
                run, "effective_config.json", [], gaps, "bench latency"):
            continue
        try:
            res = results.load_latency_result(files[0])
        except _PARSE_ERRORS as e:
            gaps.add("unreadable", str(e), run)
            continue
        s = stats.summarize(res.latencies)
        driver = lat.get((run.config, run.arm, "decode", x["batch"], x["input_len"], x["output_len"]))
        driver_median = float(np.median(driver)) if driver else None
        diff = s["median"] / driver_median - 1 if driver_median else None
        rows.append({"run_id": run.run_id, "tier": run.tier, "config": run.config, "arm": run.arm, "kind": "xcheck",
                     "engine": "vllm bench latency", "batch": x["batch"], "input_len": x["input_len"],
                     "output_len": x["output_len"], "n": s["n"], "median_s": s["median"], "p25_s": s["p25"],
                     "p75_s": s["p75"], "derived": False, "xcheck_diff": diff,
                     "note": (None if diff is None else f"{'within' if abs(diff) <= XCHECK_TOL else 'outside'} "
                                                       f"{XCHECK_TOL:.0%} of the driver")})
    return rows


def _offline(runs: Sequence[Run], pred: Mapping, gaps: _Gaps) -> tuple[list[dict], dict[tuple, list[float]]]:
    rows: list[dict] = []
    lat: dict[tuple, list[float]] = {}
    for run in _of_kind(runs, "offline"):
        points = os.path.join(run.path, "points")
        try:
            got = results.offline_rows(points)
            run_lat = _point_latencies(points)
        except _PARSE_ERRORS as e:
            gaps.add("unreadable", f"points: {e}", run)
            continue
        if not got and run.status == "done":
            gaps.add("unreadable", "done, but no point files under points/", run)
        if got and not _engine_checked(run, "effective_config.json", got, gaps, "offline points"):
            continue
        for r in got:
            r.update(run_id=run.run_id, tier=run.tier, derived=False)
            if r["note"]:
                gaps.add("partial", f"{r['kind']} b{r['batch']} o{r['output_len']}: {r['note']}", run)
            rows.append(r)
        for key, v in run_lat.items():
            lat.setdefault(key, v)          # a repeated point keeps its first session (oldest run)

    tp1 = [r for r in rows if (r["config"], r["arm"], r["kind"]) == ("TP1", "base", "decode") and finite(r["step_s"])]
    gpu1 = _smoke_gpu1_step(runs, gaps)
    gpu0 = median_or_none(r["step_s"] for r in tp1 if r["batch"] == 1)
    asym = abs(gpu1 - gpu0) / gpu0 if gpu1 is not None and gpu0 else None
    for r in tp1:                           # spec 4.1: DP2 at batch B = two TP1 ranks at B/2
        d = dict(r, config="DP2", batch=2 * r["batch"], derived=True, gpu_asym=asym,
                 note=f"derived: TP1 at batch {r['batch']}",
                 flag="gpu_asym>=1%" if asym is not None and asym >= GPU_ASYM_FLAG else None)
        rows.append(d)

    ref: dict[tuple, list[float]] = {}
    for r in rows:
        if (r["config"], r["arm"]) == ("TP1", "base") and not r["derived"] and _metric(r) is not None:
            ref.setdefault((r["kind"], r["batch"], r["input_len"]), []).append(_metric(r))
    for r in rows:
        value, base = _metric(r), median_or_none(ref.get((r["kind"], r["batch"], r["input_len"]), []))
        r["gpus"] = CONFIG_GPUS.get(r["config"], 1)
        r["tokens_per_step"] = r["batch"] if r["kind"] == "decode" else r["batch"] * r["input_len"]
        r["t_ms"] = value * 1e3 if value is not None else None
        r["speedup"] = base / value if value and base is not None else None
        r["efficiency"] = r["speedup"] / r["gpus"] if r["speedup"] is not None else None
        r["model_ms"], r["model_lo_ms"], r["model_hi_ms"] = _model_ms(pred, r)
        r.setdefault("gpu_asym", None)
        r.setdefault("flag", None)
    rows += _xcheck_rows(runs, lat, gaps)
    return rows, lat


def _boot_medians(arrays: Sequence[Sequence[float]], n_boot: int, seed: int) -> list[np.ndarray]:
    """For each array, the medians of n_boot resamples; one default_rng(seed) draws them all in order."""
    rng = np.random.default_rng(seed)
    out = []
    for a in arrays:
        x = np.asarray(a, dtype=float)
        out.append(np.median(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1))
    return out


def efficiency_delta(lat: Mapping[tuple, Sequence[float]], n_boot: int = N_BOOT, seed: int = 0) -> list[dict]:
    """H2 (AM24): e = TP1/TP2 time ratio / 2; delta = e(largest) - e(smallest) with a bootstrap 95% CI that
    resamples every point's iterations. Decode 128 vs 1 (two-length steps), prefill 8192 vs 512."""
    b_lo, b_hi, n_lo, n_hi = min(DECODE_BATCHES), max(DECODE_BATCHES), min(PREFILL_LENS), max(PREFILL_LENS)
    regimes = {
        "decode": (b_lo, b_hi, [(c, "decode", b, DECODE_INPUT_LEN, L) for b in (b_lo, b_hi) for c in ("TP1", "TP2")
                                for L in (DECODE_L1, DECODE_L2)]),
        "prefill": (n_lo, n_hi, [(c, "prefill", 1, n, 1) for n in (n_lo, n_hi) for c in ("TP1", "TP2")]),
    }
    rows = []
    for regime, (lo, hi, keys) in regimes.items():
        full = [(c, "base", kind, b, i, o) for c, kind, b, i, o in keys]
        missing = [k for k in full if not lat.get(k)]
        row = {"regime": regime, "config": "TP2", "arm": "base", "x_lo": lo, "x_hi": hi, "e_lo": None, "e_hi": None,
               "delta": None, "ci_lo": None, "ci_hi": None, "n_boot": n_boot,
               "missing": "; ".join(f"{c}/{a} {k} b{b} i{i} o{o}" for c, a, k, b, i, o in missing)}
        if not missing:
            meds = dict(zip(full, (np.array([np.median(lat[k])]) for k in full)))
            boots = dict(zip(full, _boot_medians([lat[k] for k in full], n_boot, seed)))

            def e(m: Mapping, x: int) -> np.ndarray:
                if regime == "decode":
                    t = {c: (m[(c, "base", "decode", x, DECODE_INPUT_LEN, DECODE_L2)]
                             - m[(c, "base", "decode", x, DECODE_INPUT_LEN, DECODE_L1)]) for c in ("TP1", "TP2")}
                else:
                    t = {c: m[(c, "base", "prefill", 1, x, 1)] for c in ("TP1", "TP2")}
                with np.errstate(divide="ignore", invalid="ignore"):
                    return t["TP1"] / t["TP2"] / 2

            e_lo, e_hi = e(meds, lo)[0], e(meds, hi)[0]
            d_boot = e(boots, hi) - e(boots, lo)
            d_boot = d_boot[np.isfinite(d_boot)]
            row.update(e_lo=finite(e_lo), e_hi=finite(e_hi), delta=finite(e_hi - e_lo))
            if d_boot.size:
                ci = np.percentile(d_boot, [2.5, 97.5])
                row.update(ci_lo=float(ci[0]), ci_hi=float(ci[1]))
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------ online

@dataclass
class _Sub:
    row: dict
    result: results.ServeResult


def _is_api_server(cmd: str) -> bool:
    """AM11: the API server is `vllm serve` itself (or a spawned VLLM::APIServer), never an engine core,
    worker or coordinator (vLLM renames those with setproctitle to VLLM::<name>)."""
    if "VLLM::APIServer" in cmd:
        return True
    return bool(re.search(r"\bvllm serve\b", cmd)) and "VLLM::" not in cmd


def _is_client(cmd: str) -> bool:
    return bool(re.search(r"\bbench serve\b", cmd))


def cpu_p90(path: str, window: tuple[float, float] | None = None) -> dict:
    """p90 of cpu_percent per process of a CpuSampler csv; the max over pids per role (api / client).

    With `window` = (t_wall_start, t_wall_end), only the samples inside it count (AM11: one value per client
    run). A row whose t_wall or cpu_percent does not parse, such as the truncated last line of a sampler that
    was killed mid-write, is skipped and counted in "bad_rows"; "samples" counts the rows that were used."""
    by_pid: dict[tuple[str, str], list[float]] = {}
    bad = used = 0
    with open(path, newline="", errors="replace") as fh:
        for row in csv.DictReader(fh):
            cmd = row.get("cmd") or ""
            role = "client" if _is_client(cmd) else "api" if _is_api_server(cmd) else None
            if role is None:
                continue
            try:
                cpu = float(row["cpu_percent"])
                t = float(row["t_wall"]) if window is not None else None
            except (KeyError, TypeError, ValueError):
                bad += 1
                continue
            if t is not None and not window[0] <= t <= window[1]:
                continue
            used += 1
            by_pid.setdefault((role, str(row.get("pid"))), []).append(cpu)
    out: dict = {"api": None, "client": None, "samples": used, "bad_rows": bad}
    for (role, _), vals in by_pid.items():
        p90 = float(np.percentile(vals, 90))
        out[role] = p90 if out[role] is None else max(out[role], p90)
    return out


def _result_dir_of(argv: Sequence[str]) -> str | None:
    for i, a in enumerate(argv):
        if a == "--result-dir" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--result-dir="):
            return a.split("=", 1)[1]
    return None


def client_window(sub_dir: str, session_dir: str, k: int) -> tuple[float, float] | None:
    """(t_wall_start, t_wall_end) of the `bench serve` command(s) of client run sub-<k>, from cmd.json (C2).

    A cmd.json inside sub-<k>/ covers that run alone. In the session's cmd.json, a client command belongs to
    sub-<k> when its --result-dir is that directory. DP2-rand runs two clients at once: the window spans both."""
    for d, own in ((sub_dir, True), (session_dir, False)):
        path = os.path.join(d, "cmd.json")
        if not os.path.exists(path):
            continue
        cmds = _read_json(path)
        wins = []
        for c in cmds if isinstance(cmds, list) else []:
            argv = [str(a) for a in (c.get("argv") or [])] if isinstance(c, dict) else []
            if not _is_client(" ".join(argv)):
                continue
            rdir = _result_dir_of(argv)
            if own or (rdir is not None and os.path.basename(os.path.normpath(rdir)) == f"sub-{k}"):
                start, end = finite(c.get("t_wall_start")), finite(c.get("t_wall_end"))
                if start is not None and end is not None:
                    wins.append((start, end))
        if wins:
            return min(w[0] for w in wins), max(w[1] for w in wins)
    return None


_CPU_NONE = {"api": None, "client": None, "samples": 0, "bad_rows": 0}


def _cpu_for(sub_dir: str, session_dir: str, k: int) -> tuple[dict, str | None, str | None, list[str]]:
    """AM11 CPU p90 of one client run: (values, scope, note, errors).

    scope "run": sub-<k>/cpu.csv, or the session cpu.csv inside the client's cmd.json window. scope "session":
    no window was found, so the p90 is over the whole session, startup included. Parse problems never drop the
    client result (Review Focus 2): they come back as a note and, when the file is unusable, as an error."""
    sub_csv, session_csv = os.path.join(sub_dir, "cpu.csv"), os.path.join(session_dir, "cpu.csv")
    notes: list[str] = []
    try:
        name = f"sub-{k}/cpu.csv" if os.path.exists(sub_csv) else "cpu.csv"
        if os.path.exists(sub_csv):
            cpu, scope = cpu_p90(sub_csv), "run"
        elif os.path.exists(session_csv):
            try:
                window = client_window(sub_dir, session_dir, k)
            except _PARSE_ERRORS as e:
                window = None
                notes.append(f"cmd.json: {e}")
            if window is not None:
                cpu, scope = cpu_p90(session_csv, window), "run"
                if not cpu["samples"]:
                    notes.append(f"cpu.csv: no API-server or client samples in the client window "
                                 f"[{window[0]:.3f}, {window[1]:.3f}]")
            else:
                cpu, scope = cpu_p90(session_csv), "session"
                notes.append("cpu.csv: no client window in cmd.json, p90 over the whole session")
        else:
            return dict(_CPU_NONE), None, None, []
    except (*_PARSE_ERRORS, csv.Error) as e:
        msg = f"{name}: {e}"
        return dict(_CPU_NONE), None, msg, [msg]
    if cpu["bad_rows"]:
        notes.append(f"{name}: skipped {cpu['bad_rows']} unparseable rows")
    return cpu, scope, "; ".join(notes) or None, []


def _throttled(sub_dir: str, session_dir: str) -> tuple[bool | None, str | None]:
    """Throttle flag from gpu.csv. The session gpu.csv covers the whole session (nvidia-smi timestamps are box
    local time, so they are not windowed per client run)."""
    for d in (sub_dir, session_dir):
        path = os.path.join(d, "gpu.csv")
        if os.path.exists(path):
            name = os.path.relpath(path, session_dir)
            try:
                flags = monitor.throttle_flags(path)
            except (*_PARSE_ERRORS, UnicodeDecodeError) as e:
                return None, f"{name}: {e}"
            return (bool(flags["throttled"]) if flags["rows"] else None), None
    return None, None


def _metric_deltas(sub_dir: str) -> tuple[dict[str, float] | None, str | None]:
    try:
        with open(os.path.join(sub_dir, "metrics_before.prom")) as f:
            before = promparse.scrape_summary(f.read())
        with open(os.path.join(sub_dir, "metrics_after.prom")) as f:
            after = promparse.scrape_summary(f.read())
        return promparse.deltas(before, after), None
    except _PARSE_ERRORS as e:
        return None, f"metrics: {e}"


def _sub_dirs(run_dir: str) -> list[tuple[int, str]]:
    out = []
    for path in glob.glob(os.path.join(run_dir, "sub-*")):
        m = re.fullmatch(r"sub-(\d+)", os.path.basename(path))
        if m and os.path.isdir(path):
            out.append((int(m.group(1)), path))
    return sorted(out)


def _load_sub_result(sub_dir: str) -> results.ServeResult:
    """result.json, or the halves of a DP2-rand run (result*.json) merged per D4-28."""
    files = sorted(glob.glob(os.path.join(sub_dir, "result*.json")))
    if not files:
        raise ValueError("no result.json")
    parts = [results.load_serve_result(p) for p in files]
    return parts[0] if len(parts) == 1 else results.merge_serve_results(parts)


def _sub_row(run: Run, k: int, sub: str) -> tuple[dict, results.ServeResult, list[str]]:
    """One online_runs row from a sub-<k>/ client run (ruling R4), plus the monitor files it could not read."""
    meta_path = os.path.join(sub, "meta.json")
    meta = _read_json(meta_path) if os.path.exists(meta_path) else {}
    r = _load_sub_result(sub)
    vpath = os.path.join(sub, "validation.json")
    validation = _read_json(vpath) if os.path.exists(vpath) else {}
    rate = _rate(meta.get("rate"))
    extra = {"run_id": run.run_id, "sub": k, "tier": run.tier, "phase": meta.get("phase", run.p("phase")),
             "seed": meta.get("seed"), "config": meta.get("config", run.config),
             "arm": meta.get("arm", run.arm), "round": int(meta.get("round", run.round)),
             "rate_target": rate if rate is not None else r.request_rate}
    row = results.online_row(r, extra)
    deltas, metrics_note = _metric_deltas(sub)
    cpu, cpu_scope, cpu_note, errors = _cpu_for(sub, run.path, k)
    throttled, gpu_error = _throttled(sub, run.path)
    if gpu_error:
        errors.append(gpu_error)
    flags = [str(f) for f in validation.get("flags", [])]
    preempt = deltas.get("vllm:num_preemptions_total") if deltas else None
    if preempt and "preempted" not in flags:
        flags.append("preempted")
    if throttled:
        flags.append("throttled")
    if any(cpu[role] is not None and cpu[role] > CPU_FLAG_PCT for role in ("api", "client")):
        flags.append("cpu>80%" if cpu_scope == "run" else "cpu>80% (session-wide p90)")
    violations = [v for v in (row["violations"], "; ".join(map(str, validation.get("violations", [])))) if v]
    row.update({
        "valid": bool(row["valid"] and validation.get("valid", True)),
        "violations": "; ".join(violations),
        "preemptions": preempt,
        "prefix_cache_hits": deltas.get("vllm:prefix_cache_hits_total") if deltas else None,
        "prefix_cache_queries": deltas.get("vllm:prefix_cache_queries_total") if deltas else None,
        "cpu_api_p90": cpu["api"], "cpu_client_p90": cpu["client"], "cpu_scope": cpu_scope,
        "cpu_samples": cpu["samples"], "throttled": throttled, "flags": "; ".join(flags),
        "metrics_note": metrics_note,
        "monitor_note": "; ".join(n for n in (cpu_note, gpu_error) if n) or None,
    })
    return row, r, errors


def _online(runs: Sequence[Run], gaps: _Gaps) -> list[_Sub]:
    subs = []
    for run in _of_kind(runs, "serve_session"):
        dirs = _sub_dirs(run.path)
        if not dirs and run.status == "done":
            gaps.add("unreadable", "done, but no sub-<k>/ client runs", run)
        monitor_errors: set[str] = set()
        for k, sub in dirs:
            try:
                row, r, errors = _sub_row(run, k, sub)
            except _PARSE_ERRORS as e:
                gaps.add("unreadable", f"sub-{k}: {e}", run)
                continue
            for msg in errors:
                if msg not in monitor_errors:          # a broken session file is listed once, not per client run
                    monitor_errors.add(msg)
                    gaps.add("partial", f"monitor file unreadable, client results kept: {msg}", run)
            if not row["valid"]:
                gaps.add("invalid", f"sub-{k} (rate {row['rate_target']}): {row['violations']}", run)
            subs.append(_Sub(row, r))
    return subs


def _is_sat(row: Mapping) -> bool:
    return row["rate_target"] is not None and math.isinf(row["rate_target"])


def _saturation(subs: Sequence[_Sub], grid_mu: Mapping[str, float]) -> list[dict]:
    """AM6: mu from the token-emission timeline per saturation client run; preemptions per 1K requests."""
    rows = []
    for s in subs:
        r, row = s.result, s.row
        if not _is_sat(row):
            continue
        mu = finite(stats.saturation_tps(r.start_times, r.ttfts, r.itls, r.output_lens, r.ok_mask))
        pre = row["preemptions"]
        rows.append({"row": "seed", "run_id": row["run_id"], "sub": row["sub"], "tier": row["tier"],
                     "phase": row["phase"], "config": row["config"], "arm": row["arm"], "round": row["round"],
                     "seed": row["seed"], "valid": row["valid"], "completed": r.completed, "mu_tps": mu,
                     "mu_rps": mu / ONLINE_OUTPUT_LEN if mu is not None else None, "preemptions": pre,
                     "preempt_per_1k": pre / r.completed * 1000 if pre is not None and r.completed else None,
                     "flags": row["flags"]})
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["valid"] and row["mu_tps"] is not None:
            groups.setdefault((row["config"], row["arm"], row["phase"]), []).append(row)
    for (config, arm, phase), g in sorted(groups.items()):
        mu = median_or_none(x["mu_tps"] for x in g)
        rows.append({"row": "median", "config": config, "arm": arm, "phase": phase, "n_seeds": len(g),
                     "mu_tps": mu, "mu_rps": mu / ONLINE_OUTPUT_LEN if mu is not None else None,
                     "preempt_per_1k": median_or_none(x["preempt_per_1k"] for x in g),
                     "grid_mu_rps": grid_mu.get(config) if arm == "base" and phase == "sat" else None})
    return rows


def _rate_points(subs: Sequence[_Sub]) -> dict[tuple[int, str], dict[str, list[goodput.RatePoint]]]:
    """(round, arm) -> config -> RatePoints of the valid sweep client runs."""
    out: dict[tuple[int, str], dict[str, list[goodput.RatePoint]]] = {}
    for s in subs:
        row = s.row
        if row["phase"] != "sweep" or not row["valid"] or row["rate_target"] is None or _is_sat(row):
            continue
        ms = results.request_metrics(s.result)
        point = goodput.RatePoint(row["rate_target"], tuple(m["ttft_s"] for m in ms), tuple(m["tpot_s"] for m in ms))
        out.setdefault((row["round"], row["arm"]), {}).setdefault(row["config"], []).append(point)
    return out


def _goodput(points: Mapping[tuple[int, str], Mapping[str, Sequence[goodput.RatePoint]]]) -> list[dict]:
    rows = []
    for (rnd, arm), by_config in sorted(points.items()):
        gpus = {c: CONFIG_GPUS.get(c, 1) for c in by_config}
        for ttft in (TTFT_SLO_S, *TTFT_SLO_SENSITIVITY_S):
            for row in goodput.goodput_rows(by_config, gpus, TPOT_SLOS_MS, ttft):
                rows.append({"round": rnd, "arm": arm, **row})
    return rows


def _pooled(points: Mapping[tuple[int, str], Mapping[str, Sequence[goodput.RatePoint]]],
            keys: Sequence[tuple[int, str]]) -> dict[str, list[goodput.RatePoint]]:
    """TP2/DP2 requests pooled over rounds per grid rate (the grid is frozen, AM7)."""
    pooled: dict[str, dict[float, list[goodput.RatePoint]]] = {}
    for key in keys:
        for config in ("TP2", "DP2"):
            for p in points[key].get(config, []):
                pooled.setdefault(config, {}).setdefault(round(p.rate, 6), []).append(p)
    return {c: [goodput.RatePoint(rate, tuple(x for p in ps for x in p.ttft_s), tuple(x for p in ps for x in p.tpot_s))
                for rate, ps in sorted(by_rate.items())] for c, by_rate in pooled.items()}


def _s_star(points: Mapping[tuple[int, str], Mapping[str, Sequence[goodput.RatePoint]]],
            gp_rows: Sequence[dict]) -> list[dict]:
    """AM8: s* per round, then the median, the range and a request-level bootstrap CI over pooled rounds."""
    rows = []
    keys = [k for k in sorted(points) if k[1] == "base" and "TP2" in points[k] and "DP2" in points[k]]
    for rnd, arm in keys:
        g = {c: [r["goodput_rps"] for r in sorted(gp_rows, key=lambda r: r["tpot_slo_ms"])
                 if (r["round"], r["arm"], r["config"], r["ttft_slo_s"]) == (rnd, arm, c, TTFT_SLO_S)]
             for c in ("TP2", "DP2")}
        star = goodput.crossover(TPOT_SLOS_MS, g["TP2"], g["DP2"])
        pair = {c: points[(rnd, arm)][c] for c in ("TP2", "DP2")}
        lo, hi = goodput.bootstrap_s_star(pair, TPOT_SLOS_MS, TTFT_SLO_S, n_boot=S_STAR_BOOT, seed=rnd)
        rows.append({"row": "round", "round": rnd, "arm": arm, "s_star_ms": star, "ci_lo_ms": lo, "ci_hi_ms": hi,
                     "note": None if star is not None else "no crossover in the swept TPOT SLOs"})
    if rows:
        stars = [r["s_star_ms"] for r in rows if r["s_star_ms"] is not None]
        lo, hi = goodput.bootstrap_s_star(_pooled(points, keys), TPOT_SLOS_MS, TTFT_SLO_S, n_boot=S_STAR_BOOT, seed=0)
        rows.append({"row": "all", "round": None, "arm": "base", "s_star_ms": median_or_none(stars),
                     "s_star_min": min(stars) if stars else None, "s_star_max": max(stars) if stars else None,
                     "n_rounds": len(rows), "n_crossover": len(stars), "ci_lo_ms": lo, "ci_hi_ms": hi,
                     "note": "CI: request-level bootstrap over the rounds pooled per grid rate"})
    return rows


# ------------------------------------------------------------------------------------------ comm

def _comm_point(kind: str, run: Run, d: Mapping) -> dict:
    """One tidy row from an M1/M2/M3/M4 row shape (Task 12 parsers)."""
    row = {"row": "point", "source": kind, "run_id": run.run_id, "impl": d.get("impl"), "variant": d.get("variant"),
           "mode": d.get("mode"), "bytes": int(d["bytes"]), "lat_us": None, "p25_us": d.get("p25_us"),
           "p75_us": d.get("p75_us"), "algbw_GBps": d.get("algbw_GBps"), "busbw_GBps": d.get("busbw_GBps"),
           "algo": d.get("algo"), "proto": d.get("proto"), "backend": d.get("backend"), "oneshot": d.get("oneshot")}
    if "op" in d:                                          # M2: fused AR+RMSNorm per op, ms
        row.update(impl=d["op"], lat_us=finite(d["ms"]) and d["ms"] * 1e3)
    elif "median_us" in d:                                 # M3
        row["lat_us"] = finite(d["median_us"])
        algo, _, proto = str(d.get("variant") or "").partition(":")
        row.update(algo=None if algo in ("", "none") else algo, proto=None if proto in ("", "none") else proto)
    elif "mean_us" in d:                                   # M1: graph-replay mean
        row["lat_us"] = finite(d["mean_us"])
    else:                                                  # M4: nccl-tests
        row.update(variant=d.get("place"), lat_us=finite(d.get("p50_us")) or finite(d.get("time_us")))
    return row


def _comm(runs: Sequence[Run], gaps: _Gaps) -> list[dict]:
    points = []
    for run in _of_kind(runs, "comm_m1", "comm_m2", "comm_m3", "comm_m4"):
        path = os.path.join(run.path, "comm_rows.jsonl")
        if not os.path.exists(path):
            if run.status == "done":
                gaps.add("unreadable", "done, but no comm_rows.jsonl", run)
            continue
        try:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        points.append(_comm_point(run.kind, run, json.loads(line)))
        except _PARSE_ERRORS as e:
            gaps.add("unreadable", f"comm_rows.jsonl: {e}", run)
    groups: dict[tuple, list[dict]] = {}
    for p in points:
        if p["lat_us"] is not None:
            groups.setdefault((p["source"], p["impl"], p["variant"], p["mode"]), []).append(p)
    fits = []
    for (source, impl, variant, mode), g in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        alpha, beta = stats.fit_alpha_beta([p["bytes"] for p in g], [p["lat_us"] * 1e-6 for p in g])
        fits.append({"row": "fit", "source": source, "run_id": g[0]["run_id"], "impl": impl, "variant": variant,
                     "mode": mode, "n": len(g), "alpha_us": finite(alpha) and alpha * 1e6,
                     "beta_GBps": finite(beta) and beta / 1e9})
    return points + fits


# ------------------------------------------------------------------------------------------ KV, traces

def _kv(runs: Sequence[Run], gaps: _Gaps) -> tuple[list[dict], list[tuple[Run, dict]]]:
    """Every engine boot's KV line; the first boot of each config is cold_boot (D5-22)."""
    rows, effs, boots = [], [], {}
    for run in runs:                                        # oldest first
        path = os.path.join(run.path, "effective_config.json")
        if not os.path.exists(path):
            continue
        try:
            eff = _read_json(path)
            tokens = [int(t) for t in eff.get("kv_cache_tokens") or []]
        except _PARSE_ERRORS as e:
            gaps.add("unreadable", f"effective_config.json: {e}", run)
            continue
        effs.append((run, eff))
        avail = list(eff.get("available_kv_gib") or [])
        conc = list(eff.get("max_concurrency") or [])
        base = {"run_id": run.run_id, "kind": run.kind, "tier": run.tier, "config": run.config, "arm": run.arm,
                "status": run.status, "violations": len(eff.get("violations") or []), "t_order": run.t_order}
        if not tokens:
            rows.append({**base, "engine": None, "kv_tokens": None, "boot_index": None, "cold_boot": None,
                         "note": "no 'GPU KV cache size' line"})
            continue
        index = boots.get(run.config, 0)
        boots[run.config] = index + 1
        for i, t in enumerate(tokens):
            rows.append({**base, "engine": i, "kv_tokens": t, "available_kv_gib": avail[i] if i < len(avail) else None,
                         "max_concurrency": conc[i] if i < len(conc) else None, "boot_index": index,
                         "cold_boot": index == 0, "note": None})
    return rows, effs


def h6_bounds(row: Mapping, exact_steps: int | None) -> dict[str, int]:
    """Lower/upper bounds on the steps of one rank with exactly the expected AR and AG op counts.

    exact_steps (counted from trace.sqlite) is exact. From the summary alone: a single observed value
    decides it; a mode other than the expected count means at most half the steps are exact; otherwise
    at least one step is off and nothing more is known."""
    steps = int(row["steps"] or 0)
    if exact_steps is not None:
        return {"h6_exact_min": exact_steps, "h6_exact_max": exact_steps}
    tp = int(row["tp"])
    bounds = []
    for prefix, expected in (("ar", model.allreduces_per_step(tp)), ("ag", model.allgathers_per_step(tp))):
        lo, hi, mode = row[f"{prefix}_min"], row[f"{prefix}_max"], row[f"{prefix}_mode"]
        if lo is None or hi is None:
            bounds.append((0, 0))
        elif lo == hi:
            bounds.append((steps, steps) if lo == expected else (0, 0))
        elif mode != expected:
            bounds.append((0, steps // 2))
        else:
            bounds.append((0, steps - 1))
    (ar_lo, ar_hi), (ag_lo, ag_hi) = bounds
    return {"h6_exact_min": max(0, ar_lo + ag_lo - steps), "h6_exact_max": min(ar_hi, ag_hi)}


def _pure_decode(step: Mapping) -> bool:
    return step["n_ctx_reqs"] == 0 and step["n_ctx_tokens"] == 0 and step["n_gen_reqs"] == step["n_gen_tokens"] > 0


def sqlite_steps(path: str, tp: int) -> list[list[dict]]:
    """Per rank, one flat row per step of trace.sqlite (spec 7.6 trace_steps): the op counts, GPU busy, the
    category ms attributed inside the step's own kernels (AM16: overlap goes to the earlier start), and the
    ms of vLLM's standalone fused_add_rms_norm kernels (the AM16 comm subtraction)."""
    td = traces.load_trace(path)
    ar, ag = model.allreduces_per_step(tp), model.allgathers_per_step(tp)
    out = []
    for pid, dev in traces.worker_ranks(td):
        steps = traces.assign_steps(td, pid, dev)
        rows = []
        for s in steps:
            ks = s["kernels"]
            window = (min(k.start for k in ks), max(k.end for k in ks)) if ks else (0, 0)
            cats = traces.attribute(ks, window) if ks else dict.fromkeys(kernels.CATEGORIES, 0)
            norm_ns = traces.union_ns((k.start, k.end) for k in ks if FUSED_ADD_RMS_NORM.search(k.name))
            row = {"pid": pid, "device": dev, "step": s["index"], "nvtx": s["nvtx"],
                   "n_ctx_reqs": s["n_ctx_reqs"], "n_ctx_tokens": s["n_ctx_tokens"], "n_gen_reqs": s["n_gen_reqs"],
                   "n_gen_tokens": s["n_gen_tokens"], "pure_decode": _pure_decode(s),
                   "last": s["index"] == len(steps) - 1, "step_ms": (s["end"] - s["start"]) / 1e6,
                   "gpu_busy_ms": s["gpu_busy_ns"] / 1e6, "ar_ops": s["ar_ops"], "ag_ops": s["ag_ops"],
                   "exact_counts": s["ar_ops"] == ar and s["ag_ops"] == ag}
            row.update({f"cat_{c}_ms": cats.get(c, 0) / 1e6 for c in kernels.CATEGORIES})
            row["fused_add_rms_norm_ms"] = norm_ns / 1e6
            rows.append(row)
        out.append(rows)
    return out


def _stat_mean(steps: Sequence[Mapping], key: str) -> float | None:
    """Mean over the pure decode steps when there are any (as traces.summarize_trace does), else all steps."""
    stat = [s for s in steps if s["pure_decode"]] or list(steps)
    return float(np.mean([s[key] for s in stat])) if stat else None


def _traces(runs: Sequence[Run], gaps: _Gaps) -> tuple[list[dict], list[dict]]:
    """(trace_summary rows, one per rank; trace_steps rows, one per rank and step where trace.sqlite exists)."""
    rows, step_rows = [], []
    for run in _of_kind(runs, "trace"):
        path = os.path.join(run.path, "trace_summary.json")
        if not os.path.exists(path):
            if run.status == "done":
                gaps.add("unreadable", "done, but no trace_summary.json", run)
            continue
        try:
            summ = _read_json(path)
            tp = int(run.tp or len(summ["ranks"]) or 1)
            gate = summ.get("gate") or {}
            gate_ok = gate.get("ok") is True
            sqlite_path = os.path.join(run.path, "trace.sqlite")
            by_rank = None
            if os.path.exists(sqlite_path):
                try:
                    by_rank = sqlite_steps(sqlite_path, tp)
                except (sqlite3.Error, *_PARSE_ERRORS) as e:
                    gaps.add("unreadable", f"trace.sqlite: {e}", run)
            if not gate_ok:
                gaps.add("trace_gate", ("; ".join(map(str, gate.get("reasons", []))) or "gate failed")
                         + " (excluded from H6, H7 and the AM16 comm time)", run)
            run_rows = []
            for i, rk in enumerate(summ["ranks"]):
                ar, ag = rk.get("ar_ops_per_step") or {}, rk.get("ag_ops_per_step") or {}
                frac = rk.get("category_frac") or {}
                step_ms, busy = finite(rk.get("mean_step_ms")), finite(rk.get("busy_frac"))
                row = {"run_id": run.run_id, "tier": run.tier, "config": run.config, "arm": run.arm, "tp": tp,
                       "points": run.p("points"), "gate_ok": gate_ok,
                       "gate_reasons": "; ".join(map(str, gate.get("reasons", []))),
                       "idle_est": finite(summ.get("idle_est")),
                       "ar_wire_ms": finite(summ.get("ar_wire_s_per_step")) and summ["ar_wire_s_per_step"] * 1e3,
                       "ar_sync_ms": finite(summ.get("ar_sync_wait_s_per_step")) and summ["ar_sync_wait_s_per_step"] * 1e3,
                       "launch_ts_missing": summ.get("launch_ts_missing"),
                       "rank": rk.get("rank", i), "pid": rk.get("pid"), "device": rk.get("device"),
                       "steps": rk.get("steps"), "decode_steps": rk.get("decode_steps"),
                       "ar_min": ar.get("min"), "ar_max": ar.get("max"), "ar_mode": ar.get("mode"),
                       "ag_min": ag.get("min"), "ag_max": ag.get("max"), "ag_mode": ag.get("mode"),
                       "mean_step_ms": step_ms, "busy_frac": busy, "unclassified_frac": finite(frac.get("other")),
                       "unclassified_top": json.dumps(rk.get("unclassified_top", [])[:5])}
                for c in kernels.CATEGORIES:        # per-step ms: the step split by the window's busy shares
                    row[f"cat_{c}_ms"] = (step_ms * busy * frac.get(c, 0.0)
                                          if step_ms is not None and busy is not None else None)
                row["idle_ms"] = step_ms * (1 - busy) if step_ms is not None and busy is not None else None
                steps = by_rank[i] if by_rank is not None and i < len(by_rank) else None
                if steps is not None:
                    row.update(h6_source="trace.sqlite", h6_steps=len(steps),
                               ar_ms=_stat_mean(steps, "cat_all_reduce_ms"), ar_ms_source="trace.sqlite",
                               fused_add_rms_norm_ms=_stat_mean(steps, "fused_add_rms_norm_ms"))
                    row.update(h6_bounds(row, sum(s["exact_counts"] for s in steps)))
                    for s in steps:
                        step_rows.append({"run_id": run.run_id, "config": run.config, "arm": run.arm, "tp": tp,
                                          "points": run.p("points"), "gate_ok": gate_ok, "rank": row["rank"], **s})
                else:
                    row.update(h6_source="summary", h6_steps=row["steps"], ar_ms=row["cat_all_reduce_ms"],
                               ar_ms_source="summary shares", fused_add_rms_norm_ms=None)
                    row.update(h6_bounds(row, None))
                run_rows.append(row)
            rows += run_rows
        except _PARSE_ERRORS as e:
            gaps.add("unreadable", f"trace_summary.json: {e}", run)
    _comm_time(rows)
    return rows, step_rows


def _comm_time(rows: Sequence[dict]) -> None:
    """AM16: TP2 comm time per step = fused AR time - TP1's per-step fused_add_rms_norm time.

    The default path (base, G1, G2) fuses residual add + RMSNorm into the all-reduce kernel; TP1 runs that
    work as standalone fused_add_rms_norm kernels, so their time is subtracted. AR1-AR3 run the RMSNorm as its
    own kernel, so their AR time is comm time as it stands. The TP1 reference is a gate-passing TP1 trace at
    the same points (same arm, else base), rank 0, counted from its trace.sqlite."""
    tp1 = {}
    for r in rows:
        if r["config"] == "TP1" and r["gate_ok"] and r["rank"] == 0 and r.get("fused_add_rms_norm_ms") is not None:
            tp1.setdefault((r["arm"], r["points"]), r)
    for r in rows:
        r.update(ar_fused=None, tp1_norm_ms=None, comm_ms=None, comm_note=None)
        if r["tp"] != 2:
            continue
        fused = r["arm"] not in UNFUSED_AR_ARMS
        r["ar_fused"] = fused
        if not r["gate_ok"]:
            r["comm_note"] = "completeness gate failed"
        elif r["ar_ms"] is None:
            r["comm_note"] = "no all-reduce time"
        elif not fused:
            r.update(comm_ms=r["ar_ms"], comm_note="unfused path: the AR kernels are comm only")
        else:
            ref = tp1.get((r["arm"], r["points"])) or tp1.get(("base", r["points"]))
            if ref is None:
                r["comm_note"] = (f"no gate-passing TP1 {r['points']} trace with trace.sqlite for the "
                                  "fused_add_rms_norm time")
            else:
                r.update(tp1_norm_ms=ref["fused_add_rms_norm_ms"], comm_ms=r["ar_ms"] - ref["fused_add_rms_norm_ms"],
                         comm_note=f"fused AR ({r['ar_ms_source']}) - TP1 {ref['arm']} fused_add_rms_norm")


# ------------------------------------------------------------------------------------------ gaps, evidence

def _fi_backend(effs: Sequence[tuple[Run, dict]]) -> str | None:
    for run, eff in effs:
        if run.kind == "smoke" and run.config == "TP2":
            return eff.get("fi_backend")
    return None


def _expected_missing(runs: Sequence[Run], raw_dir: str, effs: Sequence[tuple[Run, dict]],
                      skipped: Mapping[str, str], gaps: _Gaps) -> None:
    """Runs the matrix of the tiers seen expects but that have no record and were not reported skipped."""
    tiers = sorted({r.tier for r in runs if r.tier in matrix.TIERS})
    if not tiers:
        return
    grid = None
    grid_path = os.path.join(raw_dir, "rate_grid.json")
    if os.path.exists(grid_path):
        try:
            grid = _read_json(grid_path)["grid"]
        except _PARSE_ERRORS as e:
            gaps.add("unreadable", f"rate_grid.json: {e}", run_id="rate_grid.json")
    elif set(tiers) & {"P1", "P2"}:
        gaps.add("missing", "no raw/rate_grid.json: the P1 sweeps and the grid-dependent P2 sessions cannot be "
                            "listed", run_id="rate_grid.json")
    # The runner plans ROUNDS rounds (T16 default); a session that died before its last rounds leaves no record
    # of them, so the rounds seen are only a lower bound (Review Focus 2).
    sweep_rounds = [r.round for r in runs if r.kind == "serve_session" and r.p("phase") == "sweep" and r.tier == "P1"]
    rounds = max([ROUNDS, *sweep_rounds])
    try:
        specs = matrix.build_matrix(tiers, grid, _fi_backend(effs), rounds)
    except (ValueError, KeyError, TypeError) as e:
        gaps.add("unreadable", f"cannot rebuild the run matrix: {e}", run_id="matrix")
        return
    present = {r.run_id for r in runs}
    for s in specs:
        if s.run_id not in present and s.run_id not in skipped:
            gaps.add("missing", f"expected by the {s.tier} matrix, no run record", run_id=s.run_id, config=s.config,
                     kind=s.kind)


def _status_gaps(runs: Sequence[Run], skipped: Mapping[str, str], gaps: _Gaps) -> None:
    present = {r.run_id: r for r in runs}
    for r in runs:
        if r.status == "incomplete" and r.run_id in skipped:
            gaps.add("skipped", skipped[r.run_id], r)
        elif r.status in ("failed", "skipped", "incomplete"):
            gaps.add(r.status, r.reason or r.status, r)
    for run_id, reason in skipped.items():
        if run_id not in present:
            parts = run_id.split("-")
            gaps.add("skipped", reason, run_id=run_id, config=parts[2] if len(parts) > 2 else None,
                     kind=parts[1] if len(parts) > 1 else None)


CONFOUNDERS = (
    # (confounder, control, evidence captured, glob patterns under the results dir), spec section 6
    ("CUDA-graph mode", "pinned FULL_AND_PIECEWISE; arms G1/G2", "capture log lines; trace graphId per kernel",
     ("raw/*/effective_config.json", "raw/*/trace_summary.json")),
    ("Prefix caching", "off plus random prompts; arm PC-on", "/metrics prefix-cache counters before/after",
     ("raw/*/sub-*/metrics_before.prom", "raw/*/sub-*/metrics_after.prom")),
    ("Chunked prefill / batch budget", "pinned 8192 everywhere; prefill sizes <= 8192; two-length decode difference",
     "Chunked prefill is enabled with max_num_batched_tokens=8192.", ("raw/*/effective_config.json",)),
    ("Warmup / cold caches", "one throwaway start per config; offline 5/3 warmups; online --num-warmups 16",
     "per-iteration latencies kept; startup timings", ("raw/*/points/point-*.json", "raw/*/cmd.json")),
    ("Tokenizer/detokenizer CPU", "token-ID prompts offline; tokbench; per-process CPU sampling",
     "tokbench.json, cpu.csv", ("raw/*/tokbench.json", "raw/*/cpu.csv", "raw/*/sub-*/cpu.csv")),
    ("All-reduce implementation", "baseline declared (FlashInfer fused); A-AR ladder",
     "backend log lines; kernel names per step", ("raw/*/effective_config.json", "raw/*/trace_summary.json")),
    ("Scaling baseline", "TP1 on the same box, same flags; DP2 as the real alternative", "same env.json",
     ("raw/*/env.json",)),
    ("Version drift", "image digest, vLLM 0.30.0, torch 2.13.0, NCCL, FlashInfer, driver, nsys",
     "env.json, pip freeze, log lines", ("raw/*/env.json", "raw/*/*freeze*", "raw/*/effective_config.json")),
    ("NVLink topology", "NV18 hard gate", "topo.txt, nvlink.txt, M1/M3 busbw",
     ("raw/*/topo.txt", "raw/*/nvlink.txt", "raw/*/comm_rows.jsonl")),
    ("Clocks / thermals / power", "200 ms monitor during every run; throttle-flagged runs", "gpu.csv",
     ("raw/*/gpu.csv", "raw/*/sub-*/gpu.csv")),
    ("KV preemption", "max_num_seqs and the rate grid; detect", "/metrics preemption counters before/after",
     ("raw/*/sub-*/metrics_after.prom",)),
    ("Arrival process and seeds", "Poisson, seeds recorded; DP2-rand halves use different seeds",
     "client argv; result JSON", ("raw/*/sub-*/meta.json", "raw/*/sub-*/result*.json", "raw/*/cmd.json")),
    ("Sampler", "T=1.0, top_p=1.0 everywhere; --generation-config vllm", "server log (no override warning)",
     ("raw/*/effective_config.json",)),
    ("Engine-default drift (offline vs online)", "every engine flag pinned from one source",
     "effective_config.json from logs", ("raw/*/effective_config.json",)),
    ("Executor asymmetry (uni vs mp)", "declared; arm A-EXEC", "config log line", ("raw/*/effective_config.json",)),
    ("Box hygiene", "no foreign GPU processes, memory < 1 GiB before every start", "preflight.json, per-run pre-check",
     ("raw/*/preflight.json",)),
    ("Frontend capacity", "1 API server in every config; arm A-API2", "CPU% of the API-server process",
     ("raw/*/cpu.csv", "raw/*/sub-*/cpu.csv")),
    ("Drift over the session", "online rounds interleave configs (Latin square)", "round index in every row",
     ("raw/*/sub-*/meta.json",)),
)


def _confounder_checks(effs: Sequence[tuple[Run, dict]], online: Sequence[dict]) -> dict[int, str]:
    chunked = [e.get("chunked_prefill_tokens") for _, e in effs]
    executors = sorted({str(e.get("executor")) for _, e in effs})
    base_hits = sum(r["prefix_cache_hits"] or 0 for r in online if r["arm"] != "PCon")
    return {
        2: f"prefix-cache hits outside arm PCon: {base_hits:g}",
        3: f"{sum(c == logparse.EXPECTED_MAX_NUM_BATCHED_TOKENS for c in chunked)}/{len(chunked)} boots log "
           f"max_num_batched_tokens={logparse.EXPECTED_MAX_NUM_BATCHED_TOKENS}",
        10: f"{sum(bool(r['throttled']) for r in online)} throttle-flagged client runs",
        11: f"{sum(bool(r['preemptions']) for r in online)} client runs with preemptions",
        13: f"{sum(bool(e.get('sampling_override')) for _, e in effs)} boots with the sampling-override warning",
        15: f"executors seen: {', '.join(executors) if executors else 'none'}",
        17: f"{sum('cpu>80%' in (r['flags'] or '') for r in online)} client runs with a process above "
            f"{CPU_FLAG_PCT:g}% of one core",
    }


def _confounders(results_dir: str, effs: Sequence[tuple[Run, dict]], online: Sequence[dict]) -> list[dict]:
    checks = _confounder_checks(effs, online)
    rows = []
    for i, (name, control, evidence, patterns) in enumerate(CONFOUNDERS, start=1):
        found = list(dict.fromkeys(os.path.relpath(p, results_dir) for pat in patterns     # pattern order
                                   for p in sorted(glob.glob(os.path.join(results_dir, pat)))))
        rows.append({"id": i, "confounder": name, "control": control, "evidence": evidence,
                     "patterns": " ".join(patterns), "n_found": len(found),
                     "found": "; ".join(found[:MAX_EVIDENCE_PATHS]), "check": checks.get(i)})
    return rows


# ------------------------------------------------------------------------------------------ hypotheses

def _offline_step(tables: Mapping, config: str, arm: str, batch: int, kind: str = "decode",
                  input_len: int = DECODE_INPUT_LEN) -> float | None:
    return median_or_none(_metric(r) for r in tables.get("offline_points", [])
                   if (r.get("config"), r.get("arm"), r.get("kind"), r.get("batch"), r.get("input_len"))
                   == (config, arm, kind, batch, input_len) and not r.get("derived"))


def _in_band(x: float, band: Sequence[float] | None) -> bool:
    return band is not None and band[0] <= x <= band[1]


def _fmt(x: object) -> str:
    v = finite(x)
    return "n/a" if v is None else f"{v:.4g}"


def _h(id_: str, statement: str, measured: object, band: object, verdict: str, note: str) -> dict:
    return {"id": id_, "statement": statement, "measured": measured, "band": band, "verdict": verdict, "note": note}


def _ratio_hyp(tables: Mapping, band: Sequence[float] | None, id_: str, statement: str,
               num: tuple[str, str], den: tuple[str, str]) -> dict:
    """A b1 decode step ratio num/den against a model band (H1, H8)."""
    b = min(DECODE_BATCHES)
    t = {k: _offline_step(tables, *k, b) for k in (num, den)}
    missing = [f"{c}/{a}" for (c, a), v in t.items() if not v]
    if missing:
        return _h(id_, statement, None, band, "insufficient_data",
                  f"missing offline decode step at batch {b} for: {', '.join(missing)}")
    ratio = t[num] / t[den]
    if band is None:
        return _h(id_, statement, ratio, band, "insufficient_data", "no model band in predictions()")
    return _h(id_, statement, ratio, band, "hit" if _in_band(ratio, band) else "miss",
              f"{num[0]}/{num[1]} {t[num] * 1e3:.4g} ms vs {den[0]}/{den[1]} {t[den] * 1e3:.4g} ms at batch {b}")


def _h2(tables: Mapping, bands: Mapping) -> dict:
    statement = "TP2 scaling efficiency e = speedup/2 rises with tokens per step, in decode and in prefill"
    band = {"decode": bands.get("H2"), "prefill": bands.get("H2_prefill")}
    rows = {r["regime"]: r for r in tables.get("efficiency_delta", [])}
    notes, missing, ok = [], [], True
    measured = {}
    for regime in ("decode", "prefill"):
        r = rows.get(regime)
        if r is None or r.get("delta") is None:
            missing.append(f"{regime}: {r.get('missing') if r else 'no offline points'}")
            continue
        measured[regime] = r["delta"]
        ci = (r.get("ci_lo"), r.get("ci_hi"))
        notes.append(f"{regime} e({r['x_hi']}) - e({r['x_lo']}) = {_fmt(r['delta'])}, 95% CI "
                     f"[{_fmt(ci[0])}, {_fmt(ci[1])}]")
        ok &= r["delta"] > 0 and ci[0] is not None and ci[0] > 0
    if missing:
        return _h("H2", statement, measured or None, band, "insufficient_data", "missing " + "; ".join(missing))
    return _h("H2", statement, measured, band, "hit" if ok else "miss", "; ".join(notes))


def _h3(tables: Mapping, bands: Mapping) -> dict:
    statement = "DP2's saturation throughput exceeds TP2's in every seed"
    mu: dict[str, dict] = {"TP2": {}, "DP2": {}}
    seen: set = set(SAT_SEEDS)                      # the planned seeds (matrix sat params), plus any recorded
    for r in tables.get("saturation", []):
        if (r.get("row"), r.get("arm"), r.get("phase")) == ("seed", "base", "sat") and r.get("config") in mu:
            seen.add(r.get("seed"))
            if r.get("valid") and r.get("mu_tps") is not None:
                mu[r["config"]][r["seed"]] = r["mu_tps"]
    band = bands.get("H3")
    absent = [c for c, v in mu.items() if not v]
    if absent:
        return _h("H3", statement, None, band, "insufficient_data",
                  f"no valid base saturation runs for {', '.join(absent)}")
    seeds = sorted(set(mu["TP2"]) & set(mu["DP2"]), key=str)
    ratios = [mu["DP2"][s] / mu["TP2"][s] for s in seeds]
    note = "DP2/TP2 per seed: " + (", ".join(f"{s}: {x:.3f}" for s, x in zip(seeds, ratios)) or "none paired")
    lacking = {c: sorted(seen - set(mu[c]), key=str) for c in mu}
    lacking = {c: v for c, v in lacking.items() if v}
    if lacking:
        note += "; no valid run for seed(s) " + "; ".join(f"{c}: {v}" for c, v in lacking.items())
    measured = float(np.median(ratios)) if ratios else None
    if any(x <= 1 for x in ratios):                 # one counterexample decides "in every seed"
        return _h("H3", statement, measured, band, "miss", note)
    if lacking:
        return _h("H3", statement, measured, band, "insufficient_data", note)
    return _h("H3", statement, measured, band, "hit", note)


def _h4(tables: Mapping, bands: Mapping) -> dict:
    statement = "A goodput crossover s* exists (TP2 wins below, DP2 above); the measured range overlaps the band"
    band = bands.get("H4")
    rounds = [r for r in tables.get("s_star", []) if r.get("row") == "round"]
    if not rounds:
        present = {r.get("config") for r in tables.get("goodput", []) if r.get("arm") == "base"}
        absent = [c for c in ("TP2", "DP2") if c not in present]
        return _h("H4", statement, None, band, "insufficient_data",
                  "no sweep round has both TP2 and DP2" + (f" (missing: {', '.join(absent)})" if absent else ""))
    stars = [r["s_star_ms"] for r in rounds if r["s_star_ms"] is not None]
    if not stars:
        return _h("H4", statement, None, band, "miss", f"no crossover in any of {len(rounds)} rounds")
    rng = [min(stars), max(stars)]
    note = f"s* range {_fmt(rng[0])}-{_fmt(rng[1])} ms over {len(stars)}/{len(rounds)} rounds"
    if band is None:
        return _h("H4", statement, rng, band, "insufficient_data", note + "; the model predicts no crossover")
    return _h("H4", statement, rng, band, "hit" if rng[1] >= band[0] and rng[0] <= band[1] else "miss", note)


def _h5(tables: Mapping, bands: Mapping) -> dict:
    statement = "TP2's KV capacity per engine is inside the model band relative to TP1's (warm boots)"
    band = bands.get("H5")
    kv = {"TP1": [], "TP2": []}
    cold = {"TP1": 0, "TP2": 0}
    for r in tables.get("kv_capacity", []):
        if r.get("config") in kv and r.get("arm") == "base" and r.get("kv_tokens") is not None \
                and not r.get("violations"):
            if r.get("cold_boot"):
                cold[r["config"]] += 1
            else:
                kv[r["config"]].append(r["kv_tokens"])
    absent = [c for c, v in kv.items() if not v]
    if absent:
        why = [f"{c}{' (only a cold boot)' if cold[c] else ''}" for c in absent]
        return _h("H5", statement, None, band, "insufficient_data", f"no warm-boot KV line for {', '.join(why)}")
    t1, t2 = float(np.median(kv["TP1"])), float(np.median(kv["TP2"]))
    ratio = t2 / t1
    note = f"median warm KV tokens TP2 {t2:.0f} ({len(kv['TP2'])} boots) / TP1 {t1:.0f} ({len(kv['TP1'])} boots)"
    if band is None:
        return _h("H5", statement, ratio, band, "insufficient_data", note + "; no model band")
    return _h("H5", statement, ratio, band, "hit" if _in_band(ratio, band) else "miss", note)


def _h6(tables: Mapping) -> dict:
    statement = (f"Every traced TP2 step has exactly {model.allreduces_per_step(2)} all-reduce ops and "
                 f"{model.allgathers_per_step(2)} all-gather (>= {H6_MIN_EXACT:.0%} of steps, all arms)")
    tp2 = [r for r in tables.get("trace_summary", []) if r.get("tp") == 2]
    rows = [r for r in tp2 if r.get("gate_ok") is True]        # spec 4.5: a trace is valid only if the gate holds
    rejected = sorted({r["run_id"] for r in tp2 if r.get("gate_ok") is not True})
    total = sum(int(r.get("h6_steps", r.get("steps")) or 0) for r in rows)
    if not rows or not total:
        why = f"; {len(rejected)} TP2 traces failed the completeness gate: {', '.join(rejected)}" if rejected else ""
        return _h("H6", statement, None, None, "insufficient_data", "no gate-passing TP2 trace with steps" + why)
    lo = sum(r["h6_exact_min"] for r in rows) / total
    hi = sum(r["h6_exact_max"] for r in rows) / total
    runs = len({r["run_id"] for r in rows})
    note = f"{runs} traces, {total} rank-steps; exact fraction in [{lo:.4f}, {hi:.4f}]"
    if rejected:
        note += f"; {len(rejected)} traces excluded by the completeness gate"
    if lo >= H6_MIN_EXACT:
        return _h("H6", statement, lo, None, "hit", note)
    if hi < H6_MIN_EXACT:
        return _h("H6", statement, hi, None, "miss", note)
    return _h("H6", statement, None, None, "insufficient_data",
              note + "; the summaries give only min/max/mode per rank, trace.sqlite is needed for exact counts")


def _h7(tables: Mapping) -> dict:
    statement = (f"TP2 decode bs-1 GPU idle_est < {H7_BASE_MAX:.2f} (base), > {H7_G2_MIN:.2f} (G2), "
                 "ordered base < G1 < G2")
    idle: dict[str, list[float]] = {}
    tp1_g2 = []
    rejected: dict[str, int] = {}
    for r in tables.get("trace_summary", []):
        if r.get("points") != "decode:b1" or r.get("idle_est") is None:
            continue
        if r.get("gate_ok") is not True:            # spec 4.5: a trace is valid only if the gate holds
            if r.get("tp") == 2:
                rejected[r["arm"]] = rejected.get(r["arm"], 0) + 1
            continue
        if r.get("tp") == 2:
            idle.setdefault(r["arm"], []).append(r["idle_est"])
        elif r.get("config") == "TP1" and r.get("arm") == "G2":
            tp1_g2.append(r["idle_est"])
    v = {a: median_or_none(idle.get(a, [])) for a in ("base", "G1", "G2")}
    absent = [a for a, x in v.items() if x is None]
    why = f" ({len(rejected)} arm(s) with traces rejected by the completeness gate: {', '.join(sorted(rejected))})" \
        if rejected else ""
    if absent:
        return _h("H7", statement, {a: x for a, x in v.items() if x is not None} or None, None, "insufficient_data",
                  f"no gate-passing TP2 decode:b1 trace idle_est for arm(s): {', '.join(absent)}" + why)
    ok = v["base"] < H7_BASE_MAX and v["G2"] > H7_G2_MIN and v["base"] < v["G1"] < v["G2"]
    note = ", ".join(f"{a} {x:.3f}" for a, x in v.items()) + why
    if tp1_g2:
        note += f"; TP1 G2 {median_or_none(tp1_g2):.3f} (predicted < {H7_G2_MIN:.2f})"
    return _h("H7", statement, v, None, "hit" if ok else "miss", note)


def evaluate_hypotheses(tables: Mapping[str, Sequence[dict]], pred: Mapping) -> list[dict]:
    """H1..H8 against the committed bands of model.predictions() (spec 4.8 as amended by AM22-AM24)."""
    bands = pred.get("bands", {})
    return [
        _ratio_hyp(tables, bands.get("H1"), "H1", "TP2 decode speedup over TP1 at batch 1 lies inside the model band",
                   ("TP1", "base"), ("TP2", "base")),
        _h2(tables, bands),
        _h3(tables, bands),
        _h4(tables, bands),
        _h5(tables, bands),
        _h6(tables),
        _h7(tables),
        _ratio_hyp(tables, bands.get("H8"), "H8", "AR3 (pure NCCL) / AR0 decode step ratio at batch 1 lies inside "
                   "the model band", ("TP2", "AR3"), ("TP2", "base")),
    ]


# ------------------------------------------------------------------------------------------ output

def _cell(v: object) -> object:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return v


def write_csv(path: str, rows: Sequence[Mapping]) -> None:
    """Columns in first-seen order over all rows; None and NaN are empty cells, lists and dicts JSON."""
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in cols])


def _jsonable(v: object) -> object:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def analyze(results_dir: str) -> dict[str, list[dict]]:
    raw = os.path.join(results_dir, "raw")
    gaps = _Gaps()
    pred = model.predictions()
    runs = load_runs(raw, gaps)
    skipped = _skipped_from_last_run(raw)
    grid_mu = {}
    if os.path.exists(os.path.join(raw, "rate_grid.json")):
        try:
            grid_mu = dict(_read_json(os.path.join(raw, "rate_grid.json")).get("mu_rps") or {})
        except _PARSE_ERRORS:
            grid_mu = {}

    offline_rows, lat = _offline(runs, pred, gaps)
    subs = _online(runs, gaps)
    online_rows = [s.row for s in subs]
    points = _rate_points(subs)
    gp = _goodput(points)
    kv, effs = _kv(runs, gaps)
    trace_rows, trace_steps = _traces(runs, gaps)
    tables: dict[str, list[dict]] = {
        "offline_points": offline_rows,
        "online_runs": online_rows,
        "saturation": _saturation(subs, grid_mu),
        "goodput": gp,
        "s_star": _s_star(points, gp),
        "comm": _comm(runs, gaps),
        "kv_capacity": kv,
        "trace_summary": trace_rows,
        "trace_steps": trace_steps,
        "efficiency_delta": efficiency_delta(lat),
        "confounders": _confounders(results_dir, effs, online_rows),
    }
    _status_gaps(runs, skipped, gaps)
    _expected_missing(runs, raw, effs, skipped, gaps)
    tables["gaps"] = gaps.rows

    tidy = os.path.join(results_dir, "tidy")
    os.makedirs(tidy, exist_ok=True)
    for name in ALL_TABLES:
        write_csv(os.path.join(tidy, f"{name}.csv"), tables[name])
    with open(os.path.join(tidy, "hypotheses.json"), "w") as f:
        json.dump(_jsonable(evaluate_hypotheses(tables, pred)), f, indent=1)
    return tables


def is_fake(tables: Mapping[str, Sequence[dict]]) -> bool:
    """A dry run: the offline driver recorded the fake engine (C3 tpprof.engine == "fake")."""
    return any(r.get("engine") == "fake" for r in tables.get("offline_points", []))

