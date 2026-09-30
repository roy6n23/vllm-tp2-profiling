"""nsys trace runs: the exact nsys argv, export, and the retry policy of spec 4.5.

Attempt 1 captures between cudaProfilerStart/Stop. If it hangs past the timeout, it is
retried once with VLLM_ALLREDUCE_USE_SYMM_MEM=0 (vLLM #48486). If the exported trace
fails the completeness gate, it is rerun once with --capture-range=none (AM17: that
replaces both capture flags), and the analysis then uses the tpprof:measure window.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from tpprof import constants, procs, traces

CAPTURE_API = "cudaProfilerApi"
CAPTURE_NONE = "none"
_CAPTURE_FLAGS = {
    CAPTURE_API: ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"],
    CAPTURE_NONE: ["--capture-range=none"],
}
NSYS_ENV = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "60"}
SYMM_MEM_OFF = {"VLLM_ALLREDUCE_USE_SYMM_MEM": "0"}      # vLLM #48486: nsys + symm-mem hang
OUTPUT_NAME = "trace"


def nsys_profile_argv(output_base: str, target: Sequence[str], capture: str = CAPTURE_API,
                      nsys: str | None = None) -> list[str]:
    """Spec 4.5 `nsys profile` argv, token for token; capture="none" is the AM17 fallback."""
    if capture not in _CAPTURE_FLAGS:
        raise ValueError(f"capture must be one of {sorted(_CAPTURE_FLAGS)}, got {capture!r}")
    return [nsys or constants.nsys_path(), "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
            "--trace-fork-before-exec=true", "--sample=none", "--cpuctxsw=none", *_CAPTURE_FLAGS[capture],
            "--force-overwrite=true", f"--output={output_base}", *target]


def nsys_export_argv(rep: str, sqlite_path: str, nsys: str | None = None) -> list[str]:
    return [nsys or constants.nsys_path(), "export", "--type=sqlite", "--force-overwrite=true",
            f"--output={sqlite_path}", rep]


@dataclass
class TraceOutcome:
    ok: bool
    attempts: list[dict]
    sqlite_path: str | None
    summary: dict | None
    reason: str


def _command(argv: list[str], env: Mapping[str, str], overrides: Mapping[str, str], log: str, run_id: str,
             timeout_s: float) -> dict:
    """Run one command through procs (own process group, log file, timeout) and record it like cmd.json."""
    t_wall, t_mono = time.time(), time.monotonic()
    out = procs.run(argv, {**env, **overrides}, log, run_id, timeout_s)
    return {"argv": argv, "env_overrides": dict(overrides), "cwd": None,
            "t_wall_start": t_wall, "t_mono_start": t_mono, "t_wall_end": out.t_wall_end, "t_mono_end": out.t_mono_end,
            "duration_s": out.t_mono_end - t_mono, "exit_code": out.exit_code, "timed_out": out.timed_out, "log": log}


def _has_measure_range(sqlite_path: str) -> bool:
    return any(r.text == traces.MEASURE_RANGE for r in traces.load_trace(sqlite_path).ranges)


def run_trace(run_dir: str, run_id: str, target: Sequence[str], env: Mapping[str, str], tp: int, min_steps: int,
              timeout_s: float, untraced_step_s: float | None = None) -> TraceOutcome:
    """Profile target under nsys, export to <run_dir>/trace.sqlite, and summarize it (spec 4.5).

    env is the target's full environment; NSYS_ENV and NSYS_TMPDIR=<run_dir>/nsys_tmp are
    added. The nsys binary is env's TPPROF_NSYS if set, else constants.nsys_path() (AM18).
    Every attempt is returned; the caller writes the summary.
    """
    target = [str(a) for a in target]
    nsys = env.get("TPPROF_NSYS") or constants.nsys_path()
    tmpdir = os.path.join(run_dir, "nsys_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    base = os.path.join(run_dir, OUTPUT_NAME)
    rep, sqlite_path = f"{base}.nsys-rep", f"{base}.sqlite"
    overrides = {**NSYS_ENV, "NSYS_TMPDIR": tmpdir}

    attempts: list[dict] = []
    capture, symm_off = CAPTURE_API, False
    notes: list[str] = []
    summary: dict | None = None
    exported: str | None = None
    label = ""

    def fail(why: str) -> TraceOutcome:
        return TraceOutcome(False, attempts, exported, summary, "; ".join([*notes, f"{label}: {why}"]))

    while True:
        n = len(attempts) + 1
        log = os.path.join(run_dir, f"nsys-attempt{n}.log")
        env_over = {**overrides, **(SYMM_MEM_OFF if symm_off else {})}
        a = _command(nsys_profile_argv(base, target, capture, nsys), env, env_over, log, run_id, timeout_s)
        a.update({"attempt": n, "capture": capture, "export": None, "gate_ok": None, "gate_reasons": []})
        attempts.append(a)
        label = f"attempt {n} (capture-range={capture}{', VLLM_ALLREDUCE_USE_SYMM_MEM=0' if symm_off else ''})"

        if a["timed_out"]:
            if symm_off:
                return fail(f"nsys profile timed out after {timeout_s:g} s")
            notes.append(f"{label} timed out after {timeout_s:g} s; retrying with VLLM_ALLREDUCE_USE_SYMM_MEM=0")
            symm_off = True
            continue
        if a["exit_code"] != 0:
            return fail(f"nsys profile exited {a['exit_code']}; see {log}")

        exported = summary = None            # a failed export must not leave the previous attempt's trace
        if os.path.exists(sqlite_path):
            os.remove(sqlite_path)
        ex = _command(nsys_export_argv(rep, sqlite_path, nsys), env, {}, log, run_id, timeout_s)
        a["export"] = ex
        if ex["timed_out"] or ex["exit_code"] != 0 or not os.path.exists(sqlite_path):
            status = f"timed out after {timeout_s:g} s" if ex["timed_out"] else f"exited {ex['exit_code']}"
            return fail(f"nsys export {status}; see {log}")
        exported = sqlite_path
        try:
            summary = traces.summarize_trace(sqlite_path, tp, min_steps, untraced_step_s)
            reasons = list(summary["gate"]["reasons"])
            if capture == CAPTURE_NONE and not _has_measure_range(sqlite_path):
                reasons.append(f"capture-range=none trace has no {traces.MEASURE_RANGE} range to bound the window")
        except sqlite3.Error as e:
            return fail(f"trace analysis failed: {e}")
        a["gate_ok"], a["gate_reasons"] = not reasons, reasons
        summary["gate"] = {"ok": not reasons, "reasons": reasons}
        if not reasons:
            notes.append(f"{label}: completeness gate ok")
            return TraceOutcome(True, attempts, exported, summary, "; ".join(notes))
        if capture == CAPTURE_NONE:
            return fail("completeness gate failed: " + "; ".join(reasons))
        notes.append(f"{label}: completeness gate failed ({'; '.join(reasons)}); rerunning with capture-range=none")
        capture = CAPTURE_NONE
