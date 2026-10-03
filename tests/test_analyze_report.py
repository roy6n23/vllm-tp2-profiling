from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import shutil
import statistics
import sys

import pytest

from tests import make_records
from tpprof import analyze, gap, kernels, model, plots, report

CSV_TABLES = ("offline_points", "online_runs", "saturation", "goodput", "s_star", "comm", "kv_capacity",
              "trace_summary", "posteriori", "tp2_gap")
HYP_IDS = tuple(f"H{i}" for i in range(1, 9))
WATERMARK = "> FAKE DATA — dry run with fake tools; numbers are meaningless"


@pytest.fixture(scope="module")
def records(tmp_path_factory) -> str:
    d = str(tmp_path_factory.mktemp("results"))
    make_records.build(d)
    return d


@pytest.fixture(scope="module")
def tables(records) -> dict:
    return analyze.analyze(records)


@pytest.fixture(scope="module")
def hyps(tables) -> list[dict]:
    return analyze.evaluate_hypotheses(tables, model.predictions())


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _by_id(hyps: list[dict]) -> dict[str, dict]:
    return {h["id"]: h for h in hyps}


def _drop_config(results_dir: str, config: str) -> None:
    raw = os.path.join(results_dir, "raw")
    for name in os.listdir(raw):
        parts = name.split("-")
        if len(parts) > 2 and parts[2] == config:
            shutil.rmtree(os.path.join(raw, name))


# ------------------------------------------------------------------------------------------ analyze

def test_analyze_writes_all_csvs_and_hypotheses(records, tables):
    tidy = os.path.join(records, "tidy")
    for name in CSV_TABLES:
        path = os.path.join(tidy, f"{name}.csv")
        assert os.path.exists(path), name
        assert tables[name], f"{name} is empty"
        assert len(_read_csv(path)) == len(tables[name])
    with open(os.path.join(tidy, "hypotheses.json")) as f:
        doc = json.load(f)
    assert [h["id"] for h in doc] == list(HYP_IDS)


def test_offline_points_have_dp2_derived_rows_and_speedups(tables):
    rows = tables["offline_points"]
    tp1 = {r["batch"]: r for r in rows if (r["config"], r["arm"], r["kind"]) == ("TP1", "base", "decode")
           and not r["derived"]}
    dp2 = {r["batch"]: r for r in rows if r["config"] == "DP2" and r["kind"] == "decode"}
    assert dp2 and all(r["derived"] for r in dp2.values())
    assert sorted(dp2) == [2 * b for b in sorted(tp1)]
    for b, r in dp2.items():
        assert r["step_s"] == tp1[b // 2]["step_s"]                       # DP2 at B = TP1 at B/2
        if b in tp1:
            assert r["speedup"] == pytest.approx(tp1[b]["step_s"] / tp1[b // 2]["step_s"])
            assert r["efficiency"] == pytest.approx(r["speedup"] / 2)
        else:
            assert r["speedup"] is None
    tp2_b1 = next(r for r in rows if (r["config"], r["arm"], r["kind"], r["batch"]) == ("TP2", "base", "decode", 1))
    assert tp2_b1["speedup"] == pytest.approx(tp1[1]["step_s"] / tp2_b1["step_s"])
    assert tp2_b1["model_ms"] == model.predictions()["decode"]["central"]["2"]["1"]
    # AM14: the smoke's GPU1 point is 0.5% slower than GPU0, below the 1% flag
    assert dp2[2]["gpu_asym"] == pytest.approx(0.005, abs=1e-4)
    assert not dp2[2]["flag"]
    xcheck = [r for r in rows if r["kind"] == "xcheck"]
    assert {r["config"] for r in xcheck} == {"TP1", "TP2"}
    assert all(r["xcheck_diff"] == pytest.approx(0.01, abs=2e-3) for r in xcheck)


def test_online_saturation_goodput_and_s_star(tables):
    online = tables["online_runs"]
    assert len(online) == 3 * 3 + 9 * 6 + 2 * 4 + 6 + 2 * 3        # sat, sweeps, pc, DP2rand, API2
    assert all(r["valid"] for r in online)
    dp2_sat = [r for r in online if (r["config"], r["phase"], r["arm"]) == ("DP2", "sat", "base")]
    assert all(r["cpu_api_p90"] > analyze.CPU_FLAG_PCT and "cpu>80%" in r["flags"] for r in dp2_sat)
    tp2_sat = [r for r in online if (r["config"], r["phase"], r["arm"]) == ("TP2", "sat", "base")]
    assert all(r["cpu_api_p90"] < analyze.CPU_FLAG_PCT and r["cpu_client_p90"] < 30 for r in tp2_sat)
    sat = [r for r in tables["saturation"] if r["row"] == "seed" and r["arm"] == "base" and r["phase"] == "sat"]
    assert {(r["config"], r["seed"]) for r in sat} == {(c, s) for c in ("TP1", "TP2", "DP2") for s in (1, 2, 3)}
    for r in sat:
        mu = make_records.MU_RPS[r["config"]] * (1 + 0.01 * (r["seed"] - 2))
        assert r["mu_rps"] == pytest.approx(mu, rel=0.05)
        assert r["preempt_per_1k"] == (120.0 if r["config"] in ("TP1", "DP2") else 0.0)
    gp = tables["goodput"]
    assert {r["round"] for r in gp} == {1, 2, 3}
    assert {r["ttft_slo_s"] for r in gp} == {1.0, 0.5, 2.0}
    assert "DP2rand" in {r["config"] for r in gp}
    star = tables["s_star"]
    rounds = [r for r in star if r["row"] == "round"]
    assert [r["round"] for r in rounds] == [1, 2, 3]
    assert all(10 < r["s_star_ms"] < 20 for r in rounds)
    summary = next(r for r in star if r["row"] == "all")
    assert summary["s_star_min"] <= summary["s_star_ms"] <= summary["s_star_max"]
    assert summary["ci_lo_ms"] is not None and summary["ci_lo_ms"] <= summary["ci_hi_ms"]
    # the request-level CI belongs to the pooled s*, not to the median of the rounds (2026-10-02 report)
    assert summary["ci_lo_ms"] <= summary["s_star_pooled_ms"] <= summary["ci_hi_ms"]


def _session(raw: str, prefix: str) -> str:
    return os.path.join(raw, next(n for n in sorted(os.listdir(raw)) if n.startswith(prefix)))


def test_cpu_p90_is_per_client_run(records, tables):
    """AM11: each client run gets the p90 of its own benchmark phase, not of the whole session."""
    session = _session(os.path.join(records, "raw"), "P1-serve_session-TP2-base-r1")
    rows = sorted((r for r in tables["online_runs"] if r["run_id"] == os.path.basename(session)),
                  key=lambda r: r["sub"])
    assert len(rows) == 6 and all(r["cpu_scope"] == "bench" and r["cpu_samples"] for r in rows)
    levels = [make_records.api_cpu_level(make_records.matrix.RunSpec("serve_session", "TP2", "base", "P1",
                                                                     (("phase", "sweep"),)), k, 6) for k in range(6)]
    for r, level in zip(rows, levels):
        assert r["cpu_api_p90"] == pytest.approx(level, abs=1.0)        # never the startup burst
        assert 20 <= r["cpu_client_p90"] < 25
    assert ["cpu>80%" in r["flags"] for r in rows] == [False] * 5 + [True]
    whole = analyze.cpu_p90(os.path.join(session, "cpu.csv"))            # what a session-wide p90 would say
    assert whole["api"] > analyze.CPU_FLAG_PCT


def test_cpu_without_a_client_window_is_marked_session_wide(tmp_path):
    d = _small(tmp_path, tiers=("P1",))
    session = _session(os.path.join(d, "raw"), "P1-serve_session-TP2-base-r1")
    with open(os.path.join(session, "cmd.json")) as f:
        cmds = json.load(f)
    with open(os.path.join(session, "cmd.json"), "w") as f:
        json.dump([c for c in cmds if "bench" not in c["argv"]], f)           # the server command only
    rows = [r for r in analyze.analyze(d)["online_runs"] if r["run_id"] == os.path.basename(session)]
    assert rows and all(r["cpu_scope"] == "session" for r in rows)
    assert all("cpu>80% (session-wide p90)" in r["flags"] and "whole session" in r["monitor_note"] for r in rows)


def test_broken_monitor_files_keep_the_client_results(tmp_path):
    """Review Focus 2: a bad cpu.csv or gpu.csv costs the CPU/throttle columns, never the client results."""
    d = _small(tmp_path, tiers=("P1",))
    raw = os.path.join(d, "raw")
    before = analyze.analyze(d)
    tp2 = _session(raw, "P1-serve_session-TP2-base-r1")
    with open(os.path.join(tp2, "cpu.csv"), "a") as f:
        f.write(f"{make_records.T0 + 5:.3f},100,1,python3,/usr/bin/python3 /usr/local/bin/vllm serve /models,")
    dp2 = _session(raw, "P1-serve_session-DP2-base-r1")
    with open(os.path.join(dp2, "cpu.csv"), "w") as f:
        f.write("t_wall,pid,cmd,cpu_percent\n1.0,7,\"" + "x" * 200_000 + "\n")  # csv.Error: field limit
    with open(os.path.join(dp2, "gpu.csv"), "wb") as f:
        f.write(b"\xff\xfe garbage\n")
    after = analyze.analyze(d)
    assert len(after["online_runs"]) == len(before["online_runs"])
    for config in ("TP2", "DP2"):
        assert any(r["config"] == config and r["round"] == 1 for r in after["goodput"])
    tp2_rows = [r for r in after["online_runs"] if r["run_id"] == os.path.basename(tp2)]
    assert all(r["cpu_api_p90"] is not None for r in tp2_rows)
    assert any("cpu.csv: skipped 1 unparseable rows" in (r["monitor_note"] or "") for r in tp2_rows)
    dp2_rows = [r for r in after["online_runs"] if r["run_id"] == os.path.basename(dp2)]
    assert dp2_rows and all(r["cpu_api_p90"] is None and "cpu.csv" in r["monitor_note"] for r in dp2_rows)
    partial = [g for g in after["gaps"] if g["type"] == "partial" and g["run_id"] == os.path.basename(dp2)]
    assert len(partial) == 1 and "cpu.csv" in partial[0]["detail"]            # listed once per session file
    h = _by_id(analyze.evaluate_hypotheses(after, model.predictions()))
    assert h["H4"]["verdict"] != "insufficient_data"


def test_comm_rows_and_alpha_beta_fits(tables):
    comm = tables["comm"]
    assert {r["source"] for r in comm} == {"comm_m1", "comm_m2", "comm_m3", "comm_m4"}
    fit = next(r for r in comm if r["row"] == "fit" and r["source"] == "comm_m3" and r["variant"] == "none:none"
               and r["mode"] == "graph")
    assert fit["alpha_us"] == pytest.approx(6.0, rel=0.1)
    assert fit["beta_GBps"] == pytest.approx(260.0, rel=0.02)
    ring = next(r for r in comm if r["row"] == "point" and r["variant"] == "ring:LL")
    assert (ring["algo"], ring["proto"]) == ("ring", "LL")
    m2 = [r for r in comm if r["row"] == "point" and r["source"] == "comm_m2"]
    assert m2 and all(r["lat_us"] > 0 for r in m2)


def test_kv_capacity_flags_the_first_boot_of_each_config(tables):
    kv = tables["kv_capacity"]
    by_config: dict[str, list[dict]] = {}
    for r in kv:
        if r["engine"] == 0:
            by_config.setdefault(r["config"], []).append(r)
    for config, rows in by_config.items():
        rows.sort(key=lambda r: r["boot_index"])
        assert rows[0]["cold_boot"], config
        assert not any(r["cold_boot"] for r in rows[1:]), config
    assert len([r for r in kv if r["config"] == "DP2" and r["boot_index"] == 0]) == 2   # one row per engine


def test_trace_summary_is_flat_per_rank(tables):
    rows = tables["trace_summary"]
    tp2_b1 = [r for r in rows if (r["config"], r["arm"], r["points"]) == ("TP2", "base", "decode:b1")]
    assert [r["rank"] for r in tp2_b1] == [0, 1]
    assert all(r["ar_mode"] == 65 and r["ag_mode"] == 1 for r in tp2_b1)
    # this trace has a trace.sqlite, so H6 counts its steps exactly
    assert all(r["h6_source"] == "trace.sqlite" and r["h6_exact_min"] == r["h6_steps"] == make_records.SQLITE_STEPS
               for r in tp2_b1)
    assert tp2_b1[0]["unclassified_frac"] == pytest.approx(0.01)
    assert sum(tp2_b1[0][f"cat_{c}_ms"] for c in kernels.CATEGORIES) + tp2_b1[0]["idle_ms"] == \
        pytest.approx(tp2_b1[0]["mean_step_ms"])


def test_trace_steps_and_am16_comm_time(records, tables):
    steps = tables["trace_steps"]
    by_trace: dict[tuple, list[dict]] = {}
    for r in steps:
        by_trace.setdefault((r["config"], r["arm"], r["points"], r["rank"]), []).append(r)
    assert set(by_trace) == {("TP1", "base", "decode:b1", 0), ("TP2", "base", "decode:b1", 0),
                             ("TP2", "base", "decode:b1", 1)}
    assert all(len(v) == make_records.SQLITE_STEPS for v in by_trace.values())
    tp2 = by_trace[("TP2", "base", "decode:b1", 0)]
    assert all(s["ar_ops"] == 65 and s["ag_ops"] == 1 and s["exact_counts"] and s["pure_decode"] for s in tp2)
    assert all(sum(s[f"cat_{c}_ms"] for c in kernels.CATEGORIES) == pytest.approx(s["gpu_busy_ms"]) for s in tp2)
    assert _read_csv(os.path.join(records, "tidy", "trace_steps.csv"))
    # AM16: TP1 runs 65 standalone fused_add_rms_norm kernels of 2.5 us per step (tests/synth_trace.py)
    tp1_norm = 65 * 2_500 / 1e6
    assert by_trace[("TP1", "base", "decode:b1", 0)][0]["fused_add_rms_norm_ms"] == pytest.approx(tp1_norm)
    rows = {(r["config"], r["arm"], r["points"], r["rank"]): r for r in tables["trace_summary"]}
    base = rows[("TP2", "base", "decode:b1", 0)]
    assert base["ar_fused"] and base["ar_ms_source"] == "trace.sqlite"
    assert base["tp1_norm_ms"] == pytest.approx(tp1_norm)
    assert base["comm_ms"] == pytest.approx(base["ar_ms"] - tp1_norm) and 0 < base["comm_ms"] < base["ar_ms"]
    ar3 = rows[("TP2", "AR3", "decode:b1", 0)]                               # unfused: AR time is comm time
    assert ar3["ar_fused"] is False and ar3["comm_ms"] == pytest.approx(ar3["ar_ms"])
    b32 = rows[("TP2", "base", "decode:b32", 0)]                             # no TP1 decode:b32 trace.sqlite
    assert b32["comm_ms"] is None and "TP1 decode:b32" in b32["comm_note"]
    assert rows[("TP1", "base", "decode:b1", 0)]["comm_ms"] is None


def test_posteriori_table_refits_tp1_and_predicts_tp2(records, tables):
    rows = tables["posteriori"]
    c = {r["name"]: r for r in rows if r["row"] == "constant"}
    # make_records draws every step time from the central model, so the TP1 fit returns its constants.
    assert c["bw_eff"]["value"] == pytest.approx(make_records.C.bw_eff, rel=1e-6)
    assert c["t_fixed"]["value"] == pytest.approx(make_records.C.t_fixed, rel=1e-6)
    assert c["alpha"]["value"] == pytest.approx(5e-6, rel=0.05)              # M2 of the engines' backend (trtllm)
    assert "flashinfer_trtllm_fused_allreduce_rmsnorm_oneshot" in c["alpha"]["source"]
    assert c["alpha_nccl_graph"]["value"] == pytest.approx(6e-6, rel=0.1)   # M3, graph, default variant
    points = {(r["config"], r["arm"], r["batch"]): r for r in rows if r["row"] == "point"}
    assert {k[:2] for k in points} == {("TP1", "base"), ("TP2", "base"), ("TP2", "AR3")}
    assert points[("TP1", "base", 1)]["residual_ms"] == pytest.approx(0.0, abs=1e-6)
    # The central model gives TP2 an extra 0.1 ms per step; the stage sets that term to 0, so it is the residual.
    assert points[("TP2", "base", 1)]["residual_ms"] == pytest.approx(make_records.C.t_extra_tp2 * 1e3, abs=0.01)
    assert len(_read_csv(os.path.join(records, "tidy", "posteriori.csv"))) == len(rows)


def test_tp2_gap_table_splits_the_traced_batch(records, tables):
    rows = tables["tp2_gap"]
    assert {r["batch"] for r in rows} == {1}                                  # SQLITE_TRACES: decode:b1 only
    assert [r["component"] for r in rows] == ["step", *gap.COMPONENTS]
    step, parts = rows[0], rows[1:]
    assert step["tp1_ms"] == pytest.approx(make_records._decode_step("TP1", "base", 1) * 1e3)
    assert step["tp2_ms"] == pytest.approx(make_records._decode_step("TP2", "base", 1) * 1e3)
    assert step["excess_ms"] == pytest.approx(step["tp2_ms"] - step["tp1_ms"] / 2)
    assert sum(r["excess_ms"] for r in parts) == pytest.approx(step["excess_ms"])
    by = {r["component"]: r for r in parts}
    # The same AM16 subtraction as trace_summary's comm_ms (mean of the two ranks), plus the 9 us all-gather.
    am16 = [r["comm_ms"] for r in tables["trace_summary"]
            if (r["config"], r["arm"], r["points"]) == ("TP2", "base", "decode:b1")]
    assert by["comm"]["tp1_ms"] == 0.0
    assert by["comm"]["tp2_ms"] == pytest.approx(sum(am16) / 2 + 9_000 / 1e6)
    tp1_norm = 65 * 2_500 / 1e6                                               # tests/synth_trace.py
    assert by["norm_act_rope"]["tp2_ms"] - tp1_norm == pytest.approx(by["norm_act_rope"]["tp1_ms"] - tp1_norm)
    assert by["gemm"]["tp2_ms"] == pytest.approx(by["gemm"]["tp1_ms"] / 2, rel=0.01)   # synthetic GEMMs halve
def test_comm_time_of_an_eager_trace_is_its_all_reduce_time(tmp_path):
    """G2 (--enforce-eager) runs no fusion pass: its TP2 trace has the residual add + RMSNorm as standalone
    kernels, so TP1's norm time is not subtracted from its all-reduce time (the 2026-10-02 G2 trace)."""
    d = str(tmp_path / "results")
    make_records.build(d, tiers=("P0",), rounds=1,
                       sqlite_traces=(*make_records.SQLITE_TRACES, ("TP2", "G2", "decode:b1")))
    rows = {(r["config"], r["arm"], r["points"], r["rank"]): r for r in analyze.analyze(d)["trace_summary"]}
    g2 = rows[("TP2", "G2", "decode:b1", 0)]
    assert g2["fused_add_rms_norm_ms"] == pytest.approx(65 * 2_500 / 1e6)     # standalone, as in TP1
    assert g2["ar_fused"] is False and g2["tp1_norm_ms"] is None
    assert g2["comm_ms"] == pytest.approx(g2["ar_ms"])
    base = rows[("TP2", "base", "decode:b1", 0)]                              # no standalone norm kernels
    assert base["fused_add_rms_norm_ms"] == 0 and base["ar_fused"]
    assert base["comm_ms"] == pytest.approx(base["ar_ms"] - base["tp1_norm_ms"])


def _comm_row(config: str, arm: str, norm_ms: float | None, ar_ms: float = 0.4) -> dict:
    return {"config": config, "arm": arm, "tp": int(config[2]), "points": "decode:b1", "rank": 0, "gate_ok": True,
            "ar_ms": ar_ms, "ar_ms_source": "trace.sqlite" if norm_ms is not None else "summary shares",
            "fused_add_rms_norm_ms": norm_ms}


def test_comm_time_reads_fusion_from_the_trace_and_falls_back_to_the_arm():
    rows = [_comm_row("TP1", "base", 0.15, ar_ms=0.0),
            _comm_row("TP2", "base", 0.0),         # no standalone norm kernels: the all-reduce kernel does the norm
            _comm_row("TP2", "base", 0.13),        # standalone norm kernels (the fusion did not happen)
            _comm_row("TP2", "base", None),        # no trace.sqlite: base is fused by design
            _comm_row("TP2", "G2", None),          # no trace.sqlite: --enforce-eager runs no fusion pass
            _comm_row("TP2", "AR3", None)]
    analyze._comm_time(rows)
    assert [r["ar_fused"] for r in rows] == [None, True, False, True, False, False]
    assert [r["comm_ms"] for r in rows[1:]] == pytest.approx([0.25, 0.4, 0.25, 0.4, 0.4])
    assert [r["tp1_norm_ms"] for r in rows[1:]] == [0.15, None, 0.15, None, None]
    assert "standalone" in rows[2]["comm_note"]


def test_trace_summary_has_busy_time_and_idle_per_rank(tables):
    """idle_est (AM16) is one number per trace: 1 - the ranks' mean busy time per step / the untraced step. The
    per-rank values behind it show a rank whose GPU is busy spin-waiting for the other one (2026-10-03, H7)."""
    def trace(config: str, arm: str) -> list[dict]:
        return [r for r in tables["trace_summary"] if (r["config"], r["arm"], r["points"]) == (config, arm, "decode:b1")]

    steps: dict[int, list[float]] = {}
    for s in tables["trace_steps"]:
        if (s["config"], s["arm"], s["points"]) == ("TP2", "base", "decode:b1"):
            steps.setdefault(s["rank"], []).append(s["gpu_busy_ms"])
    rows = trace("TP2", "base")
    busy = [statistics.fmean(steps[r["rank"]]) for r in rows]
    assert [r["busy_ms"] for r in rows] == pytest.approx(busy) and busy[0] != busy[1]
    untraced_ms = statistics.fmean(busy) / (1 - rows[0]["idle_est"])          # the step idle_est was computed with
    assert [r["idle_rank"] for r in rows] == pytest.approx([1 - b / untraced_ms for b in busy])
    assert statistics.fmean(r["idle_rank"] for r in rows) == pytest.approx(rows[0]["idle_est"])
    [tp1] = trace("TP1", "base")
    assert tp1["busy_ms"] > 0 and tp1["idle_rank"] == pytest.approx(tp1["idle_est"])       # one rank
    assert all(r["busy_ms"] is None and r["idle_rank"] is None for r in trace("TP2", "AR3"))   # no trace.sqlite


def test_gate_failed_traces_decide_no_hypothesis(tmp_path):
    """Spec 4.5: a trace is valid only if the completeness gate holds."""
    pred = model.predictions()
    bad = lambda **kw: _trace_row(0, 0, 0, gate_ok=False, **kw)              # would be an H6 miss if counted
    h = _by_id(analyze.evaluate_hypotheses(_tables_with(trace_summary=[bad()]), pred))
    assert h["H6"]["verdict"] == "insufficient_data" and "completeness gate" in h["H6"]["note"]
    h = _by_id(analyze.evaluate_hypotheses(_tables_with(trace_summary=[bad(), _trace_row(65, 65, 65)]), pred))
    assert h["H6"]["verdict"] == "hit" and "excluded" in h["H6"]["note"]
    rows = [_trace_row(65, 65, 65, arm=a, idle_est=v) for a, v in (("base", 0.05), ("G1", 0.15))]
    rows.append(_trace_row(65, 65, 65, arm="G2", idle_est=0.01, gate_ok=False))  # would make H7 a miss
    h7 = _by_id(analyze.evaluate_hypotheses(_tables_with(trace_summary=rows), pred))["H7"]
    assert h7["verdict"] == "insufficient_data" and "G2" in h7["note"] and "completeness gate" in h7["note"]

    d = _small(tmp_path, tiers=("P0",))
    raw = os.path.join(d, "raw")
    for name in os.listdir(raw):
        if name.startswith("P0-trace-TP2"):
            path = os.path.join(raw, name, "trace_summary.json")
            with open(path) as f:
                doc = json.load(f)
            doc["gate"] = {"ok": False, "reasons": ["rank 1 has 253 steps, rank 0 has 256"]}
            with open(path, "w") as f:
                json.dump(doc, f)
    tables = analyze.analyze(d)
    tp2 = [r for r in tables["trace_summary"] if r["tp"] == 2]
    assert tp2 and not any(r["gate_ok"] for r in tp2) and all(r["comm_ms"] is None for r in tp2)
    assert any(g["type"] == "trace_gate" and "253 steps" in g["detail"] for g in tables["gaps"])
    h = _by_id(analyze.evaluate_hypotheses(tables, pred))
    assert h["H6"]["verdict"] == h["H7"]["verdict"] == "insufficient_data"


# ------------------------------------------------------------------------------------------ hypotheses

def test_evaluate_hypotheses_returns_eight_rows_with_allowed_verdicts(hyps):
    assert [h["id"] for h in hyps] == list(HYP_IDS)
    for h in hyps:
        assert set(h) >= {"id", "statement", "measured", "band", "verdict", "note"}
        assert h["verdict"] in analyze.VERDICTS == ("hit", "miss", "insufficient_data")


def test_complete_synthetic_set_hits_the_model_driven_hypotheses(hyps):
    v = {h["id"]: h["verdict"] for h in hyps}
    for h in ("H1", "H2", "H3", "H5", "H6", "H7", "H8"):
        assert v[h] == "hit", (h, _by_id(hyps)[h])
    assert v["H4"] in ("hit", "miss")
    h1 = _by_id(hyps)["H1"]
    assert h1["band"] == model.predictions()["bands"]["H1"]
    assert h1["measured"] == pytest.approx(model.predictions()["ratios"]["decode_speedup"]["central"]["1"], rel=1e-3)
    h5 = _by_id(hyps)["H5"]                     # warm boots only: the 5%-low cold boots are excluded
    assert h5["measured"] == pytest.approx(make_records.KV_TOKENS["TP2"] / make_records.KV_TOKENS["TP1"])


def _tables_with(**overrides) -> dict:
    base = {name: [] for name in analyze.ALL_TABLES}
    base.update(overrides)
    return base


def _decode_row(config: str, arm: str, batch: int, step_s: float) -> dict:
    return {"config": config, "arm": arm, "kind": "decode", "batch": batch, "input_len": 1024, "step_s": step_s,
            "median_s": None, "derived": False}


def test_h1_h8_miss_outside_band_and_insufficient_when_missing():
    pred = model.predictions()
    t = _tables_with(offline_points=[_decode_row("TP1", "base", 1, 6e-3), _decode_row("TP2", "base", 1, 3e-3),
                                     _decode_row("TP2", "AR3", 1, 4.5e-3)])
    h = _by_id(analyze.evaluate_hypotheses(t, pred))
    assert h["H1"]["verdict"] == "miss" and h["H1"]["measured"] == pytest.approx(2.0)
    assert h["H8"]["verdict"] == "miss" and h["H8"]["measured"] == pytest.approx(1.5)
    empty = _by_id(analyze.evaluate_hypotheses(_tables_with(), pred))
    assert all(x["verdict"] == "insufficient_data" and x["note"] for x in empty.values())
    assert "TP2" in empty["H1"]["note"] and "AR3" in empty["H8"]["note"]


def _sat_row(config: str, seed: int, mu: float, valid: bool = True) -> dict:
    return {"row": "seed", "config": config, "arm": "base", "phase": "sat", "seed": seed, "valid": valid,
            "mu_tps": mu}


def test_h3_needs_every_seed_of_both_configs():
    pred = model.predictions()
    h3 = lambda rows: _by_id(analyze.evaluate_hypotheses(_tables_with(saturation=rows), pred))["H3"]
    full = [_sat_row("TP2", s, 100.0) for s in (1, 2, 3)] + [_sat_row("DP2", s, 120.0) for s in (1, 2, 3)]
    assert h3(full)["verdict"] == "hit"
    one_tp2 = [_sat_row("TP2", 1, 100.0), _sat_row("TP2", 2, 90.0, valid=False)] + full[3:]
    got = h3(one_tp2)                                   # seeds 2 and 3 of TP2 are unknown
    assert got["verdict"] == "insufficient_data" and "TP2: [2, 3]" in got["note"]
    assert h3([_sat_row("TP2", 1, 130.0)] + full[3:])["verdict"] == "miss"   # one counterexample decides


def _trace_row(ar_min: int, ar_max: int, ar_mode: int, steps: int = 256, **kw) -> dict:
    row = {"run_id": "r", "config": "TP2", "arm": "base", "tp": 2, "points": "decode:b1", "rank": 0,
           "steps": steps, "ar_min": ar_min, "ar_max": ar_max, "ar_mode": ar_mode, "ag_min": 1, "ag_max": 1,
           "ag_mode": 1, "idle_est": 0.05, "gate_ok": True}
    row.update(kw)
    row.update(analyze.h6_bounds(row, None))
    return row


def test_h6_uses_bounds_from_the_summary():
    pred = model.predictions()
    h6 = lambda rows: _by_id(analyze.evaluate_hypotheses(_tables_with(trace_summary=rows), pred))["H6"]
    assert h6([_trace_row(65, 65, 65)])["verdict"] == "hit"
    assert h6([_trace_row(0, 0, 0)])["verdict"] == "miss"             # AR names not recognized at tp=2
    assert h6([_trace_row(64, 65, 64)])["verdict"] == "miss"          # the mode is off: <= 50% exact
    unknown = h6([_trace_row(64, 66, 65)])
    assert unknown["verdict"] == "insufficient_data" and "trace.sqlite" in unknown["note"]
    exact = _trace_row(64, 66, 65, h6_exact_steps=255)                # counted from trace.sqlite
    exact.update(analyze.h6_bounds(exact, 255))
    assert h6([exact])["verdict"] == "hit"


def test_h7_rule():
    pred = model.predictions()

    def h7(base: float, g1: float, g2: float) -> str:
        rows = [_trace_row(65, 65, 65, arm=a, idle_est=v) for a, v in (("base", base), ("G1", g1), ("G2", g2))]
        return _by_id(analyze.evaluate_hypotheses(_tables_with(trace_summary=rows), pred))["H7"]["verdict"]

    assert h7(0.05, 0.15, 0.40) == "hit"
    assert h7(0.12, 0.15, 0.40) == "miss"
    assert h7(0.05, 0.45, 0.40) == "miss"
    assert h7(0.05, 0.15, 0.25) == "miss"
    rows = [_trace_row(65, 65, 65, arm="base", idle_est=0.05)]
    got = _by_id(analyze.evaluate_hypotheses(_tables_with(trace_summary=rows), pred))["H7"]
    assert got["verdict"] == "insufficient_data" and "G1" in got["note"] and "G2" in got["note"]


# ------------------------------------------------------------------------------------------ report

def _small(tmp_path, tiers=("P0", "P1")) -> str:
    """A smaller record set (one sweep round) for tests that edit or delete records."""
    d = str(tmp_path / "results")
    make_records.build(d, tiers=tiers, rounds=1)
    return d


def test_report_handles_missing_configs(tmp_path):
    d = _small(tmp_path)
    _drop_config(d, "DP2")
    tables = analyze.analyze(d)
    hyps = analyze.evaluate_hypotheses(tables, model.predictions())
    h = _by_id(hyps)
    assert h["H3"]["verdict"] == "insufficient_data" and "DP2" in h["H3"]["note"]
    assert h["H4"]["verdict"] == "insufficient_data" and "DP2" in h["H4"]["note"]
    assert h["H1"]["verdict"] == "hit"
    out = os.path.join(d, "SUMMARY.md")
    report.write_summary(tables, hyps, [], out, fake=False)
    text = pathlib.Path(out).read_text(encoding="utf-8")
    gaps = text.split("## Gaps", 1)[1]
    missing = [line for line in gaps.splitlines() if "| missing |" in line]
    # besides DP2, only the P1 sweep rounds that _small never ran (it builds 1 of ROUNDS rounds)
    others = [line for line in missing if "| DP2 |" not in line]
    assert missing and all("P1-serve_session-" in line and ("-r2-" in line or "-r3-" in line) for line in others)
    assert any("P0-smoke-DP2-base" in line for line in missing)
    assert any("serve_session" in line for line in missing)


def test_rounds_that_never_started_are_listed_as_missing(tmp_path):
    """Review Focus 2: a session that died before its last rounds leaves no record of them."""
    d = _small(tmp_path, tiers=("P1",))                  # 1 of ROUNDS sweep rounds
    missing = {g["run_id"] for g in analyze.analyze(d)["gaps"] if g["type"] == "missing"}
    expected = {s.run_id for s in make_records.matrix.build_matrix(("P1",), make_records.matrix.rate_grid(
        make_records.MU_RPS), "trtllm", analyze.ROUNDS) if s.round > 1}
    assert len(expected) == 3 * (analyze.ROUNDS - 1) and missing == expected


def test_summary_sections_watermark_and_gaps(records, tables, hyps, tmp_path):
    out = str(tmp_path / "SUMMARY.md")
    report.write_summary(tables, hyps, [str(tmp_path / "figures" / "x.png")], out, fake=True)
    text = pathlib.Path(out).read_text(encoding="utf-8")
    assert text.splitlines()[0] == WATERMARK
    for section in ("## Hypotheses", "## Headline", "## Offline", "## Online", "## Communication",
                    "## A-posteriori fit", "## Traces", "## Why TP2 is not 2x", "## KV capacity",
                    "## Confounder evidence", "## Gaps"):
        assert section in text, section
    hyp_section = text.split("## Hypotheses", 1)[1].split("\n## ", 1)[0]
    assert sum(1 for line in hyp_section.splitlines() if line.startswith("| H")) == 8
    gaps = text.split("## Gaps", 1)[1]
    assert "engine_start_failed" in gaps and "EXECuni" in gaps                  # the failed run and its reason
    assert "dependency_failed:synthetic" in gaps and "tokbench" in gaps          # the skipped run
    confounders = text.split("## Confounder evidence", 1)[1].split("\n## ", 1)[0]
    assert sum(1 for line in confounders.splitlines() if line[:3].strip("| ").isdigit()) == 18
    assert "raw/" in confounders and "effective_config.json" in confounders
    fit = text.split("## A-posteriori fit", 1)[1].split("\n## ", 1)[0]
    assert "| bw_eff |" in fit and "| TP2 | AR3 |" in fit and "out of sample" in fit
    why = text.split("## Why TP2 is not 2x", 1)[1].split("\n## ", 1)[0]
    assert "### Batch 1" in why and "| all-reduce and all-gather |" in why and "Traced / untraced step" in why
    headline = text.split("## Headline", 1)[1].split("\n## ", 1)[0]
    assert "above half of TP1's" in headline
    report.write_summary(tables, hyps, [], out, fake=False)
    assert "FAKE" not in pathlib.Path(out).read_text(encoding="utf-8").splitlines()[0]


def test_traces_table_shows_busy_and_idle_by_rank(tables, hyps, tmp_path):
    out = str(tmp_path / "SUMMARY.md")
    report.write_summary(tables, hyps, [], out, fake=True)
    section = pathlib.Path(out).read_text(encoding="utf-8").split("## Traces", 1)[1].split("\n## ", 1)[0]
    lines = [[c.strip() for c in line.strip("|").split("|")] for line in section.splitlines() if line.startswith("|")]
    cells = {tuple(row[:3]): dict(zip(lines[0], row)) for row in lines[2:]}
    tp2 = [r for r in tables["trace_summary"] if (r["config"], r["arm"], r["points"]) == ("TP2", "base", "decode:b1")]
    row = cells[("TP2", "base", "decode:b1")]
    assert row["busy by rank (ms/step)"] == " / ".join(f"{r['busy_ms']:.3g}" for r in tp2)
    assert row["idle by rank"] == " / ".join(f"{r['idle_rank']:.3g}" for r in tp2)
    assert cells[("TP2", "AR3", "decode:b1")]["idle by rank"] == "n/a / n/a"       # no trace.sqlite
    assert " / " not in cells[("TP1", "base", "decode:b1")]["busy by rank (ms/step)"]   # one rank
    assert "idle_est is their mean" in section


def test_unreadable_records_become_gaps_not_crashes(tmp_path):
    d = _small(tmp_path)
    raw = os.path.join(d, "raw")
    sweep = next(n for n in sorted(os.listdir(raw)) if n.startswith("P1-serve_session-TP1-base-r1"))
    with open(os.path.join(raw, sweep, "sub-0", "result.json"), "w") as f:
        f.write('{"date": "2026')                                                # truncated JSON
    with open(os.path.join(raw, sweep, "sub-1", "result.json")) as f:
        doc = json.load(f)
    doc["failed"], doc["completed"], doc["errors"][0] = 1, doc["completed"] - 1, "HTTP 500"
    with open(os.path.join(raw, sweep, "sub-1", "result.json"), "w") as f:
        json.dump(doc, f)                                                      # a rejected client run
    trace = next(n for n in sorted(os.listdir(raw)) if n.startswith("P0-trace-TP2"))
    os.remove(os.path.join(raw, trace, "trace_summary.json"))
    os.makedirs(os.path.join(raw, "stray-dir"))
    tables = analyze.analyze(d)
    gaps = tables["gaps"]
    assert any(g["run_id"] == sweep and "sub-0" in g["detail"] for g in gaps)
    assert any(g["run_id"] == trace and "trace_summary.json" in g["detail"] for g in gaps)
    assert any(g["run_id"] == "stray-dir" and "spec.json" in g["detail"] for g in gaps)
    assert any(g["type"] == "invalid" and g["run_id"] == sweep and "failed=1" in g["detail"] for g in gaps)
    bad = next(r for r in tables["online_runs"] if r["run_id"] == sweep and r["sub"] == 1)
    assert not bad["valid"]
    tp1_r1 = [r for r in tables["goodput"] if (r["round"], r["config"]) == (1, "TP1")]
    assert tp1_r1                                        # goodput still computed from the valid rates
    assert len(tables["online_runs"]) == 3 * 3 + 3 * 6 - 1                   # sat + one sweep round, minus sub-0


def test_split_dp2rand_results_are_merged(tmp_path):
    d = _small(tmp_path, tiers=("P2",))
    raw = os.path.join(d, "raw")
    session = next(n for n in sorted(os.listdir(raw)) if n.startswith("P2-serve_session-DP2rand"))
    sub = os.path.join(raw, session, "sub-0")
    os.remove(os.path.join(sub, "result.json"))
    for half, seed in ((0, 1000), (1, 1001)):
        doc = make_records.serve_result("DP2rand", "base", 1, 4.0, seed, 20)
        with open(os.path.join(sub, f"result-{half}.json"), "w") as f:
            json.dump(doc, f)
    row = next(r for r in analyze.analyze(d)["online_runs"] if r["run_id"] == session and r["sub"] == 0)
    assert row["completed"] == 40 and row["valid"]


def test_fit_and_gap_sections_without_data_say_so_once(tmp_path):
    out = str(tmp_path / "SUMMARY.md")
    report.write_summary(_tables_with(), [], [], out, fake=False)
    text = pathlib.Path(out).read_text(encoding="utf-8")
    for section in ("## A-posteriori fit", "## Why TP2 is not 2x"):
        body = text.split(section, 1)[1].split("\n## ", 1)[0]
        assert body.count("(no data)") == 1 and "###" not in body, section


def test_report_end_to_end_detects_fake(tmp_path):
    pytest.importorskip("matplotlib")
    d = _small(tmp_path)
    path = report.report(d)
    assert path == os.path.join(d, "SUMMARY.md")
    text = pathlib.Path(path).read_text(encoding="utf-8")
    assert text.splitlines()[0] == WATERMARK                          # the synthetic points say engine "fake"
    assert "](figures/decode_step_vs_batch.png)" in text


# ------------------------------------------------------------------------------------------ figures

def test_make_figures_writes_pngs(tables, tmp_path):
    pytest.importorskip("matplotlib")
    paths = plots.make_figures(tables, str(tmp_path / "figures"))
    names = {os.path.basename(p) for p in paths}
    assert len(paths) >= 5
    assert names <= set(plots.FIGURES)
    assert {"decode_step_vs_batch.png", "goodput_vs_slo.png", "allreduce_latency_vs_size.png",
            "tp2_gap_waterfall.png"} <= names
    for p in paths:
        with open(p, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"
    PIL = pytest.importorskip("PIL.Image")
    with PIL.open(paths[0]) as img:
        assert [round(x) for x in img.info["dpi"]] == [150, 150]


def test_make_figures_returns_empty_without_matplotlib(tables, tmp_path, monkeypatch):
    for name in [m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    out = tmp_path / "figures"
    assert plots.make_figures(tables, str(out)) == []
    assert not out.exists() or not any(out.iterdir())


def test_hypotheses_json_is_strict_json(records, tables):
    with open(os.path.join(records, "tidy", "hypotheses.json")) as f:
        text = f.read()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite constant {c} in hypotheses.json"))
    assert not any(isinstance(v, float) and math.isnan(v) for h in json.loads(text) for v in h.values())


def test_points_count_only_from_an_engine_that_passed_the_check(tmp_path):
    """Review C1: a real engine's points need a clean effective-config check on record (a failed check, or none
    after a hard kill, keeps them out of every table); a failed check on the GPU1 smoke drops the GPU1 step."""
    d = str(tmp_path / "results")
    specs = make_records.build(d, tiers=("P0",), rounds=1, engine="vllm")

    def edit_eff(spec, name, violations):
        path = os.path.join(d, "raw", spec.run_id, name)
        if violations is None:
            os.remove(path)
            return
        with open(path) as f:
            doc = json.load(f)
        doc["violations"] = violations
        with open(path, "w") as f:
            json.dump(doc, f)

    def offline(config, arm):
        return next(s for s in specs if s.kind == "offline" and (s.config, s.arm) == (config, arm))

    failed_check = offline("TP2", "base")
    unchecked = offline("TP2", "AR3")
    smoke = next(s for s in specs if s.kind == "smoke" and s.p("gpu1_check"))
    edit_eff(failed_check, "effective_config.json", ['missing line "Using FlashAttention version 3" (found 2).'])
    edit_eff(unchecked, "effective_config.json", None)
    edit_eff(smoke, "effective_config_gpu1.json", ['unexpected line "Default vLLM sampling parameters"'])

    tables = analyze.analyze(d)
    used = {r["run_id"] for r in tables["offline_points"]}
    assert failed_check.run_id not in used and unchecked.run_id not in used
    assert offline("TP1", "base").run_id in used                  # a vetted run stays
    gaps = {(g["run_id"], g["type"]) for g in tables["gaps"]}
    for s in (failed_check, unchecked, smoke):
        assert (s.run_id, "effective_config") in gaps
    derived = [r for r in tables["offline_points"] if r["config"] == "DP2" and r["derived"]]
    assert derived and all(r["gpu_asym"] is None for r in derived)   # no vetted GPU1 step to compare with


def test_headline_pairs_the_bootstrap_ci_with_the_pooled_s_star():
    # 2026-10-02 report printed "s* = 17.69 ms ... bootstrap 95% CI 18.82-19.03": the median of the rounds
    # next to the CI of the pooled-request s* (18.93), which does not contain it
    star = {"row": "all", "s_star_ms": 17.69, "s_star_min": 12.31, "s_star_max": 18.04, "n_crossover": 3,
            "n_rounds": 3, "s_star_pooled_ms": 18.93, "ci_lo_ms": 18.82, "ci_hi_ms": 19.03}
    text = "\n".join(report._headline({"s_star": [star]}, []))
    assert "s\\* = 17.69 ms" in text and "range 12.31–18.04 ms" in text
    assert "pooled-request s\\* 18.93 ms (request-level bootstrap 95% CI 18.82–19.03 ms" in text
    assert "17.69 ms** (median over 3/3 rounds with a crossover; range 12.31–18.04 ms; bootstrap" not in text


def test_client_cpu_counts_only_the_benchmark_phase(tmp_path):
    # 2026-10-02 box: the client p90 read ~100% from 10-15 s of startup (imports, tokenizer, dataset) while
    # the benchmark phase never passed ~65%. The window is the last `duration` seconds before the client exits.
    session, sub = tmp_path / "run", tmp_path / "run" / "sub-0"
    sub.mkdir(parents=True)
    client = "vllm bench serve --result-dir " + str(sub)
    with open(sub / "cpu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("t_wall", "pid", "ppid", "name", "cmd", "cpu_percent", "rss_mib"))
        for t in range(100, 160):
            w.writerow((t, 7, 1, "python", client, 150.0 if t < 112 else 40.0, 900))
            w.writerow((t, 5, 1, "python", "vllm serve /m", 60.0, 900))
    (session / "cmd.json").write_text(json.dumps([{"argv": client.split(), "t_wall_start": 100.0,
                                                    "t_wall_end": 160.0}]))
    (sub / "result.json").write_text(json.dumps({"duration": 47.5}))
    cpu, scope, note, errors = analyze._cpu_for(str(sub), str(session), 0)
    assert scope == "bench" and not errors
    assert cpu["client"] == pytest.approx(40.0) and cpu["api"] == pytest.approx(60.0)
    assert cpu["samples"] == 2 * 47                    # t = 113 .. 159 inside [112.5, 160]
