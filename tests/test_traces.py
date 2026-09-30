from __future__ import annotations

import bisect
import sqlite3

import pytest

from tests import synth_trace
from tpprof import kernels, traces


def _rank(pid, device, *, tp=2, ar_backend="trtllm", batch=1, n=20, drop_last=0):
    step = [0, 0, batch, batch]
    return {"pid": pid, "device": device, "tp": tp, "ar_backend": ar_backend, "batch": batch,
            "steps": [list(step) for _ in range(n)], "drop_last": drop_last}


def _tp2(**kw):
    drop = kw.pop("drop_last_rank1", 0)
    return [_rank(42420, 0, **kw), _rank(42421, 1, drop_last=drop, **kw)]


def _db(tmp_path, ranks, **kw) -> str:
    path = str(tmp_path / "trace.sqlite")
    synth_trace.build_trace_db(path, ranks, **kw)
    return path


def test_tp2_trtllm_counts_65_ar_and_1_ag_per_step(tmp_path):
    db = _db(tmp_path, _tp2())
    td = traces.load_trace(db)
    pairs = traces.worker_ranks(td)
    assert [d for _, d in pairs] == [0, 1]
    for pid, device in pairs:
        steps = traces.assign_steps(td, pid, device)
        assert len(steps) == 20
        assert all(s["ar_ops"] == 65 for s in steps)
        assert all(s["ag_ops"] == 1 for s in steps)
        assert all(s["nvtx"] == "execute_context_0(0)_generation_1(1)" for s in steps)
        assert all((s["n_ctx_reqs"], s["n_ctx_tokens"], s["n_gen_reqs"], s["n_gen_tokens"]) == (0, 0, 1, 1)
                   for s in steps)
        assert all(s["gpu_busy_ns"] > 0 for s in steps)
    summary = traces.summarize_trace(db, tp=2, min_steps=5)
    assert summary["gate"] == {"ok": True, "reasons": []}
    assert summary["launch_ts_missing"] == 0
    for i, r in enumerate(summary["ranks"]):
        assert (r["rank"], r["device"], r["steps"], r["decode_steps"]) == (i, i, 20, 20)
        assert r["ar_ops_per_step"] == {"min": 65, "max": 65, "mode": 65}
        assert r["ag_ops_per_step"] == {"min": 1, "max": 1, "mode": 1}
        assert 0 < r["busy_frac"] < 1
        assert r["mean_step_ms"] > 0
        assert set(r["category_frac"]) == set(kernels.CATEGORIES)
        assert sum(r["category_frac"].values()) == pytest.approx(1.0)
        assert r["category_frac"]["memcpy"] > 0
    assert summary["idle_est"] is None


def test_two_shot_counts_one_op(tmp_path):
    db = _db(tmp_path, _tp2(ar_backend="mnnvl", batch=128))
    td = traces.load_trace(db)
    for pid, device in traces.worker_ranks(td):
        steps = traces.assign_steps(td, pid, device)
        assert len(steps) == 20
        for s in steps:
            assert s["ar_ops"] == 65
            assert sum(kernels.is_ar_tail(k.name) for k in s["kernels"]) == 65
            window = (min(k.start for k in s["kernels"]), max(k.end for k in s["kernels"]))
            with_tails = traces.attribute(s["kernels"], window)["all_reduce"]
            no_tails = traces.attribute([k for k in s["kernels"] if not kernels.is_ar_tail(k.name)],
                                        window)["all_reduce"]
            assert with_tails > no_tails > 0
    summary = traces.summarize_trace(db, tp=2, min_steps=5)
    assert summary["gate"]["ok"]
    assert summary["ranks"][0]["ar_ops_per_step"]["mode"] == 65


def test_correlation_ids_collide_across_pids(tmp_path):
    db = _db(tmp_path, _tp2(), collide_correlation_ids=True)
    con = sqlite3.connect(db)
    ids = {}
    for gpid, cid in con.execute("SELECT globalPid, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        ids.setdefault(gpid >> 24, set()).add(cid)
    a, b = ids.values()
    assert a & b, "the synthetic trace must reuse correlationIds across pids"
    runtime = {(gtid >> 24, cid): start for gtid, cid, start in
               con.execute("SELECT globalTid, correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    con.close()

    td = traces.load_trace(db)
    pids = sorted(ids)
    for k in td.kernels:
        other = pids[1 - pids.index(k.pid)]
        cid = next(c for (p, c), s in runtime.items() if p == k.pid and s == k.launch_ts)
        # the launch time came from this pid's record, and the other pid's record differs
        assert runtime[(k.pid, cid)] == k.launch_ts
        if (other, cid) in runtime:
            assert runtime[(other, cid)] != k.launch_ts
    for pid, device in traces.worker_ranks(td):
        steps = traces.assign_steps(td, pid, device)
        assert [s["ar_ops"] for s in steps] == [65] * 20
        assert [s["ag_ops"] for s in steps] == [1] * 20


def test_correlation_ids_disjoint_option(tmp_path):
    db = _db(tmp_path, _tp2(), collide_correlation_ids=False)
    con = sqlite3.connect(db)
    ids = {}
    for gpid, cid in con.execute("SELECT globalPid, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        ids.setdefault(gpid >> 24, set()).add(cid)
    con.close()
    a, b = ids.values()
    assert not a & b
    assert traces.summarize_trace(db, tp=2, min_steps=5)["gate"]["ok"]


def test_step_assignment_uses_launch_time_not_gpu_start(tmp_path):
    db = _db(tmp_path, _tp2())
    td = traces.load_trace(db)
    for pid, device in traces.worker_ranks(td):
        starts = [r.start for r in traces.step_ranges(td, pid)]
        mine = [k for k in td.kernels if k.pid == pid and k.device == device]
        # the GPU runs behind the CPU: many kernels start on the GPU after the next range began
        lagging = [k for k in mine
                   if bisect.bisect_right(starts, k.start) > bisect.bisect_right(starts, k.launch_ts)]
        assert len(lagging) > len(mine) // 4
        # assigning by GPU start would give wrong per-step AR counts
        by_gpu = [0] * len(starts)
        for k in mine:
            if kernels.is_ar_launch(k.name):
                by_gpu[min(bisect.bisect_right(starts, k.start) - 1, len(starts) - 1)] += 1
        assert by_gpu != [65] * len(starts)
        steps = traces.assign_steps(td, pid, device)
        assert [s["ar_ops"] for s in steps] == [65] * 20
        assert [s["ag_ops"] for s in steps] == [1] * 20


@pytest.mark.parametrize("tp, ar_backend, batch", [
    (2, "trtllm", 1), (2, "nccl", 1), (1, "none", 1),
])
def test_synthetic_gpu_lag_is_bounded_by_one_step(tmp_path, tp, ar_backend, batch):
    # the GPU lags by up to one step, over the spec's ~256-step window, without a growing backlog
    n = 256
    ranks = [_rank(42420 + r, r, tp=tp, ar_backend=ar_backend, batch=batch, n=n) for r in range(tp)]
    td = traces.load_trace(_db(tmp_path, ranks))
    for pid, device in traces.worker_ranks(td):
        steps = traces.assign_steps(td, pid, device)
        assert len(steps) == n
        lags = [(min(e.start for e in s["kernels"]) - s["start"]) / (nxt["start"] - s["start"])
                for s, nxt in zip(steps, steps[1:])]
        assert 0.2 < min(lags) and max(lags) < 1
        assert max(lags[-20:]) <= max(lags[:20]) + 0.05


def test_gate_fails_when_rank_loses_tail(tmp_path):
    db = _db(tmp_path, _tp2(drop_last_rank1=3))
    gate = traces.completeness_gate(traces.load_trace(db), tp=2, min_steps=5)
    assert not gate.ok
    assert "rank 1 has 17 steps, rank 0 has 20" in gate.reasons
    assert all("rank 1" in r for r in gate.reasons)
    summary = traces.summarize_trace(db, tp=2, min_steps=5)
    assert summary["gate"]["ok"] is False
    assert [r["steps"] for r in summary["ranks"]] == [20, 17]


def test_gate_fails_below_min_steps(tmp_path):
    db = _db(tmp_path, _tp2(n=4))
    gate = traces.completeness_gate(traces.load_trace(db), tp=2, min_steps=5)
    assert not gate.ok
    assert any("rank 0 has 4 steps" in r and "5" in r for r in gate.reasons)


def test_gate_fails_on_wrong_rank_count(tmp_path):
    db = _db(tmp_path, [_rank(4240, 0, tp=1, ar_backend="none")])
    gate = traces.completeness_gate(traces.load_trace(db), tp=2, min_steps=5)
    assert not gate.ok
    assert any("expected 2 (pid, device) pairs, found 1" in r for r in gate.reasons)


def test_gate_fails_on_unequal_ar_counts(tmp_path):
    db = _db(tmp_path, _tp2())
    con = sqlite3.connect(db)
    # drop rank 1's first all-reduce kernel (step 0)
    con.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid = (SELECT k.rowid FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON s.id = k.demangledName WHERE k.globalPid = ? AND s.value = ? "
                "ORDER BY k.start LIMIT 1)", (42421 << 24, synth_trace.FI_ONESHOT))
    con.commit()
    con.close()
    gate = traces.completeness_gate(traces.load_trace(db), tp=2, min_steps=5)
    assert not gate.ok
    assert gate.reasons == ["rank 1 per-step AR op counts differ from rank 0 in 1 steps; first at step 0: 64 vs 65"]


def test_gate_fails_when_last_step_has_no_kernels(tmp_path):
    db = _db(tmp_path, _tp2())
    last = traces.step_ranges(traces.load_trace(db), 42421)[-1].start
    con = sqlite3.connect(db)
    # delete rank 1's last-step kernels only; its input memcpy stays and must not satisfy (d)
    con.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid = ? AND correlationId IN (SELECT "
                "correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid >> 24 = ? AND start >= ?)",
                (42421 << 24, 42421, last))
    con.commit()
    con.close()
    td = traces.load_trace(db)
    last_step = traces.assign_steps(td, 42421, 1)[-1]
    assert [k.name for k in last_step["kernels"]] == [traces.MEMCPY_NAME]
    gate = traces.completeness_gate(td, tp=2, min_steps=5)
    assert not gate.ok
    assert "rank 1 has no kernels in its last step (step 19)" in gate.reasons


def test_gate_passes_tp1(tmp_path):
    db = _db(tmp_path, [_rank(4240, 0, tp=1, ar_backend="none")])
    td = traces.load_trace(db)
    assert traces.completeness_gate(td, tp=1, min_steps=5).ok
    steps = traces.assign_steps(td, 4240, 0)
    assert [s["ar_ops"] for s in steps] == [0] * 20
    assert [s["ag_ops"] for s in steps] == [0] * 20
    assert all(sum("fused_add_rms_norm_kernel" in k.name for k in s["kernels"]) == 65 for s in steps)
    summary = traces.summarize_trace(db, tp=1, min_steps=5)
    assert summary["gate"]["ok"]
    assert summary["ar_wire_s_per_step"] is None
    assert summary["ar_sync_wait_s_per_step"] is None


@pytest.mark.parametrize("table", [
    "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL",
    "NVTX_EVENTS", "StringIds",
])
def test_missing_tables_tolerated(tmp_path, table):
    db = _db(tmp_path, _tp2(), omit_tables=[table])
    td = traces.load_trace(db)
    assert table not in td.tables
    summary = traces.summarize_trace(db, tp=2, min_steps=5)
    if table == "CUPTI_ACTIVITY_KIND_MEMCPY":
        assert summary["gate"] == {"ok": True, "reasons": []}
        assert td.copies == []
        assert summary["ranks"][0]["category_frac"]["memcpy"] == 0
    elif table == "CUPTI_ACTIVITY_KIND_RUNTIME":
        # launch times fall back to the GPU start, and every fallback is counted
        assert summary["launch_ts_missing"] == len(td.kernels) + len(td.copies) > 0
    else:
        # no kernels, no step ranges, or no kernel names: the gate names the missing table
        assert summary["gate"]["ok"] is False
        assert f"trace is missing required tables: {table}" in summary["gate"]["reasons"]


def test_kernel_names_fall_back_to_short_name(tmp_path):
    db = _db(tmp_path, _tp2(n=6))
    con = sqlite3.connect(db)
    con.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName = -1")   # an id missing from StringIds
    con.commit()
    con.close()
    td = traces.load_trace(db)
    assert "ncclDevKernel_AllGather_RING_LL" in {k.name for k in td.kernels}


def test_unknown_kernels_reported(tmp_path):
    summary = traces.summarize_trace(_db(tmp_path, _tp2()), tp=2, min_steps=5)
    for r in summary["ranks"]:
        top = dict(r["unclassified_top"])
        assert top["some_unknown_kernel_xyz"] > 0
        assert r["category_frac"]["other"] == pytest.approx(sum(top.values()))
        assert len(r["unclassified_top"]) <= 20


def test_attribution_sums_to_window_busy(tmp_path):
    td = traces.load_trace(_db(tmp_path, _tp2()))
    pid, device = traces.worker_ranks(td)[0]
    ks = [k for k in td.kernels + td.copies if k.pid == pid and k.device == device]
    lo, hi = min(k.start for k in ks), max(k.end for k in ks)
    for window in ((lo, hi), (lo + (hi - lo) // 3, lo + 2 * (hi - lo) // 3)):
        clipped = [(max(k.start, window[0]), min(k.end, window[1])) for k in ks
                   if k.end > window[0] and k.start < window[1]]
        shares = traces.attribute(ks, window)
        assert set(shares) == set(kernels.CATEGORIES)
        assert sum(shares.values()) == traces.union_ns(clipped)
    # PDL-style overlap exists, so summed durations exceed the union
    assert sum(k.end - k.start for k in ks) > traces.union_ns((k.start, k.end) for k in ks)


def test_attribute_gives_overlap_to_earlier_start():
    a = traces.Kernel(1, 0, 0, 100, "nvjet_tst_x", 0, None)
    b = traces.Kernel(1, 0, 60, 160, "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 1, None)
    assert traces.attribute([b, a], (0, 1000)) == {**dict.fromkeys(kernels.CATEGORIES, 0),
                                                   "gemm": 100, "all_reduce": 60}
    assert traces.attribute([b, a], (50, 120)) == {**dict.fromkeys(kernels.CATEGORIES, 0),
                                                   "gemm": 50, "all_reduce": 20}


def test_union_ns():
    assert traces.union_ns([]) == 0
    assert traces.union_ns([(0, 10), (5, 20), (30, 40), (40, 45)]) == 35


def test_ar_wire_and_sync_wait_per_op(tmp_path):
    summary = traces.summarize_trace(_db(tmp_path, _tp2(ar_backend="mnnvl", batch=128)), tp=2, min_steps=5)
    ar_ns = sum(d for n, d in synth_trace.kernels_for_step(2, "mnnvl", 128)
                if kernels.categorize(n) == "all_reduce")
    assert summary["ar_wire_s_per_step"] == pytest.approx(ar_ns / 1e9)
    assert summary["ar_sync_wait_s_per_step"] == pytest.approx(65 * synth_trace.SYNC_WAIT_NS / 1e9)


def test_idle_est_uses_untraced_step(tmp_path):
    db = _db(tmp_path, [_rank(4240, 0, tp=1, ar_backend="none")])
    td = traces.load_trace(db)
    steps = traces.assign_steps(td, 4240, 0)
    busy_per_step = sum(s["gpu_busy_ns"] for s in steps) / len(steps) / 1e9
    untraced = 2 * busy_per_step
    summary = traces.summarize_trace(db, tp=1, min_steps=5, untraced_step_s=untraced)
    assert summary["idle_est"] == pytest.approx(1 - busy_per_step / untraced)
    assert summary["idle_est"] == pytest.approx(0.5)


def test_measure_range_sets_the_window(tmp_path):
    ranks = _tp2()
    lo, hi = synth_trace.trace_span(ranks)
    db = _db(tmp_path, ranks, measure=(lo, hi))
    td = traces.load_trace(db)
    assert any(r.text == traces.MEASURE_RANGE for r in td.ranges)
    summary = traces.summarize_trace(db, tp=2, min_steps=5)
    assert summary["window_ns"] == [lo, hi]
    assert summary["gate"]["ok"]
    assert [r["steps"] for r in summary["ranks"]] == [20, 20]


def test_measure_range_filters_steps_outside_it(tmp_path):
    ranks = _tp2()
    probe = str(tmp_path / "probe.sqlite")
    synth_trace.build_trace_db(probe, ranks)
    starts = [r.start for r in traces.step_ranges(traces.load_trace(probe), 42420)]
    # a window from step 5's range start to just before step 15's range start
    db = _db(tmp_path, ranks, measure=(starts[5], starts[15] - 1))
    summary = traces.summarize_trace(db, tp=2, min_steps=5)
    assert [r["steps"] for r in summary["ranks"]] == [10, 10]
    assert summary["window_ns"] == [starts[5], starts[15] - 1]


def test_prefill_steps_are_not_decode_steps(tmp_path):
    ranks = _tp2(n=8)
    for r in ranks:
        r["steps"][0] = [1, 2048, 0, 0]
    summary = traces.summarize_trace(_db(tmp_path, ranks), tp=2, min_steps=5)
    assert [r["decode_steps"] for r in summary["ranks"]] == [7, 7]
    td = traces.load_trace(str(tmp_path / "trace.sqlite"))
    first = traces.assign_steps(td, 42420, 0)[0]
    assert (first["n_ctx_reqs"], first["n_ctx_tokens"], first["n_gen_reqs"], first["n_gen_tokens"]) == (1, 2048, 0, 0)


def test_step_re():
    assert traces.STEP_RE.match("execute_context_1(4)_generation_1(1)").groups() == ("1", "4", "1", "1")
    assert traces.STEP_RE.match("execute_5_context_1(sq4sk4sqsq16sqsk16)_generation_1(sq1sk11sqsq1sqsk11)") is None
