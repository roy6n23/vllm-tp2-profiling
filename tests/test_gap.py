from __future__ import annotations

import pytest

from tpprof import gap, kernels

# Per-step GPU ms by kernel category. TP1 runs the residual add + RMSNorm as standalone kernels (0.14 ms,
# counted inside norm_act_rope); the TP2 baseline runs that work inside the fused all-reduce kernel.
TP1 = {"gemm": 5.3, "attention": 0.4, "norm_act_rope": 0.32, "sampling": 0.02, "memcpy": 0.01, "other": 0.01}
TP1_NORM = 0.14
TP2_RANKS = ({"all_reduce": 0.34, "all_gather": 0.01, "gemm": 2.96, "attention": 0.40, "norm_act_rope": 0.17,
              "sampling": 0.02, "memcpy": 0.01, "other": 0.01},
             {"all_reduce": 0.38, "all_gather": 0.01, "gemm": 2.90, "attention": 0.40, "norm_act_rope": 0.17,
              "sampling": 0.02, "memcpy": 0.01, "other": 0.01})
TP1_STEP, TP2_STEP = 6.3, 4.1          # untraced medians, ms


def _steps(config: str, rank: int, cats: dict, norm: float, batch: int = 1, n: int = 4, arm: str = "base",
           gate_ok: bool = True, pure: bool = True, traced_step: float = 5.0) -> list[dict]:
    out = []
    for i in range(n):
        row = {"config": config, "arm": arm, "points": f"decode:b{batch}", "rank": rank, "gate_ok": gate_ok,
               "step": i, "pure_decode": pure, "last": i == n - 1, "step_ms": traced_step,
               "fused_add_rms_norm_ms": norm}
        row.update({f"cat_{c}_ms": cats.get(c, 0.0) for c in kernels.CATEGORIES})
        row["gpu_busy_ms"] = sum(cats.values())
        out.append(row)
    return out


def _point(config: str, batch: int, t_ms: float, arm: str = "base", derived: bool = False) -> dict:
    return {"config": config, "arm": arm, "kind": "decode", "batch": batch, "t_ms": t_ms, "step_s": t_ms / 1e3,
            "derived": derived}


def _tables(steps: list[dict] | None = None, points: list[dict] | None = None) -> dict:
    if steps is None:
        steps = (_steps("TP1", 0, TP1, TP1_NORM, traced_step=6.4) + _steps("TP2", 0, TP2_RANKS[0], 0.0, traced_step=4.2)
                 + _steps("TP2", 1, TP2_RANKS[1], 0.0, traced_step=4.2))
    if points is None:
        points = [_point("TP1", 1, TP1_STEP), _point("TP2", 1, TP2_STEP)]
    return {"trace_steps": steps, "offline_points": points}


def _by_component(rows: list[dict], batch: int = 1) -> dict[str, dict]:
    return {r["component"]: r for r in rows if r["batch"] == batch}


def test_components_add_up_to_the_gap_exactly():
    rows = gap.rows(_tables())
    got = _by_component(rows)
    assert list(got) == ["step", *gap.COMPONENTS]
    step = got["step"]
    assert (step["tp1_ms"], step["tp2_ms"], step["half_tp1_ms"]) == (TP1_STEP, TP2_STEP, pytest.approx(3.15))
    assert step["excess_ms"] == pytest.approx(0.95) and step["share"] == 1.0
    parts = [got[c] for c in gap.COMPONENTS]
    assert sum(r["excess_ms"] for r in parts) == pytest.approx(step["excess_ms"])
    assert sum(r["share"] for r in parts) == pytest.approx(1.0)
    assert sum(r["tp1_ms"] for r in parts) == pytest.approx(TP1_STEP)
    assert sum(r["tp2_ms"] for r in parts) == pytest.approx(TP2_STEP)
    for r in parts:
        assert r["half_tp1_ms"] == pytest.approx(r["tp1_ms"] / 2)
        assert r["excess_ms"] == pytest.approx(r["tp2_ms"] - r["tp1_ms"] / 2)
        assert r["label"]


def test_component_definitions():
    got = _by_component(gap.rows(_tables()))
    assert (got["gemm"]["tp1_ms"], got["gemm"]["tp2_ms"]) == (5.3, pytest.approx(2.93))       # mean of the ranks
    assert got["gemm"]["excess_ms"] == pytest.approx(0.28)
    assert got["attention"]["excess_ms"] == pytest.approx(0.20)
    # AM16: the fused all-reduce kernel also does TP1's standalone residual add + RMSNorm; that time is norm
    # work, and the rest of the kernel plus the all-gather is communication.
    assert got["comm"]["tp1_ms"] == 0.0
    assert got["comm"]["tp2_ms"] == pytest.approx(0.36 - TP1_NORM + 0.01)
    assert got["norm_act_rope"]["tp2_ms"] == pytest.approx(0.17 + TP1_NORM)
    assert got["other_gpu"]["tp1_ms"] == pytest.approx(0.04) and got["other_gpu"]["tp2_ms"] == pytest.approx(0.04)
    # What is left of the untraced step after every kernel: CPU time and launch gaps.
    assert got["not_gpu"]["tp1_ms"] == pytest.approx(TP1_STEP - sum(TP1.values()))
    tp2_busy = (sum(TP2_RANKS[0].values()) + sum(TP2_RANKS[1].values())) / 2
    assert got["not_gpu"]["tp2_ms"] == pytest.approx(TP2_STEP - tp2_busy)


def test_step_row_reports_how_far_the_traced_runs_are_from_the_untraced_ones():
    step = _by_component(gap.rows(_tables()))["step"]
    assert step["tp1_traced_over_untraced"] == pytest.approx(6.4 / TP1_STEP)
    assert step["tp2_traced_over_untraced"] == pytest.approx(4.2 / TP2_STEP)


def test_an_unfused_tp2_keeps_its_own_norm_kernels():
    unfused = [dict(c, norm_act_rope=0.30) for c in TP2_RANKS]              # the norm runs as its own kernel
    steps = (_steps("TP1", 0, TP1, TP1_NORM) + _steps("TP2", 0, unfused[0], 0.13) + _steps("TP2", 1, unfused[1], 0.13))
    got = _by_component(gap.rows(_tables(steps=steps)))
    assert got["comm"]["tp2_ms"] == pytest.approx(0.36 + 0.01)
    assert got["norm_act_rope"]["tp2_ms"] == pytest.approx(0.30)
    assert sum(got[c]["excess_ms"] for c in gap.COMPONENTS) == pytest.approx(got["step"]["excess_ms"])


def test_only_pure_decode_steps_of_gate_passing_baseline_traces_count():
    noise = (_steps("TP1", 0, {"gemm": 99.0}, 9.0, pure=False)               # the prefill step of the same trace
             + _steps("TP2", 0, {"gemm": 99.0}, 0.0, arm="G2")               # another arm
             + _steps("TP2", 0, {"gemm": 99.0}, 0.0, batch=32))              # a batch with no TP1 trace
    rows = gap.rows(_tables(steps=_tables()["trace_steps"] + noise,
                            points=_tables()["offline_points"] + [_point("TP1", 32, 8.2), _point("TP2", 32, 5.1),
                                                                  _point("TP2", 1, 99.0, arm="G2"),
                                                                  _point("DP2", 1, 99.0, derived=True)]))
    assert {r["batch"] for r in rows} == {1}
    assert _by_component(rows)["gemm"]["tp1_ms"] == 5.3
    failed = (_steps("TP1", 0, TP1, TP1_NORM, gate_ok=False) + _steps("TP2", 0, TP2_RANKS[0], 0.0)
              + _steps("TP2", 1, TP2_RANKS[1], 0.0))
    assert gap.rows(_tables(steps=failed)) == []


def test_a_trace_without_pure_decode_steps_uses_every_step():
    steps = (_steps("TP1", 0, TP1, TP1_NORM, pure=False) + _steps("TP2", 0, TP2_RANKS[0], 0.0, pure=False)
             + _steps("TP2", 1, TP2_RANKS[1], 0.0, pure=False))
    assert _by_component(gap.rows(_tables(steps=steps)))["gemm"]["tp1_ms"] == 5.3


def test_every_traced_batch_gets_its_rows_in_batch_order():
    b32 = (_steps("TP1", 0, dict(TP1, attention=2.0), 0.15, batch=32) + _steps("TP2", 0, TP2_RANKS[0], 0.0, batch=32)
           + _steps("TP2", 1, TP2_RANKS[1], 0.0, batch=32))
    rows = gap.rows(_tables(steps=b32 + _tables()["trace_steps"],
                            points=[_point("TP1", 32, 8.2), _point("TP2", 32, 5.1), _point("TP1", 1, TP1_STEP),
                                    _point("TP2", 1, TP2_STEP)]))
    assert [r["batch"] for r in rows] == [1] * 7 + [32] * 7
    assert _by_component(rows, 32)["step"]["excess_ms"] == pytest.approx(5.1 - 4.1)


def test_no_rows_without_both_traces_and_both_untraced_points():
    assert gap.rows({}) == []
    assert gap.rows(_tables(points=[_point("TP1", 1, TP1_STEP)])) == []
    assert gap.rows(_tables(steps=_steps("TP1", 0, TP1, TP1_NORM))) == []
