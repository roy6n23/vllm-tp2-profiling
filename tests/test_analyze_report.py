from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import shutil
import sys

import pytest

from tests import make_records
from tpprof import analyze, kernels, model, plots, report

CSV_TABLES = ("offline_points", "online_runs", "saturation", "goodput", "s_star", "comm", "kv_capacity",
              "trace_summary")
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
    assert all(r["ar_mode"] == 65 and r["ag_mode"] == 1 and r["h6_exact_min"] == r["steps"] for r in tp2_b1)
    assert tp2_b1[0]["unclassified_frac"] == pytest.approx(0.01)
    assert sum(tp2_b1[0][f"cat_{c}_ms"] for c in kernels.CATEGORIES) + tp2_b1[0]["idle_ms"] == \
        pytest.approx(tp2_b1[0]["mean_step_ms"])


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


def _trace_row(ar_min: int, ar_max: int, ar_mode: int, steps: int = 256, **kw) -> dict:
    row = {"run_id": "r", "config": "TP2", "arm": "base", "tp": 2, "points": "decode:b1", "rank": 0,
           "steps": steps, "ar_min": ar_min, "ar_max": ar_max, "ar_mode": ar_mode, "ag_min": 1, "ag_max": 1,
           "ag_mode": 1, "idle_est": 0.05}
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
    assert missing and all("| DP2 |" in line for line in missing)
    assert any("P0-smoke-DP2-base" in line for line in missing)
    assert any("serve_session" in line for line in missing)


def test_summary_sections_watermark_and_gaps(records, tables, hyps, tmp_path):
    out = str(tmp_path / "SUMMARY.md")
    report.write_summary(tables, hyps, [str(tmp_path / "figures" / "x.png")], out, fake=True)
    text = pathlib.Path(out).read_text(encoding="utf-8")
    assert text.splitlines()[0] == WATERMARK
    for section in ("## Hypotheses", "## Headline", "## Offline", "## Online", "## Communication", "## Traces",
                    "## KV capacity", "## Confounder evidence", "## Gaps"):
        assert section in text, section
    hyp_section = text.split("## Hypotheses", 1)[1].split("\n## ", 1)[0]
    assert sum(1 for line in hyp_section.splitlines() if line.startswith("| H")) == 8
    gaps = text.split("## Gaps", 1)[1]
    assert "engine_start_failed" in gaps and "EXECuni" in gaps                  # the failed run and its reason
    assert "dependency_failed:synthetic" in gaps and "tokbench" in gaps          # the skipped run
    confounders = text.split("## Confounder evidence", 1)[1].split("\n## ", 1)[0]
    assert sum(1 for line in confounders.splitlines() if line[:3].strip("| ").isdigit()) == 18
    assert "raw/" in confounders and "effective_config.json" in confounders
    report.write_summary(tables, hyps, [], out, fake=False)
    assert "FAKE" not in pathlib.Path(out).read_text(encoding="utf-8").splitlines()[0]


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
    assert {"decode_step_vs_batch.png", "goodput_vs_slo.png", "allreduce_latency_vs_size.png"} <= names
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
