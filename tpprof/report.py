"""SUMMARY.md: hypothesis verdicts, headline numbers, the tables, confounder evidence and every gap
(spec 7.6, 1.3 S6/S7). write_summary only formats the tables that analyze() built.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence

from tpprof import analyze, model, plots
from tpprof.constants import DECODE_BATCHES, DECODE_INPUT_LEN, PREFILL_LENS, TPOT_SLOS_MS, TTFT_SLO_S

FAKE_WATERMARK = "> FAKE DATA — dry run with fake tools; numbers are meaningless"
SUMMARY_NAME = "SUMMARY.md"
FIGURES_DIR = "figures"

Tables = Mapping[str, Sequence[dict]]


def _f(x: object, digits: int = 4) -> str:
    v = analyze.finite(x)
    return "n/a" if v is None else f"{v:.{digits}g}"


def _ms(x: object) -> str:
    v = analyze.finite(x)
    return "n/a" if v is None else f"{v * 1e3:.4g}"


def _esc(x: object) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    rows = list(rows)
    if not rows:
        return ["(no data)", ""]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_esc(c) for c in r) + " |" for r in rows]
    return out + [""]


def _measured(m: object) -> str:
    if m is None:
        return "n/a"
    if isinstance(m, dict):
        return ", ".join(f"{k} {_f(v)}" for k, v in m.items())
    if isinstance(m, (list, tuple)):
        return "–".join(_f(v) for v in m)
    return _f(m)


def _band(b: object) -> str:
    if b is None:
        return "—"
    if isinstance(b, dict):
        return ", ".join(f"{k} {_band(v)}" for k, v in b.items())
    return f"{_f(b[0])}–{_f(b[1])}"


# ------------------------------------------------------------------------------------------ sections

def _hypotheses(hyps: Sequence[dict]) -> list[str]:
    out = ["## Hypotheses", "",
           "Bands come from `model.predictions()` (the committed `predictions.md`). Decision rules: spec 4.8 as "
           "amended by AM22-AM24.", ""]
    return out + _table(("ID", "Statement", "Measured", "Band", "Verdict", "Note"),
                        ((h["id"], h["statement"], _measured(h["measured"]), _band(h["band"]), h["verdict"], h["note"])
                         for h in hyps))


def _offline_value(t: Tables, config: str, arm: str, kind: str, batch: int, input_len: int,
                   derived: bool = False, key: str = "t_ms") -> float | None:
    vals = [r.get(key) for r in t.get("offline_points", [])
            if (r.get("config"), r.get("arm"), r.get("kind"), r.get("batch"), r.get("input_len")) ==
            (config, arm, kind, batch, input_len) and bool(r.get("derived")) == derived]
    return analyze.median_or_none(vals)


def _headline(t: Tables, hyps: Sequence[dict]) -> list[str]:
    out = ["## Headline", ""]
    star = next((r for r in t.get("s_star", []) if r.get("row") == "all"), None)
    if star and star.get("s_star_ms") is not None:
        out.append(f"- **s\\* = {_f(star['s_star_ms'])} ms** (median over {star['n_crossover']}/{star['n_rounds']} "
                   f"rounds with a crossover; range {_f(star['s_star_min'])}–{_f(star['s_star_max'])} ms, which is "
                   "the run-to-run uncertainty). Below s\\*, TP2 gets more goodput from two GPUs than two replicas "
                   f"do. Pooling every round's requests: pooled-request s\\* {_f(star.get('s_star_pooled_ms'))} ms "
                   f"(request-level bootstrap 95% CI {_f(star['ci_lo_ms'])}–{_f(star['ci_hi_ms'])} ms; within-run "
                   "noise only).")
    else:
        out.append("- s\\*: not measured (see H4 and Gaps).")
    b_lo, b_hi = min(DECODE_BATCHES), max(DECODE_BATCHES)
    for b in (b_lo, b_hi):
        s = _offline_value(t, "TP2", "base", "decode", b, DECODE_INPUT_LEN, key="speedup")
        out.append(f"- TP2 decode speedup over TP1 at batch {b}: {_f(s, 3)}x (efficiency {_f(s and s / 2, 3)})")
    for n in (min(PREFILL_LENS), max(PREFILL_LENS)):
        s = _offline_value(t, "TP2", "base", "prefill", 1, n, key="speedup")
        out.append(f"- TP2 prefill speedup over TP1 at {n} tokens: {_f(s, 3)}x")
    h3 = next((h for h in hyps if h["id"] == "H3"), None)
    if h3 is not None:
        out.append(f"- DP2 / TP2 saturation throughput: {_measured(h3['measured'])} (H3 {h3['verdict']})")
    return out + [""]


def _offline(t: Tables) -> list[str]:
    out = ["## Offline", "", "### Decode step (ms, two-length method, input 1024)", ""]
    rows = []
    for b in sorted({r["batch"] for r in t.get("offline_points", []) if r.get("kind") == "decode"}):
        tp2_row = [r for r in t["offline_points"] if (r.get("config"), r.get("arm"), r.get("kind"), r.get("batch"))
                   == ("TP2", "base", "decode", b) and not r.get("derived")]
        rows.append((b, _f(_offline_value(t, "TP1", "base", "decode", b, DECODE_INPUT_LEN)),
                     _f(_offline_value(t, "TP2", "base", "decode", b, DECODE_INPUT_LEN)),
                     _f(_offline_value(t, "DP2", "base", "decode", b, DECODE_INPUT_LEN, derived=True)),
                     _f(analyze.median_or_none(r.get("speedup") for r in tp2_row), 3),
                     _f(analyze.median_or_none(r.get("efficiency") for r in tp2_row), 3),
                     _f(_offline_value(t, "TP1", "base", "decode", b, DECODE_INPUT_LEN, key="model_ms")),
                     _f(_offline_value(t, "TP2", "base", "decode", b, DECODE_INPUT_LEN, key="model_ms"))))
    out += _table(("batch", "TP1", "TP2", "DP2 (derived)", "TP2 speedup", "TP2 efficiency", "model TP1",
                   "model TP2"), rows)
    asym = next((r for r in t.get("offline_points", []) if r.get("derived") and r.get("gpu_asym") is not None), None)
    if asym:
        out += [f"GPU0/GPU1 TP1 decode bs-1 difference (AM14): {asym['gpu_asym']:.2%}"
                f"{' — the DP2 derivation is flagged' if asym.get('flag') else ''}.", ""]
    out += ["### Prefill (ms, median of iterations, batch 1)", ""]
    rows = [(n, _f(_offline_value(t, "TP1", "base", "prefill", 1, n)), _f(_offline_value(t, "TP2", "base", "prefill", 1, n)),
             _f(_offline_value(t, "TP2", "base", "prefill", 1, n, key="speedup"), 3),
             _f(_offline_value(t, "TP1", "base", "prefill", 1, n, key="model_ms")),
             _f(_offline_value(t, "TP2", "base", "prefill", 1, n, key="model_ms")))
            for n in sorted({r["input_len"] for r in t.get("offline_points", []) if r.get("kind") == "prefill"})]
    out += _table(("tokens", "TP1", "TP2", "TP2 speedup", "model TP1", "model TP2"), rows)
    out += ["### Arms (decode step ms)", ""]
    arms = sorted({(r["config"], r["arm"]) for r in t.get("offline_points", []) if r.get("kind") == "decode"
                   and r.get("arm") != "base" and not r.get("derived")})
    rows = []
    for config, arm in arms:
        for b in sorted({r["batch"] for r in t["offline_points"] if (r.get("config"), r.get("arm"), r.get("kind"))
                         == (config, arm, "decode")}):
            v = _offline_value(t, config, arm, "decode", b, DECODE_INPUT_LEN)
            base = _offline_value(t, config, "base", "decode", b, DECODE_INPUT_LEN)
            rows.append((config, arm, b, _f(v), _f(base), _f(v / base if v and base else None, 3)))
    out += _table(("config", "arm", "batch", "step", "base step", "arm / base"), rows)
    xc = [r for r in t.get("offline_points", []) if r.get("kind") == "xcheck"]
    out += ["### `vllm bench latency` cross-check (batch 8, input 1024, output 64)", ""]
    out += _table(("config", "bench latency median (ms)", "vs driver", "note"),
                  ((r["config"], _ms(r["median_s"]), f"{r['xcheck_diff']:+.2%}" if r.get("xcheck_diff") is not None
                    else "n/a", r.get("note") or "") for r in xc))
    return out


def _online(t: Tables) -> list[str]:
    out = ["## Online", "", "### Saturation (AM6: token-emission timeline, middle 80%)", ""]
    sat = t.get("saturation", [])
    out += _table(("config", "arm", "phase", "seeds", "mu (tok/s)", "mu (req/s)", "preemptions / 1K req",
                   "grid mu (req/s)"),
                  ((r["config"], r["arm"], r["phase"], r["n_seeds"], _f(r["mu_tps"]), _f(r["mu_rps"]),
                    _f(r["preempt_per_1k"]), _f(r.get("grid_mu_rps"))) for r in sat if r.get("row") == "median"))
    out += ["### Client runs", "", "CPU p90 is per client run, over the benchmark phase only: the last `duration` "
            "seconds before the client exited, which leaves out the client's own startup (AM11). A value marked "
            "`(run)` had no benchmark duration and includes the client's startup; `(session)` had no window at all "
            "and covers the whole session. 100% is one core.", ""]
    runs = sorted(t.get("online_runs", []), key=lambda r: (str(r["phase"]), str(r["config"]), str(r["arm"]),
                                                           r["round"] or 0, r["rate_target"] or 0))
    out += _table(("phase", "config", "arm", "round", "rate", "valid", "tput (tok/s)", "TTFT p50/p99 (ms)",
                   "TPOT p50/p99 (ms)", "preemptions", "CPU p90 api/client (%)", "flags"),
                  ((r["phase"], r["config"], r["arm"], r["round"], _f(r["rate_target"]), r["valid"],
                    _f(r["tput_uniform_tps"]), f"{_ms(r['ttft_p50_s'])} / {_ms(r['ttft_p99_s'])}",
                    f"{_ms(r['tpot_p50_s'])} / {_ms(r['tpot_p99_s'])}", _f(r["preemptions"]),
                    f"{_f(r['cpu_api_p90'], 3)} / {_f(r['cpu_client_p90'], 3)}"
                    + {"run": " (run)", "session": " (session)"}.get(r.get("cpu_scope"), ""), r["flags"] or "")
                   for r in runs))
    out += [f"### Goodput (req/s, TTFT <= {TTFT_SLO_S:g} s, median over rounds)", ""]
    gp = [r for r in t.get("goodput", []) if r.get("ttft_slo_s") == TTFT_SLO_S and r.get("arm") == "base"]
    configs = [c for c in plots.ONLINE_CONFIGS if any(r["config"] == c for r in gp)]
    rows = []
    for slo in TPOT_SLOS_MS:
        rows.append((slo, *(_f(analyze.median_or_none(r["goodput_rps"] for r in gp
                                                 if r["config"] == c and r["tpot_slo_ms"] == slo)) for c in configs)))
    out += _table(("TPOT SLO (ms)", *configs), rows if configs else [])
    out += ["### s* per round", ""]
    out += ["Per-round CIs resample requests within one run, so they leave out run-to-run variance; compare "
            "the rounds instead.", ""]
    out += _table(("round", "s* (ms)", "pooled-request s* (ms)", "bootstrap CI (ms)", "note"),
                  ((r["round"] if r["row"] == "round" else "all (median)", _f(r["s_star_ms"]),
                    _f(r.get("s_star_pooled_ms")) if r["row"] == "all" else "",
                    f"{_f(r['ci_lo_ms'])}–{_f(r['ci_hi_ms'])}", r.get("note") or "") for r in t.get("s_star", [])))
    return out


def _communication(t: Tables) -> list[str]:
    out = ["## Communication", "", "alpha from sizes <= 64 KiB, beta from sizes >= 8 MiB (spec 4.6). M2 is the "
           "production fused kernel (AM25).", ""]
    fits = [r for r in t.get("comm", []) if r.get("row") == "fit"]
    return out + _table(("source", "impl", "variant", "mode", "points", "alpha (us)", "beta (GB/s)"),
                        ((r["source"], r["impl"], r.get("variant") or "", r.get("mode") or "", r["n"],
                          _f(r["alpha_us"]), _f(r["beta_GBps"])) for r in fits))


def _traces(t: Tables) -> list[str]:
    out = ["## Traces", "", "Traced shares are fractions of the traced window; nsys step times are not headline "
           "latency (spec 4.5). idle_est = 1 - traced busy per step / untraced median step (AM16). A trace that "
           "fails the completeness gate is listed but decides no hypothesis (spec 4.5). TP2 comm = fused AR time "
           "- TP1's per-step fused_add_rms_norm time (AM16); AR1-AR3 are unfused, so their AR time is comm. "
           "Per-step rows are in `tidy/trace_steps.csv`.", ""]
    rows = [r for r in t.get("trace_summary", []) if r.get("rank") == 0]
    return out + _table(("config", "arm", "points", "gate", "steps", "AR/AG per step (mode)", "idle_est",
                         "unclassified", "AR wire / sync (ms/step)", "AR / TP1 norm / comm (ms/step)", "run"),
                        ((r["config"], r["arm"], r["points"], "ok" if r["gate_ok"] else f"FAIL: {r['gate_reasons']}",
                          r["steps"], f"{r['ar_mode']} / {r['ag_mode']}", _f(r["idle_est"], 3),
                          _f(r["unclassified_frac"], 3), f"{_f(r['ar_wire_ms'], 3)} / {_f(r['ar_sync_ms'], 3)}",
                          (f"{_f(r.get('ar_ms'), 3)} / {_f(r.get('tp1_norm_ms'), 3)} / {_f(r.get('comm_ms'), 3)}"
                           + (f" ({r['comm_note']})" if r.get("comm_ms") is None and r.get("comm_note") else ""))
                          if r.get("tp") == 2 else "", r["run_id"]) for r in rows))


def _kv(t: Tables) -> list[str]:
    out = ["## KV capacity", "", "Every engine boot's `GPU KV cache size` line. The first boot of each config runs "
           "on a cold compile cache and is flagged (D5-22); H5 uses warm boots only.", ""]
    rows = [r for r in t.get("kv_capacity", [])]
    return out + _table(("config", "arm", "kind", "boot", "engine", "KV tokens", "available KV (GiB)", "cold boot",
                         "run"),
                        ((r["config"], r["arm"], r["kind"], r["boot_index"], r["engine"], r["kv_tokens"] or "n/a",
                          _f(r.get("available_kv_gib")), r["cold_boot"], r["run_id"]) for r in rows))


def _confounders(t: Tables) -> list[str]:
    out = ["## Confounder evidence", "", "Spec section 6. Paths are relative to the results directory; the first "
           f"{analyze.MAX_EVIDENCE_PATHS} matches are shown.", ""]
    return out + _table(("#", "Confounder", "Control", "Evidence", "Found", "Check"),
                        ((r["id"], r["confounder"], r["control"], r["evidence"],
                          f"{r['n_found']} file{'' if r['n_found'] == 1 else 's'}"
                          + (f": {r['found']}" if r["found"] else ""), r.get("check") or "")
                         for r in t.get("confounders", [])))


def _gaps(t: Tables, hyps: Sequence[dict]) -> list[str]:
    out = ["## Gaps", "", "Every failed, skipped, incomplete, invalid or unreadable run, every matrix run without a "
           "record, and every hypothesis without enough data.", ""]
    rows = [(g["type"], g.get("run_id") or "", g.get("config") or "", g.get("kind") or "", g["detail"])
            for g in t.get("gaps", [])]
    rows += [("insufficient_data", h["id"], "", "hypothesis", h["note"]) for h in hyps
             if h["verdict"] == "insufficient_data"]
    if not rows:
        return out + ["None.", ""]
    return out + _table(("Type", "Run", "Config", "Kind", "Detail"), rows)


def _figures(figures: Sequence[str], out_dir: str) -> list[str]:
    out = ["## Figures", ""]
    if not figures:
        return out + ["None (matplotlib is not installed, or there was no data to plot).", ""]
    for p in figures:
        rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
        out.append(f"- [{os.path.basename(p)}]({rel})")
    return out + [""]


def write_summary(tables: Tables, hyps: Sequence[dict], figures: Sequence[str], out_path: str, fake: bool) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    lines = [FAKE_WATERMARK, ""] if fake else []
    lines += ["# Results: TP=2 vs DP=2 on vLLM 0.30.0 (Llama-3.1-8B, 2x H100 SXM)", "",
              "Generated by `python -m tpprof report` from the run records under `raw/`; the tidy tables are in "
              "`tidy/`.", ""]
    lines += _hypotheses(hyps)
    lines += _headline(tables, hyps)
    lines += _offline(tables)
    lines += _online(tables)
    lines += _communication(tables)
    lines += _traces(tables)
    lines += _kv(tables)
    lines += _figures(figures, out_dir)
    lines += _confounders(tables)
    lines += _gaps(tables, hyps)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def report(results_dir: str, fake: bool | None = None) -> str:
    """analyze -> figures -> SUMMARY.md in results_dir. fake=None detects a dry run from the records."""
    tables = analyze.analyze(results_dir)
    hyps = analyze.evaluate_hypotheses(tables, model.predictions())
    figures = plots.make_figures(tables, os.path.join(results_dir, FIGURES_DIR))
    out = os.path.join(results_dir, SUMMARY_NAME)
    write_summary(tables, hyps, figures, out, analyze.is_fake(tables) if fake is None else fake)
    return out

