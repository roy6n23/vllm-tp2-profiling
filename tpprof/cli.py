"""`python -m tpprof <subcommand>`: preflight, env, matrix, run, traces, analyze, report, predict.

Every tpprof import happens inside its handler, so `preflight --quick` needs only the standard
library and nvidia-smi (AM27): it runs right after SSH, before numpy or psutil are known to work.
`--model-dir` defaults to $TPPROF_MODEL_DIR; `--results-dir` to $TPPROF_RESULTS_DIR, else `results`
(`run --dry-run` defaults to `results/dryrun` and ignores $TPPROF_RESULTS_DIR).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TIER_NAMES = ("P0", "P1", "P2")
TRACE_UNCLASSIFIED_MAX = 0.01      # AM19: the unclassified share must stay below 1 %


def _results_dir(args: argparse.Namespace, dry_run: bool = False) -> str:
    """--results-dir, else $TPPROF_RESULTS_DIR, else results. A dry run never falls back to the real results
    dir: without --results-dir it writes to results/dryrun (git-ignored)."""
    if getattr(args, "results_dir", None):
        return args.results_dir
    if dry_run:
        return os.path.join("results", "dryrun")
    return os.environ.get("TPPROF_RESULTS_DIR") or "results"


def _tiers(text: str) -> list[str]:
    """`P0,P1`, `all`; repeated --tier options are joined by the caller."""
    names = [t.strip() for t in text.split(",") if t.strip()]
    if "all" in names:
        return list(TIER_NAMES)
    bad = [t for t in names if t not in TIER_NAMES]
    if bad or not names:
        raise argparse.ArgumentTypeError(f"unknown tier {', '.join(bad) or text!r}; use P0, P1, P2 or all")
    return names


def _joined(values: list[list[str]] | None, default: list[str]) -> list[str]:
    if not values:
        return list(default)
    return list(dict.fromkeys(t for group in values for t in group))


def _names(values: list[list[str]] | None) -> list[str]:
    """`--only a b --only c,d` -> [a, b, c, d]: repeatable, space- or comma-separated, duplicates dropped."""
    return list(dict.fromkeys(n.strip() for group in values or [] for item in group for n in item.split(",")
                              if n.strip()))


def _add_gate_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--accept-topology", action="store_true", help="accept a GPU0-GPU1 link other than NV18 (recorded)")
    p.add_argument("--skip-gate", action="append", nargs="+", metavar="NAME",
                   help="override failed hard gates by name (recorded in preflight.json)")


# ---------------------------------------------------------------- handlers


def _cmd_preflight(args: argparse.Namespace) -> int:
    from tpprof import preflight

    ctx = preflight.PreflightContext(model_dir=args.model_dir or os.environ.get("TPPROF_MODEL_DIR", ""),
                                     results_dir=_results_dir(args), accept_topology=args.accept_topology,
                                     skip_gates=tuple(_names(args.skip_gate)))
    checks = preflight.quick_checks(ctx) if args.quick else preflight.full_checks(ctx)
    ok, text = preflight.verdict(checks)
    print(text)
    return 0 if ok else 1


def _cmd_env(args: argparse.Namespace) -> int:
    from tpprof import envcapture, preflight

    ctx = preflight.PreflightContext(model_dir=os.environ.get("TPPROF_MODEL_DIR", ""), results_dir=_results_dir(args))
    print(json.dumps(envcapture.capture_env(ctx), indent=2, sort_keys=True))
    return 0


def _read_json(path: str) -> object:
    with open(path) as f:
        return json.load(f)


def _cmd_matrix(args: argparse.Namespace) -> int:
    from tpprof import matrix

    tiers = _joined(args.tier, list(TIER_NAMES))
    raw = os.path.join(_results_dir(args), "raw")
    grid_path = os.path.join(raw, "rate_grid.json")
    if os.path.exists(grid_path):
        doc = _read_json(grid_path)
        grid, mu, source = doc["grid"], doc["mu_rps"], f"rate grid: measured ({grid_path})"
    else:
        mu = matrix.model_mu_rps()
        grid, source = matrix.rate_grid(mu), "rate grid: a-priori model (no raw/rate_grid.json yet)"
    fi_backend = "mnnvl"               # unknown before the TP2 smoke: assume the larger P2 (A-FIB included)
    for s in matrix.p0_specs():
        path = os.path.join(raw, s.run_id, "effective_config.json")
        if s.kind == "smoke" and s.config == "TP2" and os.path.exists(path):
            fi_backend = _read_json(path).get("fi_backend")
    specs = matrix.build_matrix(tiers, grid, fi_backend)
    print(source)
    if args.list or not args.estimate:
        for s in specs:
            print(f"{s.run_id:<60} {s.kind:<21} {s.config:<8} {s.arm:<9} {dict(s.params)}")
    if args.estimate:
        print(matrix.format_estimate(matrix.estimate(specs, mu)), end="")
    return 0


def _cmd_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from tpprof import matrix, runner

    tiers = _joined(args.tier, [])
    only = _names(args.only)
    bad = [k for k in only if k not in matrix.KINDS]
    if bad:
        parser.error(f"--only: unknown kind {', '.join(bad)}; expected some of {', '.join(matrix.KINDS)}")
    results_dir = _results_dir(args, args.dry_run)
    if args.dry_run:
        ctx = runner.dry_run_context(results_dir)
        ctx.skip_gates = tuple(dict.fromkeys([*ctx.skip_gates, *_names(args.skip_gate)]))
    else:
        model_dir = args.model_dir or os.environ.get("TPPROF_MODEL_DIR")
        if not model_dir:
            parser.error("run needs --model-dir (or TPPROF_MODEL_DIR), except with --dry-run")
        ctx = runner.RunContext(results_dir=results_dir, model_dir=model_dir, base_env=dict(os.environ),
                                skip_gates=tuple(_names(args.skip_gate)))
    ctx.accept_topology = args.accept_topology
    if args.rounds is not None:
        ctx.rounds = args.rounds
    if args.port_base is not None:
        ctx.port_base = args.port_base
    r = runner.Runner(ctx)
    summary = r.run_tiers(tiers, only, retry_failed=args.retry_failed)
    print(f"{len(summary['done'])} done, {len(summary['failed'])} failed, {len(summary['skipped'])} skipped "
          f"({os.path.join(ctx.results_dir, 'raw', '_last_run.json')})")
    for run_id in summary["failed"]:
        print(f"  failed  {run_id}")
    if r.interrupted:
        print("interrupted: rerun the same command to resume")
        return 130
    return 1 if summary["failed"] else 0


def _trace_lines(run_id: str, spec: dict, summ: dict) -> tuple[list[str], bool]:
    from tpprof import model

    tp = int((spec.get("engine") or {}).get("tp") or len(summ.get("ranks") or []) or 1)
    want_ar, want_ag = model.allreduces_per_step(tp), model.allgathers_per_step(tp)
    gate = summ.get("gate") or {}
    ok = True
    lines = [f"{run_id}  {spec.get('config')}/{spec.get('arm')} {dict(spec.get('params') or []).get('points')}  "
             f"gate {'ok' if gate.get('ok') else 'FAILED: ' + '; '.join(map(str, gate.get('reasons', [])))}"]
    for rk in summ.get("ranks") or []:
        other = float((rk.get("category_frac") or {}).get("other", 0.0))
        ar, ag = rk.get("ar_ops_per_step") or {}, rk.get("ag_ops_per_step") or {}
        h6 = all(d.get(k) == want for d, want in ((ar, want_ar), (ag, want_ag)) for k in ("min", "max", "mode"))
        bad_share = other >= TRACE_UNCLASSIFIED_MAX
        ok &= h6 and not bad_share
        top = ", ".join(f"{name} {frac:.2%}" for name, frac in (rk.get("unclassified_top") or [])[:3])
        lines.append(f"    rank {rk.get('rank')}: unclassified {other:.2%}{' (>= 1%)' if bad_share else ''}"
                     f"{' [' + top + ']' if bad_share and top else ''}; "
                     f"H6 {'ok' if h6 else 'off'}: AR/step {ar.get('min')}..{ar.get('max')} (want {want_ar}), "
                     f"AG/step {ag.get('min')}..{ag.get('max')} (want {want_ag})")
    return lines, ok


def _cmd_traces(args: argparse.Namespace) -> int:
    raw = os.path.join(_results_dir(args), "raw")
    found, all_ok = 0, True
    for name in sorted(os.listdir(raw)) if os.path.isdir(raw) else []:
        d = os.path.join(raw, name)
        spec_path, summ_path = os.path.join(d, "spec.json"), os.path.join(d, "trace_summary.json")
        if name.startswith("_") or not (os.path.exists(spec_path) and os.path.exists(summ_path)):
            continue
        spec = _read_json(spec_path)
        if spec.get("kind") != "trace":
            continue
        found += 1
        lines, ok = _trace_lines(name, spec, _read_json(summ_path))
        all_ok &= ok
        print("\n".join(lines))
    if not found:
        print(f"no trace summaries under {raw}")
        return 1
    print(f"traces --check: {found} traces, {'OK' if all_ok else 'FAILED (unclassified >= 1% or H6 off)'}")
    return 0 if all_ok else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    from tpprof import analyze

    results_dir = _results_dir(args)
    tables = analyze.analyze(results_dir)
    for name, rows in tables.items():
        print(f"{name:<16} {len(rows)} rows")
    print(f"wrote {os.path.join(results_dir, 'tidy')}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from tpprof import analyze, model, plots, report

    results_dir = _results_dir(args)
    tables = analyze.analyze(results_dir)
    hyps = analyze.evaluate_hypotheses(tables, model.predictions())
    figures = plots.make_figures(tables, os.path.join(results_dir, "figures"))
    fake = os.path.exists(os.path.join(results_dir, "raw", "_dry_run.json"))
    out = os.path.join(results_dir, "SUMMARY.md")
    report.write_summary(tables, hyps, figures, out, fake)
    print(f"wrote {out} ({len(figures)} figures{', FAKE data' if fake else ''})")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from tpprof import predict

    return predict.main(["--out", args.out])


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tpprof", description="TP=2 vs DP=2 profiling harness for vLLM 0.30.0")
    sub = ap.add_subparsers(dest="command", required=True, metavar="SUBCOMMAND")

    p = sub.add_parser("preflight", help="hard gates and soft warnings (--quick: stdlib + nvidia-smi only)")
    p.add_argument("--quick", action="store_true", help="hardware gates only, before any download (AM27)")
    _add_gate_options(p)
    p.add_argument("--model-dir")
    p.set_defaults(func=_cmd_preflight)

    p = sub.add_parser("env", help="print env.json")
    p.set_defaults(func=_cmd_env)

    p = sub.add_parser("matrix", help="the run matrix and its time/cost estimate")
    p.add_argument("--tier", action="append", type=_tiers, help="P0,P1,P2 or all (default all)")
    p.add_argument("--list", action="store_true", help="one line per run spec (the default)")
    p.add_argument("--estimate", action="store_true", help="hours and cost per tier (AM21)")
    p.add_argument("--results-dir", help="where raw/rate_grid.json is read from")
    p.set_defaults(func=_cmd_matrix)

    p = run_p = sub.add_parser("run", help="run tiers of the matrix (resumable)")
    p.add_argument("--tier", action="append", type=_tiers, required=True, help="P0,P1,P2 or all")
    p.add_argument("--only", action="append", nargs="+", metavar="KIND", help="only these run kinds")
    p.add_argument("--rounds", type=int, help="P1 sweep rounds (default 3)")
    p.add_argument("--dry-run", action="store_true", help="fake tools, fake engine, scaled time (spec 7.4)")
    p.add_argument("--retry-failed", action="store_true", help="rerun runs that have failed.json")
    p.add_argument("--results-dir")
    p.add_argument("--model-dir")
    p.add_argument("--port-base", type=int, help="first server port (default 8000)")
    _add_gate_options(p)
    p.set_defaults(func=lambda a: _cmd_run(a, run_p))       # usage errors name the run subcommand

    p = sub.add_parser("traces", help="check trace summaries (AM19)")
    p.add_argument("--check", action="store_true", required=True,
                   help="gates, unclassified share and H6 per trace; exit 1 if a share >= 1%% or an H6 count is off")
    p.add_argument("--results-dir")
    p.set_defaults(func=_cmd_traces)

    for name, func, text in (("analyze", _cmd_analyze, "tidy CSVs and hypotheses.json"),
                             ("report", _cmd_report, "figures and SUMMARY.md")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--results-dir")
        p.set_defaults(func=func)

    p = sub.add_parser("predict", help="write predictions.md from tpprof/model.py")
    p.add_argument("--out", default="predictions.md")
    p.set_defaults(func=_cmd_predict)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)
