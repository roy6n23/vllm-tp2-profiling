from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tests.conftest import ROOT
from tpprof import constants as c
from tpprof import engine as e
from tpprof import offline
from tpprof.offline import Point
from tpprof.results import LATENCY_PERCENTILES, load_latency_result

PY = sys.executable
C3_META_KEYS = {"kind", "batch", "input_len", "output_len", "warmup", "iters", "config", "arm",
                "t_wall_start", "t_mono_start", "engine"}


def _env(**overrides: str) -> dict[str, str]:
    """TP2 base environment (the runner sets it for the driver) plus the fake-engine switches."""
    env = e.base_config("TP2").environment(os.environ)
    env.update(TPPROF_FAKE="1", FAKE_TIME_SCALE="0.001")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)
    env.pop("FAKE_NSYS_EVENTS", None)
    env.update(overrides)
    return env


def _driver(tmp_path, extra: list[str], cfg: e.EngineConfig | None = None, **env: str):
    cfg = cfg or e.base_config("TP2")
    meta = json.dumps({"config": cfg.name, "arm": cfg.arm})
    argv = [PY, "-m", "tpprof.offline", "--out", str(tmp_path / "points"), "--meta", meta, *extra,
            "--", *cfg.offline_args("/models/llama")]
    return subprocess.run(argv, cwd=tmp_path, env=_env(**env), capture_output=True, text=True,
                          timeout=120)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("TPPROF_FAKE", "1")
    monkeypatch.setenv("FAKE_TIME_SCALE", "0.001")
    monkeypatch.delenv("FAKE_NSYS_EVENTS", raising=False)
    for name in ("VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "FAKE_VLLM_NO_MULTICAST"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------------- points

def test_default_points_count_and_values():
    pts = offline.default_points()
    assert len(pts) == 3 + 16
    prefill = [p for p in pts if p.kind == "prefill"]
    assert prefill == [Point("prefill", 1, n, 1, c.PREFILL_WARMUP, c.PREFILL_ITERS) for n in c.PREFILL_LENS]
    decode = [p for p in pts if p.kind == "decode"]
    assert decode == [Point("decode", b, c.DECODE_INPUT_LEN, n, c.DECODE_WARMUP, c.DECODE_ITERS)
                      for b in c.DECODE_BATCHES for n in (c.DECODE_L1, c.DECODE_L2)]
    assert (c.PREFILL_WARMUP, c.PREFILL_ITERS, c.DECODE_WARMUP, c.DECODE_ITERS) == (5, 20, 3, 10)


def test_point_filename_is_c3_name():
    assert Point("decode", 8, 1024, 64, 3, 10).filename() == "point-decode-b8-i1024-o64.json"
    assert Point("prefill", 1, 2048, 1, 5, 20).filename() == "point-prefill-b1-i2048-o1.json"


def test_parse_points_mixed_spec():
    pts = offline.parse_points("decode:b1,b32;prefill:2048")
    assert pts == [Point("decode", 1, 1024, 64, 3, 10), Point("decode", 1, 1024, 320, 3, 10),
                   Point("decode", 32, 1024, 64, 3, 10), Point("decode", 32, 1024, 320, 3, 10),
                   Point("prefill", 1, 2048, 1, 5, 20)]


def test_parse_points_kind_words():
    assert offline.parse_points("all") == offline.default_points()
    assert offline.parse_points("prefill") == [p for p in offline.default_points() if p.kind == "prefill"]
    assert offline.parse_points("decode") == [p for p in offline.default_points() if p.kind == "decode"]
    assert offline.parse_points("decode:b1,b1;decode:b1") == offline.parse_points("decode:b1")


@pytest.mark.parametrize("spec", ["", "bogus", "decode:32", "decode:bx", "prefill:b2048", "decode:b0",
                                  "prefill:", "decode:b1;;prefill:512"])
def test_parse_points_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        offline.parse_points(spec)


# ---------------------------------------------------------------------------------- fake engine

def test_fake_engine_ar_backend_follows_config_env(fake):
    tp2 = e.base_config("TP2").offline_args("/m")
    assert offline.make_engine(tp2, False, {"arm": "base"}).ar_backend == "mnnvl"
    assert offline.make_engine(e.base_config("TP1").offline_args("/m"), False, {"arm": "base"}).ar_backend == "none"
    fake.setenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0")
    assert offline.make_engine(e.arm_config("TP2", "AR2").offline_args("/m"), False,
                               {"arm": "AR2"}).ar_backend == "custom"
    ar3 = offline.make_engine(e.arm_config("TP2", "AR3").offline_args("/m"), False, {"arm": "AR3"})
    assert (ar3.ar_backend, ar3.tp, ar3.name) == ("nccl", 2, "fake")


def test_fake_engine_trtllm_when_forced_or_no_multicast(fake):
    args = e.arm_config("TP2", "FIBtrtllm").offline_args("/m")
    fake.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "trtllm")
    assert offline.make_engine(args, False, {"arm": "FIBtrtllm"}).ar_backend == "trtllm"
    fake.delenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND")
    fake.setenv("FAKE_VLLM_NO_MULTICAST", "1")
    assert offline.make_engine(e.base_config("TP2").offline_args("/m"), False, {"arm": "base"}).ar_backend == "trtllm"


def test_fake_engine_reads_graph_flags(fake):
    g2 = offline.make_engine(e.arm_config("TP2", "G2").offline_args("/m"), False, {"arm": "G2"})
    assert (g2.enforce_eager, g2.cudagraph_mode) == (True, "NONE")
    g1 = offline.make_engine(e.arm_config("TP1", "G1").offline_args("/m"), False, {"arm": "G1"})
    assert (g1.enforce_eager, g1.cudagraph_mode) == (False, "NONE")
    base = offline.make_engine(e.base_config("TP2").offline_args("/m"), False, {"arm": "base"})
    assert base.cudagraph_mode == "FULL_AND_PIECEWISE"


def test_fake_engine_rejects_what_the_llm_class_rejects(fake):
    with pytest.raises(ValueError, match="data_parallel_size"):
        offline.make_engine(e.base_config("DP2").offline_args("/m"), False, {"arm": "base"})
    with pytest.raises(ValueError, match="--compilation-config"):
        offline.make_engine(["--tensor-parallel-size", "1", "--compilation-config", "{bad"], False, {})
    eng = offline.make_engine(["--tensor-parallel-size", "1"], False, {})
    with pytest.raises(RuntimeError, match="Profiling is not enabled"):
        eng.start_profile()


def test_make_engine_without_fake_uses_vllm(monkeypatch):
    monkeypatch.delenv("TPPROF_FAKE", raising=False)
    try:
        import vllm  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("vllm is installed; this checks the Mac/CI path where it is not")
    with pytest.raises(ImportError, match="vllm"):
        offline.make_engine(e.base_config("TP1").offline_args("/m"), False, {"arm": "base"})


def test_import_does_not_pull_in_vllm_or_torch():
    code = "import sys, tpprof.offline; print(sorted(m for m in ('vllm', 'torch') if m in sys.modules))"
    out = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------------------- run_points

def test_run_points_writes_c3_files(fake, tmp_path):
    meta = {"config": "TP2", "arm": "base"}
    eng = offline.make_engine(e.base_config("TP2").offline_args("/m"), False, meta)
    pts = offline.parse_points("decode:b2;prefill:512")
    written = offline.run_points(eng, pts, str(tmp_path), meta)
    assert written == [str(tmp_path / p.filename()) for p in pts]
    for p in pts:
        res = load_latency_result(str(tmp_path / p.filename()))
        assert len(res.latencies) == p.iters and all(x > 0 for x in res.latencies)
        assert set(res.percentiles) == {str(q) for q in LATENCY_PERCENTILES}
        assert set(res.meta) == C3_META_KEYS
        assert {k: res.meta[k] for k in ("kind", "batch", "input_len", "output_len", "warmup", "iters")} == {
            "kind": p.kind, "batch": p.batch, "input_len": p.input_len, "output_len": p.output_len,
            "warmup": p.warmup, "iters": p.iters}
        assert (res.meta["config"], res.meta["arm"], res.meta["engine"]) == ("TP2", "base", "fake")


def test_decode_lengths_are_interleaved(fake, tmp_path):
    log: list = []
    meta = {"config": "TP1", "arm": "base", "call_log": log}
    eng = offline.make_engine(e.base_config("TP1").offline_args("/m"), False, meta)
    offline.run_points(eng, offline.parse_points("decode:b4"), str(tmp_path), meta)
    gens = [entry for entry in log if entry[0] == "generate"]
    assert all(entry[1:3] == (4, c.DECODE_INPUT_LEN) for entry in gens)
    lens = [entry[3] for entry in gens]
    l1, l2 = c.DECODE_L1, c.DECODE_L2
    assert lens == [l1] * 3 + [l2] * 3 + [l1, l2] * 10
    assert "call_log" not in load_latency_result(str(tmp_path / "point-decode-b4-i1024-o64.json")).meta


def test_prompts_are_seeded_and_reused(fake, tmp_path):
    log: list = []
    meta = {"config": "TP1", "arm": "base", "call_log": log}
    eng = offline.make_engine(["--tensor-parallel-size", "1"], False, meta)
    pt = Point("prefill", 2, 16, 1, 2, 3)
    offline.run_points(eng, [pt], str(tmp_path), meta, seed=7)
    want = np.random.default_rng(7).integers(0, 10000, size=(2, 16)).tolist()
    prompts = [entry[4] for entry in log if entry[0] == "generate"]
    assert len(prompts) == 5 and all(p == want for p in prompts)


def test_run_points_resumes_per_point(fake, tmp_path):
    meta = {"config": "TP1", "arm": "base", "call_log": []}
    eng = offline.make_engine(["--tensor-parallel-size", "1"], False, meta)
    pts = offline.parse_points("decode:b1;prefill:512")
    (tmp_path / pts[0].filename()).write_text('{"avg_latency": 1.0, "latencies": [1.0], "percentiles"')  # truncated
    offline.run_points(eng, pts[1:], str(tmp_path), meta)
    meta["call_log"].clear()
    written = offline.run_points(eng, pts, str(tmp_path), meta)
    assert written == [str(tmp_path / pts[0].filename())]                    # truncated file redone, rest skipped
    assert {entry[3] for entry in meta["call_log"] if entry[0] == "generate"} == {c.DECODE_L1}


# ---------------------------------------------------------------------------------- CLI

def test_cli_writes_points_then_resumes(tmp_path):
    out = tmp_path / "points"
    r = _driver(tmp_path, ["--points", "decode:b1;prefill:512"])
    assert r.returncode == 0, r.stderr
    names = sorted(os.listdir(out))
    assert names == ["point-decode-b1-i1024-o320.json", "point-decode-b1-i1024-o64.json",
                     "point-prefill-b1-i512-o1.json"]
    for name in names:
        doc = json.loads((out / name).read_text())
        assert set(doc) == {"avg_latency", "latencies", "percentiles", "tpprof"}
        assert set(doc["tpprof"]) == C3_META_KEYS
    mtimes = {n: (out / n).stat().st_mtime_ns for n in names}
    r2 = _driver(tmp_path, ["--points", "decode:b1;prefill:512"])
    assert r2.returncode == 0, r2.stderr
    assert {n: (out / n).stat().st_mtime_ns for n in names} == mtimes
    assert "all 3 points already done" in r2.stderr


def test_cli_fails_loudly(tmp_path):
    r = _driver(tmp_path, ["--points", "decode:b1"], cfg=e.base_config("DP2"))
    assert r.returncode != 0
    assert "Traceback" in r.stderr and "data_parallel_size" in r.stderr
    r = subprocess.run([PY, "-m", "tpprof.offline", "--out", str(tmp_path), "--meta", "{}", "--", "--model", "/m"],
                       cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "config" in r.stderr


def test_cli_profile_writes_c4_lines(tmp_path):
    events = tmp_path / "events.jsonl"
    r = _driver(tmp_path, ["--profile", "decode:b4", "--profile-warmup", "1", "--profile-iters", "2"],
                FAKE_NSYS_EVENTS=str(events))
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in events.read_text().splitlines()]
    assert [x["rank"] for x in lines] == [0, 1]
    assert [x["device"] for x in lines] == [0, 1]
    assert all(x["tp"] == 2 and x["ar_backend"] == "mnnvl" and x["batch"] == 4 for x in lines)
    assert lines[0]["pid"] != lines[1]["pid"] and lines[1]["pid"] - lines[0]["pid"] == 1
    out_len = c.DECODE_L2 - c.DECODE_L1
    one_iter = [[4, 4 * c.DECODE_INPUT_LEN, 0, 0]] + [[0, 0, 4, 4]] * (out_len - 1)
    assert all(x["steps"] == one_iter * 2 for x in lines)                  # warmup is outside the window
    assert not (tmp_path / "points").exists() or not list((tmp_path / "points").glob("point-*.json"))


def test_cli_profile_prefill_tp1(tmp_path):
    events = tmp_path / "events.jsonl"
    r = _driver(tmp_path, ["--profile", "prefill:2048", "--profile-warmup", "2", "--profile-iters", "5"],
                cfg=e.base_config("TP1"), FAKE_NSYS_EVENTS=str(events))
    assert r.returncode == 0, r.stderr
    (line,) = [json.loads(x) for x in events.read_text().splitlines()]
    assert (line["rank"], line["tp"], line["ar_backend"], line["batch"]) == (0, 1, "none", 1)
    assert line["steps"] == [[1, 2048, 0, 0]] * 5


def test_cli_profile_needs_exactly_one_shape(tmp_path):
    r = _driver(tmp_path, ["--profile", "decode:b1,b32", "--profile-warmup", "1", "--profile-iters", "1"])
    assert r.returncode != 0 and "exactly one" in r.stderr
    r = _driver(tmp_path, ["--profile", "decode:b1"])
    assert r.returncode != 0 and "--profile-iters" in r.stderr


def test_run_profile_order(fake, tmp_path):
    log: list = []
    meta = {"config": "TP2", "arm": "base", "call_log": log}
    eng = offline.make_engine(e.base_config("TP2").offline_args("/m"), True, meta)
    offline.run_profile(eng, Point("decode", 1, 8, 4, 3, 10), warmup=2, iters=3, sleep_before_stop_s=0.0)
    assert [entry[0] for entry in log] == (["generate"] * 2 + ["start_profile", "range_push"] + ["generate"] * 3
                                           + ["range_pop", "stop_profile"])
    assert log[3][1] == "tpprof:measure"


# ---------------------------------------------------------------------------------- spawn guard (AM32, build-7)

def test_spawn_guard(tmp_path):
    """The fake engine starts one spawn child, like vLLM's workers; `python -m tpprof.offline` survives it."""
    r = _driver(tmp_path, ["--points", "prefill:512"])
    assert r.returncode == 0, r.stderr
    assert "spawn child ok" in r.stderr


def test_spawn_guard_catches_a_missing_main_guard(tmp_path):
    """Negative control: an entry point without `if __name__ == "__main__"` fails under spawn."""
    script = tmp_path / "noguard.py"
    script.write_text(textwrap.dedent(f"""\
        from tpprof.offline import main
        raise SystemExit(main(["--out", {str(tmp_path / "p")!r}, "--meta", '{{"config": "TP1", "arm": "base"}}',
                               "--points", "prefill:512", "--", "--tensor-parallel-size", "1"]))
        """))
    r = subprocess.run([PY, str(script)], cwd=tmp_path, env=_env(), capture_output=True, text=True,
                       timeout=120)
    assert r.returncode != 0
    assert "spawn child" in r.stderr and "__main__" in r.stderr
