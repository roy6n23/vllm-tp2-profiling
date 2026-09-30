from __future__ import annotations

from tests.conftest import FIXTURES
from tpprof import engine as e
from tpprof import helpflags as h

SERVE = (FIXTURES / "serve_help_all.txt").read_text()
LAT = (FIXTURES / "bench_latency_help.txt").read_text()


def test_parse_real_help_has_known_flags_and_no_log_junk():
    flags = h.parse_help_flags(SERVE)
    for f in ("--enable-prefix-caching", "--no-enable-prefix-caching", "--tensor-parallel-size", "-tp",
              "--max-num-batched-tokens", "--compilation-config", "-cc", "--attention-config", "--api-server-count"):
        assert f in flags, f
    assert not any(x.startswith("-cc.") for x in flags)
    assert "--disable-log-requests" not in flags


def test_parse_ignores_description_and_log_lines():
    text = (
        "INFO 09-30 08:07:50 [importing.py:98] --not-a-flag here\n"
        "WARNING 09-30 08:08:01 [cpu.py:631] --also-not\n"
        "  --real-flag REAL_FLAG, -rf REAL_FLAG\n"
        "                        see -cc.mode=none`. and --other-flag. (default: None)\n"
        "                        -cc.cudagraph_mode=none`. (default: False)\n"
        "                        --data-parallel-start-rank. (default: False)\n"
        "  --bool-flag, --no-bool-flag\n"
        "  --choice {a,b}        Pick one. (default: a)\n"
    )
    assert h.parse_help_flags(text) == {"--real-flag", "-rf", "--bool-flag", "--no-bool-flag", "--choice"}


def test_parse_real_bench_latency_help():
    flags = h.parse_help_flags(LAT)
    for f in ("--batch-size", "--input-len", "--output-len", "--num-iters-warmup", "--num-iters", "--output-json",
              "--model", "--enforce-eager", "--disable-custom-all-reduce"):
        assert f in flags, f
    assert "--api-server-count" not in flags and "--port" not in flags


def test_flags_in_argv_canonicalizes():
    argv = ["vllm", "serve", "/m", "--max_num_seqs", "8", "--profiler-config.profiler", "cuda", "-O2", "--x=1"]
    assert h.flags_in_argv(argv) == ["--max-num-seqs", "--profiler-config", "-O2", "--x"]


def test_flags_in_argv_skips_negative_numbers_and_values():
    argv = ["--seed", "-1", "--temp", "-0.5", "--cfg", '{"a":1}', "-cc.mode=3", "-tp", "2"]
    assert h.flags_in_argv(argv) == ["--seed", "--temp", "--cfg", "-cc", "-tp"]


def test_missing_flags_reports_unknown_in_order():
    argv = ["vllm", "serve", "/m", "--disable-log-requests", "--seed", "0", "-O2", "--bogus", "--disable-log-stats"]
    assert h.missing_flags(argv, SERVE) == ["--disable-log-requests", "--bogus"]


def test_every_generated_serve_argv_is_known_to_real_vllm_serve():
    for cfg in e.all_configs():
        assert h.missing_flags(cfg.serve_argv("/m", 8000), SERVE) == [], cfg.name + "/" + cfg.arm


def test_every_generated_offline_argv_is_known_to_real_bench_latency():
    for cfg in e.all_configs():
        if cfg.dp == 1:
            assert h.missing_flags(cfg.offline_args("/m"), LAT) == [], cfg.name + "/" + cfg.arm


def test_every_generated_bench_latency_argv_is_known_to_real_bench_latency():
    for cfg in e.all_configs():
        if cfg.dp == 1:
            argv = cfg.bench_latency_argv("/m", 8, 1024, 64, 5, 20, "/r/x.json")
            assert h.missing_flags(argv, LAT) == [], cfg.name + "/" + cfg.arm
