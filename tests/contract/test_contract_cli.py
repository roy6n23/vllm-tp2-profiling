"""Contract tests: tpprof's argv, parsers and the fakes against the REAL vLLM 0.30.0 CLI (spec 8, S2).

They run only inside docker/contract.Dockerfile (`make contract`), where the real `vllm` is on PATH
(CPU build, VLLM_TARGET_DEVICE=cpu) and the Llama-3.1 tokenizer files are in /opt/tok. The default
pytest options exclude the `contract` marker.
"""
from __future__ import annotations

import difflib
import inspect
import os
import re
import subprocess
import sys

import pytest

from tests.conftest import FAKE_BIN, FIXTURES
from tpprof import engine, helpflags, procs, results
from tpprof.client import ClientSpec, client_argv, run_clients
from tpprof.offline import PROFILER_CONFIG
from tpprof.server import start_server

pytestmark = [
    pytest.mark.contract,
    # Raised by torch while vllm imports; not ours to fix.
    pytest.mark.filterwarnings("ignore:`torch.jit.script_method` is deprecated:DeprecationWarning"),
]

TOKENIZER_DIR = "/opt/tok"
FAKE_VLLM = str(FAKE_BIN / "vllm")
FAKE_PORT = 18000
HELP_TIMEOUT_S = 300
# Log lines around the help text: vLLM's `%(levelname)s %(asctime)s` prefix (C6) and torch's glog-style
# `W0930 13:13:44.534000` prefix. Their timestamps differ on every run.
_LOG_LINE = re.compile(r"^((DEBUG|INFO|WARNING|ERROR|CRITICAL) \d\d-\d\d |[DIWEF]\d{4} )\d\d:\d\d:\d\d")

HELP_FIXTURES = {
    ("serve",): "serve_help_all.txt",
    ("bench", "latency"): "bench_latency_help.txt",
    ("bench", "serve"): "bench_serve_help.txt",
}
CONFIGS = engine.all_configs()
CONFIG_IDS = [f"{c.name}-{c.arm}" for c in CONFIGS]
OFFLINE_CONFIGS = [c for c in CONFIGS if c.dp == 1]
OFFLINE_IDS = [f"{c.name}-{c.arm}" for c in OFFLINE_CONFIGS]
_help_cache: dict[tuple[str, ...], str] = {}


def live_help(sub: tuple[str, ...]) -> str:
    """`vllm <sub> --help=all` from the real CLI, stdout and stderr merged like the fixture captures."""
    if sub not in _help_cache:
        p = subprocess.run(["vllm", *sub, "--help=all"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=HELP_TIMEOUT_S)
        assert p.returncode == 0, f"vllm {' '.join(sub)} --help=all exited {p.returncode}:\n{p.stdout[-4000:]}"
        _help_cache[sub] = p.stdout
    return _help_cache[sub]


def sample_client_spec(result_dir: str) -> ClientSpec:
    return ClientSpec(base_url=f"http://127.0.0.1:{FAKE_PORT}", tokenizer=TOKENIZER_DIR, input_len=64,
                      output_len=16, prefix_len=0, num_prompts=8, request_rate=4.0, seed=0,
                      result_dir=result_dir, result_filename="contract.json",
                      request_id_prefix="contract-client-", metadata=(("tpprof_config", "TP1"),))


def parse_engine_args(args: list[str]):
    """vLLM's own EngineArgs parse, exactly as tpprof.offline.VllmEngine does it."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    return EngineArgs.from_cli_args(parser.parse_args(args))


def enum_name(value: object) -> object:
    """An enum member's name, or the value itself when vLLM keeps the string."""
    return getattr(value, "name", value)


# ---------------------------------------------------------------- 1. flags known to the real CLI


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_serve_argv_flags_known_to_real_serve(cfg):
    argv = cfg.serve_argv(TOKENIZER_DIR, FAKE_PORT)
    assert helpflags.missing_flags(argv, live_help(("serve",))) == []


@pytest.mark.parametrize("cfg", OFFLINE_CONFIGS, ids=OFFLINE_IDS)
def test_offline_argv_flags_known_to_real_bench_latency(cfg):
    help_text = live_help(("bench", "latency"))
    assert helpflags.missing_flags(cfg.offline_args(TOKENIZER_DIR), help_text) == []
    argv = cfg.bench_latency_argv(TOKENIZER_DIR, batch=8, input_len=1024, output_len=64, warmup=3, iters=10,
                                  output_json="/tmp/lat.json")
    assert helpflags.missing_flags(argv, help_text) == []


def test_client_argv_flags_known_to_real_bench_serve(tmp_path):
    argv = client_argv(sample_client_spec(str(tmp_path)))
    assert helpflags.missing_flags(argv, live_help(("bench", "serve"))) == []


@pytest.mark.parametrize("sub", list(HELP_FIXTURES), ids=["-".join(s) for s in HELP_FIXTURES])
def test_help_fixtures_match_real_cli(sub):
    """The fixtures (which the fake `vllm --help=all` prints) list exactly the real CLI's flags.

    Text differences outside vLLM's log lines are printed and the fixture is rewritten with the
    live text; only the flag sets are asserted.
    """
    path = FIXTURES / HELP_FIXTURES[sub]
    live, fixture = live_help(sub), path.read_text()

    def body(text: str) -> list[str]:
        return [line for line in text.splitlines(keepends=True) if not _LOG_LINE.match(line)]

    if body(live) != body(fixture):
        sys.stdout.writelines(difflib.unified_diff(body(fixture), body(live), f"fixture/{path.name}",
                                                   f"live/vllm {' '.join(sub)} --help=all"))
        path.write_text(live)
    live_flags, fixture_flags = helpflags.parse_help_flags(live), helpflags.parse_help_flags(fixture)
    assert live_flags == fixture_flags, (f"only in the real CLI: {sorted(live_flags - fixture_flags)}; "
                                         f"only in the fixture: {sorted(fixture_flags - live_flags)}")


# ---------------------------------------------------------------- 2. EngineArgs parses the offline argv


def expected_cudagraph_mode(cfg) -> str | None:
    if cfg.arm == "G2":
        return None                # no --compilation-config: --enforce-eager decides
    return "NONE" if cfg.arm == "G1" else "FULL_AND_PIECEWISE"


def expected_fuse_allreduce_rms(cfg) -> bool | None:
    if cfg.name != "TP2" or cfg.arm == "G2":
        return None                # no pass_config given
    return cfg.arm not in ("AR1", "AR2", "AR3")


@pytest.mark.parametrize("cfg", OFFLINE_CONFIGS, ids=OFFLINE_IDS)
def test_engine_args_parse_offline_argv(cfg):
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    ea = parse_engine_args(cfg.offline_args(TOKENIZER_DIR))
    assert ea.model == TOKENIZER_DIR
    assert ea.max_num_seqs == 1024
    assert ea.max_num_batched_tokens == 8192
    assert ea.enable_prefix_caching is (cfg.arm == "PCon")
    cc = ea.compilation_config
    mode = cc.cudagraph_mode
    assert (None if mode is None else enum_name(mode)) == expected_cudagraph_mode(cfg)
    assert cc.pass_config.fuse_allreduce_rms is expected_fuse_allreduce_rms(cfg)
    assert ea.enforce_eager is (cfg.arm == "G2")
    # EngineArgs keeps the CLI string; vLLM resolves it with this validator when it builds the config.
    assert AttentionConfig.validate_backend_before(ea.attention_backend) is AttentionBackendEnum.FLASH_ATTN
    assert ea.attention_config.flash_attn_version == 3
    assert ea.generation_config == "vllm"


# ---------------------------------------------------------------- 3. Python API used by tpprof.offline


def test_llm_api_signatures():
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    assert vllm.__version__ == "0.30.0"
    assert "use_tqdm" in inspect.signature(LLM.generate).parameters
    for name in ("from_engine_args", "start_profile", "stop_profile"):
        assert callable(getattr(LLM, name, None)), f"LLM.{name} is missing"
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=4, detokenize=True)
    assert (sp.n, sp.max_tokens, sp.ignore_eos, sp.detokenize) == (1, 4, True, True)
    prompt = TokensPrompt(prompt_token_ids=[1, 2])
    assert prompt["prompt_token_ids"] == [1, 2]


# ---------------------------------------------------------------- 4. real client against the fake server


def test_real_client_against_fake_server(tmp_path):
    cfg = engine.arm_config("TP1", "base")
    base_env = {**os.environ, "FAKE_VLLM_TOKENIZER": TOKENIZER_DIR}
    server = start_server(cfg, TOKENIZER_DIR, FAKE_PORT, str(tmp_path), base_env, "contract-server",
                          startup_timeout_s=120, vllm_bin=FAKE_VLLM)
    try:
        spec = sample_client_spec(str(tmp_path / "client"))
        [outcome] = run_clients([spec], os.environ, str(tmp_path), "contract-client", timeout_s=300,
                                vllm_bin="vllm")
    finally:
        procs.stop(server.proc)
    log = procs.tail(outcome.log_path, 60)
    assert not outcome.timed_out and outcome.exit_code == 0, f"real vllm bench serve failed:\n{log}"
    r = results.load_serve_result(spec.result_path)
    assert results.validate_serve(r, expect_in=64, expect_out=16) == [], log
    assert r.metadata == {"tpprof_config": "TP1"}
    assert r.request_rate == 4.0


# ---------------------------------------------------------------- 5. profiler config


def test_profiler_config_parses():
    cfg = engine.arm_config("TP2", "base")
    args = [*cfg.offline_args(TOKENIZER_DIR), "--profiler-config", PROFILER_CONFIG]
    assert PROFILER_CONFIG == '{"profiler":"cuda"}'
    assert parse_engine_args(args).profiler_config.profiler == "cuda"
