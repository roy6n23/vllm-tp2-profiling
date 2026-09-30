from __future__ import annotations

import re

import pytest

from tests.conftest import FIXTURES
from tpprof.logparse import (
    FALLBACK_WARNINGS,
    HARD_FAILURES,
    Expectation,
    check,
    expectation_for,
    parse_engine_log,
)

LOGS = FIXTURES / "logs"


def read(name: str) -> str:
    return (LOGS / name).read_text()


def parse(name: str):
    return parse_engine_log(read(name))


def drop_lines(text: str, needle: str) -> str:
    kept = [ln for ln in text.split("\n") if needle not in ln]
    assert len(kept) < len(text.split("\n")), f"fixture has no line containing {needle!r}"
    return "\n".join(kept)


# --- parsing -----------------------------------------------------------------

def test_tp2_mnnvl_fields():
    eff = parse("tp2_base_mnnvl.log")
    assert eff.vllm_version == "0.30.0"
    assert "tensor_parallel_size=2" in eff.engine_config_banner
    assert eff.kv_cache_tokens == [955000]
    assert eff.max_concurrency == [103.62]
    assert eff.available_kv_gib == [58.29]
    assert eff.model_loading_gib == [7.51, 7.51]
    assert eff.chunked_prefill_tokens == 8192
    assert eff.v2_model_runner is True
    assert eff.attention_backend == "FLASH_ATTN"
    assert eff.attention_explicit is True
    assert eff.flash_attn_version == 3
    assert eff.ar_backends == ["FLASHINFER", "CUSTOM", "SYMM_MEM", "PYNCCL"]
    assert eff.fusions_line == "allreduce_rms"
    assert eff.fi_backend == "mnnvl"
    assert eff.fi_backend_fallback is False
    assert eff.nccl_version == "2.30.7"
    assert eff.executor == "mp"
    assert eff.enforce_eager is False
    assert eff.sampling_override is False
    assert eff.mrv2_fallback is False
    assert eff.jit_after_warmup == []
    assert eff.hard_failures == []
    assert eff.startup_complete is True
    assert eff.graph_capture == ["Graph capturing finished in 3 secs, took 0.61 GiB"] * 2 + [
        "Graph capturing finished in 14 secs, took 0.61 GiB"] * 2


def test_tp1_fields_and_glued_tqdm_line():
    eff = parse("tp1_base.log")
    assert eff.kv_cache_tokens == [420959]
    assert eff.max_concurrency == [45.68]
    assert eff.ar_backends is None
    assert eff.fi_backend is None
    assert eff.nccl_version is None
    assert eff.executor == "mp"
    # The second capture line sits after a tqdm bar and a bare '\r' (D5-6); it must still be found.
    assert eff.graph_capture == ["Graph capturing finished in 3 secs, took 0.52 GiB",
                                 "Graph capturing finished in 18 secs, took 0.52 GiB"]


def test_line_glued_after_tqdm_without_newline():
    # Real shape from D5-6: another process's line glued onto a tqdm bar on one physical line.
    glued = ("Loading safetensors checkpoint shards:   0% Completed | 0/7 [00:00<?, ?it/s](Worker_TP1 pid=4302) "
             "WARNING 09-30 10:00:09 [symm_mem.py:100] SymmMemCommunicator: symmetric memory initialization "
             "failed: boom\n")
    eff = parse_engine_log(glued)
    assert eff.hard_failures == ["SymmMemCommunicator: symmetric memory initialization failed: boom"]


def test_process_prefixes_do_not_change_the_parse():
    # C6 describes TP1 lines under (EngineCore pid=...); with the AM2 `mp` pin they come from a plain `Worker`
    # (multiproc_executor.py:1062-1095). Parsing is on the message only, so both shapes give the same result.
    text = read("tp1_base.log")
    as_engine_core = text.replace("(Worker pid=4301)", "(EngineCore pid=4230)")
    assert as_engine_core != text
    assert parse_engine_log(as_engine_core) == parse_engine_log(text)


def test_exec_uni_fixture_fields():
    eff = parse("tp1_exec_uni.log")
    assert eff.executor == "uni"
    assert eff.kv_cache_tokens == [420959]
    assert eff.graph_capture == ["Graph capturing finished in 3 secs, took 0.52 GiB",
                                 "Graph capturing finished in 18 secs, took 0.52 GiB"]


def test_dp2_has_one_kv_line_per_engine():
    eff = parse("dp2_base.log")
    assert eff.kv_cache_tokens == [420959, 421103]
    assert eff.max_concurrency == [45.68, 45.69]
    assert eff.available_kv_gib == [51.39, 51.39]
    assert "data_parallel_size=2" in eff.engine_config_banner


def test_fallback_log_is_recorded_not_failing():
    eff = parse("tp2_base_trtllm_fallback.log")
    assert eff.fi_backend == "trtllm"
    assert eff.fi_backend_fallback is True
    assert eff.hard_failures == []
    assert check(eff, expectation_for("TP2", "base", True)) == []


def test_hard_fail_log_is_rejected():
    eff = parse("tp2_hard_fail.log")
    assert eff.hard_failures == [
        "Failed to initialize FlashInfer Allreduce norm fusion workspace with backend=mnnvl"]
    violations = check(eff, expectation_for("TP2", "base", True))
    assert any("Failed to initialize FlashInfer Allreduce norm fusion workspace with backend=mnnvl" in v
               for v in violations)
    assert any("Initialized FlashInfer Allreduce norm fusion workspace with backend=" in v for v in violations)


def test_g2_fields():
    eff = parse("tp2_g2.log")
    assert eff.enforce_eager is True
    assert eff.graph_capture == []


def test_offline_log_without_prefixes_parses():
    # Offline LLM prints main-process lines without a '(Name pid=N) ' prefix (D5-7).
    text = re.sub(r"^\(\w+ pid=\d+\) ", "", read("tp1_base.log"), flags=re.M)
    text = drop_lines(text, "INFO:     ")
    eff = parse_engine_log(text)
    assert eff.kv_cache_tokens == [420959]
    assert eff.startup_complete is False
    assert check(eff, expectation_for("TP1", "base", False)) == []
    assert any("Application startup complete." in v for v in check(eff, expectation_for("TP1", "base", True)))


def test_executor_from_banner_takes_precedence():
    text = read("tp1_base.log").replace("tensor_parallel_size=1,",
                                        "tensor_parallel_size=1, distributed_executor_backend=uni,")
    assert parse_engine_log(text).executor == "uni"


def test_jit_after_warmup_is_recorded():
    jit = ("(Worker_TP0 pid=4301) WARNING 09-30 10:05:00 [jit_monitor.py:141] Triton JIT compile during "
           "inference: _fwd_kernel (BLOCK_M=64). This causes a latency spike; consider extending warmup "
           "to cover this shape/config.")
    eff = parse_engine_log(read("tp2_base_mnnvl.log") + jit + "\n")
    assert len(eff.jit_after_warmup) == 1
    assert "_fwd_kernel" in eff.jit_after_warmup[0]


@pytest.mark.parametrize("msg", [
    "(EngineCore pid=4230) ERROR 09-30 10:00:09 [core.py:1366] EngineCore failed to start.",
    "(APIServer pid=4101) RuntimeError: Engine core initialization failed. See root cause above. "
    "Failed core proc(s): {'EngineCore': 1}",
    "(EngineCore pid=4230) ERROR 09-30 10:00:09 [multiproc_executor.py:315] Worker proc VllmWorker-1 died "
    "unexpectedly (exit code: -9), shutting down executor.",
    "(APIServer pid=4101) TimeoutError: Timed out waiting for engine core processes to start. This is often "
    "caused by slow weight loading.",
    "(Worker pid=4301) WARNING 09-30 10:00:09 [symm_mem.py:100] SymmMemCommunicator: symmetric memory "
    "initialization failed: CUDA error Communicator is not available. To suppress this warning set "
    "VLLM_ALLREDUCE_USE_SYMM_MEM=0",
    "(Worker pid=4301) WARNING 09-30 10:00:09 [allreduce_rms_fusion.py:1136] AllReduce fusion pass is disabled.",
    "(Worker pid=4301) ERROR 09-30 10:00:09 [shm_broadcast.py:244] Insufficient space in /dev/shm: 64 MiB free",
])
def test_hard_failure_strings_are_caught(msg):
    eff = parse_engine_log(read("tp2_base_mnnvl.log") + msg + "\n")
    assert len(eff.hard_failures) == 1
    violations = check(eff, expectation_for("TP2", "base", True))
    assert len(violations) == 1 and "hard failure" in violations[0]


def test_failure_string_lists_match_spec_am4():
    assert len(HARD_FAILURES) == len(set(HARD_FAILURES))
    for s in ("Failed to initialize FlashInfer Allreduce norm fusion workspace with backend=",
              "Flashinfer is not installed or comm module not found, skipping allreduce fusion pass",
              "Failed to initialize Flashinfer allreduce workspace. Flashinfer allreduce-norm fusion will be disabled.",
              "AllReduce fusion pass is disabled.",
              "Custom allreduce is disabled because",
              "Custom allreduce is disabled due to an unsupported world size",
              "SymmMemCommunicator: symmetric memory initialization failed",
              "Insufficient space in /dev/shm"):
        assert s in HARD_FAILURES
    assert FALLBACK_WARNINGS == ("FlashInfer MNNVL multicast is unavailable on the current topology",
                                 "Failed to initialize FlashInfer All Reduce workspace:",
                                 "FlashInfer mnnvl allreduce workspace unavailable")
    # The tp_size <= 1 notice is not the hard 'AllReduce fusion pass is disabled.' line.
    eff = parse_engine_log("WARNING 09-30 10:00:00 [allreduce_rms_fusion.py:996] "
                           "AllReduce fusion pass is disabled for tp_size <= 1.\n")
    assert eff.hard_failures == []


# --- expectations and check --------------------------------------------------

@pytest.mark.parametrize("log, config, arm", [
    ("tp1_base.log", "TP1", "base"),
    ("tp1_base.log", "DP2rand0", "base"),
    ("tp1_base.log", "DP2rand1", "base"),
    ("tp1_base.log", "TP1", "G1"),
    ("tp1_exec_uni.log", "TP1", "EXECuni"),
    ("tp2_base_mnnvl.log", "TP2", "base"),
    ("tp2_base_mnnvl.log", "TP2", "G1"),
    ("tp2_base_mnnvl.log", "TP2", "PCon"),
    ("tp2_base_mnnvl.log", "TP2", "API2"),
    ("tp2_base_trtllm_fallback.log", "TP2", "FIBtrtllm"),
    ("tp2_ar1.log", "TP2", "AR1"),
    ("tp2_ar2.log", "TP2", "AR2"),
    ("tp2_ar3.log", "TP2", "AR3"),
    ("tp2_g2.log", "TP2", "G2"),
    ("dp2_base.log", "DP2", "base"),
    ("dp2_base.log", "DP2", "API2"),
])
def test_fixture_passes_its_expectation(log, config, arm):
    assert check(parse(log), expectation_for(config, arm, True)) == []


def test_ar2_passes_ar2_and_fails_base():
    eff = parse("tp2_ar2.log")
    assert check(eff, expectation_for("TP2", "AR2", True)) == []
    violations = check(eff, expectation_for("TP2", "base", True))
    assert any("FLASHINFER" in v and "['CUSTOM', 'PYNCCL']" in v for v in violations)
    assert any("Initialized FlashInfer Allreduce norm fusion workspace with backend=" in v for v in violations)


def test_ar3_fails_ar2_and_ar2_fails_ar3():
    ar3 = check(parse("tp2_ar3.log"), expectation_for("TP2", "AR2", True))
    assert len(ar3) == 1 and "['CUSTOM', 'PYNCCL']" in ar3[0] and "['PYNCCL']" in ar3[0]
    ar2 = check(parse("tp2_ar2.log"), expectation_for("TP2", "AR3", True))
    assert len(ar2) == 1 and "['PYNCCL']" in ar2[0]


def test_g2_requires_enforce_eager_line():
    violations = check(parse("tp2_base_mnnvl.log"), expectation_for("TP2", "G2", True))
    assert violations == ['missing line "Enforce eager set, disabling torch.compile and CUDAGraphs" '
                          "(required by --enforce-eager)."]
    assert check(parse("tp1_base.log"), expectation_for("TP1", "G2", True)) == violations


def test_fibtrtllm_rejects_mnnvl():
    violations = check(parse("tp2_base_mnnvl.log"), expectation_for("TP2", "FIBtrtllm", True))
    assert len(violations) == 1
    assert "Initialized FlashInfer Allreduce norm fusion workspace with backend=trtllm" in violations[0]
    assert "mnnvl" in violations[0]


def test_dp2_log_fails_single_engine_expectation_and_vice_versa():
    one = check(parse("dp2_base.log"), expectation_for("TP1", "base", True))
    # DP2 passes no executor flag, so its log records no executor value, and a missing value is not a violation.
    assert parse("dp2_base.log").executor is None
    assert len(one) == 1 and "GPU KV cache size:" in one[0] and "expected 1" in one[0]
    two = check(parse("tp1_base.log"), expectation_for("DP2", "base", True))
    assert len(two) == 1 and "expected 2" in two[0]


def test_tp1_log_fails_tp2_expectation():
    violations = check(parse("tp1_base.log"), expectation_for("TP2", "base", True))
    assert any("all-reduce backends (in dispatch order) for group 'tp:0'" in v for v in violations)
    assert any("Initialized FlashInfer Allreduce norm fusion workspace with backend=" in v for v in violations)


@pytest.mark.parametrize("log, needle, named", [
    ("tp1_base.log", "Using V2 Model Runner", "Using V2 Model Runner"),
    ("tp1_base.log", "Initializing a V1 LLM engine", "Initializing a V1 LLM engine (v0.30.0)"),
    ("tp1_base.log", "Chunked prefill is enabled", "Chunked prefill is enabled with max_num_batched_tokens=8192."),
    ("tp1_base.log", "Using AttentionBackendEnum", "Using AttentionBackendEnum.FLASH_ATTN backend."),
    ("tp1_base.log", "Using FlashAttention version", "Using FlashAttention version 3"),
    ("tp1_base.log", "KV cache size:", "GPU KV cache size:"),
    ("tp1_base.log", "Application startup complete.", "Application startup complete."),
    ("tp2_base_mnnvl.log", "all-reduce backends", "all-reduce backends (in dispatch order) for group 'tp:0'"),
    ("tp2_base_mnnvl.log", "Initialized FlashInfer Allreduce norm fusion workspace",
     "Initialized FlashInfer Allreduce norm fusion workspace with backend="),
])
def test_missing_required_line_is_named(log, needle, named):
    config = "TP2" if log.startswith("tp2") else "TP1"
    eff = parse_engine_log(drop_lines(read(log), needle))
    violations = check(eff, expectation_for(config, "base", True))
    assert len(violations) == 1
    assert named in violations[0]


def test_wrong_values_are_named():
    text = (read("tp1_base.log")
            .replace("(v0.30.0)", "(v0.29.1)")
            .replace("max_num_batched_tokens=8192.", "max_num_batched_tokens=2048.")
            .replace("Using AttentionBackendEnum.FLASH_ATTN backend.", "Using AttentionBackendEnum.FLASHINFER backend.")
            .replace("Using FlashAttention version 3", "Using FlashAttention version 2"))
    violations = check(parse_engine_log(text), expectation_for("TP1", "base", True))
    assert len(violations) == 4
    joined = "\n".join(violations)
    for found in ("0.29.1", "2048", "FLASHINFER", "found version 2"):
        assert found in joined


AUTO_ATTENTION = "Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER']."


def test_auto_selected_attention_form_is_parsed_but_rejected():
    # FLASH_ATTN is also the SM90 auto default (D5-12), so the auto form means --attention-backend was dropped.
    text = read("tp1_base.log").replace("Using AttentionBackendEnum.FLASH_ATTN backend.", AUTO_ATTENTION)
    eff = parse_engine_log(text)
    assert eff.attention_backend == "FLASH_ATTN"
    assert eff.attention_explicit is False
    violations = check(eff, expectation_for("TP1", "base", True))
    assert len(violations) == 1
    assert 'missing line "Using AttentionBackendEnum.FLASH_ATTN backend."' in violations[0]
    assert "auto-selection" in violations[0] and "--attention-backend was not applied" in violations[0]


def test_auto_attention_line_next_to_explicit_one_is_rejected():
    worker = "(Worker pid=4301) INFO 09-30 10:00:07 [cuda.py:539] "
    eff = parse_engine_log(read("tp1_base.log") + worker + AUTO_ATTENTION + "\n")
    assert eff.attention_explicit is False
    violations = check(eff, expectation_for("TP1", "base", True))
    assert len(violations) == 1 and "Using AttentionBackendEnum.FLASH_ATTN backend." in violations[0]


def test_exec_uni_requires_uni_and_base_requires_mp():
    uni = check(parse("tp1_base.log"), expectation_for("TP1", "EXECuni", True))
    assert uni == ["expected \"'distributed_executor_backend': 'uni'\" in the \"non-default args:\" line "
                   "(found executor mp)."]
    mp = check(parse("tp1_exec_uni.log"), expectation_for("TP1", "base", True))
    assert len(mp) == 1 and "'distributed_executor_backend': 'mp'" in mp[0] and "found executor uni" in mp[0]
    for config in ("DP2rand0", "DP2rand1"):
        assert len(check(parse("tp1_exec_uni.log"), expectation_for(config, "base", True))) == 1


NONDEFAULT_HEADER = "(APIServer pid=4101) INFO 09-30 10:00:00 [api_utils.py:286] non-default args: "


def with_nondefault_line(name: str, args: str) -> str:
    lines = read(name).split("\n")
    assert "non-default args: " in lines[0]
    return "\n".join([NONDEFAULT_HEADER + args] + lines[1:])


@pytest.mark.parametrize("args", ["{...}", "{'model_tag': '/models/llama'}"])
@pytest.mark.parametrize("name,config,arm", [
    ("tp1_base.log", "TP1", "base"), ("tp1_base.log", "DP2rand0", "base"), ("tp1_base.log", "DP2rand1", "base"),
    ("tp1_exec_uni.log", "TP1", "EXECuni"), ("tp2_base_mnnvl.log", "TP2", "base"), ("tp2_ar2.log", "TP2", "AR2"),
])
def test_log_without_an_executor_value_records_none_and_passes(name, config, arm, args):
    # C6 shows the line as "non-default args: {...}", and neither C6's banner nor the real 0.30.0 banner has an
    # executor field, so a log that follows C6 may carry no executor value at all. It is recorded as None and
    # is not a violation (AM2: logparse records the executor). Only a recorded value that differs is flagged.
    eff = parse_engine_log(with_nondefault_line(name, args))
    assert eff.executor is None
    assert check(eff, expectation_for(config, arm, True)) == []


# contracts.md C6, verbatim: the messages "as the fake emits them", with C6's process prefixes (TP1-like and DP2
# lines under EngineCore / EngineCore_DP<i>, TP2 worker lines under Worker_TP0, uvicorn under APIServer).
C6_AR_TAIL = (" all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: "
              "['FLASHINFER_PCIE_IPC', 'FLASHINFER', 'NCCL_SYMM_MEM', 'QUICK_REDUCE', 'AITER_CUSTOM', 'CUSTOM', "
              "'SYMM_MEM', 'PYNCCL'].")
C6_AR_DEFAULT = "['FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL']"   # C6's list; AR2 and AR3 change it (AM3)
C6_AR_LIST = {"AR2": "['CUSTOM', 'PYNCCL']", "AR3": "['PYNCCL']"}
C6_PAIRS = ([(c, "base") for c in ("TP1", "DP2rand0", "DP2rand1", "TP2", "DP2")]
            + [("TP2", a) for a in ("AR1", "AR2", "AR3", "G1", "G2", "PCon", "FIBtrtllm", "API2")]
            + [("TP1", "G1"), ("TP1", "G2"), ("TP1", "EXECuni"), ("DP2", "API2")])


def c6_log(config: str, arm: str) -> str:
    tp2, dp = config == "TP2", 2 if config == "DP2" else 1
    lines = [("APIServer", "non-default args: {...}")]
    for eng in ([f"EngineCore_DP{i}" for i in range(dp)] if dp > 1 else ["EngineCore"]):
        w = "Worker_TP0" if tp2 else eng
        lines += [(eng, "Initializing a V1 LLM engine (v0.30.0) with config: model='/models/llama', ..., "
                        f"tensor_parallel_size={2 if tp2 else 1}, data_parallel_size={dp}, "
                        f"disable_custom_all_reduce={arm == 'AR3'}, enforce_eager={arm == 'G2'}, ..."),
                  (eng, "Chunked prefill is enabled with max_num_batched_tokens=8192."),
                  (w, "Using V2 Model Runner"),
                  (w, "Using AttentionBackendEnum.FLASH_ATTN backend."),
                  (w, "Using FlashAttention version 3")]
        if tp2:
            lines += [(w, "vLLM is using nccl==2.30.7"),
                      (w, "Using " + C6_AR_LIST.get(arm, C6_AR_DEFAULT) + C6_AR_TAIL)]
            if arm in ("base", "G1"):
                lines.append((w, "Enabled custom fusions: allreduce_rms"))
            if arm not in ("AR2", "AR3", "G2"):
                lines.append((w, "Initialized FlashInfer Allreduce norm fusion workspace with backend="
                                 + ("trtllm" if arm == "FIBtrtllm" else "mnnvl")))
        lines += [(w, "Model loading took 14.99 GiB memory and 12.345678 seconds"),
                  (w, "Available KV cache memory: 51.39 GiB"),
                  (eng, "GPU KV cache size: 420,959 tokens, Maximum concurrency for 9,216 tokens per request: 45.68x")]
        lines.append((eng, "Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting "
                           "-cc.mode=none -cc.cudagraph_mode=none") if arm == "G2"
                     else (w, "Graph capturing finished in 18 secs, took 0.52 GiB"))
    pids = {proc: 4101 + i for i, proc in enumerate(dict.fromkeys(proc for proc, _ in lines))}
    out = [f"({proc} pid={pids[proc]}) INFO 09-30 10:00:00 [fake.py:1] {msg}" for proc, msg in lines]
    return "\n".join(out + ["(APIServer pid=4101) INFO:     Application startup complete."]) + "\n"


@pytest.mark.parametrize("config, arm", C6_PAIRS)
def test_c6_literal_log_passes_its_expectation(config, arm):
    # The review finding: a fake that prints C6 literally carries no executor value and must still pass check().
    eff = parse_engine_log(c6_log(config, arm))
    assert eff.executor is None
    assert check(eff, expectation_for(config, arm, True)) == []


@pytest.mark.parametrize("args", [
    "{'model_tag': '/m', 'tensor_parallel_size': 2, 'distributed_executor_backend': 'mp'}",   # dict repr (real)
    '{"model_tag": "/m", "tensor_parallel_size": 2, "distributed_executor_backend": "mp"}',   # JSON rendering
])
def test_executor_rendered_from_argv_in_either_quote_style(args):
    eff = parse_engine_log(with_nondefault_line("tp2_base_mnnvl.log", args))
    assert eff.executor == "mp"
    assert check(eff, expectation_for("TP2", "base", True)) == []


def test_sampling_override_is_a_violation():
    eff = parse("tp1_sampling_override.log")
    assert eff.sampling_override is True
    violations = check(eff, expectation_for("TP1", "base", True))
    assert len(violations) == 1
    assert "Default vLLM sampling parameters have been overridden" in violations[0]


def test_mrv2_fallback_is_a_violation():
    line = ("(APIServer pid=4101) WARNING 09-30 10:00:03 [vllm.py:717] Model Runner V2 does not yet support "
            "prompt_logprobs; using the V1 model runner instead.")
    eff = parse_engine_log(line + "\n" + read("tp1_base.log"))
    assert eff.mrv2_fallback is True
    violations = check(eff, expectation_for("TP1", "base", True))
    assert len(violations) == 1 and "Model Runner V2 does not yet support" in violations[0]


def test_empty_log_names_every_always_required_line():
    violations = check(parse_engine_log(""), expectation_for("TP2", "base", True))
    joined = "\n".join(violations)
    for named in ("Initializing a V1 LLM engine (v0.30.0)", "Using V2 Model Runner",
                  "Using AttentionBackendEnum.FLASH_ATTN backend.", "Using FlashAttention version 3",
                  "Chunked prefill is enabled with max_num_batched_tokens=8192.", "GPU KV cache size:",
                  "Application startup complete.", "for group 'tp:0'",
                  "Initialized FlashInfer Allreduce norm fusion workspace with backend="):
        assert named in joined


def test_expectation_for_configs():
    assert expectation_for("TP1", "base", True) == Expectation(engines=1, serve=True, tp2=False, executor="mp")
    assert expectation_for("TP1", "EXECuni", False) == Expectation(engines=1, serve=False, tp2=False, executor="uni")
    for rand in ("DP2rand0", "DP2rand1"):
        assert expectation_for(rand, "base", False) == Expectation(engines=1, serve=False, tp2=False, executor="mp")
    # DP2 passes no --distributed-executor-backend (C1), so there is no executor value in its log to check.
    assert expectation_for("DP2", "base", True) == Expectation(engines=2, serve=True, tp2=False)
    base = Expectation(engines=1, serve=True, tp2=True, ar_first="FLASHINFER", require_fi_workspace=True,
                       executor="mp")
    assert expectation_for("TP2", "base", True) == base
    assert expectation_for("TP2", "AR2", True).ar_exact == (("CUSTOM", "PYNCCL"),)
    assert expectation_for("TP2", "AR3", True).ar_exact == (("PYNCCL",),)
    g2 = expectation_for("TP2", "G2", True)
    assert (g2.ar_first, g2.require_fi_workspace, g2.require_enforce_eager) == ("FLASHINFER", False, True)
    assert expectation_for("TP2", "FIBtrtllm", True).fi_backend_exact == "trtllm"


@pytest.mark.parametrize("config, arm", [("TP3", "base"), ("TP1", "AR9"), ("TP1", "AR2"), ("DP2", "G2"),
                                         ("TP2", "EXECuni"), ("DP2rand0", "G1"),
                                         ("TP2", "AR0")])
def test_expectation_for_rejects_unknown_or_inapplicable(config, arm):
    with pytest.raises(ValueError, match=re.escape(repr(config if config == "TP3" else arm))):
        expectation_for(config, arm, True)
