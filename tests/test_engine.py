from __future__ import annotations

import dataclasses
import json

import pytest

from tpprof import engine as e


def test_tp2_serve_argv_exact():
    cfg = e.base_config("TP2")
    argv = cfg.serve_argv("/models/llama", 8000)
    assert argv[:3] == ["vllm", "serve", "/models/llama"]
    joined = " ".join(argv)
    for frag in ["--dtype bfloat16", "--max-model-len 9216", "--gpu-memory-utilization 0.90",
                 "--max-num-seqs 1024", "--max-num-batched-tokens 8192", "--block-size 16",
                 "--kv-cache-dtype auto", "--seed 0", "--no-enable-prefix-caching", "--enable-chunked-prefill",
                 "--async-scheduling", "--stream-interval 1", "--no-enable-dbo", "--no-enable-batch-sharded-sampling",
                 "--performance-mode balanced", "--optimization-level 2", "--attention-backend FLASH_ATTN",
                 '--attention-config {"flash_attn_version":3}', "--generation-config vllm",
                 "--fail-on-environ-validation", "--tensor-parallel-size 2", "--distributed-executor-backend mp",
                 '--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_allreduce_rms":true}}',
                 "--served-model-name llama-3.1-8b-instruct", "--host 127.0.0.1", "--port 8000",
                 "--api-server-count 1", "--disable-uvicorn-access-log", "--no-enable-log-requests"]:
        assert frag in joined, frag
    assert "--model" not in argv


def test_tp2_serve_argv_token_for_token():
    """C1 fixes the order: COMMON_FLAGS, parallelism, --compilation-config, then serve-only flags."""
    argv = e.base_config("TP2").serve_argv("/models/llama", 8000, vllm_bin="/usr/bin/vllm")
    assert argv == [
        "/usr/bin/vllm", "serve", "/models/llama",
        "--dtype", "bfloat16", "--max-model-len", "9216", "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "1024", "--max-num-batched-tokens", "8192", "--block-size", "16",
        "--kv-cache-dtype", "auto", "--seed", "0", "--no-enable-prefix-caching", "--enable-chunked-prefill",
        "--async-scheduling", "--stream-interval", "1", "--no-enable-dbo", "--no-enable-batch-sharded-sampling",
        "--performance-mode", "balanced", "--optimization-level", "2", "--attention-backend", "FLASH_ATTN",
        "--attention-config", '{"flash_attn_version":3}', "--generation-config", "vllm",
        "--fail-on-environ-validation",
        "--tensor-parallel-size", "2", "--distributed-executor-backend", "mp",
        "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_allreduce_rms":true}}',
        "--served-model-name", "llama-3.1-8b-instruct", "--host", "127.0.0.1", "--port", "8000",
        "--api-server-count", "1", "--disable-uvicorn-access-log", "--no-enable-log-requests",
    ]


def test_tp1_and_dp2_parallel_flags_and_gpus():
    tp1, dp2 = e.base_config("TP1"), e.base_config("DP2")
    assert tp1.gpus == (0,) and dp2.gpus == (0, 1)
    assert "--distributed-executor-backend mp" in " ".join(tp1.engine_args())
    assert "--data-parallel-size 2" in " ".join(dp2.engine_args())
    assert "--distributed-executor-backend" not in dp2.engine_args()
    assert '"pass_config"' not in " ".join(tp1.engine_args())


def test_parallel_tail_per_config():
    tails = {
        "TP1": ["--tensor-parallel-size", "1", "--distributed-executor-backend", "mp",
                "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'],
        "DP2": ["--tensor-parallel-size", "1", "--data-parallel-size", "2",
                "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'],
    }
    tails["DP2rand0"] = tails["DP2rand1"] = tails["TP1"]
    for name, tail in tails.items():
        assert e.base_config(name).engine_args()[-len(tail):] == tail, name


def test_gpus_tp_dp_per_config():
    expected = {"TP1": (1, 1, (0,)), "TP2": (2, 1, (0, 1)), "DP2": (1, 2, (0, 1)),
                "DP2rand0": (1, 1, (0,)), "DP2rand1": (1, 1, (1,))}
    for name, (tp, dp, gpus) in expected.items():
        cfg = e.base_config(name)
        assert (cfg.name, cfg.arm, cfg.tp, cfg.dp, cfg.gpus, cfg.api_servers) == (name, "base", tp, dp, gpus, 1)


def test_dp2rand_engine_args_equal_tp1():
    tp1 = e.base_config("TP1").engine_args()
    assert e.base_config("DP2rand0").engine_args() == tp1 == e.base_config("DP2rand1").engine_args()
    assert e.base_config("DP2rand1").environment({})["CUDA_VISIBLE_DEVICES"] == "1"


def test_offline_args_have_no_serve_only_flags():
    args = e.base_config("TP2").offline_args("/m")
    assert args[:2] == ["--model", "/m"]
    for f in ("--port", "--host", "--api-server-count", "--served-model-name", "--disable-uvicorn-access-log",
              "--no-enable-log-requests"):
        assert f not in args


def test_bench_latency_argv():
    cfg = e.base_config("TP2")
    argv = cfg.bench_latency_argv("/m", batch=8, input_len=1024, output_len=64, warmup=5, iters=20,
                                  output_json="/r/x.json", vllm_bin="/bin/vllm")
    assert argv[:3] == ["/bin/vllm", "bench", "latency"]
    assert argv[3:3 + len(cfg.offline_args("/m"))] == cfg.offline_args("/m")
    assert argv[3 + len(cfg.offline_args("/m")):] == [
        "--batch-size", "8", "--input-len", "1024", "--output-len", "64",
        "--num-iters-warmup", "5", "--num-iters", "20", "--output-json", "/r/x.json"]


def test_environment_sets_and_unsets():
    env = e.base_config("TP2").environment({"OPENAI_API_KEY": "x", "VLLM_USE_V2_MODEL_RUNNER": "1", "KEEP": "1"})
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1" and env["KEEP"] == "1"
    assert "OPENAI_API_KEY" not in env and "VLLM_USE_V2_MODEL_RUNNER" not in env
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn" and env["VLLM_ALLREDUCE_USE_FLASHINFER"] == "1"


def test_base_env_and_unset_exact():
    cfg = e.base_config("TP1")
    assert cfg.env == (
        ("HF_HUB_OFFLINE", "1"), ("TOKENIZERS_PARALLELISM", "false"),
        ("VLLM_ALLREDUCE_USE_FLASHINFER", "1"), ("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "0"),
        ("VLLM_ALLREDUCE_USE_SYMM_MEM", "1"), ("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "auto"),
        ("VLLM_LOGGING_LEVEL", "INFO"), ("VLLM_USE_NCCL_SYMM_MEM", "0"), ("VLLM_USE_RUST_BENCH", "0"),
        ("VLLM_USE_RUST_FRONTEND", "0"), ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
        ("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "60"))
    assert cfg.unset_env == ("OPENAI_API_KEY", "SAVE_TO_PYTORCH_BENCHMARK_FORMAT", "VLLM_ATTENTION_BACKEND",
                             "VLLM_USE_V2_MODEL_RUNNER")
    env = cfg.environment({"VLLM_ATTENTION_BACKEND": "X", "SAVE_TO_PYTORCH_BENCHMARK_FORMAT": "1",
                           "CUDA_VISIBLE_DEVICES": "3", "VLLM_LOGGING_LEVEL": "DEBUG"})
    assert env == {**dict(cfg.env), "CUDA_VISIBLE_DEVICES": "0"}


@pytest.mark.parametrize("arm,frag,absent", [
    ("AR1", '"fuse_allreduce_rms":false', None),
    ("AR3", "--disable-custom-all-reduce", None),
    ("G1", '"cudagraph_mode":"NONE"', None),
    ("G2", "--enforce-eager", "--compilation-config"),
    ("PCon", "--enable-prefix-caching", "--no-enable-prefix-caching"),
])
def test_arm_transforms(arm, frag, absent):
    cfg = e.arm_config("TP2", arm)
    joined = " ".join(cfg.engine_args())
    assert frag in joined
    if absent:
        assert absent not in cfg.engine_args()


def test_ar2_env_and_g2_drops_optimization_level():
    ar2 = e.arm_config("TP2", "AR2").environment({})
    assert ar2["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0" and ar2["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0"
    assert "--optimization-level" not in e.arm_config("TP1", "G2").engine_args()


def test_ar_ladder_composes():
    ar1, ar2, ar3 = (e.arm_config("TP2", a) for a in ("AR1", "AR2", "AR3"))
    cc_off = '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_allreduce_rms":false}}'
    assert ar1.engine_args()[-2:] == ["--compilation-config", cc_off]
    assert ar1.env == e.base_config("TP2").env
    assert ar2.engine_args() == ar1.engine_args()
    assert ar3.env == ar2.env
    assert ar3.engine_args()[-3:] == ["--disable-custom-all-reduce", "--compilation-config", cc_off]


def test_g1_keeps_fusion_on_tp2_and_g2_ends_with_enforce_eager():
    assert e.arm_config("TP2", "G1").engine_args()[-1] == \
        '{"cudagraph_mode":"NONE","pass_config":{"fuse_allreduce_rms":true}}'
    assert e.arm_config("TP1", "G1").engine_args()[-1] == '{"cudagraph_mode":"NONE"}'
    g2 = e.arm_config("TP1", "G2").engine_args()
    assert g2[-1] == "--enforce-eager" and "--compilation-config" not in g2


def test_pcon_replaces_in_place():
    base, pcon = e.base_config("TP2").engine_args(), e.arm_config("TP2", "PCon").engine_args()
    i = base.index("--no-enable-prefix-caching")
    assert pcon[i] == "--enable-prefix-caching"
    assert pcon[:i] + pcon[i + 1:] == base[:i] + base[i + 1:]


def test_execuni_and_fibtrtllm():
    uni = e.arm_config("TP1", "EXECuni").engine_args()
    assert uni[uni.index("--distributed-executor-backend") + 1] == "uni"
    assert len(uni) == len(e.base_config("TP1").engine_args())
    fib = e.arm_config("TP2", "FIBtrtllm")
    assert fib.environment({})["VLLM_FLASHINFER_ALLREDUCE_BACKEND"] == "trtllm"
    assert fib.engine_args() == e.base_config("TP2").engine_args()


def test_only_single_api_server_configs_expose_gauges():
    assert e.base_config("TP2").exposes_gauges
    assert not e.arm_config("TP2", "API2").exposes_gauges
    assert not e.arm_config("DP2", "API2").exposes_gauges


def test_api2_only_changes_serve_count():
    base, api2 = e.base_config("DP2"), e.arm_config("DP2", "API2")
    assert base.engine_args() == api2.engine_args()
    assert "--api-server-count 2" in " ".join(api2.serve_argv("/m", 8000))
    assert api2.to_dict() != base.to_dict()


def test_invalid_arm_config_pairs_raise():
    with pytest.raises(ValueError):
        e.arm_config("TP1", "AR1")
    with pytest.raises(ValueError):
        e.arm_config("TP2", "EXECuni")


def test_unknown_names_raise():
    with pytest.raises(ValueError, match="TP4"):
        e.base_config("TP4")
    with pytest.raises(ValueError, match="AR9"):
        e.arm_config("TP2", "AR9")


def test_arms_table_matches_contract():
    assert set(e.ARMS) == set(e.ARM_NAMES)
    assert e.ARMS["base"] == frozenset(e.CONFIG_NAMES)
    for arm in ("AR1", "AR2", "AR3", "PCon", "FIBtrtllm"):
        assert e.ARMS[arm] == {"TP2"}, arm
    assert e.ARMS["G1"] == e.ARMS["G2"] == {"TP1", "TP2"}
    assert e.ARMS["EXECuni"] == {"TP1"} and e.ARMS["API2"] == {"TP2", "DP2"}


def test_all_configs_is_every_valid_pair():
    pairs = [(c.name, c.arm) for c in e.all_configs()]
    assert len(pairs) == len(set(pairs)) == sum(len(v) for v in e.ARMS.values())
    assert ("TP2", "AR3") in pairs and ("TP1", "AR1") not in pairs


def test_validate_rejects_forbidden():
    cfg = e.base_config("TP1")
    bad = cfg.with_arm("base", set_flags={"--disable-log-requests": None})
    with pytest.raises(ValueError, match="disable-log-requests"):
        e.validate(bad)
    with pytest.raises(ValueError, match="VLLM_ATTENTION_BACKEND"):
        e.validate(cfg.with_arm("base", env={"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}))


@pytest.mark.parametrize("change,match", [
    ({"set_flags": {"--model": "/m"}}, "--model"),
    ({"set_flags": {"--disable-log-stats": None}}, "disable-log-stats"),
    ({"set_flags": {"-cc.cudagraph_mode": "NONE"}}, "-cc.cudagraph_mode"),
    ({"set_flags": {"--compilation-config.mode": "3"}}, "dotted"),
    ({"set_flags": {"--compilation-config": "{not json"}}, "--compilation-config"),
    ({"set_flags": {"--attention-config": None}}, "--attention-config"),
    ({"env": {"VLLM_USE_V2_MODEL_RUNNER": "1"}}, "VLLM_USE_V2_MODEL_RUNNER"),
])
def test_validate_rejects(change, match):
    with pytest.raises(ValueError, match=match):
        e.validate(e.base_config("TP2").with_arm("base", **change))


def test_validate_rejects_duplicates_gpu_mismatch_and_bad_arm():
    cfg = e.base_config("TP1")
    dup = dataclasses.replace(cfg, flags=cfg.flags + (("--seed", "1"),))
    with pytest.raises(ValueError, match="duplicate.*--seed"):
        e.validate(dup)
    with pytest.raises(ValueError, match="gpus"):
        e.validate(dataclasses.replace(cfg, gpus=(0, 1)))
    with pytest.raises(ValueError, match="AR1"):
        e.validate(cfg.with_arm("AR1"))


def test_with_arm_set_drop_env():
    cfg = e.base_config("TP1")
    new = cfg.with_arm("x", set_flags={"--seed": "7", "--foo": None}, drop_flags=["--stream-interval"],
                       env={"ZZZ": "1", "AAA": "2"})
    args = new.engine_args()
    assert new.arm == "x" and cfg.arm == "base"
    assert args.index("--seed") == cfg.engine_args().index("--seed") and args[args.index("--seed") + 1] == "7"
    assert "--stream-interval" not in args
    assert args[-3] == "--foo" and args[-2] == "--compilation-config"
    assert [k for k, _ in new.env] == sorted(k for k, _ in new.env)
    assert ("ZZZ", "1") in new.env and ("AAA", "2") in new.env
    with pytest.raises(dataclasses.FrozenInstanceError):
        new.arm = "y"


def test_json_flag_values_parse():
    for cfg in e.all_configs():
        a = cfg.engine_args()
        for flag in ("--compilation-config", "--attention-config"):
            if flag in a:
                json.loads(a[a.index(flag) + 1])


def test_to_dict_is_stable_and_serializable():
    d1, d2 = e.base_config("TP2").to_dict(), e.base_config("TP2").to_dict()
    assert d1 == d2 and json.loads(json.dumps(d1)) == d1


def test_to_dict_distinguishes_every_config():
    dumps = {json.dumps(c.to_dict(), sort_keys=True) for c in e.all_configs()}
    assert len(dumps) == len(e.all_configs())
