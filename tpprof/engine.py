"""Engine configurations: the one source of truth for vLLM flags and environment (spec 4.1, 4.2, 4.7).

Every engine tpprof starts, whether `vllm serve`, `vllm bench latency` or the offline
driver, is rendered from an `EngineConfig` built here (contract C1, AM1, AM2).
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from tpprof.constants import SERVED_MODEL_NAME

Flag = tuple[str, "str | None"]

CONFIG_NAMES = ("TP1", "TP2", "DP2", "DP2rand0", "DP2rand1")
ARM_NAMES = ("base", "AR1", "AR2", "AR3", "G1", "G2", "PCon", "EXECuni", "FIBtrtllm", "API2")

CC_FLAG = "--compilation-config"


def dump_json(obj: object) -> str:
    """Canonical JSON for flag values (C1)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compilation_config(cudagraph_mode: str = "FULL_AND_PIECEWISE", fuse_allreduce_rms: bool | None = None) -> str:
    """`--compilation-config` value; `pass_config` is present only when `fuse_allreduce_rms` is given."""
    cc: dict[str, object] = {"cudagraph_mode": cudagraph_mode}
    if fuse_allreduce_rms is not None:
        cc["pass_config"] = {"fuse_allreduce_rms": fuse_allreduce_rms}
    return dump_json(cc)


COMMON_FLAGS: tuple[Flag, ...] = (
    ("--dtype", "bfloat16"),
    ("--max-model-len", "9216"),
    ("--gpu-memory-utilization", "0.90"),
    ("--max-num-seqs", "1024"),                      # AM1
    ("--max-num-batched-tokens", "8192"),
    ("--block-size", "16"),
    ("--kv-cache-dtype", "auto"),
    ("--seed", "0"),
    ("--no-enable-prefix-caching", None),
    ("--enable-chunked-prefill", None),
    ("--async-scheduling", None),
    ("--stream-interval", "1"),
    ("--no-enable-dbo", None),
    ("--no-enable-batch-sharded-sampling", None),
    ("--performance-mode", "balanced"),
    ("--optimization-level", "2"),
    ("--attention-backend", "FLASH_ATTN"),
    ("--attention-config", dump_json({"flash_attn_version": 3})),
    ("--generation-config", "vllm"),
    ("--fail-on-environ-validation", None),
)

_TP1_PARALLEL: tuple[Flag, ...] = (("--tensor-parallel-size", "1"), ("--distributed-executor-backend", "mp"))  # AM2

# config -> (tp, dp, gpus, parallelism flags, --compilation-config value)
PARALLEL: dict[str, tuple[int, int, tuple[int, ...], tuple[Flag, ...], str]] = {
    "TP1": (1, 1, (0,), _TP1_PARALLEL, compilation_config()),
    "TP2": (2, 1, (0, 1), (("--tensor-parallel-size", "2"), ("--distributed-executor-backend", "mp")),
            compilation_config(fuse_allreduce_rms=True)),
    "DP2": (1, 2, (0, 1), (("--tensor-parallel-size", "1"), ("--data-parallel-size", "2")), compilation_config()),
    "DP2rand0": (1, 1, (0,), _TP1_PARALLEL, compilation_config()),
    "DP2rand1": (1, 1, (1,), _TP1_PARALLEL, compilation_config()),
}

BASE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_ALLREDUCE_USE_FLASHINFER": "1",
    "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "0",
    "VLLM_ALLREDUCE_USE_SYMM_MEM": "1",
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND": "auto",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_USE_NCCL_SYMM_MEM": "0",
    "VLLM_USE_RUST_BENCH": "0",
    "VLLM_USE_RUST_FRONTEND": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "60",
}

UNSET_ALWAYS = ("OPENAI_API_KEY", "SAVE_TO_PYTORCH_BENCHMARK_FORMAT", "VLLM_ATTENTION_BACKEND",
                "VLLM_USE_V2_MODEL_RUNNER")

FORBIDDEN_FLAGS = ("--disable-log-requests", "--disable-log-stats", "--model")
FORBIDDEN_ENV = ("VLLM_ATTENTION_BACKEND", "VLLM_USE_V2_MODEL_RUNNER")
# `*-config` flags whose value is not JSON.
_NON_JSON_CONFIG_FLAGS = frozenset({"--config", "--generation-config"})

_TP2 = frozenset({"TP2"})
ARMS: dict[str, frozenset[str]] = {
    "base": frozenset(CONFIG_NAMES),
    "AR1": _TP2,
    "AR2": _TP2,
    "AR3": _TP2,
    "G1": frozenset({"TP1", "TP2"}),
    "G2": frozenset({"TP1", "TP2"}),
    "PCon": _TP2,
    "EXECuni": frozenset({"TP1"}),
    "FIBtrtllm": _TP2,
    "API2": frozenset({"TP2", "DP2"}),
}

_SERVE_HOST = "127.0.0.1"


@dataclass(frozen=True)
class EngineConfig:
    name: str
    arm: str
    tp: int
    dp: int
    gpus: tuple[int, ...]
    flags: tuple[Flag, ...]
    env: tuple[tuple[str, str], ...]
    unset_env: tuple[str, ...]
    api_servers: int = 1

    @property
    def exposes_gauges(self) -> bool:
        """vLLM 0.30.0 disables stats logging, and with it the /metrics gauges, when --api-server-count > 1
        ("AsyncLLM created with api_server_count more than 1; disabling stats logging", 2026-10-02 box)."""
        return self.api_servers == 1

    def engine_args(self) -> list[str]:
        args: list[str] = []
        for flag, value in self.flags:
            args.append(flag)
            if value is not None:
                args.append(value)
        return args

    def serve_argv(self, model_dir: str, port: int, vllm_bin: str = "vllm") -> list[str]:
        serve_only = ["--served-model-name", SERVED_MODEL_NAME, "--host", _SERVE_HOST, "--port", str(port),
                      "--api-server-count", str(self.api_servers), "--disable-uvicorn-access-log",
                      "--no-enable-log-requests"]
        return [vllm_bin, "serve", model_dir, *self.engine_args(), *serve_only]

    def offline_args(self, model_dir: str) -> list[str]:
        return ["--model", model_dir, *self.engine_args()]

    def bench_latency_argv(self, model_dir: str, batch: int, input_len: int, output_len: int,
                           warmup: int, iters: int, output_json: str, vllm_bin: str = "vllm") -> list[str]:
        return [vllm_bin, "bench", "latency", *self.offline_args(model_dir),
                "--batch-size", str(batch), "--input-len", str(input_len), "--output-len", str(output_len),
                "--num-iters-warmup", str(warmup), "--num-iters", str(iters), "--output-json", output_json]

    def environment(self, base: Mapping[str, str]) -> dict[str, str]:
        env = {k: v for k, v in base.items() if k not in self.unset_env}
        env.update(self.env)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
        return env

    def with_arm(self, arm: str, set_flags: Mapping[str, str | None] | None = None,
                 drop_flags: Iterable[str] = (), env: Mapping[str, str] | None = None) -> EngineConfig:
        """A copy with `arm` set. Does not validate; `arm_config` does.

        `set_flags` replaces a flag's value in place, or appends the flag before
        `--compilation-config` (at the end if there is none). `drop_flags` removes flags.
        """
        flags = list(self.flags)
        for flag, value in (set_flags or {}).items():
            names = [f for f, _ in flags]
            if flag in names:
                flags[names.index(flag)] = (flag, value)
            else:
                at = names.index(CC_FLAG) if CC_FLAG in names else len(flags)
                flags.insert(at, (flag, value))
        dropped = set(drop_flags)
        flags = [(f, v) for f, v in flags if f not in dropped]
        new_env = {**dict(self.env), **(env or {})}
        return dataclasses.replace(self, arm=arm, flags=tuple(flags), env=tuple(sorted(new_env.items())))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arm": self.arm,
            "tp": self.tp,
            "dp": self.dp,
            "gpus": list(self.gpus),
            "flags": [[f, v] for f, v in self.flags],
            "env": dict(self.env),
            "unset_env": list(self.unset_env),
            "api_servers": self.api_servers,
        }


def base_config(name: str) -> EngineConfig:
    if name not in PARALLEL:
        raise ValueError(f"unknown engine config {name!r}; expected one of {CONFIG_NAMES}")
    tp, dp, gpus, parallel, cc = PARALLEL[name]
    cfg = EngineConfig(name=name, arm="base", tp=tp, dp=dp, gpus=gpus,
                       flags=(*COMMON_FLAGS, *parallel, (CC_FLAG, cc)),
                       env=tuple(sorted(BASE_ENV.items())), unset_env=tuple(sorted(UNSET_ALWAYS)))
    validate(cfg)
    return cfg


def _with_cc(cfg: EngineConfig, **changes: object) -> EngineConfig:
    cc = json.loads(dict(cfg.flags)[CC_FLAG])
    for key, value in changes.items():
        if key == "fuse_allreduce_rms":
            cc.setdefault("pass_config", {})["fuse_allreduce_rms"] = value
        else:
            cc[key] = value
    return cfg.with_arm(cfg.arm, set_flags={CC_FLAG: dump_json(cc)})


def _ar1(cfg: EngineConfig) -> EngineConfig:
    return _with_cc(cfg, fuse_allreduce_rms=False)


def _ar2(cfg: EngineConfig) -> EngineConfig:
    return _ar1(cfg).with_arm(cfg.arm, env={"VLLM_ALLREDUCE_USE_FLASHINFER": "0", "VLLM_ALLREDUCE_USE_SYMM_MEM": "0"})


def _ar3(cfg: EngineConfig) -> EngineConfig:
    return _ar2(cfg).with_arm(cfg.arm, set_flags={"--disable-custom-all-reduce": None})


def _g1(cfg: EngineConfig) -> EngineConfig:
    return _with_cc(cfg, cudagraph_mode="NONE")


def _g2(cfg: EngineConfig) -> EngineConfig:
    return cfg.with_arm(cfg.arm, drop_flags=(CC_FLAG, "--optimization-level")).with_arm(
        cfg.arm, set_flags={"--enforce-eager": None})


def _pcon(cfg: EngineConfig) -> EngineConfig:
    flags = tuple(("--enable-prefix-caching", None) if f == "--no-enable-prefix-caching" else (f, v)
                  for f, v in cfg.flags)
    return dataclasses.replace(cfg, flags=flags)


def _execuni(cfg: EngineConfig) -> EngineConfig:
    return cfg.with_arm(cfg.arm, set_flags={"--distributed-executor-backend": "uni"})


def _fibtrtllm(cfg: EngineConfig) -> EngineConfig:
    return cfg.with_arm(cfg.arm, env={"VLLM_FLASHINFER_ALLREDUCE_BACKEND": "trtllm"})


def _api2(cfg: EngineConfig) -> EngineConfig:
    return dataclasses.replace(cfg, api_servers=2)


_TRANSFORMS = {
    "base": lambda cfg: cfg,
    "AR1": _ar1,
    "AR2": _ar2,
    "AR3": _ar3,
    "G1": _g1,
    "G2": _g2,
    "PCon": _pcon,
    "EXECuni": _execuni,
    "FIBtrtllm": _fibtrtllm,
    "API2": _api2,
}


def _check_arm(name: str, arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARM_NAMES}")
    if name not in ARMS[arm]:
        raise ValueError(f"arm {arm!r} does not apply to config {name!r}; it applies to {sorted(ARMS[arm])}")


def arm_config(name: str, arm: str) -> EngineConfig:
    """`base_config(name)` with the arm transform applied (spec 4.7, C1), validated."""
    _check_arm(name, arm)
    cfg = dataclasses.replace(_TRANSFORMS[arm](base_config(name)), arm=arm)
    validate(cfg)
    return cfg


def validate(cfg: EngineConfig) -> None:
    """Raise ValueError naming the first problem that would make the engine command wrong."""
    if cfg.name not in CONFIG_NAMES:
        raise ValueError(f"unknown engine config {cfg.name!r}; expected one of {CONFIG_NAMES}")
    _check_arm(cfg.name, cfg.arm)
    seen: set[str] = set()
    for flag, value in cfg.flags:
        if flag in FORBIDDEN_FLAGS:
            raise ValueError(f"{cfg.name}/{cfg.arm}: forbidden engine flag {flag}")
        if flag.startswith("-cc.") or "." in flag:
            raise ValueError(f"{cfg.name}/{cfg.arm}: dotted flag {flag}; use the JSON form")
        if flag in seen:
            raise ValueError(f"{cfg.name}/{cfg.arm}: duplicate flag {flag}")
        seen.add(flag)
        if flag.endswith("-config") and flag not in _NON_JSON_CONFIG_FLAGS:
            try:
                parsed = json.loads(value) if value is not None else None
            except json.JSONDecodeError as exc:
                raise ValueError(f"{cfg.name}/{cfg.arm}: {flag} value is not valid JSON: {exc}") from None
            if not isinstance(parsed, dict):
                raise ValueError(f"{cfg.name}/{cfg.arm}: {flag} value must be a JSON object, got {value!r}")
    for key, _ in cfg.env:
        if key in FORBIDDEN_ENV:
            raise ValueError(f"{cfg.name}/{cfg.arm}: forbidden environment variable {key}")
    if cfg.tp * cfg.dp != len(cfg.gpus):
        raise ValueError(f"{cfg.name}/{cfg.arm}: tp*dp = {cfg.tp * cfg.dp} but gpus = {cfg.gpus}")


def all_configs() -> list[EngineConfig]:
    """Every valid (config, arm) pair, in CONFIG_NAMES then ARM_NAMES order."""
    return [arm_config(name, arm) for name in CONFIG_NAMES for arm in ARM_NAMES if name in ARMS[arm]]
