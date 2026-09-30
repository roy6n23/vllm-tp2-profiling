"""Effective-config verification from vLLM 0.30.0 engine logs (spec 4.2, AM3-AM5; contract C6).

The log formats are the ones verified against the v0.30.0 source (research D2-16, D5-6..D5-24). Matching
is always on the message text after the optional '(<Name> pid=<pid>) ' prefix and the
'LEVEL mm-dd HH:MM:SS [file:line] ' header, never on file:line. uvicorn lines have no vLLM header and are
matched by substring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tpprof.constants import VLLM_VERSION

# The engine flag value that `Chunked prefill is enabled with max_num_batched_tokens=%d.` must echo (C1).
EXPECTED_MAX_NUM_BATCHED_TOKENS = 8192
EXPECTED_ATTENTION_BACKEND = "FLASH_ATTN"
EXPECTED_FLASH_ATTN_VERSION = 3

# Spec AM4 hard list, then the D5-24 startup and worker-death signatures.
HARD_FAILURES: tuple[str, ...] = (
    "Failed to initialize FlashInfer Allreduce norm fusion workspace with backend=",
    "Flashinfer is not installed or comm module not found, skipping allreduce fusion pass",
    "Failed to initialize Flashinfer allreduce workspace. Flashinfer allreduce-norm fusion will be disabled.",
    "AllReduce fusion pass is disabled.",
    "Custom allreduce is disabled because",
    "Custom allreduce is disabled due to an unsupported world size",
    "SymmMemCommunicator: symmetric memory initialization failed",
    "Insufficient space in /dev/shm",
    "EngineCore failed to start.",
    "Engine core initialization failed",
    "Timed out waiting for engine core processes to start",
    "died unexpectedly",
)

# Spec AM4 recorded list: the normal mnnvl -> trtllm fallback. These set fi_backend_fallback, never fail.
FALLBACK_WARNINGS: tuple[str, ...] = (
    "FlashInfer MNNVL multicast is unavailable on the current topology",
    "Failed to initialize FlashInfer All Reduce workspace:",
    "FlashInfer mnnvl allreduce workspace unavailable",
)

SAMPLING_OVERRIDE = "Default vLLM sampling parameters have been overridden"
MRV2_FALLBACK = "Model Runner V2 does not yet support"
V2_RUNNER = "Using V2 Model Runner"
ENFORCE_EAGER = "Enforce eager set, disabling torch.compile and CUDAGraphs"
STARTUP_COMPLETE = "Application startup complete."
FUSIONS_PREFIX = "Enabled custom fusions: "

_HEADER = re.compile(
    r"(?:\((?P<proc>\w+) pid=(?P<pid>\d+)\) )?"
    r"(?P<lvl>DEBUG|INFO|WARNING|ERROR|CRITICAL) (?P<ts>\d\d-\d\d \d\d:\d\d:\d\d) "
    r"\[(?P<src>[^\]]+):(?P<ln>\d+)\] (?P<msg>.*)")
_BANNER = re.compile(r"Initializing a V1 LLM engine \(v(?P<v>[^)]+)\) with config: (?P<cfg>.*)")
_BANNER_EXECUTOR = re.compile(r"distributed_executor_backend=(\w+)")
_NONDEFAULT_EXECUTOR = re.compile(r"non-default args: .*'distributed_executor_backend': '(\w+)'")
_KV = re.compile(r"(?P<dev>\w+) KV cache size: (?P<tok>[\d,]+) tokens, "
                 r"Maximum concurrency for (?P<len>[\d,]+) tokens per request: (?P<conc>[\d.]+)x")
_AVAILABLE_KV = re.compile(r"Available KV cache memory: (?P<g>[\d.]+) GiB")
_MODEL_LOADING = re.compile(r"Model loading took (?P<g>[\d.]+) GiB memory")
_CHUNKED = re.compile(r"Chunked prefill is enabled with max_num_batched_tokens=(?P<n>\d+)\.")
# Explicit form (--attention-backend applied, cuda.py:478) and auto-selection form (cuda.py:539-546). With the
# explicit flag 0.30.0 prints only the first, so the second means the flag was not applied (AM5, D5-12).
_ATTENTION_EXPLICIT = re.compile(r"Using AttentionBackendEnum\.(?P<b>[A-Z_]+) backend\.")
_ATTENTION_AUTO = re.compile(r"Using (?P<b>[A-Z_]+) attention backend out of potential backends")
_FA_VERSION = re.compile(r"Using FlashAttention version (?P<v>\d+)")
_AR_LIST = re.compile(r"Using \[(?P<l>[^\]]*)\] all-reduce backends \(in dispatch order\) for group 'tp:0'")
_AR_ITEM = re.compile(r"'([A-Z_]+)'")
_FI_WORKSPACE = re.compile(r"Initialized FlashInfer Allreduce norm fusion workspace with backend=(?P<b>\w+)")
_NCCL = re.compile(r"vLLM is using nccl==(?P<v>[\d.]+)")
_JIT = re.compile(r"during inference: .*This causes a latency spike")
_GRAPH = re.compile(r"Graph capturing finished in .*")


@dataclass
class EffectiveConfig:
    vllm_version: str | None = None
    engine_config_banner: str | None = None
    kv_cache_tokens: list[int] = field(default_factory=list)       # one per engine, commas stripped
    max_concurrency: list[float] = field(default_factory=list)
    available_kv_gib: list[float] = field(default_factory=list)
    model_loading_gib: list[float] = field(default_factory=list)
    chunked_prefill_tokens: int | None = None
    v2_model_runner: bool = False
    attention_backend: str | None = None                           # "FLASH_ATTN"
    attention_explicit: bool = False    # every attention line seen was the explicit AM5 form (none auto-selected)
    flash_attn_version: int | None = None
    ar_backends: list[str] | None = None
    fusions_line: str | None = None
    fi_backend: str | None = None                                  # "mnnvl" | "trtllm"
    fi_backend_fallback: bool = False
    nccl_version: str | None = None
    executor: str | None = None                                    # from banner: distributed_executor_backend=...
    enforce_eager: bool = False
    sampling_override: bool = False
    mrv2_fallback: bool = False
    jit_after_warmup: list[str] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)
    startup_complete: bool = False
    graph_capture: list[str] = field(default_factory=list)


def _messages(text: str) -> list[str]:
    """Message part of every line. Splits on \\r too: tqdm bars glue other lines on with '\\r' (D5-6)."""
    out = []
    for piece in re.split(r"[\r\n]+", text):
        if not piece.strip():
            continue
        m = _HEADER.search(piece)
        out.append(m.group("msg") if m else piece)
    return out


def parse_engine_log(text: str) -> EffectiveConfig:
    eff = EffectiveConfig()
    nondefault_executor = None
    attention_forms: set[str] = set()
    for msg in _messages(text):
        if m := _BANNER.search(msg):
            if eff.vllm_version is None:
                eff.vllm_version, eff.engine_config_banner = m.group("v"), m.group("cfg")
                if ex := _BANNER_EXECUTOR.search(m.group("cfg")):
                    eff.executor = ex.group(1)
        elif m := _NONDEFAULT_EXECUTOR.search(msg):
            nondefault_executor = m.group(1)
        elif m := _KV.search(msg):
            eff.kv_cache_tokens.append(int(m.group("tok").replace(",", "")))
            eff.max_concurrency.append(float(m.group("conc")))
        elif m := _AVAILABLE_KV.search(msg):
            eff.available_kv_gib.append(float(m.group("g")))
        elif m := _MODEL_LOADING.search(msg):
            eff.model_loading_gib.append(float(m.group("g")))
        elif m := _CHUNKED.search(msg):
            eff.chunked_prefill_tokens = int(m.group("n"))
        elif m := _FA_VERSION.search(msg):
            eff.flash_attn_version = int(m.group("v"))
        elif m := _AR_LIST.search(msg):
            eff.ar_backends = _AR_ITEM.findall(m.group("l"))
        elif m := _FI_WORKSPACE.search(msg):
            eff.fi_backend = m.group("b")
        elif m := _NCCL.search(msg):
            eff.nccl_version = m.group("v")
        elif m := _GRAPH.search(msg):
            eff.graph_capture.append(m.group(0))
        elif _JIT.search(msg):
            eff.jit_after_warmup.append(msg)
        elif FUSIONS_PREFIX in msg:
            eff.fusions_line = msg.split(FUSIONS_PREFIX, 1)[1]
        for form, rx in (("explicit", _ATTENTION_EXPLICIT), ("auto", _ATTENTION_AUTO)):
            if m := rx.search(msg):
                eff.attention_backend = m.group("b")
                attention_forms.add(form)
        eff.v2_model_runner |= V2_RUNNER in msg
        eff.enforce_eager |= ENFORCE_EAGER in msg
        eff.sampling_override |= SAMPLING_OVERRIDE in msg
        eff.mrv2_fallback |= MRV2_FALLBACK in msg
        eff.startup_complete |= STARTUP_COMPLETE in msg
        eff.fi_backend_fallback |= any(s in msg for s in FALLBACK_WARNINGS)
        if any(s in msg for s in HARD_FAILURES):
            eff.hard_failures.append(msg)
    eff.attention_explicit = attention_forms == {"explicit"}
    if eff.executor is None:
        # The real 0.30.0 banner has no executor field; the CLI input always carries it because the
        # flag's default is None (D5-8), and every config pins it (AM2).
        eff.executor = nondefault_executor
    return eff


@dataclass(frozen=True)
class Expectation:
    engines: int                        # number of KV lines expected (DP2: 2)
    serve: bool                         # require "Application startup complete."
    tp2: bool
    ar_first: str | None = None         # "FLASHINFER" (AR0, AR1, G1, G2, PCon, FIBtrtllm, API2)
    ar_exact: tuple[tuple[str, ...], ...] | None = None   # AR2: (("CUSTOM","PYNCCL"),); AR3: (("PYNCCL",),)
    require_fi_workspace: bool = False  # AR0, AR1, G1, PCon, FIBtrtllm, API2
    require_enforce_eager: bool = False # G2
    fi_backend_exact: str | None = None # FIBtrtllm: "trtllm"
    executor: str | None = None         # pinned --distributed-executor-backend (AM2): "mp", or "uni" for EXECuni


# Config -> (engines, tp2, pinned executor). DP2rand0/1 are single-engine TP1-like configs (ruling R6). C1 pins
# `--distributed-executor-backend mp` for TP1, DP2rand0/1 and TP2 (AM2); DP2 passes no executor flag, so its
# logs carry no executor value to check.
_CONFIGS = {"TP1": (1, False, "mp"), "DP2rand0": (1, False, "mp"), "DP2rand1": (1, False, "mp"),
            "TP2": (1, True, "mp"), "DP2": (2, False, None)}
# Arm -> configs it applies to (C1 arms table; the TP2 base arm is the spec's AR0 and is named "base" here).
_ARM_CONFIGS = {
    "base": frozenset(_CONFIGS),
    "AR1": frozenset({"TP2"}), "AR2": frozenset({"TP2"}), "AR3": frozenset({"TP2"}),
    "G1": frozenset({"TP1", "TP2"}), "G2": frozenset({"TP1", "TP2"}), "PCon": frozenset({"TP2"}),
    "EXECuni": frozenset({"TP1"}), "FIBtrtllm": frozenset({"TP2"}), "API2": frozenset({"TP2", "DP2"}),
}
# Per-arm TP2 expectations (AM3). G2's enforce-eager line is required for TP1 and TP2 alike.
_TP2_ARMS = {
    "base": dict(ar_first="FLASHINFER", require_fi_workspace=True),
    "AR1": dict(ar_first="FLASHINFER", require_fi_workspace=True),
    "AR2": dict(ar_exact=(("CUSTOM", "PYNCCL"),)),
    "AR3": dict(ar_exact=(("PYNCCL",),)),
    "G1": dict(ar_first="FLASHINFER", require_fi_workspace=True),
    "G2": dict(ar_first="FLASHINFER"),
    "PCon": dict(ar_first="FLASHINFER", require_fi_workspace=True),
    "FIBtrtllm": dict(ar_first="FLASHINFER", require_fi_workspace=True, fi_backend_exact="trtllm"),
    "API2": dict(ar_first="FLASHINFER", require_fi_workspace=True),
}


def expectation_for(config: str, arm: str, serve: bool) -> Expectation:
    if config not in _CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {sorted(_CONFIGS)}")
    if arm not in _ARM_CONFIGS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(_ARM_CONFIGS)}")
    if config not in _ARM_CONFIGS[arm]:
        raise ValueError(f"arm {arm!r} does not apply to config {config!r}")
    engines, tp2, executor = _CONFIGS[config]
    extra = dict(_TP2_ARMS.get(arm, {})) if tp2 else {}
    if arm == "G2":
        extra["require_enforce_eager"] = True
    if arm == "EXECuni":
        executor = "uni"
    return Expectation(engines=engines, serve=serve, tp2=tp2, executor=executor, **extra)


def _fmt(backends: list[str] | tuple[str, ...]) -> str:
    return "[" + ", ".join(f"'{b}'" for b in backends) + "]"


def check(eff: EffectiveConfig, exp: Expectation) -> list[str]:
    """Violations of the effective-config rules (spec 4.2, AM3-AM5); [] means OK."""
    v: list[str] = []
    if eff.vllm_version != VLLM_VERSION:
        v.append(f'missing line "Initializing a V1 LLM engine (v{VLLM_VERSION})" '
                 f"(found version {eff.vllm_version}).")
    if not eff.v2_model_runner:
        v.append(f'missing line "{V2_RUNNER}".')
    if eff.attention_backend != EXPECTED_ATTENTION_BACKEND or not eff.attention_explicit:
        found = (f"attention backend {eff.attention_backend}" if eff.attention_explicit or eff.attention_backend is None
                 else f'the auto-selection line "Using {eff.attention_backend} attention backend out of potential '
                      f'backends", so --attention-backend was not applied')
        v.append(f'missing line "Using AttentionBackendEnum.{EXPECTED_ATTENTION_BACKEND} backend." (found {found}).')
    if eff.flash_attn_version != EXPECTED_FLASH_ATTN_VERSION:
        v.append(f'missing line "Using FlashAttention version {EXPECTED_FLASH_ATTN_VERSION}" '
                 f"(found version {eff.flash_attn_version}).")
    if eff.chunked_prefill_tokens != EXPECTED_MAX_NUM_BATCHED_TOKENS:
        v.append(f'missing line "Chunked prefill is enabled with '
                 f'max_num_batched_tokens={EXPECTED_MAX_NUM_BATCHED_TOKENS}." '
                 f"(found {eff.chunked_prefill_tokens}).")
    if len(eff.kv_cache_tokens) != exp.engines:
        v.append(f'expected {exp.engines} "GPU KV cache size: N tokens" line(s), one per engine, '
                 f"found {len(eff.kv_cache_tokens)}.")
    for msg in eff.hard_failures:
        v.append(f'hard failure: "{msg}".')
    if eff.sampling_override:
        v.append(f'unexpected line "{SAMPLING_OVERRIDE}" (the engine must run with --generation-config vllm).')
    if eff.mrv2_fallback:
        v.append(f'unexpected line "{MRV2_FALLBACK}" (the engine fell back to the V1 model runner).')
    if exp.serve and not eff.startup_complete:
        v.append(f'missing line "{STARTUP_COMPLETE}".')
    if exp.executor is not None and eff.executor != exp.executor:
        v.append(f"expected \"'distributed_executor_backend': '{exp.executor}'\" in the \"non-default args:\" line "
                 f"(found executor {eff.executor}).")
    if exp.require_enforce_eager and not eff.enforce_eager:
        v.append(f'missing line "{ENFORCE_EAGER}" (required by --enforce-eager).')
    if exp.tp2:
        v += _check_tp2(eff, exp)
    return v


def _check_tp2(eff: EffectiveConfig, exp: Expectation) -> list[str]:
    v: list[str] = []
    if eff.ar_backends is None:
        v.append('missing line "Using [...] all-reduce backends (in dispatch order) for group \'tp:0\'".')
    else:
        if exp.ar_first is not None and eff.ar_backends[:1] != [exp.ar_first]:
            v.append(f"expected {exp.ar_first} first in the all-reduce backends for group 'tp:0', "
                     f"found {_fmt(eff.ar_backends)}.")
        if exp.ar_exact is not None and tuple(eff.ar_backends) not in exp.ar_exact:
            wanted = " or ".join(_fmt(a) for a in exp.ar_exact)
            v.append(f"expected all-reduce backends exactly {wanted} for group 'tp:0', "
                     f"found {_fmt(eff.ar_backends)}.")
    if exp.fi_backend_exact is not None:
        if eff.fi_backend != exp.fi_backend_exact:
            v.append(f'missing line "Initialized FlashInfer Allreduce norm fusion workspace with '
                     f'backend={exp.fi_backend_exact}" (found backend {eff.fi_backend}).')
    elif exp.require_fi_workspace and eff.fi_backend not in ("mnnvl", "trtllm"):
        v.append('missing line "Initialized FlashInfer Allreduce norm fusion workspace with '
                 f'backend=<mnnvl|trtllm>" (found backend {eff.fi_backend}).')
    return v
