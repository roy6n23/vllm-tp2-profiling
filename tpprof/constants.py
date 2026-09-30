"""Pinned versions, model identity, hardware constants and workload constants.

Every value here comes from the design spec (sections 3, 4 and the amendments).
Other modules import these names instead of repeating literals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

VLLM_VERSION = "0.30.0"
VLLM_COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
VLLM_IMAGE = "vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
VLLM_IMAGE_TAG = "vllm/vllm-openai:v0.30.0"
TORCH_VERSION = "2.13.0"
TORCH_CUDA = "13.0"
NCCL_EXPECTED = "2.30.7"
MIN_DRIVER_MAJOR = 580
NSYS_VERSION = "2026.5.1"
NSYS_DEFAULT_PATH = "/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys"
NSYS_APT_PACKAGE = "nsight-systems-cli-2026.5.1"


def nsys_path() -> str:
    """nsys executable; the dry run points TPPROF_NSYS at the fake (spec AM18)."""
    return os.environ.get("TPPROF_NSYS", NSYS_DEFAULT_PATH)


@dataclass(frozen=True)
class ModelRef:
    repo: str
    revision: str


MODEL = ModelRef("NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77")
MODEL_META = ModelRef("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659")
SERVED_MODEL_NAME = "llama-3.1-8b-instruct"
# On-disk shard file sizes (safetensors header included) and the tensor payload total (spec AM26).
MODEL_SHARD_SIZES = {
    "model-00001-of-00004.safetensors": 4976698672,
    "model-00002-of-00004.safetensors": 4999802720,
    "model-00003-of-00004.safetensors": 4915916176,
    "model-00004-of-00004.safetensors": 1168138808,
}
MODEL_SHARD_FILES_TOTAL = 16060556376
MODEL_TENSOR_BYTES_TOTAL = 16060522496
MODEL_REQUIRED_FILES = ("config.json", "generation_config.json", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json",
                        "model.safetensors.index.json")


@dataclass(frozen=True)
class GpuSpec:
    name: str
    sm_count: int
    hbm_bw: float            # bytes/s, peak
    bf16_dense_flops: float  # FLOP/s, peak
    power_limit_w: int
    memory_total_mib: int


H100_SXM = GpuSpec("NVIDIA H100 80GB HBM3", 132, 3.352e12, 989.4e12, 700, 81559)
H100_PCIE = GpuSpec("NVIDIA H100 PCIe", 114, 2.039e12, 756e12, 350, 81559)
H100_NVL = GpuSpec("NVIDIA H100 NVL", 132, 3.938e12, 835.5e12, 400, 95830)
NVLINK_DIR_BW = 450e9    # bytes/s per direction on H100 SXM (18 links x 25 GB/s)


@dataclass(frozen=True)
class ModelShape:
    hidden: int = 4096
    intermediate: int = 14336
    layers: int = 32
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128      # config.json has no head_dim; 4096 / 32
    vocab: int = 128256
    bytes_per_param: int = 2


LLAMA31_8B = ModelShape()

# Offline workload (spec 4.3, AM13)
PREFILL_LENS = (512, 2048, 8192)
PREFILL_WARMUP, PREFILL_ITERS = 5, 20
DECODE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
DECODE_INPUT_LEN = 1024
DECODE_L1, DECODE_L2 = 64, 320
DECODE_WARMUP, DECODE_ITERS = 3, 10
XCHECK = {"batch": 8, "input_len": 1024, "output_len": 64, "warmup": 5, "iters": 20}

# Online workload (spec 4.4, AM6-AM11)
ONLINE_INPUT_LEN, ONLINE_OUTPUT_LEN = 1024, 256
SAT_NUM_PROMPTS = 3000
SAT_SEEDS = (1, 2, 3)
RATE_FRACTIONS = (0.20, 0.40, 0.60, 0.75, 0.85, 0.95)
SWEEP_WINDOW_S = 90
MIN_PROMPTS = 200
ROUNDS = 3
LATIN_SQUARE = (("TP1", "TP2", "DP2"), ("TP2", "DP2", "TP1"), ("DP2", "TP1", "TP2"))
NUM_WARMUPS = 16
READY_CHECK_TIMEOUT_S = 60
PC_LOAD_FRACTION = 0.60
TTFT_SLO_S = 1.0
TPOT_SLOS_MS = (5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60)
TTFT_SLO_SENSITIVITY_S = (0.5, 2.0)
ATTAINMENT = 0.90
METRIC_PERCENTILES = (25, 50, 75, 90, 99)
PORT_BASE = 8000

# Box resources (spec 7.5, AM12, AM27)
MIN_SHM_BYTES = 1 << 30
MIN_DISK_BYTES = 60 * 10**9
MIN_NOFILE_HARD = 8192
GPU_FREE_MIB = 1024
