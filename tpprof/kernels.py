"""Kernel-name categorizer for nsys traces (spec 4.5, research D3-9, D7-5, D7-8).

The regexes run in a fixed order so that collective kernels win over the
broader patterns: FlashInfer all-reduce names contain "flash", and FA3
attention names contain "cutlass".
"""
from __future__ import annotations

import re

CATEGORIES = ("all_reduce", "all_gather", "gemm", "attention", "norm_act_rope", "sampling", "memcpy", "other")

# nsys names copy and set records "[CUDA memcpy ...]" / "[CUDA memset]"; traces.py uses the same names.
MEMCPY = re.compile(r"^\[CUDA mem(?:cpy|set)")
# One kernel per all-reduce op.
AR_LAUNCH = re.compile(
    r"ncclDevKernel_AllReduce_|ncclSymkDevKernel_AllReduce_|cross_device_reduce_[12]stage"
    r"|allreduce_fusion_kernel_(?:oneshot_lamport|twoshot_sync)|oneshotAllreduceFusionKernel"
    r"|twoshotAllreduceKernel|(?:one|two)_shot_all_reduce_kernel"
    r"|multimem_(?:all_reduce|one_shot_reduce)_kernel|lamport_allreduce")
# Second kernel of the mnnvl two-shot fused op: all-reduce time, not a new op.
AR_TAIL = re.compile(r"rmsNormLamport")
ALL_GATHER = re.compile(
    r"ncclDevKernel_AllGather|ncclSymkDevKernel_AllGather|cross_device_all_gather"
    r"|mnnvl_lamport_all_gather|multimem_all_gather_kernel")
OTHER_COMM = re.compile(r"ncclDevKernel_|ncclSymkDevKernel_|cross_device_reduce_scatter|mnnvl_lamport_reduce_scatter")
ATTENTION = re.compile(r"FlashAttn|flash_fwd|flash::|fmha|paged_attention|attention_kernel|unified_attention")
GEMM = re.compile(r"gemm|Gemm|GEMM|nvjet|xmma|cublas|cutlass(?!.*FlashAttn)|matmul|_mm_|triton_.*mm")
SAMPLING = re.compile(r"sampl|topk|top_k|top_p|topp|argmax|ArgMax|exponential|gumbel|multinomial|softmax")
NORM_ACT_ROPE = re.compile(
    r"rms_norm|rmsnorm|RMSNorm|layernorm|fused_add|silu|act_and_mul|gelu|rotary|rope"
    r"|triton_poi|triton_red|triton_per|elementwise|reshape_and_cache|copy_kernel|index_select|embedding")

_ORDER = (
    (MEMCPY, "memcpy"),
    (AR_LAUNCH, "all_reduce"),
    (AR_TAIL, "all_reduce"),
    (ALL_GATHER, "all_gather"),
    (OTHER_COMM, "all_reduce"),
    (ATTENTION, "attention"),
    (GEMM, "gemm"),
    (SAMPLING, "sampling"),
    (NORM_ACT_ROPE, "norm_act_rope"),
)


def categorize(name: str) -> str:
    for rx, category in _ORDER:
        if rx.search(name):
            return category
    return "other"


def is_ar_launch(name: str) -> bool:
    """One per all-reduce op."""
    return AR_LAUNCH.search(name) is not None


def is_ar_tail(name: str) -> bool:
    """rmsNormLamport: part of the preceding all-reduce op (its time counts, it is not a new op)."""
    return AR_TAIL.search(name) is not None


def is_all_gather(name: str) -> bool:
    return ALL_GATHER.search(name) is not None
