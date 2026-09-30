from __future__ import annotations

import pytest

from tests.conftest import FIXTURES
from tpprof import kernels


def _rows() -> list[tuple[str, str, bool]]:
    rows = []
    for line in (FIXTURES / "kernel_names.tsv").read_text().splitlines():
        if line.strip():
            name, category, ar_launch = line.split("\t")
            rows.append((name, category, ar_launch == "1"))
    return rows


ROWS = _rows()


def test_fixture_covers_every_non_memcpy_category():
    assert {c for _, c, _ in ROWS} == set(kernels.CATEGORIES) - {"memcpy"}


@pytest.mark.parametrize("name,category,ar_launch", ROWS, ids=[r[0][:60] for r in ROWS])
def test_real_kernel_names(name, category, ar_launch):
    assert kernels.categorize(name) == category
    assert kernels.is_ar_launch(name) is ar_launch


def test_rmsnorm_lamport_is_an_ar_tail_not_a_launch():
    name = next(n for n, _, _ in ROWS if "rmsNormLamport" in n)
    assert kernels.is_ar_tail(name)
    assert not kernels.is_ar_launch(name)
    assert kernels.categorize(name) == "all_reduce"
    assert not any(kernels.is_ar_tail(n) for n, _, _ in ROWS if n != name)


def test_all_gather_detection():
    gathers = [n for n, _, _ in ROWS if kernels.is_all_gather(n)]
    assert gathers == ["ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"]


def test_flashinfer_comm_names_are_not_attention():
    # "flashinfer" contains "flash"; the AR regexes run before ATTENTION.
    for name, category, _ in ROWS:
        if "flashinfer" in name:
            assert kernels.categorize(name) == "all_reduce"


def test_fa3_cutlass_name_is_attention_not_gemm():
    fa3 = next(n for n, _, _ in ROWS if "FlashAttnFwdSm90" in n)
    assert "cutlass" in fa3
    assert kernels.categorize(fa3) == "attention"


def test_cutlass_gemm_without_flashattn_is_gemm():
    assert kernels.categorize("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_tn_align8>(Params)") == "gemm"


def test_memcpy_and_memset_records():
    assert kernels.categorize("[CUDA memcpy Host-to-Device]") == "memcpy"
    assert kernels.categorize("[CUDA memset]") == "memcpy"


@pytest.mark.parametrize("name,category", [
    ("ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)", "all_reduce"),
    ("ncclSymkDevKernel_AllGather_LL(ncclSymkDevWorkArgs4K)", "all_gather"),
    ("void vllm::cross_device_reduce_2stage<__nv_bfloat16, 2>(...)", "all_reduce"),
    ("void multimem_all_reduce_kernel<__nv_bfloat16>(...)", "all_reduce"),
])
def test_other_comm_names(name, category):
    assert kernels.categorize(name) == category
