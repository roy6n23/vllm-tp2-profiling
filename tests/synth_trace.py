"""Synthetic Nsight Systems SQLite export (schema 3.30.1 subset, research D6-7).

Models what the analysis depends on:
- one worker process per rank, `globalPid = pid << 24`, `globalTid = (pid << 24) | tid`;
- per step, a vLLM `execute_context_*` push/pop range on the worker's main thread. Inside it
  come an input memcpy and one `cudaGraphLaunch` whose graph-node kernels all share its
  correlationId. After the range end, `sample_tokens` launches the lm_head GEMM, the logits
  all-gather and the sampler eagerly (research D6-4, D6-21);
- the GPU runs behind the CPU by a lag of 25-75% of a step that varies per step, so a
  step's late kernels start on the GPU after the next range began (AM15). The step period
  covers the worst-case GPU time of a step plus an idle gap, so the backlog never builds up:
  a step's first GPU event always starts before the next range does, at any step count;
- FlashInfer all-reduce kernels overlap their predecessor by PDL_OVERLAP_NS (research D3-14);
- rank r spin-waits SYNC_WAIT_NS longer in all-reduce ops j with j % tp == r (AM16);
- correlationIds restart at 1 in every process when collide_correlation_ids is True (D6-8).
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Sequence

from tpprof import constants

T0_NS = 1_000_000_000
RANK_SKEW_NS = 3_000
RANGE_NS = 30_000
IDLE_GAP_NS = 20_000
LAUNCH_LATENCY_NS = 4_000
GRAPH_LAUNCH_NS = 5_000          # the forward's cudaGraphLaunch, after the range start
PDL_OVERLAP_NS = 500
SYNC_WAIT_NS = 2_000
MEASURE_PAD_NS = 1_000

# Real kernel names (tests/fixtures/kernel_names.tsv; research D3-9, D7-8).
FI_ONESHOT = ("void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::"
              "trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>(flashinfer::"
              "trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>)")
FI_TWOSHOT = ("void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_twoshot_sync<(flashinfer::"
              "trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true>(flashinfer::"
              "trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>)")
MNNVL_ONESHOT = ("void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)2, "
                 "__nv_bfloat16, true, (flashinfer::QuantType)0>(flashinfer::trtllm_mnnvl_allreduce::"
                 "AllReduceFusionParams)")
MNNVL_TWOSHOT = ("void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)2, "
                 "__nv_bfloat16, true>(flashinfer::trtllm_mnnvl_allreduce::AllReduceFusionParams)")
MNNVL_TAIL = ("void flashinfer::trtllm_mnnvl_allreduce::rmsNormLamport<__nv_bfloat16, (flashinfer::QuantType)0, "
              "true, 8>(flashinfer::trtllm_mnnvl_allreduce::RMSNormParams)")
CUSTOM_AR = ("void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, "
             "vllm::Signal*, __nv_bfloat16*, int, int)")
NCCL_AR = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
NCCL_AG = "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
FUSED_ADD_RMS_NORM = ("void vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>(c10::BFloat16*, c10::BFloat16*, "
                      "c10::BFloat16 const*, float, int, int)")
FA3 = ("void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::"
       "CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1>>, cute::tuple<cute::C<128>, "
       "cute::C<128>, cute::C<128>>, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, "
       "true, false, false, true, true, false, true, false, false>>>>(flash::FlashAttnFwdSm90<...>::Params)")
CACHE_WRITE = ("void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)"
               "0>(__nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, long const*, int, "
               "int, int, int, int, int, float const*, float const*)")
GEMM_WIDE = "nvjet_tst_128x256_64x4_1x2_h_bz_coopA_TNT"
GEMM_NARROW = "nvjet_tst_64x64_64x8_2x1_v_bz_splitK_TNT"
LM_HEAD = ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off"
           "_kernel__5x_cublas")
SILU = "triton_poi_fused_mul_silu_2"
EXPONENTIAL = ("void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<float, 4, "
               "at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, "
               "double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda(at::cuda::detail::"
               "Philox4_32_10&)#1}>(long, at::PhiloxCudaState, ...)")
ARGMAX = ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, "
          "unsigned int, long, 4>>(at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4>)")
UNKNOWN = "some_unknown_kernel_xyz"

LAYERS = constants.LLAMA31_8B.layers
_PDL = (FI_ONESHOT, FI_TWOSHOT, MNNVL_ONESHOT, MNNVL_TWOSHOT)

TABLES = {
    "StringIds": "id INTEGER PRIMARY KEY, value TEXT NOT NULL",
    "CUPTI_ACTIVITY_KIND_KERNEL": (
        "start INTEGER NOT NULL, end INTEGER NOT NULL, deviceId INTEGER NOT NULL, contextId INTEGER NOT NULL, "
        "streamId INTEGER NOT NULL, correlationId INTEGER, globalPid INTEGER, demangledName INTEGER NOT NULL, "
        "shortName INTEGER NOT NULL, mangledName INTEGER, graphNodeId INTEGER, graphId INTEGER, "
        "launchType INTEGER"),
    "CUPTI_ACTIVITY_KIND_RUNTIME": (
        "start INTEGER NOT NULL, end INTEGER NOT NULL, eventClass INTEGER NOT NULL, globalTid INTEGER, "
        "correlationId INTEGER, nameId INTEGER NOT NULL, returnValue INTEGER NOT NULL"),
    "CUPTI_ACTIVITY_KIND_MEMCPY": (
        "start INTEGER NOT NULL, end INTEGER NOT NULL, deviceId INTEGER NOT NULL, streamId INTEGER NOT NULL, "
        "correlationId INTEGER, globalPid INTEGER, bytes INTEGER NOT NULL, copyKind INTEGER NOT NULL, "
        "graphNodeId INTEGER"),
    "NVTX_EVENTS": (
        "start INTEGER NOT NULL, end INTEGER, eventType INTEGER NOT NULL, rangeId INTEGER, category INTEGER, "
        "color INTEGER, text TEXT, globalTid INTEGER, endGlobalTid INTEGER, textId INTEGER, domainId INTEGER"),
    "PROCESSES": "globalPid INTEGER, pid INTEGER, name TEXT",
    "TARGET_INFO_GPU": "id INTEGER NOT NULL, name TEXT, smCount INTEGER",
}


def _ar_op(tp: int, ar_backend: str, batch: int) -> list[tuple[str, int]]:
    """The kernels of one all-reduce op (fused with residual add + RMSNorm where vLLM fuses it)."""
    if ar_backend == "none":
        return [(FUSED_ADD_RMS_NORM, 2_500)]
    if ar_backend == "trtllm":
        return [(FI_ONESHOT, 6_000)] if batch <= 4096 else [(FI_TWOSHOT, 12_000)]
    if ar_backend == "mnnvl":
        return [(MNNVL_ONESHOT, 5_000)] if batch <= 64 else [(MNNVL_TWOSHOT, 9_000), (MNNVL_TAIL, 3_000)]
    if ar_backend == "custom":
        return [(CUSTOM_AR, 7_000), (FUSED_ADD_RMS_NORM, 2_500)]
    if ar_backend == "nccl":
        return [(NCCL_AR, 14_000), (FUSED_ADD_RMS_NORM, 2_500)]
    raise ValueError(f"unknown ar_backend {ar_backend!r}")


def kernels_for_step(tp: int, ar_backend: str, batch: int) -> list[tuple[str, int]]:
    """(name, duration_ns) of one decode step in launch order: the forward, then sample_tokens."""
    if (tp == 1) != (ar_backend == "none"):
        raise ValueError(f"ar_backend {ar_backend!r} does not match tp={tp}")
    scale = 1 + batch / 256

    def k(name: str, ns: float) -> tuple[str, int]:
        return name, int(ns * scale)

    ar = _ar_op(tp, ar_backend, batch)
    out = list(ar)                                    # embedding all-reduce
    for _ in range(LAYERS):
        out += [k(GEMM_WIDE, 12_000 / tp), k(CACHE_WRITE, 2_500), k(FA3, 9_000 / tp), k(GEMM_NARROW, 8_000 / tp)]
        out += ar                                     # o_proj
        out += [k(GEMM_WIDE, 30_000 / tp), k(SILU, 3_000), k(GEMM_NARROW, 16_000 / tp)]
        out += ar                                     # down_proj; the last one is fused with the final norm
    out.append((UNKNOWN, 1_000))
    out.append(k(LM_HEAD, 60_000 / tp))              # sample_tokens: compute_logits
    if tp > 1:
        out.append((NCCL_AG, 9_000))                  # logits all-gather
    out += [k(EXPONENTIAL, 3_000), k(ARGMAX, 4_000)]
    return out


def _short(name: str) -> str:
    base = name[5:] if name.startswith("void ") else name
    for sep in ("<", "("):
        base = base.split(sep, 1)[0]
    return base.rsplit("::", 1)[-1]


class _Writer:
    def __init__(self, con: sqlite3.Connection, omit: set[str]):
        self.con, self.omit = con, omit
        self.strings: dict[str, int] = {}
        self.rows: dict[str, list[tuple]] = {t: [] for t in TABLES}

    def sid(self, value: str) -> int:
        if value not in self.strings:
            self.strings[value] = len(self.strings) + 1
        return self.strings[value]

    def add(self, table: str, row: tuple) -> None:
        self.rows[table].append(row)

    def flush(self) -> None:
        self.rows["StringIds"] = [(i, v) for v, i in self.strings.items()]
        for table, cols in TABLES.items():
            if table in self.omit:
                continue
            self.con.execute(f"CREATE TABLE {table} ({cols})")
            rows = self.rows[table]
            if rows:
                marks = ",".join("?" * len(rows[0]))
                self.con.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
        self.con.commit()


def _lag(k: int, period: int) -> int:
    return int(period * (0.25 + 0.5 * ((k * 37) % 11) / 10))


def _layout(w: _Writer, ri: int, rank: dict, collide: bool) -> tuple[int, int]:
    """Writes one rank's rows; returns (first range start, last GPU end)."""
    pid, device, tp = rank["pid"], rank["device"], rank["tp"]
    gpid = pid << 24
    gtid = gpid | (pid & 0xFFFFFF)
    ks = kernels_for_step(tp, rank["ar_backend"], rank["batch"])
    split = next(i for i, (n, _) in enumerate(ks) if n == LM_HEAD)
    work = sum(d for _, d in ks)
    n_ops = sum(n in _PDL or n in (CUSTOM_AR, NCCL_AR) for n, _ in ks[:split])
    sync = -(-n_ops // tp) * SYNC_WAIT_NS if tp > 1 else 0   # the most spin-waits any rank adds
    # the input copy and its gap fit before GRAPH_LAUNCH_NS; PDL overlap only shortens the step
    period = GRAPH_LAUNCH_NS + work + sync + IDLE_GAP_NS
    n_keep = len(rank["steps"]) - rank.get("drop_last", 0)
    short = set(rank.get("short_ar_steps", ()))              # steps missing their first (embedding) all-reduce op
    n_first_ar = len(_ar_op(tp, rank["ar_backend"], rank["batch"]))
    cid = 0 if collide else ri * 1_000_000
    cpu = T0_NS + ri * RANK_SKEW_NS
    gpu = 0
    first = cpu
    graph_id = ri + 1

    def launch(api: str, t: int, dur: int) -> int:
        nonlocal cid
        cid += 1
        w.add("CUPTI_ACTIVITY_KIND_RUNTIME", (t, t + dur, 0, gtid, cid, w.sid(api), 0))
        return cid

    def kernel(name: str, dur: int, earliest: int, c: int, node: int | None, overlap: int = 0) -> None:
        nonlocal gpu
        start = max(gpu - overlap, earliest)
        w.add("CUPTI_ACTIVITY_KIND_KERNEL", (start, start + dur, device, 1, 7, c, gpid, w.sid(name),
                                             w.sid(_short(name)), w.sid("_Z" + _short(name)), node,
                                             graph_id if node is not None else None, 0))
        gpu = max(gpu, start + dur)

    for k, (nc, nct, ng, ngt) in enumerate(rank["steps"][:n_keep]):
        step_ks = ks[n_first_ar:] if k in short else ks
        step_split = split - n_first_ar if k in short else split
        lag = _lag(k, period)
        w.add("NVTX_EVENTS", (cpu, cpu + RANGE_NS, 59, None, None, None,
                              f"execute_context_{nc}({nct})_generation_{ng}({ngt})", gtid, None, None, 0))
        # input ids HtoD copy
        t = cpu + 1_000
        c = launch("cudaMemcpyAsync_v3020", t, 2_000)
        start = max(gpu, t + lag)
        w.add("CUPTI_ACTIVITY_KIND_MEMCPY", (start, start + 1_500, device, 7, c, gpid, 8 * rank["batch"], 1, None))
        gpu = start + 1_500
        # the forward: one CUDA graph launch, all node kernels share its correlationId
        t = cpu + GRAPH_LAUNCH_NS
        c = launch("cudaGraphLaunch_v10000", t, 8_000)
        op = 0
        for node, (name, dur) in enumerate(step_ks[:step_split], start=1):
            if name in _PDL or name in (CUSTOM_AR, NCCL_AR):
                if tp > 1 and op % tp == ri % tp:
                    dur += SYNC_WAIT_NS
                op += 1
            kernel(name, dur, t + lag, c, node, PDL_OVERLAP_NS if name in _PDL else 0)
        # sample_tokens: eager launches after the execute range
        for i, (name, dur) in enumerate(step_ks[step_split:], start=1):
            t = cpu + RANGE_NS + 2_000 * i
            c = launch("cudaLaunchKernel_v7000", t, 1_500)
            kernel(name, dur, t + LAUNCH_LATENCY_NS, c, None)
        cpu += period
    return first, gpu


def _processes_and_gpus(w: _Writer, ranks: list[dict], driver_pid: int) -> None:
    for r in ranks:
        w.add("PROCESSES", (r["pid"] << 24, r["pid"], "python3"))
    w.add("PROCESSES", (driver_pid << 24, driver_pid, "python3"))
    for device in sorted({r["device"] for r in ranks}):
        w.add("TARGET_INFO_GPU", (device, constants.H100_SXM.name, constants.H100_SXM.sm_count))


def build_trace_db(path: str, ranks: list[dict], measure: tuple[int, int] | None = None,
                   omit_tables: Sequence[str] = (), collide_correlation_ids: bool = True) -> None:
    """ranks: [{"pid", "device", "tp", "ar_backend", "batch", "steps": [[nc, nct, ng, ngt], ...], "drop_last": 0}]
    (optional "short_ar_steps": step indexes that lack their first all-reduce op).

    `measure` adds the driver's `tpprof:measure` range (a StartEndRange with a registered string).
    """
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    try:
        w = _Writer(con, set(omit_tables))
        for ri, rank in enumerate(ranks):
            _layout(w, ri, rank, collide_correlation_ids)
        driver_pid = max((r["pid"] for r in ranks), default=0) + 1
        if measure is not None:
            dtid = (driver_pid << 24) | driver_pid
            w.add("NVTX_EVENTS", (measure[0], measure[1], 60, 1, None, None, None, dtid, dtid,
                                  w.sid("tpprof:measure"), 0))
        _processes_and_gpus(w, ranks, driver_pid)
        w.flush()
    finally:
        con.close()


def trace_span(ranks: list[dict]) -> tuple[int, int]:
    """(first execute-range start, last GPU end + pad) over all ranks, ignoring drop_last:
    where the offline driver's `tpprof:measure` range would sit."""
    con = sqlite3.connect(":memory:")
    w = _Writer(con, set())
    spans = [_layout(w, ri, {**r, "drop_last": 0}, True) for ri, r in enumerate(ranks)]
    con.close()
    return min(s for s, _ in spans), max(e for _, e in spans) + MEASURE_PAD_NS
