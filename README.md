# vllm-tp2-profiling

What does a second H100 buy? This repository measures tensor parallelism
(TP=2) against two independent replicas (DP=2) and a single GPU (TP=1) for
Llama-3.1-8B-Instruct in bf16 on vLLM 0.30.0, on one 2× H100 SXM box. The
harness, `tpprof`, runs offline step-time sweeps, online serving sweeps, Nsight
Systems traces and all-reduce microbenchmarks, then turns the run records into
tidy CSVs, figures and a hypothesis report.

> **Status (2026-09-30):** the harness is complete and tested without a GPU:
> unit tests, integration tests against fake `vllm` / `nvidia-smi` / `nsys` /
> `torchrun` executables, a full P0–P2 dry run through analysis and report, and
> contract tests against the real vLLM 0.30.0 CLI in a CPU container. **No GPU
> run has happened yet, so no performance numbers exist yet.** The a-priori
> predictions are committed in [predictions.md](predictions.md) (git tag
> `predictions-v1`) before any hardware is rented.

## The question

Llama-3.1-8B-Instruct in bf16 (16.06 GB of weights) fits on one H100 80GB.
What does a second H100 buy, and what does it cost?

| Config | GPUs | What it should buy | What it pays |
|---|---|---|---|
| **TP1** | 1 | baseline, and the per-GPU normalizer | none |
| **TP2** | 2 | lower per-token latency: each GPU streams half the weights and half the KV cache | 65 all-reduces + 1 all-gather per step; per-step work that does not shrink |
| **DP2** | 2 | about 2× throughput and 2× KV capacity, with no communication | per-token latency stays that of TP1 |

The two deliverables:

1. The TPOT SLO `s*` below which TP2 gets more goodput out of two GPUs than
   two replicas do, and above which the replicas win.
2. A mechanism-level account, taken from traces, of why TP2 falls short of 2×:
   communication time, fixed per-step work, and GEMV efficiency.

## What was learned before any GPU

The design went through a research pass: 10 dimensions, each high-impact
finding re-checked by a second agent against the vLLM v0.30.0 source (commit
`ced6857afa0e`). Three findings changed the design.

1. **The default TP2 all-reduce is neither NCCL nor vLLM's custom all-reduce.**
   At the default optimization level (`-O2`) on Hopper, a compile pass fuses
   all 65 all-reduces of a step with the residual add and RMSNorm that follow
   them into FlashInfer `allreduce_fusion` kernels
   (`enable_allreduce_rms_fusion`, `vllm/config/vllm.py:178-202`). The flag
   `--disable-custom-all-reduce` only removes vLLM's custom all-reduce, which the
   fused path never uses, so on its own it changes nothing in the baseline. The
   planned "custom AR off = NCCL" ablation therefore became a four-arm ladder,
   AR0 → AR3, from the fused FlashInfer path down to pure NCCL. The fusion also
   depends on the install: it needs `nvcc` or `flashinfer-cubin`, so a plain
   `pip install vllm` on a box without `nvcc` runs unfused without saying so.
   The environment is therefore the official image, pinned by digest.
2. **`--enforce-eager` turns off torch.compile and every fusion, not just CUDA
   graphs** (`vllm/config/vllm.py:1546-1552`). The graph ablation is split in
   two: G1 sets `cudagraph_mode` to `NONE` and keeps compilation and fusion; G2
   is full eager.
3. **vLLM's built-in data parallelism is the replica control.** For a dense
   model, `--data-parallel-size 2` runs two fully independent engines (no
   lockstep) behind one API server that routes each request by queue load.
   That is the realistic two-replica setup, so it is the primary DP2 config.
   Two separate servers with the client splitting requests at random became a
   sensitivity arm (A-RAND).

The model is `NousResearch/Meta-Llama-3.1-8B-Instruct` at revision
`d10aef7999a2b5ba950ab3974312feeedbfe0b77`. It is ungated, and its weights,
`config.json`, `generation_config.json` and `tokenizer.json` are byte-identical
to Meta's `meta-llama/Llama-3.1-8B-Instruct` at `0e9e39f2`, checked by comparing
git blob oids in both repositories. Only `tokenizer_config.json` (the chat
template) differs, and raw-prompt `/v1/completions` requests do not use it. The
`unsloth` mirror was rejected because its `config.json` is modified.

An adversarial review of the spec then produced 40 findings and 32 amendments
before any code was written. [WORKLOG.md](WORKLOG.md) lists the ones that
changed the measurements.

## The design

### Configurations

| id | GPUs | Launch | Role |
|---|---|---|---|
| **TP1** | 1 | `vllm serve M -tp 1 --distributed-executor-backend mp`, `CUDA_VISIBLE_DEVICES=0` | baseline |
| **TP2** | 2 | `vllm serve M -tp 2 --distributed-executor-backend mp` | tensor parallel |
| **DP2** | 2 | `vllm serve M -tp 1 -dp 2 --api-server-count 1` | primary replica control |
| DP2-rand (P2) | 2 | two `vllm serve M -tp 1`, on GPU0 and GPU1, two clients at half the rate with different seeds | sensitivity arm: random split |

Offline, DP2 is derived rather than run: a DP2 batch of B requests is two
independent TP1 engines at B/2, so the TP1 decode grid includes the half-batch
points.

### Pinned engine settings

Every engine flag and environment variable has one source,
`tpprof/engine.py`. It renders the `vllm serve` argv and the offline driver's
argv, and the offline driver parses its argv with vLLM's own `EngineArgs`.
Every default that could differ between offline and online use is written out:

- `--dtype bfloat16 --max-model-len 9216 --gpu-memory-utilization 0.90`
- `--max-num-seqs 1024 --max-num-batched-tokens 8192 --block-size 16 --kv-cache-dtype auto --seed 0`
- `--no-enable-prefix-caching` (on only in arm PCon), `--enable-chunked-prefill`, `--async-scheduling`, `--stream-interval 1`
- `--no-enable-dbo --no-enable-batch-sharded-sampling`, which keep the per-step collective count at 65 + 1
- `--performance-mode balanced --optimization-level 2`
- `--attention-backend FLASH_ATTN --attention-config {"flash_attn_version":3}` (FA3 on SM90)
- `--generation-config vllm`, which stops the server from applying the model's T=0.6 / top_p=0.9 sampler defaults
- `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}`, plus `"pass_config":{"fuse_allreduce_rms":true}` for TP2
- `--fail-on-environ-validation`, so an unknown `VLLM_*` variable is an error
- environment: `VLLM_WORKER_MULTIPROC_METHOD=spawn`, the all-reduce selection variables at their defaults, `HF_HUB_OFFLINE=1`; `VLLM_USE_V2_MODEL_RUNNER` stays unset

After every engine start, `tpprof/logparse.py` reads the engine log and fails
the run unless the expected lines are present: the vLLM version, the chunked
prefill budget, `Using V2 Model Runner`, the FLASH_ATTN backend with
FlashAttention 3, the KV cache size, and for TP2 the all-reduce backend list,
the enabled `allreduce_rms` fusion and the FlashInfer workspace line. Known
failure strings (a disabled fusion, a disabled custom all-reduce, a failed
symmetric-memory init, a full `/dev/shm`) also fail the run. The normal
FlashInfer fallback from the `mnnvl` to the `trtllm` backend is recorded, not
failed.

### Workloads

**Offline** (`tpprof/offline.py`, the `LLM` class, mirroring `vllm bench latency`):
- Prefill: batch 1 × {512, 2048, 8192} input tokens, output 1; 5 warmup + 20 measured iterations.
- Decode: batch {1, 2, 4, 8, 16, 32, 64, 128}, input 1024, two output lengths
  64 and 320; step time = (median latency at 320 − median latency at 64) / 256.
  Chunked-prefill mixing and the shrinking-batch tail are identical in both
  runs and cancel in the difference. The two lengths are interleaved, 3 warmup
  + 10 measured each, with a bootstrap CI of the difference.
- A cross-check against the official `vllm bench latency` at batch 8.
- KV capacity from each engine's `GPU KV cache size: N tokens` log line.

**Online** (`vllm bench serve`, random 1024-token prompts, 256 output tokens,
`--ignore-eos`, T=1.0, top_p=1.0, Poisson arrivals):
- Saturation first: 3 runs at `--request-rate inf` per config. Saturation
  throughput μ comes from the token-emission timeline, between its 10th and
  90th percentile.
- A rate grid per config at {0.20, 0.40, 0.60, 0.75, 0.85, 0.95} × μ, written
  once to `results/raw/rate_grid.json` and never recomputed.
- 3 rounds, with the config order rotating as a Latin square, so drift over the
  session spreads across configs.
- Goodput G(c, s) per round: the rate at which 90% of requests meet TTFT ≤ 1 s
  and TPOT ≤ s, for s from 5 to 60 ms. `s*` is where G(DP2, s) − G(TP2, s)
  changes sign.
- A run is rejected, never dropped silently, if any request failed or has the
  wrong input or output length. It is flagged if preemptions increased, a GPU
  throttle reason appeared, or the API server or client used more than 80% of
  a core.

### Traces

Nsight Systems 2026.5.1 with `--cuda-graph-trace=node` (kernels inside CUDA
graphs are invisible otherwise), capturing between vLLM's `cudaProfilerStart`
and `cudaProfilerStop`. `tpprof/traces.py` reads the exported SQLite:
- A completeness gate: one process per GPU, equal per-step NVTX range and
  all-reduce counts on every rank, kernels on every rank in the last step. On
  failure the trace is rerun once with `--capture-range=none`; on a hang, once
  with `VLLM_ALLREDUCE_USE_SYMM_MEM=0`.
- Kernels are assigned to a step by the CPU time of the runtime call that
  launched them, not by GPU start time. With async scheduling the CPU runs up
  to one step ahead of the GPU, and GPU-time windows would split steps.
- Kernels are grouped into categories (all-reduce, all-gather, GEMM,
  attention, norm/activation/RoPE, sampling, memcpy, other), with busy time as
  the union of intervals. The unclassified share must stay below 1%.

### Communication microbenchmarks

| id | What | Sizes |
|---|---|---|
| M2 (P0) | vLLM's own `benchmark_fused_collective.py` (vendored, v0.30.0): the FlashInfer fused all-reduce + RMSNorm kernel that the TP2 baseline uses, vs vLLM all-reduce + RMSNorm | 1 to 8192 tokens (8 KiB to 64 MiB) |
| M3 (P0 default, P2 variants) | `tpprof/comm_bench.py`: a `torch.distributed` NCCL all-reduce loop, eager and CUDA graph; P2 adds per-function `NCCL_ALGO` × `NCCL_PROTO` variants | 2 KiB to 256 MiB |
| M1 (P2) | vLLM's `benchmark_device_communicators.py` (vendored): custom AR, PyNccl, symmetric memory, FlashInfer standalone | 1 to 8192 tokens |
| M4 (P2) | nccl-tests v2.20.0 `all_reduce_perf`, built against the image's NCCL | 8 B to 256 MiB |

α and β for the model's a-posteriori stage come from M2 for the default path
and from M3 for pure NCCL; M1 and M4 are cross-checks.

### Ablations

| Arm | Config | Change |
|---|---|---|
| AR0 | TP2 | none: FlashInfer fused all-reduce + RMSNorm |
| AR1 | TP2 | fusion off: standalone FlashInfer all-reduce, separate RMSNorm |
| AR2 | TP2 | AR1 + FlashInfer and symmetric-memory all-reduce off: vLLM custom all-reduce below 8 MiB, NCCL above |
| AR3 | TP2 | AR2 + `--disable-custom-all-reduce`: pure NCCL |
| G1 | TP1, TP2 | `cudagraph_mode: NONE`, compilation and fusion stay on |
| G2 | TP1, TP2 | `--enforce-eager`: no compile, no fusion, no graphs |
| PCon | TP2 | prefix caching on, with a shared 512-token prefix |
| EXECuni | TP1 | `--distributed-executor-backend uni` instead of the baseline `mp` |
| FIBtrtllm | TP2 | FlashInfer `trtllm` backend, only if the baseline chose `mnnvl` |
| API2 | TP2, DP2 | `--api-server-count 2`, saturation only |
| A-RAND | DP2-rand | two servers with a random split, 1 round |

### Tiers

The run matrix is split so that a partial rental still gives a coherent result:
- **P0** decides the offline hypotheses: preflight, smoke starts, M2 and M3,
  the TP1/TP2 offline grids, the AR3 and G2 cells, the bench-latency
  cross-check and the P0 traces.
- **P1** is the online study: saturation, then 3 rounds × 3 configs × 6 rates,
  and the tokenizer benchmark.
- **P2** holds the remaining ablations, traces and microbenchmarks.

## Confounder ledger

The evidence paths are filled in from the run records after the GPU run.

| # | Confounder | Control | Evidence captured |
|---|---|---|---|
| 1 | CUDA-graph mode | pinned FULL_AND_PIECEWISE; arms G1/G2 | capture log lines; trace `graphId` per kernel |
| 2 | Prefix caching | off plus random prompts; arm PCon | `/metrics` prefix-cache counters before/after |
| 3 | Chunked prefill / batch budget | pinned 8192 everywhere; prefill sizes ≤ 8192; two-length decode difference | `Chunked prefill is enabled with max_num_batched_tokens=8192.` |
| 4 | Warmup / cold caches | one throwaway start per config (compile, FlashInfer, Triton caches); offline 5/3 warmups; online `--num-warmups 16` | per-iteration latencies kept; startup timings |
| 5 | Tokenizer/detokenizer CPU | offline uses token-ID prompts; online includes it (realistic); `tokbench` measures it; per-process CPU sampling | `tokbench.json`, `cpu.csv` |
| 6 | All-reduce implementation | baseline declared (FlashInfer fused); A-AR ladder | backend log lines; kernel names per step |
| 7 | Scaling baseline | TP1 on the same box, same flags; DP2 as the real alternative | same `env.json` |
| 8 | Version drift | image digest, vLLM 0.30.0, torch 2.13.0, NCCL, FlashInfer, driver, nsys | `env.json`, `pip freeze`, log lines |
| 9 | NVLink topology | NV18 hard gate | `topo.txt`, `nvlink.txt`, M1/M3 busbw |
| 10 | Clocks / thermals / power | cannot lock (container) → 200 ms monitor during every run; throttle-flagged runs | `gpu.csv` |
| 11 | KV preemption | `--max-num-seqs 1024` and the per-config rate grid; detect | `/metrics` preemption counters before/after |
| 12 | Arrival process and seeds | Poisson, seeds recorded; DP2-rand halves use different seeds | client argv; result JSON |
| 13 | Sampler | T=1.0, top_p=1.0 everywhere; `--generation-config vllm`; greedy never mixed in | server log (no override warning); request bodies |
| 14 | Engine-default drift (offline vs online) | every engine flag pinned from one source | `effective_config.json` from logs |
| 15 | Executor asymmetry (uni vs mp) | TP1 pinned to `mp` like TP2; arm EXECuni | config log line |
| 16 | Box hygiene | no foreign GPU processes, memory < 1 GiB before every start; page cache warmed by a throwaway start | `preflight.json`, per-run pre-check |
| 17 | Frontend capacity | 1 API server in every config; arm API2 | CPU% of the API-server process and the client |
| 18 | Drift over the session | online rounds interleave configs | round index in every row |

## Hypotheses

The bands below are copied from [predictions.md](predictions.md), which
`tpprof/model.py` generates (`python -m tpprof predict`) and a test keeps
byte-identical to the model. A band is the minimum and maximum over the
optimistic, central and pessimistic constant sets. These are predictions made
before any measurement, not results.

| id | Statement | Predicted band (central) | Decision rule |
|---|---|---|---|
| H1 | TP2 decode step speedup over TP1 at batch 1 | 1.499–1.671 (1.555) | hit if the measured TP1/TP2 step ratio is inside the band |
| H2 | TP2 scaling efficiency e = speedup / 2 rises with tokens per step: decode e(128) − e(1), prefill e(8192) − e(512) | decode 0.052–0.067 (0.067); prefill 0.080–0.100 (0.099) | hit if both differences are > 0 with a bootstrap CI excluding 0 |
| H3 | DP2 saturation throughput exceeds TP2's | DP2 / TP2 = 1.126–1.220 (1.155) | hit if μ(DP2) > μ(TP2) in every saturation run |
| H4 | A goodput crossover exists: TP2 wins below `s*`, DP2 above | `s*` = 13.92–15.30 ms (14.31) | hit if the measured `s*` range overlaps the band |
| H5 | TP2's KV capacity per engine relative to TP1's | 2.258–2.269 (2.269) | hit if the warm-boot ratio is inside the band |
| H6 | Per rank per step: exactly 65 all-reduce ops and 1 all-gather, in every arm | structural, no band | hit if exact in ≥ 99% of traced steps |
| H7 | GPU idle fraction, TP2 decode batch 1: < 10% in the baseline, > 30% in G2, ordering baseline < G1 < G2 | no model band | hit if all three statements hold |
| H8 | AR3 (pure NCCL, unfused) slows TP2 decode at batch 1 relative to the fused default | AR3 / AR0 = 1.050–1.092 (1.060) | hit if the measured ratio is inside the band |

A miss is a result, not a failure: every miss gets a WORKLOG entry naming the
model term that was wrong.

## How to reproduce

The box procedure, command by command, with time and cost per tier, is in
[RUN_ON_GPU.md](RUN_ON_GPU.md).

Without a GPU (macOS or Linux, Python ≥ 3.10):

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
make test        # unit and integration tests against the fakes
make dry-run     # the full P0-P2 matrix against the fakes, then analyze and report (FAKE watermark)
make contract    # contract tests against the real vLLM 0.30.0 CLI (needs Docker)
make predict     # regenerate predictions.md from tpprof/model.py
```

## Repository layout

```
README.md, RUN_ON_GPU.md, WORKLOG.md
predictions.md              a-priori predictions, generated by tpprof/model.py (tag predictions-v1)
tpprof/                     the harness (python -m tpprof <subcommand>)
  constants.py              pinned versions, image digest, model revision, GPU specs, workload sizes
  model.py, predict.py      prediction model and the predictions.md renderer
  engine.py, helpflags.py   engine configs and arms; argv/env rendering; --help=all flag checks
  matrix.py                 RunSpec, run_id, tiers P0-P2, rate grid, time and cost estimator
  runner.py, cli.py         resumable execution of the matrix; the CLI
  server.py, client.py      vllm serve lifecycle; vllm bench serve argv and runner
  offline.py                offline driver (LLM class, or a fake engine in tests)
  profile.py, traces.py, kernels.py   nsys runs, trace SQLite analysis, kernel categories
  comm_bench.py, vendored.py, nccltests.py   microbenchmarks M3, M1/M2, M4
  preflight.py, envcapture.py         hard gates and env.json
  procs.py, monitor.py      process groups and timeouts; GPU and CPU monitors
  logparse.py, promparse.py, results.py, stats.py, goodput.py   parsing, validity, statistics
  analyze.py, plots.py, report.py     tidy CSVs, figures, results/SUMMARY.md
scripts/                    runpod.py (RunPod REST), sync_to_box.sh, bootstrap_box.sh
third_party/vllm_benchmarks/  vendored vLLM v0.30.0 benchmark scripts (Apache-2.0, see SOURCE.md)
tests/                      unit, integration and contract tests; fixtures from the real 0.30.0 CLI
  fake_bin/                 fake vllm, nvidia-smi, nsys, torchrun
docker/                     CPU image with the real vLLM 0.30.0 CLI for the contract tests
results/                    run records from the box (after the GPU run)
```

## Limitations known in advance

- **No clock locking.** Rented containers cannot lock GPU clocks. Clocks,
  power, temperature and throttle reasons are sampled every 200 ms during every
  run, and throttled runs are flagged, not corrected.
- **Node-level trace overhead.** `--cuda-graph-trace=node` adds overhead on
  H100. Traces are used for counts and shares, never for headline latency.
  Each traced step time is compared with the untraced median, and the ratio is
  reported.
- **One box, one provider.** All numbers will come from one rented 2× H100 SXM
  slice on RunPod Secure Cloud, in one session. Box-to-box variation is not
  measured.
- **DP routing is vLLM's internal load balancer.** DP2 results include vLLM's
  queue-load routing. The random-split arm (A-RAND) shows how much that
  routing matters, but other routers are not tested.

## License

MIT, see [LICENSE](LICENSE). The vendored vLLM benchmark scripts in
`third_party/vllm_benchmarks/` keep their Apache-2.0 headers.

**Built with Llama.** The experiments run Meta's Llama 3.1 8B Instruct. Llama
3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta
Platforms, Inc. All Rights Reserved. This repository does not contain or
redistribute the model weights: `scripts/bootstrap_box.sh` downloads them on
the rented box, and their use is subject to that license and Meta's Acceptable
Use Policy.
