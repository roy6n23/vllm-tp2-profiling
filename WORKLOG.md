# WORKLOG

Dated, append-only. Record what broke, how it was found, and what it taught.
Keep entries factual, including the ones that make me look slow.

## 2026-09-30: research pass and spec review, before any code

**Context.** The first design for the question "what does a second H100
buy?" assumed the textbook picture: TP2 all-reduces go through NCCL or
vLLM's custom all-reduce, `--enforce-eager` is the CUDA-graph switch, and DP2
means two servers with the client splitting the load. Before writing code I
ran a research pass over 10 dimensions (engine arguments, the TP communication
path, bench CLIs and result formats, server observability, nsys, NCCL
microbenchmarks, model and hardware constants, release and environment,
rental logistics, a CPU container for the real CLI). Each high-impact finding
was checked again by a second agent against the vLLM v0.30.0 source at
`ced6857afa0e`.

**What the source says, and what changed.**
- **The default TP2 all-reduce is FlashInfer, fused with residual add +
  RMSNorm.** At `-O2` on SM90 with FlashInfer usable,
  `enable_allreduce_rms_fusion` (`vllm/config/vllm.py:178-202`) turns on a
  compile pass that replaces all 65 all-reduce → RMSNorm pairs of a step with
  FlashInfer `allreduce_fusion` kernels. NCCL and vLLM's custom all-reduce are
  not on the per-layer path at all. `--disable-custom-all-reduce` only clears
  the custom all-reduce, which the fusion pass never checks, so the planned
  "custom AR off = NCCL" ablation would have changed nothing. It is now a
  4-arm ladder: AR0 fused (the default), AR1 fusion off, AR2 FlashInfer and
  symmetric memory off (vLLM custom all-reduce), AR3 also
  `--disable-custom-all-reduce` (pure NCCL, verified by the
  `Using ['PYNCCL'] ...` log line).
- **`--enforce-eager` also disables torch.compile and every fusion**
  (`vllm/config/vllm.py:1546-1552` sets compilation mode and cudagraph mode
  both to NONE). An eager run therefore changes graphs, compilation and the
  all-reduce kernel at once. The graph ablation is split: G1 is
  `cudagraph_mode: NONE` with compilation on, G2 is `--enforce-eager`.
- **Built-in DP2.** `--data-parallel-size 2` for a dense model runs
  independent engine ranks (no lockstep) and routes each request to the rank
  with the lowest queue score. That is the realistic replica setup, so it is
  the primary DP2 config; the two-server random split is a P2 arm.
- **The model mirror.** The `unsloth` mirror's `config.json` is modified
  (scalar eos, a pad token, a `head_dim` key). The NousResearch mirror at
  `d10aef79` is byte-identical to Meta's repo by git blob oid: the 4 LFS
  pointer files (which hold the content sha256), `config.json`,
  `generation_config.json`, `tokenizer.json`, `special_tokens_map.json` and
  the safetensors index are equal. Only `tokenizer_config.json` (the chat
  template) differs, and raw-prompt completions do not use it.
- **Install sensitivity.** The fusion needs `has_flashinfer()`, which needs
  `nvcc` or `flashinfer-cubin`. A plain `pip install vllm` without `nvcc` runs
  unfused, and it also gets a different NCCL (2.29.7 from pip vs 2.30.7 in
  the image). The environment is the official image, pinned by digest.

**The adversarial spec review** (three lenses: facts, methodology,
implementability) produced 40 findings; 32 became binding amendments. Two of
them changed what gets measured:
- **`--max-num-seqs` back to 1024** (vLLM's default, made explicit). The draft
  pinned 512 "to match the CUDA-graph capture ceiling". That reason was wrong:
  the capture ceiling is min(2 × max_num_seqs, 512), which is 512 either way.
  And 512 is applied per engine, so it would have capped TP2 below the number
  of requests its KV cache can hold while each DP2 rank stayed KV-limited.
  That would have biased the saturation comparison (H3) and with it the upper
  half of the goodput crossover (H4).
- **Kernel-to-step assignment is by CPU launch time.** Async scheduling is on,
  so the worker's CPU runs up to one step ahead of the GPU. Step windows are
  CPU-side NVTX ranges. Placing kernels by GPU start time would cut GPU steps
  in half at the window boundaries (counts like 30 and 35 instead of 65) and
  fail H6 on correct code. Each kernel is now placed by the timestamp of the
  runtime call that launched it (for graph nodes, the `cudaGraphLaunch`), and
  GPU busy/idle is computed on the GPU timeline over the whole measured
  window.

**Lesson.** Every one of these came from reading the pinned source, not the
docs or my memory of older vLLM versions. Each ablation now has a log line or
a kernel-name check that proves the arm did what it claims.

## 2026-09-30: the review found a gate that would have failed on a correct download

**Problem.** The spec's model gate said "4 shards totaling 16,060,522,496 B".
That number is `total_size` from `model.safetensors.index.json`: the tensor
payload, 8,030,261,248 parameters × 2 bytes. Each safetensors file also
carries a JSON header, so the 4 files on disk add up to 16,060,556,376 B,
33,880 B more. A gate that summed file sizes would have failed every correct
download. Combined with the stop-loss rule ("preflight fails its hard gates
twice → destroy the pod"), it could have thrown away a healthy box.

**How it was found.** Two of the three review lenses flagged it
independently (implementability as critical, facts as important). The tell is
that the number is exactly 2 bytes per parameter, which no set of files with
headers can be.

**Fix.** The gate checks each of the 4 shard files at its exact size
(4,976,698,672 / 4,999,802,720 / 4,915,916,176 / 1,168,138,808 B) and,
separately, that the index's `total_size` is 16,060,522,496. Both totals live
in `tpprof/constants.py` under names that say which is which. There is also a
recorded `--skip-gate NAME` override, so a gate bug found on the box does not
trigger the destroy-the-pod rule.

**Lesson.** A size from a manifest is not a size on disk. Before a number
becomes a hard gate, check which quantity it measures.

The same review also caught that the normal FlashInfer `mnnvl` → `trtllm`
fallback prints warnings that the draft listed as failure strings, which would
have rejected every TP2 start on a box without NVSwitch multicast. The
fallback warnings are now recorded (`fi_backend_fallback`), and only the real
failure strings fail a run. Each module then went through code review before
merge; the fix rounds are in the git history.

**Open for the first GPU run:**
- [x] Quick preflight on a real box: NV18, 700 W, fabric registered, `/dev/shm`; record the multicast attribute. (2026-10-02: all pass; multicast True)
- [x] Which FlashInfer backend the TP2 baseline picks (`mnnvl` or `trtllm`), from the workspace log line. It decides whether A-FIB and the NVLS M3 variant run. (`mnnvl`, no fallback)
- [x] The effective-config log lines on the box match the patterns taken from source, per arm (backend list, `Enabled custom fusions: allreduce_rms`, the workspace line). (no `effective_config` failure on the US-NE-1 box)
- [x] Real kernel names in the P0 traces vs the categorizer: `traces --check` must show an unclassified share < 1% and H6 (>= 99% of steps exact). The name fixtures come from source, not from a real trace. (unclassified <= 0.29%, H6 100%)
- [x] Trace overhead ratio (traced vs untraced step time) with node-level graph tracing. (traced mean step 1.1-6.4x the untraced median, varying by run; traces are used for counts and shares only)
- [x] The model file gate passes on the real download.
- [x] TP1 decode batch 1 on GPU1 vs GPU0 (the DP2 derivation is flagged if they differ by ≥ 1%). (0.40%)
- [x] The offline driver vs `vllm bench latency` cross-check at batch 8 (within 3% expected). (TP1 -0.01%, TP2 +2.02%)
- [ ] Estimated vs actual wall time per tier; update the estimator if it is off. (P0 0.7 h vs 1.3 h estimated, P1 2.2 vs 2.3, P2 1.2 vs 2.1: comm and trace runs are overestimated; estimator not updated yet)
- [ ] The a-posteriori fit: memory efficiency and fixed per-step time from TP1 only, α and β from M2 and M3; TP2 as an out-of-sample prediction.
- [x] Mark H1–H8 hit or miss against `predictions.md`, with one entry here per miss. (H5 and H7 missed, entry below)
- [x] Fill in the evidence paths of the confounder ledger. (results/SUMMARY.md, "Confounder evidence")

## 2026-10-02: the GPU run (three boxes, one usable)

**Boxes.** The first two RunPod pods (AP-IN-1) passed the quick preflight but
every TP2 engine died in `ncclCommInitRank`: NVLS multicast binding failed with
CUDA error 401 ("the operation cannot be performed in the present state"),
while the same all-reduce worked with `NCCL_NVLS_ENABLE=0`. The `multicast`
check only read the attribute. The full preflight now runs a real two-GPU
all-reduce (`nccl_allreduce`) and, if it fails, retries with NVLS off to name
the fault; it caught the second box. The third pod (US-NE-1) was clean. No
result comes from the first two.

**Harness fixes found on the box** (each with a test that failed first):
- The trace completeness gate counted vLLM's empty scheduler step
  (`execute_context_0(0)_generation_0(0)`, opened after the last request
  finishes, no kernels) as a truncated step and failed every TP1 trace.
- `flashinfer_jit_cache` compared versions exactly; the image ships `+cu130`.
- `scripts/runpod.py` got 403 / Cloudflare 1010 for urllib's default User-Agent.
- With `--api-server-count 2`, vLLM 0.30.0 disables stats logging and exposes no
  `/metrics` gauges, so both API2 sessions failed in the gauge poller. Such
  servers now run on the counters alone (`no_gauges`); the two sessions were
  rerun.
- The report paired the median of the round s\* values with the bootstrap CI of
  the pooled-request s\*, a different estimator (17.69 vs CI 18.82-19.03 around
  18.93). Both are now reported, each with its own label.
- The client CPU p90 included 10-15 s of client startup at 100%+, which put
  ~100% on every run; it is now taken over the benchmark phase (12-67%).

**H5 miss (KV capacity ratio 2.283 vs 2.258-2.269).** The wrong term is the
non-KV memory per GPU. Measured from the engines' available-KV lines:
71.20 GiB requested - weights - available KV = 2.51 GiB for TP1 and 2.40 GiB
for TP2. The model's central values were 4.82 and 5.40 GiB, and every constant
set gave TP2 more non-KV memory than TP1 (NCCL and communication buffers). On
this box TP2 has less: halving the per-GPU activations saves more than the
communication buffers cost. The band was narrow because it only varied the
size of that overhead, not the sign of TP2 - TP1.

**H7 miss (G2 idle 0.252, predicted > 0.30).** The other two statements held
(base 0.039 < 0.10; 0.039 < 0.208 < 0.252). The threshold assumed `--enforce-eager`
only adds CPU launch gaps on top of the same GPU work. From the traces, GPU
busy time per step is about 4.08 x (1 - 0.039) = 3.9 ms in the baseline and
13.8 x (1 - 0.252) = 10.3 ms in G2, so eager mode also does about 2.6x the GPU
work (no compile, no fusion). More GPU work per step leaves a smaller idle
share. Not explained yet: G1 (graphs off, fusion kept) also shows about
11.1 x (1 - 0.208) = 8.8 ms of GPU busy time. Next step: compare kernel counts
and categories between the base, G1 and G2 traces.

**H4 is a hit only through round 2.** The rule is range overlap: 12.31-18.04
overlaps 13.92-15.30. Rounds 1 and 3, the median and the pooled value are
2.4-3.6 ms above the band. Round 2's TP2 run at 29.66 req/s (seed 2003) had
arrivals about 18% above the mean for ~18 s (deciles 7-8). There, 80% of
requests had TPOT above 15 ms, so its attainment at 15 ms was 0.813. The
model's s\* is 2.4-3.6 ms too low (median and pooled value vs the band's top).

**Cost.** RunPod billed $53.16 for the three pods (pod 3: 6.8 h, of which
about 1.4 h was idle while nobody drove the session).

**Open:**
- [ ] G1's GPU busy time (above).
- [ ] Update the estimator for comm and trace runs.
- [ ] The a-posteriori fit: memory efficiency and fixed per-step time from TP1 only, α and β from M2 and M3; TP2 as an out-of-sample prediction.
