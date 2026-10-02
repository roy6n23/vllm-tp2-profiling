# Running on the GPU box

The full procedure for one rental of 2× H100 SXM on RunPod Secure Cloud, in
order. Commands marked **Mac** run in a checkout of this repository on your
own machine; commands marked **box** run over SSH on the pod. Replace `<id>`,
`<ip>` and `<port>` with the values that steps 1 and 2 print.

Every long phase runs inside `tmux`, so a dropped SSH connection does not stop
it. Reattach with `tmux attach -t tp2`; detach with `Ctrl-b d`.

## Time and cost

`python -m tpprof matrix --estimate` is the authoritative estimate. Run it on
the box before each block (step 4). Before any measurement it uses the
model's saturation rates; once P1 has measured them, the P1 and P2 serving
sessions are sized from `results/raw/rate_grid.json`. At the time of writing it
prints:

| Tier | Runs | Hours | Cost at $6.98/h |
|---|---|---|---|
| P0 | 21 | 1.32 | $9.21 |
| P1 | 13 | 2.28 | $15.91 |
| P2 | 26–28 | 1.98–2.09 | $13.82–14.62 |
| **P0–P2** | 60–62 | **5.58–5.69** | **$38.94–39.74** |

P2 has 28 runs if the TP2 smoke start picks the FlashInfer `mnnvl` backend
(that adds the FIBtrtllm arm and the NVLS variant of M3), 26 otherwise.

The same rental also carries the Liger-Kernel and Project 1 GPU blocks (step
6), about 2.5 h, and the bootstrap takes about 30 min. The whole session is
about 8.7 h, roughly $61 at $6.98/h. **The hard cap is $80.** Stop-loss rules:

- If the preflight fails its hard gates twice, terminate the pod and switch machines.
- If P0 overruns its estimate by 50%, skip P2.
- Stop at the last completed tier when the $80 cap comes near. Every tier
  leaves a coherent partial result.

## 0. Prerequisites

- A RunPod account with at least $80 of credit, and a RunPod API key.
- An SSH key pair. `scripts/runpod.py create` reads `~/.ssh/id_ed25519.pub`
  by default; pass `--pubkey PATH` for another key.
- `rsync`, `ssh` and Python ≥ 3.10 on the Mac, and a local venv for the
  report (step 10):

**Mac:**

```bash
cd vllm-tp2-profiling
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

Keep this venv active in the Mac terminal for the whole procedure.

## 1. Create the pod

**Mac:**

```bash
export RUNPOD_API_KEY=...
python scripts/runpod.py create
python scripts/runpod.py wait <id>
```

`create` prints `created pod <id>`. `wait` polls every 10 s until the pod has a
public IP and a mapped SSH port, then prints `ssh -p <port> root@<ip>`. The pod
is billed from `create` on. It runs the pinned image
`vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
on 2× `NVIDIA H100 80GB HBM3`, with `allowedCudaVersions ["13.0"]` (host driver
≥ 580), a 200 GB volume at `/workspace`, and sshd started from your public key.

## 2. Copy the code to the box

**Mac:**

```bash
scripts/sync_to_box.sh <ip> <port>
```

This rsyncs this checkout to `/workspace/vllm-tp2-profiling/` and, if present,
`~/Documents/Personal/triton-fa2-forward` to `/workspace/triton-fa2-forward/`
(set `TPPROF_FA2_DIR` if that checkout lives elsewhere).
It uses `--delete`: a re-sync removes box files that are missing on the Mac,
including run records under `results/`. **Pull `results/` back (step 9) before
any re-sync during the rental.**

## 3. Bootstrap the box

**Box:**

```bash
ssh -p <port> root@<ip>
tmux new -s tp2
bash /workspace/vllm-tp2-profiling/scripts/bootstrap_box.sh
```

The bootstrap runs, in order: the quick preflight (GPU name, count and SMs,
driver, NV18, 700 W, fabric, `/dev/shm`, disk, CPUs, open-files limit), the
nsys 2026.5.1 install, `pip install --no-deps -e .` into the image Python, the
model download
(`NousResearch/Meta-Llama-3.1-8B-Instruct@d10aef7999a2b5ba950ab3974312feeedbfe0b77`
to `/workspace/models/llama31-8b-instruct`), the nccl-tests build in the
background, `/workspace/tpprof.env`, and the full preflight. It is safe to
rerun.

**If the quick preflight fails, the box is not usable.** On the Mac:

```bash
python scripts/runpod.py terminate <id>
```

Then start again at step 1. See Troubleshooting for the one exception (NV12).

## 4. Check the plan

**Box** (inside tmux):

```bash
source /workspace/tpprof.env
cd /workspace/vllm-tp2-profiling
python -m tpprof matrix --estimate
```

## 5. P0

**Box:**

```bash
python -m tpprof run --tier P0 2>&1 | tee -ai results/p0.log
python -m tpprof traces --check
```

`tee -i` ignores Ctrl-C, so the runner's cleanup lines still reach the log
after you stop a run.

`traces --check` prints the completeness gate, the unclassified kernel share
and the H6 counts for every trace, and exits 1 if any unclassified share is
≥ 1% or if fewer than 99% of a trace's steps have exactly 65 all-reduce ops
and 1 all-gather (H6, counted from `trace.sqlite`). **It must pass before P1.** If it fails, fix the
kernel regexes in `tpprof/kernels.py` while the box is up (see
Troubleshooting).

## 6. Liger-Kernel and Project 1 blocks

These run on the same box between P0 and P1 and use both GPUs, so no `tpprof
run` may be active. Their commands are in the private rental-day runbook,
which is not part of this repository.

## 7. P1, then P2

**Box:**

```bash
python -m tpprof run --tier P1 2>&1 | tee -ai results/p1.log
python -m tpprof matrix --estimate
python -m tpprof run --tier P2 2>&1 | tee -ai results/p2.log
```

Run P2 only as the budget allows (see Time and cost). P1 writes
`results/raw/rate_grid.json` once its saturation runs are done; the P1 sweeps
and the P2 serving arms read it.

**Resuming.** After any interruption, rerun the same `run` command. Runs with a
`done.json` are skipped. A run stopped by Ctrl-C, by `kill` (SIGTERM) or by a
closed tmux window (SIGHUP) is marked failed with reason `interrupted`, and the
rerun redoes it. So does a run that a crash or `kill -9` left with no record,
once the leftover processes tagged with it are killed. A failed preflight is
also rerun every time. Add `--retry-failed` to redo the other failed runs:

```bash
python -m tpprof run --tier P1 --retry-failed 2>&1 | tee -ai results/p1.log
```

## 8. Analyze on the box

**Box:**

```bash
python -m tpprof analyze
```

This writes `results/tidy/*.csv` and `results/tidy/hypotheses.json` from the
run records. Run it before step 9, which compresses the trace SQLite files.

## 9. Pull the results back

**Box**, compress the trace SQLite files first:

```bash
cd /workspace/vllm-tp2-profiling
find results -name '*.sqlite' -exec zstd -q --rm {} \;
```

**Mac:**

```bash
rsync -az -e "ssh -p <port>" root@<ip>:/workspace/vllm-tp2-profiling/results/ ./results/
```

Check that `results/raw/` on the Mac has one directory per run before going
on. `*.sqlite.zst` and `*.nsys-rep` are git-ignored; everything else under
`results/` is committed.

## 10. Report locally

**Mac:**

```bash
python -m tpprof report
```

This writes `results/SUMMARY.md` and the figures from the tidy tables.

## 11. Terminate the pod

**Mac:**

```bash
python scripts/runpod.py terminate <id>
```

This deletes the pod and its `/workspace` volume. Billing stops here. Confirm
in the RunPod console that the pod is gone.

## Troubleshooting

| Symptom | Action |
|---|---|
| Quick preflight: `driver` < 580 | Wrong host (`allowedCudaVersions ["13.0"]` should prevent it). Terminate and create a new pod. |
| Quick preflight: `gpu_name`, `gpu_count` or `power_limit` | Not an uncapped 2× H100 SXM. Terminate and retry. |
| Quick preflight: `topology` is NV12 (not NV18) | Terminate and retry. Accept it only if no NV18 host is available, and then note it in the README (see below). |
| Quick preflight: `fabric` not registered | Wait a minute and rerun `bootstrap_box.sh`. If it persists, terminate and retry. |
| Quick preflight: `shm`, `/dev/shm` < 1 GiB | The container cannot change it. Switch provider (the fallback is a Lambda `gpu_2x_h100_sxm5` VM with `docker run --gpus all --ipc=host --entrypoint bash`). |
| Full preflight: `nccl_allreduce` says NVLS is broken on this host | Multicast is advertised but binding it fails (CUDA error 401), so every TP2 engine dies in `ncclCommInitRank`. A host fault, not a gate bug: terminate and create a new pod. Do not skip it, and do not set `NCCL_NVLS_ENABLE=0`, which changes every communication number. |
| A trace run hangs | Handled automatically: the trace is retried once with `VLLM_ALLREDUCE_USE_SYMM_MEM=0` (vLLM #48486), and the retry is recorded in the run's notes. If the retry also hangs, the run fails and the tier goes on. |
| `traces --check` fails on the unclassified share | It lists the top unclassified kernel names. Add regexes for them in `/workspace/vllm-tp2-profiling/tpprof/kernels.py` on the box (tpprof is installed editable), rerun `python -m tpprof traces --check`, then copy the file back: on the Mac, `scp -P <port> root@<ip>:/workspace/vllm-tp2-profiling/tpprof/kernels.py tpprof/kernels.py`. |
| A preflight gate is wrong, not the box | Override that one gate with a recorded `--skip-gate NAME` on `preflight` and on `run`, e.g. `python -m tpprof run --tier P0 --skip-gate model_files`. The override is written to `preflight.json`. |
| The SSH connection dropped | The run goes on inside tmux. `ssh -p <port> root@<ip>`, then `tmux attach -t tp2`. |
| A run failed with `gpus_busy` | GPU memory is held by a process tpprof did not start or could not stop, so the rest of that invocation's GPU runs were skipped (`gpus_busy:<run_id>`). Find it with `nvidia-smi`, kill it, and rerun the same command. |
| A run failed with `effective_config` | The engine came up configured differently from the spec; the reason names the log line. The run's offline points, if any, are moved to `rejected-points-<n>/` and never analyzed. Fix the cause, then rerun with `--retry-failed`. |
| A serve session failed with `client_timeout` | A client ran past 3× its own estimate while the server was up, so the engine is probably wedged. The server was stopped; read its `server-*.log`, then rerun with `--retry-failed`. |
| An M3 run failed with `nccl_unsupported` or `nccl_variant_mismatch` | NCCL rejected the forced algorithm/protocol, or did not use it. `nccl_verify.json` records what NCCL logged. This is a result, not a box problem; it is not retried. |

**Accepting NV12.** `bootstrap_box.sh` takes no arguments, so run a copy with
`--accept-topology` added to both preflights, and pass it to every `run`:

```bash
sed -e 's/preflight --quick;/preflight --quick --accept-topology;/' \
    -e 's/-m tpprof preflight$/-m tpprof preflight --accept-topology/' \
    /workspace/vllm-tp2-profiling/scripts/bootstrap_box.sh > /tmp/bootstrap_nv12.sh
bash /tmp/bootstrap_nv12.sh
python -m tpprof run --tier P0 --accept-topology 2>&1 | tee -ai results/p0.log
```

The override is recorded in `preflight.json`. Add a note to the README that
the GPUs were joined by NV12, since that changes every communication number.
