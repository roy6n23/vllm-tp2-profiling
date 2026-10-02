"""Preflight: hard gates and soft warnings before any measurement (spec 7.5, AM12, AM18, AM26, AM27).

`quick_checks` needs only the standard library and nvidia-smi, so it can run right after SSH,
before any download. `full_checks` adds versions, CLI flags, the model snapshot, nsys and imports;
its on-box checks import torch, vllm and transformers lazily. Every check returns a `Check`,
never raises: a check that crashes is itself reported as failed.
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from tpprof import constants, engine, helpflags

NVIDIA_SMI_TIMEOUT_S = 30
VLLM_TIMEOUT_S = 180            # the real CLI imports torch before printing anything
TOOL_TIMEOUT_S = 60
GPU_QUERY = ("--query-gpu=index,name,driver_version,memory.total,memory.used,power.limit,pci.bus_id",
             "--format=csv,noheader,nounits")
APPS_QUERY = ("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
BASE_IMPORTS = ("tpprof", "numpy", "psutil", "sqlite3")
BOX_IMPORTS = ("third_party.vllm_benchmarks.benchmark_device_communicators",)
NETWORK_FS = frozenset({"nfs", "nfs4", "cifs", "smb3", "smbfs", "ceph", "glusterfs", "9p", "lustre",
                        "fuse", "mfs", "moosefs"})
CLIENT_LABEL = "bench serve client"

SOFT = frozenset({"cpus", "ram", "nsys_status", "monitor_field", "multicast", "model_volume"})
_TERMINATE = "Wrong host: `python scripts/runpod.py terminate <pod>` and rent another 2x H100 SXM pod."
_IMAGE = f"Run inside the pinned image {constants.VLLM_IMAGE}."
FIXES: dict[str, str] = {
    "gpu_count": "Exactly 2 GPUs must be visible (check CUDA_VISIBLE_DEVICES). " + _TERMINATE,
    "gpu_name": f"Both GPUs must be {constants.H100_SXM.name} (H100 SXM). " + _TERMINATE,
    "power_limit": f"A power limit below {constants.H100_SXM.power_limit_w} W shifts every number. " + _TERMINATE,
    "driver": f"The image needs driver >= {constants.MIN_DRIVER_MAJOR} (CUDA {constants.TORCH_CUDA}); set "
              f"allowedCudaVersions [\"{constants.TORCH_CUDA}\"] on the pod. " + _TERMINATE,
    "topology": f"GPU0 and GPU1 must be joined by {constants.EXPECTED_LINK}. " + _TERMINATE
                + f" Only if no {constants.EXPECTED_LINK} host exists: rerun with --accept-topology and note it "
                  "in the README.",
    "fabric": "The NVLink fabric is not registered (fabric manager). Wait a minute and retry; "
              "if it stays so, " + _TERMINATE,
    "gpu_idle": "Stop the foreign GPU processes (nvidia-smi --query-compute-apps=pid,used_memory) "
                f"and wait until memory.used < {constants.GPU_FREE_MIB} MiB, or pick another host.",
    "shm": f"/dev/shm needs >= {constants.MIN_SHM_BYTES / 2**30:g} GiB free for TP2: start the container "
           "with --ipc=host or a larger --shm-size.",
    "disk": f"Needs >= {constants.MIN_DISK_BYTES / 1e9:g} GB free under the results directory "
            "(model, compile cache, traces): free space or attach a larger volume.",
    "nofile": f"Raise the hard open-files limit to >= {constants.MIN_NOFILE_HARD} (`ulimit -Hn 65535` as "
              "root, or docker --ulimit nofile=65535:65535).",
    "cpus": f"Fewer than {constants.MIN_CPUS} CPUs can make the API server or the client the bottleneck; "
            "check the CPU columns of the saturation runs.",
    "ram": f"Less than {constants.MIN_RAM_BYTES / 1e9:g} GB RAM: model loading and page cache may be slow; "
           "watch for swapping.",
    "vllm_version": _IMAGE,
    "help_flags": "A generated argv uses a flag this vLLM does not know: fix tpprof/engine.py or "
                  "tpprof/client.py, or confirm the vLLM version.",
    "torch": _IMAGE + f" It ships torch {constants.TORCH_VERSION} built for CUDA {constants.TORCH_CUDA}; "
                      f"an SM count other than {constants.H100_SXM.sm_count} means the GPU is not an H100 SXM.",
    "vllm_env_names": "Unset the unknown VLLM_* variables: --fail-on-environ-validation makes vLLM refuse "
                      "to start with them.",
    "json_configs": "Fix the JSON config in tpprof/engine.py; vLLM rejects it.",
    "nvcc": _IMAGE + " Without nvcc FlashInfer cannot JIT and the allreduce fusion is silently off.",
    "flashinfer_jit_cache": _IMAGE + " It ships {}=={}.".format(*constants.FLASHINFER_JIT_CACHE),
    "model_files": f"Download {constants.MODEL.repo}@{constants.MODEL.revision} again into the model dir "
                   "(scripts/bootstrap_box.sh); a partial or wrong snapshot changes the model.",
    "model_volume": "The model is on a network volume: startup and page-cache warmup are slower. "
                    "Copy it to local disk if startups time out.",
    "tokenizer_offline": "The tokenizer files must load with HF_HUB_OFFLINE=1: re-download the snapshot.",
    "nsys": f"Install {constants.NSYS_APT_PACKAGE} (scripts/bootstrap_box.sh) so nsys "
            f"{constants.NSYS_VERSION} is at {constants.NSYS_DEFAULT_PATH}; on the box TPPROF_NSYS must be unset.",
    "nsys_status": "nsys status --environment reports a problem; CPU sampling may be off in traces.",
    "monitor_field": "The GPU monitor uses the legacy clocks_throttle_reasons.active field (older driver).",
    "multicast": "No NVSwitch multicast: recorded as a run attribute (FlashInfer falls back from mnnvl to "
                 "trtllm; NVLS is unavailable). No action needed.",
    "imports": "Install tpprof into the image Python: `pip install --no-deps -e .` from the repo root.",
    "nccl_allreduce": "Every TP2 run needs this all-reduce. If the detail names NVLS, the host's NVSwitch "
                      "multicast is broken (fabric manager): " + _TERMINATE,
}


@dataclass
class Check:
    name: str
    ok: bool
    hard: bool
    detail: str
    fix: str
    skipped: bool = False    # a failed hard gate overridden by --skip-gate (set from ctx.skip_gates)


@dataclass
class PreflightContext:
    nvidia_smi: str = "nvidia-smi"
    vllm_bin: str = "vllm"
    model_dir: str = ""
    results_dir: str = "results"
    shm_path: str = "/dev/shm"
    on_box: bool = True
    accept_topology: bool = False
    skip_gates: tuple[str, ...] = ()     # --skip-gate names; quick/full_checks mark those gates `skipped`
    env: Mapping[str, str] | None = None


# ================================================================ helpers


def _check(name: str, ok: bool, detail: str, hard: bool | None = None) -> Check:
    return Check(name, ok, name not in SOFT if hard is None else hard, detail, "" if ok else FIXES[name])


def _guarded(name: str, fn: Callable[..., Check], *args: object) -> Check:
    try:
        return fn(*args)
    except Exception as exc:  # a crashing check is a failed check, never a crashed preflight
        return _check(name, False, f"the check raised {type(exc).__name__}: {exc}")


def environ(ctx: PreflightContext) -> dict[str, str]:
    """The environment the checked tools run with."""
    return dict(os.environ if ctx.env is None else ctx.env)


def nsys_path(ctx: PreflightContext) -> str:
    """`constants.nsys_path()` evaluated in the context's environment (AM18)."""
    if ctx.env is None:
        return constants.nsys_path()
    return ctx.env.get("TPPROF_NSYS", constants.NSYS_DEFAULT_PATH)


def run_text(argv: Sequence[str], env: Mapping[str, str], timeout_s: float) -> tuple[int | None, str, str]:
    """(exit code, stdout, stderr); exit code None and the reason in stderr if it could not run."""
    try:
        r = subprocess.run(list(argv), env=dict(env), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           errors="replace", timeout=timeout_s, start_new_session=True)
    except OSError as exc:
        return None, "", f"cannot run {argv[0]}: {exc}"
    except subprocess.TimeoutExpired:
        return None, "", f"{' '.join(argv)} timed out after {timeout_s} s"
    return r.returncode, r.stdout, r.stderr


def tool_output(argv: Sequence[str], env: Mapping[str, str], timeout_s: float) -> tuple[str | None, str]:
    """(stdout, "") on exit 0, else (None, reason)."""
    code, out, err = run_text(argv, env, timeout_s)
    if code == 0:
        return out, ""
    if code is None:
        return None, err
    return None, f"{' '.join(argv)} exited {code}: {(err or out).strip()[-500:]}"


def query_gpus(ctx: PreflightContext) -> tuple[list[dict[str, str]], str]:
    """Rows of the C8 `--query-gpu` form as dicts, and an error ("" if the query worked)."""
    out, err = tool_output([ctx.nvidia_smi, *GPU_QUERY], environ(ctx), NVIDIA_SMI_TIMEOUT_S)
    if out is None:
        return [], f"nvidia-smi query failed: {err}"
    keys = ("index", "name", "driver_version", "memory.total", "memory.used", "power.limit", "pci.bus_id")
    rows = [dict(zip(keys, (f.strip() for f in line.split(",")))) for line in out.splitlines() if line.strip()]
    return [r for r in rows if len(r) == len(keys)], ""


def _float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _per_gpu(gpus: Sequence[Mapping[str, str]], key: str, unit: str = "") -> str:
    return "; ".join(f"GPU{g['index']}: {g[key]}{unit}" for g in gpus)


# ================================================================ quick checks


def _gpu_count(gpus: list[dict[str, str]], err: str) -> Check:
    if err:
        return _check("gpu_count", False, err)
    return _check("gpu_count", len(gpus) == 2, f"{len(gpus)} GPU(s) visible (need exactly 2)")


def _gpu_name(gpus: list[dict[str, str]], err: str) -> Check:
    if err or not gpus:
        return _check("gpu_name", False, err or "no GPU listed")
    ok = all(g["name"] == constants.H100_SXM.name for g in gpus)
    return _check("gpu_name", ok, _per_gpu(gpus, "name"))


def _power_limit(gpus: list[dict[str, str]], err: str) -> Check:
    if err or not gpus:
        return _check("power_limit", False, err or "no GPU listed")
    limits = [_float(g["power.limit"]) for g in gpus]
    ok = all(w is not None and w >= constants.H100_SXM.power_limit_w for w in limits)
    return _check("power_limit", ok, _per_gpu(gpus, "power.limit", " W")
                  + f" (need >= {constants.H100_SXM.power_limit_w} W)")


def _driver(gpus: list[dict[str, str]], err: str) -> Check:
    if err or not gpus:
        return _check("driver", False, err or "no GPU listed")
    versions = sorted({g["driver_version"] for g in gpus})
    majors = [re.match(r"(\d+)", v) for v in versions]
    ok = all(m is not None and int(m.group(1)) >= constants.MIN_DRIVER_MAJOR for m in majors)
    return _check("driver", ok, f"driver {', '.join(versions)} (need >= {constants.MIN_DRIVER_MAJOR})")


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def topology_link(topo_text: str, a: str = "GPU0", b: str = "GPU1") -> str | None:
    """The `nvidia-smi topo -m` cell for row `a`, column `b` (e.g. "NV18"), or None if absent."""
    header: list[str] | None = None
    for line in _ANSI.sub("", topo_text).splitlines():
        cells = [c.strip() for c in line.split("\t")]
        if header is None:
            if a in cells and b in cells:
                header = cells
            continue
        if cells and cells[0] == a:
            col = header.index(b)
            return cells[col] if col < len(cells) and cells[col] else None
    return None


def _topology(ctx: PreflightContext) -> Check:
    out, err = tool_output([ctx.nvidia_smi, "topo", "-m"], environ(ctx), NVIDIA_SMI_TIMEOUT_S)
    if out is None:
        return _check("topology", False, f"nvidia-smi topo -m failed: {err}")
    link = topology_link(out)
    if link is None:
        return _check("topology", False, "no GPU0-GPU1 cell in nvidia-smi topo -m")
    if link == constants.EXPECTED_LINK:
        return _check("topology", True, f"GPU0-GPU1 link {link}")
    detail = f"GPU0-GPU1 link {link}, expected {constants.EXPECTED_LINK}"
    if ctx.accept_topology:
        return _check("topology", False, detail + "; accepted by --accept-topology (recorded)", hard=False)
    return _check("topology", False, detail)


_KV = re.compile(r"^\s*(State|Status)\s*:\s*(.*?)\s*$")
_GPU_HEADER = re.compile(r"^GPU\s+(\S+)\s*$")


def fabric_states(q_text: str) -> list[tuple[str, str, str]]:
    """[(gpu bus id, State, Status)] for each `Fabric` block in `nvidia-smi -q` output."""
    out: list[tuple[str, str, str]] = []
    gpu = "?"
    lines = q_text.splitlines()
    for i, line in enumerate(lines):
        m = _GPU_HEADER.match(line)
        if m:
            gpu = m.group(1)
        if line.strip() != "Fabric":
            continue
        indent = len(line) - len(line.lstrip())
        found: dict[str, str] = {}
        for sub in lines[i + 1:]:
            if not sub.strip() or len(sub) - len(sub.lstrip()) <= indent:
                break
            kv = _KV.match(sub)
            if kv:
                found.setdefault(kv.group(1), kv.group(2))
        out.append((gpu, found.get("State", "?"), found.get("Status", "?")))
    return out


def _fabric(ctx: PreflightContext) -> Check:
    out, err = tool_output([ctx.nvidia_smi, "-q"], environ(ctx), NVIDIA_SMI_TIMEOUT_S)
    if out is None:
        return _check("fabric", False, f"nvidia-smi -q failed: {err}")
    states = fabric_states(out)
    if not states:
        return _check("fabric", False, "no Fabric block in nvidia-smi -q")
    ok = all(state == "Completed" and status == "Success" for _, state, status in states)
    detail = "; ".join(f"GPU {gpu}: State {state}, Status {status}" for gpu, state, status in states)
    return _check("fabric", ok, detail + ("" if ok else " (need State Completed, Status Success)"))


def _gpu_idle(ctx: PreflightContext, gpus: list[dict[str, str]], err: str) -> Check:
    if err or not gpus:
        return _check("gpu_idle", False, err or "no GPU listed")
    out, apps_err = tool_output([ctx.nvidia_smi, *APPS_QUERY], environ(ctx), NVIDIA_SMI_TIMEOUT_S)
    if out is None:
        return _check("gpu_idle", False, f"nvidia-smi compute-apps query failed: {apps_err}")
    apps = [[f.strip() for f in line.split(",")] for line in out.splitlines() if line.strip()]
    used = [_float(g["memory.used"]) for g in gpus]
    ok = not apps and all(u is not None and u < constants.GPU_FREE_MIB for u in used)
    detail = f"memory.used {_per_gpu(gpus, 'memory.used', ' MiB')} (need < {constants.GPU_FREE_MIB} MiB); "
    detail += ("compute apps: " + ", ".join(f"pid {a[0]} ({a[1] if len(a) > 1 else '?'} MiB)" for a in apps)
               if apps else "no compute apps")
    return _check("gpu_idle", ok, detail)


def _shm(ctx: PreflightContext) -> Check:
    try:
        free = shutil.disk_usage(ctx.shm_path).free
    except OSError as exc:
        return _check("shm", False, f"cannot stat {ctx.shm_path}: {exc}")
    return _check("shm", free >= constants.MIN_SHM_BYTES,
                  f"{free / 2**20:.1f} MiB free at {ctx.shm_path} (need >= {constants.MIN_SHM_BYTES >> 20} MiB)")


def _existing_ancestor(path: str) -> str:
    path = os.path.abspath(path)
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    return path


def _disk(ctx: PreflightContext) -> Check:
    where = _existing_ancestor(ctx.results_dir)
    try:
        free = shutil.disk_usage(where).free
    except OSError as exc:
        return _check("disk", False, f"cannot stat {where}: {exc}")
    return _check("disk", free >= constants.MIN_DISK_BYTES,
                  f"{free / 1e9:.1f} GB free under {where} (need >= {constants.MIN_DISK_BYTES / 1e9:.0f} GB)")


def _nofile() -> Check:
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    ok = hard == resource.RLIM_INFINITY or hard >= constants.MIN_NOFILE_HARD
    shown = "unlimited" if hard == resource.RLIM_INFINITY else str(hard)
    return _check("nofile", ok, f"hard RLIMIT_NOFILE {shown} (need >= {constants.MIN_NOFILE_HARD})")


def _cpus() -> Check:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 0)
    return _check("cpus", n >= constants.MIN_CPUS, f"{n} CPUs usable (want >= {constants.MIN_CPUS})")


def _ram() -> Check:
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    want = constants.MIN_RAM_BYTES
    return _check("ram", total >= want, f"{total / 1e9:.1f} GB RAM (want >= {want / 1e9:.0f} GB)")


def quick_checks(ctx: PreflightContext) -> list[Check]:
    """AM27: standard library and nvidia-smi only; runs in seconds, before any download."""
    gpus, err = query_gpus(ctx)
    return _apply_skips(ctx, [
        _guarded("gpu_count", _gpu_count, gpus, err),
        _guarded("gpu_name", _gpu_name, gpus, err),
        _guarded("power_limit", _power_limit, gpus, err),
        _guarded("driver", _driver, gpus, err),
        _guarded("topology", _topology, ctx),
        _guarded("fabric", _fabric, ctx),
        _guarded("gpu_idle", _gpu_idle, ctx, gpus, err),
        _guarded("shm", _shm, ctx),
        _guarded("disk", _disk, ctx),
        _guarded("nofile", _nofile),
        _guarded("cpus", _cpus),
        _guarded("ram", _ram),
    ])


# ================================================================ full checks


def _vllm_version(ctx: PreflightContext) -> Check:
    out, err = tool_output([ctx.vllm_bin, "--version"], environ(ctx), VLLM_TIMEOUT_S)
    if out is None:
        return _check("vllm_version", False, err)
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    got = lines[-1] if lines else ""
    return _check("vllm_version", got == constants.VLLM_VERSION,
                  f"vllm --version: {got!r} (need {constants.VLLM_VERSION})")


def _sample_client_argv(ctx: PreflightContext, model_dir: str) -> list[str]:
    from tpprof.client import ClientSpec, client_argv  # T10; imported lazily

    spec = ClientSpec(base_url=f"http://127.0.0.1:{constants.PORT_BASE}", tokenizer=model_dir,
                      input_len=constants.ONLINE_INPUT_LEN, output_len=constants.ONLINE_OUTPUT_LEN, prefix_len=0,
                      num_prompts=constants.SAT_NUM_PROMPTS, request_rate=math.inf, seed=constants.SAT_SEEDS[0],
                      result_dir=ctx.results_dir, result_filename="preflight-client.json",
                      request_id_prefix="preflight-", metadata=(("tpprof_run_id", "preflight"),))
    return client_argv(spec, ctx.vllm_bin)


def help_flag_problems(ctx: PreflightContext) -> dict[str, list[str]]:
    """{argv label: flags unknown to the matching `--help=all`}; an entry "error: ..." if it could not check.

    Labels: `serve <cfg>/<arm>` for every config, `bench latency <cfg>/<arm>` for every dp == 1 config
    (its `bench_latency_argv`, which contains `offline_args`), and `bench serve client`.
    """
    env = environ(ctx)
    model_dir = ctx.model_dir or "/model"
    helps: dict[str, tuple[str | None, str]] = {}
    for sub in ("serve", "bench latency", "bench serve"):
        helps[sub] = tool_output([ctx.vllm_bin, *sub.split(), "--help=all"], env, VLLM_TIMEOUT_S)

    def missing(sub: str, argv: Sequence[str]) -> list[str]:
        text, err = helps[sub]
        return [f"error: vllm {sub} --help=all: {err}"] if text is None else helpflags.missing_flags(argv, text)

    problems: dict[str, list[str]] = {}
    for cfg in engine.all_configs():
        problems[f"serve {cfg.name}/{cfg.arm}"] = missing(
            "serve", cfg.serve_argv(model_dir, constants.PORT_BASE, ctx.vllm_bin))
        if cfg.dp == 1:
            x = constants.XCHECK
            problems[f"bench latency {cfg.name}/{cfg.arm}"] = missing("bench latency", cfg.bench_latency_argv(
                model_dir, x["batch"], x["input_len"], x["output_len"], x["warmup"], x["iters"],
                os.path.join(ctx.results_dir, "preflight-latency.json"), ctx.vllm_bin))
    try:
        client = _sample_client_argv(ctx, model_dir)
    except Exception as exc:
        problems[CLIENT_LABEL] = [f"error: cannot build a client argv: {type(exc).__name__}: {exc}"]
    else:
        problems[CLIENT_LABEL] = missing("bench serve", client)
    return problems


def _help_flags(ctx: PreflightContext) -> Check:
    problems = help_flag_problems(ctx)
    bad = {label: flags for label, flags in problems.items() if flags}
    if not bad:
        return _check("help_flags", True, f"{len(problems)} argvs checked against --help=all; all flags known")
    return _check("help_flags", False, "; ".join(f"{label}: {', '.join(flags)}" for label, flags in bad.items()))


def _torch() -> Check:
    import torch

    problems = []
    if not torch.__version__.startswith(constants.TORCH_VERSION):
        problems.append(f"torch {torch.__version__} (need {constants.TORCH_VERSION})")
    if torch.version.cuda != constants.TORCH_CUDA:
        problems.append(f"torch.version.cuda {torch.version.cuda} (need {constants.TORCH_CUDA})")
    n = torch.cuda.device_count()
    sms = [torch.cuda.get_device_properties(i).multi_processor_count for i in range(n)]
    if n < 2 or any(sm != constants.H100_SXM.sm_count for sm in sms[:2]):
        problems.append(f"SM counts {sms} (need {constants.H100_SXM.sm_count} on 2 GPUs)")
    nccl = ".".join(map(str, torch.cuda.nccl.version()))
    detail = (f"torch {torch.__version__}, cuda {torch.version.cuda}, SMs {sms}, "
              f"torch NCCL {nccl} (vLLM expects {constants.NCCL_EXPECTED}; recorded)")
    return _check("torch", not problems, "; ".join(problems) + (" | " if problems else "") + detail)


def vllm_env_names_used(ctx: PreflightContext) -> list[str]:
    """Every VLLM_* name in the context environment and in each generated engine environment."""
    base = environ(ctx)
    names = {k for k in base if k.startswith("VLLM_")}
    for cfg in engine.all_configs():
        names |= {k for k in cfg.environment(base) if k.startswith("VLLM_")}
    return sorted(names)


def _vllm_env_names(ctx: PreflightContext) -> Check:
    import vllm.envs

    used = vllm_env_names_used(ctx)
    unknown = [k for k in used if k not in vllm.envs.environment_variables]
    return _check("vllm_env_names", not unknown,
                  f"unknown to vllm.envs: {', '.join(unknown)}" if unknown else f"{len(used)} VLLM_* names known")


def _json_configs() -> Check:
    from vllm.config import AttentionConfig, CompilationConfig

    classes = {"--compilation-config": CompilationConfig, "--attention-config": AttentionConfig}
    values = sorted({(f, v) for cfg in engine.all_configs() for f, v in cfg.flags if f in classes and v})
    errors = []
    for flag, value in values:
        try:
            classes[flag](**json.loads(value))
        except Exception as exc:
            errors.append(f"{flag} {value}: {type(exc).__name__}: {exc}")
    return _check("json_configs", not errors, "; ".join(errors) or f"{len(values)} JSON values accepted")


def _nvcc(ctx: PreflightContext) -> Check:
    path = shutil.which("nvcc", path=environ(ctx).get("PATH"))
    return _check("nvcc", path is not None, f"nvcc at {path}" if path else "nvcc not on PATH")


def _flashinfer_jit_cache() -> Check:
    dist, want = constants.FLASHINFER_JIT_CACHE
    try:
        got = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return _check("flashinfer_jit_cache", False, f"{dist} is not installed (need {want})")
    # compare the public release only: the pinned image's wheel carries a local label (+cu130)
    return _check("flashinfer_jit_cache", got.split("+", 1)[0] == want, f"{dist} {got} (need {want})")


def check_model_files(model_dir: str) -> Check:
    """AM26: the 4 shards at their exact sizes, the required files, and the index total_size."""
    if not model_dir:
        return _check("model_files", False, "no model dir given (--model-dir or TPPROF_MODEL_DIR)")
    problems = []
    for name, size in constants.MODEL_SHARD_SIZES.items():
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            problems.append(f"{name} missing")
        elif os.path.getsize(path) != size:
            problems.append(f"{name} is {os.path.getsize(path)} B (need {size})")
    for name in constants.MODEL_REQUIRED_FILES:
        if not os.path.isfile(os.path.join(model_dir, name)):
            problems.append(f"{name} missing")
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index) as fh:
                total = json.load(fh)["metadata"]["total_size"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"model.safetensors.index.json has no metadata.total_size ({exc})")
        else:
            if total != constants.MODEL_TENSOR_BYTES_TOTAL:
                problems.append(f"index total_size {total} (need {constants.MODEL_TENSOR_BYTES_TOTAL})")
    if problems:
        return _check("model_files", False, f"{model_dir}: " + "; ".join(problems))
    return _check("model_files", True, f"{model_dir}: 4 shards, {constants.MODEL_SHARD_FILES_TOTAL} B, "
                                       f"index total_size {constants.MODEL_TENSOR_BYTES_TOTAL}")


def _mount_fstype(path: str, mounts_file: str = "/proc/mounts") -> str | None:
    """File system type of the longest mount point containing `path` (Linux), or None."""
    try:
        with open(mounts_file) as fh:
            mounts = [line.split()[1:3] for line in fh if len(line.split()) >= 3]
    except OSError:
        return None
    path = os.path.realpath(path)
    best: tuple[int, str] | None = None
    for point, fstype in mounts:
        point = point.replace("\\040", " ")
        if (path == point or path.startswith(point.rstrip("/") + "/")) and (best is None or len(point) > best[0]):
            best = (len(point), fstype)
    return best[1] if best else None


def _model_volume(ctx: PreflightContext) -> Check:
    fstype = _mount_fstype(ctx.model_dir or ".")
    if fstype is None:
        return _check("model_volume", True, "mount type unknown (no /proc/mounts entry)")
    network = fstype in NETWORK_FS or fstype.startswith("fuse.")
    return _check("model_volume", not network, f"{ctx.model_dir} is on a {fstype} file system")


def _tokenizer_offline(ctx: PreflightContext) -> Check:
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(ctx.model_dir, local_files_only=True)
    return _check("tokenizer_offline", True, f"{type(tok).__name__}, vocab {len(tok)}")


def check_nsys(ctx: PreflightContext) -> Check:
    """nsys at the pinned path with the pinned version; TPPROF_NSYS unset on the box (AM18)."""
    path = nsys_path(ctx)
    problems = []
    version = ""
    if ctx.on_box and "TPPROF_NSYS" in environ(ctx):
        problems.append(f"TPPROF_NSYS is set ({environ(ctx)['TPPROF_NSYS']}); the box must use "
                        f"{constants.NSYS_DEFAULT_PATH}")
    if not os.access(path, os.X_OK):
        problems.append(f"{path} is not an executable file")
    else:
        out, err = tool_output([path, "--version"], environ(ctx), TOOL_TIMEOUT_S)
        if out is None:
            problems.append(err)
        elif constants.NSYS_VERSION not in out:
            problems.append(f"{path} --version: {out.strip()!r} (need {constants.NSYS_VERSION})")
        else:
            version = out.strip()
    if problems:
        return _check("nsys", False, "; ".join(problems))
    return _check("nsys", True, f"{path}: {version}")


def _nsys_status(ctx: PreflightContext) -> Check:
    out, err = tool_output([nsys_path(ctx), "status", "--environment"], environ(ctx), TOOL_TIMEOUT_S)
    return _check("nsys_status", out is not None, out.strip() if out is not None else err)


def monitor_field(ctx: PreflightContext) -> str | None:
    """The throttle-reasons field this nvidia-smi answers: the current name, the legacy one, or None."""
    from tpprof.monitor import EVENT_REASONS_FIELD, LEGACY_EVENT_REASONS_FIELD

    for field in (EVENT_REASONS_FIELD, LEGACY_EVENT_REASONS_FIELD):
        out, _ = tool_output([ctx.nvidia_smi, f"--query-gpu={field}", "--format=csv,noheader"], environ(ctx),
                             NVIDIA_SMI_TIMEOUT_S)
        if out is not None and out.strip():
            return field
    return None


def _monitor_field(ctx: PreflightContext) -> Check:
    from tpprof.monitor import EVENT_REASONS_FIELD

    field = monitor_field(ctx)
    if field is None:
        return _check("monitor_field", False, "neither clocks_event_reasons.active nor "
                                              "clocks_throttle_reasons.active can be queried")
    return _check("monitor_field", field == EVENT_REASONS_FIELD, field)


def _multicast() -> Check:
    import torch

    try:
        results = [bool(torch._C._distributed_c10d._SymmetricMemory.has_multicast_support(
            torch._C._autograd.DeviceType.CUDA, i)) for i in range(2)]
    except Exception as exc:
        return _check("multicast", False, f"probe failed: {type(exc).__name__}: {exc}")
    return _check("multicast", all(results), "has_multicast_support: " +
                  ", ".join(f"GPU{i} {r}" for i, r in enumerate(results)))


NCCL_TIMEOUT_S = 180
NCCL_OK = "NCCL_AR_OK"
# One all-reduce over both GPUs with the NCCL setup vLLM gets. `multicast` only reads the attribute;
# on the 2026-10-01 box it said True while binding multicast memory failed (CUDA error 401), so every
# TP2 engine died in ncclCommInitRank.
NCCL_PROBE = f"""
import socket, torch, torch.distributed as dist, torch.multiprocessing as mp

def work(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{{port}}", rank=rank, world_size=2)
    x = torch.ones(1 << 20, device="cuda")
    dist.all_reduce(x)
    torch.cuda.synchronize()
    assert x[0].item() == 2.0, x[0].item()
    print("rank", rank, "{NCCL_OK}", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(work, args=(port,), nprocs=2)
"""


def _nccl_reason(out: str, err: str) -> str:
    lines = [line.strip() for line in (err + "\n" + out).splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if "NCCL WARN" in line:
            return line[-300:]
        if line.startswith("Last error"):   # the cause follows on the next line
            return " ".join(lines[i:i + 2])[-300:]
    return (lines or ["no output"])[0][-300:]


def _nccl_allreduce(ctx: PreflightContext) -> Check:
    env = environ(ctx)
    env.pop("NCCL_NVLS_ENABLE", None)
    # a script file, not `python -c`: mp.spawn children re-import __main__ to find `work`
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "nccl_probe.py")
        with open(script, "w") as f:
            f.write(NCCL_PROBE)
        argv = [sys.executable, script]
        code, out, err = run_text(argv, env, NCCL_TIMEOUT_S)
        if code == 0 and out.count(NCCL_OK) == 2:
            return _check("nccl_allreduce", True, "2-GPU NCCL all-reduce OK (default NCCL, NVLS allowed)")
        reason = _nccl_reason(out, err)
        code2, out2, _ = run_text(argv, {**env, "NCCL_NVLS_ENABLE": "0"}, NCCL_TIMEOUT_S)
    if code2 == 0 and out2.count(NCCL_OK) == 2:
        return _check("nccl_allreduce", False, f"fails with NVLS, works with NCCL_NVLS_ENABLE=0: NVLS is broken "
                                               f"on this host: {reason}")
    return _check("nccl_allreduce", False, f"2-GPU NCCL all-reduce fails ({reason}); "
                                           "also fails with NCCL_NVLS_ENABLE=0")


def _imports(ctx: PreflightContext) -> Check:
    errors = []
    for name in BASE_IMPORTS + (BOX_IMPORTS if ctx.on_box else ()):
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return _check("imports", not errors, "; ".join(errors) or "all imports work")


def full_checks(ctx: PreflightContext) -> list[Check]:
    """`quick_checks` plus versions, flags, model, nsys and imports; on-box checks only when `ctx.on_box`."""
    box = ctx.on_box
    plan: list[tuple[str, Callable[..., Check], tuple[object, ...], bool]] = [
        ("vllm_version", _vllm_version, (ctx,), True),
        ("help_flags", _help_flags, (ctx,), True),
        ("torch", _torch, (), box),
        ("vllm_env_names", _vllm_env_names, (ctx,), box),
        ("json_configs", _json_configs, (), box),
        ("nvcc", _nvcc, (ctx,), True),
        ("flashinfer_jit_cache", _flashinfer_jit_cache, (), True),
        ("model_files", check_model_files, (ctx.model_dir,), True),
        ("model_volume", _model_volume, (ctx,), box),
        ("tokenizer_offline", _tokenizer_offline, (ctx,), box),
        ("nsys", check_nsys, (ctx,), True),
        ("nsys_status", _nsys_status, (ctx,), True),
        ("monitor_field", _monitor_field, (ctx,), True),
        ("multicast", _multicast, (), box),
        ("nccl_allreduce", _nccl_allreduce, (ctx,), box),
        ("imports", _imports, (ctx,), True),
    ]
    full = [_guarded(name, fn, *args) for name, fn, args, run in plan if run]
    return _apply_skips(ctx, quick_checks(ctx) + full)


# ================================================================ verdict and report


def _apply_skips(ctx: PreflightContext, checks: list[Check]) -> list[Check]:
    """Mark the failed hard gates named in `ctx.skip_gates`, so verdict() and write_report() agree without
    being handed the names again. The raw `ok` stays False: the override is recorded, not hidden."""
    for c in checks:
        c.skipped = c.skipped or (c.hard and not c.ok and c.name in ctx.skip_gates)
    return checks


def _skip_names(checks: Sequence[Check], skip: Sequence[str]) -> tuple[str, ...]:
    """The explicit `skip` names plus every check already marked `skipped` via ctx.skip_gates."""
    return tuple(dict.fromkeys([*skip, *(c.name for c in checks if c.skipped)]))


def _skipped(c: Check, skip: Sequence[str]) -> bool:
    return c.hard and not c.ok and c.name in skip


def verdict(checks: Sequence[Check], skip: Sequence[str] = ()) -> tuple[bool, str]:
    """(ok, a table of every check). A failed hard gate is overridden and shown as SKIP when it is marked
    `skipped` (ctx.skip_gates) or named in `skip`."""
    skip = _skip_names(checks, skip)
    rows = []
    for c in checks:
        status = "PASS" if c.ok else "SKIP" if _skipped(c, skip) else "FAIL" if c.hard else "WARN"
        first = c.detail.splitlines()[0] if c.detail else ""
        rows.append(f"{status}  {c.name:<22} {first}")
        if status == "SKIP":
            rows.append(f"      overridden by --skip-gate {c.name} (recorded)")
        elif status != "PASS":
            rows.append(f"      fix: {c.fix}")
    names = {c.name for c in checks}
    for name in dict.fromkeys(skip):
        if name not in names:
            rows.append(f"note: --skip-gate {name} matches no check")
        elif not any(_skipped(c, skip) for c in checks if c.name == name):
            rows.append(f"note: --skip-gate {name} had no effect (the gate did not fail hard)")
    failed = [c.name for c in checks if c.hard and not c.ok and c.name not in skip]
    skipped = [c.name for c in checks if _skipped(c, skip)]
    warned = [c.name for c in checks if not c.hard and not c.ok]
    ok = not failed
    summary = f"preflight {'OK' if ok else 'FAILED'}: {len(checks)} checks"
    if failed:
        summary += f", hard gates failed: {', '.join(failed)}"
    if skipped:
        summary += f", skipped gates: {', '.join(skipped)}"
    if warned:
        summary += f", warnings: {', '.join(warned)}"
    return ok, "\n".join([*rows, summary])


def write_report(checks: Sequence[Check], path: str, skip: Sequence[str] = ()) -> None:
    """preflight.json: the verdict, the --skip-gate names, and every check.

    The skipped gates come from the checks themselves (marked from ctx.skip_gates), so
    `write_report(checks, path)` agrees with `verdict(checks)`; the optional `skip` adds names, as in verdict().
    """
    skip = _skip_names(checks, skip)
    ok, _ = verdict(checks, skip)
    report = {
        "ok": ok,
        "t_wall": time.time(),
        "t_mono": time.monotonic(),
        "skipped_gates": list(skip),
        "checks": [{**dataclasses.asdict(c), "skipped": _skipped(c, skip)} for c in checks],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
