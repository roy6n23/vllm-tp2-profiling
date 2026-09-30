"""M4 (P2, cross-check): nccl-tests v2.20.0 ``all_reduce_perf`` against the image's NCCL (spec 4.6, AM31, D7-11).

One thread per GPU (``-t 2 -g 1``): ``-g 2`` distorts small-message latency. Two passes: eager with per-iteration
timing and NCCL tuning columns (``-I 1 -U 1``), and CUDA graph (``-G``). Results come from the ``-J`` JSON.
nccl-tests uses P2P/direct pointer inside one process, whereas torchrun and vLLM use P2P/CUMEM, so M3 stays the
like-for-like measurement.
"""
from __future__ import annotations

import json

from tpprof.constants import BOX_WORKSPACE as WORKSPACE
from tpprof.constants import NCCL_TESTS_BIN_DIR as BIN_DIR
from tpprof.constants import NCCL_TESTS_NCCL_HOME as NCCL_HOME
from tpprof.constants import NCCL_TESTS_TARBALL, NCCL_TESTS_VERSION, TORCH_CUDA
# The pip NCCL package dir (nvidia.nccl is a namespace package without __file__, so ask importlib; D7-11).
NCCL_PKG_CMD = ("python3 -c \"import importlib.util as u;"
                "print(list(u.find_spec('nvidia.nccl').submodule_search_locations)[0])\"")


def build_script(nccl_home_cmd: str = NCCL_PKG_CMD) -> str:
    """Bash that builds all_reduce_perf (idempotent). ``nccl_home_cmd`` prints the directory holding the NCCL
    ``include/`` and ``lib/libnccl.so.2``; nccl-tests links ``-lnccl``, so an unversioned symlink is added.
    The vLLM image is ``nvidia/cuda:*-base`` plus nvcc, so make, g++ and the cudart dev library are installed
    from apt when missing."""
    cudart_dev = f"cuda-cudart-dev-{TORCH_CUDA.replace('.', '-')}"
    return f"""#!/usr/bin/env bash
# nccl-tests v{NCCL_TESTS_VERSION} all_reduce_perf, MPI=0, sm_90 only, against the image's NCCL (tpprof M4).
set -euo pipefail
if [ -x {BIN_DIR}/all_reduce_perf ]; then echo "nccl-tests already built: {BIN_DIR}"; exit 0; fi
need=""
command -v make >/dev/null || need="$need make"
command -v g++ >/dev/null || need="$need g++"
[ -e /usr/local/cuda/lib64/libcudart.so ] || need="$need {cudart_dev}"
if [ -n "$need" ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $need
fi
NCCL_PKG="$({nccl_home_cmd})"
mkdir -p {NCCL_HOME}/lib
ln -sfn "$NCCL_PKG/include" {NCCL_HOME}/include
ln -sf "$NCCL_PKG/lib/libnccl.so.2" {NCCL_HOME}/lib/libnccl.so.2
ln -sf libnccl.so.2 {NCCL_HOME}/lib/libnccl.so
cd {WORKSPACE}
curl -fsSL -o nccl-tests-{NCCL_TESTS_VERSION}.tar.gz {NCCL_TESTS_TARBALL}
tar xzf nccl-tests-{NCCL_TESTS_VERSION}.tar.gz
cd nccl-tests-{NCCL_TESTS_VERSION}
make -j MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME={NCCL_HOME} NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
test -x {BIN_DIR}/all_reduce_perf
echo "nccl-tests built: {BIN_DIR}"
"""


def run_env(nccl_home: str = NCCL_HOME) -> dict[str, str]:
    """Environment overrides so all_reduce_perf loads the image's libnccl.so.2, not a system copy."""
    return {"LD_LIBRARY_PATH": f"{nccl_home}/lib"}


def run_argv(bin_dir: str, out_json: str, graph: bool) -> list[str]:
    mode = ["-G", "20"] if graph else ["-I", "1", "-U", "1"]
    return [f"{bin_dir}/all_reduce_perf", "-b", "8", "-e", "256M", "-f", "2", "-t", "2", "-g", "1",
            "-d", "bfloat16", "-w", "20", "-n", "200", *mode, "-J", out_json]


def parse_json(path: str) -> list[dict]:
    """The ``-J`` JSON -> one row per (size, place): ``{"impl": "nccl_tests", "mode", "place", "bytes",
    "time_us", "p50_us", "algbw_GBps", "busbw_GBps", "nwrong", "algo", "proto", "nccl_version"}``.
    p50_us needs ``-I 1``; algo and proto need ``-U 1`` (None otherwise). nccl_version is the loaded library's
    code (e.g. 23007), to be checked against the expected NCCL."""
    with open(path) as f:
        doc = json.load(f)
    for key in ("nccl_version", "config", "results"):
        if key not in doc:
            raise ValueError(f"{path}: nccl-tests JSON lacks {key!r}")
    mode = "graph" if doc["config"].get("graph") else "eager"
    rows = []
    for res in doc["results"]:
        tuning = res.get("tuning") or {}
        for place in ("out_of_place", "in_place"):
            d = res.get(place)
            if not d:
                continue
            per_iter = res.get(f"{place}_per_iter") or {}
            rows.append({"impl": "nccl_tests", "mode": mode, "place": place, "bytes": res["size"],
                         "time_us": d["time"] if "time" in d else d["cpu_time"], "p50_us": per_iter.get("p50_us"),
                         "algbw_GBps": d["alg_bw"], "busbw_GBps": d["bus_bw"], "nwrong": d.get("nwrong"),
                         "algo": tuning.get("algo"), "proto": tuning.get("proto"),
                         "nccl_version": doc["nccl_version"]})
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows
