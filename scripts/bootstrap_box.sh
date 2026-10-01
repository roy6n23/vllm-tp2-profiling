#!/usr/bin/env bash
# Prepare the rented box. Run on the box, inside tmux, after scripts/sync_to_box.sh:
#     bash /workspace/vllm-tp2-profiling/scripts/bootstrap_box.sh
# Safe to rerun: every step either skips work already done or repeats it harmlessly.
set -euo pipefail

REPO=/workspace/vllm-tp2-profiling
MODEL_DIR=/workspace/models/llama31-8b-instruct
RESULTS_DIR=$REPO/results
ENV_FILE=/workspace/tpprof.env
NCCL_SCRIPT=/workspace/nccl-tests.sh
NCCL_LOG=/workspace/nccl-tests-build.log
NCCL_PID=/workspace/nccl-tests-build.pid

step() { printf '\n==> [%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf 'bootstrap_box.sh: %s\n' "$*" >&2; exit 1; }

# 1. The Docker ENV of the image; SSH sessions do not inherit it (AM28).
step "1/9 image environment"
[ -r /etc/profile.d/00-image-env.sh ] || die "/etc/profile.d/00-image-env.sh is missing; was the pod created by scripts/runpod.py create?"
# shellcheck disable=SC1091
source /etc/profile.d/00-image-env.sh
# Every Python step runs `python`, as the brief and AM27 write it. The vLLM image links only
# /usr/bin/python3 (vLLM v0.30.0 docker/Dockerfile:786-788) and has no `python` command, so point one at
# python3 once. No download; a rerun finds it already there.
if ! command -v python >/dev/null 2>&1; then
    py3=$(command -v python3) || die "neither python nor python3 is on PATH"
    ln -sf "$py3" /usr/local/bin/python
    echo "linked /usr/local/bin/python -> $py3"
fi
cd "$REPO"

# Pinned values come from tpprof/constants.py only (importable from the repo dir before any install).
# Assigned before eval so that a failing import aborts the script under set -e.
pinned=$(python - <<'EOF'
import shlex
from tpprof import constants as c
for name, value in (("NSYS_PKG", c.NSYS_APT_PACKAGE), ("NSYS_VERSION", c.NSYS_VERSION),
                    ("NSYS", c.nsys_path()), ("MODEL_REPO", c.MODEL.repo), ("MODEL_REV", c.MODEL.revision)):
    print(f"{name}={shlex.quote(value)}")
EOF
)
eval "$pinned"

# 2. Quick hardware gates before any download (AM27).
step "2/9 quick preflight"
if ! python -m tpprof preflight --quick; then
    echo "Quick preflight failed: this box is not usable. On your Mac run: python3 scripts/runpod.py terminate <POD_ID>" >&2
    echo "Then create a new pod and retry." >&2
    exit 1
fi

# 3. Nsight Systems from the NVIDIA devtools apt repo (D6-15), invoked by absolute path.
step "3/9 nsys $NSYS_VERSION"
installed=""
if [ -x "$NSYS" ]; then installed=$("$NSYS" --version 2>/dev/null || true); fi
if [[ "$installed" == *"$NSYS_VERSION"* ]]; then
    echo "already installed: $NSYS"
else
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends gnupg2 wget ca-certificates
    wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
        | gpg --dearmor | tee /usr/share/keyrings/nvidia-devtools-keyring.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu$(. /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /" \
        | tee /etc/apt/sources.list.d/nvidia-devtools.list
    apt-get update -qq
    apt-get install -y -qq "$NSYS_PKG"
fi
nsys_version=$("$NSYS" --version)
echo "$nsys_version"
case "$nsys_version" in
    *"$NSYS_VERSION"*) ;;
    *) die "expected nsys $NSYS_VERSION at $NSYS, got: $nsys_version" ;;
esac

# 4. tpprof into the image Python, keeping the image's numpy/psutil and NCCL pin (AM30).
step "4/9 pip install --no-deps -e $REPO"
python -m pip install --no-deps -e "$REPO"

# 5. Model weights on the volume disk; hf skips files that are already complete.
step "5/9 model $MODEL_REPO@$MODEL_REV"
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
t0=$SECONDS
hf download "$MODEL_REPO" --revision "$MODEL_REV" --exclude "original/*" --local-dir "$MODEL_DIR"
echo "model download took $((SECONDS - t0)) s"

# 6. nccl-tests build in the background (AM31); the full preflight does not wait for it.
step "6/9 nccl-tests build (background)"
if [ -f "$NCCL_PID" ] && kill -0 "$(cat "$NCCL_PID")" 2>/dev/null; then
    echo "already running: pid $(cat "$NCCL_PID"), log $NCCL_LOG"
else
    python -c 'from tpprof.nccltests import build_script; print(build_script())' > "$NCCL_SCRIPT"
    nohup bash "$NCCL_SCRIPT" > "$NCCL_LOG" 2>&1 &
    echo $! > "$NCCL_PID"
    echo "started: pid $!, log $NCCL_LOG"
fi

# 7. Paths every later tpprof command reads.
step "7/9 $ENV_FILE"
mkdir -p "$RESULTS_DIR"
cat > "$ENV_FILE" <<EOF
export TPPROF_MODEL_DIR=$MODEL_DIR
export TPPROF_RESULTS_DIR=$RESULTS_DIR
EOF
cat "$ENV_FILE"
# shellcheck disable=SC1090
source "$ENV_FILE"

# 8. Full preflight: versions, flags, model files, nsys, imports.
step "8/9 full preflight"
python -m tpprof preflight

step "9/9 done"
echo "next: source $ENV_FILE && cd $REPO && python -m tpprof matrix --estimate"
echo "then: python -m tpprof run --tier P0 2>&1 | tee -ai results/p0.log"
