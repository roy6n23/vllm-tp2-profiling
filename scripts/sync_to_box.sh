#!/usr/bin/env bash
# Copy this repo (and Project 1, if present) to the rented box over the mapped SSH port (spec AM29).
# Usage: scripts/sync_to_box.sh HOST PORT     (HOST and PORT come from `python3 scripts/runpod.py wait POD_ID`)
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: scripts/sync_to_box.sh HOST PORT" >&2
    exit 2
fi
HOST=$1
PORT=$2

cd "$(dirname "$0")/.."

# The brief's rsync command, unchanged. --delete makes the box copy identical to this checkout, so any
# box file that is missing here is deleted, including run records under the box's results/ that were not
# pulled back yet. Pull results/ back from the box before re-syncing during a rental.
rsync -az --delete --exclude .venv --exclude results/dryrun -e "ssh -p $PORT" ./ "root@$HOST:/workspace/vllm-tp2-profiling/"
echo "synced $(pwd) -> root@$HOST:/workspace/vllm-tp2-profiling/"

FA2="${TPPROF_FA2_DIR:-$HOME/Documents/Personal/triton-fa2-forward}"   # Project 1 checkout; override per machine
if [ -d "$FA2" ]; then
    rsync -az --delete --exclude .venv -e "ssh -p $PORT" "$FA2/" "root@$HOST:/workspace/triton-fa2-forward/"
    echo "synced $FA2 -> root@$HOST:/workspace/triton-fa2-forward/"
else
    echo "skipped Project 1: $FA2 not found"
fi

echo "Liger: this script does not copy Liger-Kernel. Its optimization/ directory is git-excluded and lives on"
echo "the other machine; copy it to the box separately before the Liger block."
echo "Re-sync: --delete removes box files missing here, including results/; pull results/ back first."
echo "next: ssh -p $PORT root@$HOST, then: tmux new -s tp2, then: bash /workspace/vllm-tp2-profiling/scripts/bootstrap_box.sh"
