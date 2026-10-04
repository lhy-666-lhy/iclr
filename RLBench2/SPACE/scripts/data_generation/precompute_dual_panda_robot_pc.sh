#!/usr/bin/env bash
# Precompute DualPanda full-link robot PCs (incl. gripper + finger1/2).
set -euo pipefail

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PPI_ROOT"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:?"set COPPELIASIM_ROOT"}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM=offscreen

TASK="${1:-bimanual_lift_ball}"
START="${2:-0}"
END="${3:-99}"
M="${4:-16}"

echo "[$(date)] precompute robot_pc task=$TASK ep=$START-$END M=$M"
python scripts/data_generation/precompute_dual_panda_robot_pc.py \
  --task "$TASK" --start "$START" --end "$END" --points_per_link "$M"
