#!/usr/bin/env bash
# Worker: precompute one (task, episode). Called by parallel launcher.
set -euo pipefail
TASK="${1:?task}"
EP="${2:?episode}"
M="${3:-16}"

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PPI_ROOT"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:?"set COPPELIASIM_ROOT"}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM=offscreen
export PYTHONUNBUFFERED=1

OVERWRITE_ARGS=()
[[ "${PRECOMPUTE_OVERWRITE:-0}" == "1" ]] && OVERWRITE_ARGS=(--overwrite)

python scripts/data_generation/precompute_dual_panda_robot_pc.py \
  --task "$TASK" --start "$EP" --end "$EP" --points_per_link "$M" \
  "${OVERWRITE_ARGS[@]}"
