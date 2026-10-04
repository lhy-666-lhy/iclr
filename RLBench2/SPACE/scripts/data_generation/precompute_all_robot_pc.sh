#!/usr/bin/env bash
# Precompute DualPanda full-link robot_pc for all Open-PPI RLBench2 tasks.
# CPU only. Skips existing npz unless PRECOMPUTE_OVERWRITE=1.
#
# Usage:
#   nohup bash scripts/data_generation/precompute_all_robot_pc.sh > Logs/precompute_robot_pc/all.log 2>&1 &
#   tail -f Logs/precompute_robot_pc/all.log
set -euo pipefail

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PPI_ROOT"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:?"set COPPELIASIM_ROOT"}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM=offscreen

LOG_DIR="$PPI_ROOT/Logs/precompute_robot_pc"
mkdir -p "$LOG_DIR"

TASKS=(
  bimanual_handover_item_easy
  bimanual_lift_ball
  bimanual_lift_tray
  bimanual_pick_laptop
  bimanual_push_box
  bimanual_put_item_in_drawer
  bimanual_sweep_to_dustpan
)

START="${PRECOMPUTE_START:-0}"
END="${PRECOMPUTE_END:-104}"
M="${PRECOMPUTE_M:-16}"
OVERWRITE_FLAG=""
[[ "${PRECOMPUTE_OVERWRITE:-0}" == "1" ]] && OVERWRITE_FLAG="--overwrite"

echo "[$(date)] ===== precompute ALL robot_pc start ====="
echo "tasks=${#TASKS[@]} ep=${START}-${END} M=${M} overwrite=${PRECOMPUTE_OVERWRITE:-0}"

for task in "${TASKS[@]}"; do
  log="$LOG_DIR/${task}_ep${START}-${END}_m${M}.log"
  echo "[$(date)] >>> task=$task log=$log"
  python scripts/data_generation/precompute_dual_panda_robot_pc.py \
    --task "$task" --start "$START" --end "$END" --points_per_link "$M" \
    $OVERWRITE_FLAG 2>&1 | tee "$log"
  echo "[$(date)] <<< done task=$task"
done

echo "[$(date)] ===== ALL DONE ====="
find data/training_processed/robot_pc -name '*.npz' | wc -l | xargs echo 'total_npz='
