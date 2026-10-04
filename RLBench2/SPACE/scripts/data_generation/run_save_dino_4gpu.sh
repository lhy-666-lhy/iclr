#!/usr/bin/env bash
# Launch save_dino on 4 GPUs for all 7 Open-PPI tasks.
# Usage: bash scripts/data_generation/run_save_dino_4gpu.sh
set -euo pipefail

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PPI_ROOT"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:?"set COPPELIASIM_ROOT"}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export PYTHONUNBUFFERED=1

PY="${PY:-python}"
BATCH_STEPS="${BATCH_STEPS:-80}"
LOGDIR="$PPI_ROOT/Logs/save_dino"
mkdir -p "$LOGDIR"
echo "[$(date)] BATCH_STEPS=$BATCH_STEPS (dino imgs=$((BATCH_STEPS * 6))/forward)"

TASKS=(
  bimanual_lift_ball
  bimanual_lift_tray
  bimanual_push_box
  bimanual_pick_laptop
  bimanual_put_item_in_drawer
  bimanual_sweep_to_dustpan
  bimanual_handover_item_easy
)

NUM_GPU=4
# each GPU: shard episodes 0-24, 25-49, 50-74, 75-99 across tasks sequentially via a simple queue

# Build job list: task start end
JOBS=()
for t in "${TASKS[@]}"; do
  JOBS+=("$t 0 24")
  JOBS+=("$t 25 49")
  JOBS+=("$t 50 74")
  JOBS+=("$t 75 99")
done

echo "[$(date)] total jobs=${#JOBS[@]} logdir=$LOGDIR"

# Round-robin assign jobs to GPUs: each GPU runs its jobs sequentially in background
for gpu in $(seq 0 $((NUM_GPU - 1))); do
  (
    idx=0
    for job in "${JOBS[@]}"; do
      # pick jobs where idx % NUM_GPU == gpu
      if (( idx % NUM_GPU == gpu )); then
        read -r task start end <<<"$job"
        log="$LOGDIR/${task}_ep${start}_${end}_gpu${gpu}.log"
        echo "[$(date)] GPU$gpu START $task $start-$end" | tee -a "$LOGDIR/master.log"
        "$PY" scripts/data_generation/save_dino_worker.py \
          --task "$task" --device "cuda:$gpu" --start "$start" --end "$end" \
          --batch-steps "$BATCH_STEPS" \
          >"$log" 2>&1 \
          && echo "[$(date)] GPU$gpu DONE $task $start-$end" | tee -a "$LOGDIR/master.log" \
          || echo "[$(date)] GPU$gpu FAIL $task $start-$end" | tee -a "$LOGDIR/master.log"
      fi
      idx=$((idx + 1))
    done
  ) &
  echo "launched GPU$gpu worker pid=$!"
done

wait
echo "[$(date)] ALL save_dino JOBS FINISHED" | tee -a "$LOGDIR/master.log"
