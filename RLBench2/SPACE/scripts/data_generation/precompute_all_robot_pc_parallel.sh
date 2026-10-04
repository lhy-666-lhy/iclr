#!/usr/bin/env bash
# Parallel precompute DualPanda robot_pc on many CPU cores.
# One PyRep/Coppelia instance per worker; each worker handles one (task, episode).
#
# Usage (128-core node, coexist with eval):
#   cd .../RLBench2/SPACE
#   export N_JOBS=88
#   nohup bash scripts/data_generation/precompute_all_robot_pc_parallel.sh \
#     > Logs/precompute_robot_pc/parallel.log 2>&1 &
set -euo pipefail

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PPI_ROOT"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:?"set COPPELIASIM_ROOT"}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM=offscreen
export PYTHONUNBUFFERED=1

N_JOBS="${N_JOBS:-90}"
M="${PRECOMPUTE_M:-16}"
START="${PRECOMPUTE_START:-0}"
END="${PRECOMPUTE_END:-104}"
LOG_DIR="$PPI_ROOT/Logs/precompute_robot_pc"
mkdir -p "$LOG_DIR"

JOBFILE="$LOG_DIR/joblist_ep${START}-${END}.tsv"
WORKER="$PPI_ROOT/scripts/data_generation/precompute_robot_pc_one_episode.sh"

OVERWRITE_ARGS=()
[[ "${PRECOMPUTE_OVERWRITE:-0}" == "1" ]] && OVERWRITE_ARGS=(--overwrite)

echo "[$(date)] generate job list..."
python scripts/data_generation/gen_robot_pc_joblist.py \
  --start "$START" --end "$END" "${OVERWRITE_ARGS[@]}" > "$JOBFILE"
python scripts/data_generation/gen_robot_pc_joblist.py \
  --start "$START" --end "$END" "${OVERWRITE_ARGS[@]}" --count_only

NJ=$(wc -l < "$JOBFILE")
if [[ "$NJ" -eq 0 ]]; then
  echo "[$(date)] nothing to do"
  exit 0
fi

echo "[$(date)] launch $NJ jobs with N_JOBS=$N_JOBS M=$M"

# Job pool semaphore (works without GNU parallel)
SEM="$LOG_DIR/sem.pipe"
rm -f "$SEM"
mkfifo "$SEM"
exec 9<>"$SEM"
for ((i = 0; i < N_JOBS; i++)); do echo >&9; done

done_cnt=0
fail_cnt=0
while IFS=$'\t' read -r task ep; do
  read -r -u 9
  {
    log="$LOG_DIR/${task}_ep${ep}.log"
    if bash "$WORKER" "$task" "$ep" "$M" >"$log" 2>&1; then
      echo "[ok] $task ep$ep"
    else
      echo "[FAIL] $task ep$ep (see $log)"
    fi
    echo >&9
  } &
done < "$JOBFILE"

wait
exec 9>&-

echo "[$(date)] ALL DONE"
find data/training_processed/robot_pc -name '*.npz' | wc -l | xargs echo 'total_npz='
