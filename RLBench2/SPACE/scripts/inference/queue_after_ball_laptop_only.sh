#!/usr/bin/env bash
# After local ball e250/e300 finishes, eval laptop 700 750 only (no box).
set -euo pipefail
PPI_ROOT="RLBench2/SPACE"
cd "$PPI_ROOT"
mkdir -p Logs
WAIT_PID="${WAIT_PID:-484160}"
LOG="Logs/queue_after_ball_laptop700750_$(date +%Y%m%d_%H%M%S).log"
EPOCHS_SH="scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] queue laptop-only start HOST=$(hostname) WAIT_PID=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] parent gone; wait eval_ppi clear"
while pgrep -f "inference-for-rlbench2/eval_ppi_rp1.py" >/dev/null 2>&1; do sleep 30; done
sleep 5
export DISPLAY_NUM="${DISPLAY_NUM:-97}" NGPU=4 NUM_INFER=1000 TOTAL_EPS=100
echo "[$(date)] === laptop 700 750 ==="
TASK=laptop CKPT_EPOCHS="700 750" bash "$EPOCHS_SH"
echo "[$(date)] laptop queue DONE exit=$?"
