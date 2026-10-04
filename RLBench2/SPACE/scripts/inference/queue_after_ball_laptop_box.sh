#!/usr/bin/env bash
# After local ball e250/e300 finishes, eval:
#   laptop: 700 750
#   box:    450 500 550 600
set -euo pipefail

PPI_ROOT="RLBench2/SPACE"
cd "$PPI_ROOT"
mkdir -p Logs

WAIT_PID="${WAIT_PID:-484160}"   # ball epochs_4gpu parent
LOG="Logs/queue_after_ball_laptop700750_box450-600_$(date +%Y%m%d_%H%M%S).log"
EPOCHS_SH="scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh"

exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "[$(date)] queue start HOST=$(hostname)"
echo "WAIT_PID=$WAIT_PID  LOG=$LOG"
echo "Plan: wait ball -> laptop 700 750 -> box 450 500 550 600"
echo "=============================================="

echo "[$(date)] waiting for ball parent PID=$WAIT_PID ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date)] parent $WAIT_PID gone"

echo "[$(date)] waiting until no eval_ppi_rp1.py on this host ..."
while pgrep -f "inference-for-rlbench2/eval_ppi_rp1.py" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date)] GPUs clear of eval_ppi_rp1.py"
sleep 5

export DISPLAY_NUM="${DISPLAY_NUM:-97}"
export NGPU=4
export NUM_INFER=1000
export TOTAL_EPS=100

echo "[$(date)] === laptop epochs 700 750 ==="
TASK=laptop CKPT_EPOCHS="700 750" bash "$EPOCHS_SH"
ec1=$?
echo "[$(date)] laptop done exit=$ec1"
if [ "$ec1" -ne 0 ]; then
  echo "[FATAL] laptop eval failed; stop before box"
  exit "$ec1"
fi

echo "[$(date)] === box epochs 450 500 550 600 ==="
TASK=box CKPT_EPOCHS="450 500 550 600" bash "$EPOCHS_SH"
ec2=$?
echo "[$(date)] box done exit=$ec2"

echo "[$(date)] ALL QUEUE DONE laptop_ec=$ec1 box_ec=$ec2"
exit "$ec2"
