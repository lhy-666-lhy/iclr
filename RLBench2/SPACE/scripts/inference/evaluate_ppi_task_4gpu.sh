#!/usr/bin/env bash
# 4-GPU original PPI eval with PerAct2 fixed test seeds (low_dim only, no video).
# Usage:
#   TASK=drawer CKPT_NAME=epoch450_model bash scripts/inference/evaluate_ppi_task_4gpu.sh
set -euo pipefail

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PPI_ROOT"

export PYTHONPATH="${PPI_ROOT}:${PPI_ROOT}/inference-for-rlbench2:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:?"set COPPELIASIM_ROOT"}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export DISPLAY="${DISPLAY:-:99}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASK="${TASK:-drawer}"
CKPT_NAME="${CKPT_NAME:-epoch450_model}"
NUM_INFER="${NUM_INFER:-1000}"
TOTAL_EPS="${TOTAL_EPS:-100}"
NGPU="${NGPU:-4}"
DEMO_PATH="${DEMO_PATH:-${PPI_ROOT}/data/rlbench2_official/peract2_test_ppi}"

case "$TASK" in
  drawer)
    TASK_NAME="drawer"
    RL_TASK="bimanual_put_item_in_drawer"
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_put_item_in_drawer"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_8gpus_drawer_ppi_20260831-drawer_seed0}"
    QUERY_FREQ=20
    EP_LEN=300
    TEXT_PROMPT="a small white cube."
    PROMPT_TYPE="box"
    SAM_CAMERAS="[over_shoulder_left]"
    ;;
  laptop)
    TASK_NAME="laptop"
    RL_TASK="bimanual_pick_laptop"
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_pick_laptop"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_8gpus_laptop_ppi_20260831-laptop_seed0}"
    QUERY_FREQ=15
    EP_LEN=300
    TEXT_PROMPT="a black rectangle laptop"
    PROMPT_TYPE="point"
    SAM_CAMERAS="[front]"
    ;;
  handover_easy)
    TASK_NAME="handover_easy"
    RL_TASK="bimanual_handover_item_easy"
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_handover_item_easy"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_8gpus_handover_easy_ppi_20260831-handover_easy_seed0}"
    QUERY_FREQ=20
    EP_LEN=300
    TEXT_PROMPT="a small red cube"
    PROMPT_TYPE="box"
    SAM_CAMERAS="[front]"
    ;;
  *)
    echo "Unknown TASK=$TASK (supported: drawer|laptop|handover_easy)"
    exit 1
    ;;
esac

CKPT_PATH="${WEIGHTS_DIR}/${WEIGHT_NAME}/checkpoints/${CKPT_NAME}.pth.tar"
if [ ! -f "$CKPT_PATH" ]; then
  echo "Missing ckpt: $CKPT_PATH"
  echo "Original PPI drawer currently only has early epochs until training reaches 450."
  ls -lt "${WEIGHTS_DIR}/${WEIGHT_NAME}/checkpoints/" 2>/dev/null | head -20 || true
  exit 1
fi

if [ ! -d "$DEMO_PATH" ]; then
  echo "Missing DEMO_PATH: $DEMO_PATH"
  exit 1
fi

LOGDIR="exp_logs/eval_runs/ppi_${TASK_NAME}_4gpu_${CKPT_NAME}_peract2seeds_infer${NUM_INFER}"
mkdir -p "$LOGDIR"
echo "Using DEMO_PATH=$DEMO_PATH"

per=$(( TOTAL_EPS / NGPU ))
rem=$(( TOTAL_EPS % NGPU ))
start=0
pids=()
for gpu in $(seq 0 $((NGPU - 1))); do
  n=$per
  if [ "$gpu" -lt "$rem" ]; then n=$((n + 1)); fi
  if [ "$n" -le 0 ]; then continue; fi
  echo "GPU $gpu: original PPI task=$TASK_NAME ckpt=$CKPT_NAME from=$start n=$n"
  CUDA_VISIBLE_DEVICES="$gpu" python inference-for-rlbench2/eval_ppi.py \
    framework.eval_from_eps_number="$start" \
    framework.eval_episodes="$n" \
    framework.csv_logging=True \
    framework.tensorboard_logging=false \
    framework.eval_type=1101 \
    framework.weight_name="$WEIGHT_NAME" \
    framework.ckpt_name="$CKPT_NAME" \
    framework.jump_step=1 \
    framework.weightsdir="$WEIGHTS_DIR" \
    framework.logdir="exp_logs/eval_ppi_${TASK_NAME}_${CKPT_NAME}_gpu${gpu}" \
    framework.record_every_n=0 \
    rlbench.headless=True \
    rlbench.episode_length="$EP_LEN" \
    rlbench.task_name="$TASK_NAME" \
    rlbench.tasks="[${RL_TASK}]" \
    rlbench.demo_path="$DEMO_PATH" \
    rlbench.include_lang_goal_in_obs=true \
    rlbench.query_freq="$QUERY_FREQ" \
    method.policy.horizon_keyframe=4 \
    method.policy.horizon_continuous=50 \
    method.policy.n_obs_steps=1 \
    method.policy.n_action_steps=54 \
    method.policy.bounding_box="[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]" \
    method.policy.fps_num=6144 \
    method.policy.prediction_type='keyframe_continuous' \
    method.policy.what_condition='ppi' \
    method.policy.pointflow_num=200 \
    method.policy.text_prompt="$TEXT_PROMPT" \
    method.policy.prompt_type="$PROMPT_TYPE" \
    method.policy.sample_type="rps" \
    method.policy.num_inference_steps="$NUM_INFER" \
    method.policy.sam_cameras="$SAM_CAMERAS" \
    cinematic_recorder.enabled=False \
    > "${LOGDIR}/gpu${gpu}_eps${start}_n${n}.log" 2>&1 &
  pids+=($!)
  echo "  PID=${pids[-1]} log=${LOGDIR}/gpu${gpu}_eps${start}_n${n}.log"
  start=$((start + n))
done

echo "Launched ${#pids[@]} jobs. METHOD=PPI TASK=$TASK CKPT=$CKPT_NAME TOTAL_EPS=$TOTAL_EPS"
printf '%s\n' "${pids[@]}" > "${LOGDIR}/pids.txt"
echo "LOGDIR=$LOGDIR"
