#!/usr/bin/env bash
# 4-GPU PPI_rp1 eval with PerAct2 fixed test seeds (low_dim only, no video).
# Usage:
#   TASK=drawer CKPT_NAME=epoch450_model bash scripts/inference/evaluate_ppi_rp1_task_4gpu.sh
#   TASK=laptop CKPT_NAME=epoch450_model bash scripts/inference/evaluate_ppi_rp1_task_4gpu.sh
#   TASK=handover_easy CKPT_NAME=epoch450_model bash scripts/inference/evaluate_ppi_rp1_task_4gpu.sh
#   TASK=ball CKPT_NAME=epoch450_model bash scripts/inference/evaluate_ppi_rp1_task_4gpu.sh
#   TASK=box  CKPT_NAME=epoch450_model bash scripts/inference/evaluate_ppi_rp1_task_4gpu.sh
set -euo pipefail

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PPI_ROOT"

export PYTHONPATH="${PPI_ROOT}:${PPI_ROOT}/inference-for-rlbench2:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
_DEFAULT_COPPELIASIM="${COPPELIASIM_ROOT:-}"
if [ -n "${COPPELIASIM_ROOT:-}" ] && { [ -e "${COPPELIASIM_ROOT}/libcoppeliaSim.so" ] || [ -e "${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" ]; }; then
  :
else
  COPPELIASIM_ROOT="$_DEFAULT_COPPELIASIM"
fi
export COPPELIASIM_ROOT
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
export DISPLAY="${DISPLAY:-:${DISPLAY_NUM}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Ensure virtual display (do NOT rely on xdpyinfo; many nodes lack x11-utils).
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix 2>/dev/null || true
_disp="${DISPLAY#:}"
if [ ! -S "/tmp/.X11-unix/X${_disp}" ]; then
  echo "[setup] DISPLAY $DISPLAY socket missing; starting Xvfb"
  rm -f "/tmp/.X${_disp}-lock" 2>/dev/null || true
  Xvfb ":${_disp}" -screen 0 1024x768x24 -ac +extension GLX -nolisten tcp \
    >"/tmp/xvfb_${_disp}.log" 2>&1 &
  _ok=0
  for _i in $(seq 1 40); do
    if [ -S "/tmp/.X11-unix/X${_disp}" ]; then _ok=1; break; fi
    sleep 0.25
  done
  if [ "$_ok" -ne 1 ]; then
    # fallback displays
    for _alt in 90 91 92 93 94 95 96 97 98; do
      if [ -S "/tmp/.X11-unix/X${_alt}" ]; then
        export DISPLAY=":${_alt}"; DISPLAY_NUM="$_alt"; _ok=1; break
      fi
      [ -e "/tmp/.X${_alt}-lock" ] && continue
      Xvfb ":${_alt}" -screen 0 1024x768x24 -ac +extension GLX -nolisten tcp \
        >"/tmp/xvfb_${_alt}.log" 2>&1 &
      for _i in $(seq 1 40); do
        if [ -S "/tmp/.X11-unix/X${_alt}" ]; then
          export DISPLAY=":${_alt}"; DISPLAY_NUM="$_alt"; _ok=1; break
        fi
        sleep 0.25
      done
      [ "$_ok" -eq 1 ] && break
    done
  fi
  if [ "$_ok" -ne 1 ]; then
    echo "FATAL: cannot start Xvfb (see /tmp/xvfb_*.log)"
    exit 1
  fi
fi
echo "[setup] using DISPLAY=$DISPLAY"

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
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_drawer_rp1_ppi_rp1_20260831-drawer-fullmesh_seed0}"
    QUERY_FREQ=20
    EP_LEN=300
    JUMP_STEP=1
    TEXT_PROMPT="a small white cube."
    PROMPT_TYPE="box"
    SAM_CAMERAS="[over_shoulder_left]"
    BOUNDING_BOX="[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]"
    ;;
  laptop)
    TASK_NAME="laptop"
    RL_TASK="bimanual_pick_laptop"
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_pick_laptop"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_laptop_rp1_ppi_rp1_20260831-laptop-fullmesh_seed0}"
    QUERY_FREQ=15
    EP_LEN=300
    JUMP_STEP=1
    TEXT_PROMPT="a black rectangle laptop"
    PROMPT_TYPE="point"
    SAM_CAMERAS="[front]"
    BOUNDING_BOX="[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]"
    ;;
  handover_easy)
    TASK_NAME="handover_easy"
    RL_TASK="bimanual_handover_item_easy"
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_handover_item_easy"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_handover_easy_rp1_ppi_rp1_20260831-handover_easy-fullmesh_seed0}"
    QUERY_FREQ=20
    EP_LEN=300
    JUMP_STEP=1
    TEXT_PROMPT="a small red cube"
    PROMPT_TYPE="box"
    SAM_CAMERAS="[front]"
    BOUNDING_BOX="[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]"
    ;;
  ball)
    TASK_NAME="ball"
    RL_TASK="bimanual_lift_ball"
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_lift_ball"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_ball_rp1_ppi_rp1_20260902-ball-fullmesh_seed0}"
    QUERY_FREQ=5
    EP_LEN=100
    JUMP_STEP=3
    TEXT_PROMPT="a white ball"
    PROMPT_TYPE="box"
    SAM_CAMERAS="[over_shoulder_right]"
    BOUNDING_BOX="[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]"
    ;;
  box)
    TASK_NAME="box"
    RL_TASK="bimanual_push_box"
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_push_box"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_box_rp1_ppi_rp1_20260902-box-fullmesh_seed0}"
    QUERY_FREQ=10
    EP_LEN=250
    JUMP_STEP=1
    TEXT_PROMPT="a white rectangle box"
    PROMPT_TYPE="box"
    SAM_CAMERAS="[over_shoulder_right]"
    BOUNDING_BOX="[[-0.5,-0.55,0.75],[1.1,0.55,1.98]]"
    ;;
  dustpan|sweep)
    TASK_NAME="dustpan"
    RL_TASK="bimanual_sweep_to_dustpan"
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_sweep_to_dustpan"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_dustpan_rp1_ppi_rp1_20260902-dustpan-fullmesh_seed0}"
    QUERY_FREQ=25
    EP_LEN=400
    JUMP_STEP=1
    TEXT_PROMPT="a top-down view of a brown T-shaped pole and its bottom."
    PROMPT_TYPE="box"
    SAM_CAMERAS="[over_shoulder_left]"
    BOUNDING_BOX="[[-0.5,-0.55,0.75],[1.1,0.55,1.98]]"
    ;;
  tray)
    TASK_NAME="tray"
    RL_TASK="bimanual_lift_tray"
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_lift_tray"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_lift_tray_rp1_ppi_rp1_20260902-tray-fullmesh_seed0}"
    QUERY_FREQ=10
    EP_LEN=300
    JUMP_STEP=1
    TEXT_PROMPT="a grey tray."
    PROMPT_TYPE="box"
    SAM_CAMERAS="[front]"
    BOUNDING_BOX="[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]"
    ;;
  *)
    echo "Unknown TASK=$TASK (supported: drawer|laptop|handover_easy|ball|box|dustpan|sweep|tray)"
    exit 1
    ;;
esac

CKPT_PATH="${WEIGHTS_DIR}/${WEIGHT_NAME}/checkpoints/${CKPT_NAME}.pth.tar"
if [ ! -f "$CKPT_PATH" ]; then
  echo "Missing ckpt: $CKPT_PATH"
  exit 1
fi

if [ ! -d "$DEMO_PATH" ]; then
  echo "Missing DEMO_PATH: $DEMO_PATH"
  exit 1
fi

# Keep under exp_logs/. Allow override so sequential runners never overwrite.
LOGDIR="${LOGDIR_OVERRIDE:-exp_logs/eval_runs/${TASK_NAME}_4gpu_${CKPT_NAME}_peract2seeds_infer${NUM_INFER}}"
mkdir -p "$LOGDIR"
echo "Using DEMO_PATH=$DEMO_PATH"
echo "Using LOGDIR=$LOGDIR DISPLAY=$DISPLAY COPPELIASIM_ROOT=$COPPELIASIM_ROOT"

per=$(( TOTAL_EPS / NGPU ))
rem=$(( TOTAL_EPS % NGPU ))
start=0
pids=()
for gpu in $(seq 0 $((NGPU - 1))); do
  n=$per
  if [ "$gpu" -lt "$rem" ]; then n=$((n + 1)); fi
  if [ "$n" -le 0 ]; then continue; fi
  echo "GPU $gpu: task=$TASK_NAME ckpt=$CKPT_NAME from=$start n=$n"
  CUDA_VISIBLE_DEVICES="$gpu" python inference-for-rlbench2/eval_ppi_rp1.py \
    framework.eval_from_eps_number="$start" \
    framework.eval_episodes="$n" \
    framework.csv_logging=True \
    framework.tensorboard_logging=false \
    framework.eval_type=1101 \
    framework.weight_name="$WEIGHT_NAME" \
    framework.ckpt_name="$CKPT_NAME" \
    framework.jump_step="$JUMP_STEP" \
    framework.weightsdir="$WEIGHTS_DIR" \
    framework.logdir="exp_logs/eval_rp1_${TASK_NAME}_${CKPT_NAME}_gpu${gpu}" \
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
    method.policy.bounding_box="$BOUNDING_BOX" \
    method.policy.fps_num=6144 \
    method.policy.prediction_type='keyframe_continuous' \
    method.policy.what_condition='ppi' \
    method.policy.pointflow_num=200 \
    method.policy.text_prompt="$TEXT_PROMPT" \
    method.policy.prompt_type="$PROMPT_TYPE" \
    method.policy.sample_type="rps" \
    method.policy.num_inference_steps="$NUM_INFER" \
    method.policy.robot_points_per_link=16 \
    method.policy.sam_cameras="$SAM_CAMERAS" \
    cinematic_recorder.enabled=False \
    > "${LOGDIR}/gpu${gpu}_eps${start}_n${n}.log" 2>&1 &
  pids+=($!)
  echo "  PID=${pids[-1]} log=${LOGDIR}/gpu${gpu}_eps${start}_n${n}.log"
  start=$((start + n))
done

echo "Launched ${#pids[@]} jobs. TASK=$TASK CKPT=$CKPT_NAME TOTAL_EPS=$TOTAL_EPS NUM_INFER=$NUM_INFER"
printf '%s\n' "${pids[@]}" > "${LOGDIR}/pids.txt"
echo "LOGDIR=$LOGDIR"

# Optional blocking wait (used by sequential epoch launcher).
if [ "${WAIT_FOR_JOBS:-0}" = "1" ]; then
  echo "WAIT_FOR_JOBS=1: waiting for ${#pids[@]} workers ..."
  fail=0
  for p in "${pids[@]}"; do
    if ! wait "$p"; then
      echo "WARNING: worker PID=$p exited non-zero"
      fail=1
    fi
  done
  # Early abort if all workers died before producing any Score.
  if ! grep -q 'Score:' "${LOGDIR}"/gpu*.log 2>/dev/null; then
    echo "FATAL: no Score: lines found after wait; startup likely failed."
    echo "---- last 30 lines of gpu0 log ----"
    tail -n 30 "${LOGDIR}"/gpu0_*.log 2>/dev/null || true
    exit 1
  fi
  if [ "$fail" -ne 0 ]; then
    echo "WARNING: some workers failed, but Score lines exist; check logs."
  fi
  echo "All workers finished for CKPT=$CKPT_NAME"
fi
