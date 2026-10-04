#!/usr/bin/env bash
# 4-GPU PPI_rp1 drawer eval: random online init (no stored demos), NUM_INFER=1000
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

WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_drawer_rp1_ppi_rp1_20260831-drawer-fullmesh_seed0}"
CKPT_NAME="${CKPT_NAME:-epoch450_model}"
# Empty: do not load stored demos; env samples random variations online.
DEMO_PATH="${DEMO_PATH:-}"
NUM_INFER="${NUM_INFER:-1000}"
TOTAL_EPS="${TOTAL_EPS:-100}"
NGPU="${NGPU:-4}"

LOGDIR="Logs/eval_rp1/drawer_4gpu_random_infer${NUM_INFER}"
mkdir -p "$LOGDIR"

per=$(( TOTAL_EPS / NGPU ))
rem=$(( TOTAL_EPS % NGPU ))
start=0
pids=()
for gpu in $(seq 0 $((NGPU - 1))); do
  n=$per
  if [ "$gpu" -lt "$rem" ]; then n=$((n + 1)); fi
  if [ "$n" -le 0 ]; then continue; fi
  echo "GPU $gpu: eval_from_eps_number=$start eval_episodes=$n (random online)"
  CUDA_VISIBLE_DEVICES="$gpu" python inference-for-rlbench2/eval_ppi_rp1.py \
    framework.eval_from_eps_number="$start" \
    framework.eval_episodes="$n" \
    framework.csv_logging=True \
    framework.tensorboard_logging=false \
    framework.eval_type=1101 \
    framework.weight_name="$WEIGHT_NAME" \
    framework.ckpt_name="$CKPT_NAME" \
    framework.jump_step=1 \
    framework.weightsdir="exp_logs/ckpt/bimanual_put_item_in_drawer" \
    framework.logdir="exp_logs/eval_rp1_random_gpu${gpu}" \
    framework.record_every_n=0 \
    rlbench.headless=True \
    rlbench.episode_length=300 \
    rlbench.task_name="drawer" \
    rlbench.tasks=[bimanual_put_item_in_drawer] \
    rlbench.demo_path="$DEMO_PATH" \
    rlbench.include_lang_goal_in_obs=true \
    rlbench.query_freq=20 \
    method.policy.horizon_keyframe=4 \
    method.policy.horizon_continuous=50 \
    method.policy.n_obs_steps=1 \
    method.policy.n_action_steps=54 \
    method.policy.bounding_box="[[-0.5,-0.55,0.77],[1.1,0.55,1.98]]" \
    method.policy.fps_num=6144 \
    method.policy.prediction_type='keyframe_continuous' \
    method.policy.what_condition='ppi' \
    method.policy.pointflow_num=200 \
    method.policy.text_prompt="a small white cube." \
    method.policy.prompt_type="box" \
    method.policy.sample_type="rps" \
    method.policy.num_inference_steps="$NUM_INFER" \
    method.policy.robot_points_per_link=16 \
    method.policy.sam_cameras="[over_shoulder_left]" \
    cinematic_recorder.enabled=False \
    > "${LOGDIR}/gpu${gpu}_eps${start}_n${n}.log" 2>&1 &
  pids+=($!)
  echo "  PID=${pids[-1]} log=${LOGDIR}/gpu${gpu}_eps${start}_n${n}.log"
  start=$((start + n))
done

echo "Launched ${#pids[@]} jobs. TOTAL_EPS=$TOTAL_EPS NUM_INFER=$NUM_INFER mode=random_online"
printf '%s\n' "${pids[@]}" > "${LOGDIR}/pids.txt"
