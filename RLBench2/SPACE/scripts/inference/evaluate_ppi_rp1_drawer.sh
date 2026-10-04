#!/usr/bin/env bash
# PPI_rp1 eval: bimanual_put_item_in_drawer (fast: no video / no viz)
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
CKPT_NAME="${CKPT_NAME:-epoch800_model}"
DEMO_PATH="${DEMO_PATH:-}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
# Paper/sim default: 1000 DDPM steps (override with NUM_INFER=100 for faster debug)
NUM_INFER="${NUM_INFER:-1000}"

CUDA_VISIBLE_DEVICES="$GPU" python inference-for-rlbench2/eval_ppi_rp1.py \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=100 \
    framework.csv_logging=True \
    framework.tensorboard_logging=false \
    framework.eval_type=1101 \
    framework.weight_name="$WEIGHT_NAME" \
    framework.ckpt_name="$CKPT_NAME" \
    framework.jump_step=1 \
    framework.weightsdir="exp_logs/ckpt/bimanual_put_item_in_drawer" \
    framework.logdir="exp_logs/eval_rp1" \
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
    cinematic_recorder.enabled=False
